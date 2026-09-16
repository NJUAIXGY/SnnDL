#include <sst/core/sst_config.h>
#include "TraceNoCSourceV5.h"
#include "../../../../../../sst-core/external/nlohmann/json.hpp"

#include <algorithm>
#include <cinttypes>

namespace SST { namespace SnnDL { namespace v5 {
using json = nlohmann::json;

TraceNoCSourceV5::TraceNoCSourceV5(SST::ComponentId_t id, SST::Params& p)
    : Component(id), out_("SnnDL.TraceNoCSourceV5", 0, 0, Output::STDOUT),
      trace_file_(p.find<std::string>("trace_file", "")),
      output_json_(p.find<std::string>("output_json", "")),
      observation_json_(p.find<std::string>("observation_json", "")),
      multicast_mode_(p.find<std::string>("multicast_mode", "source_replication")),
      source_pe_(p.find<std::uint32_t>("source_pe", 0)),
      source_core_(p.find<std::uint32_t>("source_core", 0)),
      source_id_(p.find<std::uint32_t>("source_id", 0)),
      coordinator_source_id_(p.find<std::uint32_t>("coordinator_source_id", source_id_)),
      external_control_(p.find<int>("external_control", 0) != 0),
      max_line_bytes_(std::max<std::uint64_t>(128, p.find<std::uint64_t>("max_line_bytes", 1024 * 1024))),
      issue_width_(std::max<std::uint64_t>(1, p.find<std::uint64_t>("issue_width_per_source", 1))),
      max_outstanding_(std::max<std::uint64_t>(1, p.find<std::uint64_t>("max_outstanding_per_source", 1))),
      lookahead_limit_(std::max<std::uint64_t>(1, p.find<std::uint64_t>("lookahead_records", 64))) {
    out_.setVerboseLevel(p.find<int>("verbose", 0));
    if (trace_file_.empty()) out_.fatal(CALL_INFO, -1, "TraceNoCSourceV5 requires trace_file\n");
    trace_stream_.open(trace_file_);
    if (!trace_stream_.good()) out_.fatal(CALL_INFO, -1, "cannot open NoC trace_file=%s\n", trace_file_.c_str());
    if (!observation_json_.empty()) {
        observation_stream_.open(observation_json_, std::ios::out | std::ios::trunc);
        if (!observation_stream_.good()) out_.fatal(CALL_INFO, -1, "cannot open observation_json=%s\n", observation_json_.c_str());
    }
    inject_ = configureLink("inject");
    ack_ = configureLink("ack", new Event::Handler2<TraceNoCSourceV5, &TraceNoCSourceV5::handleAck_>(this));
    if (!inject_ || !ack_) out_.fatal(CALL_INFO, -1, "TraceNoCSourceV5 requires inject and ack links\n");
    if (external_control_) {
        command_ = configureLink("command", new Event::Handler2<TraceNoCSourceV5, &TraceNoCSourceV5::handleControl_>(this));
        status_ = configureLink("status");
        if (!command_ || !status_) out_.fatal(CALL_INFO, -1, "externally controlled TraceNoCSourceV5 requires command/status links\n");
    }
    registerClock(p.find<std::string>("clock", "1GHz"), new Clock::Handler2<TraceNoCSourceV5, &TraceNoCSourceV5::tick_>(this));
    registerAsPrimaryComponent();
    primaryComponentDoNotEndSim();
}

TraceNoCSourceV5::~TraceNoCSourceV5() {
    if (observation_stream_.is_open()) observation_stream_.flush();
}

bool TraceNoCSourceV5::readNext_(Record& result) {
    result = Record{};
    std::string line;
    while (std::getline(trace_stream_, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        if (line.size() > max_line_bytes_)
            out_.fatal(CALL_INFO, -1, "NoC trace record exceeds max_line_bytes=%" PRIu64 "\n", max_line_bytes_);
        try {
            const auto value = json::parse(line);
            if (!value.is_object() || value.value("schema_version", std::string()) != "snndl-logical-noc-record/v1")
                out_.fatal(CALL_INFO, -1, "TraceNoCSourceV5 requires snndl-logical-noc-record/v1\n");
            const auto source = value.at("source");
            const auto traffic = value.at("traffic");
            const auto multicast = value.at("multicast");
            const auto ordering = value.at("ordering");
            const auto release = value.at("release");
            result.event_id = value.at("event_id").get<std::string>();
            result.token = value.value("event_token", 0ULL);
            result.timestep = value.at("timestep").get<std::uint64_t>();
            result.source_pe = source.at("pe").get<std::uint32_t>();
            result.source_core = source.at("core").get<std::uint32_t>();
            result.source_neuron = source.at("neuron").get<std::uint64_t>();
            result.source_event_seq = source.at("event_sequence").get<std::uint64_t>();
            result.payload_bytes = traffic.at("payload_bytes").get<std::uint32_t>();
            result.virtual_network = traffic.at("virtual_network").get<std::uint32_t>();
            if (multicast.contains("route_id") && !multicast.at("route_id").is_null())
                result.route_id = multicast.at("route_id").get<std::uint64_t>();
            result.stream_sequence = ordering.at("stream_sequence").get<std::uint64_t>();
            result.release_tick = release.at("tick").get<std::uint64_t>();
            if (release.contains("depends_on") && release.at("depends_on").is_array())
                for (const auto& dependency : release.at("depends_on")) result.depends_on.push_back(dependency.get<std::string>());
            if (result.source_pe != source_pe_ || result.source_core != source_core_ || result.event_id.empty() || result.token == 0)
                out_.fatal(CALL_INFO, -1, "NoC trace record does not match source owner or token\n");
            if (have_sequence_ && result.stream_sequence <= last_sequence_)
                out_.fatal(CALL_INFO, -1, "NoC stream_sequence is not strictly increasing\n");
            have_sequence_ = true;
            last_sequence_ = result.stream_sequence;
            for (const auto& destination : value.at("destinations")) {
                result.destination_pes.push_back(destination.at("pe").get<std::uint32_t>());
                result.destination_masks.push_back(destination.at("core_mask").get<std::uint64_t>());
            }
            if (result.destination_pes.empty() || result.destination_pes.size() != result.destination_masks.size())
                out_.fatal(CALL_INFO, -1, "NoC trace record has no destination set\n");
            if (multicast_mode_ == "unicast") result.mode = TraceNoCMulticastMode::Unicast;
            else if (multicast_mode_ == "native_tree") result.mode = TraceNoCMulticastMode::NativeTree;
            else if (multicast_mode_ == "source_replication") result.mode = TraceNoCMulticastMode::SourceReplication;
            else out_.fatal(CALL_INFO, -1, "unknown NoC multicast_mode=%s\n", multicast_mode_.c_str());
            ++records_;
            return true;
        } catch (const std::exception& error) {
            out_.fatal(CALL_INFO, -1, "invalid logical NoC trace record: %s\n", error.what());
        }
    }
    return false;
}

TraceNoCInjectionV5Event* TraceNoCSourceV5::makeEvent_(const Record& r) const {
    auto* event = new TraceNoCInjectionV5Event();
    event->event_id = r.event_id;
    event->event_token = r.token;
    event->timestep = r.timestep;
    event->source_pe = r.source_pe;
    event->source_core = r.source_core;
    event->source_neuron = r.source_neuron;
    event->source_event_seq = r.source_event_seq;
    event->route_id = r.route_id;
    event->payload_bytes = r.payload_bytes;
    event->virtual_network = r.virtual_network;
    event->multicast_mode = r.mode;
    event->destination_pes = r.destination_pes;
    event->destination_core_masks = r.destination_masks;
    return event;
}

void TraceNoCSourceV5::emit_(const Record& record, const char* phase, std::uint64_t runtime_id,
                             std::uint64_t tick, std::uint64_t retries, std::uint32_t physical_packets) {
    if (!observation_stream_.good()) return;
    observation_stream_ << json{
        {"schema_version", "snndl-noc-replay-observation/v1"}, {"domain", "noc"},
        {"event_id", record.event_id}, {"event_token", record.token}, {"phase", phase},
        {"tick", tick}, {"runtime_id", runtime_id}, {"retry_count", retries},
        {"source_id", source_id_}, {"source_pe", record.source_pe}, {"source_core", record.source_core},
        {"route_id", record.route_id},
        {"timestep", record.timestep}, {"stream_sequence", record.stream_sequence},
        {"release_tick", record.release_tick}, {"physical_packets", physical_packets},
        {"multicast_mode", multicast_mode_}
    }.dump() << '\n';
    observation_stream_.flush();
}

void TraceNoCSourceV5::sendStatus_(TraceStatusOp operation) {
    if (!status_) return;
    auto* status = new TraceStatusEvent();
    status->operation = operation;
    status->source_id = coordinator_source_id_;
    status->expected_records = records_;
    status->eligible = eligible_records_;
    status->offered = offered_;
    status->injected = injected_;
    status->completed = completed_;
    status->outstanding = outstanding_.size();
    status->cycle = cycle_;
    status_->send(status);
}

bool TraceNoCSourceV5::dependenciesReady_(const Record& record) const {
    for (const auto& dependency : record.depends_on)
        if (completed_event_ids_.find(dependency) == completed_event_ids_.end()) return false;
    return true;
}

void TraceNoCSourceV5::handleControl_(SST::Event* raw) {
    auto* control = dynamic_cast<TraceControlEvent*>(raw);
    if (!control || control->source_id != coordinator_source_id_) {
        delete raw;
        out_.fatal(CALL_INFO, -1, "TraceNoCSourceV5 received an invalid control event\n");
    }
    if (control->operation == TraceControlOp::Start) {
        if (started_ || finished_) {
            delete control;
            out_.fatal(CALL_INFO, -1, "TraceNoCSourceV5 received duplicate Start\n");
        }
        started_ = true;
    } else {
        delete control;
        out_.fatal(CALL_INFO, -1, "TraceNoCSourceV5 aborted by replay coordinator\n");
    }
    delete control;
}

bool TraceNoCSourceV5::tick_(SST::Cycle_t) {
    if (finished_) return false;
    ++cycle_;
    if (external_control_) {
        if (!ready_sent_) {
            ready_sent_ = true;
            sendStatus_(TraceStatusOp::Ready);
            return false;
        }
        if (!started_) return false;
    }
    if (!have_next_ && !eof_) {
        have_next_ = readNext_(next_);
        eof_ = !have_next_;
    }
    while (have_next_ && next_.release_tick <= cycle_ && eligible_.size() < lookahead_limit_) {
        emit_(next_, "eligible", 0, cycle_, 0);
        ++eligible_records_;
        eligible_.push_back(next_);
        have_next_ = readNext_(next_);
        eof_ = !have_next_;
    }
    std::uint64_t issued = 0;
    while (issued < issue_width_ && outstanding_.size() < max_outstanding_ && !eligible_.empty()) {
        if (!dependenciesReady_(eligible_.front())) break;
        Record record = eligible_.front();
        eligible_.pop_front();
        if (!record.offered_before) {
            ++offered_;
            record.offered_before = true;
            emit_(record, "offered", 0, cycle_, 0);
        }
        outstanding_.emplace(record.token, InFlight{record, cycle_, 0});
        inject_->send(makeEvent_(record));
        emit_(record, "injected", record.token, cycle_, 0);
        ++issued;
    }
    if (eof_ && !have_next_ && eligible_.empty() && outstanding_.empty() && !finished_) {
        finished_ = true;
        drained_sent_ = true;
        if (external_control_) sendStatus_(TraceStatusOp::Drained);
        else primaryComponentOKToEndSim();
    }
    return false;
}

void TraceNoCSourceV5::handleAck_(SST::Event* raw) {
    auto* ack = dynamic_cast<TraceNoCInjectionAckV5Event*>(raw);
    if (!ack) {
        delete raw;
        out_.fatal(CALL_INFO, -1, "TraceNoCSourceV5 received an invalid injection ACK\n");
    }
    const auto found = outstanding_.find(ack->event_token);
    if (found == outstanding_.end() || ack->event_id != found->second.record.event_id) {
        delete ack;
        out_.fatal(CALL_INFO, -1, "TraceNoCSourceV5 received an unknown injection ACK\n");
    }
    InFlight flight = found->second;
    if (ack->accepted) {
        ++injected_;
        ++completed_;
        physical_packets_ += ack->physical_packets;
        accepted_event_ids_.push_back(flight.record.event_id);
        completed_event_ids_.insert(flight.record.event_id);
        emit_(flight.record, "accepted", flight.record.token, cycle_, flight.retries, ack->physical_packets);
        emit_(flight.record, "completed", flight.record.token, cycle_, flight.retries, ack->physical_packets);
        outstanding_.erase(found);
    } else if (ack->retryable) {
        ++retries_;
        emit_(flight.record, "retry", 0, cycle_, flight.retries + 1);
        outstanding_.erase(found);
        ++flight.retries;
        flight.record.offered_before = true;
        eligible_.push_front(flight.record);
    } else {
        delete ack;
        out_.fatal(CALL_INFO, -1, "TraceNoCSourceV5 injection was permanently rejected\n");
    }
    delete ack;
}

void TraceNoCSourceV5::writeEvidence_() const {
    if (output_json_.empty()) return;
    std::ofstream out(output_json_);
    out << "{\n  \"schema_version\": \"snndl-noc-source-evidence/v2\",\n"
        << "  \"source_id\": " << source_id_ << ",\n"
        << "  \"expected_logical_events\": " << records_ << ",\n"
        << "  \"eligible_events\": " << eligible_records_ << ",\n"
        << "  \"offered_events\": " << offered_ << ",\n"
        << "  \"injected_events\": " << injected_ << ",\n"
        << "  \"completed_events\": " << completed_ << ",\n"
        << "  \"accepted_events\": " << accepted_event_ids_.size() << ",\n"
        << "  \"retry_events\": " << retries_ << ",\n"
        << "  \"physical_source_packets\": " << physical_packets_ << ",\n"
        << "  \"outstanding_at_end\": " << outstanding_.size() << ",\n"
        << "  \"accepted_event_ids\": [";
    for (std::size_t index = 0; index < accepted_event_ids_.size(); ++index) {
        if (index != 0) out << ", ";
        out << json(accepted_event_ids_[index]).dump();
    }
    out << "],\n  \"status\": \""
        << ((finished_ && completed_ == records_) ? "PASS" : "INCOMPLETE") << "\"\n}\n";
}

void TraceNoCSourceV5::finish() {
    if (observation_stream_.is_open()) observation_stream_.flush();
    writeEvidence_();
    if (!finished_ || !eligible_.empty() || !outstanding_.empty() || completed_ != records_ || offered_ != records_)
        out_.fatal(CALL_INFO, -1, "TraceNoCSourceV5 finished before source replay drained\n");
}

}}}
