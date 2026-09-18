#include <sst/core/sst_config.h>

#include "TraceSramSourceV5.h"

#include "../../../../../../sst-core/external/nlohmann/json.hpp"

#include <algorithm>
#include <cinttypes>
#include <stdexcept>

namespace SST { namespace SnnDL { namespace v5 {

using json = nlohmann::json;

TraceSramSourceV5::TraceSramSourceV5(SST::ComponentId_t id, SST::Params& params)
    : Component(id), out_("SnnDL.TraceSramSourceV5", 0, 0, Output::STDOUT),
      trace_file_(params.find<std::string>("trace_file", "")),
      output_json_(params.find<std::string>("output_json", "")),
      observation_json_(params.find<std::string>("observation_json", "")),
      execution_mode_(params.find<std::string>("execution_mode", "trace_open_loop")),
      source_id_(params.find<std::uint32_t>("source_id", 0)),
      coordinator_source_id_(params.find<std::uint32_t>("coordinator_source_id", source_id_)),
      mixed_source_stream_(params.find<int>("mixed_source_stream", 0) != 0),
      external_control_(params.find<int>("external_control", 0) != 0),
      max_line_bytes_(std::max<std::uint64_t>(128, params.find<std::uint64_t>("max_line_bytes", 1024 * 1024))),
      issue_width_(std::max<std::uint64_t>(1, params.find<std::uint64_t>("issue_width_per_source", 1))),
      max_outstanding_(std::max<std::uint64_t>(1, params.find<std::uint64_t>("max_outstanding_per_source", 1))),
      lookahead_limit_(std::max<std::uint64_t>(1, params.find<std::uint64_t>("lookahead_records", 64))) {
    out_.setVerboseLevel(params.find<int>("verbose", 0));
    if (trace_file_.empty()) out_.fatal(CALL_INFO, -1, "TraceSramSourceV5 requires trace_file\n");
    trace_stream_.open(trace_file_);
    if (!trace_stream_.good()) out_.fatal(CALL_INFO, -1, "cannot open SRAM trace_file=%s\n", trace_file_.c_str());
    if (!observation_json_.empty()) {
        observation_stream_.open(observation_json_, std::ios::out | std::ios::trunc);
        if (!observation_stream_.good()) out_.fatal(CALL_INFO, -1, "cannot open observation_json=%s\n", observation_json_.c_str());
    }
    request_link_ = configureLink("request");
    response_link_ = configureLink("response", new Event::Handler2<TraceSramSourceV5, &TraceSramSourceV5::handleResponse_>(this));
    if (!request_link_ || !response_link_) out_.fatal(CALL_INFO, -1, "TraceSramSourceV5 requires request/response links\n");
    if (external_control_) {
        command_link_ = configureLink("command", new Event::Handler2<TraceSramSourceV5, &TraceSramSourceV5::handleControl_>(this));
        status_link_ = configureLink("status");
        if (!command_link_ || !status_link_) out_.fatal(CALL_INFO, -1, "externally controlled TraceSramSourceV5 requires command/status links\n");
    }
    registerClock(params.find<std::string>("clock", "1GHz"),
                  new Clock::Handler2<TraceSramSourceV5, &TraceSramSourceV5::clockTick_>(this));
    registerAsPrimaryComponent();
    primaryComponentDoNotEndSim();
}

TraceSramSourceV5::~TraceSramSourceV5() {
    // Preserve the append-only lifecycle prefix on abnormal SST teardown.
    if (observation_stream_.is_open()) observation_stream_.flush();
}

bool TraceSramSourceV5::readNext_(Record& result) {
    std::string line;
    while (std::getline(trace_stream_, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue;
        if (line.size() > max_line_bytes_) out_.fatal(CALL_INFO, -1, "SRAM trace record exceeds max_line_bytes=%" PRIu64 "\n", max_line_bytes_);
        try {
            const auto value = json::parse(line);
            if (!value.is_object() || value.value("schema_version", std::string()) != "snndl-materialized-sram-request/v1")
                out_.fatal(CALL_INFO, -1, "TraceSramSourceV5 requires snndl-materialized-sram-request/v1 records\n");
            result.request_id = value.at("request_id").get<std::uint64_t>();
            result.access_id = value.at("access_id").get<std::string>();
            result.request_class = value.value("request_class", std::string("sram-read"));
            result.region_id = value.at("region_id").get<std::string>();
            result.source_pe = value.value("source_pe", 0U);
            result.source_core = value.at("source_core").get<std::uint32_t>();
            result.address = value.at("address").get<std::uint64_t>();
            result.bytes = value.at("bytes").get<std::uint64_t>();
            result.release_tick = value.value("release_tick", 0ULL);
            result.timestep = value.value("timestep", 0ULL);
            result.stream_sequence = value.at("stream_sequence").get<std::uint64_t>();
            result.logical_bank = value.value("bank", 0U);
            result.write = value.value("write", false);
            const auto data = value.at("data");
            if (!data.is_array() || data.size() != result.bytes) out_.fatal(CALL_INFO, -1, "SRAM trace data length does not match bytes\n");
            result.data.clear();
            result.data.reserve(data.size());
            for (const auto& byte : data) {
                const auto parsed = byte.get<unsigned int>();
                if (parsed > 255) out_.fatal(CALL_INFO, -1, "SRAM trace payload byte is outside uint8 range\n");
                result.data.push_back(static_cast<std::uint8_t>(parsed));
            }
            if (result.request_id == 0 || result.access_id.empty() || result.region_id.empty() || result.bytes == 0 ||
                (!mixed_source_stream_ && result.source_core != source_id_))
                out_.fatal(CALL_INFO, -1, "invalid materialized SRAM request identity or size\n");
            if (!seen_request_ids_.insert(result.request_id).second)
                out_.fatal(CALL_INFO, -1, "duplicate SRAM request_id=%" PRIu64 "\n", result.request_id);
            if (have_sequence_ && result.stream_sequence <= last_stream_sequence_)
                out_.fatal(CALL_INFO, -1, "SRAM stream_sequence is not strictly increasing for source=%u\n", source_id_);
            have_sequence_ = true;
            last_stream_sequence_ = result.stream_sequence;
            ++records_;
            return true;
        } catch (const std::exception& error) {
            out_.fatal(CALL_INFO, -1, "invalid materialized SRAM trace record: %s\n", error.what());
        }
    }
    return false;
}

void TraceSramSourceV5::emit_(const Record& record, const char* phase, std::uint64_t runtime_id,
                              std::uint64_t tick, std::uint64_t retries, std::uint32_t bank,
                              std::uint32_t port, std::uint64_t service, std::uint64_t completion,
                              const char* status) {
    if (!observation_stream_.good()) return;
    observation_stream_ << json{
        {"schema_version", "snndl-sram-replay-observation/v1"},
        {"request_id", record.request_id}, {"access_id", record.access_id},
        {"component", "trace_sram_source.core." + std::to_string(source_id_)},
        {"domain", "sram"}, {"phase", phase}, {"tick", tick},
        {"runtime_id", runtime_id}, {"retry_count", retries}, {"status", status},
        {"source_id", source_id_}, {"source_pe", record.source_pe},
        {"source_core", record.source_core}, {"timestep", record.timestep},
        {"byte_address", record.address}, {"bytes", record.bytes},
        {"stream_sequence", record.stream_sequence}, {"release_tick", record.release_tick},
        {"region_id", record.region_id}, {"request_class", record.request_class},
        {"logical_bank", record.logical_bank}, {"physical_bank", bank}, {"port", port},
        {"service_cycle", service}, {"completion_cycle", completion}, {"write", record.write}
    }.dump() << '\n';
}

void TraceSramSourceV5::sendStatus_(TraceStatusOp operation) {
    if (!status_link_) return;
    auto* status = new TraceStatusEvent();
    status->operation = operation;
    status->source_id = coordinator_source_id_;
    status->expected_records = records_;
    status->eligible = eligible_records_;
    status->offered = offered_;
    status->injected = injected_;
    status->completed = completed_;
    status->request_bytes = bytes_;
    status->outstanding = outstanding_.size();
    status->cycle = cycle_;
    status_link_->send(status);
}

void TraceSramSourceV5::handleControl_(SST::Event* raw) {
    auto* control = dynamic_cast<TraceControlEvent*>(raw);
    if (!control) { delete raw; out_.fatal(CALL_INFO, -1, "TraceSramSourceV5 received an unexpected control event\n"); }
    if (control->source_id != coordinator_source_id_) { delete control; out_.fatal(CALL_INFO, -1, "TraceSramSourceV5 received a command for another source\n"); }
    if (control->operation == TraceControlOp::Start) {
        if (started_ || finished_) { delete control; out_.fatal(CALL_INFO, -1, "TraceSramSourceV5 received duplicate Start\n"); }
        started_ = true;
    } else {
        delete control;
        out_.fatal(CALL_INFO, -1, "TraceSramSourceV5 aborted by replay coordinator\n");
    }
    delete control;
}

void TraceSramSourceV5::handleResponse_(SST::Event* raw) {
    auto* response = dynamic_cast<SramResponseEvent*>(raw);
    if (!response) { delete raw; out_.fatal(CALL_INFO, -1, "TraceSramSourceV5 received an unexpected response\n"); }
    const auto found = outstanding_.find(response->request_id);
    if (found == outstanding_.end()) { delete response; out_.fatal(CALL_INFO, -1, "TraceSramSourceV5 received unknown response id=%" PRIu64 "\n", response->request_id); }
    InFlight flight = found->second;
    if (response->address != flight.record.address) { delete response; out_.fatal(CALL_INFO, -1, "TraceSramSourceV5 response address mismatch\n"); }
    if (!response->accepted) {
        if (!response->retryable) { delete response; out_.fatal(CALL_INFO, -1, "TraceSramSourceV5 SRAM request was permanently rejected\n"); }
        ++flight.retries;
        ++retries_;
        emit_(flight.record, "retry", 0, cycle_, flight.retries, response->bank, response->port, 0, 0, "retryable");
        outstanding_.erase(found);
        eligible_.push_front(flight.record);
        delete response;
        return;
    }
    ++injected_;
    bytes_ += flight.record.bytes;
    if (response->completed) {
        ++completed_;
        emit_(flight.record, "accepted", flight.record.request_id, cycle_, flight.retries,
              response->bank, response->port, response->service_cycle,
              response->completion_cycle);
        emit_(flight.record, "completed", flight.record.request_id, cycle_, flight.retries,
              response->bank, response->port, response->service_cycle,
              response->completion_cycle);
        outstanding_.erase(found);
    } else {
        emit_(flight.record, "accepted", flight.record.request_id, cycle_, flight.retries,
              response->bank, response->port, response->service_cycle,
              response->completion_cycle);
    }
    delete response;
}

bool TraceSramSourceV5::clockTick_(SST::Cycle_t) {
    if (finished_) return false;
    ++cycle_;
    if (external_control_) {
        if (!ready_sent_) { ready_sent_ = true; sendStatus_(TraceStatusOp::Ready); return false; }
        if (!started_) return false;
    }
    if (!have_next_ && !eof_) { have_next_ = readNext_(next_); eof_ = !have_next_; }
    while (have_next_ && next_.release_tick <= cycle_ && eligible_.size() < lookahead_limit_) {
        emit_(next_, "eligible", 0, cycle_, 0, next_.logical_bank);
        ++eligible_records_;
        eligible_.push_back(next_);
        have_next_ = readNext_(next_);
        eof_ = !have_next_;
    }
    std::uint64_t issued = 0;
    while (issued < issue_width_ && outstanding_.size() < max_outstanding_ && !eligible_.empty()) {
        Record record = eligible_.front();
        eligible_.pop_front();
        if (!record.offered_before) {
            ++offered_;
            record.offered_before = true;
            emit_(record, "offered", 0, cycle_, 0, record.logical_bank);
        }
        auto* request = new SramRequestEvent();
        request->request_id = record.request_id;
        request->address = record.address;
        request->data = record.data;
        request->write = record.write;
        outstanding_.emplace(record.request_id, InFlight{record, cycle_, 0});
        request_link_->send(request);
        emit_(record, "injected", record.request_id, cycle_, 0, record.logical_bank);
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

void TraceSramSourceV5::writeSummary_() const {
    if (output_json_.empty()) return;
    std::ofstream out(output_json_, std::ios::out | std::ios::trunc);
    out << "{\n  \"schema_version\": \"snndl-sram-replay-summary/v1\",\n"
        << "  \"mode\": \"" << execution_mode_ << "\",\n"
        << "  \"source_id\": " << source_id_ << ",\n"
        << "  \"expected_records\": " << records_ << ",\n"
        << "  \"eligible_records\": " << eligible_records_ << ",\n"
        << "  \"offered\": " << offered_ << ",\n"
        << "  \"injected\": " << injected_ << ",\n"
        << "  \"completed\": " << completed_ << ",\n"
        << "  \"retry_events\": " << retries_ << ",\n"
        << "  \"request_bytes\": " << bytes_ << ",\n"
        << "  \"outstanding_at_end\": " << outstanding_.size() << ",\n"
        << "  \"status\": \"" << ((finished_ && records_ == completed_) ? "PASS" : "INCOMPLETE") << "\"\n}\n";
}

void TraceSramSourceV5::finish() {
    if (observation_stream_.is_open()) observation_stream_.flush();
    writeSummary_();
    if (!finished_ || !eligible_.empty() || !outstanding_.empty() || records_ != completed_)
        out_.fatal(CALL_INFO, -1, "TraceSramSourceV5 finished before replay drained\n");
}

}}}
