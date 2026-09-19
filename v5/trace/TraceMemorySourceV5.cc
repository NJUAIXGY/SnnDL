#include <sst/core/sst_config.h>

#include "TraceMemorySourceV5.h"

#include "v5/trace/TraceJsonlReader.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cinttypes>
#include <limits>
#include <stdexcept>

namespace SST { namespace SnnDL { namespace v5 {

using json = nlohmann::json;

TraceMemorySourceV5::TraceMemorySourceV5(SST::ComponentId_t id, SST::Params& params)
    : Component(id), out_("SnnDL.TraceMemorySourceV5", 0, 0, Output::STDOUT),
      trace_file_(params.find<std::string>("trace_file", "")),
      output_json_(params.find<std::string>("output_json", "")),
      observation_json_(params.find<std::string>("observation_json", "")),
      execution_mode_(params.find<std::string>("execution_mode", "trace_open_loop")),
      source_id_(params.find<std::uint32_t>("source_id", 0)),
      coordinator_source_id_(params.find<std::uint32_t>("coordinator_source_id", source_id_)),
      external_control_(params.find<int>("external_control", 0) != 0),
      max_line_bytes_(normalizeTraceMaxLineBytes(params.find<std::uint64_t>("max_line_bytes", kTraceMaxLineBytesDefault))),
      issue_width_(std::max<std::uint64_t>(1, params.find<std::uint64_t>("issue_width_per_source", 1))),
      max_outstanding_(std::max<std::uint64_t>(1, params.find<std::uint64_t>("max_outstanding_per_source", 1))),
      lookahead_limit_(std::max<std::uint64_t>(1, params.find<std::uint64_t>("lookahead_records", 64))) {
    out_.setVerboseLevel(params.find<int>("verbose", 0));
    if (trace_file_.empty()) out_.fatal(CALL_INFO, -1, "TRACE-CONFIG: TraceMemorySourceV5 requires trace_file\n");
    trace_stream_.open(trace_file_);
    if (!trace_stream_.good()) out_.fatal(CALL_INFO, -1, "TRACE-IO: cannot open trace_file=%s\n", trace_file_.c_str());
    if (!observation_json_.empty()) {
        observation_stream_.open(observation_json_, std::ios::out | std::ios::trunc);
        if (!observation_stream_.good()) out_.fatal(CALL_INFO, -1, "TRACE-IO: cannot open observation_json=%s\n", observation_json_.c_str());
    }
    memory_ = loadUserSubComponent<SST::Interfaces::StandardMem>(
        "memory", ComponentInfo::SHARE_NONE, registerTimeBase("1ns"),
        new SST::Interfaces::StandardMem::Handler2<TraceMemorySourceV5, &TraceMemorySourceV5::handleMemory_>(this));
    if (!memory_) out_.fatal(CALL_INFO, -1, "TRACE-CONFIG: TraceMemorySourceV5 requires StandardMem slot 'memory'\n");
    if (external_control_) {
        command_link_ = configureLink(
            "command", new SST::Event::Handler2<TraceMemorySourceV5, &TraceMemorySourceV5::handleControl_>(this));
        status_link_ = configureLink("status");
        if (!command_link_ || !status_link_) {
            out_.fatal(CALL_INFO, -1, "TRACE-CONFIG: externally controlled TraceMemorySourceV5 requires command/status links\n");
        }
    }
    registerClock(params.find<std::string>("clock", "1GHz"),
                  new Clock::Handler2<TraceMemorySourceV5, &TraceMemorySourceV5::clockTick_>(this));
    if (!external_control_) {
        started_ = true;
        registerAsPrimaryComponent();
        primaryComponentDoNotEndSim();
    }
}

TraceMemorySourceV5::~TraceMemorySourceV5() {
    // Keep bounded observation shards readable even when SST tears down after
    // a failed simulation rather than reaching the normal acceptance path.
    if (observation_stream_.is_open()) observation_stream_.flush();
}
void TraceMemorySourceV5::init(unsigned int phase) { memory_->init(phase); }
void TraceMemorySourceV5::setup() { memory_->setup(); }

bool TraceMemorySourceV5::readNext_(Record& result) {
    nlohmann::json value;
    std::string parse_error;
    const auto status = reader_.next(trace_stream_, max_line_bytes_, value, parse_error);
    if (status == TraceJsonlReader::Status::End) return false;
    if (status == TraceJsonlReader::Status::Oversize)
        out_.fatal(CALL_INFO, -1, "TRACE-LIMIT: trace record exceeds max_line_bytes=%" PRIu64 "\n", max_line_bytes_);
    if (status == TraceJsonlReader::Status::ParseError)
        out_.fatal(CALL_INFO, -1, "TRACE-SCHEMA: invalid trace record: %s\n", parse_error.c_str());
    try {
        if (!value.is_object() || value.value("schema_version", std::string()) != "snndl-materialized-request/v1")
            out_.fatal(CALL_INFO, -1, "TRACE-SCHEMA: TraceMemorySourceV5 requires snndl-materialized-request/v1 records\n");
        if (value.value("operation", std::string()) != "read")
            out_.fatal(CALL_INFO, -1, "TRACE-SCHEMA: TRACE-1 only supports read operations\n");
        result.request_id = value.at("request_id").get<std::string>();
        result.access_id = value.at("access_id").get<std::string>();
        result.request_class = value.at("request_class").get<std::string>();
        result.region_id = value.at("region_id").get<std::string>();
        result.source_pe = value.at("source_pe").get<std::uint32_t>();
        result.source_core = value.at("source_core").get<std::uint32_t>();
        result.address = value.at("byte_address").get<std::uint64_t>();
        result.bytes = value.at("bytes").get<std::uint64_t>();
        result.release_tick = value.at("release_tick").get<std::uint64_t>();
        result.timestep = value.at("timestep").get<std::uint64_t>();
        result.stream_sequence = value.at("stream_sequence").get<std::uint64_t>();
        if (result.request_id.empty() || result.access_id.empty() || result.request_class.empty() ||
            result.region_id.empty() || result.bytes == 0 || result.source_core != source_id_)
            out_.fatal(CALL_INFO, -1, "TRACE-IDENTITY: invalid materialized request identity or size\n");
        if (!seen_request_ids_.insert(result.request_id).second)
            out_.fatal(CALL_INFO, -1, "TRACE-IDENTITY: duplicate materialized request_id=%s\n", result.request_id.c_str());
        if (have_sequence_ && result.stream_sequence <= last_stream_sequence_)
            out_.fatal(CALL_INFO, -1, "TRACE-ORDER: stream_sequence is not strictly increasing for source=%u\n", source_id_);
        have_sequence_ = true;
        last_stream_sequence_ = result.stream_sequence;
        ++records_;
        return true;
    } catch (const std::exception& error) {
        out_.fatal(CALL_INFO, -1, "TRACE-SCHEMA: invalid trace record: %s\n", error.what());
    }
    return false;
}

void TraceMemorySourceV5::emit_(const Record& record, const char* phase,
                                std::uint64_t runtime_id, std::uint64_t tick, std::uint64_t retry) {
    if (!observation_stream_.good()) return;
        observation_stream_ << json{{"schema_version", "snndl-replay-observation/v1"},
        {"request_id", record.request_id}, {"access_id", record.access_id},
        {"component", "trace_memory_source.core." + std::to_string(source_id_)},
        {"phase", phase}, {"tick", tick}, {"runtime_id", runtime_id},
        {"retry_count", retry}, {"status", "ok"}, {"source_id", source_id_},
        {"source_pe", record.source_pe}, {"source_core", record.source_core},
        {"timestep", record.timestep}, {"byte_address", record.address},
        {"bytes", record.bytes}, {"stream_sequence", record.stream_sequence},
        {"release_tick", record.release_tick}, {"region_id", record.region_id},
        {"request_class", record.request_class}}.dump() << '\n';
}

void TraceMemorySourceV5::sendStatus_(TraceStatusOp operation) {
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

void TraceMemorySourceV5::handleControl_(SST::Event* event) {
    auto* control = dynamic_cast<TraceControlEvent*>(event);
    if (!control) {
        delete event;
        out_.fatal(CALL_INFO, -1, "TRACE-CONTROL: TraceMemorySourceV5 received an unexpected control event\n");
    }
    if (control->source_id != coordinator_source_id_) {
        delete control;
        out_.fatal(CALL_INFO, -1, "TRACE-CONTROL: TraceMemorySourceV5 received a command for source=%u\n", control->source_id);
    }
    switch (control->operation) {
    case TraceControlOp::Start:
        if (started_ || finished_) {
            delete control;
            out_.fatal(CALL_INFO, -1, "TRACE-CONTROL: TraceMemorySourceV5 received duplicate Start\n");
        }
        started_ = true;
        break;
    case TraceControlOp::Abort:
        delete control;
        out_.fatal(CALL_INFO, -1, "TRACE-CONTROL: TraceMemorySourceV5 aborted by replay coordinator\n");
        break;
    }
    delete control;
}

bool TraceMemorySourceV5::clockTick_(SST::Cycle_t) {
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
        emit_(next_, "eligible", 0, cycle_);
        ++eligible_records_;
        eligible_.push_back(next_);
        have_next_ = readNext_(next_);
        eof_ = !have_next_;
    }
    std::uint64_t issued = 0;
    while (issued < issue_width_ && outstanding_.size() < max_outstanding_ && !eligible_.empty()) {
        Record record = eligible_.front();
        eligible_.pop_front();
        ++offered_;
        emit_(record, "offered", 0, cycle_);
        auto* request = new SST::Interfaces::StandardMem::Read(record.address, record.bytes);
        const auto runtime_id = static_cast<std::uint64_t>(request->getID());
        outstanding_.emplace(runtime_id, InFlight{record, runtime_id});
        memory_->send(request);
        ++injected_;
        bytes_ += record.bytes;
        emit_(record, "injected", runtime_id, cycle_);
        ++issued;
    }
    if (eof_ && !have_next_ && eligible_.empty() && outstanding_.empty() && !finished_) {
        finished_ = true;
        drained_sent_ = true;
        if (external_control_) {
            sendStatus_(TraceStatusOp::Drained);
        } else {
            primaryComponentOKToEndSim();
        }
    }
    return false;
}

void TraceMemorySourceV5::handleMemory_(SST::Interfaces::StandardMem::Request* request) {
    auto* response = dynamic_cast<SST::Interfaces::StandardMem::ReadResp*>(request);
    if (!response) {
        delete request;
        out_.fatal(CALL_INFO, -1, "TRACE-ADMISSION: TraceMemorySourceV5 received non-read response\n");
    }
    const auto runtime_id = static_cast<std::uint64_t>(response->getID());
    const auto found = outstanding_.find(runtime_id);
    if (found == outstanding_.end()) {
        delete response;
        out_.fatal(CALL_INFO, -1, "TRACE-IDENTITY: TraceMemorySourceV5 received unknown response id=%" PRIu64 "\n", runtime_id);
    }
    const auto record = found->second.record;
    if (response->pAddr != record.address || response->size != record.bytes) {
        delete response;
        out_.fatal(CALL_INFO, -1, "TRACE-IDENTITY: TraceMemorySourceV5 response address/size mismatch\n");
    }
    outstanding_.erase(found);
    ++completed_;
    emit_(record, "completed", runtime_id, cycle_);
    delete response;
}

void TraceMemorySourceV5::writeSummary_() const {
    if (output_json_.empty()) return;
    std::ofstream out(output_json_, std::ios::out | std::ios::trunc);
    out << "{\n  \"schema_version\": \"snndl-trace-replay-summary/v1\",\n"
        << "  \"mode\": \"" << execution_mode_ << "\",\n"
        << "  \"source_id\": " << source_id_ << ",\n"
        << "  \"expected_records\": " << records_ << ",\n"
        << "  \"eligible_records\": " << eligible_records_ << ",\n"
        << "  \"offered\": " << offered_ << ",\n"
        << "  \"injected\": " << injected_ << ",\n"
        << "  \"completed\": " << completed_ << ",\n"
        << "  \"request_bytes\": " << bytes_ << ",\n"
        << "  \"outstanding_at_end\": " << outstanding_.size() << ",\n"
        << "  \"status\": \"" << ((finished_ && records_ == injected_ && completed_ == injected_) ? "PASS" : "INCOMPLETE") << "\"\n}\n";
}

void TraceMemorySourceV5::finish() {
    if (observation_stream_.is_open()) observation_stream_.flush();
    writeSummary_();
    if (!finished_ || !eligible_.empty() || !outstanding_.empty() || records_ != injected_ || completed_ != injected_)
        out_.fatal(CALL_INFO, -1, "TRACE-DRAIN: TraceMemorySourceV5 finished before replay drained\n");
}

}}}
