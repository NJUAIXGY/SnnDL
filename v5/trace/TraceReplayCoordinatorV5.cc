#include <sst/core/sst_config.h>

#include "TraceReplayCoordinatorV5.h"

#include <algorithm>
#include <cinttypes>
#include <fstream>

namespace SST { namespace SnnDL { namespace v5 {

TraceReplayCoordinatorV5::TraceReplayCoordinatorV5(SST::ComponentId_t id, SST::Params& params)
    : Component(id), out_("SnnDL.TraceReplayCoordinatorV5", 0, 0, Output::STDOUT),
      sources_(std::max<std::uint32_t>(1, params.find<std::uint32_t>("sources", 1))),
      expected_records_(params.find<std::uint64_t>("expected_records", 0)),
      timeout_cycles_(std::max<std::uint64_t>(1, params.find<std::uint64_t>("timeout_cycles", 1000000))),
      execution_mode_(params.find<std::string>("execution_mode", "trace_open_loop")),
      output_json_(params.find<std::string>("output_json", "")) {
    out_.setVerboseLevel(params.find<int>("verbose", 0));
    if (execution_mode_ != "trace_open_loop" && execution_mode_ != "trace_dependency_closed_loop") {
        out_.fatal(CALL_INFO, -1, "TRACE-CONFIG: TraceReplayCoordinatorV5 received unsupported execution_mode=%s\n",
                   execution_mode_.c_str());
    }
    command_links_.reserve(sources_);
    status_links_.reserve(sources_);
    sources_state_.resize(sources_);
    for (std::uint32_t source = 0; source < sources_; ++source) {
        auto* command = configureLink("command_source" + std::to_string(source));
        auto* status = configureLink(
            "status_source" + std::to_string(source),
            new SST::Event::Handler2<TraceReplayCoordinatorV5, &TraceReplayCoordinatorV5::handleStatus_>(this));
        if (!command || !status) {
            out_.fatal(CALL_INFO, -1, "TRACE-CONFIG: TraceReplayCoordinatorV5 requires command/status links for source=%u\n", source);
        }
        command_links_.push_back(command);
        status_links_.push_back(status);
    }
    registerClock(params.find<std::string>("clock", "1GHz"),
                  new Clock::Handler2<TraceReplayCoordinatorV5, &TraceReplayCoordinatorV5::tick_>(this));
    registerAsPrimaryComponent();
    primaryComponentDoNotEndSim();
}

void TraceReplayCoordinatorV5::sendStart_() {
    if (start_sent_) return;
    start_sent_ = true;
    last_progress_cycle_ = cycle_;
    for (std::uint32_t source = 0; source < sources_; ++source) {
        command_links_[source]->send(new TraceControlEvent(TraceControlOp::Start, source, 0));
    }
}

void TraceReplayCoordinatorV5::sendAbort_() {
    for (std::uint32_t source = 0; source < sources_; ++source) {
        command_links_[source]->send(new TraceControlEvent(TraceControlOp::Abort, source, 0));
    }
}

void TraceReplayCoordinatorV5::handleStatus_(SST::Event* event) {
    auto* status = dynamic_cast<TraceStatusEvent*>(event);
    if (!status) {
        delete event;
        out_.fatal(CALL_INFO, -1, "TRACE-CONTROL: TraceReplayCoordinatorV5 received an unexpected status event\n");
    }
    if (status->source_id >= sources_) {
        delete status;
        out_.fatal(CALL_INFO, -1, "TRACE-IDENTITY: TraceReplayCoordinatorV5 received status from invalid source=%u\n",
                   status->source_id);
    }
    auto& state = sources_state_[status->source_id];
    last_progress_cycle_ = cycle_;
    switch (status->operation) {
    case TraceStatusOp::Ready:
        if (state.ready) {
            delete status;
            out_.fatal(CALL_INFO, -1, "TRACE-CONTROL: duplicate Ready status from source=%u\n", status->source_id);
        }
        if (start_sent_) {
            delete status;
            out_.fatal(CALL_INFO, -1, "TRACE-CONTROL: Ready status arrived after replay start\n");
        }
        state.ready = true;
        state.status = *status;
        if (std::all_of(sources_state_.begin(), sources_state_.end(),
                        [](const SourceState& item) { return item.ready; })) {
            sendStart_();
        }
        break;
    case TraceStatusOp::Drained:
        if (!start_sent_ || !state.ready || state.drained) {
            delete status;
            out_.fatal(CALL_INFO, -1, "TRACE-CONTROL: invalid Drained status from source=%u\n", status->source_id);
        }
        state.drained = true;
        state.status = *status;
        if (std::all_of(sources_state_.begin(), sources_state_.end(),
                        [](const SourceState& item) { return item.drained; })) {
            complete_();
        }
        break;
    case TraceStatusOp::Failed:
        {
            const auto source = status->source_id;
            const auto error = status->error;
            delete status;
            out_.fatal(CALL_INFO, -1, "TRACE-CONTROL: Trace replay source=%u failed: %s\n", source, error.c_str());
        }
        break;
    }
    delete status;
}

void TraceReplayCoordinatorV5::complete_() {
    if (finished_) return;
    std::uint64_t expected = 0;
    std::uint64_t offered = 0;
    std::uint64_t injected = 0;
    std::uint64_t completed = 0;
    for (const auto& state : sources_state_) {
        expected += state.status.expected_records;
        offered += state.status.offered;
        injected += state.status.injected;
        completed += state.status.completed;
    }
    const auto manifest_match = expected_records_ == 0 || expected_records_ == expected;
    if (!manifest_match || expected != offered || offered != injected || injected != completed) {
        out_.fatal(CALL_INFO, -1,
                   "TRACE-DRAIN: Trace replay request conservation failed expected=%" PRIu64
                   " offered=%" PRIu64 " injected=%" PRIu64 " completed=%" PRIu64 "\n",
                   expected, offered, injected, completed);
    }
    finished_ = true;
    writeSummary_();
    primaryComponentOKToEndSim();
}

bool TraceReplayCoordinatorV5::tick_(SST::Cycle_t) {
    ++cycle_;
    if (!finished_ && cycle_ - last_progress_cycle_ > timeout_cycles_) {
        sendAbort_();
        writeSummary_();
        out_.fatal(CALL_INFO, -1, "TRACE-DRAIN: TraceReplayCoordinatorV5 timeout after %" PRIu64 " cycles\n", timeout_cycles_);
    }
    return false;
}

void TraceReplayCoordinatorV5::writeSummary_() const {
    if (output_json_.empty()) return;
    std::ofstream out(output_json_, std::ios::out | std::ios::trunc);
    if (!out.good()) return;
    std::uint64_t expected = 0, offered = 0, injected = 0, completed = 0;
    for (const auto& state : sources_state_) {
        expected += state.status.expected_records;
        offered += state.status.offered;
        injected += state.status.injected;
        completed += state.status.completed;
    }
    const bool source_drain = std::all_of(sources_state_.begin(), sources_state_.end(),
                                          [](const SourceState& state) { return state.drained; });
    const bool conservation = expected == offered && offered == injected && injected == completed &&
                              (expected_records_ == 0 || expected_records_ == expected);
    out << "{\n"
        << "  \"schema_version\": \"snndl-trace-coordinator-summary/v1\",\n"
        << "  \"execution_mode\": \"" << execution_mode_ << "\",\n"
        << "  \"sources\": " << sources_ << ",\n"
        << "  \"expected_records\": " << expected << ",\n"
        << "  \"offered_records\": " << offered << ",\n"
        << "  \"injected_records\": " << injected << ",\n"
        << "  \"completed_records\": " << completed << ",\n"
        << "  \"source_drain\": " << (source_drain ? "true" : "false") << ",\n"
        << "  \"coordinator_drain\": " << (finished_ ? "true" : "false") << ",\n"
        << "  \"request_conservation\": " << (conservation ? "true" : "false") << ",\n"
        << "  \"cycle\": " << cycle_ << ",\n"
        << "  \"status\": \"" << (finished_ && source_drain && conservation ? "PASS" : "INCOMPLETE") << "\",\n"
        << "  \"per_source\": [\n";
    for (std::uint32_t source = 0; source < sources_; ++source) {
        const auto& state = sources_state_[source];
        out << "    {\"source_id\": " << source
            << ", \"ready\": " << (state.ready ? "true" : "false")
            << ", \"drained\": " << (state.drained ? "true" : "false")
            << ", \"expected_records\": " << state.status.expected_records
            << ", \"offered\": " << state.status.offered
            << ", \"injected\": " << state.status.injected
            << ", \"completed\": " << state.status.completed
            << ", \"outstanding\": " << state.status.outstanding << "}";
        if (source + 1 != sources_) out << ",";
        out << "\n";
    }
    out << "  ]\n}\n";
}

void TraceReplayCoordinatorV5::finish() {
    if (!finished_) writeSummary_();
}

}}}
