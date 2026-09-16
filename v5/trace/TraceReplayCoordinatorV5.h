#ifndef SST_SNN_DL_V5_TRACE_REPLAY_COORDINATOR_V5_H
#define SST_SNN_DL_V5_TRACE_REPLAY_COORDINATOR_V5_H

#include "v5/events/TraceEvents.h"

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>

#include <cstdint>
#include <string>
#include <vector>

namespace SST { namespace SnnDL { namespace v5 {

class TraceReplayCoordinatorV5 final : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(TraceReplayCoordinatorV5, "SnnDL", "TraceReplayCoordinatorV5",
        SST_ELI_ELEMENT_VERSION(1,0,0), "TRACE-1 global replay lifetime coordinator", COMPONENT_CATEGORY_PROCESSOR)
    SST_ELI_DOCUMENT_PARAMS(
        {"sources", "Number of TraceMemorySourceV5 participants", "1"},
        {"expected_records", "Manifest record count (zero derives from source status)", "0"},
        {"timeout_cycles", "Maximum cycles without replay progress", "1000000"},
        {"execution_mode", "trace_open_loop or trace_dependency_closed_loop", "trace_open_loop"},
        {"output_json", "Coordinator summary JSON", ""},
        {"clock", "Coordinator clock", "1GHz"},
        {"verbose", "Verbose level", "0"})
    SST_ELI_DOCUMENT_PORTS(
        {"command_source%(sources)d", "Direct start/abort command to each source", {"SnnDL.TraceControlEvent"}},
        {"status_source%(sources)d", "Direct readiness/drain status from each source", {"SnnDL.TraceStatusEvent"}})

    TraceReplayCoordinatorV5(SST::ComponentId_t, SST::Params&);
    ~TraceReplayCoordinatorV5() override = default;
    void finish() override;

private:
    struct SourceState {
        bool ready = false;
        bool drained = false;
        TraceStatusEvent status;
    };

    void handleStatus_(SST::Event*);
    bool tick_(SST::Cycle_t);
    void sendStart_();
    void sendAbort_();
    void complete_();
    void writeSummary_() const;

    SST::Output out_;
    std::vector<SST::Link*> command_links_;
    std::vector<SST::Link*> status_links_;
    std::vector<SourceState> sources_state_;
    std::uint32_t sources_ = 1;
    std::uint64_t expected_records_ = 0;
    std::uint64_t timeout_cycles_ = 1000000;
    std::string execution_mode_ = "trace_open_loop";
    std::uint64_t cycle_ = 0;
    std::uint64_t last_progress_cycle_ = 0;
    bool start_sent_ = false;
    bool finished_ = false;
    std::string output_json_;
};

}}}

#endif
