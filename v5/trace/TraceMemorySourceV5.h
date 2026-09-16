#ifndef SST_SNN_DL_V5_TRACE_MEMORY_SOURCE_V5_H
#define SST_SNN_DL_V5_TRACE_MEMORY_SOURCE_V5_H

#include "v5/events/TraceEvents.h"

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/interfaces/stdMem.h>
#include <sst/core/output.h>

#include <cstdint>
#include <deque>
#include <fstream>
#include <map>
#include <set>
#include <string>

namespace SST { namespace SnnDL { namespace v5 {

class TraceMemorySourceV5 final : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(TraceMemorySourceV5, "SnnDL", "TraceMemorySourceV5",
        SST_ELI_ELEMENT_VERSION(1,0,0), "TRACE-1 open-loop StandardMem read replay", COMPONENT_CATEGORY_PROCESSOR)
    SST_ELI_DOCUMENT_PARAMS(
        {"trace_file", "Materialized request JSONL shard", ""},
        {"output_json", "Replay summary JSON", ""},
        {"observation_json", "Append-only source observation JSONL", ""},
        {"execution_mode", "trace_open_loop or trace_dependency_closed_loop", "trace_open_loop"},
        {"source_id", "Stable ordinal used by the replay coordinator", "0"},
        {"coordinator_source_id", "Global ordinal used by a shared replay coordinator (defaults to source_id)", "0"},
        {"external_control", "Use TraceReplayCoordinatorV5 for lifetime control", "0"},
        {"issue_width_per_source", "Maximum reads offered per source tick", "1"},
        {"max_outstanding_per_source", "Maximum injected reads awaiting response", "1"},
        {"lookahead_records", "Bounded records held ahead of the issue cursor", "64"},
        {"max_line_bytes", "Maximum accepted JSONL record length", "1048576"},
        {"clock", "Replay source clock", "1GHz"},
        {"verbose", "Verbose level", "0"})
    SST_ELI_DOCUMENT_PORTS(
        {"command", "Direct replay start/abort command", {"SnnDL.TraceControlEvent"}},
        {"status", "Direct replay readiness/drain status", {"SnnDL.TraceStatusEvent"}})
    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS(
        {"memory", "StandardMem client toward private L1", "SST::Interfaces::StandardMem"})

    TraceMemorySourceV5(SST::ComponentId_t, SST::Params&);
    ~TraceMemorySourceV5() override;
    void init(unsigned int phase) override;
    void setup() override;
    void finish() override;

private:
    struct Record {
        std::string request_id;
        std::string access_id;
        std::string request_class;
        std::string region_id;
        std::uint32_t source_pe = 0;
        std::uint32_t source_core = 0;
        std::uint64_t address = 0;
        std::uint64_t bytes = 0;
        std::uint64_t release_tick = 0;
        std::uint64_t timestep = 0;
        std::uint64_t stream_sequence = 0;
    };
    struct InFlight { Record record; std::uint64_t runtime_id = 0; };

    bool clockTick_(SST::Cycle_t);
    void handleMemory_(SST::Interfaces::StandardMem::Request*);
    void handleControl_(SST::Event*);
    bool readNext_(Record&);
    void emit_(const Record&, const char*, std::uint64_t, std::uint64_t, std::uint64_t = 0);
    void sendStatus_(TraceStatusOp);
    void writeSummary_() const;

    SST::Output out_;
    SST::Interfaces::StandardMem* memory_ = nullptr;
    SST::Link* command_link_ = nullptr;
    SST::Link* status_link_ = nullptr;
    std::ifstream trace_stream_;
    std::ofstream observation_stream_;
    std::string trace_file_, output_json_, observation_json_, execution_mode_ = "trace_open_loop";
    std::uint32_t source_id_ = 0;
    std::uint32_t coordinator_source_id_ = 0;
    bool external_control_ = false;
    std::uint64_t max_line_bytes_ = 1024 * 1024;
    std::uint64_t issue_width_ = 1, max_outstanding_ = 1, lookahead_limit_ = 64, cycle_ = 0;
    std::uint64_t records_ = 0, eligible_records_ = 0, offered_ = 0, injected_ = 0;
    std::uint64_t completed_ = 0, bytes_ = 0;
    bool eof_ = false, have_next_ = false, finished_ = false, ready_sent_ = false;
    bool started_ = false, drained_sent_ = false, have_sequence_ = false;
    std::uint64_t last_stream_sequence_ = 0;
    Record next_;
    std::set<std::string> seen_request_ids_;
    std::map<std::uint64_t, InFlight> outstanding_;
    std::deque<Record> eligible_;
};

}}}

#endif
