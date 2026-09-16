#ifndef SST_SNN_DL_V5_TRACE_SRAM_SOURCE_V5_H
#define SST_SNN_DL_V5_TRACE_SRAM_SOURCE_V5_H

#include "v5/events/StorageEvents.h"
#include "v5/events/TraceEvents.h"

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>

#include <cstdint>
#include <deque>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace SST { namespace SnnDL { namespace v5 {

class TraceSramSourceV5 final : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(TraceSramSourceV5, "SnnDL", "TraceSramSourceV5",
        SST_ELI_ELEMENT_VERSION(1,0,0), "TRACE open-loop banked SRAM request replay", COMPONENT_CATEGORY_PROCESSOR)
    SST_ELI_DOCUMENT_PARAMS(
        {"trace_file", "Materialized SRAM request JSONL shard", ""},
        {"output_json", "SRAM replay summary JSON", ""},
        {"observation_json", "Append-only source observation JSONL", ""},
        {"execution_mode", "trace_open_loop or trace_dependency_closed_loop", "trace_open_loop"},
        {"source_id", "Stable source ordinal", "0"},
        {"coordinator_source_id", "Global ordinal used by a shared replay coordinator (defaults to source_id)", "0"},
        {"mixed_source_stream", "Allow one bounded stream to carry records from multiple Core owners", "0"},
        {"external_control", "Use TraceReplayCoordinatorV5 lifetime control", "0"},
        {"issue_width_per_source", "Maximum requests offered per source tick", "1"},
        {"max_outstanding_per_source", "Maximum requests awaiting SRAM response", "1"},
        {"lookahead_records", "Bounded records held ahead of the issue cursor", "64"},
        {"max_line_bytes", "Maximum accepted JSONL record length", "1048576"},
        {"clock", "Replay source clock", "1GHz"},
        {"verbose", "Verbose level", "0"})
    SST_ELI_DOCUMENT_PORTS(
        {"request", "SRAM request output", {"SnnDL.SramRequestEvent"}},
        {"response", "SRAM completion or retry response input", {"SnnDL.SramResponseEvent"}},
        {"command", "Direct replay start/abort command", {"SnnDL.TraceControlEvent"}},
        {"status", "Direct replay readiness/drain status", {"SnnDL.TraceStatusEvent"}})

    TraceSramSourceV5(SST::ComponentId_t, SST::Params&);
    ~TraceSramSourceV5() override;
    void finish() override;

private:
    struct Record {
        std::uint64_t request_id = 0;
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
        std::uint32_t logical_bank = 0;
        bool write = false;
        std::vector<std::uint8_t> data;
        bool offered_before = false;
    };
    struct InFlight {
        Record record;
        std::uint64_t issue_cycle = 0;
        std::uint64_t retries = 0;
    };

    bool clockTick_(SST::Cycle_t);
    void handleResponse_(SST::Event*);
    void handleControl_(SST::Event*);
    bool readNext_(Record&);
    void emit_(const Record&, const char*, std::uint64_t runtime_id,
               std::uint64_t tick, std::uint64_t retries,
               std::uint32_t bank = 0, std::uint32_t port = 0,
               std::uint64_t service = 0, std::uint64_t completion = 0,
               const char* status = "ok");
    void sendStatus_(TraceStatusOp);
    void writeSummary_() const;

    SST::Output out_;
    SST::Link* request_link_ = nullptr;
    SST::Link* response_link_ = nullptr;
    SST::Link* command_link_ = nullptr;
    SST::Link* status_link_ = nullptr;
    std::ifstream trace_stream_;
    std::ofstream observation_stream_;
    std::string trace_file_, output_json_, observation_json_, execution_mode_ = "trace_open_loop";
    std::uint32_t source_id_ = 0;
    std::uint32_t coordinator_source_id_ = 0;
    bool mixed_source_stream_ = false;
    bool external_control_ = false;
    std::uint64_t max_line_bytes_ = 1024 * 1024;
    std::uint64_t issue_width_ = 1, max_outstanding_ = 1, lookahead_limit_ = 64;
    std::uint64_t cycle_ = 0;
    std::uint64_t records_ = 0, eligible_records_ = 0, offered_ = 0, injected_ = 0;
    std::uint64_t completed_ = 0, bytes_ = 0, retries_ = 0;
    bool eof_ = false, have_next_ = false, finished_ = false, ready_sent_ = false;
    bool started_ = false, have_sequence_ = false, drained_sent_ = false;
    std::uint64_t last_stream_sequence_ = 0;
    Record next_;
    std::set<std::uint64_t> seen_request_ids_;
    std::map<std::uint64_t, InFlight> outstanding_;
    std::deque<Record> eligible_;
};

}}}

#endif
