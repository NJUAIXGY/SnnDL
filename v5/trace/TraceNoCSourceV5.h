#ifndef SST_SNN_DL_V5_TRACE_NOC_SOURCE_V5_H
#define SST_SNN_DL_V5_TRACE_NOC_SOURCE_V5_H

#include "v5/events/TraceEvents.h"
#include "v5/trace/TraceJsonlReader.h"

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

class TraceNoCSourceV5 final : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(TraceNoCSourceV5, "SnnDL", "TraceNoCSourceV5",
        SST_ELI_ELEMENT_VERSION(1,0,0), "TRACE-4 logical NoC event source", COMPONENT_CATEGORY_PROCESSOR)
    SST_ELI_DOCUMENT_PARAMS(
        {"trace_file", "Logical NoC JSONL shard", ""},
        {"source_pe", "Source PE owner", "0"},
        {"source_core", "Source Core owner", "0"},
        {"multicast_mode", "unicast, source_replication, or native_tree", "source_replication"},
        {"output_json", "NoC source evidence", ""},
        {"observation_json", "Append-only source observation JSONL", ""},
        {"source_id", "Global replay source ordinal", "0"},
        {"coordinator_source_id", "Global ordinal used by a shared replay coordinator (defaults to source_id)", "0"},
        {"external_control", "Use TraceReplayCoordinatorV5 lifetime control", "0"},
        {"issue_width_per_source", "Maximum logical events offered per source tick", "1"},
        {"max_outstanding_per_source", "Maximum events awaiting endpoint admission ACK", "1"},
        {"lookahead_records", "Bounded records held ahead of the issue cursor", "64"},
        // The default below is a string literal only for the ELI table; it mirrors
        // kTraceMaxLineBytesDefault in v5/trace/TraceJsonlReader.h, which is the value
        // actually used when the parameter is absent.
        {"max_line_bytes", "Maximum accepted JSONL record length", "1048576"},
        {"clock", "Source clock", "1GHz"},
        {"verbose", "Verbose level", "0"})
    SST_ELI_DOCUMENT_PORTS(
        {"inject", "Logical event injection into the local PE endpoint", {"SnnDL.TraceNoCInjectionV5Event"}},
        {"ack", "Endpoint atomic admission ACK", {"SnnDL.TraceNoCInjectionAckV5Event"}},
        {"command", "Direct replay start/abort command", {"SnnDL.TraceControlEvent"}},
        {"status", "Direct replay readiness/drain status", {"SnnDL.TraceStatusEvent"}})

    TraceNoCSourceV5(SST::ComponentId_t, SST::Params&);
    ~TraceNoCSourceV5() override;
    void finish() override;

private:
    struct Record {
        std::string event_id;
        std::uint64_t token = 0, timestep = 0, source_neuron = 0, source_event_seq = 0;
        // payload_bytes has no component-side default: the record must carry it
        // because the traffic-class width is a contract-layer policy, not a
        // property of this source.
        std::uint32_t source_pe = 0, source_core = 0, payload_bytes = 0, virtual_network = 0;
        std::uint64_t route_id = 0, release_tick = 0, stream_sequence = 0;
        TraceNoCMulticastMode mode = TraceNoCMulticastMode::SourceReplication;
        std::vector<std::uint32_t> destination_pes;
        std::vector<std::uint64_t> destination_masks;
        std::vector<std::string> depends_on;
        bool offered_before = false;
    };
    struct InFlight {
        Record record;
        std::uint64_t issue_cycle = 0;
        std::uint64_t retries = 0;
    };

    bool tick_(SST::Cycle_t);
    void handleAck_(SST::Event*);
    void handleControl_(SST::Event*);
    bool readNext_(Record&);
    TraceNoCInjectionV5Event* makeEvent_(const Record&) const;
    void emit_(const Record&, const char*, std::uint64_t, std::uint64_t, std::uint64_t,
               std::uint32_t = 0);
    void sendStatus_(TraceStatusOp);
    bool dependenciesReady_(const Record&) const;
    void writeEvidence_() const;

    SST::Output out_;
    SST::Link* inject_ = nullptr;
    SST::Link* ack_ = nullptr;
    SST::Link* command_ = nullptr;
    SST::Link* status_ = nullptr;
    std::ifstream trace_stream_;
    TraceJsonlReader reader_;
    std::ofstream observation_stream_;
    std::string trace_file_, output_json_, observation_json_, multicast_mode_;
    std::uint32_t source_pe_ = 0, source_core_ = 0, source_id_ = 0;
    std::uint32_t coordinator_source_id_ = 0;
    bool external_control_ = false;
    std::uint64_t max_line_bytes_ = kTraceMaxLineBytesDefault;
    std::uint64_t issue_width_ = 1, max_outstanding_ = 1, lookahead_limit_ = 64;
    std::uint64_t cycle_ = 0, records_ = 0, eligible_records_ = 0, offered_ = 0;
    std::uint64_t injected_ = 0, completed_ = 0, retries_ = 0, physical_packets_ = 0;
    std::uint64_t last_sequence_ = 0;
    std::vector<std::string> accepted_event_ids_;
    std::set<std::string> completed_event_ids_;
    std::set<std::string> seen_event_ids_;
    bool have_sequence_ = false, eof_ = false, have_next_ = false, finished_ = false;
    bool ready_sent_ = false, started_ = false, drained_sent_ = false;
    Record next_;
    std::map<std::uint64_t, InFlight> outstanding_;
    std::deque<Record> eligible_;
};

}}}
#endif
