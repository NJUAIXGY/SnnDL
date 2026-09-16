#ifndef SST_SNN_DL_V5_TRACE_NOC_SINK_V5_H
#define SST_SNN_DL_V5_TRACE_NOC_SINK_V5_H

#include "v5/events/TraceEvents.h"

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>

#include <cstdint>
#include <deque>
#include <fstream>
#include <string>
#include <vector>

namespace SST { namespace SnnDL { namespace v5 {

class TraceNoCSinkV5 final : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(TraceNoCSinkV5, "SnnDL", "TraceNoCSinkV5",
        SST_ELI_ELEMENT_VERSION(1,0,0), "TRACE-4 Core-facing logical NoC sink", COMPONENT_CATEGORY_PROCESSOR)
    SST_ELI_DOCUMENT_PARAMS(
        {"destination_pe", "Destination PE owner", "0"},
        {"destination_core", "Destination Core owner", "0"},
        {"queue_entries", "Finite sink queue", "16"},
        {"service_latency_cycles", "Cycles before delivery ACK", "1"},
        {"drain_grace_cycles", "Cycles to wait after final ACK for endpoint return", "2"},
        {"expected_deliveries", "Expected destination Core deliveries", "0"},
        {"output_json", "NoC sink evidence", ""},
        {"clock", "Sink clock", "1GHz"},
        {"verbose", "Verbose level", "0"})
    SST_ELI_DOCUMENT_PORTS(
        {"delivery", "Core-facing logical delivery input", {"SnnDL.TraceNoCDeliveryV5Event"}},
        {"ack", "Core-facing delivery ACK", {"SnnDL.TraceNoCDeliveryAckV5Event"}})

    TraceNoCSinkV5(SST::ComponentId_t, SST::Params&);
    ~TraceNoCSinkV5() override;
    void finish() override;

private:
    struct Pending { TraceNoCDeliveryV5Event* event = nullptr; std::uint64_t ready_cycle = 0; };
    void handleDelivery_(SST::Event*);
    bool tick_(SST::Cycle_t);
    void writeEvidence_() const;

    SST::Output out_;
    SST::Link* delivery_ = nullptr;
    SST::Link* ack_ = nullptr;
    std::string output_json_;
    std::uint32_t destination_pe_ = 0, destination_core_ = 0, queue_entries_ = 16;
    std::uint64_t service_latency_ = 1, drain_grace_cycles_ = 2, expected_deliveries_ = 0, cycle_ = 0, completed_ = 0, retries_ = 0;
    std::uint64_t drain_ready_cycle_ = 0;
    std::uint64_t peak_queue_ = 0;
    std::vector<std::string> completed_event_ids_;
    bool finished_ = false;
    std::deque<Pending> queue_;
};

}}}
#endif
