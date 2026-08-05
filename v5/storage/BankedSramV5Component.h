#ifndef SST_SNN_DL_V5_BANKED_SRAM_V5_COMPONENT_H
#define SST_SNN_DL_V5_BANKED_SRAM_V5_COMPONENT_H

#include "BankedSramV5.h"
#include "v5/events/StorageEvents.h"

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>

#include <cstdint>
#include <map>
#include <string>

namespace SST {
namespace SnnDL {
namespace v5 {

class BankedSramV5Component final : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(
        BankedSramV5Component,
        "SnnDL",
        "BankedSramV5",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "P2 request-driven banked SRAM with finite queues and backing",
        COMPONENT_CATEGORY_MEMORY
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"capacity_bytes", "Resident SRAM capacity", "4096"},
        {"banks", "Number of independent banks", "1"},
        {"ports_per_bank", "Service ports per bank", "1"},
        {"interleave_bytes", "Low-order bank interleave", "4"},
        {"read_latency_cycles", "Read latency after bank service", "1"},
        {"write_latency_cycles", "Write latency after bank service", "1"},
        {"request_queue_entries", "Finite accepted request capacity", "16"},
        {"response_queue_entries", "Finite completion queue capacity", "16"},
        {"stats_json", "Optional raw SRAM statistics output path", ""},
        {"clock", "SRAM clock", "1GHz"},
        {"verbose", "Verbose logging level", "0"}
    )

    SST_ELI_DOCUMENT_STATISTICS(
        {"storage.sram.requests.accepted", "Accepted SRAM requests", "requests", 1},
        {"storage.sram.requests.retryable_rejects", "Requests rejected by finite queues", "requests", 1},
        {"storage.sram.requests.capacity_rejects", "Requests rejected by capacity bounds", "requests", 1},
        {"storage.sram.requests.issued", "Requests assigned to a bank port", "requests", 1},
        {"storage.sram.requests.completed", "Completed SRAM requests", "requests", 1},
        {"storage.sram.reads", "Completed SRAM reads", "requests", 1},
        {"storage.sram.writes", "Completed SRAM writes", "requests", 1},
        {"storage.sram.bytes_read", "Bytes returned by SRAM reads", "bytes", 1},
        {"storage.sram.bytes_written", "Bytes committed by SRAM writes", "bytes", 1},
        {"storage.sram.queue_occupancy", "Peak SRAM request occupancy", "requests", 1},
        {"storage.sram.response_occupancy", "Peak SRAM response occupancy", "requests", 1},
        {"storage.sram.bank_conflicts", "Cycles with bank oversubscription", "cycles", 1},
        {"storage.sram.port_stall_cycles", "Cycles with no free bank port", "cycles", 1},
        {"storage.sram.busy_cycles", "Bank busy cycles", "cycles", 1},
        {"storage.sram.latency_cycles", "Aggregate enqueue-to-completion latency", "cycles", 1},
        {"storage.sram.capacity_bytes", "Configured resident capacity", "bytes", 1},
        {"storage.sram.resident_bytes", "Resident backing bytes", "bytes", 1}
    )

    SST_ELI_DOCUMENT_PORTS(
        {"request", "Finite request input", {"SnnDL.SramRequestEvent"}},
        {"response", "Completion or retry response output", {"SnnDL.SramResponseEvent"}}
    )

    BankedSramV5Component(SST::ComponentId_t id, SST::Params& params);
    ~BankedSramV5Component() override;

    void init(unsigned int phase) override;
    void setup() override;
    void finish() override;

private:
    void handleRequest_(SST::Event* event);
    bool clockTick_(SST::Cycle_t cycle);
    void sendResponse_(const BankedSramV5Response& response);
    void publishStatistics_();

    SST::Output out_;
    SST::Link* request_link_ = nullptr;
    SST::Link* response_link_ = nullptr;
    BankedSramV5 model_;
    std::uint64_t current_cycle_ = 0;
    std::string stats_json_;
    std::map<std::string, Statistic<std::uint64_t>*> statistics_;
};

} // namespace v5
} // namespace SnnDL
} // namespace SST

#endif
