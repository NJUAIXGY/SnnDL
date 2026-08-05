#ifndef SST_SNN_DL_V5_DMA_ENGINE_H
#define SST_SNN_DL_V5_DMA_ENGINE_H

#include "v5/events/StorageEvents.h"

#include <sst/core/component.h>
#include <sst/core/interfaces/stdMem.h>
#include <sst/core/link.h>
#include <sst/core/output.h>

#include <cstdint>
#include <deque>
#include <map>
#include <string>

namespace SST {
namespace SnnDL {
namespace v5 {

class SnnDmaEngineV5 final : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(
        SnnDmaEngineV5,
        "SnnDL",
        "SnnDmaEngineV5",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "P3 finite PE DMA engine using StandardMem MoveData",
        COMPONENT_CATEGORY_MEMORY
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"channels", "Parallel DMA issue channels", "1"},
        {"queue_entries", "Finite descriptor queue capacity", "16"},
        {"max_outstanding", "Finite outstanding burst capacity", "8"},
        {"burst_bytes", "Default burst size", "64"},
        {"bytes_per_cycle", "Configured byte issue budget for evidence", "16"},
        {"setup_cycles", "Descriptor setup delay", "1"},
        {"expected_descriptors", "Primary completion count; zero disables primary ownership", "0"},
        {"clock", "DMA clock", "1GHz"},
        {"stats_json", "Optional raw DMA statistics output path", ""},
        {"verbose", "Verbose logging level", "0"}
    )

    SST_ELI_DOCUMENT_STATISTICS(
        {"dma.descriptors.accepted", "Accepted DMA descriptors", "descriptors", 1},
        {"dma.descriptors.completed", "Completed DMA descriptors", "descriptors", 1},
        {"dma.descriptors.retryable_rejects", "Descriptors rejected by finite queue", "descriptors", 1},
        {"dma.bursts.issued", "MoveData bursts issued", "bursts", 1},
        {"dma.bursts.completed", "MoveData bursts completed", "bursts", 1},
        {"dma.bytes.issued", "Bytes represented by MoveData requests", "bytes", 1},
        {"dma.bytes.completed", "Bytes completed by MoveData responses", "bytes", 1},
        {"dma.queue.occupancy", "Peak descriptor queue occupancy", "descriptors", 1},
        {"dma.outstanding.occupancy", "Peak outstanding burst occupancy", "bursts", 1},
        {"dma.setup_cycles", "Aggregate descriptor setup cycles", "cycles", 1},
        {"dma.elapsed_cycles", "Aggregate descriptor elapsed cycles", "cycles", 1}
    )

    SST_ELI_DOCUMENT_PORTS(
        {"descriptor", "DMA descriptor input", {"SnnDL.DmaDescriptorEvent"}},
        {"completion", "DMA completion or retry output", {"SnnDL.DmaCompletionEvent"}}
    )

    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS(
        {"memory", "StandardMem interface to private L1", "SST::Interfaces::StandardMem"}
    )

    SnnDmaEngineV5(SST::ComponentId_t id, SST::Params& params);
    ~SnnDmaEngineV5() override;

    void init(unsigned int phase) override;
    void setup() override;
    void finish() override;

private:
    struct DescriptorState {
        DmaDescriptorEvent descriptor;
        std::uint64_t burst_bytes = 0;
        std::uint64_t burst_count = 0;
        std::uint64_t next_offset = 0;
        std::uint64_t issued = 0;
        std::uint64_t completed = 0;
        std::uint64_t start_cycle = 0;
        std::uint64_t ready_cycle = 0;
    };

    struct RequestState {
        std::uint64_t descriptor_id = 0;
        std::uint64_t bytes = 0;
    };

    void handleDescriptor_(SST::Event* event);
    void handleMemory_(SST::Interfaces::StandardMem::Request* request);
    bool clockTick_(SST::Cycle_t cycle);
    void issueBursts_();
    void completeDescriptor_(std::uint64_t descriptor_id);
    void sendCompletion_(const DescriptorState& state, bool accepted, bool completed,
                         bool retryable, std::uint8_t error);
    void publishStatistics_();
    void writeStats_() const;

    SST::Output out_;
    SST::Link* descriptor_link_ = nullptr;
    SST::Link* completion_link_ = nullptr;
    SST::Interfaces::StandardMem* memory_ = nullptr;
    std::uint32_t channels_ = 1;
    std::size_t queue_entries_ = 16;
    std::size_t max_outstanding_ = 8;
    std::uint64_t default_burst_bytes_ = 64;
    std::uint64_t bytes_per_cycle_ = 16;
    std::uint64_t setup_cycles_ = 1;
    std::uint64_t expected_descriptors_ = 0;
    std::uint64_t current_cycle_ = 0;
    std::uint64_t completed_descriptors_ = 0;
    std::deque<std::uint64_t> order_;
    std::map<std::uint64_t, DescriptorState> descriptors_;
    std::map<std::uint64_t, RequestState> requests_;
    std::map<std::string, Statistic<std::uint64_t>*> statistics_;
    std::uint64_t descriptors_accepted_ = 0;
    std::uint64_t descriptors_completed_ = 0;
    std::uint64_t descriptor_retries_ = 0;
    std::uint64_t bursts_issued_ = 0;
    std::uint64_t bursts_completed_ = 0;
    std::uint64_t bytes_issued_ = 0;
    std::uint64_t bytes_completed_ = 0;
    std::uint64_t queue_peak_ = 0;
    std::uint64_t outstanding_peak_ = 0;
    std::uint64_t setup_cycles_total_ = 0;
    std::uint64_t elapsed_cycles_total_ = 0;
    std::string stats_json_;
};

} // namespace v5
} // namespace SnnDL
} // namespace SST

#endif
