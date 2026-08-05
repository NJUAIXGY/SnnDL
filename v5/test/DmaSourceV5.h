#ifndef SST_SNN_DL_V5_DMA_SOURCE_H
#define SST_SNN_DL_V5_DMA_SOURCE_H

#include "v5/events/StorageEvents.h"

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>

#include <cstdint>
#include <deque>
#include <map>
#include <string>

namespace SST {
namespace SnnDL {
namespace v5 {

class DmaSourceV5 final : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(
        DmaSourceV5,
        "SnnDL",
        "DmaSourceV5",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "P3 deterministic DMA descriptor source and evidence sink",
        COMPONENT_CATEGORY_PROCESSOR
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"descriptors", "Number of descriptors to issue", "1"},
        {"source_base", "First source address", "0"},
        {"destination_base", "First destination address", "4096"},
        {"descriptor_bytes", "Bytes per descriptor", "256"},
        {"burst_bytes", "Bytes per DMA burst", "64"},
        {"stride_bytes", "Address stride between descriptors", "256"},
        {"source_space", "Source address-space id for provenance", "0"},
        {"destination_space", "Destination address-space id for provenance", "4"},
        {"source_owner", "Source owner id", "0"},
        {"destination_owner", "Destination owner id", "0"},
        {"clock", "Source clock", "1GHz"},
        {"output_json", "Evidence JSON path", ""},
        {"verbose", "Verbose logging level", "0"}
    )

    SST_ELI_DOCUMENT_PORTS(
        {"descriptor", "Descriptor output", {"SnnDL.DmaDescriptorEvent"}},
        {"completion", "Completion input", {"SnnDL.DmaCompletionEvent"}},
        {"ready", "Optional final preload completion output", {"SnnDL.DmaCompletionEvent"}}
    )

    DmaSourceV5(SST::ComponentId_t id, SST::Params& params);
    ~DmaSourceV5() override;

    void init(unsigned int) override {}
    void setup() override {}
    void finish() override;

private:
    void handleCompletion_(SST::Event* event);
    bool clockTick_(SST::Cycle_t cycle);
    void sendDescriptor_(const DmaDescriptorEvent& descriptor);
    void writeEvidence_() const;

    SST::Output out_;
    SST::Link* descriptor_link_ = nullptr;
    SST::Link* completion_link_ = nullptr;
    SST::Link* ready_link_ = nullptr;
    std::uint64_t descriptor_count_ = 1;
    std::uint64_t source_base_ = 0;
    std::uint64_t destination_base_ = 4096;
    std::uint64_t descriptor_bytes_ = 256;
    std::uint64_t burst_bytes_ = 64;
    std::uint64_t stride_bytes_ = 256;
    std::uint8_t source_space_ = 0;
    std::uint8_t destination_space_ = 4;
    std::uint32_t source_owner_ = 0;
    std::uint32_t destination_owner_ = 0;
    std::uint64_t generated_ = 0;
    std::uint64_t attempts_ = 0;
    std::uint64_t completed_ = 0;
    std::uint64_t completed_bytes_ = 0;
    std::uint64_t completed_bursts_ = 0;
    std::uint64_t retryable_rejects_ = 0;
    bool ready_sent_ = false;
    std::uint64_t current_cycle_ = 0;
    std::map<std::uint64_t, DmaDescriptorEvent> outstanding_;
    std::deque<std::uint64_t> retry_queue_;
    std::string output_json_;
};

} // namespace v5
} // namespace SnnDL
} // namespace SST

#endif
