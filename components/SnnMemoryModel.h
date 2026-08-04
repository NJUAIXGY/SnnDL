#ifndef SST_SNN_DL_MEMORY_MODEL_H
#define SST_SNN_DL_MEMORY_MODEL_H

#include "events/SnnMeshEvents.h"

#include <sst/core/clock.h>
#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>

#include <cstdint>
#include <deque>
#include <vector>

namespace SST { namespace SnnDL {

// A shared chip DRAM endpoint.  It is intentionally an SST component rather
// than a Python delay: requests occupy a timed queue and responses traverse
// the same SST links back to their PE.
class SnnMemoryModel final : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(
        SnnMemoryModel,
        "SnnDL",
        "SnnMemoryModel",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "Timed shared memory endpoint for the v4 2D SNN model",
        COMPONENT_CATEGORY_MEMORY
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"total_pes", "Number of PE request ports", "1"},
        {"latency_cycles", "Read response latency in memory clock cycles", "10"},
        {"queue_capacity", "Maximum number of outstanding requests", "4096"},
        {"clock", "Memory clock", "1GHz"},
        {"verbose", "Verbose logging level", "0"}
    )

    SST_ELI_DOCUMENT_PORTS(
        {"pe%(pe)d", "Bidirectional request/response port for each PE", {"SST::Event"}}
    )

    SnnMemoryModel(SST::ComponentId_t id, SST::Params& params);
    ~SnnMemoryModel() override;

    void init(unsigned int phase) override;
    void setup() override;
    void finish() override;

private:
    struct Pending {
        SST::Cycle_t ready_cycle = 0;
        std::uint32_t source_pe = 0;
        std::uint64_t request_id = 0;
        std::uint64_t timestep = 0;
        std::uint64_t address = 0;
        float value = 0.0f;
    };

    void handleRequest_(SST::Event* event);
    bool clockTick_(SST::Cycle_t cycle);

    SST::Output out_;
    std::vector<SST::Link*> pe_links_;
    std::deque<Pending> pending_;
    std::uint64_t latency_cycles_ = 10;
    std::size_t queue_capacity_ = 4096;
    std::uint64_t requests_ = 0;
    std::uint64_t responses_ = 0;
};

}} // namespace SST::SnnDL

#endif
