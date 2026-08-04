#ifndef SST_SNN_DL_TIMESTEP_TYPES_H
#define SST_SNN_DL_TIMESTEP_TYPES_H

#include <cstdint>

namespace SST {
namespace SnnDL {

using TimestepId = std::uint64_t;

struct SpikeMessage {
    TimestepId timestep = 0;
    std::uint32_t source_neuron = 0;
    std::uint32_t source_node = 0;
    std::uint16_t source_core = 0;
    std::uint64_t source_event_seq = 0;
};

struct SynapseContribution {
    TimestepId timestep = 0;
    std::uint32_t pre_global = 0;
    std::uint32_t post_local = 0;
    float weight = 0.0f;
    std::uint64_t stable_order = 0;
};

} // namespace SnnDL
} // namespace SST

#endif
