#ifndef SST_SNN_DL_CORE_SNN_CORE_TILE_H
#define SST_SNN_DL_CORE_SNN_CORE_TILE_H

#include "api/TimestepTypes.h"
#include "snn/compute/DeltaAccumulator.h"
#include "snn/compute/NextGenNeuronEngine.h"

#include <cstdint>
#include <vector>

namespace SST { namespace SnnDL {

// The core tile is a domain object owned by MeshPE2D.  It deliberately has no
// SST links; transport and memory timing remain in the owning SST component.
class SnnCoreTile {
public:
    SnnCoreTile(std::uint32_t pe_id, std::uint16_t core_id,
                std::uint32_t neurons_per_core, float dt_ms,
                float tau_mem_ms, float threshold, float reset,
                std::uint32_t refractory_timesteps);

    void begin(TimestepId timestep);
    bool accepts(TimestepId timestep, std::uint32_t local_neuron) const;
    void addDelta(TimestepId timestep, std::uint32_t local_neuron,
                  float weight, std::uint64_t stable_order);
    std::vector<SpikeMessage> commit(TimestepId timestep);
    const std::vector<NeuronState>& state() const { return neuron_engine_.state(); }
    std::uint32_t neurons() const { return neurons_per_core_; }
    std::uint16_t coreId() const { return core_id_; }

private:
    std::uint32_t neurons_per_core_ = 0;
    std::uint16_t core_id_ = 0;
    NextGenNeuronEngine neuron_engine_;
    DeltaAccumulator delta_accumulator_;
};

}} // namespace SST::SnnDL

#endif
