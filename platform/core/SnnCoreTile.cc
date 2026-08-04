#include "SnnCoreTile.h"

#include <stdexcept>

namespace SST { namespace SnnDL {

SnnCoreTile::SnnCoreTile(
    std::uint32_t pe_id, std::uint16_t core_id,
    std::uint32_t neurons_per_core, float dt_ms, float tau_mem_ms,
    float threshold, float reset, std::uint32_t refractory_timesteps)
    : neurons_per_core_(neurons_per_core),
      core_id_(core_id),
      neuron_engine_(neurons_per_core, pe_id, core_id, dt_ms, tau_mem_ms,
                     threshold, reset, refractory_timesteps),
      delta_accumulator_(neurons_per_core) {
    if (neurons_per_core == 0) {
        throw std::invalid_argument("SnnCoreTile requires at least one neuron");
    }
}

void SnnCoreTile::begin(TimestepId timestep) {
    neuron_engine_.beginTimestep(timestep);
    delta_accumulator_.begin(timestep);
}

bool SnnCoreTile::accepts(TimestepId timestep, std::uint32_t local_neuron) const {
    return local_neuron < neurons_per_core_ &&
           neuron_engine_.acceptsInput(timestep, local_neuron);
}

void SnnCoreTile::addDelta(TimestepId timestep, std::uint32_t local_neuron,
                           float weight, std::uint64_t stable_order) {
    delta_accumulator_.add(SynapseContribution{
        timestep, 0, local_neuron, weight, stable_order});
}

std::vector<SpikeMessage> SnnCoreTile::commit(TimestepId timestep) {
    const auto deltas = delta_accumulator_.view(timestep);
    auto fired = neuron_engine_.commitTimestep(timestep, deltas);
    delta_accumulator_.commitDone(timestep);
    return fired;
}

}} // namespace SST::SnnDL
