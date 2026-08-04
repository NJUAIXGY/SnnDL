#include "NextGenNeuronEngine.h"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace SST {
namespace SnnDL {

NextGenNeuronEngine::NextGenNeuronEngine(
    std::uint32_t neurons, std::uint32_t source_node,
    std::uint16_t source_core, float dt_ms, float tau_mem_ms,
    float threshold, float reset, std::uint32_t refractory_timesteps)
    : source_node_(source_node), source_core_(source_core),
      leak_(std::exp(-dt_ms / tau_mem_ms)), threshold_(threshold),
      reset_(reset), refractory_timesteps_(refractory_timesteps),
      state_(neurons) {
    if (neurons == 0 || dt_ms <= 0.0f || tau_mem_ms <= 0.0f || threshold <= 0.0f) {
        throw std::invalid_argument("invalid neuron engine parameters");
    }
}

void NextGenNeuronEngine::beginTimestep(TimestepId timestep) {
    if (active_) throw std::logic_error("neuron timestep is already active");
    active_timestep_ = timestep;
    active_ = true;
}

bool NextGenNeuronEngine::acceptsInput(TimestepId timestep, std::uint32_t neuron) const {
    if (!active_ || timestep != active_timestep_ || neuron >= state_.size()) return false;
    return state_[neuron].refractory == 0;
}

std::vector<SpikeMessage> NextGenNeuronEngine::commitTimestep(
    TimestepId timestep, const std::vector<float>& deltas) {
    if (!active_ || timestep != active_timestep_) {
        throw std::logic_error("neuron commit has the wrong timestep");
    }
    if (deltas.size() != state_.size()) {
        throw std::invalid_argument("delta view size does not match neuron state");
    }

    std::vector<SpikeMessage> fired;
    fired.reserve(state_.size());
    for (std::size_t neuron = 0; neuron < state_.size(); ++neuron) {
        auto& current = state_[neuron];
        if (current.refractory > 0) {
            --current.refractory;
            current.v_mem = reset_;
            continue;
        }
        const float integrated = current.v_mem * leak_ + deltas[neuron];
        if (integrated >= threshold_) {
            current.v_mem = reset_;
            current.refractory = refractory_timesteps_;
            fired.push_back(SpikeMessage{timestep + 1, static_cast<std::uint32_t>(neuron),
                                         source_node_, source_core_, event_seq_++});
        } else {
            current.v_mem = integrated;
        }
    }
    held_spikes_[timestep + 1] = fired;
    active_ = false;
    return fired;
}

std::vector<SpikeMessage> NextGenNeuronEngine::releaseHeldSpikes(TimestepId timestep) {
    auto it = held_spikes_.find(timestep);
    if (it == held_spikes_.end()) return {};
    auto result = std::move(it->second);
    held_spikes_.erase(it);
    return result;
}

} // namespace SnnDL
} // namespace SST
