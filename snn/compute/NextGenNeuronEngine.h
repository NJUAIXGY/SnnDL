#ifndef SST_SNN_DL_NEXTGEN_NEURON_ENGINE_H
#define SST_SNN_DL_NEXTGEN_NEURON_ENGINE_H

#include "api/TimestepTypes.h"

#include <cstdint>
#include <map>
#include <vector>

namespace SST {
namespace SnnDL {

struct NeuronState {
    float v_mem = 0.0f;
    std::uint32_t refractory = 0;
};

class NextGenNeuronEngine {
public:
    NextGenNeuronEngine(std::uint32_t neurons, std::uint32_t source_node,
                        std::uint16_t source_core, float dt_ms,
                        float tau_mem_ms, float threshold, float reset,
                        std::uint32_t refractory_timesteps);

    void beginTimestep(TimestepId timestep);
    bool acceptsInput(TimestepId timestep, std::uint32_t neuron) const;
    std::vector<SpikeMessage> commitTimestep(TimestepId timestep,
                                             const std::vector<float>& deltas);
    std::vector<SpikeMessage> releaseHeldSpikes(TimestepId timestep);
    const std::vector<NeuronState>& state() const { return state_; }

private:
    std::uint32_t source_node_ = 0;
    std::uint16_t source_core_ = 0;
    float leak_ = 1.0f;
    float threshold_ = 1.0f;
    float reset_ = 0.0f;
    std::uint32_t refractory_timesteps_ = 0;
    TimestepId active_timestep_ = 0;
    bool active_ = false;
    std::uint64_t event_seq_ = 0;
    std::vector<NeuronState> state_;
    std::map<TimestepId, std::vector<SpikeMessage>> held_spikes_;
};

} // namespace SnnDL
} // namespace SST

#endif
