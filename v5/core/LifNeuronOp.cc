#include "LifNeuronOp.h"

#include <cmath>
#include <stdexcept>

namespace SST {
namespace SnnDL {
namespace v5 {

LifNeuronOp::LifNeuronOp(const Config& config) : config_(config) {
    if (config_.dt_ms <= 0.0f || config_.tau_mem_ms <= 0.0f || config_.threshold <= 0.0f) {
        throw std::invalid_argument("invalid v5 LIF parameters");
    }
    leak_ = std::exp(-config_.dt_ms / config_.tau_mem_ms);
}

LifNeuronResult LifNeuronOp::evaluate(const LifNeuronState& state, float delta) const {
    LifNeuronResult result{state, false};
    if (state.refractory > 0) {
        result.state.refractory = state.refractory - 1;
        result.state.membrane = config_.reset;
        return result;
    }
    const float integrated = state.membrane * leak_ + delta;
    if (integrated >= config_.threshold) {
        result.state.membrane = config_.reset;
        result.state.refractory = config_.refractory_timesteps;
        result.fired = true;
    } else {
        result.state.membrane = integrated;
        result.state.refractory = 0;
    }
    return result;
}

} // namespace v5
} // namespace SnnDL
} // namespace SST
