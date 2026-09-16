#include "IfNeuronOp.h"

#include <cmath>
#include <stdexcept>

namespace SST {
namespace SnnDL {
namespace v5 {

IfNeuronOp::IfNeuronOp(const Config& config) : config_(config) {
    if (!std::isfinite(config_.resistance) || !std::isfinite(config_.threshold) ||
        !std::isfinite(config_.reset) || config_.threshold <= 0.0f) {
        throw std::invalid_argument("invalid v5 IF parameters");
    }
}

LifNeuronResult IfNeuronOp::evaluate(const LifNeuronState& state, float input_value) const {
    if (!std::isfinite(state.membrane) || !std::isfinite(input_value)) {
        throw std::invalid_argument("non-finite v5 IF state or input");
    }
    const float integrated = state.membrane + config_.resistance * input_value;
    if (integrated > config_.threshold) {
        return LifNeuronResult{LifNeuronState{config_.reset, 0}, true};
    }
    return LifNeuronResult{LifNeuronState{integrated, 0}, false};
}

} // namespace v5
} // namespace SnnDL
} // namespace SST
