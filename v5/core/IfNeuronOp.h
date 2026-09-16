#ifndef SST_SNN_DL_V5_IF_NEURON_OP_H
#define SST_SNN_DL_V5_IF_NEURON_OP_H

#include "LifNeuronOp.h"

namespace SST {
namespace SnnDL {
namespace v5 {

// Discrete NIR IF: v <- v + R * input; fire iff v > threshold; then v <- reset.
// The state shares the CoreState encoding used by LIF, with refractory fixed to zero.
class IfNeuronOp {
public:
    struct Config {
        float resistance = 1.0f;
        float threshold = 1.0f;
        float reset = 0.0f;
    };

    explicit IfNeuronOp(const Config& config);

    LifNeuronResult evaluate(const LifNeuronState& state, float input_value) const;
    const Config& config() const { return config_; }

private:
    Config config_;
};

} // namespace v5
} // namespace SnnDL
} // namespace SST

#endif
