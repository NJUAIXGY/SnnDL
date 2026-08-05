#ifndef SST_SNN_DL_V5_LIF_NEURON_OP_H
#define SST_SNN_DL_V5_LIF_NEURON_OP_H

#include <cstdint>

namespace SST {
namespace SnnDL {
namespace v5 {

struct LifNeuronState {
    float membrane = 0.0f;
    std::uint32_t refractory = 0;
};

struct LifNeuronResult {
    LifNeuronState state;
    bool fired = false;
};

class LifNeuronOp {
public:
    struct Config {
        float dt_ms = 1.0f;
        float tau_mem_ms = 20.0f;
        float threshold = 1.0f;
        float reset = 0.0f;
        std::uint32_t refractory_timesteps = 0;
    };

    explicit LifNeuronOp(const Config& config);

    LifNeuronResult evaluate(const LifNeuronState& state, float delta) const;
    float leak() const { return leak_; }
    const Config& config() const { return config_; }

private:
    Config config_;
    float leak_ = 1.0f;
};

} // namespace v5
} // namespace SnnDL
} // namespace SST

#endif
