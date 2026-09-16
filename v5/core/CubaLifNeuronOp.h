#ifndef SST_SNN_DL_V5_CUBA_LIF_NEURON_OP_H
#define SST_SNN_DL_V5_CUBA_LIF_NEURON_OP_H

#include <array>
#include <cstddef>
#include <cstdint>

namespace SST {
namespace SnnDL {
namespace v5 {

struct CubaLifNeuronState {
    float synaptic_current = 0.0f;
    float membrane = 0.0f;
};

struct CubaLifNeuronResult {
    CubaLifNeuronState state;
    bool fired = false;
};

class CubaLifNeuronOp {
public:
    enum class ResetMode : std::uint8_t {
        Value = 0,
        Subtract = 1,
    };

    struct Config {
        float dt_seconds = 1.0e-3f;
        float tau_syn_seconds = 2.0e-3f;
        float tau_mem_seconds = 3.0e-3f;
        float resistance = 1.0f;
        float v_leak = 0.0f;
        float threshold = 1.0f;
        float reset = 0.0f;
        ResetMode reset_mode = ResetMode::Value;
        float input_weight = 1.0f;
    };

    static constexpr std::size_t kStateBytes = sizeof(float) * 2;

    explicit CubaLifNeuronOp(const Config& config);

    CubaLifNeuronResult evaluate(const CubaLifNeuronState& state, float input_value) const;
    const Config& config() const { return config_; }
    float synapticDecay() const { return synaptic_decay_; }
    float membraneDecay() const { return membrane_decay_; }

    // Canonical little-endian state encoding for a future CubaLIF
    // CoreState region. It is separate from CoreStorageV5's LIF record.
    static void encodeState(const CubaLifNeuronState& state,
                            std::array<std::uint8_t, kStateBytes>& bytes);
    static bool decodeState(const std::uint8_t* bytes, std::size_t size,
                            CubaLifNeuronState& state);

private:
    Config config_;
    float synaptic_decay_ = 0.0f;
    float membrane_decay_ = 0.0f;
};

} // namespace v5
} // namespace SnnDL
} // namespace SST

#endif
