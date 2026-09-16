#include "v5/core/CubaLifNeuronOp.h"

#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>

using namespace SST::SnnDL::v5;

namespace {

std::uint32_t bitsOf(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

CubaLifNeuronOp::Config vectorConfig() {
    CubaLifNeuronOp::Config config;
    config.dt_seconds = 1.0e-3f;
    config.tau_syn_seconds = 2.0e-3f;
    config.tau_mem_seconds = 3.0e-3f;
    config.resistance = 2.0f;
    config.v_leak = -0.1f;
    config.threshold = 1.0f;
    config.reset = 0.2f;
    config.input_weight = 1.5f;
    return config;
}

void testPythonReferenceVector() {
    CubaLifNeuronOp op(vectorConfig());
    CubaLifNeuronState state;
    const float inputs[] = {0.75f, 0.0f, 1.2f};
    const float currents[] = {1.125f, 0.6823469996452322f, 2.213864326477051f};
    const float membranes[] = {0.6094577312469482f, 0.7951967120170593f, 0.2f};
    const std::uint32_t current_bits[] = {0x3f900000u, 0x3f2eae4bu, 0x400daff4u};
    const std::uint32_t membrane_bits[] = {0x3f1c056cu, 0x3f4b9203u, 0x3e4ccccdu};
    const bool fired[] = {false, false, true};
    for (int index = 0; index < 3; ++index) {
        const auto result = op.evaluate(state, inputs[index]);
        assert(result.state.synaptic_current == currents[index]);
        assert(result.state.membrane == membranes[index]);
        assert(bitsOf(result.state.synaptic_current) == current_bits[index]);
        assert(bitsOf(result.state.membrane) == membrane_bits[index]);
        assert(result.fired == fired[index]);
        state = result.state;
    }
}

void testResetAndStrictThreshold() {
    auto config = vectorConfig();
    config.dt_seconds = 1.0f;
    config.tau_syn_seconds = 1.0f;
    config.tau_mem_seconds = 1.0f;
    config.resistance = 1.0f;
    config.v_leak = 0.0f;
    config.threshold = 0.1f;
    config.reset = -0.25f;
    config.input_weight = 1.0f;
    config.reset_mode = CubaLifNeuronOp::ResetMode::Value;
    const auto value = CubaLifNeuronOp(config).evaluate(CubaLifNeuronState{}, 1.0f);
    assert(value.fired && value.state.membrane == -0.25f);

    config.reset_mode = CubaLifNeuronOp::ResetMode::Subtract;
    const auto subtract = CubaLifNeuronOp(config).evaluate(CubaLifNeuronState{}, 1.0f);
    assert(subtract.fired);
    assert(std::fabs(subtract.state.membrane - 0.5321205258369446f) < 1.0e-7f);

    config.resistance = 0.0f;
    config.v_leak = 1.0f;
    config.threshold = 1.0f;
    const auto equal = CubaLifNeuronOp(config).evaluate(CubaLifNeuronState{0.0f, 1.0f}, 0.0f);
    assert(!equal.fired);
}

void testStateCodec() {
    const CubaLifNeuronState expected{1.125f, -0.25f};
    std::array<std::uint8_t, CubaLifNeuronOp::kStateBytes> bytes{};
    CubaLifNeuronOp::encodeState(expected, bytes);
    CubaLifNeuronState restored;
    assert(CubaLifNeuronOp::decodeState(bytes.data(), bytes.size(), restored));
    assert(restored.synaptic_current == expected.synaptic_current);
    assert(restored.membrane == expected.membrane);
    assert(!CubaLifNeuronOp::decodeState(bytes.data(), bytes.size() - 1, restored));
    const CubaLifNeuronState invalid{std::numeric_limits<float>::quiet_NaN(), 0.0f};
    bool rejected_encode = false;
    try {
        CubaLifNeuronOp::encodeState(invalid, bytes);
    } catch (const std::invalid_argument&) {
        rejected_encode = true;
    }
    assert(rejected_encode);
}

} // namespace

int main() {
    testPythonReferenceVector();
    testResetAndStrictThreshold();
    testStateCodec();
    std::cout << "v5 CubaLIF operator: PASS\n";
    return 0;
}
