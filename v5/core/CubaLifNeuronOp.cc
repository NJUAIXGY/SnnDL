#include "CubaLifNeuronOp.h"

#include <cmath>
#include <cstring>
#include <stdexcept>

namespace SST {
namespace SnnDL {
namespace v5 {

namespace {

float f32(float value) {
    // Force the float32 rounding boundary used by the Python reference.
    volatile float narrowed = value;
    return narrowed;
}

bool finite(float value) {
    return std::isfinite(static_cast<double>(value));
}

std::uint32_t bitsOfFloat(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return bits;
}

float floatOfBits(std::uint32_t bits) {
    float value = 0.0f;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

void writeU32(std::uint32_t value, std::uint8_t* bytes) {
    for (std::size_t index = 0; index < sizeof(value); ++index) {
        bytes[index] = static_cast<std::uint8_t>(value >> (index * 8));
    }
}

std::uint32_t readU32(const std::uint8_t* bytes) {
    std::uint32_t value = 0;
    for (std::size_t index = 0; index < sizeof(value); ++index) {
        value |= static_cast<std::uint32_t>(bytes[index]) << (index * 8);
    }
    return value;
}

} // namespace

CubaLifNeuronOp::CubaLifNeuronOp(const Config& config) : config_(config) {
    if (!finite(config_.dt_seconds) || config_.dt_seconds <= 0.0f ||
        !finite(config_.tau_syn_seconds) || config_.tau_syn_seconds <= 0.0f ||
        !finite(config_.tau_mem_seconds) || config_.tau_mem_seconds <= 0.0f ||
        !finite(config_.resistance) || !finite(config_.v_leak) ||
        !finite(config_.threshold) || config_.threshold <= 0.0f ||
        !finite(config_.reset) || !finite(config_.input_weight)) {
        throw std::invalid_argument("invalid v5 CubaLIF parameters");
    }
    synaptic_decay_ = f32(static_cast<float>(std::exp(
        -static_cast<double>(config_.dt_seconds) / config_.tau_syn_seconds)));
    membrane_decay_ = f32(static_cast<float>(std::exp(
        -static_cast<double>(config_.dt_seconds) / config_.tau_mem_seconds)));
}

CubaLifNeuronResult CubaLifNeuronOp::evaluate(const CubaLifNeuronState& state,
                                               float input_value) const {
    if (!finite(state.synaptic_current) || !finite(state.membrane) || !finite(input_value)) {
        throw std::invalid_argument("v5 CubaLIF state and input must be finite");
    }

    const float decayed_current = f32(state.synaptic_current * synaptic_decay_);
    const float injected_current = f32(config_.input_weight * input_value);
    const float current = f32(static_cast<float>(
        static_cast<double>(decayed_current) + injected_current));
    const float one_minus_membrane_decay = f32(1.0f - membrane_decay_);
    const float leak_term = f32(config_.v_leak * one_minus_membrane_decay);
    const float decayed_membrane = f32(state.membrane * membrane_decay_);
    const float membrane_delta = f32(static_cast<float>(
        static_cast<double>(config_.resistance) * current * one_minus_membrane_decay));
    const float membrane = f32(static_cast<float>(
        static_cast<double>(decayed_membrane) + leak_term + membrane_delta));
    float updated_membrane = membrane;

    const bool fired = membrane > config_.threshold;
    if (fired) {
        updated_membrane = config_.reset_mode == ResetMode::Value
                               ? f32(config_.reset)
                               : f32(membrane - config_.threshold);
    }
    return CubaLifNeuronResult{CubaLifNeuronState{current, updated_membrane}, fired};
}

void CubaLifNeuronOp::encodeState(const CubaLifNeuronState& state,
                                  std::array<std::uint8_t, kStateBytes>& bytes) {
    if (!finite(state.synaptic_current) || !finite(state.membrane)) {
        throw std::invalid_argument("v5 CubaLIF state must be finite");
    }
    writeU32(bitsOfFloat(state.synaptic_current), bytes.data());
    writeU32(bitsOfFloat(state.membrane), bytes.data() + sizeof(float));
}

bool CubaLifNeuronOp::decodeState(const std::uint8_t* bytes, std::size_t size,
                                  CubaLifNeuronState& state) {
    if (bytes == nullptr || size < kStateBytes) return false;
    const CubaLifNeuronState decoded{
        floatOfBits(readU32(bytes)),
        floatOfBits(readU32(bytes + sizeof(float))),
    };
    if (!finite(decoded.synaptic_current) || !finite(decoded.membrane)) return false;
    state = decoded;
    return true;
}

} // namespace v5
} // namespace SnnDL
} // namespace SST
