#include "v5/core/IfNeuronOp.h"

#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>

using namespace SST::SnnDL::v5;

int main() {
    const IfNeuronOp op(IfNeuronOp::Config{1.0f, 1.0f, -0.25f});
    const auto threshold = op.evaluate(LifNeuronState{1.0f, 7}, 0.0f);
    assert(!threshold.fired);
    assert(threshold.state.membrane == 1.0f);
    assert(threshold.state.refractory == 0);
    const auto fired = op.evaluate(LifNeuronState{0.75f, 0}, 0.5f);
    assert(fired.fired);
    assert(fired.state.membrane == -0.25f);
    assert(fired.state.refractory == 0);
    bool rejected = false;
    try {
        (void)op.evaluate(LifNeuronState{std::numeric_limits<float>::quiet_NaN(), 0}, 0.0f);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
    return 0;
}
