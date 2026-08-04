#include "api/TimestepTypes.h"
#include "snn/compute/DeltaAccumulator.h"
#include "snn/compute/NextGenNeuronEngine.h"
#include "snn/timestep/TimestepTracker.h"

#include <cassert>
#include <stdexcept>

using namespace SST::SnnDL;

int main() {
    TimestepTracker tracker;
    tracker.open(7);
    tracker.acquire(7, WorkKind::WeightRead, 2);
    tracker.release(7, WorkKind::WeightRead, 1);
    tracker.sealIngress(7);
    assert(!tracker.locallyDrained(7));
    tracker.release(7, WorkKind::WeightRead, 1);
    assert(tracker.locallyDrained(7));
    bool underflow = false;
    try {
        tracker.release(7, WorkKind::WeightRead, 1);
    } catch (const std::logic_error&) {
        underflow = true;
    }
    assert(underflow);

    DeltaAccumulator accumulator(2);
    accumulator.begin(7);
    accumulator.add(SynapseContribution{7, 0, 0, 0.25f, 2});
    accumulator.add(SynapseContribution{7, 1, 0, 0.75f, 1});
    accumulator.add(SynapseContribution{7, 2, 1, 0.5f, 3});
    auto values = accumulator.view(7);
    assert(values.size() == 2);
    assert(values[0] > 0.999f && values[0] < 1.001f);
    assert(values[1] > 0.499f && values[1] < 0.501f);
    accumulator.commitDone(7);

    NextGenNeuronEngine neurons(2, 3, 0, 1.0f, 20.0f, 1.0f, 0.0f, 1);
    neurons.beginTimestep(7);
    assert(neurons.acceptsInput(7, 0));
    auto fired = neurons.commitTimestep(7, std::vector<float>{1.1f, 0.0f});
    assert(fired.size() == 1);
    assert(fired.front().timestep == 8);
    assert(neurons.releaseHeldSpikes(7).empty());
    auto held = neurons.releaseHeldSpikes(8);
    assert(held.size() == 1);

    // Exercise the intended chip-scale mapping: 16 PEs, four cores per PE,
    // and 64 neurons per core (256 neurons per PE, 4096 total).
    constexpr std::uint32_t kPes = 16;
    constexpr std::uint32_t kCoresPerPe = 4;
    constexpr std::uint32_t kNeuronsPerCore = 64;
    std::vector<NextGenNeuronEngine> pe_cores;
    pe_cores.reserve(kPes * kCoresPerPe);
    std::size_t total_fired = 0;
    for (std::uint32_t pe = 0; pe < kPes; ++pe) {
        for (std::uint16_t core = 0; core < kCoresPerPe; ++core) {
            pe_cores.emplace_back(kNeuronsPerCore, pe, core, 1.0f, 20.0f,
                                  1.0f, 0.0f, 0);
            auto& core_engine = pe_cores.back();
            core_engine.beginTimestep(11);
            std::vector<float> core_deltas(kNeuronsPerCore, 1.1f);
            auto core_fired = core_engine.commitTimestep(11, core_deltas);
            assert(core_engine.state().size() == kNeuronsPerCore);
            assert(core_fired.size() == kNeuronsPerCore);
            total_fired += core_fired.size();
        }
    }
    assert(pe_cores.size() == kPes * kCoresPerPe);
    assert(total_fired == kPes * kCoresPerPe * kNeuronsPerCore);

    return 0;
}
