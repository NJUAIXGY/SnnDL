#include "v5/storage/CoreStorageV5.h"

#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

using namespace SST::SnnDL::v5;
using AddressSpaceId = SnnDL::v5::AddressSpaceId;

namespace {

CoreStorageV5Config config() {
    CoreStorageV5Config value;
    value.core_id = 3;
    value.pe_id = 2;
    value.neurons = 2;
    value.max_delta_entries_per_neuron = 2;
    value.index_bytes = 32;
    value.route_bytes = 32;
    value.state_sram.banks = 2;
    value.delta_sram.banks = 2;
    value.index_sram.banks = 2;
    value.route_sram.banks = 2;
    return value;
}

void testTypedRegionsAndState() {
    CoreStorageV5 storage(config());
    const auto& state_region = storage.region(AddressSpaceId::CoreState);
    const auto& delta_region = storage.region(AddressSpaceId::CoreDelta);
    const auto& index_region = storage.region(AddressSpaceId::CoreIndex);
    const auto& route_region = storage.region(AddressSpaceId::PeRoute);
    assert(state_region.owner_id == 3);
    assert(delta_region.owner_id == 3);
    assert(index_region.owner_id == 3);
    assert(route_region.owner_id == 2);
    assert(state_region.space == AddressSpaceId::CoreState);
    assert(delta_region.space == AddressSpaceId::CoreDelta);
    assert(index_region.space == AddressSpaceId::CoreIndex);
    assert(route_region.space == AddressSpaceId::PeRoute);

    LifNeuronState initial;
    assert(storage.readState(1, initial));
    assert(initial.membrane == 0.0f && initial.refractory == 0);
    assert(storage.writeState(1, LifNeuronState{0.75f, 4}));
    LifNeuronState round_trip;
    assert(storage.readState(1, round_trip));
    assert(std::fabs(round_trip.membrane - 0.75f) < 1.0e-6f);
    assert(round_trip.refractory == 4);

    std::vector<std::uint8_t> index;
    std::vector<std::uint8_t> route;
    assert(storage.readIndex(0, 8, index) && index.size() == 8);
    assert(storage.readRoute(0, 8, route) && route.size() == 8);
    assert(storage.stats(AddressSpaceId::CoreIndex).reads == 1);
    assert(storage.stats(AddressSpaceId::PeRoute).reads == 1);
}

void testDeltaIsBackedAndBounded() {
    CoreStorageV5 storage(config());
    const RetireEntry first{RetireKey{0, 9, 1}, 0, 0.25f};
    const RetireEntry second{RetireKey{0, 3, 2}, 0, 0.5f};
    const RetireEntry overflow{RetireKey{0, 12, 0}, 0, 1.0f};
    assert(storage.appendDelta(first));
    assert(storage.appendDelta(second));
    assert(!storage.appendDelta(overflow));

    std::vector<RetireEntry> entries;
    assert(storage.readDeltaEntries(0, entries));
    assert(entries.size() == 2);
    assert(entries[0].key.source_event_seq == 3);
    assert(entries[1].key.source_event_seq == 9);
    assert(std::fabs(entries[0].weight - 0.5f) < 1.0e-6f);
    assert(storage.stats(AddressSpaceId::CoreDelta).reads >= 3);
    assert(storage.stats(AddressSpaceId::CoreDelta).writes >= 4);

    assert(storage.clearDelta(0));
    entries.clear();
    assert(storage.readDeltaEntries(0, entries));
    assert(entries.empty());
    storage.resetTimestep();
}

} // namespace

int main() {
    testTypedRegionsAndState();
    testDeltaIsBackedAndBounded();
    std::cout << "v5 core storage binding: PASS\n";
    return 0;
}
