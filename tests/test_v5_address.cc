#include <cassert>
#include <cstdint>

#include "v5/api/AddressSpace.h"
#include "v5/api/ArtifactContract.h"

int main() {
    static_assert(SnnDL::v5::kArtifactHeaderBytes == 72, "v5 header ABI changed");
    static_assert(SnnDL::v5::kArtifactFormatVersion == 1, "v5 format changed");

    const SnnDL::v5::RegionDescriptor region{
        SnnDL::v5::AddressSpaceId::ChipDram, 0, 0x1000, 0x100, true};
    const SnnDL::v5::TypedAddress address{
        SnnDL::v5::AddressSpaceId::ChipDram, 0, 0x20};
    std::uint64_t physical = 0;
    assert(SnnDL::v5::addressInRegion(address, region));
    assert(SnnDL::v5::resolveChipDramAddress(address, region, physical));
    assert(physical == 0x1020);
    assert(!SnnDL::v5::resolveChipDramAddress(
        {SnnDL::v5::AddressSpaceId::ChipDram, 0, 0x100}, region, physical));
    assert(!SnnDL::v5::resolveChipDramAddress(
        {SnnDL::v5::AddressSpaceId::CoreIndex, 0, 0x20}, region, physical));

    const SnnDL::v5::RegionDescriptor weights{
        SnnDL::v5::AddressSpaceId::PeWeightSpm, 7, 0, 16, false};
    SnnDL::v5::TypedAddress weight_address{};
    assert(SnnDL::v5::typedAddressForElement(weights, 3, 4, weight_address));
    assert(weight_address.space == SnnDL::v5::AddressSpaceId::PeWeightSpm);
    assert(weight_address.owner_id == 7);
    assert(weight_address.byte_offset == 12);
    assert(SnnDL::v5::resolveRegionAddress(weight_address, weights, physical));
    assert(physical == 12);
    assert(!SnnDL::v5::typedAddressForElement(weights, 4, 4, weight_address));
    assert(!SnnDL::v5::typedAddressForElement(weights, UINT64_MAX, 2, weight_address));

    assert(SnnDL::v5::artifactMagic()[0] == 'S');
    return 0;
}
