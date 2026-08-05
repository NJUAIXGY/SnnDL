#pragma once

#include <cstdint>

#include "V5Types.h"

namespace SnnDL {
namespace v5 {

struct TypedAddress final {
    AddressSpaceId space;
    OwnerId owner_id;
    ByteOffset byte_offset;
};

struct RegionDescriptor final {
    AddressSpaceId space;
    OwnerId owner_id;
    std::uint64_t base;
    std::uint64_t size_bytes;
    bool has_chip_dram_backing;
};

inline bool addressInRegion(const TypedAddress& address, const RegionDescriptor& region) {
    return address.space == region.space && address.owner_id == region.owner_id &&
           address.byte_offset < region.size_bytes;
}

inline bool typedAddressForElement(const RegionDescriptor& region,
                                   std::uint64_t element_index,
                                   std::uint64_t element_bytes,
                                   TypedAddress& address) {
    if (element_bytes == 0 || element_index > UINT64_MAX / element_bytes) return false;
    const auto byte_offset = element_index * element_bytes;
    if (byte_offset >= region.size_bytes ||
        element_bytes > region.size_bytes - byte_offset) {
        return false;
    }
    address = TypedAddress{region.space, region.owner_id, byte_offset};
    return true;
}

inline bool resolveRegionAddress(const TypedAddress& address,
                                 const RegionDescriptor& region,
                                 std::uint64_t& physical) {
    if (!addressInRegion(address, region) ||
        address.byte_offset > UINT64_MAX - region.base) {
        return false;
    }
    physical = region.base + address.byte_offset;
    return true;
}

inline bool resolveChipDramAddress(const TypedAddress& address,
                                   const RegionDescriptor& region,
                                   std::uint64_t& physical) {
    if (address.space != AddressSpaceId::ChipDram ||
        region.space != AddressSpaceId::ChipDram || !region.has_chip_dram_backing ||
        !resolveRegionAddress(address, region, physical)) {
        return false;
    }
    return true;
}

}  // namespace v5
}  // namespace SnnDL
