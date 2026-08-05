#pragma once

#include <cstdint>

namespace SnnDL {
namespace v5 {

using ByteOffset = std::uint64_t;
using OwnerId = std::uint32_t;
using RegionId = std::uint32_t;

enum class AddressSpaceId : std::uint8_t {
    CoreState = 0,
    CoreDelta = 1,
    CoreIndex = 2,
    PeRoute = 3,
    PeWeightSpm = 4,
    ChipDram = 5,
};

}  // namespace v5
}  // namespace SnnDL
