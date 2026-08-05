#include "CoreStorageV5.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace SST {
namespace SnnDL {
namespace v5 {

namespace {
using AddressSpaceId = ::SnnDL::v5::AddressSpaceId;
using RegionDescriptor = ::SnnDL::v5::RegionDescriptor;

constexpr std::uint64_t kRegionStride = 0x100000000ULL;
constexpr std::uint64_t kDeltaBase = 0x10000000ULL;
constexpr std::uint64_t kIndexBase = 0x20000000ULL;
constexpr std::uint64_t kRouteBase = 0x30000000ULL;

std::uint64_t checkedMultiply(std::uint64_t lhs, std::uint64_t rhs, const char* what) {
    if (lhs != 0 && rhs > std::numeric_limits<std::uint64_t>::max() / lhs) {
        throw std::invalid_argument(std::string("P2 storage size overflows ") + what);
    }
    return lhs * rhs;
}

std::uint64_t checkedAdd(std::uint64_t lhs, std::uint64_t rhs, const char* what) {
    if (rhs > std::numeric_limits<std::uint64_t>::max() - lhs) {
        throw std::invalid_argument(std::string("P2 storage size overflows ") + what);
    }
    return lhs + rhs;
}

std::uint32_t readU32At(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    if (offset + sizeof(std::uint32_t) > bytes.size()) {
        throw std::invalid_argument("P2 storage byte record is truncated");
    }
    return static_cast<std::uint32_t>(bytes[offset]) |
           (static_cast<std::uint32_t>(bytes[offset + 1]) << 8) |
           (static_cast<std::uint32_t>(bytes[offset + 2]) << 16) |
           (static_cast<std::uint32_t>(bytes[offset + 3]) << 24);
}

std::uint64_t readU64At(const std::vector<std::uint8_t>& bytes, std::size_t offset) {
    if (offset + sizeof(std::uint64_t) > bytes.size()) {
        throw std::invalid_argument("P2 storage byte record is truncated");
    }
    std::uint64_t value = 0;
    for (std::size_t index = 0; index < sizeof(std::uint64_t); ++index) {
        value |= static_cast<std::uint64_t>(bytes[offset + index]) << (index * 8);
    }
    return value;
}

void writeU32At(std::uint32_t value, std::vector<std::uint8_t>& bytes, std::size_t offset) {
    if (offset + sizeof(std::uint32_t) > bytes.size()) {
        throw std::invalid_argument("P2 storage byte record is truncated");
    }
    for (std::size_t index = 0; index < sizeof(std::uint32_t); ++index) {
        bytes[offset + index] = static_cast<std::uint8_t>(value >> (index * 8));
    }
}

void writeU64At(std::uint64_t value, std::vector<std::uint8_t>& bytes, std::size_t offset) {
    if (offset + sizeof(std::uint64_t) > bytes.size()) {
        throw std::invalid_argument("P2 storage byte record is truncated");
    }
    for (std::size_t index = 0; index < sizeof(std::uint64_t); ++index) {
        bytes[offset + index] = static_cast<std::uint8_t>(value >> (index * 8));
    }
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

} // namespace

CoreStorageV5Config CoreStorageV5::normalize_(CoreStorageV5Config config) {
    if (config.neurons == 0) throw std::invalid_argument("P2 storage neurons must be positive");
    if (config.max_delta_entries_per_neuron == 0) {
        throw std::invalid_argument("P2 storage max_delta_entries_per_neuron must be positive");
    }
    if (config.index_bytes == 0 || config.route_bytes == 0) {
        throw std::invalid_argument("P2 storage index and route regions must be positive");
    }
    const auto state_bytes = regionBytes_(config, AddressSpaceId::CoreState);
    const auto delta_bytes = regionBytes_(config, AddressSpaceId::CoreDelta);
    if (config.state_sram.capacity_bytes == 0) config.state_sram.capacity_bytes = state_bytes;
    if (config.delta_sram.capacity_bytes == 0) config.delta_sram.capacity_bytes = delta_bytes;
    if (config.index_sram.capacity_bytes == 0) config.index_sram.capacity_bytes = config.index_bytes;
    if (config.route_sram.capacity_bytes == 0) config.route_sram.capacity_bytes = config.route_bytes;
    if (config.state_sram.capacity_bytes < state_bytes || config.delta_sram.capacity_bytes < delta_bytes ||
        config.index_sram.capacity_bytes < config.index_bytes || config.route_sram.capacity_bytes < config.route_bytes) {
        throw std::invalid_argument("P2 storage SRAM capacity is smaller than its typed region");
    }
    return config;
}

std::uint64_t CoreStorageV5::regionBytes_(const CoreStorageV5Config& config, AddressSpaceId space) {
    switch (space) {
    case AddressSpaceId::CoreState:
        return checkedMultiply(config.neurons, kStateBytes, "CoreState");
    case AddressSpaceId::CoreDelta: {
        const auto slots = checkedMultiply(config.neurons, config.max_delta_entries_per_neuron, "CoreDelta slots");
        return checkedAdd(checkedMultiply(config.neurons, kDeltaCountBytes, "CoreDelta counts"),
                         checkedMultiply(slots, kDeltaEntryBytes, "CoreDelta entries"), "CoreDelta");
    }
    case AddressSpaceId::CoreIndex:
        return config.index_bytes;
    case AddressSpaceId::PeRoute:
        return config.route_bytes;
    default:
        throw std::invalid_argument("P2 CoreStorageV5 only binds CoreState/CoreDelta/CoreIndex/PeRoute");
    }
}

std::uint64_t CoreStorageV5::regionBase_(const CoreStorageV5Config& config, AddressSpaceId space) {
    const auto owner = (space == AddressSpaceId::PeRoute) ? config.pe_id : config.core_id;
    if (owner > std::numeric_limits<std::uint64_t>::max() / kRegionStride) {
        throw std::invalid_argument("P2 storage owner overflows region base");
    }
    const auto base = checkedMultiply(owner, kRegionStride, "region base");
    const auto suffix = space == AddressSpaceId::CoreState ? 0ULL
                       : space == AddressSpaceId::CoreDelta ? kDeltaBase
                       : space == AddressSpaceId::CoreIndex ? kIndexBase
                       : kRouteBase;
    return checkedAdd(base, suffix, "region base");
}

RegionDescriptor CoreStorageV5::descriptor_(const CoreStorageV5Config& config, AddressSpaceId space) {
    return RegionDescriptor{
        space,
        space == AddressSpaceId::PeRoute ? config.pe_id : config.core_id,
        regionBase_(config, space),
        regionBytes_(config, space),
        false,
    };
}

BankedSramV5Config CoreStorageV5::sramConfig_(const CoreStorageV5Config& config, AddressSpaceId space) {
    switch (space) {
    case AddressSpaceId::CoreState: return config.state_sram;
    case AddressSpaceId::CoreDelta: return config.delta_sram;
    case AddressSpaceId::CoreIndex: return config.index_sram;
    case AddressSpaceId::PeRoute: return config.route_sram;
    default: throw std::invalid_argument("invalid P2 CoreStorageV5 region");
    }
}

CoreStorageV5::CoreStorageV5(const CoreStorageV5Config& config)
    : config_(normalize_(config)),
      state_(descriptor_(config_, AddressSpaceId::CoreState), sramConfig_(config_, AddressSpaceId::CoreState)),
      delta_(descriptor_(config_, AddressSpaceId::CoreDelta), sramConfig_(config_, AddressSpaceId::CoreDelta)),
      index_(descriptor_(config_, AddressSpaceId::CoreIndex), sramConfig_(config_, AddressSpaceId::CoreIndex)),
      route_(descriptor_(config_, AddressSpaceId::PeRoute), sramConfig_(config_, AddressSpaceId::PeRoute)) {}

CoreStorageV5::Region& CoreStorageV5::region_(AddressSpaceId space) {
    switch (space) {
    case AddressSpaceId::CoreState: return state_;
    case AddressSpaceId::CoreDelta: return delta_;
    case AddressSpaceId::CoreIndex: return index_;
    case AddressSpaceId::PeRoute: return route_;
    default: throw std::invalid_argument("invalid P2 CoreStorageV5 region");
    }
}

const CoreStorageV5::Region& CoreStorageV5::region_(AddressSpaceId space) const {
    return const_cast<CoreStorageV5*>(this)->region_(space);
}

const RegionDescriptor& CoreStorageV5::region(AddressSpaceId space) const {
    return region_(space).descriptor;
}

const BankedSramV5Stats& CoreStorageV5::stats(AddressSpaceId space) const {
    return region_(space).sram.stats();
}

bool CoreStorageV5::transfer_(Region& region, std::uint64_t byte_offset,
                               const std::vector<std::uint8_t>& input, bool write,
                               std::vector<std::uint8_t>& output) {
    const ::SnnDL::v5::TypedAddress address{region.descriptor.space, region.descriptor.owner_id, byte_offset};
    std::uint64_t physical = 0;
    if (!::SnnDL::v5::resolveRegionAddress(address, region.descriptor, physical)) return false;
    (void)physical;

    BankedSramV5Request request;
    request.request_id = region.next_request_id++;
    request.address = byte_offset;
    request.data = input;
    request.write = write;
    BankedSramV5Response rejection;
    if (!region.sram.accept(request, region.cycle, &rejection)) return false;

    // The binding exposes a blocking typed operation to CorePipeline.  The
    // underlying model still performs finite-queue admission, bank service,
    // latency and byte-backed completion before this call returns.
    for (std::uint64_t guard = 0; guard < 1000000; ++guard) {
        region.sram.tick(region.cycle);
        auto responses = region.sram.takeResponses();
        for (auto& response : responses) {
            if (response.request_id != request.request_id) continue;
            if (!response.accepted || !response.completed) return false;
            output = std::move(response.data);
            return true;
        }
        if (region.cycle == std::numeric_limits<std::uint64_t>::max()) return false;
        ++region.cycle;
    }
    return false;
}

bool CoreStorageV5::readBytes_(Region& region, std::uint64_t byte_offset, std::size_t bytes,
                               std::vector<std::uint8_t>& output) {
    if (bytes == 0) return false;
    return transfer_(region, byte_offset, std::vector<std::uint8_t>(bytes, 0), false, output);
}

bool CoreStorageV5::writeBytes_(Region& region, std::uint64_t byte_offset,
                                const std::vector<std::uint8_t>& input) {
    if (input.empty()) return false;
    std::vector<std::uint8_t> ignored;
    return transfer_(region, byte_offset, input, true, ignored);
}

void CoreStorageV5::resetTimestep() {
    std::vector<std::uint8_t> zero(kDeltaCountBytes, 0);
    for (std::uint32_t neuron = 0; neuron < config_.neurons; ++neuron) {
        if (!writeBytes_(delta_, deltaCountOffset_(neuron), zero)) {
            throw std::logic_error("P2 CoreDelta reset request failed");
        }
    }
}

std::uint32_t CoreStorageV5::readU32_(const std::vector<std::uint8_t>& bytes) const {
    return readU32At(bytes, 0);
}

void CoreStorageV5::writeU32_(std::uint32_t value, std::vector<std::uint8_t>& bytes) const {
    bytes.assign(sizeof(std::uint32_t), 0);
    writeU32At(value, bytes, 0);
}

void CoreStorageV5::encodeState_(const LifNeuronState& state, std::vector<std::uint8_t>& bytes) {
    bytes.assign(kStateBytes, 0);
    writeU32At(bitsOfFloat(state.membrane), bytes, 0);
    writeU32At(state.refractory, bytes, sizeof(std::uint32_t));
}

LifNeuronState CoreStorageV5::decodeState_(const std::vector<std::uint8_t>& bytes) {
    return LifNeuronState{floatOfBits(readU32At(bytes, 0)), readU32At(bytes, sizeof(std::uint32_t))};
}

void CoreStorageV5::encodeDelta_(const RetireEntry& entry, std::vector<std::uint8_t>& bytes) {
    bytes.assign(kDeltaEntryBytes, 0);
    writeU64At(entry.key.source_event_seq, bytes, 0);
    writeU64At(entry.key.edge_ordinal, bytes, 8);
    writeU64At(entry.timestep, bytes, 16);
    writeU32At(bitsOfFloat(entry.weight), bytes, 24);
}

RetireEntry CoreStorageV5::decodeDelta_(std::uint32_t post_neuron,
                                        const std::vector<std::uint8_t>& bytes) {
    RetireEntry entry;
    entry.key.post_neuron = post_neuron;
    entry.key.source_event_seq = readU64At(bytes, 0);
    entry.key.edge_ordinal = readU64At(bytes, 8);
    entry.timestep = readU64At(bytes, 16);
    entry.weight = floatOfBits(readU32At(bytes, 24));
    return entry;
}

std::uint64_t CoreStorageV5::deltaCountOffset_(std::uint32_t neuron) const {
    return checkedMultiply(neuron, kDeltaCountBytes, "CoreDelta count offset");
}

std::uint64_t CoreStorageV5::deltaEntryOffset_(std::uint32_t neuron, std::uint32_t slot) const {
    const auto slots_before = checkedMultiply(neuron, config_.max_delta_entries_per_neuron, "CoreDelta entry offset");
    const auto slot_index = checkedAdd(slots_before, slot, "CoreDelta entry slot");
    return checkedAdd(checkedMultiply(config_.neurons, kDeltaCountBytes, "CoreDelta entry base"),
                     checkedMultiply(slot_index, kDeltaEntryBytes, "CoreDelta entry offset"),
                     "CoreDelta entry offset");
}

bool CoreStorageV5::readState(std::uint32_t neuron, LifNeuronState& state) {
    if (neuron >= config_.neurons) return false;
    std::vector<std::uint8_t> bytes;
    if (!readBytes_(state_, checkedMultiply(neuron, kStateBytes, "CoreState offset"), kStateBytes, bytes) ||
        bytes.size() != kStateBytes) return false;
    state = decodeState_(bytes);
    return true;
}

bool CoreStorageV5::writeState(std::uint32_t neuron, const LifNeuronState& state) {
    if (neuron >= config_.neurons) return false;
    std::vector<std::uint8_t> bytes;
    encodeState_(state, bytes);
    return writeBytes_(state_, checkedMultiply(neuron, kStateBytes, "CoreState offset"), bytes);
}

bool CoreStorageV5::appendDelta(const RetireEntry& entry) {
    if (entry.key.post_neuron >= config_.neurons) return false;
    std::vector<std::uint8_t> count_bytes;
    if (!readBytes_(delta_, deltaCountOffset_(entry.key.post_neuron), kDeltaCountBytes, count_bytes)) return false;
    const auto count = readU32_(count_bytes);
    if (count >= config_.max_delta_entries_per_neuron) return false;
    std::vector<std::uint8_t> entry_bytes;
    encodeDelta_(entry, entry_bytes);
    if (!writeBytes_(delta_, deltaEntryOffset_(entry.key.post_neuron, count), entry_bytes)) return false;
    writeU32_(count + 1, count_bytes);
    return writeBytes_(delta_, deltaCountOffset_(entry.key.post_neuron), count_bytes);
}

bool CoreStorageV5::readDeltaEntries(std::uint32_t neuron, std::vector<RetireEntry>& entries) {
    entries.clear();
    if (neuron >= config_.neurons) return false;
    std::vector<std::uint8_t> count_bytes;
    if (!readBytes_(delta_, deltaCountOffset_(neuron), kDeltaCountBytes, count_bytes)) return false;
    const auto count = readU32_(count_bytes);
    if (count > config_.max_delta_entries_per_neuron) return false;
    entries.reserve(count);
    for (std::uint32_t slot = 0; slot < count; ++slot) {
        std::vector<std::uint8_t> bytes;
        if (!readBytes_(delta_, deltaEntryOffset_(neuron, slot), kDeltaEntryBytes, bytes)) return false;
        entries.push_back(decodeDelta_(neuron, bytes));
    }
    std::stable_sort(entries.begin(), entries.end(), [](const RetireEntry& lhs, const RetireEntry& rhs) {
        return lhs.key < rhs.key;
    });
    return true;
}

bool CoreStorageV5::clearDelta(std::uint32_t neuron) {
    if (neuron >= config_.neurons) return false;
    std::vector<std::uint8_t> zero(kDeltaCountBytes, 0);
    return writeBytes_(delta_, deltaCountOffset_(neuron), zero);
}

bool CoreStorageV5::readIndex(std::uint64_t byte_offset, std::size_t bytes,
                              std::vector<std::uint8_t>& data) {
    return readBytes_(index_, byte_offset, bytes, data);
}

bool CoreStorageV5::readRoute(std::uint64_t byte_offset, std::size_t bytes,
                              std::vector<std::uint8_t>& data) {
    return readBytes_(route_, byte_offset, bytes, data);
}

} // namespace v5
} // namespace SnnDL
} // namespace SST
