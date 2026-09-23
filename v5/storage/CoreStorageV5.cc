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

// The execution SRAM record family has no SST-time coordinate: issue,
// service and completion cycles below are region-local counters that this
// binding advances itself.  The only workload coordinate a record can carry
// is the pipeline timestep declared through CoreStorageV5::beginTimestep(),
// and an access made outside a declared timestep stays explicitly null rather
// than being reported as timestep 0.
std::string timestepField(bool have_timestep, std::uint64_t timestep) {
    return have_timestep ? std::to_string(timestep) : std::string("null");
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
      layout_(StateDeltaLayoutV5::bind(config_)),
      state_(descriptor_(config_, AddressSpaceId::CoreState), sramConfig_(config_, AddressSpaceId::CoreState)),
      delta_(descriptor_(config_, AddressSpaceId::CoreDelta), sramConfig_(config_, AddressSpaceId::CoreDelta)),
      index_(descriptor_(config_, AddressSpaceId::CoreIndex), sramConfig_(config_, AddressSpaceId::CoreIndex)),
      route_(descriptor_(config_, AddressSpaceId::PeRoute), sramConfig_(config_, AddressSpaceId::PeRoute)) {
    if (layout_.identityOrder()) {
        for (std::uint32_t neuron = 0; neuron < config_.neurons; ++neuron) {
            const auto spans = layout_.stateSpans(neuron);
            if (spans.size() != 1 || spans[0].offset != static_cast<std::uint64_t>(neuron) * kStateBytes ||
                spans[0].bytes != kStateBytes || layout_.deltaCountOffset(neuron) != deltaCountOffset_(neuron)) {
                throw std::logic_error("CA-5B identity descriptor diverges from the historical state or count offset");
            }
            for (std::uint32_t slot = 0; slot < config_.max_delta_entries_per_neuron; ++slot) {
                if (layout_.deltaEntryOffset(neuron, slot) != deltaEntryOffset_(neuron, slot)) {
                    throw std::logic_error("CA-5B identity descriptor diverges from the historical delta entry offset");
                }
            }
        }
    }
    if (!config_.trace_json.empty()) {
        trace_stream_.open(config_.trace_json, std::ios::out | std::ios::trunc);
        if (!trace_stream_.good()) {
            throw std::invalid_argument("cannot open CoreStorageV5 execution trace: " + config_.trace_json);
        }
    }
}

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
    // Legacy compatibility profile only.  The async path must not spin a
    // private region.cycle inside one CorePipeline tick.
    requireLegacy_("transfer");
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
    const auto issue_cycle = region.cycle;
    if (!region.sram.accept(request, region.cycle, &rejection)) {
        if (trace_stream_.good()) {
            trace_stream_ << "{\"schema_version\":\"snndl-execution-sram-request/v1\""
                          << ",\"request_id\":" << request.request_id
                          << ",\"core_id\":" << config_.core_id
                          << ",\"pe_id\":" << config_.pe_id
                          << ",\"region_id\":" << static_cast<unsigned>(region.descriptor.space)
                          << ",\"address\":" << byte_offset << ",\"bytes\":" << input.size()
                          << ",\"write\":" << (write ? "true" : "false")
                          << ",\"issue_cycle\":" << issue_cycle
                          << ",\"timestep\":" << timestepField(have_timestep_, current_timestep_)
                          << ",\"accepted\":false,\"completed\":false"
                          << ",\"retryable\":" << (rejection.retryable ? "true" : "false")
                          << ",\"bank\":" << rejection.bank
                          << ",\"service_cycle\":" << rejection.service_cycle
                          << ",\"completion_cycle\":" << rejection.completion_cycle << "}\n";
        }
        return false;
    }

    // The binding exposes a blocking typed operation to CorePipeline.  The
    // underlying model still performs finite-queue admission, bank service,
    // latency and byte-backed completion before this call returns.
    for (std::uint64_t guard = 0; guard < 1000000; ++guard) {
        region.sram.tick(region.cycle);
        auto responses = region.sram.takeResponses();
        for (auto& response : responses) {
            if (response.request_id != request.request_id) continue;
            if (!response.accepted || !response.completed) return false;
            if (trace_stream_.good()) {
                trace_stream_ << "{\"schema_version\":\"snndl-execution-sram-request/v1\""
                              << ",\"request_id\":" << request.request_id
                              << ",\"core_id\":" << config_.core_id
                              << ",\"pe_id\":" << config_.pe_id
                              << ",\"region_id\":" << static_cast<unsigned>(region.descriptor.space)
                              << ",\"address\":" << byte_offset << ",\"bytes\":" << input.size()
                              << ",\"write\":" << (write ? "true" : "false")
                              << ",\"issue_cycle\":" << issue_cycle
                              << ",\"timestep\":" << timestepField(have_timestep_, current_timestep_)
                              << ",\"accepted\":true,\"completed\":true"
                              << ",\"retryable\":false,\"bank\":" << response.bank
                              << ",\"port\":" << response.port
                              << ",\"service_cycle\":" << response.service_cycle
                              << ",\"completion_cycle\":" << response.completion_cycle << "}\n";
                trace_stream_.flush();
            }
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

void CoreStorageV5::beginTimestep(std::uint64_t timestep) {
    if (have_timestep_ && timestep < current_timestep_) {
        throw std::invalid_argument("P2 storage timestep must not move backwards");
    }
    have_timestep_ = true;
    current_timestep_ = timestep;
}

void CoreStorageV5::resetTimestep() {
    requireLegacy_("resetTimestep");
    std::vector<std::uint8_t> zero(kDeltaCountBytes, 0);
    for (std::uint32_t neuron = 0; neuron < config_.neurons; ++neuron) {
        if (!writeBytes_(delta_, deltaCountByteOffset(neuron), zero)) {
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

bool CoreStorageV5::readLogicalState_(std::uint32_t neuron, std::vector<std::uint8_t>& logical) {
    logical.assign(kStateBytes, 0);
    for (const auto& span : layout_.stateSpans(neuron)) {
        std::vector<std::uint8_t> part;
        if (!readBytes_(state_, span.offset, span.bytes, part) || part.size() != span.bytes ||
            span.logical_offset + span.bytes > logical.size()) {
            return false;
        }
        std::copy(part.begin(), part.end(), logical.begin() + span.logical_offset);
    }
    return true;
}

bool CoreStorageV5::writeLogicalState_(std::uint32_t neuron, const std::vector<std::uint8_t>& logical) {
    if (logical.size() != kStateBytes) return false;
    for (const auto& span : layout_.stateSpans(neuron)) {
        if (span.logical_offset + span.bytes > logical.size()) return false;
        const std::vector<std::uint8_t> slice(logical.begin() + span.logical_offset,
                                              logical.begin() + span.logical_offset + span.bytes);
        if (!writeBytes_(state_, span.offset, slice)) return false;
    }
    return true;
}

bool CoreStorageV5::peekLogicalState_(std::uint32_t neuron, std::vector<std::uint8_t>& logical) const {
    logical.assign(kStateBytes, 0);
    for (const auto& span : layout_.stateSpans(neuron)) {
        std::vector<std::uint8_t> part;
        if (!copyCompleted_(AddressSpaceId::CoreState, span.offset, span.bytes, part) || part.size() != span.bytes ||
            span.logical_offset + span.bytes > logical.size()) {
            return false;
        }
        std::copy(part.begin(), part.end(), logical.begin() + span.logical_offset);
    }
    return true;
}

bool CoreStorageV5::readState(std::uint32_t neuron, LifNeuronState& state) {
    if (neuron >= config_.neurons) return false;
    std::vector<std::uint8_t> bytes;
    if (!readLogicalState_(neuron, bytes)) return false;
    state = decodeState_(bytes);
    return true;
}

bool CoreStorageV5::writeState(std::uint32_t neuron, const LifNeuronState& state) {
    if (neuron >= config_.neurons) return false;
    std::vector<std::uint8_t> bytes;
    encodeState_(state, bytes);
    return writeLogicalState_(neuron, bytes);
}

bool CoreStorageV5::readCubaLifState(std::uint32_t neuron, CubaLifNeuronState& state) {
    if (neuron >= config_.neurons) return false;
    std::vector<std::uint8_t> bytes;
    if (!readLogicalState_(neuron, bytes)) return false;
    return CubaLifNeuronOp::decodeState(bytes.data(), bytes.size(), state);
}

bool CoreStorageV5::writeCubaLifState(std::uint32_t neuron, const CubaLifNeuronState& state) {
    if (neuron >= config_.neurons) return false;
    std::array<std::uint8_t, CubaLifNeuronOp::kStateBytes> bytes{};
    CubaLifNeuronOp::encodeState(state, bytes);
    return writeLogicalState_(neuron, std::vector<std::uint8_t>(bytes.begin(), bytes.end()));
}

bool CoreStorageV5::appendDelta(const RetireEntry& entry) {
    if (entry.key.post_neuron >= config_.neurons) return false;
    std::vector<std::uint8_t> count_bytes;
    if (!readBytes_(delta_, deltaCountByteOffset(entry.key.post_neuron), kDeltaCountBytes, count_bytes)) return false;
    const auto count = readU32_(count_bytes);
    if (count >= config_.max_delta_entries_per_neuron) return false;
    std::vector<std::uint8_t> entry_bytes;
    encodeDelta_(entry, entry_bytes);
    if (!writeBytes_(delta_, deltaEntryByteOffset(entry.key.post_neuron, count), entry_bytes)) return false;
    writeU32_(count + 1, count_bytes);
    return writeBytes_(delta_, deltaCountByteOffset(entry.key.post_neuron), count_bytes);
}

bool CoreStorageV5::readDeltaEntries(std::uint32_t neuron, std::vector<RetireEntry>& entries) {
    entries.clear();
    if (neuron >= config_.neurons) return false;
    std::vector<std::uint8_t> count_bytes;
    if (!readBytes_(delta_, deltaCountByteOffset(neuron), kDeltaCountBytes, count_bytes)) return false;
    const auto count = readU32_(count_bytes);
    if (count > config_.max_delta_entries_per_neuron) return false;
    entries.reserve(count);
    for (std::uint32_t slot = 0; slot < count; ++slot) {
        std::vector<std::uint8_t> bytes;
        if (!readBytes_(delta_, deltaEntryByteOffset(neuron, slot), kDeltaEntryBytes, bytes)) return false;
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
    return writeBytes_(delta_, deltaCountByteOffset(neuron), zero);
}

bool CoreStorageV5::readIndex(std::uint64_t byte_offset, std::size_t bytes,
                              std::vector<std::uint8_t>& data) {
    try {
        return readBytes_(index_, layout_.indexOffset(byte_offset, bytes), bytes, data);
    } catch (const std::invalid_argument&) {
        return false;
    }
}

bool CoreStorageV5::readRoute(std::uint64_t byte_offset, std::size_t bytes,
                              std::vector<std::uint8_t>& data) {
    try {
        return readBytes_(route_, layout_.routeOffset(byte_offset, bytes), bytes, data);
    } catch (const std::invalid_argument&) {
        return false;
    }
}

void CoreStorageV5::requireAsync_(const char* what) const {
    if (config_.execution_profile != CoreStorageExecutionProfile::AsyncCompletion) {
        throw std::logic_error(std::string("CA-5A ") + what + " requires the async completion profile");
    }
}

void CoreStorageV5::requireLegacy_(const char* what) const {
    if (config_.execution_profile != CoreStorageExecutionProfile::LegacyBlocking) {
        throw std::logic_error(std::string("CA-5A async profile forbids blocking ") + what);
    }
}

std::uint64_t CoreStorageV5::advanceCount(AddressSpaceId space) const {
    return region_(space).advance_count;
}

std::vector<StateDeltaLayoutV5::Span> CoreStorageV5::stateSpans(std::uint32_t neuron) const {
    return layout_.stateSpans(neuron);
}

std::uint64_t CoreStorageV5::stateByteOffset(std::uint32_t neuron) const {
    const auto spans = stateSpans(neuron);
    if (spans.size() != 1 || spans.front().bytes != kStateBytes || spans.front().logical_offset != 0) {
        throw std::logic_error("CA-5B state object is not one contiguous record");
    }
    return spans.front().offset;
}

std::uint64_t CoreStorageV5::deltaCountByteOffset(std::uint32_t neuron) const {
    return layout_.deltaCountOffset(neuron);
}

std::uint64_t CoreStorageV5::deltaEntryByteOffset(std::uint32_t neuron, std::uint32_t slot) const {
    return layout_.deltaEntryOffset(neuron, slot);
}

std::uint64_t CoreStorageV5::requireIndexOffset(std::uint64_t byte_offset, std::size_t bytes) const {
    return layout_.indexOffset(byte_offset, bytes);
}

std::uint64_t CoreStorageV5::requireRouteOffset(std::uint64_t byte_offset, std::size_t bytes) const {
    return layout_.routeOffset(byte_offset, bytes);
}

std::vector<std::uint8_t> CoreStorageV5::encodeLifRecord(const LifNeuronState& state) {
    std::vector<std::uint8_t> bytes;
    encodeState_(state, bytes);
    return bytes;
}

LifNeuronState CoreStorageV5::decodeLifRecord(const std::vector<std::uint8_t>& bytes) {
    return decodeState_(bytes);
}

std::vector<std::uint8_t> CoreStorageV5::encodeDeltaRecord(const RetireEntry& entry) {
    std::vector<std::uint8_t> bytes;
    encodeDelta_(entry, bytes);
    return bytes;
}

RetireEntry CoreStorageV5::decodeDeltaRecord(std::uint32_t post_neuron, const std::vector<std::uint8_t>& bytes) {
    return decodeDelta_(post_neuron, bytes);
}

std::vector<std::uint8_t> CoreStorageV5::encodeCountRecord(std::uint32_t value) {
    std::vector<std::uint8_t> bytes;
    bytes.assign(kDeltaCountBytes, 0);
    writeU32At(value, bytes, 0);
    return bytes;
}

std::uint32_t CoreStorageV5::decodeCountRecord(const std::vector<std::uint8_t>& bytes) {
    return readU32At(bytes, 0);
}

bool CoreStorageV5::copyCompleted_(AddressSpaceId space, std::uint64_t byte_offset, std::size_t bytes,
                                   std::vector<std::uint8_t>& out) const {
    return region_(space).sram.copyCompletedBytes(byte_offset, bytes, out);
}

bool CoreStorageV5::readCompletedState(std::uint32_t neuron, LifNeuronState& state) const {
    if (neuron >= config_.neurons) return false;
    std::vector<std::uint8_t> bytes;
    if (!peekLogicalState_(neuron, bytes)) return false;
    state = decodeState_(bytes);
    return true;
}

bool CoreStorageV5::readCompletedCubaLifState(std::uint32_t neuron, CubaLifNeuronState& state) const {
    if (neuron >= config_.neurons) return false;
    std::vector<std::uint8_t> bytes;
    if (!peekLogicalState_(neuron, bytes)) return false;
    return CubaLifNeuronOp::decodeState(bytes.data(), bytes.size(), state);
}

void CoreStorageV5::beginEpoch() {
    if (!drained()) {
        throw std::logic_error("CA-5A refuses a new epoch while SRAM requests are outstanding");
    }
    if (epoch_ == std::numeric_limits<std::uint64_t>::max()) {
        throw std::logic_error("CA-5A storage epoch overflow");
    }
    ++epoch_;
}

std::size_t CoreStorageV5::completedUnconsumed_(AddressSpaceId space) const {
    std::size_t count = 0;
    for (const auto& entry : ops_) {
        if (entry.second.region == space && entry.second.phase == StorageRequestPhase::Completed) ++count;
    }
    return count;
}

bool CoreStorageV5::responseSaturated_(const Region& region) const {
    const auto held = completedUnconsumed_(region.descriptor.space) + region.sram.responses();
    const auto occupied = region.sram.pending() + region.sram.inFlight() + held;
    return region.sram.pending() + region.sram.inFlight() >= region.sram.config().request_queue_entries ||
           occupied >= region.sram.config().response_queue_entries;
}

void CoreStorageV5::notePeak_() {
    pending_peak_ = std::max<std::uint64_t>(pending_peak_, static_cast<std::uint64_t>(ops_.size()));
}

void CoreStorageV5::traceOp_(const OpRecord& op, bool accepted, bool completed) const {
    if (!trace_stream_.good()) return;
    const char* stall = op.stall_reason.empty() ? "none" : op.stall_reason.c_str();
    trace_stream_ << "{\"schema_version\":\"snndl-execution-sram-request/v1\""
                  << ",\"timebase\":\"core_pipeline_cycle\""
                  << ",\"request_id\":" << op.physical_request_id
                  << ",\"logical_operation_id\":" << op.logical_operation_id
                  << ",\"attempt_id\":" << op.attempt_id
                  << ",\"epoch\":" << op.epoch
                  << ",\"core_id\":" << config_.core_id
                  << ",\"pe_id\":" << config_.pe_id
                  << ",\"region_id\":" << static_cast<unsigned>(op.region)
                  << ",\"address\":" << op.offset
                  << ",\"bytes\":" << op.payload.size()
                  << ",\"write\":" << (op.write ? "true" : "false")
                  << ",\"issue_cycle\":" << (accepted ? op.accept_cycle : 0)
                  << ",\"service_cycle\":" << op.service_cycle
                  << ",\"completion_cycle\":" << op.completion_cycle
                  << ",\"consume_cycle\":" << op.consume_cycle
                  << ",\"timestep\":" << timestepField(have_timestep_, current_timestep_)
                  << ",\"accepted\":" << (accepted ? "true" : "false")
                  << ",\"completed\":" << (completed ? "true" : "false")
                  << ",\"retryable\":" << (!accepted && !op.terminal_reject ? "true" : "false")
                  << ",\"bank\":" << op.bank
                  << ",\"port\":" << op.port
                  << ",\"stall_reason\":\"" << stall << "\""
                  << ",\"pending_peak\":" << pending_peak_ << "}\n";
    trace_stream_.flush();
}

StorageCompletion CoreStorageV5::completionFrom_(const OpRecord& op) const {
    StorageCompletion completion;
    completion.token = op.token;
    completion.epoch = op.epoch;
    completion.region = op.region;
    completion.logical_operation_id = op.logical_operation_id;
    completion.attempt_id = op.attempt_id;
    completion.physical_request_id = op.physical_request_id;
    completion.issue_cycle = op.accept_cycle;
    completion.service_cycle = op.service_cycle;
    completion.completion_cycle = op.completion_cycle;
    completion.eligible_service_cycle = op.eligible_service_cycle;
    completion.bank = op.bank;
    completion.port = op.port;
    completion.write = op.write;
    completion.ok = op.ok;
    completion.data = op.response;
    completion.stall_reason = op.stall_reason.empty() ? "none" : op.stall_reason;
    return completion;
}

void CoreStorageV5::tryAdmit_(OpRecord& op, std::uint64_t now) {
    if (op.phase != StorageRequestPhase::Created || op.terminal_reject) return;
    Region& region = region_(op.region);
    ++op.attempt_id;
    if (responseSaturated_(region)) {
        op.stall_reason = "queue_full";
        op.bank = region.sram.bankForAddress(op.offset);
        notePeak_();
        traceOp_(op, false, false);
        return;
    }
    BankedSramV5Request request;
    request.request_id = region.next_request_id++;
    request.address = op.offset;
    request.data = op.payload;
    request.write = op.write;
    BankedSramV5Response rejection;
    op.physical_request_id = request.request_id;
    if (!region.sram.accept(request, now, &rejection)) {
        op.stall_reason = rejection.retryable ? "queue_full"
                        : rejection.reject == BankedSramV5Reject::Capacity ? "capacity" : "invalid";
        op.bank = rejection.bank;
        op.terminal_reject = !rejection.retryable;
        notePeak_();
        traceOp_(op, false, false);
        if (op.terminal_reject) {
            op.phase = StorageRequestPhase::Completed;
            op.ok = false;
        }
        return;
    }
    op.phase = StorageRequestPhase::Accepted;
    op.accept_cycle = now;
    op.eligible_service_cycle = (has_advance_ && last_advance_now_ == now) ? now + 1 : now;
    op.bank = region.sram.bankForAddress(op.offset);
    op.stall_reason.clear();
    region.physical_to_token.emplace(request.request_id, op.token.id);
    notePeak_();
}

StorageSubmitResult CoreStorageV5::submit_(AddressSpaceId space, std::uint64_t byte_offset,
                                           std::vector<std::uint8_t> data, bool write, std::uint64_t now) {
    requireAsync_("submit");
    StorageSubmitResult result;
    Region& region = region_(space);
    const ::SnnDL::v5::TypedAddress address{region.descriptor.space, region.descriptor.owner_id, byte_offset};
    std::uint64_t physical = 0;
    if (data.empty() || !::SnnDL::v5::resolveRegionAddress(address, region.descriptor, physical) ||
        byte_offset > region.descriptor.size_bytes ||
        data.size() > region.descriptor.size_bytes - byte_offset) {
        result.reject = BankedSramV5Reject::Invalid;
        result.stall_reason = "invalid";
        return result;
    }
    (void)physical;
    OpRecord op;
    op.token.id = next_token_++;
    op.epoch = epoch_;
    op.region = space;
    op.logical_operation_id = next_logical_++;
    op.phase = StorageRequestPhase::Created;
    op.write = write;
    op.offset = byte_offset;
    op.payload = std::move(data);
    const auto token = op.token.id;
    ops_.emplace(token, std::move(op));
    tryAdmit_(ops_.at(token), now);
    const auto& stored = ops_.at(token);
    result.token = stored.token;
    result.accepted = stored.phase == StorageRequestPhase::Accepted;
    result.retryable = !result.accepted && !stored.terminal_reject;
    result.reject = result.accepted ? BankedSramV5Reject::None
                  : stored.stall_reason == "capacity" ? BankedSramV5Reject::Capacity
                  : stored.stall_reason == "invalid" ? BankedSramV5Reject::Invalid
                  : BankedSramV5Reject::QueueFull;
    result.stall_reason = stored.stall_reason.empty() ? "none" : stored.stall_reason;
    if (stored.terminal_reject) {
        ops_.erase(token);
        result.token = {};
    }
    return result;
}

StorageSubmitResult CoreStorageV5::submitRead(AddressSpaceId space, std::uint64_t byte_offset,
                                              std::size_t bytes, std::uint64_t now) {
    return submit_(space, byte_offset, std::vector<std::uint8_t>(bytes, 0), false, now);
}

StorageSubmitResult CoreStorageV5::submitWrite(AddressSpaceId space, std::uint64_t byte_offset,
                                               const std::vector<std::uint8_t>& data, std::uint64_t now) {
    return submit_(space, byte_offset, data, true, now);
}

void CoreStorageV5::admit(std::uint64_t now) {
    requireAsync_("admit");
    std::vector<std::uint64_t> created;
    created.reserve(ops_.size());
    for (const auto& entry : ops_) {
        if (entry.second.phase == StorageRequestPhase::Created && !entry.second.terminal_reject) {
            created.push_back(entry.first);
        }
    }
    for (const auto token : created) tryAdmit_(ops_.at(token), now);
}

void CoreStorageV5::advance(std::uint64_t now) {
    requireAsync_("advance");
    if (has_advance_) {
        if (now == last_advance_now_) {
            throw std::logic_error("CA-5A advanced SRAM twice in one logical tick");
        }
        if (now != last_advance_now_ + 1) {
            throw std::logic_error("CA-5A advance must move one logical cycle at a time");
        }
    }
    has_advance_ = true;
    last_advance_now_ = now;
    const AddressSpaceId spaces[] = {
        AddressSpaceId::CoreState, AddressSpaceId::CoreDelta,
        AddressSpaceId::CoreIndex, AddressSpaceId::PeRoute};
    for (const auto space : spaces) {
        auto& region = region_(space);
        region.sram.tick(now);
        ++region.advance_count;
        for (auto& entry : ops_) {
            auto& op = entry.second;
            if (op.region != space) continue;
            if (op.phase != StorageRequestPhase::Accepted && op.phase != StorageRequestPhase::Servicing) continue;
            if (region.sram.locate(op.physical_request_id) == BankedSramV5::Location::InFlight) {
                op.phase = StorageRequestPhase::Servicing;
            }
        }
    }
    notePeak_();
}

std::vector<StorageCompletion> CoreStorageV5::takeCompletions() {
    requireAsync_("takeCompletions");
    std::vector<StorageCompletion> ready;
    const AddressSpaceId spaces[] = {
        AddressSpaceId::CoreState, AddressSpaceId::CoreDelta,
        AddressSpaceId::CoreIndex, AddressSpaceId::PeRoute};
    for (const auto space : spaces) {
        auto& region = region_(space);
        for (auto& response : region.sram.takeResponses()) {
            const auto found = region.physical_to_token.find(response.request_id);
            if (found == region.physical_to_token.end()) {
                throw std::logic_error("CA-5A SRAM completion has no CoreStorage token");
            }
            auto& op = ops_.at(found->second);
            op.phase = StorageRequestPhase::Completed;
            op.service_cycle = response.service_cycle;
            op.completion_cycle = response.completion_cycle;
            op.bank = response.bank;
            op.port = response.port;
            op.ok = response.accepted && response.completed;
            op.response = std::move(response.data);
            if (op.service_cycle > op.eligible_service_cycle) op.stall_reason = "port_busy";
            region.physical_to_token.erase(found);
            ready.push_back(completionFrom_(op));
        }
    }
    notePeak_();
    return ready;
}

bool CoreStorageV5::consume(StorageToken token) {
    requireAsync_("consume");
    const auto found = ops_.find(token.id);
    if (found == ops_.end() || found->second.phase != StorageRequestPhase::Completed) return false;
    found->second.phase = StorageRequestPhase::Consumed;
    found->second.consume_cycle = has_advance_ ? last_advance_now_ : 0;
    traceOp_(found->second, true, found->second.ok);
    ops_.erase(found);
    return true;
}

std::size_t CoreStorageV5::pending() const {
    return ops_.size();
}

bool CoreStorageV5::drained() const {
    if (!ops_.empty()) return false;
    const AddressSpaceId spaces[] = {
        AddressSpaceId::CoreState, AddressSpaceId::CoreDelta,
        AddressSpaceId::CoreIndex, AddressSpaceId::PeRoute};
    for (const auto space : spaces) {
        const auto& sram = region_(space).sram;
        if (sram.pending() != 0 || sram.inFlight() != 0 || sram.responses() != 0) return false;
    }
    return true;
}

} // namespace v5
} // namespace SnnDL
} // namespace SST
