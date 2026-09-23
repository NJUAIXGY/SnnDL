#include "StateDeltaLayoutV5.h"

#include "CoreStorageV5.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace SST {
namespace SnnDL {
namespace v5 {

namespace {

constexpr std::uint32_t kShaK[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

std::uint32_t rotr(std::uint32_t value, std::uint32_t bits) {
    return (value >> bits) | (value << (32 - bits));
}

class Sha256 {
public:
    Sha256() { reset(); }

    void update(const std::uint8_t* data, std::size_t length) {
        for (std::size_t index = 0; index < length; ++index) {
            block_[used_++] = data[index];
            if (used_ == 64) {
                transform(block_);
                bits_ += 512;
                used_ = 0;
            }
        }
    }

    std::string hex() {
        const std::uint64_t total_bits = bits_ + static_cast<std::uint64_t>(used_) * 8;
        block_[used_++] = 0x80;
        if (used_ > 56) {
            while (used_ < 64) block_[used_++] = 0;
            transform(block_);
            used_ = 0;
        }
        while (used_ < 56) block_[used_++] = 0;
        for (int shift = 56; shift >= 0; shift -= 8) block_[used_++] = static_cast<std::uint8_t>(total_bits >> shift);
        transform(block_);
        std::ostringstream out;
        out << std::hex;
        for (const auto word : state_) {
            for (int shift = 28; shift >= 0; shift -= 4) {
                out << ((word >> shift) & 0xf);
            }
        }
        return out.str();
    }

private:
    void reset() {
        state_[0] = 0x6a09e667;
        state_[1] = 0xbb67ae85;
        state_[2] = 0x3c6ef372;
        state_[3] = 0xa54ff53a;
        state_[4] = 0x510e527f;
        state_[5] = 0x9b05688c;
        state_[6] = 0x1f83d9ab;
        state_[7] = 0x5be0cd19;
        bits_ = 0;
        used_ = 0;
    }

    void transform(const std::uint8_t block[64]) {
        std::uint32_t words[64];
        for (int index = 0; index < 16; ++index) {
            words[index] = (std::uint32_t(block[index * 4]) << 24) | (std::uint32_t(block[index * 4 + 1]) << 16) |
                           (std::uint32_t(block[index * 4 + 2]) << 8) | std::uint32_t(block[index * 4 + 3]);
        }
        for (int index = 16; index < 64; ++index) {
            const auto s0 = rotr(words[index - 15], 7) ^ rotr(words[index - 15], 18) ^ (words[index - 15] >> 3);
            const auto s1 = rotr(words[index - 2], 17) ^ rotr(words[index - 2], 19) ^ (words[index - 2] >> 10);
            words[index] = words[index - 16] + s0 + words[index - 7] + s1;
        }
        auto a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        auto e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (int index = 0; index < 64; ++index) {
            const auto s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
            const auto ch = (e & f) ^ ((~e) & g);
            const auto temp1 = h + s1 + ch + kShaK[index] + words[index];
            const auto s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
            const auto maj = (a & b) ^ (a & c) ^ (b & c);
            const auto temp2 = s0 + maj;
            h = g;
            g = f;
            f = e;
            e = d + temp1;
            d = c;
            c = b;
            b = a;
            a = temp1 + temp2;
        }
        state_[0] += a;
        state_[1] += b;
        state_[2] += c;
        state_[3] += d;
        state_[4] += e;
        state_[5] += f;
        state_[6] += g;
        state_[7] += h;
    }

    std::uint32_t state_[8]{};
    std::uint64_t bits_ = 0;
    std::uint8_t block_[64]{};
    std::size_t used_ = 0;
};

std::string sha256Hex(const std::string& text) {
    Sha256 hasher;
    hasher.update(reinterpret_cast<const std::uint8_t*>(text.data()), text.size());
    const auto digest = hasher.hex();
    if (text.empty() && digest != "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855") {
        throw std::logic_error("CA-5B SHA-256 self-check failed");
    }
    return digest;
}

std::string canonicalDump(nlohmann::json value) {
    if (value.is_object()) value.erase("digest");
    const auto dumped = value.dump(-1, ' ', true);
    std::string canonical;
    canonical.reserve(dumped.size());
    for (std::size_t index = 0; index < dumped.size(); ++index) {
        if (dumped[index] == '\\' && index + 1 < dumped.size() && dumped[index + 1] == '/') {
            canonical.push_back('/');
            ++index;
        } else {
            canonical.push_back(dumped[index]);
        }
    }
    return canonical;
}

std::string requireDigest(const nlohmann::json& value, const char* what) {
    const auto digest = sha256Hex(canonicalDump(value));
    if (!value.contains("digest") || !value["digest"].is_string() || value["digest"].get<std::string>() != digest) {
        throw std::invalid_argument(std::string("CA-5B ") + what + " digest does not match its content");
    }
    return digest;
}

std::uint64_t requireU64(const nlohmann::json& value, const char* key) {
    if (!value.contains(key) || !value[key].is_number_unsigned() && !value[key].is_number_integer()) {
        throw std::invalid_argument(std::string("CA-5B descriptor is missing ") + key);
    }
    return value[key].get<std::uint64_t>();
}

void requireDomain(const nlohmann::json& section, const char* key, std::uint64_t count, const char* what) {
    if (!section.contains(key) || !section[key].is_array() || section[key].size() != 2 ||
        section[key][0].get<std::uint64_t>() != 0 || section[key][1].get<std::uint64_t>() != count) {
        throw std::invalid_argument(std::string("CA-5B ") + what + " domain is not the explicit half-open range");
    }
}

struct Placed {
    std::uint32_t neuron = 0;
    std::uint32_t slot = 0;
    std::uint32_t field = 0;
    std::uint64_t offset = 0;
    std::uint32_t width = 0;
};

void rejectOverlap(std::vector<std::pair<std::uint64_t, std::uint64_t>> ranges, const char* what) {
    std::sort(ranges.begin(), ranges.end());
    std::uint64_t previous = 0;
    bool have = false;
    for (const auto& range : ranges) {
        if (range.second <= range.first) throw std::invalid_argument(std::string("CA-5B ") + what + " placement is empty");
        if (have && range.first < previous) {
            throw std::invalid_argument(std::string("CA-5B ") + what + " placements alias");
        }
        previous = range.second;
        have = true;
    }
}

std::uint64_t legacyEntry(std::uint32_t neuron, std::uint32_t slot, std::uint32_t neurons, std::uint32_t slots) {
    return static_cast<std::uint64_t>(neurons) * 4u +
           (static_cast<std::uint64_t>(neuron) * slots + slot) * 32u;
}

} // namespace

StateDeltaLayoutV5 StateDeltaLayoutV5::bind(const CoreStorageV5Config& config) {
    nlohmann::json document;
    bool check_digest = false;
    std::string path;
    if (!config.descriptor_json.empty()) {
        if (config.execution_profile != CoreStorageExecutionProfile::AsyncCompletion) {
            throw std::invalid_argument("CA-5B state/delta descriptor requires AsyncCompletion");
        }
        std::ifstream input(config.descriptor_json);
        if (!input) throw std::invalid_argument("CA-5B cannot open state/delta descriptor: " + config.descriptor_json);
        document = nlohmann::json::parse(input, nullptr, true, true);
        check_digest = true;
        path = config.descriptor_json;
    } else {
        document = {
            {"schema_version", "snndl-state-delta-layout/v1"},
            {"execution_profile", "async_completion"},
            {"bank_input_rule", "region-relative-start"},
            {"neurons", config.neurons},
            {"max_delta_entries_per_neuron", config.max_delta_entries_per_neuron},
            {"owners", nlohmann::json::array({{{"core_id", config.core_id}, {"pe_id", config.pe_id}}})},
            {"state", {{"region", "CoreState"}, {"object_kind", "neuron_state"}, {"order", "neuron-major"},
                       {"version", "snndl-state-delta-layout/v1"}, {"neuron_domain", {0, config.neurons}},
                       {"element_width", 8}, {"alignment", 4}, {"capacity", config.neurons * 8},
                       {"fields", {{{"name", "word0"}, {"width", 4}, {"alignment", 4}},
                                   {{"name", "word1"}, {"width", 4}, {"alignment", 4}}}}}},
            {"delta_count", {{"region", "CoreDelta"}, {"object_kind", "delta_count"}, {"order", "neuron-major"},
                             {"version", "snndl-state-delta-layout/v1"}, {"field", "count"},
                             {"neuron_domain", {0, config.neurons}}, {"element_width", 4}, {"alignment", 4},
                             {"capacity", config.neurons * 4}}},
            {"delta_entry", {{"region", "CoreDelta"}, {"object_kind", "delta_entry"}, {"order", "neuron-major"},
                             {"version", "snndl-state-delta-layout/v1"},
                             {"neuron_domain", {0, config.neurons}},
                             {"slot_domain", {0, config.max_delta_entries_per_neuron}},
                             {"element_width", 32}, {"alignment", 4},
                             {"entry_base", config.neurons * 4},
                             {"capacity", config.neurons * config.max_delta_entries_per_neuron * 32}}},
            {"index", {{"region", "CoreIndex"}, {"object_kind", "index"}, {"order", "identity"},
                       {"version", "snndl-state-delta-layout/v1"}, {"alignment", 1},
                       {"capacity", config.index_bytes}}},
            {"route", {{"region", "PeRoute"}, {"object_kind", "route"}, {"order", "identity"},
                       {"version", "snndl-state-delta-layout/v1"}, {"alignment", 1},
                       {"capacity", config.route_bytes}}},
        };
    }

    if (document.value("schema_version", "") != "snndl-state-delta-layout/v1") {
        throw std::invalid_argument("CA-5B schema_version is not snndl-state-delta-layout/v1");
    }
    if (document.value("execution_profile", "") != "async_completion") {
        throw std::invalid_argument("CA-5B descriptor execution_profile must be async_completion");
    }
    if (document.value("bank_input_rule", "") != "region-relative-start") {
        throw std::invalid_argument("CA-5B bank input must be region-relative-start, without a global base");
    }
    const auto neurons = static_cast<std::uint32_t>(requireU64(document, "neurons"));
    const auto slots = static_cast<std::uint32_t>(requireU64(document, "max_delta_entries_per_neuron"));
    if (neurons != config.neurons || slots != config.max_delta_entries_per_neuron) {
        throw std::invalid_argument("CA-5B descriptor neuron or slot domain does not match CoreStorage");
    }
    bool owned = false;
    for (const auto& owner : document.at("owners")) {
        if (owner.at("core_id").get<std::uint32_t>() == config.core_id &&
            owner.at("pe_id").get<std::uint32_t>() == config.pe_id) {
            owned = true;
        }
    }
    if (!owned) throw std::invalid_argument("CA-5B descriptor does not own this core");

    const auto& state = document.at("state");
    const auto& counts = document.at("delta_count");
    const auto& entries = document.at("delta_entry");
    const auto& index = document.at("index");
    const auto& route = document.at("route");
    requireDomain(state, "neuron_domain", neurons, "state");
    requireDomain(counts, "neuron_domain", neurons, "delta count");
    requireDomain(entries, "neuron_domain", neurons, "delta entry");
    requireDomain(entries, "slot_domain", slots, "delta entry slot");
    if (index.value("order", "") != "identity" || route.value("order", "") != "identity") {
        throw std::invalid_argument("CA-5B index and route stay on the identity address map");
    }
    if (counts.value("order", "") != "neuron-major") {
        throw std::invalid_argument("CA-5B delta count order must stay neuron-major");
    }
    const auto state_order = state.at("order").get<std::string>();
    const auto entry_order = entries.at("order").get<std::string>();
    if (state_order != "neuron-major" && state_order != "field-major") {
        throw std::invalid_argument("CA-5B state order is not a supported candidate");
    }
    if (entry_order != "neuron-major" && entry_order != "slot-major") {
        throw std::invalid_argument("CA-5B delta entry order is not a supported candidate");
    }
    if (requireU64(state, "capacity") != static_cast<std::uint64_t>(neurons) * 8 ||
        requireU64(state, "element_width") != 8) {
        throw std::invalid_argument("CA-5B state capacity does not match the logical records");
    }
    const auto entry_base = requireU64(entries, "entry_base");
    if (entry_base != static_cast<std::uint64_t>(neurons) * 4 || requireU64(counts, "capacity") != entry_base) {
        throw std::invalid_argument("CA-5B delta count prefix does not match the entry base");
    }
    if (requireU64(entries, "capacity") != static_cast<std::uint64_t>(neurons) * slots * 32 ||
        requireU64(entries, "element_width") != 32) {
        throw std::invalid_argument("CA-5B delta entry capacity does not match the slot domain");
    }
    if (requireU64(index, "capacity") != config.index_bytes || requireU64(route, "capacity") != config.route_bytes) {
        throw std::invalid_argument("CA-5B index or route capacity does not match CoreStorage");
    }

    std::string state_digest;
    std::string count_digest;
    std::string entry_digest;
    std::string plan_digest;
    if (check_digest) {
        for (const char* name : {"state", "delta_count", "delta_entry", "index", "route"}) {
            if (document.at(name).value("version", "") != "snndl-state-delta-layout/v1") {
                throw std::invalid_argument("CA-5B section version does not match the plan");
            }
        }
        state_digest = requireDigest(state, "state");
        count_digest = requireDigest(counts, "delta count");
        entry_digest = requireDigest(entries, "delta entry");
        requireDigest(index, "index");
        requireDigest(route, "route");
        plan_digest = requireDigest(document, "plan");
    }

    auto place_from = [](const nlohmann::json& section, std::uint32_t neuron_count, std::uint32_t slot_count,
                         std::uint32_t field_count, const std::string& order, std::uint64_t base,
                         std::uint32_t width) {
        std::vector<Placed> placed;
        if (section.contains("placements")) {
            for (const auto& item : section.at("placements")) {
                placed.push_back(Placed{item.at("neuron").get<std::uint32_t>(), item.value("slot", 0u),
                                         item.value("field", 0u), item.at("offset").get<std::uint64_t>(),
                                         item.at("width").get<std::uint32_t>()});
            }
            return placed;
        }
        if (field_count == 2) {
            const auto fields = section.at("fields");
            for (std::uint32_t neuron = 0; neuron < neuron_count; ++neuron) {
                std::uint32_t local = 0;
                for (std::uint32_t field = 0; field < fields.size(); ++field) {
                    const auto field_width = fields.at(field).at("width").get<std::uint32_t>();
                    std::uint64_t offset = 0;
                    if (order == "neuron-major") offset = static_cast<std::uint64_t>(neuron) * width + local;
                    else offset = static_cast<std::uint64_t>(field) * neuron_count * field_width +
                                  static_cast<std::uint64_t>(neuron) * field_width;
                    placed.push_back(Placed{neuron, 0, field, offset, field_width});
                    local += field_width;
                }
            }
            return placed;
        }
        for (std::uint32_t neuron = 0; neuron < neuron_count; ++neuron) {
            for (std::uint32_t slot = 0; slot < slot_count; ++slot) {
                std::uint64_t offset = base;
                if (slot_count == 1) offset = static_cast<std::uint64_t>(neuron) * width;
                else if (order == "neuron-major") {
                    offset = base + (static_cast<std::uint64_t>(neuron) * slot_count + slot) * width;
                } else {
                    offset = base + (static_cast<std::uint64_t>(slot) * neuron_count + neuron) * width;
                }
                placed.push_back(Placed{neuron, slot, 0, offset, width});
            }
        }
        return placed;
    };

    const auto state_placed = place_from(state, neurons, 1, 2, state_order, 0, 8);
    const auto count_placed = place_from(counts, neurons, 1, 1, "neuron-major", 0, 4);
    const auto entry_placed = place_from(entries, neurons, slots, 1, entry_order, entry_base, 32);
    auto cover = [](const std::vector<Placed>& placed, std::uint32_t neuron_count, std::uint32_t slot_count,
                    std::uint32_t field_count, const char* what) {
        std::vector<char> seen(static_cast<std::size_t>(neuron_count) * slot_count * field_count, 0);
        for (const auto& item : placed) {
            if (item.neuron >= neuron_count || item.slot >= slot_count || item.field >= field_count) {
                throw std::invalid_argument(std::string("CA-5B ") + what + " placement is outside its domain");
            }
            const auto index = (item.neuron * slot_count + item.slot) * field_count + item.field;
            if (seen[index]) throw std::invalid_argument(std::string("CA-5B ") + what + " placement is duplicated");
            seen[index] = 1;
        }
        if (std::find(seen.begin(), seen.end(), 0) != seen.end()) {
            throw std::invalid_argument(std::string("CA-5B ") + what + " domain is incomplete");
        }
    };
    cover(state_placed, neurons, 1, 2, "state");
    cover(count_placed, neurons, 1, 1, "delta count");
    cover(entry_placed, neurons, slots, 1, "delta entry");

    std::vector<std::pair<std::uint64_t, std::uint64_t>> state_ranges;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> delta_ranges;
    StateDeltaLayoutV5 layout;
    layout.neurons_ = neurons;
    layout.max_slots_ = slots;
    layout.index_capacity_ = config.index_bytes;
    layout.route_capacity_ = config.route_bytes;
    layout.identity_order_ = state_order == "neuron-major" && entry_order == "neuron-major" &&
                             !state.contains("placements") && !entries.contains("placements");
    layout.digest_ = plan_digest;
    layout.state_digest_ = state_digest;
    layout.count_digest_ = count_digest;
    layout.entry_digest_ = entry_digest;
    layout.path_ = std::move(path);
    layout.state_.assign(static_cast<std::size_t>(neurons) * 2, {});
    layout.counts_.assign(neurons, 0);
    layout.entries_.assign(static_cast<std::size_t>(neurons) * slots, 0);
    for (const auto& item : state_placed) {
        if (item.width != 4 || item.offset % 4 != 0 || item.offset + item.width > requireU64(state, "capacity")) {
            throw std::invalid_argument("CA-5B state placement is unaligned or out of capacity");
        }
        layout.state_[item.neuron * 2 + item.field] = FieldAddress{item.offset, item.width};
        state_ranges.emplace_back(item.offset, item.offset + item.width);
    }
    for (const auto& item : count_placed) {
        if (item.width != 4 || item.offset % 4 != 0 || item.offset + item.width > entry_base) {
            throw std::invalid_argument("CA-5B delta count placement aliases entries or is unaligned");
        }
        layout.counts_[item.neuron] = item.offset;
        delta_ranges.emplace_back(item.offset, item.offset + item.width);
    }
    const auto entry_limit = entry_base + requireU64(entries, "capacity");
    for (const auto& item : entry_placed) {
        if (item.width != 32 || item.offset % 4 != 0 || item.offset < entry_base || item.offset + item.width > entry_limit) {
            throw std::invalid_argument("CA-5B delta entry placement is outside the entry area");
        }
        layout.entries_[static_cast<std::size_t>(item.neuron) * slots + item.slot] = item.offset;
        delta_ranges.emplace_back(item.offset, item.offset + item.width);
    }
    rejectOverlap(std::move(state_ranges), "state");
    rejectOverlap(std::move(delta_ranges), "delta");
    if (layout.identity_order_) {
        for (std::uint32_t neuron = 0; neuron < neurons; ++neuron) {
            if (layout.state_[neuron * 2].offset != static_cast<std::uint64_t>(neuron) * 8 ||
                layout.state_[neuron * 2 + 1].offset != static_cast<std::uint64_t>(neuron) * 8 + 4 ||
                layout.counts_[neuron] != static_cast<std::uint64_t>(neuron) * 4) {
                throw std::invalid_argument("CA-5B identity descriptor diverges from the historical offsets");
            }
            for (std::uint32_t slot = 0; slot < slots; ++slot) {
                if (layout.entries_[static_cast<std::size_t>(neuron) * slots + slot] != legacyEntry(neuron, slot, neurons, slots)) {
                    throw std::invalid_argument("CA-5B identity delta entry diverges from the historical offset");
                }
            }
        }
    }
    return layout;
}

std::vector<StateDeltaLayoutV5::Span> StateDeltaLayoutV5::stateSpans(std::uint32_t neuron) const {
    if (neuron >= neurons_) throw std::invalid_argument("CA-5B state neuron is outside the descriptor domain");
    std::vector<Span> spans;
    std::uint32_t logical = 0;
    for (std::uint32_t field = 0; field < 2; ++field) {
        const auto& address = state_[static_cast<std::size_t>(neuron) * 2 + field];
        if (!spans.empty() && spans.back().offset + spans.back().bytes == address.offset &&
            spans.back().logical_offset + spans.back().bytes == logical) {
            spans.back().bytes += address.width;
        } else {
            spans.push_back(Span{address.offset, address.width, logical});
        }
        logical += address.width;
    }
    return spans;
}

std::uint64_t StateDeltaLayoutV5::deltaCountOffset(std::uint32_t neuron) const {
    if (neuron >= neurons_) throw std::invalid_argument("CA-5B delta count neuron is outside the descriptor domain");
    return counts_[neuron];
}

std::uint64_t StateDeltaLayoutV5::deltaEntryOffset(std::uint32_t neuron, std::uint32_t slot) const {
    if (neuron >= neurons_ || slot >= max_slots_) {
        throw std::invalid_argument("CA-5B delta entry is outside the descriptor domain");
    }
    return entries_[static_cast<std::size_t>(neuron) * max_slots_ + slot];
}

std::uint64_t StateDeltaLayoutV5::indexOffset(std::uint64_t byte_offset, std::size_t bytes) const {
    if (bytes == 0 || byte_offset > index_capacity_ || bytes > index_capacity_ - byte_offset) {
        throw std::invalid_argument("CA-5B index address is outside the descriptor");
    }
    return byte_offset;
}

std::uint64_t StateDeltaLayoutV5::routeOffset(std::uint64_t byte_offset, std::size_t bytes) const {
    if (bytes == 0 || byte_offset > route_capacity_ || bytes > route_capacity_ - byte_offset) {
        throw std::invalid_argument("CA-5B route address is outside the descriptor");
    }
    return byte_offset;
}

} // namespace v5
} // namespace SnnDL
} // namespace SST
