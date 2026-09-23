#ifndef SST_SNN_DL_V5_STATE_DELTA_LAYOUT_V5_H
#define SST_SNN_DL_V5_STATE_DELTA_LAYOUT_V5_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace SST {
namespace SnnDL {
namespace v5 {

struct CoreStorageV5Config;

// Compiler-owned region-relative addresses for CoreState and CoreDelta.
// Hot-path lookups read the bound table.  The historical offset formula is
// only a parity check for the identity candidate.
class StateDeltaLayoutV5 {
public:
    struct Span {
        std::uint64_t offset = 0;
        std::uint32_t bytes = 0;
        std::uint32_t logical_offset = 0;
    };

    static StateDeltaLayoutV5 bind(const CoreStorageV5Config& config);

    std::vector<Span> stateSpans(std::uint32_t neuron) const;
    std::uint64_t deltaCountOffset(std::uint32_t neuron) const;
    std::uint64_t deltaEntryOffset(std::uint32_t neuron, std::uint32_t slot) const;
    std::uint64_t indexOffset(std::uint64_t byte_offset, std::size_t bytes) const;
    std::uint64_t routeOffset(std::uint64_t byte_offset, std::size_t bytes) const;

    bool identityOrder() const { return identity_order_; }
    const std::string& digest() const { return digest_; }
    const std::string& stateDigest() const { return state_digest_; }
    const std::string& deltaCountDigest() const { return count_digest_; }
    const std::string& deltaEntryDigest() const { return entry_digest_; }
    const std::string& path() const { return path_; }

private:
    struct FieldAddress {
        std::uint64_t offset = 0;
        std::uint32_t width = 0;
    };

    std::uint32_t neurons_ = 0;
    std::uint32_t max_slots_ = 0;
    std::uint64_t index_capacity_ = 0;
    std::uint64_t route_capacity_ = 0;
    bool identity_order_ = true;
    std::string digest_;
    std::string state_digest_;
    std::string count_digest_;
    std::string entry_digest_;
    std::string path_;
    std::vector<FieldAddress> state_;
    std::vector<std::uint64_t> counts_;
    std::vector<std::uint64_t> entries_;
};

} // namespace v5
} // namespace SnnDL
} // namespace SST

#endif
