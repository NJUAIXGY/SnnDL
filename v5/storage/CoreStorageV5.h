#ifndef SST_SNN_DL_V5_CORE_STORAGE_V5_H
#define SST_SNN_DL_V5_CORE_STORAGE_V5_H

#include "BankedSramV5.h"
#include "v5/api/AddressSpace.h"
#include "v5/core/DeterministicRetireQueue.h"
#include "v5/core/LifNeuronOp.h"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace SST {
namespace SnnDL {
namespace v5 {

// P2's typed local-memory owner.  Each region has one byte backing and all
// functional reads/writes use BankedSramV5's request path.  The class is
// deliberately independent of SST links so the same binding can be exercised
// by the deterministic CorePipeline unit tests and by SnnCoreV5.
struct CoreStorageV5Config {
    std::uint32_t core_id = 0;
    std::uint32_t pe_id = 0;
    std::uint32_t neurons = 1;
    std::size_t max_delta_entries_per_neuron = 32;
    std::uint64_t index_bytes = 4096;
    std::uint64_t route_bytes = 4096;
    BankedSramV5Config state_sram;
    BankedSramV5Config delta_sram;
    BankedSramV5Config index_sram;
    BankedSramV5Config route_sram;
};

class CoreStorageV5 final {
public:
    static constexpr std::size_t kStateBytes = sizeof(float) + sizeof(std::uint32_t);
    static constexpr std::size_t kDeltaCountBytes = sizeof(std::uint32_t);
    static constexpr std::size_t kDeltaEntryBytes = 32;

    explicit CoreStorageV5(const CoreStorageV5Config& config);

    const CoreStorageV5Config& config() const { return config_; }
    const ::SnnDL::v5::RegionDescriptor& region(::SnnDL::v5::AddressSpaceId space) const;
    const BankedSramV5Stats& stats(::SnnDL::v5::AddressSpaceId space) const;

    // Timestep state is persistent; only the delta counts are reset here.
    void resetTimestep();

    bool readState(std::uint32_t neuron, LifNeuronState& state);
    bool writeState(std::uint32_t neuron, const LifNeuronState& state);

    // Delta entries are resident in CoreDelta.  The count and entries are
    // stored in the same region, so there is no vector shadow of accumulated
    // weights.
    bool appendDelta(const RetireEntry& entry);
    bool readDeltaEntries(std::uint32_t neuron, std::vector<RetireEntry>& entries);
    bool clearDelta(std::uint32_t neuron);

    bool readIndex(std::uint64_t byte_offset, std::size_t bytes,
                   std::vector<std::uint8_t>& data);
    bool readRoute(std::uint64_t byte_offset, std::size_t bytes,
                   std::vector<std::uint8_t>& data);

private:
    struct Region final {
        Region(const ::SnnDL::v5::RegionDescriptor& descriptor,
               const BankedSramV5Config& config)
            : descriptor(descriptor), sram(config) {}

        ::SnnDL::v5::RegionDescriptor descriptor;
        BankedSramV5 sram;
        std::uint64_t next_request_id = 1;
        std::uint64_t cycle = 0;
    };

    static CoreStorageV5Config normalize_(CoreStorageV5Config config);
    static std::uint64_t regionBytes_(const CoreStorageV5Config& config,
                                      ::SnnDL::v5::AddressSpaceId space);
    static std::uint64_t regionBase_(const CoreStorageV5Config& config,
                                     ::SnnDL::v5::AddressSpaceId space);
    static ::SnnDL::v5::RegionDescriptor descriptor_(const CoreStorageV5Config& config,
                                                     ::SnnDL::v5::AddressSpaceId space);
    static BankedSramV5Config sramConfig_(const CoreStorageV5Config& config,
                                          ::SnnDL::v5::AddressSpaceId space);

    Region& region_(::SnnDL::v5::AddressSpaceId space);
    const Region& region_(::SnnDL::v5::AddressSpaceId space) const;
    bool transfer_(Region& region, std::uint64_t byte_offset,
                   const std::vector<std::uint8_t>& input, bool write,
                   std::vector<std::uint8_t>& output);
    bool readBytes_(Region& region, std::uint64_t byte_offset, std::size_t bytes,
                    std::vector<std::uint8_t>& output);
    bool writeBytes_(Region& region, std::uint64_t byte_offset,
                     const std::vector<std::uint8_t>& input);
    std::uint32_t readU32_(const std::vector<std::uint8_t>& bytes) const;
    void writeU32_(std::uint32_t value, std::vector<std::uint8_t>& bytes) const;
    static void encodeState_(const LifNeuronState& state, std::vector<std::uint8_t>& bytes);
    static LifNeuronState decodeState_(const std::vector<std::uint8_t>& bytes);
    static void encodeDelta_(const RetireEntry& entry, std::vector<std::uint8_t>& bytes);
    static RetireEntry decodeDelta_(std::uint32_t post_neuron,
                                    const std::vector<std::uint8_t>& bytes);
    std::uint64_t deltaCountOffset_(std::uint32_t neuron) const;
    std::uint64_t deltaEntryOffset_(std::uint32_t neuron, std::uint32_t slot) const;

    CoreStorageV5Config config_;
    Region state_;
    Region delta_;
    Region index_;
    Region route_;
};

} // namespace v5
} // namespace SnnDL
} // namespace SST

#endif
