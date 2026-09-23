#ifndef SST_SNN_DL_V5_CORE_STORAGE_V5_H
#define SST_SNN_DL_V5_CORE_STORAGE_V5_H

#include "BankedSramV5.h"
#include "StateDeltaLayoutV5.h"
#include "v5/api/AddressSpace.h"
#include "v5/core/DeterministicRetireQueue.h"
#include "v5/core/CubaLifNeuronOp.h"
#include "v5/core/LifNeuronOp.h"

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <map>
#include <string>
#include <vector>

namespace SST {
namespace SnnDL {
namespace v5 {

// CA-5A execution profiles.  LegacyBlocking keeps the historical in-call SRAM
// spin so functional comparisons have a stable baseline.  AsyncCompletion is
// the CorePipeline execution path: one shared logical clock, no internal spin.
enum class CoreStorageExecutionProfile : std::uint8_t {
    LegacyBlocking = 0,
    AsyncCompletion = 1,
};

enum class StorageRequestPhase : std::uint8_t {
    Created = 0,
    Accepted,
    Servicing,
    Completed,
    Consumed,
};

struct StorageToken {
    std::uint64_t id = 0;
    explicit operator bool() const { return id != 0; }
};

struct StorageSubmitResult {
    StorageToken token;
    bool accepted = false;
    bool retryable = false;
    BankedSramV5Reject reject = BankedSramV5Reject::None;
    std::string stall_reason;
};

struct StorageCompletion {
    StorageToken token;
    std::uint64_t epoch = 0;
    ::SnnDL::v5::AddressSpaceId region = ::SnnDL::v5::AddressSpaceId::CoreState;
    std::uint64_t logical_operation_id = 0;
    std::uint32_t attempt_id = 0;
    std::uint64_t physical_request_id = 0;
    std::uint64_t issue_cycle = 0;
    std::uint64_t service_cycle = 0;
    std::uint64_t completion_cycle = 0;
    std::uint64_t eligible_service_cycle = 0;
    std::uint32_t bank = 0;
    std::uint32_t port = 0;
    bool write = false;
    bool ok = false;
    std::vector<std::uint8_t> data;
    std::string stall_reason;
};

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
    std::string trace_json;
    // Direct storage callers keep the blocking profile.  CorePipeline selects
    // AsyncCompletion as its real execution path.  A descriptor file is rejected
    // unless this profile is AsyncCompletion, so a layout experiment cannot
    // silently fall back to the blocking clock.
    CoreStorageExecutionProfile execution_profile = CoreStorageExecutionProfile::LegacyBlocking;
    std::string descriptor_json;
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
    CoreStorageExecutionProfile executionProfile() const { return config_.execution_profile; }
    const ::SnnDL::v5::RegionDescriptor& region(::SnnDL::v5::AddressSpaceId space) const;
    const BankedSramV5Stats& stats(::SnnDL::v5::AddressSpaceId space) const;
    std::uint64_t epoch() const { return epoch_; }
    std::uint64_t advanceCount(::SnnDL::v5::AddressSpaceId space) const;
    std::uint64_t pendingPeak() const { return pending_peak_; }

    // Declares the logical v5 pipeline timestep that owns every following
    // typed access so the execution SRAM trace carries a real coordinate.
    // This is the network timestep, never the CorePipeline cycle and never
    // SST time.  Accesses made before the first declaration are traced with
    // an explicit null timestep instead of an invented zero.
    void beginTimestep(std::uint64_t timestep);

    // Opens the next storage epoch.  Refuses while any request is accepted,
    // in service, completed-but-unconsumed, or waiting for admission.
    void beginEpoch();

    // Non-blocking CA-5A path.  submit creates one logical operation.  A
    // retryable reject keeps that operation and only bumps attempt_id.
    // advance ticks every region exactly once.  takeCompletions does not
    // consume.  consume releases a completed token exactly once.
    StorageSubmitResult submitRead(::SnnDL::v5::AddressSpaceId space, std::uint64_t byte_offset,
                                   std::size_t bytes, std::uint64_t now);
    StorageSubmitResult submitWrite(::SnnDL::v5::AddressSpaceId space, std::uint64_t byte_offset,
                                    const std::vector<std::uint8_t>& data, std::uint64_t now);
    void advance(std::uint64_t now);
    void admit(std::uint64_t now);
    std::vector<StorageCompletion> takeCompletions();
    bool consume(StorageToken token);
    std::size_t pending() const;
    bool drained() const;

    const StateDeltaLayoutV5& layout() const { return layout_; }
    std::vector<StateDeltaLayoutV5::Span> stateSpans(std::uint32_t neuron) const;
    // Contiguous identity records only.  Field-major state has more than one span.
    std::uint64_t stateByteOffset(std::uint32_t neuron) const;
    std::uint64_t deltaCountByteOffset(std::uint32_t neuron) const;
    std::uint64_t deltaEntryByteOffset(std::uint32_t neuron, std::uint32_t slot) const;
    std::uint64_t requireIndexOffset(std::uint64_t byte_offset, std::size_t bytes) const;
    std::uint64_t requireRouteOffset(std::uint64_t byte_offset, std::size_t bytes) const;
    static std::vector<std::uint8_t> encodeLifRecord(const LifNeuronState& state);
    static LifNeuronState decodeLifRecord(const std::vector<std::uint8_t>& bytes);
    static std::vector<std::uint8_t> encodeDeltaRecord(const RetireEntry& entry);
    static RetireEntry decodeDeltaRecord(std::uint32_t post_neuron, const std::vector<std::uint8_t>& bytes);
    static std::vector<std::uint8_t> encodeCountRecord(std::uint32_t value);
    static std::uint32_t decodeCountRecord(const std::vector<std::uint8_t>& bytes);

    // Completed-byte observation.  Does not admit a request or advance SRAM.
    bool readCompletedState(std::uint32_t neuron, LifNeuronState& state) const;
    bool readCompletedCubaLifState(std::uint32_t neuron, CubaLifNeuronState& state) const;

    // Legacy blocking profile only.  AsyncCompletion rejects these so a caller
    // cannot spin the SRAM clock inside one Core tick.
    void resetTimestep();

    bool readState(std::uint32_t neuron, LifNeuronState& state);
    bool writeState(std::uint32_t neuron, const LifNeuronState& state);
    bool readCubaLifState(std::uint32_t neuron, CubaLifNeuronState& state);
    bool writeCubaLifState(std::uint32_t neuron, const CubaLifNeuronState& state);

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
        // LegacyBlocking time base only.  AsyncCompletion uses the caller's now.
        std::uint64_t cycle = 0;
        std::uint64_t advance_count = 0;
        std::map<std::uint64_t, std::uint64_t> physical_to_token;
    };

    struct OpRecord {
        StorageToken token;
        std::uint64_t epoch = 0;
        ::SnnDL::v5::AddressSpaceId region = ::SnnDL::v5::AddressSpaceId::CoreState;
        std::uint64_t logical_operation_id = 0;
        std::uint32_t attempt_id = 0;
        std::uint64_t physical_request_id = 0;
        StorageRequestPhase phase = StorageRequestPhase::Created;
        bool write = false;
        bool terminal_reject = false;
        std::uint64_t offset = 0;
        std::vector<std::uint8_t> payload;
        std::uint64_t accept_cycle = 0;
        std::uint64_t eligible_service_cycle = 0;
        std::uint64_t service_cycle = 0;
        std::uint64_t completion_cycle = 0;
        std::uint64_t consume_cycle = 0;
        std::uint32_t bank = 0;
        std::uint32_t port = 0;
        bool ok = false;
        std::vector<std::uint8_t> response;
        std::string stall_reason;
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

    void requireAsync_(const char* what) const;
    void requireLegacy_(const char* what) const;
    StorageSubmitResult submit_(::SnnDL::v5::AddressSpaceId space, std::uint64_t byte_offset,
                                std::vector<std::uint8_t> data, bool write, std::uint64_t now);
    void tryAdmit_(OpRecord& op, std::uint64_t now);
    bool responseSaturated_(const Region& region) const;
    std::size_t completedUnconsumed_(::SnnDL::v5::AddressSpaceId space) const;
    void notePeak_();
    void traceOp_(const OpRecord& op, bool accepted, bool completed) const;
    StorageCompletion completionFrom_(const OpRecord& op) const;
    bool copyCompleted_(::SnnDL::v5::AddressSpaceId space, std::uint64_t byte_offset, std::size_t bytes,
                        std::vector<std::uint8_t>& out) const;

    bool readLogicalState_(std::uint32_t neuron, std::vector<std::uint8_t>& logical);
    bool writeLogicalState_(std::uint32_t neuron, const std::vector<std::uint8_t>& logical);
    bool peekLogicalState_(std::uint32_t neuron, std::vector<std::uint8_t>& logical) const;

    CoreStorageV5Config config_;
    StateDeltaLayoutV5 layout_;
    bool have_timestep_ = false;
    std::uint64_t current_timestep_ = 0;
    std::uint64_t epoch_ = 0;
    std::uint64_t next_token_ = 1;
    std::uint64_t next_logical_ = 1;
    std::uint64_t pending_peak_ = 0;
    bool has_advance_ = false;
    std::uint64_t last_advance_now_ = 0;
    std::map<std::uint64_t, OpRecord> ops_;
    mutable std::ofstream trace_stream_;
    Region state_;
    Region delta_;
    Region index_;
    Region route_;
};

} // namespace v5
} // namespace SnnDL
} // namespace SST

#endif
