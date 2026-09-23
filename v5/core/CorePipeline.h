#ifndef SST_SNN_DL_V5_CORE_PIPELINE_H
#define SST_SNN_DL_V5_CORE_PIPELINE_H

#include "DeterministicRetireQueue.h"
#include "CubaLifNeuronOp.h"
#include "IfNeuronOp.h"
#include "LifNeuronOp.h"
#include "v5/storage/CoreStorageV5.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

namespace SST {
namespace SnnDL {
namespace v5 {

struct StageTiming {
    std::uint32_t latency_cycles = 1;
    std::uint32_t width = 1;
};

struct ScheduleStageDescriptor {
    std::string id;
    std::string operation;
    std::string resource;
    std::string request_class;
    std::vector<std::string> dependencies;
    std::uint32_t earliest_cycle = 0;
    std::string retry_policy;
    std::string issue_group_id;
    std::size_t max_inflight = 1;
};

enum class NeuronOperatorKind : std::uint8_t {
    Lif = 0,
    CubaLif = 1,
    Padding = 2,
    If = 3,
};

struct NeuronBinding {
    NeuronOperatorKind kind = NeuronOperatorKind::Lif;
    LifNeuronOp::Config lif;
    CubaLifNeuronOp::Config cuba_lif;
    IfNeuronOp::Config if_op;
};

struct CorePipelineConfig {
    std::uint32_t neurons = 1;
    std::size_t ingress_entries = 16;
    std::size_t row_entries = 16;
    std::size_t synapse_entries = 32;
    std::size_t retire_entries = 32;
    std::size_t accumulator_entries = 32;
    std::size_t held_spike_entries = 32;
    // Compiler SchedulePlan admission reservations.  A zero reservation
    // means "use the physical queue capacity" for compatibility profiles.
    bool schedule_admission_enabled = false;
    std::size_t schedule_ingress_entries = 0;
    std::size_t schedule_synapse_entries = 0;
    std::vector<ScheduleStageDescriptor> schedule_stages;
    StageTiming ingress;
    StageTiming row_lookup;
    StageTiming synapse;
    StageTiming retire;
    StageTiming accumulator;
    StageTiming neuron;
    NeuronOperatorKind neuron_operator = NeuronOperatorKind::Lif;
    LifNeuronOp::Config lif;
    CubaLifNeuronOp::Config cuba_lif;
    IfNeuronOp::Config if_op;
    std::vector<NeuronBinding> neuron_bindings;
    // CA-5A real path.  LegacyBlocking keeps the historical in-call SRAM spin.
    CoreStorageExecutionProfile execution_profile = CoreStorageExecutionProfile::AsyncCompletion;
    CoreStorageV5Config storage;
};

struct SpikeInput {
    std::uint64_t timestep = 0;
    std::uint64_t source_neuron = 0;
    std::uint64_t source_event_seq = 0;
};

struct RowRequest {
    std::uint64_t timestep = 0;
    std::uint64_t source_neuron = 0;
    std::uint64_t source_event_seq = 0;
    std::uint64_t row_id = 0;
};

struct SynapseResponse {
    std::uint64_t timestep = 0;
    std::uint64_t source_neuron = 0;
    std::uint64_t source_event_seq = 0;
    std::uint32_t post_neuron = 0;
    std::uint64_t edge_ordinal = 0;
    float weight = 0.0f;
    bool row_complete = false;
    std::uint32_t row_edge_count = 0;
};

struct RowDone {
    std::uint64_t timestep = 0;
    std::uint64_t source_neuron = 0;
    std::uint64_t source_event_seq = 0;
    std::uint32_t edge_count = 0;
};

struct FiredSpike {
    std::uint64_t timestep = 0;
    std::uint32_t post_neuron = 0;
    std::uint64_t source_event_seq = 0;
};

struct CorePipelineStats {
    struct Stage {
        std::uint64_t accepted = 0;
        std::uint64_t issued = 0;
        std::uint64_t completed = 0;
        std::uint64_t occupancy = 0;
        std::uint64_t busy_cycles = 0;
        std::uint64_t full_cycles = 0;
        std::uint64_t stall_cycles = 0;
    };

    std::uint64_t cycles = 0;
    std::uint64_t ingress_accepted = 0;
    std::uint64_t row_requests = 0;
    std::uint64_t rows_completed = 0;
    std::uint64_t synapse_issued = 0;
    std::uint64_t retire_retired = 0;
    std::uint64_t accumulator_updates = 0;
    std::uint64_t neurons_evaluated = 0;
    std::uint64_t neurons_fired = 0;
    std::uint64_t held_released = 0;
    std::uint64_t ingress_full_cycles = 0;
    std::uint64_t row_stall_cycles = 0;
    std::uint64_t synapse_stall_cycles = 0;
    std::uint64_t retire_stall_cycles = 0;
    std::uint64_t accumulator_stall_cycles = 0;
    std::uint64_t held_full_cycles = 0;
    Stage ingress;
    Stage row_lookup;
    Stage synapse;
    Stage retire;
    Stage accumulator;
    Stage neuron;
    Stage held_spike;
};

class CorePipeline {
public:
    explicit CorePipeline(const CorePipelineConfig& config);

    void start(std::uint64_t timestep);
    bool submitSpike(const SpikeInput& spike);
    bool acceptSynapseResponse(const SynapseResponse& response);
    bool acceptRowDone(const RowDone& done);
    void sealIngress();
    bool tick();
    bool readyToCommit() const;
    void commit();

    std::vector<RowRequest> takeRowRequests();
    std::vector<FiredSpike> takeReleasedSpikes();
    const CorePipelineStats& stats() const { return stats_; }
    std::uint64_t timestep() const { return active_timestep_; }
    std::uint64_t functionalHash() const;
    std::size_t pendingEntries() const;
    std::size_t heldEntries() const { return held_count_; }
    bool active() const { return active_; }
    std::size_t scheduleStageCount() const { return config_.schedule_stages.size(); }
    bool sealed() const { return sealed_; }
    // Compatibility snapshot for tests and evidence only.  The bytes come from
    // SRAM writes that have already completed.  state() does not advance SRAM.
    const std::vector<LifNeuronState>& state() const;
    const std::vector<CubaLifNeuronState>& cubaState() const;
    const CoreStorageV5& storage() const { return *storage_; }

private:
    struct RowKey {
        std::uint64_t source_neuron = 0;
        std::uint64_t source_event_seq = 0;
        bool operator<(const RowKey& other) const {
            if (source_neuron != other.source_neuron) return source_neuron < other.source_neuron;
            return source_event_seq < other.source_event_seq;
        }
    };

    struct RowState {
        std::uint32_t expected = 0;
        std::uint32_t received = 0;
        bool done = false;
        std::set<std::uint64_t> ordinals;
    };

    template <typename T>
    struct Timed {
        T value;
        std::uint64_t ready_cycle = 0;
    };

    enum class AppendPhase : std::uint8_t { ReadCount, WriteEntry, WriteCount, Done, Overflow };
    enum class NeuronPhase : std::uint8_t { Load, LoadEntries, Ready, Write, Finished };

    struct IndexWait {
        SpikeInput spike;
        StorageToken token;
        bool done = false;
    };
    struct AppendTxn {
        RetireEntry entry;
        AppendPhase phase = AppendPhase::ReadCount;
        StorageToken token;
        bool submitted = false;
        std::uint32_t count = 0;
    };
    struct NeuronTxn {
        std::uint32_t neuron = 0;
        NeuronPhase phase = NeuronPhase::Load;
        struct SpanWait {
            StorageToken token;
            std::uint32_t logical_offset = 0;
            std::uint32_t bytes = 0;
            bool done = false;
            std::vector<std::uint8_t> data;
        };
        std::vector<SpanWait> state_reads;
        std::vector<SpanWait> state_writes;
        StorageToken count_token;
        bool state_done = false;
        bool count_done = false;
        std::uint32_t delta_count = 0;
        bool entries_submitted = false;
        std::vector<StorageToken> entry_tokens;
        std::vector<std::uint8_t> entry_done;
        std::vector<RetireEntry> deltas;
        LifNeuronState lif_state;
        CubaLifNeuronState cuba_state;
        LifNeuronResult lif_result;
        CubaLifNeuronResult cuba_result;
        bool fired = false;
        StorageToken clear_token;
        bool write_submitted = false;
        bool clear_submitted = false;
        bool write_done = false;
        bool clear_done = false;
    };

    static std::uint64_t effectiveLatency_(std::uint32_t latency);
    static std::uint64_t hashMix_(std::uint64_t hash, std::uint64_t value);
    bool asyncStorage_() const;
    bool storageIdle_() const;
    bool pollCompletion_(StorageToken token, StorageCompletion& completion);
    void clearPipelineState_();
    void submitEpochRequests_();
    void tickAsync_();
    void consumeEpochBarrier_();
    void consumeIndexReads_();
    void consumeAppends_();
    void consumeNeuronChain_();
    void issueIndexReads_();
    void issueAppends_();
    void issueNeuronChain_();
    void evaluateReadyNeurons_();
    void retireFinishedNeurons_();
    RowKey keyFor_(std::uint64_t source_neuron, std::uint64_t source_event_seq) const;
    bool allRowsComplete_() const;
    bool queuesEmpty_() const;
    void processNeuron_();
    void processAccumulator_();
    void processRetire_();
    void processSynapse_();
    void processRows_();
    void processIngress_();
    void scheduleNeuron_();
    void resetTimestep_();
    void recordStageCycles_();
    void recordOccupancy_();
    std::size_t scheduleCapacity_(const char* stage_id, std::size_t fallback) const;
    std::size_t scheduleWidth_(const char* stage_id, std::size_t fallback) const;
    bool scheduleReady_(const char* stage_id) const;

    CorePipelineConfig config_;
    LifNeuronOp lif_;
    CubaLifNeuronOp cuba_lif_;
    IfNeuronOp if_;
    std::uint64_t active_timestep_ = 0;
    std::uint64_t last_timestep_ = 0;
    // cycle_ is the shared Core/SRAM logical clock and never restarts.
    // timestep_origin_ makes SchedulePlan earliest_cycle timestep-local.
    std::uint64_t timestep_origin_ = 0;
    bool has_timestep_ = false;
    bool active_ = false;
    bool sealed_ = false;
    bool scan_done_ = false;
    std::uint64_t cycle_ = 0;
    std::uint64_t next_row_id_ = 1;
    std::size_t rows_issued_ = 0;
    std::size_t held_count_ = 0;

    std::deque<Timed<SpikeInput>> ingress_q_;
    std::deque<Timed<SpikeInput>> row_q_;
    std::deque<RowRequest> row_request_out_;
    std::deque<Timed<SynapseResponse>> synapse_q_;
    DeterministicRetireQueue retire_q_;
    std::deque<Timed<RetireEntry>> accumulator_q_;
    std::deque<FiredSpike> released_out_;
    std::map<std::uint64_t, std::vector<FiredSpike>> held_spikes_;
    std::map<RowKey, RowState> rows_;
    std::unique_ptr<CoreStorageV5> storage_;
    std::map<std::uint64_t, StorageCompletion> bound_;
    std::vector<StorageToken> epoch_reset_tokens_;
    std::vector<std::uint8_t> epoch_reset_done_;
    StorageToken route_token_;
    bool route_submitted_ = false;
    bool route_done_ = false;
    bool epoch_barrier_ready_ = false;
    std::deque<IndexWait> index_waits_;
    std::deque<AppendTxn> append_txns_;
    std::vector<NeuronTxn> neuron_txns_;
    bool neuron_evaluated_ = false;
    std::vector<std::uint8_t> neuron_busy_;
    mutable std::vector<LifNeuronState> state_snapshot_;
    mutable std::vector<CubaLifNeuronState> cuba_state_snapshot_;
    std::uint32_t next_neuron_ = 0;
    bool neuron_batch_pending_ = false;
    std::uint32_t neuron_batch_begin_ = 0;
    std::uint32_t neuron_batch_count_ = 0;
    std::uint64_t neuron_batch_ready_ = 0;
    CorePipelineStats stats_;
};

} // namespace v5
} // namespace SnnDL
} // namespace SST

#endif
