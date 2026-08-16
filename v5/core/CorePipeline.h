#ifndef SST_SNN_DL_V5_CORE_PIPELINE_H
#define SST_SNN_DL_V5_CORE_PIPELINE_H

#include "DeterministicRetireQueue.h"
#include "LifNeuronOp.h"
#include "v5/storage/CoreStorageV5.h"

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <set>
#include <vector>

namespace SST {
namespace SnnDL {
namespace v5 {

struct StageTiming {
    std::uint32_t latency_cycles = 1;
    std::uint32_t width = 1;
};

struct CorePipelineConfig {
    std::uint32_t neurons = 1;
    std::size_t ingress_entries = 16;
    std::size_t row_entries = 16;
    std::size_t synapse_entries = 32;
    std::size_t retire_entries = 32;
    std::size_t accumulator_entries = 32;
    std::size_t held_spike_entries = 32;
    StageTiming ingress;
    StageTiming row_lookup;
    StageTiming synapse;
    StageTiming retire;
    StageTiming accumulator;
    StageTiming neuron;
    LifNeuronOp::Config lif;
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
    bool sealed() const { return sealed_; }
    // Compatibility snapshot for tests and evidence only.  Functional reads
    // in the pipeline go through CoreStorageV5, never through this vector.
    const std::vector<LifNeuronState>& state() const;
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

    static std::uint64_t effectiveLatency_(std::uint32_t latency);
    static std::uint64_t hashMix_(std::uint64_t hash, std::uint64_t value);
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

    CorePipelineConfig config_;
    LifNeuronOp lif_;
    std::uint64_t active_timestep_ = 0;
    std::uint64_t last_timestep_ = 0;
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
    mutable std::vector<LifNeuronState> state_snapshot_;
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
