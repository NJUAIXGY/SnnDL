#include "CorePipeline.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace SST {
namespace SnnDL {
namespace v5 {

namespace {
std::uint32_t positiveWidth(std::uint32_t width) {
    return std::max<std::uint32_t>(1, width);
}
}

CorePipeline::CorePipeline(const CorePipelineConfig& config)
    : config_(config), lif_(config.lif), retire_q_(config.retire_entries),
      storage_(nullptr), state_snapshot_(config.neurons) {
    if (config_.neurons == 0 || config_.ingress_entries == 0 || config_.row_entries == 0 ||
        config_.synapse_entries == 0 || config_.accumulator_entries == 0 ||
        config_.held_spike_entries == 0) {
        throw std::invalid_argument("v5 core queue and neuron capacities must be positive");
    }
    config_.ingress.width = positiveWidth(config_.ingress.width);
    config_.row_lookup.width = positiveWidth(config_.row_lookup.width);
    config_.synapse.width = positiveWidth(config_.synapse.width);
    config_.retire.width = positiveWidth(config_.retire.width);
    config_.accumulator.width = positiveWidth(config_.accumulator.width);
    config_.neuron.width = positiveWidth(config_.neuron.width);
    config_.storage.neurons = config_.neurons;
    // CoreDelta is resident state, not a transient retire queue.  Keep its
    // fallback independent from pipeline backpressure so a small retire queue
    // cannot make a valid multi-edge row impossible to commit.
    if (config_.storage.max_delta_entries_per_neuron == 0) {
        config_.storage.max_delta_entries_per_neuron =
            CoreStorageV5Config{}.max_delta_entries_per_neuron;
    }
    storage_ = std::make_unique<CoreStorageV5>(config_.storage);
}

std::uint64_t CorePipeline::effectiveLatency_(std::uint32_t latency) {
    // A zero-latency configuration is accepted for contract probes, but an
    // item still crosses a clock boundary so stages can never bypass in one tick.
    return std::max<std::uint32_t>(1, latency);
}

std::uint64_t CorePipeline::hashMix_(std::uint64_t hash, std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    return hash;
}

CorePipeline::RowKey CorePipeline::keyFor_(std::uint32_t source_neuron,
                                           std::uint64_t source_event_seq) const {
    return RowKey{source_neuron, source_event_seq};
}

void CorePipeline::resetTimestep_() {
    ingress_q_.clear();
    row_q_.clear();
    row_request_out_.clear();
    synapse_q_.clear();
    retire_q_.clear();
    accumulator_q_.clear();
    rows_.clear();
    storage_->resetTimestep();
    stats_ = CorePipelineStats{};
    cycle_ = 0;
    next_row_id_ = 1;
    rows_issued_ = 0;
    next_neuron_ = 0;
    neuron_batch_pending_ = false;
    neuron_batch_begin_ = 0;
    neuron_batch_count_ = 0;
    neuron_batch_ready_ = 0;
    scan_done_ = false;
    sealed_ = false;
}

void CorePipeline::recordOccupancy_() {
    const auto update = [](CorePipelineStats::Stage& stage, std::uint64_t occupancy) {
        stage.occupancy = std::max(stage.occupancy, occupancy);
    };
    update(stats_.ingress, ingress_q_.size());
    update(stats_.row_lookup, row_q_.size() + row_request_out_.size());
    update(stats_.synapse, synapse_q_.size());
    update(stats_.retire, retire_q_.size());
    update(stats_.accumulator, accumulator_q_.size());
    update(stats_.neuron, neuron_batch_pending_ ? neuron_batch_count_ : 0);
    update(stats_.held_spike, held_count_);
}

void CorePipeline::recordStageCycles_() {
    recordOccupancy_();
    const auto mark = [](CorePipelineStats::Stage& stage, bool busy, bool full, bool stall) {
        if (busy) ++stage.busy_cycles;
        if (full) ++stage.full_cycles;
        if (stall) ++stage.stall_cycles;
    };
    const bool ingress_ready = !ingress_q_.empty() && ingress_q_.front().ready_cycle <= cycle_;
    const bool row_ready = !row_q_.empty() && row_q_.front().ready_cycle <= cycle_;
    const bool synapse_ready = !synapse_q_.empty() && synapse_q_.front().ready_cycle <= cycle_;
    mark(stats_.ingress, !ingress_q_.empty(), ingress_q_.size() >= config_.ingress_entries,
         ingress_ready && row_q_.size() >= config_.row_entries);
    mark(stats_.row_lookup, !row_q_.empty() || !row_request_out_.empty(),
         row_q_.size() >= config_.row_entries || row_request_out_.size() >= config_.row_entries,
         row_ready && row_request_out_.size() >= config_.row_entries);
    mark(stats_.synapse, !synapse_q_.empty(), synapse_q_.size() >= config_.synapse_entries,
         synapse_ready && retire_q_.full());
    mark(stats_.retire, !retire_q_.empty(), retire_q_.size() >= config_.retire_entries,
         !retire_q_.empty() && accumulator_q_.size() >= config_.accumulator_entries);
    mark(stats_.accumulator, !accumulator_q_.empty(),
         accumulator_q_.size() >= config_.accumulator_entries, false);
    mark(stats_.neuron, neuron_batch_pending_, false,
         neuron_batch_pending_ && held_count_ >= config_.held_spike_entries);
    mark(stats_.held_spike, held_count_ != 0, held_count_ >= config_.held_spike_entries,
         held_count_ >= config_.held_spike_entries);
}

void CorePipeline::start(std::uint64_t timestep) {
    if (active_) throw std::logic_error("v5 core Start arrived while active");
    if (has_timestep_ && timestep != last_timestep_ + 1) {
        throw std::logic_error("v5 core timesteps must advance exactly by one");
    }
    resetTimestep_();
    std::vector<std::uint8_t> route_byte;
    if (!storage_->readRoute(0, 1, route_byte)) {
        throw std::logic_error("v5 PeRoute binding is not readable");
    }
    active_timestep_ = timestep;
    last_timestep_ = timestep;
    has_timestep_ = true;
    active_ = true;

    auto held = held_spikes_.find(timestep);
    if (held != held_spikes_.end()) {
        for (const auto& spike : held->second) {
            released_out_.push_back(spike);
            ++stats_.held_spike.issued;
            ++stats_.held_spike.completed;
        }
        stats_.held_released = held->second.size();
        stats_.held_spike.completed += 0;
        held_count_ -= held->second.size();
        held_spikes_.erase(held);
    }
}

bool CorePipeline::submitSpike(const SpikeInput& spike) {
    if (!active_ || sealed_ || spike.timestep != active_timestep_) return false;
    if (spike.source_neuron >= config_.neurons || ingress_q_.size() >= config_.ingress_entries) {
        ++stats_.ingress_full_cycles;
        return false;
    }
    const RowKey key = keyFor_(spike.source_neuron, spike.source_event_seq);
    if (rows_.find(key) != rows_.end()) return false;
    ingress_q_.push_back(Timed<SpikeInput>{spike, cycle_ + effectiveLatency_(config_.ingress.latency_cycles)});
    ++stats_.ingress_accepted;
    ++stats_.ingress.accepted;
    return true;
}

bool CorePipeline::acceptSynapseResponse(const SynapseResponse& response) {
    if (!active_ || response.timestep != active_timestep_ ||
        response.post_neuron >= config_.neurons || synapse_q_.size() >= config_.synapse_entries) {
        return false;
    }
    const RowKey key = keyFor_(response.source_neuron, response.source_event_seq);
    auto row = rows_.find(key);
    if (row == rows_.end() || row->second.ordinals.find(response.edge_ordinal) != row->second.ordinals.end()) {
        return false;
    }
    row->second.ordinals.insert(response.edge_ordinal);
    ++row->second.received;
    if (response.row_complete) {
        row->second.done = true;
        row->second.expected = response.row_edge_count;
    }
    synapse_q_.push_back(Timed<SynapseResponse>{
        response, cycle_ + effectiveLatency_(config_.synapse.latency_cycles)});
    ++stats_.synapse.accepted;
    return true;
}

bool CorePipeline::acceptRowDone(const RowDone& done) {
    if (!active_ || done.timestep != active_timestep_) return false;
    const RowKey key = keyFor_(done.source_neuron, done.source_event_seq);
    auto row = rows_.find(key);
    if (row == rows_.end() || row->second.done) return false;
    row->second.done = true;
    row->second.expected = done.edge_count;
    ++stats_.rows_completed;
    ++stats_.row_lookup.completed;
    return true;
}

void CorePipeline::sealIngress() {
    if (!active_) throw std::logic_error("v5 core SealIngress without active timestep");
    sealed_ = true;
}

void CorePipeline::processNeuron_() {
    if (!neuron_batch_pending_ || neuron_batch_ready_ > cycle_) return;
    std::vector<LifNeuronResult> results;
    results.reserve(neuron_batch_count_);
    std::size_t new_fires = 0;
    for (std::uint32_t i = 0; i < neuron_batch_count_; ++i) {
        const auto neuron = neuron_batch_begin_ + i;
        LifNeuronState state;
        std::vector<RetireEntry> ordered;
        if (!storage_->readState(neuron, state) || !storage_->readDeltaEntries(neuron, ordered)) {
            ++stats_.neuron.stall_cycles;
            return;
        }
        float delta = 0.0f;
        for (const auto& entry : ordered) delta += entry.weight;
        results.push_back(lif_.evaluate(state, delta));
        if (results.back().fired) ++new_fires;
    }
    if (held_count_ + new_fires > config_.held_spike_entries) {
        ++stats_.held_full_cycles;
        return;
    }
    for (std::uint32_t i = 0; i < neuron_batch_count_; ++i) {
        const auto neuron = neuron_batch_begin_ + i;
        if (!storage_->writeState(neuron, results[i].state) || !storage_->clearDelta(neuron)) {
            throw std::logic_error("v5 CoreState/CoreDelta write failed");
        }
        ++stats_.neuron.issued;
        ++stats_.neuron.completed;
        if (results[i].fired) {
            const FiredSpike spike{active_timestep_ + 1, neuron, 0};
            held_spikes_[active_timestep_ + 1].push_back(spike);
            ++held_count_;
            ++stats_.neurons_fired;
            ++stats_.held_spike.accepted;
        }
        ++stats_.neurons_evaluated;
    }
    next_neuron_ += neuron_batch_count_;
    neuron_batch_pending_ = false;
}

void CorePipeline::processAccumulator_() {
    std::uint32_t processed = 0;
    while (processed < config_.accumulator.width && !accumulator_q_.empty()) {
        auto& item = accumulator_q_.front();
        if (item.ready_cycle > cycle_) break;
        if (!storage_->appendDelta(item.value)) {
            ++stats_.accumulator_stall_cycles;
            break;
        }
        accumulator_q_.pop_front();
        ++processed;
        ++stats_.accumulator_updates;
        ++stats_.accumulator.issued;
        ++stats_.accumulator.completed;
    }
    if (!accumulator_q_.empty() && accumulator_q_.front().ready_cycle <= cycle_ && processed == 0) {
        ++stats_.accumulator_stall_cycles;
    }
}

void CorePipeline::processRetire_() {
    std::uint32_t processed = 0;
    while (processed < config_.retire.width && !retire_q_.empty()) {
        if (accumulator_q_.size() >= config_.accumulator_entries) {
            ++stats_.retire_stall_cycles;
            break;
        }
        const auto entry = retire_q_.pop();
        accumulator_q_.push_back(Timed<RetireEntry>{
            entry, cycle_ + effectiveLatency_(config_.accumulator.latency_cycles)});
        ++processed;
        ++stats_.retire_retired;
        ++stats_.retire.issued;
        ++stats_.retire.completed;
        ++stats_.accumulator.accepted;
    }
}

void CorePipeline::processSynapse_() {
    std::uint32_t processed = 0;
    while (processed < config_.synapse.width && !synapse_q_.empty()) {
        auto& item = synapse_q_.front();
        if (item.ready_cycle > cycle_) break;
        if (retire_q_.full()) {
            ++stats_.synapse_stall_cycles;
            break;
        }
        const auto& response = item.value;
        const RetireEntry entry{
            RetireKey{response.post_neuron, response.source_event_seq, response.edge_ordinal},
            response.timestep,
            response.weight};
        if (!retire_q_.push(entry)) {
            ++stats_.synapse_stall_cycles;
            break;
        }
        synapse_q_.pop_front();
        ++processed;
        ++stats_.synapse_issued;
        ++stats_.synapse.issued;
        ++stats_.synapse.completed;
        ++stats_.retire.accepted;
    }
}

void CorePipeline::processRows_() {
    std::uint32_t processed = 0;
    while (processed < config_.row_lookup.width && !row_q_.empty()) {
        auto& item = row_q_.front();
        if (item.ready_cycle > cycle_) break;
        if (row_request_out_.size() >= config_.row_entries) {
            ++stats_.row_stall_cycles;
            break;
        }
        const auto& spike = item.value;
        const RowKey key = keyFor_(spike.source_neuron, spike.source_event_seq);
        if (rows_.find(key) != rows_.end()) {
            throw std::logic_error("duplicate v5 row request");
        }
        // Row lookup owns the CoreIndex address path even while the provider
        // remains the P1 ideal transport.  This prevents a second vector copy
        // from becoming the implicit index truth.
        std::vector<std::uint8_t> index_byte;
        if (!storage_->readIndex(0, 1, index_byte)) {
            ++stats_.row_lookup.stall_cycles;
            break;
        }
        rows_.emplace(key, RowState{});
        row_request_out_.push_back(RowRequest{
            spike.timestep, spike.source_neuron, spike.source_event_seq, next_row_id_++});
        row_q_.pop_front();
        ++rows_issued_;
        ++stats_.row_requests;
        ++stats_.row_lookup.issued;
        ++processed;
    }
}

void CorePipeline::processIngress_() {
    if (ingress_q_.size() >= config_.ingress_entries) ++stats_.ingress_full_cycles;
    std::uint32_t processed = 0;
    while (processed < config_.ingress.width && !ingress_q_.empty()) {
        auto& item = ingress_q_.front();
        if (item.ready_cycle > cycle_) break;
        if (row_q_.size() >= config_.row_entries) {
            ++stats_.row_stall_cycles;
            break;
        }
        row_q_.push_back(Timed<SpikeInput>{
            item.value, cycle_ + effectiveLatency_(config_.row_lookup.latency_cycles)});
        ingress_q_.pop_front();
        ++processed;
        ++stats_.ingress.issued;
        ++stats_.ingress.completed;
        ++stats_.row_lookup.accepted;
    }
}

bool CorePipeline::allRowsComplete_() const {
    if (rows_.size() != rows_issued_) return false;
    for (const auto& item : rows_) {
        if (!item.second.done || item.second.received != item.second.expected) return false;
    }
    return true;
}

bool CorePipeline::queuesEmpty_() const {
    return ingress_q_.empty() && row_q_.empty() && row_request_out_.empty() && synapse_q_.empty() &&
           retire_q_.empty() && accumulator_q_.empty();
}

void CorePipeline::scheduleNeuron_() {
    if (!sealed_ || !allRowsComplete_() || !queuesEmpty_() || scan_done_ || neuron_batch_pending_) return;
    if (next_neuron_ >= config_.neurons) {
        scan_done_ = true;
        return;
    }
    neuron_batch_begin_ = next_neuron_;
    neuron_batch_count_ = std::min<std::uint32_t>(config_.neuron.width, config_.neurons - next_neuron_);
    neuron_batch_ready_ = cycle_ + effectiveLatency_(config_.neuron.latency_cycles);
    neuron_batch_pending_ = true;
    stats_.neuron.accepted += neuron_batch_count_;
}

bool CorePipeline::tick() {
    if (!active_) return false;
    const auto before = pendingEntries();
    recordStageCycles_();
    // Reverse order is intentional: newly produced items cannot be observed by
    // a later stage until the next clock tick.
    processNeuron_();
    processAccumulator_();
    processRetire_();
    processSynapse_();
    processRows_();
    processIngress_();
    scheduleNeuron_();
    recordOccupancy_();
    ++cycle_;
    ++stats_.cycles;
    return before != pendingEntries() || readyToCommit();
}

bool CorePipeline::readyToCommit() const {
    return active_ && sealed_ && scan_done_ && !neuron_batch_pending_ && allRowsComplete_() && queuesEmpty_();
}

void CorePipeline::commit() {
    if (!readyToCommit()) throw std::logic_error("v5 core Commit before pipeline drained");
    active_ = false;
}

std::vector<RowRequest> CorePipeline::takeRowRequests() {
    std::vector<RowRequest> result;
    result.reserve(row_request_out_.size());
    while (!row_request_out_.empty()) {
        result.push_back(row_request_out_.front());
        row_request_out_.pop_front();
    }
    return result;
}

std::vector<FiredSpike> CorePipeline::takeReleasedSpikes() {
    std::vector<FiredSpike> result;
    result.reserve(released_out_.size());
    while (!released_out_.empty()) {
        result.push_back(released_out_.front());
        released_out_.pop_front();
    }
    return result;
}

std::size_t CorePipeline::pendingEntries() const {
    return ingress_q_.size() + row_q_.size() + row_request_out_.size() + synapse_q_.size() +
           retire_q_.size() + accumulator_q_.size() + (neuron_batch_pending_ ? 1u : 0u);
}

const std::vector<LifNeuronState>& CorePipeline::state() const {
    for (std::uint32_t neuron = 0; neuron < config_.neurons; ++neuron) {
        if (!storage_->readState(neuron, state_snapshot_[neuron])) {
            throw std::logic_error("v5 CoreState snapshot read failed");
        }
    }
    return state_snapshot_;
}

std::uint64_t CorePipeline::functionalHash() const {
    std::uint64_t hash = 0x6a09e667f3bcc909ULL;
    const auto& snapshot = state();
    for (std::size_t neuron = 0; neuron < snapshot.size(); ++neuron) {
        std::uint32_t membrane_bits = 0;
        std::memcpy(&membrane_bits, &snapshot[neuron].membrane, sizeof(membrane_bits));
        hash = hashMix_(hash, neuron);
        hash = hashMix_(hash, membrane_bits);
        hash = hashMix_(hash, snapshot[neuron].refractory);
    }
    return hash;
}

} // namespace v5
} // namespace SnnDL
} // namespace SST
