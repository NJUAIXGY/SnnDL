#include "CorePipeline.h"

#include <algorithm>
#include <array>
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
    : config_(config), lif_(config.lif), cuba_lif_(config.cuba_lif), if_(config.if_op),
      retire_q_(config.retire_entries), storage_(nullptr),
      state_snapshot_(config.neurons), cuba_state_snapshot_(config.neurons) {
    if (config_.neurons == 0 || config_.ingress_entries == 0 || config_.row_entries == 0 ||
        config_.synapse_entries == 0 || config_.accumulator_entries == 0 ||
        config_.held_spike_entries == 0) {
        throw std::invalid_argument("v5 core queue and neuron capacities must be positive");
    }
    if (!config_.schedule_stages.empty()) {
        static const char* expected_ids[] = {
            "preload", "ingress", "row-lookup", "synapse-issue", "neuron-update",
            "retire", "seal", "commit", "drain",
        };
        if (config_.schedule_stages.size() != sizeof(expected_ids) / sizeof(expected_ids[0])) {
            throw std::invalid_argument("SchedulePlan descriptor must contain exactly nine stages");
        }
        for (std::size_t index = 0; index < config_.schedule_stages.size(); ++index) {
            const auto& stage = config_.schedule_stages[index];
            if (stage.id != expected_ids[index] || stage.operation.empty() ||
                stage.resource.empty() || stage.request_class.empty() ||
                stage.retry_policy.empty() || stage.issue_group_id.empty() ||
                stage.max_inflight == 0) {
                throw std::invalid_argument("SchedulePlan descriptor has an invalid stage");
            }
            for (const auto& dependency : stage.dependencies) {
                if (dependency == stage.id) {
                    throw std::invalid_argument("SchedulePlan descriptor contains a self dependency");
                }
            }
            if (index == 0 && !stage.dependencies.empty()) {
                throw std::invalid_argument("SchedulePlan preload stage must have no dependencies");
            }
            if (index > 0 && (stage.dependencies.size() != 1 ||
                              stage.dependencies.front() != expected_ids[index - 1])) {
                throw std::invalid_argument("SchedulePlan descriptor dependencies are not a linear timestep DAG");
            }
        }
    }
    config_.ingress.width = positiveWidth(config_.ingress.width);
    config_.row_lookup.width = positiveWidth(config_.row_lookup.width);
    config_.synapse.width = positiveWidth(config_.synapse.width);
    config_.retire.width = positiveWidth(config_.retire.width);
    config_.accumulator.width = positiveWidth(config_.accumulator.width);
    config_.neuron.width = positiveWidth(config_.neuron.width);
    if (config_.neuron_bindings.empty()) {
        config_.neuron_bindings.resize(config_.neurons);
        for (auto& binding : config_.neuron_bindings) {
            binding.kind = config_.neuron_operator;
            binding.lif = config_.lif;
            binding.cuba_lif = config_.cuba_lif;
            binding.if_op = config_.if_op;
        }
    } else if (config_.neuron_bindings.size() != config_.neurons) {
        throw std::invalid_argument("v5 neuron_bindings must cover every local neuron");
    }
    config_.storage.neurons = config_.neurons;
    config_.storage.execution_profile = config_.execution_profile;
    // CoreDelta is resident state, not a transient retire queue.  Keep its
    // fallback independent from pipeline backpressure so a small retire queue
    // cannot make a valid multi-edge row impossible to commit.
    if (config_.storage.max_delta_entries_per_neuron == 0) {
        config_.storage.max_delta_entries_per_neuron =
            CoreStorageV5Config{}.max_delta_entries_per_neuron;
    }
    storage_ = std::make_unique<CoreStorageV5>(config_.storage);
    neuron_busy_.assign(config_.neurons, 0);
}

bool CorePipeline::asyncStorage_() const {
    return config_.execution_profile == CoreStorageExecutionProfile::AsyncCompletion;
}

bool CorePipeline::storageIdle_() const {
    return epoch_barrier_ready_ && index_waits_.empty() && append_txns_.empty() && neuron_txns_.empty();
}

bool CorePipeline::pollCompletion_(StorageToken token, StorageCompletion& completion) {
    if (!token) return false;
    const auto found = bound_.find(token.id);
    if (found == bound_.end()) return false;
    completion = std::move(found->second);
    bound_.erase(found);
    if (!storage_->consume(token)) throw std::logic_error("CA-5A completion consumed twice");
    if (!completion.ok) {
        throw std::logic_error("CA-5A SRAM request failed closed: " + completion.stall_reason);
    }
    return true;
}

std::size_t CorePipeline::scheduleCapacity_(const char* stage_id, std::size_t fallback) const {
    if (!config_.schedule_admission_enabled || config_.schedule_stages.empty()) return fallback;
    for (const auto& stage : config_.schedule_stages) {
        if (stage.id == stage_id) return std::min(fallback, stage.max_inflight);
    }
    throw std::logic_error(std::string("SchedulePlan descriptor is missing stage ") + stage_id);
}

std::size_t CorePipeline::scheduleWidth_(const char* stage_id, std::size_t fallback) const {
    if (!config_.schedule_admission_enabled || config_.schedule_stages.empty()) return fallback;
    for (const auto& stage : config_.schedule_stages) {
        if (stage.id == stage_id) return std::min(fallback, stage.max_inflight);
    }
    throw std::logic_error(std::string("SchedulePlan descriptor is missing stage ") + stage_id);
}

bool CorePipeline::scheduleReady_(const char* stage_id) const {
    if (!config_.schedule_admission_enabled || config_.schedule_stages.empty()) return true;
    for (const auto& stage : config_.schedule_stages) {
        if (stage.id == stage_id) return cycle_ - timestep_origin_ >= stage.earliest_cycle;
    }
    throw std::logic_error(std::string("SchedulePlan descriptor is missing stage ") + stage_id);
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

CorePipeline::RowKey CorePipeline::keyFor_(std::uint64_t source_neuron,
                                           std::uint64_t source_event_seq) const {
    return RowKey{source_neuron, source_event_seq};
}

void CorePipeline::clearPipelineState_() {
    ingress_q_.clear();
    row_q_.clear();
    row_request_out_.clear();
    synapse_q_.clear();
    retire_q_.clear();
    accumulator_q_.clear();
    rows_.clear();
    stats_ = CorePipelineStats{};
    next_row_id_ = 1;
    rows_issued_ = 0;
    next_neuron_ = 0;
    neuron_batch_pending_ = false;
    neuron_batch_begin_ = 0;
    neuron_batch_count_ = 0;
    neuron_batch_ready_ = 0;
    scan_done_ = false;
    sealed_ = false;
    bound_.clear();
    epoch_reset_tokens_.clear();
    epoch_reset_done_.clear();
    route_token_ = {};
    route_submitted_ = false;
    route_done_ = false;
    epoch_barrier_ready_ = false;
    index_waits_.clear();
    append_txns_.clear();
    neuron_txns_.clear();
    neuron_evaluated_ = false;
    neuron_busy_.assign(config_.neurons, 0);
}

void CorePipeline::resetTimestep_() {
    clearPipelineState_();
    storage_->resetTimestep();
}

void CorePipeline::submitEpochRequests_() {
    using AddressSpaceId = ::SnnDL::v5::AddressSpaceId;
    const auto route = storage_->submitRead(
        AddressSpaceId::PeRoute, storage_->requireRouteOffset(0, 1), 1, cycle_);
    if (!route.token) throw std::logic_error("v5 PeRoute binding is not readable");
    route_token_ = route.token;
    route_submitted_ = true;
    const auto zeros = CoreStorageV5::encodeCountRecord(0);
    epoch_reset_tokens_.assign(config_.neurons, StorageToken{});
    epoch_reset_done_.assign(config_.neurons, 0);
    for (std::uint32_t neuron = 0; neuron < config_.neurons; ++neuron) {
        const auto submitted = storage_->submitWrite(
            AddressSpaceId::CoreDelta, storage_->deltaCountByteOffset(neuron), zeros, cycle_);
        if (!submitted.token) throw std::logic_error("P2 CoreDelta reset request failed");
        epoch_reset_tokens_[neuron] = submitted.token;
    }
}

void CorePipeline::tickAsync_() {
    storage_->advance(cycle_);
    for (auto& completion : storage_->takeCompletions()) {
        const auto inserted = bound_.emplace(completion.token.id, std::move(completion));
        if (!inserted.second) throw std::logic_error("CA-5A duplicate completion token");
    }
    consumeEpochBarrier_();
    consumeIndexReads_();
    consumeAppends_();
    consumeNeuronChain_();
    storage_->admit(cycle_);
    issueIndexReads_();
    issueAppends_();
    issueNeuronChain_();
    processRetire_();
    processSynapse_();
    processIngress_();
    scheduleNeuron_();
}

void CorePipeline::consumeEpochBarrier_() {
    if (!route_submitted_ || epoch_barrier_ready_) return;
    if (!route_done_) {
        StorageCompletion completion;
        if (pollCompletion_(route_token_, completion)) route_done_ = true;
    }
    bool all = route_done_;
    for (std::size_t index = 0; index < epoch_reset_tokens_.size(); ++index) {
        if (epoch_reset_done_[index]) continue;
        StorageCompletion completion;
        if (pollCompletion_(epoch_reset_tokens_[index], completion)) epoch_reset_done_[index] = 1;
        else all = false;
    }
    if (!route_done_) all = false;
    if (all && epoch_reset_tokens_.size() == config_.neurons) epoch_barrier_ready_ = true;
}

void CorePipeline::consumeIndexReads_() {
    for (auto& wait : index_waits_) {
        if (wait.done) continue;
        StorageCompletion completion;
        if (!pollCompletion_(wait.token, completion)) continue;
        wait.done = true;
    }
    while (!index_waits_.empty() && index_waits_.front().done) {
        if (row_request_out_.size() >= config_.row_entries) {
            ++stats_.row_stall_cycles;
            ++stats_.row_lookup.stall_cycles;
            break;
        }
        const auto spike = index_waits_.front().spike;
        const RowKey key = keyFor_(spike.source_neuron, spike.source_event_seq);
        if (rows_.find(key) != rows_.end()) throw std::logic_error("duplicate v5 row request");
        rows_.emplace(key, RowState{});
        row_request_out_.push_back(RowRequest{
            spike.timestep, spike.source_neuron, spike.source_event_seq, next_row_id_++});
        index_waits_.pop_front();
        ++rows_issued_;
        ++stats_.row_requests;
        ++stats_.row_lookup.issued;
    }
}

void CorePipeline::consumeAppends_() {
    for (auto& txn : append_txns_) {
        if (!txn.submitted || txn.phase == AppendPhase::Done || txn.phase == AppendPhase::Overflow) continue;
        StorageCompletion completion;
        if (!pollCompletion_(txn.token, completion)) continue;
        txn.submitted = false;
        if (txn.phase == AppendPhase::ReadCount) {
            txn.count = CoreStorageV5::decodeCountRecord(completion.data);
            txn.phase = txn.count >= config_.storage.max_delta_entries_per_neuron
                            ? AppendPhase::Overflow : AppendPhase::WriteEntry;
        } else if (txn.phase == AppendPhase::WriteEntry) {
            txn.phase = AppendPhase::WriteCount;
        } else if (txn.phase == AppendPhase::WriteCount) {
            neuron_busy_[txn.entry.key.post_neuron] = 0;
            txn.phase = AppendPhase::Done;
            ++stats_.accumulator_updates;
            ++stats_.accumulator.issued;
            ++stats_.accumulator.completed;
        }
    }
    std::deque<AppendTxn> kept;
    for (auto& txn : append_txns_) {
        if (txn.phase != AppendPhase::Done) kept.push_back(std::move(txn));
    }
    append_txns_.swap(kept);
}

void CorePipeline::evaluateReadyNeurons_() {
    if (neuron_evaluated_ || neuron_txns_.empty()) return;
    for (const auto& txn : neuron_txns_) {
        if (txn.phase != NeuronPhase::Ready) return;
    }
    std::size_t fires = 0;
    for (auto& txn : neuron_txns_) {
        float delta = 0.0f;
        for (const auto& entry : txn.deltas) delta += entry.weight;
        const auto& binding = config_.neuron_bindings[txn.neuron];
        if (binding.kind == NeuronOperatorKind::CubaLif) {
            CubaLifNeuronOp op(binding.cuba_lif);
            txn.cuba_result = op.evaluate(txn.cuba_state, delta);
            txn.fired = txn.cuba_result.fired;
        } else if (binding.kind == NeuronOperatorKind::Padding) {
            txn.lif_result = LifNeuronResult{txn.lif_state, false};
            txn.fired = false;
        } else if (binding.kind == NeuronOperatorKind::If) {
            IfNeuronOp op(binding.if_op);
            txn.lif_result = op.evaluate(txn.lif_state, delta);
            txn.fired = txn.lif_result.fired;
        } else {
            LifNeuronOp op(binding.lif);
            txn.lif_result = op.evaluate(txn.lif_state, delta);
            txn.fired = txn.lif_result.fired;
        }
        if (txn.fired) ++fires;
    }
    if (held_count_ + fires > config_.held_spike_entries) {
        ++stats_.held_full_cycles;
        ++stats_.neuron.stall_cycles;
        return;
    }
    neuron_evaluated_ = true;
    for (auto& txn : neuron_txns_) txn.phase = NeuronPhase::Write;
}

void CorePipeline::retireFinishedNeurons_() {
    if (neuron_txns_.empty() || !neuron_evaluated_) return;
    for (const auto& txn : neuron_txns_) {
        if (txn.phase != NeuronPhase::Finished) return;
    }
    for (const auto& txn : neuron_txns_) {
        ++stats_.neuron.issued;
        ++stats_.neuron.completed;
        if (txn.fired) {
            held_spikes_[active_timestep_ + 1].push_back(
                FiredSpike{active_timestep_ + 1, txn.neuron, 0});
            ++held_count_;
            ++stats_.neurons_fired;
            ++stats_.held_spike.accepted;
        }
        ++stats_.neurons_evaluated;
        neuron_busy_[txn.neuron] = 0;
    }
    next_neuron_ += neuron_batch_count_;
    neuron_batch_pending_ = false;
    neuron_txns_.clear();
    neuron_evaluated_ = false;
}

void CorePipeline::consumeNeuronChain_() {
    for (auto& txn : neuron_txns_) {
        if (txn.phase == NeuronPhase::Load) {
            if (!txn.state_done) {
                bool ready = !txn.state_reads.empty();
                for (auto& span : txn.state_reads) {
                    if (span.done) continue;
                    StorageCompletion completion;
                    if (!pollCompletion_(span.token, completion)) {
                        ready = false;
                        continue;
                    }
                    span.data = std::move(completion.data);
                    span.done = true;
                }
                if (ready) {
                    std::vector<std::uint8_t> logical(CoreStorageV5::kStateBytes, 0);
                    for (const auto& span : txn.state_reads) {
                        if (span.data.size() != span.bytes || span.logical_offset + span.bytes > logical.size()) {
                            throw std::logic_error("v5 CoreState read failed");
                        }
                        std::copy(span.data.begin(), span.data.end(), logical.begin() + span.logical_offset);
                    }
                    const auto& binding = config_.neuron_bindings[txn.neuron];
                    if (binding.kind == NeuronOperatorKind::CubaLif) {
                        if (!CubaLifNeuronOp::decodeState(logical.data(), logical.size(), txn.cuba_state)) {
                            throw std::logic_error("v5 CubaLIF CoreState read failed");
                        }
                    } else {
                        txn.lif_state = CoreStorageV5::decodeLifRecord(logical);
                    }
                    txn.state_done = true;
                }
            }
            if (!txn.count_done) {
                StorageCompletion completion;
                if (pollCompletion_(txn.count_token, completion)) {
                    txn.delta_count = CoreStorageV5::decodeCountRecord(completion.data);
                    if (txn.delta_count > config_.storage.max_delta_entries_per_neuron) {
                        throw std::logic_error("v5 CoreDelta count is out of range");
                    }
                    txn.count_done = true;
                }
            }
            if (txn.state_done && txn.count_done) {
                txn.phase = txn.delta_count == 0 ? NeuronPhase::Ready : NeuronPhase::LoadEntries;
            }
        } else if (txn.phase == NeuronPhase::LoadEntries) {
            for (std::size_t slot = 0; slot < txn.entry_tokens.size(); ++slot) {
                if (txn.entry_done[slot]) continue;
                StorageCompletion completion;
                if (!pollCompletion_(txn.entry_tokens[slot], completion)) continue;
                txn.deltas[slot] = CoreStorageV5::decodeDeltaRecord(txn.neuron, completion.data);
                txn.entry_done[slot] = 1;
            }
            bool all = txn.entries_submitted && !txn.entry_done.empty();
            for (const auto done : txn.entry_done) if (!done) all = false;
            if (all) {
                std::stable_sort(txn.deltas.begin(), txn.deltas.end(),
                                 [](const RetireEntry& lhs, const RetireEntry& rhs) {
                                     return lhs.key < rhs.key;
                                 });
                txn.phase = NeuronPhase::Ready;
            }
        } else if (txn.phase == NeuronPhase::Write) {
            if (txn.write_submitted && !txn.write_done) {
                bool ready = !txn.state_writes.empty();
                for (auto& span : txn.state_writes) {
                    if (span.done) continue;
                    StorageCompletion completion;
                    if (!pollCompletion_(span.token, completion)) {
                        ready = false;
                        continue;
                    }
                    span.done = true;
                }
                if (ready) txn.write_done = true;
            }
            if (txn.clear_submitted && !txn.clear_done) {
                StorageCompletion completion;
                if (pollCompletion_(txn.clear_token, completion)) txn.clear_done = true;
            }
            if (txn.write_done && txn.clear_done) txn.phase = NeuronPhase::Finished;
        }
    }
    evaluateReadyNeurons_();
    retireFinishedNeurons_();
}

void CorePipeline::issueIndexReads_() {
    if (!scheduleReady_("row-lookup")) return;
    std::size_t outstanding = 0;
    for (const auto& wait : index_waits_) if (!wait.done) ++outstanding;
    const auto width = scheduleWidth_("row-lookup", config_.row_lookup.width);
    while (outstanding < width && !row_q_.empty()) {
        auto& item = row_q_.front();
        if (item.ready_cycle > cycle_) break;
        if (row_request_out_.size() >= config_.row_entries) {
            ++stats_.row_stall_cycles;
            ++stats_.row_lookup.stall_cycles;
            break;
        }
        const auto submitted = storage_->submitRead(
            ::SnnDL::v5::AddressSpaceId::CoreIndex, storage_->requireIndexOffset(0, 1), 1, cycle_);
        if (!submitted.token) throw std::logic_error("v5 CoreIndex binding is not readable");
        IndexWait wait;
        wait.spike = item.value;
        wait.token = submitted.token;
        index_waits_.push_back(wait);
        row_q_.pop_front();
        ++outstanding;
    }
}

void CorePipeline::issueAppends_() {
    using AddressSpaceId = ::SnnDL::v5::AddressSpaceId;
    std::uint32_t started = 0;
    bool stalled = false;
    while (started < config_.accumulator.width && !accumulator_q_.empty()) {
        auto& item = accumulator_q_.front();
        if (item.ready_cycle > cycle_) break;
        if (!epoch_barrier_ready_ || neuron_busy_[item.value.key.post_neuron]) {
            stalled = true;
            break;
        }
        AppendTxn txn;
        txn.entry = item.value;
        txn.phase = AppendPhase::ReadCount;
        neuron_busy_[item.value.key.post_neuron] = 1;
        append_txns_.push_back(std::move(txn));
        accumulator_q_.pop_front();
        ++started;
    }
    if (stalled) ++stats_.accumulator_stall_cycles;
    for (auto& txn : append_txns_) {
        if (txn.submitted || txn.phase == AppendPhase::Done || txn.phase == AppendPhase::Overflow) continue;
        StorageSubmitResult submitted;
        if (txn.phase == AppendPhase::ReadCount) {
            submitted = storage_->submitRead(
                AddressSpaceId::CoreDelta, storage_->deltaCountByteOffset(txn.entry.key.post_neuron),
                CoreStorageV5::kDeltaCountBytes, cycle_);
        } else if (txn.phase == AppendPhase::WriteEntry) {
            submitted = storage_->submitWrite(
                AddressSpaceId::CoreDelta,
                storage_->deltaEntryByteOffset(txn.entry.key.post_neuron, txn.count),
                CoreStorageV5::encodeDeltaRecord(txn.entry), cycle_);
        } else {
            submitted = storage_->submitWrite(
                AddressSpaceId::CoreDelta, storage_->deltaCountByteOffset(txn.entry.key.post_neuron),
                CoreStorageV5::encodeCountRecord(txn.count + 1), cycle_);
        }
        if (!submitted.token) throw std::logic_error("v5 CoreDelta append request rejected");
        txn.token = submitted.token;
        txn.submitted = true;
    }
}

void CorePipeline::issueNeuronChain_() {
    using AddressSpaceId = ::SnnDL::v5::AddressSpaceId;
    if (!neuron_batch_pending_ || neuron_batch_ready_ > cycle_) return;
    if (!epoch_barrier_ready_) {
        ++stats_.neuron.stall_cycles;
        return;
    }
    if (neuron_txns_.empty()) {
        for (std::uint32_t index = 0; index < neuron_batch_count_; ++index) {
            if (neuron_busy_[neuron_batch_begin_ + index]) {
                ++stats_.neuron.stall_cycles;
                return;
            }
        }
        neuron_txns_.reserve(neuron_batch_count_);
        for (std::uint32_t index = 0; index < neuron_batch_count_; ++index) {
            const auto neuron = neuron_batch_begin_ + index;
            NeuronTxn txn;
            txn.neuron = neuron;
            for (const auto& span : storage_->stateSpans(neuron)) {
                const auto state = storage_->submitRead(
                    AddressSpaceId::CoreState, span.offset, span.bytes, cycle_);
                if (!state.token) throw std::logic_error("v5 CoreState/CoreDelta read rejected");
                NeuronTxn::SpanWait wait;
                wait.token = state.token;
                wait.logical_offset = span.logical_offset;
                wait.bytes = span.bytes;
                txn.state_reads.push_back(std::move(wait));
            }
            const auto count = storage_->submitRead(
                AddressSpaceId::CoreDelta, storage_->deltaCountByteOffset(neuron),
                CoreStorageV5::kDeltaCountBytes, cycle_);
            if (!count.token || txn.state_reads.empty()) {
                throw std::logic_error("v5 CoreState/CoreDelta read rejected");
            }
            txn.count_token = count.token;
            neuron_busy_[neuron] = 1;
            neuron_txns_.push_back(std::move(txn));
        }
        return;
    }
    for (auto& txn : neuron_txns_) {
        if (txn.phase == NeuronPhase::LoadEntries && !txn.entries_submitted) {
            txn.entry_tokens.resize(txn.delta_count);
            txn.entry_done.assign(txn.delta_count, 0);
            txn.deltas.resize(txn.delta_count);
            for (std::uint32_t slot = 0; slot < txn.delta_count; ++slot) {
                const auto submitted = storage_->submitRead(
                    AddressSpaceId::CoreDelta, storage_->deltaEntryByteOffset(txn.neuron, slot),
                    CoreStorageV5::kDeltaEntryBytes, cycle_);
                if (!submitted.token) throw std::logic_error("v5 CoreDelta entry read rejected");
                txn.entry_tokens[slot] = submitted.token;
            }
            txn.entries_submitted = true;
        }
        if (txn.phase == NeuronPhase::Write && !txn.write_submitted) {
            std::vector<std::uint8_t> bytes;
            if (config_.neuron_bindings[txn.neuron].kind == NeuronOperatorKind::CubaLif) {
                std::array<std::uint8_t, CubaLifNeuronOp::kStateBytes> encoded{};
                CubaLifNeuronOp::encodeState(txn.cuba_result.state, encoded);
                bytes.assign(encoded.begin(), encoded.end());
            } else {
                bytes = CoreStorageV5::encodeLifRecord(txn.lif_result.state);
            }
            for (const auto& span : storage_->stateSpans(txn.neuron)) {
                if (span.logical_offset + span.bytes > bytes.size()) {
                    throw std::logic_error("v5 CoreState/CoreDelta write failed");
                }
                const std::vector<std::uint8_t> slice(bytes.begin() + span.logical_offset,
                                                      bytes.begin() + span.logical_offset + span.bytes);
                const auto submitted = storage_->submitWrite(
                    AddressSpaceId::CoreState, span.offset, slice, cycle_);
                if (!submitted.token) throw std::logic_error("v5 CoreState/CoreDelta write failed");
                NeuronTxn::SpanWait wait;
                wait.token = submitted.token;
                wait.logical_offset = span.logical_offset;
                wait.bytes = span.bytes;
                txn.state_writes.push_back(std::move(wait));
            }
            txn.write_submitted = true;
        }
        if (txn.phase == NeuronPhase::Write && !txn.clear_submitted) {
            const auto submitted = storage_->submitWrite(
                AddressSpaceId::CoreDelta, storage_->deltaCountByteOffset(txn.neuron),
                CoreStorageV5::encodeCountRecord(0), cycle_);
            if (!submitted.token) throw std::logic_error("v5 CoreState/CoreDelta write failed");
            txn.clear_token = submitted.token;
            txn.clear_submitted = true;
        }
    }
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
    // Hand the declared timestep to the typed storage binding before any access
    // it traces, so the delta-count reset and the PeRoute probe below are
    // attributed to this timestep rather than to an invented zero.
    storage_->beginTimestep(timestep);
    if (asyncStorage_()) {
        if (!storage_->drained()) {
            throw std::logic_error("CA-5A refuses a new epoch while SRAM requests are outstanding");
        }
        storage_->beginEpoch();
        clearPipelineState_();
        submitEpochRequests_();
    } else {
        resetTimestep_();
        std::vector<std::uint8_t> route_byte;
        if (!storage_->readRoute(0, 1, route_byte)) {
            throw std::logic_error("v5 PeRoute binding is not readable");
        }
    }
    timestep_origin_ = cycle_;
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
    const auto ingress_capacity = config_.schedule_stages.empty()
        ? (config_.schedule_admission_enabled && config_.schedule_ingress_entries > 0
            ? std::min(config_.ingress_entries, config_.schedule_ingress_entries)
            : config_.ingress_entries)
        : scheduleCapacity_("ingress", config_.ingress_entries);
    if (ingress_q_.size() >= ingress_capacity) {
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
    const auto synapse_capacity = config_.schedule_stages.empty()
        ? (config_.schedule_admission_enabled && config_.schedule_synapse_entries > 0
            ? std::min(config_.synapse_entries, config_.schedule_synapse_entries)
            : config_.synapse_entries)
        : scheduleCapacity_("synapse-issue", config_.synapse_entries);
    if (!active_ || response.timestep != active_timestep_ ||
        response.post_neuron >= config_.neurons || synapse_q_.size() >= synapse_capacity) {
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
    struct EvaluatedNeuron {
        NeuronOperatorKind kind = NeuronOperatorKind::Lif;
        LifNeuronResult lif;
        CubaLifNeuronResult cuba_lif;
        bool fired = false;
    };
    std::size_t new_fires = 0;
    std::vector<EvaluatedNeuron> results;
    results.reserve(neuron_batch_count_);
    for (std::uint32_t i = 0; i < neuron_batch_count_; ++i) {
        const auto neuron = neuron_batch_begin_ + i;
        const auto& binding = config_.neuron_bindings[neuron];
        std::vector<RetireEntry> ordered;
        float delta = 0.0f;
        if (binding.kind == NeuronOperatorKind::CubaLif) {
            CubaLifNeuronState state;
            if (!storage_->readCubaLifState(neuron, state) ||
                !storage_->readDeltaEntries(neuron, ordered)) {
                ++stats_.neuron.stall_cycles;
                return;
            }
            for (const auto& entry : ordered) delta += entry.weight;
            CubaLifNeuronOp op(binding.cuba_lif);
            const auto result = op.evaluate(state, delta);
            results.push_back(EvaluatedNeuron{binding.kind, {}, result, result.fired});
        } else {
            LifNeuronState state;
            if (!storage_->readState(neuron, state) || !storage_->readDeltaEntries(neuron, ordered)) {
                ++stats_.neuron.stall_cycles;
                return;
            }
            for (const auto& entry : ordered) delta += entry.weight;
            if (binding.kind == NeuronOperatorKind::Padding) {
                results.push_back(EvaluatedNeuron{binding.kind, LifNeuronResult{state, false}, {}, false});
            } else if (binding.kind == NeuronOperatorKind::If) {
                IfNeuronOp op(binding.if_op);
                const auto result = op.evaluate(state, delta);
                results.push_back(EvaluatedNeuron{binding.kind, result, {}, result.fired});
            } else {
                LifNeuronOp op(binding.lif);
                const auto result = op.evaluate(state, delta);
                results.push_back(EvaluatedNeuron{binding.kind, result, {}, result.fired});
            }
        }
        if (results.back().fired) ++new_fires;
    }
    if (held_count_ + new_fires > config_.held_spike_entries) {
        ++stats_.held_full_cycles;
        return;
    }
    for (std::uint32_t i = 0; i < neuron_batch_count_; ++i) {
        const auto neuron = neuron_batch_begin_ + i;
        const bool state_written = results[i].kind == NeuronOperatorKind::CubaLif
                                       ? storage_->writeCubaLifState(neuron, results[i].cuba_lif.state)
                                       : storage_->writeState(neuron, results[i].lif.state);
        if (!state_written || !storage_->clearDelta(neuron)) {
            throw std::logic_error("v5 CoreState/CoreDelta write failed");
        }
        ++stats_.neuron.issued;
        ++stats_.neuron.completed;
        const bool fired = results[i].fired;
        if (fired) {
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
    if (!scheduleReady_("synapse-issue")) return;
    std::uint32_t processed = 0;
    while (processed < scheduleWidth_("synapse-issue", config_.synapse.width) && !synapse_q_.empty()) {
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
    if (!scheduleReady_("row-lookup")) return;
    std::uint32_t processed = 0;
    while (processed < scheduleWidth_("row-lookup", config_.row_lookup.width) && !row_q_.empty()) {
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
    if (!scheduleReady_("ingress")) return;
    if (ingress_q_.size() >= config_.ingress_entries) ++stats_.ingress_full_cycles;
    std::uint32_t processed = 0;
    while (processed < scheduleWidth_("ingress", config_.ingress.width) && !ingress_q_.empty()) {
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
    if (!scheduleReady_("neuron-update")) return;
    if (!sealed_ || !allRowsComplete_() || !queuesEmpty_() || scan_done_ || neuron_batch_pending_) return;
    if (asyncStorage_() && !storageIdle_()) return;
    if (next_neuron_ >= config_.neurons) {
        scan_done_ = true;
        return;
    }
    neuron_batch_begin_ = next_neuron_;
    neuron_batch_count_ = std::min<std::uint32_t>(
        static_cast<std::uint32_t>(scheduleWidth_("neuron-update", config_.neuron.width)),
        config_.neurons - next_neuron_);
    neuron_batch_ready_ = cycle_ + effectiveLatency_(config_.neuron.latency_cycles);
    neuron_batch_pending_ = true;
    stats_.neuron.accepted += neuron_batch_count_;
}

bool CorePipeline::tick() {
    if (!active_) return false;
    const auto before = pendingEntries();
    recordStageCycles_();
    if (asyncStorage_()) {
        // One logical tick, one SRAM advance per region:
        // advance, collect, bind, consume, admit/issue, then the caller
        // increments the Core cycle.
        tickAsync_();
    } else {
        // Reverse order is intentional: newly produced items cannot be observed by
        // a later stage until the next clock tick.
        processNeuron_();
        processAccumulator_();
        processRetire_();
        processSynapse_();
        processRows_();
        processIngress_();
        scheduleNeuron_();
    }
    recordOccupancy_();
    ++cycle_;
    ++stats_.cycles;
    return before != pendingEntries() || readyToCommit();
}

bool CorePipeline::readyToCommit() const {
    const bool structural = active_ && sealed_ && scan_done_ && !neuron_batch_pending_ &&
                            allRowsComplete_() && queuesEmpty_();
    if (!structural) return false;
    if (!asyncStorage_()) return true;
    return storageIdle_() && storage_->drained();
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
           retire_q_.size() + accumulator_q_.size() + index_waits_.size() + append_txns_.size() +
           (neuron_batch_pending_ ? 1u : 0u);
}

const std::vector<LifNeuronState>& CorePipeline::state() const {
    for (const auto& binding : config_.neuron_bindings) {
        if (binding.kind != NeuronOperatorKind::Lif) {
            throw std::logic_error("v5 CorePipeline state() is only valid for all-LIF bindings");
        }
    }
    if (asyncStorage_() && !storage_->drained()) {
        throw std::logic_error("v5 state() requires CoreStorage to be drained");
    }
    for (std::uint32_t neuron = 0; neuron < config_.neurons; ++neuron) {
        if (!storage_->readCompletedState(neuron, state_snapshot_[neuron])) {
            throw std::logic_error("v5 CoreState snapshot read failed");
        }
    }
    return state_snapshot_;
}

const std::vector<CubaLifNeuronState>& CorePipeline::cubaState() const {
    for (const auto& binding : config_.neuron_bindings) {
        if (binding.kind != NeuronOperatorKind::CubaLif) {
            throw std::logic_error("v5 CorePipeline cubaState() is only valid for all-CubaLIF bindings");
        }
    }
    if (asyncStorage_() && !storage_->drained()) {
        throw std::logic_error("v5 cubaState() requires CoreStorage to be drained");
    }
    for (std::uint32_t neuron = 0; neuron < config_.neurons; ++neuron) {
        if (!storage_->readCompletedCubaLifState(neuron, cuba_state_snapshot_[neuron])) {
            throw std::logic_error("v5 CubaLIF CoreState snapshot read failed");
        }
    }
    return cuba_state_snapshot_;
}

std::uint64_t CorePipeline::functionalHash() const {
    if (asyncStorage_() && !storage_->drained()) {
        throw std::logic_error("v5 functionalHash() requires CoreStorage to be drained");
    }
    std::uint64_t hash = 0x6a09e667f3bcc909ULL;
    for (std::size_t neuron = 0; neuron < config_.neuron_bindings.size(); ++neuron) {
        hash = hashMix_(hash, neuron);
        hash = hashMix_(hash, static_cast<std::uint8_t>(config_.neuron_bindings[neuron].kind));
        if (config_.neuron_bindings[neuron].kind == NeuronOperatorKind::CubaLif) {
            CubaLifNeuronState snapshot;
            if (!storage_->readCompletedCubaLifState(static_cast<std::uint32_t>(neuron), snapshot)) {
                throw std::logic_error("v5 CubaLIF CoreState hash read failed");
            }
            std::uint32_t current_bits = 0;
            std::uint32_t membrane_bits = 0;
            std::memcpy(&current_bits, &snapshot.synaptic_current, sizeof(current_bits));
            std::memcpy(&membrane_bits, &snapshot.membrane, sizeof(membrane_bits));
            hash = hashMix_(hash, current_bits);
            hash = hashMix_(hash, membrane_bits);
        } else {
            LifNeuronState snapshot;
            if (!storage_->readCompletedState(static_cast<std::uint32_t>(neuron), snapshot)) {
                throw std::logic_error("v5 LIF CoreState hash read failed");
            }
            std::uint32_t membrane_bits = 0;
            std::memcpy(&membrane_bits, &snapshot.membrane, sizeof(membrane_bits));
            hash = hashMix_(hash, membrane_bits);
            hash = hashMix_(hash, snapshot.refractory);
        }
    }
    return hash;
}

} // namespace v5
} // namespace SnnDL
} // namespace SST
