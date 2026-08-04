// -*- c++ -*-

#include <sst/core/sst_config.h>

#include "workloads/tensor/TensorWorkload.h"

#include <sst/core/output.h>

#include "INocTransport.h"
#include "NocPacketEvent.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace SST { namespace SnnDL {

namespace {

inline bool has_collective_magic_(const std::vector<uint8_t>& payload) {
    return payload.size() >= 4 &&
           payload[0] == static_cast<uint8_t>('C') &&
           payload[1] == static_cast<uint8_t>('O') &&
           payload[2] == static_cast<uint8_t>('L') &&
           payload[3] == static_cast<uint8_t>('L');
}

inline bool has_magic4_(const std::vector<uint8_t>& payload, char a, char b, char c, char d) {
    return payload.size() >= 4 &&
           payload[0] == static_cast<uint8_t>(a) &&
           payload[1] == static_cast<uint8_t>(b) &&
           payload[2] == static_cast<uint8_t>(c) &&
           payload[3] == static_cast<uint8_t>(d);
}

inline uint32_t read_u32_le_(const std::vector<uint8_t>& payload, size_t off) {
    if (off + 4 > payload.size()) return 0;
    return static_cast<uint32_t>(payload[off]) |
           (static_cast<uint32_t>(payload[off + 1]) << 8) |
           (static_cast<uint32_t>(payload[off + 2]) << 16) |
           (static_cast<uint32_t>(payload[off + 3]) << 24);
}

inline void write_u32_le_(std::vector<uint8_t>& payload, size_t off, uint32_t v) {
    if (off + 4 > payload.size()) return;
    payload[off] = static_cast<uint8_t>(v & 0xffu);
    payload[off + 1] = static_cast<uint8_t>((v >> 8) & 0xffu);
    payload[off + 2] = static_cast<uint8_t>((v >> 16) & 0xffu);
    payload[off + 3] = static_cast<uint8_t>((v >> 24) & 0xffu);
}


} // namespace

bool TensorWorkload::collectivePendingActive_() const {
    if (!collective_pending_active_) return false;
    return collective_pending_sent_bytes_ < collective_pending_total_bytes_;
}

uint32_t TensorWorkload::collectiveRingParticipantCount_() const {
    const uint32_t total_nodes = (rt_.total_nodes > 0) ? static_cast<uint32_t>(rt_.total_nodes) : 1u;
    return std::max<uint32_t>(total_nodes, 1u);
}

uint32_t TensorWorkload::collectiveRingStepsPerChunk_() const {
    const uint32_t participants = collectiveRingParticipantCount_();
    const uint32_t hop = (participants > 1) ? (participants - 1u) : 1u;
    if (cfg_.collective_type == "allreduce") {
        return hop * 2u;
    }
    if (cfg_.collective_type == "allgather" || cfg_.collective_type == "reducescatter") {
        return hop;
    }
    return hop;
}

uint32_t TensorWorkload::collectiveRingChunkPayloadBytes_(uint32_t chunk_index) const {
    const uint32_t base_chunk = (cfg_.collective_chunk_bytes > 0)
                                    ? cfg_.collective_chunk_bytes
                                    : std::max<uint32_t>(cfg_.collective_packet_bytes, 8u);
    if (base_chunk == 0) return 0;
    if (collective_ring_chunks_total_ == 0) return base_chunk;
    if (chunk_index + 1u < collective_ring_chunks_total_) return base_chunk;

    const uint64_t total = (collective_active_bytes_ != 0) ? collective_active_bytes_ : cfg_.collective_bytes;
    const uint64_t rem = (base_chunk > 0) ? (total % static_cast<uint64_t>(base_chunk)) : 0ull;
    if (rem == 0ull) return base_chunk;
    return static_cast<uint32_t>(rem);
}

uint32_t TensorWorkload::collectiveRingStepPayloadBytes_(uint32_t chunk_index, uint32_t step_in_chunk) const {
    const uint32_t chunk_payload = collectiveRingChunkPayloadBytes_(chunk_index);
    if (chunk_payload == 0) return 0;

    if (collective_2d_active_) {
        // 2D staged allreduce: split chunk payload along X (row), then split the kept row-segment along Y (col).
        const uint32_t dim_x = std::max<uint32_t>(collective_2d_dim_x_ ? collective_2d_dim_x_ : 1u, 1u);
        const uint32_t dim_y = std::max<uint32_t>(collective_2d_dim_y_ ? collective_2d_dim_y_ : 1u, 1u);
        const uint32_t row_hop = (collective_2d_row_hop_ > 0) ? collective_2d_row_hop_ : 1u;
        const uint32_t col_hop = (collective_2d_col_hop_ > 0) ? collective_2d_col_hop_ : 1u;

        const uint32_t row_base = chunk_payload / dim_x;
        const uint32_t row_rem = chunk_payload % dim_x;
        const uint32_t row_last_id = (dim_x > 0) ? (dim_x - 1u) : 0u;
        const uint32_t row_last_bytes = row_base + ((row_last_id < row_rem) ? 1u : 0u);

        const uint32_t col_base = (dim_y > 0) ? (row_last_bytes / dim_y) : row_last_bytes;
        const uint32_t col_rem = (dim_y > 0) ? (row_last_bytes % dim_y) : 0u;
        const uint32_t col_last_id = (dim_y > 0) ? (dim_y - 1u) : 0u;
        const uint32_t col_last_bytes = col_base + ((col_last_id < col_rem) ? 1u : 0u);

        const Collective2dStage stage = collective2dStageForStep_(step_in_chunk);
        if (stage == Collective2dStage::RowRS || stage == Collective2dStage::RowAG) {
            uint32_t local_step = step_in_chunk;
            if (stage == Collective2dStage::RowAG) {
                local_step = (step_in_chunk >= (row_hop + 2u * col_hop)) ? (step_in_chunk - (row_hop + 2u * col_hop)) : 0u;
            }
            const uint32_t seg_id = (row_hop > 0) ? (local_step % row_hop) : 0u;
            return row_base + ((seg_id < row_rem) ? 1u : 0u);
        }

        // ColRS / ColAG
        uint32_t local_step = 0;
        if (stage == Collective2dStage::ColRS) {
            local_step = (step_in_chunk >= row_hop) ? (step_in_chunk - row_hop) : 0u;
        } else {
            // ColAG
            local_step = (step_in_chunk >= (row_hop + col_hop)) ? (step_in_chunk - (row_hop + col_hop)) : 0u;
        }
        const uint32_t seg_id = (col_hop > 0) ? (local_step % col_hop) : 0u;
        (void)col_last_bytes; // keep for symmetry / documentation; excluded by seg_id range when dim_y>1.
        return col_base + ((seg_id < col_rem) ? 1u : 0u);
    }

    // Ring-like chunked collective: split each chunk into P segments and send one segment per step.
    const uint32_t participants = collectiveRingParticipantCount_();
    const uint32_t parts = std::max<uint32_t>(participants, 1u);
    const uint32_t base = chunk_payload / parts;
    const uint32_t rem = chunk_payload % parts;
    const uint32_t hop = (parts > 1u) ? (parts - 1u) : 1u;
    const uint32_t seg_id = (hop > 0) ? (step_in_chunk % hop) : 0u;
    return base + ((seg_id < rem) ? 1u : 0u);
}

uint32_t TensorWorkload::collectiveRingNextDestNode_() const {
    const uint32_t total_nodes = (rt_.total_nodes > 0) ? static_cast<uint32_t>(rt_.total_nodes) : 1u;
    if (total_nodes <= 1u) return static_cast<uint32_t>(rt_.node_id);

    if ((cfg_.collective_pattern == "mesh_x" || cfg_.collective_pattern == "mesh_xy") && total_nodes > 1u) {
        const uint32_t mesh_size = static_cast<uint32_t>(std::sqrt(static_cast<double>(total_nodes)));
        if (mesh_size > 0 && mesh_size * mesh_size == total_nodes) {
            const uint32_t node_x = static_cast<uint32_t>(rt_.node_id) % mesh_size;
            const uint32_t node_y = static_cast<uint32_t>(rt_.node_id) / mesh_size;
            const uint32_t east = node_y * mesh_size + ((node_x + 1u) % mesh_size);
            if (cfg_.collective_pattern == "mesh_xy" && (collective_ring_step_index_ & 1u)) {
                return ((node_y + 1u) % mesh_size) * mesh_size + node_x;
            }
            return east;
        }
    }
    return (static_cast<uint32_t>(rt_.node_id) + 1u) % total_nodes;
}

TensorWorkload::Collective2dStage TensorWorkload::collective2dStageForStep_(uint32_t step_in_chunk) const {
    const uint32_t row_hop = (collective_2d_row_hop_ > 0) ? collective_2d_row_hop_ : 1u;
    const uint32_t col_hop = (collective_2d_col_hop_ > 0) ? collective_2d_col_hop_ : 1u;
    if (step_in_chunk < row_hop) return Collective2dStage::RowRS;
    if (step_in_chunk < row_hop + col_hop) return Collective2dStage::ColRS;
    if (step_in_chunk < row_hop + 2u * col_hop) return Collective2dStage::ColAG;
    return Collective2dStage::RowAG;
}

uint32_t TensorWorkload::collective2dNextDestNodeForStep_(uint32_t step_in_chunk) const {
    const uint32_t total_nodes = (rt_.total_nodes > 0) ? static_cast<uint32_t>(rt_.total_nodes) : 1u;
    const uint32_t node = static_cast<uint32_t>(rt_.node_id);
    if (total_nodes <= 1u) return node;

    const uint32_t dim_x = (collective_2d_dim_x_ > 0) ? collective_2d_dim_x_ : total_nodes;
    const uint32_t dim_y = (collective_2d_dim_y_ > 0) ? collective_2d_dim_y_ : 1u;
    if (dim_x == 0u || dim_y == 0u) return (node + 1u) % total_nodes;
    const uint64_t prod = static_cast<uint64_t>(dim_x) * static_cast<uint64_t>(dim_y);
    if (prod != static_cast<uint64_t>(total_nodes)) {
        return (node + 1u) % total_nodes;
    }

    uint32_t x = 0;
    uint32_t y = 0;
    if (cfg_.collective_2d_row_major) {
        x = node % dim_x;
        y = node / dim_x;
    } else {
        y = node % dim_y;
        x = node / dim_y;
    }
    auto to_node = [&](uint32_t xx, uint32_t yy) -> uint32_t {
        if (cfg_.collective_2d_row_major) return yy * dim_x + xx;
        return xx * dim_y + yy;
    };

    const Collective2dStage stage = collective2dStageForStep_(step_in_chunk);
    if (stage == Collective2dStage::RowRS || stage == Collective2dStage::RowAG) {
        const uint32_t nx = (x + 1u) % dim_x;
        return to_node(nx, y);
    }
    const uint32_t ny = (y + 1u) % dim_y;
    return to_node(x, ny);
}


bool TensorWorkload::collectiveUseEventCreditReturn_() const {
    return cfg_.collective_credit_enable &&
           cfg_.collective_algo == "ring_chunked" &&
           cfg_.collective_credit_return_mode == "event_on_recv";
}

void TensorWorkload::onCollectiveCreditIssue_(uint32_t seq,
                                              uint32_t chunk,
                                              uint32_t step,
                                              uint32_t dst_node,
                                              uint16_t dst_core,
                                              uint64_t now_cycle) {
    if (!cfg_.collective_credit_enable) return;
    collective_credit_inflight_chunks_ = saturatingAddU64_(collective_credit_inflight_chunks_, 1ull);
    if (collective_credit_inflight_chunks_ > tensor_collective_inflight_chunks_max_) {
        tensor_collective_inflight_chunks_max_ = collective_credit_inflight_chunks_;
    }

    if (!collectiveUseEventCreditReturn_()) return;
    const CollectiveCreditKey key{seq, chunk, step, dst_node, dst_core};
    auto& state = collective_credit_outstanding_[key];
    state.outstanding = saturatingAddU64_(state.outstanding, 1ull);
    state.issue_cycles.push_back(now_cycle);
}

void TensorWorkload::onCollectiveCreditReturn_(uint32_t seq,
                                               uint32_t chunk,
                                               uint32_t step,
                                               uint32_t credits,
                                               uint32_t src_node,
                                               uint16_t src_core,
                                               uint64_t now_cycle) {
    if (!cfg_.collective_credit_enable) return;
    if (credits == 0) return;

    const CollectiveCreditKey key{seq, chunk, step, src_node, src_core};
    auto it = collective_credit_outstanding_.find(key);
    if (it == collective_credit_outstanding_.end() || it->second.outstanding == 0) {
        if (collective_credit_return_seen_.find(key) != collective_credit_return_seen_.end()) {
            tensor_collective_credit_return_dup_total_ += static_cast<uint64_t>(credits);
        } else {
            tensor_collective_credit_return_orphan_total_ += static_cast<uint64_t>(credits);
        }
        collective_credit_return_seen_.insert(key);
        return;
    }

    collective_credit_return_seen_.insert(key);
    uint64_t applied = 0;
    const uint64_t req = static_cast<uint64_t>(credits);
    while (applied < req && it->second.outstanding > 0) {
        it->second.outstanding -= 1;
        applied += 1;
        if (!it->second.issue_cycles.empty()) {
            const uint64_t issue_cycle = it->second.issue_cycles.front();
            it->second.issue_cycles.pop_front();
            const uint64_t latency = (now_cycle >= issue_cycle) ? (now_cycle - issue_cycle) : 0ull;
            tensor_collective_credit_return_latency_cycles_total_ =
                saturatingAddU64_(tensor_collective_credit_return_latency_cycles_total_, latency);
            if (latency > tensor_collective_credit_return_latency_cycles_max_) {
                tensor_collective_credit_return_latency_cycles_max_ = latency;
            }
        }
    }

    if (it->second.outstanding == 0) {
        it->second.issue_cycles.clear();
        collective_credit_outstanding_.erase(it);
        if (collective_credit_outstanding_.empty()) {
            collective_credit_return_seen_.clear();
        }
    }

    if (applied < req) {
        const uint64_t extra = req - applied;
        tensor_collective_credit_return_orphan_total_ = saturatingAddU64_(
            tensor_collective_credit_return_orphan_total_, extra);
    }

    if (collective_credit_inflight_chunks_ >= applied) {
        collective_credit_inflight_chunks_ -= applied;
    } else {
        collective_credit_inflight_chunks_ = 0;
    }
}

void TensorWorkload::maybeEmitCollectiveCreditReturn_(const NocPacketEvent* packet, uint64_t now_cycle) {
    (void)now_cycle;
    if (!packet) return;
    if (!collectiveUseEventCreditReturn_()) return;
    if (!has_collective_magic_(packet->payload)) return;
    if (packet->payload.size() < 16) return;

    const uint32_t seq = read_u32_le_(packet->payload, 4);
    const uint32_t chunk = read_u32_le_(packet->payload, 8);
    const uint32_t step = read_u32_le_(packet->payload, 12);
    const uint32_t dst_node = packet->src_node;
    const uint16_t dst_core = packet->src_endpoint;
    const uint32_t credits = 1u;

    const CollectiveCreditKey key{seq, chunk, step, dst_node, dst_core};
    auto it = collective_credit_return_pending_credits_.find(key);
    if (it == collective_credit_return_pending_credits_.end()) {
        collective_credit_return_pending_queue_.push_back(key);
        collective_credit_return_pending_credits_.emplace(key, static_cast<uint64_t>(credits));
    } else {
        it->second = saturatingAddU64_(it->second, static_cast<uint64_t>(credits));
    }
}

uint64_t TensorWorkload::emitCollectiveCreditReturnTraffic_(uint64_t now_cycle,
                                                            uint64_t noc_budget_bytes,
                                                            bool& budget_blocked) {
    budget_blocked = false;
    if (!collectiveUseEventCreditReturn_()) return 0;
    if (collective_credit_return_pending_queue_.empty()) return 0;

    constexpr uint64_t kPacketBytes = 20ull;
    const bool uncapped = (noc_budget_bytes == std::numeric_limits<uint64_t>::max());
    uint64_t budget = noc_budget_bytes;
    uint64_t sent_bytes = 0;

    size_t rounds = collective_credit_return_pending_queue_.size();
    while (rounds > 0 && !collective_credit_return_pending_queue_.empty()) {
        rounds -= 1;
        const CollectiveCreditKey key = collective_credit_return_pending_queue_.front();
        collective_credit_return_pending_queue_.pop_front();

        auto it = collective_credit_return_pending_credits_.find(key);
        if (it == collective_credit_return_pending_credits_.end() || it->second == 0) {
            if (it != collective_credit_return_pending_credits_.end()) {
                collective_credit_return_pending_credits_.erase(it);
            }
            continue;
        }

        if (!uncapped && budget < kPacketBytes) {
            budget_blocked = true;
            collective_credit_return_pending_queue_.push_front(key);
            break;
        }

        const uint32_t credits = static_cast<uint32_t>(
            std::min<uint64_t>(it->second, static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())));
        if (credits == 0) {
            collective_credit_return_pending_credits_.erase(it);
            continue;
        }

        const uint32_t dst_node = key.peer_node;
        const uint16_t dst_core = key.peer_core;
        const bool self_is_dst =
            (dst_node == static_cast<uint32_t>(rt_.node_id) && dst_core == static_cast<uint16_t>(rt_.core_id));

        if (self_is_dst) {
            tensor_collective_credit_return_pkts_sent_total_ += 1;
            tensor_collective_credit_return_pkts_recv_total_ += 1;
            onCollectiveCreditReturn_(key.seq, key.chunk, key.step, credits, dst_node, dst_core, now_cycle);
        } else {
            if (!rt_.noc) {
                budget_blocked = true;
                collective_credit_return_pending_queue_.push_front(key);
                break;
            }

            std::vector<uint8_t> payload(static_cast<size_t>(kPacketBytes), 0);
            payload[0] = static_cast<uint8_t>('T');
            payload[1] = static_cast<uint8_t>('C');
            payload[2] = static_cast<uint8_t>('C');
            payload[3] = static_cast<uint8_t>('R');
            write_u32_le_(payload, 4, key.seq);
            write_u32_le_(payload, 8, key.chunk);
            write_u32_le_(payload, 12, key.step);
            write_u32_le_(payload, 16, credits);

            auto* pkt = new NocPacketEvent(rt_.node_id,
                                           dst_node,
                                           static_cast<uint16_t>(rt_.core_id),
                                           dst_core,
                                           NocPacketKind::Control,
                                           now_cycle);
            pkt->payload = payload;

            tensor_pkt_sent_total_ += 1;
            tensor_pkt_bytes_sent_total_ += static_cast<uint64_t>(pkt->payload.size());
            tensor_collective_credit_return_pkts_sent_total_ += 1;

            if (dst_node == static_cast<uint32_t>(rt_.node_id)) {
                rt_.noc->injectLocal(static_cast<int>(dst_core), pkt);
            } else {
                rt_.noc->sendFromCore(static_cast<int>(rt_.core_id), pkt);
            }
        }

        sent_bytes = saturatingAddU64_(sent_bytes, kPacketBytes);
        if (!uncapped) {
            budget = (budget >= kPacketBytes) ? (budget - kPacketBytes) : 0ull;
        }

        if (it->second > static_cast<uint64_t>(credits)) {
            it->second -= static_cast<uint64_t>(credits);
            collective_credit_return_pending_queue_.push_back(key);
        } else {
            collective_credit_return_pending_credits_.erase(it);
        }
    }

    if (!collective_credit_return_pending_queue_.empty() && !uncapped && budget < kPacketBytes) {
        budget_blocked = true;
    }

    return sent_bytes;
}

uint64_t TensorWorkload::emitCollectiveTraffic_(uint64_t now_cycle, uint64_t noc_budget_bytes, bool& budget_blocked) {
    budget_blocked = false;
    bool backpressure_stalled = false;
    const bool soft_backpressure = (cfg_.collective_backpressure_mode == "soft");
    const bool credit_enabled = cfg_.collective_credit_enable;
    const bool event_credit_return = collectiveUseEventCreditReturn_();
    const uint64_t credit_window = credit_enabled
                                       ? static_cast<uint64_t>(std::max<uint32_t>(
                                             cfg_.collective_credit_window_chunks ? cfg_.collective_credit_window_chunks
                                                                                  : cfg_.collective_max_inflight_chunks,
                                             1u))
                                       : std::numeric_limits<uint64_t>::max();
    if (credit_enabled && !event_credit_return && collective_credit_inflight_chunks_ > 0) {
        collective_credit_inflight_chunks_ -= 1;
    }
    auto try_consume_credit = [&]() -> bool {
        if (!credit_enabled) return true;
        if (collective_credit_inflight_chunks_ < credit_window) return true;
        tensor_collective_credit_stall_cycles_total_ += 1;
        backpressure_stalled = true;
        if (!soft_backpressure) {
            budget_blocked = true;
            return false;
        }
        return true;
    };
    auto on_credit_issue = [&](uint32_t seq,
                               uint32_t chunk,
                               uint32_t step,
                               uint32_t dst_node,
                               uint16_t dst_core) {
        if (!credit_enabled) return;
        onCollectiveCreditIssue_(seq, chunk, step, dst_node, dst_core, now_cycle);
    };

    if (!collectiveReady_()) {
        if (cfg_.collective_type != "none" && cfg_.strict && !rt_.noc && rt_.log) {
            rt_.log->fatal(CALL_INFO, -1, "tensor fatal: collective enabled but INocTransport is null (core=%u)\n", rt_.core_id);
        }
        return 0;
    }

    // Program mode uses explicit collective ops; never auto-start based on period.
    if (cfg_.exec_mode == "program" && !collectivePendingActive_()) {
        return 0;
    }

    if (!collectivePendingActive_()) {
        if (cfg_.collective_blocking && collective_epoch_active_) {
            if (cfg_.collective_scope == "per_core") {
                if (collective_epoch_recv_bytes_ < collective_epoch_expected_recv_bytes_) return 0;
                // Completed: allow starting a new epoch.
                markCollectiveEpochDone_(now_cycle);
                const uint32_t done_seq = collective_epoch_seq_;
                collective_epoch_active_ = false;
                collective_epoch_expected_recv_bytes_ = 0;
                collective_epoch_recv_bytes_ = 0;
                collective_recv_bytes_by_seq_.erase(done_seq);
            } else {
                // Group-scoped blocking collective: wait for explicit RELEASE (leader/root broadcast).
                return 0;
            }
        }
        if (now_cycle - collective_last_cycle_ < cfg_.collective_period_cycles) return 0;

        const uint32_t seq32 = static_cast<uint32_t>(collective_seq_ & 0xffffffffu);
        collective_pending_active_ = true;
        collective_pending_seq_ = seq32;
        collective_pending_sent_bytes_ = 0;
        collective_pending_next_dest_ = 0;
        collective_pending_dest_nodes_.clear();
        collective_pending_dest_remaining_bytes_.clear();
        collective_ring_active_ = false;
        collective_ring_total_payload_bytes_ = 0;
        collective_ring_sent_payload_bytes_ = 0;
        collective_ring_reduce_wait_cycles_remaining_ = 0;
        collective_2d_active_ = false;
        collective_2d_dim_x_ = 0;
        collective_2d_dim_y_ = 0;
        collective_2d_row_hop_ = 0;
        collective_2d_col_hop_ = 0;

        if (cfg_.collective_algo == "ring_chunked" || cfg_.collective_algo == "torus_2d_rs_ag") {
            collective_active_bytes_ = cfg_.collective_bytes;
            const uint32_t chunk_bytes = std::max<uint32_t>(
                cfg_.collective_chunk_bytes ? cfg_.collective_chunk_bytes : cfg_.collective_packet_bytes,
                8u);
            uint32_t steps_per_chunk = 0;
            if (cfg_.collective_algo == "torus_2d_rs_ag") {
                collective_2d_active_ = true;
                const uint32_t total_nodes = (rt_.total_nodes > 0) ? static_cast<uint32_t>(rt_.total_nodes) : 1u;
                uint32_t dim_x = cfg_.collective_2d_dim_x;
                uint32_t dim_y = cfg_.collective_2d_dim_y;
                auto fallback_dims = [&]() {
                    const uint32_t sq = static_cast<uint32_t>(std::sqrt(static_cast<double>(total_nodes)));
                    if (sq > 0 && sq * sq == total_nodes) {
                        dim_x = sq;
                        dim_y = sq;
                    } else {
                        dim_x = total_nodes;
                        dim_y = 1u;
                    }
                };
                if (dim_x == 0u || dim_y == 0u ||
                    (static_cast<uint64_t>(dim_x) * static_cast<uint64_t>(dim_y) != static_cast<uint64_t>(total_nodes))) {
                    fallback_dims();
                }
                if (dim_x == 0u) dim_x = 1u;
                if (dim_y == 0u) dim_y = 1u;
                collective_2d_dim_x_ = dim_x;
                collective_2d_dim_y_ = dim_y;
                collective_2d_row_hop_ = (dim_x > 1u) ? (dim_x - 1u) : 1u;
                collective_2d_col_hop_ = (dim_y > 1u) ? (dim_y - 1u) : 1u;
                steps_per_chunk = 2u * (collective_2d_row_hop_ + collective_2d_col_hop_);
            } else {
                steps_per_chunk = collectiveRingStepsPerChunk_();
            }
            steps_per_chunk = std::max<uint32_t>(steps_per_chunk, 1u);
            const uint32_t chunks_total = static_cast<uint32_t>(
                std::max<uint64_t>(1ull, ceilDivU64_(collective_active_bytes_, static_cast<uint64_t>(chunk_bytes))));

            collective_ring_active_ = true;
            collective_ring_seq_ = seq32;
            collective_ring_chunks_total_ = chunks_total;
            collective_ring_steps_per_chunk_ = steps_per_chunk;
            collective_ring_chunk_index_ = 0;
            collective_ring_step_index_ = 0;
            collective_ring_step_remaining_bytes_ = collectiveRingStepPayloadBytes_(0, 0);
            collective_ring_max_inflight_chunks_ = std::max<uint32_t>(cfg_.collective_max_inflight_chunks, 1u);

            uint64_t total_payload = 0;
            for (uint32_t chunk_idx = 0; chunk_idx < chunks_total; ++chunk_idx) {
                for (uint32_t step = 0; step < steps_per_chunk; ++step) {
                    const uint64_t payload = static_cast<uint64_t>(collectiveRingStepPayloadBytes_(chunk_idx, step));
                    total_payload = saturatingAddU64_(total_payload, payload);
                }
            }
            collective_ring_total_payload_bytes_ = total_payload;
            collective_pending_total_bytes_ = total_payload;
            tensor_collective_chunk_groups_total_ += static_cast<uint64_t>(chunks_total);
        } else {
            const uint32_t total_nodes = (rt_.total_nodes > 0) ? static_cast<uint32_t>(rt_.total_nodes) : 1u;
            std::vector<uint32_t> dest_nodes;
            dest_nodes.reserve(2);

            const bool use_mesh_x = (cfg_.collective_pattern == "mesh_x");
            const bool use_mesh_xy = (cfg_.collective_pattern == "mesh_xy");
            if ((use_mesh_x || use_mesh_xy) && total_nodes > 1) {
                const uint32_t mesh_size = static_cast<uint32_t>(std::sqrt(static_cast<double>(total_nodes)));
                if (mesh_size > 0 && mesh_size * mesh_size == total_nodes) {
                    const uint32_t node_x = static_cast<uint32_t>(rt_.node_id) % mesh_size;
                    const uint32_t node_y = static_cast<uint32_t>(rt_.node_id) / mesh_size;
                    const uint32_t east = node_y * mesh_size + ((node_x + 1u) % mesh_size);
                    dest_nodes.push_back(east);
                    if (use_mesh_xy) {
                        const uint32_t south = ((node_y + 1u) % mesh_size) * mesh_size + node_x;
                        if (south != east) {
                            dest_nodes.push_back(south);
                        }
                    }
                }
            }
            if (dest_nodes.empty()) {
                dest_nodes.push_back((static_cast<uint32_t>(rt_.node_id) + 1u) % total_nodes);
            }

            // Ensure we can embed epoch seq in every packet (payload[0..7]).
            const uint64_t min_total_bytes = static_cast<uint64_t>(dest_nodes.size()) * 8ull;
            const uint64_t total_bytes = std::max<uint64_t>(cfg_.collective_bytes, min_total_bytes);
            collective_active_bytes_ = total_bytes;
            const uint64_t dest_count = static_cast<uint64_t>(dest_nodes.size());
            if (dest_count == 0 || total_bytes == 0) return 0;

            const uint64_t base_bytes = total_bytes / dest_count;
            const uint64_t remainder = total_bytes % dest_count;
            collective_pending_total_bytes_ = total_bytes;
            collective_pending_dest_nodes_ = dest_nodes;
            collective_pending_dest_remaining_bytes_.assign(dest_nodes.size(), 0);
            for (size_t dest_index = 0; dest_index < dest_nodes.size(); ++dest_index) {
                collective_pending_dest_remaining_bytes_[dest_index] = base_bytes + ((dest_index < remainder) ? 1u : 0u);
            }
        }

        collective_last_cycle_ = now_cycle;
            tensor_collective_cycles_total_ += 1;
            if (cfg_.collective_blocking) {
                collective_epoch_active_ = true;
                collective_epoch_seq_ = seq32;
                collective_epoch_expected_recv_bytes_ = collective_pending_total_bytes_;
                collective_epoch_recv_bytes_ = collectiveRecvBytesForSeq_(seq32);
                collective_epoch_start_cycle_ = now_cycle;
                collective_done_notified_ = false;
            }
            if (!cfg_.collective_blocking || cfg_.collective_scope == "per_core") {
                collective_seq_ += 1;
            }
    }

    if (!collectivePendingActive_()) return 0;

    uint64_t budget = noc_budget_bytes;
    const bool uncapped = (noc_budget_bytes == std::numeric_limits<uint64_t>::max());
    uint64_t bytes_sent = 0;
    const uint32_t packet_bytes = cfg_.collective_packet_bytes ? cfg_.collective_packet_bytes : 256u;
    const uint32_t seq32 = collective_pending_seq_;
    if (cfg_.collective_algo == "ring_chunked" || cfg_.collective_algo == "torus_2d_rs_ag") {
        uint32_t issued_packets = 0;
        while (collectivePendingActive_()) {
            if (!uncapped && budget < 8) {
                budget_blocked = true;
                backpressure_stalled = true;
                break;
            }
            if (collective_ring_max_inflight_chunks_ > 0 && issued_packets >= collective_ring_max_inflight_chunks_) {
                break;
            }
            if (collective_ring_reduce_wait_cycles_remaining_ > 0) {
                collective_ring_reduce_wait_cycles_remaining_ -= 1;
                tensor_collective_reduce_wait_cycles_total_ += 1;
                if (collective_2d_active_) {
                    tensor_collective_2d_reduce_wait_cycles_total_ += 1;
                }
                break;
            }
            if (!collective_ring_active_ || collective_ring_step_remaining_bytes_ == 0) {
                collective_pending_sent_bytes_ = collective_pending_total_bytes_;
                break;
            }
            if (!try_consume_credit()) {
                break;
            }

            uint64_t max_payload = std::min<uint64_t>(packet_bytes, collective_ring_step_remaining_bytes_);
            if (!uncapped) {
                max_payload = std::min<uint64_t>(max_payload, budget);
            }
            const uint32_t payload_bytes = static_cast<uint32_t>(max_payload);
            if (payload_bytes == 0) {
                budget_blocked = !uncapped;
                backpressure_stalled = backpressure_stalled || !uncapped;
                break;
            }

            const uint32_t step_in_chunk = collective_ring_step_index_;
            const Collective2dStage stage =
                collective_2d_active_ ? collective2dStageForStep_(step_in_chunk) : Collective2dStage::RowRS;
            const uint32_t dest_node =
                collective_2d_active_ ? collective2dNextDestNodeForStep_(step_in_chunk) : collectiveRingNextDestNode_();
            auto* pkt = new NocPacketEvent(rt_.node_id,
                                           dest_node,
                                           static_cast<uint16_t>(rt_.core_id),
                                           static_cast<uint16_t>(rt_.core_id),
                                           NocPacketKind::RawBytes,
                                           now_cycle);
            pkt->payload.resize(payload_bytes);
            fillBytesDeterministic_(cfg_.seed_base ^
                                        (static_cast<uint64_t>(rt_.node_id) << 32) ^
                                        (static_cast<uint64_t>(rt_.core_id) << 16),
                                    0,
                                    seq32 ^ collective_ring_chunk_index_ ^ (collective_ring_step_index_ << 16),
                                    pkt->payload);
            if (pkt->payload.size() >= 4) {
                pkt->payload[0] = static_cast<uint8_t>('C');
                pkt->payload[1] = static_cast<uint8_t>('O');
                pkt->payload[2] = static_cast<uint8_t>('L');
                pkt->payload[3] = static_cast<uint8_t>('L');
            }
            if (pkt->payload.size() >= 8) {
                write_u32_le_(pkt->payload, 4, seq32);
            }
            if (pkt->payload.size() >= 12) {
                write_u32_le_(pkt->payload, 8, collective_ring_chunk_index_);
            }
            if (pkt->payload.size() >= 16) {
                write_u32_le_(pkt->payload, 12, collective_ring_step_index_);
            }

            tensor_pkt_sent_total_ += 1;
            tensor_pkt_bytes_sent_total_ += static_cast<uint64_t>(pkt->payload.size());
            tensor_collective_pkts_sent_total_ += 1;
            tensor_collective_bytes_sent_total_ += static_cast<uint64_t>(pkt->payload.size());
            if (collective_2d_active_) {
                const uint64_t sent = static_cast<uint64_t>(pkt->payload.size());
                switch (stage) {
                    case Collective2dStage::RowRS:
                        tensor_collective_2d_row_rs_bytes_sent_total_ += sent;
                        break;
                    case Collective2dStage::ColRS:
                        tensor_collective_2d_col_rs_bytes_sent_total_ += sent;
                        break;
                    case Collective2dStage::ColAG:
                        tensor_collective_2d_col_ag_bytes_sent_total_ += sent;
                        break;
                    case Collective2dStage::RowAG:
                    default:
                        tensor_collective_2d_row_ag_bytes_sent_total_ += sent;
                        break;
                }
            }
            if (dest_node == static_cast<uint32_t>(rt_.node_id)) {
                rt_.noc->injectLocal(static_cast<int>(rt_.core_id), pkt);
            } else {
                rt_.noc->sendFromCore(static_cast<int>(rt_.core_id), pkt);
            }

            issued_packets += 1;
            on_credit_issue(seq32,
                            collective_ring_chunk_index_,
                            collective_ring_step_index_,
                            dest_node,
                            static_cast<uint16_t>(rt_.core_id));
            collective_ring_step_remaining_bytes_ -= payload_bytes;
            collective_ring_sent_payload_bytes_ += static_cast<uint64_t>(payload_bytes);
            collective_pending_sent_bytes_ += static_cast<uint64_t>(payload_bytes);
            bytes_sent += static_cast<uint64_t>(payload_bytes);
            if (!uncapped) {
                budget -= static_cast<uint64_t>(payload_bytes);
            }

            if (collective_ring_step_remaining_bytes_ == 0) {
                if (collective_2d_active_) {
                    const Collective2dStage completed = collective2dStageForStep_(collective_ring_step_index_);
                    switch (completed) {
                        case Collective2dStage::RowRS:
                            tensor_collective_2d_row_rs_steps_total_ += 1;
                            break;
                        case Collective2dStage::ColRS:
                            tensor_collective_2d_col_rs_steps_total_ += 1;
                            break;
                        case Collective2dStage::ColAG:
                            tensor_collective_2d_col_ag_steps_total_ += 1;
                            break;
                        case Collective2dStage::RowAG:
                        default:
                            tensor_collective_2d_row_ag_steps_total_ += 1;
                            break;
                    }
                }
                tensor_collective_ring_steps_total_ += 1;
                collective_ring_step_index_ += 1;
                if (collective_ring_step_index_ >= collective_ring_steps_per_chunk_) {
                    collective_ring_step_index_ = 0;
                    collective_ring_chunk_index_ += 1;
                }

                if (collective_ring_chunk_index_ >= collective_ring_chunks_total_) {
                    collective_ring_active_ = false;
                    collective_pending_sent_bytes_ = collective_pending_total_bytes_;
                    break;
                }

                collective_ring_step_remaining_bytes_ =
                    collectiveRingStepPayloadBytes_(collective_ring_chunk_index_, collective_ring_step_index_);
                if (cfg_.collective_type == "allreduce" && cfg_.collective_reduce_overhead_cycles > 0) {
                    if (collective_2d_active_) {
                        const uint32_t stage2_start = collective_2d_row_hop_ + collective_2d_col_hop_;
                        if (collective_ring_step_index_ == stage2_start) {
                            collective_ring_reduce_wait_cycles_remaining_ = cfg_.collective_reduce_overhead_cycles;
                        }
                    } else {
                        const uint32_t participants = collectiveRingParticipantCount_();
                        const uint32_t rs_steps = (participants > 1) ? (participants - 1u) : 1u;
                        if (collective_ring_step_index_ == rs_steps) {
                            collective_ring_reduce_wait_cycles_remaining_ = cfg_.collective_reduce_overhead_cycles;
                        }
                    }
                }
            }
        }
    } else {
        while (collectivePendingActive_()) {
            if (!uncapped && budget < 8) {
                budget_blocked = true;
                backpressure_stalled = true;
                break;
            }
            if (!try_consume_credit()) {
                break;
            }

            bool found_dest = false;
            size_t dest_index = collective_pending_next_dest_;
            for (size_t i = 0; i < collective_pending_dest_nodes_.size(); ++i) {
                const size_t idx = (dest_index + i) % collective_pending_dest_nodes_.size();
                if (collective_pending_dest_remaining_bytes_[idx] > 0) {
                    dest_index = idx;
                    found_dest = true;
                    break;
                }
            }
            if (!found_dest) break;

            uint64_t remaining = collective_pending_dest_remaining_bytes_[dest_index];
            uint64_t max_payload = std::min<uint64_t>(packet_bytes, remaining);
            if (!uncapped) {
                if (budget == 0) {
                    budget_blocked = true;
                    backpressure_stalled = true;
                    break;
                }
                max_payload = std::min<uint64_t>(max_payload, budget);
            }
            const uint32_t payload_bytes = static_cast<uint32_t>(max_payload);
            if (payload_bytes == 0) {
                budget_blocked = !uncapped;
                backpressure_stalled = backpressure_stalled || !uncapped;
                break;
            }

            const uint32_t dest_node = collective_pending_dest_nodes_[dest_index];
            auto* pkt = new NocPacketEvent(rt_.node_id,
                                           dest_node,
                                           static_cast<uint16_t>(rt_.core_id),
                                           static_cast<uint16_t>(rt_.core_id),
                                           NocPacketKind::RawBytes,
                                           now_cycle);
            pkt->payload.resize(payload_bytes);
            fillBytesDeterministic_(cfg_.seed_base ^
                                        (static_cast<uint64_t>(rt_.node_id) << 32) ^
                                        (static_cast<uint64_t>(rt_.core_id) << 16),
                                    0,
                                    seq32,
                                    pkt->payload);
            if (pkt->payload.size() >= 4) {
                pkt->payload[0] = static_cast<uint8_t>('C');
                pkt->payload[1] = static_cast<uint8_t>('O');
                pkt->payload[2] = static_cast<uint8_t>('L');
                pkt->payload[3] = static_cast<uint8_t>('L');
            }
            if (pkt->payload.size() >= 8) {
                write_u32_le_(pkt->payload, 4, seq32);
            }

            tensor_pkt_sent_total_ += 1;
            tensor_pkt_bytes_sent_total_ += static_cast<uint64_t>(pkt->payload.size());
            tensor_collective_pkts_sent_total_ += 1;
            tensor_collective_bytes_sent_total_ += static_cast<uint64_t>(pkt->payload.size());
            if (dest_node == static_cast<uint32_t>(rt_.node_id)) {
                rt_.noc->injectLocal(static_cast<int>(rt_.core_id), pkt);
            } else {
                rt_.noc->sendFromCore(static_cast<int>(rt_.core_id), pkt);
            }

            collective_pending_dest_remaining_bytes_[dest_index] -= static_cast<uint64_t>(payload_bytes);
            on_credit_issue(seq32, 0u, 0u, dest_node, static_cast<uint16_t>(rt_.core_id));
            collective_pending_sent_bytes_ += static_cast<uint64_t>(payload_bytes);
            collective_pending_next_dest_ = (dest_index + 1) % collective_pending_dest_nodes_.size();
            bytes_sent += static_cast<uint64_t>(payload_bytes);
            if (!uncapped) {
                budget -= static_cast<uint64_t>(payload_bytes);
                if (budget == 0) break;
            }
        }
    }

    if (!collectivePendingActive_()) {
        collective_pending_active_ = false;
        collective_active_bytes_ = 0;
        collective_pending_total_bytes_ = 0;
        collective_pending_sent_bytes_ = 0;
        collective_pending_dest_nodes_.clear();
        collective_pending_dest_remaining_bytes_.clear();
        collective_pending_next_dest_ = 0;
        collective_ring_active_ = false;
        collective_ring_total_payload_bytes_ = 0;
        collective_ring_sent_payload_bytes_ = 0;
        collective_ring_reduce_wait_cycles_remaining_ = 0;
        collective_2d_active_ = false;
        collective_2d_dim_x_ = 0;
        collective_2d_dim_y_ = 0;
        collective_2d_row_hop_ = 0;
        collective_2d_col_hop_ = 0;
        if (!event_credit_return) {
            collective_credit_inflight_chunks_ = 0;
            collective_credit_outstanding_.clear();
            collective_credit_return_seen_.clear();
            collective_credit_return_pending_credits_.clear();
            collective_credit_return_pending_queue_.clear();
        } else if (collective_credit_outstanding_.empty()) {
            collective_credit_return_seen_.clear();
        }
    }

    if (backpressure_stalled) {
        tensor_collective_backpressure_stall_cycles_total_ += 1;
    }

    return bytes_sent;
}

uint64_t TensorWorkload::emitCommTraffic_(uint64_t now_cycle, uint64_t noc_budget_bytes, bool& budget_blocked) {
    budget_blocked = false;
    if (!commReady_()) {
        if (cfg_.comm_enable && cfg_.strict && !rt_.noc && rt_.log) {
            rt_.log->fatal(CALL_INFO, -1, "tensor fatal: comm_enable=1 but INocTransport is null (core=%u)\n", rt_.core_id);
        }
        return 0;
    }
    if (now_cycle - comm_last_cycle_ < cfg_.comm_period_cycles) return 0;

    if (cfg_.noc_bandwidth_bytes_per_cycle > 0 &&
        static_cast<uint64_t>(cfg_.comm_payload_bytes) > noc_budget_bytes) {
        budget_blocked = true;
        return 0;
    }

    const uint32_t dst_node =
        (rt_.total_nodes > 1) ? ((rt_.node_id + 1u) % rt_.total_nodes) : rt_.node_id;
    auto* pkt = new NocPacketEvent(rt_.node_id,
                                   dst_node,
                                   static_cast<uint16_t>(rt_.core_id),
                                   static_cast<uint16_t>(rt_.core_id),
                                   NocPacketKind::RawBytes,
                                   now_cycle);
    pkt->payload.resize(cfg_.comm_payload_bytes);
    fillBytesDeterministic_(cfg_.seed_base ^
                                (static_cast<uint64_t>(rt_.node_id) << 32) ^
                                (static_cast<uint64_t>(rt_.core_id) << 16),
                            0,
                            iter_seq_,
                            pkt->payload);
    tensor_pkt_sent_total_ += 1;
    tensor_pkt_bytes_sent_total_ += static_cast<uint64_t>(pkt->payload.size());
    if (dst_node == rt_.node_id) {
        rt_.noc->injectLocal(static_cast<int>(rt_.core_id), pkt);
    } else {
        rt_.noc->sendFromCore(static_cast<int>(rt_.core_id), pkt);
    }
    comm_last_cycle_ = now_cycle;
    return static_cast<uint64_t>(pkt->payload.size());
}

void TensorWorkload::issueNocTraffic_(uint64_t now_cycle,
                                      uint64_t& noc_budget,
                                      bool& did,
                                      bool& did_collective,
                                      bool& did_comm,
                                      bool& noc_budget_blocked) {
    const uint64_t uncapped_budget = std::numeric_limits<uint64_t>::max();
    auto consume_budget = [&](uint64_t sent_bytes) {
        if (cfg_.noc_bandwidth_bytes_per_cycle > 0 && noc_budget != uncapped_budget) {
            noc_budget = (noc_budget >= sent_bytes) ? (noc_budget - sent_bytes) : 0;
        }
    };

    auto issue_collective = [&]() {
        bool blocked_collective = false;
        const uint64_t collective_sent = emitCollectiveTraffic_(now_cycle, noc_budget, blocked_collective);
        if (collective_sent > 0) {
            did = true;
            did_collective = true;
            tensor_collective_issue_cycles_total_ += 1;
            consume_budget(collective_sent);
        }
        if (collectivePendingActive_()) {
            tensor_collective_pending_cycles_total_ += 1;
        }
        noc_budget_blocked = noc_budget_blocked || blocked_collective;
    };

    auto issue_credit_return = [&]() {
        bool blocked_credit_return = false;
        const uint64_t return_sent = emitCollectiveCreditReturnTraffic_(now_cycle, noc_budget, blocked_credit_return);
        if (return_sent > 0) {
            did = true;
            consume_budget(return_sent);
        }
        noc_budget_blocked = noc_budget_blocked || blocked_credit_return;
    };

    auto issue_comm = [&]() {
        bool blocked_comm = false;
        const uint64_t comm_sent = emitCommTraffic_(now_cycle, noc_budget, blocked_comm);
        if (comm_sent > 0) {
            did = true;
            did_comm = true;
            consume_budget(comm_sent);
        }
        noc_budget_blocked = noc_budget_blocked || blocked_comm;
    };

    issue_credit_return();

    if (cfg_.collective_issue_priority == "payload_first") {
        issue_comm();
        issue_collective();
    } else {
        issue_collective();
        issue_comm();
    }
}

uint64_t TensorWorkload::collectiveRecvBytesForSeq_(uint32_t seq) const {
    auto it = collective_recv_bytes_by_seq_.find(seq);
    if (it == collective_recv_bytes_by_seq_.end()) return 0;
    return it->second;
}

void TensorWorkload::markCollectiveEpochDone_(uint64_t now_cycle) {
    if (!cfg_.collective_blocking) return;
    if (!collective_epoch_active_) return;
    if (collective_epoch_expected_recv_bytes_ == 0) return;

    const uint64_t start = collective_epoch_start_cycle_;
    const uint64_t lat = (now_cycle >= start) ? (now_cycle - start) : 0ull;
    tensor_collective_epoch_done_total_ += 1ull;
    tensor_collective_epoch_latency_cycles_total_ = saturatingAddU64_(tensor_collective_epoch_latency_cycles_total_, lat);
    if (lat > tensor_collective_epoch_latency_cycles_max_) {
        tensor_collective_epoch_latency_cycles_max_ = lat;
    }
    collective_epoch_start_cycle_ = 0;
}

void TensorWorkload::serviceCollectiveBarrier_(uint64_t now_cycle) {
    if (!cfg_.collective_blocking) return;

    // Per-core blocking barrier: completion is purely local (no RELEASE/control traffic).
    // We service it here so program-mode workloads can observe completion without relying
    // on the bulk/tile scheduler paths.
    if (cfg_.collective_scope == "per_core") {
        if (!collective_epoch_active_) return;
        if (collective_epoch_expected_recv_bytes_ == 0) return;
        if (collective_epoch_recv_bytes_ < collective_epoch_expected_recv_bytes_) return;
        markCollectiveEpochDone_(now_cycle);
        const uint32_t done_seq = collective_epoch_seq_;
        collective_epoch_active_ = false;
        collective_epoch_expected_recv_bytes_ = 0;
        collective_epoch_recv_bytes_ = 0;
        collective_recv_bytes_by_seq_.erase(done_seq);
        return;
    }

    if (!collectiveReady_()) return;

    maybeNotifyCollectiveDone_(now_cycle);
    maybeEmitCollectiveRelease_(now_cycle);
}

void TensorWorkload::maybeNotifyCollectiveDone_(uint64_t now_cycle) {
    if (!cfg_.collective_blocking) return;
    if (cfg_.collective_scope == "per_core") return;
    if (!collective_epoch_active_) return;
    if (collective_done_notified_) return;
    if (collective_epoch_expected_recv_bytes_ == 0) return;
    if (collective_epoch_recv_bytes_ < collective_epoch_expected_recv_bytes_) return;

    // Mark notified before sending to avoid accidental duplicates.
    collective_done_notified_ = true;

    std::vector<uint8_t> payload(16, 0);
    payload[0] = static_cast<uint8_t>('T');
    payload[1] = static_cast<uint8_t>('C');
    payload[2] = static_cast<uint8_t>('D');
    payload[3] = static_cast<uint8_t>('N');
    write_u32_le_(payload, 4, collective_epoch_seq_);
    write_u32_le_(payload, 8, static_cast<uint32_t>(rt_.node_id));
    write_u32_le_(payload, 12, static_cast<uint32_t>(rt_.core_id));

    const uint32_t dst_node = (cfg_.collective_scope == "per_pe") ? static_cast<uint32_t>(rt_.node_id) : 0u;
    const uint32_t dst_core = 0u;

    const bool self_is_dst = (dst_node == static_cast<uint32_t>(rt_.node_id) && dst_core == static_cast<uint32_t>(rt_.core_id));
    if (self_is_dst) {
        onCollectiveControlPacket_(payload, now_cycle);
        return;
    }

    if (!rt_.noc) return;
    auto* pkt = new NocPacketEvent(rt_.node_id,
                                   dst_node,
                                   static_cast<uint16_t>(rt_.core_id),
                                   static_cast<uint16_t>(dst_core),
                                   NocPacketKind::Control,
                                   now_cycle);
    pkt->payload = payload;
    tensor_pkt_sent_total_ += 1;
    tensor_pkt_bytes_sent_total_ += static_cast<uint64_t>(pkt->payload.size());
    if (dst_node == static_cast<uint32_t>(rt_.node_id)) {
        rt_.noc->injectLocal(static_cast<int>(dst_core), pkt);
    } else {
        rt_.noc->sendFromCore(static_cast<int>(rt_.core_id), pkt);
    }
}

void TensorWorkload::maybeEmitCollectiveRelease_(uint64_t now_cycle) {
    if (!collective_release_pending_) return;

    const bool is_pe_leader = (cfg_.collective_scope == "per_pe" && rt_.core_id == 0u);
    const bool is_sys_root = (cfg_.collective_scope == "per_system" && rt_.node_id == 0u && rt_.core_id == 0u);
    if (!is_pe_leader && !is_sys_root) return;
    if (!rt_.noc) return;

    const uint32_t seq = collective_release_pending_seq_;
    const uint32_t next = seq + 1u;

    std::vector<uint8_t> payload(12, 0);
    payload[0] = static_cast<uint8_t>('T');
    payload[1] = static_cast<uint8_t>('C');
    payload[2] = static_cast<uint8_t>('R');
    payload[3] = static_cast<uint8_t>('L');
    write_u32_le_(payload, 4, seq);
    write_u32_le_(payload, 8, next);

    if (is_pe_leader) {
        const uint32_t node = static_cast<uint32_t>(rt_.node_id);
        const uint32_t cores = total_cores_cfg_ ? total_cores_cfg_ : 1u;
        for (uint32_t core = 0; core < cores; ++core) {
            if (core == static_cast<uint32_t>(rt_.core_id)) continue;
            auto* pkt = new NocPacketEvent(rt_.node_id,
                                           node,
                                           static_cast<uint16_t>(rt_.core_id),
                                           static_cast<uint16_t>(core),
                                           NocPacketKind::Control,
                                           now_cycle);
            pkt->payload = payload;
            tensor_pkt_sent_total_ += 1;
            tensor_pkt_bytes_sent_total_ += static_cast<uint64_t>(pkt->payload.size());
            rt_.noc->injectLocal(static_cast<int>(core), pkt);
        }
        onCollectiveRelease_(seq, next, now_cycle);
    } else {
        const uint32_t total_nodes = (rt_.total_nodes > 0) ? static_cast<uint32_t>(rt_.total_nodes) : 1u;
        const uint32_t cores = total_cores_cfg_ ? total_cores_cfg_ : 1u;
        for (uint32_t node = 0; node < total_nodes; ++node) {
            for (uint32_t core = 0; core < cores; ++core) {
                if (node == static_cast<uint32_t>(rt_.node_id) && core == static_cast<uint32_t>(rt_.core_id)) continue;
                auto* pkt = new NocPacketEvent(rt_.node_id,
                                               node,
                                               static_cast<uint16_t>(rt_.core_id),
                                               static_cast<uint16_t>(core),
                                               NocPacketKind::Control,
                                               now_cycle);
                pkt->payload = payload;
                tensor_pkt_sent_total_ += 1;
                tensor_pkt_bytes_sent_total_ += static_cast<uint64_t>(pkt->payload.size());
                if (node == static_cast<uint32_t>(rt_.node_id)) {
                    rt_.noc->injectLocal(static_cast<int>(core), pkt);
                } else {
                    rt_.noc->sendFromCore(static_cast<int>(rt_.core_id), pkt);
                }
            }
        }
        onCollectiveRelease_(seq, next, now_cycle);
    }

    collective_release_pending_ = false;
    collective_release_pending_seq_ = 0;
    collective_barrier_done_count_ = 0;
    collective_barrier_done_bitmap_.clear();
}

void TensorWorkload::onCollectiveControlPacket_(const std::vector<uint8_t>& payload, uint64_t now_cycle) {
    (void)now_cycle;
    if (!cfg_.collective_blocking) return;
    if (cfg_.collective_scope == "per_core") return;

    if (has_magic4_(payload, 'T', 'C', 'D', 'N')) {
        if (payload.size() < 16) return;
        const uint32_t seq = read_u32_le_(payload, 4);
        const uint32_t sender_node = read_u32_le_(payload, 8);
        const uint32_t sender_core = read_u32_le_(payload, 12);
        if (sender_core >= (total_cores_cfg_ ? total_cores_cfg_ : 1u)) return;

        const uint32_t cur_seq = static_cast<uint32_t>(collective_seq_ & 0xffffffffu);
        if (seq != cur_seq) return;

        size_t expected = 0;
        size_t idx = 0;
        if (cfg_.collective_scope == "per_pe") {
            if (rt_.core_id != 0u) return;
            if (sender_node != static_cast<uint32_t>(rt_.node_id)) return;
            expected = static_cast<size_t>(total_cores_cfg_ ? total_cores_cfg_ : 1u);
            idx = static_cast<size_t>(sender_core);
        } else if (cfg_.collective_scope == "per_system") {
            if (!(rt_.node_id == 0u && rt_.core_id == 0u)) return;
            const uint32_t total_nodes = (rt_.total_nodes > 0) ? static_cast<uint32_t>(rt_.total_nodes) : 1u;
            if (sender_node >= total_nodes) return;
            expected = static_cast<size_t>(total_nodes) * static_cast<size_t>(total_cores_cfg_ ? total_cores_cfg_ : 1u);
            idx = static_cast<size_t>(sender_node) * static_cast<size_t>(total_cores_cfg_ ? total_cores_cfg_ : 1u) +
                  static_cast<size_t>(sender_core);
        } else {
            return;
        }

        if (expected == 0) return;
        if (collective_barrier_done_bitmap_.size() != expected) {
            collective_barrier_done_bitmap_.assign(expected, 0);
            collective_barrier_done_count_ = 0;
        }
        if (idx >= collective_barrier_done_bitmap_.size()) return;

        if (collective_barrier_done_bitmap_[idx] == 0) {
            collective_barrier_done_bitmap_[idx] = 1;
            collective_barrier_done_count_ += 1;
        }

        if (collective_barrier_done_count_ >= expected && !collective_release_pending_) {
            collective_release_pending_ = true;
            collective_release_pending_seq_ = seq;
        }
        return;
    }

    if (has_magic4_(payload, 'T', 'C', 'R', 'L')) {
        if (payload.size() < 12) return;
        const uint32_t seq = read_u32_le_(payload, 4);
        const uint32_t next = read_u32_le_(payload, 8);
        onCollectiveRelease_(seq, next, now_cycle);
        return;
    }
}

void TensorWorkload::onCollectiveRelease_(uint32_t seq, uint32_t next_seq, uint64_t now_cycle) {
    if (!cfg_.collective_blocking) return;
    if (cfg_.collective_scope == "per_core") return;

    const uint32_t cur_seq = static_cast<uint32_t>(collective_seq_ & 0xffffffffu);
    if (seq != cur_seq) return;

    markCollectiveEpochDone_(now_cycle);
    collective_seq_ = static_cast<uint64_t>(next_seq);
    collective_epoch_active_ = false;
    collective_epoch_expected_recv_bytes_ = 0;
    collective_epoch_recv_bytes_ = 0;
    collective_done_notified_ = false;
    collective_recv_bytes_by_seq_.erase(seq);
}

bool TensorWorkload::deliverPacket(NocPacketEvent* packet) {
    if (!packet) return true;
    const NocPacketKind kind = packet->packetKind();
    if (kind == NocPacketKind::RawBytes || kind == NocPacketKind::Control) {
        tensor_pkt_recv_total_ += 1;
        tensor_pkt_bytes_recv_total_ += static_cast<uint64_t>(packet->payload.size());
    }
    if (kind == NocPacketKind::RawBytes) {
        if (has_collective_magic_(packet->payload)) {
            tensor_collective_pkts_recv_total_ += 1;
            tensor_collective_bytes_recv_total_ += static_cast<uint64_t>(packet->payload.size());
            const uint32_t seq = read_u32_le_(packet->payload, 4);
            if (cfg_.collective_blocking) {
                collective_recv_bytes_by_seq_[seq] += static_cast<uint64_t>(packet->payload.size());
                if (collective_epoch_active_ && seq == collective_epoch_seq_) {
                    collective_epoch_recv_bytes_ = collectiveRecvBytesForSeq_(collective_epoch_seq_);
                }
            }
            maybeEmitCollectiveCreditReturn_(packet, packet->timestamp);
        }
    } else if (kind == NocPacketKind::Control) {
        if (has_magic4_(packet->payload, 'T', 'C', 'C', 'R')) {
            if (packet->payload.size() >= 20) {
                const uint32_t seq = read_u32_le_(packet->payload, 4);
                const uint32_t chunk = read_u32_le_(packet->payload, 8);
                const uint32_t step = read_u32_le_(packet->payload, 12);
                const uint32_t credits = read_u32_le_(packet->payload, 16);
                tensor_collective_credit_return_pkts_recv_total_ += 1;
                onCollectiveCreditReturn_(seq,
                                          chunk,
                                          step,
                                          credits,
                                          packet->src_node,
                                          packet->src_endpoint,
                                          packet->timestamp);
            }
        } else {
            onCollectiveControlPacket_(packet->payload, packet->timestamp);
        }
    }
    delete packet;
    return true;
}


}} // namespace SST::SnnDL

