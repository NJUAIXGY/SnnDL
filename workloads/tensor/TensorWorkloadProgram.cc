// -*- c++ -*-

#include <sst/core/sst_config.h>

#include "workloads/tensor/TensorWorkload.h"

#include <sst/core/output.h>

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace SST { namespace SnnDL {

namespace {

std::string toLowerCopy(const std::string& value) {
    std::string result;
    result.reserve(value.size());
    for (char ch : value) {
        result.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    return result;
}

} // namespace

bool TensorWorkload::parseProgramDsl_(const std::string& dsl, std::vector<ProgramOp>& out_ops) const {
    out_ops.clear();

    auto trim_copy = [](const std::string& s) -> std::string {
        size_t b = 0;
        while (b < s.size() && std::isspace(static_cast<unsigned char>(s[b]))) b++;
        size_t e = s.size();
        while (e > b && std::isspace(static_cast<unsigned char>(s[e - 1]))) e--;
        return s.substr(b, e - b);
    };
    auto parse_u64 = [](const std::string& s, uint64_t& out) -> bool {
        try {
            size_t idx = 0;
            const unsigned long long v = std::stoull(s, &idx, 0);
            if (idx != s.size()) return false;
            out = static_cast<uint64_t>(v);
            return true;
        } catch (...) {
            return false;
        }
    };
    auto parse_bool01 = [&](const std::string& s, bool& out) -> bool {
        const std::string v = toLowerCopy(trim_copy(s));
        if (v == "1" || v == "true" || v == "yes" || v == "y") {
            out = true;
            return true;
        }
        if (v == "0" || v == "false" || v == "no" || v == "n") {
            out = false;
            return true;
        }
        return false;
    };

    size_t pos = 0;
    while (pos < dsl.size()) {
        const size_t next = dsl.find(';', pos);
        const size_t len = (next == std::string::npos) ? (dsl.size() - pos) : (next - pos);
        std::string token = trim_copy(dsl.substr(pos, len));
        pos = (next == std::string::npos) ? dsl.size() : (next + 1);
        if (token.empty()) continue;

        std::string op_str = token;
        std::string args_str;
        const size_t colon = token.find(':');
        if (colon != std::string::npos) {
            op_str = token.substr(0, colon);
            args_str = token.substr(colon + 1);
        }
        const std::string op = toLowerCopy(trim_copy(op_str));
        if (op.empty()) continue;

        ProgramOp out{};
        if (op == "gemm") {
            out.kind = ProgramOpKind::Gemm;
        } else if (op == "allreduce") {
            out.kind = ProgramOpKind::Allreduce;
            out.blocking = true;
        } else if (op == "softmax") {
            out.kind = ProgramOpKind::Softmax;
        } else if (op == "dma_read") {
            out.kind = ProgramOpKind::DmaRead;
        } else if (op == "dma_write") {
            out.kind = ProgramOpKind::DmaWrite;
        } else if (op == "fence") {
            out.kind = ProgramOpKind::Fence;
        } else if (op == "gemm_ub") {
            out.kind = ProgramOpKind::GemmUb;
        } else {
            return false;
        }

        args_str = trim_copy(args_str);
        if (!args_str.empty()) {
            size_t ap = 0;
            while (ap < args_str.size()) {
                const size_t anext = args_str.find(',', ap);
                const size_t alen = (anext == std::string::npos) ? (args_str.size() - ap) : (anext - ap);
                std::string arg = trim_copy(args_str.substr(ap, alen));
                ap = (anext == std::string::npos) ? args_str.size() : (anext + 1);
                if (arg.empty()) continue;

                std::string k = arg;
                std::string v;
                const size_t eq = arg.find('=');
                if (eq != std::string::npos) {
                    k = arg.substr(0, eq);
                    v = arg.substr(eq + 1);
                } else {
                    v = "";
                }
                k = toLowerCopy(trim_copy(k));
                v = trim_copy(v);

                if (out.kind == ProgramOpKind::Allreduce) {
                    if (k == "bytes") {
                        uint64_t bv = 0;
                        if (!parse_u64(v, bv)) return false;
                        out.bytes = bv;
                    } else if (k == "blocking") {
                        bool b = true;
                        if (v.empty()) {
                            b = true;
                        } else if (!parse_bool01(v, b)) {
                            return false;
                        }
                        out.blocking = b;
                    } else {
                        return false;
                    }
                } else if (out.kind == ProgramOpKind::Softmax) {
                    if (k == "elems") {
                        uint64_t ev = 0;
                        if (!parse_u64(v, ev) || ev == 0) return false;
                        out.elems = ev;
                    } else {
                        return false;
                    }
                } else if (out.kind == ProgramOpKind::DmaRead || out.kind == ProgramOpKind::DmaWrite) {
                    if (k == "bytes") {
                        uint64_t bv = 0;
                        if (!parse_u64(v, bv) || bv == 0) return false;
                        out.bytes = bv;
                    } else if (k == "buf") {
                        uint64_t bv = 0;
                        if (!parse_u64(v, bv)) return false;
                        if (bv > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) return false;
                        const uint32_t buf = static_cast<uint32_t>(bv);
                        if (buf >= std::max<uint32_t>(cfg_.program_ub_buffers, 1u)) return false;
                        out.buf = buf;
                    } else if (k == "ub_addr") {
                        uint64_t av = 0;
                        if (!parse_u64(v, av)) return false;
                        out.ub_addr_present = true;
                        out.ub_addr = av;
                    } else if (out.kind == ProgramOpKind::DmaRead && k == "reset") {
                        bool b = true;
                        if (v.empty()) {
                            b = true;
                        } else if (!parse_bool01(v, b)) {
                            return false;
                        }
                        out.reset = b;
                    } else if (out.kind == ProgramOpKind::DmaWrite && k == "consume") {
                        bool b = true;
                        if (v.empty()) {
                            b = true;
                        } else if (!parse_bool01(v, b)) {
                            return false;
                        }
                        out.consume = b;
                    } else {
                        return false;
                    }
                } else if (out.kind == ProgramOpKind::GemmUb) {
                    if (k == "cycles") {
                        uint64_t cv = 0;
                        if (!parse_u64(v, cv)) return false;
                        out.cycles = cv;
                    } else if (k == "ub_read" || k == "ub_read_bytes") {
                        uint64_t bv = 0;
                        if (!parse_u64(v, bv)) return false;
                        out.ub_read_bytes = bv;
                    } else if (k == "ub_write" || k == "ub_write_bytes") {
                        uint64_t bv = 0;
                        if (!parse_u64(v, bv)) return false;
                        out.ub_write_bytes = bv;
                    } else if (k == "buf") {
                        uint64_t bv = 0;
                        if (!parse_u64(v, bv)) return false;
                        if (bv > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) return false;
                        const uint32_t buf = static_cast<uint32_t>(bv);
                        if (buf >= std::max<uint32_t>(cfg_.program_ub_buffers, 1u)) return false;
                        out.buf = buf;
                    } else if (k == "ub_read_addr") {
                        uint64_t av = 0;
                        if (!parse_u64(v, av)) return false;
                        out.ub_read_addr_present = true;
                        out.ub_read_addr = av;
                    } else if (k == "ub_write_addr") {
                        uint64_t av = 0;
                        if (!parse_u64(v, av)) return false;
                        out.ub_write_addr_present = true;
                        out.ub_write_addr = av;
                    } else if (k == "m" || k == "tm") {
                        uint64_t mv = 0;
                        if (!parse_u64(v, mv)) return false;
                        if (mv > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) return false;
                        out.m = static_cast<uint32_t>(mv);
                    } else if (k == "n" || k == "tn") {
                        uint64_t nv = 0;
                        if (!parse_u64(v, nv)) return false;
                        if (nv > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) return false;
                        out.n = static_cast<uint32_t>(nv);
                    } else if (k == "k" || k == "tk") {
                        uint64_t kvv = 0;
                        if (!parse_u64(v, kvv)) return false;
                        if (kvv > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max())) return false;
                        out.k = static_cast<uint32_t>(kvv);
                    } else {
                        return false;
                    }
                } else {
                    // GEMM/FENCE: no args.
                    return false;
                }
            }
        } else {
            if (out.kind == ProgramOpKind::DmaRead || out.kind == ProgramOpKind::DmaWrite) {
                return false;
            }
        }

        if ((out.kind == ProgramOpKind::DmaRead || out.kind == ProgramOpKind::DmaWrite) && out.bytes == 0) {
            return false;
        }
        if (out.kind == ProgramOpKind::Softmax && out.elems == 0) {
            return false;
        }
        if (out.kind == ProgramOpKind::GemmUb && out.cycles == 0) {
            if (out.m == 0 || out.n == 0 || out.k == 0) {
                return false;
            }
        }

        out_ops.push_back(out);
    }

    return !out_ops.empty();
}

uint64_t TensorWorkload::estimateGemmUbCycles_(uint32_t m, uint32_t n, uint32_t k) const {
    if (m == 0u || n == 0u || k == 0u) return 1ull;

    const uint64_t mm = static_cast<uint64_t>(m);
    const uint64_t nn = static_cast<uint64_t>(n);
    const uint64_t kk = static_cast<uint64_t>(k);
    const uint64_t macs = saturatingMulU64_(saturatingMulU64_(mm, nn), kk);
    const uint64_t thr = effectivePeakMacsPerCycle_();
    const uint64_t math = ceilDivU64_(std::max<uint64_t>(1ull, macs), std::max<uint64_t>(1ull, thr));
    const uint64_t pipe = static_cast<uint64_t>(compute_pipeline_latency_cycles_effective_);
    const uint64_t base_total = saturatingAddU64_(math ? math : 1ull, pipe);

    uint64_t extra = 0;
    if (cfg_.mxu_wavefront_enable && cfg_.mxu_wavefront_alpha > 0.0f) {
        const uint64_t am = clampNonZero_(static_cast<uint64_t>(cfg_.array_m), 1ull);
        const uint64_t an = clampNonZero_(static_cast<uint64_t>(cfg_.array_n), 1ull);
        const uint64_t mb = ceilDivU64_(mm, am);
        const uint64_t nb = ceilDivU64_(nn, an);
        const uint64_t m_blk = std::min<uint64_t>(mm, am);
        const uint64_t n_blk = std::min<uint64_t>(nn, an);
        const uint64_t span_blk = (m_blk > 0 && n_blk > 0 && kk > 0) ? (m_blk + n_blk + kk - 2ull) : 0ull;
        const uint64_t wf_span = saturatingMulU64_(saturatingMulU64_(mb, nb), span_blk);
        const double wf_body_d = static_cast<double>(cfg_.mxu_wavefront_alpha) * static_cast<double>(wf_span);
        const uint64_t wf_body = static_cast<uint64_t>(std::ceil(std::max(0.0, wf_body_d)));
        const uint64_t wf_total = saturatingAddU64_((wf_body ? wf_body : 1ull), pipe);
        if (wf_total > base_total) {
            extra = wf_total - base_total;
        }
    }

    const uint64_t total = saturatingAddU64_(base_total, extra);
    return total ? total : 1ull;
}

bool TensorWorkload::startProgramCollective_(uint64_t now_cycle, uint64_t bytes, bool blocking) {
    (void)blocking;
    if (collectivePendingActive_()) return true;
    if (cfg_.collective_type == "none") return false;
    if (!rt_.noc) {
        if (cfg_.strict && rt_.log) {
            rt_.log->fatal(CALL_INFO, -1, "tensor fatal: collective enabled but INocTransport is null (core=%u)\n", rt_.core_id);
        }
        return false;
    }

    const uint64_t collective_bytes = (bytes > 0) ? bytes : cfg_.collective_bytes;
    if (collective_bytes == 0) return false;

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
        collective_active_bytes_ = collective_bytes;
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

        const uint64_t min_total_bytes = static_cast<uint64_t>(dest_nodes.size()) * 8ull;
        const uint64_t total_bytes = std::max<uint64_t>(collective_bytes, min_total_bytes);
        collective_active_bytes_ = total_bytes;
        const uint64_t dest_count = static_cast<uint64_t>(dest_nodes.size());
        if (dest_count == 0 || total_bytes == 0) return false;

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

    return true;
}


bool TensorWorkload::onClockTickProgramM7_(uint64_t now_cycle) {
    bool did = false;
    bool did_mem = false;
    bool did_compute = false;
    bool did_collective = false;
    bool did_comm = false;
    bool noc_budget_blocked = false;
    uint64_t noc_budget = (cfg_.noc_bandwidth_bytes_per_cycle > 0)
                              ? cfg_.noc_bandwidth_bytes_per_cycle
                              : std::numeric_limits<uint64_t>::max();
    resetOnchipCycleState_(now_cycle);

    auto issue_noc = [&]() {
        issueNocTraffic_(now_cycle, noc_budget, did, did_collective, did_comm, noc_budget_blocked);
        if (noc_budget_blocked && !did_collective && !did_comm) {
            tensor_stall_noc_budget_cycles_total_ += 1;
        }
    };

    auto program_ub_total_occupancy_bytes = [&]() -> uint64_t {
        if (!program_addr_aware_enable_) {
            uint64_t sum = 0;
            for (const uint64_t v : program_ub_reserved_bytes_by_buf_) {
                sum = saturatingAddU64_(sum, v);
            }
            for (const uint64_t v : program_ub_valid_bytes_by_buf_) {
                sum = saturatingAddU64_(sum, v);
            }
            return sum;
        }
        uint64_t sum = 0;
        for (const auto& mp : program_ub_regions_by_buf_) {
            for (const auto& kv : mp) {
                sum = saturatingAddU64_(sum, kv.second.reserved_bytes);
                sum = saturatingAddU64_(sum, kv.second.valid_bytes);
            }
        }
        return sum;
    };
    auto update_program_ub_occupancy_max = [&]() {
        const uint64_t occ = program_ub_total_occupancy_bytes();
        if (occ > tensor_program_ub_occupancy_bytes_max_) {
            tensor_program_ub_occupancy_bytes_max_ = occ;
        }
    };

    // Step-gated mode: only run when a step is open, but still flush credit-return traffic.
    if (step_gated_ && !step_open_) {
        issue_noc();
        if (did) active_cycles_++;
        return did;
    }

    if (!started_) {
        if (now_cycle < cfg_.start_cycle) {
            issue_noc();
            if (did) active_cycles_++;
            return did;
        }
        started_ = true;
    }

    serviceCollectiveBarrier_(now_cycle);

    // Stop after requested program iterations (0 = run forever).
    if (cfg_.iterations > 0 && program_iter_done_ >= cfg_.iterations) {
        issue_noc();
        if (did) active_cycles_++;
        return did;
    }

    if (program_ops_.empty()) {
        // Program mode with an empty program is allowed when strict=0 (acts like idle workload).
        issue_noc();
        if (did) active_cycles_++;
        return did;
    }

    const bool collective_blocks_compute =
        (!cfg_.collective_overlap_with_compute && (program_coll_slot_.active || collectivePendingActive_()));

    // === Execute: DMA ===
    if (program_dma_read_slot_.active || program_dma_write_slot_.active) {
        tensor_program_dma_busy_cycles_total_ += 1;
        tensor_dma_cycles_total_ += 1;

        const uint64_t kUncapped = std::numeric_limits<uint64_t>::max();
        uint64_t budget = dmaBudgetBytesPerCycle_(now_cycle);
        const bool capped = (budget != kUncapped);
        const bool budget_initially_zero = capped && (budget == 0);
        const auto max_out = static_cast<size_t>(cfg_.mem_max_outstanding);

        bool program_mem_stall_marked = false;
        bool budget_stall_marked = false;
        bool outstanding_stall_marked = false;
        bool onchip_port_stall_marked = false;
        bool onchip_bank_stall_marked = false;
        bool hbm_channel_stall_marked = false;
        auto mark_program_mem_stall = [&]() {
            if (!program_mem_stall_marked) {
                tensor_program_mem_stall_cycles_total_ += 1;
                program_mem_stall_marked = true;
            }
        };
        auto mark_budget_stall = [&]() {
            if (!budget_stall_marked) {
                tensor_stall_dma_budget_cycles_total_ += 1;
                budget_stall_marked = true;
            }
            mark_program_mem_stall();
        };
        auto mark_outstanding_stall = [&]() {
            if (!outstanding_stall_marked) {
                tensor_stall_mem_outstanding_cycles_total_ += 1;
                outstanding_stall_marked = true;
            }
            mark_program_mem_stall();
        };
        auto mark_onchip_port_stall = [&]() {
            if (!onchip_port_stall_marked) {
                tensor_stall_onchip_port_cycles_total_ += 1;
                onchip_port_stall_marked = true;
            }
            mark_program_mem_stall();
        };
        auto mark_onchip_bank_stall = [&]() {
            if (!onchip_bank_stall_marked) {
                tensor_stall_onchip_bank_conflict_cycles_total_ += 1;
                onchip_bank_stall_marked = true;
            }
            mark_program_mem_stall();
        };
        auto mark_hbm_channel_stall = [&]() {
            if (!hbm_channel_stall_marked) {
                tensor_stall_dma_hbm_channel_budget_cycles_total_ += 1;
                hbm_channel_stall_marked = true;
            }
            mark_program_mem_stall();
        };

        auto try_mark_ub_bank_access = [&](uint64_t tag_seed) -> bool {
            if (!cfg_.onchip_model_enable) return true;
            if (!cfg_.onchip_bank_model_enable) return true;
            const uint32_t bank = selectUbBank_(tag_seed);
            if (bank >= onchip_ub_bank_queue_occupancy_.size()) return true;

            if (cfg_.ub_bank_conflict_mode == "queue") {
                if (onchip_ub_bank_queue_occupancy_[bank] >= cfg_.bank_queue_depth) return false;
                onchip_ub_bank_queue_occupancy_[bank] += 1;
                updateBankQueueOccupancyMax_();
                return true;
            }

            // hard: only 1 access per bank per cycle
            if (onchip_ub_bank_queue_occupancy_[bank] > 0) return false;
            onchip_ub_bank_queue_occupancy_[bank] = 1;
            updateBankQueueOccupancyMax_();
            return true;
        };

        auto service_dma_slot = [&](ProgramDmaSlot& slot, MemTag tag) {
            if (!slot.active) return;

            // M29: optional burst/setup model. When enabled, a slot issues at most one burst at a time,
            // and pays a fixed setup cost before each burst can start issuing bytes.
            const bool burst_enable = (cfg_.dma_burst_bytes > 0);
            if (burst_enable) {
                if (slot.setup_cycles_rem > 0) {
                    slot.setup_cycles_rem -= 1;
                    return;
                }
                if (slot.burst_bytes_rem == 0 && slot.issued_bytes < slot.total_bytes) {
                    slot.burst_bytes_rem =
                        std::min<uint64_t>(cfg_.dma_burst_bytes, slot.total_bytes - slot.issued_bytes);
                    if (cfg_.dma_setup_cycles > 0) {
                        slot.setup_cycles_rem = cfg_.dma_setup_cycles;
                        // Count this cycle as setup.
                        slot.setup_cycles_rem -= 1;
                        return;
                    }
                }
            }

            // M29: optional per-cycle issue-lane and per-slot inflight limits.
            const uint32_t engines_cfg = slot.is_read ? cfg_.dma_read_engines : cfg_.dma_write_engines;
            const uint32_t engines = (engines_cfg > 0) ? engines_cfg : std::numeric_limits<uint32_t>::max();
            const uint64_t slot_inflight_cap =
                (engines_cfg > 0 && cfg_.dma_max_inflight_per_engine > 0)
                    ? (static_cast<uint64_t>(engines_cfg) * static_cast<uint64_t>(cfg_.dma_max_inflight_per_engine))
                    : 0ull;

            bool slot_issued_this_cycle = false;
            uint32_t reqs_issued_this_cycle = 0;
            while (slot.issued_bytes < slot.total_bytes && reqs_issued_this_cycle < engines) {
                const uint64_t rem = slot.total_bytes - slot.issued_bytes;
                uint32_t want = static_cast<uint32_t>(std::min<uint64_t>(cfg_.mem_req_bytes, rem));
                if (burst_enable && slot.burst_bytes_rem > 0) {
                    want = static_cast<uint32_t>(std::min<uint64_t>(want, slot.burst_bytes_rem));
                }
                if (want == 0) break;
                if (capped) {
                    if (budget == 0) {
                        if (budget_initially_zero) {
                            mark_budget_stall();
                        }
                        break;
                    }
                    want = static_cast<uint32_t>(std::min<uint64_t>(want, budget));
                    if (want == 0) {
                        if (budget_initially_zero) {
                            mark_budget_stall();
                        }
                        break;
                    }
                }
                if (inflight_.size() >= max_out) {
                    mark_outstanding_stall();
                    break;
                }
                if (slot_inflight_cap > 0 && static_cast<uint64_t>(slot.inflight_reqs) >= slot_inflight_cap) {
                    mark_outstanding_stall();
                    break;
                }

                uint32_t hbm_ch = 0;
                want = clampBytesByHbmChannelBudget_(
                    now_cycle,
                    slot.is_read ? ReqKind::Read : ReqKind::Write,
                    want,
                    hbm_ch);
                if (want == 0) {
                    // Mirror "budget_initially_zero" semantics: only count stall when this slot
                    // cannot issue any bytes in this cycle due to channel budget contention.
                    if (!slot_issued_this_cycle) {
                        mark_hbm_channel_stall();
                    }
                    break;
                }

                // M19: program-mode on-chip UB port + bank conflict modeling.
                if (cfg_.onchip_model_enable) {
                    if (slot.is_read) {
                        // DRAM -> UB: consumes UB write ports.
                        if (!acquireOnchipWritePorts_(1, 0)) {
                            mark_onchip_port_stall();
                            break;
                        }
                    } else {
                        // UB -> DRAM: consumes UB read ports.
                        if (!acquireOnchipReadPorts_(1, 0)) {
                            mark_onchip_port_stall();
                            break;
                        }
                    }
                    const uint64_t addr_seed = (slot.ub_addr_present ? slot.ub_addr : 0ull) + slot.issued_bytes;
                    const uint64_t tag_seed =
                        (static_cast<uint64_t>(slot.buf) << 48) ^ addr_seed ^ (static_cast<uint64_t>(slot.epoch) << 1);
                    if (!try_mark_ub_bank_access(tag_seed)) {
                        mark_onchip_bank_stall();
                        break;
                    }
                }

                uint32_t issued = 0;
                if (slot.is_read) {
                    issued = issueMemReadTagged_(want, tag, slot.epoch);
                } else {
                    issued = issueMemWriteTagged_(want, tag, slot.epoch);
                }
                if (issued == 0) {
                    mark_outstanding_stall();
                    break;
                }

                slot.issued_bytes += static_cast<uint64_t>(issued);
                slot.inflight_reqs += 1;
                reqs_issued_this_cycle += 1;
                if (burst_enable && slot.burst_bytes_rem > 0) {
                    slot.burst_bytes_rem = (slot.burst_bytes_rem >= issued) ? (slot.burst_bytes_rem - issued) : 0;
                }
                consumeHbmChannelBudget_(now_cycle, hbm_ch, issued);
                did = true;
                did_mem = true;
                slot_issued_this_cycle = true;
                if (capped) {
                    budget = (budget >= issued) ? (budget - issued) : 0;
                    if (budget == 0) break;
                }

                // M29: one burst at a time; next burst (with setup) begins next cycle.
                if (burst_enable && slot.burst_bytes_rem == 0 && slot.issued_bytes < slot.total_bytes) {
                    break;
                }
            }

            if (slot.done_bytes >= slot.total_bytes) {
                if (slot.is_read) {
                    const uint64_t total = slot.total_bytes;
                    const uint32_t buf = slot.buf;
                    if (!program_addr_aware_enable_) {
                        if (buf < program_ub_reserved_bytes_by_buf_.size() && buf < program_ub_valid_bytes_by_buf_.size()) {
                            auto& rsv = program_ub_reserved_bytes_by_buf_[buf];
                            auto& val = program_ub_valid_bytes_by_buf_[buf];
                            rsv = (rsv >= total) ? (rsv - total) : 0;
                            val = saturatingAddU64_(val, total);
                            update_program_ub_occupancy_max();
                        }
                    } else if (buf < program_ub_regions_by_buf_.size() && slot.ub_addr_present) {
                        auto& mp = program_ub_regions_by_buf_[buf];
                        auto it = mp.find(slot.ub_addr);
                        if (it != mp.end()) {
                            auto& reg = it->second;
                            reg.reserved_bytes = (reg.reserved_bytes >= total) ? (reg.reserved_bytes - total) : 0;
                            reg.valid_bytes = saturatingAddU64_(reg.valid_bytes, total);
                            if (reg.size_bytes > 0 && reg.valid_bytes > reg.size_bytes) {
                                reg.valid_bytes = reg.size_bytes;
                            }
                            update_program_ub_occupancy_max();
                        }
                    }
                }

                slot = ProgramDmaSlot{};
                tensor_program_ops_total_ += 1;
            }
        };

        // Shared DMA budget across read+write slots: issue reads first, then writes.
        service_dma_slot(program_dma_read_slot_, MemTag::ProgramDmaRead);
        service_dma_slot(program_dma_write_slot_, MemTag::ProgramDmaWrite);
    }

    // === Execute: MXU (compute placeholder) ===
    if (program_mxu_slot_.active) {
        if (collective_blocks_compute) {
            tensor_stall_collective_cycles_total_ += 1;
        } else if (program_mxu_slot_.rem_cycles > 0 ||
                   program_mxu_slot_.ub_read_a_bytes_rem > 0 ||
                   program_mxu_slot_.ub_read_b_bytes_rem > 0 ||
                   program_mxu_slot_.ub_write_bytes_rem > 0) {
            bool ok = true;
            bool onchip_port_stall_marked = false;
            bool onchip_bank_stall_marked = false;
            auto mark_onchip_port_stall = [&]() {
                if (!onchip_port_stall_marked) {
                    tensor_stall_onchip_port_cycles_total_ += 1;
                    onchip_port_stall_marked = true;
                }
            };
            auto mark_onchip_bank_stall = [&]() {
                if (!onchip_bank_stall_marked) {
                    tensor_stall_onchip_bank_conflict_cycles_total_ += 1;
                    onchip_bank_stall_marked = true;
                }
            };

            auto try_acquire_ub_banks2 = [&](uint64_t seed0, uint64_t seed1) -> bool {
                if (!cfg_.onchip_model_enable) return true;
                if (!cfg_.onchip_bank_model_enable) return true;
                const uint32_t bank0 = selectUbBank_(seed0);
                const uint32_t bank1 = selectUbBank_(seed1);
                if (bank0 >= onchip_ub_bank_queue_occupancy_.size()) return true;
                if (bank1 >= onchip_ub_bank_queue_occupancy_.size()) return true;
                const uint32_t depth = std::max<uint32_t>(cfg_.bank_queue_depth, 1u);

                if (cfg_.ub_bank_conflict_mode == "queue") {
                    if (bank0 == bank1) {
                        const uint32_t occ = onchip_ub_bank_queue_occupancy_[bank0];
                        if (occ + 2u > depth) return false;
                        onchip_ub_bank_queue_occupancy_[bank0] = occ + 2u;
                    } else {
                        const uint32_t occ0 = onchip_ub_bank_queue_occupancy_[bank0];
                        const uint32_t occ1 = onchip_ub_bank_queue_occupancy_[bank1];
                        if (occ0 + 1u > depth) return false;
                        if (occ1 + 1u > depth) return false;
                        onchip_ub_bank_queue_occupancy_[bank0] = occ0 + 1u;
                        onchip_ub_bank_queue_occupancy_[bank1] = occ1 + 1u;
                    }
                    updateBankQueueOccupancyMax_();
                    return true;
                }

                // hard: only 1 access per bank per cycle
                if (bank0 == bank1) return false;
                if (onchip_ub_bank_queue_occupancy_[bank0] > 0) return false;
                if (onchip_ub_bank_queue_occupancy_[bank1] > 0) return false;
                onchip_ub_bank_queue_occupancy_[bank0] = 1;
                onchip_ub_bank_queue_occupancy_[bank1] = 1;
                updateBankQueueOccupancyMax_();
                return true;
            };

            auto try_acquire_acc_bank = [&](uint64_t seed) -> bool {
                if (!cfg_.onchip_model_enable) return true;
                if (!cfg_.onchip_bank_model_enable) return true;
                const uint32_t bank = selectAccBank_(seed);
                if (bank >= onchip_acc_bank_queue_occupancy_.size()) return true;
                const uint32_t depth = std::max<uint32_t>(cfg_.bank_queue_depth, 1u);

                if (cfg_.acc_bank_conflict_mode == "queue") {
                    if (onchip_acc_bank_queue_occupancy_[bank] >= depth) return false;
                    onchip_acc_bank_queue_occupancy_[bank] += 1;
                    updateBankQueueOccupancyMax_();
                    return true;
                }

                // hard: only 1 access per bank per cycle
                if (onchip_acc_bank_queue_occupancy_[bank] > 0) return false;
                onchip_acc_bank_queue_occupancy_[bank] = 1;
                updateBankQueueOccupancyMax_();
                return true;
            };

            if (cfg_.onchip_model_enable) {
                // Conservative per-cycle demand: 2x UB read + 1x ACC write.
                if (!acquireOnchipReadPorts_(2, 0)) {
                    mark_onchip_port_stall();
                    ok = false;
                } else if (!acquireOnchipWritePorts_(0, 1)) {
                    mark_onchip_port_stall();
                    ok = false;
                }

                if (ok) {
                    const uint64_t tag_base =
                        (static_cast<uint64_t>(program_mxu_slot_.buf) << 48) ^
                        (static_cast<uint64_t>(program_pc_) << 16) ^
                        (static_cast<uint64_t>(program_iter_done_) << 1) ^
                        cfg_.seed_base;
                    if (!try_acquire_ub_banks2(tag_base ^ 0x1001ull, tag_base ^ 0x1002ull)) {
                        mark_onchip_bank_stall();
                        ok = false;
                    } else if (!try_acquire_acc_bank(tag_base ^ 0x2001ull)) {
                        mark_onchip_bank_stall();
                        ok = false;
                    }
                }
            }

            if (ok) {
                const bool have_math = (program_mxu_slot_.rem_cycles > 0);
                const bool have_io =
                    (program_mxu_slot_.ub_read_a_bytes_rem > 0) ||
                    (program_mxu_slot_.ub_read_b_bytes_rem > 0) ||
                    (program_mxu_slot_.ub_write_bytes_rem > 0);

                tensor_program_mxu_busy_cycles_total_ += 1;
                tensor_compute_cycles_total_ += 1;

                if (have_math) {
                    tensor_compute_math_cycles_total_ += 1;
                    tensor_mac_ops_total_ += effectivePeakMacsPerCycle_();
                    program_mxu_slot_.rem_cycles -= 1;
                }

                if (have_io) {
                    tensor_mxu_io_busy_cycles_total_ += 1;
                    const uint64_t a_bpc = cfg_.mxu_a_bytes_per_cycle;
                    const uint64_t b_bpc = cfg_.mxu_b_bytes_per_cycle;
                    const uint64_t c_bpc = cfg_.mxu_c_bytes_per_cycle;
                    auto consume = [&](uint64_t& rem, uint64_t bpc) {
                        if (rem == 0) return;
                        const uint64_t cut = (bpc > 0) ? std::min<uint64_t>(rem, bpc) : rem;
                        rem -= cut;
                    };
                    consume(program_mxu_slot_.ub_read_a_bytes_rem, a_bpc);
                    consume(program_mxu_slot_.ub_read_b_bytes_rem, b_bpc);
                    consume(program_mxu_slot_.ub_write_bytes_rem, c_bpc);
                }

                did = true;
                did_compute = true;
            }
        }
        if (program_mxu_slot_.active &&
            program_mxu_slot_.rem_cycles == 0 &&
            program_mxu_slot_.ub_read_a_bytes_rem == 0 &&
            program_mxu_slot_.ub_read_b_bytes_rem == 0 &&
            program_mxu_slot_.ub_write_bytes_rem == 0) {
            const uint64_t w = program_mxu_slot_.ub_write_bytes;
            const uint64_t rsv = program_mxu_slot_.ub_write_reserved_bytes;
            const uint32_t buf = program_mxu_slot_.buf;
            if (!program_addr_aware_enable_) {
                if (buf < program_ub_reserved_bytes_by_buf_.size() && buf < program_ub_valid_bytes_by_buf_.size()) {
                    auto& rsv_bytes = program_ub_reserved_bytes_by_buf_[buf];
                    auto& val_bytes = program_ub_valid_bytes_by_buf_[buf];
                    if (rsv > 0) {
                        rsv_bytes = (rsv_bytes >= rsv) ? (rsv_bytes - rsv) : 0;
                    }
                    if (w > 0) {
                        val_bytes = saturatingAddU64_(val_bytes, w);
                    }
                    update_program_ub_occupancy_max();
                }
            } else if (buf < program_ub_regions_by_buf_.size() && program_mxu_slot_.ub_write_addr_present) {
                auto& mp = program_ub_regions_by_buf_[buf];
                auto it = mp.find(program_mxu_slot_.ub_write_addr);
                if (it != mp.end()) {
                    auto& reg = it->second;
                    if (rsv > 0) {
                        reg.reserved_bytes = (reg.reserved_bytes >= rsv) ? (reg.reserved_bytes - rsv) : 0;
                    }
                    if (w > 0) {
                        reg.valid_bytes = saturatingAddU64_(reg.valid_bytes, w);
                        if (reg.size_bytes > 0 && reg.valid_bytes > reg.size_bytes) {
                            reg.valid_bytes = reg.size_bytes;
                        }
                    }
                    update_program_ub_occupancy_max();
                }
            }
            program_mxu_slot_ = ProgramMxuSlot{};
            tensor_program_ops_total_ += 1;
        }
    }

    // === Execute: Vector ===
    if (program_vec_slot_.active) {
        if (collective_blocks_compute) {
            tensor_stall_collective_cycles_total_ += 1;
        } else if (program_vec_slot_.rem_cycles > 0) {
            tensor_program_vec_busy_cycles_total_ += 1;
            tensor_vector_cycles_total_ += 1;
            program_vec_slot_.rem_cycles -= 1;
            did = true;
            did_compute = true;
        }
        if (program_vec_slot_.active && program_vec_slot_.rem_cycles == 0) {
            program_vec_slot_ = ProgramVecSlot{};
            tensor_program_ops_total_ += 1;
        }
    }

    // === Execute: Collective/Comm/credit-return ===
    issue_noc();
    if (program_coll_slot_.active) {
        tensor_program_coll_busy_cycles_total_ += 1;
        bool done = !collectivePendingActive_();
        if (done && program_coll_slot_.blocking && cfg_.collective_blocking && collective_epoch_active_) {
            tensor_stall_collective_cycles_total_ += 1;
            done = false;
        }
        if (done) {
            program_coll_slot_ = ProgramCollSlot{};
            tensor_program_ops_total_ += 1;
        }
    }

    if (cfg_.collective_overlap_with_compute && did_compute && did_collective) {
        tensor_overlap_compute_collective_cycles_total_ += 1;
    }
    if (did_compute && did_mem) {
        tensor_overlap_compute_mem_cycles_total_ += 1;
    }
    if (did_mem || did_compute || did_collective || did_comm) {
        tensor_program_any_busy_cycles_total_ += 1;
    }

    // === Fence ===
    if (program_fence_pending_) {
        bool ok = true;
        if (program_dma_read_slot_.active || program_dma_write_slot_.active || program_mxu_slot_.active || program_vec_slot_.active ||
            program_coll_slot_.active)
            ok = false;
        if (!inflight_.empty()) ok = false;
        if (collectivePendingActive_()) ok = false;
        if (ok) {
            program_fence_pending_ = false;
            tensor_program_fence_count_total_ += 1;
            tensor_program_ops_total_ += 1;
        } else {
            tensor_program_fence_wait_cycles_total_ += 1;
        }
    }

    // === Issue ===
    if (!program_fence_pending_) {
        const uint32_t ub_bufs = std::max<uint32_t>(cfg_.program_ub_buffers, 1u);
        const uint64_t ub_part =
            (program_addr_aware_enable_ && ub_bufs > 0) ? (cfg_.ub_bytes / static_cast<uint64_t>(ub_bufs)) : 0ull;
        auto clear_valid_in_buf = [&](uint32_t buf) {
            if (buf >= program_ub_regions_by_buf_.size()) return;
            auto& mp = program_ub_regions_by_buf_[buf];
            for (auto& kv : mp) {
                kv.second.valid_bytes = 0;
            }
        };
        auto get_or_create_region = [&](uint32_t buf, uint64_t addr, uint64_t size_bytes) -> ProgramUbRegion* {
            if (buf >= program_ub_regions_by_buf_.size()) return nullptr;
            auto& mp = program_ub_regions_by_buf_[buf];
            auto it = mp.find(addr);
            if (it == mp.end()) {
                auto ins = mp.emplace(addr, ProgramUbRegion{size_bytes, 0ull, 0ull});
                it = ins.first;
            } else if (it->second.size_bytes != size_bytes) {
                if (cfg_.strict && rt_.log) {
                    rt_.log->fatal(
                        CALL_INFO,
                        -1,
                        "tensor fatal: program addr-aware region size mismatch (core=%u buf=%u ub_addr=%llu expected=%llu got=%llu)\n",
                        rt_.core_id,
                        buf,
                        (unsigned long long)addr,
                        (unsigned long long)it->second.size_bytes,
                        (unsigned long long)size_bytes);
                }
                return nullptr;
            }
            return &it->second;
        };

        // M28: attribute a subset of program-mode UB stalls into the shared stall counters
        // so exec_mode=program becomes comparable with exec_mode=tile in summaries.
        bool wait_read_stall_marked = false;
        bool onchip_capacity_stall_marked = false;
        auto mark_wait_read_stall = [&]() {
            if (!wait_read_stall_marked) {
                tensor_stall_wait_read_cycles_total_ += 1;
                wait_read_stall_marked = true;
            }
            tensor_program_ub_stall_cycles_total_ += 1;
        };
        auto mark_onchip_capacity_stall = [&]() {
            if (!onchip_capacity_stall_marked) {
                tensor_stall_onchip_capacity_cycles_total_ += 1;
                onchip_capacity_stall_marked = true;
            }
            tensor_program_ub_stall_cycles_total_ += 1;
        };

        uint32_t issued = 0;
        while (issued < cfg_.program_issue_width && program_pc_ < program_ops_.size()) {
            const ProgramOp& op = program_ops_[program_pc_];
            if (op.kind == ProgramOpKind::Fence) {
                program_fence_pending_ = true;
                program_pc_ += 1;
                break;
            }

            bool ok = false;
            if (op.kind == ProgramOpKind::DmaRead) {
                const uint32_t buf = op.buf;
                const bool any_dma_active = program_dma_read_slot_.active || program_dma_write_slot_.active;
                if (program_dma_read_slot_.active || (!cfg_.program_dma_dual_enable && any_dma_active)) {
                    ok = false;
                } else {
                    if (!program_addr_aware_enable_) {
                        if (buf >= program_ub_reserved_bytes_by_buf_.size() || buf >= program_ub_valid_bytes_by_buf_.size()) {
                            ok = false;
                        } else if (cfg_.ub_bytes > 0 &&
                                   ([&]() -> bool {
                                       uint64_t occ = program_ub_total_occupancy_bytes();
                                       if (op.reset) {
                                           const uint64_t v = program_ub_valid_bytes_by_buf_[buf];
                                           occ = (occ >= v) ? (occ - v) : 0;
                                       }
                                       return saturatingAddU64_(occ, op.bytes) > cfg_.ub_bytes;
                                   })()) {
                            mark_onchip_capacity_stall();
                            ok = false;
                        } else {
                            program_dma_read_slot_.active = true;
                            program_dma_read_slot_.is_read = true;
                            program_dma_read_slot_.buf = buf;
                            program_dma_read_slot_.ub_addr_present = op.ub_addr_present;
                            program_dma_read_slot_.ub_addr = op.ub_addr;
                            program_dma_read_slot_.epoch = program_dma_epoch_next_++;
                            program_dma_read_slot_.total_bytes = op.bytes;
                            program_dma_read_slot_.issued_bytes = 0;
                            program_dma_read_slot_.done_bytes = 0;
                            if (op.reset) {
                                program_ub_valid_bytes_by_buf_[buf] = 0;
                            }
                            program_ub_reserved_bytes_by_buf_[buf] =
                                saturatingAddU64_(program_ub_reserved_bytes_by_buf_[buf], op.bytes);
                            update_program_ub_occupancy_max();
                            ok = true;
                        }
                    } else {
                        if (buf >= program_ub_regions_by_buf_.size() || !op.ub_addr_present) {
                            ok = false;
                        } else if (ub_part == 0 || op.ub_addr + op.bytes > ub_part) {
                            if (cfg_.strict && rt_.log) {
                                rt_.log->fatal(
                                    CALL_INFO,
                                    -1,
                                    "tensor fatal: program dma_read ub_addr range overflow (core=%u buf=%u ub_addr=%llu bytes=%llu ub_part=%llu)\n",
                                    rt_.core_id,
                                    buf,
                                    (unsigned long long)op.ub_addr,
                                    (unsigned long long)op.bytes,
                                    (unsigned long long)ub_part);
                            }
                            ok = false;
                        } else {
                            if (op.reset) {
                                clear_valid_in_buf(buf);
                            }
                            ProgramUbRegion* reg = get_or_create_region(buf, op.ub_addr, op.bytes);
                            if (!reg) {
                                ok = false;
                            } else if (reg->reserved_bytes > 0 || reg->valid_bytes > 0) {
                                mark_onchip_capacity_stall();
                                ok = false;
                            } else {
                                reg->reserved_bytes = saturatingAddU64_(reg->reserved_bytes, op.bytes);
                                if (reg->size_bytes > 0 && reg->reserved_bytes > reg->size_bytes) {
                                    reg->reserved_bytes = reg->size_bytes;
                                }
                                update_program_ub_occupancy_max();

                                program_dma_read_slot_.active = true;
                                program_dma_read_slot_.is_read = true;
                                program_dma_read_slot_.buf = buf;
                                program_dma_read_slot_.ub_addr_present = true;
                                program_dma_read_slot_.ub_addr = op.ub_addr;
                                program_dma_read_slot_.epoch = program_dma_epoch_next_++;
                                program_dma_read_slot_.total_bytes = op.bytes;
                                program_dma_read_slot_.issued_bytes = 0;
                                program_dma_read_slot_.done_bytes = 0;
                                ok = true;
                            }
                        }
                    }
                }
            } else if (op.kind == ProgramOpKind::DmaWrite) {
                const uint32_t buf = op.buf;
                const bool any_dma_active = program_dma_read_slot_.active || program_dma_write_slot_.active;
                if (program_dma_write_slot_.active || (!cfg_.program_dma_dual_enable && any_dma_active)) {
                    ok = false;
                } else {
                    if (!program_addr_aware_enable_) {
                        if (buf >= program_ub_valid_bytes_by_buf_.size()) {
                            ok = false;
                        } else if (program_ub_valid_bytes_by_buf_[buf] < op.bytes) {
                            mark_wait_read_stall();
                            ok = false;
                        } else {
                            if (op.consume) {
                                program_ub_valid_bytes_by_buf_[buf] -= op.bytes;
                            }
                            program_dma_write_slot_.active = true;
                            program_dma_write_slot_.is_read = false;
                            program_dma_write_slot_.buf = buf;
                            program_dma_write_slot_.ub_addr_present = op.ub_addr_present;
                            program_dma_write_slot_.ub_addr = op.ub_addr;
                            program_dma_write_slot_.epoch = program_dma_epoch_next_++;
                            program_dma_write_slot_.total_bytes = op.bytes;
                            program_dma_write_slot_.issued_bytes = 0;
                            program_dma_write_slot_.done_bytes = 0;
                            ok = true;
                        }
                    } else {
                        if (buf >= program_ub_regions_by_buf_.size() || !op.ub_addr_present) {
                            ok = false;
                        } else if (ub_part == 0 || op.ub_addr + op.bytes > ub_part) {
                            if (cfg_.strict && rt_.log) {
                                rt_.log->fatal(
                                    CALL_INFO,
                                    -1,
                                    "tensor fatal: program dma_write ub_addr range overflow (core=%u buf=%u ub_addr=%llu bytes=%llu ub_part=%llu)\n",
                                    rt_.core_id,
                                    buf,
                                    (unsigned long long)op.ub_addr,
                                    (unsigned long long)op.bytes,
                                    (unsigned long long)ub_part);
                            }
                            ok = false;
                        } else {
                            ProgramUbRegion* reg = get_or_create_region(buf, op.ub_addr, op.bytes);
                            if (!reg) {
                                ok = false;
                            } else if (reg->valid_bytes < op.bytes) {
                                mark_wait_read_stall();
                                ok = false;
                            } else {
                                if (op.consume) {
                                    reg->valid_bytes -= op.bytes;
                                }
                                update_program_ub_occupancy_max();

                                program_dma_write_slot_.active = true;
                                program_dma_write_slot_.is_read = false;
                                program_dma_write_slot_.buf = buf;
                                program_dma_write_slot_.ub_addr_present = true;
                                program_dma_write_slot_.ub_addr = op.ub_addr;
                                program_dma_write_slot_.epoch = program_dma_epoch_next_++;
                                program_dma_write_slot_.total_bytes = op.bytes;
                                program_dma_write_slot_.issued_bytes = 0;
                                program_dma_write_slot_.done_bytes = 0;
                                ok = true;
                            }
                        }
                    }
                }
            } else if (op.kind == ProgramOpKind::GemmUb) {
                const uint32_t buf = op.buf;
                if (program_mxu_slot_.active) {
                    ok = false;
                } else {
                    uint64_t cycles = op.cycles;
                    if (cycles == 0) {
                        cycles = estimateGemmUbCycles_(op.m, op.n, op.k);
                    }

                    if (!program_addr_aware_enable_) {
                        if (buf >= program_ub_reserved_bytes_by_buf_.size() || buf >= program_ub_valid_bytes_by_buf_.size()) {
                            ok = false;
                        } else if (program_ub_valid_bytes_by_buf_[buf] < op.ub_read_bytes) {
                            mark_wait_read_stall();
                            ok = false;
                        } else if (cfg_.ub_bytes > 0 &&
                                   saturatingAddU64_(program_ub_total_occupancy_bytes(), op.ub_write_bytes) > cfg_.ub_bytes) {
                            mark_onchip_capacity_stall();
                            ok = false;
                        } else {
                            if (op.ub_write_bytes > 0) {
                                program_ub_reserved_bytes_by_buf_[buf] =
                                    saturatingAddU64_(program_ub_reserved_bytes_by_buf_[buf], op.ub_write_bytes);
                                update_program_ub_occupancy_max();
                            }
                            program_mxu_slot_.active = true;
                            program_mxu_slot_.buf = buf;
                            program_mxu_slot_.ub_write_addr_present = op.ub_write_addr_present;
                            program_mxu_slot_.ub_write_addr = op.ub_write_addr;
                            program_mxu_slot_.rem_cycles = cycles;
                            program_mxu_slot_.ub_read_bytes = op.ub_read_bytes;
                            program_mxu_slot_.ub_write_bytes = op.ub_write_bytes;
                            program_mxu_slot_.ub_write_reserved_bytes = op.ub_write_bytes;
                            {
                                const uint64_t total_in = op.ub_read_bytes;
                                const uint64_t a_bytes = total_in / 2;
                                const uint64_t b_bytes = total_in - a_bytes;
                                program_mxu_slot_.ub_read_a_bytes_rem = a_bytes;
                                program_mxu_slot_.ub_read_b_bytes_rem = b_bytes;
                                program_mxu_slot_.ub_write_bytes_rem = op.ub_write_bytes;
                            }
                            ok = true;
                        }
                    } else {
                        if (buf >= program_ub_regions_by_buf_.size() || !op.ub_read_addr_present || !op.ub_write_addr_present) {
                            ok = false;
                        } else if (ub_part == 0 ||
                                   (op.ub_read_bytes > 0 && op.ub_read_addr + op.ub_read_bytes > ub_part) ||
                                   (op.ub_write_bytes > 0 && op.ub_write_addr + op.ub_write_bytes > ub_part)) {
                            if (cfg_.strict && rt_.log) {
                                rt_.log->fatal(
                                    CALL_INFO,
                                    -1,
                                    "tensor fatal: program gemm_ub ub_addr range overflow (core=%u buf=%u ub_part=%llu)\n",
                                    rt_.core_id,
                                    buf,
                                    (unsigned long long)ub_part);
                            }
                            ok = false;
                        } else {
                            ok = true;
                            if (op.ub_read_bytes > 0) {
                                ProgramUbRegion* in = get_or_create_region(buf, op.ub_read_addr, op.ub_read_bytes);
                                if (!in) {
                                    ok = false;
                                } else if (in->valid_bytes < op.ub_read_bytes) {
                                    mark_wait_read_stall();
                                    ok = false;
                                }
                            }
                            if (ok && op.ub_write_bytes > 0) {
                                ProgramUbRegion* out = get_or_create_region(buf, op.ub_write_addr, op.ub_write_bytes);
                                if (!out) {
                                    ok = false;
                                } else if (out->reserved_bytes > 0 || out->valid_bytes > 0) {
                                    mark_onchip_capacity_stall();
                                    ok = false;
                                } else {
                                    out->reserved_bytes = saturatingAddU64_(out->reserved_bytes, op.ub_write_bytes);
                                    if (out->size_bytes > 0 && out->reserved_bytes > out->size_bytes) {
                                        out->reserved_bytes = out->size_bytes;
                                    }
                                    update_program_ub_occupancy_max();
                                }
                            }
                            if (ok) {
                                program_mxu_slot_.active = true;
                                program_mxu_slot_.buf = buf;
                                program_mxu_slot_.ub_write_addr_present = op.ub_write_addr_present;
                                program_mxu_slot_.ub_write_addr = op.ub_write_addr;
                                    program_mxu_slot_.rem_cycles = cycles;
                                    program_mxu_slot_.ub_read_bytes = op.ub_read_bytes;
                                    program_mxu_slot_.ub_write_bytes = op.ub_write_bytes;
                                    program_mxu_slot_.ub_write_reserved_bytes = op.ub_write_bytes;
                                    {
                                        const uint64_t total_in = op.ub_read_bytes;
                                        const uint64_t a_bytes = total_in / 2;
                                        const uint64_t b_bytes = total_in - a_bytes;
                                        program_mxu_slot_.ub_read_a_bytes_rem = a_bytes;
                                        program_mxu_slot_.ub_read_b_bytes_rem = b_bytes;
                                        program_mxu_slot_.ub_write_bytes_rem = op.ub_write_bytes;
                                    }
                                    ok = true;
                                }
                            }
                        }
                }
            } else if (op.kind == ProgramOpKind::Softmax) {
                if (program_vec_slot_.active) {
                    ok = false;
                } else {
                    const uint64_t denom = static_cast<uint64_t>(std::max<uint32_t>(cfg_.vector_elems_per_cycle, 1u));
                    const uint64_t body = ceilDivU64_(std::max<uint64_t>(1ull, op.elems), denom);
                    const uint64_t pipe = static_cast<uint64_t>(cfg_.vector_pipeline_latency_cycles);
                    uint64_t cycles = saturatingAddU64_(body, pipe);
                    if (cycles == 0) cycles = 1;
                    program_vec_slot_.active = true;
                    program_vec_slot_.rem_cycles = cycles;
                    ok = true;
                }
            } else if (op.kind == ProgramOpKind::Allreduce) {
                if (program_coll_slot_.active) {
                    ok = false;
                } else {
                    const uint64_t bytes = (op.bytes > 0) ? op.bytes : cfg_.collective_bytes;
                    if (!startProgramCollective_(now_cycle, bytes, op.blocking)) {
                        if (cfg_.strict && rt_.log) {
                            rt_.log->fatal(CALL_INFO, -1, "tensor fatal: failed to start program allreduce (core=%u)\n", rt_.core_id);
                        }
                        ok = false;
                    } else {
                        program_coll_slot_.active = true;
                        program_coll_slot_.bytes = bytes;
                        program_coll_slot_.blocking = op.blocking;
                        ok = true;
                    }
                }
            } else {
                // Unknown / unsupported op in M7 pipeline.
                ok = false;
            }

            if (!ok) break;
            program_pc_ += 1;
            issued += 1;
        }
    }

    // === Iteration completion ===
    if (program_pc_ >= program_ops_.size() &&
        !program_fence_pending_ &&
        !program_dma_read_slot_.active &&
        !program_dma_write_slot_.active &&
        !program_mxu_slot_.active &&
        !program_vec_slot_.active &&
        !program_coll_slot_.active &&
        inflight_.empty() &&
        !collectivePendingActive_()) {
        tensor_program_iters_total_ += 1;
        program_iter_done_ += 1;
        if (step_gated_) {
            step_open_ = false;
        }
        if (program_loop_enable_ && (cfg_.iterations == 0 || program_iter_done_ < cfg_.iterations)) {
            program_pc_ = 0;
        }
    }

    if (did) active_cycles_++;
    return did;
}

bool TensorWorkload::onClockTickProgram_(uint64_t now_cycle) {
    if (program_m7_enable_) {
        return onClockTickProgramM7_(now_cycle);
    }

    // Always allow NoC control traffic (credit return) to make forward progress.
    bool did = false;
    bool did_collective = false;
    bool did_comm = false;
    bool noc_budget_blocked = false;
    uint64_t noc_budget = (cfg_.noc_bandwidth_bytes_per_cycle > 0)
                              ? cfg_.noc_bandwidth_bytes_per_cycle
                              : std::numeric_limits<uint64_t>::max();

    auto issue_noc = [&]() {
        issueNocTraffic_(now_cycle, noc_budget, did, did_collective, did_comm, noc_budget_blocked);
        if (noc_budget_blocked && !did_collective && !did_comm) {
            tensor_stall_noc_budget_cycles_total_ += 1;
        }
    };

    // Step-gated mode: only run when a step is open, but still flush credit-return traffic.
    if (step_gated_ && !step_open_) {
        issue_noc();
        if (did) active_cycles_++;
        return did;
    }

    if (!started_) {
        if (now_cycle < cfg_.start_cycle) {
            issue_noc();
            if (did) active_cycles_++;
            return did;
        }
        started_ = true;
    }

    serviceCollectiveBarrier_(now_cycle);

    // Stop after requested program iterations (0 = run forever).
    if (cfg_.iterations > 0 && program_iter_done_ >= cfg_.iterations) {
        issue_noc();
        if (did) active_cycles_++;
        return did;
    }

    if (program_ops_.empty()) {
        // Program mode with an empty program is allowed when strict=0 (acts like idle workload).
        issue_noc();
        if (did) active_cycles_++;
        return did;
    }

    if (program_pc_ >= program_ops_.size()) {
        // Program finished (or waiting for the next global step). Stay idle but keep draining NoC control traffic.
        issue_noc();
        if (did) active_cycles_++;
        return did;
    }

    auto finish_program_iter = [&]() {
        tensor_program_iters_total_ += 1;
        program_iter_done_ += 1;
        if (step_gated_) {
            step_open_ = false;
        }
        if (program_loop_enable_ && (cfg_.iterations == 0 || program_iter_done_ < cfg_.iterations)) {
            program_pc_ = 0;
        }
    };

    const ProgramOp op = program_ops_[program_pc_];
    if (op.kind == ProgramOpKind::Gemm) {
        const bool did_gemm = onClockTickTile_(now_cycle);
        if (iter_active_) {
            program_gemm_started_ = true;
        }
        if (program_gemm_started_ && !iter_active_) {
            program_gemm_started_ = false;
            program_op_started_ = false;
            program_collective_started_ = false;
            program_softmax_rem_cycles_ = 0;
            tensor_program_ops_total_ += 1;
            program_pc_ += 1;
            if (program_pc_ >= program_ops_.size()) {
                finish_program_iter();
            }
        }
        return did_gemm;
    }

    if (op.kind == ProgramOpKind::Allreduce) {
        if (!program_collective_started_) {
            const uint64_t bytes = (op.bytes > 0) ? op.bytes : cfg_.collective_bytes;
            if (!startProgramCollective_(now_cycle, bytes, op.blocking)) {
                if (cfg_.strict && rt_.log) {
                    rt_.log->fatal(CALL_INFO, -1, "tensor fatal: failed to start program allreduce (core=%u)\n", rt_.core_id);
                }
                issue_noc();
                if (did) active_cycles_++;
                return did;
            }
            program_collective_started_ = true;
            program_op_started_ = true;
        }

        issue_noc();

        bool done = program_collective_started_ && !collectivePendingActive_();
        if (done && op.blocking && cfg_.collective_blocking && collective_epoch_active_) {
            // Blocking collective: wait for recv completion when enabled.
            tensor_stall_collective_cycles_total_ += 1;
            done = false;
        }
        if (done) {
            program_collective_started_ = false;
            program_op_started_ = false;
            tensor_program_ops_total_ += 1;
            program_pc_ += 1;
            if (program_pc_ >= program_ops_.size()) {
                finish_program_iter();
            }
        }
        if (did) active_cycles_++;
        return did;
    }

    // Softmax/vector placeholder: compute-only cycles + NoC credit-return flushing.
    if (!program_op_started_) {
        const uint64_t elems = op.elems;
        const uint64_t denom = static_cast<uint64_t>(std::max<uint32_t>(cfg_.vector_elems_per_cycle, 1u));
        const uint64_t body = ceilDivU64_(std::max<uint64_t>(1ull, elems), denom);
        const uint64_t pipe = static_cast<uint64_t>(cfg_.vector_pipeline_latency_cycles);
        program_softmax_rem_cycles_ = saturatingAddU64_(body, pipe);
        if (program_softmax_rem_cycles_ == 0) program_softmax_rem_cycles_ = 1;
        program_op_started_ = true;
    }

    issue_noc();

    if (program_softmax_rem_cycles_ > 0) {
        program_softmax_rem_cycles_ -= 1;
        tensor_vector_cycles_total_ += 1;
        did = true;
    }
    if (program_op_started_ && program_softmax_rem_cycles_ == 0) {
        program_op_started_ = false;
        tensor_program_ops_total_ += 1;
        program_pc_ += 1;
        if (program_pc_ >= program_ops_.size()) {
            finish_program_iter();
        }
    }
    if (did) active_cycles_++;
    return did;
}


}} // namespace SST::SnnDL

