// -*- c++ -*-

#include <sst/core/sst_config.h>

#include "workload/tensor/TensorWorkload.h"

#include <sst/core/output.h>
#include <sst/core/params.h>

#include "IMemoryAccess.h"
#include "INocTransport.h"
#include "NocPacketEvent.h"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <cstdint>
#include <cstring>
#include <limits>

namespace SST { namespace SnnDL {

namespace {

inline uint64_t clamp_u64_(uint64_t v, uint64_t lo, uint64_t hi) {
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

inline std::string to_lower_copy_(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char ch : s) out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    return out;
}

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

uint64_t TensorWorkload::ceilDivU64_(uint64_t a, uint64_t b) {
    if (b == 0) return a;
    return (a + b - 1) / b;
}

uint64_t TensorWorkload::clampNonZero_(uint64_t v, uint64_t fallback) {
    if (v != 0) return v;
    return fallback ? fallback : 1ull;
}

uint64_t TensorWorkload::splitmix64_next_(uint64_t& x) {
    x += 0x9e3779b97f4a7c15ULL;
    uint64_t z = x;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

void TensorWorkload::fillBytesDeterministic_(uint64_t seed,
                                            uint64_t addr,
                                            uint32_t seq,
                                            std::vector<uint8_t>& out) {
    uint64_t rng = seed ^ addr ^ (static_cast<uint64_t>(seq) << 1) ^ 0x54454e534f52ULL; // "TENSOR"
    size_t pos = 0;
    while (pos < out.size()) {
        const uint64_t v = splitmix64_next_(rng);
        const size_t n = std::min<size_t>(8, out.size() - pos);
        std::memcpy(out.data() + pos, &v, n);
        pos += n;
    }
}

void TensorWorkload::configureFromParams(const SST::Params& params) {
    cfg_.m = params.find<uint32_t>("tensor_m", cfg_.m);
    cfg_.n = params.find<uint32_t>("tensor_n", cfg_.n);
    cfg_.k = params.find<uint32_t>("tensor_k", cfg_.k);
    cfg_.element_bytes = params.find<uint32_t>("tensor_element_bytes", cfg_.element_bytes);

    cfg_.array_m = params.find<uint32_t>("tensor_array_m", cfg_.array_m);
    cfg_.array_n = params.find<uint32_t>("tensor_array_n", cfg_.array_n);
    cfg_.compute_efficiency = params.find<float>("tensor_compute_efficiency", cfg_.compute_efficiency);

    cfg_.overlap_enable = params.find<int>("tensor_overlap_enable", cfg_.overlap_enable ? 1 : 0) != 0;
    cfg_.start_cycle = params.find<uint64_t>("tensor_start_cycle", cfg_.start_cycle);
    cfg_.iterations = params.find<uint32_t>("tensor_iterations", cfg_.iterations);

    cfg_.mem_enable = params.find<int>("tensor_mem_enable", cfg_.mem_enable ? 1 : 0) != 0;
    cfg_.mem_region_bytes = params.find<uint64_t>("tensor_mem_region_bytes", cfg_.mem_region_bytes);
    cfg_.mem_req_bytes = params.find<uint32_t>("tensor_mem_req_bytes", cfg_.mem_req_bytes);
    cfg_.mem_max_outstanding = params.find<uint32_t>("tensor_mem_max_outstanding", cfg_.mem_max_outstanding);

    cfg_.dataflow = to_lower_copy_(params.find<std::string>("tensor_dataflow", cfg_.dataflow));
    cfg_.tile_m = params.find<uint32_t>("tensor_tile_m", cfg_.tile_m);
    cfg_.tile_n = params.find<uint32_t>("tensor_tile_n", cfg_.tile_n);
    cfg_.tile_k = params.find<uint32_t>("tensor_tile_k", cfg_.tile_k);
    cfg_.exec_mode = to_lower_copy_(params.find<std::string>("tensor_exec_mode", cfg_.exec_mode));
    cfg_.tile_schedule = to_lower_copy_(params.find<std::string>("tensor_tile_schedule", cfg_.tile_schedule));
    cfg_.writeback_policy = to_lower_copy_(params.find<std::string>("tensor_writeback_policy", cfg_.writeback_policy));
    cfg_.ub_bytes = params.find<uint64_t>("tensor_ub_bytes", cfg_.ub_bytes);
    cfg_.acc_bytes = params.find<uint64_t>("tensor_acc_bytes", cfg_.acc_bytes);
    cfg_.dma_bandwidth_bytes_per_cycle =
        params.find<uint64_t>("tensor_dma_bandwidth_bytes_per_cycle", cfg_.dma_bandwidth_bytes_per_cycle);
    cfg_.double_buffer = params.find<int>("tensor_double_buffer", cfg_.double_buffer ? 1 : 0) != 0;
    if (cfg_.exec_mode != "bulk" && cfg_.exec_mode != "tile") {
        cfg_.exec_mode = "bulk";
    }
    if (cfg_.tile_schedule != "auto" &&
        cfg_.tile_schedule != "mnk" &&
        cfg_.tile_schedule != "mkn" &&
        cfg_.tile_schedule != "nkm") {
        cfg_.tile_schedule = "auto";
    }
    if (cfg_.writeback_policy != "at_end_of_k") {
        cfg_.writeback_policy = "at_end_of_k";
    }
    if (cfg_.dataflow != "os" && cfg_.dataflow != "ws" && cfg_.dataflow != "is") {
        cfg_.dataflow = "os";
    }

    cfg_.collective_type = to_lower_copy_(params.find<std::string>("tensor_collective_type", cfg_.collective_type));
    cfg_.collective_blocking = params.find<int>("tensor_collective_blocking", cfg_.collective_blocking ? 1 : 0) != 0;
    cfg_.collective_scope = to_lower_copy_(params.find<std::string>("tensor_collective_scope", cfg_.collective_scope));
    cfg_.collective_bytes = params.find<uint64_t>("tensor_collective_bytes", cfg_.collective_bytes);
    cfg_.collective_period_cycles = params.find<uint64_t>("tensor_collective_period_cycles", cfg_.collective_period_cycles);
    cfg_.collective_pattern = to_lower_copy_(params.find<std::string>("tensor_collective_pattern", cfg_.collective_pattern));
    cfg_.collective_packet_bytes = params.find<uint32_t>("tensor_collective_packet_bytes", cfg_.collective_packet_bytes);
    if (cfg_.collective_scope != "per_core" &&
        cfg_.collective_scope != "per_pe" &&
        cfg_.collective_scope != "per_system") {
        cfg_.collective_scope = "per_core";
    }
    if (cfg_.collective_type != "none" &&
        cfg_.collective_type != "allreduce" &&
        cfg_.collective_type != "allgather" &&
        cfg_.collective_type != "reducescatter") {
        cfg_.collective_type = "none";
    }
    if (cfg_.collective_pattern != "ring" &&
        cfg_.collective_pattern != "mesh_x" &&
        cfg_.collective_pattern != "mesh_xy") {
        cfg_.collective_pattern = "ring";
    }

    cfg_.comm_enable = params.find<int>("tensor_comm_enable", cfg_.comm_enable ? 1 : 0) != 0;
    cfg_.comm_period_cycles = params.find<uint64_t>("tensor_comm_period_cycles", cfg_.comm_period_cycles);
    cfg_.comm_payload_bytes = params.find<uint32_t>("tensor_comm_payload_bytes", cfg_.comm_payload_bytes);

    cfg_.strict = params.find<int>("tensor_strict", cfg_.strict ? 1 : 0) != 0;
    cfg_.seed_base = params.find<uint64_t>("tensor_seed", cfg_.seed_base);
    total_cores_cfg_ = params.find<uint32_t>("total_cores", total_cores_cfg_);
    if (total_cores_cfg_ == 0) total_cores_cfg_ = 1;

    // === Hard bounds (avoid pathological allocations / invalid params) ===
    if (cfg_.m == 0) cfg_.m = 1;
    if (cfg_.n == 0) cfg_.n = 1;
    if (cfg_.k == 0) cfg_.k = 1;
    cfg_.element_bytes = static_cast<uint32_t>(clamp_u64_(cfg_.element_bytes, 1, 16));
    if (cfg_.array_m == 0) cfg_.array_m = 1;
    if (cfg_.array_n == 0) cfg_.array_n = 1;
    cfg_.mem_req_bytes = static_cast<uint32_t>(clamp_u64_(cfg_.mem_req_bytes, 1, 1024ull * 1024ull));
    cfg_.mem_max_outstanding = static_cast<uint32_t>(clamp_u64_(cfg_.mem_max_outstanding, 1, 4096));
    cfg_.mem_region_bytes = clampNonZero_(cfg_.mem_region_bytes, 4096);
    cfg_.start_cycle = clampNonZero_(cfg_.start_cycle, 1);
    cfg_.comm_payload_bytes = static_cast<uint32_t>(clamp_u64_(cfg_.comm_payload_bytes, 0, 1024ull * 1024ull));
    cfg_.collective_packet_bytes = static_cast<uint32_t>(clamp_u64_(cfg_.collective_packet_bytes, 1, 1024ull * 1024ull));
    if (cfg_.collective_packet_bytes < 8) {
        cfg_.collective_packet_bytes = 8;
    }

    // Derived per-iteration sizes/ops (GEMM)
    const unsigned __int128 a_elems = static_cast<unsigned __int128>(cfg_.m) * static_cast<unsigned __int128>(cfg_.k);
    const unsigned __int128 b_elems = static_cast<unsigned __int128>(cfg_.k) * static_cast<unsigned __int128>(cfg_.n);
    const unsigned __int128 c_elems = static_cast<unsigned __int128>(cfg_.m) * static_cast<unsigned __int128>(cfg_.n);
    const unsigned __int128 eb = static_cast<unsigned __int128>(cfg_.element_bytes);
    const unsigned __int128 macs = c_elems * static_cast<unsigned __int128>(cfg_.k);
    mac_ops_per_iter_ =
        (macs > static_cast<unsigned __int128>(std::numeric_limits<uint64_t>::max()))
            ? std::numeric_limits<uint64_t>::max()
            : static_cast<uint64_t>(macs);
    compute_cycles_per_iter_ = ceilDivU64_(mac_ops_per_iter_, effectivePeakMacsPerCycle_());

    const uint64_t tm = clampNonZero_(std::min<uint64_t>(cfg_.tile_m ? cfg_.tile_m : cfg_.m, cfg_.m), 1);
    const uint64_t tn = clampNonZero_(std::min<uint64_t>(cfg_.tile_n ? cfg_.tile_n : cfg_.n, cfg_.n), 1);
    const uint64_t tk = clampNonZero_(std::min<uint64_t>(cfg_.tile_k ? cfg_.tile_k : cfg_.k, cfg_.k), 1);
    const uint64_t mt = ceilDivU64_(cfg_.m, tm);
    const uint64_t nt = ceilDivU64_(cfg_.n, tn);
    const uint64_t kt = ceilDivU64_(cfg_.k, tk);
    tile_count_per_iter_ = mt * nt * kt;

    const uint64_t a_tile = tm * tk * static_cast<uint64_t>(cfg_.element_bytes);
    const uint64_t b_tile = tk * tn * static_cast<uint64_t>(cfg_.element_bytes);
    const uint64_t c_tile = tm * tn * static_cast<uint64_t>(cfg_.element_bytes);

    const bool dataflow_is = (cfg_.dataflow == "is");
    const bool dataflow_ws = (cfg_.dataflow == "ws");
    const bool dataflow_os = (!dataflow_is && !dataflow_ws);

    // Persist tile-derived params for tile exec mode.
    tile_tm_ = static_cast<uint32_t>(tm);
    tile_tn_ = static_cast<uint32_t>(tn);
    tile_tk_ = static_cast<uint32_t>(tk);
    tile_mt_ = static_cast<uint32_t>(mt);
    tile_nt_ = static_cast<uint32_t>(nt);
    tile_kt_ = static_cast<uint32_t>(kt);
    tile_a_bytes_ = a_tile;
    tile_b_bytes_ = b_tile;
    tile_c_bytes_ = c_tile;

    tile_schedule_eff_ = cfg_.tile_schedule;
    if (tile_schedule_eff_.empty()) tile_schedule_eff_ = "auto";
    if (cfg_.exec_mode == "tile" && tile_schedule_eff_ == "auto") {
        if (dataflow_is) {
            tile_schedule_eff_ = "mkn";
        } else if (dataflow_ws) {
            tile_schedule_eff_ = "nkm";
        } else {
            tile_schedule_eff_ = "mnk";
        }
    } else if (tile_schedule_eff_ == "auto") {
        tile_schedule_eff_ = "mnk";
    }

    const bool can_keep_a = (cfg_.ub_bytes > 0 && cfg_.ub_bytes >= a_tile);
    const bool can_keep_b = (cfg_.ub_bytes > 0 && cfg_.ub_bytes >= b_tile);
    tile_keep_a_ = (cfg_.exec_mode == "tile" && dataflow_is && can_keep_a && tile_schedule_eff_ == "mkn");
    tile_keep_b_ = (cfg_.exec_mode == "tile" && dataflow_ws && can_keep_b && tile_schedule_eff_ == "nkm");
    const bool keep_a = (cfg_.exec_mode == "tile") ? tile_keep_a_ : can_keep_a;
    const bool keep_b = (cfg_.exec_mode == "tile") ? tile_keep_b_ : can_keep_b;

    uint64_t total_a = 0;
    uint64_t total_b = 0;
    if (dataflow_is) {
        const uint64_t a_factor = keep_a ? 1 : nt;
        total_a = mt * kt * a_tile * a_factor;
        total_b = mt * nt * kt * b_tile;
    } else if (dataflow_ws) {
        const uint64_t b_factor = keep_b ? 1 : mt;
        total_a = mt * nt * kt * a_tile;
        total_b = kt * nt * b_tile * b_factor;
    } else if (dataflow_os) {
        total_a = mt * nt * kt * a_tile;
        total_b = mt * nt * kt * b_tile;
    }

    const uint64_t total_c = mt * nt * c_tile;
    bytes_read_per_iter_ = total_a + total_b;
    bytes_write_per_iter_ = total_c;
    dram_bytes_per_iter_ = bytes_read_per_iter_ + bytes_write_per_iter_;
    onchip_bytes_per_iter_ = bytes_read_per_iter_ + bytes_write_per_iter_;
    if (cfg_.dma_bandwidth_bytes_per_cycle > 0) {
        dma_cycles_per_iter_ = ceilDivU64_(dram_bytes_per_iter_, cfg_.dma_bandwidth_bytes_per_cycle);
    } else {
        dma_cycles_per_iter_ = 0;
    }
    if (tile_count_per_iter_ > 0) {
        tile_seg_cycles_base_ = compute_cycles_per_iter_ / tile_count_per_iter_;
        tile_seg_cycles_remainder_ = compute_cycles_per_iter_ % tile_count_per_iter_;
    } else {
        tile_seg_cycles_base_ = 0;
        tile_seg_cycles_remainder_ = 0;
    }

    configured_ = true;
}

void TensorWorkload::bindRuntime(const Runtime& rt) {
    rt_ = rt;
    if (inflight_.size() > static_cast<size_t>(cfg_.mem_max_outstanding) * 8u) {
        inflight_.clear();
    }
}

void TensorWorkload::onGlobalStepStart(uint32_t seq) {
    (void)seq;
    step_gated_ = true;
    step_open_ = true;
    step_seq_ = seq;
}

void TensorWorkload::startIteration_() {
    iter_active_ = true;
    compute_started_ = (cfg_.exec_mode == "bulk") ? (cfg_.overlap_enable || cfg_.double_buffer) : false;
    rem_read_bytes_ = (cfg_.mem_enable && rt_.mem) ? bytes_read_per_iter_ : 0;
    rem_write_bytes_ = (cfg_.mem_enable && rt_.mem) ? bytes_write_per_iter_ : 0;
    rem_compute_cycles_ = compute_cycles_per_iter_;
    rem_macs_ = mac_ops_per_iter_;

    // Deterministic address offsets per-iteration (avoid trivially reusing the same cache lines).
    const uint64_t region = clampNonZero_(cfg_.mem_region_bytes, 4096);
    uint64_t rng = cfg_.seed_base ^
                   (static_cast<uint64_t>(rt_.node_id) << 32) ^
                   (static_cast<uint64_t>(rt_.core_id) << 16) ^
                   static_cast<uint64_t>(iter_seq_) ^
                   (static_cast<uint64_t>(step_seq_) << 48);
    read_off_ = (region > 0) ? (splitmix64_next_(rng) % region) : 0;
    write_off_ = (region > 0) ? ((read_off_ + (region / 2u)) % region) : 0;

    tensor_tile_count_total_ += tile_count_per_iter_;
    tensor_dram_bytes_total_ += dram_bytes_per_iter_;
    tensor_onchip_bytes_total_ += onchip_bytes_per_iter_;
    tensor_dma_cycles_total_ += dma_cycles_per_iter_;

    tile_mode_active_ = (cfg_.exec_mode == "tile");
    if (tile_mode_active_) {
        resetTileIteration_();
    }

    iter_seq_++;
}

void TensorWorkload::resetTileIteration_() {
    tile_mode_active_ = true;

    tile_seg_gen_index_ = 0;
    tile_gen_mi_ = 0;
    tile_gen_ni_ = 0;
    tile_gen_ki_ = 0;
    tile_gen_done_ = false;
    tile_seg_done_ = 0;
    tile_cur_ = TileSegState{};
    tile_next_ = TileSegState{};
    tile_writeback_queue_.clear();
    tile_writebacks_.clear();

    (void)generateNextTileSeg_(tile_cur_);
    const bool prefetch = (cfg_.overlap_enable || cfg_.double_buffer);
    if (prefetch) {
        (void)generateNextTileSeg_(tile_next_);
    }
}

bool TensorWorkload::advanceTileIndices_(uint32_t& mi, uint32_t& ni, uint32_t& ki) const {
    if (tile_schedule_eff_ == "mkn") {
        // m outer, k middle, n inner
        ni++;
        if (ni < tile_nt_) return true;
        ni = 0;
        ki++;
        if (ki < tile_kt_) return true;
        ki = 0;
        mi++;
        return (mi < tile_mt_);
    }
    if (tile_schedule_eff_ == "nkm") {
        // n outer, k middle, m inner
        mi++;
        if (mi < tile_mt_) return true;
        mi = 0;
        ki++;
        if (ki < tile_kt_) return true;
        ki = 0;
        ni++;
        return (ni < tile_nt_);
    }
    // default: mnk (m outer, n middle, k inner)
    ki++;
    if (ki < tile_kt_) return true;
    ki = 0;
    ni++;
    if (ni < tile_nt_) return true;
    ni = 0;
    mi++;
    return (mi < tile_mt_);
}

uint64_t TensorWorkload::tileSegComputeCycles_(uint64_t seg_index) const {
    const uint64_t extra = (seg_index < tile_seg_cycles_remainder_) ? 1ull : 0ull;
    return tile_seg_cycles_base_ + extra;
}

uint64_t TensorWorkload::tileNeedReadABytes_(uint32_t /*mi*/, uint32_t ni, uint32_t /*ki*/) const {
    if (!cfg_.mem_enable || !rt_.mem) return 0;
    if (cfg_.dataflow == "is" && tile_keep_a_) {
        return (ni == 0) ? tile_a_bytes_ : 0;
    }
    return tile_a_bytes_;
}

uint64_t TensorWorkload::tileNeedReadBBytes_(uint32_t mi, uint32_t /*ni*/, uint32_t /*ki*/) const {
    if (!cfg_.mem_enable || !rt_.mem) return 0;
    if (cfg_.dataflow == "ws" && tile_keep_b_) {
        return (mi == 0) ? tile_b_bytes_ : 0;
    }
    return tile_b_bytes_;
}

bool TensorWorkload::generateNextTileSeg_(TileSegState& out) {
    if (tile_gen_done_) {
        out = TileSegState{};
        return false;
    }
    if (tile_seg_gen_index_ >= tile_count_per_iter_) {
        tile_gen_done_ = true;
        out = TileSegState{};
        return false;
    }

    out = TileSegState{};
    out.valid = true;
    out.seg_index = tile_seg_gen_index_;
    out.epoch = out.seg_index + 1ull;
    out.mi = tile_gen_mi_;
    out.ni = tile_gen_ni_;
    out.ki = tile_gen_ki_;
    out.need_a_bytes = tileNeedReadABytes_(out.mi, out.ni, out.ki);
    out.need_b_bytes = tileNeedReadBBytes_(out.mi, out.ni, out.ki);
    out.rem_compute_cycles = tileSegComputeCycles_(out.seg_index);

    tile_seg_gen_index_++;
    const bool ok = advanceTileIndices_(tile_gen_mi_, tile_gen_ni_, tile_gen_ki_);
    if (!ok) {
        tile_gen_done_ = true;
    }
    return true;
}

void TensorWorkload::scheduleWriteback_(uint32_t mi, uint32_t ni) {
    if (!cfg_.mem_enable || !rt_.mem) return;
    if (tile_c_bytes_ == 0) return;
    const uint64_t epoch = (1ull << 63) | (static_cast<uint64_t>(mi) << 32) | static_cast<uint64_t>(ni);
    if (tile_writebacks_.find(epoch) != tile_writebacks_.end()) return;

    WritebackState st;
    st.epoch = epoch;
    st.mi = mi;
    st.ni = ni;
    st.total_bytes = tile_c_bytes_;
    st.issued_bytes = 0;
    st.done_bytes = 0;
    tile_writebacks_.emplace(epoch, st);
    tile_writeback_queue_.push_back(epoch);
}

void TensorWorkload::retireCompletedWritebacks_() {
    while (!tile_writeback_queue_.empty()) {
        const uint64_t epoch = tile_writeback_queue_.front();
        auto it = tile_writebacks_.find(epoch);
        if (it == tile_writebacks_.end()) {
            tile_writeback_queue_.pop_front();
            continue;
        }
        if (it->second.done_bytes < it->second.total_bytes) break;
        tile_writebacks_.erase(it);
        tile_writeback_queue_.pop_front();
    }
}

uint32_t TensorWorkload::issueMemReadTagged_(uint32_t max_bytes, MemTag tag, uint64_t epoch) {
    if (!cfg_.mem_enable) return 0;
    if (!rt_.mem) {
        if (cfg_.strict && rt_.log) {
            rt_.log->fatal(CALL_INFO, -1, "tensor fatal: mem_enable=1 but IMemoryAccess is null (core=%u)\n", rt_.core_id);
        }
        return 0;
    }
    if (max_bytes == 0) return 0;
    if (inflight_.size() >= static_cast<size_t>(cfg_.mem_max_outstanding)) return 0;

    const uint32_t bytes = static_cast<uint32_t>(std::min<uint32_t>(max_bytes, cfg_.mem_req_bytes));
    const uint64_t region = clampNonZero_(cfg_.mem_region_bytes, 4096);
    uint64_t off = read_off_;
    if (off + bytes > region) off = 0;
    const uint64_t addr = rt_.base_addr + off;
    read_off_ = off + bytes;
    if (read_off_ >= region) read_off_ = 0;

    if (rt_.reporting.report_mem_issue) {
        rt_.reporting.report_mem_issue(rt_.reporting.ctx, bytes);
    }
    memory_requests_++;
    tensor_mem_reads_issued_total_ += 1;
    tensor_mem_bytes_read_total_ += static_cast<uint64_t>(bytes);

    const auto req_id = rt_.mem->read(
        addr, bytes,
        [this, addr, bytes, tag, epoch](IMemoryAccess::RequestId cb_id, uint64_t /*addr_cb*/, std::vector<uint8_t>&& got) {
            if (cb_id != 0) {
                inflight_.erase(static_cast<uint64_t>(cb_id));
            }
            if (cb_id == 0 || got.size() != bytes) {
                if (cfg_.strict && rt_.log) {
                    rt_.log->fatal(
                        CALL_INFO, -1,
                        "tensor fatal: read failed (core=%u addr=0x%llx bytes=%u got=%zu)\n",
                        rt_.core_id, (unsigned long long)addr, bytes, got.size());
                }
                return;
            }
            onMemComplete_(ReqKind::Read, tag, epoch, bytes);
        });
    if (req_id != 0) {
        inflight_[static_cast<uint64_t>(req_id)] = InflightReq{ReqKind::Read, tag, epoch, bytes};
    } else if (cfg_.strict && rt_.log) {
        rt_.log->fatal(
            CALL_INFO, -1,
            "tensor fatal: read issue failed (core=%u addr=0x%llx bytes=%u)\n",
            rt_.core_id, (unsigned long long)addr, bytes);
    }

    return bytes;
}

uint32_t TensorWorkload::issueMemWriteTagged_(uint32_t max_bytes, MemTag tag, uint64_t epoch) {
    if (!cfg_.mem_enable) return 0;
    if (!rt_.mem) {
        if (cfg_.strict && rt_.log) {
            rt_.log->fatal(CALL_INFO, -1, "tensor fatal: mem_enable=1 but IMemoryAccess is null (core=%u)\n", rt_.core_id);
        }
        return 0;
    }
    if (max_bytes == 0) return 0;
    if (inflight_.size() >= static_cast<size_t>(cfg_.mem_max_outstanding)) return 0;

    const uint32_t bytes = static_cast<uint32_t>(std::min<uint32_t>(max_bytes, cfg_.mem_req_bytes));
    const uint64_t region = clampNonZero_(cfg_.mem_region_bytes, 4096);
    uint64_t off = write_off_;
    if (off + bytes > region) off = 0;
    const uint64_t addr = rt_.base_addr + off;
    write_off_ = off + bytes;
    if (write_off_ >= region) write_off_ = 0;

    std::vector<uint8_t> data;
    data.resize(bytes);
    fillBytesDeterministic_(cfg_.seed_base ^
                                (static_cast<uint64_t>(rt_.node_id) << 32) ^
                                (static_cast<uint64_t>(rt_.core_id) << 16),
                            addr, static_cast<uint32_t>(epoch), data);

    if (rt_.reporting.report_mem_issue) {
        rt_.reporting.report_mem_issue(rt_.reporting.ctx, bytes);
    }
    memory_requests_++;
    tensor_mem_writes_issued_total_ += 1;
    tensor_mem_bytes_write_total_ += static_cast<uint64_t>(bytes);

    const auto req_id = rt_.mem->write(
        addr, data,
        [this, addr, bytes, tag, epoch](IMemoryAccess::RequestId cb_id, uint64_t /*addr_cb*/) {
            if (cb_id != 0) {
                inflight_.erase(static_cast<uint64_t>(cb_id));
            }
            if (cb_id == 0) {
                if (cfg_.strict && rt_.log) {
                    rt_.log->fatal(
                        CALL_INFO, -1,
                        "tensor fatal: write failed (core=%u addr=0x%llx bytes=%u)\n",
                        rt_.core_id, (unsigned long long)addr, bytes);
                }
                return;
            }
            onMemComplete_(ReqKind::Write, tag, epoch, bytes);
        });
    if (req_id != 0) {
        inflight_[static_cast<uint64_t>(req_id)] = InflightReq{ReqKind::Write, tag, epoch, bytes};
    } else if (cfg_.strict && rt_.log) {
        rt_.log->fatal(
            CALL_INFO, -1,
            "tensor fatal: write issue failed (core=%u addr=0x%llx bytes=%u)\n",
            rt_.core_id, (unsigned long long)addr, bytes);
    }

    return bytes;
}

void TensorWorkload::onMemComplete_(ReqKind kind, MemTag tag, uint64_t epoch, uint32_t bytes) {
    if (!tile_mode_active_) return;

    if (kind == ReqKind::Read) {
        if (tag == MemTag::ReadA) {
            if (tile_cur_.valid && tile_cur_.epoch == epoch) {
                tile_cur_.done_a_bytes += static_cast<uint64_t>(bytes);
                return;
            }
            if (tile_next_.valid && tile_next_.epoch == epoch) {
                tile_next_.done_a_bytes += static_cast<uint64_t>(bytes);
                return;
            }
        } else if (tag == MemTag::ReadB) {
            if (tile_cur_.valid && tile_cur_.epoch == epoch) {
                tile_cur_.done_b_bytes += static_cast<uint64_t>(bytes);
                return;
            }
            if (tile_next_.valid && tile_next_.epoch == epoch) {
                tile_next_.done_b_bytes += static_cast<uint64_t>(bytes);
                return;
            }
        }
        return;
    }

    if (kind == ReqKind::Write && tag == MemTag::WriteC) {
        auto it = tile_writebacks_.find(epoch);
        if (it != tile_writebacks_.end()) {
            it->second.done_bytes += static_cast<uint64_t>(bytes);
        }
        return;
    }
}

uint32_t TensorWorkload::issueMemRead_() {
    if (!cfg_.mem_enable) return 0;
    if (rem_read_bytes_ == 0) return 0;
    const uint32_t want = static_cast<uint32_t>(std::min<uint64_t>(cfg_.mem_req_bytes, rem_read_bytes_));
    const uint32_t issued = issueMemReadTagged_(want, MemTag::Generic, /*epoch*/0);
    if (issued == 0) return 0;
    rem_read_bytes_ -= issued;
    return issued;
}

uint32_t TensorWorkload::issueMemWrite_() {
    if (!cfg_.mem_enable) return 0;
    if (rem_write_bytes_ == 0) return 0;
    const uint32_t want = static_cast<uint32_t>(std::min<uint64_t>(cfg_.mem_req_bytes, rem_write_bytes_));
    const uint32_t issued = issueMemWriteTagged_(want, MemTag::Generic, /*epoch*/0);
    if (issued == 0) return 0;
    rem_write_bytes_ -= issued;
    return issued;
}

bool TensorWorkload::tickCompute_() {
    if (!compute_started_) return false;
    if (rem_compute_cycles_ == 0) return false;

    rem_compute_cycles_--;
    tensor_compute_cycles_total_ += 1;

    const uint64_t per = effectivePeakMacsPerCycle_();
    const uint64_t done = (rem_macs_ >= per) ? per : rem_macs_;
    tensor_mac_ops_total_ += done;
    rem_macs_ = (rem_macs_ >= done) ? (rem_macs_ - done) : 0;

    return true;
}

bool TensorWorkload::emitCollectiveTraffic_(uint64_t now_cycle) {
    if (!collectiveReady_()) {
        if (cfg_.collective_type != "none" && cfg_.strict && !rt_.noc && rt_.log) {
            rt_.log->fatal(CALL_INFO, -1, "tensor fatal: collective enabled but INocTransport is null (core=%u)\n", rt_.core_id);
        }
        return false;
    }

    if (cfg_.collective_blocking && collective_epoch_active_) {
        if (cfg_.collective_scope == "per_core") {
            if (collective_epoch_recv_bytes_ < collective_epoch_expected_recv_bytes_) return false;
            // Completed: allow starting a new epoch.
            const uint32_t done_seq = collective_epoch_seq_;
            collective_epoch_active_ = false;
            collective_epoch_expected_recv_bytes_ = 0;
            collective_epoch_recv_bytes_ = 0;
            collective_recv_bytes_by_seq_.erase(done_seq);
        } else {
            // Group-scoped blocking collective: wait for explicit RELEASE (leader/root broadcast).
            return false;
        }
    }

    if (now_cycle - collective_last_cycle_ < cfg_.collective_period_cycles) return false;

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
    const uint64_t dest_count = static_cast<uint64_t>(dest_nodes.size());
    if (dest_count == 0 || total_bytes == 0) return false;

    const uint64_t base_bytes = total_bytes / dest_count;
    const uint64_t remainder = total_bytes % dest_count;
    const uint32_t packet_bytes = cfg_.collective_packet_bytes ? cfg_.collective_packet_bytes : 256u;
    const uint32_t seq32 = static_cast<uint32_t>(collective_seq_ & 0xffffffffu);

    bool did = false;
    for (size_t dest_index = 0; dest_index < dest_nodes.size(); ++dest_index) {
        const uint32_t dest_node = dest_nodes[dest_index];
        uint64_t bytes_for_dest = base_bytes + ((dest_index < remainder) ? 1u : 0u);
        while (bytes_for_dest > 0) {
            const uint32_t payload_bytes = static_cast<uint32_t>(std::min<uint64_t>(packet_bytes, bytes_for_dest));
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
                                    0, static_cast<uint32_t>(collective_seq_), pkt->payload);
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
            bytes_for_dest -= payload_bytes;
            did = true;
        }
    }

    if (did) {
        collective_last_cycle_ = now_cycle;
        tensor_collective_cycles_total_ += 1;
        if (cfg_.collective_blocking) {
            collective_epoch_active_ = true;
            collective_epoch_seq_ = seq32;
            collective_epoch_expected_recv_bytes_ = total_bytes;
            collective_epoch_recv_bytes_ = collectiveRecvBytesForSeq_(seq32);
            collective_done_notified_ = false;
        }
        if (!cfg_.collective_blocking || cfg_.collective_scope == "per_core") {
            collective_seq_ += 1;
        }
    }
    return did;
}

uint64_t TensorWorkload::collectiveRecvBytesForSeq_(uint32_t seq) const {
    auto it = collective_recv_bytes_by_seq_.find(seq);
    if (it == collective_recv_bytes_by_seq_.end()) return 0;
    return it->second;
}

void TensorWorkload::serviceCollectiveBarrier_(uint64_t now_cycle) {
    if (!cfg_.collective_blocking) return;
    if (cfg_.collective_scope == "per_core") return;
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
        onCollectiveRelease_(seq, next);
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
        onCollectiveRelease_(seq, next);
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
        onCollectiveRelease_(seq, next);
        return;
    }
}

void TensorWorkload::onCollectiveRelease_(uint32_t seq, uint32_t next_seq) {
    if (!cfg_.collective_blocking) return;
    if (cfg_.collective_scope == "per_core") return;

    const uint32_t cur_seq = static_cast<uint32_t>(collective_seq_ & 0xffffffffu);
    if (seq != cur_seq) return;

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
        }
    } else if (kind == NocPacketKind::Control) {
        onCollectiveControlPacket_(packet->payload, packet->timestamp);
    }
    delete packet;
    return true;
}

bool TensorWorkload::onClockTickTile_(uint64_t now_cycle) {
    // Start iteration when ready (and not blocked by a pending blocking-collective epoch).
    if (!iter_active_) {
        if (cfg_.iterations > 0 && iter_done_ >= cfg_.iterations) return false;

        if (cfg_.collective_blocking && collective_epoch_active_) {
            if (cfg_.collective_scope == "per_core") {
                if (collective_epoch_recv_bytes_ < collective_epoch_expected_recv_bytes_) {
                    tensor_stall_collective_cycles_total_ += 1;
                    if (!inflight_.empty()) {
                        // Still allow progress due to in-flight completions; this tick itself does not do work.
                    }
                    return false;
                }
                // Epoch completed: clear barrier.
                const uint32_t done_seq = collective_epoch_seq_;
                collective_epoch_active_ = false;
                collective_epoch_expected_recv_bytes_ = 0;
                collective_epoch_recv_bytes_ = 0;
                collective_recv_bytes_by_seq_.erase(done_seq);
            } else {
                // Group scope: explicit RELEASE drives epoch completion.
                tensor_stall_collective_cycles_total_ += 1;
                return false;
            }
        }

        if (inflight_.empty()) {
            startIteration_();
        }
    }

    bool did = false;

    // Optional RawBytes NoC traffic (only when step is active)
    if (iter_active_ && commReady_()) {
        if (now_cycle - comm_last_cycle_ >= cfg_.comm_period_cycles) {
            const uint32_t dst_node =
                (rt_.total_nodes > 1) ? ((rt_.node_id + 1u) % rt_.total_nodes) : rt_.node_id;
            auto* pkt = new NocPacketEvent(rt_.node_id,
                                           dst_node,
                                           static_cast<uint16_t>(rt_.core_id),
                                           static_cast<uint16_t>(rt_.core_id),
                                           NocPacketKind::RawBytes,
                                           /*ts*/now_cycle);
            pkt->payload.resize(cfg_.comm_payload_bytes);
            fillBytesDeterministic_(cfg_.seed_base ^
                                        (static_cast<uint64_t>(rt_.node_id) << 32) ^
                                        (static_cast<uint64_t>(rt_.core_id) << 16),
                                    /*addr*/0, iter_seq_, pkt->payload);
            tensor_pkt_sent_total_ += 1;
            tensor_pkt_bytes_sent_total_ += static_cast<uint64_t>(pkt->payload.size());
            if (dst_node == rt_.node_id) {
                rt_.noc->injectLocal(static_cast<int>(rt_.core_id), pkt);
            } else {
                rt_.noc->sendFromCore(static_cast<int>(rt_.core_id), pkt);
            }
            comm_last_cycle_ = now_cycle;
            did = true;
        }
    } else if (cfg_.comm_enable && cfg_.strict && !rt_.noc && rt_.log) {
        rt_.log->fatal(CALL_INFO, -1, "tensor fatal: comm_enable=1 but INocTransport is null (core=%u)\n", rt_.core_id);
    }

    if (iter_active_ && emitCollectiveTraffic_(now_cycle)) {
        did = true;
    }

    if (!iter_active_) {
        if (did) active_cycles_++;
        return did;
    }

    tensor_iter_cycles_total_ += 1;

    // DMA budgeting (bytes/cycle); if disabled, issue as much as possible until outstanding limit.
    uint64_t budget = (cfg_.dma_bandwidth_bytes_per_cycle > 0) ? cfg_.dma_bandwidth_bytes_per_cycle
                                                               : std::numeric_limits<uint64_t>::max();
    bool blocked_budget = false;
    bool blocked_outstanding = false;

    const auto max_out = static_cast<size_t>(cfg_.mem_max_outstanding);
    const bool prefetch = (cfg_.overlap_enable || cfg_.double_buffer);

    retireCompletedWritebacks_();

    // === Mandatory reads for current tile-seg ===
    if (cfg_.mem_enable && rt_.mem && tile_cur_.valid) {
        while (budget > 0) {
            uint64_t pending = 0;
            MemTag tag = MemTag::Generic;
            if (tile_cur_.issued_a_bytes < tile_cur_.need_a_bytes) {
                pending = tile_cur_.need_a_bytes - tile_cur_.issued_a_bytes;
                tag = MemTag::ReadA;
            } else if (tile_cur_.issued_b_bytes < tile_cur_.need_b_bytes) {
                pending = tile_cur_.need_b_bytes - tile_cur_.issued_b_bytes;
                tag = MemTag::ReadB;
            } else {
                break;
            }

            if (inflight_.size() >= max_out) {
                blocked_outstanding = true;
                break;
            }

            const uint32_t chunk = static_cast<uint32_t>(std::min<uint64_t>(cfg_.mem_req_bytes, pending));
            if (chunk == 0) break;
            if (chunk > budget) {
                blocked_budget = true;
                break;
            }
            const uint32_t issued = issueMemReadTagged_(chunk, tag, tile_cur_.epoch);
            if (issued == 0) {
                if (inflight_.size() >= max_out) blocked_outstanding = true;
                break;
            }
            if (tag == MemTag::ReadA) {
                tile_cur_.issued_a_bytes += issued;
            } else if (tag == MemTag::ReadB) {
                tile_cur_.issued_b_bytes += issued;
            }
            rem_read_bytes_ = (rem_read_bytes_ >= issued) ? (rem_read_bytes_ - issued) : 0;
            budget -= issued;
            did = true;
        }
    }

    // === Writebacks (oldest-first) ===
    if (cfg_.mem_enable && rt_.mem) {
        while (budget > 0 && !tile_writeback_queue_.empty()) {
            const uint64_t epoch = tile_writeback_queue_.front();
            auto it = tile_writebacks_.find(epoch);
            if (it == tile_writebacks_.end()) {
                tile_writeback_queue_.pop_front();
                continue;
            }
            WritebackState& st = it->second;
            if (st.issued_bytes >= st.total_bytes) {
                // Issued all; wait for completions.
                break;
            }
            if (inflight_.size() >= max_out) {
                blocked_outstanding = true;
                break;
            }
            const uint64_t pending = st.total_bytes - st.issued_bytes;
            const uint32_t chunk = static_cast<uint32_t>(std::min<uint64_t>(cfg_.mem_req_bytes, pending));
            if (chunk == 0) break;
            if (chunk > budget) {
                blocked_budget = true;
                break;
            }
            const uint32_t issued = issueMemWriteTagged_(chunk, MemTag::WriteC, epoch);
            if (issued == 0) {
                if (inflight_.size() >= max_out) blocked_outstanding = true;
                break;
            }
            st.issued_bytes += issued;
            rem_write_bytes_ = (rem_write_bytes_ >= issued) ? (rem_write_bytes_ - issued) : 0;
            budget -= issued;
            did = true;
        }
    }

    // === Prefetch reads for next tile-seg (optional) ===
    if (prefetch && cfg_.mem_enable && rt_.mem && tile_next_.valid) {
        while (budget > 0) {
            uint64_t pending = 0;
            MemTag tag = MemTag::Generic;
            if (tile_next_.issued_a_bytes < tile_next_.need_a_bytes) {
                pending = tile_next_.need_a_bytes - tile_next_.issued_a_bytes;
                tag = MemTag::ReadA;
            } else if (tile_next_.issued_b_bytes < tile_next_.need_b_bytes) {
                pending = tile_next_.need_b_bytes - tile_next_.issued_b_bytes;
                tag = MemTag::ReadB;
            } else {
                break;
            }

            if (inflight_.size() >= max_out) {
                blocked_outstanding = true;
                break;
            }

            const uint32_t chunk = static_cast<uint32_t>(std::min<uint64_t>(cfg_.mem_req_bytes, pending));
            if (chunk == 0) break;
            if (chunk > budget) {
                blocked_budget = true;
                break;
            }
            const uint32_t issued = issueMemReadTagged_(chunk, tag, tile_next_.epoch);
            if (issued == 0) {
                if (inflight_.size() >= max_out) blocked_outstanding = true;
                break;
            }
            if (tag == MemTag::ReadA) {
                tile_next_.issued_a_bytes += issued;
            } else if (tag == MemTag::ReadB) {
                tile_next_.issued_b_bytes += issued;
            }
            rem_read_bytes_ = (rem_read_bytes_ >= issued) ? (rem_read_bytes_ - issued) : 0;
            budget -= issued;
            did = true;
        }
    }

    // === Compute for current tile-seg (requires reads ready) ===
    const bool cur_reads_ready =
        (!tile_cur_.valid) ||
        (tile_cur_.done_a_bytes >= tile_cur_.need_a_bytes && tile_cur_.done_b_bytes >= tile_cur_.need_b_bytes);
    if (tile_cur_.valid && cur_reads_ready && tile_cur_.rem_compute_cycles > 0) {
        tile_cur_.rem_compute_cycles--;
        if (rem_compute_cycles_ > 0) rem_compute_cycles_--;
        tensor_compute_cycles_total_ += 1;

        const uint64_t per = effectivePeakMacsPerCycle_();
        const uint64_t done = (rem_macs_ >= per) ? per : rem_macs_;
        tensor_mac_ops_total_ += done;
        rem_macs_ = (rem_macs_ >= done) ? (rem_macs_ - done) : 0;

        did = true;
    }

    // === Tile-seg complete → advance ===
    if (tile_cur_.valid && cur_reads_ready && tile_cur_.rem_compute_cycles == 0) {
        tile_seg_done_ += 1;
        if (tile_cur_.ki + 1u == tile_kt_) {
            scheduleWriteback_(tile_cur_.mi, tile_cur_.ni);
        }

        // Advance current seg.
        if (tile_next_.valid) {
            tile_cur_ = tile_next_;
            tile_next_ = TileSegState{};
        } else {
            tile_cur_ = TileSegState{};
            (void)generateNextTileSeg_(tile_cur_);
        }
        // Maintain a single lookahead seg for prefetch.
        if (prefetch && !tile_next_.valid) {
            (void)generateNextTileSeg_(tile_next_);
        }
        did = true;
    }

    retireCompletedWritebacks_();

    // === Iteration complete ===
    if (iter_active_ &&
        tile_seg_done_ >= tile_count_per_iter_ &&
        rem_read_bytes_ == 0 &&
        rem_write_bytes_ == 0 &&
        rem_compute_cycles_ == 0 &&
        tile_writeback_queue_.empty() &&
        inflight_.empty()) {
        iter_done_++;
        iter_active_ = false;
        compute_started_ = false;
        tile_mode_active_ = false;
        tile_cur_ = TileSegState{};
        tile_next_ = TileSegState{};
        tile_writeback_queue_.clear();
        tile_writebacks_.clear();
        if (step_gated_) {
            step_open_ = false;
        }
    }

    if (!did && (iter_active_ || !inflight_.empty())) {
        tensor_dma_stall_cycles_total_ += 1;
        const bool pending_mem_issue =
            (tile_cur_.valid && ((tile_cur_.issued_a_bytes < tile_cur_.need_a_bytes) || (tile_cur_.issued_b_bytes < tile_cur_.need_b_bytes))) ||
            (prefetch && tile_next_.valid &&
             ((tile_next_.issued_a_bytes < tile_next_.need_a_bytes) || (tile_next_.issued_b_bytes < tile_next_.need_b_bytes))) ||
            (!tile_writeback_queue_.empty() &&
             ([&]() -> bool {
                 const uint64_t epoch = tile_writeback_queue_.front();
                 auto it = tile_writebacks_.find(epoch);
                 return it != tile_writebacks_.end() && it->second.issued_bytes < it->second.total_bytes;
             })());

        if (blocked_outstanding && pending_mem_issue) {
            tensor_stall_mem_outstanding_cycles_total_ += 1;
        } else if (blocked_budget && pending_mem_issue) {
            tensor_stall_dma_budget_cycles_total_ += 1;
        } else if (tile_cur_.valid && !cur_reads_ready && tile_cur_.rem_compute_cycles > 0) {
            tensor_stall_wait_read_cycles_total_ += 1;
        } else if (rem_compute_cycles_ == 0 && (!tile_writeback_queue_.empty() || !inflight_.empty())) {
            tensor_stall_wait_write_cycles_total_ += 1;
        }
    }

    if (did) active_cycles_++;
    return did;
}

bool TensorWorkload::onClockTick(uint64_t now_cycle) {
    total_cycles_++;
    if (!configured_) return false;

    // Step-gated mode: only run when a step is open.
    if (step_gated_ && !step_open_) return false;

    if (!started_) {
        if (now_cycle < cfg_.start_cycle) return false;
        started_ = true;
    }

    serviceCollectiveBarrier_(now_cycle);

    if (cfg_.exec_mode == "tile") {
        return onClockTickTile_(now_cycle);
    }

    if (!iter_active_) {
        if (cfg_.iterations > 0 && iter_done_ >= cfg_.iterations) return false;
        if (cfg_.collective_blocking && collective_epoch_active_) {
            if (cfg_.collective_scope == "per_core") {
                if (collective_epoch_recv_bytes_ < collective_epoch_expected_recv_bytes_) {
                    tensor_stall_collective_cycles_total_ += 1;
                    return false;
                }
                const uint32_t done_seq = collective_epoch_seq_;
                collective_epoch_active_ = false;
                collective_epoch_expected_recv_bytes_ = 0;
                collective_epoch_recv_bytes_ = 0;
                collective_recv_bytes_by_seq_.erase(done_seq);
            } else {
                tensor_stall_collective_cycles_total_ += 1;
                return false;
            }
        }
        if (inflight_.empty()) {
            startIteration_();
        }
    }

    bool did = false;

    // === DMA reads ===
    if (tileModelEnabled_() && cfg_.dma_bandwidth_bytes_per_cycle > 0) {
        uint64_t budget = cfg_.dma_bandwidth_bytes_per_cycle;
        while (rem_read_bytes_ > 0) {
            const uint32_t next_bytes = static_cast<uint32_t>(std::min<uint64_t>(cfg_.mem_req_bytes, rem_read_bytes_));
            if (next_bytes == 0 || next_bytes > budget) break;
            const uint32_t issued = issueMemRead_();
            if (issued == 0) break;
            budget -= issued;
            did = true;
        }
    } else {
        while (issueMemRead_()) {
            did = true;
        }
    }

    // === Compute ===
    if (!compute_started_) {
        // Non-overlap mode: wait for all read requests to complete.
        if (!cfg_.mem_enable || !rt_.mem) {
            compute_started_ = true;
        } else if (rem_read_bytes_ == 0) {
            bool any_read_inflight = false;
            for (const auto& kv : inflight_) {
                if (kv.second.kind == ReqKind::Read) {
                    any_read_inflight = true;
                    break;
                }
            }
            if (!any_read_inflight) compute_started_ = true;
        }
    }
    if (tickCompute_()) {
        did = true;
    }

    // === DMA writes (after compute completes) ===
    if (rem_compute_cycles_ == 0) {
        if (tileModelEnabled_() && cfg_.dma_bandwidth_bytes_per_cycle > 0) {
            uint64_t budget = cfg_.dma_bandwidth_bytes_per_cycle;
            while (rem_write_bytes_ > 0) {
                const uint32_t next_bytes = static_cast<uint32_t>(std::min<uint64_t>(cfg_.mem_req_bytes, rem_write_bytes_));
                if (next_bytes == 0 || next_bytes > budget) break;
                const uint32_t issued = issueMemWrite_();
                if (issued == 0) break;
                budget -= issued;
                did = true;
            }
        } else {
            while (issueMemWrite_()) {
                did = true;
            }
        }
    }

    // === Optional RawBytes NoC traffic (only when step is active) ===
    if (iter_active_ && commReady_()) {
        if (now_cycle - comm_last_cycle_ >= cfg_.comm_period_cycles) {
            const uint32_t dst_node =
                (rt_.total_nodes > 1) ? ((rt_.node_id + 1u) % rt_.total_nodes) : rt_.node_id;
            auto* pkt = new NocPacketEvent(rt_.node_id,
                                           dst_node,
                                           static_cast<uint16_t>(rt_.core_id),
                                           static_cast<uint16_t>(rt_.core_id),
                                           NocPacketKind::RawBytes,
                                           /*ts*/now_cycle);
            pkt->payload.resize(cfg_.comm_payload_bytes);
            fillBytesDeterministic_(cfg_.seed_base ^
                                        (static_cast<uint64_t>(rt_.node_id) << 32) ^
                                        (static_cast<uint64_t>(rt_.core_id) << 16),
                                    /*addr*/0, iter_seq_, pkt->payload);
            tensor_pkt_sent_total_ += 1;
            tensor_pkt_bytes_sent_total_ += static_cast<uint64_t>(pkt->payload.size());
            if (dst_node == rt_.node_id) {
                rt_.noc->injectLocal(static_cast<int>(rt_.core_id), pkt);
            } else {
                rt_.noc->sendFromCore(static_cast<int>(rt_.core_id), pkt);
            }
            comm_last_cycle_ = now_cycle;
            did = true;
        }
    } else if (cfg_.comm_enable && cfg_.strict && !rt_.noc && rt_.log) {
        rt_.log->fatal(CALL_INFO, -1, "tensor fatal: comm_enable=1 but INocTransport is null (core=%u)\n", rt_.core_id);
    }

    if (iter_active_ && emitCollectiveTraffic_(now_cycle)) {
        did = true;
    }

    // === Iteration complete ===
    if (iter_active_ &&
        rem_read_bytes_ == 0 &&
        rem_write_bytes_ == 0 &&
        rem_compute_cycles_ == 0 &&
        inflight_.empty()) {
        iter_done_++;
        iter_active_ = false;
        compute_started_ = false;
        if (step_gated_) {
            // In barrier/step-gated runs, execute at most one iteration per step.
            step_open_ = false;
        }
    }

    if (!did && (iter_active_ || !inflight_.empty())) {
        tensor_dma_stall_cycles_total_ += 1;
    }

    if (did) active_cycles_++;
    return did;
}

bool TensorWorkload::hasWork() const {
    if (!configured_) return false;
    // Before start_cycle is reached, still report "work pending" so barrier/drain policies
    // do not accidentally advance global steps without executing the workload.
    if (!started_) return true;
    if (!inflight_.empty()) return true;
    if (iter_active_) return true;
    if (step_gated_) {
        if (!step_open_) return false;
        if (cfg_.iterations > 0 && iter_done_ >= cfg_.iterations) return false;
        return true;
    }
    if (cfg_.iterations > 0 && iter_done_ >= cfg_.iterations) return false;
    return true;
}

double TensorWorkload::getUtilization() const {
    if (total_cycles_ == 0) return 0.0;
    return static_cast<double>(tensor_compute_cycles_total_) / static_cast<double>(total_cycles_);
}

void TensorWorkload::getStatistics(std::map<std::string, uint64_t>& stats) const {
    // Preserve legacy keys expected by MultiCorePE/analysis.
    stats["spikes_received"] = 0;
    stats["spikes_generated"] = 0;
    stats["neurons_fired"] = 0;
    stats["memory_requests"] = memory_requests_;
    stats["total_cycles"] = total_cycles_;
    stats["active_cycles"] = active_cycles_;
    stats["cycles_update_neuron"] = 0;
    stats["synaptic_accesses"] = 0;

    // Tensor workload counters (pulled by MultiCorePE for PE-level aggregation).
    stats["tensor_mem_reads_issued_total"] = tensor_mem_reads_issued_total_;
    stats["tensor_mem_writes_issued_total"] = tensor_mem_writes_issued_total_;
    stats["tensor_mem_bytes_read_total"] = tensor_mem_bytes_read_total_;
    stats["tensor_mem_bytes_write_total"] = tensor_mem_bytes_write_total_;
    stats["tensor_compute_cycles_total"] = tensor_compute_cycles_total_;
    stats["tensor_mac_ops_total"] = tensor_mac_ops_total_;
    stats["tensor_dma_stall_cycles_total"] = tensor_dma_stall_cycles_total_;
    stats["tensor_iter_cycles_total"] = tensor_iter_cycles_total_;
    stats["tensor_stall_dma_budget_cycles_total"] = tensor_stall_dma_budget_cycles_total_;
    stats["tensor_stall_mem_outstanding_cycles_total"] = tensor_stall_mem_outstanding_cycles_total_;
    stats["tensor_stall_wait_read_cycles_total"] = tensor_stall_wait_read_cycles_total_;
    stats["tensor_stall_wait_write_cycles_total"] = tensor_stall_wait_write_cycles_total_;
    stats["tensor_stall_collective_cycles_total"] = tensor_stall_collective_cycles_total_;
    stats["tensor_dma_cycles_total"] = tensor_dma_cycles_total_;
    stats["tensor_dram_bytes_total"] = tensor_dram_bytes_total_;
    stats["tensor_onchip_bytes_total"] = tensor_onchip_bytes_total_;
    stats["tensor_tile_count_total"] = tensor_tile_count_total_;
    stats["tensor_collective_bytes_sent_total"] = tensor_collective_bytes_sent_total_;
    stats["tensor_collective_bytes_recv_total"] = tensor_collective_bytes_recv_total_;
    stats["tensor_collective_pkts_sent_total"] = tensor_collective_pkts_sent_total_;
    stats["tensor_collective_pkts_recv_total"] = tensor_collective_pkts_recv_total_;
    stats["tensor_collective_cycles_total"] = tensor_collective_cycles_total_;
    stats["tensor_pkt_sent_total"] = tensor_pkt_sent_total_;
    stats["tensor_pkt_recv_total"] = tensor_pkt_recv_total_;
    stats["tensor_pkt_bytes_sent_total"] = tensor_pkt_bytes_sent_total_;
    stats["tensor_pkt_bytes_recv_total"] = tensor_pkt_bytes_recv_total_;
}

}} // namespace SST::SnnDL
