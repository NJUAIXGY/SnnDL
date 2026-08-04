// -*- c++ -*-

#include <sst/core/sst_config.h>

#include "workloads/tensor/TensorWorkload.h"

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

struct DmaSharedBudgetState_ {
    uint64_t cycle_tag = 0;
    uint32_t rr_start = 0;
    bool rr_init = false;
};

// Per-rank (per-process) shared DMA budget state keyed by PE/node_id.
// This models a PE-local shared DMA/HBM bandwidth budget; no cross-rank coherence is required.
static std::unordered_map<uint64_t, DmaSharedBudgetState_> g_dma_shared_budget_state_;

struct HbmChannelBudgetState_ {
    uint64_t cycle_tag = 0;
    std::vector<uint64_t> left{};
};

// Per-rank (per-process) shared HBM channel budget state keyed by PE/node_id.
// This models per-channel bandwidth contention inside a PE; no cross-rank coherence is required.
static std::unordered_map<uint64_t, HbmChannelBudgetState_> g_hbm_channel_budget_state_;

inline std::string to_lower_copy_(const std::string& s) {
    std::string out;
    out.reserve(s.size());
    for (char ch : s) out.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    return out;
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

uint64_t TensorWorkload::saturatingAddU64_(uint64_t a, uint64_t b) {
    const uint64_t kMax = std::numeric_limits<uint64_t>::max();
    if (kMax - a < b) return kMax;
    return a + b;
}

uint64_t TensorWorkload::saturatingMulU64_(uint64_t a, uint64_t b) {
    if (a == 0 || b == 0) return 0;
    const uint64_t kMax = std::numeric_limits<uint64_t>::max();
    if (a > (kMax / b)) return kMax;
    return a * b;
}

uint64_t TensorWorkload::saturatingMulU64ByU32_(uint64_t a, uint32_t b) {
    return saturatingMulU64_(a, static_cast<uint64_t>(b));
}

uint32_t TensorWorkload::clampPipelineLatencyCycles_(uint32_t v) {
    constexpr uint32_t kMaxPipelineLatency = 4096u;
    return static_cast<uint32_t>(std::min<uint64_t>(v, kMaxPipelineLatency));
}

TensorWorkload::ComputeProfile TensorWorkload::resolveComputeProfile_(const Config& cfg) {
    ComputeProfile out{};
    std::string precision = to_lower_copy_(cfg.compute_precision);

    if (precision == "bf16") {
        out.profile_id = 1;
        out.throughput_scale = 1.0f;
        out.pipeline_latency_cycles = 0;
    } else if (precision == "fp32") {
        out.profile_id = 2;
        out.throughput_scale = 0.5f;
        out.pipeline_latency_cycles = 2;
    } else if (precision == "tf32") {
        out.profile_id = 3;
        out.throughput_scale = 0.75f;
        out.pipeline_latency_cycles = 1;
    } else if (precision == "int8") {
        out.profile_id = 4;
        out.throughput_scale = 2.0f;
        out.pipeline_latency_cycles = 0;
    } else if (precision == "fp8") {
        out.profile_id = 5;
        out.throughput_scale = 2.0f;
        out.pipeline_latency_cycles = 0;
    } else {
        // default/fallback: fp16
        out.profile_id = 0;
        out.throughput_scale = 1.0f;
        out.pipeline_latency_cycles = 0;
    }

    if (cfg.compute_profile_override_enable) {
        const float override_scale = cfg.compute_throughput_scale;
        out.throughput_scale = (override_scale > 0.0f) ? override_scale : 1.0f;
        out.pipeline_latency_cycles = clampPipelineLatencyCycles_(cfg.compute_pipeline_latency_cycles);
    } else {
        out.pipeline_latency_cycles = clampPipelineLatencyCycles_(out.pipeline_latency_cycles);
    }

    return out;
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

uint64_t TensorWorkload::nowNs_() const {
    if (rt_.time.now_ns) return rt_.time.now_ns(rt_.time.ctx);
    // Fallback: best-effort (legacy behavior assumes 1GHz => 1 cycle == 1ns anyway).
    return now_cycle_cached_;
}


uint64_t TensorWorkload::dmaSharedQuotaBytesPerCycle_(uint64_t now_cycle) const {
    if (cfg_.dma_shared_bandwidth_bytes_per_cycle == 0) {
        return std::numeric_limits<uint64_t>::max();
    }

    const uint32_t cores = total_cores_cfg_ ? total_cores_cfg_ : 1u;
    if (cores <= 1u) {
        return cfg_.dma_shared_bandwidth_bytes_per_cycle;
    }

    const uint64_t key = static_cast<uint64_t>(rt_.node_id);
    DmaSharedBudgetState_& st = g_dma_shared_budget_state_[key];
    if (st.cycle_tag != now_cycle) {
        st.cycle_tag = now_cycle;
        if (!st.rr_init || st.rr_start >= cores) {
            st.rr_start = 0;
            st.rr_init = true;
        } else {
            st.rr_start = (st.rr_start + 1u) % cores;
        }
    } else if (!st.rr_init || st.rr_start >= cores) {
        st.rr_start = 0;
        st.rr_init = true;
    }

    const uint64_t total = cfg_.dma_shared_bandwidth_bytes_per_cycle;
    const uint64_t base = total / static_cast<uint64_t>(cores);
    const uint64_t rem = total % static_cast<uint64_t>(cores);
    const uint32_t cid = static_cast<uint32_t>(rt_.core_id) % cores;
    const uint32_t rr = st.rr_start % cores;
    const uint32_t pos = (cid >= rr) ? (cid - rr) : (cid + cores - rr);
    return base + ((static_cast<uint64_t>(pos) < rem) ? 1ull : 0ull);
}

uint64_t TensorWorkload::dmaBudgetBytesPerCycle_(uint64_t now_cycle) const {
    const uint64_t local =
        (cfg_.dma_bandwidth_bytes_per_cycle > 0)
            ? cfg_.dma_bandwidth_bytes_per_cycle
            : std::numeric_limits<uint64_t>::max();
    const uint64_t shared = dmaSharedQuotaBytesPerCycle_(now_cycle);
    return std::min<uint64_t>(local, shared);
}

bool TensorWorkload::hbmChannelBudgetEnabled_() const {
    return cfg_.dma_hbm_channel_bandwidth_bytes_per_cycle > 0;
}

uint32_t TensorWorkload::hbmChannelCount_() const {
    return std::max<uint32_t>(cfg_.dma_hbm_channels, 1u);
}

bool TensorWorkload::memTimingProxyEnabled_() const {
    return cfg_.mem_enable &&
           (cfg_.mem_timing_model == "proxy_v2" || cfg_.mem_timing_model == "proxy_v3");
}

uint32_t TensorWorkload::memTimingBanksPerChannel_() const {
    const uint64_t groups = static_cast<uint64_t>(std::max<uint32_t>(cfg_.mem_bank_groups_per_channel, 1u));
    const uint64_t per_group = static_cast<uint64_t>(std::max<uint32_t>(cfg_.mem_banks_per_group, 1u));
    const uint64_t banks = groups * per_group;
    return static_cast<uint32_t>(std::max<uint64_t>(1ull, banks));
}

uint64_t TensorWorkload::memTimingServiceQuantumCycles_() const {
    return std::max<uint64_t>(1ull, static_cast<uint64_t>(cfg_.mem_t_burst_cycles));
}

uint32_t TensorWorkload::memTimingBankIndex_(uint64_t off) const {
    const uint64_t interleave = std::max<uint64_t>(cfg_.dma_hbm_channel_interleave_bytes, 1ull);
    const uint32_t channels = hbmChannelCount_();
    const uint64_t stripe = off / interleave;
    const uint64_t stripe_in_ch = stripe / static_cast<uint64_t>(std::max<uint32_t>(channels, 1u));
    const uint32_t banks_per_channel = memTimingBanksPerChannel_();
    if (banks_per_channel <= 1u) return 0u;
    return static_cast<uint32_t>(stripe_in_ch % static_cast<uint64_t>(banks_per_channel));
}

uint32_t TensorWorkload::memTimingBankGroupIndex_(uint64_t off) const {
    const uint32_t banks_per_group = std::max<uint32_t>(cfg_.mem_banks_per_group, 1u);
    const uint32_t groups = std::max<uint32_t>(cfg_.mem_bank_groups_per_channel, 1u);
    const uint32_t bank = memTimingBankIndex_(off);
    const uint32_t group = bank / banks_per_group;
    if (groups <= 1u) return 0u;
    return static_cast<uint32_t>(group % groups);
}

uint64_t TensorWorkload::memTimingRowIndex_(uint64_t off) const {
    const uint64_t interleave = std::max<uint64_t>(cfg_.dma_hbm_channel_interleave_bytes, 1ull);
    const uint32_t channels = hbmChannelCount_();
    const uint64_t stripe = off / interleave;
    const uint64_t stripe_off = off % interleave;
    const uint64_t stripe_in_ch = stripe / static_cast<uint64_t>(std::max<uint32_t>(channels, 1u));
    const uint64_t channel_local_off = stripe_in_ch * interleave + stripe_off;
    const uint64_t row_bytes = clampNonZero_(cfg_.mem_row_bytes, 256ull);
    return channel_local_off / row_bytes;
}

uint64_t TensorWorkload::memTimingQueueDepthForReq_() const {
    return std::max<uint64_t>(1ull, static_cast<uint64_t>(cfg_.mem_bank_queue_depth));
}

uint64_t TensorWorkload::memTimingRefreshDelayCycles_(MemTimingBankState& st, uint64_t now_cycle) {
    const uint64_t interval = static_cast<uint64_t>(cfg_.mem_refresh_interval_cycles);
    const uint64_t block = static_cast<uint64_t>(cfg_.mem_refresh_block_cycles);
    if (interval == 0 || block == 0) return 0;
    if (st.last_refresh_cycle == 0) {
        st.last_refresh_cycle = now_cycle;
        return 0;
    }
    if (now_cycle <= st.last_refresh_cycle) return 0;
    const uint64_t elapsed = now_cycle - st.last_refresh_cycle;
    const uint64_t periods = elapsed / interval;
    if (periods == 0) return 0;
    st.last_refresh_cycle = st.last_refresh_cycle + periods * interval;
    const uint64_t delay = periods * block;
    tensor_mem_refresh_block_cycles_total_ = saturatingAddU64_(tensor_mem_refresh_block_cycles_total_, delay);
    return delay;
}

uint64_t TensorWorkload::memTimingProxyDelayCycles_(ReqKind kind, uint64_t off, uint32_t bytes, uint64_t now_cycle) {
    (void)kind;
    (void)bytes;
    if (!memTimingProxyEnabled_()) return 0;
    if (mem_timing_banks_.empty()) resetMemTimingState_();
    if (mem_timing_banks_.empty()) return 0;

    const uint32_t channels = hbmChannelCount_();
    const uint32_t banks_per_channel = memTimingBanksPerChannel_();
    const uint32_t ch = memOffsetToHbmChannel_(off);
    const uint32_t bank_local = memTimingBankIndex_(off);
    const uint64_t bank_global_u64 =
        static_cast<uint64_t>(ch % std::max<uint32_t>(channels, 1u)) * static_cast<uint64_t>(banks_per_channel) +
        static_cast<uint64_t>(bank_local % std::max<uint32_t>(banks_per_channel, 1u));
    if (bank_global_u64 >= static_cast<uint64_t>(mem_timing_banks_.size())) return 0;
    MemTimingBankState& st = mem_timing_banks_[static_cast<size_t>(bank_global_u64)];

    const uint64_t burst = memTimingServiceQuantumCycles_();
    uint64_t queue_wait = (st.busy_until_cycle > now_cycle) ? (st.busy_until_cycle - now_cycle) : 0ull;
    if (cfg_.mem_sched_policy == "frfcfs" && st.row_open && st.open_row == memTimingRowIndex_(off) && queue_wait > 0) {
        const uint64_t cut = std::min<uint64_t>(queue_wait, burst);
        queue_wait -= cut;
    }

    const uint64_t depth = memTimingQueueDepthForReq_();
    uint64_t queue_slots = (burst > 0) ? ceilDivU64_(queue_wait, burst) : 0ull;
    if (queue_slots >= depth) {
        tensor_mem_bank_queue_full_total_ = saturatingAddU64_(tensor_mem_bank_queue_full_total_, 1ull);
        const uint64_t cap = (depth > 0) ? ((depth - 1ull) * burst) : 0ull;
        queue_wait = std::min<uint64_t>(queue_wait, cap);
        queue_slots = (burst > 0) ? ceilDivU64_(queue_wait, burst) : 0ull;
    }
    tensor_mem_cmd_queue_slots_total_ =
        saturatingAddU64_(tensor_mem_cmd_queue_slots_total_, queue_slots);
    if (queue_slots > tensor_mem_cmd_queue_depth_max_) {
        tensor_mem_cmd_queue_depth_max_ = queue_slots;
    }

    const uint64_t row = memTimingRowIndex_(off);
    const uint64_t t_rcd = static_cast<uint64_t>(cfg_.mem_t_rcd_cycles);
    const uint64_t t_cl = static_cast<uint64_t>(cfg_.mem_t_cl_cycles);
    const uint64_t t_rp = static_cast<uint64_t>(cfg_.mem_t_rp_cycles);
    uint64_t row_delay = 0;
    uint32_t cmd_issue_count = 0;
    if (!st.row_open) {
        tensor_mem_row_miss_total_ = saturatingAddU64_(tensor_mem_row_miss_total_, 1ull);
        tensor_mem_cmd_act_total_ = saturatingAddU64_(tensor_mem_cmd_act_total_, 1ull);
        tensor_mem_cmd_rdwr_total_ = saturatingAddU64_(tensor_mem_cmd_rdwr_total_, 1ull);
        row_delay = t_rcd + t_cl + burst;
        cmd_issue_count = 2u;
    } else if (st.open_row == row) {
        tensor_mem_row_hit_total_ = saturatingAddU64_(tensor_mem_row_hit_total_, 1ull);
        tensor_mem_cmd_rdwr_total_ = saturatingAddU64_(tensor_mem_cmd_rdwr_total_, 1ull);
        row_delay = t_cl + burst;
        cmd_issue_count = 1u;
    } else {
        tensor_mem_row_conflict_total_ = saturatingAddU64_(tensor_mem_row_conflict_total_, 1ull);
        tensor_mem_cmd_pre_total_ = saturatingAddU64_(tensor_mem_cmd_pre_total_, 1ull);
        tensor_mem_cmd_act_total_ = saturatingAddU64_(tensor_mem_cmd_act_total_, 1ull);
        tensor_mem_cmd_rdwr_total_ = saturatingAddU64_(tensor_mem_cmd_rdwr_total_, 1ull);
        row_delay = t_rp + t_rcd + t_cl + burst;
        cmd_issue_count = 3u;
    }
    tensor_mem_row_service_cycles_total_ =
        saturatingAddU64_(tensor_mem_row_service_cycles_total_, row_delay);
    tensor_mem_cmd_issue_total_ =
        saturatingAddU64_(tensor_mem_cmd_issue_total_, static_cast<uint64_t>(std::max<uint32_t>(1u, cmd_issue_count)));

    uint64_t cmd_bus_wait = 0;
    if (cfg_.mem_timing_model == "proxy_v3" && !mem_timing_channels_.empty()) {
        const uint32_t ch_idx = ch % std::max<uint32_t>(channels, 1u);
        if (ch_idx < static_cast<uint32_t>(mem_timing_channels_.size())) {
            MemTimingChannelState& ch_state = mem_timing_channels_[static_cast<size_t>(ch_idx)];
            const uint32_t bank_group = memTimingBankGroupIndex_(off);
            const uint64_t tccd_s = std::max<uint64_t>(1ull, static_cast<uint64_t>(cfg_.mem_t_ccd_s_cycles));
            const uint64_t tccd_l = std::max<uint64_t>(tccd_s, static_cast<uint64_t>(cfg_.mem_t_ccd_l_cycles));

            uint64_t cmd_cycle = std::max<uint64_t>(now_cycle, ch_state.cmd_bus_ready_cycle);
            uint64_t first_cmd_cycle = cmd_cycle;
            const uint32_t cmd_total = std::max<uint32_t>(1u, cmd_issue_count);
            for (uint32_t i = 0; i < cmd_total; ++i) {
                uint64_t issue_cycle = cmd_cycle;
                if (ch_state.last_cmd_valid) {
                    const bool same_bg = (ch_state.last_cmd_bank_group == bank_group);
                    const uint64_t tccd = same_bg ? tccd_l : tccd_s;
                    issue_cycle = std::max<uint64_t>(issue_cycle, ch_state.last_cmd_cycle + tccd);
                    if (!same_bg) {
                        tensor_mem_cmd_bus_bg_switch_total_ =
                            saturatingAddU64_(tensor_mem_cmd_bus_bg_switch_total_, 1ull);
                    }
                }
                if (i == 0u) {
                    first_cmd_cycle = issue_cycle;
                }
                ch_state.last_cmd_cycle = issue_cycle;
                ch_state.last_cmd_bank_group = bank_group;
                ch_state.last_cmd_valid = true;
                ch_state.cmd_bus_ready_cycle = issue_cycle + 1ull;
                cmd_cycle = ch_state.cmd_bus_ready_cycle;
            }
            if (first_cmd_cycle > now_cycle) {
                cmd_bus_wait = first_cmd_cycle - now_cycle;
            }
            tensor_mem_cmd_bus_wait_cycles_total_ =
                saturatingAddU64_(tensor_mem_cmd_bus_wait_cycles_total_, cmd_bus_wait);
        }
    }

    const uint64_t service_start = now_cycle + queue_wait + cmd_bus_wait;
    const uint64_t refresh_delay = memTimingRefreshDelayCycles_(st, service_start);
    const uint64_t proxy_delay = queue_wait + cmd_bus_wait + refresh_delay + row_delay;
    const uint64_t active_span = std::max<uint64_t>(burst, row_delay);
    const uint64_t ready_cycle = service_start + refresh_delay + active_span;

    st.busy_until_cycle = ready_cycle;
    st.row_open = true;
    st.open_row = row;

    tensor_mem_bank_queue_wait_cycles_total_ =
        saturatingAddU64_(tensor_mem_bank_queue_wait_cycles_total_, queue_wait);
    if (cfg_.mem_sched_policy == "frfcfs") {
        tensor_mem_sched_frfcfs_pick_total_ = saturatingAddU64_(tensor_mem_sched_frfcfs_pick_total_, 1ull);
    } else {
        tensor_mem_sched_fifo_pick_total_ = saturatingAddU64_(tensor_mem_sched_fifo_pick_total_, 1ull);
    }
    tensor_mem_proxy_delay_cycles_total_ =
        saturatingAddU64_(tensor_mem_proxy_delay_cycles_total_, proxy_delay);
    if (proxy_delay > tensor_mem_proxy_delay_cycles_max_) {
        tensor_mem_proxy_delay_cycles_max_ = proxy_delay;
    }
    tensor_mem_bank_active_cycles_total_ =
        saturatingAddU64_(tensor_mem_bank_active_cycles_total_, active_span);
    return proxy_delay;
}

void TensorWorkload::resetMemTimingState_() {
    mem_timing_banks_.clear();
    mem_timing_channels_.clear();
    if (!memTimingProxyEnabled_()) return;
    const uint64_t banks =
        static_cast<uint64_t>(hbmChannelCount_()) * static_cast<uint64_t>(memTimingBanksPerChannel_());
    const size_t count = static_cast<size_t>(std::max<uint64_t>(banks, 1ull));
    mem_timing_banks_.assign(count, MemTimingBankState{});
    const size_t channels = static_cast<size_t>(std::max<uint32_t>(hbmChannelCount_(), 1u));
    mem_timing_channels_.assign(channels, MemTimingChannelState{});
}

uint64_t TensorWorkload::peekNextMemOffset_(ReqKind kind, uint32_t bytes) const {
    const uint64_t region = clampNonZero_(cfg_.mem_region_bytes, 4096);
    uint64_t off = (kind == ReqKind::Read) ? read_off_ : write_off_;
    if (off + static_cast<uint64_t>(bytes) > region) off = 0;
    return off;
}

uint32_t TensorWorkload::memOffsetToHbmChannel_(uint64_t off) const {
    const uint64_t interleave = std::max<uint64_t>(cfg_.dma_hbm_channel_interleave_bytes, 1ull);
    const uint32_t channels = hbmChannelCount_();
    if (channels <= 1u) return 0;
    const uint64_t idx = off / interleave;
    return static_cast<uint32_t>(idx % static_cast<uint64_t>(channels));
}

uint64_t TensorWorkload::memOffsetToPhysicalAddr_(uint64_t off) const {
    const uint64_t region = clampNonZero_(cfg_.mem_region_bytes, 4096);
    const uint32_t channels = hbmChannelCount_();
    if (channels <= 1u) return rt_.base_addr + off;

    const uint64_t interleave = std::max<uint64_t>(cfg_.dma_hbm_channel_interleave_bytes, 1ull);
    const uint64_t stripe = off / interleave;
    const uint64_t stripe_off = off % interleave;
    const uint32_t ch = static_cast<uint32_t>(stripe % static_cast<uint64_t>(channels));
    const uint64_t stripe_in_ch = stripe / static_cast<uint64_t>(channels);

    // Deinterleave: map a striped (logical) address space into contiguous per-channel
    // physical partitions (as built by mem_memhierarchy.build_pe_memory_systems()).
    const uint64_t base = region / static_cast<uint64_t>(channels);
    const uint64_t rem = region % static_cast<uint64_t>(channels);
    const uint64_t ch_base =
        base * static_cast<uint64_t>(ch) + std::min<uint64_t>(static_cast<uint64_t>(ch), rem);
    const uint64_t phys_off = ch_base + stripe_in_ch * interleave + stripe_off;
    return rt_.base_addr + (region ? (phys_off % region) : phys_off);
}

uint64_t TensorWorkload::hbmChannelBudgetLeftBytes_(uint64_t now_cycle, uint32_t channel) const {
    if (!hbmChannelBudgetEnabled_()) return std::numeric_limits<uint64_t>::max();

    const uint32_t channels = hbmChannelCount_();
    const uint64_t key = static_cast<uint64_t>(rt_.node_id);
    HbmChannelBudgetState_& st = g_hbm_channel_budget_state_[key];
    if (st.cycle_tag != now_cycle || st.left.size() != static_cast<size_t>(channels)) {
        st.cycle_tag = now_cycle;
        st.left.assign(static_cast<size_t>(channels), cfg_.dma_hbm_channel_bandwidth_bytes_per_cycle);
    }
    if (st.left.empty()) return 0;
    const uint32_t ch = (channels > 0u) ? (channel % channels) : 0u;
    const size_t idx = static_cast<size_t>(ch);
    if (idx >= st.left.size()) return 0;
    return st.left[idx];
}

void TensorWorkload::consumeHbmChannelBudget_(uint64_t now_cycle, uint32_t channel, uint32_t bytes) {
    if (!hbmChannelBudgetEnabled_()) return;

    const uint32_t channels = hbmChannelCount_();
    const uint64_t key = static_cast<uint64_t>(rt_.node_id);
    HbmChannelBudgetState_& st = g_hbm_channel_budget_state_[key];
    if (st.cycle_tag != now_cycle || st.left.size() != static_cast<size_t>(channels)) {
        st.cycle_tag = now_cycle;
        st.left.assign(static_cast<size_t>(channels), cfg_.dma_hbm_channel_bandwidth_bytes_per_cycle);
    }
    if (st.left.empty()) return;
    const uint32_t ch = (channels > 0u) ? (channel % channels) : 0u;
    const size_t idx = static_cast<size_t>(ch);
    if (idx >= st.left.size()) return;
    uint64_t& left = st.left[idx];
    const uint64_t cut = std::min<uint64_t>(left, static_cast<uint64_t>(bytes));
    left -= cut;
}

uint32_t TensorWorkload::clampBytesByHbmChannelBudget_(uint64_t now_cycle,
                                                      ReqKind kind,
                                                      uint32_t want,
                                                      uint32_t& out_channel) const {
    out_channel = 0;
    if (!hbmChannelBudgetEnabled_() || want == 0) return want;

    uint32_t bytes = want;
    // The address depends on wrap-around (off+bytes>region). Clamping bytes can change
    // whether wrap-around happens, so iterate to reach a stable (addr,channel,bytes).
    for (int it = 0; it < 3; ++it) {
        if (bytes == 0) return 0;
        const uint64_t off = peekNextMemOffset_(kind, bytes);
        const uint32_t ch = memOffsetToHbmChannel_(off);
        out_channel = ch;
        const uint64_t left = hbmChannelBudgetLeftBytes_(now_cycle, ch);
        const uint32_t clamped = static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(bytes), left));
        if (clamped == bytes) return bytes;
        bytes = clamped;
    }
    return bytes;
}

void TensorWorkload::bindRuntime(const Runtime& rt) {
    rt_ = rt;
    total_cores_cfg_ = std::max<uint32_t>(rt_.total_cores, 1u);
    if (inflight_.size() > static_cast<size_t>(cfg_.mem_max_outstanding) * 8u) {
        inflight_.clear();
    }
    resetMemTimingState_();

    // Program mode: parse DSL after runtime is available (for strict fatal logging).
    program_ops_.clear();
    program_loop_enable_ = cfg_.program_loop;
    program_iter_done_ = 0;
    program_pc_ = 0;
    program_op_started_ = false;
    program_softmax_rem_cycles_ = 0;
    program_gemm_started_ = false;
    program_collective_started_ = false;
    program_m7_enable_ = false;
    program_fence_pending_ = false;
    program_addr_aware_enable_ = false;
    const uint32_t ub_bufs = std::max<uint32_t>(cfg_.program_ub_buffers, 1u);
    program_ub_reserved_bytes_by_buf_.assign(ub_bufs, 0ull);
    program_ub_valid_bytes_by_buf_.assign(ub_bufs, 0ull);
    program_ub_regions_by_buf_.assign(ub_bufs, std::unordered_map<uint64_t, ProgramUbRegion>{});
    tensor_program_ub_occupancy_bytes_max_ = 0;
    program_dma_epoch_next_ = 1;
    program_dma_read_slot_ = ProgramDmaSlot{};
    program_dma_write_slot_ = ProgramDmaSlot{};
    program_mxu_slot_ = ProgramMxuSlot{};
    program_vec_slot_ = ProgramVecSlot{};
    program_coll_slot_ = ProgramCollSlot{};

    if (cfg_.exec_mode == "program") {
        std::vector<ProgramOp> ops;
        const std::string dsl = cfg_.program_dsl;
        if (!dsl.empty() && parseProgramDsl_(dsl, ops)) {
            program_ops_ = std::move(ops);
            for (const auto& op : program_ops_) {
                if (op.ub_addr_present || op.ub_read_addr_present || op.ub_write_addr_present) {
                    program_addr_aware_enable_ = true;
                }
                if (op.kind == ProgramOpKind::DmaRead ||
                    op.kind == ProgramOpKind::DmaWrite ||
                    op.kind == ProgramOpKind::Fence ||
                    op.kind == ProgramOpKind::GemmUb) {
                    program_m7_enable_ = true;
                }
            }

            if (program_addr_aware_enable_) {
                const uint64_t ub_bytes = cfg_.ub_bytes;
                if (ub_bytes == 0 || (ub_bytes % static_cast<uint64_t>(ub_bufs)) != 0) {
                    if (cfg_.strict && rt_.log) {
                        rt_.log->fatal(
                            CALL_INFO,
                            -1,
                            "tensor fatal: program address-aware mode requires tensor_ub_bytes divisible by tensor_program_ub_buffers "
                            "(core=%u ub_bytes=%llu ub_buffers=%u)\n",
                            rt_.core_id,
                            (unsigned long long)ub_bytes,
                            ub_bufs);
                    }
                    program_addr_aware_enable_ = false;
                }
            }
        } else if (cfg_.strict && rt_.log) {
            rt_.log->fatal(
                CALL_INFO,
                -1,
                "tensor fatal: exec_mode=program but tensor_program_dsl is empty or invalid (core=%u)\n",
                rt_.core_id);
        }
    }
}

void TensorWorkload::onGlobalStepStart(uint32_t seq) {
    (void)seq;
    step_gated_ = true;
    step_open_ = true;
    step_seq_ = seq;
    if (cfg_.exec_mode == "program") {
        program_pc_ = 0;
        program_op_started_ = false;
        program_softmax_rem_cycles_ = 0;
        program_gemm_started_ = false;
        program_collective_started_ = false;
    }
}

void TensorWorkload::startIteration_() {
    iter_active_ = true;
    compute_started_ = (cfg_.exec_mode == "bulk") ? (cfg_.overlap_enable || cfg_.double_buffer) : false;
    rem_read_bytes_ = (cfg_.mem_enable && rt_.mem) ? bytes_read_per_iter_ : 0;
    rem_write_bytes_ = (cfg_.mem_enable && rt_.mem) ? bytes_write_per_iter_ : 0;
    rem_compute_math_cycles_ = compute_math_cycles_per_iter_;
    rem_compute_pipeline_cycles_ = compute_pipeline_cycles_per_iter_;
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
    onchip_ub_occupancy_bytes_ = 0;
    onchip_weight_occupancy_bytes_ = 0;
    onchip_acc_occupancy_bytes_ = 0;
    std::fill(onchip_ub_bank_occupancy_bytes_.begin(), onchip_ub_bank_occupancy_bytes_.end(), 0ull);
    std::fill(onchip_weight_bank_occupancy_bytes_.begin(), onchip_weight_bank_occupancy_bytes_.end(), 0ull);
    std::fill(onchip_acc_bank_occupancy_bytes_.begin(), onchip_acc_bank_occupancy_bytes_.end(), 0ull);
    std::fill(onchip_ub_bank_queue_occupancy_.begin(), onchip_ub_bank_queue_occupancy_.end(), 0u);
    std::fill(onchip_weight_bank_queue_occupancy_.begin(), onchip_weight_bank_queue_occupancy_.end(), 0u);
    std::fill(onchip_acc_bank_queue_occupancy_.begin(), onchip_acc_bank_queue_occupancy_.end(), 0u);
    onchip_ub_bank_rr_ = 0;
    onchip_weight_bank_rr_ = 0;
    onchip_acc_bank_rr_ = 0;
    if (!collectiveUseEventCreditReturn_()) {
        collective_credit_inflight_chunks_ = 0;
        collective_credit_outstanding_.clear();
        collective_credit_return_seen_.clear();
        collective_credit_return_pending_credits_.clear();
        collective_credit_return_pending_queue_.clear();
    }
    acc_reserved_tiles_.clear();
    a_resident_tiles_.clear();
    b_resident_tiles_.clear();

    tile_mode_active_ = (cfg_.exec_mode == "tile" || cfg_.exec_mode == "program");
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

uint64_t TensorWorkload::tileSegMathCycles_(uint64_t seg_index) const {
    const uint64_t extra = (seg_index < tile_seg_math_cycles_remainder_) ? 1ull : 0ull;
    return tile_seg_math_cycles_base_ + extra;
}

uint64_t TensorWorkload::tileSegComputeCycles_(uint64_t seg_index) const {
    const uint64_t math_cycles = tileSegMathCycles_(seg_index);
    return saturatingAddU64_(math_cycles, static_cast<uint64_t>(compute_pipeline_latency_cycles_effective_));
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
    out.rem_compute_math_cycles = tileSegMathCycles_(out.seg_index);
    out.rem_compute_pipeline_cycles = static_cast<uint64_t>(compute_pipeline_latency_cycles_effective_);
    out.rem_compute_wavefront_cycles = 0;
    const uint64_t base_total = out.rem_compute_math_cycles + out.rem_compute_pipeline_cycles;
    if (cfg_.mxu_wavefront_enable && cfg_.mxu_wavefront_alpha > 0.0f) {
        const uint64_t m0 = static_cast<uint64_t>(out.mi) * static_cast<uint64_t>(tile_tm_);
        const uint64_t n0 = static_cast<uint64_t>(out.ni) * static_cast<uint64_t>(tile_tn_);
        const uint64_t k0 = static_cast<uint64_t>(out.ki) * static_cast<uint64_t>(tile_tk_);
        const uint64_t m_eff = (m0 < cfg_.m) ? std::min<uint64_t>(tile_tm_, static_cast<uint64_t>(cfg_.m) - m0) : 0ull;
        const uint64_t n_eff = (n0 < cfg_.n) ? std::min<uint64_t>(tile_tn_, static_cast<uint64_t>(cfg_.n) - n0) : 0ull;
        const uint64_t k_eff = (k0 < cfg_.k) ? std::min<uint64_t>(tile_tk_, static_cast<uint64_t>(cfg_.k) - k0) : 0ull;
        uint64_t wf_span = 0;
        if (m_eff > 0 && n_eff > 0 && k_eff > 0) {
            // Approximate systolic fill/drain latency. If the tile exceeds the physical
            // array dimensions, model it as multiple array-sized blocks (e.g., 64x64 on
            // a 32x32 array => 2x2 blocks).
            const uint64_t am = clampNonZero_(static_cast<uint64_t>(cfg_.array_m), 1ull);
            const uint64_t an = clampNonZero_(static_cast<uint64_t>(cfg_.array_n), 1ull);
            const uint64_t mb = ceilDivU64_(m_eff, am);
            const uint64_t nb = ceilDivU64_(n_eff, an);
            const uint64_t m_blk = std::min<uint64_t>(m_eff, am);
            const uint64_t n_blk = std::min<uint64_t>(n_eff, an);
            const uint64_t span_blk = m_blk + n_blk + k_eff - 2ull;
            wf_span = saturatingMulU64_(saturatingMulU64_(mb, nb), span_blk);
        }
        const double wf_body_d = static_cast<double>(cfg_.mxu_wavefront_alpha) * static_cast<double>(wf_span);
        const uint64_t wf_body = static_cast<uint64_t>(std::ceil(std::max(0.0, wf_body_d)));
        const uint64_t wf_total = (wf_body ? wf_body : 1ull) + out.rem_compute_pipeline_cycles;
        if (wf_total > base_total) {
            out.rem_compute_wavefront_cycles = wf_total - base_total;
        }
    }
    out.rem_compute_cycles = out.rem_compute_math_cycles + out.rem_compute_pipeline_cycles + out.rem_compute_wavefront_cycles;

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
        releaseAccTile_(it->second.mi, it->second.ni);
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

    uint32_t bytes = static_cast<uint32_t>(std::min<uint32_t>(max_bytes, cfg_.mem_req_bytes));
    const uint64_t region = clampNonZero_(cfg_.mem_region_bytes, 4096);
    uint64_t off = read_off_;
    const uint32_t channels = hbmChannelCount_();
    if (channels > 1u) {
        const uint64_t interleave = std::max<uint64_t>(cfg_.dma_hbm_channel_interleave_bytes, 1ull);
        const uint64_t rem = interleave - (off % interleave);
        bytes = static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(bytes), rem));
    }
    if (off + bytes > region) off = 0;
    const uint64_t proxy_delay_cycles = memTimingProxyDelayCycles_(ReqKind::Read, off, bytes, now_cycle_cached_);
    const uint64_t addr = memOffsetToPhysicalAddr_(off);
    read_off_ = off + bytes;
    if (read_off_ >= region) read_off_ = 0;

    if (rt_.reporting.report_mem_issue) {
        rt_.reporting.report_mem_issue(rt_.reporting.ctx, bytes);
    }
    memory_requests_++;
    tensor_mem_reads_issued_total_ += 1;
    tensor_mem_bytes_read_total_ += static_cast<uint64_t>(bytes);
    const uint64_t issue_ns = nowNs_();

    const auto req_id = rt_.mem->read(
        addr, bytes,
        [this, addr, bytes, tag, epoch](IMemoryAccess::RequestId cb_id, uint64_t /*addr_cb*/, std::vector<uint8_t>&& got) {
            if (cb_id == 0 || got.size() != bytes) {
                if (cfg_.strict && rt_.log) {
                    rt_.log->fatal(
                        CALL_INFO, -1,
                        "tensor fatal: read failed (core=%u addr=0x%llx bytes=%u got=%zu)\n",
                        rt_.core_id, (unsigned long long)addr, bytes, got.size());
                }
                return;
            }
            onMemResponse_(static_cast<uint64_t>(cb_id), ReqKind::Read, tag, epoch, bytes);
        });
    if (req_id != 0) {
        inflight_[static_cast<uint64_t>(req_id)] = InflightReq{
            ReqKind::Read,
            tag,
            epoch,
            bytes,
            issue_ns,
            proxy_delay_cycles,
        };
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

    uint32_t bytes = static_cast<uint32_t>(std::min<uint32_t>(max_bytes, cfg_.mem_req_bytes));
    const uint64_t region = clampNonZero_(cfg_.mem_region_bytes, 4096);
    uint64_t off = write_off_;
    const uint32_t channels = hbmChannelCount_();
    if (channels > 1u) {
        const uint64_t interleave = std::max<uint64_t>(cfg_.dma_hbm_channel_interleave_bytes, 1ull);
        const uint64_t rem = interleave - (off % interleave);
        bytes = static_cast<uint32_t>(std::min<uint64_t>(static_cast<uint64_t>(bytes), rem));
    }
    if (off + bytes > region) off = 0;
    const uint64_t proxy_delay_cycles = memTimingProxyDelayCycles_(ReqKind::Write, off, bytes, now_cycle_cached_);
    const uint64_t addr = memOffsetToPhysicalAddr_(off);
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
    const uint64_t issue_ns = nowNs_();

    const auto req_id = rt_.mem->write(
        addr, data,
        [this, addr, bytes, tag, epoch](IMemoryAccess::RequestId cb_id, uint64_t /*addr_cb*/) {
            if (cb_id == 0) {
                if (cfg_.strict && rt_.log) {
                    rt_.log->fatal(
                        CALL_INFO, -1,
                        "tensor fatal: write failed (core=%u addr=0x%llx bytes=%u)\n",
                        rt_.core_id, (unsigned long long)addr, bytes);
                }
                return;
            }
            onMemResponse_(static_cast<uint64_t>(cb_id), ReqKind::Write, tag, epoch, bytes);
        });
    if (req_id != 0) {
        inflight_[static_cast<uint64_t>(req_id)] = InflightReq{
            ReqKind::Write,
            tag,
            epoch,
            bytes,
            issue_ns,
            proxy_delay_cycles,
        };
    } else if (cfg_.strict && rt_.log) {
        rt_.log->fatal(
            CALL_INFO, -1,
            "tensor fatal: write issue failed (core=%u addr=0x%llx bytes=%u)\n",
            rt_.core_id, (unsigned long long)addr, bytes);
    }

    return bytes;
}

void TensorWorkload::onMemResponse_(uint64_t req_id, ReqKind kind, MemTag tag, uint64_t epoch, uint32_t bytes) {
    uint64_t issue_ns = 0;
    uint64_t proxy_delay_cycles = 0;
    auto it = inflight_.find(req_id);
    if (it != inflight_.end()) {
        issue_ns = it->second.issue_ns;
        proxy_delay_cycles = it->second.proxy_delay_cycles;
    }
    const uint64_t done_ns = nowNs_();
    uint64_t lat = 0;
    if (done_ns >= issue_ns) {
        lat = done_ns - issue_ns;
    }
    lat = saturatingAddU64_(lat, proxy_delay_cycles);

    if (kind == ReqKind::Read) {
        tensor_mem_read_latency_cycles_total_ = saturatingAddU64_(tensor_mem_read_latency_cycles_total_, lat);
        if (lat > tensor_mem_read_latency_cycles_max_) tensor_mem_read_latency_cycles_max_ = lat;
        tensor_mem_read_latency_samples_total_ = saturatingAddU64_(tensor_mem_read_latency_samples_total_, 1);
    } else {
        tensor_mem_write_latency_cycles_total_ = saturatingAddU64_(tensor_mem_write_latency_cycles_total_, lat);
        if (lat > tensor_mem_write_latency_cycles_max_) tensor_mem_write_latency_cycles_max_ = lat;
        tensor_mem_write_latency_samples_total_ = saturatingAddU64_(tensor_mem_write_latency_samples_total_, 1);
    }

    if (it != inflight_.end()) {
        inflight_.erase(it);
    }
    onMemComplete_(kind, tag, epoch, bytes);
}

void TensorWorkload::onMemComplete_(ReqKind kind, MemTag tag, uint64_t epoch, uint32_t bytes) {
    // Program-mode explicit DMA (M7): update slot progress even when tile model is disabled.
    if (cfg_.exec_mode == "program" && program_m7_enable_) {
        if (program_dma_read_slot_.active && program_dma_read_slot_.epoch == epoch) {
            if (kind == ReqKind::Read && tag == MemTag::ProgramDmaRead) {
                program_dma_read_slot_.done_bytes = std::min<uint64_t>(
                    program_dma_read_slot_.total_bytes,
                    program_dma_read_slot_.done_bytes + static_cast<uint64_t>(bytes));
                if (program_dma_read_slot_.inflight_reqs > 0) {
                    program_dma_read_slot_.inflight_reqs -= 1;
                }
            }
        }
        if (program_dma_write_slot_.active && program_dma_write_slot_.epoch == epoch) {
            if (kind == ReqKind::Write && tag == MemTag::ProgramDmaWrite) {
                program_dma_write_slot_.done_bytes = std::min<uint64_t>(
                    program_dma_write_slot_.total_bytes,
                    program_dma_write_slot_.done_bytes + static_cast<uint64_t>(bytes));
                if (program_dma_write_slot_.inflight_reqs > 0) {
                    program_dma_write_slot_.inflight_reqs -= 1;
                }
            }
        }
    }

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

    tensor_compute_cycles_total_ += 1;
    if (rem_compute_math_cycles_ > 0) {
        rem_compute_math_cycles_--;
        rem_compute_cycles_ = (rem_compute_cycles_ > 0) ? (rem_compute_cycles_ - 1) : 0;
        tensor_compute_math_cycles_total_ += 1;

        const uint64_t per = effectivePeakMacsPerCycle_();
        const uint64_t done = (rem_macs_ >= per) ? per : rem_macs_;
        tensor_mac_ops_total_ += done;
        rem_macs_ = (rem_macs_ >= done) ? (rem_macs_ - done) : 0;
        return true;
    }

    if (rem_compute_pipeline_cycles_ > 0) {
        rem_compute_pipeline_cycles_--;
        rem_compute_cycles_ = (rem_compute_cycles_ > 0) ? (rem_compute_cycles_ - 1) : 0;
        tensor_compute_pipeline_cycles_total_ += 1;
        return true;
    }

    rem_compute_cycles_ = 0;
    return false;
}

uint32_t TensorWorkload::selectUbBank_(uint64_t tag_seed) {
    const uint32_t banks = std::max<uint32_t>(cfg_.ub_bank_count, 1u);
    if (banks <= 1u) return 0u;

    if (cfg_.ub_bank_select_policy == "rr") {
        const uint32_t bank = onchip_ub_bank_rr_ % banks;
        onchip_ub_bank_rr_ = (bank + 1u) % banks;
        return bank;
    }
    if (cfg_.ub_bank_select_policy == "hash") {
        uint64_t x = tag_seed ^ (static_cast<uint64_t>(rt_.node_id) << 32) ^
                     (static_cast<uint64_t>(rt_.core_id) << 16) ^ cfg_.seed_base;
        return static_cast<uint32_t>(splitmix64_next_(x) % static_cast<uint64_t>(banks));
    }

    const uint64_t gran = std::max<uint64_t>(cfg_.mem_req_bytes, 1u);
    return static_cast<uint32_t>((tag_seed / gran) % static_cast<uint64_t>(banks));
}

uint32_t TensorWorkload::selectWeightBank_(uint64_t tag_seed) {
    const uint32_t banks = std::max<uint32_t>(cfg_.ub_bank_count, 1u);
    if (banks <= 1u) return 0u;

    if (cfg_.ub_bank_select_policy == "rr") {
        const uint32_t bank = onchip_weight_bank_rr_ % banks;
        onchip_weight_bank_rr_ = (bank + 1u) % banks;
        return bank;
    }
    if (cfg_.ub_bank_select_policy == "hash") {
        uint64_t x = tag_seed ^ (static_cast<uint64_t>(rt_.node_id) << 32) ^
                     (static_cast<uint64_t>(rt_.core_id) << 16) ^ cfg_.seed_base;
        return static_cast<uint32_t>(splitmix64_next_(x) % static_cast<uint64_t>(banks));
    }

    const uint64_t gran = std::max<uint64_t>(cfg_.mem_req_bytes, 1u);
    return static_cast<uint32_t>((tag_seed / gran) % static_cast<uint64_t>(banks));
}

uint32_t TensorWorkload::selectAccBank_(uint64_t tag_seed) {
    const uint32_t banks = std::max<uint32_t>(cfg_.acc_bank_count, 1u);
    if (banks <= 1u) return 0u;

    if (cfg_.acc_bank_select_policy == "rr") {
        const uint32_t bank = onchip_acc_bank_rr_ % banks;
        onchip_acc_bank_rr_ = (bank + 1u) % banks;
        return bank;
    }
    if (cfg_.acc_bank_select_policy == "hash") {
        uint64_t x = tag_seed ^ (static_cast<uint64_t>(tile_seg_done_) << 20) ^
                     (static_cast<uint64_t>(rt_.node_id) << 32) ^
                     (static_cast<uint64_t>(rt_.core_id) << 16) ^ cfg_.seed_base;
        return static_cast<uint32_t>(splitmix64_next_(x) % static_cast<uint64_t>(banks));
    }

    const uint64_t gran = std::max<uint64_t>(tile_c_bytes_, 1ull);
    return static_cast<uint32_t>((tag_seed / gran) % static_cast<uint64_t>(banks));
}

void TensorWorkload::updateBankQueueOccupancyMax_() {
    uint64_t max_occ = 0;
    for (uint32_t v : onchip_ub_bank_queue_occupancy_) {
        if (static_cast<uint64_t>(v) > max_occ) max_occ = static_cast<uint64_t>(v);
    }
    for (uint32_t v : onchip_weight_bank_queue_occupancy_) {
        if (static_cast<uint64_t>(v) > max_occ) max_occ = static_cast<uint64_t>(v);
    }
    for (uint32_t v : onchip_acc_bank_queue_occupancy_) {
        if (static_cast<uint64_t>(v) > max_occ) max_occ = static_cast<uint64_t>(v);
    }
    if (max_occ > tensor_bank_queue_occupancy_max_) {
        tensor_bank_queue_occupancy_max_ = max_occ;
    }
}

void TensorWorkload::resetOnchipCycleState_(uint64_t now_cycle) {
    if (onchip_cycle_tag_ == now_cycle) return;
    onchip_cycle_tag_ = now_cycle;
    onchip_ub_read_ports_used_ = 0;
    onchip_ub_write_ports_used_ = 0;
    onchip_acc_read_ports_used_ = 0;
    onchip_acc_write_ports_used_ = 0;
    if (cfg_.onchip_bank_model_enable) {
        for (uint32_t& v : onchip_ub_bank_queue_occupancy_) {
            if (v > 0) v -= 1;
        }
        for (uint32_t& v : onchip_weight_bank_queue_occupancy_) {
            if (v > 0) v -= 1;
        }
        for (uint32_t& v : onchip_acc_bank_queue_occupancy_) {
            if (v > 0) v -= 1;
        }
    }
}

bool TensorWorkload::acquireOnchipReadPorts_(uint32_t ub_ports_needed, uint32_t acc_ports_needed) {
    if (!cfg_.onchip_model_enable) return true;
    const uint32_t ub_limit = cfg_.ub_read_ports;
    const uint32_t acc_limit = cfg_.acc_read_ports;

    if (ub_limit > 0 && onchip_ub_read_ports_used_ + ub_ports_needed > ub_limit) return false;
    if (acc_limit > 0 && onchip_acc_read_ports_used_ + acc_ports_needed > acc_limit) return false;
    onchip_ub_read_ports_used_ += ub_ports_needed;
    onchip_acc_read_ports_used_ += acc_ports_needed;
    return true;
}

bool TensorWorkload::acquireOnchipWritePorts_(uint32_t ub_ports_needed, uint32_t acc_ports_needed) {
    if (!cfg_.onchip_model_enable) return true;
    const uint32_t ub_limit = cfg_.ub_write_ports;
    const uint32_t acc_limit = cfg_.acc_write_ports;

    if (ub_limit > 0 && onchip_ub_write_ports_used_ + ub_ports_needed > ub_limit) return false;
    if (acc_limit > 0 && onchip_acc_write_ports_used_ + acc_ports_needed > acc_limit) return false;
    onchip_ub_write_ports_used_ += ub_ports_needed;
    onchip_acc_write_ports_used_ += acc_ports_needed;
    return true;
}

bool TensorWorkload::reserveUbBytes_(uint64_t bytes,
                                     uint64_t& spill_budget,
                                     bool& spilled,
                                     bool& spill_budget_blocked,
                                     bool& bank_conflict_blocked,
                                     std::vector<std::pair<uint32_t, uint64_t>>* bank_allocs) {
    spilled = false;
    spill_budget_blocked = false;
    bank_conflict_blocked = false;
    if (bytes == 0) return true;
    if (!cfg_.onchip_model_enable || cfg_.ub_bytes == 0) return true;

    auto spill_or_fail = [&](bool bank_conflict) -> bool {
        if (!cfg_.spill_enable) {
            bank_conflict_blocked = bank_conflict;
            return false;
        }
        const bool spill_budget_capped =
            cfg_.spill_share_noc_budget &&
            cfg_.noc_bandwidth_bytes_per_cycle > 0 &&
            spill_budget != std::numeric_limits<uint64_t>::max();
        if (spill_budget_capped && spill_budget < bytes) {
            spill_budget_blocked = true;
            return false;
        }

        spilled = true;
        tensor_spill_bytes_total_ += bytes;
        tensor_spill_pkts_total_ += ceilDivU64_(bytes, std::max<uint64_t>(cfg_.spill_packet_bytes, 8ull));
        if (spill_budget_capped) {
            spill_budget -= bytes;
        }
        return true;
    };

    if (!cfg_.onchip_bank_model_enable || onchip_ub_bank_occupancy_bytes_.empty()) {
        if (onchip_ub_occupancy_bytes_ + bytes <= cfg_.ub_bytes) {
            onchip_ub_occupancy_bytes_ += bytes;
            return true;
        }
        return spill_or_fail(false);
    }

    const uint32_t bank = selectUbBank_(onchip_ub_occupancy_bytes_ + bytes + tensor_mem_reads_issued_total_);
    if (bank >= onchip_ub_bank_occupancy_bytes_.size() || bank >= onchip_ub_bank_queue_occupancy_.size()) {
        return spill_or_fail(false);
    }

    bool queue_marked = false;
    if (cfg_.ub_bank_conflict_mode == "queue") {
        if (onchip_ub_bank_queue_occupancy_[bank] >= cfg_.bank_queue_depth) {
            bank_conflict_blocked = true;
            return false;
        }
        onchip_ub_bank_queue_occupancy_[bank] += 1;
        queue_marked = true;
        updateBankQueueOccupancyMax_();
    } else {
        if (onchip_ub_bank_queue_occupancy_[bank] > 0) {
            bank_conflict_blocked = true;
            return false;
        }
        onchip_ub_bank_queue_occupancy_[bank] = 1;
        queue_marked = true;
        updateBankQueueOccupancyMax_();
    }

    const uint64_t bank_cap = (cfg_.ub_bank_bytes > 0)
                                  ? cfg_.ub_bank_bytes
                                  : ceilDivU64_(cfg_.ub_bytes, static_cast<uint64_t>(std::max<uint32_t>(cfg_.ub_bank_count, 1u)));
    const bool global_fit = onchip_ub_occupancy_bytes_ + bytes <= cfg_.ub_bytes;
    const bool bank_fit = onchip_ub_bank_occupancy_bytes_[bank] + bytes <= bank_cap;
    if (global_fit && bank_fit) {
        onchip_ub_occupancy_bytes_ += bytes;
        onchip_ub_bank_occupancy_bytes_[bank] += bytes;
        if (bank_allocs) bank_allocs->emplace_back(bank, bytes);
        return true;
    }

    if (queue_marked && onchip_ub_bank_queue_occupancy_[bank] > 0) {
        onchip_ub_bank_queue_occupancy_[bank] -= 1;
    }
    return spill_or_fail(!bank_fit);
}

bool TensorWorkload::reserveWeightBytes_(uint64_t bytes,
                                         uint64_t& spill_budget,
                                         bool& spilled,
                                         bool& spill_budget_blocked,
                                         bool& bank_conflict_blocked,
                                         std::vector<std::pair<uint32_t, uint64_t>>* bank_allocs) {
    spilled = false;
    spill_budget_blocked = false;
    bank_conflict_blocked = false;
    if (bytes == 0) return true;
    if (!cfg_.onchip_model_enable || cfg_.weight_bytes == 0) return true;

    auto spill_or_fail = [&](bool bank_conflict) -> bool {
        if (!cfg_.spill_enable) {
            bank_conflict_blocked = bank_conflict;
            return false;
        }
        const bool spill_budget_capped =
            cfg_.spill_share_noc_budget &&
            cfg_.noc_bandwidth_bytes_per_cycle > 0 &&
            spill_budget != std::numeric_limits<uint64_t>::max();
        if (spill_budget_capped && spill_budget < bytes) {
            spill_budget_blocked = true;
            return false;
        }

        spilled = true;
        tensor_spill_bytes_total_ += bytes;
        tensor_spill_pkts_total_ += ceilDivU64_(bytes, std::max<uint64_t>(cfg_.spill_packet_bytes, 8ull));
        if (spill_budget_capped) {
            spill_budget -= bytes;
        }
        return true;
    };

    if (!cfg_.onchip_bank_model_enable || onchip_weight_bank_occupancy_bytes_.empty()) {
        if (onchip_weight_occupancy_bytes_ + bytes <= cfg_.weight_bytes) {
            onchip_weight_occupancy_bytes_ += bytes;
            if (onchip_weight_occupancy_bytes_ > tensor_onchip_weight_occupancy_bytes_max_) {
                tensor_onchip_weight_occupancy_bytes_max_ = onchip_weight_occupancy_bytes_;
            }
            return true;
        }
        return spill_or_fail(false);
    }

    const uint32_t bank = selectWeightBank_(onchip_weight_occupancy_bytes_ + bytes + tensor_mem_reads_issued_total_);
    if (bank >= onchip_weight_bank_occupancy_bytes_.size() || bank >= onchip_weight_bank_queue_occupancy_.size()) {
        return spill_or_fail(false);
    }

    bool queue_marked = false;
    if (cfg_.ub_bank_conflict_mode == "queue") {
        if (onchip_weight_bank_queue_occupancy_[bank] >= cfg_.bank_queue_depth) {
            bank_conflict_blocked = true;
            return false;
        }
        onchip_weight_bank_queue_occupancy_[bank] += 1;
        queue_marked = true;
        updateBankQueueOccupancyMax_();
    } else {
        if (onchip_weight_bank_queue_occupancy_[bank] > 0) {
            bank_conflict_blocked = true;
            return false;
        }
        onchip_weight_bank_queue_occupancy_[bank] = 1;
        queue_marked = true;
        updateBankQueueOccupancyMax_();
    }

    const uint64_t bank_cap = (cfg_.ub_bank_bytes > 0)
                                  ? cfg_.ub_bank_bytes
                                  : ceilDivU64_(cfg_.weight_bytes, static_cast<uint64_t>(std::max<uint32_t>(cfg_.ub_bank_count, 1u)));
    const bool global_fit = onchip_weight_occupancy_bytes_ + bytes <= cfg_.weight_bytes;
    const bool bank_fit = onchip_weight_bank_occupancy_bytes_[bank] + bytes <= bank_cap;
    if (global_fit && bank_fit) {
        onchip_weight_occupancy_bytes_ += bytes;
        onchip_weight_bank_occupancy_bytes_[bank] += bytes;
        if (onchip_weight_occupancy_bytes_ > tensor_onchip_weight_occupancy_bytes_max_) {
            tensor_onchip_weight_occupancy_bytes_max_ = onchip_weight_occupancy_bytes_;
        }
        if (onchip_weight_bank_occupancy_bytes_[bank] > tensor_onchip_weight_bank_occupancy_bytes_max_) {
            tensor_onchip_weight_bank_occupancy_bytes_max_ = onchip_weight_bank_occupancy_bytes_[bank];
        }
        if (bank_allocs) bank_allocs->emplace_back(bank, bytes);
        return true;
    }

    if (queue_marked && onchip_weight_bank_queue_occupancy_[bank] > 0) {
        onchip_weight_bank_queue_occupancy_[bank] -= 1;
    }
    return spill_or_fail(!bank_fit);
}

void TensorWorkload::releaseUbBytes_(uint64_t bytes) {
    if (!cfg_.onchip_model_enable || cfg_.ub_bytes == 0 || bytes == 0) return;
    onchip_ub_occupancy_bytes_ = (onchip_ub_occupancy_bytes_ >= bytes) ? (onchip_ub_occupancy_bytes_ - bytes) : 0;
}

void TensorWorkload::releaseUbBankAllocs_(std::vector<std::pair<uint32_t, uint64_t>>& bank_allocs) {
    if (!cfg_.onchip_model_enable || cfg_.ub_bytes == 0) {
        bank_allocs.clear();
        return;
    }
    for (const auto& alloc : bank_allocs) {
        const uint32_t bank = alloc.first;
        const uint64_t bytes = alloc.second;
        if (bytes == 0) continue;
        onchip_ub_occupancy_bytes_ = (onchip_ub_occupancy_bytes_ >= bytes) ? (onchip_ub_occupancy_bytes_ - bytes) : 0;
        if (bank < onchip_ub_bank_occupancy_bytes_.size()) {
            uint64_t& occ = onchip_ub_bank_occupancy_bytes_[bank];
            occ = (occ >= bytes) ? (occ - bytes) : 0;
        }
    }
    bank_allocs.clear();
}

void TensorWorkload::releaseWeightBytes_(uint64_t bytes) {
    if (!cfg_.onchip_model_enable || cfg_.weight_bytes == 0 || bytes == 0) return;
    onchip_weight_occupancy_bytes_ =
        (onchip_weight_occupancy_bytes_ >= bytes) ? (onchip_weight_occupancy_bytes_ - bytes) : 0;
}

void TensorWorkload::releaseWeightBankAllocs_(std::vector<std::pair<uint32_t, uint64_t>>& bank_allocs) {
    if (!cfg_.onchip_model_enable || cfg_.weight_bytes == 0) {
        bank_allocs.clear();
        return;
    }
    for (const auto& alloc : bank_allocs) {
        const uint32_t bank = alloc.first;
        const uint64_t bytes = alloc.second;
        if (bytes == 0) continue;
        onchip_weight_occupancy_bytes_ =
            (onchip_weight_occupancy_bytes_ >= bytes) ? (onchip_weight_occupancy_bytes_ - bytes) : 0;
        if (bank < onchip_weight_bank_occupancy_bytes_.size()) {
            uint64_t& occ = onchip_weight_bank_occupancy_bytes_[bank];
            occ = (occ >= bytes) ? (occ - bytes) : 0;
        }
    }
    bank_allocs.clear();
}

bool TensorWorkload::reserveAccTile_(uint32_t mi,
                                     uint32_t ni,
                                     uint64_t& spill_budget,
                                     bool& spilled,
                                     bool& spill_budget_blocked,
                                     bool& bank_conflict_blocked) {
    spilled = false;
    spill_budget_blocked = false;
    bank_conflict_blocked = false;
    if (!cfg_.onchip_model_enable || cfg_.acc_bytes == 0 || tile_c_bytes_ == 0) {
        return true;
    }

    const uint64_t key = (static_cast<uint64_t>(mi) << 32) | static_cast<uint64_t>(ni);
    if (acc_reserved_tiles_.find(key) != acc_reserved_tiles_.end()) {
        return true;
    }

    auto spill_or_fail = [&](bool bank_conflict) -> bool {
        if (!cfg_.spill_enable) {
            bank_conflict_blocked = bank_conflict;
            return false;
        }
        const bool spill_budget_capped =
            cfg_.spill_share_noc_budget &&
            cfg_.noc_bandwidth_bytes_per_cycle > 0 &&
            spill_budget != std::numeric_limits<uint64_t>::max();
        if (spill_budget_capped && spill_budget < tile_c_bytes_) {
            spill_budget_blocked = true;
            return false;
        }

        spilled = true;
        tensor_spill_bytes_total_ += tile_c_bytes_;
        tensor_spill_pkts_total_ += ceilDivU64_(tile_c_bytes_, std::max<uint64_t>(cfg_.spill_packet_bytes, 8ull));
        if (spill_budget_capped) {
            spill_budget -= tile_c_bytes_;
        }
        return true;
    };

    if (!cfg_.onchip_bank_model_enable || onchip_acc_bank_occupancy_bytes_.empty()) {
        if (onchip_acc_occupancy_bytes_ + tile_c_bytes_ <= cfg_.acc_bytes) {
            onchip_acc_occupancy_bytes_ += tile_c_bytes_;
            acc_reserved_tiles_.emplace(key, AccTileAlloc{tile_c_bytes_, 0u, 0u});
            return true;
        }
        return spill_or_fail(false);
    }

    const uint32_t bank = selectAccBank_(key);
    if (bank >= onchip_acc_bank_occupancy_bytes_.size() || bank >= onchip_acc_bank_queue_occupancy_.size()) {
        return spill_or_fail(false);
    }

    bool queue_marked = false;
    if (cfg_.acc_bank_conflict_mode == "queue") {
        if (onchip_acc_bank_queue_occupancy_[bank] >= cfg_.bank_queue_depth) {
            bank_conflict_blocked = true;
            return false;
        }
        onchip_acc_bank_queue_occupancy_[bank] += 1;
        queue_marked = true;
        updateBankQueueOccupancyMax_();
    } else {
        if (onchip_acc_bank_queue_occupancy_[bank] > 0) {
            bank_conflict_blocked = true;
            return false;
        }
        onchip_acc_bank_queue_occupancy_[bank] = 1;
        queue_marked = true;
        updateBankQueueOccupancyMax_();
    }

    const uint64_t bank_cap = (cfg_.acc_bank_bytes > 0)
                                  ? cfg_.acc_bank_bytes
                                  : ceilDivU64_(cfg_.acc_bytes, static_cast<uint64_t>(std::max<uint32_t>(cfg_.acc_bank_count, 1u)));
    const bool global_fit = onchip_acc_occupancy_bytes_ + tile_c_bytes_ <= cfg_.acc_bytes;
    const bool bank_fit = onchip_acc_bank_occupancy_bytes_[bank] + tile_c_bytes_ <= bank_cap;
    if (global_fit && bank_fit) {
        onchip_acc_occupancy_bytes_ += tile_c_bytes_;
        onchip_acc_bank_occupancy_bytes_[bank] += tile_c_bytes_;
        acc_reserved_tiles_.emplace(key, AccTileAlloc{tile_c_bytes_, bank, queue_marked ? 1u : 0u});
        return true;
    }

    if (queue_marked && onchip_acc_bank_queue_occupancy_[bank] > 0) {
        onchip_acc_bank_queue_occupancy_[bank] -= 1;
    }
    return spill_or_fail(!bank_fit);
}

void TensorWorkload::releaseAccTile_(uint32_t mi, uint32_t ni) {
    if (!cfg_.onchip_model_enable || cfg_.acc_bytes == 0) return;
    const uint64_t key = (static_cast<uint64_t>(mi) << 32) | static_cast<uint64_t>(ni);
    auto it = acc_reserved_tiles_.find(key);
    if (it == acc_reserved_tiles_.end()) return;
    const AccTileAlloc alloc = it->second;
    const uint64_t bytes = alloc.bytes;
    onchip_acc_occupancy_bytes_ = (onchip_acc_occupancy_bytes_ >= bytes) ? (onchip_acc_occupancy_bytes_ - bytes) : 0;
    if (alloc.bank < onchip_acc_bank_occupancy_bytes_.size()) {
        uint64_t& occ = onchip_acc_bank_occupancy_bytes_[alloc.bank];
        occ = (occ >= bytes) ? (occ - bytes) : 0;
    }
    acc_reserved_tiles_.erase(it);
}

bool TensorWorkload::onClockTickTile_(uint64_t now_cycle) {
    // Start iteration when ready (and not blocked by a pending blocking-collective epoch).
    if (!iter_active_) {
        if (cfg_.exec_mode != "program" && cfg_.iterations > 0 && iter_done_ >= cfg_.iterations) return false;

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
                markCollectiveEpochDone_(now_cycle);
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
    bool did_mem = false;
    bool did_compute = false;
    bool did_collective = false;
    bool did_comm = false;
    bool noc_budget_blocked = false;
    uint64_t noc_budget = (cfg_.noc_bandwidth_bytes_per_cycle > 0)
                              ? cfg_.noc_bandwidth_bytes_per_cycle
                              : std::numeric_limits<uint64_t>::max();
    resetOnchipCycleState_(now_cycle);

    if (iter_active_) {
        issueNocTraffic_(now_cycle, noc_budget, did, did_collective, did_comm, noc_budget_blocked);
    }
    if (noc_budget_blocked && !did_collective && !did_comm) {
        tensor_stall_noc_budget_cycles_total_ += 1;
    }

    if (!iter_active_) {
        if (did) active_cycles_++;
        return did;
    }

    tensor_iter_cycles_total_ += 1;

    // DMA budgeting (bytes/cycle); if disabled, issue as much as possible until outstanding limit.
    const uint64_t kUncapped = std::numeric_limits<uint64_t>::max();
    uint64_t budget = dmaBudgetBytesPerCycle_(now_cycle);
    const bool dma_capped = (budget != kUncapped);
    bool blocked_budget = dma_capped && (budget == 0);
    bool blocked_hbm_channel_budget = false;
    bool blocked_outstanding = false;
    bool blocked_onchip_capacity = false;
    bool blocked_onchip_port = false;
    bool blocked_onchip_bank_conflict = false;
    bool blocked_spill_budget = false;
    uint64_t spill_budget = (cfg_.spill_share_noc_budget && cfg_.noc_bandwidth_bytes_per_cycle > 0)
                                ? noc_budget
                                : std::numeric_limits<uint64_t>::max();

    const auto max_out = static_cast<size_t>(cfg_.mem_max_outstanding);
    const bool prefetch = (cfg_.overlap_enable || cfg_.double_buffer);
    auto rollback_ub_alloc = [&](std::vector<std::pair<uint32_t, uint64_t>>& allocs, uint64_t bytes) {
        uint64_t rem = bytes;
        while (rem > 0 && !allocs.empty()) {
            std::pair<uint32_t, uint64_t>& tail = allocs.back();
            const uint64_t cut = std::min<uint64_t>(rem, tail.second);
            if (cut > 0) {
                onchip_ub_occupancy_bytes_ = (onchip_ub_occupancy_bytes_ >= cut) ? (onchip_ub_occupancy_bytes_ - cut) : 0;
                if (tail.first < onchip_ub_bank_occupancy_bytes_.size()) {
                    uint64_t& occ = onchip_ub_bank_occupancy_bytes_[tail.first];
                    occ = (occ >= cut) ? (occ - cut) : 0;
                }
                tail.second -= cut;
                rem -= cut;
            } else {
                rem = 0;
            }
            if (tail.second == 0) allocs.pop_back();
        }
    };
    auto rollback_weight_alloc = [&](std::vector<std::pair<uint32_t, uint64_t>>& allocs, uint64_t bytes) {
        uint64_t rem = bytes;
        while (rem > 0 && !allocs.empty()) {
            std::pair<uint32_t, uint64_t>& tail = allocs.back();
            const uint64_t cut = std::min<uint64_t>(rem, tail.second);
            if (cut > 0) {
                onchip_weight_occupancy_bytes_ =
                    (onchip_weight_occupancy_bytes_ >= cut) ? (onchip_weight_occupancy_bytes_ - cut) : 0;
                if (tail.first < onchip_weight_bank_occupancy_bytes_.size()) {
                    uint64_t& occ = onchip_weight_bank_occupancy_bytes_[tail.first];
                    occ = (occ >= cut) ? (occ - cut) : 0;
                }
                tail.second -= cut;
                rem -= cut;
            } else {
                rem = 0;
            }
            if (tail.second == 0) allocs.pop_back();
        }
    };

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

            uint32_t chunk = static_cast<uint32_t>(std::min<uint64_t>(cfg_.mem_req_bytes, pending));
            if (chunk == 0) break;
            if (chunk > budget) {
                blocked_budget = true;
                break;
            }

            uint32_t hbm_ch = 0;
            chunk = clampBytesByHbmChannelBudget_(now_cycle, ReqKind::Read, chunk, hbm_ch);
            if (chunk == 0) {
                blocked_hbm_channel_budget = true;
                break;
            }

            const bool use_weight_pool = (tag == MemTag::ReadB && cfg_.weight_bytes > 0);
            bool onchip_spilled = false;
            if (cfg_.onchip_model_enable) {
                if (!acquireOnchipWritePorts_(1, 0)) {
                    blocked_onchip_port = true;
                    break;
                }
                bool spill_budget_blocked_local = false;
                bool bank_conflict_blocked_local = false;
                std::vector<std::pair<uint32_t, uint64_t>>* bank_allocs =
                    (cfg_.onchip_bank_model_enable
                         ? ((tag == MemTag::ReadA) ? &tile_cur_.reserved_a_bank_allocs
                                                   : (use_weight_pool ? &tile_cur_.reserved_b_weight_bank_allocs
                                                                      : &tile_cur_.reserved_b_bank_allocs))
                         : nullptr);
                const bool ok = use_weight_pool
                                    ? reserveWeightBytes_(chunk,
                                                          spill_budget,
                                                          onchip_spilled,
                                                          spill_budget_blocked_local,
                                                          bank_conflict_blocked_local,
                                                          bank_allocs)
                                    : reserveUbBytes_(chunk,
                                                      spill_budget,
                                                      onchip_spilled,
                                                      spill_budget_blocked_local,
                                                      bank_conflict_blocked_local,
                                                      bank_allocs);
                if (!ok) {
                    if (spill_budget_blocked_local) {
                        blocked_spill_budget = true;
                    } else if (bank_conflict_blocked_local) {
                        blocked_onchip_bank_conflict = true;
                    } else {
                        blocked_onchip_capacity = true;
                    }
                    break;
                }
            }

            const uint32_t issued = issueMemReadTagged_(chunk, tag, tile_cur_.epoch);
            if (issued == 0) {
                if (cfg_.onchip_model_enable && !onchip_spilled) {
                    if (cfg_.onchip_bank_model_enable) {
                        if (tag == MemTag::ReadA) {
                            rollback_ub_alloc(tile_cur_.reserved_a_bank_allocs, chunk);
                        } else if (tag == MemTag::ReadB) {
                            if (use_weight_pool) {
                                rollback_weight_alloc(tile_cur_.reserved_b_weight_bank_allocs, chunk);
                            } else {
                                rollback_ub_alloc(tile_cur_.reserved_b_bank_allocs, chunk);
                            }
                        }
                    } else {
                        if (use_weight_pool) {
                            releaseWeightBytes_(chunk);
                        } else {
                            releaseUbBytes_(chunk);
                        }
                    }
                }
                if (inflight_.size() >= max_out) blocked_outstanding = true;
                break;
            }
            consumeHbmChannelBudget_(now_cycle, hbm_ch, issued);
            if (cfg_.onchip_model_enable && !onchip_spilled && issued < chunk) {
                const uint64_t rollback = static_cast<uint64_t>(chunk - issued);
                if (cfg_.onchip_bank_model_enable) {
                    if (tag == MemTag::ReadA) {
                        rollback_ub_alloc(tile_cur_.reserved_a_bank_allocs, rollback);
                    } else if (tag == MemTag::ReadB) {
                        if (use_weight_pool) {
                            rollback_weight_alloc(tile_cur_.reserved_b_weight_bank_allocs, rollback);
                        } else {
                            rollback_ub_alloc(tile_cur_.reserved_b_bank_allocs, rollback);
                        }
                    }
                } else {
                    if (use_weight_pool) {
                        releaseWeightBytes_(rollback);
                    } else {
                        releaseUbBytes_(rollback);
                    }
                }
            }
            if (tag == MemTag::ReadA) {
                tile_cur_.issued_a_bytes += issued;
                if (cfg_.onchip_model_enable && !onchip_spilled) {
                    tile_cur_.reserved_a_bytes += issued;
                }
            } else if (tag == MemTag::ReadB) {
                tile_cur_.issued_b_bytes += issued;
                if (cfg_.onchip_model_enable && !onchip_spilled) {
                    if (use_weight_pool) {
                        tile_cur_.reserved_b_weight_bytes += issued;
                    } else {
                        tile_cur_.reserved_b_bytes += issued;
                    }
                }
            }
            rem_read_bytes_ = (rem_read_bytes_ >= issued) ? (rem_read_bytes_ - issued) : 0;
            budget -= issued;
            did = true;
            did_mem = true;
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
            uint32_t chunk = static_cast<uint32_t>(std::min<uint64_t>(cfg_.mem_req_bytes, pending));
            if (chunk == 0) break;
            if (chunk > budget) {
                blocked_budget = true;
                break;
            }

            uint32_t hbm_ch = 0;
            chunk = clampBytesByHbmChannelBudget_(now_cycle, ReqKind::Write, chunk, hbm_ch);
            if (chunk == 0) {
                blocked_hbm_channel_budget = true;
                break;
            }
            if (cfg_.onchip_model_enable && cfg_.acc_bytes > 0) {
                const uint64_t acc_key = (static_cast<uint64_t>(st.mi) << 32) | static_cast<uint64_t>(st.ni);
                if (acc_reserved_tiles_.find(acc_key) != acc_reserved_tiles_.end()) {
                    if (!acquireOnchipReadPorts_(0, 1)) {
                        blocked_onchip_port = true;
                        break;
                    }
                }
            }
            const uint32_t issued = issueMemWriteTagged_(chunk, MemTag::WriteC, epoch);
            if (issued == 0) {
                if (inflight_.size() >= max_out) blocked_outstanding = true;
                break;
            }
            consumeHbmChannelBudget_(now_cycle, hbm_ch, issued);
            st.issued_bytes += issued;
            rem_write_bytes_ = (rem_write_bytes_ >= issued) ? (rem_write_bytes_ - issued) : 0;
            budget -= issued;
            did = true;
            did_mem = true;
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

            uint32_t chunk = static_cast<uint32_t>(std::min<uint64_t>(cfg_.mem_req_bytes, pending));
            if (chunk == 0) break;
            if (chunk > budget) {
                blocked_budget = true;
                break;
            }

            uint32_t hbm_ch = 0;
            chunk = clampBytesByHbmChannelBudget_(now_cycle, ReqKind::Read, chunk, hbm_ch);
            if (chunk == 0) {
                blocked_hbm_channel_budget = true;
                break;
            }

            const bool use_weight_pool = (tag == MemTag::ReadB && cfg_.weight_bytes > 0);
            bool onchip_spilled = false;
            if (cfg_.onchip_model_enable) {
                if (!acquireOnchipWritePorts_(1, 0)) {
                    blocked_onchip_port = true;
                    break;
                }
                bool spill_budget_blocked_local = false;
                bool bank_conflict_blocked_local = false;
                std::vector<std::pair<uint32_t, uint64_t>>* bank_allocs =
                    (cfg_.onchip_bank_model_enable
                         ? ((tag == MemTag::ReadA) ? &tile_next_.reserved_a_bank_allocs
                                                   : (use_weight_pool ? &tile_next_.reserved_b_weight_bank_allocs
                                                                      : &tile_next_.reserved_b_bank_allocs))
                         : nullptr);
                const bool ok = use_weight_pool
                                    ? reserveWeightBytes_(chunk,
                                                          spill_budget,
                                                          onchip_spilled,
                                                          spill_budget_blocked_local,
                                                          bank_conflict_blocked_local,
                                                          bank_allocs)
                                    : reserveUbBytes_(chunk,
                                                      spill_budget,
                                                      onchip_spilled,
                                                      spill_budget_blocked_local,
                                                      bank_conflict_blocked_local,
                                                      bank_allocs);
                if (!ok) {
                    if (spill_budget_blocked_local) {
                        blocked_spill_budget = true;
                    } else if (bank_conflict_blocked_local) {
                        blocked_onchip_bank_conflict = true;
                    } else {
                        blocked_onchip_capacity = true;
                    }
                    break;
                }
            }

            const uint32_t issued = issueMemReadTagged_(chunk, tag, tile_next_.epoch);
            if (issued == 0) {
                if (cfg_.onchip_model_enable && !onchip_spilled) {
                    if (cfg_.onchip_bank_model_enable) {
                        if (tag == MemTag::ReadA) {
                            rollback_ub_alloc(tile_next_.reserved_a_bank_allocs, chunk);
                        } else if (tag == MemTag::ReadB) {
                            if (use_weight_pool) {
                                rollback_weight_alloc(tile_next_.reserved_b_weight_bank_allocs, chunk);
                            } else {
                                rollback_ub_alloc(tile_next_.reserved_b_bank_allocs, chunk);
                            }
                        }
                    } else {
                        if (use_weight_pool) {
                            releaseWeightBytes_(chunk);
                        } else {
                            releaseUbBytes_(chunk);
                        }
                    }
                }
                if (inflight_.size() >= max_out) blocked_outstanding = true;
                break;
            }
            consumeHbmChannelBudget_(now_cycle, hbm_ch, issued);
            if (cfg_.onchip_model_enable && !onchip_spilled && issued < chunk) {
                const uint64_t rollback = static_cast<uint64_t>(chunk - issued);
                if (cfg_.onchip_bank_model_enable) {
                    if (tag == MemTag::ReadA) {
                        rollback_ub_alloc(tile_next_.reserved_a_bank_allocs, rollback);
                    } else if (tag == MemTag::ReadB) {
                        if (use_weight_pool) {
                            rollback_weight_alloc(tile_next_.reserved_b_weight_bank_allocs, rollback);
                        } else {
                            rollback_ub_alloc(tile_next_.reserved_b_bank_allocs, rollback);
                        }
                    }
                } else {
                    if (use_weight_pool) {
                        releaseWeightBytes_(rollback);
                    } else {
                        releaseUbBytes_(rollback);
                    }
                }
            }
            if (tag == MemTag::ReadA) {
                tile_next_.issued_a_bytes += issued;
                if (cfg_.onchip_model_enable && !onchip_spilled) {
                    tile_next_.reserved_a_bytes += issued;
                }
            } else if (tag == MemTag::ReadB) {
                tile_next_.issued_b_bytes += issued;
                if (cfg_.onchip_model_enable && !onchip_spilled) {
                    if (use_weight_pool) {
                        tile_next_.reserved_b_weight_bytes += issued;
                    } else {
                        tile_next_.reserved_b_bytes += issued;
                    }
                }
            }
            rem_read_bytes_ = (rem_read_bytes_ >= issued) ? (rem_read_bytes_ - issued) : 0;
            budget -= issued;
            did = true;
            did_mem = true;
        }
    }

    // === Compute for current tile-seg (requires reads ready) ===
    const bool cur_reads_ready =
        (!tile_cur_.valid) ||
        (tile_cur_.done_a_bytes >= tile_cur_.need_a_bytes && tile_cur_.done_b_bytes >= tile_cur_.need_b_bytes);
    const bool collective_blocks_compute =
        (!cfg_.collective_overlap_with_compute && (collectivePendingActive_() || did_collective));
    bool onchip_compute_ready = true;
    if (tile_cur_.valid && cur_reads_ready && tile_cur_.rem_compute_cycles > 0 && !collective_blocks_compute && cfg_.onchip_model_enable) {
        if (!tile_cur_.acc_reserved) {
            bool acc_spilled = false;
            bool spill_budget_blocked_local = false;
            bool bank_conflict_blocked_local = false;
            if (!reserveAccTile_(tile_cur_.mi,
                                 tile_cur_.ni,
                                 spill_budget,
                                 acc_spilled,
                                 spill_budget_blocked_local,
                                 bank_conflict_blocked_local)) {
                onchip_compute_ready = false;
                if (spill_budget_blocked_local) {
                    blocked_spill_budget = true;
                } else if (bank_conflict_blocked_local) {
                    blocked_onchip_bank_conflict = true;
                } else {
                    blocked_onchip_capacity = true;
                }
            } else {
                tile_cur_.acc_reserved = true;
            }
        }
        if (onchip_compute_ready) {
            const uint32_t ub_ports_needed =
                std::max<uint32_t>(1u, ((tile_a_bytes_ > 0) ? 1u : 0u) + ((tile_b_bytes_ > 0) ? 1u : 0u));
            if (!acquireOnchipReadPorts_(ub_ports_needed, 1)) {
                onchip_compute_ready = false;
                blocked_onchip_port = true;
            }
        }
    }

    if (tile_cur_.valid &&
        cur_reads_ready &&
        tile_cur_.rem_compute_cycles > 0 &&
        !collective_blocks_compute &&
        onchip_compute_ready) {
        if (rem_compute_cycles_ > 0) rem_compute_cycles_--;
        tensor_compute_cycles_total_ += 1;
        if (tile_cur_.rem_compute_math_cycles > 0) {
            tile_cur_.rem_compute_math_cycles--;
            tensor_compute_math_cycles_total_ += 1;

            const uint64_t per = effectivePeakMacsPerCycle_();
            const uint64_t done = (rem_macs_ >= per) ? per : rem_macs_;
            tensor_mac_ops_total_ += done;
            rem_macs_ = (rem_macs_ >= done) ? (rem_macs_ - done) : 0;
        } else if (tile_cur_.rem_compute_pipeline_cycles > 0) {
            tile_cur_.rem_compute_pipeline_cycles--;
            tensor_compute_pipeline_cycles_total_ += 1;
        } else if (tile_cur_.rem_compute_wavefront_cycles > 0) {
            tile_cur_.rem_compute_wavefront_cycles--;
            tensor_mxu_wavefront_cycles_total_ += 1;
        }
        tile_cur_.rem_compute_cycles =
            tile_cur_.rem_compute_math_cycles + tile_cur_.rem_compute_pipeline_cycles + tile_cur_.rem_compute_wavefront_cycles;

        did = true;
        did_compute = true;
    } else if (tile_cur_.valid && cur_reads_ready && tile_cur_.rem_compute_cycles > 0 && collective_blocks_compute) {
        tensor_stall_collective_cycles_total_ += 1;
    }

    // === Tile-seg complete → advance ===
    if (tile_cur_.valid && cur_reads_ready && tile_cur_.rem_compute_cycles == 0) {
        tile_seg_done_ += 1;

        // M15: materialize keep-a/keep-b as real on-chip residency (not just reduced DRAM reads).
        if (cfg_.onchip_model_enable) {
            if (cfg_.dataflow == "is" && tile_keep_a_ && tile_cur_.ni == 0) {
                const uint64_t key = (static_cast<uint64_t>(tile_cur_.mi) << 32) | static_cast<uint64_t>(tile_cur_.ki);
                if (tile_cur_.reserved_a_bytes > 0 || !tile_cur_.reserved_a_bank_allocs.empty()) {
                    ResidentTileAlloc& alloc = a_resident_tiles_[key];
                    alloc.pool = OnchipPoolKind::Ub;
                    alloc.bytes += tile_cur_.reserved_a_bytes;
                    for (const auto& p : tile_cur_.reserved_a_bank_allocs) {
                        alloc.bank_allocs.emplace_back(p);
                    }
                    tile_cur_.reserved_a_bytes = 0;
                    tile_cur_.reserved_a_bank_allocs.clear();
                    if (static_cast<uint64_t>(a_resident_tiles_.size()) > tensor_onchip_a_resident_tiles_max_) {
                        tensor_onchip_a_resident_tiles_max_ = static_cast<uint64_t>(a_resident_tiles_.size());
                    }
                }
            }
            if (cfg_.dataflow == "ws" && tile_keep_b_ && tile_cur_.mi == 0) {
                const uint64_t key = (static_cast<uint64_t>(tile_cur_.ni) << 32) | static_cast<uint64_t>(tile_cur_.ki);
                const bool b_is_weight = (cfg_.weight_bytes > 0);
                const bool has_b_alloc = b_is_weight
                                            ? (tile_cur_.reserved_b_weight_bytes > 0 || !tile_cur_.reserved_b_weight_bank_allocs.empty())
                                            : (tile_cur_.reserved_b_bytes > 0 || !tile_cur_.reserved_b_bank_allocs.empty());
                if (has_b_alloc) {
                    ResidentTileAlloc& alloc = b_resident_tiles_[key];
                    alloc.pool = b_is_weight ? OnchipPoolKind::Weight : OnchipPoolKind::Ub;
                    if (b_is_weight) {
                        alloc.bytes += tile_cur_.reserved_b_weight_bytes;
                        for (const auto& p : tile_cur_.reserved_b_weight_bank_allocs) {
                            alloc.bank_allocs.emplace_back(p);
                        }
                        tile_cur_.reserved_b_weight_bytes = 0;
                        tile_cur_.reserved_b_weight_bank_allocs.clear();
                    } else {
                        alloc.bytes += tile_cur_.reserved_b_bytes;
                        for (const auto& p : tile_cur_.reserved_b_bank_allocs) {
                            alloc.bank_allocs.emplace_back(p);
                        }
                        tile_cur_.reserved_b_bytes = 0;
                        tile_cur_.reserved_b_bank_allocs.clear();
                    }
                    if (static_cast<uint64_t>(b_resident_tiles_.size()) > tensor_onchip_b_resident_tiles_max_) {
                        tensor_onchip_b_resident_tiles_max_ = static_cast<uint64_t>(b_resident_tiles_.size());
                    }
                }
            }
        }

        if (cfg_.onchip_bank_model_enable) {
            releaseUbBankAllocs_(tile_cur_.reserved_a_bank_allocs);
            releaseUbBankAllocs_(tile_cur_.reserved_b_bank_allocs);
            releaseWeightBankAllocs_(tile_cur_.reserved_b_weight_bank_allocs);
        } else {
            releaseUbBytes_(tile_cur_.reserved_a_bytes);
            releaseUbBytes_(tile_cur_.reserved_b_bytes);
            releaseWeightBytes_(tile_cur_.reserved_b_weight_bytes);
        }

        // Release resident tiles at their last reuse point.
        if (cfg_.onchip_model_enable) {
            if (cfg_.dataflow == "is" && tile_keep_a_ && (tile_cur_.ni + 1u == tile_nt_)) {
                const uint64_t key = (static_cast<uint64_t>(tile_cur_.mi) << 32) | static_cast<uint64_t>(tile_cur_.ki);
                auto it = a_resident_tiles_.find(key);
                if (it != a_resident_tiles_.end()) {
                    if (cfg_.onchip_bank_model_enable) {
                        releaseUbBankAllocs_(it->second.bank_allocs);
                    } else {
                        releaseUbBytes_(it->second.bytes);
                    }
                    a_resident_tiles_.erase(it);
                }
            }
            if (cfg_.dataflow == "ws" && tile_keep_b_ && (tile_cur_.mi + 1u == tile_mt_)) {
                const uint64_t key = (static_cast<uint64_t>(tile_cur_.ni) << 32) | static_cast<uint64_t>(tile_cur_.ki);
                auto it = b_resident_tiles_.find(key);
                if (it != b_resident_tiles_.end()) {
                    if (it->second.pool == OnchipPoolKind::Weight) {
                        if (cfg_.onchip_bank_model_enable) {
                            releaseWeightBankAllocs_(it->second.bank_allocs);
                        } else {
                            releaseWeightBytes_(it->second.bytes);
                        }
                    } else {
                        if (cfg_.onchip_bank_model_enable) {
                            releaseUbBankAllocs_(it->second.bank_allocs);
                        } else {
                            releaseUbBytes_(it->second.bytes);
                        }
                    }
                    b_resident_tiles_.erase(it);
                }
            }
        }
        if (tile_cur_.ki + 1u == tile_kt_) {
            scheduleWriteback_(tile_cur_.mi, tile_cur_.ni);
            if (!cfg_.mem_enable || !rt_.mem) {
                releaseAccTile_(tile_cur_.mi, tile_cur_.ni);
            }
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
        !collectivePendingActive_() &&
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
        acc_reserved_tiles_.clear();
        a_resident_tiles_.clear();
        b_resident_tiles_.clear();
        onchip_ub_occupancy_bytes_ = 0;
        onchip_weight_occupancy_bytes_ = 0;
        onchip_acc_occupancy_bytes_ = 0;
        std::fill(onchip_ub_bank_occupancy_bytes_.begin(), onchip_ub_bank_occupancy_bytes_.end(), 0ull);
        std::fill(onchip_weight_bank_occupancy_bytes_.begin(), onchip_weight_bank_occupancy_bytes_.end(), 0ull);
        std::fill(onchip_acc_bank_occupancy_bytes_.begin(), onchip_acc_bank_occupancy_bytes_.end(), 0ull);
        std::fill(onchip_ub_bank_queue_occupancy_.begin(), onchip_ub_bank_queue_occupancy_.end(), 0u);
        std::fill(onchip_weight_bank_queue_occupancy_.begin(), onchip_weight_bank_queue_occupancy_.end(), 0u);
        std::fill(onchip_acc_bank_queue_occupancy_.begin(), onchip_acc_bank_queue_occupancy_.end(), 0u);
        onchip_ub_bank_rr_ = 0;
        onchip_weight_bank_rr_ = 0;
        onchip_acc_bank_rr_ = 0;
        if (step_gated_ && close_step_on_iter_done_) {
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

        if (blocked_onchip_port) {
            tensor_stall_onchip_port_cycles_total_ += 1;
        } else if (blocked_spill_budget) {
            tensor_stall_spill_budget_cycles_total_ += 1;
        } else if (blocked_onchip_bank_conflict) {
            tensor_stall_onchip_bank_conflict_cycles_total_ += 1;
        } else if (blocked_onchip_capacity) {
            tensor_stall_onchip_capacity_cycles_total_ += 1;
        } else if (blocked_hbm_channel_budget && pending_mem_issue) {
            tensor_stall_dma_hbm_channel_budget_cycles_total_ += 1;
        } else if (blocked_outstanding && pending_mem_issue) {
            tensor_stall_mem_outstanding_cycles_total_ += 1;
        } else if (blocked_budget && pending_mem_issue) {
            tensor_stall_dma_budget_cycles_total_ += 1;
        } else if (tile_cur_.valid && !cur_reads_ready && tile_cur_.rem_compute_cycles > 0) {
            tensor_stall_wait_read_cycles_total_ += 1;
        } else if (rem_compute_cycles_ == 0 && (!tile_writeback_queue_.empty() || !inflight_.empty())) {
            tensor_stall_wait_write_cycles_total_ += 1;
        }
    }

    if (cfg_.collective_overlap_with_compute && did_compute && did_collective) {
        tensor_overlap_compute_collective_cycles_total_ += 1;
    }
    if (did_compute && did_mem) {
        tensor_overlap_compute_mem_cycles_total_ += 1;
    }

    if (did) active_cycles_++;
    return did;
}

bool TensorWorkload::onClockTick(uint64_t now_cycle) {
    now_cycle_cached_ = now_cycle;
    total_cycles_++;
    if (!configured_) return false;

    if (cfg_.exec_mode == "program") {
        return onClockTickProgram_(now_cycle);
    }

    bool did_credit_return = false;
    if (!iter_active_) {
        bool credit_budget_blocked = false;
        const uint64_t credit_budget = (cfg_.noc_bandwidth_bytes_per_cycle > 0)
                                           ? cfg_.noc_bandwidth_bytes_per_cycle
                                           : std::numeric_limits<uint64_t>::max();
        const uint64_t credit_sent =
            emitCollectiveCreditReturnTraffic_(now_cycle, credit_budget, credit_budget_blocked);
        did_credit_return = (credit_sent > 0);
        if (credit_budget_blocked && !did_credit_return) {
            tensor_stall_noc_budget_cycles_total_ += 1;
        }
    }

    // Step-gated mode: only run when a step is open.
    if (step_gated_ && !step_open_) {
        if (did_credit_return) active_cycles_++;
        return did_credit_return;
    }

    if (!started_) {
        if (now_cycle < cfg_.start_cycle) {
            if (did_credit_return) active_cycles_++;
            return did_credit_return;
        }
        started_ = true;
    }

    serviceCollectiveBarrier_(now_cycle);

    if (cfg_.exec_mode == "tile") {
        const bool did_tile = onClockTickTile_(now_cycle);
        if (did_credit_return && !did_tile) active_cycles_++;
        return did_credit_return || did_tile;
    }

    if (!iter_active_) {
        if (cfg_.exec_mode != "program" && cfg_.iterations > 0 && iter_done_ >= cfg_.iterations) {
            if (did_credit_return) active_cycles_++;
            return did_credit_return;
        }
        if (cfg_.collective_blocking && collective_epoch_active_) {
            if (cfg_.collective_scope == "per_core") {
                if (collective_epoch_recv_bytes_ < collective_epoch_expected_recv_bytes_) {
                    tensor_stall_collective_cycles_total_ += 1;
                    if (did_credit_return) active_cycles_++;
                    return did_credit_return;
                }
                markCollectiveEpochDone_(now_cycle);
                const uint32_t done_seq = collective_epoch_seq_;
                collective_epoch_active_ = false;
                collective_epoch_expected_recv_bytes_ = 0;
                collective_epoch_recv_bytes_ = 0;
                collective_recv_bytes_by_seq_.erase(done_seq);
            } else {
                tensor_stall_collective_cycles_total_ += 1;
                if (did_credit_return) active_cycles_++;
                return did_credit_return;
            }
        }
        if (inflight_.empty()) {
            startIteration_();
        }
    }

    bool did = did_credit_return;
    bool did_mem = false;
    bool did_compute = false;
    bool did_collective = false;
    bool did_comm = false;
    bool noc_budget_blocked = false;
    uint64_t noc_budget = (cfg_.noc_bandwidth_bytes_per_cycle > 0)
                              ? cfg_.noc_bandwidth_bytes_per_cycle
                              : std::numeric_limits<uint64_t>::max();

    // === DMA reads ===
    const bool dma_shared = (cfg_.dma_shared_bandwidth_bytes_per_cycle > 0);
    uint64_t shared_budget_left = dma_shared ? dmaBudgetBytesPerCycle_(now_cycle) : std::numeric_limits<uint64_t>::max();
    if (dma_shared) {
        uint64_t& budget = shared_budget_left;
        while (rem_read_bytes_ > 0) {
            const uint32_t next_bytes = static_cast<uint32_t>(std::min<uint64_t>(cfg_.mem_req_bytes, rem_read_bytes_));
            if (next_bytes == 0 || next_bytes > budget) break;
            const uint32_t issued = issueMemRead_();
            if (issued == 0) break;
            budget -= issued;
            did = true;
            did_mem = true;
        }
    } else if (tileModelEnabled_() && cfg_.dma_bandwidth_bytes_per_cycle > 0) {
        uint64_t budget = cfg_.dma_bandwidth_bytes_per_cycle;
        while (rem_read_bytes_ > 0) {
            const uint32_t next_bytes = static_cast<uint32_t>(std::min<uint64_t>(cfg_.mem_req_bytes, rem_read_bytes_));
            if (next_bytes == 0 || next_bytes > budget) break;
            const uint32_t issued = issueMemRead_();
            if (issued == 0) break;
            budget -= issued;
            did = true;
            did_mem = true;
        }
    } else {
        while (issueMemRead_()) {
            did = true;
            did_mem = true;
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
    const bool collective_blocks_compute = (!cfg_.collective_overlap_with_compute && collectivePendingActive_());
    if (!collective_blocks_compute && tickCompute_()) {
        did = true;
        did_compute = true;
    } else if (collective_blocks_compute && rem_compute_cycles_ > 0) {
        tensor_stall_collective_cycles_total_ += 1;
    }

    // === DMA writes (after compute completes) ===
    if (rem_compute_cycles_ == 0) {
        if (dma_shared) {
            uint64_t& budget = shared_budget_left;
            while (rem_write_bytes_ > 0) {
                const uint32_t next_bytes = static_cast<uint32_t>(std::min<uint64_t>(cfg_.mem_req_bytes, rem_write_bytes_));
                if (next_bytes == 0 || next_bytes > budget) break;
                const uint32_t issued = issueMemWrite_();
                if (issued == 0) break;
                budget -= issued;
                did = true;
                did_mem = true;
            }
        } else if (tileModelEnabled_() && cfg_.dma_bandwidth_bytes_per_cycle > 0) {
            uint64_t budget = cfg_.dma_bandwidth_bytes_per_cycle;
            while (rem_write_bytes_ > 0) {
                const uint32_t next_bytes = static_cast<uint32_t>(std::min<uint64_t>(cfg_.mem_req_bytes, rem_write_bytes_));
                if (next_bytes == 0 || next_bytes > budget) break;
                const uint32_t issued = issueMemWrite_();
                if (issued == 0) break;
                budget -= issued;
                did = true;
                did_mem = true;
            }
        } else {
            while (issueMemWrite_()) {
                did = true;
                did_mem = true;
            }
        }
    }

    if (iter_active_) {
        issueNocTraffic_(now_cycle, noc_budget, did, did_collective, did_comm, noc_budget_blocked);
    }

    if (noc_budget_blocked && !did_collective && !did_comm) {
        tensor_stall_noc_budget_cycles_total_ += 1;
    }

    // === Iteration complete ===
    if (iter_active_ &&
        rem_read_bytes_ == 0 &&
        rem_write_bytes_ == 0 &&
        rem_compute_cycles_ == 0 &&
        !collectivePendingActive_() &&
        inflight_.empty()) {
        iter_done_++;
        iter_active_ = false;
        compute_started_ = false;
        if (step_gated_ && close_step_on_iter_done_) {
            // In barrier/step-gated runs, execute at most one iteration per step.
            step_open_ = false;
        }
    }

    if (!did && (iter_active_ || !inflight_.empty())) {
        tensor_dma_stall_cycles_total_ += 1;
    }

    if (cfg_.collective_overlap_with_compute && did_compute && did_collective) {
        tensor_overlap_compute_collective_cycles_total_ += 1;
    }
    if (did_compute && did_mem) {
        tensor_overlap_compute_mem_cycles_total_ += 1;
    }

    if (did) active_cycles_++;
    return did;
}


}} // namespace SST::SnnDL
