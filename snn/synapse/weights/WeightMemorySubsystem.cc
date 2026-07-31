// -*- c++ -*-
//
// WeightMemorySubsystem.cc:
// Phase E: 将 StandardMem pending/回调/解析/BCSR 缓存等数据路径收敛到内存子系统，
// 控制层仅保留 GAS/窗口编排与统计汇总。

#include <sst/core/sst_config.h>

#include "WeightMemorySubsystem.h"

#include "SnnBcsrWeightManager.h"
#include "SnnDLStringUtil.h"
#include "research/local_storage/PeInternalPodShadowGate.h"
#include "research/local_storage/PodOwnerServiceTable.h"

#include <sst/core/output.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace SST;
using namespace SST::SnnDL;

namespace {
// NOTE(Universal-core experiments):
// Keep BCSR cache/prefetch/populate optimizations disabled so GAS remains the only
// externally credited optimizer; keep in-flight coalescing available via runtime
// config for request dedup and stable window forward progress.
static constexpr bool kEnableBcsrOptimizations = false;
static constexpr bool kEnableBcsrInflightCoalescing = true;

static inline bool isPow2(uint32_t v) { return v != 0 && (v & (v - 1u)) == 0; }
static inline uint64_t alignDownU64(uint64_t v, uint64_t a) { return a ? (v & ~(a - 1u)) : v; }
static inline uint64_t alignUpU64(uint64_t v, uint64_t a) {
    if (!a) return v;
    const uint64_t m = a - 1u;
    return (v + m) & ~m;
}

static inline void prepareAlignedRead(uint64_t addr, size_t size, uint32_t line_size,
                                      uint64_t& out_addr, size_t& out_size, size_t& out_slice_off) {
    out_addr = addr;
    out_size = size;
    out_slice_off = 0;
    if (size == 0) return;
    if (!isPow2(line_size)) return;
    const uint64_t ls = static_cast<uint64_t>(line_size);
    const uint64_t end = addr + static_cast<uint64_t>(size);
    const uint64_t a0 = alignDownU64(addr, ls);
    const uint64_t a1 = alignUpU64(end, ls);
    if (a1 <= a0) return;
    out_addr = a0;
    out_size = static_cast<size_t>(a1 - a0);
    out_slice_off = static_cast<size_t>(addr - a0);
}

static std::string resolveWeightsTemplatePath(const std::string& tmpl, uint32_t pe, uint32_t core) {
    if (tmpl.empty()) return "";
    return resolvePeCoreTemplate(tmpl, pe, core);
}

static bool readFileSlice(const std::string& path, uint64_t offset, size_t bytes, std::vector<uint8_t>& out) {
    out.clear();
    if (bytes == 0) return false;
    std::ifstream f(path, std::ios::in | std::ios::binary);
    if (!f.is_open()) return false;
    f.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    if (!f.good()) return false;
    out.resize(bytes, 0);
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(bytes));
    return static_cast<size_t>(f.gcount()) == bytes;
}

struct RowIndexPrefetchPolicy {
    std::string mode;
    uint32_t block_rows = 0;
    uint32_t n_block_rows = 0;
    bool enabled = false;
    bool auto_all_rows = false;
};

static RowIndexPrefetchPolicy computeRowIndexPrefetchPolicy(const WeightMemorySubsystem::OrchestratorConfig& orch) {
    RowIndexPrefetchPolicy p{};
    if (!orch.use_bcsr || !orch.bcsr_mgr) return p;
    if (!orch.bcsr_mgr->isRowptrReady()) return p;

    p.mode = toLowerCopy(orch.bcsr_row_index_prefetch_mode);
    if (p.mode.empty()) p.mode = "auto";
    if (p.mode == "off") return p;

    p.block_rows = orch.bcsr_mgr->effectiveBlockRows();
    p.n_block_rows =
        p.block_rows ? ((orch.num_neurons + p.block_rows - 1u) / p.block_rows) : static_cast<uint32_t>(orch.num_neurons);
    if (p.n_block_rows == 0) return p;

    p.enabled = true;
    if (p.mode == "auto" && p.n_block_rows <= orch.bcsr_row_index_prefetch_all_rows_threshold) {
        uint32_t last_start = 0;
        uint32_t last_end = 0;
        if (orch.bcsr_mgr->rowBounds(p.n_block_rows - 1u, last_start, last_end) && last_end > 0) {
            const size_t colidx_bytes = orch.bcsr_mgr->colIndexBytes(last_end);
            p.auto_all_rows = (colidx_bytes <= orch.bcsr_row_index_prefetch_all_rows_max_bytes);
        }
    }
    return p;
}
} // namespace

void WeightMemorySubsystem::onClockTick(uint64_t now_cycle) {
    setNowCycle(now_cycle);
    if (usePerPostRetire_()) {
        tryRetireEdgesPerPost_();
    }
    noteShadowRecoverableOnTick_();
    drainShadowPerPostRetire_();
    updateRetireHolStatsOnTick_();
    if (weight_sram_enable_) {
        idx_sram_model_.onClockTick(now_cycle);
        l0_sram_model_.onClockTick(now_cycle);
        weight_sram_stall_budget_cycles_ = std::min<uint64_t>(
            std::numeric_limits<uint64_t>::max(),
            weight_sram_stall_budget_cycles_ + idx_sram_model_.consumeLastCyclePredictedExtraCycles());
        weight_sram_stall_budget_cycles_ = std::min<uint64_t>(
            std::numeric_limits<uint64_t>::max(),
            weight_sram_stall_budget_cycles_ + l0_sram_model_.consumeLastCyclePredictedExtraCycles());
    }
    if (shared_weight_object_plane_) {
        shared_weight_object_plane_->onClockTick(now_cycle);
    }
    if (pe_local_service_object_table_ && pe_local_service_object_table_->enabled()) {
        pe_local_service_object_table_->onClockTick(now_cycle);
    }
    if (weight_sram_stall_budget_cycles_ > 0) {
        --weight_sram_stall_budget_cycles_;
        weight_sram_stall_cycles_total_ = std::min<uint64_t>(
            std::numeric_limits<uint64_t>::max(),
            weight_sram_stall_cycles_total_ + 1ull);
        if (!observeOnlyWeightSramStall_()) {
            return;
        }
    }
    // BCSR rowptr 预取与后续 prefetchAll 都在内存子系统内闭环
    maybeIssueBcsrRowptrPrefetch_();
    drainPendingReads_();
    drainRowIndexPrefetch_();
    drainPendingDirectReads_();
    drainPendingBcsrRowptrWaiters_();
}

void WeightMemorySubsystem::noteIdxSramReadMirror_(uint64_t addr, size_t bytes) {
    if (weight_sram_enable_ && weight_idx_sram_enable_) {
        idx_sram_model_.noteRead(now_cycle_, addr, bytes);
    }
    if (shared_weight_object_plane_) {
        shared_weight_object_plane_->noteIdxRead(now_cycle_, addr, bytes);
    }
}

void WeightMemorySubsystem::noteIdxSramResidentMirror_(uint64_t bytes) {
    if (shared_weight_object_plane_ && shared_weight_object_plane_residency_authority_) {
        shared_weight_object_plane_->noteResidentIdxBytes(bytes);
        return;
    }
    if (weight_sram_enable_ && weight_idx_sram_enable_) {
        idx_sram_model_.noteResidentBytes(bytes);
    }
    if (shared_weight_object_plane_) {
        shared_weight_object_plane_->noteResidentIdxBytes(bytes);
    }
}

void WeightMemorySubsystem::noteL0SramReadMirror_(uint64_t addr) {
    if (weight_sram_enable_ && weight_l0_sram_enable_) {
        l0_sram_model_.noteRead(now_cycle_, sram_layout_.l0SlotAddr(addr), sizeof(float));
    }
    if (shared_weight_object_plane_) {
        shared_weight_object_plane_->noteL0Read(now_cycle_, addr);
    }
}

void WeightMemorySubsystem::noteL0SramWriteMirror_(uint64_t addr) {
    if (weight_sram_enable_ && weight_l0_sram_enable_) {
        l0_sram_model_.noteWrite(now_cycle_, sram_layout_.l0SlotAddr(addr), sizeof(float));
    }
    if (shared_weight_object_plane_) {
        shared_weight_object_plane_->noteL0Write(now_cycle_, addr);
    }
}

void WeightMemorySubsystem::noteL0SramResidentMirror_(uint64_t bytes) {
    if (shared_weight_object_plane_ && shared_weight_object_plane_residency_authority_) {
        shared_weight_object_plane_->noteResidentL0Bytes(bytes);
        return;
    }
    if (weight_sram_enable_ && weight_l0_sram_enable_) {
        l0_sram_model_.noteResidentBytes(bytes);
    }
    if (shared_weight_object_plane_) {
        shared_weight_object_plane_->noteResidentL0Bytes(bytes);
    }
}

WeightMemorySubsystem::IssueStatus
WeightMemorySubsystem::tryIssueRead_(PendingMeta meta, bool count_budget, bool budget_reserved) {
    if (!mem_access_ || meta.size == 0) {
        if (meta.has_single_cb && meta.single_cb) meta.single_cb(0.0f);
        if (meta.has_bytes_cb && meta.bytes_cb) { const std::vector<uint8_t> empty{}; meta.bytes_cb(empty); }
        return IssueStatus::Failed;
    }

    const bool in_window_budget = count_budget && (window_seq_ != 0);
    const bool budget_limited =
        (!budget_reserved) &&
        in_window_budget &&
        (window_.budget != 0) &&
        (window_.issued + 1u > window_.budget);
    // Semantic seal: window_read_budget is a soft scheduling hint only.
    // Correctness must not depend on whether requests were coalesced.
    (void)budget_limited;

    const bool inflight_ok =
        (window_.max_outstanding == 0u) ||
        (window_.outstanding + 1u <= window_.max_outstanding);
    if (!inflight_ok) {
        // Hard throttle remains max_outstanding.
        return IssueStatus::DeferredInflight;
    }

    // Issue now: update per-window issued counter for observability and inflight accounting.
    if (!budget_reserved && in_window_budget) {
        window_.issued += 1u;
    }
    window_.outstanding += 1u;
    if (window_.outstanding > window_.peak_outstanding) {
        window_.peak_outstanding = window_.outstanding;
    }
    if (orch_.update_pending_peak) orch_.update_pending_peak(peakOutstanding());

    const size_t tracked_retire_seq = meta.retire_seq;
    const int tracked_kind = meta.bcsr_kind;
    meta.counted_inflight = true;
    const uint64_t req = issueRead_(std::move(meta));
    if (req != 0) {
        if (tracked_kind == 4 && tracked_retire_seq != std::numeric_limits<size_t>::max()) {
            setEdgeRetireIssued_(tracked_retire_seq);
        }
        return IssueStatus::Issued;
    }

    // Roll back accounting on issue failure.
    if (!budget_reserved && in_window_budget) {
        if (window_.issued > 0) window_.issued -= 1u;
        else window_.issued = 0;
    }
    if (window_.outstanding > 0) window_.outstanding -= 1u;
    else window_.outstanding = 0;
    return IssueStatus::Failed;
}

void WeightMemorySubsystem::drainPendingReads_() {
    if (drain_pending_in_progress_) return;
    drain_pending_in_progress_ = true;

    auto fail_waiters = [](auto& waiters) {
        for (auto& ww : waiters) {
            if (ww.cb) ww.cb(0.0f);
        }
    };

    auto drain_inflight_queue =
        [this, &fail_waiters](auto& pending_queue, auto& inflight_map, auto&& make_meta, auto&& on_issued) -> bool {
        while (!pending_queue.empty()) {
            const uint64_t inflight_key = pending_queue.front();
            auto it = inflight_map.find(inflight_key);
            if (it == inflight_map.end()) {
                pending_queue.pop_front();
                continue;
            }
            auto& inflight = it->second;
            if (inflight.issued) {
                inflight.queued = false;
                pending_queue.pop_front();
                continue;
            }
            if (!orch_.use_bcsr || !orch_.bcsr_mgr) {
                auto waiters = std::move(inflight.waiters);
                inflight_map.erase(it);
                pending_queue.pop_front();
                fail_waiters(waiters);
                continue;
            }

            PendingMeta meta = make_meta(inflight);
            const IssueStatus st = tryIssueRead_(std::move(meta), inflight.count_budget, /*budget_reserved*/true);
            if (st == IssueStatus::Issued) {
                inflight.issued = true;
                inflight.queued = false;
                pending_queue.pop_front();
                on_issued();
                return true;
            }
            if (st == IssueStatus::DeferredInflight) return false;
            if (st == IssueStatus::DeferredBudget) return false;

            // Failed
            auto waiters = std::move(inflight.waiters);
            inflight_map.erase(it);
            pending_queue.pop_front();
            fail_waiters(waiters);
            return true;
        }
        return false;
    };

    auto drain_colidx = [this, &drain_inflight_queue]() -> bool {
        auto make_meta = [this](const ColidxInflight& inflight) -> PendingMeta {
            const uint32_t block_count = (inflight.row_end > inflight.row_start)
                                             ? (inflight.row_end - inflight.row_start)
                                             : 0;
            const size_t bytes = orch_.bcsr_mgr ? orch_.bcsr_mgr->colIndexBytes(block_count) : 0;
            const uint64_t addr = orch_.bcsr_mgr ? orch_.bcsr_mgr->colIndexAddr(inflight.row_start) : 0;

            PendingMeta meta{};
            meta.window_seq = inflight.window_seq;
            meta.address = addr;
            meta.size = bytes;
            meta.orig_address = addr;
            meta.orig_size = bytes;
            meta.slice_offset = 0;
            meta.issue_cycle = now_cycle_;
            meta.bcsr_kind = 2;
            meta.bcsr_block_row = inflight.block_row;
            meta.bcsr_target_block_col = UINT32_MAX;
            meta.bcsr_row_start = inflight.row_start;
            meta.bcsr_prefetch_all = false;
            meta.has_single_cb = false;
            meta.is_weight = true;
            meta.count_weight_read = false;
            return meta;
        };

        auto on_issued = [this]() {
            if (diag_debug_ && diag_window_active_) diag_win_.rowidx_miss += 1;
        };

        return drain_inflight_queue(pending_colidx_reads_, inflight_colidx_, make_meta, on_issued);
    };

    auto drain_block = [this, &drain_inflight_queue]() -> bool {
        auto make_meta = [this](const BlockInflight& inflight) -> PendingMeta {
            const size_t block_bytes = orch_.bcsr_mgr ? orch_.bcsr_mgr->blockBytes() : 0;
            const size_t row_bytes = orch_.bcsr_mgr ? orch_.bcsr_mgr->blockRowBytes() : 0;
            uint64_t addr = 0;
            size_t req_orig_size = block_bytes;
            if (orch_.bcsr_mgr) {
                addr = orch_.bcsr_mgr->blockDataAddrByRow(inflight.block_row, inflight.idx_in_row);
                if (bcsr_block_fetch_mode_ == BcsrBlockFetchMode::RowCacheline) {
                    addr += static_cast<uint64_t>(inflight.intra_row) * static_cast<uint64_t>(row_bytes);
                    req_orig_size = row_bytes;
                }
            }
            uint64_t req_addr = addr;
            size_t req_size = req_orig_size;
            size_t slice_off = 0;
            prepareAlignedRead(addr, req_orig_size, orch_.line_size_bytes, req_addr, req_size, slice_off);

            PendingMeta meta{};
            meta.window_seq = inflight.window_seq;
            meta.address = req_addr;
            meta.size = req_size;
            meta.orig_address = addr;
            meta.orig_size = req_orig_size;
            meta.slice_offset = slice_off;
            meta.issue_cycle = now_cycle_;
            meta.bcsr_kind = 3;
            meta.bcsr_block_row = inflight.block_row;
            meta.bcsr_target_block_col = inflight.block_col;
            meta.bcsr_idx_in_row = inflight.idx_in_row;
            meta.bcsr_intra_row = inflight.intra_row;
            meta.bcsr_global_block_index = inflight.global_block_index;
            meta.bcsr_row_slice_fetch = (bcsr_block_fetch_mode_ == BcsrBlockFetchMode::RowCacheline);
            meta.bcsr_prefetch_all = false;
            meta.has_single_cb = false;
            meta.is_weight = true;
            meta.count_weight_read = true;
            return meta;
        };

        auto on_issued = [this]() {
            if (diag_debug_ && diag_window_active_) diag_win_.block_miss += 1;
        };

        return drain_inflight_queue(pending_block_reads_, inflight_block_, make_meta, on_issued);
    };

    // Priority: colidx first (unblocks many blocks), then blockdata.
    while (true) {
        bool progressed = false;
        progressed |= drain_colidx();
        progressed |= drain_block();
        if (!progressed) break;
        if (window_.max_outstanding != 0 && window_.outstanding >= window_.max_outstanding) break;
    }

    drain_pending_in_progress_ = false;
}

void WeightMemorySubsystem::drainPendingDirectReads_() {
    if (pending_direct_reads_.empty()) return;

    while (!pending_direct_reads_.empty()) {
        if (window_.max_outstanding != 0 && window_.outstanding >= window_.max_outstanding) break;

        PendingMeta meta = std::move(pending_direct_reads_.front());
        pending_direct_reads_.pop_front();

        // Direct reads can be used by naive baseline and by BCSR paths when coalescing is disabled.
        const bool count_budget = (window_seq_ != 0);
        const IssueStatus st = tryIssueRead_(meta, /*count_budget*/count_budget, /*budget_reserved*/false);
        if (st == IssueStatus::Issued) {
            continue;
        }
        if (st == IssueStatus::DeferredInflight || st == IssueStatus::DeferredBudget) {
            pending_direct_reads_.push_front(std::move(meta));
            break;
        }
        // Failed: preserve forward progress by failing the request (callback returns 0.0f).
        if (meta.has_single_cb && meta.single_cb) meta.single_cb(0.0f);
        if (meta.has_bytes_cb && meta.bytes_cb) {
            const std::vector<uint8_t> empty{};
            meta.bytes_cb(empty);
        }
    }
}

bool WeightMemorySubsystem::experimentalIdx2IngressAllowApplyCarry_() const {
    return false;
}

bool WeightMemorySubsystem::shouldTailGuardIdx2RespDrop_() const {
    return false;
}

bool WeightMemorySubsystem::observeOnlyWeightSramStall_() const {
    return false;
}

uint32_t WeightMemorySubsystem::computeExperimentalIdx2IngressBudget_() {
    return 0u;
}

uint32_t WeightMemorySubsystem::experimentalIdx2IngressPrefetchMaxInflight_() const {
    return 0u;
}

void WeightMemorySubsystem::maybeEnqueueRowIndexPrefetchAllRows_() {
    if (!kEnableBcsrOptimizations) return;
    const auto p = computeRowIndexPrefetchPolicy(orch_);
    if (!p.enabled) return;

    const bool do_all = (p.mode == "all_rows") || p.auto_all_rows;
    if (!do_all) return;
    if (row_index_prefetch_all_done_) return;

    // Bulk prefetch marker: drainRowIndexPrefetch_ will issue a single colidx read covering all block_rows.
    row_index_prefetch_rows_.push_back(UINT32_MAX);
    row_index_prefetch_all_done_ = true;
    row_index_prefetch_bulk_pending_ = true;
}

void WeightMemorySubsystem::maybeEnqueueRowIndexPrefetchPostsPrev_() {
    if (!kEnableBcsrOptimizations) return;
    const auto p = computeRowIndexPrefetchPolicy(orch_);
    if (!p.enabled) return;

    const bool do_posts_prev = (p.mode == "posts_prev") || (p.mode == "auto" && !p.auto_all_rows);
    if (!do_posts_prev) return;

    const auto& posts = postsPrev();
    if (posts.empty()) return;

    std::vector<uint8_t> seen(p.n_block_rows, 0);
    std::vector<uint32_t> rows;
    rows.reserve(std::min<size_t>(posts.size(), p.n_block_rows));
    for (uint32_t post_local : posts) {
        const uint32_t block_row = p.block_rows ? (post_local / p.block_rows) : post_local;
        if (block_row >= p.n_block_rows) continue;
        if (seen[block_row]) continue;
        seen[block_row] = 1;
        rows.push_back(block_row);
    }
    std::sort(rows.begin(), rows.end());
    for (uint32_t r : rows) row_index_prefetch_rows_.push_back(r);
}

void WeightMemorySubsystem::drainRowIndexPrefetch_() {
    if (!kEnableBcsrOptimizations) return;
    const auto p = computeRowIndexPrefetchPolicy(orch_);
    if (!p.enabled) return;

    while (!row_index_prefetch_rows_.empty()) {
        if (window_.max_outstanding != 0 && window_.outstanding >= window_.max_outstanding) break;
        const uint32_t block_row = row_index_prefetch_rows_.front();
        row_index_prefetch_rows_.pop_front();

        if (block_row == UINT32_MAX) {
            if (row_index_prefetch_bulk_inflight_) continue;
            uint32_t last_start = 0;
            uint32_t last_end = 0;
            if (!orch_.bcsr_mgr->rowBounds(p.n_block_rows - 1u, last_start, last_end)) continue;
            const uint32_t total_blocks = last_end;
            if (total_blocks == 0) continue;

            const size_t bytes = orch_.bcsr_mgr->colIndexBytes(total_blocks);
            const uint64_t addr = orch_.bcsr_mgr->colIndexAddr(0);

            PendingMeta meta{};
            meta.window_seq = window_seq_;
            meta.address = addr;
            meta.size = bytes;
            meta.orig_address = addr;
            meta.orig_size = bytes;
            meta.slice_offset = 0;
            meta.issue_cycle = now_cycle_;
            meta.bcsr_kind = 2;
            meta.bcsr_block_row = UINT32_MAX;
            meta.bcsr_row_start = 0;
            meta.bcsr_target_block_col = UINT32_MAX;
            meta.bcsr_prefetch_all = false;
            meta.bcsr_colidx_bulk_all_rows = true;
            meta.has_single_cb = false;
            meta.is_weight = true;
            meta.count_weight_read = false;

            const IssueStatus st = tryIssueRead_(std::move(meta), /*count_budget*/false, /*budget_reserved*/false);
            if (st == IssueStatus::Issued) {
                row_index_prefetch_bulk_pending_ = false;
                row_index_prefetch_bulk_inflight_ = true;
                continue;
            }
            if (st == IssueStatus::DeferredInflight) {
                row_index_prefetch_rows_.push_front(UINT32_MAX);
                break;
            }
            // Failed/DeferredBudget (shouldn't happen): retry later to preserve forward progress.
            row_index_prefetch_rows_.push_front(UINT32_MAX);
            break;
        }

        std::vector<uint32_t> cached_cols;
        if (orch_.bcsr_mgr->rowIndexGet(block_row, cached_cols)) continue;

        uint32_t start = 0;
        uint32_t end = 0;
        if (!orch_.bcsr_mgr->rowBounds(block_row, start, end)) continue;
        if (end <= start) continue;

        const uint32_t block_count = end - start;
        const size_t bytes = orch_.bcsr_mgr->colIndexBytes(block_count);
        const uint64_t addr = orch_.bcsr_mgr->colIndexAddr(start);

        PendingMeta meta{};
        meta.window_seq = window_seq_;
        meta.address = addr;
        meta.size = bytes;
        meta.orig_address = addr;
        meta.orig_size = bytes;
        meta.slice_offset = 0;
        meta.issue_cycle = now_cycle_;
        meta.bcsr_kind = 2;
        meta.bcsr_block_row = block_row;
        meta.bcsr_row_start = start;
        meta.bcsr_target_block_col = UINT32_MAX; // prefetch-only: do not request a specific block
        meta.bcsr_prefetch_all = false;
        meta.has_single_cb = false;
        meta.is_weight = true;
        meta.count_weight_read = false;

        const IssueStatus st = tryIssueRead_(std::move(meta), /*count_budget*/false, /*budget_reserved*/false);
        if (st == IssueStatus::Issued) continue;
        if (st == IssueStatus::DeferredInflight) {
            row_index_prefetch_rows_.push_front(block_row);
            break;
        }
        // Failed/DeferredBudget: drop this row and continue (budget should not apply to prefetch).
    }
}

void WeightMemorySubsystem::maybeAutoTuneBlockCache_() {
    if (!kEnableBcsrOptimizations) return;
    if (!orch_.use_bcsr || !orch_.bcsr_mgr) return;
    if (!orch_.bcsr_block_cache_auto_tune) return;
    if (orch_.bcsr_block_cache_max_bytes == 0) return;

    const uint32_t misses = block_miss_window_;
    const uint32_t hits = block_hit_window_;
    const uint32_t total = hits + misses;
    if (total == 0) return;
    if (misses < orch_.bcsr_block_cache_tune_min_misses) return;

    const float ratio = static_cast<float>(misses) / static_cast<float>(total);
    if (ratio < orch_.bcsr_block_cache_tune_miss_ratio) return;

    const size_t block_bytes = orch_.bcsr_mgr->blockBytes();
    if (block_bytes == 0) return;
    const uint64_t max_entries_u64 = orch_.bcsr_block_cache_max_bytes / static_cast<uint64_t>(block_bytes);
    if (max_entries_u64 == 0) return;

    uint32_t cap = orch_.bcsr_mgr->blockCacheCapacity();
    if (cap == 0) return; // respect explicit disable

    const uint32_t max_cap =
        (max_entries_u64 > static_cast<uint64_t>(UINT32_MAX))
            ? UINT32_MAX
            : static_cast<uint32_t>(max_entries_u64);
    if (cap >= max_cap) return;

    uint32_t new_cap = cap;
    const uint32_t grow2 = (cap > (UINT32_MAX / 2u)) ? UINT32_MAX : (cap * 2u);
    new_cap = std::max<uint32_t>(grow2, cap + 64u);
    if (new_cap > max_cap) new_cap = max_cap;
    if (new_cap <= cap) return;

    orch_.bcsr_mgr->setBlockCacheCapacity(new_cap);

    static int dbg_tune = 0;
    if (diag_out_ && diag_out_->getVerboseLevel() >= 2 &&
        diag_node_id_ == 0 && diag_core_id_ == 0 && dbg_tune < 32) {
        diag_out_->verbose(CALL_INFO, 2, 0,
            "[bcsr] auto-tune block_cache_cap %u -> %u (miss=%u hit=%u ratio=%.3f max_bytes=%" PRIu64 " block_bytes=%zu)\n",
            cap, new_cap, misses, hits, ratio,
            static_cast<uint64_t>(orch_.bcsr_block_cache_max_bytes),
            block_bytes);
        ++dbg_tune;
    }
}

void WeightMemorySubsystem::noteReadSourceIssue_(int bcsr_kind, size_t req_bytes) {
    const uint64_t bytes = static_cast<uint64_t>(req_bytes);
    switch (bcsr_kind) {
        case 1:
            read_src_rowptr_reqs_ += 1;
            read_src_rowptr_bytes_ += bytes;
            break;
        case 2:
            read_src_colidx_reqs_ += 1;
            read_src_colidx_bytes_ += bytes;
            break;
        case 3:
            read_src_blockdata_reqs_ += 1;
            read_src_blockdata_bytes_ += bytes;
            break;
        default:
            read_src_dense_reqs_ += 1;
            read_src_dense_bytes_ += bytes;
            break;
    }
}

uint64_t WeightMemorySubsystem::issueRead_(PendingMeta meta) {
    if (!mem_access_ || meta.size == 0) {
        if (meta.has_single_cb && meta.single_cb) meta.single_cb(0.0f);
        if (meta.has_bytes_cb && meta.bytes_cb) { const std::vector<uint8_t> empty{}; meta.bytes_cb(empty); }
        return 0;
    }
    // Window diagnostic (debug-only): attribute raw issue traffic to current window.
    if (diag_debug_ && diag_window_active_ && diag_out_) {
        diag_win_.issue_cnt_total += 1;
        diag_win_.issue_bytes_total += static_cast<uint64_t>(meta.size);
        switch (meta.bcsr_kind) {
            case 0:
                diag_win_.issue_cnt_dense += 1;
                diag_win_.issue_bytes_dense += static_cast<uint64_t>(meta.size);
                break;
            case 1:
                diag_win_.issue_cnt_rowptr += 1;
                diag_win_.issue_bytes_rowptr += static_cast<uint64_t>(meta.size);
                break;
            case 2:
                diag_win_.issue_cnt_colidx += 1;
                diag_win_.issue_bytes_colidx += static_cast<uint64_t>(meta.size);
                break;
            case 3:
                diag_win_.issue_cnt_block += 1;
                diag_win_.issue_bytes_block += static_cast<uint64_t>(meta.size);
                break;
            default:
                break;
        }
        // bounded duplicate detection on request address
        if (diag_req_addrs_.size() < 65536) {
            const uint64_t key = meta.address ? meta.address : meta.orig_address;
            auto it = diag_req_addrs_.insert(key);
            if (it.second) diag_win_.issue_unique_addrs += 1;
            else diag_win_.issue_dup_addrs += 1;
        }
    }
    if (orch_.report_mem_issue) {
        const size_t report_bytes = meta.report_bytes ? meta.report_bytes : meta.size;
        orch_.report_mem_issue(report_bytes, meta.count_weight_read);
    }
    const int bcsr_kind = meta.bcsr_kind;
    const uint64_t addr = meta.address;
    const size_t bytes = meta.size;
    auto cb = [this, meta = std::move(meta)](uint64_t req_id, uint64_t resp_addr, std::vector<uint8_t>&& data) mutable {
        handleReadResp_(req_id, resp_addr, std::move(meta), std::move(data));
    };

    const uint64_t req_id = mem_access_->read(addr, bytes, std::move(cb));
    if (req_id != 0) {
        noteReadSourceIssue_(bcsr_kind, bytes);
    }
    return req_id;
}

bool WeightMemorySubsystem::prepareDenseRead_(uint32_t row, uint32_t col, uint32_t width,
                                             uint64_t& req_addr, size_t& req_size,
                                             bool& is_row, uint32_t& col_start, uint32_t& count_floats) const {
    const uint32_t bpf = sizeof(float);
    auto denseAddr = [&](uint32_t r, uint32_t c) -> uint64_t {
        if (!dense_phys_enable_) {
            return orch_.base_addr +
                   (static_cast<uint64_t>(r) * static_cast<uint64_t>(width) + static_cast<uint64_t>(c)) * bpf;
        }
        return orch_.base_addr + densePhysV1Offset(r, c, dense_phys_);
    };

    req_addr = denseAddr(row, col);
    req_size = sizeof(float);
    is_row = false;
    col_start = col;
    count_floats = 1;
    if (orch_.read_force_single) return true;

    const bool merge_row = orch_.merge_read_row;
    const bool merge_cl = orch_.merge_read_cacheline;
    const bool merge_auto = orch_.merge_read_auto;

    if (merge_auto) {
        const uint32_t fpl = std::max<uint32_t>(1, orch_.line_size_bytes / bpf);
        const size_t bytes_row = static_cast<size_t>(width) * static_cast<size_t>(bpf);
        const size_t bytes_cl = static_cast<size_t>(fpl) * static_cast<size_t>(bpf);
        const bool choose_row = merge_row && (bytes_row <= bytes_cl);
        if (choose_row) {
            is_row = true;
            col_start = 0;
            count_floats = width;
            req_addr = denseAddr(row, 0);
            req_size = static_cast<size_t>(count_floats) * bpf;
        } else if (merge_cl) {
            col_start = (col / fpl) * fpl;
            count_floats = std::min<uint32_t>(fpl, width - col_start);
            req_addr = denseAddr(row, col_start);
            req_size = static_cast<size_t>(count_floats) * bpf;
        }
        return true;
    }

    if (merge_row) {
        is_row = true;
        col_start = 0;
        count_floats = width;
        req_addr = denseAddr(row, 0);
        req_size = static_cast<size_t>(count_floats) * bpf;
        return true;
    }

    if (merge_cl) {
        const uint32_t fpl = std::max<uint32_t>(1, orch_.line_size_bytes / bpf);
        col_start = (col / fpl) * fpl;
        count_floats = std::min<uint32_t>(fpl, width - col_start);
        req_addr = denseAddr(row, col_start);
        req_size = static_cast<size_t>(count_floats) * bpf;
        return true;
    }

    return true;
}

void WeightMemorySubsystem::issueDenseResolved_(uint32_t row, uint32_t col, uint32_t cb_col,
                                               std::function<void(float)> cb) {
    if (!mem_access_) {
        if (cb) cb(orch_.init_default_weight);
        return;
    }
    const uint32_t width = orch_.use_post_row_pre_col ? orch_.weights_cols : orch_.num_neurons;
    if (width == 0) {
        if (cb) cb(orch_.init_default_weight);
        return;
    }

    uint64_t req_addr = 0;
    size_t req_size = sizeof(float);
    bool is_row = false;
    uint32_t col_start = col;
    uint32_t count_floats = 1;
    (void)prepareDenseRead_(row, col, width, req_addr, req_size, is_row, col_start, count_floats);

    PendingMeta meta{};
    meta.window_seq = 0;
    meta.address = req_addr;
    meta.size = req_size;
    meta.is_row = is_row;
    meta.pre = row;
    meta.post_start = col_start;
    meta.count_floats = count_floats;
    meta.has_single_cb = (cb != nullptr);
    meta.cb_post = cb_col;
    meta.single_cb = std::move(cb);
    meta.issue_cycle = now_cycle_;
    meta.bcsr_kind = 0;
    meta.is_weight = (req_addr >= orch_.base_addr && req_addr < orch_.weight_region_end);
    meta.count_weight_read = true;
    (void)issueRead_(std::move(meta));
}

void WeightMemorySubsystem::requestDense_(uint32_t pre, uint32_t post, std::function<void(float)> cb) {
    // 兼容：当启用 BCSR 且仍通过 Dense 接口发起时，自动走 BCSR。
    if (orch_.use_bcsr && orch_.use_post_row_pre_col) {
        requestBCSR_(pre, post, std::move(cb));
        return;
    }
    if (!orch_.accessor) {
        if (cb) cb(orch_.init_default_weight);
        return;
    }
    uint32_t req_pre = 0;
    uint32_t req_post = 0;
    uint64_t cache_key = 0;
    if (!orch_.accessor->resolve(pre, post, req_pre, req_post, cache_key)) {
        if (cb) cb(orch_.init_default_weight);
        return;
    }
    const uint32_t row_idx = orch_.use_post_row_pre_col ? req_post : req_pre;
    const uint32_t col_idx = orch_.use_post_row_pre_col ? req_pre : req_post;
    issueDenseResolved_(row_idx, col_idx, /*cb_col*/col_idx, std::move(cb));
}

void WeightMemorySubsystem::noteIdxSramLookup_(uint32_t pre_global, uint32_t post_local, bool idx2_mode) {
    idx_lookup_total_ += 1;
    if (idx2_mode) idx_lookup_idx2_total_ += 1;
    else idx_lookup_legacy_total_ += 1;

    if (!(weight_sram_enable_ && weight_idx_sram_enable_) && !shared_weight_object_plane_) return;

    const uint64_t row_addr = sram_layout_.idxRowBaseAddr(post_local);
    noteIdxSramReadMirror_(row_addr + 0ull, sizeof(uint32_t));   // row_base
    noteIdxSramReadMirror_(row_addr + 4ull, sizeof(uint16_t));   // row_len
    noteIdxSramReadMirror_(row_addr + 6ull, sizeof(uint16_t));   // row_bucket_count
    noteIdxSramReadMirror_(row_addr + 8ull, sizeof(uint32_t));   // row_seed
    noteIdxSramReadMirror_(row_addr + 12ull, sizeof(uint32_t));  // row_bucket_off
    if (idx2_mode) {
        const uint32_t bucket = (pre_global ^ post_local) & 0xffu;
        noteIdxSramReadMirror_(sram_layout_.idxPilotAddr(post_local, bucket), sizeof(uint8_t));
    } else {
        const uint64_t legacy_addr = sram_layout_.idxLegacyLookupAddr(pre_global, post_local);
        noteIdxSramReadMirror_(legacy_addr, sizeof(uint32_t));
        noteIdxSramReadMirror_(legacy_addr + 4ull, sizeof(uint32_t));
    }
}

void WeightMemorySubsystem::noteL0SramLookup_(uint64_t addr, bool hit) {
    l0_lookup_total_ += 1;
    if (hit) l0_hit_total_ += 1;
    else l0_miss_total_ += 1;
    if (!(weight_sram_enable_ && weight_l0_sram_enable_) && !shared_weight_object_plane_) return;
    noteL0SramReadMirror_(addr);
}

void WeightMemorySubsystem::noteL0SramFill_(uint64_t addr) {
    l0_fill_total_ += 1;
    if (!(weight_sram_enable_ && weight_l0_sram_enable_) && !shared_weight_object_plane_) return;
    if (shared_weight_object_plane_ && shared_weight_object_plane_residency_authority_) {
        if (weight_sram_enable_ && weight_l0_sram_enable_) {
            l0_sram_model_.noteWrite(now_cycle_, sram_layout_.l0SlotAddr(addr), sizeof(float));
        }
        shared_weight_object_plane_->noteL0Fill(now_cycle_, addr);
        return;
    }
    noteL0SramWriteMirror_(addr);
}

void WeightMemorySubsystem::noteL0SramEvict_(uint64_t addr) {
    l0_evict_total_ += 1;
    if (!(weight_sram_enable_ && weight_l0_sram_enable_) && !shared_weight_object_plane_) return;
    if (shared_weight_object_plane_ && shared_weight_object_plane_residency_authority_) {
        if (weight_sram_enable_ && weight_l0_sram_enable_) {
            l0_sram_model_.noteWrite(now_cycle_, sram_layout_.l0SlotAddr(addr), sizeof(float));
        }
        shared_weight_object_plane_->noteL0Evict(now_cycle_, addr);
        return;
    }
    noteL0SramWriteMirror_(addr);
}

bool WeightMemorySubsystem::peInternalPodShadowEnabled_() const {
    PeInternalPodShadowGateConfig cfg{};
    cfg.pe_internal_cpe_enable = orch_.pe_internal_cpe_enable;
    cfg.pe_internal_pod_enable = orch_.pe_internal_pod_enable;
    cfg.pe_internal_pod_metadata_enable = orch_.pe_internal_pod_metadata_enable;
    cfg.pe_internal_pod_owner_enable = orch_.pe_internal_pod_owner_enable;
    cfg.pod_count = orch_.pe_internal_pod_count;

    PeInternalPodShadowGateBindings bindings{};
    bindings.metadata_plane = pod_metadata_object_plane_;
    bindings.owner_table = pod_owner_service_table_;
    return PeInternalPodShadowGate::enabled(cfg, bindings);
}

void WeightMemorySubsystem::observePeInternalPodMetadataObject_(
    PodMetadataObjectPlane::MetadataKind kind,
    uint64_t object_id) {
    const uint64_t object_key =
        PodMetadataObjectPlane::composeObjectKey(kind, object_id);

    PeInternalPodShadowGateConfig cfg{};
    cfg.pe_internal_cpe_enable = orch_.pe_internal_cpe_enable;
    cfg.pe_internal_pod_enable = orch_.pe_internal_pod_enable;
    cfg.pe_internal_pod_metadata_enable = orch_.pe_internal_pod_metadata_enable;
    cfg.pe_internal_pod_owner_enable = orch_.pe_internal_pod_owner_enable;
    cfg.core_id = orch_.core_id;
    cfg.pod_id = orch_.pe_internal_pod_id;
    cfg.pod_count = orch_.pe_internal_pod_count;
    cfg.window_seq = window_seq_;

    PeInternalPodShadowGateBindings bindings{};
    bindings.metadata_plane = pod_metadata_object_plane_;
    bindings.owner_table = pod_owner_service_table_;

    PeInternalPodShadowGateCounters counters{};
    counters.guard_drop_total = pe_internal_pod_guard_drop_total_;
    counters.guard_disabled_total = pe_internal_pod_guard_disabled_total_;
    counters.guard_missing_metadata_plane_total =
        pe_internal_pod_guard_missing_metadata_plane_total_;
    counters.guard_missing_owner_table_total =
        pe_internal_pod_guard_missing_owner_table_total_;
    counters.guard_zero_pod_count_total =
        pe_internal_pod_guard_zero_pod_count_total_;
    counters.guard_window_zero_total =
        pe_internal_pod_guard_window_zero_total_;
    counters.guard_invalid_cfg_pod_total =
        pe_internal_pod_guard_invalid_cfg_pod_total_;
    counters.guard_rowdescriptor_disabled_total =
        pe_internal_pod_guard_rowdescriptor_disabled_total_;
    counters.guard_rowdescriptor_missing_metadata_plane_total =
        pe_internal_pod_guard_rowdescriptor_missing_metadata_plane_total_;
    counters.guard_rowdescriptor_missing_owner_table_total =
        pe_internal_pod_guard_rowdescriptor_missing_owner_table_total_;
    counters.guard_rowdescriptor_zero_pod_count_total =
        pe_internal_pod_guard_rowdescriptor_zero_pod_count_total_;
    counters.guard_rowdescriptor_window_zero_total =
        pe_internal_pod_guard_rowdescriptor_window_zero_total_;
    counters.guard_rowdescriptor_invalid_cfg_pod_total =
        pe_internal_pod_guard_rowdescriptor_invalid_cfg_pod_total_;
    counters.guard_base_total = pe_internal_pod_guard_base_total_;
    counters.guard_band_total = pe_internal_pod_guard_band_total_;
    counters.guard_other_total = pe_internal_pod_guard_other_total_;
    counters.guard_idx2row_total = pe_internal_pod_guard_idx2row_total_;
    counters.guard_rowindex_total = pe_internal_pod_guard_rowindex_total_;
    counters.guard_rowdescriptor_total = pe_internal_pod_guard_rowdescriptor_total_;
    counters.frontier_export_total = pe_internal_pod_frontier_export_total_;
    counters.frontier_consumer_count_sum_total =
        pe_internal_pod_frontier_consumer_count_sum_total_;
    counters.frontier_overlap_strength_sum_total =
        pe_internal_pod_frontier_overlap_strength_sum_total_;
    counters.frontier_base_consumer_count_sum_total =
        pe_internal_pod_frontier_base_consumer_count_sum_total_;
    counters.frontier_base_overlap_strength_sum_total =
        pe_internal_pod_frontier_base_overlap_strength_sum_total_;
    counters.frontier_band_consumer_count_sum_total =
        pe_internal_pod_frontier_band_consumer_count_sum_total_;
    counters.frontier_band_overlap_strength_sum_total =
        pe_internal_pod_frontier_band_overlap_strength_sum_total_;
    counters.owner_lookup_total = pe_internal_pod_owner_lookup_total_;
    counters.owner_alloc_total = pe_internal_pod_owner_alloc_total_;
    counters.owner_alloc_idx2row_total =
        pe_internal_pod_owner_alloc_idx2row_total_;
    counters.owner_alloc_rowindex_total =
        pe_internal_pod_owner_alloc_rowindex_total_;
    counters.owner_alloc_rowdescriptor_total =
        pe_internal_pod_owner_alloc_rowdescriptor_total_;
    counters.owner_hit_total = pe_internal_pod_owner_hit_total_;
    counters.owner_hit_idx2row_total = pe_internal_pod_owner_hit_idx2row_total_;
    counters.owner_hit_rowindex_total =
        pe_internal_pod_owner_hit_rowindex_total_;
    counters.owner_hit_rowdescriptor_total =
        pe_internal_pod_owner_hit_rowdescriptor_total_;
    counters.owner_reject_total = pe_internal_pod_owner_reject_total_;
    counters.owner_disabled_reject_total =
        pe_internal_pod_owner_disabled_reject_total_;
    counters.owner_invalid_pod_reject_total =
        pe_internal_pod_owner_invalid_pod_reject_total_;
    counters.owner_table_full_reject_total =
        pe_internal_pod_owner_table_full_reject_total_;
    counters.join_request_total = pe_internal_pod_join_request_total_;
    counters.join_grant_total = pe_internal_pod_join_grant_total_;
    counters.join_reject_total = pe_internal_pod_join_reject_total_;
    counters.join_table_disabled_reject_total =
        pe_internal_pod_join_table_disabled_reject_total_;
    counters.join_duplicate_consumer_reject_total =
        pe_internal_pod_join_duplicate_consumer_reject_total_;
    counters.join_table_full_reject_total =
        pe_internal_pod_join_table_full_reject_total_;
    counters.join_before_private_issue_total =
        pe_internal_pod_join_before_private_issue_total_;
    counters.owner_first_issue_deferred_total =
        pe_internal_pod_owner_first_issue_deferred_total_;
    counters.owner_first_issue_deferred_idx2row_total =
        pe_internal_pod_owner_first_issue_deferred_idx2row_total_;
    counters.owner_first_issue_deferred_rowindex_total =
        pe_internal_pod_owner_first_issue_deferred_rowindex_total_;
    counters.owner_first_issue_deferred_rowdescriptor_total =
        pe_internal_pod_owner_first_issue_deferred_rowdescriptor_total_;
    counters.owner_first_private_issue_avoided_total =
        pe_internal_pod_owner_first_private_issue_avoided_total_;
    counters.owner_first_private_issue_avoided_idx2row_total =
        pe_internal_pod_owner_first_private_issue_avoided_idx2row_total_;
    counters.owner_first_private_issue_avoided_rowindex_total =
        pe_internal_pod_owner_first_private_issue_avoided_rowindex_total_;
    counters.owner_first_private_issue_avoided_rowdescriptor_total =
        pe_internal_pod_owner_first_private_issue_avoided_rowdescriptor_total_;
    counters.reject_base_total = pe_internal_pod_reject_base_total_;
    counters.reject_band_total = pe_internal_pod_reject_band_total_;
    counters.reject_other_total = pe_internal_pod_reject_other_total_;
    counters.reject_idx2row_total = pe_internal_pod_reject_idx2row_total_;
    counters.reject_rowindex_total = pe_internal_pod_reject_rowindex_total_;
    counters.reject_rowdescriptor_total = pe_internal_pod_reject_rowdescriptor_total_;
    counters.useful_total = pe_internal_pod_useful_total_;
    counters.useful_join_grant_total =
        pe_internal_pod_useful_join_grant_total_;
    counters.useful_duplicate_replay_elide_total =
        pe_internal_pod_useful_duplicate_replay_elide_total_;
    counters.useful_base_total = pe_internal_pod_useful_base_total_;
    counters.useful_band_total = pe_internal_pod_useful_band_total_;
    counters.useful_other_total = pe_internal_pod_useful_other_total_;
    counters.useful_idx2row_total = pe_internal_pod_useful_idx2row_total_;
    counters.useful_rowindex_total = pe_internal_pod_useful_rowindex_total_;
    counters.useful_rowdescriptor_total = pe_internal_pod_useful_rowdescriptor_total_;
    counters.duplicate_metadata_replay_elided_total =
        pe_internal_pod_duplicate_metadata_replay_elided_total_;
    counters.duplicate_metadata_issue_elided_total =
        pe_internal_pod_duplicate_metadata_issue_elided_total_;
    counters.fallback_private_issue_total =
        pe_internal_pod_fallback_private_issue_total_;

    const uint64_t prev_owner_alloc_total = counters.owner_alloc_total;
    const uint64_t prev_join_grant_total = counters.join_grant_total;

    PeInternalPodShadowGate::observe(cfg, bindings, kind, object_id, counters);

    if (pe_local_service_object_table_ != nullptr &&
        pe_local_service_object_table_->enabled()) {
        const auto note_kind_counter =
            [&](uint64_t& idx2row_counter,
                uint64_t& rowindex_counter,
                uint64_t& rowdescriptor_counter) {
                switch (kind) {
                    case PodMetadataObjectPlane::MetadataKind::Idx2Row:
                        idx2row_counter += 1u;
                        break;
                    case PodMetadataObjectPlane::MetadataKind::RowIndex:
                        rowindex_counter += 1u;
                        break;
                    case PodMetadataObjectPlane::MetadataKind::RowDescriptor:
                        rowdescriptor_counter += 1u;
                        break;
                    default:
                        break;
                }
            };

        if (counters.owner_alloc_total > prev_owner_alloc_total) {
            PeLocalServiceObjectTable::OwnerRequest owner{};
            owner.pod_id = orch_.pe_internal_pod_id;
            owner.window_seq = window_seq_;
            owner.object_key = object_key;
            owner.owner_core_id = orch_.core_id;
            (void)pe_local_service_object_table_->noteOwnerForm(owner);
        }

        if (counters.join_grant_total > prev_join_grant_total) {
            PeLocalServiceObjectTable::JoinRequest join{};
            join.pod_id = orch_.pe_internal_pod_id;
            join.window_seq = window_seq_;
            join.object_key = object_key;
            join.consumer_core_id = orch_.core_id;
            const auto join_result = pe_local_service_object_table_->join(join);
            if (join_result.joined_live) {
                pe_internal_pod_service_join_live_total_ += 1u;
                note_kind_counter(
                    pe_internal_pod_service_join_live_idx2row_total_,
                    pe_internal_pod_service_join_live_rowindex_total_,
                    pe_internal_pod_service_join_live_rowdescriptor_total_);
            }
            if (join_result.joined_ready) {
                pe_internal_pod_service_join_ready_total_ += 1u;
                note_kind_counter(
                    pe_internal_pod_service_join_ready_idx2row_total_,
                    pe_internal_pod_service_join_ready_rowindex_total_,
                    pe_internal_pod_service_join_ready_rowdescriptor_total_);
            }
            if (join_result.late_join) {
                pe_internal_pod_service_late_join_total_ += 1u;
                note_kind_counter(
                    pe_internal_pod_service_late_join_idx2row_total_,
                    pe_internal_pod_service_late_join_rowindex_total_,
                    pe_internal_pod_service_late_join_rowdescriptor_total_);
            }
            if (join_result.joined_live || join_result.joined_ready) {
                pe_internal_pod_service_potential_private_service_elide_total_ += 1u;
                note_kind_counter(
                    pe_internal_pod_service_potential_private_service_elide_idx2row_total_,
                    pe_internal_pod_service_potential_private_service_elide_rowindex_total_,
                    pe_internal_pod_service_potential_private_service_elide_rowdescriptor_total_);
                pe_internal_pod_owner_first_service_elide_total_ += 1u;
                note_kind_counter(
                    pe_internal_pod_owner_first_service_elide_idx2row_total_,
                    pe_internal_pod_owner_first_service_elide_rowindex_total_,
                    pe_internal_pod_owner_first_service_elide_rowdescriptor_total_);
            }
        }
    }

    pe_internal_pod_guard_drop_total_ = counters.guard_drop_total;
    pe_internal_pod_guard_disabled_total_ = counters.guard_disabled_total;
    pe_internal_pod_guard_missing_metadata_plane_total_ =
        counters.guard_missing_metadata_plane_total;
    pe_internal_pod_guard_missing_owner_table_total_ =
        counters.guard_missing_owner_table_total;
    pe_internal_pod_guard_zero_pod_count_total_ =
        counters.guard_zero_pod_count_total;
    pe_internal_pod_guard_window_zero_total_ =
        counters.guard_window_zero_total;
    pe_internal_pod_guard_invalid_cfg_pod_total_ =
        counters.guard_invalid_cfg_pod_total;
    pe_internal_pod_guard_rowdescriptor_disabled_total_ =
        counters.guard_rowdescriptor_disabled_total;
    pe_internal_pod_guard_rowdescriptor_missing_metadata_plane_total_ =
        counters.guard_rowdescriptor_missing_metadata_plane_total;
    pe_internal_pod_guard_rowdescriptor_missing_owner_table_total_ =
        counters.guard_rowdescriptor_missing_owner_table_total;
    pe_internal_pod_guard_rowdescriptor_zero_pod_count_total_ =
        counters.guard_rowdescriptor_zero_pod_count_total;
    pe_internal_pod_guard_rowdescriptor_window_zero_total_ =
        counters.guard_rowdescriptor_window_zero_total;
    pe_internal_pod_guard_rowdescriptor_invalid_cfg_pod_total_ =
        counters.guard_rowdescriptor_invalid_cfg_pod_total;
    pe_internal_pod_guard_base_total_ = counters.guard_base_total;
    pe_internal_pod_guard_band_total_ = counters.guard_band_total;
    pe_internal_pod_guard_other_total_ = counters.guard_other_total;
    pe_internal_pod_guard_idx2row_total_ = counters.guard_idx2row_total;
    pe_internal_pod_guard_rowindex_total_ = counters.guard_rowindex_total;
    pe_internal_pod_guard_rowdescriptor_total_ = counters.guard_rowdescriptor_total;
    pe_internal_pod_frontier_export_total_ = counters.frontier_export_total;
    pe_internal_pod_frontier_consumer_count_sum_total_ =
        counters.frontier_consumer_count_sum_total;
    pe_internal_pod_frontier_overlap_strength_sum_total_ =
        counters.frontier_overlap_strength_sum_total;
    pe_internal_pod_frontier_base_consumer_count_sum_total_ =
        counters.frontier_base_consumer_count_sum_total;
    pe_internal_pod_frontier_base_overlap_strength_sum_total_ =
        counters.frontier_base_overlap_strength_sum_total;
    pe_internal_pod_frontier_band_consumer_count_sum_total_ =
        counters.frontier_band_consumer_count_sum_total;
    pe_internal_pod_frontier_band_overlap_strength_sum_total_ =
        counters.frontier_band_overlap_strength_sum_total;
    pe_internal_pod_owner_lookup_total_ = counters.owner_lookup_total;
    pe_internal_pod_owner_alloc_total_ = counters.owner_alloc_total;
    pe_internal_pod_owner_alloc_idx2row_total_ =
        counters.owner_alloc_idx2row_total;
    pe_internal_pod_owner_alloc_rowindex_total_ =
        counters.owner_alloc_rowindex_total;
    pe_internal_pod_owner_alloc_rowdescriptor_total_ =
        counters.owner_alloc_rowdescriptor_total;
    pe_internal_pod_owner_hit_total_ = counters.owner_hit_total;
    pe_internal_pod_owner_hit_idx2row_total_ =
        counters.owner_hit_idx2row_total;
    pe_internal_pod_owner_hit_rowindex_total_ =
        counters.owner_hit_rowindex_total;
    pe_internal_pod_owner_hit_rowdescriptor_total_ =
        counters.owner_hit_rowdescriptor_total;
    pe_internal_pod_owner_reject_total_ = counters.owner_reject_total;
    pe_internal_pod_owner_disabled_reject_total_ =
        counters.owner_disabled_reject_total;
    pe_internal_pod_owner_invalid_pod_reject_total_ =
        counters.owner_invalid_pod_reject_total;
    pe_internal_pod_owner_table_full_reject_total_ =
        counters.owner_table_full_reject_total;
    pe_internal_pod_join_request_total_ = counters.join_request_total;
    pe_internal_pod_join_grant_total_ = counters.join_grant_total;
    pe_internal_pod_join_reject_total_ = counters.join_reject_total;
    pe_internal_pod_join_table_disabled_reject_total_ =
        counters.join_table_disabled_reject_total;
    pe_internal_pod_join_duplicate_consumer_reject_total_ =
        counters.join_duplicate_consumer_reject_total;
    pe_internal_pod_join_table_full_reject_total_ =
        counters.join_table_full_reject_total;
    pe_internal_pod_join_before_private_issue_total_ =
        counters.join_before_private_issue_total;
    pe_internal_pod_owner_first_issue_deferred_total_ =
        counters.owner_first_issue_deferred_total;
    pe_internal_pod_owner_first_issue_deferred_idx2row_total_ =
        counters.owner_first_issue_deferred_idx2row_total;
    pe_internal_pod_owner_first_issue_deferred_rowindex_total_ =
        counters.owner_first_issue_deferred_rowindex_total;
    pe_internal_pod_owner_first_issue_deferred_rowdescriptor_total_ =
        counters.owner_first_issue_deferred_rowdescriptor_total;
    pe_internal_pod_owner_first_private_issue_avoided_total_ =
        counters.owner_first_private_issue_avoided_total;
    pe_internal_pod_owner_first_private_issue_avoided_idx2row_total_ =
        counters.owner_first_private_issue_avoided_idx2row_total;
    pe_internal_pod_owner_first_private_issue_avoided_rowindex_total_ =
        counters.owner_first_private_issue_avoided_rowindex_total;
    pe_internal_pod_owner_first_private_issue_avoided_rowdescriptor_total_ =
        counters.owner_first_private_issue_avoided_rowdescriptor_total;
    pe_internal_pod_reject_base_total_ = counters.reject_base_total;
    pe_internal_pod_reject_band_total_ = counters.reject_band_total;
    pe_internal_pod_reject_other_total_ = counters.reject_other_total;
    pe_internal_pod_reject_idx2row_total_ = counters.reject_idx2row_total;
    pe_internal_pod_reject_rowindex_total_ = counters.reject_rowindex_total;
    pe_internal_pod_reject_rowdescriptor_total_ = counters.reject_rowdescriptor_total;
    pe_internal_pod_useful_total_ = counters.useful_total;
    pe_internal_pod_useful_join_grant_total_ =
        counters.useful_join_grant_total;
    pe_internal_pod_useful_duplicate_replay_elide_total_ =
        counters.useful_duplicate_replay_elide_total;
    pe_internal_pod_useful_base_total_ = counters.useful_base_total;
    pe_internal_pod_useful_band_total_ = counters.useful_band_total;
    pe_internal_pod_useful_other_total_ = counters.useful_other_total;
    pe_internal_pod_useful_idx2row_total_ = counters.useful_idx2row_total;
    pe_internal_pod_useful_rowindex_total_ = counters.useful_rowindex_total;
    pe_internal_pod_useful_rowdescriptor_total_ = counters.useful_rowdescriptor_total;
    pe_internal_pod_duplicate_metadata_replay_elided_total_ =
        counters.duplicate_metadata_replay_elided_total;
    pe_internal_pod_duplicate_metadata_issue_elided_total_ =
        counters.duplicate_metadata_issue_elided_total;
    pe_internal_pod_fallback_private_issue_total_ =
        counters.fallback_private_issue_total;
}

bool WeightMemorySubsystem::notePeInternalPodServiceObjectReady_(
    PodMetadataObjectPlane::MetadataKind kind,
    uint64_t object_id,
    uint32_t window_seq) {
    const uint32_t effective_window_seq = (window_seq != 0u) ? window_seq : window_seq_;
    if (effective_window_seq == 0u) return false;
    if (pe_local_service_object_table_ == nullptr ||
        !pe_local_service_object_table_->enabled()) {
        return false;
    }

    PeLocalServiceObjectTable::ReadyRequest ready{};
    ready.pod_id = orch_.pe_internal_pod_id;
    ready.window_seq = effective_window_seq;
    ready.object_key = PodMetadataObjectPlane::composeObjectKey(kind, object_id);
    const auto ready_result = pe_local_service_object_table_->markReady(ready);
    if (!ready_result.valid || !ready_result.transitioned) {
        return false;
    }

    pe_internal_pod_service_ready_transition_total_ += 1u;
    pe_internal_pod_service_ready_fanout_total_ += 1u;
    pe_internal_pod_service_ready_fanout_consumers_sum_total_ +=
        static_cast<uint64_t>(ready_result.consumer_count);

    switch (kind) {
        case PodMetadataObjectPlane::MetadataKind::Idx2Row:
            pe_internal_pod_service_ready_transition_idx2row_total_ += 1u;
            pe_internal_pod_service_ready_fanout_idx2row_total_ += 1u;
            pe_internal_pod_service_ready_fanout_consumers_sum_idx2row_total_ +=
                static_cast<uint64_t>(ready_result.consumer_count);
            break;
        case PodMetadataObjectPlane::MetadataKind::RowIndex:
            pe_internal_pod_service_ready_transition_rowindex_total_ += 1u;
            pe_internal_pod_service_ready_fanout_rowindex_total_ += 1u;
            pe_internal_pod_service_ready_fanout_consumers_sum_rowindex_total_ +=
                static_cast<uint64_t>(ready_result.consumer_count);
            break;
        case PodMetadataObjectPlane::MetadataKind::RowDescriptor:
            pe_internal_pod_service_ready_transition_rowdescriptor_total_ += 1u;
            pe_internal_pod_service_ready_fanout_rowdescriptor_total_ += 1u;
            pe_internal_pod_service_ready_fanout_consumers_sum_rowdescriptor_total_ +=
                static_cast<uint64_t>(ready_result.consumer_count);
            break;
        default:
            break;
    }

    if (ready_result.released_after_transition) {
        pe_internal_pod_service_ready_release_total_ += 1u;
        switch (kind) {
            case PodMetadataObjectPlane::MetadataKind::Idx2Row:
                pe_internal_pod_service_ready_release_idx2row_total_ += 1u;
                pe_internal_pod_service_released_idx2row_total_ += 1u;
                break;
            case PodMetadataObjectPlane::MetadataKind::RowIndex:
                pe_internal_pod_service_ready_release_rowindex_total_ += 1u;
                pe_internal_pod_service_released_rowindex_total_ += 1u;
                break;
            case PodMetadataObjectPlane::MetadataKind::RowDescriptor:
                pe_internal_pod_service_ready_release_rowdescriptor_total_ += 1u;
                pe_internal_pod_service_released_rowdescriptor_total_ += 1u;
                break;
            default:
                break;
        }
    }

    return true;
}

void WeightMemorySubsystem::notePeInternalPodServiceObjectReleased_(
    PodMetadataObjectPlane::MetadataKind kind,
    uint64_t object_id,
    uint32_t window_seq,
    bool defer_until_ready) {
    const uint32_t effective_window_seq = (window_seq != 0u) ? window_seq : window_seq_;
    if (effective_window_seq == 0u) return;
    if (pe_local_service_object_table_ == nullptr ||
        !pe_local_service_object_table_->enabled()) {
        return;
    }

    PeLocalServiceObjectTable::ReleaseRequest release{};
    release.pod_id = orch_.pe_internal_pod_id;
    release.window_seq = effective_window_seq;
    release.object_key = PodMetadataObjectPlane::composeObjectKey(kind, object_id);
    release.defer_until_ready = defer_until_ready;
    const auto release_result = pe_local_service_object_table_->release(release);
    const bool released = release_result.valid && release_result.released;
    const bool deferred = release_result.valid && release_result.deferred;
    if (deferred) {
        pe_internal_pod_service_release_deferred_total_ += 1u;
    }
    switch (kind) {
        case PodMetadataObjectPlane::MetadataKind::Idx2Row:
            if (released) {
                pe_internal_pod_service_released_idx2row_total_ += 1u;
            } else if (deferred) {
                pe_internal_pod_service_release_deferred_idx2row_total_ += 1u;
            } else if (!deferred) {
                pe_internal_pod_service_release_missing_idx2row_total_ += 1u;
            }
            break;
        case PodMetadataObjectPlane::MetadataKind::RowIndex:
            if (released) {
                pe_internal_pod_service_released_rowindex_total_ += 1u;
            } else if (deferred) {
                pe_internal_pod_service_release_deferred_rowindex_total_ += 1u;
            } else if (!deferred) {
                pe_internal_pod_service_release_missing_rowindex_total_ += 1u;
            }
            break;
        case PodMetadataObjectPlane::MetadataKind::RowDescriptor:
            if (released) {
                pe_internal_pod_service_released_rowdescriptor_total_ += 1u;
            } else if (deferred) {
                pe_internal_pod_service_release_deferred_rowdescriptor_total_ += 1u;
            } else if (!deferred) {
                pe_internal_pod_service_release_missing_rowdescriptor_total_ += 1u;
            }
            break;
        default:
            break;
    }
}

float WeightMemorySubsystem::applyWeightGuards_(float w) const {
    if (orch_.bcsr_weight_guard_enable) {
        if (!std::isfinite(w) || std::fabs(w) > orch_.bcsr_weight_abs_max) {
            return 0.0f;
        }
    }
    return applyReadRespZeroFallback_(w);
}

SST::Output* WeightMemorySubsystem::diagOutOrFallback_() const {
    static SST::Output fallback("WeightMemorySubsystem[@p:@l]: ", 0, 0, SST::Output::STDERR);
    return diag_out_ ? diag_out_ : &fallback;
}

void WeightMemorySubsystem::enqueueBcsrBlockReadCoalesced_(uint32_t win_seq,
                                                          uint32_t block_row,
                                                          uint32_t block_col,
                                                          uint32_t global_block_index,
                                                          uint32_t idx_in_row,
                                                          uint32_t intra_row,
                                                          uint32_t intra_col,
                                                          uint32_t pre_global,
                                                          uint32_t post_local,
                                                          std::function<void(float)> cb) {
    if (!orch_.use_bcsr || !orch_.bcsr_mgr || !mem_access_) {
        if (cb) cb(0.0f);
        return;
    }

    const bool row_slice_fetch = (bcsr_block_fetch_mode_ == BcsrBlockFetchMode::RowCacheline);
    const bool block_inflight_coalesce =
        kEnableBcsrInflightCoalescing && orch_.bcsr_block_inflight_coalesce_enable;

    if (!block_inflight_coalesce) {
        // Coalescing disabled: keep one-request-per-edge path (format-only BCSR behavior).
        const size_t block_bytes = orch_.bcsr_mgr->blockBytes();
        const size_t row_bytes = orch_.bcsr_mgr->blockRowBytes();
        uint64_t addr = orch_.bcsr_mgr->blockDataAddrByRow(block_row, idx_in_row);
        size_t req_orig_size = block_bytes;
        if (row_slice_fetch) {
            addr += static_cast<uint64_t>(intra_row) * static_cast<uint64_t>(row_bytes);
            req_orig_size = row_bytes;
        }
        uint64_t req_addr = addr;
        size_t req_size = req_orig_size;
        size_t slice_off = 0;
        prepareAlignedRead(addr, req_orig_size, orch_.line_size_bytes, req_addr, req_size, slice_off);

        PendingMeta meta{};
        meta.window_seq = win_seq;
        meta.address = req_addr;
        meta.size = req_size;
        meta.orig_address = addr;
        meta.orig_size = req_orig_size;
        meta.slice_offset = slice_off;
        meta.issue_cycle = now_cycle_;
        meta.bcsr_kind = 3;
        meta.bcsr_block_row = block_row;
        meta.bcsr_target_block_col = block_col;
        meta.bcsr_idx_in_row = idx_in_row;
        meta.bcsr_intra_row = intra_row;
        meta.bcsr_intra_col = intra_col;
        meta.bcsr_global_block_index = global_block_index;
        meta.bcsr_row_slice_fetch = row_slice_fetch;
        meta.bcsr_prefetch_all = false;
        meta.has_single_cb = true;
        meta.single_cb = std::move(cb);
        meta.is_weight = true;
        meta.count_weight_read = true;

        const bool count_budget = (win_seq != 0);
        const IssueStatus st = tryIssueRead_(meta, /*count_budget*/count_budget, /*budget_reserved*/false);
        if (st == IssueStatus::Issued) return;
        if (st == IssueStatus::DeferredInflight) {
            pending_direct_reads_.push_back(std::move(meta));
            return;
        }
        if (meta.has_single_cb && meta.single_cb) meta.single_cb(0.0f);
        return;
    }

    const uint32_t inflight_intra_row = row_slice_fetch ? intra_row : 0;
    const uint64_t inflight_key = makeBlockInflightKey_(win_seq, global_block_index, inflight_intra_row);
    auto& inflight = inflight_block_[inflight_key];
    if (!inflight.issued) {
        inflight.window_seq = win_seq;
        inflight.block_row = block_row;
        inflight.block_col = block_col;
        inflight.global_block_index = global_block_index;
        inflight.idx_in_row = idx_in_row;
        inflight.intra_row = intra_row;
        inflight.count_budget = (win_seq != 0);
    }

    BlockWaiter waiter{};
    waiter.pre_global = pre_global;
    waiter.post_local = post_local;
    waiter.intra_row = intra_row;
    waiter.intra_col = intra_col;
    waiter.cb = std::move(cb);
    inflight.waiters.push_back(std::move(waiter));

    if (inflight.issued) return;
    if (inflight.queued) return;
    if (inflight.waiters.size() == 1) block_miss_window_ += 1;

    const size_t block_bytes = orch_.bcsr_mgr->blockBytes();
    const size_t row_bytes = orch_.bcsr_mgr->blockRowBytes();
    uint64_t addr = orch_.bcsr_mgr->blockDataAddrByRow(block_row, idx_in_row);
    size_t req_orig_size = block_bytes;
    if (row_slice_fetch) {
        addr += static_cast<uint64_t>(intra_row) * static_cast<uint64_t>(row_bytes);
        req_orig_size = row_bytes;
    }
    uint64_t req_addr = addr;
    size_t req_size = req_orig_size;
    size_t slice_off = 0;
    prepareAlignedRead(addr, req_orig_size, orch_.line_size_bytes, req_addr, req_size, slice_off);

    PendingMeta meta{};
    meta.window_seq = win_seq;
    meta.address = req_addr;
    meta.size = req_size;
    meta.orig_address = addr;
    meta.orig_size = req_orig_size;
    meta.slice_offset = slice_off;
    meta.issue_cycle = now_cycle_;
    meta.bcsr_kind = 3;
    meta.bcsr_block_row = block_row;
    meta.bcsr_target_block_col = block_col;
    meta.bcsr_idx_in_row = idx_in_row;
    meta.bcsr_intra_row = intra_row;
    meta.bcsr_global_block_index = global_block_index;
    meta.bcsr_row_slice_fetch = row_slice_fetch;
    meta.bcsr_prefetch_all = false;
    meta.has_single_cb = false;
    meta.is_weight = true;
    meta.count_weight_read = true;

    const IssueStatus st = tryIssueRead_(std::move(meta), inflight.count_budget, /*budget_reserved*/false);
    if (st == IssueStatus::Issued) {
        inflight.issued = true;
        inflight.queued = false;
        if (diag_debug_ && diag_window_active_) diag_win_.block_miss += 1;
        return;
    }
    if (st == IssueStatus::DeferredInflight) {
        inflight.queued = true;
        pending_block_reads_.push_back(inflight_key);
        return;
    }

    auto waiters = std::move(inflight.waiters);
    inflight_block_.erase(inflight_key);
    for (auto& ww : waiters) {
        if (ww.cb) ww.cb(0.0f);
    }
}

void WeightMemorySubsystem::requestBCSR_(uint32_t pre_global, uint32_t post_local, std::function<void(float)> cb) {
    if (!orch_.use_bcsr || !orch_.bcsr_mgr) {
        if (cb) cb(0.0f);
        return;
    }
    if (!mem_access_) {
        if (cb) cb(0.0f);
        return;
    }
    if (orch_.bcsr_force_file_read && orch_.read_bcsr_from_file) {
        float w = orch_.read_bcsr_from_file(post_local, pre_global);
        if (cb) cb(applyWeightGuards_(w));
        return;
    }
    if (orch_.ensure_loader_ready && !orch_.ensure_loader_ready()) {
        // Non-window path: defer until WeightLoader completes; do not silently return 0.
        if (window_seq_ == 0) {
            PendingBcsrRowptrWaiter w{};
            w.pre_global = pre_global;
            w.post_local = post_local;
            w.cb = std::move(cb);
            pending_bcsr_rowptr_waiters_.push_back(std::move(w));
            return;
        }
        if (cb) cb(0.0f);
        return;
    }
    if (orch_.bcsr_rowptr_ready && !orch_.bcsr_rowptr_ready()) {
        // Non-window path: defer until rowptr is ready; rowptr prefetch will be issued by onClockTick().
        if (window_seq_ == 0) {
            PendingBcsrRowptrWaiter w{};
            w.pre_global = pre_global;
            w.post_local = post_local;
            w.cb = std::move(cb);
            pending_bcsr_rowptr_waiters_.push_back(std::move(w));
            maybeIssueBcsrRowptrPrefetch_();
            return;
        }
        if (cb) cb(0.0f);
        return;
    }

    const uint32_t br = orch_.bcsr_mgr->effectiveBlockRows();
    const uint32_t bc = orch_.bcsr_mgr->effectiveBlockCols();
    const uint32_t block_row = (br ? (post_local / br) : 0);
    const uint32_t intra_row = (br ? (post_local % br) : 0);
    const uint32_t block_col = (bc ? (pre_global / bc) : 0);
    const uint32_t intra_col = (bc ? (pre_global % bc) : 0);

    uint32_t start = 0;
    uint32_t end = 0;
    if (!orch_.bcsr_mgr->rowBounds(block_row, start, end)) {
        if (cb) cb(0.0f);
        return;
    }
    if (end <= start) {
        if (cb) cb(0.0f);
        return;
    }

    std::vector<uint32_t> cols;
    if (!orch_.bcsr_mgr->rowIndexGet(block_row, cols)) {
        const uint32_t block_count = end - start;
        const size_t bytes = orch_.bcsr_mgr->colIndexBytes(block_count);
        const uint64_t addr = orch_.bcsr_mgr->colIndexAddr(start);

        const bool colidx_inflight_coalesce =
            kEnableBcsrInflightCoalescing && orch_.bcsr_colidx_inflight_coalesce_enable;
        if (!colidx_inflight_coalesce) {
            // Coalescing disabled: one-request-per-edge path.
            PendingMeta meta{};
            meta.window_seq = window_seq_;
            meta.address = addr;
            meta.size = bytes;
            meta.orig_address = addr;
            meta.orig_size = bytes;
            meta.slice_offset = 0;
            meta.issue_cycle = now_cycle_;
            meta.bcsr_kind = 2;
            meta.bcsr_block_row = block_row;
            meta.bcsr_target_block_col = block_col;
            meta.bcsr_intra_row = intra_row;
            meta.bcsr_intra_col = intra_col;
            meta.bcsr_row_start = start;
            meta.bcsr_prefetch_all = false;
            meta.has_single_cb = true;
            meta.single_cb = std::move(cb);
            meta.is_weight = true;
            meta.count_weight_read = false;

            const bool count_budget = (window_seq_ != 0);
            const IssueStatus st = tryIssueRead_(meta, /*count_budget*/count_budget, /*budget_reserved*/false);
            if (st == IssueStatus::Issued) {
                if (diag_debug_ && diag_window_active_) diag_win_.rowidx_miss += 1;
                return;
            }
            if (st == IssueStatus::DeferredInflight) {
                pending_direct_reads_.push_back(std::move(meta));
                return;
            }
            if (meta.has_single_cb && meta.single_cb) meta.single_cb(0.0f);
            return;
        }

        const uint32_t win_seq = window_seq_;
        const uint64_t inflight_key = makeInflightKey_(win_seq, block_row);
        auto& inflight = inflight_colidx_[inflight_key];
        if (!inflight.issued) {
            inflight.window_seq = win_seq;
            inflight.block_row = block_row;
            inflight.row_start = start;
            inflight.row_end = end;
            inflight.count_budget = (win_seq != 0);
        }

        ColidxWaiter waiter{};
        waiter.pre_global = pre_global;
        waiter.post_local = post_local;
        waiter.target_block_col = block_col;
        waiter.intra_row = intra_row;
        waiter.intra_col = intra_col;
        waiter.cb = std::move(cb);
        inflight.waiters.push_back(std::move(waiter));

        if (inflight.issued) return;
        if (inflight.queued) return;

        PendingMeta meta{};
        meta.window_seq = win_seq;
        meta.address = addr;
        meta.size = bytes;
        meta.orig_address = addr;
        meta.orig_size = bytes;
        meta.slice_offset = 0;
        meta.issue_cycle = now_cycle_;
        meta.bcsr_kind = 2;
        meta.bcsr_block_row = block_row;
        meta.bcsr_target_block_col = UINT32_MAX;
        meta.bcsr_row_start = start;
        meta.bcsr_prefetch_all = false;
        meta.has_single_cb = false;
        meta.is_weight = true;
        meta.count_weight_read = false;

        const IssueStatus st = tryIssueRead_(std::move(meta), inflight.count_budget, /*budget_reserved*/false);
        if (st == IssueStatus::Issued) {
            inflight.issued = true;
            inflight.queued = false;
            if (diag_debug_ && diag_window_active_) diag_win_.rowidx_miss += 1;
            return;
        }
        if (st == IssueStatus::DeferredInflight) {
            inflight.queued = true;
            pending_colidx_reads_.push_back(inflight_key);
            return;
        }

        auto waiters = std::move(inflight.waiters);
        inflight_colidx_.erase(inflight_key);
        for (auto& ww : waiters) {
            if (ww.cb) ww.cb(0.0f);
        }
        return;
    }
    if (diag_debug_ && diag_window_active_) diag_win_.rowidx_hit += 1;
    uint32_t idx_in_row = 0;
    bool found = false;
    for (size_t i = 0; i < cols.size(); ++i) {
        if (cols[i] == block_col) {
            idx_in_row = static_cast<uint32_t>(i);
            found = true;
            break;
        }
    }
    if (!found) {
        if (cb) cb(0.0f);
        return;
    }

    const uint32_t global_block_index = start + idx_in_row;
    enqueueBcsrBlockReadCoalesced_(window_seq_,
                                   block_row, block_col, global_block_index,
                                   idx_in_row,
                                   intra_row, intra_col,
                                   pre_global, post_local,
                                   std::move(cb));
    return;
}

void WeightMemorySubsystem::maybeIssueBcsrRowptrPrefetch_() {
    if (!orch_.use_bcsr || !orch_.bcsr_mgr) return;
    if (!mem_access_) return;
    if (orch_.bcsr_mgr->isRowptrReady()) return;
    if (orch_.bcsr_mgr->isRowptrReadPending()) return;
    if (orch_.ensure_loader_ready && !orch_.ensure_loader_ready()) return;
    const bool urgent = !pending_bcsr_rowptr_waiters_.empty();
    if (!urgent && now_cycle_ < orch_.memory_warmup_cycles) return;
    if (orch_.loader_barrier_cycles != 0 && now_cycle_ < orch_.loader_barrier_cycles) return;

    const uint32_t br = orch_.bcsr_mgr->effectiveBlockRows();
    const uint32_t nBlockRows = (br ? ((orch_.num_neurons + br - 1) / br) : 0);
    const size_t bytes = static_cast<size_t>(nBlockRows + 1) * sizeof(uint32_t);
    const uint64_t addr = orch_.bcsr_mgr->rowptrAddr();

    // Keep rowptr reads strictly at original addr/bytes (do NOT extend to cacheline boundaries).
    // Rationale: under some non-coherent cache configurations, extended multi-cacheline reads may
    // return all-zero payloads, while the same address range read without extension is correct.
    const uint64_t req_addr = addr;
    const size_t req_size = bytes;
    const size_t slice_off = 0;

    PendingMeta meta{};
    meta.address = req_addr;
    meta.size = req_size;
    meta.orig_address = addr;
    meta.orig_size = bytes;
    meta.slice_offset = slice_off;
    meta.issue_cycle = now_cycle_;
    meta.bcsr_kind = 1;
    meta.is_weight = true;
    meta.count_weight_read = false;
    if (diag_out_ && diag_out_->getVerboseLevel() >= 2) {
        diag_out_->verbose(CALL_INFO, 2, 0,
            "[diag-bcsr-rowptr-issue] node=%d core=%d addr=0x%llx bytes=%zu nBlockRows=%u\n",
            diag_node_id_, diag_core_id_,
            (unsigned long long)addr, bytes,
            nBlockRows);
    }
    const IssueStatus st = tryIssueRead_(std::move(meta), /*count_budget*/false, /*budget_reserved*/false);
    if (st == IssueStatus::Issued) {
        orch_.bcsr_mgr->setRowptrReadPending(true);
    }
}

void WeightMemorySubsystem::drainPendingBcsrRowptrWaiters_() {
    if (window_seq_ != 0) return; // only used by non-window workloads
    if (pending_bcsr_rowptr_waiters_.empty()) return;
    if (orch_.ensure_loader_ready && !orch_.ensure_loader_ready()) return;
    if (orch_.use_bcsr && orch_.bcsr_mgr && !orch_.bcsr_mgr->isRowptrReady()) {
        maybeIssueBcsrRowptrPrefetch_();
        return;
    }

    static constexpr size_t kMaxDrainPerTick = 1024;
    size_t drained = 0;
    while (drained < kMaxDrainPerTick && !pending_bcsr_rowptr_waiters_.empty()) {
        auto w = std::move(pending_bcsr_rowptr_waiters_.front());
        pending_bcsr_rowptr_waiters_.pop_front();
        requestBCSR_(w.pre_global, w.post_local, std::move(w.cb));
        ++drained;
    }
}

void WeightMemorySubsystem::bcsrPrefetchAll_() {
    if (!kEnableBcsrOptimizations) return;
    if (!orch_.use_bcsr || !orch_.bcsr_mgr) return;
    if (!orch_.bcsr_prefetch_all) return;
    if (bcsr_prefetch_issued_) return;
    if (!mem_access_) return;
    if (!orch_.bcsr_mgr->isRowptrReady()) return;

    const uint32_t br = orch_.bcsr_mgr->effectiveBlockRows();
    const uint32_t nBlockRows = (br ? ((orch_.num_neurons + br - 1) / br) : 0);
    for (uint32_t block_row = 0; block_row < nBlockRows; ++block_row) {
        uint32_t start = 0;
        uint32_t end = 0;
        if (!orch_.bcsr_mgr->rowBounds(block_row, start, end)) break;
        if (end <= start) continue;

        std::vector<uint32_t> cached_cols;
        if (orch_.bcsr_mgr->rowIndexGet(block_row, cached_cols)) {
            bcsrPrefetchRowBlocks_(block_row, cached_cols, start);
            continue;
        }

        const uint32_t block_count = end - start;
        const size_t bytes = orch_.bcsr_mgr->colIndexBytes(block_count);
        const uint64_t addr = orch_.bcsr_mgr->colIndexAddr(start);
        uint64_t req_addr = addr;
        size_t req_size = bytes;
        size_t slice_off = 0;
        prepareAlignedRead(addr, bytes, orch_.line_size_bytes, req_addr, req_size, slice_off);
        PendingMeta meta{};
        meta.address = req_addr;
        meta.size = req_size;
        meta.orig_address = addr;
        meta.orig_size = bytes;
        meta.slice_offset = slice_off;
        meta.issue_cycle = now_cycle_;
        meta.bcsr_kind = 2;
        meta.bcsr_block_row = block_row;
        meta.bcsr_row_start = start;
        meta.bcsr_prefetch_all = true;
        meta.bcsr_target_block_col = UINT32_MAX;
        meta.is_weight = true;
        meta.count_weight_read = false;
        (void)issueRead_(std::move(meta));
    }
    bcsr_prefetch_issued_ = true;
}

void WeightMemorySubsystem::bcsrPrefetchRowBlocks_(uint32_t block_row, const std::vector<uint32_t>& cols, uint32_t row_start) {
    if (!kEnableBcsrOptimizations) return;
    if (!orch_.use_bcsr || !orch_.bcsr_mgr) return;
    if (!mem_access_) return;
    const size_t block_bytes = orch_.bcsr_mgr->blockBytes();
    for (size_t i = 0; i < cols.size(); ++i) {
        uint32_t block_col = cols[i];
        if (orch_.bcsr_mgr->hasBlock(block_row, block_col)) continue;
        const uint32_t global_block_index = row_start + static_cast<uint32_t>(i);
        const uint64_t addr = orch_.bcsr_mgr->blockDataAddr(global_block_index);
        uint64_t req_addr = addr;
        size_t req_size = block_bytes;
        size_t slice_off = 0;
        prepareAlignedRead(addr, block_bytes, orch_.line_size_bytes, req_addr, req_size, slice_off);
        PendingMeta meta{};
        meta.address = req_addr;
        meta.size = req_size;
        meta.orig_address = addr;
        meta.orig_size = block_bytes;
        meta.slice_offset = slice_off;
        meta.issue_cycle = now_cycle_;
        meta.bcsr_kind = 3;
        meta.bcsr_block_row = block_row;
        meta.bcsr_target_block_col = block_col;
        meta.bcsr_row_start = row_start;
        meta.bcsr_idx_in_row = static_cast<uint32_t>(i);
        meta.bcsr_global_block_index = global_block_index;
        meta.bcsr_prefetch_all = true;
        meta.is_weight = true;
        meta.count_weight_read = true;
        (void)issueRead_(std::move(meta));
    }
}

void WeightMemorySubsystem::bcsrPopulateWeightCache_(uint32_t block_row, uint32_t block_col, const std::vector<float>& blk) {
    if (!kEnableBcsrOptimizations) return;
    if (!orch_.use_bcsr || !orch_.bcsr_mgr) return;
    if (!orch_.bcsr_populate_weight_cache_enable) return;
    if (blk.empty()) return;
    if (orch_.weights_cols == 0) return;
    const uint32_t br = orch_.bcsr_mgr->effectiveBlockRows();
    const uint32_t bc = orch_.bcsr_mgr->effectiveBlockCols();
    const uint32_t row_base = block_row * br;
    const uint32_t col_base = block_col * bc;
    for (uint32_t rr = 0; rr < br; ++rr) {
        const uint32_t post_local = row_base + rr;
        if (post_local >= orch_.num_neurons) break;
        for (uint32_t cc = 0; cc < bc; ++cc) {
            const uint32_t pre_global = col_base + cc;
            if (pre_global >= orch_.weights_cols) break;
            const size_t idx = static_cast<size_t>(rr) * static_cast<size_t>(bc) + cc;
            if (idx >= blk.size()) break;
            const float w = blk[idx];
            const uint64_t key = static_cast<uint64_t>(post_local) * static_cast<uint64_t>(orch_.weights_cols) +
                                 static_cast<uint64_t>(pre_global);
            if (orch_.cache_put) orch_.cache_put(key, w);
            else if (cache_put_fn_) cache_put_fn_(key, w);
        }
    }
}

void WeightMemorySubsystem::handleReadResp_(uint64_t req_id, uint64_t addr, PendingMeta meta, std::vector<uint8_t>&& data) {
    (void)req_id;
    meta.address = addr;

    // Decrement in-flight read counter once per response; always attempt to drain deferred reads on exit.
    if (meta.counted_inflight) {
        noteComplete(/*n*/1);
    }
    struct DrainGuard {
        WeightMemorySubsystem* self;
        ~DrainGuard() {
            self->drainPendingReads_();
            self->drainPendingDirectReads_();
        }
    } guard{this};

    if (orch_.report_mem_latency && now_cycle_ >= meta.issue_cycle) {
        orch_.report_mem_latency(static_cast<uint64_t>(now_cycle_ - meta.issue_cycle), meta.is_weight);
    }

    const std::vector<uint8_t>& bytes = data;
    const size_t float_count = bytes.size() / sizeof(float);
    const float* fptr = (float_count > 0) ? reinterpret_cast<const float*>(bytes.data()) : nullptr;

    if (diag_debug_ && diag_window_active_ && diag_out_) {
        diag_win_.resp_cnt_total += 1;
        diag_win_.resp_bytes_total += static_cast<uint64_t>(bytes.size());
        if (bytes.size() < meta.size) diag_win_.resp_short_total += 1;
        switch (meta.bcsr_kind) {
            case 0:
                diag_win_.resp_cnt_dense += 1;
                diag_win_.resp_bytes_dense += static_cast<uint64_t>(bytes.size());
                break;
            case 1:
                diag_win_.resp_cnt_rowptr += 1;
                diag_win_.resp_bytes_rowptr += static_cast<uint64_t>(bytes.size());
                break;
            case 2:
                diag_win_.resp_cnt_colidx += 1;
                diag_win_.resp_bytes_colidx += static_cast<uint64_t>(bytes.size());
                break;
            case 3:
                diag_win_.resp_cnt_block += 1;
                diag_win_.resp_bytes_block += static_cast<uint64_t>(bytes.size());
                break;
            default:
                break;
        }
    }

    // Byte-exact correctness (dense-only): validate every returned byte before any slice/caching logic.
    if (meta.bcsr_kind == 0 && meta.is_weight && byteExactVerifyEnabled_()) {
        verifyDenseReadBytes_(addr, meta.size, bytes);
    }

    // ---- BCSR ----
    if (meta.bcsr_kind == 1) {
        // rowptr
        if (orch_.bcsr_mgr) {
            orch_.bcsr_mgr->setRowptrReadPending(false);
            const size_t slice_off = meta.slice_offset;
            const size_t slice_len = meta.orig_size ? meta.orig_size : meta.size;
            const bool slice_ok = (bytes.size() >= slice_off + slice_len);
            const uint8_t* slice_ptr = slice_ok ? (bytes.data() + slice_off) : bytes.data();
            const size_t slice_bytes = slice_ok ? slice_len : bytes.size();
            bool ok = orch_.bcsr_mgr->installRowptrFromBytes(slice_ptr, slice_bytes, orch_.num_neurons);
            if (!ok && orch_.bcsr_rowptr_file_fallback_enable && !orch_.weights_template.empty()) {
                // Fallback: load rowptr (+ optional colidx all_rows) from the BCSR file.
                // This is a robustness path for environments where stdmem multi-cacheline reads are unreliable.
                bcsr_rowptr_file_preload_attempted_ = true;
                const std::string path = resolveWeightsTemplatePath(orch_.weights_template, orch_.node_id, orch_.core_id);
                if (!path.empty() && orch_.base_addr <= orch_.bcsr_mgr->rowptrAddr()) {
                    const uint64_t rp_off = orch_.bcsr_mgr->rowptrAddr() - orch_.base_addr;
                    std::vector<uint8_t> rp_bytes;
                    const size_t expect = orch_.bcsr_mgr->expectedRowptrBytes(orch_.num_neurons);
                    if (readFileSlice(path, rp_off, expect, rp_bytes)) {
                        ok = orch_.bcsr_mgr->installRowptrFromBytes(rp_bytes.data(), rp_bytes.size(), orch_.num_neurons);
                        if (ok && diag_out_ && diag_out_->getVerboseLevel() >= 2) {
                            diag_out_->verbose(CALL_INFO, 2, 0,
                                "[diag-bcsr-rowptr-file] node=%d core=%d fallback ok path=%s bytes=%zu\n",
                                diag_node_id_, diag_core_id_, path.c_str(), rp_bytes.size());
                        }
                    }

                    // Optional: keep all row-index metadata resident (format metadata); preload all colidx once.
                    std::string mode = toLowerCopy(orch_.bcsr_row_index_prefetch_mode);
                    if (mode.empty()) mode = "auto";
                    if (ok && !bcsr_rowidx_file_preloaded_ && mode == "all_rows" && orch_.bcsr_mgr->rowIndexCacheCapacity() > 0) {
                        const uint32_t idx_bytes = orch_.bcsr_mgr->effectiveIdxBytes();
                        const uint64_t ci_base_addr = orch_.bcsr_mgr->colIndexAddr(0);
                        if (orch_.base_addr <= ci_base_addr) {
                            const uint32_t br = orch_.bcsr_mgr->effectiveBlockRows();
                            const uint32_t nBlockRows =
                                br ? ((orch_.num_neurons + br - 1u) / br) : static_cast<uint32_t>(orch_.num_neurons);
                            uint32_t last_start = 0;
                            uint32_t last_end = 0;
                            const bool bounds_ok =
                                (nBlockRows > 0) && orch_.bcsr_mgr->rowBounds(nBlockRows - 1u, last_start, last_end);
                            const uint32_t total_blocks = bounds_ok ? last_end : 0;
                            const size_t total_bytes = total_blocks ? orch_.bcsr_mgr->colIndexBytes(total_blocks) : 0;
                            const uint64_t ci_off = ci_base_addr - orch_.base_addr;
                            std::vector<uint8_t> all;
                            if (bounds_ok && total_bytes > 0 && (idx_bytes == 2 || idx_bytes == 4) &&
                                readFileSlice(path, ci_off, total_bytes, all)) {
                                for (uint32_t block_row = 0; block_row < nBlockRows; ++block_row) {
                                    uint32_t start = 0;
                                    uint32_t end = 0;
                                    if (!orch_.bcsr_mgr->rowBounds(block_row, start, end)) break;
                                    if (end <= start) continue;
                                    const uint32_t block_count = end - start;
                                    const size_t off = static_cast<size_t>(start) * static_cast<size_t>(idx_bytes);
                                    const size_t need = static_cast<size_t>(block_count) * static_cast<size_t>(idx_bytes);
                                    if (off + need > all.size()) break;

                                    std::vector<uint32_t> cols(block_count);
                                    if (idx_bytes == 2) {
                                        for (uint32_t i = 0; i < block_count; ++i) {
                                            uint16_t v = 0;
                                            std::memcpy(&v, all.data() + off + static_cast<size_t>(i) * 2u, 2u);
                                            cols[i] = v;
                                        }
                                    } else {
                                        for (uint32_t i = 0; i < block_count; ++i) {
                                            uint32_t v = 0;
                                            std::memcpy(&v, all.data() + off + static_cast<size_t>(i) * 4u, 4u);
                                            cols[i] = v;
                                        }
                                    }
                                    orch_.bcsr_mgr->rowIndexPut(block_row, std::move(cols));
                                }
                                bcsr_rowidx_file_preloaded_ = true;
                                if (diag_out_ && diag_out_->getVerboseLevel() >= 2) {
                                    diag_out_->verbose(CALL_INFO, 2, 0,
                                        "[diag-bcsr-colidx-file] node=%d core=%d preload all_rows ok bytes=%zu\n",
                                        diag_node_id_, diag_core_id_, all.size());
                                }
                            }
                        }
                    }
                }
            }
            if (!ok && orch_.ensure_rowptr_ready_or_fatal) {
                orch_.ensure_rowptr_ready_or_fatal("rowptr load failed (dram)");
            }
            if (orch_.bcsr_mgr->isRowptrReady()) {
                // Rowptr 就绪后仅恢复 Apply 阶段发起（BCSR 预取/缓存优化已全局禁用）
                if (orch_.resume_issue_after_rowptr_ready) {
                    orch_.resume_issue_after_rowptr_ready();
                }
            }
        }
        return;
    }

    if (meta.bcsr_kind == 2) {
        // colidx
        if (!orch_.bcsr_mgr) {
            if (meta.has_single_cb && meta.single_cb) meta.single_cb(0.0f);
            return;
        }
        if (bytes.size() < meta.size) {
            if (meta.has_single_cb && meta.single_cb) meta.single_cb(0.0f);
            return;
        }
        const size_t slice_off = meta.slice_offset;
        const size_t slice_len = meta.orig_size ? meta.orig_size : meta.size;
        if (bytes.size() < slice_off + slice_len) {
            if (meta.has_single_cb && meta.single_cb) meta.single_cb(0.0f);
            return;
        }
        const uint8_t* slice = bytes.data() + slice_off;
        const uint32_t idx_bytes = orch_.bcsr_mgr->effectiveIdxBytes();
        const bool rowindex_prefetch_only =
            !meta.has_single_cb &&
            meta.bcsr_target_block_col == UINT32_MAX &&
            !meta.bcsr_prefetch_all &&
            !meta.bcsr_colidx_bulk_all_rows;
        auto noteRowidxReadyFromResponse = [&]() {
            (void)notePeInternalPodServiceObjectReady_(
                PodMetadataObjectPlane::MetadataKind::RowIndex,
                static_cast<uint64_t>(meta.bcsr_block_row),
                meta.window_seq);
        };

        if (meta.bcsr_colidx_bulk_all_rows) {
            const uint32_t br = orch_.bcsr_mgr->effectiveBlockRows();
            const uint32_t nBlockRows =
                br ? ((orch_.num_neurons + br - 1u) / br) : static_cast<uint32_t>(orch_.num_neurons);
            uint32_t last_start = 0;
            uint32_t last_end = 0;
            const bool bounds_ok =
                (nBlockRows > 0) && orch_.bcsr_mgr->rowBounds(nBlockRows - 1u, last_start, last_end);
            const uint32_t total_blocks = bounds_ok ? last_end : 0;
            const size_t expect_bytes = total_blocks ? orch_.bcsr_mgr->colIndexBytes(total_blocks) : 0;

            const bool ok = bounds_ok && expect_bytes > 0 && slice_len >= expect_bytes && (idx_bytes == 2 || idx_bytes == 4);
            if (!ok) {
                row_index_prefetch_bulk_inflight_ = false;
                // Fail any waiters in this window to avoid deadlock.
                std::vector<uint64_t> keys;
                keys.reserve(inflight_colidx_.size());
                for (const auto& kv : inflight_colidx_) {
                    if (kv.second.window_seq == meta.window_seq) keys.push_back(kv.first);
                }
                for (uint64_t k : keys) {
                    auto it = inflight_colidx_.find(k);
                    if (it == inflight_colidx_.end()) continue;
                    auto waiters = std::move(it->second.waiters);
                    inflight_colidx_.erase(it);
                    for (auto& w : waiters) {
                        if (w.cb) w.cb(0.0f);
                    }
                }
                return;
            }

            // Populate row_index_cache_ for all rows from a single contiguous colidx slice.
            for (uint32_t block_row = 0; block_row < nBlockRows; ++block_row) {
                uint32_t start = 0;
                uint32_t end = 0;
                if (!orch_.bcsr_mgr->rowBounds(block_row, start, end)) break;
                if (end <= start) continue;
                const uint32_t block_count = end - start;
                const size_t off = static_cast<size_t>(start) * static_cast<size_t>(idx_bytes);
                const size_t need = static_cast<size_t>(block_count) * static_cast<size_t>(idx_bytes);
                if (off + need > slice_len) break;

                std::vector<uint32_t> cols(block_count);
                if (idx_bytes == 2) {
                    for (uint32_t i = 0; i < block_count; ++i) {
                        uint16_t v = 0;
                        std::memcpy(&v, slice + off + static_cast<size_t>(i) * 2u, 2u);
                        cols[i] = v;
                    }
                } else {
                    for (uint32_t i = 0; i < block_count; ++i) {
                        uint32_t v = 0;
                        std::memcpy(&v, slice + off + static_cast<size_t>(i) * 4u, 4u);
                        cols[i] = v;
                    }
                }
                orch_.bcsr_mgr->rowIndexPut(block_row, std::move(cols));
            }

            row_index_prefetch_bulk_inflight_ = false;

            // A: resolve any coalesced waiters for this window.
            std::vector<uint64_t> keys;
            keys.reserve(inflight_colidx_.size());
            for (const auto& kv : inflight_colidx_) {
                if (kv.second.window_seq == meta.window_seq) keys.push_back(kv.first);
            }
            for (uint64_t k : keys) {
                auto inflight_it = inflight_colidx_.find(k);
                if (inflight_it == inflight_colidx_.end()) continue;
                ColidxInflight inflight = std::move(inflight_it->second);
                inflight_colidx_.erase(inflight_it);

                std::vector<uint32_t> cached_cols;
                if (!orch_.bcsr_mgr->rowIndexGet(inflight.block_row, cached_cols) || cached_cols.empty()) {
                    for (auto& w : inflight.waiters) {
                        if (w.cb) w.cb(0.0f);
                    }
                    continue;
                }

                std::unordered_map<uint32_t, uint32_t> idx_lookup;
                if (inflight.waiters.size() > 1) {
                    idx_lookup.reserve(cached_cols.size());
                    for (size_t i = 0; i < cached_cols.size(); ++i) {
                        idx_lookup[cached_cols[i]] = static_cast<uint32_t>(i);
                    }
                }
                auto findIdx = [&](uint32_t target, uint32_t& out_idx) -> bool {
                    if (!idx_lookup.empty()) {
                        auto it = idx_lookup.find(target);
                        if (it == idx_lookup.end()) return false;
                        out_idx = it->second;
                        return true;
                    }
                    for (size_t i = 0; i < cached_cols.size(); ++i) {
                        if (cached_cols[i] == target) {
                            out_idx = static_cast<uint32_t>(i);
                            return true;
                        }
                    }
                    return false;
                };

                for (auto& w : inflight.waiters) {
                    uint32_t idx_in_row = 0;
                    if (!findIdx(w.target_block_col, idx_in_row)) {
                        if (w.cb) w.cb(0.0f);
                        continue;
                    }
                    const uint32_t global_block_index = inflight.row_start + idx_in_row;
                    enqueueBcsrBlockReadCoalesced_(meta.window_seq,
                                                   inflight.block_row,
                                                   w.target_block_col,
                                                   global_block_index,
                                                   idx_in_row,
                                                   w.intra_row,
                                                   w.intra_col,
                                                   w.pre_global,
                                                   w.post_local,
                                                   std::move(w.cb));
                }
            }
            return;
        }

        const size_t n = idx_bytes ? (slice_len / idx_bytes) : 0;
        std::vector<uint32_t> cols(n);
        if (idx_bytes == 2) {
            for (size_t i = 0; i < n; ++i) cols[i] = reinterpret_cast<const uint16_t*>(slice)[i];
        } else {
            for (size_t i = 0; i < n; ++i) cols[i] = reinterpret_cast<const uint32_t*>(slice)[i];
        }
        // 诊断：node0/core0 仅采样部分 colidx 内容，验证与文件偏移一致
        static int dbg_colidx_dump = 0;
        if (diag_debug_ && diag_out_ && diag_out_->getVerboseLevel() >= 2 &&
            diag_node_id_ == 0 && diag_core_id_ == 0 && dbg_colidx_dump < 16 && !cols.empty()) {
            const size_t dump = std::min<size_t>(cols.size(), 8);
            std::string first_vals;
            for (size_t i = 0; i < dump; ++i) {
                if (i) first_vals.push_back(',');
                first_vals += std::to_string(cols[i]);
            }
            const size_t dump_bytes = std::min<size_t>(slice_len, 16);
            std::string first_bytes;
            for (size_t i = 0; i < dump_bytes; ++i) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "%02x", slice[i]);
                if (i) first_bytes.push_back(' ');
                first_bytes += buf;
            }
            diag_out_->verbose(CALL_INFO, 2, 0,
                "[diag-bcsr-colidx-dump] node=0 core=0 block_row=%u start=%u end=%u count=%zu first=%s idx_bytes=%u bytes=[%s] slice_off=%zu\n",
                meta.bcsr_block_row, meta.bcsr_row_start, meta.bcsr_row_start + (uint32_t)cols.size(),
                cols.size(), first_vals.c_str(), idx_bytes, first_bytes.c_str(), slice_off);
            ++dbg_colidx_dump;
        }
        // 重要：
        // - 某些实验会全局关闭 BCSR 缓存（rowIndex/block cache），此时 rowIndexPut/get 为 no-op。
        // - 但 colidx 回包仍必须能定位 target_block_col；否则所有 BCSR 权重读都会退化为 0（功能错误）。
        // 因此：本次回包逻辑始终使用本地解析得到的 `cols`；仅在缓存启用时再写入 rowIndex cache。
        const uint32_t row_index_cap = orch_.bcsr_mgr->rowIndexCacheCapacity();
        if (meta.bcsr_prefetch_all) {
            bcsrPrefetchRowBlocks_(meta.bcsr_block_row, cols, meta.bcsr_row_start);
            if (row_index_cap != 0) orch_.bcsr_mgr->rowIndexPut(meta.bcsr_block_row, std::move(cols));
            return;
        }

        const uint64_t inflight_key = makeInflightKey_(meta.window_seq, meta.bcsr_block_row);

        // A: coalesced waiters for this (window, block_row)
        auto inflight_it = inflight_colidx_.find(inflight_key);
        if (inflight_it != inflight_colidx_.end()) {
            auto waiters = std::move(inflight_it->second.waiters);
            const uint32_t row_start = inflight_it->second.row_start;
            inflight_colidx_.erase(inflight_it);
            // Build quick index (block_col -> idx_in_row) once for this row.
            std::unordered_map<uint32_t, uint32_t> idx_lookup;
            if (waiters.size() > 1 && !cols.empty()) {
                idx_lookup.reserve(cols.size());
                for (size_t i = 0; i < cols.size(); ++i) {
                    idx_lookup[cols[i]] = static_cast<uint32_t>(i);
                }
            }
            auto findIdx = [&](uint32_t target, uint32_t& out_idx) -> bool {
                if (!idx_lookup.empty()) {
                    auto it = idx_lookup.find(target);
                    if (it == idx_lookup.end()) return false;
                    out_idx = it->second;
                    return true;
                }
                for (size_t i = 0; i < cols.size(); ++i) {
                    if (cols[i] == target) {
                        out_idx = static_cast<uint32_t>(i);
                        return true;
                    }
                }
                return false;
            };

            for (auto& w : waiters) {
                uint32_t idx_in_row = 0;
                if (!findIdx(w.target_block_col, idx_in_row)) {
                    if (w.cb) w.cb(0.0f);
                    continue;
                }
                const uint32_t global_block_index = row_start + idx_in_row;
                enqueueBcsrBlockReadCoalesced_(meta.window_seq,
                                               meta.bcsr_block_row,
                                               w.target_block_col,
                                               global_block_index,
                                               idx_in_row,
                                               w.intra_row,
                                               w.intra_col,
                                               w.pre_global,
                                               w.post_local,
                                               std::move(w.cb));
            }
            if (rowindex_prefetch_only) {
                noteRowidxReadyFromResponse();
            }
            if (row_index_cap != 0) orch_.bcsr_mgr->rowIndexPut(meta.bcsr_block_row, std::move(cols));
            return;
        }

        uint32_t idx_in_row = 0;
        bool found = false;
        for (size_t i = 0; i < cols.size(); ++i) {
            if (cols[i] == meta.bcsr_target_block_col) {
                idx_in_row = static_cast<uint32_t>(i);
                found = true;
                break;
            }
        }
        if (!found) {
            if (rowindex_prefetch_only) noteRowidxReadyFromResponse();
            if (meta.has_single_cb && meta.single_cb) meta.single_cb(0.0f);
            if (row_index_cap != 0) orch_.bcsr_mgr->rowIndexPut(meta.bcsr_block_row, std::move(cols));
            return;
        }
        const uint32_t global_block_index = meta.bcsr_row_start + idx_in_row;
        if (row_index_cap != 0) orch_.bcsr_mgr->rowIndexPut(meta.bcsr_block_row, std::move(cols));
        if (rowindex_prefetch_only) noteRowidxReadyFromResponse();
        enqueueBcsrBlockReadCoalesced_(meta.window_seq,
                                       meta.bcsr_block_row,
                                       meta.bcsr_target_block_col,
                                       global_block_index,
                                       idx_in_row,
                                       meta.bcsr_intra_row,
                                       meta.bcsr_intra_col,
                                       /*pre_global=*/meta.bcsr_target_block_col * orch_.bcsr_mgr->effectiveBlockCols() + meta.bcsr_intra_col,
                                       /*post_local=*/meta.bcsr_block_row * orch_.bcsr_mgr->effectiveBlockRows() + meta.bcsr_intra_row,
                                       std::move(meta.single_cb));
        return;
    }

    if (meta.bcsr_kind == 3) {
        // block data
        if (!orch_.bcsr_mgr) {
            if (meta.has_single_cb && meta.single_cb) meta.single_cb(0.0f);
            return;
        }
        const uint32_t br = orch_.bcsr_mgr->effectiveBlockRows();
        const uint32_t bc = orch_.bcsr_mgr->effectiveBlockCols();
        const size_t slice_off = meta.slice_offset;
        const size_t slice_len = meta.orig_size ? meta.orig_size : meta.size;
        const size_t row_expect_bytes = static_cast<size_t>(bc) * sizeof(float);
        const size_t full_expect_bytes = static_cast<size_t>(br) * static_cast<size_t>(bc) * sizeof(float);
        const bool row_cacheline_fetch = meta.bcsr_row_slice_fetch;
        const size_t expect_bytes = row_cacheline_fetch ? row_expect_bytes : full_expect_bytes;
        const bool ok = (bytes.size() >= slice_off + slice_len) && (slice_len >= expect_bytes);
        if (diag_debug_ && diag_out_ && diag_out_->getVerboseLevel() >= 2 &&
            diag_node_id_ == 0 && diag_core_id_ == 0) {
            static int dbg_block_resp = 0;
            if (dbg_block_resp < 2048) {
                float f0 = 0.0f;
                if (bytes.size() >= slice_off + sizeof(float)) {
                    std::memcpy(&f0, bytes.data() + slice_off, sizeof(float));
                }
                diag_out_->verbose(CALL_INFO, 2, 0,
                    "[diag-bcsr-block-resp] node=%d core=%d block_row=%u block_col=%u gbi=%u addr=0x%llx bytes=%zu slice_off=%zu f0=%.6f ok=%d\n",
                    diag_node_id_, diag_core_id_, meta.bcsr_block_row, meta.bcsr_target_block_col,
                    meta.bcsr_global_block_index, (unsigned long long)meta.address,
                    bytes.size(), slice_off, f0, ok ? 1 : 0);
                ++dbg_block_resp;
            }
        }
        if (ok && expect_bytes > 0) {
            if (row_cacheline_fetch) {
                std::vector<float> row_vals(static_cast<size_t>(bc), 0.0f);
                std::memcpy(row_vals.data(), bytes.data() + slice_off, row_expect_bytes);

                const uint64_t inflight_key = makeBlockInflightKey_(
                    meta.window_seq, meta.bcsr_global_block_index, meta.bcsr_intra_row);
                auto inflight_it = inflight_block_.find(inflight_key);
                if (inflight_it != inflight_block_.end()) {
                    auto waiters = std::move(inflight_it->second.waiters);
                    inflight_block_.erase(inflight_it);
                    for (auto& w : waiters) {
                        float ww = 0.0f;
                        if (w.intra_row == meta.bcsr_intra_row) {
                            const uint32_t off = w.intra_col;
                            ww = (off < row_vals.size()) ? row_vals[off] : 0.0f;
                        } else if (orch_.read_bcsr_from_file) {
                            ww = orch_.read_bcsr_from_file(w.post_local, w.pre_global);
                        }
                        const float guarded = applyWeightGuards_(ww);
                        verifyBcsrEdgeWeight_(w.pre_global, w.post_local, guarded);
                        if (w.cb) w.cb(guarded);
                    }
                } else if (meta.has_single_cb && meta.single_cb) {
                    const uint32_t off = meta.bcsr_intra_col;
                    float w = (off < row_vals.size()) ? row_vals[off] : 0.0f;
                    const float guarded = applyWeightGuards_(w);
                    verifyBcsrEdgeWeight_(
                        /*pre_global=*/meta.bcsr_target_block_col * bc + meta.bcsr_intra_col,
                        /*post_local=*/meta.bcsr_block_row * br + meta.bcsr_intra_row,
                        guarded);
                    meta.single_cb(guarded);
                }
            } else {
                const size_t n = static_cast<size_t>(br) * static_cast<size_t>(bc);
                std::vector<float> blk(n, 0.0f);
                std::memcpy(blk.data(), bytes.data() + slice_off, full_expect_bytes);

                // Deliver callbacks directly from the returned block bytes to support "block_cache_cap=0"
                // (naive baseline). Caching, if enabled, is applied after callbacks.
                const uint64_t inflight_key = makeBlockInflightKey_(
                    meta.window_seq, meta.bcsr_global_block_index, /*intra_row=*/0);
                auto inflight_it = inflight_block_.find(inflight_key);
                if (inflight_it != inflight_block_.end()) {
                    auto waiters = std::move(inflight_it->second.waiters);
                    inflight_block_.erase(inflight_it);
                    for (auto& w : waiters) {
                        const uint32_t off = w.intra_row * bc + w.intra_col;
                        const float ww = (off < blk.size()) ? blk[off] : 0.0f;
                        const float guarded = applyWeightGuards_(ww);
                        verifyBcsrEdgeWeight_(w.pre_global, w.post_local, guarded);
                        if (w.cb) w.cb(guarded);
                    }
                } else if (meta.has_single_cb && meta.single_cb) {
                    const uint32_t off = meta.bcsr_intra_row * bc + meta.bcsr_intra_col;
                    float w = (off < blk.size()) ? blk[off] : 0.0f;
                    const float guarded = applyWeightGuards_(w);
                    verifyBcsrEdgeWeight_(
                        /*pre_global=*/meta.bcsr_target_block_col * bc + meta.bcsr_intra_col,
                        /*post_local=*/meta.bcsr_block_row * br + meta.bcsr_intra_row,
                        guarded);
                    meta.single_cb(guarded);
                }

                // Optional: populate dense weight cache from this returned block (optimization).
                bcsrPopulateWeightCache_(meta.bcsr_block_row, meta.bcsr_target_block_col, blk);

                // Optional: cache this block for subsequent hits.
                if (orch_.bcsr_mgr->blockCacheCapacity() != 0) {
                    orch_.bcsr_mgr->blockPut(meta.bcsr_block_row, meta.bcsr_target_block_col, std::move(blk));
                }
            }
        } else {
            const uint64_t inflight_key = makeBlockInflightKey_(
                meta.window_seq,
                meta.bcsr_global_block_index,
                row_cacheline_fetch ? meta.bcsr_intra_row : 0);
            auto inflight_it = inflight_block_.find(inflight_key);
            if (inflight_it != inflight_block_.end()) {
                auto waiters = std::move(inflight_it->second.waiters);
                inflight_block_.erase(inflight_it);
                for (auto& w : waiters) {
                    float ww = 0.0f;
                    if (orch_.read_bcsr_from_file) {
                        ww = orch_.read_bcsr_from_file(w.post_local, w.pre_global);
                    }
                    if (w.cb) w.cb(applyWeightGuards_(ww));
                }
                return;
            }

            // 兜底：仅回调单个权重（不强行回填整个块），避免破坏性能/复杂性
            if (meta.has_single_cb && meta.single_cb) {
                float w = 0.0f;
                if (orch_.read_bcsr_from_file) {
                    const uint32_t post_local = meta.bcsr_block_row * br + meta.bcsr_intra_row;
                    const uint32_t pre_global = meta.bcsr_target_block_col * bc + meta.bcsr_intra_col;
                    w = orch_.read_bcsr_from_file(post_local, pre_global);
                }
                meta.single_cb(applyWeightGuards_(w));
            }
        }
        return;
    }

    // ---- Dense ----
    const uint32_t width = orch_.use_post_row_pre_col ? orch_.weights_cols : orch_.num_neurons;
    if (width > 0 && fptr) {
        for (size_t i = 0; i < float_count; ++i) {
            const uint32_t col_idx = meta.post_start + static_cast<uint32_t>(i);
            if (col_idx >= width) break;
            float w = applyReadRespZeroFallback_(fptr[i]);
            const uint64_t key = static_cast<uint64_t>(meta.pre) * static_cast<uint64_t>(width) +
                                 static_cast<uint64_t>(col_idx);
            if (orch_.cache_put) orch_.cache_put(key, w);
            else if (cache_put_fn_) cache_put_fn_(key, w);
        }
    }
    if (meta.has_single_cb && meta.single_cb) {
        float w = 0.0f;
        const uint32_t target_col = meta.cb_post;
        if (fptr && width > 0 &&
            target_col >= meta.post_start &&
            (target_col - meta.post_start) < float_count &&
            target_col < width) {
            w = fptr[static_cast<size_t>(target_col - meta.post_start)];
        }
        meta.single_cb(applyReadRespZeroFallback_(w));
    }
}
