// -*- c++ -*-
//
// WeightMemorySubsystem:
// - 向 compute core 提供统一的 IWeightReader（dense/BCSR + cache）
// - 承载 window-read 的集合/预算/并发/outstanding 与发起编排（Phase A）
//   使控制层仅在窗口边界与 recordEdge/recordTouch 上触发即可。

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cinttypes>
#include <functional>
#include <deque>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "synapse/gas/GasEdgeCollector.h"
#include "SnnWeightReader.h"
#include "IMemoryAccess.h"
#include "WeightAccessor.h"

namespace SST { class Output; }

namespace SST { namespace SnnDL {

class BcsrWeightManager;

class WeightMemorySubsystem : public IWeightReader {
public:
    using CacheTryFn = std::function<bool(uint64_t,float&)>;
    using CachePutFn = std::function<void(uint64_t,float)>;

    using RequestFn = std::function<bool(uint32_t,uint32_t,std::function<void(float)>)>;
    using AccUpdateFn = std::function<void(uint32_t,float)>;
    using DiagEdgeFn = std::function<void(const char*,uint32_t,uint32_t,float,uint32_t)>;
    using CacheReportFn = std::function<void(bool)>;
    using BoolFn = std::function<bool()>;
    using PendingPeakFn = std::function<void(uint32_t)>;
    using ReadBcsrFileFn = std::function<float(uint32_t,uint32_t)>;
    using MemIssueFn = std::function<void(size_t,bool)>;
    using MemLatencyFn = std::function<void(uint64_t,bool)>;
    using VoidFn = std::function<void()>;
    using EnsureRowptrFn = std::function<void(const char*)>;

    struct WindowCounters {
        uint32_t budget = 0;
        uint32_t issued = 0;
        uint32_t outstanding = 0;
        uint32_t peak_outstanding = 0;
        uint32_t max_outstanding = 0;
    };

    struct OrchestratorConfig {
        WeightAccessor* accessor = nullptr;
        RequestFn request_dense;
        RequestFn request_bcsr;
        CacheTryFn cache_try;
        CachePutFn cache_put;
        AccUpdateFn acc_update;
        DiagEdgeFn diag_edge_weight;
        CacheReportFn report_cache_access;
        PendingPeakFn update_pending_peak;
        MemIssueFn report_mem_issue;
        MemLatencyFn report_mem_latency;
        VoidFn on_scheme1_prefetch_resp;
        BoolFn ensure_loader_ready;
        BoolFn bcsr_rowptr_ready;
        EnsureRowptrFn ensure_rowptr_ready_or_fatal;
        VoidFn resume_issue_after_rowptr_ready;
        ReadBcsrFileFn read_bcsr_from_file;
        bool use_bcsr = false;
        bool bcsr_force_file_read = false;
        bool bcsr_rowptr_file_fallback_enable = false;
        bool bcsr_prefetch_all = false;
        // PhaseX: naive baseline needs a way to disable in-flight coalescing without disabling BCSR format.
        bool bcsr_colidx_inflight_coalesce_enable = true;
        bool bcsr_block_inflight_coalesce_enable = true;
        // RowIndex(colidx) 冷启动/热路径优化：默认启用 "auto"（当 colidx 总量很小才用 all_rows；否则用 posts_prev）。
        std::string bcsr_row_index_prefetch_mode = "auto"; // off/all_rows/posts_prev/auto
        uint32_t bcsr_row_index_prefetch_all_rows_threshold = 1024;
        uint64_t bcsr_row_index_prefetch_all_rows_max_bytes = 64ull * 1024ull;
        // Block 缓存稳定性/性能：默认启用 LRU（由 BcsrWeightManager 托管），并允许按 miss 比率自适应扩容。
        bool bcsr_block_cache_auto_tune = true;
        uint64_t bcsr_block_cache_max_bytes = 64ull * 1024ull * 1024ull;
        float bcsr_block_cache_tune_miss_ratio = 0.05f;
        uint32_t bcsr_block_cache_tune_min_misses = 64;
        // Populate dense weight cache from returned BCSR blocks (optimization; disable for naive_raw).
        bool bcsr_populate_weight_cache_enable = true;
        bool bcsr_weight_guard_enable = true;
        float bcsr_weight_abs_max = 10.0f;
        // BCSR semantic correctness (sampled): compare runtime weights against file-backed reference.
        // This is a debug/regression-only feature; keep max_edges small to avoid performance impact.
        bool bcsr_semantic_verify_enable = false;
        uint32_t bcsr_semantic_verify_max_edges = 64;
        uint32_t bcsr_semantic_verify_max_mismatch = 8;
        float bcsr_semantic_verify_abs_tol = 1e-6f;
        float bcsr_semantic_verify_rel_tol = 1e-6f;
        bool readresp_zero_fallback = false;
        float init_default_weight = 0.5f;
        uint32_t num_neurons = 0;
        uint32_t weights_cols = 0;
        bool use_post_row_pre_col = false;
        uint64_t base_addr = 0;
        uint64_t weight_region_end = 0;
        bool read_force_single = false;
        bool merge_read_cacheline = true;
        bool merge_read_row = false;
        bool merge_read_auto = false;
        uint32_t line_size_bytes = 64;
        uint64_t memory_warmup_cycles = 0;
        uint64_t loader_barrier_cycles = 0;
        uint32_t node_id = 0;
        uint32_t core_id = 0;
        std::string weights_template;
        BcsrWeightManager* bcsr_mgr = nullptr;
        bool byte_exact_verify_enable = false;
        std::string byte_exact_verify_mode;
        uint32_t byte_exact_verify_row_scale = 1024;
        uint32_t byte_exact_verify_max_mismatch = 8;
    };

    WeightMemorySubsystem() = default;

    // ===== Runtime binding (Phase1) =====
    void bindMemory(IMemoryAccess* mem) { mem_access_ = mem; }
    size_t pendingSize() const { return mem_access_ ? mem_access_->pendingSize() : 0; }
    bool hasDeferredWork() const {
        return !pending_bcsr_rowptr_waiters_.empty() ||
               row_index_prefetch_bulk_pending_ ||
               row_index_prefetch_bulk_inflight_ ||
               !row_index_prefetch_rows_.empty() ||
               !pending_colidx_reads_.empty() ||
               !pending_block_reads_.empty() ||
               !pending_direct_reads_.empty();
    }
    void setNowCycle(uint64_t now_cycle) { now_cycle_ = now_cycle; }
    void onClockTick(uint64_t now_cycle);
    // 低层 dense 预取发起（scheme1 baseline 使用）：由调用方提供地址/大小与 row/col 起点。
    bool issueDensePrefetchRaw(uint64_t req_addr, size_t req_size,
                               uint32_t row, uint32_t col_start, uint32_t count_floats,
                               bool scheme1_prefetch);

    void configure(CacheTryFn cache_try_fn,
                   CachePutFn cache_put_fn) {
        cache_try_fn_ = std::move(cache_try_fn);
        cache_put_fn_ = std::move(cache_put_fn);
    }

    void configureOrchestrator(OrchestratorConfig cfg) {
        orch_ = std::move(cfg);
        ensureWindowTracking(orch_.num_neurons);
        if (orch_.bcsr_semantic_verify_enable) {
            bcsr_sem_verified_edges_ = 0;
            bcsr_sem_mismatch_count_ = 0;
            bcsr_sem_mismatch_logged_ = 0;
            bcsr_sem_pass_logged_ = false;
            bcsr_sem_inconclusive_ = false;
            bcsr_sem_inconclusive_reason_.clear();
        }
    }

    // Phase4-Task6.4: allow workload to rebind accumulator callback without rebuilding all orchestrator config.
    // This is used when GAS/window orchestration moves into workload=snn and accumulator state is owned there.
    void overrideAccUpdate(AccUpdateFn fn) { orch_.acc_update = std::move(fn); }

    // ===== Read issuance counters (budget/inflight) =====
    // NOTE:
    // - issued: per-window issued "real reads" (issueRead_ calls), used for budget.
    // - outstanding: global in-flight reads (issued but not yet resp), used for max_outstanding_requests throttle.
    // This avoids the previous "edge-count throttle" bug: after colidx/block coalescing, many edges map to
    // few real reads; throttling by edges would underfill inflight and inflate Apply latency.
    void configureWindow(uint32_t budget_reads, uint32_t max_inflight_reads) {
        window_.budget = budget_reads;
        window_.max_outstanding = max_inflight_reads;
    }
    void beginWindow() {
        window_.issued = 0;
        window_.peak_outstanding = window_.outstanding;
    }
    bool canIssue(uint32_t n = 1, bool count_budget = true) const {
        const bool budget_ok =
            (!count_budget) ||
            (window_seq_ == 0) ||
            (window_.budget == 0) ||
            (window_.issued + n <= window_.budget);
        const bool ostd_ok = (window_.outstanding + n <= window_.max_outstanding);
        return budget_ok && ostd_ok;
    }
    void noteIssue(uint32_t n = 1, bool count_budget = true) {
        if (count_budget && window_seq_ != 0) {
            window_.issued += n;
        }
        window_.outstanding += n;
        if (window_.outstanding > window_.peak_outstanding) {
            window_.peak_outstanding = window_.outstanding;
        }
    }
    void noteComplete(uint32_t n = 1) {
        if (window_.outstanding >= n) window_.outstanding -= n;
        else window_.outstanding = 0;
    }
    uint32_t issued() const { return window_.issued; }
    uint32_t outstanding() const { return window_.outstanding; }
    uint32_t peakOutstanding() const { return window_.peak_outstanding; }
    uint32_t budget() const { return window_.budget; }
    uint32_t maxOutstanding() const { return window_.max_outstanding; }

    // ===== Window tracking (posts/pres) =====
    void reserveWindowContainers(uint32_t num_neurons) {
        const size_t post_cap = std::max<size_t>(64, static_cast<size_t>(num_neurons) / 8);
        posts_list_window_.reserve(post_cap);
        posts_list_prev_window_.reserve(post_cap);
        active_pre_window_.reserve(256);
        active_pre_prev_window_.reserve(256);
    }

    void ensureWindowTracking(uint32_t num_neurons) {
        if (num_neurons == 0) return;
        if (posts_seen_window_.size() != num_neurons) {
            posts_seen_window_.assign(num_neurons, 0);
        }
        if (posts_seen_prev_window_.size() != num_neurons) {
            posts_seen_prev_window_.assign(num_neurons, 0);
        }
    }

    // 在 BeginGather 时调用：将上一个窗口的 curr 迁移到 prev，并清空 curr。
    void beginGatherWindow(bool window_read_enable, uint32_t num_neurons) {
        if (!window_read_enable) return;
        ensureWindowTracking(num_neurons);
        posts_seen_prev_window_ = posts_seen_window_;
        std::fill(posts_seen_window_.begin(), posts_seen_window_.end(), 0);
        posts_list_prev_window_.swap(posts_list_window_);
        posts_list_window_.clear();
        active_pre_prev_window_ = std::move(active_pre_window_);
        active_pre_window_.clear();
    }

    void notePostLocal(uint32_t post_local, uint32_t num_neurons) {
        if (post_local >= num_neurons) return;
        ensureWindowTracking(num_neurons);
        if (!posts_seen_window_[post_local]) {
            posts_seen_window_[post_local] = 1;
            posts_list_window_.push_back(post_local);
        }
    }

    void notePreGlobal(uint32_t pre_global) {
        active_pre_window_.insert(pre_global);
    }

    void noteWindowTouch(uint32_t post_local, uint32_t pre_global, uint32_t num_neurons) {
        notePostLocal(post_local, num_neurons);
        notePreGlobal(pre_global);
    }

    size_t postsCurrSize() const { return posts_list_window_.size(); }
    size_t postsPrevSize() const { return posts_list_prev_window_.size(); }
    size_t presCurrSize() const { return active_pre_window_.size(); }
    size_t presPrevSize() const { return active_pre_prev_window_.size(); }
    const std::vector<uint32_t>& postsCurr() const { return posts_list_window_; }
    const std::vector<uint32_t>& postsPrev() const { return posts_list_prev_window_; }
    const std::unordered_set<uint32_t>& presCurr() const { return active_pre_window_; }
    const std::unordered_set<uint32_t>& presPrev() const { return active_pre_prev_window_; }

    // ===== Edge collection =====
    void recordEdge(uint32_t post_local, uint32_t pre_global) {
        uint64_t key = (static_cast<uint64_t>(post_local) << 32) | static_cast<uint64_t>(pre_global);
        edge_collector_.curr[key] += 1;
    }
    size_t edgesCurrSize() const { return edge_collector_.currSize(); }
    size_t edgesPrevSize() const { return edge_collector_.prevSize(); }
    size_t edgesPrevIter() const { return edge_collector_.prevIter(); }
    bool edgesPrevEmpty() const { return edge_collector_.prevEmpty(); }
    void flipEdgesForApply(bool debug, SST::Output* out, int core_id, uint32_t seq) {
        edge_collector_.flipForApply(debug, out, core_id, seq);
    }
    bool nextPrevEdge(uint64_t& key, uint32_t& count) { return edge_collector_.nextPrev(key, count); }

    // ===== Orchestration entrypoints =====
    void beginApplyWindow(uint32_t seq, bool debug, SST::Output* out, int core_id) {
        window_seq_ = seq;
        // 若上一窗口未显式收尾（例如仿真提前结束/卡在 BeginApply），在新窗口开启前先输出上一窗口诊断摘要。
        if (diag_debug_ && diag_window_active_ && diag_seq_ != 0 && diag_seq_ != seq) {
            dumpWindowDiagSummary_("[diag-window-weights] rollover");
            resetWindowDiag_();
        }

        diag_out_ = out;
        diag_debug_ = debug;
        diag_core_id_ = core_id;
        diag_seq_ = seq;
        if (diag_debug_) {
            diag_window_active_ = true;
            diag_window_seq_ = seq;
            resetWindowDiag_();
        } else {
            diag_window_active_ = false;
            diag_window_seq_ = 0;
        }
        if (orch_.use_bcsr && orch_.bcsr_rowptr_ready && !orch_.bcsr_rowptr_ready()) {
            // defer until rowptr ready (preserve old behavior)
        }
        flipEdgesForApply(debug, out, core_id, seq);
        beginWindow();
        block_hit_window_ = 0;
        block_miss_window_ = 0;
        resetEdgeRetire_();
        if (byteExactVerifyEnabled_()) {
            byte_exact_mismatch_count_ = 0;
            byte_exact_mismatch_logged_ = 0;
            byte_exact_verified_reads_ = 0;
            byte_exact_verified_edges_ = 0;
            byte_exact_pass_logged_ = false;
        }
        // RowIndex(colidx) 预取：用于降低 window1/2 的 rowidx_miss，以及提升后续窗口命中率（不改语义，仅提前读）。
        if (orch_.use_bcsr) {
            maybeEnqueueRowIndexPrefetchPostsPrev_();
            drainRowIndexPrefetch_();
        }
        diag_node_id_ = static_cast<int>(orch_.node_id);
    }

    // 仅用于调试：在 EndScatter 或仿真结束前输出当前窗口的权重读摘要。
    // 该函数无行为副作用；仅打印并清空本地诊断计数器。
    void endScatterWindow(uint32_t seq) {
        (void)seq;
        // Always perform non-diagnostic housekeeping (applies to all cores, not only debug targets).
        maybeAutoTuneBlockCache_();
        window_seq_ = 0;

        emitByteExactPassMarker_("EndScatter", seq);
        // When semantic verification is enabled, emit a PASS marker once we have verified enough edges.
        if (orch_.bcsr_semantic_verify_enable &&
            !bcsr_sem_pass_logged_ &&
            (bcsr_sem_verified_edges_ >= static_cast<uint64_t>(orch_.bcsr_semantic_verify_max_edges))) {
            emitBcsrSemanticVerifyMarker_("EndScatter", seq);
        }

        if (!diag_debug_ || !diag_out_) return;
        if (!diag_window_active_) return;
        if (diag_window_seq_ != 0 && seq != diag_window_seq_) return;
        dumpWindowDiagSummary_("[diag-window-weights] EndScatter");
        resetWindowDiag_();
        diag_window_active_ = false;
        diag_window_seq_ = 0;
    }

    // 调试兜底：在仿真结束/提前退出时输出当前窗口（可能未完成）的权重读摘要。
    void finishWindowDiag() {
        emitByteExactPassMarker_("finish", /*seq*/0);
        finishSemanticVerify();

        if (!diag_debug_ || !diag_out_) return;
        if (!diag_window_active_) return;
        dumpWindowDiagSummary_("[diag-window-weights] finish");
        resetWindowDiag_();
        diag_window_active_ = false;
        diag_window_seq_ = 0;
        window_seq_ = 0;
    }

    // BCSR semantic correctness marker (independent of window_read_debug).
    // Called from component finish to ensure a positive marker exists even if EndScatter never happens.
    void finishSemanticVerify() { emitBcsrSemanticVerifyMarker_("finish", /*seq*/0); }

    void issueFromEdges() {
        // 防止同步回调导致的深递归/栈溢出：
        // requestBCSR_/requestDense_ 在 cache-hit/块命中等场景可能同步触发回调；
        // 若回调里再次调用 issueFromEdges()，会形成链式递归并最终破坏内存池。
        if (issue_edges_in_progress_) {
            issue_edges_again_ = true;
            return;
        }
        issue_edges_in_progress_ = true;
        do {
            issue_edges_again_ = false;
            issueFromEdgesOnce_();
        } while (issue_edges_again_);
        issue_edges_in_progress_ = false;
    }

private:
    void issueFromEdgesOnce_() {
        if (!orch_.accessor) return;
        if (orch_.ensure_loader_ready && !orch_.ensure_loader_ready()) {
            if (diag_debug_ && diag_out_) {
                diag_out_->verbose(CALL_INFO, 0, 0,
                    "[diag-window-read] BeginApply: core=%u window=%u defer edge issue (loader not ready)\n",
                    static_cast<uint32_t>(diag_core_id_), diag_seq_);
            }
            return;
        }
        if (orch_.use_bcsr && orch_.bcsr_rowptr_ready && !orch_.bcsr_rowptr_ready()) {
            if (diag_debug_ && diag_out_) {
                diag_out_->verbose(CALL_INFO, 0, 0,
                    "[diag-window-read] BeginApply: core=%u window=%u defer edge issue (BCSR rowptr not ready)\n",
                    static_cast<uint32_t>(diag_core_id_), diag_seq_);
            }
            return;
        }

        while (canIssueMoreReads_()) {
            uint64_t key = 0;
            uint32_t count = 0;
            if (!nextPrevEdge(key, count)) {
                if (diag_debug_ && diag_out_) {
                    diag_out_->verbose(CALL_INFO, 0, 0,
                        "[diag-edge-loop] core=%d window=%u no-prev size=%zu iter=%zu",
                        diag_core_id_, diag_seq_, edgesPrevSize(), edgesPrevIter());
                }
                break;
            }
            uint32_t post_local = static_cast<uint32_t>(key >> 32);
            uint32_t pre_global = static_cast<uint32_t>(key & 0xffffffffu);
            if (orch_.num_neurons > 0 && post_local >= orch_.num_neurons) continue;
            const size_t seq = registerEdgeRetire_(post_local, pre_global, count,
                                                  orch_.use_bcsr ? EdgeSrc::BCSR : EdgeSrc::Dense);

            if (orch_.use_bcsr) {
                if (orch_.bcsr_force_file_read && orch_.read_bcsr_from_file) {
                    float resolved = orch_.read_bcsr_from_file(post_local, pre_global);
                    if (orch_.readresp_zero_fallback && resolved == 0.0f) resolved = orch_.init_default_weight;
                    setEdgeRetireReady_(seq, resolved, EdgeSrc::BCSRFile);
                    tryRetireEdges_();
                    continue;
                }
                if (diag_debug_ && diag_out_) {
                    diag_out_->verbose(CALL_INFO, 0, 0,
                        "[diag-read-issue-WMS] core=%d pre=%u post=%u window=%u seq=%zu use_bcsr=1\n",
                        diag_core_id_, pre_global, post_local, diag_seq_, seq);
                }
                auto cb = [this, seq](float w) {
                    float resolved = w;
                    if (orch_.readresp_zero_fallback && resolved == 0.0f) resolved = orch_.init_default_weight;
                    setEdgeRetireReady_(seq, resolved, EdgeSrc::BCSR);
                    tryRetireEdges_();
                    issueFromEdges();
                };
                requestBCSR_(pre_global, post_local, std::move(cb));
                continue;
            }

            uint32_t req_pre = 0;
            uint32_t req_post = 0;
            uint64_t cache_key = 0;
            if (!orch_.accessor->resolve(pre_global, post_local, req_pre, req_post, cache_key)) {
                // 解析失败：不应阻塞 retire；按“默认权重=0”退役（保持不产生更新的语义）
                setEdgeRetireReady_(seq, 0.0f, EdgeSrc::Dense);
                tryRetireEdges_();
                continue;
            }

            float cached = 0.0f;
            const bool cache_hit = orch_.cache_try ? orch_.cache_try(cache_key, cached)
                                                   : (cache_try_fn_ ? cache_try_fn_(cache_key, cached) : false);
            if (cache_hit) {
                if (orch_.readresp_zero_fallback && cached == 0.0f) cached = orch_.init_default_weight;
                setEdgeRetireReady_(seq, cached, EdgeSrc::Cache);
                tryRetireEdges_();
                if (orch_.report_cache_access) orch_.report_cache_access(true);
                continue;
            }
            if (orch_.report_cache_access) orch_.report_cache_access(false);

            noteIssue();
            if (orch_.update_pending_peak) orch_.update_pending_peak(outstanding());
            auto miss_cb = [this, seq, cache_key](float w) {
                float resolved = w;
                if (orch_.readresp_zero_fallback && resolved == 0.0f) resolved = orch_.init_default_weight;
                if (orch_.cache_put) orch_.cache_put(cache_key, resolved);
                else if (cache_put_fn_) cache_put_fn_(cache_key, resolved);
                setEdgeRetireReady_(seq, resolved, EdgeSrc::Miss);
                noteComplete();
                tryRetireEdges_();
                issueFromEdges();
            };

            const uint32_t row_idx = orch_.use_post_row_pre_col ? req_post : req_pre;
            const uint32_t col_idx = orch_.use_post_row_pre_col ? req_pre : req_post;
            issueDenseResolved_(row_idx, col_idx, /*cb_col*/col_idx, std::move(miss_cb));
        }
    }

public:
    void issueFromSets(const std::vector<uint32_t>* posts_to_use,
                       const std::unordered_set<uint32_t>* pres_to_use) {
        if (!orch_.accessor || !posts_to_use || !pres_to_use) return;
        if (orch_.use_bcsr) {
            issueFromSetsBcsr(posts_to_use, pres_to_use);
            return;
        }
        // 确定性发起顺序：unordered_set 的遍历顺序在不同运行中不稳定；
        // 对 pre 集合与 post 列表排序，保证内存请求发起顺序稳定。
        std::vector<uint32_t> pres_sorted;
        pres_sorted.reserve(pres_to_use->size());
        for (const auto& pre_g : *pres_to_use) pres_sorted.push_back(pre_g);
        std::sort(pres_sorted.begin(), pres_sorted.end());
        std::vector<uint32_t> posts_sorted = *posts_to_use;
        std::sort(posts_sorted.begin(), posts_sorted.end());
        uint32_t issued = 0;
        for (const auto& pre_g : pres_sorted) {
            for (uint32_t post_l : posts_sorted) {
                if (!canIssueMoreReads_()) break;
                uint32_t req_pre = 0;
                uint32_t req_post = 0;
                uint64_t cache_key = 0;
                if (!orch_.accessor->resolve(pre_g, post_l, req_pre, req_post, cache_key)) continue;
                if (orch_.report_cache_access) orch_.report_cache_access(false);
                noteIssue();
                if (orch_.update_pending_peak) orch_.update_pending_peak(outstanding());
                issued++;
                auto cache_cb = [this, cache_key](float w) {
                    if (orch_.cache_put) orch_.cache_put(cache_key, w);
                    else if (cache_put_fn_) cache_put_fn_(cache_key, w);
                    noteComplete();
                };
                const uint32_t row_idx = orch_.use_post_row_pre_col ? req_post : req_pre;
                const uint32_t col_idx = orch_.use_post_row_pre_col ? req_pre : req_post;
                issueDenseResolved_(row_idx, col_idx, /*cb_col*/col_idx, std::move(cache_cb));
            }
            if (!canIssueMoreReads_()) break;
        }
        if (diag_debug_ && diag_out_) {
            diag_out_->verbose(CALL_INFO, 0, 0,
                "[diag-window-read] BeginApply: core=%u window=%u issued=%u (this window) outstanding_reqs=%u\n",
                static_cast<uint32_t>(diag_core_id_), diag_seq_, issued, outstanding());
        }
    }

    void issueFromSetsBcsr(const std::vector<uint32_t>* posts_to_use,
                           const std::unordered_set<uint32_t>* pres_to_use) {
        if (!posts_to_use || !pres_to_use) return;
        if (orch_.bcsr_rowptr_ready && !orch_.bcsr_rowptr_ready()) {
            if (diag_debug_ && diag_out_) {
                diag_out_->verbose(CALL_INFO, 0, 0,
                    "[diag-window-read] BeginApply: core=%u window=%u skip set-priming (BCSR rowptr not ready)\n",
                    static_cast<uint32_t>(diag_core_id_), diag_seq_);
            }
            return;
        }
        // 确定性发起顺序：对 pre 集合与 post 列表排序，保证内存请求发起顺序稳定。
        std::vector<uint32_t> pres_sorted;
        pres_sorted.reserve(pres_to_use->size());
        for (const auto& pre_g : *pres_to_use) pres_sorted.push_back(pre_g);
        std::sort(pres_sorted.begin(), pres_sorted.end());
        std::vector<uint32_t> posts_sorted = *posts_to_use;
        std::sort(posts_sorted.begin(), posts_sorted.end());
        uint32_t primed = 0;
        for (const auto& pre_g : pres_sorted) {
            for (uint32_t post_l : posts_sorted) {
                if (!canIssueMoreReads_()) break;
                if (orch_.report_cache_access) orch_.report_cache_access(false);
                primed++;
                auto cb = [](float) {};
                requestBCSR_(pre_g, post_l, std::move(cb));
            }
            if (!canIssueMoreReads_()) break;
        }
        if (diag_debug_ && diag_out_) {
            diag_out_->verbose(CALL_INFO, 0, 0,
                "[diag-window-read] BCSR priming: core=%u window=%u issued=%u outstanding=%u\n",
                static_cast<uint32_t>(diag_core_id_), diag_seq_, primed, outstanding());
        }
    }

    void issueFallbackReadsIfNeeded(bool strict_gas_active) {
        bool need_sets = false;
        if (strict_gas_active) {
            need_sets = edgesPrevEmpty();
        } else {
            need_sets = true;
        }
        if (!need_sets) return;

        const bool have_posts_prev = !postsPrev().empty();
        const bool have_pres_prev  = !presPrev().empty();
        const bool have_posts_curr = !postsCurr().empty();
        const bool have_pres_curr  = !presCurr().empty();

        const bool use_fallback = (!have_posts_prev || !have_pres_prev) && (have_posts_curr && have_pres_curr);
        if ((!have_posts_prev && !have_posts_curr) || (!have_pres_prev && !have_pres_curr)) {
            if (diag_debug_ && diag_out_) {
                diag_out_->verbose(CALL_INFO, 0, 0,
                    "[diag-window-read] BeginApply: core=%u window=%u skip read (both windows empty)\n",
                    static_cast<uint32_t>(diag_core_id_), diag_seq_);
            }
            return;
        }

        const std::vector<uint32_t>* posts = use_fallback ? &postsCurr() : &postsPrev();
        const std::unordered_set<uint32_t>* pres = use_fallback ? &presCurr() : &presPrev();
        issueFromSets(posts, pres);
    }

    // ===== IWeightReader =====
    void requestDense(uint32_t pre, uint32_t post, std::function<void(float)> cb) override {
        requestDense_(pre, post, std::move(cb));
    }

    void requestBCSR(uint32_t pre_global, uint32_t post_local, std::function<void(float)> cb) override {
        requestBCSR_(pre_global, post_local, std::move(cb));
    }

    bool tryCache(uint64_t key, float& out) override {
        return cache_try_fn_ ? cache_try_fn_(key, out) : false;
    }

    void putCache(uint64_t key, float value) override {
        if (cache_put_fn_) cache_put_fn_(key, value);
    }

private:
    bool canIssueMoreReads_() const {
        if (canIssue(/*n*/1, /*count_budget*/true)) return true;
        if (diag_debug_ && diag_out_) {
            diag_out_->verbose(CALL_INFO, 0, 0,
                "[diag-edge-loop] core=%d budget/ostd stop issued=%u outstanding=%u budget=%u limit=%u\n",
                diag_core_id_, issued(), outstanding(), budget(), maxOutstanding());
        }
        return false;
    }

		    CacheTryFn cache_try_fn_;
		    CachePutFn cache_put_fn_;

    WindowCounters window_;

    // Window-read tracking
    std::vector<uint8_t> posts_seen_window_;
    std::vector<uint8_t> posts_seen_prev_window_;
    std::vector<uint32_t> posts_list_window_;
    std::vector<uint32_t> posts_list_prev_window_;
    std::unordered_set<uint32_t> active_pre_window_;
    std::unordered_set<uint32_t> active_pre_prev_window_;
    GasEdgeCollector edge_collector_;

    // Orchestrator config
    OrchestratorConfig orch_{};

    // ===== Deterministic retire for edge-driven acc_update =====
    // 由于 StandardMem 回调到达顺序可能在 MPI 多 rank 下抖动，直接在回调里 acc_update
    // 会导致 float 累加顺序变化，从而在“极稀疏发放/阈值临界”场景出现 0/非0 跳变。
    // 这里将每条 edge 的 (post,pre,count) 按发起顺序编号，并在结果 ready 后按序退役。
    enum class EdgeSrc : uint8_t { Dense, Cache, Miss, BCSR, BCSRFile };
    struct EdgeRetireEntry {
        uint32_t post_local = 0;
        uint32_t pre_global = 0;
        uint32_t count = 0;
        float weight = 0.0f;
        bool ready = false;
        EdgeSrc src = EdgeSrc::Dense;
    };

    std::vector<EdgeRetireEntry> edge_retire_;
    size_t next_retire_seq_ = 0;

    bool issue_edges_in_progress_ = false;
    bool issue_edges_again_ = false;

    void resetEdgeRetire_() {
        edge_retire_.clear();
        next_retire_seq_ = 0;
        edge_retire_.reserve(edgesPrevSize());
    }

    size_t registerEdgeRetire_(uint32_t post_local, uint32_t pre_global, uint32_t count, EdgeSrc src) {
        const size_t seq = edge_retire_.size();
        EdgeRetireEntry e;
        e.post_local = post_local;
        e.pre_global = pre_global;
        e.count = count;
        e.src = src;
        edge_retire_.push_back(e);
        return seq;
    }

    static const char* edgeSrcTag_(EdgeSrc src) {
        switch (src) {
            case EdgeSrc::Cache:    return "cache";
            case EdgeSrc::Miss:     return "miss";
            case EdgeSrc::BCSR:     return "bcsr";
            case EdgeSrc::BCSRFile: return "bcsr-file";
            case EdgeSrc::Dense:    return "dense";
        }
        return "edge";
    }

    void setEdgeRetireReady_(size_t seq, float weight, EdgeSrc src) {
        if (seq >= edge_retire_.size()) return;
        edge_retire_[seq].weight = weight;
        edge_retire_[seq].ready = true;
        edge_retire_[seq].src = src;
    }

    void tryRetireEdges_() {
        while (next_retire_seq_ < edge_retire_.size()) {
            const auto& e = edge_retire_[next_retire_seq_];
            if (!e.ready) break;
            if (byteExactVerifyEnabled_() && e.src == EdgeSrc::Dense) {
                verifyDenseEdgeWeight_(e.pre_global, e.post_local, e.count, e.weight);
            }
            if (orch_.acc_update) orch_.acc_update(e.post_local, e.weight * static_cast<float>(e.count));
            if (orch_.diag_edge_weight) orch_.diag_edge_weight(edgeSrcTag_(e.src), e.post_local, e.pre_global, e.weight, e.count);
            next_retire_seq_++;
        }
    }

    // Diag context (per-window)
    SST::Output* diag_out_ = nullptr;
    bool diag_debug_ = false;
    int diag_node_id_ = 0;
    int diag_core_id_ = 0;
    uint32_t diag_seq_ = 0;

    struct WindowDiag {
        // issue side (bytes == meta.size)
        uint64_t issue_cnt_total = 0;
        uint64_t issue_bytes_total = 0;
        uint64_t issue_cnt_dense = 0;
        uint64_t issue_bytes_dense = 0;
        uint64_t issue_cnt_rowptr = 0;
        uint64_t issue_bytes_rowptr = 0;
        uint64_t issue_cnt_colidx = 0;
        uint64_t issue_bytes_colidx = 0;
        uint64_t issue_cnt_block = 0;
        uint64_t issue_bytes_block = 0;

        // response side (bytes == resp bytes from mem)
        uint64_t resp_cnt_total = 0;
        uint64_t resp_bytes_total = 0;
        uint64_t resp_cnt_dense = 0;
        uint64_t resp_bytes_dense = 0;
        uint64_t resp_cnt_rowptr = 0;
        uint64_t resp_bytes_rowptr = 0;
        uint64_t resp_cnt_colidx = 0;
        uint64_t resp_bytes_colidx = 0;
        uint64_t resp_cnt_block = 0;
        uint64_t resp_bytes_block = 0;

        // quality checks
        uint64_t resp_short_total = 0; // bytes < meta.size

        // semantic cache behavior (BCSR)
        uint64_t rowidx_hit = 0;
        uint64_t rowidx_miss = 0;
        uint64_t block_hit = 0;
        uint64_t block_miss = 0;

        // coarse duplicate detection (per window, bounded)
        uint64_t issue_unique_addrs = 0;
        uint64_t issue_dup_addrs = 0;
    };

    bool diag_window_active_ = false;
    uint32_t diag_window_seq_ = 0;
    WindowDiag diag_win_{};
    std::unordered_set<uint64_t> diag_req_addrs_{};
    uint32_t byte_exact_mismatch_count_ = 0;
    uint32_t byte_exact_mismatch_logged_ = 0;
    uint64_t byte_exact_verified_reads_ = 0;
    uint64_t byte_exact_verified_edges_ = 0;
    bool byte_exact_pass_logged_ = false;
    uint64_t bcsr_sem_verified_edges_ = 0;
    uint32_t bcsr_sem_mismatch_count_ = 0;
    uint32_t bcsr_sem_mismatch_logged_ = 0;
    bool bcsr_sem_pass_logged_ = false;
    bool bcsr_sem_inconclusive_ = false;
    std::string bcsr_sem_inconclusive_reason_;

    void resetWindowDiag_() {
        diag_win_ = WindowDiag{};
        diag_req_addrs_.clear();
    }

    bool byteExactVerifyEnabled_() const;
    float expectedDenseWeight_(uint32_t row, uint32_t col) const;
    void verifyDenseReadBytes_(uint64_t addr, size_t req_size, const std::vector<uint8_t>& bytes);
    void verifyDenseEdgeWeight_(uint32_t pre_global, uint32_t post_local, uint32_t count, float weight);
    void emitByteExactPassMarker_(const char* where, uint32_t seq);
    bool bcsrSemanticVerifyEnabled_() const;
    void verifyBcsrEdgeWeight_(uint32_t pre_global, uint32_t post_local, float weight);
    void emitBcsrSemanticVerifyMarker_(const char* where, uint32_t seq);

    void dumpWindowDiagSummary_(const char* tag) {
        if (!diag_out_) return;
        const uint32_t seq = diag_window_seq_;
        diag_out_->verbose(
            CALL_INFO, 0, 0,
            "%s node=%d core=%d window=%u "
            "issue(cnt=%" PRIu64 " bytes=%" PRIu64 " dense=%" PRIu64 " rowptr=%" PRIu64 " colidx=%" PRIu64 " block=%" PRIu64 ") "
            "resp(cnt=%" PRIu64 " bytes=%" PRIu64 " dense=%" PRIu64 " rowptr=%" PRIu64 " colidx=%" PRIu64 " block=%" PRIu64 " short=%" PRIu64 ") "
            "bcsr(rowidx_hit=%" PRIu64 " rowidx_miss=%" PRIu64 " block_hit=%" PRIu64 " block_miss=%" PRIu64 ") "
            "addr(unique=%" PRIu64 " dup=%" PRIu64 ")\n",
            tag ? tag : "[diag-window-weights]",
            diag_node_id_,
            diag_core_id_,
            seq,
            diag_win_.issue_cnt_total,
            diag_win_.issue_bytes_total,
            diag_win_.issue_bytes_dense,
            diag_win_.issue_bytes_rowptr,
            diag_win_.issue_bytes_colidx,
            diag_win_.issue_bytes_block,
            diag_win_.resp_cnt_total,
            diag_win_.resp_bytes_total,
            diag_win_.resp_bytes_dense,
            diag_win_.resp_bytes_rowptr,
            diag_win_.resp_bytes_colidx,
            diag_win_.resp_bytes_block,
            diag_win_.resp_short_total,
            diag_win_.rowidx_hit,
            diag_win_.rowidx_miss,
            diag_win_.block_hit,
            diag_win_.block_miss,
            diag_win_.issue_unique_addrs,
            diag_win_.issue_dup_addrs);
    }

    // ===== Phase1: memory is pure; semantic pending stays in this subsystem =====
    struct PendingMeta {
        // Apply/window sequence id (from beginApplyWindow); used to scope in-flight coalescing to a single window.
        uint32_t window_seq = 0;
        uint64_t address = 0;
        size_t size = 0;
        uint64_t orig_address = 0;
        size_t orig_size = 0;
        size_t slice_offset = 0;
        bool is_row = false;
        uint32_t pre = 0;
        uint32_t post_start = 0;
        uint32_t count_floats = 0;
        bool is_weight = true;
        bool count_weight_read = true;
        bool counted_inflight = false;
        uint64_t issue_cycle = 0;
        // 0=dense, 1=rowptr, 2=colidx, 3=blockdata
        int bcsr_kind = 0;
        uint32_t bcsr_block_row = 0;
        uint32_t bcsr_target_block_col = 0;
        uint32_t bcsr_intra_row = 0;
        uint32_t bcsr_intra_col = 0;
        uint32_t bcsr_row_start = 0;
        uint32_t bcsr_idx_in_row = 0;
        uint32_t bcsr_global_block_index = 0;
        bool bcsr_prefetch_all = false;
        bool bcsr_colidx_bulk_all_rows = false;
        bool scheme1_prefetch = false;
        bool has_single_cb = false;
        uint32_t cb_post = 0;
        std::function<void(float)> single_cb;
    };

    IMemoryAccess* mem_access_ = nullptr;
    uint64_t now_cycle_ = 0;
    bool bcsr_prefetch_issued_ = false;
    bool row_index_prefetch_all_done_ = false;
    bool row_index_prefetch_bulk_pending_ = false;
    bool row_index_prefetch_bulk_inflight_ = false;
    std::deque<uint32_t> row_index_prefetch_rows_{};
    uint32_t window_seq_ = 0;
    bool bcsr_rowptr_file_preload_attempted_ = false;
    bool bcsr_rowidx_file_preloaded_ = false;

    // Lightweight always-on window counters (for auto-tune; independent of debug-only diag_win_).
    uint32_t block_hit_window_ = 0;
    uint32_t block_miss_window_ = 0;

    // ===== Phase A/B: in-flight coalescing (per-window) =====
    // 目的：消除同一窗口内对同一 colidx/blockdata 的重复读发起（大量 edge 在 colidx/block 未就绪时会抖动）。
    // 约束：只在同一 window_seq 内合并；不跨窗共享（保持严格 GAS 边界语义）。
    struct ColidxWaiter {
        uint32_t pre_global = 0;
        uint32_t post_local = 0;
        uint32_t target_block_col = 0;
        uint32_t intra_row = 0;
        uint32_t intra_col = 0;
        std::function<void(float)> cb;
    };

    struct ColidxInflight {
        uint32_t window_seq = 0;
        uint32_t block_row = 0;
        uint32_t row_start = 0;
        uint32_t row_end = 0;
        bool issued = false;
        bool queued = false;
        bool count_budget = true;
        std::vector<ColidxWaiter> waiters;
    };

    struct BlockWaiter {
        uint32_t pre_global = 0;
        uint32_t post_local = 0;
        uint32_t intra_row = 0;
        uint32_t intra_col = 0;
        std::function<void(float)> cb;
    };

    struct BlockInflight {
        uint32_t window_seq = 0;
        uint32_t block_row = 0;
        uint32_t block_col = 0;
        uint32_t global_block_index = 0;
        bool issued = false;
        bool queued = false;
        bool count_budget = true;
        std::vector<BlockWaiter> waiters;
    };

    std::unordered_map<uint64_t, ColidxInflight> inflight_colidx_{};
    std::unordered_map<uint64_t, BlockInflight> inflight_block_{};
    std::deque<uint64_t> pending_colidx_reads_{};
    std::deque<uint64_t> pending_block_reads_{};
    struct PendingBcsrRowptrWaiter {
        uint32_t pre_global = 0;
        uint32_t post_local = 0;
        std::function<void(float)> cb;
    };
    std::deque<PendingBcsrRowptrWaiter> pending_bcsr_rowptr_waiters_{};
    // Direct (non-coalesced) read queue: used by naive baseline when coalescing is disabled, while still respecting inflight limits.
    std::deque<PendingMeta> pending_direct_reads_{};
    bool drain_pending_in_progress_ = false;

    enum class IssueStatus : uint8_t { Issued, DeferredInflight, DeferredBudget, Failed };
    IssueStatus tryIssueRead_(PendingMeta meta, bool count_budget, bool budget_reserved);
    void drainPendingReads_();
    void drainPendingDirectReads_();

    static uint64_t makeInflightKey_(uint32_t window_seq, uint32_t id) {
        return (static_cast<uint64_t>(window_seq) << 32) | static_cast<uint64_t>(id);
    }

    void handleReadResp_(uint64_t req_id, uint64_t addr, PendingMeta meta, std::vector<uint8_t>&& data);
    uint64_t issueRead_(PendingMeta meta);

    bool prepareDenseRead_(uint32_t row, uint32_t col, uint32_t width,
                           uint64_t& req_addr, size_t& req_size,
                           bool& is_row, uint32_t& col_start, uint32_t& count_floats) const;
    void issueDenseResolved_(uint32_t row, uint32_t col, uint32_t cb_col,
                             std::function<void(float)> cb);
    void requestDense_(uint32_t pre, uint32_t post, std::function<void(float)> cb);
    void enqueueBcsrBlockReadCoalesced_(uint32_t win_seq,
                                        uint32_t block_row,
                                        uint32_t block_col,
                                        uint32_t global_block_index,
                                        uint32_t intra_row,
                                        uint32_t intra_col,
                                        uint32_t pre_global,
                                        uint32_t post_local,
                                        std::function<void(float)> cb);
    void requestBCSR_(uint32_t pre_global, uint32_t post_local, std::function<void(float)> cb);
    void maybeIssueBcsrRowptrPrefetch_();
    void drainPendingBcsrRowptrWaiters_();
    void maybeEnqueueRowIndexPrefetchAllRows_();
    void maybeEnqueueRowIndexPrefetchPostsPrev_();
    void drainRowIndexPrefetch_();
    void maybeAutoTuneBlockCache_();
    void bcsrPrefetchAll_();
    void bcsrPrefetchRowBlocks_(uint32_t block_row, const std::vector<uint32_t>& cols, uint32_t row_start);
    void bcsrPopulateWeightCache_(uint32_t block_row, uint32_t block_col, const std::vector<float>& blk);
};

}} // namespace SST::SnnDL
