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
#include <functional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "GasEdgeCollector.h"
#include "SnnWeightReader.h"
#include "WeightAccessor.h"

namespace SST { class Output; }

namespace SST { namespace SnnDL {

class WeightMemorySubsystem : public IWeightReader {
public:
    using DenseReqFn = std::function<void(uint32_t,uint32_t,std::function<void(float)>)>;
    using BcsrReqFn  = std::function<void(uint32_t,uint32_t,std::function<void(float)>)>;
    using CacheTryFn = std::function<bool(uint64_t,float&)>;
    using CachePutFn = std::function<void(uint64_t,float)>;

    using RequestFn = std::function<bool(uint32_t,uint32_t,std::function<void(float)>)>;
    using AccUpdateFn = std::function<void(uint32_t,float)>;
    using DiagEdgeFn = std::function<void(const char*,uint32_t,uint32_t,float,uint32_t)>;
    using CacheReportFn = std::function<void(bool)>;
    using BoolFn = std::function<bool()>;
    using PendingPeakFn = std::function<void(uint32_t)>;
    using ReadBcsrFileFn = std::function<float(uint32_t,uint32_t)>;

    struct WindowCounters {
        uint32_t budget = 0;
        uint32_t issued = 0;
        uint32_t outstanding = 0;
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
        BoolFn ensure_loader_ready;
        BoolFn bcsr_rowptr_ready;
        ReadBcsrFileFn read_bcsr_from_file;
        bool use_bcsr = false;
        bool bcsr_force_file_read = false;
        bool readresp_zero_fallback = false;
        float init_default_weight = 0.5f;
        uint32_t num_neurons = 0;
    };

    WeightMemorySubsystem() = default;

    void configure(DenseReqFn dense_fn,
                   BcsrReqFn bcsr_fn,
                   CacheTryFn cache_try_fn,
                   CachePutFn cache_put_fn) {
        dense_fn_ = std::move(dense_fn);
        bcsr_fn_ = std::move(bcsr_fn);
        cache_try_fn_ = std::move(cache_try_fn);
        cache_put_fn_ = std::move(cache_put_fn);
    }

    void configureOrchestrator(OrchestratorConfig cfg) {
        orch_ = std::move(cfg);
        ensureWindowTracking(orch_.num_neurons);
    }

    // ===== Window counters (budget/outstanding) =====
    void configureWindow(uint32_t budget, uint32_t max_outstanding) {
        window_.budget = budget;
        window_.max_outstanding = max_outstanding;
    }
    void beginWindow() {
        window_.issued = 0;
        window_.outstanding = 0;
    }
    bool canIssue(uint32_t n = 1) const {
        const bool budget_ok = (window_.budget == 0) || (window_.issued + n <= window_.budget);
        const bool ostd_ok = (window_.outstanding + n <= window_.max_outstanding);
        return budget_ok && ostd_ok;
    }
    void noteIssue(uint32_t n = 1) {
        window_.issued += n;
        window_.outstanding += n;
    }
    void noteComplete(uint32_t n = 1) {
        if (window_.outstanding >= n) window_.outstanding -= n;
        else window_.outstanding = 0;
    }
    uint32_t issued() const { return window_.issued; }
    uint32_t outstanding() const { return window_.outstanding; }
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
        diag_out_ = out;
        diag_debug_ = debug;
        diag_core_id_ = core_id;
        diag_seq_ = seq;
        if (orch_.use_bcsr && orch_.bcsr_rowptr_ready && !orch_.bcsr_rowptr_ready()) {
            // defer until rowptr ready (preserve old behavior)
        }
        flipEdgesForApply(debug, out, core_id, seq);
        beginWindow();
    }

    void issueFromEdges() {
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

            if (orch_.use_bcsr) {
                if (orch_.bcsr_force_file_read && orch_.read_bcsr_from_file) {
                    float resolved = orch_.read_bcsr_from_file(post_local, pre_global);
                    if (orch_.readresp_zero_fallback && resolved == 0.0f) resolved = orch_.init_default_weight;
                    if (orch_.acc_update) orch_.acc_update(post_local, resolved * static_cast<float>(count));
                    if (orch_.diag_edge_weight) orch_.diag_edge_weight("bcsr-file", post_local, pre_global, resolved, count);
                    continue;
                }
                noteIssue();
                if (orch_.update_pending_peak) orch_.update_pending_peak(outstanding());
                auto cb = [this, post_local, pre_global, count](float w) {
                    float resolved = w;
                    if (orch_.readresp_zero_fallback && resolved == 0.0f) resolved = orch_.init_default_weight;
                    if (orch_.acc_update) orch_.acc_update(post_local, resolved * static_cast<float>(count));
                    if (orch_.diag_edge_weight) orch_.diag_edge_weight("bcsr", post_local, pre_global, resolved, count);
                    noteComplete();
                    issueFromEdges();
                };
                if (orch_.request_bcsr) (void)orch_.request_bcsr(pre_global, post_local, std::move(cb));
                else if (bcsr_fn_) bcsr_fn_(pre_global, post_local, std::move(cb));
                continue;
            }

            uint32_t req_pre = 0;
            uint32_t req_post = 0;
            uint64_t cache_key = 0;
            if (!orch_.accessor->resolve(pre_global, post_local, req_pre, req_post, cache_key)) {
                continue;
            }

            float cached = 0.0f;
            const bool cache_hit = orch_.cache_try ? orch_.cache_try(cache_key, cached)
                                                   : (cache_try_fn_ ? cache_try_fn_(cache_key, cached) : false);
            if (cache_hit) {
                if (orch_.readresp_zero_fallback && cached == 0.0f) cached = orch_.init_default_weight;
                if (orch_.acc_update) orch_.acc_update(post_local, cached * static_cast<float>(count));
                if (orch_.diag_edge_weight) orch_.diag_edge_weight("cache", post_local, pre_global, cached, count);
                if (orch_.report_cache_access) orch_.report_cache_access(true);
                continue;
            }
            if (orch_.report_cache_access) orch_.report_cache_access(false);

            noteIssue();
            if (orch_.update_pending_peak) orch_.update_pending_peak(outstanding());
            auto miss_cb = [this, post_local, pre_global, count, cache_key](float w) {
                float resolved = w;
                if (orch_.readresp_zero_fallback && resolved == 0.0f) resolved = orch_.init_default_weight;
                if (orch_.cache_put) orch_.cache_put(cache_key, resolved);
                else if (cache_put_fn_) cache_put_fn_(cache_key, resolved);
                if (orch_.acc_update) orch_.acc_update(post_local, resolved * static_cast<float>(count));
                if (orch_.diag_edge_weight) orch_.diag_edge_weight("miss", post_local, pre_global, resolved, count);
                noteComplete();
                issueFromEdges();
            };

            bool issued = false;
            if (orch_.request_dense) issued = orch_.request_dense(req_pre, req_post, miss_cb);
            else if (dense_fn_) { dense_fn_(req_pre, req_post, miss_cb); issued = true; }
            if (!issued && dense_fn_) dense_fn_(req_pre, req_post, miss_cb);
        }
    }

    void issueFromSets(const std::vector<uint32_t>* posts_to_use,
                       const std::unordered_set<uint32_t>* pres_to_use) {
        if (!orch_.accessor || !posts_to_use || !pres_to_use) return;
        if (orch_.use_bcsr) {
            issueFromSetsBcsr(posts_to_use, pres_to_use);
            return;
        }
        uint32_t issued = 0;
        for (const auto& pre_g : *pres_to_use) {
            for (uint32_t post_l : *posts_to_use) {
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
                bool ok = false;
                if (orch_.request_dense) ok = orch_.request_dense(req_pre, req_post, cache_cb);
                else if (dense_fn_) { dense_fn_(req_pre, req_post, cache_cb); ok = true; }
                if (!ok && dense_fn_) dense_fn_(req_pre, req_post, cache_cb);
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
        uint32_t primed = 0;
        for (const auto& pre_g : *pres_to_use) {
            for (uint32_t post_l : *posts_to_use) {
                if (!canIssueMoreReads_()) break;
                if (orch_.report_cache_access) orch_.report_cache_access(false);
                noteIssue();
                if (orch_.update_pending_peak) orch_.update_pending_peak(outstanding());
                primed++;
                auto cb = [this](float) { noteComplete(); };
                if (orch_.request_bcsr) (void)orch_.request_bcsr(pre_g, post_l, std::move(cb));
                else if (bcsr_fn_) bcsr_fn_(pre_g, post_l, std::move(cb));
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
        if (dense_fn_) dense_fn_(pre, post, std::move(cb));
    }

    void requestBCSR(uint32_t pre_global, uint32_t post_local, std::function<void(float)> cb) override {
        if (bcsr_fn_) bcsr_fn_(pre_global, post_local, std::move(cb));
    }

    bool tryCache(uint64_t key, float& out) override {
        return cache_try_fn_ ? cache_try_fn_(key, out) : false;
    }

    void putCache(uint64_t key, float value) override {
        if (cache_put_fn_) cache_put_fn_(key, value);
    }

private:
    bool canIssueMoreReads_() const {
        if (canIssue()) return true;
        if (diag_debug_ && diag_out_) {
            diag_out_->verbose(CALL_INFO, 0, 0,
                "[diag-edge-loop] core=%d budget/ostd stop issued=%u outstanding=%u budget=%u limit=%u\n",
                diag_core_id_, issued(), outstanding(), budget(), maxOutstanding());
        }
        return false;
    }

    DenseReqFn dense_fn_;
    BcsrReqFn bcsr_fn_;
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

    // Diag context (per-window)
    SST::Output* diag_out_ = nullptr;
    bool diag_debug_ = false;
    int diag_core_id_ = 0;
    uint32_t diag_seq_ = 0;
};

}} // namespace SST::SnnDL
