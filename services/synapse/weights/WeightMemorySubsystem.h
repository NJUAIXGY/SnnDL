// -*- c++ -*-
//
// WeightMemorySubsystem:
// - 向 compute core 提供统一的 IWeightReader（dense/BCSR + cache）
// - 承载 window-read 的集合/预算/并发/outstanding 与发起编排（Phase A）
//   使控制层仅在窗口边界与 recordEdge/recordTouch 上触发即可。

#pragma once

#include <algorithm>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cinttypes>
#include <cmath>
#include <cstdlib>
#include <functional>
#include <deque>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "synapse/gas/GasEdgeCollector.h"
#include "synapse/weights/GcssIndexTable.h"
#include "synapse/weights/GcssIndexRowMphf.h"
#include "synapse/weights/GcssIndexPreMphf.h"
#include "SnnWeightReader.h"
#include "IMemoryAccess.h"
#include "WeightAccessor.h"
#include "DenseWeightLayout.h"
#include "services/memory/sram_sim/layout/VirtualSramLayout.h"
#include "services/memory/sram_sim/model/BankedSramModel.h"
#include "IPeAggregation.h"

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
    using SubmitTassLfP0WindowReportFn = std::function<void(const TassLfP0WindowReport&)>;
    using SubmitTassNaiveWindowRequestFn = std::function<void(const TassNaiveWindowRequest&)>;

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
        // BCSR blockdata fetch granularity:
        // - full_block: read full br*bc block bytes (legacy behavior)
        // - row_cacheline: read only one block-row slice (intra_row), typically bc*val_bytes
        std::string bcsr_block_fetch_mode = "full_block"; // full_block|row_cacheline
        // RowIndex(colidx) 冷启动/热路径优化：默认启用 "auto"（当 colidx 总量很小才用 all_rows；否则用 posts_prev）。
        std::string bcsr_row_index_prefetch_mode = "auto"; // off/all_rows/posts_prev/auto
        uint32_t bcsr_row_index_prefetch_all_rows_threshold = 1024;
        uint64_t bcsr_row_index_prefetch_all_rows_max_bytes = 64ull * 1024ull;
        // Experimental NoC×memory coupling (STORM-PIF): prefetch BCSR row-index (colidx)
        // rows from Gather-stage touch signals before Apply critical path.
        bool experimental_noc_rowidx_prefetch_enable = false;
        uint32_t experimental_noc_rowidx_prefetch_budget_per_tick = 4;
        uint32_t experimental_noc_rowidx_cache_rows = 1024;
        bool experimental_noc_rowidx_prefetch_gather_only = true;
        // STORM-PIF v8: only enqueue block_row prefetch after it is touched N times in-window.
        uint32_t experimental_noc_rowidx_hot_touch_min = 1;
        // STORM-PIF v8: adapt prefetch budget by queue depth / inflight headroom.
        bool experimental_noc_rowidx_budget_adapt_enable = false;
        uint32_t experimental_noc_rowidx_budget_adapt_max_per_tick = 32;
        uint32_t experimental_noc_rowidx_budget_adapt_q_depth = 16;
        // Experimental NoC×memory coupling (STORM-NIP p0):
        // prefetch GCSSIDX2 values as soon as ingress touches are observed in Gather.
        bool experimental_idx2_ingress_prefetch_enable = false;
        uint32_t experimental_idx2_ingress_prefetch_budget_per_tick = 4;
        uint32_t experimental_idx2_ingress_prefetch_cache_entries = 4096;
        bool experimental_idx2_ingress_prefetch_gather_only = true;
        bool experimental_idx2_ingress_tail_guard_enable = false;
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
        // Dense weights layout (default: row_major).
        DenseLayoutMode dense_layout_mode = DenseLayoutMode::RowMajor;
        // Required when dense_layout_mode=PhysV1. Must match the offline weights_phys generator.
        uint32_t dense_phys_dram_row_bytes = 0;
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
        std::string gcss_index_template;
        std::string synapse_weight_mode = "bcsr_gas"; // bcsr_gas | gcss_valueonly_dstcore | gcss_valueonly_dstcore_idx2 | gcss_valueonly_dstcore_vlf_premphf
        // Observe-only SRAM model for weight-side index/L0 studies.
        bool weight_sram_model_enable = false;
        bool weight_idx_sram_enable = false;
        bool weight_l0_sram_enable = false;
        uint64_t weight_idx_sram_capacity_bytes = 0;
        uint64_t weight_l0_sram_capacity_bytes = 0;
        uint32_t weight_idx_sram_banks = 16;
        uint32_t weight_l0_sram_banks = 8;
        uint32_t weight_sram_ports_per_bank = 1;
        uint64_t weight_sram_bank_interleave_bytes = 4;
        uint32_t weight_sram_t_read_cycles = 1;
        uint32_t weight_sram_t_write_cycles = 1;
        uint32_t weight_sram_sample_log2 = 0;
        uint64_t weight_idx_sram_base = 0x100000000ull;
        uint64_t weight_l0_sram_base = 0x200000000ull;
        uint64_t weight_l0_sram_slots = (1ull << 20);
        // Experimental retire policy (default keeps historical global in-order semantics).
        // Allowed values: global_inorder | per_post
        std::string experimental_retire_policy = "global_inorder";
        bool experimental_pre_window_profile_export_enable = false;
        std::string experimental_pre_window_profile_export_dir;
        bool experimental_tass_lf_p0_enable = false;
        uint32_t tass_lf_p0_mesh_rows = 1;
        uint32_t tass_lf_p0_mesh_cols = 1;
        uint32_t tass_lf_p0_block_h = 2;
        uint32_t tass_lf_p0_block_w = 2;
        uint32_t tass_lf_p0_cores_per_pe = 1;
        SubmitTassLfP0WindowReportFn submit_tass_lf_p0_window_report;
        SubmitTassNaiveWindowRequestFn submit_tass_naive_window_request;
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
               !experimental_noc_rowidx_pending_rows_.empty() ||
               !experimental_idx2_ingress_pending_addrs_.empty() ||
               !experimental_idx2_ingress_inflight_.empty() ||
               !gcss_vlf_issue_queue_.empty() ||
               !pending_colidx_reads_.empty() ||
               !pending_block_reads_.empty() ||
               !pending_direct_reads_.empty() ||
               (retired_edges_count_ < edge_retire_.size()) ||
               pending_tass_naive_responses_ != 0;
    }
    void setNowCycle(uint64_t now_cycle) { now_cycle_ = now_cycle; }
    void onClockTick(uint64_t now_cycle);
    struct ExperimentalNocRowidxStats {
        uint64_t touch_events_total = 0;
        uint64_t rows_touched_enqueued = 0;
        uint64_t rows_filtered_cold = 0;
        uint64_t prefetch_rows_issued = 0;
        uint64_t prefetch_rows_deferred = 0;
        uint64_t prefetch_rows_failed = 0;
        uint64_t prefetch_bytes_issued = 0;
        uint64_t budget_ticks_total = 0;
        uint64_t budget_effective_total = 0;
        uint64_t budget_adapt_ticks = 0;
        uint64_t cache_hits = 0;
        uint64_t cache_misses = 0;
        uint64_t cache_fills = 0;
        uint64_t cache_full_drop = 0;
        uint64_t cache_entries = 0;
    };
    ExperimentalNocRowidxStats experimentalNocRowidxStats() const {
        ExperimentalNocRowidxStats s = experimental_noc_rowidx_stats_;
        s.cache_entries = static_cast<uint64_t>(experimental_noc_rowidx_cache_.size());
        return s;
    }
    struct ExperimentalIdx2IngressStats {
        uint64_t touch_events_total = 0;
        uint64_t lookup_miss_total = 0;
        uint64_t enqueued_total = 0;
        uint64_t dedup_pending_total = 0;
        uint64_t dedup_inflight_total = 0;
        uint64_t dedup_cache_total = 0;
        uint64_t prefetch_issued_total = 0;
        uint64_t prefetch_bytes_total = 0;
        uint64_t prefetch_deferred_total = 0;
        uint64_t prefetch_failed_total = 0;
        uint64_t prefetch_resp_ok_total = 0;
        uint64_t prefetch_resp_short_total = 0;
        uint64_t prefetch_resp_drop_tail_total = 0;
        uint64_t prefetch_complete_inflight_miss_total = 0;
        uint64_t prefetch_complete_zero_waiters_total = 0;
        uint64_t prefetch_complete_waiters_total = 0;
        uint64_t demand_hit_total = 0;
        uint64_t demand_join_total = 0;
        uint64_t demand_join_cb_nonnull_total = 0;
        uint64_t demand_join_cb_null_total = 0;
        uint64_t demand_fallback_total = 0;
        uint64_t waiters_served_total = 0;
        uint64_t cache_fill_total = 0;
        uint64_t cache_evict_total = 0;
        uint64_t cache_entries = 0;
    };
    ExperimentalIdx2IngressStats experimentalIdx2IngressStats() const {
        ExperimentalIdx2IngressStats s = experimental_idx2_ingress_stats_;
        s.cache_entries = static_cast<uint64_t>(experimental_idx2_ingress_value_cache_.size());
        return s;
    }
    struct GcssLookupStats {
        uint64_t hit_total = 0;
        uint64_t miss_total = 0;
    };
    GcssLookupStats gcssLookupStats() const {
        GcssLookupStats s{};
        s.hit_total = gcss_lookup_hit_;
        s.miss_total = gcss_lookup_miss_;
        return s;
    }
    struct ReadSourceStats {
        uint64_t dense_reqs_total = 0;
        uint64_t dense_bytes_total = 0;
        uint64_t rowptr_reqs_total = 0;
        uint64_t rowptr_bytes_total = 0;
        uint64_t colidx_reqs_total = 0;
        uint64_t colidx_bytes_total = 0;
        uint64_t blockdata_reqs_total = 0;
        uint64_t blockdata_bytes_total = 0;
        uint64_t gcss_reqs_total = 0;
        uint64_t gcss_bytes_total = 0;
    };
    ReadSourceStats readSourceStats() const {
        ReadSourceStats s{};
        s.dense_reqs_total = read_src_dense_reqs_;
        s.dense_bytes_total = read_src_dense_bytes_;
        s.rowptr_reqs_total = read_src_rowptr_reqs_;
        s.rowptr_bytes_total = read_src_rowptr_bytes_;
        s.colidx_reqs_total = read_src_colidx_reqs_;
        s.colidx_bytes_total = read_src_colidx_bytes_;
        s.blockdata_reqs_total = read_src_blockdata_reqs_;
        s.blockdata_bytes_total = read_src_blockdata_bytes_;
        s.gcss_reqs_total = read_src_gcss_reqs_;
        s.gcss_bytes_total = read_src_gcss_bytes_;
        return s;
    }
    struct SramObservabilityStats {
        BankedSramStats idx_sram{};
        BankedSramStats l0_sram{};
        uint64_t idx_lookup_total = 0;
        uint64_t idx_lookup_idx2_total = 0;
        uint64_t idx_lookup_legacy_total = 0;
        uint64_t l0_lookup_total = 0;
        uint64_t l0_hit_total = 0;
        uint64_t l0_miss_total = 0;
        uint64_t l0_fill_total = 0;
        uint64_t l0_evict_total = 0;
    };
    SramObservabilityStats sramObservabilityStats() const {
        SramObservabilityStats s{};
        s.idx_sram = idx_sram_model_.stats();
        s.l0_sram = l0_sram_model_.stats();
        s.idx_lookup_total = idx_lookup_total_;
        s.idx_lookup_idx2_total = idx_lookup_idx2_total_;
        s.idx_lookup_legacy_total = idx_lookup_legacy_total_;
        s.l0_lookup_total = l0_lookup_total_;
        s.l0_hit_total = l0_hit_total_;
        s.l0_miss_total = l0_miss_total_;
        s.l0_fill_total = l0_fill_total_;
        s.l0_evict_total = l0_evict_total_;
        return s;
    }
    struct RetireObservabilityStats {
        uint64_t global_hol_cycles_total = 0;
        uint64_t ready_but_blocked_edges_total = 0;
        uint64_t per_post_progress_total = 0;
    };
    RetireObservabilityStats retireObservabilityStats() const {
        RetireObservabilityStats s{};
        s.global_hol_cycles_total = retire_global_hol_cycles_total_;
        s.ready_but_blocked_edges_total = retire_ready_but_blocked_edges_total_;
        s.per_post_progress_total = retire_per_post_progress_total_;
        return s;
    }
    struct TassLfP0Stats {
        uint64_t block_epochs_total = 0;
        uint64_t block_active_pres_total = 0;
        uint64_t block_shared_pres_total = 0;
        uint64_t cross_core_joins_total = 0;
        uint64_t payload_bytes_total = 0;
        uint64_t current_vlf_line_groups_total = 0;
        uint64_t block_naive_line_count_total = 0;
        uint64_t block_fused_lb_line_count_total = 0;
        uint64_t response_fanout_total = 0;
        uint64_t reports_flushed_total = 0;
        uint64_t reports_nonzero_payload_total = 0;
        uint64_t reports_pre_entries_total = 0;
        uint64_t reports_via_callback_total = 0;
        uint64_t reports_via_fallback_total = 0;
    };
    TassLfP0Stats tassLfP0Stats() {
        flushTassLfP0WindowIfNeeded_(true);
        harvestTassLfP0Completions_();
        return tass_lf_p0_stats_;
    }
    void flushSramObservability(uint64_t now_cycle) {
        if (!weight_sram_enable_) return;
        idx_sram_model_.onClockTick(now_cycle + 1);
        l0_sram_model_.onClockTick(now_cycle + 1);
    }
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
        gcss_index_loaded_ = false;
        gcss_index_load_failed_ = false;
        gcss_index_path_.clear();
        gcss_index_.clear();
        gcss_idx2_index_loaded_ = false;
        gcss_idx2_index_load_failed_ = false;
        gcss_idx2_index_path_.clear();
        gcss_idx2_index_.clear();
        gcss_premphf_index_loaded_ = false;
        gcss_premphf_index_load_failed_ = false;
        gcss_premphf_index_path_.clear();
        gcss_premphf_index_.clear();
        experimental_idx2_ingress_pending_addrs_.clear();
        experimental_idx2_ingress_pending_set_.clear();
        experimental_idx2_ingress_inflight_.clear();
        experimental_idx2_ingress_value_cache_.clear();
        experimental_idx2_ingress_value_cache_lru_.clear();
        edge_pre_rank_curr_.clear();
        edge_pre_rank_prev_.clear();
        gcss_vlf_issue_queue_.clear();
        gcss_vlf_issue_queue_prepared_ = false;
        pre_touch_order_window_.clear();
        gcss_vlf_issue_prepare_total_ = 0;
        gcss_vlf_issue_edges_total_ = 0;
        gcss_vlf_issue_reorder_trigger_total_ = 0;
        gcss_vlf_issue_line_groups_total_ = 0;
        experimental_idx2_ingress_stats_ = ExperimentalIdx2IngressStats{};
        gcss_lookup_hit_ = 0;
        gcss_lookup_miss_ = 0;
        read_src_dense_reqs_ = 0;
        read_src_dense_bytes_ = 0;
        read_src_rowptr_reqs_ = 0;
        read_src_rowptr_bytes_ = 0;
        read_src_colidx_reqs_ = 0;
        read_src_colidx_bytes_ = 0;
        read_src_blockdata_reqs_ = 0;
        read_src_blockdata_bytes_ = 0;
        read_src_gcss_reqs_ = 0;
        read_src_gcss_bytes_ = 0;
        idx_lookup_total_ = 0;
        idx_lookup_idx2_total_ = 0;
        idx_lookup_legacy_total_ = 0;
        l0_lookup_total_ = 0;
        l0_hit_total_ = 0;
        l0_miss_total_ = 0;
        l0_fill_total_ = 0;
        l0_evict_total_ = 0;
        retire_global_hol_cycles_total_ = 0;
        retire_ready_but_blocked_edges_total_ = 0;
        retire_per_post_progress_total_ = 0;
        tass_lf_p0_stats_ = TassLfP0Stats{};
        tass_lf_p0_window_started_ = false;
        tass_lf_p0_window_seq_ = 0;
        tass_lf_p0_window_pre_payload_bytes_.clear();
        tass_lf_p0_window_payload_bytes_ = 0;
        tass_lf_p0_window_current_vlf_line_groups_ = 0;
        ready_uncommitted_edges_ = 0;
        {
            std::string policy = orch_.experimental_retire_policy;
            std::transform(policy.begin(), policy.end(), policy.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (policy == "per_post" || policy == "per_post_deterministic") {
                retire_policy_ = RetirePolicy::PerPostDeterministic;
            } else {
                retire_policy_ = RetirePolicy::GlobalInOrder;
            }
        }
        per_post_retire_seq_.clear();
        per_post_head_idx_.clear();
        per_post_touched_.clear();
        per_post_touched_list_.clear();
        per_post_ready_posts_.clear();
        tass_lf_p0_enabled_ = orch_.experimental_tass_lf_p0_enable;
        pending_tass_naive_responses_ = 0;
        tass_naive_window_request_submitted_ = false;
        weight_sram_enable_ = orch_.weight_sram_model_enable;
        weight_idx_sram_enable_ = orch_.weight_idx_sram_enable;
        weight_l0_sram_enable_ = orch_.weight_l0_sram_enable;
        VirtualSramLayoutConfig lcfg{};
        lcfg.idx_base = orch_.weight_idx_sram_base;
        lcfg.l0_base = orch_.weight_l0_sram_base;
        lcfg.l0_slots = orch_.weight_l0_sram_slots;
        sram_layout_.configure(lcfg);
        BankedSramConfig idx_cfg{};
        idx_cfg.enable = weight_sram_enable_ && weight_idx_sram_enable_;
        idx_cfg.name = "weight_idx_sram";
        idx_cfg.capacity_bytes = orch_.weight_idx_sram_capacity_bytes;
        idx_cfg.banks = orch_.weight_idx_sram_banks;
        idx_cfg.ports_per_bank = orch_.weight_sram_ports_per_bank;
        idx_cfg.bank_interleave_bytes = orch_.weight_sram_bank_interleave_bytes;
        idx_cfg.t_read_cycles = orch_.weight_sram_t_read_cycles;
        idx_cfg.t_write_cycles = orch_.weight_sram_t_write_cycles;
        idx_cfg.sample_log2 = orch_.weight_sram_sample_log2;
        idx_sram_model_.configure(idx_cfg);
        BankedSramConfig l0_cfg{};
        l0_cfg.enable = weight_sram_enable_ && weight_l0_sram_enable_;
        l0_cfg.name = "weight_l0_sram";
        l0_cfg.capacity_bytes = orch_.weight_l0_sram_capacity_bytes;
        l0_cfg.banks = orch_.weight_l0_sram_banks;
        l0_cfg.ports_per_bank = orch_.weight_sram_ports_per_bank;
        l0_cfg.bank_interleave_bytes = orch_.weight_sram_bank_interleave_bytes;
        l0_cfg.t_read_cycles = orch_.weight_sram_t_read_cycles;
        l0_cfg.t_write_cycles = orch_.weight_sram_t_write_cycles;
        l0_cfg.sample_log2 = orch_.weight_sram_sample_log2;
        l0_sram_model_.configure(l0_cfg);
        {
            std::string mode = orch_.bcsr_block_fetch_mode;
            std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
                return static_cast<char>(std::tolower(c));
            });
            if (mode == "row_cacheline") bcsr_block_fetch_mode_ = BcsrBlockFetchMode::RowCacheline;
            else bcsr_block_fetch_mode_ = BcsrBlockFetchMode::FullBlock;
        }
        dense_cols_effective_ = orch_.use_post_row_pre_col ? orch_.weights_cols : orch_.num_neurons;
        dense_phys_enable_ = (orch_.dense_layout_mode == DenseLayoutMode::PhysV1);
        if (dense_phys_enable_) {
            if (orch_.dense_phys_dram_row_bytes == 0) {
                diagOutOrFallback_()->fatal(CALL_INFO, -1,
                                            "WeightMemorySubsystem fatal: dense_layout_mode=phys_v1 requires dense_phys_dram_row_bytes>0\n");
                std::abort();
            }
            if (orch_.line_size_bytes == 0) orch_.line_size_bytes = 64;
            const bool ok = computeDensePhysV1Derived(
                orch_.num_neurons,
                dense_cols_effective_,
                orch_.line_size_bytes,
                orch_.dense_phys_dram_row_bytes,
                dense_phys_);
            if (!ok) {
                diagOutOrFallback_()->fatal(CALL_INFO, -1,
                                            "WeightMemorySubsystem fatal: invalid dense phys_v1 layout params rows=%u cols=%u line=%u row_bytes=%u\n",
                                            orch_.num_neurons,
                                            dense_cols_effective_,
                                            orch_.line_size_bytes,
                                            orch_.dense_phys_dram_row_bytes);
                std::abort();
            }
            if (orch_.dense_phys_dram_row_bytes != 0 &&
                (orch_.base_addr % static_cast<uint64_t>(orch_.dense_phys_dram_row_bytes)) != 0) {
                diagOutOrFallback_()->fatal(CALL_INFO, -1,
                                            "WeightMemorySubsystem fatal: phys_v1 requires base_addr aligned to dram_row_bytes "
                                            "(base=0x%llx row_bytes=%u)\n",
                                            (unsigned long long)orch_.base_addr,
                                            orch_.dense_phys_dram_row_bytes);
                std::abort();
            }
        }
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
        pre_touch_order_window_.reserve(256);
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
        pre_touch_order_window_.clear();
        resetExperimentalNocRowidxWindow_();
    }

    void notePostLocal(uint32_t post_local, uint32_t num_neurons) {
        if (post_local >= num_neurons) return;
        ensureWindowTracking(num_neurons);
        if (!posts_seen_window_[post_local]) {
            posts_seen_window_[post_local] = 1;
            posts_list_window_.push_back(post_local);
        }
    }

    void setSubmitTassLfP0WindowReportFn(SubmitTassLfP0WindowReportFn fn) {
        orch_.submit_tass_lf_p0_window_report = std::move(fn);
    }

    void setSubmitTassNaiveWindowRequestFn(SubmitTassNaiveWindowRequestFn fn) {
        orch_.submit_tass_naive_window_request = std::move(fn);
    }

    void completeTassNaiveResponses(const std::vector<TassNaiveResponseEntry>& entries) {
        if (entries.empty()) return;
        for (const auto& entry : entries) {
            if (entry.window_seq != window_seq_) {
                SST::Output* out = diagOutOrFallback_();
                out->fatal(CALL_INFO, -1,
                           "WeightMemorySubsystem fatal: naive_tass response window mismatch (node=%u core=%u resp_window=%u active_window=%u)\n",
                           orch_.node_id,
                           orch_.core_id,
                           entry.window_seq,
                           window_seq_);
            }
            const size_t seq = static_cast<size_t>(entry.retire_seq);
            if (seq >= edge_retire_.size()) {
                SST::Output* out = diagOutOrFallback_();
                out->fatal(CALL_INFO, -1,
                           "WeightMemorySubsystem fatal: naive_tass retire_seq out of range (node=%u core=%u seq=%zu edge_retire_size=%zu window=%u)\n",
                           orch_.node_id,
                           orch_.core_id,
                           seq,
                           edge_retire_.size(),
                           window_seq_);
            }
            const bool was_ready = edge_retire_[seq].ready;
            const float resolved = applyReadRespZeroFallback_(entry.weight);
            setEdgeRetireReady_(seq, resolved, EdgeSrc::Dense);
            if (!was_ready && pending_tass_naive_responses_ > 0) pending_tass_naive_responses_ -= 1;
        }
        tryRetireEdges_();
        issueFromEdges();
    }

    void notePreGlobal(uint32_t pre_global) {
        auto inserted = active_pre_window_.insert(pre_global);
        if (inserted.second) pre_touch_order_window_.push_back(pre_global);
    }

    void noteWindowTouch(uint32_t post_local, uint32_t pre_global, uint32_t num_neurons) {
        notePostLocal(post_local, num_neurons);
        notePreGlobal(pre_global);
        noteExperimentalNocRowidxTouch_(post_local);
        noteExperimentalIdx2IngressTouch_(post_local, pre_global);
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
        const uint64_t key = (static_cast<uint64_t>(post_local) << 32) | static_cast<uint64_t>(pre_global);
        edge_collector_.curr[key] += 1;
    }
    void recordEdgeWithWeight(uint32_t post_local, uint32_t pre_global, float weight) {
        (void)weight;
        const uint64_t key = (static_cast<uint64_t>(post_local) << 32) | static_cast<uint64_t>(pre_global);
        edge_collector_.curr[key] += 1;
    }
    void recordEdgeWithPreRank(uint32_t post_local, uint32_t pre_global, float weight, uint32_t pre_rank) {
        (void)weight;
        const uint64_t key = (static_cast<uint64_t>(post_local) << 32) | static_cast<uint64_t>(pre_global);
        edge_collector_.curr[key] += 1;
        auto it = edge_pre_rank_curr_.find(key);
        if (it == edge_pre_rank_curr_.end()) {
            edge_pre_rank_curr_.emplace(key, pre_rank);
            return;
        }
        if (it->second == pre_rank) return;
        // Keep deterministic behavior even if duplicated records carry inconsistent rank.
        it->second = std::min<uint32_t>(it->second, pre_rank);
    }
    size_t edgesCurrSize() const { return edge_collector_.currSize(); }
    size_t edgesPrevSize() const { return edge_collector_.prevSize(); }
    size_t edgesPrevIter() const { return edge_collector_.prevIter(); }
    bool edgesPrevEmpty() const { return edge_collector_.prevEmpty(); }
    void flipEdgesForApply(bool debug, SST::Output* out, int core_id, uint32_t seq) {
        edge_collector_.flipForApply(debug, out, core_id, seq);
        edge_pre_rank_prev_.swap(edge_pre_rank_curr_);
        edge_pre_rank_curr_.clear();
    }
    bool nextPrevEdge(uint64_t& key, uint32_t& count) { return edge_collector_.nextPrev(key, count); }

    // ===== Orchestration entrypoints =====
    void beginApplyWindow(uint32_t seq, bool debug, SST::Output* out, int core_id) {
        window_seq_ = seq;
        maybeExportPreWindowProfile_(seq);
        if (tass_lf_p0_enabled_ && tass_lf_p0_window_started_ && tass_lf_p0_window_seq_ != seq) {
            flushTassLfP0WindowIfNeeded_(true);
        }
        // Gather-only experimental prefetch: stop enqueueing once Apply begins.
        if (orch_.experimental_noc_rowidx_prefetch_gather_only) {
            experimental_noc_rowidx_pending_rows_.clear();
        }
        if (orch_.experimental_idx2_ingress_prefetch_gather_only) {
            experimental_idx2_ingress_pending_addrs_.clear();
            experimental_idx2_ingress_pending_set_.clear();
        }
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
        resetGcssVlfIssueQueue_();
        pending_tass_naive_responses_ = 0;
        tass_naive_window_request_submitted_ = false;
        if (tass_lf_p0_enabled_) {
            tass_lf_p0_window_started_ = true;
            tass_lf_p0_window_seq_ = seq;
            tass_lf_p0_window_pre_payload_bytes_.clear();
            tass_lf_p0_window_payload_bytes_ = 0;
            tass_lf_p0_window_current_vlf_line_groups_ = 0;
        }
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
        flushTassLfP0WindowIfNeeded_(false);
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
        flushTassLfP0WindowIfNeeded_(true);

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
        const bool gcss_mode = isGcssValueOnlyMode_();
        if (orch_.ensure_loader_ready && !orch_.ensure_loader_ready()) {
            if (diag_debug_ && diag_out_) {
                diag_out_->verbose(CALL_INFO, 0, 0,
                    "[diag-window-read] BeginApply: core=%u window=%u defer edge issue (loader not ready)\n",
                    static_cast<uint32_t>(diag_core_id_), diag_seq_);
            }
            return;
        }
        if (!gcss_mode && orch_.use_bcsr && orch_.bcsr_rowptr_ready && !orch_.bcsr_rowptr_ready()) {
            if (diag_debug_ && diag_out_) {
                diag_out_->verbose(CALL_INFO, 0, 0,
                    "[diag-window-read] BeginApply: core=%u window=%u defer edge issue (BCSR rowptr not ready)\n",
                    static_cast<uint32_t>(diag_core_id_), diag_seq_);
            }
            return;
        }

        if (gcss_mode && isGcssValueOnlyPreMphfMode_()) {
            if (!gcss_vlf_issue_queue_prepared_) {
                prepareGcssVlfIssueQueue_();
            }
            while (canIssueMoreReads_() && !gcss_vlf_issue_queue_.empty()) {
                GcssVlfEdgeIssueEntry e = std::move(gcss_vlf_issue_queue_.front());
                gcss_vlf_issue_queue_.pop_front();
                auto cb = [this, seq = e.retire_seq](float w) {
                    float resolved = applyReadRespZeroFallback_(w);
                    setEdgeRetireReady_(seq, resolved, EdgeSrc::Dense);
                    tryRetireEdges_();
                    issueFromEdges();
                };
                issueGcssByAddr_(e.addr, std::move(cb), /*count_weight_read*/true);
            }
            return;
        }

        if (gcss_mode && isGcssValueOnlyBlockNaiveTassMode_()) {
            if (tass_naive_window_request_submitted_) return;
            tass_naive_window_request_submitted_ = true;
            TassNaiveWindowRequest request{};
            request.mesh_rows = std::max<uint32_t>(1u, orch_.tass_lf_p0_mesh_rows);
            request.mesh_cols = std::max<uint32_t>(1u, orch_.tass_lf_p0_mesh_cols);
            request.block_h = std::max<uint32_t>(1u, orch_.tass_lf_p0_block_h);
            request.block_w = std::max<uint32_t>(1u, orch_.tass_lf_p0_block_w);
            request.source_node = orch_.node_id;
            request.source_core = orch_.core_id;
            request.cores_per_pe = std::max<uint32_t>(1u, orch_.tass_lf_p0_cores_per_pe);
            request.window_seq = window_seq_;
            request.line_size_bytes = std::max<uint32_t>(1u, orch_.line_size_bytes);
            while (true) {
                uint64_t key = 0;
                uint32_t count = 0;
                if (!nextPrevEdge(key, count)) break;
                const uint32_t post_local = static_cast<uint32_t>(key >> 32);
                const uint32_t pre_global = static_cast<uint32_t>(key & 0xffffffffu);
                if (orch_.num_neurons > 0 && post_local >= orch_.num_neurons) continue;
                const size_t seq = registerEdgeRetire_(post_local, pre_global, count,
                                                      orch_.use_bcsr ? EdgeSrc::BCSR : EdgeSrc::Dense);
                const uint32_t block_post_local = computeNaiveTassBlockPostLocal_(post_local);
                uint32_t widx = 0;
                if (!lookupGcssWidx_(pre_global, block_post_local, widx)) {
                    SST::Output* out = diagOutOrFallback_();
                    out->fatal(CALL_INFO, -1,
                               "WeightMemorySubsystem fatal: naive_tass strict lookup miss (node=%u core=%u pre=%u post=%u block_post=%u mode=%s)\n",
                               orch_.node_id,
                               orch_.core_id,
                               pre_global,
                               post_local,
                               block_post_local,
                               orch_.synapse_weight_mode.c_str());
                    return;
                }
                TassNaiveRequestEntry entry{};
                entry.retire_seq = static_cast<uint32_t>(seq);
                entry.widx = widx;
                request.entries.push_back(entry);
                read_src_gcss_reqs_ += 1;
                read_src_gcss_bytes_ += sizeof(float);
                if (orch_.report_mem_issue) {
                    orch_.report_mem_issue(sizeof(float), /*count_weight_read*/true);
                }
            }
            pending_tass_naive_responses_ = static_cast<uint32_t>(request.entries.size());
            if (!request.entries.empty()) {
                if (!orch_.submit_tass_naive_window_request) {
                    SST::Output* out = diagOutOrFallback_();
                    out->fatal(CALL_INFO, -1,
                               "WeightMemorySubsystem fatal: naive_tass mode requires submit_tass_naive_window_request callback (node=%u core=%u)\n",
                               orch_.node_id,
                               orch_.core_id);
                    return;
                }
                orch_.submit_tass_naive_window_request(request);
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

            if (gcss_mode) {
                uint32_t widx = 0;
                if (!lookupGcssWidx_(pre_global, post_local, widx)) {
                    setEdgeRetireReady_(seq, 0.0f, EdgeSrc::Dense);
                    tryRetireEdges_();
                    continue;
                }
                auto cb = [this, seq](float w) {
                    float resolved = applyReadRespZeroFallback_(w);
                    setEdgeRetireReady_(seq, resolved, EdgeSrc::Dense);
                    tryRetireEdges_();
                    issueFromEdges();
                };
                requestGCSS_(widx, std::move(cb));
                continue;
            }

            if (orch_.use_bcsr) {
                if (orch_.bcsr_force_file_read && orch_.read_bcsr_from_file) {
                    float resolved = applyWeightGuards_(orch_.read_bcsr_from_file(post_local, pre_global));
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
                    float resolved = applyReadRespZeroFallback_(w);
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
                cached = applyReadRespZeroFallback_(cached);
                setEdgeRetireReady_(seq, cached, EdgeSrc::Cache);
                tryRetireEdges_();
                if (orch_.report_cache_access) orch_.report_cache_access(true);
                continue;
            }
            if (orch_.report_cache_access) orch_.report_cache_access(false);

            noteIssue();
            if (orch_.update_pending_peak) orch_.update_pending_peak(outstanding());
            auto miss_cb = [this, seq, cache_key](float w) {
                float resolved = applyReadRespZeroFallback_(w);
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
        if (isGcssValueOnlyMode_()) {
            // GCSS value-only path is edge-driven by design.
            // Do not fallback to Cartesian set reads, which would change semantics/cost model.
            return;
        }
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
    bool isGcssValueOnlyMode_() const;
    bool isGcssValueOnlyIdx2Mode_() const;
    bool isGcssValueOnlyPreMphfMode_() const;
    bool isGcssValueOnlyBlockNaiveTassMode_() const;
    uint32_t computeNaiveTassBlockPostLocal_(uint32_t post_local) const;
    bool ensureGcssIndexLoaded_();
    bool lookupGcssWidx_(uint32_t pre_global, uint32_t post_local, uint32_t& out_widx);
    bool lookupGcssPreBaseLen_(uint32_t pre_global, uint32_t& out_base, uint32_t& out_len);
    void requestGCSS_(uint32_t widx, std::function<void(float)> cb);
    void prepareGcssVlfIssueQueue_();
    void noteTassLfP0PreparedWindow_(const std::unordered_map<uint32_t, uint64_t>& pre_payload_bytes,
                                     uint64_t payload_bytes,
                                     uint64_t current_vlf_line_groups);
    void flushTassLfP0WindowIfNeeded_(bool force);
    void harvestTassLfP0Completions_();
    void resetGcssVlfIssueQueue_();

    float applyReadRespZeroFallback_(float w) const {
        if (orch_.readresp_zero_fallback && w == 0.0f) return orch_.init_default_weight;
        return w;
    }

    float applyWeightGuards_(float w) const;
    SST::Output* diagOutOrFallback_() const;

    bool canIssueMoreReads_() const {
        // Semantic seal: budget is soft; hard gating uses outstanding only.
        if (canIssue(/*n*/1, /*count_budget*/false)) return true;
        if (diag_debug_ && diag_out_) {
            diag_out_->verbose(CALL_INFO, 0, 0,
                "[diag-edge-loop] core=%d outstanding stop issued=%u outstanding=%u budget=%u limit=%u\n",
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
    std::vector<uint32_t> pre_touch_order_window_;
    std::unordered_map<uint64_t, uint32_t> edge_pre_rank_curr_{};
    std::unordered_map<uint64_t, uint32_t> edge_pre_rank_prev_{};
    GasEdgeCollector edge_collector_;

    // Orchestrator config
    OrchestratorConfig orch_{};

    // Dense layout (derived). Cached to avoid recomputing row_stride/packing per request.
    bool dense_phys_enable_ = false;
    uint32_t dense_cols_effective_ = 0;
    DensePhysV1Derived dense_phys_{};

    void maybeExportPreWindowProfile_(uint32_t seq);

    // ===== Deterministic retire for edge-driven acc_update =====
    // 由于 StandardMem 回调到达顺序可能在 MPI 多 rank 下抖动，直接在回调里 acc_update
    // 会导致 float 累加顺序变化，从而在“极稀疏发放/阈值临界”场景出现 0/非0 跳变。
    // 这里将每条 edge 的 (post,pre,count) 按发起顺序编号，并在结果 ready 后按序退役。
    enum class EdgeSrc : uint8_t { Dense, Cache, Miss, BCSR, BCSRFile };
    enum class RetirePolicy : uint8_t { GlobalInOrder, PerPostDeterministic };
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
    size_t retired_edges_count_ = 0;
    // Number of ready edges that are not committed yet (used for HOL diagnostics).
    uint64_t ready_uncommitted_edges_ = 0;

    RetirePolicy retire_policy_ = RetirePolicy::GlobalInOrder;
    // Per-post deterministic retire structures (experimental path).
    std::vector<std::vector<size_t>> per_post_retire_seq_{};
    std::vector<size_t> per_post_head_idx_{};
    std::vector<uint8_t> per_post_touched_{};
    std::vector<uint32_t> per_post_touched_list_{};
    std::set<uint32_t> per_post_ready_posts_{};

    // Retire observability counters (run totals).
    uint64_t retire_global_hol_cycles_total_ = 0;
    uint64_t retire_ready_but_blocked_edges_total_ = 0;
    uint64_t retire_per_post_progress_total_ = 0;

    bool issue_edges_in_progress_ = false;
    bool issue_edges_again_ = false;
    uint32_t pending_tass_naive_responses_ = 0;
    bool tass_naive_window_request_submitted_ = false;

    bool usePerPostRetire_() const { return retire_policy_ == RetirePolicy::PerPostDeterministic; }

    void ensurePerPostRetireStorage_() {
        const size_t n = static_cast<size_t>(orch_.num_neurons);
        if (per_post_retire_seq_.size() == n &&
            per_post_head_idx_.size() == n &&
            per_post_touched_.size() == n) {
            return;
        }
        per_post_retire_seq_.assign(n, std::vector<size_t>{});
        per_post_head_idx_.assign(n, 0);
        per_post_touched_.assign(n, 0);
        per_post_touched_list_.clear();
        per_post_ready_posts_.clear();
    }

    void resetEdgeRetire_() {
        edge_retire_.clear();
        next_retire_seq_ = 0;
        retired_edges_count_ = 0;
        ready_uncommitted_edges_ = 0;
        edge_retire_.reserve(edgesPrevSize());
        per_post_ready_posts_.clear();
        if (usePerPostRetire_()) {
            ensurePerPostRetireStorage_();
            for (uint32_t post : per_post_touched_list_) {
                if (post >= per_post_retire_seq_.size()) continue;
                per_post_retire_seq_[post].clear();
                per_post_head_idx_[post] = 0;
                per_post_touched_[post] = 0;
            }
            per_post_touched_list_.clear();
        }
    }

    size_t registerEdgeRetire_(uint32_t post_local, uint32_t pre_global, uint32_t count, EdgeSrc src) {
        const size_t seq = edge_retire_.size();
        EdgeRetireEntry e;
        e.post_local = post_local;
        e.pre_global = pre_global;
        e.count = count;
        e.src = src;
        edge_retire_.push_back(e);
        if (usePerPostRetire_() && post_local < static_cast<uint32_t>(orch_.num_neurons)) {
            ensurePerPostRetireStorage_();
            if (post_local < per_post_touched_.size() && per_post_touched_[post_local] == 0) {
                per_post_touched_[post_local] = 1;
                per_post_touched_list_.push_back(post_local);
                per_post_head_idx_[post_local] = 0;
                per_post_retire_seq_[post_local].clear();
            }
            if (post_local < per_post_retire_seq_.size()) {
                per_post_retire_seq_[post_local].push_back(seq);
            }
        }
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
        auto& e = edge_retire_[seq];
        const bool was_ready = e.ready;
        e.weight = weight;
        e.ready = true;
        e.src = src;
        if (!was_ready) {
            ready_uncommitted_edges_ += 1;
        }
        if (usePerPostRetire_()) {
            const uint32_t post = e.post_local;
            if (post < per_post_retire_seq_.size() && post < per_post_head_idx_.size()) {
                const auto& seqs = per_post_retire_seq_[post];
                const size_t head_idx = per_post_head_idx_[post];
                if (head_idx < seqs.size() && seqs[head_idx] == seq) {
                    per_post_ready_posts_.insert(post);
                }
            }
        }
    }

    void commitRetireEntry_(const EdgeRetireEntry& e) {
        if (byteExactVerifyEnabled_() && e.src == EdgeSrc::Dense) {
            verifyDenseEdgeWeight_(e.pre_global, e.post_local, e.count, e.weight);
        }
        if (orch_.acc_update) orch_.acc_update(e.post_local, e.weight * static_cast<float>(e.count));
        if (orch_.diag_edge_weight) orch_.diag_edge_weight(edgeSrcTag_(e.src), e.post_local, e.pre_global, e.weight, e.count);
    }

    void tryRetireEdgesPerPost_() {
        while (!per_post_ready_posts_.empty()) {
            auto it = per_post_ready_posts_.begin();
            const uint32_t post = *it;
            if (post >= per_post_retire_seq_.size() || post >= per_post_head_idx_.size()) {
                per_post_ready_posts_.erase(it);
                continue;
            }
            auto& seqs = per_post_retire_seq_[post];
            size_t head_idx = per_post_head_idx_[post];
            if (head_idx >= seqs.size()) {
                per_post_ready_posts_.erase(it);
                continue;
            }
            const size_t seq = seqs[head_idx];
            if (seq >= edge_retire_.size()) {
                per_post_ready_posts_.erase(it);
                continue;
            }
            const auto& e = edge_retire_[seq];
            if (!e.ready) {
                per_post_ready_posts_.erase(it);
                continue;
            }
            commitRetireEntry_(e);
            if (ready_uncommitted_edges_ > 0) ready_uncommitted_edges_--;
            retired_edges_count_++;
            retire_per_post_progress_total_++;
            per_post_head_idx_[post] = head_idx + 1;
            head_idx = per_post_head_idx_[post];
            if (head_idx < seqs.size()) {
                const size_t next_seq = seqs[head_idx];
                if (next_seq < edge_retire_.size() && edge_retire_[next_seq].ready) {
                    // keep in ready set
                } else {
                    per_post_ready_posts_.erase(it);
                }
            } else {
                per_post_ready_posts_.erase(it);
            }
        }
    }

    void updateRetireHolStatsOnTick_() {
        if (usePerPostRetire_()) return;
        if (edge_retire_.empty()) return;
        if (next_retire_seq_ >= edge_retire_.size()) return;
        const auto& head = edge_retire_[next_retire_seq_];
        if (!head.ready && ready_uncommitted_edges_ > 0) {
            retire_global_hol_cycles_total_ += 1;
            retire_ready_but_blocked_edges_total_ += ready_uncommitted_edges_;
        }
    }

    void tryRetireEdges_() {
        if (usePerPostRetire_()) {
            tryRetireEdgesPerPost_();
            return;
        }
        while (next_retire_seq_ < edge_retire_.size()) {
            const auto& e = edge_retire_[next_retire_seq_];
            if (!e.ready) break;
            commitRetireEntry_(e);
            if (ready_uncommitted_edges_ > 0) ready_uncommitted_edges_--;
            retired_edges_count_++;
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
        // 0=dense, 1=rowptr, 2=colidx, 3=blockdata, 4=gcss_valueonly
        int bcsr_kind = 0;
        uint32_t bcsr_block_row = 0;
        uint32_t bcsr_target_block_col = 0;
        uint32_t bcsr_intra_row = 0;
        uint32_t bcsr_intra_col = 0;
        uint32_t bcsr_row_start = 0;
        uint32_t bcsr_idx_in_row = 0;
        uint32_t bcsr_global_block_index = 0;
        bool bcsr_row_slice_fetch = false;
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
    struct ExperimentalNocRowidxCacheEntry {
        uint32_t row_start = 0;
        std::vector<uint32_t> cols;
    };
    std::unordered_map<uint32_t, ExperimentalNocRowidxCacheEntry> experimental_noc_rowidx_cache_{};
    std::deque<uint32_t> experimental_noc_rowidx_pending_rows_{};
    std::vector<uint8_t> experimental_noc_rowidx_touched_rows_{};
    std::vector<uint16_t> experimental_noc_rowidx_touch_counts_{};
    ExperimentalNocRowidxStats experimental_noc_rowidx_stats_{};
    struct ExperimentalIdx2IngressInflightEntry {
        bool issued = false;
        std::vector<std::function<void(float)>> waiters;
    };
    std::deque<uint64_t> experimental_idx2_ingress_pending_addrs_{};
    std::unordered_set<uint64_t> experimental_idx2_ingress_pending_set_{};
    std::unordered_map<uint64_t, ExperimentalIdx2IngressInflightEntry> experimental_idx2_ingress_inflight_{};
    std::unordered_map<uint64_t, float> experimental_idx2_ingress_value_cache_{};
    std::deque<uint64_t> experimental_idx2_ingress_value_cache_lru_{};
    ExperimentalIdx2IngressStats experimental_idx2_ingress_stats_{};
    uint32_t window_seq_ = 0;
    bool bcsr_rowptr_file_preload_attempted_ = false;
    bool bcsr_rowidx_file_preloaded_ = false;
    bool gcss_index_loaded_ = false;
    bool gcss_index_load_failed_ = false;
    std::string gcss_index_path_;
    GcssIndexTable gcss_index_;
    bool gcss_idx2_index_loaded_ = false;
    bool gcss_idx2_index_load_failed_ = false;
    std::string gcss_idx2_index_path_;
    GcssIndexRowMphf gcss_idx2_index_;
    bool gcss_premphf_index_loaded_ = false;
    bool gcss_premphf_index_load_failed_ = false;
    std::string gcss_premphf_index_path_;
    GcssIndexPreMphf gcss_premphf_index_;
    struct GcssVlfEdgeIssueEntry {
        size_t retire_seq = 0;
        uint32_t post_local = 0;
        uint32_t pre_global = 0;
        uint32_t pre_rank = 0;
        uint32_t count = 0;
        uint64_t addr = 0;
    };
    std::deque<GcssVlfEdgeIssueEntry> gcss_vlf_issue_queue_{};
    bool gcss_vlf_issue_queue_prepared_ = false;
    uint64_t gcss_vlf_issue_prepare_total_ = 0;
    uint64_t gcss_vlf_issue_edges_total_ = 0;
    uint64_t gcss_vlf_issue_reorder_trigger_total_ = 0;
    uint64_t gcss_vlf_issue_line_groups_total_ = 0;
    bool tass_lf_p0_enabled_ = false;
    bool tass_lf_p0_window_started_ = false;
    uint32_t tass_lf_p0_window_seq_ = 0;
    std::unordered_map<uint32_t, uint64_t> tass_lf_p0_window_pre_payload_bytes_{};
    uint64_t tass_lf_p0_window_payload_bytes_ = 0;
    uint64_t tass_lf_p0_window_current_vlf_line_groups_ = 0;
    bool tass_lf_p0_flush_branch_logged_ = false;
    TassLfP0Stats tass_lf_p0_stats_{};
    uint64_t gcss_lookup_hit_ = 0;
    uint64_t gcss_lookup_miss_ = 0;
    uint64_t read_src_dense_reqs_ = 0;
    uint64_t read_src_dense_bytes_ = 0;
    uint64_t read_src_rowptr_reqs_ = 0;
    uint64_t read_src_rowptr_bytes_ = 0;
    uint64_t read_src_colidx_reqs_ = 0;
    uint64_t read_src_colidx_bytes_ = 0;
    uint64_t read_src_blockdata_reqs_ = 0;
    uint64_t read_src_blockdata_bytes_ = 0;
    uint64_t read_src_gcss_reqs_ = 0;
    uint64_t read_src_gcss_bytes_ = 0;

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
        uint32_t idx_in_row = 0;
        uint32_t intra_row = 0;
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
    enum class BcsrBlockFetchMode : uint8_t { FullBlock = 0, RowCacheline = 1 };
    IssueStatus tryIssueRead_(PendingMeta meta, bool count_budget, bool budget_reserved);
    void drainPendingReads_();
    void drainPendingDirectReads_();
    void drainExperimentalNocRowidxPrefetch_();
    void drainExperimentalIdx2IngressPrefetch_();
    uint32_t computeExperimentalNocRowidxBudget_();
    uint32_t computeExperimentalIdx2IngressBudget_() const;

    static uint64_t makeInflightKey_(uint32_t window_seq, uint32_t id) {
        return (static_cast<uint64_t>(window_seq) << 32) | static_cast<uint64_t>(id);
    }
    static uint64_t makeBlockInflightKey_(uint32_t window_seq,
                                          uint32_t global_block_index,
                                          uint32_t intra_row) {
        return (static_cast<uint64_t>(window_seq) << 48) |
               ((static_cast<uint64_t>(intra_row) & 0xffffull) << 32) |
               static_cast<uint64_t>(global_block_index);
    }

    void handleReadResp_(uint64_t req_id, uint64_t addr, PendingMeta meta, std::vector<uint8_t>&& data);
    void noteReadSourceIssue_(int bcsr_kind, size_t req_bytes);
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
                                        uint32_t idx_in_row,
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
    bool experimentalNocRowidxPrefetchEnabled_() const;
    void resetExperimentalNocRowidxWindow_();
    void noteExperimentalNocRowidxTouch_(uint32_t post_local);
    bool lookupExperimentalNocRowidxCache_(uint32_t block_row,
                                           uint32_t& row_start_out,
                                           const std::vector<uint32_t>*& cols_out);
    void storeExperimentalNocRowidxCache_(uint32_t block_row,
                                          uint32_t row_start,
                                          const std::vector<uint32_t>& cols);
    void maybeAutoTuneBlockCache_();
    void bcsrPrefetchAll_();
    void bcsrPrefetchRowBlocks_(uint32_t block_row, const std::vector<uint32_t>& cols, uint32_t row_start);
    void bcsrPopulateWeightCache_(uint32_t block_row, uint32_t block_col, const std::vector<float>& blk);
    bool experimentalIdx2IngressPrefetchEnabled_() const;
    void noteExperimentalIdx2IngressTouch_(uint32_t post_local, uint32_t pre_global);
    bool tryServeExperimentalIdx2Ingress_(uint64_t addr, std::function<void(float)> cb);
    bool lookupExperimentalIdx2IngressCache_(uint64_t addr, float& value_out);
    void storeExperimentalIdx2IngressCache_(uint64_t addr, float value);
    bool shouldTailGuardIdx2RespDrop_() const;
    void completeExperimentalIdx2IngressPrefetch_(uint64_t addr, bool ok, float value);
    void issueGcssByAddr_(uint64_t addr, std::function<void(float)> cb, bool count_weight_read);
    void noteIdxSramLookup_(uint32_t pre_global, uint32_t post_local, bool idx2_mode);
    void noteL0SramLookup_(uint64_t addr, bool hit);
    void noteL0SramFill_(uint64_t addr);
    void noteL0SramEvict_(uint64_t addr);

    BcsrBlockFetchMode bcsr_block_fetch_mode_ = BcsrBlockFetchMode::FullBlock;
    VirtualSramLayout sram_layout_{};
    bool weight_sram_enable_ = false;
    bool weight_idx_sram_enable_ = false;
    bool weight_l0_sram_enable_ = false;
    BankedSramModel idx_sram_model_{};
    BankedSramModel l0_sram_model_{};
    uint64_t idx_lookup_total_ = 0;
    uint64_t idx_lookup_idx2_total_ = 0;
    uint64_t idx_lookup_legacy_total_ = 0;
    uint64_t l0_lookup_total_ = 0;
    uint64_t l0_hit_total_ = 0;
    uint64_t l0_miss_total_ = 0;
    uint64_t l0_fill_total_ = 0;
    uint64_t l0_evict_total_ = 0;
};

}} // namespace SST::SnnDL
