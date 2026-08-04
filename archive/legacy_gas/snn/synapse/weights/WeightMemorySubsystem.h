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
#include <limits>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "snn/synapse/gas/GasEdgeCollector.h"
#include "SnnWeightReader.h"
#include "IMemoryAccess.h"
#include "WeightAccessor.h"
#include "WeightCacheOps.h"
#include "DenseWeightLayout.h"
#include "research/local_storage/PeLocalServiceObjectTable.h"
#include "research/local_storage/PodMetadataObjectPlane.h"
#include "research/local_storage/PeWeightObjectPlane.h"
#include "platform/memory/sram_sim/layout/VirtualSramLayout.h"
#include "platform/memory/sram_sim/model/BankedSramModel.h"
#include "IPeAggregation.h"

namespace SST { class Output; }

namespace SST { namespace SnnDL {

class BcsrWeightManager;
class PeWeightObjectPlane;
class PeLocalServiceObjectTable;
class PodMetadataObjectPlane;
class PodOwnerServiceTable;

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
        uint32_t total_cores = 1;
        std::string weights_template;
        BcsrWeightManager* bcsr_mgr = nullptr;
        bool byte_exact_verify_enable = false;
        std::string byte_exact_verify_mode;
        uint32_t byte_exact_verify_row_scale = 1024;
        uint32_t byte_exact_verify_max_mismatch = 8;
        std::string synapse_weight_mode = "bcsr_gas";
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
        bool experimental_retire_shadow_per_post_enable = false;
        bool pe_internal_cpe_enable = false;
        bool pe_internal_pod_enable = false;
        bool pe_internal_pod_metadata_enable = false;
        bool pe_internal_pod_owner_enable = false;
        uint32_t pe_internal_pod_id = 0;
        uint32_t pe_internal_pod_count = 1;
        uint32_t pe_internal_pod_size = 1;
    };

    WeightMemorySubsystem() = default;

    // ===== Runtime binding (Phase1) =====
    void bindMemory(IMemoryAccess* mem) { mem_access_ = mem; }
    void bindSharedWeightObjectPlane(PeWeightObjectPlane* plane) {
        shared_weight_object_plane_ = plane;
    }
    void bindPodMetadataObjectPlane(PodMetadataObjectPlane* plane) {
        pod_metadata_object_plane_ = plane;
    }
    void bindPodOwnerServiceTable(PodOwnerServiceTable* table) {
        pod_owner_service_table_ = table;
    }
    void bindPeLocalServiceObjectTable(PeLocalServiceObjectTable* table) {
        pe_local_service_object_table_ = table;
    }
    void setPeInternalPodRuntimeConfig(bool pe_internal_cpe_enable,
                                       bool pe_internal_pod_enable,
                                       bool pe_internal_pod_metadata_enable,
                                       bool pe_internal_pod_owner_enable,
                                       uint32_t pod_id,
                                       uint32_t pod_count,
                                       uint32_t pod_size,
                                       uint32_t core_id) {
        orch_.core_id = core_id;
        orch_.pe_internal_cpe_enable = pe_internal_cpe_enable;
        orch_.pe_internal_pod_enable =
            orch_.pe_internal_cpe_enable && pe_internal_pod_enable;
        orch_.pe_internal_pod_metadata_enable =
            orch_.pe_internal_pod_enable && pe_internal_pod_metadata_enable;
        orch_.pe_internal_pod_owner_enable =
            orch_.pe_internal_pod_enable && pe_internal_pod_owner_enable;
        orch_.pe_internal_pod_count =
            orch_.pe_internal_pod_enable ? std::max<uint32_t>(1u, pod_count) : 1u;
        orch_.pe_internal_pod_size = std::max<uint32_t>(1u, pod_size);
        orch_.pe_internal_pod_id =
            orch_.pe_internal_pod_enable
                ? std::min<uint32_t>(pod_id, orch_.pe_internal_pod_count - 1u)
                : 0u;
    }
    void setSharedWeightObjectPlaneResidencyAuthority(bool enable) {
        shared_weight_object_plane_residency_authority_ = enable;
    }
    size_t pendingSize() const { return mem_access_ ? mem_access_->pendingSize() : 0; }
    struct DeferredWorkBreakdown {
        size_t pending_bcsr_rowptr_waiters = 0;
        uint64_t weight_sram_stall_budget_cycles = 0;
        bool row_index_prefetch_bulk_pending = false;
        bool row_index_prefetch_bulk_inflight = false;
        size_t row_index_prefetch_rows = 0;
        size_t pending_colidx_reads = 0;
        size_t pending_block_reads = 0;
        size_t pending_direct_reads = 0;
        size_t edge_retire_total = 0;
        size_t edge_retire_retired = 0;
    };
    DeferredWorkBreakdown deferredWorkBreakdown() const {
        DeferredWorkBreakdown d{};
        d.pending_bcsr_rowptr_waiters = pending_bcsr_rowptr_waiters_.size();
        d.weight_sram_stall_budget_cycles = weight_sram_stall_budget_cycles_;
        d.row_index_prefetch_bulk_pending = row_index_prefetch_bulk_pending_;
        d.row_index_prefetch_bulk_inflight = row_index_prefetch_bulk_inflight_;
        d.row_index_prefetch_rows = row_index_prefetch_rows_.size();
        d.pending_colidx_reads = pending_colidx_reads_.size();
        d.pending_block_reads = pending_block_reads_.size();
        d.pending_direct_reads = pending_direct_reads_.size();
        d.edge_retire_total = edge_retire_.size();
        d.edge_retire_retired = retired_edges_count_;
        return d;
    }
    bool hasDeferredWork() const {
        return !pending_bcsr_rowptr_waiters_.empty() ||
               weight_sram_stall_budget_cycles_ != 0 ||
               row_index_prefetch_bulk_pending_ ||
               row_index_prefetch_bulk_inflight_ ||
               !row_index_prefetch_rows_.empty() ||
               !pending_colidx_reads_.empty() ||
               !pending_block_reads_.empty() ||
               !pending_direct_reads_.empty() ||
               (retired_edges_count_ < edge_retire_.size());
    }
    void setNowCycle(uint64_t now_cycle) { now_cycle_ = now_cycle; }
    void onClockTick(uint64_t now_cycle);
    struct ExperimentalPeInternalPodPathAlignmentLane {
        uint64_t producer_touch_events_total = 0;
        uint64_t producer_enqueued_total = 0;
        uint64_t seam_owner_form_total = 0;
        uint64_t seam_joiner_hit_total = 0;
        uint64_t seam_joiner_useful_total = 0;
        uint64_t seam_owner_live_join_total = 0;
        uint64_t seam_owner_ready_join_total = 0;
        uint64_t seam_ready_transition_total = 0;
        uint64_t seam_ready_fanout_total = 0;
        uint64_t seam_ready_fanout_consumers_sum = 0;
        uint64_t seam_late_join_total = 0;
        uint64_t seam_potential_private_service_elide_total = 0;
        uint64_t seam_owner_first_issue_deferred_total = 0;
        uint64_t seam_owner_first_private_issue_avoided_total = 0;
        uint64_t seam_owner_first_service_elide_total = 0;
        uint64_t seam_guard_total = 0;
        uint64_t seam_guard_disabled_total = 0;
        uint64_t seam_guard_missing_metadata_plane_total = 0;
        uint64_t seam_guard_missing_owner_table_total = 0;
        uint64_t seam_guard_zero_pod_count_total = 0;
        uint64_t seam_guard_window_zero_total = 0;
        uint64_t seam_guard_invalid_cfg_pod_total = 0;
        uint64_t seam_reject_total = 0;
        uint64_t seam_useful_total = 0;
        uint64_t seam_attempted_total = 0;
    };
    struct ExperimentalPeInternalPodPathAlignmentStats {
        ExperimentalPeInternalPodPathAlignmentLane idx2row{};
        ExperimentalPeInternalPodPathAlignmentLane rowindex{};
        ExperimentalPeInternalPodPathAlignmentLane rowdescriptor{};
    };
    ExperimentalPeInternalPodPathAlignmentStats
    experimentalPeInternalPodPathAlignmentStats() const {
        const auto pod = peInternalPodStats();

        ExperimentalPeInternalPodPathAlignmentStats s{};
        s.idx2row.producer_touch_events_total = 0;
        s.idx2row.producer_enqueued_total = 0;
        s.idx2row.seam_owner_form_total = pod.owner_alloc_idx2row_total;
        s.idx2row.seam_joiner_hit_total = pod.owner_hit_idx2row_total;
        s.idx2row.seam_joiner_useful_total = pod.useful_idx2row_total;
        s.idx2row.seam_owner_live_join_total = pod.service_join_live_idx2row_total;
        s.idx2row.seam_owner_ready_join_total = pod.service_join_ready_idx2row_total;
        s.idx2row.seam_ready_transition_total =
            pod.service_ready_transition_idx2row_total;
        s.idx2row.seam_ready_fanout_total =
            pod.service_ready_fanout_idx2row_total;
        s.idx2row.seam_ready_fanout_consumers_sum =
            pod.service_ready_fanout_consumers_sum_idx2row_total;
        s.idx2row.seam_late_join_total = pod.service_late_join_idx2row_total;
        s.idx2row.seam_potential_private_service_elide_total =
            pod.service_potential_private_service_elide_idx2row_total;
        s.idx2row.seam_owner_first_issue_deferred_total =
            pod.owner_first_issue_deferred_idx2row_total;
        s.idx2row.seam_owner_first_private_issue_avoided_total =
            pod.owner_first_private_issue_avoided_idx2row_total;
        s.idx2row.seam_owner_first_service_elide_total =
            pod.owner_first_service_elide_idx2row_total;
        s.idx2row.seam_guard_total = pod.guard_idx2row_total;
        s.idx2row.seam_reject_total = pod.reject_idx2row_total;
        s.idx2row.seam_useful_total = pod.useful_idx2row_total;
        s.idx2row.seam_attempted_total = pod.attempted_idx2row_total;

        s.rowindex.producer_touch_events_total = 0;
        s.rowindex.producer_enqueued_total = 0;
        s.rowindex.seam_owner_form_total = pod.owner_alloc_rowindex_total;
        s.rowindex.seam_joiner_hit_total = pod.owner_hit_rowindex_total;
        s.rowindex.seam_joiner_useful_total = pod.useful_rowindex_total;
        s.rowindex.seam_owner_live_join_total = pod.service_join_live_rowindex_total;
        s.rowindex.seam_owner_ready_join_total = pod.service_join_ready_rowindex_total;
        s.rowindex.seam_ready_transition_total =
            pod.service_ready_transition_rowindex_total;
        s.rowindex.seam_ready_fanout_total =
            pod.service_ready_fanout_rowindex_total;
        s.rowindex.seam_ready_fanout_consumers_sum =
            pod.service_ready_fanout_consumers_sum_rowindex_total;
        s.rowindex.seam_late_join_total = pod.service_late_join_rowindex_total;
        s.rowindex.seam_potential_private_service_elide_total =
            pod.service_potential_private_service_elide_rowindex_total;
        s.rowindex.seam_owner_first_issue_deferred_total =
            pod.owner_first_issue_deferred_rowindex_total;
        s.rowindex.seam_owner_first_private_issue_avoided_total =
            pod.owner_first_private_issue_avoided_rowindex_total;
        s.rowindex.seam_owner_first_service_elide_total =
            pod.owner_first_service_elide_rowindex_total;
        s.rowindex.seam_guard_total = pod.guard_rowindex_total;
        s.rowindex.seam_reject_total = pod.reject_rowindex_total;
        s.rowindex.seam_useful_total = pod.useful_rowindex_total;
        s.rowindex.seam_attempted_total = pod.attempted_rowindex_total;

        s.rowdescriptor.producer_touch_events_total = 0;
        s.rowdescriptor.producer_enqueued_total = 0;
        s.rowdescriptor.seam_owner_form_total = pod.owner_alloc_rowdescriptor_total;
        s.rowdescriptor.seam_joiner_hit_total = pod.owner_hit_rowdescriptor_total;
        s.rowdescriptor.seam_joiner_useful_total = pod.useful_rowdescriptor_total;
        s.rowdescriptor.seam_owner_live_join_total =
            pod.service_join_live_rowdescriptor_total;
        s.rowdescriptor.seam_owner_ready_join_total =
            pod.service_join_ready_rowdescriptor_total;
        s.rowdescriptor.seam_ready_transition_total =
            pod.service_ready_transition_rowdescriptor_total;
        s.rowdescriptor.seam_ready_fanout_total =
            pod.service_ready_fanout_rowdescriptor_total;
        s.rowdescriptor.seam_ready_fanout_consumers_sum =
            pod.service_ready_fanout_consumers_sum_rowdescriptor_total;
        s.rowdescriptor.seam_late_join_total =
            pod.service_late_join_rowdescriptor_total;
        s.rowdescriptor.seam_potential_private_service_elide_total =
            pod.service_potential_private_service_elide_rowdescriptor_total;
        s.rowdescriptor.seam_owner_first_issue_deferred_total =
            pod.owner_first_issue_deferred_rowdescriptor_total;
        s.rowdescriptor.seam_owner_first_private_issue_avoided_total =
            pod.owner_first_private_issue_avoided_rowdescriptor_total;
        s.rowdescriptor.seam_owner_first_service_elide_total =
            pod.owner_first_service_elide_rowdescriptor_total;
        s.rowdescriptor.seam_guard_total = pod.guard_rowdescriptor_total;
        s.rowdescriptor.seam_guard_disabled_total =
            pod.guard_rowdescriptor_disabled_total;
        s.rowdescriptor.seam_guard_missing_metadata_plane_total =
            pod.guard_rowdescriptor_missing_metadata_plane_total;
        s.rowdescriptor.seam_guard_missing_owner_table_total =
            pod.guard_rowdescriptor_missing_owner_table_total;
        s.rowdescriptor.seam_guard_zero_pod_count_total =
            pod.guard_rowdescriptor_zero_pod_count_total;
        s.rowdescriptor.seam_guard_window_zero_total =
            pod.guard_rowdescriptor_window_zero_total;
        s.rowdescriptor.seam_guard_invalid_cfg_pod_total =
            pod.guard_rowdescriptor_invalid_cfg_pod_total;
        s.rowdescriptor.seam_reject_total = pod.reject_rowdescriptor_total;
        s.rowdescriptor.seam_useful_total = pod.useful_rowdescriptor_total;
        s.rowdescriptor.seam_attempted_total = pod.attempted_rowdescriptor_total;
        return s;
    }
    struct PeInternalPodStats {
        uint64_t guard_drop_total = 0;
        uint64_t guard_disabled_total = 0;
        uint64_t guard_missing_metadata_plane_total = 0;
        uint64_t guard_missing_owner_table_total = 0;
        uint64_t guard_zero_pod_count_total = 0;
        uint64_t guard_window_zero_total = 0;
        uint64_t guard_invalid_cfg_pod_total = 0;
        uint64_t guard_rowdescriptor_disabled_total = 0;
        uint64_t guard_rowdescriptor_missing_metadata_plane_total = 0;
        uint64_t guard_rowdescriptor_missing_owner_table_total = 0;
        uint64_t guard_rowdescriptor_zero_pod_count_total = 0;
        uint64_t guard_rowdescriptor_window_zero_total = 0;
        uint64_t guard_rowdescriptor_invalid_cfg_pod_total = 0;
        uint64_t guard_base_total = 0;
        uint64_t guard_band_total = 0;
        uint64_t guard_other_total = 0;
        uint64_t guard_idx2row_total = 0;
        uint64_t guard_rowindex_total = 0;
        uint64_t guard_rowdescriptor_total = 0;
        uint64_t frontier_export_total = 0;
        uint64_t frontier_consumer_count_sum_total = 0;
        uint64_t frontier_overlap_strength_sum_total = 0;
        uint64_t frontier_base_consumer_count_sum_total = 0;
        uint64_t frontier_base_overlap_strength_sum_total = 0;
        uint64_t frontier_band_consumer_count_sum_total = 0;
        uint64_t frontier_band_overlap_strength_sum_total = 0;
        uint64_t owner_lookup_total = 0;
        uint64_t owner_alloc_total = 0;
        uint64_t owner_alloc_idx2row_total = 0;
        uint64_t owner_alloc_rowindex_total = 0;
        uint64_t owner_alloc_rowdescriptor_total = 0;
        uint64_t owner_hit_total = 0;
        uint64_t owner_hit_idx2row_total = 0;
        uint64_t owner_hit_rowindex_total = 0;
        uint64_t owner_hit_rowdescriptor_total = 0;
        uint64_t owner_reject_total = 0;
        uint64_t owner_disabled_reject_total = 0;
        uint64_t owner_invalid_pod_reject_total = 0;
        uint64_t owner_table_full_reject_total = 0;
        uint64_t join_request_total = 0;
        uint64_t join_grant_total = 0;
        uint64_t join_reject_total = 0;
        uint64_t join_table_disabled_reject_total = 0;
        uint64_t join_duplicate_consumer_reject_total = 0;
        uint64_t join_table_full_reject_total = 0;
        uint64_t join_before_private_issue_total = 0;
        uint64_t owner_first_issue_deferred_total = 0;
        uint64_t owner_first_issue_deferred_idx2row_total = 0;
        uint64_t owner_first_issue_deferred_rowindex_total = 0;
        uint64_t owner_first_issue_deferred_rowdescriptor_total = 0;
        uint64_t owner_first_private_issue_avoided_total = 0;
        uint64_t owner_first_private_issue_avoided_idx2row_total = 0;
        uint64_t owner_first_private_issue_avoided_rowindex_total = 0;
        uint64_t owner_first_private_issue_avoided_rowdescriptor_total = 0;
        uint64_t reject_base_total = 0;
        uint64_t reject_band_total = 0;
        uint64_t reject_other_total = 0;
        uint64_t reject_idx2row_total = 0;
        uint64_t reject_rowindex_total = 0;
        uint64_t reject_rowdescriptor_total = 0;
        uint64_t useful_total = 0;
        uint64_t useful_join_grant_total = 0;
        uint64_t useful_duplicate_replay_elide_total = 0;
        uint64_t useful_base_total = 0;
        uint64_t useful_band_total = 0;
        uint64_t useful_other_total = 0;
        uint64_t useful_idx2row_total = 0;
        uint64_t useful_rowindex_total = 0;
        uint64_t useful_rowdescriptor_total = 0;
        uint64_t service_join_live_total = 0;
        uint64_t service_join_live_idx2row_total = 0;
        uint64_t service_join_live_rowindex_total = 0;
        uint64_t service_join_live_rowdescriptor_total = 0;
        uint64_t service_join_ready_total = 0;
        uint64_t service_join_ready_idx2row_total = 0;
        uint64_t service_join_ready_rowindex_total = 0;
        uint64_t service_join_ready_rowdescriptor_total = 0;
        uint64_t service_ready_transition_total = 0;
        uint64_t service_ready_transition_idx2row_total = 0;
        uint64_t service_ready_transition_rowindex_total = 0;
        uint64_t service_ready_transition_rowdescriptor_total = 0;
        uint64_t service_ready_fanout_total = 0;
        uint64_t service_ready_fanout_idx2row_total = 0;
        uint64_t service_ready_fanout_rowindex_total = 0;
        uint64_t service_ready_fanout_rowdescriptor_total = 0;
        uint64_t service_ready_fanout_consumers_sum_total = 0;
        uint64_t service_ready_fanout_consumers_sum_idx2row_total = 0;
        uint64_t service_ready_fanout_consumers_sum_rowindex_total = 0;
        uint64_t service_ready_fanout_consumers_sum_rowdescriptor_total = 0;
        uint64_t service_release_deferred_total = 0;
        uint64_t service_release_deferred_idx2row_total = 0;
        uint64_t service_release_deferred_rowindex_total = 0;
        uint64_t service_release_deferred_rowdescriptor_total = 0;
        uint64_t service_ready_release_total = 0;
        uint64_t service_ready_release_idx2row_total = 0;
        uint64_t service_ready_release_rowindex_total = 0;
        uint64_t service_ready_release_rowdescriptor_total = 0;
        uint64_t service_late_join_total = 0;
        uint64_t service_late_join_idx2row_total = 0;
        uint64_t service_late_join_rowindex_total = 0;
        uint64_t service_late_join_rowdescriptor_total = 0;
        uint64_t service_potential_private_service_elide_total = 0;
        uint64_t service_potential_private_service_elide_idx2row_total = 0;
        uint64_t service_potential_private_service_elide_rowindex_total = 0;
        uint64_t service_potential_private_service_elide_rowdescriptor_total = 0;
        uint64_t owner_first_service_elide_total = 0;
        uint64_t owner_first_service_elide_idx2row_total = 0;
        uint64_t owner_first_service_elide_rowindex_total = 0;
        uint64_t owner_first_service_elide_rowdescriptor_total = 0;
        uint64_t attempted_total = 0;
        uint64_t attempted_guard_total = 0;
        uint64_t attempted_reject_total = 0;
        uint64_t attempted_useful_total = 0;
        uint64_t attempted_base_total = 0;
        uint64_t attempted_band_total = 0;
        uint64_t attempted_other_total = 0;
        uint64_t attempted_idx2row_total = 0;
        uint64_t attempted_rowindex_total = 0;
        uint64_t attempted_rowdescriptor_total = 0;
        uint64_t duplicate_metadata_replay_elided_total = 0;
        uint64_t duplicate_metadata_issue_elided_total = 0;
        uint64_t fallback_private_issue_total = 0;

        void refreshAttemptedView() {
            attempted_guard_total = guard_drop_total;
            attempted_reject_total = owner_reject_total + join_reject_total;
            attempted_useful_total = useful_total;
            attempted_base_total = guard_base_total + reject_base_total + useful_base_total;
            attempted_band_total = guard_band_total + reject_band_total + useful_band_total;
            attempted_other_total = guard_other_total + reject_other_total + useful_other_total;
            attempted_idx2row_total =
                guard_idx2row_total + reject_idx2row_total + useful_idx2row_total;
            attempted_rowindex_total =
                guard_rowindex_total + reject_rowindex_total + useful_rowindex_total;
            attempted_rowdescriptor_total =
                guard_rowdescriptor_total + reject_rowdescriptor_total + useful_rowdescriptor_total;
            attempted_total =
                attempted_guard_total + attempted_reject_total + attempted_useful_total;
        }
    };
    PeInternalPodStats peInternalPodStats() const {
        PeInternalPodStats s{};
        s.guard_drop_total = pe_internal_pod_guard_drop_total_;
        s.guard_disabled_total = pe_internal_pod_guard_disabled_total_;
        s.guard_missing_metadata_plane_total =
            pe_internal_pod_guard_missing_metadata_plane_total_;
        s.guard_missing_owner_table_total =
            pe_internal_pod_guard_missing_owner_table_total_;
        s.guard_zero_pod_count_total =
            pe_internal_pod_guard_zero_pod_count_total_;
        s.guard_window_zero_total = pe_internal_pod_guard_window_zero_total_;
        s.guard_invalid_cfg_pod_total =
            pe_internal_pod_guard_invalid_cfg_pod_total_;
        s.guard_rowdescriptor_disabled_total =
            pe_internal_pod_guard_rowdescriptor_disabled_total_;
        s.guard_rowdescriptor_missing_metadata_plane_total =
            pe_internal_pod_guard_rowdescriptor_missing_metadata_plane_total_;
        s.guard_rowdescriptor_missing_owner_table_total =
            pe_internal_pod_guard_rowdescriptor_missing_owner_table_total_;
        s.guard_rowdescriptor_zero_pod_count_total =
            pe_internal_pod_guard_rowdescriptor_zero_pod_count_total_;
        s.guard_rowdescriptor_window_zero_total =
            pe_internal_pod_guard_rowdescriptor_window_zero_total_;
        s.guard_rowdescriptor_invalid_cfg_pod_total =
            pe_internal_pod_guard_rowdescriptor_invalid_cfg_pod_total_;
        s.guard_base_total = pe_internal_pod_guard_base_total_;
        s.guard_band_total = pe_internal_pod_guard_band_total_;
        s.guard_other_total = pe_internal_pod_guard_other_total_;
        s.guard_idx2row_total = pe_internal_pod_guard_idx2row_total_;
        s.guard_rowindex_total = pe_internal_pod_guard_rowindex_total_;
        s.guard_rowdescriptor_total = pe_internal_pod_guard_rowdescriptor_total_;
        s.frontier_export_total = pe_internal_pod_frontier_export_total_;
        s.frontier_consumer_count_sum_total =
            pe_internal_pod_frontier_consumer_count_sum_total_;
        s.frontier_overlap_strength_sum_total =
            pe_internal_pod_frontier_overlap_strength_sum_total_;
        s.frontier_base_consumer_count_sum_total =
            pe_internal_pod_frontier_base_consumer_count_sum_total_;
        s.frontier_base_overlap_strength_sum_total =
            pe_internal_pod_frontier_base_overlap_strength_sum_total_;
        s.frontier_band_consumer_count_sum_total =
            pe_internal_pod_frontier_band_consumer_count_sum_total_;
        s.frontier_band_overlap_strength_sum_total =
            pe_internal_pod_frontier_band_overlap_strength_sum_total_;
        s.owner_lookup_total = pe_internal_pod_owner_lookup_total_;
        s.owner_alloc_total = pe_internal_pod_owner_alloc_total_;
        s.owner_alloc_idx2row_total = pe_internal_pod_owner_alloc_idx2row_total_;
        s.owner_alloc_rowindex_total = pe_internal_pod_owner_alloc_rowindex_total_;
        s.owner_alloc_rowdescriptor_total = pe_internal_pod_owner_alloc_rowdescriptor_total_;
        s.owner_hit_total = pe_internal_pod_owner_hit_total_;
        s.owner_hit_idx2row_total = pe_internal_pod_owner_hit_idx2row_total_;
        s.owner_hit_rowindex_total = pe_internal_pod_owner_hit_rowindex_total_;
        s.owner_hit_rowdescriptor_total = pe_internal_pod_owner_hit_rowdescriptor_total_;
        s.owner_reject_total = pe_internal_pod_owner_reject_total_;
        s.owner_disabled_reject_total =
            pe_internal_pod_owner_disabled_reject_total_;
        s.owner_invalid_pod_reject_total =
            pe_internal_pod_owner_invalid_pod_reject_total_;
        s.owner_table_full_reject_total =
            pe_internal_pod_owner_table_full_reject_total_;
        s.join_request_total = pe_internal_pod_join_request_total_;
        s.join_grant_total = pe_internal_pod_join_grant_total_;
        s.join_reject_total = pe_internal_pod_join_reject_total_;
        s.join_table_disabled_reject_total =
            pe_internal_pod_join_table_disabled_reject_total_;
        s.join_duplicate_consumer_reject_total =
            pe_internal_pod_join_duplicate_consumer_reject_total_;
        s.join_table_full_reject_total =
            pe_internal_pod_join_table_full_reject_total_;
        s.join_before_private_issue_total =
            pe_internal_pod_join_before_private_issue_total_;
        s.owner_first_issue_deferred_total =
            pe_internal_pod_owner_first_issue_deferred_total_;
        s.owner_first_issue_deferred_idx2row_total =
            pe_internal_pod_owner_first_issue_deferred_idx2row_total_;
        s.owner_first_issue_deferred_rowindex_total =
            pe_internal_pod_owner_first_issue_deferred_rowindex_total_;
        s.owner_first_issue_deferred_rowdescriptor_total =
            pe_internal_pod_owner_first_issue_deferred_rowdescriptor_total_;
        s.owner_first_private_issue_avoided_total =
            pe_internal_pod_owner_first_private_issue_avoided_total_;
        s.owner_first_private_issue_avoided_idx2row_total =
            pe_internal_pod_owner_first_private_issue_avoided_idx2row_total_;
        s.owner_first_private_issue_avoided_rowindex_total =
            pe_internal_pod_owner_first_private_issue_avoided_rowindex_total_;
        s.owner_first_private_issue_avoided_rowdescriptor_total =
            pe_internal_pod_owner_first_private_issue_avoided_rowdescriptor_total_;
        s.reject_base_total = pe_internal_pod_reject_base_total_;
        s.reject_band_total = pe_internal_pod_reject_band_total_;
        s.reject_other_total = pe_internal_pod_reject_other_total_;
        s.reject_idx2row_total = pe_internal_pod_reject_idx2row_total_;
        s.reject_rowindex_total = pe_internal_pod_reject_rowindex_total_;
        s.reject_rowdescriptor_total = pe_internal_pod_reject_rowdescriptor_total_;
        s.useful_total = pe_internal_pod_useful_total_;
        s.useful_join_grant_total =
            pe_internal_pod_useful_join_grant_total_;
        s.useful_duplicate_replay_elide_total =
            pe_internal_pod_useful_duplicate_replay_elide_total_;
        s.useful_base_total = pe_internal_pod_useful_base_total_;
        s.useful_band_total = pe_internal_pod_useful_band_total_;
        s.useful_other_total = pe_internal_pod_useful_other_total_;
        s.useful_idx2row_total = pe_internal_pod_useful_idx2row_total_;
        s.useful_rowindex_total = pe_internal_pod_useful_rowindex_total_;
        s.useful_rowdescriptor_total = pe_internal_pod_useful_rowdescriptor_total_;
        s.service_join_live_total = pe_internal_pod_service_join_live_total_;
        s.service_join_live_idx2row_total = pe_internal_pod_service_join_live_idx2row_total_;
        s.service_join_live_rowindex_total = pe_internal_pod_service_join_live_rowindex_total_;
        s.service_join_live_rowdescriptor_total =
            pe_internal_pod_service_join_live_rowdescriptor_total_;
        s.service_join_ready_total = pe_internal_pod_service_join_ready_total_;
        s.service_join_ready_idx2row_total = pe_internal_pod_service_join_ready_idx2row_total_;
        s.service_join_ready_rowindex_total =
            pe_internal_pod_service_join_ready_rowindex_total_;
        s.service_join_ready_rowdescriptor_total =
            pe_internal_pod_service_join_ready_rowdescriptor_total_;
        s.service_ready_transition_total =
            pe_internal_pod_service_ready_transition_total_;
        s.service_ready_transition_idx2row_total =
            pe_internal_pod_service_ready_transition_idx2row_total_;
        s.service_ready_transition_rowindex_total =
            pe_internal_pod_service_ready_transition_rowindex_total_;
        s.service_ready_transition_rowdescriptor_total =
            pe_internal_pod_service_ready_transition_rowdescriptor_total_;
        s.service_ready_fanout_total =
            pe_internal_pod_service_ready_fanout_total_;
        s.service_ready_fanout_idx2row_total =
            pe_internal_pod_service_ready_fanout_idx2row_total_;
        s.service_ready_fanout_rowindex_total =
            pe_internal_pod_service_ready_fanout_rowindex_total_;
        s.service_ready_fanout_rowdescriptor_total =
            pe_internal_pod_service_ready_fanout_rowdescriptor_total_;
        s.service_ready_fanout_consumers_sum_total =
            pe_internal_pod_service_ready_fanout_consumers_sum_total_;
        s.service_ready_fanout_consumers_sum_idx2row_total =
            pe_internal_pod_service_ready_fanout_consumers_sum_idx2row_total_;
        s.service_ready_fanout_consumers_sum_rowindex_total =
            pe_internal_pod_service_ready_fanout_consumers_sum_rowindex_total_;
        s.service_ready_fanout_consumers_sum_rowdescriptor_total =
            pe_internal_pod_service_ready_fanout_consumers_sum_rowdescriptor_total_;
        s.service_release_deferred_total =
            pe_internal_pod_service_release_deferred_total_;
        s.service_release_deferred_idx2row_total =
            pe_internal_pod_service_release_deferred_idx2row_total_;
        s.service_release_deferred_rowindex_total =
            pe_internal_pod_service_release_deferred_rowindex_total_;
        s.service_release_deferred_rowdescriptor_total =
            pe_internal_pod_service_release_deferred_rowdescriptor_total_;
        s.service_ready_release_total =
            pe_internal_pod_service_ready_release_total_;
        s.service_ready_release_idx2row_total =
            pe_internal_pod_service_ready_release_idx2row_total_;
        s.service_ready_release_rowindex_total =
            pe_internal_pod_service_ready_release_rowindex_total_;
        s.service_ready_release_rowdescriptor_total =
            pe_internal_pod_service_ready_release_rowdescriptor_total_;
        s.service_late_join_total = pe_internal_pod_service_late_join_total_;
        s.service_late_join_idx2row_total = pe_internal_pod_service_late_join_idx2row_total_;
        s.service_late_join_rowindex_total =
            pe_internal_pod_service_late_join_rowindex_total_;
        s.service_late_join_rowdescriptor_total =
            pe_internal_pod_service_late_join_rowdescriptor_total_;
        s.service_potential_private_service_elide_total =
            pe_internal_pod_service_potential_private_service_elide_total_;
        s.service_potential_private_service_elide_idx2row_total =
            pe_internal_pod_service_potential_private_service_elide_idx2row_total_;
        s.service_potential_private_service_elide_rowindex_total =
            pe_internal_pod_service_potential_private_service_elide_rowindex_total_;
        s.service_potential_private_service_elide_rowdescriptor_total =
            pe_internal_pod_service_potential_private_service_elide_rowdescriptor_total_;
        s.owner_first_service_elide_total =
            pe_internal_pod_owner_first_service_elide_total_;
        s.owner_first_service_elide_idx2row_total =
            pe_internal_pod_owner_first_service_elide_idx2row_total_;
        s.owner_first_service_elide_rowindex_total =
            pe_internal_pod_owner_first_service_elide_rowindex_total_;
        s.owner_first_service_elide_rowdescriptor_total =
            pe_internal_pod_owner_first_service_elide_rowdescriptor_total_;
        s.duplicate_metadata_replay_elided_total =
            pe_internal_pod_duplicate_metadata_replay_elided_total_;
        s.duplicate_metadata_issue_elided_total =
            pe_internal_pod_duplicate_metadata_issue_elided_total_;
        s.fallback_private_issue_total =
            pe_internal_pod_fallback_private_issue_total_;
        s.refreshAttemptedView();
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
        uint64_t enforced_stall_cycles_total = 0;
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
        s.enforced_stall_cycles_total = weight_sram_stall_cycles_total_;
        return s;
    }
    struct RetireObservabilityStats {
        uint64_t global_hol_cycles_total = 0;
        uint64_t ready_but_blocked_edges_total = 0;
        uint64_t per_post_progress_total = 0;
        uint64_t wait_cycles_total = 0;
        uint64_t wait_cycles_due_to_hol_total = 0;
        uint64_t wait_cycles_due_to_barrier_total = 0;
        uint64_t wait_cycles_due_to_not_ready_total = 0;
        uint64_t samepost_blocked_edges_total = 0;
        uint64_t crosspost_blocked_edges_total = 0;
        uint64_t policy_loss_cycles_total = 0;
        uint64_t policy_loss_edges_total = 0;
        uint64_t head_hol_cycles_dense_total = 0;
        uint64_t head_hol_cycles_cache_total = 0;
        uint64_t head_hol_cycles_miss_total = 0;
        uint64_t head_hol_cycles_bcsr_total = 0;
        uint64_t head_hol_cycles_bcsr_file_total = 0;
        uint64_t head_blocked_edges_dense_total = 0;
        uint64_t head_blocked_edges_cache_total = 0;
        uint64_t head_blocked_edges_miss_total = 0;
        uint64_t head_blocked_edges_bcsr_total = 0;
        uint64_t head_blocked_edges_bcsr_file_total = 0;
        uint64_t shadow_per_post_recoverable_cycles_total = 0;
        uint64_t shadow_per_post_recoverable_edges_total = 0;
        uint64_t shadow_per_post_ready_posts_peak = 0;
        uint64_t shadow_per_post_committable_edges_peak = 0;
        uint64_t begin_apply_windows_total = 0;
        uint64_t begin_apply_prev_edges_total = 0;
        uint64_t begin_apply_outstanding_carryin_total = 0;
        uint64_t begin_apply_outstanding_carryin_windows_total = 0;
        uint64_t begin_apply_loader_not_ready_windows_total = 0;
        uint64_t edge_retire_registered_total = 0;
        uint64_t edge_retire_retired_total = 0;
        uint64_t end_scatter_pending_direct_reads_residual_total = 0;
        uint64_t end_scatter_outstanding_residual_total = 0;
        uint64_t end_scatter_residual_work_windows_total = 0;
        uint64_t ready_queue_peak = 0;
        uint64_t unblock_events_total = 0;
    };
    RetireObservabilityStats retireObservabilityStats() const {
        RetireObservabilityStats s{};
        s.global_hol_cycles_total = retire_global_hol_cycles_total_;
        s.ready_but_blocked_edges_total = retire_ready_but_blocked_edges_total_;
        s.per_post_progress_total = retire_per_post_progress_total_;
        s.wait_cycles_total = retire_wait_cycles_total_;
        s.wait_cycles_due_to_hol_total = retire_wait_cycles_due_to_hol_total_;
        s.wait_cycles_due_to_barrier_total = retire_wait_cycles_due_to_barrier_total_;
        s.wait_cycles_due_to_not_ready_total = retire_wait_cycles_due_to_not_ready_total_;
        s.samepost_blocked_edges_total = retire_samepost_blocked_edges_total_;
        s.crosspost_blocked_edges_total = retire_crosspost_blocked_edges_total_;
        s.policy_loss_cycles_total = retire_policy_loss_cycles_total_;
        s.policy_loss_edges_total = retire_policy_loss_edges_total_;
        s.head_hol_cycles_dense_total = retire_head_hol_cycles_dense_total_;
        s.head_hol_cycles_cache_total = retire_head_hol_cycles_cache_total_;
        s.head_hol_cycles_miss_total = retire_head_hol_cycles_miss_total_;
        s.head_hol_cycles_bcsr_total = retire_head_hol_cycles_bcsr_total_;
        s.head_hol_cycles_bcsr_file_total = retire_head_hol_cycles_bcsr_file_total_;
        s.head_blocked_edges_dense_total = retire_head_blocked_edges_dense_total_;
        s.head_blocked_edges_cache_total = retire_head_blocked_edges_cache_total_;
        s.head_blocked_edges_miss_total = retire_head_blocked_edges_miss_total_;
        s.head_blocked_edges_bcsr_total = retire_head_blocked_edges_bcsr_total_;
        s.head_blocked_edges_bcsr_file_total = retire_head_blocked_edges_bcsr_file_total_;
        s.begin_apply_windows_total = retire_begin_apply_windows_total_;
        s.begin_apply_prev_edges_total = retire_begin_apply_prev_edges_total_;
        s.begin_apply_outstanding_carryin_total = retire_begin_apply_outstanding_carryin_total_;
        s.begin_apply_outstanding_carryin_windows_total = retire_begin_apply_outstanding_carryin_windows_total_;
        s.begin_apply_loader_not_ready_windows_total = retire_begin_apply_loader_not_ready_windows_total_;
        s.edge_retire_registered_total = retire_edge_retire_registered_total_;
        s.edge_retire_retired_total = retire_edge_retire_retired_total_;
        s.end_scatter_pending_direct_reads_residual_total =
            retire_end_scatter_pending_direct_reads_residual_total_;
        s.end_scatter_outstanding_residual_total =
            retire_end_scatter_outstanding_residual_total_;
        s.end_scatter_residual_work_windows_total =
            retire_end_scatter_residual_work_windows_total_;
        s.ready_queue_peak = retire_ready_queue_peak_;
        s.unblock_events_total = retire_unblock_events_total_;
        return s;
    }
    void flushSramObservability(uint64_t now_cycle) {
        if (!weight_sram_enable_) return;
        idx_sram_model_.onClockTick(now_cycle + 1);
        l0_sram_model_.onClockTick(now_cycle + 1);
        if (shared_weight_object_plane_) {
            shared_weight_object_plane_->onClockTick(now_cycle + 1);
        }
    }
    void configure(CacheTryFn cache_try_fn,
                   CachePutFn cache_put_fn) {
        cache_try_fn_ = std::move(cache_try_fn);
        cache_put_fn_ = std::move(cache_put_fn);
    }

    void configureOrchestrator(OrchestratorConfig cfg) {
        orch_ = std::move(cfg);
        pre_touch_order_window_.clear();
        pre_touch_rank_window_.clear();
        pe_internal_pod_guard_drop_total_ = 0;
        pe_internal_pod_guard_disabled_total_ = 0;
        pe_internal_pod_guard_missing_metadata_plane_total_ = 0;
        pe_internal_pod_guard_missing_owner_table_total_ = 0;
        pe_internal_pod_guard_zero_pod_count_total_ = 0;
        pe_internal_pod_guard_window_zero_total_ = 0;
        pe_internal_pod_guard_invalid_cfg_pod_total_ = 0;
        pe_internal_pod_guard_rowdescriptor_disabled_total_ = 0;
        pe_internal_pod_guard_rowdescriptor_missing_metadata_plane_total_ = 0;
        pe_internal_pod_guard_rowdescriptor_missing_owner_table_total_ = 0;
        pe_internal_pod_guard_rowdescriptor_zero_pod_count_total_ = 0;
        pe_internal_pod_guard_rowdescriptor_window_zero_total_ = 0;
        pe_internal_pod_guard_rowdescriptor_invalid_cfg_pod_total_ = 0;
        pe_internal_pod_guard_base_total_ = 0;
        pe_internal_pod_guard_band_total_ = 0;
        pe_internal_pod_guard_other_total_ = 0;
        pe_internal_pod_guard_idx2row_total_ = 0;
        pe_internal_pod_guard_rowindex_total_ = 0;
        pe_internal_pod_guard_rowdescriptor_total_ = 0;
        pe_internal_pod_frontier_export_total_ = 0;
        pe_internal_pod_frontier_consumer_count_sum_total_ = 0;
        pe_internal_pod_frontier_overlap_strength_sum_total_ = 0;
        pe_internal_pod_frontier_base_consumer_count_sum_total_ = 0;
        pe_internal_pod_frontier_base_overlap_strength_sum_total_ = 0;
        pe_internal_pod_frontier_band_consumer_count_sum_total_ = 0;
        pe_internal_pod_frontier_band_overlap_strength_sum_total_ = 0;
        pe_internal_pod_owner_lookup_total_ = 0;
        pe_internal_pod_owner_alloc_total_ = 0;
        pe_internal_pod_owner_alloc_idx2row_total_ = 0;
        pe_internal_pod_owner_alloc_rowindex_total_ = 0;
        pe_internal_pod_owner_alloc_rowdescriptor_total_ = 0;
        pe_internal_pod_owner_hit_total_ = 0;
        pe_internal_pod_owner_hit_idx2row_total_ = 0;
        pe_internal_pod_owner_hit_rowindex_total_ = 0;
        pe_internal_pod_owner_hit_rowdescriptor_total_ = 0;
        pe_internal_pod_owner_reject_total_ = 0;
        pe_internal_pod_owner_disabled_reject_total_ = 0;
        pe_internal_pod_owner_invalid_pod_reject_total_ = 0;
        pe_internal_pod_owner_table_full_reject_total_ = 0;
        pe_internal_pod_join_request_total_ = 0;
        pe_internal_pod_join_grant_total_ = 0;
        pe_internal_pod_join_reject_total_ = 0;
        pe_internal_pod_join_table_disabled_reject_total_ = 0;
        pe_internal_pod_join_duplicate_consumer_reject_total_ = 0;
        pe_internal_pod_join_table_full_reject_total_ = 0;
        pe_internal_pod_join_before_private_issue_total_ = 0;
        pe_internal_pod_owner_first_issue_deferred_total_ = 0;
        pe_internal_pod_owner_first_issue_deferred_idx2row_total_ = 0;
        pe_internal_pod_owner_first_issue_deferred_rowindex_total_ = 0;
        pe_internal_pod_owner_first_issue_deferred_rowdescriptor_total_ = 0;
        pe_internal_pod_owner_first_private_issue_avoided_total_ = 0;
        pe_internal_pod_owner_first_private_issue_avoided_idx2row_total_ = 0;
        pe_internal_pod_owner_first_private_issue_avoided_rowindex_total_ = 0;
        pe_internal_pod_owner_first_private_issue_avoided_rowdescriptor_total_ = 0;
        pe_internal_pod_reject_base_total_ = 0;
        pe_internal_pod_reject_band_total_ = 0;
        pe_internal_pod_reject_other_total_ = 0;
        pe_internal_pod_reject_idx2row_total_ = 0;
        pe_internal_pod_reject_rowindex_total_ = 0;
        pe_internal_pod_reject_rowdescriptor_total_ = 0;
        pe_internal_pod_useful_total_ = 0;
        pe_internal_pod_useful_join_grant_total_ = 0;
        pe_internal_pod_useful_duplicate_replay_elide_total_ = 0;
        pe_internal_pod_useful_base_total_ = 0;
        pe_internal_pod_useful_band_total_ = 0;
        pe_internal_pod_useful_other_total_ = 0;
        pe_internal_pod_useful_idx2row_total_ = 0;
        pe_internal_pod_useful_rowindex_total_ = 0;
        pe_internal_pod_useful_rowdescriptor_total_ = 0;
        pe_internal_pod_service_join_live_total_ = 0;
        pe_internal_pod_service_join_live_idx2row_total_ = 0;
        pe_internal_pod_service_join_live_rowindex_total_ = 0;
        pe_internal_pod_service_join_live_rowdescriptor_total_ = 0;
        pe_internal_pod_service_join_ready_total_ = 0;
        pe_internal_pod_service_join_ready_idx2row_total_ = 0;
        pe_internal_pod_service_join_ready_rowindex_total_ = 0;
        pe_internal_pod_service_join_ready_rowdescriptor_total_ = 0;
        pe_internal_pod_service_ready_transition_total_ = 0;
        pe_internal_pod_service_ready_transition_idx2row_total_ = 0;
        pe_internal_pod_service_ready_transition_rowindex_total_ = 0;
        pe_internal_pod_service_ready_transition_rowdescriptor_total_ = 0;
        pe_internal_pod_service_ready_fanout_total_ = 0;
        pe_internal_pod_service_ready_fanout_idx2row_total_ = 0;
        pe_internal_pod_service_ready_fanout_rowindex_total_ = 0;
        pe_internal_pod_service_ready_fanout_rowdescriptor_total_ = 0;
        pe_internal_pod_service_ready_fanout_consumers_sum_total_ = 0;
        pe_internal_pod_service_ready_fanout_consumers_sum_idx2row_total_ = 0;
        pe_internal_pod_service_ready_fanout_consumers_sum_rowindex_total_ = 0;
        pe_internal_pod_service_ready_fanout_consumers_sum_rowdescriptor_total_ = 0;
        pe_internal_pod_service_release_deferred_total_ = 0;
        pe_internal_pod_service_release_deferred_idx2row_total_ = 0;
        pe_internal_pod_service_release_deferred_rowindex_total_ = 0;
        pe_internal_pod_service_release_deferred_rowdescriptor_total_ = 0;
        pe_internal_pod_service_ready_release_total_ = 0;
        pe_internal_pod_service_ready_release_idx2row_total_ = 0;
        pe_internal_pod_service_ready_release_rowindex_total_ = 0;
        pe_internal_pod_service_ready_release_rowdescriptor_total_ = 0;
        pe_internal_pod_service_released_idx2row_total_ = 0;
        pe_internal_pod_service_released_rowindex_total_ = 0;
        pe_internal_pod_service_released_rowdescriptor_total_ = 0;
        pe_internal_pod_service_release_missing_idx2row_total_ = 0;
        pe_internal_pod_service_release_missing_rowindex_total_ = 0;
        pe_internal_pod_service_release_missing_rowdescriptor_total_ = 0;
        pe_internal_pod_service_late_join_total_ = 0;
        pe_internal_pod_service_late_join_idx2row_total_ = 0;
        pe_internal_pod_service_late_join_rowindex_total_ = 0;
        pe_internal_pod_service_late_join_rowdescriptor_total_ = 0;
        pe_internal_pod_service_potential_private_service_elide_total_ = 0;
        pe_internal_pod_service_potential_private_service_elide_idx2row_total_ = 0;
        pe_internal_pod_service_potential_private_service_elide_rowindex_total_ = 0;
        pe_internal_pod_service_potential_private_service_elide_rowdescriptor_total_ = 0;
        pe_internal_pod_owner_first_service_elide_total_ = 0;
        pe_internal_pod_owner_first_service_elide_idx2row_total_ = 0;
        pe_internal_pod_owner_first_service_elide_rowindex_total_ = 0;
        pe_internal_pod_owner_first_service_elide_rowdescriptor_total_ = 0;
        pe_internal_pod_duplicate_metadata_replay_elided_total_ = 0;
        pe_internal_pod_duplicate_metadata_issue_elided_total_ = 0;
        pe_internal_pod_fallback_private_issue_total_ = 0;
        read_src_dense_reqs_ = 0;
        read_src_dense_bytes_ = 0;
        read_src_rowptr_reqs_ = 0;
        read_src_rowptr_bytes_ = 0;
        read_src_colidx_reqs_ = 0;
        read_src_colidx_bytes_ = 0;
        read_src_blockdata_reqs_ = 0;
        read_src_blockdata_bytes_ = 0;
        idx_lookup_total_ = 0;
        idx_lookup_idx2_total_ = 0;
        idx_lookup_legacy_total_ = 0;
        l0_lookup_total_ = 0;
        l0_hit_total_ = 0;
        l0_miss_total_ = 0;
        l0_fill_total_ = 0;
        l0_evict_total_ = 0;
        weight_sram_stall_cycles_total_ = 0;
        retire_global_hol_cycles_total_ = 0;
        retire_ready_but_blocked_edges_total_ = 0;
        retire_per_post_progress_total_ = 0;
        retire_wait_cycles_total_ = 0;
        retire_wait_cycles_due_to_hol_total_ = 0;
        retire_wait_cycles_due_to_barrier_total_ = 0;
        retire_wait_cycles_due_to_not_ready_total_ = 0;
        retire_samepost_blocked_edges_total_ = 0;
        retire_crosspost_blocked_edges_total_ = 0;
        retire_policy_loss_cycles_total_ = 0;
        retire_policy_loss_edges_total_ = 0;
        retire_head_hol_cycles_dense_total_ = 0;
        retire_head_hol_cycles_cache_total_ = 0;
        retire_head_hol_cycles_miss_total_ = 0;
        retire_head_hol_cycles_bcsr_total_ = 0;
        retire_head_hol_cycles_bcsr_file_total_ = 0;
        retire_head_blocked_edges_dense_total_ = 0;
        retire_head_blocked_edges_cache_total_ = 0;
        retire_head_blocked_edges_miss_total_ = 0;
        retire_head_blocked_edges_bcsr_total_ = 0;
        retire_head_blocked_edges_bcsr_file_total_ = 0;
        retire_begin_apply_windows_total_ = 0;
        retire_begin_apply_prev_edges_total_ = 0;
        retire_begin_apply_outstanding_carryin_total_ = 0;
        retire_begin_apply_outstanding_carryin_windows_total_ = 0;
        retire_begin_apply_loader_not_ready_windows_total_ = 0;
        retire_edge_retire_registered_total_ = 0;
        retire_edge_retire_retired_total_ = 0;
        retire_end_scatter_pending_direct_reads_residual_total_ = 0;
        retire_end_scatter_outstanding_residual_total_ = 0;
        retire_end_scatter_residual_work_windows_total_ = 0;
        retire_ready_queue_peak_ = 0;
        retire_unblock_events_total_ = 0;
        ready_uncommitted_edges_ = 0;
        retire_waiting_prev_ = false;
        retire_begin_apply_loader_not_ready_marked_ = false;
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
        ready_uncommitted_by_post_.clear();
        ready_uncommitted_touched_.clear();
        ready_uncommitted_touched_list_.clear();
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
        if (idx_cfg.enable) {
            idx_sram_model_.configure(idx_cfg);
        } else {
            idx_sram_model_.disable();
        }
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
        if (l0_cfg.enable) {
            l0_sram_model_.configure(l0_cfg);
        } else {
            l0_sram_model_.disable();
        }
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
        // Normal BCSR reads come from simulated memory, but the bytes written
        // by WeightLoader must still identify the same source as routing.
        assertBcsrSourceContract_();
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
        // Re-sync the software inflight counter at each Apply window boundary.
        // Older implementations zeroed outstanding here; the current code keeps
        // it across windows so long-latency prefetches can be reflected.
        // However, strict GAS step boundaries assume the next Apply starts from
        // the actual memory pending set. If the software counter drifts above
        // the real mem_access_ pending size, later windows can deadlock with
        // canIssueMoreReads()==false while no request is actually in flight.
        const size_t pending_now = pendingSize();
        window_.outstanding = static_cast<uint32_t>(
            std::min<size_t>(pending_now, std::numeric_limits<uint32_t>::max()));
        window_.peak_outstanding = window_.outstanding;
    }
    bool canIssue(uint32_t n = 1, bool count_budget = true) const {
        const bool budget_ok =
            (!count_budget) ||
            (window_seq_ == 0) ||
            (window_.budget == 0) ||
            (window_.issued + n <= window_.budget);
        const bool ostd_ok =
            (window_.max_outstanding == 0u) ||
            (window_.outstanding + n <= window_.max_outstanding);
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
        pre_touch_rank_window_.reserve(256);
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
        withExperimentalWindowStateSerialization_([&]() {
            if (!window_read_enable) return;
            ensureWindowTracking(num_neurons);
            posts_seen_prev_window_ = posts_seen_window_;
            std::fill(posts_seen_window_.begin(), posts_seen_window_.end(), 0);
            posts_list_prev_window_.swap(posts_list_window_);
            posts_list_window_.clear();
            active_pre_prev_window_ = std::move(active_pre_window_);
            active_pre_window_.clear();
            pre_touch_order_window_.clear();
            pre_touch_rank_window_.clear();
        });
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
        auto inserted = active_pre_window_.insert(pre_global);
        if (!inserted.second) return;
        const uint32_t touch_rank =
            static_cast<uint32_t>(pre_touch_order_window_.size());
        pre_touch_order_window_.push_back(pre_global);
        pre_touch_rank_window_[pre_global] = touch_rank;
    }

    void noteWindowTouch(uint32_t post_local, uint32_t pre_global, uint32_t num_neurons) {
        withExperimentalWindowStateSerialization_([&]() {
            notePostLocal(post_local, num_neurons);
            notePreGlobal(pre_global);
        });
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
        withExperimentalWindowStateSerialization_([&]() {
            const uint64_t key =
                (static_cast<uint64_t>(post_local) << 32) |
                static_cast<uint64_t>(pre_global);
            edge_collector_.curr[key] += 1;
        });
    }
    void recordEdgeWithWeight(uint32_t post_local, uint32_t pre_global, float weight) {
        withExperimentalWindowStateSerialization_([&]() {
            (void)weight;
            const uint64_t key =
                (static_cast<uint64_t>(post_local) << 32) |
                static_cast<uint64_t>(pre_global);
            edge_collector_.curr[key] += 1;
        });
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
        withExperimentalWindowStateSerialization_([&]() {
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
            retire_begin_apply_windows_total_ += 1;
            retire_begin_apply_prev_edges_total_ += static_cast<uint64_t>(edgesPrevSize());
            if (window_.outstanding > 0) {
                retire_begin_apply_outstanding_carryin_total_ +=
                    static_cast<uint64_t>(window_.outstanding);
                retire_begin_apply_outstanding_carryin_windows_total_ += 1;
            }
            retire_begin_apply_loader_not_ready_marked_ = false;
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
            // gather prefetch path is intentionally not part of the runtime.
            diag_node_id_ = static_cast<int>(orch_.node_id);
        });
    }

    // 仅用于调试：在 EndScatter 或仿真结束前输出当前窗口的权重读摘要。
    // 该函数无行为副作用；仅打印并清空本地诊断计数器。
    void endScatterWindow(uint32_t seq) {
        withExperimentalWindowStateSerialization_([&]() {
            const uint64_t pending_direct_reads_residual =
                static_cast<uint64_t>(pending_direct_reads_.size());
            const uint64_t outstanding_residual =
                static_cast<uint64_t>(window_.outstanding);
            retire_end_scatter_pending_direct_reads_residual_total_ +=
                pending_direct_reads_residual;
            retire_end_scatter_outstanding_residual_total_ += outstanding_residual;
            if (pending_direct_reads_residual > 0 ||
                outstanding_residual > 0 ||
                retired_edges_count_ < edge_retire_.size()) {
                retire_end_scatter_residual_work_windows_total_ += 1;
            }
            // Always perform non-diagnostic housekeeping (applies to all cores, not only debug targets).
            maybeAutoTuneBlockCache_();
            window_seq_ = 0;

            emitByteExactPassMarker_("EndScatter", seq);
            // When semantic verification is enabled, emit a PASS marker once we have verified enough edges.
            if (orch_.bcsr_semantic_verify_enable &&
                !bcsr_sem_pass_logged_ &&
                (bcsr_sem_verified_edges_ >=
                 static_cast<uint64_t>(orch_.bcsr_semantic_verify_max_edges))) {
                emitBcsrSemanticVerifyMarker_("EndScatter", seq);
            }

            if (!diag_debug_ || !diag_out_) return;
            if (!diag_window_active_) return;
            if (diag_window_seq_ != 0 && seq != diag_window_seq_) return;
            dumpWindowDiagSummary_("[diag-window-weights] EndScatter");
            resetWindowDiag_();
            diag_window_active_ = false;
            diag_window_seq_ = 0;
        });
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
    void assertBcsrSourceContract_();

    void issueFromEdgesOnce_() {
        if (!orch_.accessor) return;
        tryRetireEdges_();
        if (orch_.ensure_loader_ready && !orch_.ensure_loader_ready()) {
            if (!retire_begin_apply_loader_not_ready_marked_) {
                retire_begin_apply_loader_not_ready_windows_total_ += 1;
                retire_begin_apply_loader_not_ready_marked_ = true;
            }
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
            setEdgeRetirePendingSrc_(seq, EdgeSrc::Miss);
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
    template <typename Fn>
    auto withExperimentalWindowStateSerialization_(Fn&& fn)
        -> decltype(fn()) {
        return fn();
    }

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
    std::unordered_map<uint32_t, uint32_t> pre_touch_rank_window_;
    GasEdgeCollector edge_collector_;

    // Orchestrator config
    OrchestratorConfig orch_{};

    // Dense layout (derived). Cached to avoid recomputing row_stride/packing per request.
    bool dense_phys_enable_ = false;
    uint32_t dense_cols_effective_ = 0;
    DensePhysV1Derived dense_phys_{};

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
    std::vector<uint64_t> ready_uncommitted_by_post_{};
    std::vector<uint8_t> ready_uncommitted_touched_{};
    std::vector<uint32_t> ready_uncommitted_touched_list_{};

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
    uint64_t retire_wait_cycles_total_ = 0;
    uint64_t retire_wait_cycles_due_to_hol_total_ = 0;
    uint64_t retire_wait_cycles_due_to_barrier_total_ = 0;
    uint64_t retire_wait_cycles_due_to_not_ready_total_ = 0;
    uint64_t retire_samepost_blocked_edges_total_ = 0;
    uint64_t retire_crosspost_blocked_edges_total_ = 0;
    uint64_t retire_policy_loss_cycles_total_ = 0;
    uint64_t retire_policy_loss_edges_total_ = 0;
    uint64_t retire_head_hol_cycles_dense_total_ = 0;
    uint64_t retire_head_hol_cycles_cache_total_ = 0;
    uint64_t retire_head_hol_cycles_miss_total_ = 0;
    uint64_t retire_head_hol_cycles_bcsr_total_ = 0;
    uint64_t retire_head_hol_cycles_bcsr_file_total_ = 0;
    uint64_t retire_head_blocked_edges_dense_total_ = 0;
    uint64_t retire_head_blocked_edges_cache_total_ = 0;
    uint64_t retire_head_blocked_edges_miss_total_ = 0;
    uint64_t retire_head_blocked_edges_bcsr_total_ = 0;
    uint64_t retire_head_blocked_edges_bcsr_file_total_ = 0;
    uint64_t retire_begin_apply_windows_total_ = 0;
    uint64_t retire_begin_apply_prev_edges_total_ = 0;
    uint64_t retire_begin_apply_outstanding_carryin_total_ = 0;
    uint64_t retire_begin_apply_outstanding_carryin_windows_total_ = 0;
    uint64_t retire_begin_apply_loader_not_ready_windows_total_ = 0;
    uint64_t retire_edge_retire_registered_total_ = 0;
    uint64_t retire_edge_retire_retired_total_ = 0;
    uint64_t retire_end_scatter_pending_direct_reads_residual_total_ = 0;
    uint64_t retire_end_scatter_outstanding_residual_total_ = 0;
    uint64_t retire_end_scatter_residual_work_windows_total_ = 0;
    uint64_t retire_ready_queue_peak_ = 0;
    uint64_t retire_unblock_events_total_ = 0;
    bool retire_waiting_prev_ = false;
    bool retire_begin_apply_loader_not_ready_marked_ = false;

    bool issue_edges_in_progress_ = false;
    bool issue_edges_again_ = false;

    bool usePerPostRetire_() const { return retire_policy_ == RetirePolicy::PerPostDeterministic; }
    bool shadowPerPostRetireEnabled_() const {
        return orch_.experimental_retire_shadow_per_post_enable && !usePerPostRetire_();
    }
    bool needPerPostRetireStorage_() const {
        return usePerPostRetire_() || shadowPerPostRetireEnabled_();
    }
    // Retire releases are unbounded in the supported policy. The former
    uint32_t perPostRetireReleaseBudget_() const { return 0; }
    void ensureReadyUncommittedStorage_() {
        const size_t n = static_cast<size_t>(orch_.num_neurons);
        if (ready_uncommitted_by_post_.size() == n &&
            ready_uncommitted_touched_.size() == n) {
            return;
        }
        ready_uncommitted_by_post_.assign(n, 0);
        ready_uncommitted_touched_.assign(n, 0);
        ready_uncommitted_touched_list_.clear();
    }

    void resetReadyUncommittedByPost_() {
        ensureReadyUncommittedStorage_();
        for (uint32_t post : ready_uncommitted_touched_list_) {
            if (post >= ready_uncommitted_by_post_.size()) continue;
            ready_uncommitted_by_post_[post] = 0;
            ready_uncommitted_touched_[post] = 0;
        }
        ready_uncommitted_touched_list_.clear();
    }

    void noteReadyUncommittedInc_(uint32_t post_local) {
        ensureReadyUncommittedStorage_();
        if (post_local >= ready_uncommitted_by_post_.size()) return;
        if (ready_uncommitted_touched_[post_local] == 0) {
            ready_uncommitted_touched_[post_local] = 1;
            ready_uncommitted_touched_list_.push_back(post_local);
        }
        ready_uncommitted_by_post_[post_local] += 1;
    }

    void noteReadyUncommittedDec_(uint32_t post_local) {
        if (post_local >= ready_uncommitted_by_post_.size()) return;
        if (ready_uncommitted_by_post_[post_local] > 0) {
            ready_uncommitted_by_post_[post_local] -= 1;
        }
    }

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
        resetReadyUncommittedByPost_();
        edge_retire_.reserve(edgesPrevSize());
        per_post_ready_posts_.clear();
        if (needPerPostRetireStorage_()) {
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
        retire_edge_retire_registered_total_ += 1;
        if (needPerPostRetireStorage_() && post_local < static_cast<uint32_t>(orch_.num_neurons)) {
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

    void setEdgeRetirePendingSrc_(size_t seq, EdgeSrc src) {
        if (seq >= edge_retire_.size()) return;
        auto& e = edge_retire_[seq];
        if (!e.ready) {
            e.src = src;
        }
    }

    void setEdgeRetireIssued_(size_t seq) {
        (void)seq;
    }

    void noteRetireHeadHolBySrc_(EdgeSrc src, uint64_t blocked_edges) {
        switch (src) {
            case EdgeSrc::Dense:
                retire_head_hol_cycles_dense_total_ += 1;
                retire_head_blocked_edges_dense_total_ += blocked_edges;
                break;
            case EdgeSrc::Cache:
                retire_head_hol_cycles_cache_total_ += 1;
                retire_head_blocked_edges_cache_total_ += blocked_edges;
                break;
            case EdgeSrc::Miss:
                retire_head_hol_cycles_miss_total_ += 1;
                retire_head_blocked_edges_miss_total_ += blocked_edges;
                break;
            case EdgeSrc::BCSR:
                retire_head_hol_cycles_bcsr_total_ += 1;
                retire_head_blocked_edges_bcsr_total_ += blocked_edges;
                break;
            case EdgeSrc::BCSRFile:
                retire_head_hol_cycles_bcsr_file_total_ += 1;
                retire_head_blocked_edges_bcsr_file_total_ += blocked_edges;
                break;
        }
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
            noteReadyUncommittedInc_(e.post_local);
            if (ready_uncommitted_edges_ > retire_ready_queue_peak_) {
                retire_ready_queue_peak_ = ready_uncommitted_edges_;
            }
        }
        if (needPerPostRetireStorage_()) {
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
        const uint64_t retired_before = retired_edges_count_;
        const uint32_t release_budget = perPostRetireReleaseBudget_();
        uint32_t released = 0;
        while (!per_post_ready_posts_.empty() &&
               (release_budget == 0u || released < release_budget)) {
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
            noteReadyUncommittedDec_(e.post_local);
            retired_edges_count_++;
            retire_edge_retire_retired_total_ += 1;
            retire_per_post_progress_total_++;
            released += 1;
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
        if (retired_edges_count_ > retired_before && retire_waiting_prev_) {
            retire_unblock_events_total_ += 1;
            retire_waiting_prev_ = false;
        }
    }

    uint64_t shadowPerPostCommittableEdges_() const {
        if (!shadowPerPostRetireEnabled_()) return 0;
        uint64_t total = 0;
        for (uint32_t post : per_post_ready_posts_) {
            if (post >= per_post_retire_seq_.size() || post >= per_post_head_idx_.size()) continue;
            const auto& seqs = per_post_retire_seq_[post];
            size_t head_idx = per_post_head_idx_[post];
            while (head_idx < seqs.size()) {
                const size_t seq = seqs[head_idx];
                if (seq >= edge_retire_.size() || !edge_retire_[seq].ready) break;
                total += 1;
                head_idx += 1;
            }
        }
        return total;
    }

    void noteShadowRecoverableOnTick_() {
        return;
    }

    void drainShadowPerPostRetire_() {
        if (!shadowPerPostRetireEnabled_()) return;
        while (!per_post_ready_posts_.empty()) {
            auto it = per_post_ready_posts_.begin();
            const uint32_t post = *it;
            if (post >= per_post_retire_seq_.size() || post >= per_post_head_idx_.size()) {
                per_post_ready_posts_.erase(it);
                continue;
            }
            const auto& seqs = per_post_retire_seq_[post];
            size_t head_idx = per_post_head_idx_[post];
            if (head_idx >= seqs.size()) {
                per_post_ready_posts_.erase(it);
                continue;
            }
            const size_t seq = seqs[head_idx];
            if (seq >= edge_retire_.size() || !edge_retire_[seq].ready) {
                per_post_ready_posts_.erase(it);
                continue;
            }
            per_post_head_idx_[post] = head_idx + 1;
            head_idx = per_post_head_idx_[post];
            if (head_idx >= seqs.size()) {
                per_post_ready_posts_.erase(it);
                continue;
            }
            const size_t next_seq = seqs[head_idx];
            if (next_seq >= edge_retire_.size() || !edge_retire_[next_seq].ready) {
                per_post_ready_posts_.erase(it);
            }
        }
    }

    void updateRetireHolStatsOnTick_() {
        if (usePerPostRetire_() || edge_retire_.empty() ||
            next_retire_seq_ >= edge_retire_.size()) {
            return;
        }
        const auto& head = edge_retire_[next_retire_seq_];
        if (!head.ready) {
            retire_wait_cycles_total_ += 1;
            retire_waiting_prev_ = true;
            if (ready_uncommitted_edges_ > 0) {
                uint64_t same_post_ready = 0;
                if (head.post_local < ready_uncommitted_by_post_.size()) {
                    same_post_ready = ready_uncommitted_by_post_[head.post_local];
                }
                const uint64_t cross_post_ready =
                    (ready_uncommitted_edges_ > same_post_ready)
                        ? (ready_uncommitted_edges_ - same_post_ready)
                        : 0;
                retire_global_hol_cycles_total_ += 1;
                retire_ready_but_blocked_edges_total_ += ready_uncommitted_edges_;
                retire_wait_cycles_due_to_hol_total_ += 1;
                retire_samepost_blocked_edges_total_ += same_post_ready;
                retire_crosspost_blocked_edges_total_ += cross_post_ready;
                noteRetireHeadHolBySrc_(head.src, ready_uncommitted_edges_);
                if (cross_post_ready > 0) {
                    retire_policy_loss_cycles_total_ += 1;
                    retire_policy_loss_edges_total_ += cross_post_ready;
                }
            } else {
                retire_wait_cycles_due_to_not_ready_total_ += 1;
            }
        }
    }

    void tryRetireEdges_() {
        if (usePerPostRetire_()) {
            tryRetireEdgesPerPost_();
            return;
        }
        const size_t retire_seq_before = next_retire_seq_;
        while (next_retire_seq_ < edge_retire_.size()) {
            const auto& e = edge_retire_[next_retire_seq_];
            if (!e.ready) break;
            commitRetireEntry_(e);
            if (ready_uncommitted_edges_ > 0) ready_uncommitted_edges_--;
            noteReadyUncommittedDec_(e.post_local);
            retired_edges_count_++;
            retire_edge_retire_retired_total_ += 1;
            next_retire_seq_++;
        }
        if (next_retire_seq_ > retire_seq_before && retire_waiting_prev_) {
            retire_unblock_events_total_ += 1;
            retire_waiting_prev_ = false;
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
        size_t report_bytes = 0;
        bool is_row = false;
        uint32_t pre = 0;
        uint32_t post_start = 0;
        uint32_t count_floats = 0;
        bool is_weight = true;
        bool count_weight_read = true;
        bool counted_inflight = false;
        uint64_t issue_cycle = 0;
        uint64_t pending_enqueue_cycle = std::numeric_limits<uint64_t>::max();
        size_t retire_seq = std::numeric_limits<size_t>::max();
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
        bool has_single_cb = false;
        bool has_bytes_cb = false;
        uint32_t cb_post = 0;
        std::function<void(float)> single_cb;
        std::function<void(const std::vector<uint8_t>&)> bytes_cb;
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
    mutable std::mutex experimental_window_state_mutex_;
    uint64_t read_src_dense_reqs_ = 0;
    uint64_t read_src_dense_bytes_ = 0;
    uint64_t read_src_rowptr_reqs_ = 0;
    uint64_t read_src_rowptr_bytes_ = 0;
    uint64_t read_src_colidx_reqs_ = 0;
    uint64_t read_src_colidx_bytes_ = 0;
    uint64_t read_src_blockdata_reqs_ = 0;
    uint64_t read_src_blockdata_bytes_ = 0;

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
    uint32_t computeExperimentalIdx2IngressBudget_();

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
    void maybeAutoTuneBlockCache_();
    void bcsrPrefetchAll_();
    void bcsrPrefetchRowBlocks_(uint32_t block_row, const std::vector<uint32_t>& cols, uint32_t row_start);
    void bcsrPopulateWeightCache_(uint32_t block_row, uint32_t block_col, const std::vector<float>& blk);
    bool experimentalIdx2IngressPrefetchEnabled_() const;
    bool experimentalIdx2IngressAllowApplyCarry_() const;
    uint32_t experimentalIdx2IngressPrefetchMaxInflight_() const;
    bool shouldTailGuardIdx2RespDrop_() const;
    bool observeOnlyWeightSramStall_() const;
    bool peInternalPodShadowEnabled_() const;
    void observePeInternalPodMetadataObject_(PodMetadataObjectPlane::MetadataKind kind,
                                             uint64_t object_id);
    bool notePeInternalPodServiceObjectReady_(PodMetadataObjectPlane::MetadataKind kind,
                                              uint64_t object_id,
                                              uint32_t window_seq = 0u);
    void notePeInternalPodServiceObjectReleased_(PodMetadataObjectPlane::MetadataKind kind,
                                                 uint64_t object_id,
                                                 uint32_t window_seq = 0u,
                                                 bool defer_until_ready = false);
    void noteIdxSramReadMirror_(uint64_t addr, size_t bytes);
    void noteIdxSramResidentMirror_(uint64_t bytes);
    void noteL0SramReadMirror_(uint64_t addr);
    void noteL0SramWriteMirror_(uint64_t addr);
    void noteL0SramResidentMirror_(uint64_t bytes);
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
    PeWeightObjectPlane* shared_weight_object_plane_ = nullptr;
    PodMetadataObjectPlane* pod_metadata_object_plane_ = nullptr;
    PodOwnerServiceTable* pod_owner_service_table_ = nullptr;
    PeLocalServiceObjectTable* pe_local_service_object_table_ = nullptr;
    bool shared_weight_object_plane_residency_authority_ = false;
    uint64_t pe_internal_pod_guard_drop_total_ = 0;
    uint64_t pe_internal_pod_guard_disabled_total_ = 0;
    uint64_t pe_internal_pod_guard_missing_metadata_plane_total_ = 0;
    uint64_t pe_internal_pod_guard_missing_owner_table_total_ = 0;
    uint64_t pe_internal_pod_guard_zero_pod_count_total_ = 0;
    uint64_t pe_internal_pod_guard_window_zero_total_ = 0;
    uint64_t pe_internal_pod_guard_invalid_cfg_pod_total_ = 0;
    uint64_t pe_internal_pod_guard_rowdescriptor_disabled_total_ = 0;
    uint64_t pe_internal_pod_guard_rowdescriptor_missing_metadata_plane_total_ = 0;
    uint64_t pe_internal_pod_guard_rowdescriptor_missing_owner_table_total_ = 0;
    uint64_t pe_internal_pod_guard_rowdescriptor_zero_pod_count_total_ = 0;
    uint64_t pe_internal_pod_guard_rowdescriptor_window_zero_total_ = 0;
    uint64_t pe_internal_pod_guard_rowdescriptor_invalid_cfg_pod_total_ = 0;
    uint64_t pe_internal_pod_guard_base_total_ = 0;
    uint64_t pe_internal_pod_guard_band_total_ = 0;
    uint64_t pe_internal_pod_guard_other_total_ = 0;
    uint64_t pe_internal_pod_guard_idx2row_total_ = 0;
    uint64_t pe_internal_pod_guard_rowindex_total_ = 0;
    uint64_t pe_internal_pod_guard_rowdescriptor_total_ = 0;
    uint64_t pe_internal_pod_frontier_export_total_ = 0;
    uint64_t pe_internal_pod_frontier_consumer_count_sum_total_ = 0;
    uint64_t pe_internal_pod_frontier_overlap_strength_sum_total_ = 0;
    uint64_t pe_internal_pod_frontier_base_consumer_count_sum_total_ = 0;
    uint64_t pe_internal_pod_frontier_base_overlap_strength_sum_total_ = 0;
    uint64_t pe_internal_pod_frontier_band_consumer_count_sum_total_ = 0;
    uint64_t pe_internal_pod_frontier_band_overlap_strength_sum_total_ = 0;
    uint64_t pe_internal_pod_owner_lookup_total_ = 0;
    uint64_t pe_internal_pod_owner_alloc_total_ = 0;
    uint64_t pe_internal_pod_owner_alloc_idx2row_total_ = 0;
    uint64_t pe_internal_pod_owner_alloc_rowindex_total_ = 0;
    uint64_t pe_internal_pod_owner_alloc_rowdescriptor_total_ = 0;
    uint64_t pe_internal_pod_owner_hit_total_ = 0;
    uint64_t pe_internal_pod_owner_hit_idx2row_total_ = 0;
    uint64_t pe_internal_pod_owner_hit_rowindex_total_ = 0;
    uint64_t pe_internal_pod_owner_hit_rowdescriptor_total_ = 0;
    uint64_t pe_internal_pod_owner_reject_total_ = 0;
    uint64_t pe_internal_pod_owner_disabled_reject_total_ = 0;
    uint64_t pe_internal_pod_owner_invalid_pod_reject_total_ = 0;
    uint64_t pe_internal_pod_owner_table_full_reject_total_ = 0;
    uint64_t pe_internal_pod_join_request_total_ = 0;
    uint64_t pe_internal_pod_join_grant_total_ = 0;
    uint64_t pe_internal_pod_join_reject_total_ = 0;
    uint64_t pe_internal_pod_join_table_disabled_reject_total_ = 0;
    uint64_t pe_internal_pod_join_duplicate_consumer_reject_total_ = 0;
    uint64_t pe_internal_pod_join_table_full_reject_total_ = 0;
    uint64_t pe_internal_pod_join_before_private_issue_total_ = 0;
    uint64_t pe_internal_pod_owner_first_issue_deferred_total_ = 0;
    uint64_t pe_internal_pod_owner_first_issue_deferred_idx2row_total_ = 0;
    uint64_t pe_internal_pod_owner_first_issue_deferred_rowindex_total_ = 0;
    uint64_t pe_internal_pod_owner_first_issue_deferred_rowdescriptor_total_ = 0;
    uint64_t pe_internal_pod_owner_first_private_issue_avoided_total_ = 0;
    uint64_t pe_internal_pod_owner_first_private_issue_avoided_idx2row_total_ = 0;
    uint64_t pe_internal_pod_owner_first_private_issue_avoided_rowindex_total_ = 0;
    uint64_t pe_internal_pod_owner_first_private_issue_avoided_rowdescriptor_total_ = 0;
    uint64_t pe_internal_pod_reject_base_total_ = 0;
    uint64_t pe_internal_pod_reject_band_total_ = 0;
    uint64_t pe_internal_pod_reject_other_total_ = 0;
    uint64_t pe_internal_pod_reject_idx2row_total_ = 0;
    uint64_t pe_internal_pod_reject_rowindex_total_ = 0;
    uint64_t pe_internal_pod_reject_rowdescriptor_total_ = 0;
    uint64_t pe_internal_pod_useful_total_ = 0;
    uint64_t pe_internal_pod_useful_join_grant_total_ = 0;
    uint64_t pe_internal_pod_useful_duplicate_replay_elide_total_ = 0;
    uint64_t pe_internal_pod_useful_base_total_ = 0;
    uint64_t pe_internal_pod_useful_band_total_ = 0;
    uint64_t pe_internal_pod_useful_other_total_ = 0;
    uint64_t pe_internal_pod_useful_idx2row_total_ = 0;
    uint64_t pe_internal_pod_useful_rowindex_total_ = 0;
    uint64_t pe_internal_pod_useful_rowdescriptor_total_ = 0;
    uint64_t pe_internal_pod_service_join_live_total_ = 0;
    uint64_t pe_internal_pod_service_join_live_idx2row_total_ = 0;
    uint64_t pe_internal_pod_service_join_live_rowindex_total_ = 0;
    uint64_t pe_internal_pod_service_join_live_rowdescriptor_total_ = 0;
    uint64_t pe_internal_pod_service_join_ready_total_ = 0;
    uint64_t pe_internal_pod_service_join_ready_idx2row_total_ = 0;
    uint64_t pe_internal_pod_service_join_ready_rowindex_total_ = 0;
    uint64_t pe_internal_pod_service_join_ready_rowdescriptor_total_ = 0;
    uint64_t pe_internal_pod_service_ready_transition_total_ = 0;
    uint64_t pe_internal_pod_service_ready_transition_idx2row_total_ = 0;
    uint64_t pe_internal_pod_service_ready_transition_rowindex_total_ = 0;
    uint64_t pe_internal_pod_service_ready_transition_rowdescriptor_total_ = 0;
    uint64_t pe_internal_pod_service_ready_fanout_total_ = 0;
    uint64_t pe_internal_pod_service_ready_fanout_idx2row_total_ = 0;
    uint64_t pe_internal_pod_service_ready_fanout_rowindex_total_ = 0;
    uint64_t pe_internal_pod_service_ready_fanout_rowdescriptor_total_ = 0;
    uint64_t pe_internal_pod_service_ready_fanout_consumers_sum_total_ = 0;
    uint64_t pe_internal_pod_service_ready_fanout_consumers_sum_idx2row_total_ = 0;
    uint64_t pe_internal_pod_service_ready_fanout_consumers_sum_rowindex_total_ = 0;
    uint64_t pe_internal_pod_service_ready_fanout_consumers_sum_rowdescriptor_total_ = 0;
    uint64_t pe_internal_pod_service_release_deferred_total_ = 0;
    uint64_t pe_internal_pod_service_release_deferred_idx2row_total_ = 0;
    uint64_t pe_internal_pod_service_release_deferred_rowindex_total_ = 0;
    uint64_t pe_internal_pod_service_release_deferred_rowdescriptor_total_ = 0;
    uint64_t pe_internal_pod_service_ready_release_total_ = 0;
    uint64_t pe_internal_pod_service_ready_release_idx2row_total_ = 0;
    uint64_t pe_internal_pod_service_ready_release_rowindex_total_ = 0;
    uint64_t pe_internal_pod_service_ready_release_rowdescriptor_total_ = 0;
    uint64_t pe_internal_pod_service_released_idx2row_total_ = 0;
    uint64_t pe_internal_pod_service_released_rowindex_total_ = 0;
    uint64_t pe_internal_pod_service_released_rowdescriptor_total_ = 0;
    uint64_t pe_internal_pod_service_release_missing_idx2row_total_ = 0;
    uint64_t pe_internal_pod_service_release_missing_rowindex_total_ = 0;
    uint64_t pe_internal_pod_service_release_missing_rowdescriptor_total_ = 0;
    uint64_t pe_internal_pod_service_late_join_total_ = 0;
    uint64_t pe_internal_pod_service_late_join_idx2row_total_ = 0;
    uint64_t pe_internal_pod_service_late_join_rowindex_total_ = 0;
    uint64_t pe_internal_pod_service_late_join_rowdescriptor_total_ = 0;
    uint64_t pe_internal_pod_service_potential_private_service_elide_total_ = 0;
    uint64_t pe_internal_pod_service_potential_private_service_elide_idx2row_total_ = 0;
    uint64_t pe_internal_pod_service_potential_private_service_elide_rowindex_total_ = 0;
    uint64_t pe_internal_pod_service_potential_private_service_elide_rowdescriptor_total_ = 0;
    uint64_t pe_internal_pod_owner_first_service_elide_total_ = 0;
    uint64_t pe_internal_pod_owner_first_service_elide_idx2row_total_ = 0;
    uint64_t pe_internal_pod_owner_first_service_elide_rowindex_total_ = 0;
    uint64_t pe_internal_pod_owner_first_service_elide_rowdescriptor_total_ = 0;
    uint64_t pe_internal_pod_duplicate_metadata_replay_elided_total_ = 0;
    uint64_t pe_internal_pod_duplicate_metadata_issue_elided_total_ = 0;
    uint64_t pe_internal_pod_fallback_private_issue_total_ = 0;
    uint64_t weight_sram_stall_budget_cycles_ = 0;
    uint64_t weight_sram_stall_cycles_total_ = 0;
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
