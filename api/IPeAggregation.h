// -*- c++ -*-
//
// IPeAggregation: PE 级汇聚接口（统计/阶段事件）。
//
// 目的：
// - 让 control 层（SnnPESubComponent 等）仅依赖窄接口上报统计与阶段事件；
// - 消除对 components/MultiCorePE.h 的硬依赖，降低耦合与编译依赖面。
//

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <sst/core/serialization/serialize.h>

namespace SST { namespace SnnDL {


struct TassLfP0PrePayloadEntry {
    uint32_t pre_global = 0;
    uint64_t payload_bytes = 0;

    void serialize_order(SST::Core::Serialization::serializer& ser) {
        SST_SER(pre_global);
        SST_SER(payload_bytes);
    }
};

struct TassLfP0WindowReport {
    uint32_t mesh_rows = 1;
    uint32_t mesh_cols = 1;
    uint32_t block_h = 2;
    uint32_t block_w = 2;
    uint32_t cores_per_pe = 1;
    uint32_t node_id = 0;
    uint32_t core_id = 0;
    uint32_t window_seq = 0;
    uint32_t line_size_bytes = 64;
    uint64_t payload_bytes_total = 0;
    uint64_t current_vlf_line_groups_total = 0;
    std::vector<TassLfP0PrePayloadEntry> pre_payload_entries{};

    void serialize_order(SST::Core::Serialization::serializer& ser) {
        SST_SER(mesh_rows);
        SST_SER(mesh_cols);
        SST_SER(block_h);
        SST_SER(block_w);
        SST_SER(cores_per_pe);
        SST_SER(node_id);
        SST_SER(core_id);
        SST_SER(window_seq);
        SST_SER(line_size_bytes);
        SST_SER(payload_bytes_total);
        SST_SER(current_vlf_line_groups_total);
        SST_SER(pre_payload_entries);
    }
};

struct TassNaiveRequestEntry {
    uint32_t retire_seq = 0;
    uint32_t widx = 0;

    void serialize_order(SST::Core::Serialization::serializer& ser) {
        SST_SER(retire_seq);
        SST_SER(widx);
    }
};

struct TassNaiveWindowRequest {
    uint32_t mesh_rows = 1;
    uint32_t mesh_cols = 1;
    uint32_t block_h = 2;
    uint32_t block_w = 2;
    uint32_t source_node = 0;
    uint32_t source_core = 0;
    uint32_t cores_per_pe = 1;
    uint32_t window_seq = 0;
    uint32_t line_size_bytes = 64;
    std::vector<TassNaiveRequestEntry> entries{};

    void serialize_order(SST::Core::Serialization::serializer& ser) {
        SST_SER(mesh_rows);
        SST_SER(mesh_cols);
        SST_SER(block_h);
        SST_SER(block_w);
        SST_SER(source_node);
        SST_SER(source_core);
        SST_SER(cores_per_pe);
        SST_SER(window_seq);
        SST_SER(line_size_bytes);
        SST_SER(entries);
    }
};

struct TassNaiveResponseEntry {
    uint32_t dst_core = 0;
    uint32_t window_seq = 0;
    uint32_t retire_seq = 0;
    float weight = 0.0f;

    void serialize_order(SST::Core::Serialization::serializer& ser) {
        SST_SER(dst_core);
        SST_SER(window_seq);
        SST_SER(retire_seq);
        SST_SER(weight);
    }
};

class IPeAggregation {
public:
    virtual ~IPeAggregation() = default;

    // === 阶段事件（用于 PE 级落盘/聚合） ===
    virtual void notifyStageEvent(uint32_t seq,
                                  const std::string& event,
                                  uint64_t ts_ns,
                                  uint64_t spikes_emitted,
                                  int core_id) = 0;

    // === Batch-A: memory access aggregation ===
    virtual void accumulateMemReadLatency(uint64_t latency_cycles, bool is_weight) = 0;
    virtual void accumulateIssueStats(uint64_t req_size_bytes, uint64_t inflight) = 0;

    // === GAS window stats aggregation ===
    virtual void accumulateGasStatsExt(uint64_t unique_bytes, uint64_t unique_reads,
                                       uint64_t rowwin_triggers, uint64_t rowwin_bytes,
                                       uint64_t bursts, uint64_t payload_bytes,
                                       uint64_t window_inflight_peak,
                                       uint64_t window_buffer_max_bytes,
                                       uint64_t unique_line_count,
                                       uint64_t covered_line_count,
                                       uint64_t overfetch_bytes,
                                       uint64_t apply_bank_credit_effective,
                                       uint64_t cmd_cost_veto,
                                       uint64_t cmd_cost_veto_fine_gap,
                                       uint64_t cmd_cost_veto_row_window) = 0;

    virtual void accumulateActivityF(double f) = 0;
    virtual void accumulateApplyScatterStats(uint64_t acc_updates, uint64_t posts_touched,
                                             uint64_t spikes_emitted, uint64_t hwm_bytes,
                                             uint64_t spill_records, uint64_t spilled_bytes) = 0;
    virtual void accumulateSynapseReadStats(uint64_t gcss_lookup_hit_total,
                                            uint64_t gcss_lookup_miss_total,
                                            uint64_t dense_reqs_total,
                                            uint64_t dense_bytes_total,
                                            uint64_t rowptr_reqs_total,
                                            uint64_t rowptr_bytes_total,
                                            uint64_t colidx_reqs_total,
                                            uint64_t colidx_bytes_total,
                                            uint64_t blockdata_reqs_total,
                                            uint64_t blockdata_bytes_total,
                                            uint64_t gcss_reqs_total,
                                            uint64_t gcss_bytes_total,
                                            uint64_t weight_idx_sram_reads_total,
                                            uint64_t weight_idx_sram_writes_total,
                                            uint64_t weight_idx_sram_bytes_read_total,
                                            uint64_t weight_idx_sram_bytes_write_total,
                                            uint64_t weight_idx_sram_bank_conflict_ticks_total,
                                            uint64_t weight_idx_sram_predicted_extra_cycles_total,
                                            uint64_t weight_idx_sram_resident_bytes_peak,
                                            uint64_t weight_idx_lookup_total,
                                            uint64_t weight_idx_lookup_idx2_total,
                                            uint64_t weight_l0_sram_reads_total,
                                            uint64_t weight_l0_sram_writes_total,
                                            uint64_t weight_l0_sram_bytes_read_total,
                                            uint64_t weight_l0_sram_bytes_write_total,
                                            uint64_t weight_l0_sram_bank_conflict_ticks_total,
                                            uint64_t weight_l0_sram_predicted_extra_cycles_total,
                                            uint64_t weight_l0_sram_resident_bytes_peak,
                                            uint64_t weight_l0_lookup_total,
                                            uint64_t weight_l0_hit_total,
                                            uint64_t weight_l0_fill_total,
                                            uint64_t weight_l0_evict_total,
                                            uint64_t core_state_sram_reads_total,
                                            uint64_t core_state_sram_writes_total,
                                            uint64_t core_state_sram_bytes_read_total,
                                            uint64_t core_state_sram_bytes_write_total,
                                            uint64_t core_state_sram_bank_conflict_ticks_total,
                                            uint64_t core_state_sram_predicted_extra_cycles_total,
                                            uint64_t core_state_sram_resident_bytes_peak,
                                            uint64_t gas_retire_global_hol_cycles_total,
                                            uint64_t gas_retire_ready_but_blocked_edges_total,
                                            uint64_t gas_retire_per_post_progress_total,
                                            uint64_t gas_tass_lf_p0_block_epochs_total,
                                            uint64_t gas_tass_lf_p0_block_active_pres_total,
                                            uint64_t gas_tass_lf_p0_block_shared_pres_total,
                                            uint64_t gas_tass_lf_p0_cross_core_joins_total,
                                            uint64_t gas_tass_lf_p0_payload_bytes_total,
                                            uint64_t gas_tass_lf_p0_current_vlf_line_groups_total,
                                            uint64_t gas_tass_lf_p0_block_naive_line_count_total,
                                            uint64_t gas_tass_lf_p0_block_fused_lb_line_count_total,
                                            uint64_t gas_tass_lf_p0_response_fanout_total,
                                            uint64_t gas_tass_lf_p0_reports_flushed_total,
                                            uint64_t gas_tass_lf_p0_reports_nonzero_payload_total,
                                            uint64_t gas_tass_lf_p0_reports_pre_entries_total,
                                            uint64_t gas_tass_lf_p0_reports_via_callback_total,
                                            uint64_t gas_tass_lf_p0_reports_via_fallback_total) = 0;

    virtual void submitTassLfP0WindowReport(const TassLfP0WindowReport& report) = 0;
    virtual void submitTassNaiveWindowRequest(const TassNaiveWindowRequest& request) = 0;

    virtual void accumulateUniqueNeuronFired(uint64_t cnt) = 0;
    virtual void accumulateWindowSpikes(uint32_t seq, uint64_t count) = 0;
};

}} // namespace SST::SnnDL
