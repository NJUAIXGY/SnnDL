// -*- c++ -*-

#include <sst/core/sst_config.h>

#include "components/MultiCorePE.h"

#include <algorithm>

using namespace SST;
using namespace SST::SnnDL;

void MultiCorePE::accumulateRiscvSnnRuntimeStats(uint64_t workload_selected,
                                                 uint64_t firmware_elf_present,
                                                 uint64_t firmware_loaded,
                                                 uint64_t backend_runtime_bridge,
                                                 uint64_t firmware_started_count,
                                                 uint64_t submitted_commands,
                                                 uint64_t accepted_commands,
                                                 uint64_t completion_visible_count,
                                                 uint64_t completion_consumed_count,
                                                 uint64_t fused_step_completion_count,
                                                 uint64_t fault_count,
                                                 uint64_t last_completion_status,
                                                 uint64_t last_fault_csr,
                                                 uint64_t backend_runtime_bridge_provider_bound) {
    if (stat_riscv_snn_workload_selected_) {
        stat_riscv_snn_workload_selected_->addData(workload_selected);
    }
    if (stat_riscv_snn_firmware_elf_present_) {
        stat_riscv_snn_firmware_elf_present_->addData(firmware_elf_present);
    }
    if (stat_riscv_snn_firmware_loaded_) {
        stat_riscv_snn_firmware_loaded_->addData(firmware_loaded);
    }
    if (stat_riscv_snn_backend_runtime_bridge_) {
        stat_riscv_snn_backend_runtime_bridge_->addData(backend_runtime_bridge);
    }
    if (stat_riscv_snn_firmware_started_count_) {
        stat_riscv_snn_firmware_started_count_->addData(firmware_started_count);
    }
    if (stat_riscv_snn_submitted_commands_) {
        stat_riscv_snn_submitted_commands_->addData(submitted_commands);
    }
    if (stat_riscv_snn_accepted_commands_) {
        stat_riscv_snn_accepted_commands_->addData(accepted_commands);
    }
    if (stat_riscv_snn_completion_visible_count_) {
        stat_riscv_snn_completion_visible_count_->addData(completion_visible_count);
    }
    if (stat_riscv_snn_completion_consumed_count_) {
        stat_riscv_snn_completion_consumed_count_->addData(completion_consumed_count);
    }
    if (stat_riscv_snn_fused_step_completion_count_) {
        stat_riscv_snn_fused_step_completion_count_->addData(fused_step_completion_count);
    }
    if (stat_riscv_snn_fault_count_) {
        stat_riscv_snn_fault_count_->addData(fault_count);
    }
    if (stat_riscv_snn_last_completion_status_) {
        stat_riscv_snn_last_completion_status_->addData(last_completion_status);
    }
    if (stat_riscv_snn_last_fault_csr_) {
        stat_riscv_snn_last_fault_csr_->addData(last_fault_csr);
    }
    if (stat_riscv_snn_backend_runtime_bridge_provider_bound_) {
        stat_riscv_snn_backend_runtime_bridge_provider_bound_->addData(
            backend_runtime_bridge_provider_bound);
    }
}

void MultiCorePE::accumulateMemReadLatency(uint64_t latency_cycles, bool is_weight) {
    if (stat_mem_read_latency_cycles_) stat_mem_read_latency_cycles_->addData(latency_cycles);
    if (is_weight) {
        if (stat_mem_read_latency_cycles_weights_) stat_mem_read_latency_cycles_weights_->addData(latency_cycles);
    } else {
        if (stat_mem_read_latency_cycles_state_) stat_mem_read_latency_cycles_state_->addData(latency_cycles);
    }
    // 窗口化：按当前仿真时间(ns)聚合响应时延
    if (window_stats_enable_) {
        uint64_t now_ns = getCurrentSimTimeNano();
        WindowAgg& w = getOrCreateWindow_(now_ns);
        w.read_count += 1;
        w.read_latency_sum += latency_cycles;
    }
}

void MultiCorePE::accumulateIssueStats(uint64_t req_size_bytes, uint64_t inflight) {
    if (stat_mem_req_size_bytes_) stat_mem_req_size_bytes_->addData(req_size_bytes);
    if (stat_mem_outstanding_at_issue_) stat_mem_outstanding_at_issue_->addData(inflight);
    if (stat_memory_requests_) stat_memory_requests_->addData(1);
    // 窗口化：按当前仿真时间(ns)聚合发起侧指标
    if (window_stats_enable_) {
        uint64_t now_ns = getCurrentSimTimeNano();
        WindowAgg& w = getOrCreateWindow_(now_ns);
        w.issue_count += 1;
        w.req_size_sum += req_size_bytes;
        w.outstanding_sum += inflight;
    }
}

void MultiCorePE::accumulateGasStats(uint64_t unique_bytes, uint64_t unique_reads) {
    if (unique_reads && stat_gas_unique_reads_total_) stat_gas_unique_reads_total_->addData(unique_reads);
    if (unique_bytes && stat_gas_unique_bytes_total_) stat_gas_unique_bytes_total_->addData(unique_bytes);
}

void MultiCorePE::accumulateGasStatsExt(uint64_t unique_bytes, uint64_t unique_reads,
                                        uint64_t rowwin_triggers, uint64_t rowwin_bytes,
                                        uint64_t bursts, uint64_t payload_bytes,
                                        uint64_t window_inflight_peak,
                                        uint64_t window_buffer_max_bytes,
                                        uint64_t frontend_staged_reads,
                                        uint64_t frontend_staged_line_touches,
                                        uint64_t frontend_granules_built,
                                        uint64_t unique_line_count,
                                        uint64_t covered_line_count,
                                        uint64_t overfetch_bytes,
                                        uint64_t apply_bank_credit_effective,
                                        uint64_t cmd_cost_veto,
                                        uint64_t cmd_cost_veto_fine_gap,
                                        uint64_t cmd_cost_veto_row_window,
                                        uint64_t stall_on_step_gate_cycles) {
    accumulateGasStats(unique_bytes, unique_reads);
    if (rowwin_triggers && stat_gas_rowwin_triggers_total_) stat_gas_rowwin_triggers_total_->addData(rowwin_triggers);
    if (rowwin_bytes && stat_gas_rowwin_bytes_total_) stat_gas_rowwin_bytes_total_->addData(rowwin_bytes);
    if (bursts && stat_gas_total_bursts_) stat_gas_total_bursts_->addData(bursts);
    if (payload_bytes && stat_gas_total_payload_bytes_) stat_gas_total_payload_bytes_->addData(payload_bytes);
    if (frontend_staged_reads && stat_gas_frontend_staged_reads_total_) stat_gas_frontend_staged_reads_total_->addData(frontend_staged_reads);
    if (frontend_staged_line_touches && stat_gas_frontend_staged_line_touches_total_) stat_gas_frontend_staged_line_touches_total_->addData(frontend_staged_line_touches);
    if (frontend_granules_built && stat_gas_frontend_granules_built_total_) stat_gas_frontend_granules_built_total_->addData(frontend_granules_built);
    if (unique_line_count && stat_gas_unique_line_count_total_) stat_gas_unique_line_count_total_->addData(unique_line_count);
    if (covered_line_count && stat_gas_covered_line_count_total_) stat_gas_covered_line_count_total_->addData(covered_line_count);
    if (overfetch_bytes && stat_gas_overfetch_bytes_total_) stat_gas_overfetch_bytes_total_->addData(overfetch_bytes);
    if (apply_bank_credit_effective && stat_gas_apply_bank_credit_effective_total_) stat_gas_apply_bank_credit_effective_total_->addData(apply_bank_credit_effective);
    if (cmd_cost_veto && stat_gas_cmd_cost_veto_total_) stat_gas_cmd_cost_veto_total_->addData(cmd_cost_veto);
    if (cmd_cost_veto_fine_gap && stat_gas_cmd_cost_veto_fine_gap_total_) stat_gas_cmd_cost_veto_fine_gap_total_->addData(cmd_cost_veto_fine_gap);
    if (cmd_cost_veto_row_window && stat_gas_cmd_cost_veto_row_window_total_) stat_gas_cmd_cost_veto_row_window_total_->addData(cmd_cost_veto_row_window);
    if (stall_on_step_gate_cycles && stat_stall_on_step_gate_cycles_total_) {
        stat_stall_on_step_gate_cycles_total_->addData(stall_on_step_gate_cycles);
    }
}

void MultiCorePE::accumulateActivityF(double f) {
    if (stat_gas_activity_f_) stat_gas_activity_f_->addData(f);
    if (window_stats_enable_) {
        uint64_t now_ns = getCurrentSimTimeNano();
        WindowAgg& w = getOrCreateWindow_(now_ns);
        w.activity_f_sum += f;
        w.activity_f_count += 1;
    }
}

void MultiCorePE::accumulateApplyScatterStats(uint64_t acc_updates, uint64_t posts_touched,
                                              uint64_t spikes_emitted, uint64_t hwm_bytes,
                                              uint64_t spill_records, uint64_t spilled_bytes) {
    if (acc_updates && stat_gas_apply_acc_updates_total_) stat_gas_apply_acc_updates_total_->addData(acc_updates);
    if (posts_touched && stat_gas_acc_posts_touched_total_) stat_gas_acc_posts_touched_total_->addData(posts_touched);
    if (spikes_emitted && stat_gas_scatter_spikes_emitted_total_) stat_gas_scatter_spikes_emitted_total_->addData(spikes_emitted);
    if (hwm_bytes && stat_gas_acc_hwm_bytes_total_) stat_gas_acc_hwm_bytes_total_->addData(hwm_bytes);
    if (spill_records && stat_gas_acc_spill_records_total_) stat_gas_acc_spill_records_total_->addData(spill_records);
    if (spilled_bytes && stat_gas_acc_spilled_bytes_total_) stat_gas_acc_spilled_bytes_total_->addData(spilled_bytes);
}

MultiCorePE::StepPerfAgg& MultiCorePE::getOrCreateStepPerf_(uint32_t seq) {
    return step_perf_[seq];
}

MultiCorePE::StepPerfAgg& MultiCorePE::getOrCreateCoreStepPerf_(int core_id, uint32_t seq) {
    return core_step_perf_[CoreStepKey{core_id, seq}];
}

MultiCorePE::StageMarks& MultiCorePE::getOrCreateCoreStageMarks_(int core_id, uint32_t seq) {
    return core_stage_marks_[CoreStepKey{core_id, seq}];
}

void MultiCorePE::recordStepActivationSummary(uint32_t seq,
                                              uint64_t pre_selected,
                                              uint64_t spike_attempts,
                                              uint64_t spikes_injected,
                                              uint64_t route_hits,
                                              uint64_t route_misses,
                                              uint64_t local_drops) {
    if (seq == 0) return;
    auto& perf = getOrCreateStepPerf_(seq);
    perf.seed_pre_selected += pre_selected;
    perf.seed_spike_attempts += spike_attempts;
    perf.seed_spikes_injected += spikes_injected;
    perf.seed_route_hits += route_hits;
    perf.seed_route_misses += route_misses;
    perf.seed_local_drops += local_drops;
}

void MultiCorePE::recordStepRxPacket(uint32_t seq, NocPacketKind kind, bool is_local, bool before_begin_gather) {
    if (seq == 0) return;
    auto& perf = getOrCreateStepPerf_(seq);
    perf.rx_packets_total += 1;
    if (is_local) perf.rx_local_packets_total += 1;
    else perf.rx_remote_packets_total += 1;
    if (before_begin_gather) {
        perf.rx_packets_before_bg_total += 1;
        if (is_local) perf.rx_local_packets_before_bg_total += 1;
        else perf.rx_remote_packets_before_bg_total += 1;
        if (perf.rx_packets_before_bg_total > perf.rx_gate_pending_peak) {
            perf.rx_gate_pending_peak = perf.rx_packets_before_bg_total;
        }
    } else {
        auto stage_it = stage_marks_.find(seq);
        int stage_code = 0;
        if (stage_it != stage_marks_.end()) {
            const auto& m = stage_it->second;
            if (m.bs != 0) stage_code = 3;
            else if (m.ga != 0) stage_code = 2;
            else if (m.bg != 0) stage_code = 1;
        }
        if (stage_code == 1) perf.rx_packets_during_gather_total += 1;
        else if (stage_code == 2) perf.rx_packets_during_apply_total += 1;
        else if (stage_code == 3) perf.rx_packets_during_scatter_total += 1;
    }
    switch (kind) {
        case NocPacketKind::Spike:
            perf.rx_spike_packets_total += 1;
            break;
        case NocPacketKind::SpikeKey:
            perf.rx_spikekey_packets_total += 1;
            break;
        case NocPacketKind::SpikeTileKey:
            perf.rx_spiketilekey_packets_total += 1;
            break;
        default:
            break;
    }
}

void MultiCorePE::recordStepRxGate(uint32_t seq,
                                   uint64_t accept_total,
                                   uint64_t reject_refractory_total,
                                   uint64_t direct_accept_total,
                                   uint64_t direct_reject_refractory_total,
                                   uint64_t fastpath_accept_total,
                                   uint64_t fastpath_reject_refractory_total) {
    if (seq == 0) return;
    auto& perf = getOrCreateStepPerf_(seq);
    perf.rx_gate_accept_total += accept_total;
    perf.rx_gate_reject_refractory_total += reject_refractory_total;
    perf.rx_gate_direct_accept_total += direct_accept_total;
    perf.rx_gate_direct_reject_refractory_total += direct_reject_refractory_total;
    perf.rx_gate_fastpath_accept_total += fastpath_accept_total;
    perf.rx_gate_fastpath_reject_refractory_total += fastpath_reject_refractory_total;
}

void MultiCorePE::recordStepGasStat(uint32_t seq, const GasStatEvent& st) {
    if (seq == 0) return;
    auto& perf = getOrCreateStepPerf_(seq);
    perf.unique_reads += st.unique_reads;
    perf.unique_bytes += st.unique_bytes;
    perf.rowwin_triggers += st.rowwin_triggers;
    perf.rowwin_bytes += st.rowwin_bytes;
    perf.bursts += st.bursts;
    perf.payload_bytes += st.payload_bytes;
    perf.window_inflight_peak = std::max<uint64_t>(perf.window_inflight_peak, st.window_inflight_peak);
    perf.window_buffer_max_bytes = std::max<uint64_t>(perf.window_buffer_max_bytes, st.window_buffer_max_bytes);
    perf.gap_absorbed_bytes += st.gap_absorbed_bytes;
    perf.frontend_staged_reads += st.frontend_staged_reads;
    perf.frontend_staged_line_touches += st.frontend_staged_line_touches;
    perf.frontend_granules_built += st.frontend_granules_built;
    perf.unique_line_count += st.unique_line_count;
    perf.covered_line_count += st.covered_line_count;
    perf.overfetch_bytes += st.overfetch_bytes;
    perf.apply_bank_credit_effective += st.apply_bank_credit_effective;
    perf.cmd_cost_veto += st.cmd_cost_veto;
    perf.cmd_cost_veto_fine_gap += st.cmd_cost_veto_fine_gap;
    perf.cmd_cost_veto_row_window += st.cmd_cost_veto_row_window;
    perf.apply_issue_attempt_total += st.apply_issue_attempt_total;
    perf.apply_issue_success_total += st.apply_issue_success_total;
    perf.apply_issue_block_no_ready_total += st.apply_issue_block_no_ready_total;
    perf.apply_issue_block_inflight_cap_total += st.apply_issue_block_inflight_cap_total;
    perf.apply_issue_block_bank_credit_total += st.apply_issue_block_bank_credit_total;
    perf.apply_issue_block_downstream_busy_total += st.apply_issue_block_downstream_busy_total;
    perf.apply_issue_block_retire_guard_total += st.apply_issue_block_retire_guard_total;
    perf.apply_ready_queue_peak = std::max<uint64_t>(perf.apply_ready_queue_peak, st.apply_ready_queue_peak);
    perf.apply_ready_queue_nonempty_cycles_total += st.apply_ready_queue_nonempty_cycles_total;
    perf.apply_first_issue_delay_ns += st.apply_first_issue_delay_ns;
    perf.apply_first_down_resp_delay_ns += st.apply_first_down_resp_delay_ns;
    perf.apply_first_granule_done_delay_ns += st.apply_first_granule_done_delay_ns;
    perf.apply_first_up_resp_delay_ns += st.apply_first_up_resp_delay_ns;
    perf.apply_down_resp_total += st.apply_down_resp_total;
    perf.apply_completed_granules_total += st.apply_completed_granules_total;
    perf.apply_emitted_subreads_total += st.apply_emitted_subreads_total;
    perf.apply_backlog_granules_residual =
        std::max<uint64_t>(perf.apply_backlog_granules_residual, st.apply_backlog_granules_residual);
    perf.apply_backlog_pending_up_reads_residual =
        std::max<uint64_t>(perf.apply_backlog_pending_up_reads_residual,
                           st.apply_backlog_pending_up_reads_residual);
    perf.apply_backlog_inflight_residual =
        std::max<uint64_t>(perf.apply_backlog_inflight_residual, st.apply_backlog_inflight_residual);
    perf.apply_backlog_granules_peak_after_due =
        std::max<uint64_t>(perf.apply_backlog_granules_peak_after_due,
                           st.apply_backlog_granules_peak_after_due);
    perf.apply_backlog_pending_up_reads_peak_after_due =
        std::max<uint64_t>(perf.apply_backlog_pending_up_reads_peak_after_due,
                           st.apply_backlog_pending_up_reads_peak_after_due);
    perf.apply_backlog_inflight_peak_after_due =
        std::max<uint64_t>(perf.apply_backlog_inflight_peak_after_due,
                           st.apply_backlog_inflight_peak_after_due);
}

void MultiCorePE::recordCoreStepGasStat(int core_id, uint32_t seq, const GasStatEvent& st) {
    if (seq == 0 || core_id < 0) return;
    auto& perf = getOrCreateCoreStepPerf_(core_id, seq);
    perf.unique_reads += st.unique_reads;
    perf.unique_bytes += st.unique_bytes;
    perf.rowwin_triggers += st.rowwin_triggers;
    perf.rowwin_bytes += st.rowwin_bytes;
    perf.bursts += st.bursts;
    perf.payload_bytes += st.payload_bytes;
    perf.window_inflight_peak = std::max<uint64_t>(perf.window_inflight_peak, st.window_inflight_peak);
    perf.window_buffer_max_bytes = std::max<uint64_t>(perf.window_buffer_max_bytes, st.window_buffer_max_bytes);
    perf.gap_absorbed_bytes += st.gap_absorbed_bytes;
    perf.frontend_staged_reads += st.frontend_staged_reads;
    perf.frontend_staged_line_touches += st.frontend_staged_line_touches;
    perf.frontend_granules_built += st.frontend_granules_built;
    perf.unique_line_count += st.unique_line_count;
    perf.covered_line_count += st.covered_line_count;
    perf.overfetch_bytes += st.overfetch_bytes;
    perf.apply_bank_credit_effective += st.apply_bank_credit_effective;
    perf.cmd_cost_veto += st.cmd_cost_veto;
    perf.cmd_cost_veto_fine_gap += st.cmd_cost_veto_fine_gap;
    perf.cmd_cost_veto_row_window += st.cmd_cost_veto_row_window;
    perf.apply_issue_attempt_total += st.apply_issue_attempt_total;
    perf.apply_issue_success_total += st.apply_issue_success_total;
    perf.apply_issue_block_no_ready_total += st.apply_issue_block_no_ready_total;
    perf.apply_issue_block_inflight_cap_total += st.apply_issue_block_inflight_cap_total;
    perf.apply_issue_block_bank_credit_total += st.apply_issue_block_bank_credit_total;
    perf.apply_issue_block_downstream_busy_total += st.apply_issue_block_downstream_busy_total;
    perf.apply_issue_block_retire_guard_total += st.apply_issue_block_retire_guard_total;
    perf.apply_ready_queue_peak = std::max<uint64_t>(perf.apply_ready_queue_peak, st.apply_ready_queue_peak);
    perf.apply_ready_queue_nonempty_cycles_total += st.apply_ready_queue_nonempty_cycles_total;
    perf.apply_first_issue_delay_ns += st.apply_first_issue_delay_ns;
    perf.apply_first_down_resp_delay_ns += st.apply_first_down_resp_delay_ns;
    perf.apply_first_granule_done_delay_ns += st.apply_first_granule_done_delay_ns;
    perf.apply_first_up_resp_delay_ns += st.apply_first_up_resp_delay_ns;
    perf.apply_down_resp_total += st.apply_down_resp_total;
    perf.apply_completed_granules_total += st.apply_completed_granules_total;
    perf.apply_emitted_subreads_total += st.apply_emitted_subreads_total;
    perf.apply_backlog_granules_residual =
        std::max<uint64_t>(perf.apply_backlog_granules_residual, st.apply_backlog_granules_residual);
    perf.apply_backlog_pending_up_reads_residual =
        std::max<uint64_t>(perf.apply_backlog_pending_up_reads_residual,
                           st.apply_backlog_pending_up_reads_residual);
    perf.apply_backlog_inflight_residual =
        std::max<uint64_t>(perf.apply_backlog_inflight_residual, st.apply_backlog_inflight_residual);
    perf.apply_backlog_granules_peak_after_due =
        std::max<uint64_t>(perf.apply_backlog_granules_peak_after_due,
                           st.apply_backlog_granules_peak_after_due);
    perf.apply_backlog_pending_up_reads_peak_after_due =
        std::max<uint64_t>(perf.apply_backlog_pending_up_reads_peak_after_due,
                           st.apply_backlog_pending_up_reads_peak_after_due);
    perf.apply_backlog_inflight_peak_after_due =
        std::max<uint64_t>(perf.apply_backlog_inflight_peak_after_due,
                           st.apply_backlog_inflight_peak_after_due);
}

void MultiCorePE::recordStepRetireStat(uint32_t seq,
                                       uint64_t retire_global_hol_cycles_total,
                                       uint64_t retire_ready_but_blocked_edges_total,
                                       uint64_t retire_per_post_progress_total,
                                       uint64_t retire_wait_cycles_total,
                                       uint64_t retire_wait_cycles_due_to_hol_total,
                                       uint64_t retire_wait_cycles_due_to_barrier_total,
                                       uint64_t retire_wait_cycles_due_to_not_ready_total,
                                       uint64_t retire_samepost_blocked_edges_total,
                                       uint64_t retire_crosspost_blocked_edges_total,
                                       uint64_t retire_policy_loss_cycles_total,
                                       uint64_t retire_policy_loss_edges_total,
                                       uint64_t retire_shadow_per_post_recoverable_cycles_total,
                                       uint64_t retire_shadow_per_post_recoverable_edges_total,
                                       uint64_t retire_shadow_per_post_ready_posts_peak,
                                       uint64_t retire_shadow_per_post_committable_edges_peak,
                                       uint64_t retire_head_hol_cycles_dense_total,
                                       uint64_t retire_head_hol_cycles_cache_total,
                                       uint64_t retire_head_hol_cycles_miss_total,
                                       uint64_t retire_head_hol_cycles_bcsr_total,
                                       uint64_t retire_head_hol_cycles_bcsr_file_total,
                                       uint64_t retire_head_hol_cycles_gcss_total,
                                       uint64_t retire_head_blocked_edges_dense_total,
                                       uint64_t retire_head_blocked_edges_cache_total,
                                       uint64_t retire_head_blocked_edges_miss_total,
                                       uint64_t retire_head_blocked_edges_bcsr_total,
                                       uint64_t retire_head_blocked_edges_bcsr_file_total,
                                       uint64_t retire_head_blocked_edges_gcss_total,
                                       uint64_t retire_gcss_head_queued_not_issued_cycles_total,
                                       uint64_t retire_gcss_qni_head_wait_episodes_total,
                                       uint64_t retire_gcss_qni_head_wait_cycles_max,
                                       uint64_t retire_gcss_head_queued_not_issued_blocked_edges_total,
                                       uint64_t retire_gcss_head_issued_wait_resp_cycles_total,
                                       uint64_t retire_gcss_head_issued_wait_resp_blocked_edges_total,
                                       uint64_t retire_gcss_resp_ready_but_hol_cycles_total,
                                       uint64_t retire_gcss_resp_ready_but_hol_blocked_edges_total,
                                       uint64_t retire_gcss_qni_loader_not_ready_cycles_total,
                                       uint64_t retire_gcss_qni_loader_not_ready_blocked_edges_total,
                                       uint64_t retire_gcss_qni_weight_sram_stall_cycles_total,
                                       uint64_t retire_gcss_qni_weight_sram_stall_blocked_edges_total,
                                       uint64_t retire_gcss_qni_vlf_younger_ahead_cycles_total,
                                       uint64_t retire_gcss_qni_vlf_younger_ahead_blocked_edges_total,
                                       uint64_t retire_gcss_qni_vlf_younger_ahead_depth_total,
                                       uint64_t retire_gcss_qni_vlf_younger_ahead_depth_samples_total,
                                       uint64_t retire_gcss_qni_vlf_younger_ahead_depth_max,
                                       uint64_t retire_gcss_qni_issue_deferred_total,
                                       uint64_t retire_gcss_qni_pending_direct_queue_residency_cycles_total,
                                       uint64_t retire_gcss_qni_pending_direct_queue_residency_samples_total,
                                       uint64_t retire_gcss_qni_pending_direct_queue_residency_cycles_max,
                                       uint64_t retire_gcss_qni_vlf_front_inflight_full_cycles_total,
                                       uint64_t retire_gcss_qni_vlf_front_inflight_full_blocked_edges_total,
                                       uint64_t retire_gcss_qni_vlf_front_waiting_issue_cycles_total,
                                       uint64_t retire_gcss_qni_vlf_front_waiting_issue_blocked_edges_total,
                                       uint64_t retire_gcss_qni_pending_younger_ahead_cycles_total,
                                       uint64_t retire_gcss_qni_pending_younger_ahead_blocked_edges_total,
                                       uint64_t retire_gcss_qni_pending_front_inflight_full_cycles_total,
                                       uint64_t retire_gcss_qni_pending_front_inflight_full_blocked_edges_total,
                                       uint64_t retire_gcss_qni_pending_front_waiting_tick_cycles_total,
                                       uint64_t retire_gcss_qni_pending_front_waiting_tick_blocked_edges_total,
                                       uint64_t retire_gcss_qni_unknown_cycles_total,
                                       uint64_t retire_gcss_qni_unknown_blocked_edges_total,
                                       uint64_t retire_begin_apply_windows_total,
                                       uint64_t retire_begin_apply_prev_edges_total,
                                       uint64_t retire_begin_apply_outstanding_carryin_total,
                                       uint64_t retire_begin_apply_outstanding_carryin_windows_total,
                                       uint64_t retire_begin_apply_loader_not_ready_windows_total,
                                       uint64_t retire_edge_retire_registered_total,
                                       uint64_t retire_edge_retire_retired_total,
                                       uint64_t retire_end_scatter_gcss_vlf_issue_queue_residual_total,
                                       uint64_t retire_end_scatter_pending_direct_reads_residual_total,
                                       uint64_t retire_end_scatter_outstanding_residual_total,
                                       uint64_t retire_end_scatter_residual_work_windows_total,
                                       uint64_t retire_gcss_vlf_issue_prepare_total,
                                       uint64_t retire_gcss_vlf_issue_edges_total,
                                       uint64_t retire_gcss_vlf_issue_reorder_trigger_total,
                                       uint64_t retire_gcss_vlf_issue_line_groups_total,
                                       uint64_t retire_ready_queue_peak,
                                       uint64_t retire_unblock_events_total) {
    if (seq == 0) return;
    auto& perf = getOrCreateStepPerf_(seq);
    perf.retire_global_hol_cycles_total += retire_global_hol_cycles_total;
    perf.retire_ready_but_blocked_edges_total += retire_ready_but_blocked_edges_total;
    perf.retire_per_post_progress_total += retire_per_post_progress_total;
    perf.retire_wait_cycles_total += retire_wait_cycles_total;
    perf.retire_wait_cycles_due_to_hol_total += retire_wait_cycles_due_to_hol_total;
    perf.retire_wait_cycles_due_to_barrier_total += retire_wait_cycles_due_to_barrier_total;
    perf.retire_wait_cycles_due_to_not_ready_total += retire_wait_cycles_due_to_not_ready_total;
    perf.retire_samepost_blocked_edges_total += retire_samepost_blocked_edges_total;
    perf.retire_crosspost_blocked_edges_total += retire_crosspost_blocked_edges_total;
    perf.retire_policy_loss_cycles_total += retire_policy_loss_cycles_total;
    perf.retire_policy_loss_edges_total += retire_policy_loss_edges_total;
    perf.retire_shadow_per_post_recoverable_cycles_total += retire_shadow_per_post_recoverable_cycles_total;
    perf.retire_shadow_per_post_recoverable_edges_total += retire_shadow_per_post_recoverable_edges_total;
    perf.retire_shadow_per_post_ready_posts_peak =
        std::max<uint64_t>(perf.retire_shadow_per_post_ready_posts_peak, retire_shadow_per_post_ready_posts_peak);
    perf.retire_shadow_per_post_committable_edges_peak =
        std::max<uint64_t>(perf.retire_shadow_per_post_committable_edges_peak, retire_shadow_per_post_committable_edges_peak);
    perf.retire_head_hol_cycles_dense_total += retire_head_hol_cycles_dense_total;
    perf.retire_head_hol_cycles_cache_total += retire_head_hol_cycles_cache_total;
    perf.retire_head_hol_cycles_miss_total += retire_head_hol_cycles_miss_total;
    perf.retire_head_hol_cycles_bcsr_total += retire_head_hol_cycles_bcsr_total;
    perf.retire_head_hol_cycles_bcsr_file_total += retire_head_hol_cycles_bcsr_file_total;
    perf.retire_head_hol_cycles_gcss_total += retire_head_hol_cycles_gcss_total;
    perf.retire_head_blocked_edges_dense_total += retire_head_blocked_edges_dense_total;
    perf.retire_head_blocked_edges_cache_total += retire_head_blocked_edges_cache_total;
    perf.retire_head_blocked_edges_miss_total += retire_head_blocked_edges_miss_total;
    perf.retire_head_blocked_edges_bcsr_total += retire_head_blocked_edges_bcsr_total;
    perf.retire_head_blocked_edges_bcsr_file_total += retire_head_blocked_edges_bcsr_file_total;
    perf.retire_head_blocked_edges_gcss_total += retire_head_blocked_edges_gcss_total;
    perf.retire_gcss_head_queued_not_issued_cycles_total += retire_gcss_head_queued_not_issued_cycles_total;
    perf.retire_gcss_qni_head_wait_episodes_total += retire_gcss_qni_head_wait_episodes_total;
    perf.retire_gcss_qni_head_wait_cycles_max = std::max<uint64_t>(
        perf.retire_gcss_qni_head_wait_cycles_max, retire_gcss_qni_head_wait_cycles_max);
    perf.retire_gcss_head_queued_not_issued_blocked_edges_total += retire_gcss_head_queued_not_issued_blocked_edges_total;
    perf.retire_gcss_head_issued_wait_resp_cycles_total += retire_gcss_head_issued_wait_resp_cycles_total;
    perf.retire_gcss_head_issued_wait_resp_blocked_edges_total += retire_gcss_head_issued_wait_resp_blocked_edges_total;
    perf.retire_gcss_resp_ready_but_hol_cycles_total += retire_gcss_resp_ready_but_hol_cycles_total;
    perf.retire_gcss_resp_ready_but_hol_blocked_edges_total += retire_gcss_resp_ready_but_hol_blocked_edges_total;
    perf.retire_gcss_qni_loader_not_ready_cycles_total += retire_gcss_qni_loader_not_ready_cycles_total;
    perf.retire_gcss_qni_loader_not_ready_blocked_edges_total += retire_gcss_qni_loader_not_ready_blocked_edges_total;
    perf.retire_gcss_qni_weight_sram_stall_cycles_total += retire_gcss_qni_weight_sram_stall_cycles_total;
    perf.retire_gcss_qni_weight_sram_stall_blocked_edges_total += retire_gcss_qni_weight_sram_stall_blocked_edges_total;
    perf.retire_gcss_qni_vlf_younger_ahead_cycles_total += retire_gcss_qni_vlf_younger_ahead_cycles_total;
    perf.retire_gcss_qni_vlf_younger_ahead_blocked_edges_total += retire_gcss_qni_vlf_younger_ahead_blocked_edges_total;
    perf.retire_gcss_qni_vlf_younger_ahead_depth_total += retire_gcss_qni_vlf_younger_ahead_depth_total;
    perf.retire_gcss_qni_vlf_younger_ahead_depth_samples_total += retire_gcss_qni_vlf_younger_ahead_depth_samples_total;
    perf.retire_gcss_qni_vlf_younger_ahead_depth_max = std::max<uint64_t>(
        perf.retire_gcss_qni_vlf_younger_ahead_depth_max,
        retire_gcss_qni_vlf_younger_ahead_depth_max);
    perf.retire_gcss_qni_issue_deferred_total += retire_gcss_qni_issue_deferred_total;
    perf.retire_gcss_qni_pending_direct_queue_residency_cycles_total +=
        retire_gcss_qni_pending_direct_queue_residency_cycles_total;
    perf.retire_gcss_qni_pending_direct_queue_residency_samples_total +=
        retire_gcss_qni_pending_direct_queue_residency_samples_total;
    perf.retire_gcss_qni_pending_direct_queue_residency_cycles_max = std::max<uint64_t>(
        perf.retire_gcss_qni_pending_direct_queue_residency_cycles_max,
        retire_gcss_qni_pending_direct_queue_residency_cycles_max);
    perf.retire_gcss_qni_vlf_front_inflight_full_cycles_total += retire_gcss_qni_vlf_front_inflight_full_cycles_total;
    perf.retire_gcss_qni_vlf_front_inflight_full_blocked_edges_total += retire_gcss_qni_vlf_front_inflight_full_blocked_edges_total;
    perf.retire_gcss_qni_vlf_front_waiting_issue_cycles_total += retire_gcss_qni_vlf_front_waiting_issue_cycles_total;
    perf.retire_gcss_qni_vlf_front_waiting_issue_blocked_edges_total += retire_gcss_qni_vlf_front_waiting_issue_blocked_edges_total;
    perf.retire_gcss_qni_pending_younger_ahead_cycles_total += retire_gcss_qni_pending_younger_ahead_cycles_total;
    perf.retire_gcss_qni_pending_younger_ahead_blocked_edges_total += retire_gcss_qni_pending_younger_ahead_blocked_edges_total;
    perf.retire_gcss_qni_pending_front_inflight_full_cycles_total += retire_gcss_qni_pending_front_inflight_full_cycles_total;
    perf.retire_gcss_qni_pending_front_inflight_full_blocked_edges_total += retire_gcss_qni_pending_front_inflight_full_blocked_edges_total;
    perf.retire_gcss_qni_pending_front_waiting_tick_cycles_total += retire_gcss_qni_pending_front_waiting_tick_cycles_total;
    perf.retire_gcss_qni_pending_front_waiting_tick_blocked_edges_total += retire_gcss_qni_pending_front_waiting_tick_blocked_edges_total;
    perf.retire_gcss_qni_unknown_cycles_total += retire_gcss_qni_unknown_cycles_total;
    perf.retire_gcss_qni_unknown_blocked_edges_total += retire_gcss_qni_unknown_blocked_edges_total;
    perf.retire_begin_apply_windows_total += retire_begin_apply_windows_total;
    perf.retire_begin_apply_prev_edges_total += retire_begin_apply_prev_edges_total;
    perf.retire_begin_apply_outstanding_carryin_total += retire_begin_apply_outstanding_carryin_total;
    perf.retire_begin_apply_outstanding_carryin_windows_total +=
        retire_begin_apply_outstanding_carryin_windows_total;
    perf.retire_begin_apply_loader_not_ready_windows_total +=
        retire_begin_apply_loader_not_ready_windows_total;
    perf.retire_edge_retire_registered_total += retire_edge_retire_registered_total;
    perf.retire_edge_retire_retired_total += retire_edge_retire_retired_total;
    perf.retire_end_scatter_gcss_vlf_issue_queue_residual_total +=
        retire_end_scatter_gcss_vlf_issue_queue_residual_total;
    perf.retire_end_scatter_pending_direct_reads_residual_total +=
        retire_end_scatter_pending_direct_reads_residual_total;
    perf.retire_end_scatter_outstanding_residual_total +=
        retire_end_scatter_outstanding_residual_total;
    perf.retire_end_scatter_residual_work_windows_total +=
        retire_end_scatter_residual_work_windows_total;
    perf.retire_gcss_vlf_issue_prepare_total += retire_gcss_vlf_issue_prepare_total;
    perf.retire_gcss_vlf_issue_edges_total += retire_gcss_vlf_issue_edges_total;
    perf.retire_gcss_vlf_issue_reorder_trigger_total += retire_gcss_vlf_issue_reorder_trigger_total;
    perf.retire_gcss_vlf_issue_line_groups_total += retire_gcss_vlf_issue_line_groups_total;
    perf.retire_ready_queue_peak = std::max<uint64_t>(perf.retire_ready_queue_peak, retire_ready_queue_peak);
    perf.retire_unblock_events_total += retire_unblock_events_total;
}

void MultiCorePE::recordPulseAgendaObservability(
    uint32_t seq,
    uint64_t candidates_total,
    uint64_t accepted_total,
    uint64_t rejected_total,
    uint64_t reject_gate_total,
    uint64_t correctness_ready_blocked_cycles_total,
    uint64_t correctness_scoreboard_occupancy_peak,
    uint64_t shared_service_hits_total,
    uint64_t shared_service_misses_total,
    uint64_t region_service_entries_peak,
    uint64_t ready_fanout_total,
    uint64_t rowdescriptor_ready_transition_total,
    uint64_t rowdescriptor_join_ready_total,
    uint64_t rowdescriptor_ready_join_shortcut_candidates_total,
    uint64_t rowdescriptor_ready_join_shortcut_taken_total,
    uint64_t rowdescriptor_ready_join_shortcut_late_release_taken_total,
    uint64_t rowdescriptor_ready_join_shortcut_deferred_live_park_total,
    uint64_t rowdescriptor_ready_join_shortcut_deferred_live_apply_total,
    uint64_t rowdescriptor_owner_form_deferred_park_total,
    uint64_t rowdescriptor_owner_form_deferred_activate_total,
    uint64_t rowdescriptor_ready_join_shortcut_blocked_not_ready_total,
    uint64_t rowdescriptor_ready_join_shortcut_blocked_owner_form_total,
    uint64_t rowdescriptor_ready_join_shortcut_blocked_join_live_total,
    uint64_t rowdescriptor_ready_join_shortcut_blocked_other_total,
    uint64_t rowdescriptor_ready_join_shortcut_release_deferred_total,
    uint64_t rowdescriptor_ready_join_shortcut_apply_complete_total,
    uint64_t rowdescriptor_ready_join_shortcut_release_forwarded_total,
    uint64_t rowdescriptor_ready_join_shortcut_release_missing_total,
    uint64_t rowdescriptor_ready_join_descriptor_elide_total,
    uint64_t rowdescriptor_ready_join_lines_elide_total,
    uint64_t rowdescriptor_owner_first_service_elide_join_live_total,
    uint64_t rowdescriptor_owner_first_service_elide_join_ready_total,
    uint64_t rowdescriptor_owner_first_service_elide_late_join_total,
    uint64_t actual_gate_enable_false_total,
    uint64_t actual_gate_window_zero_total,
    uint64_t actual_gate_line_too_small_total,
    uint64_t actual_gate_taken_total,
    uint64_t frontier_windows_total,
    uint64_t frontier_lines_exported_total,
    uint64_t frontier_overlap_lines_total,
    uint64_t frontier_overlap_peer_total,
    uint64_t frontier_max_exported_per_window,
    uint64_t prebase_lookup_owner_fill_total,
    uint64_t prebase_lookup_shared_hits_total,
    uint64_t prebase_lookup_entries_peak) {
    (void)seq;
    if (candidates_total && stat_pulse_agenda_candidates_total_) {
        stat_pulse_agenda_candidates_total_->addData(candidates_total);
    }
    if (accepted_total && stat_pulse_agenda_accepted_total_) {
        stat_pulse_agenda_accepted_total_->addData(accepted_total);
    }
    if (rejected_total && stat_pulse_agenda_rejected_total_) {
        stat_pulse_agenda_rejected_total_->addData(rejected_total);
    }
    if (reject_gate_total && stat_pulse_agenda_reject_gate_total_) {
        stat_pulse_agenda_reject_gate_total_->addData(reject_gate_total);
    }
    if (correctness_ready_blocked_cycles_total &&
        stat_pulse_correctness_ready_blocked_cycles_total_) {
        stat_pulse_correctness_ready_blocked_cycles_total_->addData(
            correctness_ready_blocked_cycles_total);
    }
    if (correctness_scoreboard_occupancy_peak &&
        stat_pulse_correctness_scoreboard_occupancy_peak_) {
        stat_pulse_correctness_scoreboard_occupancy_peak_->addData(
            correctness_scoreboard_occupancy_peak);
    }
    if (shared_service_hits_total && stat_pulse_shared_service_hits_total_) {
        stat_pulse_shared_service_hits_total_->addData(shared_service_hits_total);
    }
    if (shared_service_misses_total && stat_pulse_shared_service_misses_total_) {
        stat_pulse_shared_service_misses_total_->addData(shared_service_misses_total);
    }
    if (region_service_entries_peak && stat_pulse_region_service_entries_peak_) {
        stat_pulse_region_service_entries_peak_->addData(region_service_entries_peak);
    }
    if (ready_fanout_total && stat_pulse_ready_fanout_total_) {
        stat_pulse_ready_fanout_total_->addData(ready_fanout_total);
    }
    if (rowdescriptor_ready_transition_total &&
        stat_pulse_rowdescriptor_ready_transition_total_) {
        stat_pulse_rowdescriptor_ready_transition_total_->addData(
            rowdescriptor_ready_transition_total);
    }
    if (rowdescriptor_join_ready_total &&
        stat_pulse_rowdescriptor_join_ready_total_) {
        stat_pulse_rowdescriptor_join_ready_total_->addData(
            rowdescriptor_join_ready_total);
    }
    if (rowdescriptor_ready_join_shortcut_candidates_total &&
        stat_pulse_rowdescriptor_ready_join_shortcut_candidates_total_) {
        stat_pulse_rowdescriptor_ready_join_shortcut_candidates_total_->addData(
            rowdescriptor_ready_join_shortcut_candidates_total);
    }
    if (rowdescriptor_ready_join_shortcut_taken_total &&
        stat_pulse_rowdescriptor_ready_join_shortcut_taken_total_) {
        stat_pulse_rowdescriptor_ready_join_shortcut_taken_total_->addData(
            rowdescriptor_ready_join_shortcut_taken_total);
    }
    if (rowdescriptor_ready_join_shortcut_late_release_taken_total &&
        stat_pulse_rowdescriptor_ready_join_shortcut_late_release_taken_total_) {
        stat_pulse_rowdescriptor_ready_join_shortcut_late_release_taken_total_->addData(
            rowdescriptor_ready_join_shortcut_late_release_taken_total);
    }
    if (rowdescriptor_ready_join_shortcut_deferred_live_park_total &&
        stat_pulse_rowdescriptor_ready_join_shortcut_deferred_live_park_total_) {
        stat_pulse_rowdescriptor_ready_join_shortcut_deferred_live_park_total_->addData(
            rowdescriptor_ready_join_shortcut_deferred_live_park_total);
    }
    if (rowdescriptor_ready_join_shortcut_deferred_live_apply_total &&
        stat_pulse_rowdescriptor_ready_join_shortcut_deferred_live_apply_total_) {
        stat_pulse_rowdescriptor_ready_join_shortcut_deferred_live_apply_total_->addData(
            rowdescriptor_ready_join_shortcut_deferred_live_apply_total);
    }
    if (rowdescriptor_owner_form_deferred_park_total &&
        stat_pulse_rowdescriptor_owner_form_deferred_park_total_) {
        stat_pulse_rowdescriptor_owner_form_deferred_park_total_->addData(
            rowdescriptor_owner_form_deferred_park_total);
    }
    if (rowdescriptor_owner_form_deferred_activate_total &&
        stat_pulse_rowdescriptor_owner_form_deferred_activate_total_) {
        stat_pulse_rowdescriptor_owner_form_deferred_activate_total_->addData(
            rowdescriptor_owner_form_deferred_activate_total);
    }
    if (rowdescriptor_ready_join_shortcut_blocked_not_ready_total &&
        stat_pulse_rowdescriptor_ready_join_shortcut_blocked_not_ready_total_) {
        stat_pulse_rowdescriptor_ready_join_shortcut_blocked_not_ready_total_->addData(
            rowdescriptor_ready_join_shortcut_blocked_not_ready_total);
    }
    if (rowdescriptor_ready_join_shortcut_blocked_owner_form_total &&
        stat_pulse_rowdescriptor_ready_join_shortcut_blocked_owner_form_total_) {
        stat_pulse_rowdescriptor_ready_join_shortcut_blocked_owner_form_total_->addData(
            rowdescriptor_ready_join_shortcut_blocked_owner_form_total);
    }
    if (rowdescriptor_ready_join_shortcut_blocked_join_live_total &&
        stat_pulse_rowdescriptor_ready_join_shortcut_blocked_join_live_total_) {
        stat_pulse_rowdescriptor_ready_join_shortcut_blocked_join_live_total_->addData(
            rowdescriptor_ready_join_shortcut_blocked_join_live_total);
    }
    if (rowdescriptor_ready_join_shortcut_blocked_other_total &&
        stat_pulse_rowdescriptor_ready_join_shortcut_blocked_other_total_) {
        stat_pulse_rowdescriptor_ready_join_shortcut_blocked_other_total_->addData(
            rowdescriptor_ready_join_shortcut_blocked_other_total);
    }
    if (rowdescriptor_ready_join_shortcut_release_deferred_total &&
        stat_pulse_rowdescriptor_ready_join_shortcut_release_deferred_total_) {
        stat_pulse_rowdescriptor_ready_join_shortcut_release_deferred_total_->addData(
            rowdescriptor_ready_join_shortcut_release_deferred_total);
    }
    if (rowdescriptor_ready_join_shortcut_apply_complete_total &&
        stat_pulse_rowdescriptor_ready_join_shortcut_apply_complete_total_) {
        stat_pulse_rowdescriptor_ready_join_shortcut_apply_complete_total_->addData(
            rowdescriptor_ready_join_shortcut_apply_complete_total);
    }
    if (rowdescriptor_ready_join_shortcut_release_forwarded_total &&
        stat_pulse_rowdescriptor_ready_join_shortcut_release_forwarded_total_) {
        stat_pulse_rowdescriptor_ready_join_shortcut_release_forwarded_total_->addData(
            rowdescriptor_ready_join_shortcut_release_forwarded_total);
    }
    if (rowdescriptor_ready_join_shortcut_release_missing_total &&
        stat_pulse_rowdescriptor_ready_join_shortcut_release_missing_total_) {
        stat_pulse_rowdescriptor_ready_join_shortcut_release_missing_total_->addData(
            rowdescriptor_ready_join_shortcut_release_missing_total);
    }
    if (rowdescriptor_ready_join_descriptor_elide_total &&
        stat_pulse_rowdescriptor_ready_join_descriptor_elide_total_) {
        stat_pulse_rowdescriptor_ready_join_descriptor_elide_total_->addData(
            rowdescriptor_ready_join_descriptor_elide_total);
    }
    if (rowdescriptor_ready_join_lines_elide_total &&
        stat_pulse_rowdescriptor_ready_join_lines_elide_total_) {
        stat_pulse_rowdescriptor_ready_join_lines_elide_total_->addData(
            rowdescriptor_ready_join_lines_elide_total);
    }
    if (rowdescriptor_owner_first_service_elide_join_live_total &&
        stat_pulse_rowdescriptor_owner_first_service_elide_join_live_total_) {
        stat_pulse_rowdescriptor_owner_first_service_elide_join_live_total_->addData(
            rowdescriptor_owner_first_service_elide_join_live_total);
    }
    if (rowdescriptor_owner_first_service_elide_join_ready_total &&
        stat_pulse_rowdescriptor_owner_first_service_elide_join_ready_total_) {
        stat_pulse_rowdescriptor_owner_first_service_elide_join_ready_total_->addData(
            rowdescriptor_owner_first_service_elide_join_ready_total);
    }
    if (rowdescriptor_owner_first_service_elide_late_join_total &&
        stat_pulse_rowdescriptor_owner_first_service_elide_late_join_total_) {
        stat_pulse_rowdescriptor_owner_first_service_elide_late_join_total_->addData(
            rowdescriptor_owner_first_service_elide_late_join_total);
    }
    if (actual_gate_enable_false_total && stat_pulse_actual_gate_enable_false_total_) {
        stat_pulse_actual_gate_enable_false_total_->addData(actual_gate_enable_false_total);
    }
    if (actual_gate_window_zero_total && stat_pulse_actual_gate_window_zero_total_) {
        stat_pulse_actual_gate_window_zero_total_->addData(actual_gate_window_zero_total);
    }
    if (actual_gate_line_too_small_total && stat_pulse_actual_gate_line_too_small_total_) {
        stat_pulse_actual_gate_line_too_small_total_->addData(actual_gate_line_too_small_total);
    }
    if (actual_gate_taken_total && stat_pulse_actual_gate_taken_total_) {
        stat_pulse_actual_gate_taken_total_->addData(actual_gate_taken_total);
    }
    if (frontier_windows_total && stat_pulse_frontier_windows_total_) {
        stat_pulse_frontier_windows_total_->addData(frontier_windows_total);
    }
    if (frontier_lines_exported_total && stat_pulse_frontier_lines_exported_total_) {
        stat_pulse_frontier_lines_exported_total_->addData(frontier_lines_exported_total);
    }
    if (frontier_overlap_lines_total && stat_pulse_frontier_overlap_lines_total_) {
        stat_pulse_frontier_overlap_lines_total_->addData(frontier_overlap_lines_total);
    }
    if (frontier_overlap_peer_total && stat_pulse_frontier_overlap_peer_total_) {
        stat_pulse_frontier_overlap_peer_total_->addData(frontier_overlap_peer_total);
    }
    if (frontier_max_exported_per_window && stat_pulse_frontier_max_exported_per_window_) {
        stat_pulse_frontier_max_exported_per_window_->addData(
            frontier_max_exported_per_window);
    }
    if (prebase_lookup_owner_fill_total &&
        stat_pulse_prebase_lookup_owner_fill_total_) {
        stat_pulse_prebase_lookup_owner_fill_total_->addData(
            prebase_lookup_owner_fill_total);
    }
    if (prebase_lookup_shared_hits_total &&
        stat_pulse_prebase_lookup_shared_hits_total_) {
        stat_pulse_prebase_lookup_shared_hits_total_->addData(
            prebase_lookup_shared_hits_total);
    }
    if (prebase_lookup_entries_peak &&
        stat_pulse_prebase_lookup_entries_peak_) {
        stat_pulse_prebase_lookup_entries_peak_->addData(
            prebase_lookup_entries_peak);
    }
}

void MultiCorePE::recordPulsePodRowdescriptorObservability(
    uint32_t seq,
    uint64_t owner_form_total,
    uint64_t owner_hit_total,
    uint64_t join_live_total,
    uint64_t join_ready_total,
    uint64_t ready_transition_total,
    uint64_t late_join_total,
    uint64_t guard_total,
    uint64_t guard_disabled_total,
    uint64_t guard_missing_metadata_plane_total,
    uint64_t guard_missing_owner_table_total,
    uint64_t guard_zero_pod_count_total,
    uint64_t guard_window_zero_total,
    uint64_t guard_invalid_cfg_pod_total,
    uint64_t reject_total,
    uint64_t attempted_total,
    uint64_t potential_private_service_elide_total,
    uint64_t owner_first_issue_deferred_total,
    uint64_t owner_first_private_issue_avoided_total,
    uint64_t owner_first_service_elide_total) {
    (void)seq;
    if (owner_form_total && stat_pulse_pod_rowdescriptor_owner_form_total_) {
        stat_pulse_pod_rowdescriptor_owner_form_total_->addData(owner_form_total);
    }
    if (owner_hit_total && stat_pulse_pod_rowdescriptor_owner_hit_total_) {
        stat_pulse_pod_rowdescriptor_owner_hit_total_->addData(owner_hit_total);
    }
    if (join_live_total && stat_pulse_pod_rowdescriptor_join_live_total_) {
        stat_pulse_pod_rowdescriptor_join_live_total_->addData(join_live_total);
    }
    if (join_ready_total && stat_pulse_pod_rowdescriptor_join_ready_total_) {
        stat_pulse_pod_rowdescriptor_join_ready_total_->addData(join_ready_total);
    }
    if (ready_transition_total &&
        stat_pulse_pod_rowdescriptor_ready_transition_total_) {
        stat_pulse_pod_rowdescriptor_ready_transition_total_->addData(
            ready_transition_total);
    }
    if (late_join_total && stat_pulse_pod_rowdescriptor_late_join_total_) {
        stat_pulse_pod_rowdescriptor_late_join_total_->addData(late_join_total);
    }
    if (guard_total && stat_pulse_pod_rowdescriptor_guard_total_) {
        stat_pulse_pod_rowdescriptor_guard_total_->addData(guard_total);
    }
    if (guard_disabled_total && stat_pulse_pod_rowdescriptor_guard_disabled_total_) {
        stat_pulse_pod_rowdescriptor_guard_disabled_total_->addData(
            guard_disabled_total);
    }
    if (guard_missing_metadata_plane_total &&
        stat_pulse_pod_rowdescriptor_guard_missing_metadata_plane_total_) {
        stat_pulse_pod_rowdescriptor_guard_missing_metadata_plane_total_->addData(
            guard_missing_metadata_plane_total);
    }
    if (guard_missing_owner_table_total &&
        stat_pulse_pod_rowdescriptor_guard_missing_owner_table_total_) {
        stat_pulse_pod_rowdescriptor_guard_missing_owner_table_total_->addData(
            guard_missing_owner_table_total);
    }
    if (guard_zero_pod_count_total &&
        stat_pulse_pod_rowdescriptor_guard_zero_pod_count_total_) {
        stat_pulse_pod_rowdescriptor_guard_zero_pod_count_total_->addData(
            guard_zero_pod_count_total);
    }
    if (guard_window_zero_total &&
        stat_pulse_pod_rowdescriptor_guard_window_zero_total_) {
        stat_pulse_pod_rowdescriptor_guard_window_zero_total_->addData(
            guard_window_zero_total);
    }
    if (guard_invalid_cfg_pod_total &&
        stat_pulse_pod_rowdescriptor_guard_invalid_cfg_pod_total_) {
        stat_pulse_pod_rowdescriptor_guard_invalid_cfg_pod_total_->addData(
            guard_invalid_cfg_pod_total);
    }
    if (reject_total && stat_pulse_pod_rowdescriptor_reject_total_) {
        stat_pulse_pod_rowdescriptor_reject_total_->addData(reject_total);
    }
    if (attempted_total && stat_pulse_pod_rowdescriptor_attempted_total_) {
        stat_pulse_pod_rowdescriptor_attempted_total_->addData(attempted_total);
    }
    if (potential_private_service_elide_total &&
        stat_pulse_pod_rowdescriptor_potential_private_service_elide_total_) {
        stat_pulse_pod_rowdescriptor_potential_private_service_elide_total_->addData(
            potential_private_service_elide_total);
    }
    if (owner_first_issue_deferred_total &&
        stat_pulse_pod_rowdescriptor_owner_first_issue_deferred_total_) {
        stat_pulse_pod_rowdescriptor_owner_first_issue_deferred_total_->addData(
            owner_first_issue_deferred_total);
    }
    if (owner_first_private_issue_avoided_total &&
        stat_pulse_pod_rowdescriptor_owner_first_private_issue_avoided_total_) {
        stat_pulse_pod_rowdescriptor_owner_first_private_issue_avoided_total_->addData(
            owner_first_private_issue_avoided_total);
    }
    if (owner_first_service_elide_total &&
        stat_pulse_pod_rowdescriptor_owner_first_service_elide_total_) {
        stat_pulse_pod_rowdescriptor_owner_first_service_elide_total_->addData(
            owner_first_service_elide_total);
    }
}

void MultiCorePE::recordPulseOsaMetadataTxnObservability(
    const WeightMemorySubsystem::PulseOsaMetadataTxnStats& stats) {
    if (stats.export_total && stat_pulse_metadata_txn_export_total_) {
        stat_pulse_metadata_txn_export_total_->addData(stats.export_total);
    }
    if (stats.owner_launch_total && stat_pulse_metadata_txn_owner_launch_total_) {
        stat_pulse_metadata_txn_owner_launch_total_->addData(stats.owner_launch_total);
    }
    if (stats.join_live_total && stat_pulse_metadata_txn_join_live_total_) {
        stat_pulse_metadata_txn_join_live_total_->addData(stats.join_live_total);
    }
    if (stats.join_ready_total && stat_pulse_metadata_txn_join_ready_total_) {
        stat_pulse_metadata_txn_join_ready_total_->addData(stats.join_ready_total);
    }
    if (stats.late_join_total && stat_pulse_metadata_txn_late_join_total_) {
        stat_pulse_metadata_txn_late_join_total_->addData(stats.late_join_total);
    }
    if (stats.ready_lease_hit_total && stat_pulse_metadata_txn_ready_lease_hit_total_) {
        stat_pulse_metadata_txn_ready_lease_hit_total_->addData(stats.ready_lease_hit_total);
    }
    if (stats.ready_lease_expired_total &&
        stat_pulse_metadata_txn_ready_lease_expired_total_) {
        stat_pulse_metadata_txn_ready_lease_expired_total_->addData(
            stats.ready_lease_expired_total);
    }
    if (stats.envelope_size_sum_total &&
        stat_pulse_metadata_txn_envelope_size_sum_total_) {
        stat_pulse_metadata_txn_envelope_size_sum_total_->addData(
            stats.envelope_size_sum_total);
    }
    if (stats.frontier_observed_total &&
        stat_pulse_metadata_frontier_observed_total_) {
        stat_pulse_metadata_frontier_observed_total_->addData(
            stats.frontier_observed_total);
    }
    if (stats.frontier_same_window_reobserve_total &&
        stat_pulse_metadata_frontier_same_window_reobserve_total_) {
        stat_pulse_metadata_frontier_same_window_reobserve_total_->addData(
            stats.frontier_same_window_reobserve_total);
    }
    if (stats.frontier_owner_form_candidate_total &&
        stat_pulse_metadata_frontier_owner_form_candidate_total_) {
        stat_pulse_metadata_frontier_owner_form_candidate_total_->addData(
            stats.frontier_owner_form_candidate_total);
    }
    if (stats.frontier_join_ready_candidate_total &&
        stat_pulse_metadata_frontier_join_ready_candidate_total_) {
        stat_pulse_metadata_frontier_join_ready_candidate_total_->addData(
            stats.frontier_join_ready_candidate_total);
    }
    if (stats.frontier_premphf_base_observed_total &&
        stat_pulse_metadata_frontier_premphf_base_observed_total_) {
        stat_pulse_metadata_frontier_premphf_base_observed_total_->addData(
            stats.frontier_premphf_base_observed_total);
    }
    if (stats.frontier_premphf_base_same_window_reobserve_total &&
        stat_pulse_metadata_frontier_premphf_base_same_window_reobserve_total_) {
        stat_pulse_metadata_frontier_premphf_base_same_window_reobserve_total_->addData(
            stats.frontier_premphf_base_same_window_reobserve_total);
    }
    if (stats.frontier_premphf_base_owner_form_candidate_total &&
        stat_pulse_metadata_frontier_premphf_base_owner_form_candidate_total_) {
        stat_pulse_metadata_frontier_premphf_base_owner_form_candidate_total_->addData(
            stats.frontier_premphf_base_owner_form_candidate_total);
    }
    if (stats.frontier_premphf_base_join_ready_candidate_total &&
        stat_pulse_metadata_frontier_premphf_base_join_ready_candidate_total_) {
        stat_pulse_metadata_frontier_premphf_base_join_ready_candidate_total_->addData(
            stats.frontier_premphf_base_join_ready_candidate_total);
    }
    if (stats.frontier_premphf_band_observed_total &&
        stat_pulse_metadata_frontier_premphf_band_observed_total_) {
        stat_pulse_metadata_frontier_premphf_band_observed_total_->addData(
            stats.frontier_premphf_band_observed_total);
    }
    if (stats.frontier_premphf_band_same_window_reobserve_total &&
        stat_pulse_metadata_frontier_premphf_band_same_window_reobserve_total_) {
        stat_pulse_metadata_frontier_premphf_band_same_window_reobserve_total_->addData(
            stats.frontier_premphf_band_same_window_reobserve_total);
    }
    if (stats.frontier_premphf_band_owner_form_candidate_total &&
        stat_pulse_metadata_frontier_premphf_band_owner_form_candidate_total_) {
        stat_pulse_metadata_frontier_premphf_band_owner_form_candidate_total_->addData(
            stats.frontier_premphf_band_owner_form_candidate_total);
    }
    if (stats.frontier_premphf_band_join_ready_candidate_total &&
        stat_pulse_metadata_frontier_premphf_band_join_ready_candidate_total_) {
        stat_pulse_metadata_frontier_premphf_band_join_ready_candidate_total_->addData(
            stats.frontier_premphf_band_join_ready_candidate_total);
    }
    if (stats.frontier_idx2row_observed_total &&
        stat_pulse_metadata_frontier_idx2row_observed_total_) {
        stat_pulse_metadata_frontier_idx2row_observed_total_->addData(
            stats.frontier_idx2row_observed_total);
    }
    if (stats.frontier_idx2row_same_window_reobserve_total &&
        stat_pulse_metadata_frontier_idx2row_same_window_reobserve_total_) {
        stat_pulse_metadata_frontier_idx2row_same_window_reobserve_total_->addData(
            stats.frontier_idx2row_same_window_reobserve_total);
    }
    if (stats.frontier_idx2row_owner_form_candidate_total &&
        stat_pulse_metadata_frontier_idx2row_owner_form_candidate_total_) {
        stat_pulse_metadata_frontier_idx2row_owner_form_candidate_total_->addData(
            stats.frontier_idx2row_owner_form_candidate_total);
    }
    if (stats.frontier_idx2row_join_ready_candidate_total &&
        stat_pulse_metadata_frontier_idx2row_join_ready_candidate_total_) {
        stat_pulse_metadata_frontier_idx2row_join_ready_candidate_total_->addData(
            stats.frontier_idx2row_join_ready_candidate_total);
    }
    if (stats.frontier_rowindex_observed_total &&
        stat_pulse_metadata_frontier_rowindex_observed_total_) {
        stat_pulse_metadata_frontier_rowindex_observed_total_->addData(
            stats.frontier_rowindex_observed_total);
    }
    if (stats.frontier_rowindex_same_window_reobserve_total &&
        stat_pulse_metadata_frontier_rowindex_same_window_reobserve_total_) {
        stat_pulse_metadata_frontier_rowindex_same_window_reobserve_total_->addData(
            stats.frontier_rowindex_same_window_reobserve_total);
    }
    if (stats.frontier_rowindex_owner_form_candidate_total &&
        stat_pulse_metadata_frontier_rowindex_owner_form_candidate_total_) {
        stat_pulse_metadata_frontier_rowindex_owner_form_candidate_total_->addData(
            stats.frontier_rowindex_owner_form_candidate_total);
    }
    if (stats.frontier_rowindex_join_ready_candidate_total &&
        stat_pulse_metadata_frontier_rowindex_join_ready_candidate_total_) {
        stat_pulse_metadata_frontier_rowindex_join_ready_candidate_total_->addData(
            stats.frontier_rowindex_join_ready_candidate_total);
    }
}

void MultiCorePE::recordAtlasObjectObservability(
    const WeightMemorySubsystem::ExperimentalPeAtlasObjectCensus& census,
    const WeightMemorySubsystem::ExperimentalPeAtlasRowIndexLifecycleLedger& rowindex,
    const WeightMemorySubsystem::ExperimentalPeAtlasIdx2RowLifecycleLedger& idx2row,
    const WeightMemorySubsystem::ExperimentalPeAtlasPreMphfBaseProxyLedger& premphf_base,
    const WeightMemorySubsystem::ExperimentalPeAtlasPreMphfBandProxyLedger& premphf_band,
    const WeightMemorySubsystem::ExperimentalPeAtlasPhaseStats& phase,
    const WeightMemorySubsystem::PeInternalPodStats& pod_runtime) {
    const auto add_atlas_stat = [this](const char* name, uint64_t value) {
        auto it = atlas_object_plane_stats_.find(name);
        if (it == atlas_object_plane_stats_.end() || it->second == nullptr) return;
        if (value == 0u) return;
        it->second->addData(value);
    };
    const auto add_atlas_stat_sampled = [this](const char* name, uint64_t value) {
        auto it = atlas_object_plane_stats_.find(name);
        if (it == atlas_object_plane_stats_.end() || it->second == nullptr) return;
        it->second->addData(value);
    };

    add_atlas_stat(
        "atlas_census_premphf_base_frontier_events_total",
        census.premphf_base.frontier_events_total);
    add_atlas_stat(
        "atlas_census_premphf_base_producer_events_total",
        census.premphf_base.producer_events_total);
    add_atlas_stat(
        "atlas_census_premphf_base_gate_events_total",
        census.premphf_base.gate_events_total);
    add_atlas_stat(
        "atlas_census_premphf_base_service_events_total",
        census.premphf_base.service_events_total);
    add_atlas_stat(
        "atlas_census_premphf_band_frontier_events_total",
        census.premphf_band.frontier_events_total);
    add_atlas_stat(
        "atlas_census_premphf_band_producer_events_total",
        census.premphf_band.producer_events_total);
    add_atlas_stat(
        "atlas_census_premphf_band_gate_events_total",
        census.premphf_band.gate_events_total);
    add_atlas_stat(
        "atlas_census_premphf_band_service_events_total",
        census.premphf_band.service_events_total);
    add_atlas_stat(
        "atlas_census_idx2row_frontier_events_total",
        census.idx2row.frontier_events_total);
    add_atlas_stat(
        "atlas_census_idx2row_producer_events_total",
        census.idx2row.producer_events_total);
    add_atlas_stat(
        "atlas_census_idx2row_gate_events_total",
        census.idx2row.gate_events_total);
    add_atlas_stat(
        "atlas_census_idx2row_service_events_total",
        census.idx2row.service_events_total);
    add_atlas_stat(
        "atlas_census_rowindex_frontier_events_total",
        census.rowindex.frontier_events_total);
    add_atlas_stat(
        "atlas_census_rowindex_producer_events_total",
        census.rowindex.producer_events_total);
    add_atlas_stat(
        "atlas_census_rowindex_gate_events_total",
        census.rowindex.gate_events_total);
    add_atlas_stat(
        "atlas_census_rowindex_service_events_total",
        census.rowindex.service_events_total);
    add_atlas_stat(
        "atlas_census_rowdescriptor_frontier_events_total",
        census.rowdescriptor.frontier_events_total);
    add_atlas_stat(
        "atlas_census_rowdescriptor_producer_events_total",
        census.rowdescriptor.producer_events_total);
    add_atlas_stat(
        "atlas_census_rowdescriptor_gate_events_total",
        census.rowdescriptor.gate_events_total);
    add_atlas_stat(
        "atlas_census_rowdescriptor_service_events_total",
        census.rowdescriptor.service_events_total);

    add_atlas_stat(
        "atlas_proxy_idx2row_materialize_total",
        idx2row.materialize_total);
    add_atlas_stat(
        "atlas_proxy_idx2row_publicize_total",
        idx2row.publicize_total);
    add_atlas_stat(
        "atlas_proxy_idx2row_owner_form_total",
        idx2row.owner_form_total);
    add_atlas_stat(
        "atlas_proxy_idx2row_join_live_total",
        idx2row.join_live_total);
    add_atlas_stat(
        "atlas_proxy_idx2row_join_ready_total",
        idx2row.join_ready_total);
    add_atlas_stat(
        "atlas_proxy_idx2row_ready_total",
        idx2row.ready_total);
    add_atlas_stat(
        "atlas_proxy_idx2row_release_total",
        idx2row.release_total);
    add_atlas_stat(
        "atlas_proxy_idx2row_release_missing_total",
        idx2row.release_missing_total);
    add_atlas_stat(
        "atlas_proxy_idx2row_fallback_total",
        idx2row.fallback_total);

    add_atlas_stat(
        "atlas_proxy_premphf_base_materialize_total",
        premphf_base.materialize_total);
    add_atlas_stat(
        "atlas_proxy_premphf_base_publicize_total",
        premphf_base.publicize_total);
    add_atlas_stat(
        "atlas_proxy_premphf_base_owner_form_total",
        premphf_base.owner_form_total);
    add_atlas_stat(
        "atlas_proxy_premphf_base_shared_hit_total",
        premphf_base.shared_hit_total);
    add_atlas_stat(
        "atlas_proxy_premphf_base_lookup_ready_total",
        premphf_base.lookup_ready_total);
    add_atlas_stat(
        "atlas_proxy_premphf_base_proxy_only_gap_total",
        premphf_base.proxy_only_gap_total);

    add_atlas_stat(
        "atlas_proxy_premphf_band_materialize_total",
        premphf_band.materialize_total);
    add_atlas_stat(
        "atlas_proxy_premphf_band_publicize_total",
        premphf_band.publicize_total);
    add_atlas_stat(
        "atlas_proxy_premphf_band_owner_form_candidate_total",
        premphf_band.owner_form_candidate_total);
    add_atlas_stat(
        "atlas_proxy_premphf_band_join_ready_candidate_total",
        premphf_band.join_ready_candidate_total);
    add_atlas_stat(
        "atlas_proxy_premphf_band_zero_service_total",
        premphf_band.zero_service_total);

    add_atlas_stat(
        "atlas_proxy_rowindex_materialize_total",
        rowindex.materialize_total);
    add_atlas_stat(
        "atlas_proxy_rowindex_publicize_total",
        rowindex.publicize_total);
    add_atlas_stat(
        "atlas_proxy_rowindex_owner_form_total",
        rowindex.owner_form_total);
    add_atlas_stat(
        "atlas_proxy_rowindex_join_live_total",
        rowindex.join_live_total);
    add_atlas_stat(
        "atlas_proxy_rowindex_join_ready_total",
        rowindex.join_ready_total);
    add_atlas_stat(
        "atlas_proxy_rowindex_ready_total",
        rowindex.ready_total);
    add_atlas_stat(
        "atlas_proxy_rowindex_release_total",
        rowindex.release_total);
    add_atlas_stat(
        "atlas_proxy_rowindex_release_missing_total",
        rowindex.release_missing_total);
    add_atlas_stat(
        "atlas_proxy_rowindex_fallback_total",
        rowindex.fallback_total);

    const bool rowindex_requested =
        (census.rowindex.frontier_events_total +
         census.rowindex.producer_events_total +
         census.rowindex.gate_events_total +
         census.rowindex.service_events_total +
         rowindex.materialize_total +
         rowindex.publicize_total +
         rowindex.owner_form_total +
         rowindex.join_live_total +
         rowindex.join_ready_total +
         rowindex.ready_total +
         rowindex.release_total +
         rowindex.release_missing_total +
         rowindex.fallback_total) > 0u;
    const bool rowindex_effective =
        rowindex_requested &&
        pulse_osa_enable_ &&
        pulse_osa_metadata_txn_enable_ &&
        (pulse_osa_metadata_object_mask_bits_ &
         PeLocalServiceObjectTable::kMetadataKindMaskRowIndex) != 0u &&
        pe_internal_pod_enable_cfg_ &&
        pe_internal_pod_metadata_enable_cfg_ &&
        pe_internal_pod_owner_enable_cfg_ &&
        pe_local_service_object_table_ != nullptr;
    const bool rowindex_constructed =
        rowindex.materialize_total > 0u ||
        rowindex.publicize_total > 0u ||
        rowindex.owner_form_total > 0u ||
        rowindex.join_live_total > 0u ||
        rowindex.join_ready_total > 0u ||
        rowindex.ready_total > 0u;
    atlas_rowindex_requested_seen_ =
        atlas_rowindex_requested_seen_ || rowindex_requested;
    atlas_rowindex_effective_seen_ =
        atlas_rowindex_effective_seen_ || rowindex_effective;
    atlas_rowindex_constructed_seen_ =
        atlas_rowindex_constructed_seen_ || rowindex_constructed;

    add_atlas_stat_sampled(
        "atlas_control_runtime_produced_frontier_export_total",
        pod_runtime.frontier_export_total);
    add_atlas_stat_sampled(
        "atlas_control_runtime_produced_owner_announce_total",
        pod_runtime.owner_alloc_total);
    add_atlas_stat_sampled(
        "atlas_control_runtime_produced_join_request_total",
        pod_runtime.join_request_total);
    add_atlas_stat_sampled(
        "atlas_control_runtime_produced_ready_fanout_total",
        pod_runtime.service_ready_fanout_total);
    add_atlas_stat_sampled(
        "atlas_control_runtime_produced_join_reject_total",
        pod_runtime.owner_reject_total + pod_runtime.join_reject_total);
    atlas_control_runtime_produced_any_nonzero_last_ =
        pod_runtime.frontier_export_total > 0u ||
        pod_runtime.owner_alloc_total > 0u ||
        pod_runtime.join_request_total > 0u ||
        pod_runtime.service_ready_fanout_total > 0u ||
        (pod_runtime.owner_reject_total + pod_runtime.join_reject_total) > 0u;

    add_atlas_stat(
        "atlas_phase_local_begin_gather_total",
        phase.local_begin_gather_total);
    add_atlas_stat(
        "atlas_phase_local_first_touch_after_gather_open_total",
        phase.local_first_touch_after_gather_open_total);
    add_atlas_stat(
        "atlas_phase_local_touch_without_gather_open_total",
        phase.local_touch_without_gather_open_total);
    add_atlas_stat(
        "atlas_phase_local_touch_during_apply_total",
        phase.local_touch_during_apply_total);
    add_atlas_stat(
        "atlas_phase_local_begin_apply_total",
        phase.local_begin_apply_total);
    add_atlas_stat(
        "atlas_phase_local_begin_apply_with_touch_total",
        phase.local_begin_apply_with_touch_total);
    add_atlas_stat(
        "atlas_phase_local_begin_apply_with_pending_rowidx_total",
        phase.local_begin_apply_with_pending_rowidx_total);
    add_atlas_stat(
        "atlas_phase_local_begin_apply_without_pending_rowidx_total",
        phase.local_begin_apply_without_pending_rowidx_total);
    add_atlas_stat(
        "atlas_phase_local_end_scatter_total",
        phase.local_end_scatter_total);
}

void MultiCorePE::recordCoreStepRetireStat(int core_id,
                                           uint32_t seq,
                                           uint64_t retire_global_hol_cycles_total,
                                           uint64_t retire_ready_but_blocked_edges_total,
                                           uint64_t retire_per_post_progress_total,
                                           uint64_t retire_wait_cycles_total,
                                           uint64_t retire_wait_cycles_due_to_hol_total,
                                           uint64_t retire_wait_cycles_due_to_barrier_total,
                                           uint64_t retire_wait_cycles_due_to_not_ready_total,
                                           uint64_t retire_samepost_blocked_edges_total,
                                           uint64_t retire_crosspost_blocked_edges_total,
                                           uint64_t retire_policy_loss_cycles_total,
                                           uint64_t retire_policy_loss_edges_total,
                                           uint64_t retire_shadow_per_post_recoverable_cycles_total,
                                           uint64_t retire_shadow_per_post_recoverable_edges_total,
                                           uint64_t retire_shadow_per_post_ready_posts_peak,
                                           uint64_t retire_shadow_per_post_committable_edges_peak,
                                           uint64_t retire_head_hol_cycles_dense_total,
                                           uint64_t retire_head_hol_cycles_cache_total,
                                           uint64_t retire_head_hol_cycles_miss_total,
                                           uint64_t retire_head_hol_cycles_bcsr_total,
                                           uint64_t retire_head_hol_cycles_bcsr_file_total,
                                           uint64_t retire_head_hol_cycles_gcss_total,
                                           uint64_t retire_head_blocked_edges_dense_total,
                                           uint64_t retire_head_blocked_edges_cache_total,
                                           uint64_t retire_head_blocked_edges_miss_total,
                                           uint64_t retire_head_blocked_edges_bcsr_total,
                                           uint64_t retire_head_blocked_edges_bcsr_file_total,
                                           uint64_t retire_head_blocked_edges_gcss_total,
                                           uint64_t retire_gcss_head_queued_not_issued_cycles_total,
                                           uint64_t retire_gcss_qni_head_wait_episodes_total,
                                           uint64_t retire_gcss_qni_head_wait_cycles_max,
                                           uint64_t retire_gcss_head_queued_not_issued_blocked_edges_total,
                                           uint64_t retire_gcss_head_issued_wait_resp_cycles_total,
                                           uint64_t retire_gcss_head_issued_wait_resp_blocked_edges_total,
                                           uint64_t retire_gcss_resp_ready_but_hol_cycles_total,
                                           uint64_t retire_gcss_resp_ready_but_hol_blocked_edges_total,
                                           uint64_t retire_gcss_qni_loader_not_ready_cycles_total,
                                           uint64_t retire_gcss_qni_loader_not_ready_blocked_edges_total,
                                           uint64_t retire_gcss_qni_weight_sram_stall_cycles_total,
                                           uint64_t retire_gcss_qni_weight_sram_stall_blocked_edges_total,
                                           uint64_t retire_gcss_qni_vlf_younger_ahead_cycles_total,
                                           uint64_t retire_gcss_qni_vlf_younger_ahead_blocked_edges_total,
                                           uint64_t retire_gcss_qni_vlf_younger_ahead_depth_total,
                                           uint64_t retire_gcss_qni_vlf_younger_ahead_depth_samples_total,
                                           uint64_t retire_gcss_qni_vlf_younger_ahead_depth_max,
                                           uint64_t retire_gcss_qni_issue_deferred_total,
                                           uint64_t retire_gcss_qni_pending_direct_queue_residency_cycles_total,
                                           uint64_t retire_gcss_qni_pending_direct_queue_residency_samples_total,
                                           uint64_t retire_gcss_qni_pending_direct_queue_residency_cycles_max,
                                           uint64_t retire_gcss_qni_vlf_front_inflight_full_cycles_total,
                                           uint64_t retire_gcss_qni_vlf_front_inflight_full_blocked_edges_total,
                                           uint64_t retire_gcss_qni_vlf_front_waiting_issue_cycles_total,
                                           uint64_t retire_gcss_qni_vlf_front_waiting_issue_blocked_edges_total,
                                           uint64_t retire_gcss_qni_pending_younger_ahead_cycles_total,
                                           uint64_t retire_gcss_qni_pending_younger_ahead_blocked_edges_total,
                                           uint64_t retire_gcss_qni_pending_front_inflight_full_cycles_total,
                                           uint64_t retire_gcss_qni_pending_front_inflight_full_blocked_edges_total,
                                           uint64_t retire_gcss_qni_pending_front_waiting_tick_cycles_total,
                                           uint64_t retire_gcss_qni_pending_front_waiting_tick_blocked_edges_total,
                                           uint64_t retire_gcss_qni_unknown_cycles_total,
                                           uint64_t retire_gcss_qni_unknown_blocked_edges_total,
                                           uint64_t retire_begin_apply_windows_total,
                                           uint64_t retire_begin_apply_prev_edges_total,
                                           uint64_t retire_begin_apply_outstanding_carryin_total,
                                           uint64_t retire_begin_apply_outstanding_carryin_windows_total,
                                           uint64_t retire_begin_apply_loader_not_ready_windows_total,
                                           uint64_t retire_edge_retire_registered_total,
                                           uint64_t retire_edge_retire_retired_total,
                                           uint64_t retire_end_scatter_gcss_vlf_issue_queue_residual_total,
                                           uint64_t retire_end_scatter_pending_direct_reads_residual_total,
                                           uint64_t retire_end_scatter_outstanding_residual_total,
                                           uint64_t retire_end_scatter_residual_work_windows_total,
                                           uint64_t retire_gcss_vlf_issue_prepare_total,
                                           uint64_t retire_gcss_vlf_issue_edges_total,
                                           uint64_t retire_gcss_vlf_issue_reorder_trigger_total,
                                           uint64_t retire_gcss_vlf_issue_line_groups_total,
                                           uint64_t retire_ready_queue_peak,
                                           uint64_t retire_unblock_events_total) {
    if (seq == 0 || core_id < 0) return;
    auto& perf = getOrCreateCoreStepPerf_(core_id, seq);
    perf.retire_global_hol_cycles_total += retire_global_hol_cycles_total;
    perf.retire_ready_but_blocked_edges_total += retire_ready_but_blocked_edges_total;
    perf.retire_per_post_progress_total += retire_per_post_progress_total;
    perf.retire_wait_cycles_total += retire_wait_cycles_total;
    perf.retire_wait_cycles_due_to_hol_total += retire_wait_cycles_due_to_hol_total;
    perf.retire_wait_cycles_due_to_barrier_total += retire_wait_cycles_due_to_barrier_total;
    perf.retire_wait_cycles_due_to_not_ready_total += retire_wait_cycles_due_to_not_ready_total;
    perf.retire_samepost_blocked_edges_total += retire_samepost_blocked_edges_total;
    perf.retire_crosspost_blocked_edges_total += retire_crosspost_blocked_edges_total;
    perf.retire_policy_loss_cycles_total += retire_policy_loss_cycles_total;
    perf.retire_policy_loss_edges_total += retire_policy_loss_edges_total;
    perf.retire_shadow_per_post_recoverable_cycles_total += retire_shadow_per_post_recoverable_cycles_total;
    perf.retire_shadow_per_post_recoverable_edges_total += retire_shadow_per_post_recoverable_edges_total;
    perf.retire_shadow_per_post_ready_posts_peak =
        std::max<uint64_t>(perf.retire_shadow_per_post_ready_posts_peak, retire_shadow_per_post_ready_posts_peak);
    perf.retire_shadow_per_post_committable_edges_peak =
        std::max<uint64_t>(perf.retire_shadow_per_post_committable_edges_peak, retire_shadow_per_post_committable_edges_peak);
    perf.retire_head_hol_cycles_dense_total += retire_head_hol_cycles_dense_total;
    perf.retire_head_hol_cycles_cache_total += retire_head_hol_cycles_cache_total;
    perf.retire_head_hol_cycles_miss_total += retire_head_hol_cycles_miss_total;
    perf.retire_head_hol_cycles_bcsr_total += retire_head_hol_cycles_bcsr_total;
    perf.retire_head_hol_cycles_bcsr_file_total += retire_head_hol_cycles_bcsr_file_total;
    perf.retire_head_hol_cycles_gcss_total += retire_head_hol_cycles_gcss_total;
    perf.retire_head_blocked_edges_dense_total += retire_head_blocked_edges_dense_total;
    perf.retire_head_blocked_edges_cache_total += retire_head_blocked_edges_cache_total;
    perf.retire_head_blocked_edges_miss_total += retire_head_blocked_edges_miss_total;
    perf.retire_head_blocked_edges_bcsr_total += retire_head_blocked_edges_bcsr_total;
    perf.retire_head_blocked_edges_bcsr_file_total += retire_head_blocked_edges_bcsr_file_total;
    perf.retire_head_blocked_edges_gcss_total += retire_head_blocked_edges_gcss_total;
    perf.retire_gcss_head_queued_not_issued_cycles_total += retire_gcss_head_queued_not_issued_cycles_total;
    perf.retire_gcss_qni_head_wait_episodes_total += retire_gcss_qni_head_wait_episodes_total;
    perf.retire_gcss_qni_head_wait_cycles_max = std::max<uint64_t>(
        perf.retire_gcss_qni_head_wait_cycles_max, retire_gcss_qni_head_wait_cycles_max);
    perf.retire_gcss_head_queued_not_issued_blocked_edges_total += retire_gcss_head_queued_not_issued_blocked_edges_total;
    perf.retire_gcss_head_issued_wait_resp_cycles_total += retire_gcss_head_issued_wait_resp_cycles_total;
    perf.retire_gcss_head_issued_wait_resp_blocked_edges_total += retire_gcss_head_issued_wait_resp_blocked_edges_total;
    perf.retire_gcss_resp_ready_but_hol_cycles_total += retire_gcss_resp_ready_but_hol_cycles_total;
    perf.retire_gcss_resp_ready_but_hol_blocked_edges_total += retire_gcss_resp_ready_but_hol_blocked_edges_total;
    perf.retire_gcss_qni_loader_not_ready_cycles_total += retire_gcss_qni_loader_not_ready_cycles_total;
    perf.retire_gcss_qni_loader_not_ready_blocked_edges_total += retire_gcss_qni_loader_not_ready_blocked_edges_total;
    perf.retire_gcss_qni_weight_sram_stall_cycles_total += retire_gcss_qni_weight_sram_stall_cycles_total;
    perf.retire_gcss_qni_weight_sram_stall_blocked_edges_total += retire_gcss_qni_weight_sram_stall_blocked_edges_total;
    perf.retire_gcss_qni_vlf_younger_ahead_cycles_total += retire_gcss_qni_vlf_younger_ahead_cycles_total;
    perf.retire_gcss_qni_vlf_younger_ahead_blocked_edges_total += retire_gcss_qni_vlf_younger_ahead_blocked_edges_total;
    perf.retire_gcss_qni_vlf_younger_ahead_depth_total += retire_gcss_qni_vlf_younger_ahead_depth_total;
    perf.retire_gcss_qni_vlf_younger_ahead_depth_samples_total += retire_gcss_qni_vlf_younger_ahead_depth_samples_total;
    perf.retire_gcss_qni_vlf_younger_ahead_depth_max = std::max<uint64_t>(
        perf.retire_gcss_qni_vlf_younger_ahead_depth_max,
        retire_gcss_qni_vlf_younger_ahead_depth_max);
    perf.retire_gcss_qni_issue_deferred_total += retire_gcss_qni_issue_deferred_total;
    perf.retire_gcss_qni_pending_direct_queue_residency_cycles_total +=
        retire_gcss_qni_pending_direct_queue_residency_cycles_total;
    perf.retire_gcss_qni_pending_direct_queue_residency_samples_total +=
        retire_gcss_qni_pending_direct_queue_residency_samples_total;
    perf.retire_gcss_qni_pending_direct_queue_residency_cycles_max = std::max<uint64_t>(
        perf.retire_gcss_qni_pending_direct_queue_residency_cycles_max,
        retire_gcss_qni_pending_direct_queue_residency_cycles_max);
    perf.retire_gcss_qni_vlf_front_inflight_full_cycles_total += retire_gcss_qni_vlf_front_inflight_full_cycles_total;
    perf.retire_gcss_qni_vlf_front_inflight_full_blocked_edges_total += retire_gcss_qni_vlf_front_inflight_full_blocked_edges_total;
    perf.retire_gcss_qni_vlf_front_waiting_issue_cycles_total += retire_gcss_qni_vlf_front_waiting_issue_cycles_total;
    perf.retire_gcss_qni_vlf_front_waiting_issue_blocked_edges_total += retire_gcss_qni_vlf_front_waiting_issue_blocked_edges_total;
    perf.retire_gcss_qni_pending_younger_ahead_cycles_total += retire_gcss_qni_pending_younger_ahead_cycles_total;
    perf.retire_gcss_qni_pending_younger_ahead_blocked_edges_total += retire_gcss_qni_pending_younger_ahead_blocked_edges_total;
    perf.retire_gcss_qni_pending_front_inflight_full_cycles_total += retire_gcss_qni_pending_front_inflight_full_cycles_total;
    perf.retire_gcss_qni_pending_front_inflight_full_blocked_edges_total += retire_gcss_qni_pending_front_inflight_full_blocked_edges_total;
    perf.retire_gcss_qni_pending_front_waiting_tick_cycles_total += retire_gcss_qni_pending_front_waiting_tick_cycles_total;
    perf.retire_gcss_qni_pending_front_waiting_tick_blocked_edges_total += retire_gcss_qni_pending_front_waiting_tick_blocked_edges_total;
    perf.retire_gcss_qni_unknown_cycles_total += retire_gcss_qni_unknown_cycles_total;
    perf.retire_gcss_qni_unknown_blocked_edges_total += retire_gcss_qni_unknown_blocked_edges_total;
    perf.retire_begin_apply_windows_total += retire_begin_apply_windows_total;
    perf.retire_begin_apply_prev_edges_total += retire_begin_apply_prev_edges_total;
    perf.retire_begin_apply_outstanding_carryin_total += retire_begin_apply_outstanding_carryin_total;
    perf.retire_begin_apply_outstanding_carryin_windows_total +=
        retire_begin_apply_outstanding_carryin_windows_total;
    perf.retire_begin_apply_loader_not_ready_windows_total +=
        retire_begin_apply_loader_not_ready_windows_total;
    perf.retire_edge_retire_registered_total += retire_edge_retire_registered_total;
    perf.retire_edge_retire_retired_total += retire_edge_retire_retired_total;
    perf.retire_end_scatter_gcss_vlf_issue_queue_residual_total +=
        retire_end_scatter_gcss_vlf_issue_queue_residual_total;
    perf.retire_end_scatter_pending_direct_reads_residual_total +=
        retire_end_scatter_pending_direct_reads_residual_total;
    perf.retire_end_scatter_outstanding_residual_total +=
        retire_end_scatter_outstanding_residual_total;
    perf.retire_end_scatter_residual_work_windows_total +=
        retire_end_scatter_residual_work_windows_total;
    perf.retire_gcss_vlf_issue_prepare_total += retire_gcss_vlf_issue_prepare_total;
    perf.retire_gcss_vlf_issue_edges_total += retire_gcss_vlf_issue_edges_total;
    perf.retire_gcss_vlf_issue_reorder_trigger_total += retire_gcss_vlf_issue_reorder_trigger_total;
    perf.retire_gcss_vlf_issue_line_groups_total += retire_gcss_vlf_issue_line_groups_total;
    perf.retire_ready_queue_peak = std::max<uint64_t>(perf.retire_ready_queue_peak, retire_ready_queue_peak);
    perf.retire_unblock_events_total += retire_unblock_events_total;
}

void MultiCorePE::recordStepApplyScatter(uint32_t seq,
                                         uint64_t acc_updates,
                                         uint64_t posts_touched,
                                         uint64_t spikes_emitted,
                                         uint64_t hwm_bytes,
                                         uint64_t spill_records,
                                         uint64_t spilled_bytes) {
    if (seq == 0) return;
    auto& perf = getOrCreateStepPerf_(seq);
    perf.acc_updates += acc_updates;
    perf.posts_touched += posts_touched;
    perf.spill_records += spill_records;
    perf.spilled_bytes += spilled_bytes;
    perf.hwm_bytes_max = std::max<uint64_t>(perf.hwm_bytes_max, hwm_bytes);
    perf.scatter_spikes_emitted += spikes_emitted;
}

void MultiCorePE::recordCoreStepApplyScatter(int core_id,
                                             uint32_t seq,
                                             uint64_t acc_updates,
                                             uint64_t posts_touched,
                                             uint64_t spikes_emitted,
                                             uint64_t hwm_bytes,
                                             uint64_t spill_records,
                                             uint64_t spilled_bytes) {
    if (seq == 0 || core_id < 0) return;
    auto& perf = getOrCreateCoreStepPerf_(core_id, seq);
    perf.acc_updates += acc_updates;
    perf.posts_touched += posts_touched;
    perf.spill_records += spill_records;
    perf.spilled_bytes += spilled_bytes;
    perf.hwm_bytes_max = std::max<uint64_t>(perf.hwm_bytes_max, hwm_bytes);
    perf.scatter_spikes_emitted += spikes_emitted;
}

void MultiCorePE::accumulateExperimentalNocPrefetchStats(const ExperimentalNocRowidxPeStats& rowidx) {
#define ADD_STAT_VALUE(stat_ptr, value) \
    do { \
        if ((stat_ptr) != nullptr) (stat_ptr)->addData((value)); \
    } while (0)

    ADD_STAT_VALUE(stat_exp_noc_rowidx_prefetch_rows_total_, rowidx.prefetch_rows_total);
    ADD_STAT_VALUE(stat_exp_noc_rowidx_prefetch_bytes_total_, rowidx.prefetch_bytes_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_prefetch_complete_inflight_miss_total_,
        rowidx.prefetch_complete_inflight_miss_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_prefetch_complete_zero_waiters_total_,
        rowidx.prefetch_complete_zero_waiters_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_prefetch_complete_waiters_total_,
        rowidx.prefetch_complete_waiters_total);
    ADD_STAT_VALUE(stat_exp_noc_rowidx_prefetch_rows_deferred_total_, rowidx.prefetch_rows_deferred_total);
    ADD_STAT_VALUE(stat_exp_noc_rowidx_prefetch_rows_failed_total_, rowidx.prefetch_rows_failed_total);
    ADD_STAT_VALUE(stat_exp_noc_rowidx_cache_hits_total_, rowidx.cache_hits_total);
    ADD_STAT_VALUE(stat_exp_noc_rowidx_cache_misses_total_, rowidx.cache_misses_total);
    ADD_STAT_VALUE(stat_exp_noc_rowidx_cache_fills_total_, rowidx.cache_fills_total);
    ADD_STAT_VALUE(stat_exp_noc_rowidx_cache_full_drop_total_, rowidx.cache_full_drop_total);
    ADD_STAT_VALUE(stat_exp_noc_rowidx_cache_entries_final_, rowidx.cache_entries_final);
    ADD_STAT_VALUE(stat_exp_noc_rowidx_bulk_fill_total_, rowidx.bulk_fill_total);
    ADD_STAT_VALUE(stat_exp_noc_rowidx_bulk_rows_cached_total_, rowidx.bulk_rows_cached_total);
    ADD_STAT_VALUE(stat_exp_noc_rowidx_bulk_waiters_resolved_total_, rowidx.bulk_waiters_resolved_total);
    ADD_STAT_VALUE(stat_exp_noc_rowidx_touch_rows_total_, rowidx.touch_rows_total);
    ADD_STAT_VALUE(stat_exp_noc_rowidx_touch_events_total_, rowidx.touch_events_total);
    ADD_STAT_VALUE(stat_exp_noc_rowidx_rows_filtered_cold_total_, rowidx.rows_filtered_cold_total);
    ADD_STAT_VALUE(stat_exp_noc_rowidx_budget_ticks_total_, rowidx.budget_ticks_total);
    ADD_STAT_VALUE(stat_exp_noc_rowidx_budget_effective_total_, rowidx.budget_effective_total);
    ADD_STAT_VALUE(stat_exp_noc_rowidx_budget_adapt_ticks_total_, rowidx.budget_adapt_ticks_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_ready_transition_apply_promote_cached_total_,
        rowidx.ready_transition_apply_promote_cached_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_ready_signal_rowindex_response_total_,
        rowidx.ready_signal_rowindex_response_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_ready_transition_rowindex_response_total_,
        rowidx.ready_transition_rowindex_response_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_ready_signal_prefetch_response_total_,
        rowidx.ready_signal_prefetch_response_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_ready_transition_prefetch_response_total_,
        rowidx.ready_transition_prefetch_response_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_ready_signal_rowindex_response_inflight_waiters_total_,
        rowidx.ready_signal_rowindex_response_inflight_waiters_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_ready_transition_rowindex_response_inflight_waiters_total_,
        rowidx.ready_transition_rowindex_response_inflight_waiters_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_ready_signal_rowindex_response_inflight_zero_waiters_total_,
        rowidx.ready_signal_rowindex_response_inflight_zero_waiters_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_ready_transition_rowindex_response_inflight_zero_waiters_total_,
        rowidx.ready_transition_rowindex_response_inflight_zero_waiters_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_ready_signal_rowindex_response_noninflight_prefetch_only_total_,
        rowidx.ready_signal_rowindex_response_noninflight_prefetch_only_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_ready_transition_rowindex_response_noninflight_prefetch_only_total_,
        rowidx.ready_transition_rowindex_response_noninflight_prefetch_only_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_ready_signal_prefetch_response_inflight_waiters_total_,
        rowidx.ready_signal_prefetch_response_inflight_waiters_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_ready_transition_prefetch_response_inflight_waiters_total_,
        rowidx.ready_transition_prefetch_response_inflight_waiters_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_ready_signal_prefetch_response_inflight_zero_waiters_total_,
        rowidx.ready_signal_prefetch_response_inflight_zero_waiters_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_ready_transition_prefetch_response_inflight_zero_waiters_total_,
        rowidx.ready_transition_prefetch_response_inflight_zero_waiters_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_ready_signal_prefetch_response_noninflight_prefetch_only_total_,
        rowidx.ready_signal_prefetch_response_noninflight_prefetch_only_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_ready_transition_prefetch_response_noninflight_prefetch_only_total_,
        rowidx.ready_transition_prefetch_response_noninflight_prefetch_only_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_detached_demand_join_total_,
        rowidx.detached_demand_join_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_detached_demand_waiters_resolved_total_,
        rowidx.detached_demand_waiters_resolved_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_detached_demand_fallback_zero_total_,
        rowidx.detached_demand_fallback_zero_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_detached_demand_ready_signal_total_,
        rowidx.detached_demand_ready_signal_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_detached_demand_ready_transition_total_,
        rowidx.detached_demand_ready_transition_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_ready_bypass_experimental_cache_hit_total_,
        rowidx.ready_bypass_experimental_cache_hit_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_ready_bypass_rowindex_get_hit_total_,
        rowidx.ready_bypass_rowindex_get_hit_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_close_attempt_total_,
        rowidx.close_attempt_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_close_attempt_active_owner_total_,
        rowidx.close_attempt_active_owner_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_close_attempt_already_pending_total_,
        rowidx.close_attempt_already_pending_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_close_attempt_not_active_total_,
        rowidx.close_attempt_not_active_total);
    ADD_STAT_VALUE(
        stat_exp_noc_rowidx_close_attempt_not_owner_total_,
        rowidx.close_attempt_not_owner_total);

#undef ADD_STAT_VALUE
}

void MultiCorePE::accumulateSynapseReadStats(uint64_t gcss_lookup_hit_total,
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
                                             uint64_t weight_idx_sram_bank_peak_accesses_per_tick,
                                             uint64_t weight_idx_sram_energy_read_pj_total,
                                             uint64_t weight_idx_sram_energy_write_pj_total,
                                             uint64_t weight_idx_lookup_total,
                                             uint64_t weight_idx_lookup_idx2_total,
                                             uint64_t weight_l0_sram_reads_total,
                                             uint64_t weight_l0_sram_writes_total,
                                             uint64_t weight_l0_sram_bytes_read_total,
                                             uint64_t weight_l0_sram_bytes_write_total,
                                             uint64_t weight_l0_sram_bank_conflict_ticks_total,
                                             uint64_t weight_l0_sram_predicted_extra_cycles_total,
                                             uint64_t weight_l0_sram_resident_bytes_peak,
                                             uint64_t weight_l0_sram_bank_peak_accesses_per_tick,
                                             uint64_t weight_l0_sram_energy_read_pj_total,
                                             uint64_t weight_l0_sram_energy_write_pj_total,
                                             uint64_t weight_sram_enforced_stall_cycles_total,
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
                                             uint64_t core_state_sram_bank_peak_accesses_per_tick,
                                             uint64_t core_state_sram_energy_read_pj_total,
                                             uint64_t core_state_sram_energy_write_pj_total,
                                             uint64_t core_state_sram_stall_cycles_total,
                                             uint64_t gas_retire_global_hol_cycles_total,
                                             uint64_t gas_retire_ready_but_blocked_edges_total,
                                             uint64_t gas_retire_per_post_progress_total,
                                             uint64_t gas_retire_samepost_blocked_edges_total,
                                             uint64_t gas_retire_crosspost_blocked_edges_total,
                                             uint64_t gas_retire_policy_loss_cycles_total,
                                             uint64_t gas_retire_policy_loss_edges_total,
                                             uint64_t gas_retire_shadow_per_post_recoverable_cycles_total,
                                             uint64_t gas_retire_shadow_per_post_recoverable_edges_total,
                                             uint64_t gas_retire_shadow_per_post_ready_posts_peak,
                                             uint64_t gas_retire_shadow_per_post_committable_edges_peak) {
    const auto add_atlas_storage_stat = [this](const char* name, uint64_t value) {
        auto it = atlas_object_plane_stats_.find(name);
        if (it == atlas_object_plane_stats_.end() || it->second == nullptr) return;
        it->second->addData(value);
    };
    if (stat_gcss_lookup_hit_total_) stat_gcss_lookup_hit_total_->addData(gcss_lookup_hit_total);
    if (stat_gcss_lookup_miss_total_) stat_gcss_lookup_miss_total_->addData(gcss_lookup_miss_total);
    if (stat_weight_read_dense_reqs_total_) stat_weight_read_dense_reqs_total_->addData(dense_reqs_total);
    if (stat_weight_read_dense_bytes_total_) stat_weight_read_dense_bytes_total_->addData(dense_bytes_total);
    if (stat_weight_read_rowptr_reqs_total_) stat_weight_read_rowptr_reqs_total_->addData(rowptr_reqs_total);
    if (stat_weight_read_rowptr_bytes_total_) stat_weight_read_rowptr_bytes_total_->addData(rowptr_bytes_total);
    if (stat_weight_read_colidx_reqs_total_) stat_weight_read_colidx_reqs_total_->addData(colidx_reqs_total);
    if (stat_weight_read_colidx_bytes_total_) stat_weight_read_colidx_bytes_total_->addData(colidx_bytes_total);
    if (stat_weight_read_blockdata_reqs_total_) stat_weight_read_blockdata_reqs_total_->addData(blockdata_reqs_total);
    if (stat_weight_read_blockdata_bytes_total_) stat_weight_read_blockdata_bytes_total_->addData(blockdata_bytes_total);
    if (stat_weight_read_gcss_reqs_total_) stat_weight_read_gcss_reqs_total_->addData(gcss_reqs_total);
    if (stat_weight_read_gcss_bytes_total_) stat_weight_read_gcss_bytes_total_->addData(gcss_bytes_total);
    if (stat_weight_idx_sram_reads_total_) stat_weight_idx_sram_reads_total_->addData(weight_idx_sram_reads_total);
    if (stat_weight_idx_sram_writes_total_) stat_weight_idx_sram_writes_total_->addData(weight_idx_sram_writes_total);
    if (stat_weight_idx_sram_bytes_read_total_) stat_weight_idx_sram_bytes_read_total_->addData(weight_idx_sram_bytes_read_total);
    if (stat_weight_idx_sram_bytes_write_total_) stat_weight_idx_sram_bytes_write_total_->addData(weight_idx_sram_bytes_write_total);
    if (stat_weight_idx_sram_bank_conflict_ticks_total_) stat_weight_idx_sram_bank_conflict_ticks_total_->addData(weight_idx_sram_bank_conflict_ticks_total);
    if (stat_weight_idx_sram_predicted_extra_cycles_total_) stat_weight_idx_sram_predicted_extra_cycles_total_->addData(weight_idx_sram_predicted_extra_cycles_total);
    if (stat_weight_idx_sram_resident_bytes_peak_) stat_weight_idx_sram_resident_bytes_peak_->addData(weight_idx_sram_resident_bytes_peak);
    if (stat_weight_idx_sram_bank_peak_accesses_per_tick_) stat_weight_idx_sram_bank_peak_accesses_per_tick_->addData(weight_idx_sram_bank_peak_accesses_per_tick);
    if (stat_weight_idx_sram_energy_read_pj_total_) stat_weight_idx_sram_energy_read_pj_total_->addData(weight_idx_sram_energy_read_pj_total);
    if (stat_weight_idx_sram_energy_write_pj_total_) stat_weight_idx_sram_energy_write_pj_total_->addData(weight_idx_sram_energy_write_pj_total);
    if (stat_weight_idx_lookup_total_) stat_weight_idx_lookup_total_->addData(weight_idx_lookup_total);
    if (stat_weight_idx_lookup_idx2_total_) stat_weight_idx_lookup_idx2_total_->addData(weight_idx_lookup_idx2_total);
    add_atlas_storage_stat(
        "atlas_storage_map_weight_idx_private_reads_total",
        weight_idx_sram_reads_total);
    add_atlas_storage_stat(
        "atlas_storage_map_weight_idx_private_resident_bytes_peak",
        weight_idx_sram_resident_bytes_peak);
    if (stat_weight_l0_sram_reads_total_) stat_weight_l0_sram_reads_total_->addData(weight_l0_sram_reads_total);
    if (stat_weight_l0_sram_writes_total_) stat_weight_l0_sram_writes_total_->addData(weight_l0_sram_writes_total);
    if (stat_weight_l0_sram_bytes_read_total_) stat_weight_l0_sram_bytes_read_total_->addData(weight_l0_sram_bytes_read_total);
    if (stat_weight_l0_sram_bytes_write_total_) stat_weight_l0_sram_bytes_write_total_->addData(weight_l0_sram_bytes_write_total);
    if (stat_weight_l0_sram_bank_conflict_ticks_total_) stat_weight_l0_sram_bank_conflict_ticks_total_->addData(weight_l0_sram_bank_conflict_ticks_total);
    if (stat_weight_l0_sram_predicted_extra_cycles_total_) stat_weight_l0_sram_predicted_extra_cycles_total_->addData(weight_l0_sram_predicted_extra_cycles_total);
    if (stat_weight_l0_sram_resident_bytes_peak_) stat_weight_l0_sram_resident_bytes_peak_->addData(weight_l0_sram_resident_bytes_peak);
    if (stat_weight_l0_sram_bank_peak_accesses_per_tick_) stat_weight_l0_sram_bank_peak_accesses_per_tick_->addData(weight_l0_sram_bank_peak_accesses_per_tick);
    if (stat_weight_l0_sram_energy_read_pj_total_) stat_weight_l0_sram_energy_read_pj_total_->addData(weight_l0_sram_energy_read_pj_total);
    if (stat_weight_l0_sram_energy_write_pj_total_) stat_weight_l0_sram_energy_write_pj_total_->addData(weight_l0_sram_energy_write_pj_total);
    if (stat_weight_sram_enforced_stall_cycles_total_) stat_weight_sram_enforced_stall_cycles_total_->addData(weight_sram_enforced_stall_cycles_total);
    if (stat_weight_l0_lookup_total_) stat_weight_l0_lookup_total_->addData(weight_l0_lookup_total);
    if (stat_weight_l0_hit_total_) stat_weight_l0_hit_total_->addData(weight_l0_hit_total);
    if (stat_weight_l0_fill_total_) stat_weight_l0_fill_total_->addData(weight_l0_fill_total);
    if (stat_weight_l0_evict_total_) stat_weight_l0_evict_total_->addData(weight_l0_evict_total);
    add_atlas_storage_stat(
        "atlas_storage_map_weight_value_private_reads_total",
        weight_l0_sram_reads_total);
    add_atlas_storage_stat(
        "atlas_storage_map_weight_value_private_resident_bytes_peak",
        weight_l0_sram_resident_bytes_peak);
    if (stat_core_state_sram_reads_total_) stat_core_state_sram_reads_total_->addData(core_state_sram_reads_total);
    if (stat_core_state_sram_writes_total_) stat_core_state_sram_writes_total_->addData(core_state_sram_writes_total);
    if (stat_core_state_sram_bytes_read_total_) stat_core_state_sram_bytes_read_total_->addData(core_state_sram_bytes_read_total);
    if (stat_core_state_sram_bytes_write_total_) stat_core_state_sram_bytes_write_total_->addData(core_state_sram_bytes_write_total);
    if (stat_core_state_sram_bank_conflict_ticks_total_) stat_core_state_sram_bank_conflict_ticks_total_->addData(core_state_sram_bank_conflict_ticks_total);
    if (stat_core_state_sram_predicted_extra_cycles_total_) stat_core_state_sram_predicted_extra_cycles_total_->addData(core_state_sram_predicted_extra_cycles_total);
    if (stat_core_state_sram_resident_bytes_peak_) stat_core_state_sram_resident_bytes_peak_->addData(core_state_sram_resident_bytes_peak);
    if (stat_core_state_sram_bank_peak_accesses_per_tick_) stat_core_state_sram_bank_peak_accesses_per_tick_->addData(core_state_sram_bank_peak_accesses_per_tick);
    if (stat_core_state_sram_energy_read_pj_total_) stat_core_state_sram_energy_read_pj_total_->addData(core_state_sram_energy_read_pj_total);
    if (stat_core_state_sram_energy_write_pj_total_) stat_core_state_sram_energy_write_pj_total_->addData(core_state_sram_energy_write_pj_total);
    if (stat_core_state_sram_stall_cycles_total_) stat_core_state_sram_stall_cycles_total_->addData(core_state_sram_stall_cycles_total);
    if (stat_gas_retire_global_hol_cycles_total_) stat_gas_retire_global_hol_cycles_total_->addData(gas_retire_global_hol_cycles_total);
    if (stat_gas_retire_ready_but_blocked_edges_total_) stat_gas_retire_ready_but_blocked_edges_total_->addData(gas_retire_ready_but_blocked_edges_total);
    if (stat_gas_retire_per_post_progress_total_) stat_gas_retire_per_post_progress_total_->addData(gas_retire_per_post_progress_total);
    if (stat_gas_retire_samepost_blocked_edges_total_) stat_gas_retire_samepost_blocked_edges_total_->addData(gas_retire_samepost_blocked_edges_total);
    if (stat_gas_retire_crosspost_blocked_edges_total_) stat_gas_retire_crosspost_blocked_edges_total_->addData(gas_retire_crosspost_blocked_edges_total);
    if (stat_gas_retire_policy_loss_cycles_total_) stat_gas_retire_policy_loss_cycles_total_->addData(gas_retire_policy_loss_cycles_total);
    if (stat_gas_retire_policy_loss_edges_total_) stat_gas_retire_policy_loss_edges_total_->addData(gas_retire_policy_loss_edges_total);
    if (stat_gas_retire_shadow_per_post_recoverable_cycles_total_) stat_gas_retire_shadow_per_post_recoverable_cycles_total_->addData(gas_retire_shadow_per_post_recoverable_cycles_total);
    if (stat_gas_retire_shadow_per_post_recoverable_edges_total_) stat_gas_retire_shadow_per_post_recoverable_edges_total_->addData(gas_retire_shadow_per_post_recoverable_edges_total);
    if (stat_gas_retire_shadow_per_post_ready_posts_peak_) stat_gas_retire_shadow_per_post_ready_posts_peak_->addData(gas_retire_shadow_per_post_ready_posts_peak);
    if (stat_gas_retire_shadow_per_post_committable_edges_peak_) stat_gas_retire_shadow_per_post_committable_edges_peak_->addData(gas_retire_shadow_per_post_committable_edges_peak);
}

