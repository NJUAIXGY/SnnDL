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
