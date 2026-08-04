// -*- c++ -*-

#include <sst/core/sst_config.h>

#include "workloads/tensor/TensorWorkload.h"

#include <map>
#include <string>

namespace SST { namespace SnnDL {

bool TensorWorkload::hasWork() const {
    if (!configured_) return false;
    // Before start_cycle is reached, still report "work pending" so barrier/drain policies
    // do not accidentally advance global steps without executing the workload.
    if (!started_) return true;
    if (!inflight_.empty()) return true;
    if (iter_active_) return true;
    if (cfg_.exec_mode == "program") {
        if (collectivePendingActive_()) return true;
        if (step_gated_ && !step_open_) return false;
        if (cfg_.iterations > 0 && program_iter_done_ >= cfg_.iterations) return false;
        if (program_ops_.empty()) return false;
        if (!program_loop_enable_ && program_pc_ >= program_ops_.size()) return false;
        return true;
    }
    if (step_gated_) {
        if (!step_open_) return false;
        if (cfg_.exec_mode != "program" && cfg_.iterations > 0 && iter_done_ >= cfg_.iterations) return false;
        return true;
    }
    if (cfg_.exec_mode != "program" && cfg_.iterations > 0 && iter_done_ >= cfg_.iterations) return false;
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
    stats["tensor_mem_read_latency_cycles_total"] = tensor_mem_read_latency_cycles_total_;
    stats["tensor_mem_read_latency_cycles_max"] = tensor_mem_read_latency_cycles_max_;
    stats["tensor_mem_read_latency_samples_total"] = tensor_mem_read_latency_samples_total_;
    stats["tensor_mem_write_latency_cycles_total"] = tensor_mem_write_latency_cycles_total_;
    stats["tensor_mem_write_latency_cycles_max"] = tensor_mem_write_latency_cycles_max_;
    stats["tensor_mem_write_latency_samples_total"] = tensor_mem_write_latency_samples_total_;
    stats["tensor_mem_row_hit_total"] = tensor_mem_row_hit_total_;
    stats["tensor_mem_row_miss_total"] = tensor_mem_row_miss_total_;
    stats["tensor_mem_row_conflict_total"] = tensor_mem_row_conflict_total_;
    stats["tensor_mem_bank_queue_full_total"] = tensor_mem_bank_queue_full_total_;
    stats["tensor_mem_bank_queue_wait_cycles_total"] = tensor_mem_bank_queue_wait_cycles_total_;
    stats["tensor_mem_sched_fifo_pick_total"] = tensor_mem_sched_fifo_pick_total_;
    stats["tensor_mem_sched_frfcfs_pick_total"] = tensor_mem_sched_frfcfs_pick_total_;
    stats["tensor_mem_cmd_act_total"] = tensor_mem_cmd_act_total_;
    stats["tensor_mem_cmd_pre_total"] = tensor_mem_cmd_pre_total_;
    stats["tensor_mem_cmd_rdwr_total"] = tensor_mem_cmd_rdwr_total_;
    stats["tensor_mem_row_service_cycles_total"] = tensor_mem_row_service_cycles_total_;
    stats["tensor_mem_refresh_block_cycles_total"] = tensor_mem_refresh_block_cycles_total_;
    stats["tensor_mem_proxy_delay_cycles_total"] = tensor_mem_proxy_delay_cycles_total_;
    stats["tensor_mem_proxy_delay_cycles_max"] = tensor_mem_proxy_delay_cycles_max_;
    stats["tensor_mem_bank_active_cycles_total"] = tensor_mem_bank_active_cycles_total_;
    stats["tensor_mem_cmd_queue_slots_total"] = tensor_mem_cmd_queue_slots_total_;
    stats["tensor_mem_cmd_queue_depth_max"] = tensor_mem_cmd_queue_depth_max_;
    stats["tensor_mem_cmd_bus_wait_cycles_total"] = tensor_mem_cmd_bus_wait_cycles_total_;
    stats["tensor_mem_cmd_bus_bg_switch_total"] = tensor_mem_cmd_bus_bg_switch_total_;
    stats["tensor_mem_cmd_issue_total"] = tensor_mem_cmd_issue_total_;
    stats["tensor_compute_cycles_total"] = tensor_compute_cycles_total_;
    stats["tensor_compute_math_cycles_total"] = tensor_compute_math_cycles_total_;
    stats["tensor_compute_pipeline_cycles_total"] = tensor_compute_pipeline_cycles_total_;
    stats["tensor_mxu_wavefront_cycles_total"] = tensor_mxu_wavefront_cycles_total_;
    stats["tensor_mxu_io_busy_cycles_total"] = tensor_mxu_io_busy_cycles_total_;
    stats["tensor_compute_precision_profile_id"] = tensor_compute_precision_profile_id_;
    stats["tensor_mac_ops_total"] = tensor_mac_ops_total_;
    stats["tensor_dma_stall_cycles_total"] = tensor_dma_stall_cycles_total_;
    stats["tensor_iter_cycles_total"] = tensor_iter_cycles_total_;
    stats["tensor_stall_dma_budget_cycles_total"] = tensor_stall_dma_budget_cycles_total_;
    stats["tensor_stall_dma_hbm_channel_budget_cycles_total"] = tensor_stall_dma_hbm_channel_budget_cycles_total_;
    stats["tensor_stall_mem_outstanding_cycles_total"] = tensor_stall_mem_outstanding_cycles_total_;
    stats["tensor_stall_wait_read_cycles_total"] = tensor_stall_wait_read_cycles_total_;
    stats["tensor_stall_wait_write_cycles_total"] = tensor_stall_wait_write_cycles_total_;
    stats["tensor_stall_collective_cycles_total"] = tensor_stall_collective_cycles_total_;
    stats["tensor_stall_onchip_capacity_cycles_total"] = tensor_stall_onchip_capacity_cycles_total_;
    stats["tensor_stall_onchip_port_cycles_total"] = tensor_stall_onchip_port_cycles_total_;
    stats["tensor_stall_onchip_bank_conflict_cycles_total"] = tensor_stall_onchip_bank_conflict_cycles_total_;
    stats["tensor_stall_spill_budget_cycles_total"] = tensor_stall_spill_budget_cycles_total_;
    stats["tensor_dma_cycles_total"] = tensor_dma_cycles_total_;
    stats["tensor_dram_bytes_total"] = tensor_dram_bytes_total_;
    stats["tensor_onchip_bytes_total"] = tensor_onchip_bytes_total_;
    stats["tensor_cfg_ub_bytes"] = cfg_.ub_bytes;
    stats["tensor_cfg_weight_bytes"] = cfg_.weight_bytes;
    stats["tensor_onchip_weight_occupancy_bytes_max"] = tensor_onchip_weight_occupancy_bytes_max_;
    stats["tensor_onchip_weight_bank_occupancy_bytes_max"] = tensor_onchip_weight_bank_occupancy_bytes_max_;
    stats["tensor_onchip_a_resident_tiles_max"] = tensor_onchip_a_resident_tiles_max_;
    stats["tensor_onchip_b_resident_tiles_max"] = tensor_onchip_b_resident_tiles_max_;
    stats["tensor_tile_count_total"] = tensor_tile_count_total_;
    stats["tensor_spill_bytes_total"] = tensor_spill_bytes_total_;
    stats["tensor_spill_pkts_total"] = tensor_spill_pkts_total_;
    stats["tensor_collective_bytes_sent_total"] = tensor_collective_bytes_sent_total_;
    stats["tensor_collective_bytes_recv_total"] = tensor_collective_bytes_recv_total_;
    stats["tensor_collective_pkts_sent_total"] = tensor_collective_pkts_sent_total_;
    stats["tensor_collective_pkts_recv_total"] = tensor_collective_pkts_recv_total_;
    stats["tensor_collective_cycles_total"] = tensor_collective_cycles_total_;
    stats["tensor_collective_pending_cycles_total"] = tensor_collective_pending_cycles_total_;
    stats["tensor_collective_issue_cycles_total"] = tensor_collective_issue_cycles_total_;
    stats["tensor_collective_chunk_groups_total"] = tensor_collective_chunk_groups_total_;
    stats["tensor_collective_ring_steps_total"] = tensor_collective_ring_steps_total_;
    stats["tensor_collective_2d_row_rs_steps_total"] = tensor_collective_2d_row_rs_steps_total_;
    stats["tensor_collective_2d_col_rs_steps_total"] = tensor_collective_2d_col_rs_steps_total_;
    stats["tensor_collective_2d_col_ag_steps_total"] = tensor_collective_2d_col_ag_steps_total_;
    stats["tensor_collective_2d_row_ag_steps_total"] = tensor_collective_2d_row_ag_steps_total_;
    stats["tensor_collective_2d_row_rs_bytes_sent_total"] = tensor_collective_2d_row_rs_bytes_sent_total_;
    stats["tensor_collective_2d_col_rs_bytes_sent_total"] = tensor_collective_2d_col_rs_bytes_sent_total_;
    stats["tensor_collective_2d_col_ag_bytes_sent_total"] = tensor_collective_2d_col_ag_bytes_sent_total_;
    stats["tensor_collective_2d_row_ag_bytes_sent_total"] = tensor_collective_2d_row_ag_bytes_sent_total_;
    stats["tensor_collective_reduce_wait_cycles_total"] = tensor_collective_reduce_wait_cycles_total_;
    stats["tensor_collective_2d_reduce_wait_cycles_total"] = tensor_collective_2d_reduce_wait_cycles_total_;
    stats["tensor_collective_epoch_done_total"] = tensor_collective_epoch_done_total_;
    stats["tensor_collective_epoch_latency_cycles_total"] = tensor_collective_epoch_latency_cycles_total_;
    stats["tensor_collective_epoch_latency_cycles_max"] = tensor_collective_epoch_latency_cycles_max_;
    // This is a config identifier, not a throughput/volume metric. Export it once to
    // keep summary consumers stable under PE/core aggregation (avoid scaling by core count).
    const bool is_leader_core = (rt_.node_id == 0 && rt_.core_id == 0);
    stats["tensor_collective_algo_id"] = is_leader_core ? tensor_collective_algo_id_ : 0ull;
    stats["tensor_collective_credit_stall_cycles_total"] = tensor_collective_credit_stall_cycles_total_;
    stats["tensor_collective_backpressure_stall_cycles_total"] = tensor_collective_backpressure_stall_cycles_total_;
    stats["tensor_collective_inflight_chunks_max"] = tensor_collective_inflight_chunks_max_;
    stats["tensor_collective_credit_return_pkts_sent_total"] = tensor_collective_credit_return_pkts_sent_total_;
    stats["tensor_collective_credit_return_pkts_recv_total"] = tensor_collective_credit_return_pkts_recv_total_;
    stats["tensor_collective_credit_return_orphan_total"] = tensor_collective_credit_return_orphan_total_;
    stats["tensor_collective_credit_return_dup_total"] = tensor_collective_credit_return_dup_total_;
    stats["tensor_collective_credit_return_latency_cycles_total"] = tensor_collective_credit_return_latency_cycles_total_;
    stats["tensor_collective_credit_return_latency_cycles_max"] = tensor_collective_credit_return_latency_cycles_max_;
    stats["tensor_bank_queue_occupancy_max"] = tensor_bank_queue_occupancy_max_;
    stats["tensor_stall_noc_budget_cycles_total"] = tensor_stall_noc_budget_cycles_total_;
    stats["tensor_overlap_compute_collective_cycles_total"] = tensor_overlap_compute_collective_cycles_total_;
    stats["tensor_overlap_compute_mem_cycles_total"] = tensor_overlap_compute_mem_cycles_total_;
    stats["tensor_pkt_sent_total"] = tensor_pkt_sent_total_;
    stats["tensor_pkt_recv_total"] = tensor_pkt_recv_total_;
    stats["tensor_pkt_bytes_sent_total"] = tensor_pkt_bytes_sent_total_;
    stats["tensor_pkt_bytes_recv_total"] = tensor_pkt_bytes_recv_total_;
    stats["tensor_vector_cycles_total"] = tensor_vector_cycles_total_;
    stats["tensor_program_any_busy_cycles_total"] = tensor_program_any_busy_cycles_total_;
    stats["tensor_program_dma_busy_cycles_total"] = tensor_program_dma_busy_cycles_total_;
    stats["tensor_program_mxu_busy_cycles_total"] = tensor_program_mxu_busy_cycles_total_;
    stats["tensor_program_vec_busy_cycles_total"] = tensor_program_vec_busy_cycles_total_;
    stats["tensor_program_coll_busy_cycles_total"] = tensor_program_coll_busy_cycles_total_;
    stats["tensor_program_ops_total"] = tensor_program_ops_total_;
    stats["tensor_program_iters_total"] = tensor_program_iters_total_;
    stats["tensor_program_fence_count_total"] = tensor_program_fence_count_total_;
    stats["tensor_program_fence_wait_cycles_total"] = tensor_program_fence_wait_cycles_total_;
    stats["tensor_program_ub_stall_cycles_total"] = tensor_program_ub_stall_cycles_total_;
    stats["tensor_program_mem_stall_cycles_total"] = tensor_program_mem_stall_cycles_total_;
    stats["tensor_program_ub_occupancy_bytes_max"] = tensor_program_ub_occupancy_bytes_max_;
}


}} // namespace SST::SnnDL

