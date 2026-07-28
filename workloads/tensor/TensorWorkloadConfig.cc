// -*- c++ -*-

#include <sst/core/sst_config.h>

#include "workloads/tensor/TensorWorkload.h"

#include <sst/core/params.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <string>

namespace SST { namespace SnnDL {

namespace {

uint64_t clampU64(uint64_t value, uint64_t lower, uint64_t upper) {
    return std::max(lower, std::min(value, upper));
}

std::string toLowerCopy(const std::string& value) {
    std::string result;
    result.reserve(value.size());
    for (char ch : value) {
        result.push_back(static_cast<char>(std::tolower(static_cast<unsigned char>(ch))));
    }
    return result;
}

} // namespace

void TensorWorkload::configureFromParams(const SST::Params& params) {
    cfg_.m = params.find<uint32_t>("tensor_m", cfg_.m);
    cfg_.n = params.find<uint32_t>("tensor_n", cfg_.n);
    cfg_.k = params.find<uint32_t>("tensor_k", cfg_.k);
    cfg_.element_bytes = params.find<uint32_t>("tensor_element_bytes", cfg_.element_bytes);

    cfg_.array_m = params.find<uint32_t>("tensor_array_m", cfg_.array_m);
    cfg_.array_n = params.find<uint32_t>("tensor_array_n", cfg_.array_n);
    cfg_.compute_efficiency = params.find<float>("tensor_compute_efficiency", cfg_.compute_efficiency);
    cfg_.compute_precision = toLowerCopy(params.find<std::string>("tensor_compute_precision", cfg_.compute_precision));
    cfg_.compute_profile_override_enable =
        params.find<int>("tensor_compute_profile_override_enable", cfg_.compute_profile_override_enable ? 1 : 0) != 0;
    cfg_.compute_throughput_scale = params.find<float>("tensor_compute_throughput_scale", cfg_.compute_throughput_scale);
    cfg_.compute_pipeline_latency_cycles =
        params.find<uint32_t>("tensor_compute_pipeline_latency_cycles", cfg_.compute_pipeline_latency_cycles);
    cfg_.mxu_wavefront_enable =
        params.find<int>("tensor_mxu_wavefront_enable", cfg_.mxu_wavefront_enable ? 1 : 0) != 0;
    cfg_.mxu_wavefront_alpha = params.find<float>("tensor_mxu_wavefront_alpha", cfg_.mxu_wavefront_alpha);
    if (!(cfg_.mxu_wavefront_alpha >= 0.0f)) cfg_.mxu_wavefront_alpha = 0.0f;
    cfg_.mxu_a_bytes_per_cycle = params.find<uint64_t>("tensor_mxu_a_bytes_per_cycle", cfg_.mxu_a_bytes_per_cycle);
    cfg_.mxu_b_bytes_per_cycle = params.find<uint64_t>("tensor_mxu_b_bytes_per_cycle", cfg_.mxu_b_bytes_per_cycle);
    cfg_.mxu_c_bytes_per_cycle = params.find<uint64_t>("tensor_mxu_c_bytes_per_cycle", cfg_.mxu_c_bytes_per_cycle);

    cfg_.overlap_enable = params.find<int>("tensor_overlap_enable", cfg_.overlap_enable ? 1 : 0) != 0;
    cfg_.start_cycle = params.find<uint64_t>("tensor_start_cycle", cfg_.start_cycle);
    cfg_.iterations = params.find<uint32_t>("tensor_iterations", cfg_.iterations);

    cfg_.mem_enable = params.find<int>("tensor_mem_enable", cfg_.mem_enable ? 1 : 0) != 0;
    cfg_.mem_region_bytes = params.find<uint64_t>("tensor_mem_region_bytes", cfg_.mem_region_bytes);
    cfg_.mem_req_bytes = params.find<uint32_t>("tensor_mem_req_bytes", cfg_.mem_req_bytes);
    cfg_.mem_max_outstanding = params.find<uint32_t>("tensor_mem_max_outstanding", cfg_.mem_max_outstanding);
    cfg_.mem_timing_model = toLowerCopy(params.find<std::string>("tensor_mem_timing_model", cfg_.mem_timing_model));
    cfg_.mem_bank_groups_per_channel =
        params.find<uint32_t>("tensor_mem_bank_groups_per_channel", cfg_.mem_bank_groups_per_channel);
    cfg_.mem_banks_per_group = params.find<uint32_t>("tensor_mem_banks_per_group", cfg_.mem_banks_per_group);
    cfg_.mem_row_bytes = params.find<uint64_t>("tensor_mem_row_bytes", cfg_.mem_row_bytes);
    cfg_.mem_bank_queue_depth = params.find<uint32_t>("tensor_mem_bank_queue_depth", cfg_.mem_bank_queue_depth);
    cfg_.mem_sched_policy = toLowerCopy(params.find<std::string>("tensor_mem_sched_policy", cfg_.mem_sched_policy));
    cfg_.mem_t_rcd_cycles = params.find<uint32_t>("tensor_mem_t_rcd_cycles", cfg_.mem_t_rcd_cycles);
    cfg_.mem_t_cl_cycles = params.find<uint32_t>("tensor_mem_t_cl_cycles", cfg_.mem_t_cl_cycles);
    cfg_.mem_t_rp_cycles = params.find<uint32_t>("tensor_mem_t_rp_cycles", cfg_.mem_t_rp_cycles);
    cfg_.mem_t_burst_cycles = params.find<uint32_t>("tensor_mem_t_burst_cycles", cfg_.mem_t_burst_cycles);
    cfg_.mem_t_ccd_s_cycles = params.find<uint32_t>("tensor_mem_t_ccd_s_cycles", cfg_.mem_t_ccd_s_cycles);
    cfg_.mem_t_ccd_l_cycles = params.find<uint32_t>("tensor_mem_t_ccd_l_cycles", cfg_.mem_t_ccd_l_cycles);
    cfg_.mem_refresh_interval_cycles =
        params.find<uint32_t>("tensor_mem_refresh_interval_cycles", cfg_.mem_refresh_interval_cycles);
    cfg_.mem_refresh_block_cycles =
        params.find<uint32_t>("tensor_mem_refresh_block_cycles", cfg_.mem_refresh_block_cycles);

    cfg_.dataflow = toLowerCopy(params.find<std::string>("tensor_dataflow", cfg_.dataflow));
    cfg_.tile_m = params.find<uint32_t>("tensor_tile_m", cfg_.tile_m);
    cfg_.tile_n = params.find<uint32_t>("tensor_tile_n", cfg_.tile_n);
    cfg_.tile_k = params.find<uint32_t>("tensor_tile_k", cfg_.tile_k);
    cfg_.exec_mode = toLowerCopy(params.find<std::string>("tensor_exec_mode", cfg_.exec_mode));
    cfg_.program_dsl = params.find<std::string>("tensor_program_dsl", cfg_.program_dsl);
    cfg_.program_loop = params.find<int>("tensor_program_loop", cfg_.program_loop ? 1 : 0) != 0;
    cfg_.program_issue_width = params.find<uint32_t>("tensor_program_issue_width", cfg_.program_issue_width);
    cfg_.program_ub_buffers = std::max<uint32_t>(1u, params.find<uint32_t>("tensor_program_ub_buffers", cfg_.program_ub_buffers));
    cfg_.program_dma_dual_enable =
        params.find<int>("tensor_program_dma_dual_enable", cfg_.program_dma_dual_enable ? 1 : 0) != 0;
    cfg_.program_engine_priority =
        toLowerCopy(params.find<std::string>("tensor_program_engine_priority", cfg_.program_engine_priority));
    cfg_.vector_elems_per_cycle =
        params.find<uint32_t>("tensor_vector_elems_per_cycle", cfg_.vector_elems_per_cycle);
    cfg_.vector_pipeline_latency_cycles =
        params.find<uint32_t>("tensor_vector_pipeline_latency_cycles", cfg_.vector_pipeline_latency_cycles);
    cfg_.tile_schedule = toLowerCopy(params.find<std::string>("tensor_tile_schedule", cfg_.tile_schedule));
    cfg_.writeback_policy = toLowerCopy(params.find<std::string>("tensor_writeback_policy", cfg_.writeback_policy));
    cfg_.ub_bytes = params.find<uint64_t>("tensor_ub_bytes", cfg_.ub_bytes);
    cfg_.weight_bytes = params.find<uint64_t>("tensor_weight_bytes", cfg_.weight_bytes);
    cfg_.acc_bytes = params.find<uint64_t>("tensor_acc_bytes", cfg_.acc_bytes);
    cfg_.onchip_model_enable =
        params.find<int>("tensor_onchip_model_enable", cfg_.onchip_model_enable ? 1 : 0) != 0;
    cfg_.ub_bank_bytes = params.find<uint64_t>("tensor_ub_bank_bytes", cfg_.ub_bank_bytes);
    cfg_.ub_read_ports = params.find<uint32_t>("tensor_ub_read_ports", cfg_.ub_read_ports);
    cfg_.ub_write_ports = params.find<uint32_t>("tensor_ub_write_ports", cfg_.ub_write_ports);
    cfg_.onchip_bank_model_enable =
        params.find<int>("tensor_onchip_bank_model_enable", cfg_.onchip_bank_model_enable ? 1 : 0) != 0;
    cfg_.ub_bank_count = params.find<uint32_t>("tensor_ub_bank_count", cfg_.ub_bank_count);
    cfg_.ub_bank_select_policy =
        toLowerCopy(params.find<std::string>("tensor_ub_bank_select_policy", cfg_.ub_bank_select_policy));
    cfg_.ub_bank_conflict_mode =
        toLowerCopy(params.find<std::string>("tensor_ub_bank_conflict_mode", cfg_.ub_bank_conflict_mode));
    cfg_.acc_bank_bytes = params.find<uint64_t>("tensor_acc_bank_bytes", cfg_.acc_bank_bytes);
    cfg_.acc_read_ports = params.find<uint32_t>("tensor_acc_read_ports", cfg_.acc_read_ports);
    cfg_.acc_write_ports = params.find<uint32_t>("tensor_acc_write_ports", cfg_.acc_write_ports);
    cfg_.acc_bank_count = params.find<uint32_t>("tensor_acc_bank_count", cfg_.acc_bank_count);
    cfg_.acc_bank_select_policy =
        toLowerCopy(params.find<std::string>("tensor_acc_bank_select_policy", cfg_.acc_bank_select_policy));
    cfg_.acc_bank_conflict_mode =
        toLowerCopy(params.find<std::string>("tensor_acc_bank_conflict_mode", cfg_.acc_bank_conflict_mode));
    cfg_.bank_queue_depth = params.find<uint32_t>("tensor_bank_queue_depth", cfg_.bank_queue_depth);
    cfg_.spill_enable = params.find<int>("tensor_spill_enable", cfg_.spill_enable ? 1 : 0) != 0;
    cfg_.spill_packet_bytes = params.find<uint32_t>("tensor_spill_packet_bytes", cfg_.spill_packet_bytes);
    cfg_.spill_share_noc_budget =
        params.find<int>("tensor_spill_share_noc_budget", cfg_.spill_share_noc_budget ? 1 : 0) != 0;
    cfg_.dma_bandwidth_bytes_per_cycle =
        params.find<uint64_t>("tensor_dma_bandwidth_bytes_per_cycle", cfg_.dma_bandwidth_bytes_per_cycle);
    cfg_.dma_shared_bandwidth_bytes_per_cycle =
        params.find<uint64_t>("tensor_dma_shared_bandwidth_bytes_per_cycle", cfg_.dma_shared_bandwidth_bytes_per_cycle);
    cfg_.dma_burst_bytes = params.find<uint64_t>("tensor_dma_burst_bytes", cfg_.dma_burst_bytes);
    cfg_.dma_setup_cycles = params.find<uint32_t>("tensor_dma_setup_cycles", cfg_.dma_setup_cycles);
    cfg_.dma_read_engines = params.find<uint32_t>("tensor_dma_read_engines", cfg_.dma_read_engines);
    cfg_.dma_write_engines = params.find<uint32_t>("tensor_dma_write_engines", cfg_.dma_write_engines);
    cfg_.dma_max_inflight_per_engine =
        params.find<uint32_t>("tensor_dma_max_inflight_per_engine", cfg_.dma_max_inflight_per_engine);
    cfg_.dma_hbm_channels = std::max<uint32_t>(
        1u, params.find<uint32_t>("tensor_dma_hbm_channels", cfg_.dma_hbm_channels));
    cfg_.dma_hbm_channel_bandwidth_bytes_per_cycle =
        params.find<uint64_t>(
            "tensor_dma_hbm_channel_bandwidth_bytes_per_cycle",
            cfg_.dma_hbm_channel_bandwidth_bytes_per_cycle);
    cfg_.dma_hbm_channel_interleave_bytes =
        params.find<uint64_t>(
            "tensor_dma_hbm_channel_interleave_bytes",
            cfg_.dma_hbm_channel_interleave_bytes);
    if (cfg_.dma_hbm_channel_interleave_bytes == 0) {
        cfg_.dma_hbm_channel_interleave_bytes = 1;
    }
    cfg_.double_buffer = params.find<int>("tensor_double_buffer", cfg_.double_buffer ? 1 : 0) != 0;
    if (cfg_.exec_mode != "bulk" && cfg_.exec_mode != "tile" && cfg_.exec_mode != "program") {
        cfg_.exec_mode = "bulk";
    }
    close_step_on_iter_done_ = (cfg_.exec_mode != "program");
    if (cfg_.tile_schedule != "auto" &&
        cfg_.tile_schedule != "mnk" &&
        cfg_.tile_schedule != "mkn" &&
        cfg_.tile_schedule != "nkm") {
        cfg_.tile_schedule = "auto";
    }
    if (cfg_.writeback_policy != "at_end_of_k") {
        cfg_.writeback_policy = "at_end_of_k";
    }
    if (cfg_.dataflow != "os" && cfg_.dataflow != "ws" && cfg_.dataflow != "is") {
        cfg_.dataflow = "os";
    }
    if (cfg_.mem_timing_model != "off" &&
        cfg_.mem_timing_model != "proxy_v2" &&
        cfg_.mem_timing_model != "proxy_v3") {
        cfg_.mem_timing_model = "off";
    }
    if (cfg_.mem_sched_policy != "fifo" && cfg_.mem_sched_policy != "frfcfs") {
        cfg_.mem_sched_policy = "fifo";
    }
    if (cfg_.compute_precision != "fp16" &&
        cfg_.compute_precision != "bf16" &&
        cfg_.compute_precision != "fp32" &&
        cfg_.compute_precision != "tf32" &&
        cfg_.compute_precision != "int8" &&
        cfg_.compute_precision != "fp8") {
        cfg_.compute_precision = "fp16";
    }
    if (!(cfg_.compute_throughput_scale > 0.0f)) {
        cfg_.compute_throughput_scale = 1.0f;
    }
    cfg_.compute_pipeline_latency_cycles =
        clampPipelineLatencyCycles_(cfg_.compute_pipeline_latency_cycles);

    cfg_.collective_type = toLowerCopy(params.find<std::string>("tensor_collective_type", cfg_.collective_type));
    cfg_.collective_blocking = params.find<int>("tensor_collective_blocking", cfg_.collective_blocking ? 1 : 0) != 0;
    cfg_.collective_scope = toLowerCopy(params.find<std::string>("tensor_collective_scope", cfg_.collective_scope));
    cfg_.collective_bytes = params.find<uint64_t>("tensor_collective_bytes", cfg_.collective_bytes);
    cfg_.collective_period_cycles = params.find<uint64_t>("tensor_collective_period_cycles", cfg_.collective_period_cycles);
    cfg_.collective_pattern = toLowerCopy(params.find<std::string>("tensor_collective_pattern", cfg_.collective_pattern));
    cfg_.collective_packet_bytes = params.find<uint32_t>("tensor_collective_packet_bytes", cfg_.collective_packet_bytes);
    cfg_.collective_algo = toLowerCopy(params.find<std::string>("tensor_collective_algo", cfg_.collective_algo));
    cfg_.collective_chunk_bytes = params.find<uint32_t>("tensor_collective_chunk_bytes", cfg_.collective_chunk_bytes);
    cfg_.collective_reduce_overhead_cycles =
        params.find<uint32_t>("tensor_collective_reduce_overhead_cycles", cfg_.collective_reduce_overhead_cycles);
    cfg_.collective_max_inflight_chunks =
        params.find<uint32_t>("tensor_collective_max_inflight_chunks", cfg_.collective_max_inflight_chunks);
    cfg_.collective_credit_enable =
        params.find<int>("tensor_collective_credit_enable", cfg_.collective_credit_enable ? 1 : 0) != 0;
    cfg_.collective_credit_window_chunks =
        params.find<uint32_t>("tensor_collective_credit_window_chunks", cfg_.collective_credit_window_chunks);
    cfg_.collective_credit_return_mode =
        toLowerCopy(params.find<std::string>("tensor_collective_credit_return_mode", cfg_.collective_credit_return_mode));
    cfg_.collective_backpressure_mode =
        toLowerCopy(params.find<std::string>("tensor_collective_backpressure_mode", cfg_.collective_backpressure_mode));
    cfg_.noc_bandwidth_bytes_per_cycle =
        params.find<uint64_t>("tensor_noc_bandwidth_bytes_per_cycle", cfg_.noc_bandwidth_bytes_per_cycle);
    cfg_.collective_overlap_with_compute =
        params.find<int>("tensor_collective_overlap_with_compute", cfg_.collective_overlap_with_compute ? 1 : 0) != 0;
    cfg_.collective_issue_priority =
        toLowerCopy(params.find<std::string>("tensor_collective_issue_priority", cfg_.collective_issue_priority));
    cfg_.collective_2d_dim_x = params.find<uint32_t>("tensor_collective_2d_dim_x", cfg_.collective_2d_dim_x);
    cfg_.collective_2d_dim_y = params.find<uint32_t>("tensor_collective_2d_dim_y", cfg_.collective_2d_dim_y);
    cfg_.collective_2d_row_major =
        params.find<int>("tensor_collective_2d_row_major", cfg_.collective_2d_row_major ? 1 : 0) != 0;
    if (cfg_.collective_scope != "per_core" &&
        cfg_.collective_scope != "per_pe" &&
        cfg_.collective_scope != "per_system") {
        cfg_.collective_scope = "per_core";
    }
    if (cfg_.collective_type != "none" &&
        cfg_.collective_type != "allreduce" &&
        cfg_.collective_type != "allgather" &&
        cfg_.collective_type != "reducescatter") {
        cfg_.collective_type = "none";
    }
    if (cfg_.collective_pattern != "ring" &&
        cfg_.collective_pattern != "mesh_x" &&
        cfg_.collective_pattern != "mesh_xy") {
        cfg_.collective_pattern = "ring";
    }
    if (cfg_.collective_algo != "legacy_bytes" &&
        cfg_.collective_algo != "ring_chunked" &&
        cfg_.collective_algo != "torus_2d_rs_ag") {
        cfg_.collective_algo = "legacy_bytes";
    }
    if (cfg_.collective_algo == "torus_2d_rs_ag" && cfg_.collective_type != "allreduce") {
        cfg_.collective_type = "allreduce";
    }
    if (cfg_.collective_backpressure_mode != "hard" &&
        cfg_.collective_backpressure_mode != "soft") {
        cfg_.collective_backpressure_mode = "hard";
    }
    if (cfg_.collective_credit_return_mode != "event_on_recv" &&
        cfg_.collective_credit_return_mode != "legacy_tick") {
        cfg_.collective_credit_return_mode = "event_on_recv";
    }
    if (cfg_.collective_issue_priority != "control_first" &&
        cfg_.collective_issue_priority != "payload_first") {
        cfg_.collective_issue_priority = "control_first";
    }
    if (cfg_.ub_bank_select_policy != "interleave" &&
        cfg_.ub_bank_select_policy != "rr" &&
        cfg_.ub_bank_select_policy != "hash") {
        cfg_.ub_bank_select_policy = "interleave";
    }
    if (cfg_.acc_bank_select_policy != "interleave" &&
        cfg_.acc_bank_select_policy != "rr" &&
        cfg_.acc_bank_select_policy != "hash") {
        cfg_.acc_bank_select_policy = "interleave";
    }
    if (cfg_.ub_bank_conflict_mode != "queue" &&
        cfg_.ub_bank_conflict_mode != "block") {
        cfg_.ub_bank_conflict_mode = "queue";
    }
    if (cfg_.acc_bank_conflict_mode != "queue" &&
        cfg_.acc_bank_conflict_mode != "block") {
        cfg_.acc_bank_conflict_mode = "queue";
    }

    cfg_.comm_enable = params.find<int>("tensor_comm_enable", cfg_.comm_enable ? 1 : 0) != 0;
    cfg_.comm_period_cycles = params.find<uint64_t>("tensor_comm_period_cycles", cfg_.comm_period_cycles);
    cfg_.comm_payload_bytes = params.find<uint32_t>("tensor_comm_payload_bytes", cfg_.comm_payload_bytes);

    cfg_.strict = params.find<int>("tensor_strict", cfg_.strict ? 1 : 0) != 0;
    cfg_.seed_base = params.find<uint64_t>("tensor_seed", cfg_.seed_base);
    total_cores_cfg_ = params.find<uint32_t>("total_cores", total_cores_cfg_);
    if (total_cores_cfg_ == 0) total_cores_cfg_ = 1;

    // === Hard bounds (avoid pathological allocations / invalid params) ===
    if (cfg_.m == 0) cfg_.m = 1;
    if (cfg_.n == 0) cfg_.n = 1;
    if (cfg_.k == 0) cfg_.k = 1;
    cfg_.element_bytes = static_cast<uint32_t>(clampU64(cfg_.element_bytes, 1, 16));
    cfg_.program_issue_width = static_cast<uint32_t>(clampU64(cfg_.program_issue_width, 1, 1024ull));
    cfg_.vector_elems_per_cycle = static_cast<uint32_t>(clampU64(cfg_.vector_elems_per_cycle, 1, 1024ull * 1024ull));
    cfg_.vector_pipeline_latency_cycles = static_cast<uint32_t>(clampU64(cfg_.vector_pipeline_latency_cycles, 0, 4096ull));
    if (cfg_.array_m == 0) cfg_.array_m = 1;
    if (cfg_.array_n == 0) cfg_.array_n = 1;
    cfg_.mem_req_bytes = static_cast<uint32_t>(clampU64(cfg_.mem_req_bytes, 1, 1024ull * 1024ull));
    cfg_.mem_max_outstanding = static_cast<uint32_t>(clampU64(cfg_.mem_max_outstanding, 1, 4096));
    cfg_.mem_region_bytes = clampNonZero_(cfg_.mem_region_bytes, 4096);
    cfg_.mem_bank_groups_per_channel = static_cast<uint32_t>(clampU64(cfg_.mem_bank_groups_per_channel, 1, 1024));
    cfg_.mem_banks_per_group = static_cast<uint32_t>(clampU64(cfg_.mem_banks_per_group, 1, 1024));
    cfg_.mem_row_bytes = clampNonZero_(cfg_.mem_row_bytes, 256);
    cfg_.mem_bank_queue_depth = static_cast<uint32_t>(clampU64(cfg_.mem_bank_queue_depth, 1, 4096));
    cfg_.mem_t_rcd_cycles = static_cast<uint32_t>(clampU64(cfg_.mem_t_rcd_cycles, 0, 4096));
    cfg_.mem_t_cl_cycles = static_cast<uint32_t>(clampU64(cfg_.mem_t_cl_cycles, 0, 4096));
    cfg_.mem_t_rp_cycles = static_cast<uint32_t>(clampU64(cfg_.mem_t_rp_cycles, 0, 4096));
    cfg_.mem_t_burst_cycles = static_cast<uint32_t>(clampU64(cfg_.mem_t_burst_cycles, 0, 4096));
    cfg_.mem_refresh_interval_cycles = static_cast<uint32_t>(clampU64(cfg_.mem_refresh_interval_cycles, 0, (1ull << 20)));
    cfg_.mem_refresh_block_cycles = static_cast<uint32_t>(clampU64(cfg_.mem_refresh_block_cycles, 0, 4096));
    cfg_.start_cycle = clampNonZero_(cfg_.start_cycle, 1);
    cfg_.comm_payload_bytes = static_cast<uint32_t>(clampU64(cfg_.comm_payload_bytes, 0, 1024ull * 1024ull));
    cfg_.collective_packet_bytes = static_cast<uint32_t>(clampU64(cfg_.collective_packet_bytes, 1, 1024ull * 1024ull));
    if (cfg_.collective_packet_bytes < 8) {
        cfg_.collective_packet_bytes = 8;
    }
    cfg_.spill_packet_bytes = static_cast<uint32_t>(clampU64(cfg_.spill_packet_bytes, 1, 1024ull * 1024ull));
    if (cfg_.spill_packet_bytes < 8) {
        cfg_.spill_packet_bytes = 8;
    }
    cfg_.collective_chunk_bytes = static_cast<uint32_t>(clampU64(cfg_.collective_chunk_bytes, 0, 1024ull * 1024ull));
    cfg_.collective_max_inflight_chunks = static_cast<uint32_t>(clampU64(cfg_.collective_max_inflight_chunks, 1, 4096));
    cfg_.collective_credit_window_chunks = static_cast<uint32_t>(clampU64(cfg_.collective_credit_window_chunks, 0, 4096));
    cfg_.ub_bank_count = static_cast<uint32_t>(clampU64(cfg_.ub_bank_count, 1, 4096));
    cfg_.acc_bank_count = static_cast<uint32_t>(clampU64(cfg_.acc_bank_count, 1, 4096));
    cfg_.bank_queue_depth = static_cast<uint32_t>(clampU64(cfg_.bank_queue_depth, 1, 4096));
    cfg_.dma_burst_bytes = clampU64(cfg_.dma_burst_bytes, 0, 1ull << 30);
    cfg_.dma_setup_cycles = static_cast<uint32_t>(clampU64(cfg_.dma_setup_cycles, 0, 4096));
    cfg_.dma_read_engines = static_cast<uint32_t>(clampU64(cfg_.dma_read_engines, 0, 4096));
    cfg_.dma_write_engines = static_cast<uint32_t>(clampU64(cfg_.dma_write_engines, 0, 4096));
    cfg_.dma_max_inflight_per_engine = static_cast<uint32_t>(clampU64(cfg_.dma_max_inflight_per_engine, 0, 4096));
    cfg_.mxu_a_bytes_per_cycle = clampU64(cfg_.mxu_a_bytes_per_cycle, 0, 1ull << 30);
    cfg_.mxu_b_bytes_per_cycle = clampU64(cfg_.mxu_b_bytes_per_cycle, 0, 1ull << 30);
    cfg_.mxu_c_bytes_per_cycle = clampU64(cfg_.mxu_c_bytes_per_cycle, 0, 1ull << 30);
    if (cfg_.onchip_bank_model_enable) {
        cfg_.onchip_model_enable = true;
    }
    if (cfg_.onchip_model_enable) {
        if (cfg_.ub_read_ports == 0) cfg_.ub_read_ports = 2;
        if (cfg_.ub_write_ports == 0) cfg_.ub_write_ports = 1;
        if (cfg_.acc_read_ports == 0) cfg_.acc_read_ports = 1;
        if (cfg_.acc_write_ports == 0) cfg_.acc_write_ports = 1;
        if (cfg_.ub_bank_bytes == 0) {
            cfg_.ub_bank_bytes = cfg_.onchip_bank_model_enable
                                     ? ceilDivU64_(cfg_.ub_bytes, static_cast<uint64_t>(std::max<uint32_t>(cfg_.ub_bank_count, 1u)))
                                     : cfg_.ub_bytes;
        }
        if (cfg_.acc_bank_bytes == 0) {
            cfg_.acc_bank_bytes = cfg_.onchip_bank_model_enable
                                      ? ceilDivU64_(cfg_.acc_bytes, static_cast<uint64_t>(std::max<uint32_t>(cfg_.acc_bank_count, 1u)))
                                      : cfg_.acc_bytes;
        }
    }
    if (cfg_.collective_credit_enable && cfg_.collective_credit_window_chunks == 0) {
        cfg_.collective_credit_window_chunks = std::max<uint32_t>(cfg_.collective_max_inflight_chunks, 1u);
    }
    if (cfg_.collective_algo == "ring_chunked") {
        tensor_collective_algo_id_ = 1ull;
    } else if (cfg_.collective_algo == "torus_2d_rs_ag") {
        tensor_collective_algo_id_ = 2ull;
    } else {
        tensor_collective_algo_id_ = 0ull;
    }

    const ComputeProfile compute_profile = resolveComputeProfile_(cfg_);
    compute_throughput_scale_effective_ = compute_profile.throughput_scale;
    compute_pipeline_latency_cycles_effective_ = compute_profile.pipeline_latency_cycles;
    tensor_compute_precision_profile_id_ = compute_profile.profile_id;

    // Derived per-iteration sizes/ops (GEMM)
    const unsigned __int128 a_elems = static_cast<unsigned __int128>(cfg_.m) * static_cast<unsigned __int128>(cfg_.k);
    const unsigned __int128 b_elems = static_cast<unsigned __int128>(cfg_.k) * static_cast<unsigned __int128>(cfg_.n);
    const unsigned __int128 c_elems = static_cast<unsigned __int128>(cfg_.m) * static_cast<unsigned __int128>(cfg_.n);
    const unsigned __int128 eb = static_cast<unsigned __int128>(cfg_.element_bytes);
    const unsigned __int128 macs = c_elems * static_cast<unsigned __int128>(cfg_.k);
    mac_ops_per_iter_ =
        (macs > static_cast<unsigned __int128>(std::numeric_limits<uint64_t>::max()))
            ? std::numeric_limits<uint64_t>::max()
            : static_cast<uint64_t>(macs);
    compute_math_cycles_per_iter_ = ceilDivU64_(mac_ops_per_iter_, effectivePeakMacsPerCycle_());

    const uint64_t tm = clampNonZero_(std::min<uint64_t>(cfg_.tile_m ? cfg_.tile_m : cfg_.m, cfg_.m), 1);
    const uint64_t tn = clampNonZero_(std::min<uint64_t>(cfg_.tile_n ? cfg_.tile_n : cfg_.n, cfg_.n), 1);
    const uint64_t tk = clampNonZero_(std::min<uint64_t>(cfg_.tile_k ? cfg_.tile_k : cfg_.k, cfg_.k), 1);
    const uint64_t mt = ceilDivU64_(cfg_.m, tm);
    const uint64_t nt = ceilDivU64_(cfg_.n, tn);
    const uint64_t kt = ceilDivU64_(cfg_.k, tk);
    tile_count_per_iter_ = mt * nt * kt;

    const uint64_t a_tile = tm * tk * static_cast<uint64_t>(cfg_.element_bytes);
    const uint64_t b_tile = tk * tn * static_cast<uint64_t>(cfg_.element_bytes);
    const uint64_t c_tile = tm * tn * static_cast<uint64_t>(cfg_.element_bytes);

    const bool dataflow_is = (cfg_.dataflow == "is");
    const bool dataflow_ws = (cfg_.dataflow == "ws");
    const bool dataflow_os = (!dataflow_is && !dataflow_ws);

    // Persist tile-derived params for tile exec mode.
    tile_tm_ = static_cast<uint32_t>(tm);
    tile_tn_ = static_cast<uint32_t>(tn);
    tile_tk_ = static_cast<uint32_t>(tk);
    tile_mt_ = static_cast<uint32_t>(mt);
    tile_nt_ = static_cast<uint32_t>(nt);
    tile_kt_ = static_cast<uint32_t>(kt);
    tile_a_bytes_ = a_tile;
    tile_b_bytes_ = b_tile;
    tile_c_bytes_ = c_tile;

    const bool tile_like = (cfg_.exec_mode == "tile" || cfg_.exec_mode == "program");

    tile_schedule_eff_ = cfg_.tile_schedule;
    if (tile_schedule_eff_.empty()) tile_schedule_eff_ = "auto";
    if (tile_like && tile_schedule_eff_ == "auto") {
        if (dataflow_is) {
            tile_schedule_eff_ = "mkn";
        } else if (dataflow_ws) {
            tile_schedule_eff_ = "nkm";
        } else {
            tile_schedule_eff_ = "mnk";
        }
    } else if (tile_schedule_eff_ == "auto") {
        tile_schedule_eff_ = "mnk";
    }

    const bool ub_can_hold_a = (cfg_.ub_bytes > 0 && cfg_.ub_bytes >= a_tile);
    const bool ub_can_hold_a_and_b = (cfg_.ub_bytes > 0 && cfg_.ub_bytes >= saturatingAddU64_(a_tile, b_tile));
    const uint64_t b_pool_cap = (cfg_.weight_bytes > 0) ? cfg_.weight_bytes : cfg_.ub_bytes;
    const bool b_pool_can_hold_b = (b_pool_cap > 0 && b_pool_cap >= b_tile);
    const bool can_keep_a = ub_can_hold_a && (cfg_.weight_bytes > 0 ? b_pool_can_hold_b : ub_can_hold_a_and_b);
    const bool can_keep_b = b_pool_can_hold_b && (cfg_.weight_bytes > 0 ? ub_can_hold_a : ub_can_hold_a_and_b);
    tile_keep_a_ = (tile_like && dataflow_is && can_keep_a && tile_schedule_eff_ == "mkn");
    tile_keep_b_ = (tile_like && dataflow_ws && can_keep_b && tile_schedule_eff_ == "nkm");
    const bool keep_a = tile_like ? tile_keep_a_ : can_keep_a;
    const bool keep_b = tile_like ? tile_keep_b_ : can_keep_b;

    if (tile_like) {
        compute_pipeline_cycles_per_iter_ =
            saturatingMulU64ByU32_(tile_count_per_iter_, compute_pipeline_latency_cycles_effective_);
    } else {
        compute_pipeline_cycles_per_iter_ = static_cast<uint64_t>(compute_pipeline_latency_cycles_effective_);
    }
    compute_cycles_per_iter_ = saturatingAddU64_(compute_math_cycles_per_iter_, compute_pipeline_cycles_per_iter_);

    uint64_t total_a = 0;
    uint64_t total_b = 0;
    if (dataflow_is) {
        const uint64_t a_factor = keep_a ? 1 : nt;
        total_a = mt * kt * a_tile * a_factor;
        total_b = mt * nt * kt * b_tile;
    } else if (dataflow_ws) {
        const uint64_t b_factor = keep_b ? 1 : mt;
        total_a = mt * nt * kt * a_tile;
        total_b = kt * nt * b_tile * b_factor;
    } else if (dataflow_os) {
        total_a = mt * nt * kt * a_tile;
        total_b = mt * nt * kt * b_tile;
    }

    const uint64_t total_c = mt * nt * c_tile;
    bytes_read_per_iter_ = total_a + total_b;
    bytes_write_per_iter_ = total_c;
    dram_bytes_per_iter_ = bytes_read_per_iter_ + bytes_write_per_iter_;
    onchip_bytes_per_iter_ = bytes_read_per_iter_ + bytes_write_per_iter_;
    uint64_t dma_bw_eff = cfg_.dma_bandwidth_bytes_per_cycle;
    if (cfg_.dma_shared_bandwidth_bytes_per_cycle > 0) {
        const uint64_t cores = static_cast<uint64_t>(std::max<uint32_t>(total_cores_cfg_, 1u));
        const uint64_t per_core = ceilDivU64_(cfg_.dma_shared_bandwidth_bytes_per_cycle, cores);
        if (dma_bw_eff == 0) {
            dma_bw_eff = per_core;
        } else {
            dma_bw_eff = std::min<uint64_t>(dma_bw_eff, per_core);
        }
    }
    dma_cycles_per_iter_ = (dma_bw_eff > 0) ? ceilDivU64_(dram_bytes_per_iter_, dma_bw_eff) : 0;
    if (tile_count_per_iter_ > 0) {
        tile_seg_math_cycles_base_ = compute_math_cycles_per_iter_ / tile_count_per_iter_;
        tile_seg_math_cycles_remainder_ = compute_math_cycles_per_iter_ % tile_count_per_iter_;
    } else {
        tile_seg_math_cycles_base_ = 0;
        tile_seg_math_cycles_remainder_ = 0;
    }

    onchip_ub_bank_occupancy_bytes_.assign(std::max<uint32_t>(cfg_.ub_bank_count, 1u), 0ull);
    onchip_weight_bank_occupancy_bytes_.assign(std::max<uint32_t>(cfg_.ub_bank_count, 1u), 0ull);
    onchip_acc_bank_occupancy_bytes_.assign(std::max<uint32_t>(cfg_.acc_bank_count, 1u), 0ull);
    onchip_ub_bank_queue_occupancy_.assign(std::max<uint32_t>(cfg_.ub_bank_count, 1u), 0u);
    onchip_weight_bank_queue_occupancy_.assign(std::max<uint32_t>(cfg_.ub_bank_count, 1u), 0u);
    onchip_acc_bank_queue_occupancy_.assign(std::max<uint32_t>(cfg_.acc_bank_count, 1u), 0u);
    onchip_ub_bank_rr_ = 0;
    onchip_weight_bank_rr_ = 0;
    onchip_acc_bank_rr_ = 0;
    resetMemTimingState_();
    collective_credit_inflight_chunks_ = 0;
    collective_credit_outstanding_.clear();
    collective_credit_return_seen_.clear();
    collective_credit_return_pending_credits_.clear();
    collective_credit_return_pending_queue_.clear();

    configured_ = true;
}

}} // namespace SST::SnnDL

