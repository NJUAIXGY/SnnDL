// -*- c++ -*-
// Parameter parsing for the SnnDL PE subcomponent.
// Each field belongs to one explicit domain; runtime code consumes the groups
// directly instead of a second, flat compatibility mirror.

#pragma once

#include <cstdint>
#include <string>

#include <sst/core/params.h>

#include "SnnConfigGroups.h"
#include "WorkloadConfig.h"

namespace SST { namespace SnnDL {

struct SnnPESubComponentConfig {
    SnnPEIdentityConfig identity;
    SnnPEWorkloadConfig workload;
    SnnPECommunicationConfig communication;
    SnnPEStorageConfig storage;
    SnnPERoutingConfig routing;
    SnnPEGasConfig gas;
    SnnPEVerificationConfig verification;
    SnnPEDiagnosticsConfig diagnostics;
};

inline SnnPESubComponentConfig parseSnnPESubComponentConfig(const SST::Params& params) {
    SnnPESubComponentConfig c{};
    auto& identity = c.identity;
    auto& workload = c.workload;
    auto& communication = c.communication;
    auto& storage = c.storage;
    auto& routing = c.routing;
    auto& gas = c.gas;
    auto& verification = c.verification;
    auto& diagnostics = c.diagnostics;

    identity.sentinel_enable = params.find<int>("sentinel_enable", 0) != 0;
    identity.core_id = params.find<int>("core_id", 0);
    identity.total_cores = params.find<int>("total_cores", 1);
    identity.global_neuron_base = params.find<uint64_t>("global_neuron_base", 0);
    identity.num_neurons = params.find<uint32_t>("num_neurons", 64);
    identity.base_addr = params.find<uint64_t>("base_addr", 0);
    identity.node_id = params.find<uint32_t>("node_id", 0);
    identity.total_nodes = params.find<uint32_t>("total_nodes", 1);
    identity.verbose = params.find<int>("verbose", 0);

    diagnostics.enable_extended_diagnostics =
        params.find<int>("enable_extended_diagnostics", 0) != 0;
    diagnostics.profiler_enable = params.find<int>("enable_profiler", 0) != 0;
    diagnostics.profiler_csv_prefix = params.find<std::string>("profiler_csv_prefix", "");
    diagnostics.enable_detailed_map_log =
        params.find<int>("enable_detailed_map_log", 0) != 0;
    diagnostics.log_weight_details = params.find<int>("log_weight_details", 0) != 0;
    diagnostics.quiet_finish_logs = params.find<int>("quiet_finish_logs", 0) != 0;

    workload.workload_impl = workloadImplFromParamsOrEnv(params, "snn");
    workload.enable_weight_fetch = params.find<int>("enable_weight_fetch", 0) != 0;
    workload.workload_spike_input_enable =
        params.find<int>("workload_spike_input_enable", 0) != 0;
    workload.write_weights_on_init = params.find<int>("write_weights_on_init", 1) != 0;
    workload.memory_warmup_cycles = params.find<uint64_t>("memory_warmup_cycles", 1000);
    workload.init_default_weight = params.find<float>("init_default_weight", 0.5f);
    workload.readresp_zero_fallback = params.find<int>("readresp_zero_fallback", 0) != 0;
    workload.max_outstanding_requests = params.find<uint32_t>("max_outstanding_requests", 16);
    workload.synapse_weight_mode = "bcsr_gas";
    workload.weights_template = params.find<std::string>("weights_template", "");
    workload.neurons_per_pe = params.find<uint32_t>("neurons_per_pe", 0);
    workload.weights_file = params.find<std::string>("weights_file", "");

    storage.max_cache_entries = params.find<uint32_t>("max_cache_entries", 65536);
    storage.use_clock_weight_cache = params.find<int>("use_clock_weight_cache", 0) != 0;
    storage.disable_weight_cache = params.find<int>("disable_weight_cache", 0) != 0;
    storage.use_event_weight_fallback =
        params.find<int>("use_event_weight_fallback", 0) != 0;
    storage.dense_layout_mode = params.find<std::string>("dense_layout_mode", "row_major");
    storage.dense_phys_dram_row_bytes =
        params.find<uint32_t>("dense_phys_dram_row_bytes", 0);
    storage.loader_done_key = params.find<std::string>("loader_done_key", "");
    storage.loader_barrier_cycles = params.find<uint64_t>("loader_barrier_cycles", 0);
    storage.read_force_single = params.find<int>("read_force_single", 0) != 0;

    communication.route_summary_enable =
        params.find<int>("route_summary_enable", 0) != 0;
    communication.merge_read_cacheline =
        params.find<int>("merge_read_cacheline", 1) != 0;
    communication.merge_read_row = params.find<int>("merge_read_row", 0) != 0;
    communication.merge_read_auto = params.find<int>("merge_read_auto", 0) != 0;
    communication.line_size_bytes = params.find<uint32_t>("line_size_bytes", 64);
    communication.weights_cols = params.find<uint32_t>("weights_cols", 0);
    communication.index_mode = params.find<std::string>("index_mode", "pre_row_post_col");
    communication.verify_routing_weights =
        params.find<int>("verify_routing_weights", 0) != 0;

    gas.gas_enable = params.find<int>("gas_enable", 0) != 0;
    gas.gas_window_mode = params.find<int>("gas_window_mode", 0) != 0;
    gas.gas_manual_window_drive_requested =
        params.find<int>("gas_manual_window_drive", 0) != 0;
    gas.gas_window_cycles_gather =
        clampNonZeroU64(params.find<uint64_t>("gas_window_cycles_gather", 200));
    gas.window_read_enable = params.find<int>("window_read_enable", 0) != 0;
    gas.window_read_debug = params.find<int>("window_read_debug", 0) != 0;
    gas.scatter_diag_limit = params.find<uint32_t>("scatter_diag_limit", 0);
    gas.window_read_budget = params.find<uint32_t>("window_read_budget", 1024);
    gas.edge_collector_max_capacity =
        params.find<uint64_t>("edge_collector_max_capacity", 1000000);
    gas.apply_acc_enable = params.find<int>("apply_acc_enable", 0) != 0;
    gas.acc_high_watermark_bytes =
        params.find<uint64_t>("acc_high_watermark_bytes", 16ull * 1024ull * 1024ull);
    gas.acc_spill_enable = params.find<int>("acc_spill_enable", 1) != 0;
    gas.stage_events_csv = params.find<std::string>("stage_events_csv", "");
    gas.apply_dense_acc_enable = params.find<int>("apply_dense_acc_enable", 1) != 0;
    gas.acc_shadow_verify_enable =
        gas.apply_dense_acc_enable && (params.find<int>("acc_shadow_verify_enable", 0) != 0);
    {
        const int record_apply_default = (gas.gas_window_mode && gas.apply_acc_enable) ? 1 : 0;
        gas.record_edge_apply_enable =
            params.find<int>("record_edge_apply_enable", record_apply_default) != 0;
        gas.record_edge_idle_enable = params.find<int>("record_edge_idle_enable", 0) != 0;
        gas.record_edge_scatter_enable =
            params.find<int>("record_edge_scatter_enable", 0) != 0;
    }

    routing.bcsr_cols = params.find<uint32_t>("weights_cols", identity.num_neurons);
    routing.bcsr_block_rows = params.find<uint32_t>("bcsr_block_rows", 16);
    routing.bcsr_block_cols = params.find<uint32_t>("bcsr_block_cols", 16);
    routing.bcsr_idx_bytes = params.find<uint32_t>("bcsr_idx_bytes", 2);
    routing.bcsr_val_bytes = params.find<uint32_t>("bcsr_val_bytes", 4);
    routing.bcsr_rowptr_offset = params.find<uint64_t>("bcsr_rowptr_offset", 0);
    routing.bcsr_colidx_offset = params.find<uint64_t>("bcsr_colidx_offset", 0);
    routing.bcsr_blockdata_offset = params.find<uint64_t>("bcsr_blockdata_offset", 0);
    routing.bcsr_blockids_offset = params.find<uint64_t>("bcsr_blockids_offset", 0);
    routing.bcsr_layout_mode = params.find<std::string>("bcsr_layout_mode", "flat");
    routing.bcsr_colidx_row_stride_bytes =
        params.find<uint32_t>("bcsr_colidx_row_stride_bytes", 0);
    routing.bcsr_blockdata_row_stride_bytes =
        params.find<uint32_t>("bcsr_blockdata_row_stride_bytes", 0);
    routing.bcsr_blockids_row_stride_bytes =
        params.find<uint32_t>("bcsr_blockids_row_stride_bytes", 0);
    routing.bcsr_block_fetch_mode =
        params.find<std::string>("bcsr_block_fetch_mode", "full_block");
    routing.per_core_stride = params.find<uint64_t>("per_core_stride", 0);
    routing.bcsr_row_index_cache_cap = params.find<uint32_t>("bcsr_row_index_cache_cap", 64);
    routing.bcsr_block_cache_cap = params.find<uint32_t>("bcsr_block_cache_cap", 256);
    routing.bcsr_prefetch_all = params.find<int>("bcsr_prefetch_all", 0) != 0;
    routing.verify_weights = params.find<int>("verify_weights", 0) != 0;
    routing.bcsr_force_file_read = params.find<int>("bcsr_force_file_read", 0) != 0;
    routing.bcsr_rowptr_file_fallback_enable =
        params.find<int>("bcsr_rowptr_file_fallback_enable", 0) != 0;

    verification.byte_exact_verify_enable =
        params.find<int>("byte_exact_verify_enable", 0) != 0;
    verification.byte_exact_verify_mode =
        params.find<std::string>("byte_exact_verify_mode", "");
    verification.byte_exact_verify_row_scale =
        params.find<uint32_t>("byte_exact_verify_row_scale", 1024);
    verification.byte_exact_verify_max_mismatch =
        params.find<uint32_t>("byte_exact_verify_max_mismatch", 8);
    verification.bcsr_semantic_verify_enable =
        params.find<int>("bcsr_semantic_verify_enable", 0) != 0;
    verification.bcsr_semantic_verify_max_edges =
        params.find<uint32_t>("bcsr_semantic_verify_max_edges", 64);
    verification.bcsr_semantic_verify_max_mismatch =
        params.find<uint32_t>("bcsr_semantic_verify_max_mismatch", 8);
    verification.bcsr_semantic_verify_abs_tol =
        params.find<float>("bcsr_semantic_verify_abs_tol", 1e-6f);
    verification.bcsr_semantic_verify_rel_tol =
        params.find<float>("bcsr_semantic_verify_rel_tol", 1e-6f);

    return c;
}

}} // namespace SST::SnnDL
