// -*- c++ -*-
//
// MultiCorePEConfig.cc
//

#include "multicore/MultiCorePEConfig.h"

#include <sst/core/params.h>

#include "WorkloadConfig.h"

namespace SST { namespace SnnDL {

MultiCorePEConfig parseMultiCorePEConfig(const SST::Params& params) {
    MultiCorePEConfig c{};

    c.verbose_level = params.find<int>("verbose", 0);
    c.clock_freq = params.find<std::string>("clock", "1GHz");

    c.num_cores = params.find<int>("num_cores", 4);
    c.neurons_per_core = params.find<int>("neurons_per_core", 64);
    c.neurons_per_pe = params.find<uint32_t>("neurons_per_pe", 0);
    c.node_id = params.find<int>("node_id", 0);
    c.total_nodes = params.find<int>("total_nodes", 1);
    c.global_neuron_base = params.find<uint64_t>("global_neuron_base", 0);
    c.base_addr = params.find<uint64_t>("base_addr", 0);
    c.sim_stop_ns = params.find<uint64_t>("sim_stop_ns", 0);

    c.noc_lat_hist_max = params.find<uint32_t>("noc_lat_hist_max", 131072);
    if (c.noc_lat_hist_max < 64) c.noc_lat_hist_max = 64;
    if (c.noc_lat_hist_max > 2000000u) c.noc_lat_hist_max = 2000000u;

    // P2 Step3: no runtime getenv fallbacks here; only Params (+ optional cached env for workload selection).
    c.sentinel_enabled = params.find<int>("sentinel_enable", 0) != 0;
    c.progress_log_interval_ns = params.find<uint64_t>("progress_log_interval_ns", 0);
    c.progress_log_node = params.find<int>("progress_log_node", -1);
    c.step_diag_cap = params.find<long>("step_diag_cap", 0);
    c.step_diag_enable = params.find<int>("step_diag_enable", 0);

    c.weights_file = params.find<std::string>("weights_file", "");
    c.enable_numa = params.find<bool>("enable_numa", true);

    // Workload selector (used to disable step activation for non-SNN workloads).
    c.workload_impl = workloadImplFromParamsOrEnv(params, "snn");
    c.workload_kind = workloadKindFromString(c.workload_impl);
    c.workload_stats_modules = toLowerCopy(params.find<std::string>("workload_stats_modules", ""));

    // Exec mode hint (experiment observability; does not change behavior).
    c.exec_mode = execModeFromParams(params, "gas");

    c.v_thresh = params.find<float>("v_thresh", 1.0f);
    c.v_reset = params.find<float>("v_reset", 0.0f);
    c.v_rest = params.find<float>("v_rest", 0.0f);
    c.tau_mem = params.find<float>("tau_mem", 20.0f);
    c.t_ref = params.find<int>("t_ref", 2);

    c.enable_test_traffic = params.find<bool>("enable_test_traffic", false);
    c.test_target_node = params.find<int>("test_target_node", 0);
    c.test_period = params.find<int>("test_period", 100);
    c.test_spikes_per_burst = params.find<int>("test_spikes_per_burst", 4);
    c.test_weight = params.find<float>("test_weight", 0.2f);
    c.test_max_spikes = params.find<int>("test_max_spikes", 10);

    c.use_optimized_ring = params.find<bool>("use_optimized_ring", true);
    c.print_node_summary = params.find<bool>("print_node_summary", true);
    c.primary_keepalive = params.find<bool>("primary_keepalive", false);
    c.manual_core_drive_enable = params.find<bool>("manual_core_drive_enable", false);
    c.manual_gas_gather_cycles = params.find<uint64_t>("manual_gas_gather_cycles", 200);

    c.verify_weights = params.find<bool>("verify_weights", false);
    c.weight_verify_samples = params.find<uint32_t>("weight_verify_samples", 16);
    c.expected_weight_value = params.find<float>("expected_weight_value", 0.5f);
    c.verify_log_each_sample = params.find<bool>("verify_log_each_sample", false);

    c.use_event_weight_fallback = params.find<bool>("use_event_weight_fallback", false);
    c.enable_memory_weights = params.find<bool>("enable_memory_weights", true);
    c.write_weights_on_init = params.find<bool>("write_weights_on_init", true);

    c.window_stats_enable = params.find<bool>("window_stats_enable", false);
    c.window_us = params.find<uint64_t>("window_us", 20);
    c.window_csv = params.find<std::string>("window_csv", "");
    c.window_metrics_csv = params.find<std::string>("window_metrics_csv", "");
    c.diag_fire_log = params.find<bool>("diag_fire_log", false);

    c.global_step_sync_enable = params.find<bool>("global_step_sync_enable", false);
    c.global_step_done_policy = toLowerCopy(params.find<std::string>("global_step_done_policy", "endscatter"));
    c.global_step_quiescent_min_cycles = clampNonZeroU64(params.find<uint64_t>("global_step_quiescent_min_cycles", 1));
    c.global_step_drain_min_cycles = clampNonZeroU64(params.find<uint64_t>("global_step_drain_min_cycles", 200));
    c.global_step_fixed_cycles = clampNonZeroU64(params.find<uint64_t>("global_step_fixed_cycles", 100000));
    c.loader_done_key = params.find<std::string>("loader_done_key", "");
    c.global_step_ready_delay_cycles = params.find<uint64_t>("global_step_ready_delay_cycles", 0);

    c.step_activation_enable = params.find<bool>("step_activation_enable", false);
    c.step_activation_fraction = params.find<double>("step_activation_fraction", 0.0);
    c.step_activation_fanout = params.find<uint32_t>("step_activation_fanout", 0);
    c.step_activation_seed = params.find<uint64_t>("step_activation_seed", 0xdecafbadULL);
    c.step_activation_period_cycles = params.find<uint64_t>("step_activation_period_cycles", 0);
    c.step_activation_trigger_core = params.find<int>("step_activation_trigger_core", 0);
    c.step_reset_mem_each_step = params.find<bool>("step_reset_mem_each_step", false);
    c.step_activation_event_weight = params.find<double>("step_activation_event_weight", 0.0);
    c.step_activation_pre_pattern = toLowerCopy(params.find<std::string>("step_activation_pre_pattern", "bernoulli"));
    c.step_activation_pre_cluster_len = params.find<uint32_t>("step_activation_pre_cluster_len", 0);
    c.step_activation_use_bcsr_routes = params.find<bool>("step_activation_use_bcsr_routes", false);
    c.step_activation_bcsr_template = params.find<std::string>("step_activation_bcsr_template", "");
    c.step_activation_bcsr_rows_per_core = params.find<uint32_t>("step_activation_bcsr_rows_per_core", 0);
    c.step_activation_bcsr_br = params.find<uint32_t>("step_activation_bcsr_br", 16);
    c.step_activation_bcsr_bc = params.find<uint32_t>("step_activation_bcsr_bc", 16);
    c.step_activation_bcsr_idx_bytes = params.find<uint32_t>("step_activation_bcsr_idx_bytes", 2);
    c.step_activation_bcsr_val_bytes = params.find<uint32_t>("step_activation_bcsr_val_bytes", 4);
    c.step_activation_bcsr_rowptr_offset = params.find<uint64_t>("step_activation_bcsr_rowptr_offset", 0);
    c.step_activation_bcsr_colidx_offset = params.find<uint64_t>("step_activation_bcsr_colidx_offset", 0);
    c.step_activation_bcsr_blockdata_offset = params.find<uint64_t>("step_activation_bcsr_blockdata_offset", 0);
    c.step_activation_bcsr_blockids_offset = params.find<uint64_t>("step_activation_bcsr_blockids_offset", 0);
    c.step_activation_bcsr_weight_epsilon = params.find<double>("step_activation_bcsr_weight_epsilon", 0.0);
    c.step_activation_log_enable = params.find<bool>("step_activation_log_enable", false);
    c.step_activation_build_local_only = params.find<bool>("step_activation_build_local_only", true);
    c.step_activation_bcsr_align = params.find<uint64_t>("step_activation_bcsr_align", 64);

    c.stage_events_csv_path = params.find<std::string>("stage_events_csv", "");
    c.stats_csv_path = params.find<std::string>("stats_csv", "");

    return c;
}

}} // namespace SST::SnnDL
