// -*- c++ -*-
// Configuration groups owned by the PE core and its optional domains.

#pragma once

#include <cstdint>
#include <string>

namespace SST { namespace SnnDL {

struct SnnPEIdentityConfig {
    bool sentinel_enable = false;
    int core_id = 0;
    int total_cores = 1;
    uint64_t global_neuron_base = 0;
    uint32_t num_neurons = 64;
    uint64_t base_addr = 0;
    uint32_t node_id = 0;
    uint32_t total_nodes = 1;
    int verbose = 0;
};

struct SnnPEWorkloadConfig {
    std::string workload_impl = "snn";
    bool enable_weight_fetch = false;
    bool workload_spike_input_enable = false;
    bool write_weights_on_init = true;
    uint64_t memory_warmup_cycles = 1000;
    float init_default_weight = 0.5f;
    bool readresp_zero_fallback = false;
    uint32_t max_outstanding_requests = 16;
    std::string synapse_weight_mode = "bcsr_gas";
    std::string weights_template;
    uint32_t neurons_per_pe = 0;
    std::string weights_file;
};

struct SnnPECommunicationConfig {
    bool route_summary_enable = false;
    bool merge_read_cacheline = true;
    bool merge_read_row = false;
    bool merge_read_auto = false;
    uint32_t line_size_bytes = 64;
    uint32_t weights_cols = 0;
    std::string index_mode = "pre_row_post_col";
    bool verify_routing_weights = false;
};

struct SnnPEStorageConfig {
    uint32_t max_cache_entries = 65536;
    bool use_clock_weight_cache = false;
    bool disable_weight_cache = false;
    bool use_event_weight_fallback = false;
    std::string dense_layout_mode = "row_major";
    uint32_t dense_phys_dram_row_bytes = 0;
    std::string loader_done_key;
    uint64_t loader_barrier_cycles = 0;
    bool read_force_single = false;
};

struct SnnPERoutingConfig {
    uint32_t bcsr_cols = 0;
    uint32_t bcsr_block_rows = 16;
    uint32_t bcsr_block_cols = 16;
    uint32_t bcsr_idx_bytes = 2;
    uint32_t bcsr_val_bytes = 4;
    uint64_t bcsr_rowptr_offset = 0;
    uint64_t bcsr_colidx_offset = 0;
    uint64_t bcsr_blockdata_offset = 0;
    uint64_t bcsr_blockids_offset = 0;
    std::string bcsr_layout_mode = "flat";
    uint32_t bcsr_colidx_row_stride_bytes = 0;
    uint32_t bcsr_blockdata_row_stride_bytes = 0;
    uint32_t bcsr_blockids_row_stride_bytes = 0;
    std::string bcsr_block_fetch_mode = "full_block";
    uint64_t per_core_stride = 0;
    uint32_t bcsr_row_index_cache_cap = 64;
    uint32_t bcsr_block_cache_cap = 256;
    bool bcsr_prefetch_all = false;
    bool verify_weights = false;
    bool bcsr_force_file_read = false;
    bool bcsr_rowptr_file_fallback_enable = false;
};

struct SnnPEGasConfig {
    bool gas_enable = false;
    bool gas_window_mode = false;
    bool gas_manual_window_drive_requested = false;
    uint64_t gas_window_cycles_gather = 200;
    bool window_read_enable = false;
    bool window_read_debug = false;
    uint32_t scatter_diag_limit = 0;
    uint32_t window_read_budget = 1024;
    uint64_t edge_collector_max_capacity = 1000000;
    bool apply_acc_enable = false;
    uint64_t acc_high_watermark_bytes = 16ull * 1024ull * 1024ull;
    bool acc_spill_enable = true;
    bool apply_dense_acc_enable = true;
    bool acc_shadow_verify_enable = false;
    std::string stage_events_csv;
    bool record_edge_apply_enable = false;
    bool record_edge_idle_enable = false;
    bool record_edge_scatter_enable = false;
};

struct SnnPEVerificationConfig {
    bool byte_exact_verify_enable = false;
    std::string byte_exact_verify_mode;
    uint32_t byte_exact_verify_row_scale = 1024;
    uint32_t byte_exact_verify_max_mismatch = 8;
    bool bcsr_semantic_verify_enable = false;
    uint32_t bcsr_semantic_verify_max_edges = 64;
    uint32_t bcsr_semantic_verify_max_mismatch = 8;
    float bcsr_semantic_verify_abs_tol = 1e-6f;
    float bcsr_semantic_verify_rel_tol = 1e-6f;
};

struct SnnPEDiagnosticsConfig {
    bool enable_extended_diagnostics = false;
    bool profiler_enable = false;
    std::string profiler_csv_prefix;
    bool enable_detailed_map_log = false;
    bool log_weight_details = false;
    bool quiet_finish_logs = false;
};

}} // namespace SST::SnnDL
