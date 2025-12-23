// -*- c++ -*-
//
// SpikeCommSubsystem: 通信子系统，封装 fanout + 事件构造 + 传输调用。
//

#pragma once

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include <sst/core/output.h>

#include "ISpikeTransport.h"
#include "SnnRouteProvider.h"

namespace SST { namespace SnnDL {

class SpikeEvent;

struct SpikeCommRoutingConfig {
    bool routing_weight_driven = false;
    bool log_weight_details = false;
    bool verify_routing_weights = false;
    bool route_summary_enable = false;

    // Route inputs (keep consistent with legacy cache key semantics).
    uint32_t rows = 0;            // per-core rows (num_neurons)
    uint32_t cols = 0;            // global cols (weights_cols)
    uint32_t total_nodes = 16;    // total PEs
    uint32_t cores_per_pe = 1;    // total_cores
    uint32_t neurons_per_pe = 0;  // derived; used for denom & cache-key
    bool use_post_row_pre_col = false;

    std::string weights_template;
    float routing_epsilon = 1e-8f;
    uint32_t routing_topk = 0;
    uint32_t routing_topk_per_pe = 0;
    bool route_exclude_self_pe = false;
    std::string route_layers_mask;
    bool route_filter_warn = true;

    // Mapping integration
    std::string mapping_mode;
    std::string mapping_edges_file;
    bool mapping_csv_has_header = true;
    std::string mapping_csv_separator = ",";
    bool mapping_assume_block_ids = true;

    // Optional: BCSR route parsing parameters (used when weights_template points to BCSR bin).
    bool use_bcsr = false;
    uint32_t bcsr_br = 0;
    uint32_t bcsr_bc = 0;
    uint32_t bcsr_idx_bytes = 0;
    uint32_t bcsr_val_bytes = 0;
    uint64_t base_addr = 0;
    uint64_t bcsr_rowptr_addr = 0;
    uint64_t bcsr_colidx_addr = 0;
    uint64_t bcsr_blockdata_addr = 0;
    uint64_t bcsr_blockids_addr = 0;
};

struct SpikeCommGatingConfig {
    bool gating_event_mode = false;
    uint64_t gating_ttl_cycles = 1000;
    bool gating_scope_inputs_only = true;
};

struct SpikeCommRuntimeConfig {
    Output* log = nullptr;
    ISpikeTransport* transport = nullptr;
    uint32_t node_id = 0;
    uint32_t core_id = 0;
    uint64_t global_neuron_base = 0;
    uint32_t num_neurons = 0;
    uint32_t neurons_per_pe_cfg = 0;

    SST::Statistics::Statistic<uint64_t>* stat_fanout = nullptr;
    SST::Statistics::Statistic<uint64_t>* stat_routes_entries = nullptr;
};

class SpikeCommSubsystem {
public:
    using RouteMap = SnnRouteProvider::RouteMap;

    void configure(const SpikeCommRoutingConfig& routing_cfg,
                   const SpikeCommGatingConfig& gating_cfg);

    void bindRuntime(const SpikeCommRuntimeConfig& rt);

    // Build (or reuse) shared route tables and configure the internal route provider.
    // Safe to call multiple times; subsequent calls will refresh pointers/stats.
    void initRouting();

    // 常规入口：compute core 报告本地 neuron_idx
    void emitNeuronFire(uint32_t neuron_idx, uint64_t now_cycle);
    // 已知 source_global 的入口（保留扩展用途）
    void emitSource(uint32_t source_global, uint32_t source_local, uint64_t now_cycle);

    // Apply gating decision (from parent/PE). No-op when gating mode is off.
    void applyGatingDecision(uint32_t src_global,
                             const std::vector<uint32_t>& dest_pes,
                             uint64_t current_cycle,
                             uint64_t ttl_cycles);

    bool ready() const { return transport_ && route_provider_ready_; }

private:
    void emitCommon_(uint32_t source_global, uint32_t source_local, uint64_t now_cycle);
    void logRoutingSummary_(const char* phase, const char* reason = nullptr) const;

    Output* log_ = nullptr;
    ISpikeTransport* transport_ = nullptr;     // 非拥有；由外部管理生命周期
    uint32_t node_id_ = 0;
    uint32_t core_id_ = 0;
    uint64_t global_neuron_base_ = 0;
    uint32_t num_neurons_ = 0;
    uint32_t neurons_per_pe_cfg_ = 0;

    // Internal route provider + route tables (shared across cores in the same process).
    bool route_provider_ready_ = false;
    SnnRouteProvider route_provider_;
    std::shared_ptr<const RouteMap> routes_shared_;
    RouteMap routes_local_fallback_;

    // Gating state
    SpikeCommGatingConfig gating_cfg_{};
    std::unordered_map<uint32_t, GatingEntry> gating_cache_;

    // Routing build config
    SpikeCommRoutingConfig routing_cfg_{};

    // Statistics hooks (owned by control component)
    SST::Statistics::Statistic<uint64_t>* stat_fanout_per_spike_ = nullptr;
    SST::Statistics::Statistic<uint64_t>* stat_routes_entries_ = nullptr;
    bool route_summary_logged_ = false;

    // Process-wide shared route cache.
    static std::mutex s_route_cache_mtx_;
    static std::unordered_map<std::string, std::weak_ptr<const RouteMap>> s_route_cache_;
};

}} // namespace SST::SnnDL
