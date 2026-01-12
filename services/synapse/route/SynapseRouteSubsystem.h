// -*- c++ -*-
//
// SynapseRouteSubsystem:
// - 负责权重驱动路由（dense/BCSR/edges_csv）的构建与进程级共享缓存；
// - 对外提供 routes（供 fanout provider 使用）。
//
// Phase2 目标：把路由构建从 SpikeCommSubsystem 下移到 Synapse/Route 子系统，
// 保持现有扇出语义不变（仍复用 SnnRouteProvider）。

#pragma once

#include <cstdint>
#include <array>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "ISynapseRoute.h"
#include "api/MulticastLimits.h"
#include "SynapseRouteBuildConfig.h"
#include "SnnRouteProvider.h"

namespace SST { class Output; }
namespace SST { namespace Statistics { template <typename T> class Statistic; } }

namespace SST { namespace SnnDL {

class SynapseRouteSubsystem final : public ISynapseRoute {
public:
    using RouteMap = ISynapseRoute::RouteMap;

    struct BlockTarget final {
        uint32_t block_id = 0;
        uint32_t ingress_node = 0;
        std::array<uint32_t, kMaxMulticastBlockCells> core_mask{};
    };

    void configure(const SynapseRouteBuildConfig& cfg);
    void configureGating(bool gating_event_mode,
                         uint64_t gating_ttl_cycles,
                         bool gating_scope_inputs_only);

    void bindRuntime(Output* log,
                     uint32_t node_id,
                     uint32_t core_id,
                     uint32_t num_neurons,
                     uint32_t neurons_per_pe_cfg,
                     SST::Statistics::Statistic<uint64_t>* stat_routes_entries);
    void bindFanoutStat(SST::Statistics::Statistic<uint64_t>* stat_fanout_per_spike);

    // ISynapseRoute
    bool initRoutes() override;
    bool routingWeightDrivenActive() const override { return routing_weight_driven_active_; }
    std::shared_ptr<const RouteMap> routesShared() const override { return routes_shared_; }
    const RouteMap* routesLocalFallback() const override { return &routes_local_fallback_; }
    void computeFanout(uint32_t source_global, uint32_t neuron_idx,
                       uint64_t now_cycles,
                       std::vector<FanoutEntry>& out_entries,
                       bool& applied_gating) const override;
    void applyGatingDecision(uint32_t src_global,
                             const std::vector<uint32_t>& dest_pes,
                             uint64_t current_cycle,
                             uint64_t ttl_cycles) override;

    bool multicastEnabled() const;
    uint32_t multicastBlockW() const { return cfg_.multicast_block_w; }
    uint32_t multicastBlockH() const { return cfg_.multicast_block_h; }

    // 计算 blocked multicast targets（不改变 ISynapseRoute 接口，仅供 SpikeCommSubsystem 使用）。
    // applied_gating=true 表示命中 gating 且已对目标集合做过滤。
    bool computeMulticastTargets(uint32_t source_global,
                                 uint32_t neuron_idx,
                                 uint64_t now_cycles,
                                 std::vector<BlockTarget>& out_targets,
                                 bool& applied_gating) const;

private:
    void logRoutingSummary_(const char* phase, const char* reason) const;
    void configureFanoutProvider_();
    void initMulticastTargets_();

    SynapseRouteBuildConfig cfg_{};
    Output* log_ = nullptr;
    uint32_t node_id_ = 0;
    uint32_t core_id_ = 0;
    uint32_t num_neurons_ = 0;
    uint32_t neurons_per_pe_cfg_ = 0;
    SST::Statistics::Statistic<uint64_t>* stat_routes_entries_ = nullptr;
    SST::Statistics::Statistic<uint64_t>* stat_fanout_per_spike_ = nullptr;

    bool routing_weight_driven_active_ = false;
    bool route_summary_logged_ = false;
    std::shared_ptr<const RouteMap> routes_shared_;
    RouteMap routes_local_fallback_;

    // Fanout provider (Phase3: 从 SpikeCommSubsystem 下沉到 Synapse/Route)。
    bool fanout_provider_ready_ = false;
    SnnRouteProvider fanout_provider_;

    // Gating state (Phase3: 归入 Synapse/Route)。
    bool gating_event_mode_ = false;
    uint64_t gating_ttl_cycles_ = 1000;
    bool gating_scope_inputs_only_ = true;
    std::unordered_map<uint32_t, GatingEntry> gating_cache_;

    // Native multicast targets (MVP: 2x2 blocked)
    using MulticastTargetMap = std::unordered_map<uint32_t, std::vector<BlockTarget>>;
    bool multicast_ready_ = false;
    uint32_t mesh_w_ = 0;
    uint32_t mesh_h_ = 0;
    std::shared_ptr<const MulticastTargetMap> multicast_targets_shared_;
    MulticastTargetMap multicast_targets_local_;

    static std::mutex s_route_cache_mtx_;
    static std::unordered_map<std::string, std::weak_ptr<const RouteMap>> s_route_cache_;

    static std::mutex s_multicast_cache_mtx_;
    static std::unordered_map<std::string, std::weak_ptr<const MulticastTargetMap>> s_multicast_cache_;
};

}} // namespace SST::SnnDL
