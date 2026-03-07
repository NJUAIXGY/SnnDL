// -*- c++ -*-
//
// ISynapseRoute: Synapse/Route 子系统的最小路由抽象。
//
// 目标（Phase2）：
// - 将“权重驱动路由的构建 + 共享缓存”从 SpikeCommSubsystem 下移到 Synapse/Route；
// - SpikeCommSubsystem 仅消费最终 routes 来计算 fanout 并发送事件（传输语义不变）。
//
// Phase3 扩展：
// - fanout 计算与门控缓存同样属于“路由/突触语义”，下沉至 Synapse/Route；
// - SpikeCommSubsystem 退化为纯 transport façade（只做事件构造 + send）。

#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace SST { namespace SnnDL {

class ISynapseRoute {
public:
    using RouteMap = std::unordered_map<uint32_t, std::vector<uint32_t>>;

    struct FanoutEntry {
        uint32_t dest_global = 0;
        uint32_t dest_node = 0;
        float weight = 1.0f;
    };

    virtual ~ISynapseRoute() = default;

    // 初始化路由（幂等）。返回 true 表示权重驱动路由可用；false 表示应回退 fixed。
    virtual bool initRoutes() = 0;

    // 当前权重驱动路由是否处于启用/可用状态。
    virtual bool routingWeightDrivenActive() const = 0;

    // 权重驱动路由表（进程级共享）。可能为空（例如未启用或构建失败）。
    virtual std::shared_ptr<const RouteMap> routesShared() const = 0;

    // 可选：本地 fallback 路由表（Phase2 允许为空；保留以兼容现有 fanout provider）。
    virtual const RouteMap* routesLocalFallback() const = 0;

    // 基于“当前路由模式 + gating cache（若开启）”计算 fanout。
    virtual void computeFanout(uint32_t source_global, uint32_t neuron_idx,
                               uint64_t now_cycles,
                               std::vector<FanoutEntry>& out_entries,
                               bool& applied_gating) const = 0;

    // 应用门控决策（若 gating 模式关闭则为 no-op）。
    virtual void applyGatingDecision(uint32_t src_global,
                                     const std::vector<uint32_t>& dest_pes,
                                     uint64_t current_cycle,
                                     uint64_t ttl_cycles) = 0;
};

}} // namespace SST::SnnDL
