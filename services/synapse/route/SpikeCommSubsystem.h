// -*- c++ -*-
//
// SpikeCommSubsystem: 通信子系统，封装 fanout + 事件构造 + 传输调用。
//

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <sst/core/output.h>

#include "ISpikeTransport.h"
#include "ISynapseRoute.h"

namespace SST { namespace SnnDL {

class SpikeEvent;
class INocTransport;

struct SpikeCommRuntimeConfig {
    Output* log = nullptr;
    ISpikeTransport* transport = nullptr;
    INocTransport* noc = nullptr;  // optional (native multicast path uses packet injection)
    int src_core = 0;
    uint32_t node_id = 0;
    ISynapseRoute* synapse_route = nullptr;
    uint64_t global_neuron_base = 0;
};

class SpikeCommSubsystem {
public:
    void configure();

    void bindRuntime(const SpikeCommRuntimeConfig& rt);

    // Ensure Synapse/Route routing+fanout provider is initialized (idempotent).
    void initRouting();

    // 常规入口：compute core 报告本地 neuron_idx
    void emitNeuronFire(uint32_t neuron_idx, uint64_t now_cycle);
    // 批量入口：控制层已收敛好 fired 集合，仅需统一下发（不引入 compute::FireEvent 依赖）
    uint64_t emitNeuronFireBatch(const std::vector<uint32_t>& neuron_indices, uint64_t now_cycle);
    // 已知 source_global 的入口（保留扩展用途）
    void emitSource(uint32_t source_global, uint32_t source_local, uint64_t now_cycle);

    // Apply gating decision (from parent/PE). Delegated to Synapse/Route.
    void applyGatingDecision(uint32_t src_global,
                             const std::vector<uint32_t>& dest_pes,
                             uint64_t current_cycle,
                             uint64_t ttl_cycles);

    bool ready() const { return transport_ && route_provider_ready_; }

private:
    void emitCommon_(uint32_t source_global, uint32_t source_local, uint64_t now_cycle);

    Output* log_ = nullptr;
    ISpikeTransport* transport_ = nullptr;     // 非拥有；由外部管理生命周期
    INocTransport* noc_ = nullptr;            // 非拥有；由外部管理生命周期
    int src_core_ = 0;
    uint32_t node_id_ = 0;
    uint64_t global_neuron_base_ = 0;
    uint64_t emit_seq_ = 0; // monotonic per run; used to ensure group_id uniqueness under high-rate injection

    bool route_provider_ready_ = false;
    ISynapseRoute* synapse_route_ = nullptr;  // 非拥有；由控制层装配并保证生命周期

};

}} // namespace SST::SnnDL
