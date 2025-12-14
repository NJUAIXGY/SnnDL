// -*- c++ -*-
//
// SpikeCommSubsystem: 通信子系统，封装 fanout + 事件构造 + 传输调用。
//

#pragma once

#include <cstdint>
#include <vector>

#include <sst/core/output.h>

#include "ISpikeTransport.h"
#include "SnnRouteProvider.h"

namespace SST { namespace SnnDL {

class SpikeEvent;

struct SpikeCommConfig {
    Output* log = nullptr;
    ISpikeTransport* transport = nullptr;
    SnnRouteProvider* route = nullptr;
    uint32_t node_id = 0;
    uint32_t core_id = 0;
    uint64_t global_neuron_base = 0;
};

class SpikeCommSubsystem {
public:
    void init(const SpikeCommConfig& cfg);

    // 常规入口：compute core 报告本地 neuron_idx
    void emitNeuronFire(uint32_t neuron_idx, uint64_t now_cycle);
    // 已知 source_global 的入口（保留扩展用途）
    void emitSource(uint32_t source_global, uint32_t source_local, uint64_t now_cycle);

    bool ready() const { return transport_ && route_; }

private:
    void emitCommon_(uint32_t source_global, uint32_t source_local, uint64_t now_cycle);

    Output* log_ = nullptr;
    ISpikeTransport* transport_ = nullptr;     // 非拥有；由外部管理生命周期
    SnnRouteProvider* route_ = nullptr;        // 非拥有；复用控制层的路由提供器
    uint32_t node_id_ = 0;
    uint32_t core_id_ = 0;
    uint64_t global_neuron_base_ = 0;
};

}} // namespace SST::SnnDL
