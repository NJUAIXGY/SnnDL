// -*- c++ -*-
//
// SpikeCommSubsystem: fanout + 事件构造 + 传输调用
//

#include "SpikeCommSubsystem.h"
#include "SpikeEvent.h"

namespace SST { namespace SnnDL {

void SpikeCommSubsystem::init(const SpikeCommConfig& cfg) {
    log_ = cfg.log;
    transport_ = cfg.transport;
    route_ = cfg.route;
    node_id_ = cfg.node_id;
    core_id_ = cfg.core_id;
    global_neuron_base_ = cfg.global_neuron_base;
}

void SpikeCommSubsystem::emitNeuronFire(uint32_t neuron_idx, uint64_t now_cycle) {
    uint32_t source_global = static_cast<uint32_t>(global_neuron_base_ + neuron_idx);
    emitCommon_(source_global, neuron_idx, now_cycle);
}

void SpikeCommSubsystem::emitSource(uint32_t source_global, uint32_t source_local, uint64_t now_cycle) {
    emitCommon_(source_global, source_local, now_cycle);
}

void SpikeCommSubsystem::emitCommon_(uint32_t source_global, uint32_t source_local, uint64_t now_cycle) {
    if (!transport_ || !route_) return;
    std::vector<SnnRouteProvider::FanoutEntry> fanouts;
    bool applied_gating = false;
    route_->computeFanout(source_global, source_local, now_cycle, fanouts, applied_gating);
    if (fanouts.empty()) return;

    for (const auto& fe : fanouts) {
        auto* ev = new SpikeEvent(
            source_global,
            fe.dest_global,
            fe.dest_node,
            /*weight=*/1.0,
            now_cycle);
        // 传输层接管生命周期
        transport_->send(ev);
    }
}

}} // namespace SST::SnnDL
