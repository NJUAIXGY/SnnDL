// -*- c++ -*-
//
// SpikeCommSubsystem: fanout + 事件构造 + 传输调用
//

#include "SpikeCommSubsystem.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <inttypes.h>
#include <iterator>
#include <limits>
#include <sstream>
#include <unordered_set>

#include <sst/core/statapi/stataccumulator.h>

#include "INocTransport.h"
#include "SpikeNocCodec.h"
#include "SpikeEvent.h"
#include "SynapseRouteSubsystem.h"

namespace SST { namespace SnnDL {


void SpikeCommSubsystem::configure() {
    route_provider_ready_ = false;
    emit_seq_ = 0;
}

void SpikeCommSubsystem::bindRuntime(const SpikeCommRuntimeConfig& rt) {
    if (rt.log) log_ = rt.log;
    if (rt.transport) transport_ = rt.transport;
    if (rt.noc) noc_ = rt.noc;
    src_core_ = rt.src_core;
    node_id_ = rt.node_id;
    if (rt.synapse_route) synapse_route_ = rt.synapse_route;
    global_neuron_base_ = rt.global_neuron_base;
    route_provider_ready_ = false;
    emit_seq_ = 0;
}

void SpikeCommSubsystem::initRouting() {
    route_provider_ready_ = false;
    if (!synapse_route_) return;
    // Phase3：fanout provider 已下沉至 Synapse/Route；SpikeComm 仅确保其初始化。
    (void)synapse_route_->initRoutes();
    route_provider_ready_ = true;
}

void SpikeCommSubsystem::emitNeuronFire(uint32_t neuron_idx, uint64_t now_cycle) {
    uint32_t source_global = static_cast<uint32_t>(global_neuron_base_ + neuron_idx);
    emitCommon_(source_global, neuron_idx, now_cycle);
}

uint64_t SpikeCommSubsystem::emitNeuronFireBatch(const std::vector<uint32_t>& neuron_indices, uint64_t now_cycle) {
    if (neuron_indices.empty()) return 0;
    uint64_t emitted = 0;
    for (uint32_t neuron_idx : neuron_indices) {
        emitNeuronFire(neuron_idx, now_cycle);
        emitted += 1;
    }
    return emitted;
}

void SpikeCommSubsystem::emitSource(uint32_t source_global, uint32_t source_local, uint64_t now_cycle) {
    emitCommon_(source_global, source_local, now_cycle);
}

void SpikeCommSubsystem::emitCommon_(uint32_t source_global, uint32_t source_local, uint64_t now_cycle) {
    if (!route_provider_ready_ || !synapse_route_) return;

    if (noc_) {
        const auto* sr = dynamic_cast<const SynapseRouteSubsystem*>(synapse_route_);
        if (sr && sr->multicastEnabled()) {
            std::vector<SynapseRouteSubsystem::BlockTarget> targets;
            bool applied_gating = false;
            if (sr->computeMulticastTargets(source_global, source_local, now_cycle, targets, applied_gating) && !targets.empty()) {
                const uint32_t bw = sr->multicastBlockW();
                const uint32_t bh = sr->multicastBlockH();
                for (const auto& t : targets) {
                    SpikeNocCodec::WireSpikeKeyV2 ws{};
                    ws.version = 2;
                    ws.route_mode = 1; // blocked
                    ws.stage = 0;      // INTER
                    ws.block_w_h = static_cast<uint16_t>(((bw & 0xffu) << 8) | (bh & 0xffu));
                    ws.block_id = t.block_id;
                    ws.ingress_node = t.ingress_node;
                    ws.pre_global = source_global;
                    // group_id：一次仿真运行内必须“每次发射唯一”。
                    // 这里使用“单调递增序号 + pre_global”的结构化组合，避免 XOR/哈希导致的周期性重复与碰撞。
                    const uint64_t seq = ++emit_seq_;
                    ws.group_id = (seq << 32) | static_cast<uint64_t>(source_global);
                    ws.core_mask = t.core_mask;

                    auto* pkt = new NocPacketEvent();
                    pkt->src_node = node_id_;
                    pkt->dst_node = t.ingress_node;
                    pkt->src_endpoint = static_cast<uint16_t>(src_core_);
                    pkt->dst_endpoint = 0;
                    pkt->kind = static_cast<uint16_t>(NocPacketKind::SpikeKey);
                    pkt->timestamp = now_cycle;
                    SpikeNocCodec::encodeSpikeKey(ws, pkt->payload);

                    noc_->sendFromCore(src_core_, pkt);
                }
                return;
            }
        }
    }

    if (!transport_) return;
    std::vector<ISynapseRoute::FanoutEntry> fanouts;
    bool applied_gating = false;
    synapse_route_->computeFanout(source_global, source_local, now_cycle, fanouts, applied_gating);
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

void SpikeCommSubsystem::applyGatingDecision(uint32_t src_global,
                                             const std::vector<uint32_t>& dest_pes,
                                             uint64_t current_cycle,
                                             uint64_t ttl_cycles) {
    if (!synapse_route_) return;
    synapse_route_->applyGatingDecision(src_global, dest_pes, current_cycle, ttl_cycles);
}

}} // namespace SST::SnnDL
