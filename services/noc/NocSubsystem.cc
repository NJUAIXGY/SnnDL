// -*- c++ -*-
//
// NocSubsystem implementation
//

#include "NocSubsystem.h"

#include <cinttypes>

#include <sst/core/event.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <sst/core/statapi/stataccumulator.h>

#include "OptimizedInternalRing.h"
#include "SnnInterface.h"
#include "NocPacketEvent.h"

namespace SST { namespace SnnDL {

#ifndef NOC_LOG
#define NOC_LOG(lvl, ...) do { if (rt_.log && cfg_.log_enable) rt_.log->verbose(CALL_INFO, (lvl), 0, __VA_ARGS__); } while(0)
#endif

NocSubsystem::~NocSubsystem() {
    while (!incoming_queue_.empty()) {
        delete incoming_queue_.front();
        incoming_queue_.pop();
    }
}

void NocSubsystem::configure(const Config& cfg) {
    cfg_ = cfg;
}

void NocSubsystem::bindRuntime(const Runtime& rt) {
    rt_ = rt;
    if (rt_.num_cores <= 0) rt_.num_cores = 1;
}

void NocSubsystem::bindStats(const Stats& st) {
    st_ = st;
}

void NocSubsystem::enqueueIncoming_(NocPacketEvent* packet) {
    if (!packet) return;
    incoming_queue_.push(packet);
}

void NocSubsystem::sendExternalPacket_(NocPacketEvent* packet) {
    if (!packet) return;

    // 自环防护：如果目标节点就是本节点，直接丢弃，避免外部回送循环
    const int target_node = static_cast<int>(packet->dst_node);
    if (target_node == rt_.node_id) {
        delete packet;
        return;
    }

    // 优先使用 NIC；未配置则回退到 legacy output link
    if (rt_.nic) {
        rt_.nic->sendToNode(packet->dst_node, packet);
    } else if (rt_.external_spike_output_link) {
        rt_.external_spike_output_link->send(packet);
    } else {
        delete packet;
        return;
    }

    if (st_.external_spikes_sent) st_.external_spikes_sent->addData(1);
}

void NocSubsystem::forwardExternalPacket_(NocPacketEvent* packet) {
    if (!packet) return;
    // 中继转发语义：仅调用 nic->sendToNode，不更新 external_spikes_sent
    if (rt_.nic) {
        rt_.nic->sendToNode(packet->dst_node, packet);
    } else {
        delete packet;
    }
}

void NocSubsystem::routeInternalPacket_(int src_core, int dst_core, NocPacketEvent* packet) {
    if (!packet) return;

    if (src_core < 0 || src_core >= rt_.num_cores || dst_core < 0 || dst_core >= rt_.num_cores) {
        delete packet;
        return;
    }

    // 单核情况或同一核心内，直接递送
    if (rt_.num_cores <= 1 || src_core == dst_core || !rt_.optimized_ring) {
        if (rt_.deliver_to_endpoint) rt_.deliver_to_endpoint(dst_core, packet);
        else delete packet;
        return;
    }

    RingMessage msg;
    msg.type = RingMessageType::PACKET_MESSAGE;
    msg.src_unit = src_core;
    msg.dst_unit = dst_core;
    msg.timestamp = 0;
    msg.priority = 1;
    msg.payload.packet = packet;

    bool sent = rt_.optimized_ring->sendMessage(src_core, dst_core, msg, 1);
    if (sent) {
        if (st_.inter_core_messages) st_.inter_core_messages->addData(1);
    } else {
        delete packet;
    }
}

void NocSubsystem::onCoreSend(NocPacketEvent* packet) {
    if (!packet) return;

    const bool is_local = (static_cast<int>(packet->dst_node) == rt_.node_id);
    if (is_local) {
        const int dst_core = static_cast<int>(packet->dst_endpoint);
        const int src_core = static_cast<int>(packet->src_endpoint);
        if (dst_core >= 0 && dst_core < rt_.num_cores) {
            // src_core 若不合法，按单核直投语义处理
            if (src_core >= 0 && src_core < rt_.num_cores) {
                routeInternalPacket_(src_core, dst_core, packet);
            } else {
                if (rt_.deliver_to_endpoint) rt_.deliver_to_endpoint(dst_core, packet);
                else delete packet;
            }
        } else {
            delete packet;
        }
        return;
    }

    sendExternalPacket_(packet);
}

void NocSubsystem::sendFromCore(int src_core, NocPacketEvent* packet) {
    if (!packet) return;

    const bool is_local = (static_cast<int>(packet->dst_node) == rt_.node_id);
    if (is_local) {
        const int dst_core = static_cast<int>(packet->dst_endpoint);
        if (dst_core >= 0 && dst_core < rt_.num_cores) {
            if (src_core >= 0 && src_core < rt_.num_cores) {
                routeInternalPacket_(src_core, dst_core, packet);
            } else {
                if (rt_.deliver_to_endpoint) rt_.deliver_to_endpoint(dst_core, packet);
                else delete packet;
            }
        } else {
            delete packet;
        }
        return;
    }

    sendExternalPacket_(packet);
}

void NocSubsystem::injectLocal(int dst_core, NocPacketEvent* packet) {
    if (!packet) return;
    if (dst_core < 0 || dst_core >= rt_.num_cores) {
        delete packet;
        return;
    }
    if (rt_.deliver_to_endpoint) rt_.deliver_to_endpoint(dst_core, packet);
    else delete packet;
}

void NocSubsystem::sendExternal(NocPacketEvent* packet) {
    sendExternalPacket_(packet);
}

void NocSubsystem::onNicReceive(NocPacketEvent* packet) {
    if (!packet) return;
    if (st_.external_spikes_received) st_.external_spikes_received->addData(1);
    enqueueIncoming_(packet);
}

void NocSubsystem::onExternalPortEvent(SST::Event* event) {
    auto* packet = dynamic_cast<NocPacketEvent*>(event);
    if (!packet) {
        delete event;
        return;
    }

    // hop/TTL 保护（保持旧语义：避免端口回送循环）
    constexpr uint16_t kMaxHops = 10;
    if (packet->hop_count >= kMaxHops) {
        delete packet;
        return;
    }
    packet->hop_count += 1;

    if (st_.external_spikes_received) st_.external_spikes_received->addData(1);

    const uint32_t dest_node = packet->dst_node;
    const bool is_local = (dest_node == static_cast<uint32_t>(rt_.node_id));

    if (is_local) {
        enqueueIncoming_(packet);
        return;
    }

    // 目标不在本节点，走外发（保持旧语义：sendExternal 计入 external_spikes_sent）
    sendExternalPacket_(packet);
}

void NocSubsystem::onDirectionalLinkEvent(SST::Event* event, const std::string& direction) {
    if (!event) return;

    if (auto* packet = dynamic_cast<NocPacketEvent*>(event)) {
        NOC_LOG(4, "[noc] link=%s packet src_node=%u dst_node=%u src_ep=%u dst_ep=%u kind=%u hop=%u\n",
                direction.c_str(),
                packet->src_node, packet->dst_node,
                packet->src_endpoint, packet->dst_endpoint,
                packet->kind, packet->hop_count);
        onNicReceive(packet);
        return;
    }

    // 非脉冲事件，直接丢弃
    delete event;
}

void NocSubsystem::drainIncomingQueue(uint64_t /*current_cycle*/) {
    while (!incoming_queue_.empty()) {
        NocPacketEvent* packet = incoming_queue_.front();
        incoming_queue_.pop();

        const bool is_local = (static_cast<int>(packet->dst_node) == rt_.node_id);

        if (is_local) {
            const int target_unit = static_cast<int>(packet->dst_endpoint);
            if (target_unit >= 0 && target_unit < rt_.num_cores) {
                if (rt_.deliver_to_endpoint) rt_.deliver_to_endpoint(target_unit, packet);
                else delete packet;
            } else {
                delete packet;
            }
            continue;
        }

        // 中继转发（保持原 clockTick 语义：不更新 external_spikes_sent）
        forwardExternalPacket_(packet);
    }
}

void NocSubsystem::tickRing(uint64_t current_cycle) {
    tickOptimizedRing_(current_cycle);
}

void NocSubsystem::tickOptimizedRing_(uint64_t current_cycle) {
    if (!rt_.optimized_ring) return;

    rt_.optimized_ring->tick(current_cycle);

    // 与历史 MultiCorePE 行为保持一致：对每个 core 轮询 ejection queue
    for (int i = 0; i < rt_.num_cores; i++) {
        RingMessage msg;
        while (rt_.optimized_ring->receiveMessage(i, msg)) {
            if (msg.type == RingMessageType::PACKET_MESSAGE && msg.payload.packet) {
                int target_unit = msg.dst_unit;
                auto* packet = static_cast<NocPacketEvent*>(msg.payload.packet);
                if (target_unit >= 0 && target_unit < rt_.num_cores) {
                    if (rt_.deliver_to_endpoint) rt_.deliver_to_endpoint(target_unit, packet);
                    else delete packet;

                    // 与历史口径保持一致：receive 侧也计一次 inter_core_messages
                    if (st_.inter_core_messages) st_.inter_core_messages->addData(1);
                } else {
                    delete packet;
                }
            } else {
                // non-spike message: keep legacy behavior (ignore)
            }
        }
    }
}

}} // namespace SST::SnnDL
