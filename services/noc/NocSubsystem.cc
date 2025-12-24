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
#include "SpikeEvent.h"
#include "SpikeEventWrapper.h"

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

void NocSubsystem::enqueueIncoming_(SpikeEvent* spike) {
    if (!spike) return;
    incoming_queue_.push(spike);
}

void NocSubsystem::sendExternalSpike_(SpikeEvent* spike) {
    if (!spike) return;

    // 自环防护：如果目标节点就是本节点，直接丢弃，避免外部回送循环
    int target_node = static_cast<int>(spike->getDestinationNode());
    if (target_node == rt_.node_id) {
        delete spike;
        return;
    }

    // 优先使用 NIC；未配置则回退到 legacy output link
    if (rt_.nic) {
        rt_.nic->sendSpike(spike);
    } else if (rt_.external_spike_output_link) {
        rt_.external_spike_output_link->send(spike);
    } else {
        delete spike;
        return;
    }

    if (st_.external_spikes_sent) st_.external_spikes_sent->addData(1);
}

void NocSubsystem::forwardExternalSpike_(SpikeEvent* spike) {
    if (!spike) return;
    // 中继转发语义：仅调用 nic->sendSpike，不更新 external_spikes_sent
    if (rt_.nic) {
        rt_.nic->sendSpike(spike);
    } else {
        delete spike;
    }
}

void NocSubsystem::routeInternalSpike_(int src_core, int dst_core, SpikeEvent* spike) {
    if (!spike) return;

    if (src_core < 0 || src_core >= rt_.num_cores || dst_core < 0 || dst_core >= rt_.num_cores) {
        delete spike;
        return;
    }

    // 单核情况或同一核心内，直接递送
    if (rt_.num_cores <= 1 || src_core == dst_core || !rt_.optimized_ring) {
        if (rt_.deliver_to_core) rt_.deliver_to_core(dst_core, spike);
        else delete spike;
        return;
    }

    RingMessage msg;
    msg.type = RingMessageType::SPIKE_MESSAGE;
    msg.src_unit = src_core;
    msg.dst_unit = dst_core;
    msg.timestamp = 0;
    msg.priority = 1;
    msg.payload.spike_data = spike;

    bool sent = rt_.optimized_ring->sendMessage(src_core, dst_core, msg, 1);
    if (sent) {
        if (st_.inter_core_messages) st_.inter_core_messages->addData(1);
    } else {
        delete spike;
    }
}

void NocSubsystem::onCoreSend(SpikeEvent* event) {
    if (!event) return;

    const int target_unit =
        rt_.determine_target_unit ? rt_.determine_target_unit(static_cast<int>(event->getDestinationNeuron())) : -1;

    if (target_unit >= 0 && target_unit < rt_.num_cores) {
        const int src_core =
            rt_.determine_target_unit ? rt_.determine_target_unit(static_cast<int>(event->getSourceNeuron())) : -1;

        if (src_core >= 0 && src_core < rt_.num_cores) {
            routeInternalSpike_(src_core, target_unit, event);
            return;
        }
        if (rt_.deliver_to_core) rt_.deliver_to_core(target_unit, event);
        else delete event;
        return;
    }

    sendExternalSpike_(event);
}

void NocSubsystem::sendFromCore(int src_core, SpikeEvent* event) {
    if (!event) return;

    const int target_unit =
        rt_.determine_target_unit ? rt_.determine_target_unit(static_cast<int>(event->getDestinationNeuron())) : -1;

    if (target_unit >= 0 && target_unit < rt_.num_cores) {
        if (src_core >= 0 && src_core < rt_.num_cores) {
            routeInternalSpike_(src_core, target_unit, event);
        } else {
            if (rt_.deliver_to_core) rt_.deliver_to_core(target_unit, event);
            else delete event;
        }
        return;
    }

    sendExternalSpike_(event);
}

void NocSubsystem::injectLocal(int dst_core, SpikeEvent* event) {
    if (!event) return;
    if (dst_core < 0 || dst_core >= rt_.num_cores) {
        delete event;
        return;
    }
    if (rt_.deliver_to_core) rt_.deliver_to_core(dst_core, event);
    else delete event;
}

void NocSubsystem::sendExternal(SpikeEvent* event) {
    sendExternalSpike_(event);
}

void NocSubsystem::onNicReceive(SpikeEvent* spike) {
    if (!spike) return;
    if (st_.external_spikes_received) st_.external_spikes_received->addData(1);
    enqueueIncoming_(spike);
}

void NocSubsystem::onExternalPortEvent(SST::Event* event) {
    SpikeEvent* spike = dynamic_cast<SpikeEvent*>(event);
    if (!spike) {
        delete event;
        return;
    }

    // hop/TTL 保护（保持 handleExternalSpikeEvent 语义）
    if (spike->isExpired()) {
        delete spike;
        return;
    }
    spike->incrementHopCount();

    if (st_.external_spikes_received) st_.external_spikes_received->addData(1);

    const uint32_t dest_node = spike->getDestinationNode();
    const bool is_local = (dest_node == static_cast<uint32_t>(rt_.node_id));

    if (is_local) {
        enqueueIncoming_(spike);
        return;
    }

    const int target_unit =
        rt_.determine_target_unit ? rt_.determine_target_unit(static_cast<int>(spike->getDestinationNeuron())) : -1;

    if (target_unit >= 0 && target_unit < rt_.num_cores) {
        // 目标在本 PE（保持原实现：复制一份再投递，避免潜在所有权/缓存问题）
        SpikeEvent* cross_core_spike = new SpikeEvent(
            spike->getSourceNeuron(),
            spike->getDestinationNeuron(),
            spike->getDestinationNode(),
            spike->getWeight(),
            spike->getSpikeTime()
        );
        cross_core_spike->hop_count = spike->getHopCount();
        if (rt_.deliver_to_core) rt_.deliver_to_core(target_unit, cross_core_spike);
        else delete cross_core_spike;
        delete spike;
        return;
    }

    // 目标不在本 PE，走外发（保持原实现：sendExternalSpike 接管生命周期）
    sendExternalSpike_(spike);
}

SpikeEvent* NocSubsystem::extractSpikeFromWrapper_(SpikeEventWrapper* wrapper) {
    if (!wrapper) return nullptr;

    // SpikeEventWrapper 析构不释放 spike_data，接收端必须接管所有权。
    SpikeEvent* spike = wrapper->getSpikeEvent();
    wrapper->setSpikeEvent(nullptr);
    delete wrapper;

    if (!spike) return nullptr;
    // 清理 sender 侧可能遗留的本地缓存字段，避免跨 PE/跨 core 复用带来的脏状态。
    spike->clearLocalCache();
    return spike;
}

void NocSubsystem::onDirectionalLinkEvent(SST::Event* event, const std::string& direction) {
    if (!event) return;

    // 直接 SpikeEvent
    if (auto* spike_event = dynamic_cast<SpikeEvent*>(event)) {
        NOC_LOG(4, "[noc] link=%s direct spike src=%u dst=%u node=%u hop=%u\n",
                direction.c_str(),
                spike_event->getSourceNeuron(),
                spike_event->getDestinationNeuron(),
                spike_event->getDestinationNode(),
                spike_event->getHopCount());
        onNicReceive(spike_event);
        return;
    }

    // SpikeEventWrapper
    if (auto* wrapper_event = dynamic_cast<SpikeEventWrapper*>(event)) {
        SpikeEvent* extracted = extractSpikeFromWrapper_(wrapper_event);
        if (extracted) {
            NOC_LOG(4, "[noc] link=%s wrapper spike src=%u dst=%u node=%u hop=%u\n",
                    direction.c_str(),
                    extracted->getSourceNeuron(),
                    extracted->getDestinationNeuron(),
                    extracted->getDestinationNode(),
                    extracted->getHopCount());
            onNicReceive(extracted);
        }
        return;
    }

    // 非脉冲事件，直接丢弃
    delete event;
}

void NocSubsystem::drainIncomingQueue(uint64_t /*current_cycle*/) {
    while (!incoming_queue_.empty()) {
        SpikeEvent* spike = incoming_queue_.front();
        incoming_queue_.pop();

        const int target_unit =
            rt_.determine_target_unit ? rt_.determine_target_unit(static_cast<int>(spike->getDestinationNeuron())) : -1;

        if (target_unit >= 0 && target_unit < rt_.num_cores) {
            if (rt_.deliver_to_core) rt_.deliver_to_core(target_unit, spike);
            else delete spike;
            continue;
        }

        // 中继转发（保持原 clockTick 语义：不更新 external_spikes_sent）
        forwardExternalSpike_(spike);
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
            if (msg.type == RingMessageType::SPIKE_MESSAGE && msg.payload.spike_data) {
                int target_unit = msg.dst_unit;
                if (target_unit >= 0 && target_unit < rt_.num_cores) {
                    if (rt_.deliver_to_core) rt_.deliver_to_core(target_unit, msg.payload.spike_data);
                    else delete msg.payload.spike_data;

                    // 与历史口径保持一致：receive 侧也计一次 inter_core_messages
                    if (st_.inter_core_messages) st_.inter_core_messages->addData(1);
                } else {
                    delete msg.payload.spike_data;
                }
            } else {
                // non-spike message: keep legacy behavior (ignore)
            }
        }
    }
}

}} // namespace SST::SnnDL
