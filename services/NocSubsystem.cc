// -*- c++ -*-
//
// NocSubsystem implementation
//

#include "NocSubsystem.h"

#include <cinttypes>

#include <sst/core/event.h>
#include <sst/core/output.h>
#include <sst/core/statapi/stataccumulator.h>

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

void NocSubsystem::onCoreSend(SpikeEvent* event) {
    if (!event) return;

    const int target_unit =
        rt_.determine_target_unit ? rt_.determine_target_unit(static_cast<int>(event->getDestinationNeuron())) : -1;

    if (target_unit >= 0 && target_unit < rt_.num_cores) {
        const int src_core =
            rt_.determine_target_unit ? rt_.determine_target_unit(static_cast<int>(event->getSourceNeuron())) : -1;

        if (src_core >= 0 && src_core < rt_.num_cores && rt_.route_internal) {
            rt_.route_internal(src_core, target_unit, event);
            return;
        }
        if (rt_.deliver_to_core) {
            rt_.deliver_to_core(target_unit, event);
            return;
        }
        delete event;
        return;
    }

    if (rt_.send_external) {
        rt_.send_external(event);
        return;
    }
    delete event;
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
    if (rt_.send_external) {
        rt_.send_external(spike);
        return;
    }

    delete spike;
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
        if (rt_.forward_external) {
            rt_.forward_external(spike);
        } else {
            delete spike;
        }
    }
}

void NocSubsystem::tickRing(uint64_t current_cycle) {
    if (rt_.tick_ring) rt_.tick_ring(current_cycle);
}

}} // namespace SST::SnnDL

