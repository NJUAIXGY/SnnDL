// -*- c++ -*-
//
// NocSubsystem:
// - 统一收敛 NoC 输入侧（NIC 回调 / 外部端口 / mesh 方向链路）
// - 覆盖 send/recv/forward + 本地投递（跨 core ring）
//
// Phase4-A1.1：以“编排/适配层”形式落地，先通过回调复用 MultiCorePE 现有后端能力，
// 保持行为/统计口径不变，再逐步把后端实现迁出 MultiCorePE。

#pragma once

#include <cstdint>
#include <functional>
#include <queue>
#include <string>

#include "INocTransport.h"

namespace SST { class Event; }
namespace SST { class Link; }
namespace SST { class Output; }
namespace SST { namespace Statistics { template <typename T> class Statistic; } }

namespace SST { namespace SnnDL {

class OptimizedInternalRing;
class SnnInterface;
class SpikeEvent;
class SpikeEventWrapper;

class NocSubsystem final : public INocTransport {
public:
    struct Config {
        bool log_enable = false;
    };

    struct Runtime {
        SST::Output* log = nullptr;
        int node_id = 0;
        int num_cores = 1;

        // NoC backends（由 MultiCorePE 装配并保证生命周期）
        SnnInterface* nic = nullptr;
        OptimizedInternalRing* optimized_ring = nullptr;
        SST::Link* external_spike_output_link = nullptr;  // legacy fallback

        // neuron_id -> core_id；返回 -1 表示不在本 PE
        std::function<int(int /*neuron_id*/)> determine_target_unit;

        // 投递回调（由 MultiCorePE 提供实现，保持 core->deliverSpike 语义不变）
        std::function<void(int /*core_id*/, SpikeEvent*)> deliver_to_core;
    };

    struct Stats {
        SST::Statistics::Statistic<uint64_t>* external_spikes_received = nullptr;
        SST::Statistics::Statistic<uint64_t>* external_spikes_sent = nullptr;
        SST::Statistics::Statistic<uint64_t>* inter_core_messages = nullptr;
    };

    ~NocSubsystem();

    void configure(const Config& cfg);
    void bindRuntime(const Runtime& rt);
    void bindStats(const Stats& st);

    // === Output side ===
    void onCoreSend(SpikeEvent* event);

    // === INocTransport ===
    void sendFromCore(int src_core, SpikeEvent* event) override;
    void injectLocal(int dst_core, SpikeEvent* event) override;
    void sendExternal(SpikeEvent* event) override;

    // === Input side ===
    void onNicReceive(SpikeEvent* spike);
    void onExternalPortEvent(SST::Event* event);
    void onDirectionalLinkEvent(SST::Event* event, const std::string& direction);

    // === Scheduled by MultiCorePE ===
    void drainIncomingQueue(uint64_t current_cycle);
    void tickRing(uint64_t current_cycle);

    size_t incomingQueueSize() const { return incoming_queue_.size(); }

private:
    SpikeEvent* extractSpikeFromWrapper_(SpikeEventWrapper* wrapper);
    void enqueueIncoming_(SpikeEvent* spike);
    void sendExternalSpike_(SpikeEvent* spike);
    void forwardExternalSpike_(SpikeEvent* spike);
    void routeInternalSpike_(int src_core, int dst_core, SpikeEvent* spike);
    void tickOptimizedRing_(uint64_t current_cycle);

    Config cfg_{};
    Runtime rt_{};
    Stats st_{};

    std::queue<SpikeEvent*> incoming_queue_;
};

}} // namespace SST::SnnDL
