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

namespace SST { class Event; }
namespace SST { class Output; }
namespace SST { namespace Statistics { template <typename T> class Statistic; } }

namespace SST { namespace SnnDL {

class SpikeEvent;
class SpikeEventWrapper;

class NocSubsystem final {
public:
    struct Config {
        bool log_enable = false;
    };

    struct Runtime {
        SST::Output* log = nullptr;
        int node_id = 0;
        int num_cores = 1;

        // neuron_id -> core_id；返回 -1 表示不在本 PE
        std::function<int(int /*neuron_id*/)> determine_target_unit;

        // 投递/路由/外发回调（由 MultiCorePE 提供实现）
        std::function<void(int /*core_id*/, SpikeEvent*)> deliver_to_core;
        std::function<void(int /*src_core*/, int /*dst_core*/, SpikeEvent*)> route_internal;
        // 本 PE 产生的远端发送（沿用 sendExternalSpike 语义：更新 external_spikes_sent 等统计）
        std::function<void(SpikeEvent*)> send_external;
        // 中继转发（沿用 clockTick 中继语义：仅调用 external_nic_->sendSpike，不更新 external_spikes_sent）
        std::function<void(SpikeEvent*)> forward_external;

        // ring tick/receive（Phase4-A1.1 可直接复用 MultiCorePE 现有实现）
        std::function<void(uint64_t /*current_cycle*/)> tick_ring;
    };

    struct Stats {
        SST::Statistics::Statistic<uint64_t>* external_spikes_received = nullptr;
    };

    ~NocSubsystem();

    void configure(const Config& cfg);
    void bindRuntime(const Runtime& rt);
    void bindStats(const Stats& st);

    // === Output side ===
    void onCoreSend(SpikeEvent* event);

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

    Config cfg_{};
    Runtime rt_{};
    Stats st_{};

    std::queue<SpikeEvent*> incoming_queue_;
};

}} // namespace SST::SnnDL

