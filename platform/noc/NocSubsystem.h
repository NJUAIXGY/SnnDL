// -*- c++ -*-
//
// NocSubsystem:
// - 统一收敛 NoC 输入侧（NIC 回调 / 外部端口 / mesh 方向链路）
// - 覆盖 send/recv/forward + 本地投递（跨 core ring）
//
// Packet orchestration layer: callbacks keep endpoint delivery and network
// transport independent from the SST component that owns the subsystem.

#pragma once

#include <cstdint>
#include <deque>
#include <functional>
#include <queue>
#include <string>

#include "INocTransport.h"
#include "NocPacketEvent.h"

namespace SST { class Event; }
namespace SST { class Link; }
namespace SST { class Output; }
namespace SST { namespace Statistics { template <typename T> class Statistic; } }

namespace SST { namespace SnnDL {

class OptimizedInternalRing;
class SnnInterface;

class NocSubsystem final : public INocTransport {
public:
    struct Config {
        bool log_enable = false;
    };

    struct Runtime {
        SST::Output* log = nullptr;
        int node_id = 0;
        int num_cores = 1;

        // NoC backends are supplied by the owning component and outlive this object.
        SnnInterface* nic = nullptr;
        OptimizedInternalRing* optimized_ring = nullptr;
        SST::Link* external_spike_output_link = nullptr;  // legacy fallback

        // Optional: step tagging for step-limited experiments (naive_raw "no within-step cascade").
        // When active_step_seq != nullptr and *active_step_seq != 0, and packet.step_seq==0:
        //   packet.step_seq = *active_step_seq + step_seq_offset
        const uint32_t* active_step_seq = nullptr;  // 非拥有
        uint32_t step_seq_offset = 0;

        // Delivery callback: NoC only forwards a packet to its endpoint.
        std::function<void(int /*endpoint_id*/, NocPacketEvent*)> deliver_to_endpoint;
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
    void onCoreSend(NocPacketEvent* packet);

    // === INocTransport ===
    void sendFromCore(int src_core, NocPacketEvent* packet) override;
    void injectLocal(int dst_core, NocPacketEvent* packet) override;
    void sendExternal(NocPacketEvent* packet) override;

    // === Input side ===
    void onNicReceiveEvent(SST::Event* event);
    void onNicReceive(NocPacketEvent* packet);
    void onExternalPortEvent(SST::Event* event);
    void onDirectionalLinkEvent(SST::Event* event, const std::string& direction);

    // === Scheduled by the owning component ===
    void drainIncomingQueue(uint64_t current_cycle);
    void tickRing(uint64_t current_cycle);

    size_t incomingQueueSize() const { return incoming_queue_.size(); }
    bool isIdle() const;
    size_t nicPendingSendCount() const;
    int ringPendingMessageCount() const;

private:
    struct PendingRingInjection {
        int src_core = -1;
        int dst_core = -1;
        NocPacketEvent* packet = nullptr;
    };

    void enqueueIncoming_(NocPacketEvent* packet);
    void sendExternalPacket_(NocPacketEvent* packet);
    void forwardExternalPacket_(NocPacketEvent* packet);
    void routeInternalPacket_(int src_core, int dst_core, NocPacketEvent* packet);
    void retryPendingRingInjections_();
    void tickOptimizedRing_(uint64_t current_cycle);

    Config cfg_{};
    Runtime rt_{};
    Stats st_{};

    std::queue<NocPacketEvent*> incoming_queue_;
    // Owns packets until the ring accepts them.  Source VC pressure is
    // backpressure, never a packet-loss condition.
    std::deque<PendingRingInjection> pending_ring_injections_;
};

}} // namespace SST::SnnDL
