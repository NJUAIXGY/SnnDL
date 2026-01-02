// -*- c++ -*-
//
// ILegacySnnWorkloadHost: Phase4-Task5 过渡接口
// - 用于将 CoreShell 的入口委托切换到 workload，同时保持旧 SNN 逻辑可复用/可回归。
// - 后续 Phase4-C/Task6+ 将逐步把这些 legacy 入口的实现下沉到 services/workload/snn。
//

#pragma once

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace SST { namespace SnnDL {

class ISnnComputeCore;
class IWeightReader;
class SpikeEvent;

class ILegacySnnWorkloadHost {
public:
    virtual ~ILegacySnnWorkloadHost() = default;

    // Returns true if this cycle performed any work (best-effort); caller may ignore.
    virtual bool legacySnnOnClockTick(uint64_t now_cycle) = 0;

    // Phase4-Task6.2: drive weight/memory-related per-tick work from workload=snn.
    // Default no-op preserves compatibility for non-SNN workloads/hosts.
    virtual void legacySnnOnWeightsTick(uint64_t /*now_cycle*/) {}

    // Takes ownership of spike event.
    virtual void legacySnnDeliverSpike(SpikeEvent* spike) = 0;

    // Phase4-Task6.3: accounting/统计仍保留在 CoreShell，由 workload 在发送前回调该入口。
    virtual void legacySnnOnNeuronFires(const std::vector<uint32_t>& neuron_indices, uint64_t now_cycle) = 0;

    // Phase4-Task6.4: GAS/window scatter 由 workload=snn 执行；CoreShell 仅汇聚统计与对 PE 聚合上报。
    virtual void legacySnnOnGasScatterSpikesEmitted(uint32_t seq, uint64_t spikes_emitted) = 0;

    // Bind compute core instance owned by workload (non-owning view stored by host).
    virtual void legacySnnBindComputeCore(ISnnComputeCore* core) = 0;

    // Provide weight reader adapter (owned by host until Task6.2 moves it into workload).
    virtual IWeightReader* legacySnnGetWeightReader() = 0;

    // Phase4-Task6.2-Step2: transfer ownership of the weight reader/subsystem into workload=snn.
    // Called exactly once by SnnWorkload during cutover; host should return nullptr on subsequent calls.
    virtual std::unique_ptr<IWeightReader> legacySnnTakeWeightReader() = 0;

    // Optional learning writeback hook (called by compute core).
    virtual bool legacySnnWriteback(const std::unordered_map<uint64_t, float>& grads,
                                    float learning_rate,
                                    float weight_decay) = 0;

    // Metrics/reporting (Phase4-Task5-Step2 transitional)
    virtual bool legacySnnHasWork() const = 0;
    virtual double legacySnnGetUtilization() const = 0;
    virtual void legacySnnGetStatistics(std::map<std::string, uint64_t>& stats) const = 0;
};

}} // namespace SST::SnnDL
