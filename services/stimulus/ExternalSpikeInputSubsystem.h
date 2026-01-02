// -*- c++ -*-
//
// ExternalSpikeInputSubsystem:
// - 处理来自 external_spike_input 端口的 SpikeEvent* 直接注入；
// - 语义冻结：仅本地投递（不做 relay/forward），非本节点目标直接丢弃。
//

#pragma once

#include <cstdint>
#include <functional>

namespace SST { class Output; }

namespace SST { namespace SnnDL {

class SpikeEvent;

class ExternalSpikeInputSubsystem final {
public:
    struct Runtime {
        SST::Output* log = nullptr;
        int node_id = 0;
        uint64_t global_neuron_base = 0;
        int num_cores = 1;
        int neurons_per_core = 1;
        int total_neurons = 1;

        // 语义：接管 spike 生命周期；deliver_to_core 也必须接管生命周期。
        std::function<void(int /*core_id*/, SpikeEvent*)> deliver_to_core;
    };

    void bindRuntime(const Runtime& rt) { rt_ = rt; }

    // 语义：接管 spike 生命周期；若无法本地投递将直接 delete。
    void onSpike(SpikeEvent* spike);

private:
    int determineTargetUnit_(uint32_t global_neuron_id) const;

    Runtime rt_{};
};

}} // namespace SST::SnnDL

