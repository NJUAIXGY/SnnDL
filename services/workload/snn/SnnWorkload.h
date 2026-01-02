// -*- c++ -*-
//
// SnnWorkload: Phase4 的 SNN workload 插件壳（实现 ISpikeWorkload）
// - 目标：把现有默认 SNN 主链路逐步从 control/CoreShell 下沉到该 workload。
// - 现阶段（Task3）：仅提供可编译的“壳”，暂不切换默认执行入口。
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <queue>
#include <string>
#include <vector>

#include "ILegacySnnWorkloadHost.h"
#include "IGasStageSink.h"
#include "ISpikeWorkload.h"
#include "ISnnSpikeCommWorkload.h"

namespace SST { class Output; class Params; }

namespace SST { namespace SnnDL {

class SpikeEvent;
class NocPacketEvent;
class ISnnComputeCore;
class IWeightReader;
class WeightMemorySubsystem;
class AccumulatorOps;
class SynapseRouteSubsystem;
class SpikeCommSubsystem;
class ParentSpikeTransport;
class NocSpikeTransport;
struct SynapseRouteBuildConfig;

class SnnWorkload final : public ISpikeWorkload, public ISnnSpikeCommWorkload, public IGasStageSink {
public:
    SnnWorkload();
    ~SnnWorkload() override;

    void configureFromParams(const SST::Params& params) override;
    void bindRuntime(const Runtime& rt) override;
    void bindLegacyHost(ILegacySnnWorkloadHost* host) override { legacy_host_ = host; }

    bool onClockTick(uint64_t now_cycle) override;
    bool deliverPacket(NocPacketEvent* packet) override;
    void deliverSpike(SpikeEvent* spike) override;
    bool hasWork() const override;
    double getUtilization() const override;
    void getStatistics(std::map<std::string, uint64_t>& stats) const override;
    void onInitPhase(unsigned phase) override;
    void onSetup() override;
    void onFinish() override;

    // ISnnSpikeCommWorkload
    void applyGatingDecision(uint32_t src_global,
                             const std::vector<uint32_t>& dest_pes,
                             uint64_t current_cycle,
                             uint64_t ttl_cycles) override;
    void emitNeuronFire(uint32_t neuron_idx, uint64_t now_cycle) override;
    uint64_t emitNeuronFireBatch(const std::vector<uint32_t>& neuron_indices, uint64_t now_cycle) override;
    bool ready() const override;

    // IGasStageSink (Phase4-Task6.4): workload=snn 接管 GAS/window 阶段机业务动作
    void onGasStageEvent(const GasStageEvent& ev) override;
    void onGasStatEvent(const GasStatEvent& st) override;

private:
    enum class GasStage { Idle=0, Gather=1, Apply=2, Scatter=3 };

    uint64_t nowNs_() const;
    bool isWindowWorkload_() const { return apply_acc_enable_ && gas_window_mode_; }
    bool processReadySpikes_(uint64_t now_ns);
    void processLocalSpike_(SpikeEvent* spike);

    bool isPreLocal_(uint32_t pre_global) const;
    uint32_t remapPreGlobalModulo_(uint32_t pre_global) const;
    uint32_t mapPreGlobalToLocal_(uint32_t pre_global) const;

    void ensureWeightReaderOwned_();
    void ensureComputeCoreConfigured_();
    void ensureSpikeCommConfigured_();
    bool windowScatterModeActive_() const;

    Runtime rt_{};
    ILegacySnnWorkloadHost* legacy_host_ = nullptr; // non-owning (Phase4-Task5 transitional)

    // Phase4-Task6.2-Step2: weight reader/subsystem ownership moved from CoreShell into workload=snn.
    std::unique_ptr<IWeightReader> weight_reader_;
    WeightMemorySubsystem* weight_mem_subsystem_ = nullptr; // non-owning view into weight_reader_ (if it is WMS)

    // Compute core moved from CoreShell (Phase4 Task6.1). Still bridged via legacy host for now.
    std::unique_ptr<ISnnComputeCore> compute_core_;
    std::string compute_core_impl_ = "default";
    uint32_t num_neurons_ = 0;
    uint64_t global_neuron_base_ = 0;
    uint32_t neurons_per_pe_cfg_ = 0;
    uint32_t total_nodes_cfg_ = 16;
    bool apply_acc_enable_ = false;
    bool gas_window_mode_ = false;
    bool window_read_enable_ = false;
    bool window_read_debug_ = false;
    bool scheme1_enable_ = false;
    bool use_post_row_pre_col_ = false;
    bool workload_spike_input_enable_ = false;
    uint64_t now_cycle_cached_ = 0;
    GasStage gas_stage_ = GasStage::Idle;

    // Spike input handling migrated from CoreShell (Phase7 - Task1).
    std::queue<SpikeEvent*> incoming_spikes_;
    bool enable_weight_fetch_ = false;
    bool record_edge_apply_enable_ = false;
    bool record_edge_idle_enable_ = false;
    bool record_edge_scatter_enable_ = false;
    size_t edge_collector_max_capacity_ = 1'000'000;

    // Phase4-Task6.4: window accumulator moved into workload (acc_update callback rebound in WMS).
    std::unique_ptr<AccumulatorOps> acc_ops_;
    uint64_t last_scatter_spikes_emitted_ = 0;

    // Keep a local copy of params for compute_core_->configure().
    std::unique_ptr<SST::Params> params_;
    bool compute_configured_ = false;

    // Phase4-Task6.3: route/comm owned by workload=snn.
    std::unique_ptr<SynapseRouteSubsystem> synapse_route_;
    std::unique_ptr<SpikeCommSubsystem> spike_comm_;
    std::unique_ptr<ParentSpikeTransport> parent_spike_transport_;
    std::unique_ptr<NocSpikeTransport> noc_spike_transport_;
    bool spike_comm_configured_ = false;
};

}} // namespace SST::SnnDL
