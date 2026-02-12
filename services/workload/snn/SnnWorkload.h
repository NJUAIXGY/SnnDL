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
#include <unordered_map>
#include <vector>

#include "IWeightReaderAdopter.h"
#include "IGasStageSink.h"
#include "ISpikeWorkload.h"
#include "ISnnSpikeCommWorkload.h"

namespace SST { namespace Shared { template <typename T> class SharedArray; } }

namespace SST { class Output; class Params; }

namespace SST { namespace SnnDL {

class SpikeEvent;
class NocPacketEvent;
class ISnnComputeCore;
class IWeightReader;
class WeightMemorySubsystem;
class WeightCacheOps;
class BcsrWeightManager;
struct WeightAccessor;
class AccumulatorOps;
class SynapseRouteSubsystem;
class SpikeCommSubsystem;
class NocSpikeTransport;
	struct SynapseRouteBuildConfig;

	class SnnWorkload final : public ISpikeWorkload,
	                          public ISnnSpikeCommWorkload,
	                          public IGasStageSink,
	                          public IWeightReaderAdopter {
	public:
	    SnnWorkload();
	    ~SnnWorkload() override;

    void configureFromParams(const SST::Params& params) override;
    void bindRuntime(const Runtime& rt) override;

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

	    // IWeightReaderAdopter (Tier2-E): adopt CoreShell-provisioned weight reader to avoid duplicate assembly.
	    void adoptWeightReader(std::unique_ptr<IWeightReader> reader) override;

	private:
	    enum class GasStage { Idle=0, Gather=1, Apply=2, Scatter=3 };

    uint64_t nowNs_() const;
    void normalizeLayout_();
    bool isWindowWorkload_() const { return apply_acc_enable_ && gas_window_mode_; }
    bool processReadySpikes_(uint64_t now_ns);
    void processLocalSpike_(SpikeEvent* spike);

    bool isPreLocal_(uint32_t pre_global) const;
    uint32_t remapPreGlobalModulo_(uint32_t pre_global) const;
    uint32_t mapPreGlobalToLocal_(uint32_t pre_global) const;

    bool ensureLoaderReady_();
    void ensureWeightReaderOwned_();
    void ensureComputeCoreConfigured_();
    void ensureSpikeCommConfigured_();
    bool windowScatterModeActive_() const;

    Runtime rt_{};

    // Phase4-Task6.2-Step2: weight reader/subsystem ownership moved from CoreShell into workload=snn.
    std::unique_ptr<IWeightReader> weight_reader_;
    WeightMemorySubsystem* weight_mem_subsystem_ = nullptr; // non-owning view into weight_reader_ (if it is WMS)
    std::unique_ptr<WeightCacheOps> weight_cache_ops_;
    std::unique_ptr<WeightAccessor> weight_accessor_;
    std::unique_ptr<BcsrWeightManager> bcsr_mgr_;

    // Compute core owned by workload=snn.
    std::unique_ptr<ISnnComputeCore> compute_core_;
    std::string compute_core_impl_ = "default";
    // NOTE: num_neurons_ / global_neuron_base_ are normalized to "per-core" semantics in bindRuntime().
    // - num_neurons_        : neurons_per_core
    // - global_neuron_base_ : core-local global base (node_base + core_id * neurons_per_core)
    uint32_t num_neurons_ = 0;
    uint64_t global_neuron_base_ = 0;
    // Raw (as provided by scripts/configs); may be per-core or per-PE depending on config style.
    uint32_t num_neurons_param_ = 0;
    uint32_t neurons_per_pe_param_ = 0;
    uint64_t global_neuron_base_param_ = 0;
    uint64_t node_neuron_base_ = 0;
    bool layout_normalized_ = false;
    uint32_t neurons_per_pe_cfg_ = 0;
    uint32_t cores_per_pe_cfg_ = 1;
    uint32_t neurons_per_core_cfg_ = 0;
    uint32_t total_nodes_cfg_ = 16;
    bool apply_acc_enable_ = false;
    bool gas_window_mode_ = false;
    bool window_read_enable_ = false;
    bool window_read_debug_ = false;
    bool scheme1_enable_ = false;
    bool use_post_row_pre_col_ = false;
    bool use_bcsr_ = false;
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
    uint64_t total_scatter_spikes_emitted_ = 0;

    // Step-gate mode: end Gather/Scatter explicitly (load-driven), instead of fixed window_cycles_*.
    uint32_t gather_seq_ = 0;
    uint64_t gather_begin_cycle_ = 0;
    uint64_t gather_last_activity_cycle_ = 0;
    uint32_t gather_quiesce_cycles_ = 32; // end-gather after N quiet cycles since last spike/edge activity
    uint32_t gather_min_cycles_ = 1;      // avoid ending gather in the same cycle as BeginGather
    bool gather_end_requested_ = false;
    bool scatter_end_requested_ = false;

    // WeightLoader barrier (shared signal)
    std::string loader_done_key_;
    bool wait_for_loader_done_ = false;
    bool loader_ready_latched_ = false;
    bool loader_ready_logged_ = false;
    std::unique_ptr<SST::Shared::SharedArray<int>> loader_done_shared_;

    // Keep a local copy of params for compute_core_->configure().
    std::unique_ptr<SST::Params> params_;
    bool compute_configured_ = false;

    // Phase4-Task6.3: route/comm owned by workload=snn.
    std::unique_ptr<SynapseRouteSubsystem> synapse_route_;
    std::unique_ptr<SpikeCommSubsystem> spike_comm_;
    std::unique_ptr<NocSpikeTransport> noc_spike_transport_;
    bool spike_comm_configured_ = false;

    // Native multicast receive-side expansion cache: pre_global -> posts_local (for this core only).
    // Cached against the current shared routes table pointer to avoid stale reuse across reconfigure.
    std::shared_ptr<const std::unordered_map<uint32_t, std::vector<uint32_t>>> routes_shared_for_posts_cache_;
    std::unordered_map<uint32_t, std::vector<uint32_t>> pre_to_posts_local_;
};

}} // namespace SST::SnnDL
