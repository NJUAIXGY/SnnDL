// -*- c++ -*-

#include <sst/core/sst_config.h>

#include "workload/snn/SnnWorkload.h"
#include "ISnnComputeCore.h"
#include "SpikeEvent.h"
#include "SnnWeightReader.h"
#include "synapse/gas/AccumulatorOps.h"
#include "synapse/weights/WeightMemorySubsystem.h"
#include "synapse/route/SynapseRouteSubsystem.h"
#include "synapse/route/SpikeCommSubsystem.h"
#include "SynapseRouteBuildConfig.h"
#include "ISpikeTransport.h"
#include "NocSpikeTransport.h"

#include <sst/core/output.h>
#include <sst/core/params.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <utility>

namespace SST { namespace SnnDL {

namespace {

[[noreturn]] void fatal_missing_legacy_host_(SST::Output* log, const char* what) {
    if (log) {
        log->fatal(CALL_INFO, -1,
                   "SnnWorkload requires legacy_host_ during Phase4-Task5 cutover (%s)\n",
                   what ? what : "missing");
    }
    // Fallback: if log is null, aborting is still preferable to silent corruption.
    std::abort();
}

} // namespace

SnnWorkload::SnnWorkload() = default;

SnnWorkload::~SnnWorkload() {
    while (!incoming_spikes_.empty()) {
        delete incoming_spikes_.front();
        incoming_spikes_.pop();
    }
}

void SnnWorkload::configureFromParams(const SST::Params& params) {
    // Phase4 Task6.1: cache params for compute core creation/config.
    params_ = std::make_unique<SST::Params>(/*copy*/params);
    compute_core_impl_ = params.find<std::string>("compute_core_impl", "default");
    num_neurons_ = params.find<uint32_t>("num_neurons", 64);
    global_neuron_base_ = params.find<uint64_t>("global_neuron_base", 0);
    total_nodes_cfg_ = params.find<uint32_t>("total_nodes", 16);
    apply_acc_enable_ = params.find<int>("apply_acc_enable", 0) != 0;
    gas_window_mode_ = params.find<int>("gas_window_mode", 0) != 0;
    window_read_enable_ = params.find<int>("window_read_enable", 0) != 0;
    window_read_debug_ = params.find<int>("window_read_debug", 0) != 0;
    scheme1_enable_ = params.find<int>("scheme1_enable", 0) != 0;
    workload_spike_input_enable_ = params.find<int>("workload_spike_input_enable", 0) != 0;
    const std::string index_mode_str = params.find<std::string>("index_mode", "pre_row_post_col");
    use_post_row_pre_col_ =
        (index_mode_str == "post_row_pre_col") ||
        (index_mode_str == "bcsr_post_row") ||
        (index_mode_str == "csr_post_row");

    enable_weight_fetch_ = params.find<int>("enable_weight_fetch", 0) != 0;
    edge_collector_max_capacity_ = static_cast<size_t>(params.find<uint64_t>("edge_collector_max_capacity", 1'000'000));
    const int record_apply_default = (gas_window_mode_ && apply_acc_enable_) ? 1 : 0;
    record_edge_apply_enable_ = params.find<int>("record_edge_apply_enable", record_apply_default) != 0;
    record_edge_idle_enable_ = params.find<int>("record_edge_idle_enable", 0) != 0;
    record_edge_scatter_enable_ = params.find<int>("record_edge_scatter_enable", 0) != 0;

    const uint32_t cores_per_pe = params.find<uint32_t>("total_cores", 8);
    const uint32_t computed_neurons_per_pe = cores_per_pe * num_neurons_;
    const uint32_t np_from_params = params.find<uint32_t>("neurons_per_pe", 0);
    neurons_per_pe_cfg_ = (np_from_params > 0) ? np_from_params : computed_neurons_per_pe;
}

void SnnWorkload::bindRuntime(const Runtime& rt) {
    rt_ = rt;
    if (weight_mem_subsystem_) {
        weight_mem_subsystem_->bindMemory(rt_.mem);
    }
    // 若 route/comm 已装配，则更新 runtime 依赖（log/stat/transport 指针）
    if (synapse_route_) {
        synapse_route_->bindRuntime(rt_.log,
                                    rt_.node_id,
                                    rt_.core_id,
                                    num_neurons_,
                                    neurons_per_pe_cfg_,
                                    rt_.sinks.stat_routes_entries_total);
        synapse_route_->bindFanoutStat(rt_.sinks.stat_fanout_per_spike_total);
    }
    if (spike_comm_) {
        ISpikeTransport* transport = nullptr;
        if (rt_.noc) {
            if (!noc_spike_transport_) noc_spike_transport_ = std::make_unique<NocSpikeTransport>();
            noc_spike_transport_->setNocTransport(rt_.noc);
            noc_spike_transport_->setSourceCore(static_cast<int>(rt_.core_id));
            const uint32_t cores_per_pe = params_ ? params_->find<uint32_t>("total_cores", 1) : 1;
            noc_spike_transport_->configureLayout(total_nodes_cfg_, cores_per_pe, num_neurons_);
            transport = noc_spike_transport_.get();
        } else {
            if (!parent_spike_transport_) parent_spike_transport_ = std::make_unique<ParentSpikeTransport>(rt_.parent_iface);
            else parent_spike_transport_->setParent(rt_.parent_iface);
            transport = parent_spike_transport_.get();
        }
        SpikeCommRuntimeConfig crt{};
        crt.log = rt_.log;
        crt.transport = transport;
        crt.synapse_route = synapse_route_.get();
        crt.global_neuron_base = global_neuron_base_;
        spike_comm_->bindRuntime(crt);
    }
}

uint64_t SnnWorkload::nowNs_() const {
    if (rt_.time.now_ns) return rt_.time.now_ns(rt_.time.ctx);
    // Fallback: best-effort (legacy behavior assumes 1GHz => 1 cycle == 1ns anyway).
    return now_cycle_cached_;
}

bool SnnWorkload::isPreLocal_(uint32_t pre_global) const {
    return (pre_global >= global_neuron_base_) &&
           (pre_global < global_neuron_base_ + static_cast<uint64_t>(num_neurons_));
}

uint32_t SnnWorkload::remapPreGlobalModulo_(uint32_t pre_global) const {
    if (num_neurons_ == 0) return 0;
    const uint64_t width = static_cast<uint64_t>(num_neurons_);
    const uint64_t base =
        static_cast<uint64_t>(global_neuron_base_) - static_cast<uint64_t>(rt_.core_id) * width;
    const uint64_t diff = static_cast<uint64_t>(pre_global) - base;
    return static_cast<uint32_t>(diff % width);
}

uint32_t SnnWorkload::mapPreGlobalToLocal_(uint32_t pre_global) const {
    if (isPreLocal_(pre_global)) {
        return static_cast<uint32_t>(static_cast<uint64_t>(pre_global) - global_neuron_base_);
    }
    return remapPreGlobalModulo_(pre_global);
}

void SnnWorkload::ensureWeightReaderOwned_() {
    if (weight_reader_) return;
    if (!legacy_host_) {
        fatal_missing_legacy_host_(rt_.log, "bindLegacyHost missing (take weight reader)");
    }

    weight_reader_ = legacy_host_->legacySnnTakeWeightReader();
    if (!weight_reader_) {
        if (rt_.log) {
            rt_.log->fatal(CALL_INFO, -1,
                           "SnnWorkload failed to take weight reader from legacy host (legacySnnTakeWeightReader returned nullptr)\n");
        }
        std::abort();
    }

    weight_mem_subsystem_ = dynamic_cast<WeightMemorySubsystem*>(weight_reader_.get());
    if (weight_mem_subsystem_) {
        weight_mem_subsystem_->bindMemory(rt_.mem);
        // Phase4-Task6.4: window accumulator moved into workload=snn; rebind WMS acc_update callback.
        if (apply_acc_enable_ && gas_window_mode_) {
            if (!acc_ops_) {
                const bool dense_enable = params_ ? (params_->find<int>("apply_dense_acc_enable", 1) != 0) : true;
                const bool shadow_verify_enable =
                    dense_enable && (params_ ? (params_->find<int>("acc_shadow_verify_enable", 0) != 0) : false);
                AccumulatorOpsConfig acc_cfg{};
                acc_cfg.num_neurons = num_neurons_;
                acc_cfg.dense_enable = dense_enable;
                acc_cfg.spill_enable = params_ ? (params_->find<int>("acc_spill_enable", 1) != 0) : true;
                acc_cfg.high_watermark_bytes =
                    params_ ? params_->find<uint64_t>("acc_high_watermark_bytes", 16 * 1024 * 1024)
                            : (16 * 1024 * 1024);
                acc_cfg.shadow_verify_enable = shadow_verify_enable;
                acc_cfg.window_read_debug = window_read_debug_;
                acc_cfg.core_id = static_cast<int>(rt_.core_id);
                acc_cfg.verbose = rt_.log ? rt_.log->getVerboseLevel() : 0;
                acc_cfg.out = rt_.log;
                acc_ops_ = std::make_unique<AccumulatorOps>(acc_cfg);
            }
            weight_mem_subsystem_->overrideAccUpdate([this](uint32_t post_local, float dv) {
                if (acc_ops_) acc_ops_->update(post_local, dv);
            });
        }
    }
}

void SnnWorkload::ensureComputeCoreConfigured_() {
    if (compute_configured_) return;
    if (!legacy_host_) {
        fatal_missing_legacy_host_(rt_.log, "bindLegacyHost missing (compute configure)");
    }
    if (!params_) {
        if (rt_.log) {
            rt_.log->fatal(CALL_INFO, -1,
                           "SnnWorkload missing cached Params during compute configure (configureFromParams not called?)\n");
        }
        std::abort();
    }

    if (!compute_core_) {
        compute_core_ = createComputeCoreByName(compute_core_impl_);
        if (!compute_core_) {
            if (rt_.log) {
                rt_.log->verbose(CALL_INFO, 0, 0,
                                 "⚠️ 未知 compute_core_impl='%s'，回退到 default\n",
                                 compute_core_impl_.c_str());
            }
            compute_core_ = createComputeCoreByName("default");
        }
        if (!compute_core_) {
            if (rt_.log) rt_.log->fatal(CALL_INFO, -1, "createComputeCoreByName failed for both impl and default\n");
            std::abort();
        }
        // Expose compute core view to legacy host (non-owning) so existing control-plane code keeps working.
        legacy_host_->legacySnnBindComputeCore(compute_core_.get());
    }

    ComputeCoreContext ctx;
    ctx.core_id = rt_.core_id;
    ctx.node_id = rt_.node_id;
    ctx.num_neurons = num_neurons_;
    ctx.global_neuron_base = global_neuron_base_;
    ctx.neurons_per_pe_cfg = neurons_per_pe_cfg_;
    ctx.log = rt_.log;
    ensureWeightReaderOwned_();
    ctx.weight_reader = weight_reader_.get();
    ctx.writeback_fn = [this](const std::unordered_map<uint64_t, float>& grads,
                              float lr,
                              float wd) -> bool {
        return legacy_host_ ? legacy_host_->legacySnnWriteback(grads, lr, wd) : false;
    };
    compute_core_->configure(ctx, *params_);
    compute_configured_ = true;
}

bool SnnWorkload::windowScatterModeActive_() const {
    // 在这些模式下，发放/发送闭环由 legacy control-plane 主导，workload 不再做非 window 的 drain+send：
    // - GAS window scatter 模式（apply_acc_enable + gas_window_mode）
    // - scheme1（slice/superstep 驱动的 Scatter 阶段）
    const bool scheme1_enable = params_ ? (params_->find<int>("scheme1_enable", 0) != 0) : false;
    return (apply_acc_enable_ && gas_window_mode_) || scheme1_enable;
}

void SnnWorkload::ensureSpikeCommConfigured_() {
    if (spike_comm_configured_) return;
    if (!params_) {
        if (rt_.log) rt_.log->fatal(CALL_INFO, -1, "SnnWorkload missing cached Params for route/comm configure\n");
        std::abort();
    }

    if (!synapse_route_) synapse_route_ = std::make_unique<SynapseRouteSubsystem>();
    if (!spike_comm_) spike_comm_ = std::make_unique<SpikeCommSubsystem>();

    // Build routing config from Params (keep consistent with legacy CoreShell config).
    const std::string routing_mode = params_->find<std::string>("routing_mode", "fixed");
    const std::string weights_template = params_->find<std::string>("weights_template", "");
    uint32_t weights_cols = params_->find<uint32_t>("weights_cols", 0);
    if (weights_cols == 0) weights_cols = num_neurons_;
    const std::string index_mode_str = params_->find<std::string>("index_mode", "pre_row_post_col");
    const bool use_post_row_pre_col =
        (index_mode_str == "post_row_pre_col") || (index_mode_str == "bcsr_post_row") || (index_mode_str == "csr_post_row");
    const bool use_bcsr = (index_mode_str == "bcsr_post_row");

    SpikeCommRoutingConfig cfg{};
    cfg.routing_weight_driven = (routing_mode == "weight_driven");
    cfg.log_weight_details = params_->find<int>("log_weight_details", 0) != 0;
    cfg.verify_routing_weights = params_->find<int>("verify_routing_weights", 0) != 0;
    cfg.route_summary_enable = params_->find<int>("route_summary_enable", 0) != 0;
    cfg.rows = num_neurons_;
    cfg.cols = weights_cols;
    cfg.total_nodes = total_nodes_cfg_;
    cfg.cores_per_pe = params_->find<uint32_t>("total_cores", 1);
    cfg.neurons_per_pe = neurons_per_pe_cfg_;
    cfg.use_post_row_pre_col = use_post_row_pre_col;
    cfg.weights_template = weights_template;
    cfg.routing_epsilon = params_->find<float>("routing_epsilon", 1e-8f);
    cfg.routing_topk = params_->find<uint32_t>("routing_topk", 0);
    cfg.routing_topk_per_pe = params_->find<uint32_t>("routing_topk_per_pe", 0);
    cfg.route_exclude_self_pe = params_->find<int>("route_exclude_self_pe", 0) != 0;
    cfg.route_layers_mask = params_->find<std::string>("route_layers_mask", "");
    cfg.route_filter_warn = params_->find<int>("route_filter_warn", 1) != 0;
    cfg.mapping_mode = params_->find<std::string>("mapping_mode", "off");
    cfg.mapping_edges_file = params_->find<std::string>("mapping_edges_file", "");
    cfg.mapping_csv_has_header = params_->find<int>("mapping_csv_has_header", 1) != 0;
    cfg.mapping_csv_separator = params_->find<std::string>("mapping_csv_separator", ",");
    cfg.mapping_assume_block_ids = params_->find<int>("mapping_assume_block_ids", 1) != 0;

    cfg.use_bcsr = use_bcsr;
    cfg.base_addr = params_->find<uint64_t>("base_addr", 0);
    if (use_bcsr) {
        cfg.bcsr_br = params_->find<uint32_t>("bcsr_block_rows", 16);
        cfg.bcsr_bc = params_->find<uint32_t>("bcsr_block_cols", 16);
        cfg.bcsr_idx_bytes = params_->find<uint32_t>("bcsr_idx_bytes", 2);
        cfg.bcsr_val_bytes = params_->find<uint32_t>("bcsr_val_bytes", 4);
        const uint64_t rp_off = params_->find<uint64_t>("bcsr_rowptr_offset", 0);
        const uint64_t ci_off = params_->find<uint64_t>("bcsr_colidx_offset", 0);
        const uint64_t bd_off = params_->find<uint64_t>("bcsr_blockdata_offset", 0);
        const uint64_t id_off = params_->find<uint64_t>("bcsr_blockids_offset", 0);
        cfg.bcsr_rowptr_addr = cfg.base_addr + rp_off;
        cfg.bcsr_colidx_addr = cfg.base_addr + ci_off;
        cfg.bcsr_blockdata_addr = cfg.base_addr + bd_off;
        cfg.bcsr_blockids_addr = id_off ? (cfg.base_addr + id_off) : 0;
    }

    synapse_route_->configure(cfg);

    const std::string gating_mode = params_->find<std::string>("gating_mode", "off");
    const uint64_t gating_ttl_cycles = params_->find<uint64_t>("gating_ttl_cycles", 1000);
    const std::string gating_scope = params_->find<std::string>("gating_scope", "inputs");
    synapse_route_->configureGating(/*gating_event_mode=*/(gating_mode == "event"),
                                    /*gating_ttl_cycles=*/gating_ttl_cycles,
                                    /*gating_scope_inputs_only=*/(gating_scope != "all"));
    synapse_route_->bindRuntime(rt_.log,
                                rt_.node_id,
                                rt_.core_id,
                                num_neurons_,
                                neurons_per_pe_cfg_,
                                rt_.sinks.stat_routes_entries_total);
    synapse_route_->bindFanoutStat(rt_.sinks.stat_fanout_per_spike_total);

    // Choose transport backend: prefer NoC, fallback to parent interface.
    ISpikeTransport* transport = nullptr;
    if (rt_.noc) {
        if (!noc_spike_transport_) noc_spike_transport_ = std::make_unique<NocSpikeTransport>();
        noc_spike_transport_->setNocTransport(rt_.noc);
        noc_spike_transport_->setSourceCore(static_cast<int>(rt_.core_id));
        noc_spike_transport_->configureLayout(total_nodes_cfg_, cfg.cores_per_pe, num_neurons_);
        transport = noc_spike_transport_.get();
    } else {
        if (!parent_spike_transport_) parent_spike_transport_ = std::make_unique<ParentSpikeTransport>(rt_.parent_iface);
        else parent_spike_transport_->setParent(rt_.parent_iface);
        transport = parent_spike_transport_.get();
    }

    spike_comm_->configure();
    SpikeCommRuntimeConfig crt{};
    crt.log = rt_.log;
    crt.transport = transport;
    crt.synapse_route = synapse_route_.get();
    crt.global_neuron_base = global_neuron_base_;
    spike_comm_->bindRuntime(crt);
    spike_comm_->initRouting();

    spike_comm_configured_ = true;
}

bool SnnWorkload::onClockTick(uint64_t now_cycle) {
    now_cycle_cached_ = now_cycle;
    if (!legacy_host_) {
        fatal_missing_legacy_host_(rt_.log, "bindLegacyHost missing");
    }
    // Phase4-Task6.1：compute core 的 per-tick 驱动下沉到 workload=snn，
    // CoreShell 仅保留 non-owning view 以兼容 legacy 控制链路。
    ensureComputeCoreConfigured_();
    if (compute_core_) {
        compute_core_->onClockTick(now_cycle);
    }
    // Phase4-Task6.2-Step2：weights/memory 子系统所有权迁入 workload=snn，tick 由 workload 直接驱动。
    if (weight_mem_subsystem_) {
        weight_mem_subsystem_->onClockTick(now_cycle);
    }

    // Phase4-Task6.3: control-plane 仍由 legacy host 驱动（GAS/window/阶段事件等）。
    const bool did_legacy = legacy_host_->legacySnnOnClockTick(now_cycle);

    // Phase7-Task1: strict window-read spike queue processing moved into workload=snn.
    // NOTE: This path is currently opt-in via Params to avoid reintroducing non-determinism.
    bool did_spike = false;
    if (workload_spike_input_enable_ && isWindowWorkload_() && window_read_enable_ && !scheme1_enable_) {
        did_spike = processReadySpikes_(nowNs_());
    }

    // Phase4-Task6.3: 非 window 模式下的 “endCycle->drain->route/comm” 闭环迁入 workload。
    if (!windowScatterModeActive_() && compute_core_) {
        compute_core_->endCycle(now_cycle);
        std::vector<FireEvent> fired;
        compute_core_->drainOutputs(fired, /*clear=*/true);
        if (!fired.empty()) {
            std::vector<uint32_t> neuron_indices;
            neuron_indices.reserve(fired.size());
            for (const auto& ev : fired) neuron_indices.push_back(ev.neuron_idx);
            legacy_host_->legacySnnOnNeuronFires(neuron_indices, now_cycle);
            ensureSpikeCommConfigured_();
            (void)emitNeuronFireBatch(neuron_indices, now_cycle);
            return true;
        }
    }

    return did_legacy || did_spike;
}

bool SnnWorkload::deliverPacket(NocPacketEvent* /*packet*/) {
    // Task3: 未接管非 spike packet；保持由 CoreShell 处理。
    return false;
}

bool SnnWorkload::processReadySpikes_(uint64_t now_ns) {
    if (incoming_spikes_.empty()) return false;

    std::vector<SpikeEvent*> ready_spikes;
    ready_spikes.reserve(std::min<size_t>(incoming_spikes_.size(), 256));
    while (!incoming_spikes_.empty()) {
        SpikeEvent* spike = incoming_spikes_.front();
        if (!spike) {
            incoming_spikes_.pop();
            continue;
        }
        if (spike->getTimestamp() >= now_ns) break;
        incoming_spikes_.pop();
        ready_spikes.push_back(spike);
    }

    if (ready_spikes.size() > 1) {
        auto weightBits = [](const SpikeEvent* s) -> uint64_t {
            if (!s) return 0;
            uint64_t bits = 0;
            double w = s->getWeight();
            static_assert(sizeof(double) == sizeof(uint64_t), "double size unexpected");
            std::memcpy(&bits, &w, sizeof(uint64_t));
            return bits;
        };
        std::sort(ready_spikes.begin(), ready_spikes.end(),
                  [&](const SpikeEvent* a, const SpikeEvent* b) {
                      if (a == b) return false;
                      const uint64_t ta = a ? a->getTimestamp() : 0;
                      const uint64_t tb = b ? b->getTimestamp() : 0;
                      if (ta != tb) return ta < tb;
                      const uint32_t dna = a ? a->getDestinationNode() : 0;
                      const uint32_t dnb = b ? b->getDestinationNode() : 0;
                      if (dna != dnb) return dna < dnb;
                      const uint32_t da = a ? a->getDestinationNeuron() : 0;
                      const uint32_t db = b ? b->getDestinationNeuron() : 0;
                      if (da != db) return da < db;
                      const uint32_t sa = a ? a->getSourceNeuron() : 0;
                      const uint32_t sb = b ? b->getSourceNeuron() : 0;
                      if (sa != sb) return sa < sb;
                      const uint64_t wa = weightBits(a);
                      const uint64_t wb = weightBits(b);
                      return wa < wb;
                  });
    }

    for (auto* spike : ready_spikes) {
        processLocalSpike_(spike);
        delete spike;
    }

    return !ready_spikes.empty();
}

void SnnWorkload::processLocalSpike_(SpikeEvent* spike_event) {
    if (!spike_event) return;
    if (!isWindowWorkload_()) return;
    if (!weight_mem_subsystem_) return;

    // Map destination to local post index (preserve legacy semantics).
    const uint32_t dest = spike_event->getDestinationNeuron();
    uint32_t post_local = dest;
    bool post_local_valid = true;
    if (spike_event->hasCachedPostLocal()) {
        post_local = spike_event->getCachedPostLocal();
    } else if (dest >= num_neurons_) {
        const uint64_t g = static_cast<uint64_t>(dest);
        if (g >= global_neuron_base_ && g < global_neuron_base_ + static_cast<uint64_t>(num_neurons_)) {
            post_local = static_cast<uint32_t>(g - global_neuron_base_);
        } else {
            post_local_valid = false;
        }
    }
    if (!post_local_valid || post_local >= num_neurons_) return;

    // Compute core gating (kept consistent with legacy: reject before recording any edge/access).
    if (apply_acc_enable_ && compute_core_) {
        if (!compute_core_->shouldAcceptSynapticInput(post_local, now_cycle_cached_)) {
            return;
        }
    }

    // Strict window-read: record edge for BeginApply issueFromEdges() (no immediate dv application here).
    if (enable_weight_fetch_ && rt_.mem) {
        bool stage_ok = false;
        switch (gas_stage_) {
            case GasStage::Gather:
                stage_ok = true;
                break;
            case GasStage::Apply:
                stage_ok = record_edge_apply_enable_;
                break;
            case GasStage::Idle:
                stage_ok = record_edge_idle_enable_;
                break;
            case GasStage::Scatter:
                stage_ok = record_edge_scatter_enable_;
                break;
            default:
                stage_ok = false;
                break;
        }
        if (stage_ok) {
            const size_t curr_edges = weight_mem_subsystem_->edgesCurrSize();
            if (curr_edges < edge_collector_max_capacity_) {
                weight_mem_subsystem_->recordEdge(post_local, spike_event->getSourceNeuron());
            }
        }
    }

    // Record synaptic access counters (owned by CoreShell via runtime sinks).
    if (rt_.sinks.synaptic_accesses) (*rt_.sinks.synaptic_accesses)++;
    if (rt_.sinks.stat_synaptic_accesses_total) rt_.sinks.stat_synaptic_accesses_total->addData(1);
}

void SnnWorkload::deliverSpike(SpikeEvent* spike) {
    if (!spike) return;
    if (!legacy_host_) {
        fatal_missing_legacy_host_(rt_.log, "bindLegacyHost missing");
        delete spike;
        return;
    }

    // Phase7-Task1: migrate strict window-read spike input into workload=snn.
    // Keep legacy path for non-window modes and scheme1 to avoid behavior drift.
    const bool migrate =
        workload_spike_input_enable_ && isWindowWorkload_() && window_read_enable_ && !scheme1_enable_;
    if (!migrate) {
        legacy_host_->legacySnnDeliverSpike(spike);
        return;
    }

    ensureComputeCoreConfigured_();
    ensureWeightReaderOwned_();

    if (rt_.sinks.spikes_received) (*rt_.sinks.spikes_received)++;
    if (rt_.sinks.stat_spikes_received_total) rt_.sinks.stat_spikes_received_total->addData(1);

    spike->clearLocalCache();
    spike->setTimestamp(nowNs_());
    if (compute_core_) compute_core_->onSpikeDelivered(spike);

    // Cache post/pre local indices (used by strict window-read path).
    if (window_read_enable_) {
        const uint32_t dest = spike->getDestinationNeuron();
        uint32_t post_local = dest;
        bool post_local_valid = true;
        if (dest >= num_neurons_) {
            const uint64_t g = static_cast<uint64_t>(dest);
            if (g >= global_neuron_base_ && g < global_neuron_base_ + static_cast<uint64_t>(num_neurons_)) {
                post_local = static_cast<uint32_t>(g - global_neuron_base_);
            } else {
                post_local_valid = false;
            }
        }
        if (post_local_valid) {
            spike->setCachedPostLocal(post_local);
        }
        if (!use_post_row_pre_col_) {
            // For legacy index_mode=pre_row_post_col, cache pre_local mapping.
            // (Best-effort: strict window-read uses pre_global in edge keys.)
            spike->setCachedPreLocal(mapPreGlobalToLocal_(spike->getSourceNeuron()));
        }

        // Critical: update window touch sets immediately (before per-tick filtering), so BeginApply sees non-empty sets.
        if (weight_mem_subsystem_ && post_local_valid && post_local < num_neurons_) {
            weight_mem_subsystem_->noteWindowTouch(post_local, spike->getSourceNeuron(), num_neurons_);
        }
    }

    incoming_spikes_.push(spike);
}

bool SnnWorkload::hasWork() const {
    if (!legacy_host_) {
        fatal_missing_legacy_host_(rt_.log, "bindLegacyHost missing (hasWork)");
    }
    if (!incoming_spikes_.empty()) return true;
    if (compute_core_ && compute_core_->hasWork()) return true;
    return legacy_host_->legacySnnHasWork();
}

double SnnWorkload::getUtilization() const {
    if (!legacy_host_) {
        fatal_missing_legacy_host_(rt_.log, "bindLegacyHost missing (getUtilization)");
    }
    return legacy_host_->legacySnnGetUtilization();
}

void SnnWorkload::getStatistics(std::map<std::string, uint64_t>& stats) const {
    if (!legacy_host_) {
        fatal_missing_legacy_host_(rt_.log, "bindLegacyHost missing (getStatistics)");
    }
    legacy_host_->legacySnnGetStatistics(stats);
}

void SnnWorkload::onInitPhase(unsigned phase) {
    if (!legacy_host_) return;
    // Ensure compute core is configured before init() phases are forwarded.
    ensureComputeCoreConfigured_();
    if (compute_core_) {
        compute_core_->onInit(phase);
    }
    if (phase == 0) {
        // Ensure route/comm is ready before first firing window.
        ensureSpikeCommConfigured_();
    }
}

void SnnWorkload::onSetup() {
    if (!legacy_host_) return;
    ensureComputeCoreConfigured_();
    if (compute_core_) compute_core_->onSetup();
}

void SnnWorkload::onFinish() {
    if (!legacy_host_) return;
    if (compute_core_) compute_core_->onFinish();
}

void SnnWorkload::applyGatingDecision(uint32_t src_global,
                                      const std::vector<uint32_t>& dest_pes,
                                      uint64_t current_cycle,
                                      uint64_t ttl_cycles) {
    ensureSpikeCommConfigured_();
    if (!synapse_route_) return;
    synapse_route_->applyGatingDecision(src_global, dest_pes, current_cycle, ttl_cycles);
}

void SnnWorkload::emitNeuronFire(uint32_t neuron_idx, uint64_t now_cycle) {
    ensureSpikeCommConfigured_();
    if (!spike_comm_) return;
    spike_comm_->emitNeuronFire(neuron_idx, now_cycle);
}

uint64_t SnnWorkload::emitNeuronFireBatch(const std::vector<uint32_t>& neuron_indices, uint64_t now_cycle) {
    ensureSpikeCommConfigured_();
    if (!spike_comm_) return 0;
    return spike_comm_->emitNeuronFireBatch(neuron_indices, now_cycle);
}

bool SnnWorkload::ready() const {
    return spike_comm_ && spike_comm_configured_ && spike_comm_->ready();
}

void SnnWorkload::onGasStageEvent(const GasStageEvent& ev) {
    // Phase4-Task6.4: 仅在窗口端到端语义下接管（apply_acc_enable + gas_window_mode）。
    if (!(apply_acc_enable_ && gas_window_mode_)) return;
    if (!legacy_host_) {
        fatal_missing_legacy_host_(rt_.log, "bindLegacyHost missing (onGasStageEvent)");
    }

    ensureComputeCoreConfigured_();
    ensureSpikeCommConfigured_();
    ensureWeightReaderOwned_();

    switch (ev.op) {
        case GasOp::BeginGather:
            gas_stage_ = GasStage::Gather;
            if (acc_ops_) acc_ops_->reset();
            if (weight_mem_subsystem_) {
                weight_mem_subsystem_->beginGatherWindow(window_read_enable_, num_neurons_);
            }
            if (compute_core_) compute_core_->onStageBeginGather(ev.superstep);
            break;
        case GasOp::BeginApply:
            gas_stage_ = GasStage::Apply;
            if (compute_core_) compute_core_->onStageBeginApply(ev.superstep);
            if (weight_mem_subsystem_) {
                weight_mem_subsystem_->beginApplyWindow(ev.superstep, window_read_debug_, rt_.log, (int)rt_.core_id);
                weight_mem_subsystem_->issueFallbackReadsIfNeeded(/*strict_gas_active=*/true);
                weight_mem_subsystem_->issueFromEdges();
            }
            break;
        case GasOp::EndApply:
            gas_stage_ = GasStage::Apply;
            if (compute_core_) compute_core_->onStageEndApply(ev.superstep);
            break;
        case GasOp::BeginScatter: {
            gas_stage_ = GasStage::Scatter;
            if (compute_core_) {
                compute_core_->clearFiredWindow();
                compute_core_->onStageBeginScatter(ev.superstep);
            }

            last_scatter_spikes_emitted_ = 0;

            // Apply accumulated deltas deterministically (sorted by post id).
            std::vector<std::pair<uint32_t, float>> pairs;
            if (acc_ops_) acc_ops_->collectSortedPairs(pairs);
            for (const auto& pr : pairs) {
                const uint32_t post = pr.first;
                const float dv = pr.second;
                if (dv == 0.0f) continue;
                if (compute_core_) compute_core_->applySynapticDelta(post, dv);
            }

            // End cycle + drain fired neurons, then delegate route/comm.
            if (compute_core_) {
                compute_core_->endCycle(now_cycle_cached_);
                std::vector<FireEvent> fired;
                compute_core_->drainOutputs(fired, /*clear=*/true);
                if (!fired.empty()) {
                    std::vector<uint32_t> neuron_indices;
                    neuron_indices.reserve(fired.size());
                    for (const auto& fe : fired) neuron_indices.push_back(fe.neuron_idx);

                    legacy_host_->legacySnnOnNeuronFires(neuron_indices, now_cycle_cached_);
                    const uint64_t emitted = emitNeuronFireBatch(neuron_indices, now_cycle_cached_);
                    last_scatter_spikes_emitted_ = emitted ? emitted : neuron_indices.size();
                }
            }
            if (acc_ops_) acc_ops_->reset();

            // Report scatter spikes to CoreShell for per-window aggregation/stats.
            legacy_host_->legacySnnOnGasScatterSpikesEmitted(ev.superstep, last_scatter_spikes_emitted_);
            break;
        }
        case GasOp::EndScatter:
            gas_stage_ = GasStage::Idle;
            if (compute_core_) compute_core_->onStageEndScatter(ev.superstep, last_scatter_spikes_emitted_);
            break;
        default:
            break;
    }
}

void SnnWorkload::onGasStatEvent(const GasStatEvent& /*st*/) {
    // GAS stat aggregation stays in CoreShell (PE-level aggregation + SST stat handles).
}

}} // namespace SST::SnnDL
