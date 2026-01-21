// -*- c++ -*-

#include <sst/core/sst_config.h>

#include "workload/snn/SnnWorkload.h"
#include "ISnnComputeCore.h"
#include "SpikeEvent.h"
#include "SnnWeightReader.h"
#include "synapse/gas/AccumulatorOps.h"
#include "synapse/weights/WeightAccessor.h"
#include "synapse/weights/WeightCacheOps.h"
#include "synapse/weights/SnnBcsrWeightManager.h"
#include "synapse/weights/WeightMemorySubsystem.h"
#include "synapse/route/SynapseRouteSubsystem.h"
#include "synapse/route/SpikeCommSubsystem.h"
#include "synapse/route/SpikeNocCodec.h"
#include "SynapseRouteBuildConfig.h"
#include "ISpikeTransport.h"
#include "NocSpikeTransport.h"
#include "workload/layout/NormalizedNeuronLayout.h"

#include <sst/core/output.h>
#include <sst/core/params.h>
#include <sst/core/shared/sharedArray.h>
#include <sst/core/statapi/stataccumulator.h>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <inttypes.h>
#include <utility>

namespace SST { namespace SnnDL {

namespace {

inline void addCountStat_(SST::Statistics::Statistic<uint64_t>* st, uint64_t n) {
    if (!st || n == 0) return;
    for (uint64_t i = 0; i < n; ++i) st->addData(1);
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
    // Layout params are normalized in bindRuntime() because different scripts historically
    // used different "num_neurons/global_neuron_base" conventions (per-core vs per-PE).
    num_neurons_param_ = params.find<uint32_t>("num_neurons", 64);
    global_neuron_base_param_ = params.find<uint64_t>("global_neuron_base", 0);
    neurons_per_pe_param_ = params.find<uint32_t>("neurons_per_pe", 0);
    num_neurons_ = num_neurons_param_;
    global_neuron_base_ = global_neuron_base_param_;
    node_neuron_base_ = 0;
    layout_normalized_ = false;
    total_nodes_cfg_ = params.find<uint32_t>("total_nodes", 16);
    apply_acc_enable_ = params.find<int>("apply_acc_enable", 0) != 0;
    gas_window_mode_ = params.find<int>("gas_window_mode", 0) != 0;
    window_read_enable_ = params.find<int>("window_read_enable", 0) != 0;
    window_read_debug_ = params.find<int>("window_read_debug", 0) != 0;
    scheme1_enable_ = params.find<int>("scheme1_enable", 0) != 0;
    const std::string index_mode_str = params.find<std::string>("index_mode", "pre_row_post_col");
    use_post_row_pre_col_ =
        (index_mode_str == "post_row_pre_col") ||
        (index_mode_str == "bcsr_post_row") ||
        (index_mode_str == "csr_post_row");
    use_bcsr_ = (index_mode_str == "bcsr_post_row");

    // Phase10: GAS/window 模式下默认启用 strict window-read spike input（由 workload 直接记录 edge/touch）。
    // 兼容性：某些环境可能会把未显式设置的参数默认注入为 0，此时不应“误关掉”窗口路径，因此将 0 视为 auto。
    const int wsi = params.find<int>("workload_spike_input_enable", -1);
    const bool auto_enable =
        (apply_acc_enable_ && gas_window_mode_ && window_read_enable_ && !scheme1_enable_);
    if (wsi > 0) {
        workload_spike_input_enable_ = true;
    } else if (wsi == 0) {
        workload_spike_input_enable_ = auto_enable;
    } else {
        workload_spike_input_enable_ = auto_enable;
    }

    enable_weight_fetch_ = params.find<int>("enable_weight_fetch", 0) != 0;
    edge_collector_max_capacity_ = static_cast<size_t>(params.find<uint64_t>("edge_collector_max_capacity", 1'000'000));
    const int record_apply_default = (gas_window_mode_ && apply_acc_enable_) ? 1 : 0;
    record_edge_apply_enable_ = params.find<int>("record_edge_apply_enable", record_apply_default) != 0;
    record_edge_idle_enable_ = params.find<int>("record_edge_idle_enable", 0) != 0;
    record_edge_scatter_enable_ = params.find<int>("record_edge_scatter_enable", 0) != 0;

    cores_per_pe_cfg_ = params.find<uint32_t>("total_cores", 8);
    if (cores_per_pe_cfg_ == 0) cores_per_pe_cfg_ = 1;
    // Defer neurons_per_core/neurons_per_pe derivation to normalizeLayout_().
    neurons_per_core_cfg_ = 0;
    neurons_per_pe_cfg_ = neurons_per_pe_param_;

    // WeightLoader barrier (shared signal): allow memory readers to defer until loader is done.
    loader_done_key_ = params.find<std::string>("loader_done_key", "");
    wait_for_loader_done_ = !loader_done_key_.empty();
    loader_ready_latched_ = false;
    loader_ready_logged_ = false;
    if (wait_for_loader_done_) {
        loader_done_shared_ = std::make_unique<SST::Shared::SharedArray<int>>();
        loader_done_shared_->initialize(loader_done_key_, 1, 0);
    } else {
        loader_done_shared_.reset();
    }

    // Step-gate explicit end handshake knobs (optional; safe defaults).
    gather_quiesce_cycles_ = params.find<uint32_t>("gas_gather_quiesce_cycles", gather_quiesce_cycles_);
    gather_min_cycles_ = params.find<uint32_t>("gas_gather_min_cycles", gather_min_cycles_);
    if (gather_min_cycles_ == 0) gather_min_cycles_ = 1;

    // Reset window-local state.
    gather_seq_ = 0;
    gather_begin_cycle_ = 0;
    gather_last_activity_cycle_ = 0;
    gather_end_requested_ = false;
    scatter_end_requested_ = false;
}

void SnnWorkload::bindRuntime(const Runtime& rt) {
    rt_ = rt;
    normalizeLayout_();
    if (cores_per_pe_cfg_ == 0 || neurons_per_core_cfg_ == 0 || neurons_per_pe_cfg_ == 0 ||
        (neurons_per_pe_cfg_ % cores_per_pe_cfg_) != 0 ||
        static_cast<uint64_t>(neurons_per_pe_cfg_) !=
            static_cast<uint64_t>(cores_per_pe_cfg_) * static_cast<uint64_t>(neurons_per_core_cfg_)) {
        if (rt_.log) {
            rt_.log->fatal(CALL_INFO, -1,
                           "SnnWorkload fatal: 全局布局口径不一致: total_cores=%u neurons_per_core=%u neurons_per_pe=%u (raw: num_neurons=%u neurons_per_pe=%u base=0x%" PRIx64 ")\n",
                           cores_per_pe_cfg_,
                           neurons_per_core_cfg_,
                           neurons_per_pe_cfg_,
                           num_neurons_param_,
                           neurons_per_pe_param_,
                           static_cast<uint64_t>(global_neuron_base_param_));
        }
        std::abort();
    }
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
        // Strict universal-kernel boundary: workload=snn must route via INocTransport (NocPacketEvent),
        // not via legacy parent->sendSpike(SpikeEvent*) path.
        if (!rt_.noc) {
            if (rt_.log) rt_.log->fatal(CALL_INFO, -1, "SnnWorkload requires Runtime.noc (INocTransport) for spike comm\n");
            std::abort();
        }
        if (!noc_spike_transport_) noc_spike_transport_ = std::make_unique<NocSpikeTransport>();
        noc_spike_transport_->setNocTransport(rt_.noc);
        noc_spike_transport_->setSourceCore(static_cast<int>(rt_.core_id));
        noc_spike_transport_->configureLayout(total_nodes_cfg_, cores_per_pe_cfg_, neurons_per_core_cfg_);
        transport = noc_spike_transport_.get();
        SpikeCommRuntimeConfig crt{};
        crt.log = rt_.log;
        crt.transport = transport;
        crt.noc = rt_.noc;
        crt.src_core = static_cast<int>(rt_.core_id);
        crt.node_id = static_cast<uint32_t>(rt_.node_id);
        crt.synapse_route = synapse_route_.get();
        crt.global_neuron_base = global_neuron_base_;
        spike_comm_->bindRuntime(crt);
    }
}

void SnnWorkload::normalizeLayout_() {
    if (layout_normalized_) return;
    layout_normalized_ = true;

    uint32_t weights_cols = 0;
    if (params_) weights_cols = params_->find<uint32_t>("weights_cols", 0);

    const auto n =
        normalizeNeuronLayout(static_cast<uint32_t>(rt_.node_id),
                              static_cast<uint32_t>(rt_.core_id),
                              total_nodes_cfg_,
                              cores_per_pe_cfg_,
                              num_neurons_param_,
                              neurons_per_pe_param_,
                              global_neuron_base_param_,
                              weights_cols);

    total_nodes_cfg_ = n.total_nodes;
    cores_per_pe_cfg_ = n.cores_per_pe;
    neurons_per_core_cfg_ = n.neurons_per_core;
    neurons_per_pe_cfg_ = n.neurons_per_pe;
    node_neuron_base_ = n.node_neuron_base;
    num_neurons_ = neurons_per_core_cfg_;

    const uint64_t base_param = static_cast<uint64_t>(global_neuron_base_param_);
    global_neuron_base_ = n.core_neuron_base;
    if (n.base_match_score == 0 && rt_.log) {
        rt_.log->verbose(CALL_INFO, 1, 0,
                         "⚠️ SnnWorkload layout normalize: global_neuron_base(0x%" PRIx64 ") 与推导不一致，回退到 core_base=0x%" PRIx64 " (node=%u core=%u)\n",
                         base_param,
                         static_cast<uint64_t>(global_neuron_base_),
                         static_cast<uint32_t>(rt_.node_id),
                         static_cast<uint32_t>(rt_.core_id));
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

bool SnnWorkload::ensureLoaderReady_() {
    if (!wait_for_loader_done_) return true;
    if (loader_ready_latched_) return true;
    if (!loader_done_shared_) return true;
    if (loader_done_shared_->size() == 0) return false;
    const int ready = loader_done_shared_->mutex_read(0);
    if (ready != 0) {
        loader_ready_latched_ = true;
        if (window_read_debug_ && !loader_ready_logged_ && rt_.log) {
            rt_.log->verbose(CALL_INFO, 2, 0,
                             "[diag-loader] workload=snn core=%u weights_ready at cycle=%" PRIu64 "\n",
                             static_cast<uint32_t>(rt_.core_id),
                             static_cast<uint64_t>(now_cycle_cached_));
        }
        loader_ready_logged_ = true;
        return true;
    }
    return false;
}

void SnnWorkload::ensureWeightReaderOwned_() {
    if (weight_reader_) return;
    normalizeLayout_();

    if (!params_) {
        if (rt_.log) rt_.log->fatal(CALL_INFO, -1, "SnnWorkload missing cached Params (configureFromParams not called?)\n");
        std::abort();
    }

    // === Build local cache/weight accessor/BCSR manager (workload-owned) ===
    const uint32_t max_cache_entries = params_->find<uint32_t>("max_cache_entries", 65536);
    const bool use_clock_weight_cache = params_->find<int>("use_clock_weight_cache", 0) != 0;
    const bool disable_weight_cache = params_->find<int>("disable_weight_cache", 0) != 0;

    if (!weight_cache_ops_) weight_cache_ops_ = std::make_unique<WeightCacheOps>();
    {
        WeightCacheOps::Config cache_cfg{};
        cache_cfg.max_entries = max_cache_entries;
        cache_cfg.use_clock = use_clock_weight_cache;
        cache_cfg.disable_cache = disable_weight_cache;
        weight_cache_ops_->configure(cache_cfg, /*on_evict*/[](){});
        weight_cache_ops_->reserve(cache_cfg.max_entries ? cache_cfg.max_entries : 1);
    }

    if (!weight_accessor_) weight_accessor_ = std::make_unique<WeightAccessor>();
    uint32_t weights_cols = params_->find<uint32_t>("weights_cols", 0);
    if (weights_cols == 0) weights_cols = num_neurons_;
    weight_accessor_->configure(WeightAccessorConfig{
        static_cast<uint32_t>(rt_.core_id),
        static_cast<uint64_t>(global_neuron_base_),
        static_cast<uint32_t>(num_neurons_),
        static_cast<uint32_t>(weights_cols),
        use_post_row_pre_col_
    });

    if (!bcsr_mgr_) bcsr_mgr_ = std::make_unique<BcsrWeightManager>();
    if (use_bcsr_) {
        const uint64_t base_addr = params_->find<uint64_t>("base_addr", 0);
        const uint64_t rp_off = params_->find<uint64_t>("bcsr_rowptr_offset", 0);
        const uint64_t ci_off = params_->find<uint64_t>("bcsr_colidx_offset", 0);
        const uint64_t bd_off = params_->find<uint64_t>("bcsr_blockdata_offset", 0);
        const uint64_t id_off = params_->find<uint64_t>("bcsr_blockids_offset", 0);
        const uint32_t br = params_->find<uint32_t>("bcsr_block_rows", 16);
        const uint32_t bc = params_->find<uint32_t>("bcsr_block_cols", 16);
        const uint32_t idxb = params_->find<uint32_t>("bcsr_idx_bytes", 2);
        const uint32_t valb = params_->find<uint32_t>("bcsr_val_bytes", 4);
        const uint64_t rowptr_addr = base_addr + rp_off;
        const uint64_t colidx_addr = base_addr + ci_off;
        const uint64_t blockdata_addr = base_addr + bd_off;
        const uint64_t blockids_addr = id_off ? (base_addr + id_off) : 0;
        bcsr_mgr_->configure(rowptr_addr, colidx_addr, blockdata_addr, blockids_addr, br, bc, idxb, valb);
        // Row-index cache (colidx) 是 Apply 的关键路径：cap 太小会导致每窗重复 colidx burst → bursts 不降 → apply_ns 不降。
        // 默认不强制覆盖用户配置；仅当显式启用 auto_fit 时，自动扩到覆盖本 core 的全部 block rows（语义不变，仅减少重复读）。
        uint32_t row_cap = params_->find<uint32_t>("bcsr_row_index_cache_cap", 64);
        const bool row_auto_fit = params_->find<int>("bcsr_row_index_cache_auto_fit", 0) != 0;
        if (row_auto_fit && row_cap > 0) {
            const uint32_t br_eff = br ? br : 16;
            const uint32_t n_block_rows =
                br_eff ? ((num_neurons_ + br_eff - 1u) / br_eff) : static_cast<uint32_t>(num_neurons_);
            if (n_block_rows > 0 && row_cap < n_block_rows) {
                if (rt_.log) {
                    rt_.log->verbose(CALL_INFO, 2, 0,
                                     "[bcsr] auto-fit row_index_cache_cap %u -> %u (node=%u core=%u rows=%u br=%u)\n",
                                     row_cap,
                                     n_block_rows,
                                     static_cast<uint32_t>(rt_.node_id),
                                     static_cast<uint32_t>(rt_.core_id),
                                     static_cast<uint32_t>(num_neurons_),
                                     br_eff);
                }
                row_cap = n_block_rows;
            }
        }
        bcsr_mgr_->setRowIndexCacheCapacity(row_cap);
        bcsr_mgr_->setBlockCacheCapacity(params_->find<uint32_t>("bcsr_block_cache_cap", 256));
        {
            std::string pol = params_->find<std::string>("bcsr_block_cache_policy", "lru");
            for (auto& ch : pol) {
                if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
            }
            if (pol == "fifo") {
                bcsr_mgr_->setBlockCachePolicy(BcsrWeightManager::BlockCachePolicy::FIFO);
            } else if (pol == "legacy_unordered" || pol == "legacy") {
                bcsr_mgr_->setBlockCachePolicy(BcsrWeightManager::BlockCachePolicy::LegacyUnordered);
            } else {
                bcsr_mgr_->setBlockCachePolicy(BcsrWeightManager::BlockCachePolicy::LRU);
            }
        }
    }

    // === Build WeightMemorySubsystem ===
    auto wms = std::make_unique<WeightMemorySubsystem>();
    wms->configure(
        [this](uint64_t key, float& out) -> bool {
            return weight_cache_ops_ ? weight_cache_ops_->tryGet(key, out) : false;
        },
        [this](uint64_t key, float v) {
            if (weight_cache_ops_) weight_cache_ops_->store(key, v);
        });

    const uint64_t base_addr = params_->find<uint64_t>("base_addr", 0);
    const uint64_t weight_region_end =
        base_addr + static_cast<uint64_t>(num_neurons_) * static_cast<uint64_t>(weights_cols) * sizeof(float);

    WeightMemorySubsystem::OrchestratorConfig ocfg{};
    ocfg.accessor = weight_accessor_.get();
    ocfg.cache_try = [this](uint64_t key, float& out) -> bool {
        return weight_cache_ops_ ? weight_cache_ops_->tryGet(key, out) : false;
    };
    ocfg.cache_put = [this](uint64_t key, float v) {
        if (weight_cache_ops_) weight_cache_ops_->store(key, v);
    };
    ocfg.acc_update = [this](uint32_t post_local, float dv) {
        if (acc_ops_) acc_ops_->update(post_local, dv);
    };
    ocfg.report_mem_issue = [this](size_t bytes, bool /*count_weight_read*/) {
        if (rt_.reporting.report_mem_issue) rt_.reporting.report_mem_issue(rt_.reporting.ctx, bytes);
    };
    ocfg.ensure_loader_ready = [this]() { return ensureLoaderReady_(); };
    ocfg.bcsr_rowptr_ready = [this]() { return !use_bcsr_ || (bcsr_mgr_ && bcsr_mgr_->isRowptrReady()); };
    ocfg.ensure_rowptr_ready_or_fatal = [this](const char* reason) {
        if (rt_.log) {
            rt_.log->fatal(CALL_INFO, -1,
                           "SnnWorkload fatal: BCSR rowptr not ready (%s) node=%u core=%u\n",
                           reason ? reason : "unknown",
                           static_cast<uint32_t>(rt_.node_id),
                           static_cast<uint32_t>(rt_.core_id));
        }
        std::abort();
    };
    ocfg.resume_issue_after_rowptr_ready = [this]() {
        if (apply_acc_enable_ && gas_window_mode_ && gas_stage_ == GasStage::Apply && weight_mem_subsystem_) {
            weight_mem_subsystem_->issueFromEdges();
        }
    };
    ocfg.use_bcsr = use_bcsr_;
    ocfg.bcsr_prefetch_all = params_->find<int>("bcsr_prefetch_all", 0) != 0;
    ocfg.bcsr_colidx_inflight_coalesce_enable =
        params_->find<int>("bcsr_colidx_inflight_coalesce_enable", 1) != 0;
    ocfg.bcsr_block_inflight_coalesce_enable =
        params_->find<int>("bcsr_block_inflight_coalesce_enable", 1) != 0;
    ocfg.bcsr_row_index_prefetch_mode =
        params_->find<std::string>("bcsr_row_index_prefetch_mode", "auto");
    ocfg.bcsr_row_index_prefetch_all_rows_threshold =
        params_->find<uint32_t>("bcsr_row_index_prefetch_all_rows_threshold", 1024);
    ocfg.bcsr_row_index_prefetch_all_rows_max_bytes =
        params_->find<uint64_t>("bcsr_row_index_prefetch_all_rows_max_bytes", 64ull * 1024ull);
    ocfg.bcsr_block_cache_auto_tune =
        params_->find<int>("bcsr_block_cache_auto_tune", 1) != 0;
    ocfg.bcsr_block_cache_max_bytes =
        params_->find<uint64_t>("bcsr_block_cache_max_bytes", 64ull * 1024ull * 1024ull);
    ocfg.bcsr_block_cache_tune_miss_ratio =
        params_->find<float>("bcsr_block_cache_tune_miss_ratio", 0.05f);
    ocfg.bcsr_block_cache_tune_min_misses =
        params_->find<uint32_t>("bcsr_block_cache_tune_min_misses", 64);
    ocfg.bcsr_populate_weight_cache_enable =
        params_->find<int>("bcsr_populate_weight_cache_enable", 1) != 0;
    if (disable_weight_cache) ocfg.bcsr_populate_weight_cache_enable = false;
    ocfg.bcsr_force_file_read = params_->find<int>("bcsr_force_file_read", 0) != 0;
    ocfg.bcsr_rowptr_file_fallback_enable = params_->find<int>("bcsr_rowptr_file_fallback_enable", 0) != 0;
    ocfg.bcsr_weight_guard_enable = params_->find<int>("bcsr_weight_guard_enable", 1) != 0;
    ocfg.bcsr_weight_abs_max = params_->find<float>("bcsr_weight_abs_max", 10.0f);
    ocfg.bcsr_semantic_verify_enable = params_->find<int>("bcsr_semantic_verify_enable", 0) != 0;
    ocfg.bcsr_semantic_verify_max_edges = params_->find<uint32_t>("bcsr_semantic_verify_max_edges", 64);
    ocfg.bcsr_semantic_verify_max_mismatch = params_->find<uint32_t>("bcsr_semantic_verify_max_mismatch", 8);
    ocfg.bcsr_semantic_verify_abs_tol = params_->find<float>("bcsr_semantic_verify_abs_tol", 1e-6f);
    ocfg.bcsr_semantic_verify_rel_tol = params_->find<float>("bcsr_semantic_verify_rel_tol", 1e-6f);
    ocfg.readresp_zero_fallback = params_->find<int>("readresp_zero_fallback", 0) != 0;
    ocfg.init_default_weight = params_->find<float>("init_default_weight", 0.5f);
    ocfg.num_neurons = num_neurons_;
    ocfg.weights_cols = weights_cols;
    ocfg.use_post_row_pre_col = use_post_row_pre_col_;
    ocfg.base_addr = base_addr;
    ocfg.weight_region_end = weight_region_end;
    ocfg.read_force_single = params_->find<int>("read_force_single", 0) != 0;
    ocfg.merge_read_cacheline = params_->find<int>("merge_read_cacheline", 1) != 0;
    ocfg.merge_read_row = params_->find<int>("merge_read_row", 0) != 0;
    ocfg.merge_read_auto = params_->find<int>("merge_read_auto", 0) != 0;
    ocfg.line_size_bytes = params_->find<uint32_t>("line_size_bytes", 64);
    ocfg.byte_exact_verify_enable = params_->find<int>("byte_exact_verify_enable", 0) != 0;
    ocfg.byte_exact_verify_mode = params_->find<std::string>("byte_exact_verify_mode", "");
    ocfg.byte_exact_verify_row_scale = params_->find<uint32_t>("byte_exact_verify_row_scale", 1024);
    ocfg.byte_exact_verify_max_mismatch = params_->find<uint32_t>("byte_exact_verify_max_mismatch", 8);
    ocfg.memory_warmup_cycles = params_->find<uint64_t>("memory_warmup_cycles", 0);
    ocfg.loader_barrier_cycles = params_->find<uint64_t>("loader_barrier_cycles", 0);
    ocfg.node_id = rt_.node_id;
    ocfg.core_id = rt_.core_id;
    ocfg.weights_template = params_->find<std::string>("weights_template", "");
    ocfg.bcsr_mgr = bcsr_mgr_.get();
    wms->configureOrchestrator(std::move(ocfg));

    const uint32_t window_read_budget = params_->find<uint32_t>("window_read_budget", 1024);
    const uint32_t max_outstanding = params_->find<uint32_t>("max_outstanding_requests", 16);
    wms->configureWindow(window_read_budget, max_outstanding);
    if (window_read_enable_) wms->reserveWindowContainers(num_neurons_);

    wms->bindMemory(rt_.mem);
    weight_mem_subsystem_ = wms.get();
    weight_reader_ = std::move(wms);

    // Phase4-Task6.4: window accumulator moved into workload=snn; bind WMS acc_update callback.
    if (apply_acc_enable_ && gas_window_mode_) {
        if (!acc_ops_) {
            const bool dense_enable = params_->find<int>("apply_dense_acc_enable", 1) != 0;
            const bool shadow_verify_enable =
                dense_enable && (params_->find<int>("acc_shadow_verify_enable", 0) != 0);
            AccumulatorOpsConfig acc_cfg{};
            acc_cfg.num_neurons = num_neurons_;
            acc_cfg.dense_enable = dense_enable;
            acc_cfg.spill_enable = params_->find<int>("acc_spill_enable", 1) != 0;
            acc_cfg.high_watermark_bytes = params_->find<uint64_t>("acc_high_watermark_bytes", 16 * 1024 * 1024);
            acc_cfg.shadow_verify_enable = shadow_verify_enable;
            acc_cfg.window_read_debug = window_read_debug_;
            acc_cfg.core_id = static_cast<int>(rt_.core_id);
            acc_cfg.verbose = rt_.log ? rt_.log->getVerboseLevel() : 0;
            acc_cfg.out = rt_.log;
            acc_ops_ = std::make_unique<AccumulatorOps>(acc_cfg);
        }
        if (weight_mem_subsystem_) {
            weight_mem_subsystem_->overrideAccUpdate([this](uint32_t post_local, float dv) {
                if (acc_ops_) acc_ops_->update(post_local, dv);
            });
        }
    }
}

void SnnWorkload::ensureComputeCoreConfigured_() {
    if (compute_configured_) return;
    normalizeLayout_();
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
                rt_.log->verbose(CALL_INFO, 1, 0,
                                 "⚠️ 未知 compute_core_impl='%s'，回退到 default\n",
                                 compute_core_impl_.c_str());
            }
            compute_core_ = createComputeCoreByName("default");
        }
        if (!compute_core_) {
            if (rt_.log) rt_.log->fatal(CALL_INFO, -1, "createComputeCoreByName failed for both impl and default\n");
            std::abort();
        }
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
    // Phase10: legacy host removed. Learning writeback is optional; keep disabled unless a future workload hook is added.
    ctx.writeback_fn = [](const std::unordered_map<uint64_t, float>&, float, float) -> bool { return false; };
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
    normalizeLayout_();
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
    cfg.rows = neurons_per_pe_cfg_;
    cfg.cols = weights_cols;
    cfg.total_nodes = total_nodes_cfg_;
    cfg.cores_per_pe = cores_per_pe_cfg_;
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

    // Native multicast routing (default off for backward compatibility)
    cfg.multicast_enable = params_->find<int>("multicast_enable", 0) != 0;
    cfg.multicast_block_w = params_->find<uint32_t>("multicast_block_w", 2);
    cfg.multicast_block_h = params_->find<uint32_t>("multicast_block_h", 2);
    cfg.multicast_ingress_policy = params_->find<std::string>("multicast_ingress_policy", "top_left");
    cfg.multicast_inter_policy = params_->find<std::string>("multicast_inter_policy", "xy");
    cfg.multicast_intra_policy = params_->find<std::string>("multicast_intra_policy", "manhattan_x_first");

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

    // Strict universal-kernel boundary: workload=snn must route via INocTransport (NocPacketEvent),
    // not via legacy parent->sendSpike(SpikeEvent*) path.
    if (!rt_.noc) {
        if (rt_.log) rt_.log->fatal(CALL_INFO, -1, "SnnWorkload requires Runtime.noc (INocTransport) for spike comm\n");
        std::abort();
    }
    if (!noc_spike_transport_) noc_spike_transport_ = std::make_unique<NocSpikeTransport>();
    noc_spike_transport_->setNocTransport(rt_.noc);
    noc_spike_transport_->setSourceCore(static_cast<int>(rt_.core_id));
    noc_spike_transport_->configureLayout(total_nodes_cfg_, cfg.cores_per_pe, neurons_per_core_cfg_);
    ISpikeTransport* transport = noc_spike_transport_.get();

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
    bool did = false;
    // Phase4-Task6.1：compute core 的 per-tick 驱动下沉到 workload=snn。
    ensureComputeCoreConfigured_();
    if (compute_core_) {
        compute_core_->onClockTick(now_cycle);
    }
    // Phase4-Task6.2-Step2：weights/memory 子系统所有权迁入 workload=snn，tick 由 workload 直接驱动。
    if (weight_mem_subsystem_) {
        weight_mem_subsystem_->onClockTick(now_cycle);
    }

    // Phase7-Task1/Phase10: strict window-read spike queue processing moved into workload=snn.
    bool did_spike = false;
    if (workload_spike_input_enable_ && isWindowWorkload_() && window_read_enable_ && !scheme1_enable_) {
        did_spike = processReadySpikes_(nowNs_());
    }
    did = did || did_spike;

    // Phase4-Task6.3: 非 window 模式下的 “endCycle->drain->route/comm” 闭环迁入 workload。
    if (!windowScatterModeActive_() && compute_core_) {
        compute_core_->endCycle(now_cycle);
        std::vector<FireEvent> fired;
        compute_core_->drainOutputs(fired, /*clear=*/true);
        if (!fired.empty()) {
            const uint64_t fired_cnt = static_cast<uint64_t>(fired.size());
            if (rt_.sinks.neurons_fired) (*rt_.sinks.neurons_fired) += fired_cnt;
            if (rt_.sinks.spikes_generated) (*rt_.sinks.spikes_generated) += fired_cnt;
            addCountStat_(rt_.sinks.stat_neurons_fired_total, fired_cnt);
            addCountStat_(rt_.sinks.stat_spikes_generated_total, fired_cnt);
            std::vector<uint32_t> neuron_indices;
            neuron_indices.reserve(fired.size());
            for (const auto& ev : fired) neuron_indices.push_back(ev.neuron_idx);
            ensureSpikeCommConfigured_();
            (void)emitNeuronFireBatch(neuron_indices, now_cycle);
            did = true;
        }
    }

    // Step-gate explicit end handshake: end Gather when input has been quiescent for N cycles.
    // This is only acted on when the host provides callbacks (otherwise GatherBufferIF's legacy
    // cycle-driven windowing remains in effect).
    if (isWindowWorkload_() &&
        gas_stage_ == GasStage::Gather &&
        !gather_end_requested_ &&
        gather_seq_ != 0 &&
        rt_.reporting.request_gas_end_gather) {
        const uint64_t since_begin =
            (now_cycle_cached_ >= gather_begin_cycle_) ? (now_cycle_cached_ - gather_begin_cycle_) : 0;
        uint64_t quiet =
            (now_cycle_cached_ >= gather_last_activity_cycle_) ? (now_cycle_cached_ - gather_last_activity_cycle_) : 0;
        // In step-gated mode, spikes for this step may still be in flight even if this core
        // hasn't received any yet. Treat NoC non-idle as "activity" to avoid ending Gather
        // before the network has drained enough for inputs to arrive.
        if (rt_.noc && !rt_.noc->isIdle()) {
            gather_last_activity_cycle_ = now_cycle_cached_;
            quiet = 0;
        }
        if (since_begin >= static_cast<uint64_t>(gather_min_cycles_) &&
            quiet >= static_cast<uint64_t>(gather_quiesce_cycles_) &&
            incoming_spikes_.empty()) {
            rt_.reporting.request_gas_end_gather(rt_.reporting.ctx, gather_seq_);
            gather_end_requested_ = true;
        }
    }

    return did;
}

bool SnnWorkload::deliverPacket(NocPacketEvent* packet) {
    // CoreShell 统一以 packet 输入；Spike 的解码/语义处理完全由 workload 承担。
    // NOTE: packet ownership is transferred to this function (it will always delete packet).
    if (!packet) return true;

    const auto kind = packet->packetKind();
    if (kind == NocPacketKind::Spike) {
        SpikeEvent* spike = SpikeNocCodec::decode(*packet);
        delete packet;
        if (!spike) return true;

        // Reuse the existing spike entrypoint (keeps strict window-read migration gates + legacy fallback).
        deliverSpike(spike);
        return true;
    }

    if (kind == NocPacketKind::SpikeKey) {
        SpikeNocCodec::WireSpikeKeyV2 ws{};
        const uint64_t ts = packet->timestamp;
        const bool ok = SpikeNocCodec::decodeSpikeKeyAny(packet->payload, ws);
        delete packet;
        if (!ok || (ws.version != 1 && ws.version != 2)) return true;

        ensureSpikeCommConfigured_();
        if (!synapse_route_) return true;
        auto routes = synapse_route_->routesShared();
        if (!routes) return true;

        if (routes.get() != routes_shared_for_posts_cache_.get()) {
            routes_shared_for_posts_cache_ = routes;
            pre_to_posts_local_.clear();
        }

        const uint32_t pre_global = ws.pre_global;
        auto itc = pre_to_posts_local_.find(pre_global);
        if (itc == pre_to_posts_local_.end()) {
            std::vector<uint32_t> posts_local;
            auto itr = routes->find(pre_global);
            if (itr != routes->end()) {
                const uint32_t denom = (neurons_per_pe_cfg_ > 0) ? neurons_per_pe_cfg_ : num_neurons_;
                if (denom > 0) {
                    for (uint32_t dest_global : itr->second) {
                        const uint32_t dest_node = dest_global / denom;
                        if (dest_node != static_cast<uint32_t>(rt_.node_id)) continue;

                        const uint32_t local_in_pe = dest_global % denom;
                        const uint32_t dest_core = (num_neurons_ ? (local_in_pe / num_neurons_) : 0);
                        if (dest_core != static_cast<uint32_t>(rt_.core_id)) continue;

                        const uint32_t post_local = (num_neurons_ ? (local_in_pe % num_neurons_) : 0);
                        if (post_local < num_neurons_) posts_local.push_back(post_local);
                    }
                }
            }

            if (!posts_local.empty()) {
                std::sort(posts_local.begin(), posts_local.end());
                posts_local.erase(std::unique(posts_local.begin(), posts_local.end()), posts_local.end());
            }

            itc = pre_to_posts_local_.emplace(pre_global, std::move(posts_local)).first;
        }

        const auto& posts_local = itc->second;
        if (posts_local.empty()) return true;

        for (uint32_t post_local : posts_local) {
            const uint32_t dest_global = static_cast<uint32_t>(global_neuron_base_ + static_cast<uint64_t>(post_local));
            auto* spike = new SpikeEvent(pre_global,
                                         dest_global,
                                         static_cast<uint32_t>(rt_.node_id),
                                         /*weight=*/1.0,
                                         ts);
            deliverSpike(spike);
        }
        return true;
    }

    // 非 Spike packet 由其它 workload 或上层消费；SNN workload 默认丢弃以避免语义漂移。
    delete packet;
    return true;
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

    if (gas_stage_ == GasStage::Gather) {
        gather_last_activity_cycle_ = now_cycle_cached_;
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
    ensureComputeCoreConfigured_();

    if (rt_.sinks.spikes_received) (*rt_.sinks.spikes_received)++;
    if (rt_.sinks.stat_spikes_received_total) rt_.sinks.stat_spikes_received_total->addData(1);

    spike->clearLocalCache();
    spike->setTimestamp(nowNs_());
    if (compute_core_) compute_core_->onSpikeDelivered(spike);

    // Resolve post-local (legacy compatible).
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
    if (!post_local_valid || post_local >= num_neurons_) {
        delete spike;
        return;
    }

    // Compute core gating (kept consistent with legacy: reject before recording any edge/access).
    if (apply_acc_enable_ && compute_core_) {
        if (!compute_core_->shouldAcceptSynapticInput(post_local, now_cycle_cached_)) {
            delete spike;
            return;
        }
    }

    // Window/GAS mode: record touches + edges immediately (do NOT defer to per-tick queue),
    // so BeginApply can issue reads deterministically even for same-cycle arrivals.
    if (workload_spike_input_enable_ && isWindowWorkload_() && window_read_enable_ && !scheme1_enable_) {
        ensureWeightReaderOwned_();
        if (gas_stage_ == GasStage::Gather) {
            gather_last_activity_cycle_ = now_cycle_cached_;
        }
        if (weight_mem_subsystem_) {
            // Maintain window sets immediately.
            weight_mem_subsystem_->noteWindowTouch(post_local, spike->getSourceNeuron(), num_neurons_);

            // Record edge for BeginApply issueFromEdges().
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
                        weight_mem_subsystem_->recordEdge(post_local, spike->getSourceNeuron());
                    }
                }
            }
        }

        if (rt_.sinks.synaptic_accesses) (*rt_.sinks.synaptic_accesses)++;
        if (rt_.sinks.stat_synaptic_accesses_total) rt_.sinks.stat_synaptic_accesses_total->addData(1);
        delete spike;
        return;
    }

    // Non-window path: best-effort single-synapse apply.
    if (compute_core_) {
        const uint32_t pre_global = spike->getSourceNeuron();
        if (enable_weight_fetch_) {
            ensureWeightReaderOwned_();
            IWeightReader* wr = weight_reader_.get();
            if (wr) {
                // Weight fetch is async; apply when resolved.
                wr->requestDense(pre_global, post_local, [this, pre_global, post_local](float w) {
                    if (!compute_core_) return;
                    SynapticEvent ev;
                    ev.post_local = post_local;
                    ev.pre_global = pre_global;
                    ev.weight = w;
                    compute_core_->onSynapticEvent(ev);
                    if (rt_.sinks.synaptic_accesses) (*rt_.sinks.synaptic_accesses)++;
                    if (rt_.sinks.stat_synaptic_accesses_total) rt_.sinks.stat_synaptic_accesses_total->addData(1);
                });
                delete spike;
                return;
            }
        }

        SynapticEvent ev;
        ev.post_local = post_local;
        ev.pre_global = pre_global;
        ev.weight = static_cast<float>(spike->getWeight());
        compute_core_->onSynapticEvent(ev);
        if (rt_.sinks.synaptic_accesses) (*rt_.sinks.synaptic_accesses)++;
        if (rt_.sinks.stat_synaptic_accesses_total) rt_.sinks.stat_synaptic_accesses_total->addData(1);
    }

    delete spike;
}

bool SnnWorkload::hasWork() const {
    if (!incoming_spikes_.empty()) return true;
    if (weight_mem_subsystem_ &&
        (weight_mem_subsystem_->pendingSize() > 0 || weight_mem_subsystem_->hasDeferredWork())) {
        return true;
    }
    // NOTE:
    // - For non-window (naive) execution, global step barrier completion uses a quiescent policy
    //   that must reflect in-flight transactions (spike/mem/noc), NOT neuron-state values.
    // - DefaultSnnComputeCore::hasWork() historically used v_mem threshold heuristics, which can
    //   remain true for long periods and would stall step completion indefinitely.
    // - Therefore, only treat compute_core_->hasWork() as "blocking" when the workload is in a
    //   window/scatter-driven mode (GAS or scheme1), where the core's stage machine defines work.
    if (compute_core_ && windowScatterModeActive_() && compute_core_->hasWork()) return true;
    return false;
}

double SnnWorkload::getUtilization() const {
    if (compute_core_) return compute_core_->getUtilization();
    return 0.0;
}

void SnnWorkload::getStatistics(std::map<std::string, uint64_t>& stats) const {
    const uint64_t spikes_received = rt_.sinks.spikes_received ? *rt_.sinks.spikes_received : 0;
    const uint64_t spikes_generated = rt_.sinks.spikes_generated ? *rt_.sinks.spikes_generated : 0;
    const uint64_t neurons_fired = rt_.sinks.neurons_fired ? *rt_.sinks.neurons_fired : 0;
    const uint64_t syn_accesses = rt_.sinks.synaptic_accesses ? *rt_.sinks.synaptic_accesses : 0;
    stats["spikes_received"] = spikes_received;
    stats["spikes_generated"] = spikes_generated;
    stats["neurons_fired"] = neurons_fired;
    stats["synaptic_accesses"] = syn_accesses;
    stats["gas_scatter_spikes_emitted_total"] = total_scatter_spikes_emitted_;
}

void SnnWorkload::onInitPhase(unsigned phase) {
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
    ensureComputeCoreConfigured_();
    if (compute_core_) compute_core_->onSetup();
}

void SnnWorkload::onFinish() {
    // Debug-only: if the window didn't reach EndScatter (e.g., global step barrier stops early),
    // dump the in-flight window's weight-read summary for root-cause analysis.
    if (window_read_debug_ && weight_mem_subsystem_) {
        weight_mem_subsystem_->finishWindowDiag();
    }
    if (weight_mem_subsystem_) {
        weight_mem_subsystem_->finishSemanticVerify();
    }
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

    ensureComputeCoreConfigured_();
    ensureSpikeCommConfigured_();
    ensureWeightReaderOwned_();

    switch (ev.op) {
        case GasOp::BeginGather:
            gas_stage_ = GasStage::Gather;
            gather_seq_ = ev.superstep;
            gather_begin_cycle_ = now_cycle_cached_;
            gather_last_activity_cycle_ = now_cycle_cached_;
            gather_end_requested_ = false;
            scatter_end_requested_ = false;
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

                    const uint64_t emitted = emitNeuronFireBatch(neuron_indices, now_cycle_cached_);
                    last_scatter_spikes_emitted_ = emitted ? emitted : neuron_indices.size();

                    const uint64_t fired_cnt = static_cast<uint64_t>(neuron_indices.size());
                    if (rt_.sinks.neurons_fired) (*rt_.sinks.neurons_fired) += fired_cnt;
                    if (rt_.sinks.spikes_generated) (*rt_.sinks.spikes_generated) += fired_cnt;
                    addCountStat_(rt_.sinks.stat_neurons_fired_total, fired_cnt);
                    addCountStat_(rt_.sinks.stat_spikes_generated_total, fired_cnt);
                }
            }
            if (acc_ops_) acc_ops_->reset();

            // Report scatter spikes for per-window aggregation/stats.
            if (rt_.sinks.spikes_emitted_window) (*rt_.sinks.spikes_emitted_window) = last_scatter_spikes_emitted_;
            if (rt_.sinks.window_spikes_all) (*rt_.sinks.window_spikes_all) += last_scatter_spikes_emitted_;
            total_scatter_spikes_emitted_ += last_scatter_spikes_emitted_;
            if (rt_.sinks.stat_gas_scatter_spikes_emitted_total && last_scatter_spikes_emitted_ > 0) {
                rt_.sinks.stat_gas_scatter_spikes_emitted_total->addData(last_scatter_spikes_emitted_);
            }
            if (rt_.reporting.report_apply_scatter) {
                rt_.reporting.report_apply_scatter(rt_.reporting.ctx,
                                                   /*acc_updates=*/0,
                                                   /*posts_touched=*/0,
                                                   /*spikes_emitted=*/last_scatter_spikes_emitted_,
                                                   /*hwm_bytes=*/0,
                                                   /*spill_records=*/0,
                                                   /*spilled_bytes=*/0);
            }

            // Step-gate explicit end handshake: request EndScatter after finishing Scatter work.
            if (!scatter_end_requested_ && rt_.reporting.request_gas_end_scatter) {
                rt_.reporting.request_gas_end_scatter(rt_.reporting.ctx, ev.superstep);
                scatter_end_requested_ = true;
            }
            break;
        }
        case GasOp::EndScatter:
            gas_stage_ = GasStage::Idle;
            if (compute_core_) compute_core_->onStageEndScatter(ev.superstep, last_scatter_spikes_emitted_);
            if (weight_mem_subsystem_ && window_read_debug_) {
                weight_mem_subsystem_->endScatterWindow(ev.superstep);
            }
            break;
        default:
            break;
    }
}

void SnnWorkload::onGasStatEvent(const GasStatEvent& /*st*/) {
    // GAS stat aggregation stays in CoreShell (PE-level aggregation + SST stat handles).
}

}} // namespace SST::SnnDL
