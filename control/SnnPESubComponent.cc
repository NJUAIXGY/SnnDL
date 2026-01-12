// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// SnnPESubComponent.cc: SnnPE SubComponent版本实现文件
//

#include <sst/core/sst_config.h>
#include "SnnPESubComponent.h"
#include "SnnPESubComponent_impl.h"
#include <fstream>
#include "synapse/gas/GasCustomCmd.h"
#include "synapse/gas/GasPhaseController.h"
#include "synapse/gas/AccumulatorOps.h"
#include "IPeAggregation.h"
#include "IManualWindowDrive.h"
#include "IGasStepGate.h"
#include "ISnnComputeCore.h"
#include "synapse/stdmem/StdMemEndpoint.h"
#include "synapse/weights/SnnBcsrWeightManager.h"
#include "synapse/weights/WeightMemorySubsystem.h"
#include "synapse/weights/WeightCacheOps.h"
#include "synapse/weights/WeightAccessor.h"
#include "ISpikeTransport.h"
#include "NocSpikeTransport.h"
#include "NocPacketEvent.h"
#include "IMemoryAccess.h"
#include "CoreWorkloadFactory.h"
#include "ISpikeWorkload.h"
#include "ISnnSpikeCommWorkload.h"
#include "synapse/route/SpikeCommSubsystem.h"
#include "synapse/route/SynapseRouteSubsystem.h"
#include "WorkloadConfig.h"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <vector>

using namespace SST;
using namespace SST::SnnDL;

// 诊断门控改为参数化：由 enable_extended_diagnostics_ 成员控制

// Lightweight logging helpers (file-local). Use consistent style across SnnDL.
#ifndef SNNDL_LOGPTR
#define SNNDL_LOGPTR(ptr, lvl, ...) do { if (ptr) (ptr)->verbose(CALL_INFO, (lvl), 0, __VA_ARGS__); } while(0)
#endif
#ifndef SNNDL_LOG
#define SNNDL_LOG(lvl, ...) SNNDL_LOGPTR(output_, (lvl), __VA_ARGS__)
#endif

#ifdef SNNDL_ENABLE_DEBUG_LOG
#define SNNDL_DEBUG_ENABLED 1
#define SNNDL_DEBUG_LOG(lvl, ...) SNNDL_LOG(lvl, __VA_ARGS__)
#define SNNDL_DEBUG_BLOCK(stmt) do { stmt; } while(0)
#else
#define SNNDL_DEBUG_ENABLED 0
#define SNNDL_DEBUG_LOG(lvl, ...) do {} while(0)
#define SNNDL_DEBUG_BLOCK(stmt) do {} while(0)
#endif

void SnnPESubComponent::reportStreamMemIssueThunk_(void* ctx, size_t bytes) {
    auto* core = static_cast<SnnPESubComponent*>(ctx);
    if (!core) return;
    if (core->impl_) {
        core->impl_->reportMemoryIssue(bytes, /*count_weight_read=*/false);
    }
}

void SnnPESubComponent::reportSnnMemIssueThunk_(void* ctx, size_t bytes) {
    auto* core = static_cast<SnnPESubComponent*>(ctx);
    if (!core) return;
    if (core->impl_) {
        core->impl_->reportMemoryIssue(bytes, /*count_weight_read=*/true);
    }
}

void SnnPESubComponent::reportApplyScatterThunk_(void* ctx,
                                                 uint64_t acc_updates,
                                                 uint64_t posts_touched,
                                                 uint64_t spikes_emitted,
                                                 uint64_t hwm_bytes,
                                                 uint64_t spill_records,
                                                 uint64_t spilled_bytes) {
    auto* core = static_cast<SnnPESubComponent*>(ctx);
    if (!core) return;
    if (core->impl_) {
        core->impl_->reportApplyScatter(acc_updates, posts_touched, spikes_emitted,
                                        hwm_bytes, spill_records, spilled_bytes);
    }
}

void SnnPESubComponent::requestGasEndGatherThunk_(void* ctx, uint32_t superstep) {
    auto* core = static_cast<SnnPESubComponent*>(ctx);
    if (!core) return;
    if (superstep == 0) return;
    if (!core->stdmem_ep_ || !core->stdmem_ep_->available()) return;
    core->stdmem_ep_->sendGasCmd(GasOp::EndGather, superstep, /*slice*/0, /*tot*/1, /*flag*/false);
}

void SnnPESubComponent::requestGasEndScatterThunk_(void* ctx, uint32_t superstep) {
    auto* core = static_cast<SnnPESubComponent*>(ctx);
    if (!core) return;
    if (superstep == 0) return;
    if (!core->stdmem_ep_ || !core->stdmem_ep_->available()) return;
    core->stdmem_ep_->sendGasCmd(GasOp::EndScatter, superstep, /*slice*/0, /*tot*/1, /*flag*/false);
}

// === 静态共享路由缓存 / 阶段事件写入锁 ===
std::mutex SnnPESubComponent::s_stage_csv_mutex_;

// === Stage event hub (Phase5.2-A1): absorbed into Impl ===
void SnnPESubComponent::Impl::markBeginGather(uint32_t seq) {
    if (!core) return;
    if (gas_ctrl_) gas_ctrl_->onBeginGather(seq);
    if (core->use_bcsr_) {
        core->logBcsrWindowStats_("prev");
        core->resetBcsrWindowCounters_();
    }
    uint64_t now = core->getCurrentSimTimeNano();
    core->appendStageEventRow_("BeginGather", now, 0);
    t_begin_gather = now;
    have_begin_gather = true;
    have_begin_apply = false;
    have_begin_scatter = false;
    core->window_spikes_all_ = 0;
}

void SnnPESubComponent::Impl::markBeginApply(uint32_t seq) {
    if (!core) return;
    if (gas_ctrl_) gas_ctrl_->onBeginApply(seq);
    uint64_t now = core->getCurrentSimTimeNano();
    core->appendStageEventRow_("BeginApply", now, 0);
    t_begin_apply = now;
    have_begin_apply = true;
}

void SnnPESubComponent::Impl::markBeginScatter(uint32_t seq) {
    if (!core) return;
    if (gas_ctrl_) gas_ctrl_->onBeginScatter(seq);
    uint64_t now = core->getCurrentSimTimeNano();
    core->appendStageEventRow_("BeginScatter", now, 0);
    t_begin_scatter = now;
    have_begin_scatter = true;
}

void SnnPESubComponent::Impl::markEndScatter(uint32_t seq, uint64_t spikes_emitted) {
    if (!core) return;
    if (gas_ctrl_) gas_ctrl_->onEndScatter(seq, spikes_emitted);
    uint64_t now = core->getCurrentSimTimeNano();
    core->appendStageEventRow_("EndScatter", now, spikes_emitted);
    // 额外诊断：输出本窗口权重/BCSR 读流量摘要（仅 window_read_debug=1 时启用）
    if (core->window_read_debug_ && core->weight_mem_subsystem_) {
        core->weight_mem_subsystem_->endScatterWindow(seq);
    }
    reportWindowSpikes(static_cast<uint32_t>(seq), spikes_emitted);
    core->spikes_emitted_window_ = 0;
    core->window_spikes_all_ = 0;
    if (core->stat_gas_superstep_total_cycles_) {
        if (have_begin_gather) {
            uint64_t total = (now >= t_begin_gather) ? (now - t_begin_gather) : 0ULL;
            core->stat_gas_superstep_total_cycles_->addData(total);
        }
        if (have_begin_gather && have_begin_apply && core->stat_gas_superstep_gather_cycles_) {
            uint64_t g = (t_begin_apply >= t_begin_gather) ? (t_begin_apply - t_begin_gather) : 0ULL;
            core->stat_gas_superstep_gather_cycles_->addData(g);
        }
        if (have_begin_apply && core->stat_gas_superstep_apply_cycles_) {
            uint64_t a = (t_begin_scatter >= t_begin_apply) ? (t_begin_scatter - t_begin_apply) : 0ULL;
            core->stat_gas_superstep_apply_cycles_->addData(a);
        }
        if (have_begin_scatter && core->stat_gas_superstep_scatter_cycles_) {
            uint64_t s = (now >= t_begin_scatter) ? (now - t_begin_scatter) : 0ULL;
            core->stat_gas_superstep_scatter_cycles_->addData(s);
        }
    }
    have_begin_gather = have_begin_apply = have_begin_scatter = false;
}

void SnnPESubComponent::appendStageEventRow_(const char* event_name, uint64_t now_ns, uint64_t spikes_emitted) {
    // 阶段事件上报（转发给 MultiCorePE 聚合落盘）：
    // 注意：GAS 窗口阶段事件在多核之间并不严格同步，某些窗口的 BeginApply/BeginScatter
    // 可能首先出现在非 core0 上。若只允许 core0 上报，会导致 ga/bs 缺失，从而 gather/apply/scatter
    // 统计被写成 0（p95=0）。
    // 解决：所有核心都上报阶段边界事件，由 MultiCorePE::notifyStageEvent 做 min/max 聚合收敛。
    if (event_name == nullptr) return;
    std::lock_guard<std::mutex> lock(s_stage_csv_mutex_);
    // 改为通知父 PE 统一写入阶段事件（避免多核重复与多次落盘）；同时传递本窗发放数量
    if (auto* pe = parent_pe_cached_) {
        pe->notifyStageEvent(static_cast<uint32_t>(curr_stage_seq_), std::string(event_name), now_ns, spikes_emitted, core_id_);
    }
}

void SnnPESubComponent::handleStageEventWithoutApply_(const GasOpData* op) {
    if (!op) return;
    uint64_t now = getCurrentSimTimeNano();
    switch (op->op) {
        case GasOp::BeginGather:
            curr_stage_seq_ = op->superstep;
            appendStageEventRow_("BeginGather", now, 0);
            break;
        case GasOp::BeginApply:
            appendStageEventRow_("BeginApply", now, 0);
            break;
        case GasOp::EndApply:
            appendStageEventRow_("EndApply", now, 0);
            break;
        case GasOp::BeginScatter:
            appendStageEventRow_("BeginScatter", now, 0);
            break;
        case GasOp::EndScatter:
            appendStageEventRow_("EndScatter", now, 0);
            break;
        default:
            break;
    }
}

void SnnPESubComponent::prepareEdgeWindowForApply_() {
    if (!(apply_acc_enable_ && gas_window_mode_)) return;
    if (weight_mem_subsystem_) {
        weight_mem_subsystem_->beginApplyWindow(curr_stage_seq_, window_read_debug_, output_, core_id_);
    } else {
        windowStateBegin_();
    }
    orchestrateIssueFromEdgesDirect();
}

void SnnPESubComponent::diagEdgeWeight_(const char* tag, uint32_t post_local,
                                        uint32_t pre_global, float weight,
                                        uint32_t count) {
    if (!enable_extended_diagnostics_ && !window_read_debug_) return;
    if (!output_) return;
    output_->verbose(CALL_INFO, 1, 0,
        "[diag-weight] %s core=%d window=%u post_local=%u pre_global=%u weight=%.6f count=%u\n",
        tag ? tag : "edge", core_id_, curr_stage_seq_, post_local, pre_global,
        (double)weight, count);
}

void SnnPESubComponent::logBcsrWindowStats_(const char* tag) {
    if (!window_read_debug_ || !use_bcsr_ || !output_) return;
    if (bcsr_req_edges_ == 0 && bcsr_req_wait_rowptr_ == 0 &&
        bcsr_req_block_hit_ == 0 && bcsr_req_block_miss_ == 0) {
        return;
    }
    output_->verbose(CALL_INFO, 0, 0,
        "[diag-bcsr-window] core=%d window=%u tag=%s edges=%" PRIu64
        " rowptr_wait=%" PRIu64 " hits=%" PRIu64 " miss=%" PRIu64 "\n",
        core_id_, curr_stage_seq_, tag ? tag : "-",
        bcsr_req_edges_, bcsr_req_wait_rowptr_,
        bcsr_req_block_hit_, bcsr_req_block_miss_);
}

void SnnPESubComponent::resetBcsrWindowCounters_() {
    bcsr_req_edges_ = 0;
    bcsr_req_wait_rowptr_ = 0;
    bcsr_req_block_hit_ = 0;
    bcsr_req_block_miss_ = 0;
}

void SnnPESubComponent::reserveWindowContainers_() {
    if (weight_mem_subsystem_) {
        weight_mem_subsystem_->reserveWindowContainers(num_neurons_);
    }
}

bool SnnPESubComponent::ensureLoaderReady_() {
    if (!wait_for_loader_done_) return true;
    if (loader_ready_latched_) return true;
    if (!loader_done_shared_initialized_) return true;
    if (loader_done_shared_.size() == 0) return false;
    int ready = loader_done_shared_.mutex_read(0);
    if (ready != 0) {
        loader_ready_latched_ = true;
        if (window_read_debug_ && !loader_ready_logged_ && output_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-loader] core=%d weights_ready at cycle=%" PRIu64 "\n",
                core_id_, total_cycles_);
        }
        loader_ready_logged_ = true;
        return true;
    }
    return false;
}

bool SnnPESubComponent::ensureMemoryReady_() const {
    return stdmem_ep_ && stdmem_ep_->available() && memory_ready_;
}

void SnnPESubComponent::issueEdgeWeightFetches_() {
    const size_t prev_edges = weight_mem_subsystem_ ? weight_mem_subsystem_->edgesPrevSize() : 0;
    if (window_read_debug_ && output_) {
        output_->verbose(CALL_INFO, 0, 0,
            "[diag-edge-fetch] core=%d stage=%d prev_edges=%zu issued=%u outstanding=%u budget=%u\n",
            core_id_, static_cast<int>(gas_stage_), prev_edges,
            windowStateIssued_(), windowStateOutstanding_(), window_read_budget_);
    }
    issueFromEdges_();
}

void SnnPESubComponent::recordEdge_(uint32_t post_local, uint32_t pre_global) {
    if (!(enable_weight_fetch_ && ensureMemoryReady_())) {
        diag_edges_cond_skips_++;
        if (window_read_debug_ && !record_edge_cond_warned_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-edges] recordEdge skipped enable_weight_fetch=%d stdmem_ep=%d ready=%d\n",
                enable_weight_fetch_ ? 1 : 0,
                (stdmem_ep_ && stdmem_ep_->available()) ? 1 : 0,
                memory_ready_ ? 1 : 0);
            record_edge_cond_warned_ = true;
        }
        return;
    }
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
    if (!(apply_acc_enable_ && gas_window_mode_ && stage_ok)) {
        diag_edges_stage_skips_++;
        if (window_read_debug_ && !record_edge_stage_warned_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-edges] recordEdge skipped apply_acc=%d gas_window=%d stage=%d (apply_en=%d idle_en=%d scatter_en=%d)\n",
                apply_acc_enable_ ? 1 : 0, gas_window_mode_ ? 1 : 0, static_cast<int>(gas_stage_),
                record_edge_apply_enable_ ? 1 : 0, record_edge_idle_enable_ ? 1 : 0,
                record_edge_scatter_enable_ ? 1 : 0);
            record_edge_stage_warned_ = true;
        }
        return;
    }
    // 容量保护：极端情况下避免map无限增长
    const size_t curr_edges = weight_mem_subsystem_ ? weight_mem_subsystem_->edgesCurrSize() : 0;
    if (curr_edges >= edge_collector_max_capacity_) {
        if (!record_edge_capacity_warned_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-edges] ⚠️ edge_collector capacity reached (core=%d seq=%u cap=%zu), stop recording this window\n",
                core_id_, curr_stage_seq_, edge_collector_max_capacity_);
            record_edge_capacity_warned_ = true;
            if (stat_gas_edge_overflow_) stat_gas_edge_overflow_->addData(1);
        }
        return;
    }
    if (weight_mem_subsystem_) {
        weight_mem_subsystem_->recordEdge(post_local, pre_global);
    }
    diag_edges_record_hits_++;
    const size_t after_edges = weight_mem_subsystem_ ? weight_mem_subsystem_->edgesCurrSize() : 0;
    if (window_read_debug_ && after_edges <= 5) {
        output_->verbose(CALL_INFO, 0, 0,
            "[diag-edges] recordEdge sample core=%d seq=%u post_local=%u pre_global=%u size_now=%zu\n",
            core_id_, curr_stage_seq_, post_local, pre_global, after_edges);
    }
}

SnnPESubComponent::SnnPESubComponent(ComponentId_t id, Params& params)
    : SnnCoreAPI(id, params),
      output_(nullptr),
      memory_link_(nullptr) {
    // 构造期最早哨兵（P2：参数优先，未设置回退到环境变量，以保持兼容）。
    // 避免直接使用 stdout，统一走 SST Output。
    const bool kSentinelOn = [&params](){
        // P2 Step3：仅参数驱动，默认0=禁用
        int sent_p = params.find<int>("sentinel_enable", 0);
        return sent_p != 0;
    }();
    // 提前构建一个最低等级的输出对象，避免后续早期初始化路径使用 output_ 时发生空指针
    // 真实 verbose 等级稍后在解析完参数后再生效（此处仅用于早期诊断与防护）
    if (!output_) {
        try {
            output_ = new Output("SnnPESubComponent[@p:@l]: ", /*verbose*/0, 0, Output::STDOUT);
        } catch (...) {
            output_ = nullptr; // 最小化风险，保持后续分支都做空指针判定
        }
    }
    // 注意：其余子模块初始化挪到参数解析之后，避免早期未初始化成员被使用
    
    // 读取配置参数
    core_id_ = params.find<int>("core_id", 0);
    // Phase5.2：将实现类型改为指针持有，避免在 control 头文件中包含 synapse 实现头
    if (!stdmem_ep_) stdmem_ep_ = std::make_unique<StdMemEndpoint>();
    if (!bcsr_weights_) bcsr_weights_ = std::make_unique<BcsrWeightManager>();
    if (kSentinelOn && output_) {
        SNNDL_LOG(0, "[[sentinel-core-ctor]] core_ctor enter\n");
        SNNDL_LOG(0, "[[sentinel-core-ctor]] after params: core_id=%d\n", core_id_);
    }
    total_cores_ = params.find<int>("total_cores", 8);
    global_neuron_base_ = params.find<uint64_t>("global_neuron_base", 0);
    num_neurons_ = params.find<uint32_t>("num_neurons", 64);
    base_addr_ = params.find<uint64_t>("base_addr", 0);
    node_id_ = params.find<uint32_t>("node_id", 0);
    verbose_ = params.find<int>("verbose", 0);
    enable_extended_diagnostics_ = params.find<int>("enable_extended_diagnostics", 0) != 0;
    total_nodes_cfg_ = params.find<uint32_t>("total_nodes", 1);

    // Phase6：workload 选择（优先 Params，其次环境变量；默认 snn 保持回归口径不变）
    {
        std::string w = "snn";
        if (params.contains("workload_impl")) {
            w = params.find<std::string>("workload_impl", "snn");
        } else if (const char* env = workloadImplFromEnvCached()) {
            w = std::string(env);
        }
        std::transform(w.begin(), w.end(), w.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        if (w == "stream") workload_impl_ = WorkloadImpl::Stream;
        else if (w == "traffic") workload_impl_ = WorkloadImpl::Traffic;
        else workload_impl_ = WorkloadImpl::Snn;
    }
    // Phase6：通过工厂创建 workload（snn/stream/traffic）；未知值已在上方归一化为 SNN。
    {
        const char* name = isTrafficWorkload_() ? "traffic" : (isStreamWorkload_() ? "stream" : "snn");
        if (!workload_) workload_ = createWorkloadByName(std::string(name));
        if (!workload_) {
            if (output_) {
                output_->fatal(CALL_INFO, -1, "❌ createWorkloadByName(\"%s\") returned nullptr\n", name);
            }
            return;
        }
        workload_->configureFromParams(params);
        spike_workload_ = dynamic_cast<ISpikeWorkload*>(workload_.get());
        snn_comm_workload_ = dynamic_cast<ISnnSpikeCommWorkload*>(workload_.get());
        gas_stage_workload_ = dynamic_cast<IGasStageSink*>(workload_.get());
    }
    if (kSentinelOn && output_) {
        SNNDL_LOG(0, "[[sentinel-core-ctor]] after params2: node_id=%u num_neurons=%u base_addr=%" PRIu64 "\n",
                node_id_, num_neurons_, (uint64_t)base_addr_);
    }
    enable_weight_fetch_ = params.find<int>("enable_weight_fetch", 0) != 0;
    workload_spike_input_enable_ = params.find<int>("workload_spike_input_enable", 0) != 0;
    // Phase5.2-A1：StageEventHub 吸收到 Impl（不再单独编译 control/StageEventHub.*）
    if (!impl_) impl_ = std::make_unique<Impl>();
    impl_->init(this);
    impl_->gas_ctrl_ = std::make_unique<GasPhaseController>();
    if (impl_->gas_ctrl_) {
        impl_->gas_ctrl_->init(this, output_);
    }
#if SNNDL_DEBUG_ENABLED
    if (window_read_debug_) {
        SNNDL_DEBUG_LOG(1, "[diag-init] core=%d enable_weight_fetch=%d\n", core_id_, enable_weight_fetch_ ? 1 : 0);
    }
#endif
    write_weights_on_init_ = params.find<int>("write_weights_on_init", 1) != 0;
    memory_warmup_cycles_ = params.find<uint64_t>("memory_warmup_cycles", 1000);
    init_default_weight_ = params.find<float>("init_default_weight", 0.5f);
    readresp_zero_fallback_ = params.find<int>("readresp_zero_fallback", 0) != 0;
    max_outstanding_requests_ = params.find<uint32_t>("max_outstanding_requests", 16);
    const uint32_t max_cache_entries = params.find<uint32_t>("max_cache_entries", 65536);
    const bool use_clock_weight_cache = params.find<int>("use_clock_weight_cache", 0) != 0;
    const bool disable_weight_cache = params.find<int>("disable_weight_cache", 0) != 0;
    {
        WeightCacheOps::Config cache_cfg{};
        cache_cfg.max_entries = max_cache_entries;
        cache_cfg.use_clock = use_clock_weight_cache;
        cache_cfg.disable_cache = disable_weight_cache;
        if (!weight_cache_ops_) weight_cache_ops_ = std::make_unique<WeightCacheOps>();
        weight_cache_ops_->configure(cache_cfg, [this]() {
            if (stat_cache_evictions_) stat_cache_evictions_->addData(1);
            count_cache_evictions_++;
        });
        weight_cache_ops_->reserve(cache_cfg.max_entries ? cache_cfg.max_entries : 1);
    }
    use_event_weight_fallback_ = params.find<int>("use_event_weight_fallback", 0) != 0;
    event_weight_fallback_warned_ = false;
    merge_read_cacheline_ = params.find<int>("merge_read_cacheline", 1) != 0;
    merge_read_row_ = params.find<int>("merge_read_row", 0) != 0;
    gas_enable_ = params.find<int>("gas_enable", 0) != 0; // 默认关闭
    gas_window_mode_ = params.find<int>("gas_window_mode", 0) != 0; // 当为true时采用GatherBufferIF的window驱动
    // Deprecate manual window driving: read param for compatibility but force-disable
    bool manual_drive_param = params.find<int>("gas_manual_window_drive", 0) != 0;
    gas_manual_window_drive_ = false;
    if (manual_drive_param && output_) {
        output_->verbose(CALL_INFO, 0, 0,
            "[diag-gas-config] core=%d gas_manual_window_drive 已弃用，仍将使用自动窗口驱动\n",
            core_id_);
    }
    manual_gas_gather_cycles_cfg_ = params.find<uint64_t>("gas_window_cycles_gather", 200);
    if (manual_gas_gather_cycles_cfg_ == 0) manual_gas_gather_cycles_cfg_ = 1;
    // 方案1（slice顺序执行）
    scheme1_enable_ = params.find<int>("scheme1_enable", 0) != 0;
    scheme1_slices_ = params.find<uint32_t>("scheme1_slices", 8);
    scheme1_gather_cycles_cfg_ = params.find<uint64_t>("scheme1_gather_cycles", 100);
    scheme1_slice_gap_cycles_ = params.find<uint64_t>("scheme1_slice_gap_cycles", 0);
    scheme1_scatter_cycles_ = params.find<uint64_t>("scheme1_scatter_cycles", 1);
    scheme1_partition_mod_ = params.find<int>("scheme1_partition_mod", 0) != 0;
    merge_read_auto_ = params.find<int>("merge_read_auto", 0) != 0; // default off
    line_size_bytes_ = params.find<uint32_t>("line_size_bytes", 64);
    loader_done_key_ = params.find<std::string>("loader_done_key", "");
    wait_for_loader_done_ = !loader_done_key_.empty();
    if (wait_for_loader_done_) {
        loader_done_shared_.initialize(loader_done_key_, 1, 0);
        loader_done_shared_initialized_ = true;
        if (window_read_debug_ && output_) {
            output_->verbose(CALL_INFO, 1, 0,
                "[diag-loader] core=%d init loader_done_key=%s\n",
                core_id_, loader_done_key_.c_str());
        }
    }
    window_read_enable_ = params.find<int>("window_read_enable", 0) != 0;
    window_read_debug_ = params.find<int>("window_read_debug", 0) != 0;
    scatter_diag_limit_ = params.find<uint32_t>("scatter_diag_limit", 0);
    scatter_diag_count_ = 0;
    route_summary_enable_ = params.find<int>("route_summary_enable", 0) != 0;
    if (impl_ && impl_->gas_ctrl_) impl_->gas_ctrl_->setDebug(window_read_debug_, enable_extended_diagnostics_);
    window_read_budget_ = params.find<uint32_t>("window_read_budget", 1024);
    windowStateConfigure_();
    read_force_single_ = params.find<int>("read_force_single", 0) != 0;
    // 边集合容量上限（极端保护）
    edge_collector_max_capacity_ = static_cast<size_t>(params.find<uint64_t>("edge_collector_max_capacity", 1000000));
    if (window_read_enable_) {
        reserveWindowContainers_();
    if (!record_edge_idle_enable_ && !record_edge_scatter_enable_ && window_read_debug_ && output_ && enable_extended_diagnostics_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-record-edge] core=%d 仅在Gather阶段记录边 (Apply/Idle/Scatter=0)", core_id_);
        }
    } else if (window_read_debug_ && output_ && enable_extended_diagnostics_) {
        output_->verbose(CALL_INFO, 0, 0,
            "[diag-record-edge] core=%d window_read_enable=0 => 忽略 window_read_debug", core_id_);
    }
    // 全网读取扩展参数
    weights_cols_ = params.find<uint32_t>("weights_cols", 0);
    std::string index_mode_str = params.find<std::string>("index_mode", "pre_row_post_col");
    // 神经元状态布局参数已下沉到 compute core（控制层不再持有）
    verify_routing_weights_ = params.find<int>("verify_routing_weights", 0) != 0;
    use_post_row_pre_col_ = (index_mode_str == "post_row_pre_col");
    if (index_mode_str == "bcsr_post_row") {
        use_bcsr_ = true;
        use_post_row_pre_col_ = true;
    } else if (index_mode_str == "csr_post_row") {
        // CSR 模式已弃用：统一禁用
        use_post_row_pre_col_ = true;
        SNNDL_DEBUG_LOG(1, "[CSR] 索引模式已禁用，改用密集/BCSR读取\n");
    }
    if (weights_cols_ == 0) weights_cols_ = num_neurons_; // 默认沿用旧行宽
    // 配置权重索引解析器（独立于控制层实现）
    if (!weight_accessor_) weight_accessor_ = std::make_unique<WeightAccessor>();
    weight_accessor_->configure(WeightAccessorConfig{
        static_cast<uint32_t>(core_id_),
        static_cast<uint64_t>(global_neuron_base_),
        static_cast<uint32_t>(num_neurons_),
        static_cast<uint32_t>(weights_cols_),
        use_post_row_pre_col_
    });
    // 预计算dense权重区域上界（按行*列*4B）
    {
        uint64_t bytes = static_cast<uint64_t>(num_neurons_) * static_cast<uint64_t>(weights_cols_) * static_cast<uint64_t>(sizeof(float));
        weight_region_end_ = base_addr_ + bytes;
    }
    enable_detailed_map_log_ = params.find<int>("enable_detailed_map_log", 0) != 0;
    log_weight_details_ = params.find<int>("log_weight_details", 0) != 0;
    loader_barrier_cycles_ = params.find<uint64_t>("loader_barrier_cycles", 0);
    // BCSR 参数
    bcsr_layout_.rows = num_neurons_;
    bcsr_layout_.cols = params.find<uint32_t>("weights_cols", num_neurons_);
    bcsr_layout_.block_rows = params.find<uint32_t>("bcsr_block_rows", 16);
    bcsr_layout_.block_cols = params.find<uint32_t>("bcsr_block_cols", 16);
    bcsr_layout_.idx_bytes = params.find<uint32_t>("bcsr_idx_bytes", 2);
    bcsr_layout_.val_bytes = params.find<uint32_t>("bcsr_val_bytes", 4);
    bcsr_layout_.rowptr_offset = params.find<uint64_t>("bcsr_rowptr_offset", 0);
    bcsr_layout_.colidx_offset = params.find<uint64_t>("bcsr_colidx_offset", 0);
    bcsr_layout_.blockdata_offset = params.find<uint64_t>("bcsr_blockdata_offset", 0);
    bcsr_layout_.blockids_offset = params.find<uint64_t>("bcsr_blockids_offset", 0);
    bcsr_layout_.per_core_stride = params.find<uint64_t>("per_core_stride", 0);
    bcsr_layout_.validate(base_addr_, output_, (window_read_debug_ || enable_extended_diagnostics_), core_id_, node_id_);
    bcsr_br_ = bcsr_layout_.block_rows;
    bcsr_bc_ = bcsr_layout_.block_cols;
    bcsr_val_bytes_ = bcsr_layout_.val_bytes;
    bcsr_idx_bytes_ = bcsr_layout_.idx_bytes;
    uint64_t bcsr_rowptr_addr = base_addr_ + bcsr_layout_.rowptr_offset;
    bcsr_colidx_addr_ = base_addr_ + bcsr_layout_.colidx_offset;
    bcsr_blockdata_addr_ = base_addr_ + bcsr_layout_.blockdata_offset;
    bcsr_blockids_addr_ = bcsr_layout_.blockids_offset ? base_addr_ + bcsr_layout_.blockids_offset : 0;
    bcsr_weights_->configure(
        bcsr_rowptr_addr,
        bcsr_colidx_addr_,
        bcsr_blockdata_addr_,
        bcsr_blockids_addr_,
        bcsr_br_, bcsr_bc_, bcsr_idx_bytes_, bcsr_val_bytes_);
    // Revert: 默认值恢复为 64/256，避免影响发放路径；如需禁用由脚本显式传入 0
    bcsr_row_index_cache_cap_ = params.find<uint32_t>("bcsr_row_index_cache_cap", 64);
    bcsr_block_cache_cap_ = params.find<uint32_t>("bcsr_block_cache_cap", 256);
    // GAS Apply/Scatter Phase‑1
    apply_acc_enable_ = params.find<int>("apply_acc_enable", 0) != 0;
    acc_hwm_bytes_cfg_ = params.find<uint64_t>("acc_high_watermark_bytes", 16*1024*1024);
    acc_spill_enable_cfg_ = params.find<int>("acc_spill_enable", 1) != 0;
    stage_events_csv_ = params.find<std::string>("stage_events_csv", "");
    if (impl_ && impl_->gas_ctrl_) impl_->gas_ctrl_->setStageEventsCsv(stage_events_csv_);
    // aosoa_block_rows 默认推导已转移至 compute core
    // CSR 参数已移除
    bcsr_prefetch_all_ = params.find<int>("bcsr_prefetch_all", 0) != 0;
    // 权重验证开关（具体采样/文件配置由 compute core 解析并执行）
    verify_weights_ = params.find<int>("verify_weights", 0) != 0;
    bcsr_force_file_read_ = params.find<int>("bcsr_force_file_read", 0) != 0;
    bcsr_rowptr_file_fallback_enable_ = params.find<int>("bcsr_rowptr_file_fallback_enable", 0) != 0;
    quiet_finish_logs_ = params.find<int>("quiet_finish_logs", 0) != 0;
    const int record_apply_default = (gas_window_mode_ && apply_acc_enable_) ? 1 : 0;
    record_edge_apply_enable_ = params.find<int>("record_edge_apply_enable", record_apply_default) != 0;
    record_edge_idle_enable_ = params.find<int>("record_edge_idle_enable", 0) != 0;
    if (record_edge_idle_enable_ && output_ && enable_extended_diagnostics_) {
        output_->verbose(CALL_INFO, 0, 0,
            "[diag-gas-config] core=%d 已启用 record_edge_idle（诊断配置，回归默认关闭）\n",
            core_id_);
    }
    record_edge_scatter_enable_ = params.find<int>("record_edge_scatter_enable", 0) != 0;

    // ---- Profiling init (optional; minimal overhead when disabled) ----
#ifdef SNNDL_ENABLE_PROFILING
    profiler_enabled_ = params.find<int>("enable_profiler", 0) != 0;
    profiler_csv_prefix_ = params.find<std::string>("profiler_csv_prefix", "");
    if (profiler_enabled_) {
        try {
            profiler_ = new SST::SnnDL::Profiler(std::string("SnnPESubComponent_Core") + std::to_string(core_id_));
        } catch (...) {
            profiler_ = nullptr;
            profiler_enabled_ = false;
        }
    }
#endif
    // 默认启用致密累加器（与头文件参数表一致）
    const bool dense_acc_enable = params.find<int>("apply_dense_acc_enable", 1) != 0;
    // P3: 影子验证开关仅由参数决定（dense 模式才有意义）
    const bool shadow_verify_enable =
        dense_acc_enable && (params.find<int>("acc_shadow_verify_enable", 0) != 0);
    acc_dense_enable_cfg_ = dense_acc_enable;
    acc_shadow_verify_enable_cfg_ = shadow_verify_enable;
    // 构建窗口累加器（所有状态收敛到 AccumulatorOps）
    {
        AccumulatorOpsConfig acc_cfg{};
        acc_cfg.num_neurons = num_neurons_;
        acc_cfg.dense_enable = acc_dense_enable_cfg_;
        acc_cfg.spill_enable = acc_spill_enable_cfg_;
        acc_cfg.high_watermark_bytes = acc_hwm_bytes_cfg_;
        acc_cfg.shadow_verify_enable = acc_shadow_verify_enable_cfg_;
        acc_cfg.window_read_debug = window_read_debug_;
        acc_cfg.core_id = core_id_;
        acc_cfg.verbose = verbose_;
        acc_cfg.out = output_;
        acc_cfg.updates_count = &acc_updates_count_;
        acc_cfg.posts_touched_count = &acc_posts_touched_count_;
        acc_cfg.spill_records_count = &acc_spill_records_count_;
        acc_cfg.spilled_bytes_sum = &acc_spilled_bytes_sum_;
        acc_cfg.hwm_bytes_max = &acc_hwm_bytes_max_;
        // Stats pointers will be attached in initializeStatistics().
        acc_ops_ = std::make_unique<AccumulatorOps>(acc_cfg);
    }

    // Weights template is used by both synapse/weights (BCSR) and synapse/route; keep cached here.
    weights_template_ = params.find<std::string>("weights_template", "");

    // neurons_per_pe：默认按 total_cores*num_neurons 推导；但允许脚本显式覆盖（保持兼容）
    uint32_t np_from_params = params.find<uint32_t>("neurons_per_pe", 0);
    uint32_t computed_neurons_per_pe = static_cast<uint32_t>(total_cores_) * static_cast<uint32_t>(num_neurons_);
    if (np_from_params > 0) {
        neurons_per_pe_cfg_ = np_from_params;
        if (np_from_params != computed_neurons_per_pe && output_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-gas-config] core=%d neurons_per_pe=%u (脚本指定) ≠ cores*rows=%u\n",
                core_id_, np_from_params, computed_neurons_per_pe);
        }
    } else {
        neurons_per_pe_cfg_ = computed_neurons_per_pe;
    }

    // 获取权重文件路径（由 WeightLoader 负责加载；控制层仅缓存以便日志/兼容）
    weights_file_path_ = params.find<std::string>("weights_file", "");

    if (!isStreamWorkload_()) {
        // Phase4 Task6.1：compute core 创建/配置下沉到 workload=snn；
        // 控制层此处仅构建 synapse/weights 的 weight reader 子系统。
        configureWeightReaderSubsystem_(params);

    // 初始化输出对象（若前面已创建则不重复）
    if (!output_) {
        output_ = new Output("SnnPESubComponent[@p:@l]: ", verbose_, 0, Output::STDOUT);
    }
    if (window_read_debug_) {
        output_->verbose(CALL_INFO, 0, 0,
            "[diag-bcsr-base] node=%u core=%d base=0x%llx rowptr=0x%llx colidx=0x%llx blockdata=0x%llx blockids=0x%llx\n",
            node_id_, core_id_,
            (unsigned long long)base_addr_,
            (unsigned long long)bcsr_weights_->rowptrAddr(),
            (unsigned long long)bcsr_colidx_addr_,
            (unsigned long long)bcsr_blockdata_addr_,
            (unsigned long long)bcsr_blockids_addr_);
    }
    if (use_bcsr_) {
        if (weights_template_.empty() && window_read_debug_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-bcsr] node=%u core=%d warning: weights_template empty while BCSR enabled\n",
                node_id_, core_id_);
        } else if (bcsr_rowptr_file_fallback_enable_ && !weights_template_.empty() &&
                   !bcsr_weights_->isRowptrReady() && loadBcsrRowptrFromFile_()) {
            if (window_read_debug_) {
                output_->verbose(CALL_INFO, 0, 0,
                    "[diag-bcsr] core=%u preload rowptr entries=%zu first=%u second=%u\n",
                    core_id_, bcsr_weights_->rowptrHost().size(),
                    bcsr_weights_->rowptrHost().empty()?0u:bcsr_weights_->rowptrHost()[0],
                    bcsr_weights_->rowptrHost().size()>1?bcsr_weights_->rowptrHost()[1]:0u);
            }
        }
    }

    // 输出权重验证开关以便调试（采样/文件参数由 compute core 负责）
    if (output_) {
        output_->verbose(CALL_INFO, 3, 0,
            "🔍 权重验证配置: verify_weights=%d (details in compute core)\n",
            verify_weights_ ? 1 : 0);
    }
    }

    // output_->verbose(CALL_INFO, 1, 0, "🔧 初始化SnnPE SubComponent (核心%d, %u个神经元)\n", 
    //                 core_id_, num_neurons_);
    
    // 神经元状态由 compute core 维护，控制层不再初始化本地副本
    // 去重发放统计位图（默认全0）
    fired_ever_.assign(num_neurons_, 0);
    
    // 初始化内存访问
    memory_link_ = nullptr;

    // 学习窗口状态由 compute core 维护


    
    // 初始化统计变量
    total_cycles_ = 0;
    active_cycles_ = 0;
    boot_read_sent_ = false;
    boot_write_sent_ = false;
    weights_initialized_ = false;
    memory_ready_ = false;
    stat_spikes_received_ = nullptr;
    stat_spikes_generated_ = nullptr;
    stat_neurons_fired_ = nullptr;
    stat_memory_requests_ = nullptr;
    stat_weight_cache_hits_ = nullptr;
    stat_weight_cache_misses_ = nullptr;
    stat_merged_reads_rows_ = nullptr;
    stat_merged_reads_cls_ = nullptr;
    stat_weights_verify_count_ = nullptr;
    stat_weights_mismatch_count_ = nullptr;
    stat_weights_verify_sum_ = nullptr;
    
    // 初始化内部计数器
    count_spikes_received_ = 0;
    count_spikes_generated_ = 0;
    count_neurons_fired_ = 0;
    count_memory_requests_ = 0;
    count_non_spike_packets_received_ = 0;
    count_stream_mem_verify_pass_ = 0;
    count_stream_mem_verify_fail_ = 0;
    count_stream_pkt_sent_ = 0;
    count_stream_pkt_recv_ = 0;
    count_stream_pkt_bad_crc_ = 0;
    count_stream_pkt_bad_magic_ = 0;
    
    // 配置时钟
    std::string clock_freq = "1GHz";
    registerClock(clock_freq, new Clock::Handler2<SnnPESubComponent,&SnnPESubComponent::clockTick>(this));
    
    // 立即注册统计，避免在调用 getStatistics 前指针为空
    initializeStatistics();

    // output_->verbose(CALL_INFO, 2, 0, "✅ SnnPE SubComponent核心%d初始化完成\n", core_id_);
}

SnnPESubComponent::~SnnPESubComponent() {
    // output_->verbose(CALL_INFO, 1, 0, "🗑️ 销毁SnnPE SubComponent核心%d\n", core_id_);
    parent_pe_cached_ = nullptr;

    // 清理脉冲队列
    while (!incoming_spikes_.empty()) {
        delete incoming_spikes_.front();
        incoming_spikes_.pop();
    }
#ifdef SNNDL_ENABLE_PROFILING
    if (profiler_) { delete profiler_; profiler_ = nullptr; }
#endif
    // 避免析构次序竞态：不手动delete日志对象
    output_ = nullptr;
}

// === Activity f (per-window active axons ratio) ===
void SnnPESubComponent::activityFlush_() {
    if (!activity_stats_enable_) return;
    if (!parent_pe_cached_) return;
    if (weights_cols_ == 0) return;
    double f = (double)activity_pre_set_.size() / (double)weights_cols_;
    if (auto* pe = parent_pe_cached_) {
        pe->accumulateActivityF(f);
    }
    activityReset_();
}

size_t SnnPESubComponent::pendingMemSize_() const {
    size_t n = 0;
    if (stdmem_ep_ && stdmem_ep_->available()) {
        if (auto* mem = stdmem_ep_->memoryAccess()) n += mem->pendingSize();
    }
    return n;
}

void SnnPESubComponent::accReset_() {
    if (acc_ops_) acc_ops_->reset();
}

void SnnPESubComponent::accUpdate_(uint32_t post, float dv) {
    if (acc_ops_) acc_ops_->update(post, dv);
}

bool SnnPESubComponent::weightCacheTryGet_(uint64_t key, float& out) {
    if (!weight_cache_ops_) return false;
    return weight_cache_ops_->tryGet(key, out);
}

void SnnPESubComponent::weightCacheStore_(uint64_t key, float value) {
    if (!weight_cache_ops_) return;
    weight_cache_ops_->store(key, value);
}

void SnnPESubComponent::windowStateConfigure_() {
    if (weight_mem_subsystem_) {
        weight_mem_subsystem_->configureWindow(window_read_budget_, max_outstanding_requests_);
    }
}

void SnnPESubComponent::windowStateBegin_() {
    if (weight_mem_subsystem_) {
        weight_mem_subsystem_->beginWindow();
    }
}

bool SnnPESubComponent::windowStateCanIssue_(uint32_t n) const {
    return weight_mem_subsystem_ ? weight_mem_subsystem_->canIssue(n) : false;
}

void SnnPESubComponent::windowStateNoteIssue_(uint32_t n) {
    if (weight_mem_subsystem_) {
        weight_mem_subsystem_->noteIssue(n);
        if (impl_) impl_->updatePendingPeak(weight_mem_subsystem_->outstanding());
    }
}

void SnnPESubComponent::windowStateNoteComplete_(uint32_t n) {
    if (weight_mem_subsystem_) {
        weight_mem_subsystem_->noteComplete(n);
    }
}

uint32_t SnnPESubComponent::windowStateIssued_() const {
    return weight_mem_subsystem_ ? weight_mem_subsystem_->issued() : 0;
}

uint32_t SnnPESubComponent::windowStateOutstanding_() const {
    return weight_mem_subsystem_ ? weight_mem_subsystem_->outstanding() : 0;
}

void SnnPESubComponent::fillStreamRuntime_(ICoreWorkload::Runtime& rt) {
    rt.reporting.ctx = this;
    rt.reporting.report_mem_issue = &SnnPESubComponent::reportStreamMemIssueThunk_;
    rt.sinks.mem_verify_pass = &count_stream_mem_verify_pass_;
    rt.sinks.mem_verify_fail = &count_stream_mem_verify_fail_;
    rt.sinks.pkt_sent = &count_stream_pkt_sent_;
    rt.sinks.pkt_recv = &count_stream_pkt_recv_;
    rt.sinks.pkt_bad_crc = &count_stream_pkt_bad_crc_;
    rt.sinks.pkt_bad_magic = &count_stream_pkt_bad_magic_;
    rt.sinks.stat_mem_writes_issued_total = stat_stream_mem_writes_issued_total_;
    rt.sinks.stat_mem_reads_issued_total = stat_stream_mem_reads_issued_total_;
    rt.sinks.stat_mem_bytes_written_total = stat_stream_mem_bytes_written_total_;
    rt.sinks.stat_mem_bytes_read_total = stat_stream_mem_bytes_read_total_;
    rt.sinks.stat_mem_verify_pass_total = stat_stream_mem_verify_pass_total_;
    rt.sinks.stat_mem_verify_fail_total = stat_stream_mem_verify_fail_total_;
    rt.sinks.stat_pkt_sent_total = stat_stream_pkt_sent_total_;
    rt.sinks.stat_pkt_recv_total = stat_stream_pkt_recv_total_;
    rt.sinks.stat_pkt_bad_crc_total = stat_stream_pkt_bad_crc_total_;
    rt.sinks.stat_pkt_bad_magic_total = stat_stream_pkt_bad_magic_total_;
}

uint64_t SnnPESubComponent::workloadNowNsThunk_(void* ctx) {
    auto* self = static_cast<SnnPESubComponent*>(ctx);
    return self ? self->getCurrentSimTimeNano() : 0;
}

void SnnPESubComponent::bindWorkloadRuntime_() {
    if (!workload_) return;

    ICoreWorkload::Runtime rt{};
    rt.log = output_;
    rt.node_id = static_cast<uint32_t>(node_id_);
    rt.core_id = static_cast<uint32_t>(core_id_);
    rt.total_nodes = total_nodes_cfg_;
    rt.base_addr = base_addr_;
    rt.mem = (stdmem_ep_ && stdmem_ep_->available()) ? stdmem_ep_->memoryAccess() : nullptr;
    rt.noc = noc_transport_;
    rt.time.ctx = this;
    rt.time.now_ns = &SnnPESubComponent::workloadNowNsThunk_;
    rt.reporting.ctx = this;
    rt.reporting.report_mem_issue = &SnnPESubComponent::reportSnnMemIssueThunk_;
    rt.reporting.report_apply_scatter = &SnnPESubComponent::reportApplyScatterThunk_;
    rt.reporting.request_gas_end_gather = &SnnPESubComponent::requestGasEndGatherThunk_;
    rt.reporting.request_gas_end_scatter = &SnnPESubComponent::requestGasEndScatterThunk_;
    rt.sinks.spikes_received = &count_spikes_received_;
    rt.sinks.spikes_generated = &count_spikes_generated_;
    rt.sinks.neurons_fired = &count_neurons_fired_;
    rt.sinks.synaptic_accesses = &count_synaptic_accesses_;
    rt.sinks.window_spikes_all = &window_spikes_all_;
    rt.sinks.spikes_emitted_window = &spikes_emitted_window_;
    rt.sinks.stat_spikes_received_total = stat_spikes_received_;
    rt.sinks.stat_spikes_generated_total = stat_spikes_generated_;
    rt.sinks.stat_neurons_fired_total = stat_neurons_fired_;
    rt.sinks.stat_synaptic_accesses_total = stat_synaptic_accesses_;
    rt.sinks.stat_gas_scatter_spikes_emitted_total = stat_gas_scatter_spikes_emitted_total_;
    rt.sinks.stat_routes_entries_total = stat_routes_entries_;
    rt.sinks.stat_fanout_per_spike_total = stat_fanout_per_spike_;

    if (isStreamWorkload_()) {
        fillStreamRuntime_(rt);
    }

    workload_->bindRuntime(rt);
}

void SnnPESubComponent::setParentInterface(IPeAggregation* parent) {
    parent_pe_cached_ = parent;
    // output_->verbose(CALL_INFO, 2, 0, "🔗 核心%d设置父级接口\n", core_id_);
    bindWorkloadRuntime_();
}

void SnnPESubComponent::setNocTransport(INocTransport* noc) {
    noc_transport_ = noc;
    bindWorkloadRuntime_();
}

bool SnnPESubComponent::deliverPacket(NocPacketEvent* packet) {
    if (!packet) return true;
    // Phase7（B）：平台侧不再调用 deliverSpike；所有输入（含 Spike）统一以 packet 形式进入 workload。
    if (workload_) {
        return workload_->deliverPacket(packet);
    }

    if (enable_extended_diagnostics_ && output_) {
        output_->verbose(
            CALL_INFO, 1, 0,
            "[core-packet] core=%d kind=%u payload=%zu src=%u:%u dst=%u:%u\n",
            core_id_,
            static_cast<unsigned>(packet->kind),
            packet->payload.size(),
            packet->src_node,
            packet->src_endpoint,
            packet->dst_node,
            packet->dst_endpoint);
    }
    delete packet;
    return true;
}

void SnnPESubComponent::onGlobalStepStart(uint32_t seq) {
    if (isStreamWorkload_()) {
        // stream workload 不参与 SNN/GAS window 编排；
        // 但仍需要打开 memory(GatherBufferIF) 的 step gate，否则 StandardMemAccess 会拒绝请求（write/read 返回失败）。
        curr_stage_seq_ = seq;
        if (stdmem_ep_ && stdmem_ep_->available()) {
            auto* gate = stdmem_ep_->stepGate();
            if (gate) {
                gate->openStep(seq);
            }
        }
        return;
    }
    // 全局 Step 同步：打开 memory(GatherBufferIF) 的新窗口。
    // 注意：这里不直接操作 GAS 状态机，而是通过 IGasStepGate 保持边界清晰。
    if (!stdmem_ep_ || !stdmem_ep_->available()) {
        if (output_) output_->fatal(CALL_INFO, -1, "core=%d onGlobalStepStart(seq=%u) but stdmem endpoint is unavailable\n", core_id_, seq);
        return;
    }
    auto* gate = stdmem_ep_->stepGate();
    if (gate) {
        gate->openStep(seq);
        return;
    }

    // naive/non-window 模式：memory 可能是 memHierarchy.standardInterface（不实现 IGasStepGate）。
    // 仅在窗口化 GAS 模式下，缺失 step gate 才属于配置错误（fail-fast）。
    if (gas_window_mode_ && window_read_enable_) {
        if (output_) output_->fatal(
            CALL_INFO, -1,
            "core=%d onGlobalStepStart(seq=%u) requires memory to implement IGasStepGate (did you load GatherBufferIF with step_gate_enable=1?)\n",
            core_id_, seq);
        return;
    }
    curr_stage_seq_ = seq;
}

// === GAS stage/stat sink (Phase4-Task6.4) ===
void SnnPESubComponent::onGasStageEvent(const GasStageEvent& ev) {
    // CoreShell 保留最小镜像状态用于：
    // - StageEventHub 统计/CSV 口径（仍由 CoreShell 汇聚写出）
    // - activity f 等 PE 级聚合（兼容历史分析脚本）
    // 具体窗口读编排与 scatter 事务由 workload=snn 接管（通过转发完成）。

    curr_stage_seq_ = ev.superstep;

    // Mirror GAS stage enum (minimal).
    switch (ev.op) {
        case GasOp::BeginGather:
            gas_stage_ = GasStage::Gather;
            // Reset per-window diagnostics in CoreShell only (no side effects).
            record_edge_capacity_warned_ = false;
            diag_edges_record_hits_ = 0;
            diag_edges_stage_skips_ = 0;
            diag_edges_cond_skips_ = 0;
            diag_spikes_stage_gather_ = 0;
            diag_spikes_stage_apply_ = 0;
            diag_spikes_stage_scatter_ = 0;
            diag_spikes_stage_idle_ = 0;
            // Stage event bookkeeping (timing + window spikes reset) stays in CoreShell.
            if (impl_) impl_->markBeginGather(curr_stage_seq_);
            break;
        case GasOp::BeginApply:
            gas_stage_ = GasStage::Apply;
            if (impl_) impl_->markBeginApply(curr_stage_seq_);
            break;
        case GasOp::EndApply:
            gas_stage_ = GasStage::Apply; // remain until BeginScatter
            // EndApply is still recorded for stage CSV (legacy analysis scripts).
            appendStageEventRow_("EndApply", getCurrentSimTimeNano(), 0);
            break;
        case GasOp::BeginScatter:
            gas_stage_ = GasStage::Scatter;
            // BeginScatter timing stays in CoreShell; window spikes baseline used by EndScatter fallback.
            spikes_generated_base_ = count_spikes_generated_;
            if (impl_) impl_->markBeginScatter(curr_stage_seq_);
            break;
        case GasOp::EndScatter: {
            gas_stage_ = GasStage::Idle;
            uint64_t to_emit = window_spikes_all_ ? window_spikes_all_ : spikes_emitted_window_;
            if (to_emit == 0) {
                uint64_t delta = 0;
                if (count_spikes_generated_ >= spikes_generated_base_) {
                    delta = count_spikes_generated_ - spikes_generated_base_;
                }
                if (delta > 0) to_emit = delta;
            }
            if (impl_) impl_->markEndScatter(curr_stage_seq_, to_emit);
            break;
        }
        default:
            break;
    }

    // Activity-f window tracking: keep in CoreShell (PE-level aggregation).
    switch (ev.op) {
        case GasOp::BeginGather:
            activity_window_seq_ = ev.superstep;
            activityReset_();
            break;
        case GasOp::BeginApply:
        case GasOp::EndApply:
        case GasOp::BeginScatter:
            activityFlush_();
            break;
        default:
            break;
    }

    // Forward to workload (if it implements IGasStageSink).
    if (gas_stage_workload_) {
        gas_stage_workload_->onGasStageEvent(ev);
    }
}

void SnnPESubComponent::onGasStatEvent(const GasStatEvent& st) {
    // Accumulate at PE level for CSV visibility (keep identical to legacy behavior).
    if (auto* pe = parent_pe_cached_) {
        pe->accumulateGasStatsExt(st.unique_bytes, st.unique_reads,
                                  st.rowwin_triggers, st.rowwin_bytes,
                                  st.bursts, st.payload_bytes,
                                  st.window_inflight_peak, st.window_buffer_max_bytes);
    }
    // Local (per-core) copies for unique_* only (optional)
    if (stat_gas_unique_reads_total_ && st.unique_reads) stat_gas_unique_reads_total_->addData(st.unique_reads);
    if (stat_gas_unique_bytes_total_ && st.unique_bytes) stat_gas_unique_bytes_total_->addData(st.unique_bytes);

    // Optional forward (mostly no-op for workload=snn; kept for completeness).
    if (gas_stage_workload_) {
        gas_stage_workload_->onGasStatEvent(st);
    }
}

void SnnPESubComponent::configureWeightReaderSubsystem_(const Params& params) {
    // 构建权重读取子系统（Phase E：内存子系统闭环，控制层不再持有 pending/解析）
    if (!weight_reader_adapter_) {
        auto mem = std::make_unique<WeightMemorySubsystem>();
        mem->configure(
            [this](uint64_t key, float& out) { return weightCacheTryGet_(key, out); },
            [this](uint64_t key, float val) { weightCacheStore_(key, val); }
        );
        // Phase A/E：窗口读集合/预算/outstanding + 读发起/响应闭环下沉到子系统（保持行为与日志口径）
        {
            WeightMemorySubsystem::OrchestratorConfig ocfg{};
            ocfg.accessor = weight_accessor_.get();
            ocfg.cache_try = [this](uint64_t key, float& out) -> bool {
                return weightCacheTryGet_(key, out);
            };
            ocfg.cache_put = [this](uint64_t key, float v) {
                weightCacheStore_(key, v);
            };
            ocfg.acc_update = [this](uint32_t post_l, float dv) { accUpdate_(post_l, dv); };
            ocfg.diag_edge_weight = [this](const char* tag, uint32_t post_l, uint32_t pre_g, float w, uint32_t cnt) {
                diagEdgeWeight_(tag, post_l, pre_g, w, cnt);
            };
            ocfg.report_cache_access = [this](bool hit) { if (impl_) impl_->reportCacheAccess(hit); };
            ocfg.update_pending_peak = [this](uint32_t ostd) { if (impl_) impl_->updatePendingPeak(ostd); };
            ocfg.report_mem_issue = [this](size_t bytes, bool count_weight_read) {
                if (impl_) impl_->reportMemoryIssue(bytes, count_weight_read);
            };
            ocfg.report_mem_latency = [this](uint64_t lat_cycles, bool is_weight) {
                accum_mem_latency_cycles_ += lat_cycles;
                count_mem_responses_++;
                if (auto* pe = parent_pe_cached_) {
                    pe->accumulateMemReadLatency(lat_cycles, is_weight);
                }
            };
            ocfg.on_scheme1_prefetch_resp = [this]() {
                if (scheme1_pending_prefetch_ > 0) scheme1_pending_prefetch_--;
            };
            ocfg.ensure_loader_ready = [this]() { return ensureLoaderReady_(); };
            ocfg.bcsr_rowptr_ready = [this]() { return !use_bcsr_ || bcsr_weights_->isRowptrReady(); };
            ocfg.ensure_rowptr_ready_or_fatal = [this](const char* reason) { ensureRowptrReadyOrFatal_(reason); };
            ocfg.resume_issue_after_rowptr_ready = [this]() {
                if (apply_acc_enable_ && gas_window_mode_ && gas_stage_ == GasStage::Apply) {
                    issueEdgeWeightFetches_();
                }
            };
            ocfg.read_bcsr_from_file = [this](uint32_t post_l, uint32_t pre_g) { return readBcsrWeightFromFile_(post_l, pre_g); };
            ocfg.use_bcsr = use_bcsr_;
            ocfg.bcsr_force_file_read = bcsr_force_file_read_;
            ocfg.bcsr_prefetch_all = bcsr_prefetch_all_;
            ocfg.bcsr_weight_guard_enable = bcsr_weight_guard_enable_;
            ocfg.bcsr_weight_abs_max = bcsr_weight_abs_max_;
            ocfg.readresp_zero_fallback = readresp_zero_fallback_;
            ocfg.init_default_weight = init_default_weight_;
            ocfg.num_neurons = num_neurons_;
            ocfg.weights_cols = weights_cols_;
            ocfg.use_post_row_pre_col = use_post_row_pre_col_;
            ocfg.base_addr = static_cast<uint64_t>(base_addr_);
            ocfg.weight_region_end = weight_region_end_;
            ocfg.read_force_single = read_force_single_;
            ocfg.merge_read_cacheline = merge_read_cacheline_;
            ocfg.merge_read_row = merge_read_row_;
            ocfg.merge_read_auto = merge_read_auto_;
            ocfg.line_size_bytes = line_size_bytes_;
            ocfg.memory_warmup_cycles = memory_warmup_cycles_;
            ocfg.loader_barrier_cycles = loader_barrier_cycles_;
            ocfg.node_id = node_id_;
            ocfg.core_id = static_cast<uint32_t>(core_id_);
            ocfg.weights_template = weights_template_;
            ocfg.bcsr_mgr = bcsr_weights_.get();
            mem->configureOrchestrator(std::move(ocfg));
        }
        weight_mem_subsystem_ = mem.get();
        // Phase E：BCSR 缓存容量配置下沉到 BcsrWeightManager
        bcsr_weights_->setRowIndexCacheCapacity(bcsr_row_index_cache_cap_);
        bcsr_weights_->setBlockCacheCapacity(bcsr_block_cache_cap_);
        windowStateConfigure_();
        if (window_read_enable_) reserveWindowContainers_();
        weight_reader_adapter_ = std::move(mem);
    }
}

void SnnPESubComponent::init(unsigned int phase) {
    // 提前构建输出对象，避免在init早期使用output_时空指针
    if (!output_) {
        output_ = new Output("SnnPESubComponent[@p:@l]: ", verbose_, 0, Output::STDOUT);
    }
    // output_->verbose(CALL_INFO, 1, 0, "🔄 核心%d init phase %u\n", core_id_, phase);
    
    if (phase == 0) {
        // 初始化统计收集
        initializeStatistics();
        
        // 配置内存端口（可选，但不覆盖已设置的链接）
        if (!memory_link_) {
            memory_link_ = configureLink("mem_link");
            if (memory_link_) output_->verbose(CALL_INFO, 2, 0, "🔗 核心%d配置mem_link\n", core_id_);
        }
        
        initStdMemPhase0_();

        // Phase6/Phase4：workload runtime 绑定（平台：内存/NoC/parent；stream 额外绑定统计 sinks）。
        // Phase4 Task6.3：route/comm 装配已迁入 workload=snn，CoreShell 不再负责通信子系统装配。
        bindWorkloadRuntime_();

        // 权重验证所需的文件加载已下沉到 compute core（DefaultSnnComputeCore::initVerifyFile_）
    }

    // 将 init 相位转发给 StandardMem（通过 stdmem 端点转发）
    stdmem_ep_->init(phase);

    // Phase4 Task6.1：compute core init 下沉到 workload=snn（通过 onInitPhase 转发）。

    // Default weight initialization disabled, relying on WeightLoader
    if (phase == 4) {
        // 所有init阶段结束，允许后续时钟中发起访问
        memory_ready_ = true;
    }

    // Phase4：将生命周期相位转发给 workload（CoreShell 统一出口）。
    if (workload_) workload_->onInitPhase(phase);
}

void SnnPESubComponent::complete(unsigned int phase) {
    // 转发 complete 给 StandardMem：这是 memHierarchy init 握手的必要阶段（尤其当下游不是 Cache 而是 Bus/Dir）。
    stdmem_ep_->complete(phase);
}

void SnnPESubComponent::setup() {
    // output_->verbose(CALL_INFO, 1, 0, "🔧 核心%d setup 进入\n", core_id_);
    // output_->verbose(CALL_INFO, 1, 0,
    //     "🧩 参数: init_default_weight=%.3f, fallback=%d, merge_row=%d, merge_cl=%d, line=%uB, base_addr=%" PRIu64 ", N=%u\n",
    //     init_default_weight_, use_event_weight_fallback_, merge_read_row_, merge_read_cacheline_, line_size_bytes_, base_addr_, num_neurons_);
    
    // 验证组件状态
    if (!parent_pe_cached_) {
        output_->fatal(CALL_INFO, -1, "❌ 错误: 核心%d没有父级接口\n", core_id_);
    }
    // 注意：此处不直接发起内存访问，避免在setup阶段 MemLink 尚未建立时触发 memHierarchy fatal
    if (!stdmem_ep_ || !stdmem_ep_->available()) {
        // output_->verbose(CALL_INFO, 1, 0, "⚠️ 核心%d未配置StandardMem，检查是否有直接权重文件\n", core_id_);
        
        // 权重将由WeightLoader组件通过内存接口加载
        if (!weights_file_path_.empty()) {
            // output_->verbose(CALL_INFO, 1, 0, "🔧 核心%d权重文件路径: %s (将由WeightLoader加载)\n", core_id_, weights_file_path_.c_str());
        } else {
            output_->verbose(CALL_INFO, 1, 0, "⚠️ 核心%d未配置权重文件，将使用默认权重\n", core_id_);
        }
    }
    // 确保后端已构建（init阶段可能未加载到 StandardMem）

    // 打印映射模式与GAS端到端配置（一次性调试信息）
    {
        const char* idx_name = use_bcsr_ ? "bcsr_post_row" : (use_post_row_pre_col_ ? "post_row_pre_col" : "pre_row_post_col");
        int diag_lvl = window_read_debug_ ? 0 : 1;
        output_->verbose(CALL_INFO, diag_lvl, 0,
            "[GAS-Debug] core=%d index_mode=%s use_post_row_pre_col=%d apply_acc_enable=%d gas_enable=%d gas_window_mode=%d\n",
            core_id_, idx_name, use_post_row_pre_col_ ? 1 : 0, apply_acc_enable_ ? 1 : 0, gas_enable_ ? 1 : 0, gas_window_mode_ ? 1 : 0);
        output_->verbose(CALL_INFO, diag_lvl, 0,
            "[Init] core=%d global_base=%" PRIu64 " num_neurons=%u weights_cols=%u\n",
            core_id_, (uint64_t)global_neuron_base_, num_neurons_, weights_cols_);
    }
    // 配置一致性：启用窗口端到端语义时要求 window 模式的 GAS
    if (isNonSnnWorkload_()) {
        // Non-SNN workloads (stream/traffic) do not depend on GAS/Apply/Scatter.
        if (workload_) workload_->onSetup();
        return;
    }
    if (apply_acc_enable_ && (!gas_enable_ || !gas_window_mode_)) {
        output_->fatal(CALL_INFO, -1, "❌ 配置错误：apply_acc_enable=1 需要 GAS 启用且 gas_window_mode=1 (window_auto)。\n");
    }
    // Phase4 Task6.1：compute core setup 下沉到 workload=snn。
    if (workload_) workload_->onSetup();
    // output_->verbose(CALL_INFO, 1, 0, "✅ SnnPE SubComponent核心%d setup完成\n", core_id_);
}

void SnnPESubComponent::finish() {
    // 统计聚合（保持原路径）
    if (stat_pending_reqs_peak_) stat_pending_reqs_peak_->addData(pending_reqs_peak_);
    double avg_lat = (count_mem_responses_ > 0) ? ((double)accum_mem_latency_cycles_ / (double)count_mem_responses_) : 0.0;
    double utilization = (total_cycles_ > 0) ? (double)active_cycles_ / (double)total_cycles_ : 0.0;
    if (!quiet_finish_logs_) {
        // 输出统计信息（使用内部计数器获得正确值）
        output_->verbose(CALL_INFO, 1, 0, "📊 核心%d统计: 接收脉冲=%" PRIu64 ", 生成脉冲=%" PRIu64 ", 神经元发放=%" PRIu64 "\n",
                        core_id_, count_spikes_received_, count_spikes_generated_, count_neurons_fired_);
        if (verify_weights_) {
            uint64_t completed = 0;
            uint64_t mismatch = 0;
            if (compute_core_) {
                std::map<std::string, uint64_t> core_stats;
                compute_core_->getStatistics(core_stats);
                if (core_stats.count("core_verify_completed")) completed = core_stats["core_verify_completed"];
                if (core_stats.count("core_verify_mismatch_count")) mismatch = core_stats["core_verify_mismatch_count"];
            }
            output_->verbose(CALL_INFO, 1, 0, "🔍 权重验证: 完成=%" PRIu64 ", 不匹配=%" PRIu64 "\n",
                             completed, mismatch);
        }
        output_->verbose(CALL_INFO, 0, 0,
            "📈 核心%d性能摘要: total_cycles=%" PRIu64 ", active_cycles=%" PRIu64 ", utilization=%.4f, memory_req=%" PRIu64 ", cache_hit=%" PRIu64 ", cache_miss=%" PRIu64 ", pending_peak=%u, avg_mem_lat=%.2f\n",
            core_id_, total_cycles_, active_cycles_, utilization,
            count_memory_requests_, count_cache_hits_, count_cache_misses_, pending_reqs_peak_, avg_lat);
    }

#ifdef SNNDL_ENABLE_PROFILING
    if (profiler_enabled_ && profiler_) {
        // 控制台摘要
        profiler_->generate_report(std::cout, 3.0);
        // CSV 导出：prefix 优先；否则回退到工程级 analysis
        std::string csv = profiler_csv_prefix_.empty() ? std::string("analysis/profile_core") : profiler_csv_prefix_;
        csv += std::string("_c") + std::to_string(core_id_) + std::string(".csv");
        profiler_->export_csv(csv, 3.0);
    }
#endif
    // Phase4 Task6.1：compute core finish 下沉到 workload=snn。
    // 调试：若目标 core 在 Apply 阶段卡住导致窗口未完成，收尾时仍输出窗口级权重读摘要，便于定位瓶颈。
    if (window_read_debug_ && weight_mem_subsystem_) {
        weight_mem_subsystem_->finishWindowDiag();
    }
    if (workload_) workload_->onFinish();
}

bool SnnPESubComponent::clockTick(Cycle_t current_cycle) {
    (void)current_cycle; // 统一使用内部 cycle 计数，避免不同 SST 调度口径导致漂移
    total_cycles_++;
    if (!workload_) return false;
    const bool did = workload_->onClockTick(static_cast<uint64_t>(total_cycles_));
    // Phase10: active_cycles 由 workload 的返回值定义（SNN/stream 一致）。
    if (did) active_cycles_++;
    return false;
}

bool SnnPESubComponent::legacyClockTickInternal_(Cycle_t current_cycle) {
    (void)current_cycle;
    // Phase4-Task6.1：compute core 的 per-tick 驱动已下沉到 workload=snn（SnnWorkload）。
    bool has_activity = false;
    if (!clock_tick_logged_ && window_read_debug_) {
        output_->verbose(CALL_INFO, 0, 0,
            "[diag-gas] 核心%d clockTick start stage=%d gas_enable=%d window_mode=%d manual_drive=%d\n",
            core_id_, (int)gas_stage_, gas_enable_ ? 1 : 0, gas_window_mode_ ? 1 : 0,
            gas_manual_window_drive_ ? 1 : 0);
        clock_tick_logged_ = true;
    }
    // 不在时钟顶层阻断loader，允许deliver/recordEdge提前进行；
    // 在发起权重读取时（issueFromEdges_）再检查 loader 是否就绪。
    // 方案1：优先处理 slice 顺序执行路径；若启用则该函数接管整个周期流程
    if (scheme1_enable_) {
        if (scheme1Tick_()) return false; // 已完成本周期
    }
    // GAS: mark gather window start for this cycle (barrier-based)
    if (gas_enable_ && !gas_window_mode_ && ensureMemoryReady_()) {
        stdmem_ep_->sendGasCmd(GasOp::BeginGather, /*ss*/0, /*slice*/0, /*tot*/1);
    }
    if (gas_enable_ && gas_window_mode_ && gas_manual_window_drive_) {
        auto* drive = (stdmem_ep_ && stdmem_ep_->available()) ? stdmem_ep_->manualWindowDrive() : nullptr;
        if (!drive) {
            if (window_read_debug_ && output_ && !manual_window_tick_logged_) {
                output_->verbose(CALL_INFO, 0, 0,
                    "[diag-gas] core=%d manual window drive requested but unavailable (IManualWindowDrive not provided by stdmem)\n",
                    core_id_);
                manual_window_tick_logged_ = true;
            }
        } else {
            drive->manualWindowTick();
            manual_gas_counter_++;
            if (!manual_tick_sampled_ && manual_gas_counter_ <= manual_gas_gather_cycles_cfg_) {
                output_->verbose(CALL_INFO, 0, 0,
                    "[diag-gas] 核心%d manual_tick stage=%d counter=%" PRIu64 " threshold=%" PRIu64 "\n",
                    core_id_, (int)gas_stage_, manual_gas_counter_, manual_gas_gather_cycles_cfg_);
                if (manual_gas_counter_ >= manual_gas_gather_cycles_cfg_) manual_tick_sampled_ = true;
            }
            if (manual_gas_counter_ >= manual_gas_gather_cycles_cfg_) {
                stdmem_ep_->sendGasCmd(GasOp::EndGather, /*ss*/0, /*slice*/0, /*tot*/1);
                output_->verbose(CALL_INFO, 0, 0,
                    "[diag-gas] 核心%d手动发出 EndGather (stage=%d cnt=%" PRIu64 ")\n",
                    core_id_, (int)gas_stage_, manual_gas_counter_);
                manual_gas_counter_ = 0;
                manual_tick_sampled_ = false;
            }
        }
    }
    // 学习窗口边界由 compute core 驱动
    
    // 调试权重验证状态 (仅在前几个周期输出)
    /*
    if (verify_weights_ && total_cycles_ < 10) {
        output_->verbose(CALL_INFO, 2, 0, "🔍 核心%d状态检查: verify_weights=%d, memory_link=%s, memory_ready=%d, cycles=%lu, warmup=%lu\n",
                        core_id_, verify_weights_ ? 1 : 0, memory_link_ ? "yes" : "no", memory_ready_ ? 1 : 0, 
                        total_cycles_, memory_warmup_cycles_);
    }
    */
    
    // 处理输入脉冲队列：为避免与同周期 GAS 阶段切换事件产生“先后次序竞态”，
    // 仅处理时间戳严格早于当前仿真时间的脉冲；同周期到达的脉冲延后到下一tick处理。
    //
    // 同时，为消除 MPI 多 rank 下“同一时间戳事件到达顺序抖动”导致的非确定性，
    // 对本 tick 中可处理的 spike 做确定性排序（按 timestamp/dest/src/weight 位序）。
    if (workload_spike_input_enable_ &&
        apply_acc_enable_ && gas_window_mode_ && window_read_enable_ && !scheme1_enable_ &&
        !incoming_spikes_.empty()) {
        // Phase7: strict window-read spike input is owned by workload=snn.
        output_->fatal(CALL_INFO, -1,
            "core=%d legacy incoming_spikes_ is non-empty under strict window-read; "
            "this indicates an unintended fallback to legacy spike queueing.\n",
            core_id_);
    }
    const uint64_t now_ns = getCurrentSimTimeNano();
    std::vector<SpikeEvent*> ready_spikes;
    ready_spikes.reserve(std::min<size_t>(incoming_spikes_.size(), 256));
    while (!incoming_spikes_.empty()) {
        SpikeEvent* spike = incoming_spikes_.front();
        if (spike && spike->getTimestamp() >= now_ns) break;
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
        processLocalSpike(spike);
        has_activity = true;
        delete spike;
    }

    // Apply窗口内的机会式发起：若BeginApply时未能捕获到上一窗集合，
    // 但在本窗内由于deliverSpike处理使得 prev/curr 集合非空，则本窗内立即发起读取。
    // Apply阶段的窗口读发起应由 BeginApply 事件统一编排（issueFallbackReadsIfNeeded_/issueFromEdges），
    // 这里不再“机会式”发起集合读，避免占用 outstanding 影响按边读取完成度。
    
    // test-only 延迟读取示例已移除：权重通路由 compute core + 控制层窗口读编排负责
    // 权重验证与 BCSR 探针已下沉到 compute core（DefaultSnnComputeCore::onClockTick）
#if 0
    // 权重正确性验证：在暖机完成后进行固定次数采样读取，对比 expected_weight_value_
    if (verify_weights_ && memory_ && memory_ready_ && total_cycles_ >= memory_warmup_cycles_ &&
        (loader_barrier_cycles_ == 0 || total_cycles_ >= loader_barrier_cycles_)) {
        if (!verify_started_) {
            verify_started_ = true;
            // 降低默认日志级别
            output_->verbose(CALL_INFO, 3, 0, "🎯 核心%d权重验证启动: 周期=%lu, 暖机阈值=%lu\n", 
                            core_id_, total_cycles_, memory_warmup_cycles_);
        }
        // 每个周期发起至多一个样本，避免拥塞
        if (verify_completed_ < weight_verify_samples_ && verify_requested_ - verify_completed_ < max_outstanding_requests_) {
            uint32_t sample_idx = verify_requested_;
            // 采样若干 (row, col)
            uint32_t row;
            uint32_t col;
            if (verify_cluster_enable_) {
                // 将前weight_verify_samples_个样本聚类到同一cacheline：固定行0，列在一个cacheline范围内循环
                uint32_t fpl = std::max<uint32_t>(1, line_size_bytes_ / (uint32_t)sizeof(float));
                row = 0;
                col = (sample_idx % fpl);
                if (use_post_row_pre_col_) {
                    // 新模式 col 表示 pre_global；为了命中同一CL，选取一小段连续 pre_global
                    // 这里使用全局列 0..fpl-1，足够验证命中
                } else {
                    // 旧模式 col=post_local，同样聚到同一CL
                }
            } else {
                row = (sample_idx * 13) % num_neurons_;                // 本地目标行
                col = use_post_row_pre_col_ ? ((sample_idx * 7) % std::max<uint32_t>(1, weights_cols_))
                                             : ((sample_idx * 7) % num_neurons_);
            }
            // 新模式传参：(pre_global=col, post_local=row)；旧模式：(pre_local=row, post_local=col)
            uint32_t arg0 = use_post_row_pre_col_ ? col : row;
            uint32_t arg1 = use_post_row_pre_col_ ? row : col;
            requestWeight(arg0, arg1, [this, row, col](float w){
                verify_completed_++;
                verify_sum_ += static_cast<double>(w);
                bool mismatch = false;
                if (verify_against_file_ && verify_file_loaded_) {
                    // 使用文件中的期望值（row-major: row*weights_cols_ + col）
                    uint64_t idx = static_cast<uint64_t>(row) * static_cast<uint64_t>(weights_cols_) + static_cast<uint64_t>(col);
                    float expected = 0.0f;
                    if (idx < verify_file_buf_.size()) expected = verify_file_buf_[idx];
                    mismatch = (std::fabs(w - expected) > verify_epsilon_);
                    if (verify_log_each_sample_) {
                        output_->verbose(CALL_INFO, 1, 0,
                            "🔎 权重样本(FILE): row=%u col=%u value=%.6f expected=%.6f diff=%.6f %s\n",
                            row, col, w, expected, std::fabs(w-expected), (mismatch?"MISMATCH":"OK"));
                    }
                } else {
                    // 回退到常数期望（兼容旧行为）
                    mismatch = (std::fabs(w - expected_weight_value_) > verify_epsilon_);
                    if (verify_log_each_sample_) {
                        output_->verbose(CALL_INFO, 1, 0,
                            "🔎 权重样本(CONST): row=%u col=%u value=%.6f expected=%.6f diff=%.6f %s\n",
                            row, col, w, expected_weight_value_, std::fabs(w-expected_weight_value_), (mismatch?"MISMATCH":"OK"));
                    }
                }
                if (mismatch) verify_mismatch_count_++;
                // 详细调试权重读取值（禁用默认回调日志；仅当逐样本日志开启时输出）
                if (verify_log_each_sample_) {
                    output_->verbose(CALL_INFO, 2, 0,
                        "🔎 权重验证回调: core=%d row=%u col=%u value=%.6f sum=%.6f count=%u\n",
                        core_id_, row, col, w, verify_sum_, verify_completed_);
                }
                if (stat_weights_verify_count_) stat_weights_verify_count_->addData(1);
                if (verify_mismatch_count_ && stat_weights_mismatch_count_) stat_weights_mismatch_count_->addData(1);
                if (stat_weights_verify_sum_) stat_weights_verify_sum_->addData(verify_sum_);
            });
            verify_requested_++;
        }
    }

    // BCSR探针：尽力在同一窗口发起一次按边读，扫描一个块内的列，促成 1.0 样本出现（仅诊断；不影响GAS语义）
    if (verify_weights_ && use_bcsr_ && ensureMemoryReady_() && bcsr_weights_->isRowptrReady() &&
        total_cycles_ >= memory_warmup_cycles_ && (loader_barrier_cycles_ == 0 || total_cycles_ >= loader_barrier_cycles_)) {
        if (!verify_bcsr_done_ && !verify_bcsr_inflight_) {
            uint32_t br = (bcsr_br_>0? bcsr_br_:16);
            uint32_t bc = (bcsr_bc_>0? bcsr_bc_:16);
            uint32_t nBlockRows = (num_neurons_ + br - 1) / br;
            if (!verify_bcsr_started_) {
                const auto& rp = bcsr_weights_->rowptrHost();
                for (uint32_t r = 0; r < nBlockRows; ++r) {
                    if (r + 1 >= rp.size()) break;
                    uint32_t start = rp[r];
                    uint32_t end   = rp[r+1];
                    if (end > start) { verify_bcsr_post_local_ = r * br; verify_bcsr_block_col_ = 0; verify_bcsr_intra_col_ = 0; verify_bcsr_started_ = true; if (output_) output_->verbose(CALL_INFO, 1, 0, "[VERIFY][init-bcsr] core=%d post_local=%u rowptr=(%u,%u)\n", core_id_, verify_bcsr_post_local_, start, end); break; }
                }
                if (!verify_bcsr_started_) verify_bcsr_done_ = true;
            }
            if (verify_bcsr_started_ && !verify_bcsr_done_) {
                uint32_t r = verify_bcsr_post_local_ / br;
                // 若该行的colidx已缓存，解析出第一个block_col；否则先触发一次colidx读取
                if (!verify_bcsr_block_resolved_) {
                    std::vector<uint32_t> cols;
                    if (bcsrRowIndexGet_(r, cols) && !cols.empty()) {
                        verify_bcsr_block_col_ = cols[0];
                        verify_bcsr_block_resolved_ = true;
                        if (output_) output_->verbose(CALL_INFO, 1, 0, "[VERIFY][bcsr-colidx] core=%d row=%u first_block_col=%u\n", core_id_, r, verify_bcsr_block_col_);
                    } else {
                        // 文件直读一次，解析该行第一个块并且定位块内首个非零位置（诊断用途）
                        std::string bin_path = resolveWeightTemplate(node_id_, core_id_);
                        if (!bin_path.empty()) {
                            uint32_t rows=0, colsN=0, brM=0, bcM=0, idxB=0, valB=0, totalB=0;
                            uint64_t rp_off=0, ci_off=0, bd_off=0, id_off=0;
                            std::string meta_path = bin_path + ".meta.json";
                            if (parseBcsrMeta(meta_path, rows, colsN, brM, bcM, idxB, valB, rp_off, ci_off, bd_off, id_off, totalB)) {
                                std::ifstream fin(bin_path, std::ios::binary);
                                if (fin.good()) {
                                    const auto& rp = bcsr_weights_->rowptrHost();
                                    uint32_t start = (r+1 < rp.size() ? rp[r] : 0);
                                    uint32_t end   = (r+1 < rp.size() ? rp[r+1] : start);
                                    uint32_t brEff = (brM? brM : 1), bcEff = (bcM? bcM : 16);
                                    size_t blk_bytes = (size_t)brEff * (size_t)bcEff * (size_t)valB;
                                    uint32_t nblocks = (end > start ? (end - start) : 0);
                                    for (uint32_t j = 0; j < nblocks && !verify_bcsr_block_resolved_; ++j) {
                                        // 读第 j 个块列值
                                        fin.seekg(static_cast<std::streamoff>(ci_off + (size_t)(start + j) * idxB), std::ios::beg);
                                        uint32_t blk_col = 0; if (idxB == 2) { uint16_t v=0; fin.read(reinterpret_cast<char*>(&v), 2); blk_col = v; } else { fin.read(reinterpret_cast<char*>(&blk_col), 4); }
                                        if (!fin.good()) break;
                                        // 读该块数据
                                        fin.seekg(static_cast<std::streamoff>(bd_off + (uint64_t)(start + j) * blk_bytes), std::ios::beg);
                                        std::vector<float> blk(brEff*bcEff, 0.0f);
                                        if (valB == 4) fin.read(reinterpret_cast<char*>(blk.data()), blk_bytes);
                                        if (!fin.good()) break;
                                        for (uint32_t cc = 0; cc < bcEff; ++cc) {
                                            if (std::fabs(blk[cc]) > verify_epsilon_) { verify_bcsr_block_col_ = blk_col; verify_bcsr_intra_col_ = cc; verify_bcsr_block_resolved_ = true; break; }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                uint32_t intra = verify_bcsr_intra_col_;
                uint32_t pre_global = (verify_bcsr_block_resolved_ ? (verify_bcsr_block_col_ * bc + intra) : intra);
                uint32_t post_local = verify_bcsr_post_local_;
                verify_bcsr_inflight_ = true;
                uint32_t block_row = verify_bcsr_post_local_ / br;
                uint32_t bc_eff = (bcsr_bc_>0? bcsr_bc_:16);
                uint32_t blk_col = (bc_eff? (pre_global / bc_eff) : 0);
                size_t block_bytes = (size_t)(bcsr_br_>0?bcsr_br_:1) * (size_t)bc_eff * (size_t)bcsr_val_bytes_;
                uint32_t start = 0;
                uint64_t block_addr = 0;
                const auto& rp = bcsr_weights_->rowptrHost();
                if (block_row + 1 < rp.size()) {
                    start = rp[block_row];
                    block_addr = bcsr_blockdata_addr_ + (uint64_t)(start + verify_bcsr_block_col_) * block_bytes;
                }
                if (output_) {
                    output_->verbose(CALL_INFO, 0, 0,
                        "[VERIFY][mem-addr] core=%d post=%u pre=%u blk_row=%u blk_col=%u block_addr=0x%llx rowptr_start=%u block_bytes=%zu\n",
                        core_id_, post_local, pre_global, block_row, blk_col,
                        (unsigned long long)block_addr, start, block_bytes);
                }
                requestWeightBCSR(pre_global, post_local, [this, post_local, pre_global](float w){
                    if (output_) {
                        output_->verbose(CALL_INFO, 1, 0, "[VERIFY][probe-bcsr] post=%u pre=%u value=%.6f\n", post_local, pre_global, w);
                    }
                    verify_completed_++;
                    verify_sum_ += static_cast<double>(w);
                    // 仅诊断：读取文件中的同一位置，打印文件权重（不改变语义）
                    do {
                        std::string bin_path = resolveWeightTemplate(node_id_, core_id_);
                        if (bin_path.empty()) break;
                        uint32_t rows=0, colsN=0, brM=0, bcM=0, idxB=0, valB=0, totalB=0;
                        uint64_t rp_off=0, ci_off=0, bd_off=0, id_off=0;
                        std::string meta_path = bin_path + ".meta.json";
                        if (!parseBcsrMeta(meta_path, rows, colsN, brM, bcM, idxB, valB, rp_off, ci_off, bd_off, id_off, totalB)) break;
                        std::ifstream fin(bin_path, std::ios::binary);
                        if (!fin.good()) break;
                        uint32_t brEff = (brM? brM : 1), bcEff = (bcM? bcM : 16);
                        uint32_t r = (brEff? (post_local / brEff) : 0);
                        // 直接从文件读取 rowptr
                        fin.seekg(static_cast<std::streamoff>(rp_off + (size_t)r * sizeof(uint32_t)), std::ios::beg);
                        uint32_t start=0, end=0; fin.read(reinterpret_cast<char*>(&start), 4); fin.read(reinterpret_cast<char*>(&end), 4);
                        if (!fin.good() || end <= start) break;
                        uint32_t total_blocks = end - start;
                        if (total_blocks > 1'000'000) break; // 防止异常数据
                        uint32_t blk_col_target = (bcEff? (pre_global / bcEff) : 0);
                        int idx_in_row = -1;
                        for (uint32_t j=0; j < total_blocks; ++j) {
                            fin.seekg(static_cast<std::streamoff>(ci_off + (size_t)(start + j) * idxB), std::ios::beg);
                            uint32_t blk_col = 0; if (idxB == 2) { uint16_t v=0; fin.read(reinterpret_cast<char*>(&v), 2); blk_col = v; } else { fin.read(reinterpret_cast<char*>(&blk_col), 4); }
                            if (!fin.good()) break;
                            if (blk_col == blk_col_target) { idx_in_row = (int)j; break; }
                        }
                        if (idx_in_row < 0) break;
                        // 读该块数据并取 intra 列值
                        size_t blk_bytes = (size_t)brEff * (size_t)bcEff * (size_t)valB;
                        fin.seekg(static_cast<std::streamoff>(bd_off + (uint64_t)(start + (uint32_t)idx_in_row) * blk_bytes), std::ios::beg);
                        std::vector<float> blk(brEff*bcEff, 0.0f);
                        if (valB == 4) fin.read(reinterpret_cast<char*>(blk.data()), blk_bytes);
                        if (!fin.good()) break;
                        uint32_t intra = (bcEff? (pre_global % bcEff) : 0);
                        float fv = (intra < blk.size()? blk[intra] : 0.0f);
                        if (output_) output_->verbose(CALL_INFO, 1, 0, "[VERIFY][file-probe] post=%u pre=%u file_value=%.6f mem_value=%.6f\n", post_local, pre_global, fv, w);
                    } while(0);
                    if (std::fabs(w) > verify_epsilon_) {
                        verify_bcsr_done_ = true;
                    } else {
                        uint32_t bc = (bcsr_bc_>0? bcsr_bc_:16);
                        if (verify_bcsr_intra_col_ + 1 < bc) {
                            // 立即尝试下一列
                            verify_bcsr_intra_col_++;
                            uint32_t next_pre = (verify_bcsr_block_resolved_ ? (verify_bcsr_block_col_ * bc + verify_bcsr_intra_col_) : verify_bcsr_intra_col_);
                            requestWeightBCSR(next_pre, post_local, [this, post_local, next_pre](float w2){
                                if (output_) output_->verbose(CALL_INFO, 1, 0, "[VERIFY][probe-bcsr] post=%u pre=%u value=%.6f\n", post_local, next_pre, w2);
                                verify_completed_++;
                                verify_sum_ += static_cast<double>(w2);
                                if (std::fabs(w2) > verify_epsilon_) {
                                    verify_bcsr_done_ = true;
                                } else {
                                    // 若仍未命中，则留给下一tick继续
                                }
                                verify_bcsr_inflight_ = false;
                            });
                            return; // 由内层回调负责清理 inflight
                        } else {
                            verify_bcsr_done_ = true;
                        }
                    }
                    verify_bcsr_inflight_ = false;
                });
            }
        }
    }
#endif // legacy verify/probe path (moved to compute core)

    // Phase4-Task6.3：非 window 模式下的 “endCycle->drain->route/comm” 闭环已迁入 workload=snn（SnnWorkload::onClockTick）。
    // 控制层仅保留 GAS/window/阶段编排等 control-plane 逻辑，避免重复推进导致行为漂移。
    
    if (has_activity) {
        active_cycles_++;
    }
    // GAS: end of gather window for this cycle; GatherBufferIF will reply upon '读齐'
    if (gas_enable_ && !gas_window_mode_ && ensureMemoryReady_()) {
        stdmem_ep_->sendGasCmd(GasOp::EndGather, /*ss*/0, /*slice*/0, /*tot*/1);
    }

    return false;  // 继续时钟
}

bool SnnPESubComponent::legacySnnOnClockTick(uint64_t now_cycle) {
    return legacyClockTickInternal_(static_cast<Cycle_t>(now_cycle));
}

void SnnPESubComponent::legacySnnOnWeightsTick(uint64_t now_cycle) {
    if (weight_mem_subsystem_) weight_mem_subsystem_->onClockTick(now_cycle);
}

void SnnPESubComponent::legacySnnBindComputeCore(ISnnComputeCore* core) {
    compute_core_ = core;
}

IWeightReader* SnnPESubComponent::legacySnnGetWeightReader() {
    if (weight_reader_adapter_) return weight_reader_adapter_.get();
    // Phase4-Task6.2-Step2: weight reader ownership moved into workload; keep a non-owning view for legacy paths.
    return weight_mem_subsystem_;
}

std::unique_ptr<IWeightReader> SnnPESubComponent::legacySnnTakeWeightReader() {
    // Phase4-Task6.2-Step2: transfer ownership to workload=snn (called exactly once).
    return std::move(weight_reader_adapter_);
}

bool SnnPESubComponent::legacySnnWriteback(const std::unordered_map<uint64_t, float>& grads,
                                          float learning_rate,
                                          float weight_decay) {
    if (!ensureMemoryReady_()) {
        if (output_) {
            output_->verbose(CALL_INFO, 1, 0,
                "⚠️ 学习: 写回启用但内存接口不可用，跳过本窗写回\n");
        }
        return false;
    }
    return applyLocalWeightUpdates_(grads, learning_rate, weight_decay);
}

bool SnnPESubComponent::legacySnnHasWork() const {
    if (compute_core_ && compute_core_->hasWork()) return true;
    return !incoming_spikes_.empty();
}

double SnnPESubComponent::legacySnnGetUtilization() const {
    if (compute_core_) return compute_core_->getUtilization();
    if (total_cycles_ == 0) return 0.0;
    return static_cast<double>(active_cycles_) / static_cast<double>(total_cycles_);
}

void SnnPESubComponent::legacySnnGetStatistics(std::map<std::string, uint64_t>& stats) const {
    // 使用内部计数器而不是getCollectionCount()来获取正确的累计值
    if (compute_core_) {
        std::map<std::string, uint64_t> core_stats;
        compute_core_->getStatistics(core_stats);
        // 将核心计数口径映射到原有键，保持外部兼容
        if (core_stats.count("core_spikes_generated")) stats["spikes_generated"] = core_stats.at("core_spikes_generated");
        if (core_stats.count("core_neurons_fired")) stats["neurons_fired"] = core_stats.at("core_neurons_fired");
        if (core_stats.count("core_cycles_update_neuron")) stats["cycles_update_neuron"] = core_stats.at("core_cycles_update_neuron");
        if (core_stats.count("core_synaptic_accesses")) stats["synaptic_accesses"] = core_stats.at("core_synaptic_accesses");
        if (core_stats.count("core_total_cycles")) stats["total_cycles"] = core_stats.at("core_total_cycles");
        if (core_stats.count("core_active_cycles")) stats["active_cycles"] = core_stats.at("core_active_cycles");
        // 保留核心专有统计以便上层诊断
        stats.insert(core_stats.begin(), core_stats.end());
    }
    stats["spikes_received"] = count_spikes_received_;
    // 兼容旧路径：若核心未覆盖，则继续使用本地计数
    if (!stats.count("spikes_generated")) stats["spikes_generated"] = count_spikes_generated_;
    if (!stats.count("neurons_fired")) stats["neurons_fired"] = count_neurons_fired_;
    stats["memory_requests"] = count_memory_requests_;
    if (!stats.count("total_cycles")) stats["total_cycles"] = total_cycles_;
    if (!stats.count("active_cycles")) stats["active_cycles"] = active_cycles_;
    if (!stats.count("cycles_update_neuron")) stats["cycles_update_neuron"] = count_cycles_update_neuron_;
    if (!stats.count("synaptic_accesses")) stats["synaptic_accesses"] = count_synaptic_accesses_;
    if (!stats.count("core_weight_cache_hits") && stat_weight_cache_hits_) stats["core_weight_cache_hits"] = stat_weight_cache_hits_->getCollectionCount();
    if (!stats.count("core_weight_cache_misses") && stat_weight_cache_misses_) stats["core_weight_cache_misses"] = stat_weight_cache_misses_->getCollectionCount();
    if (!stats.count("core_pending_reqs_peak")) stats["core_pending_reqs_peak"] = pending_reqs_peak_;
    if (enable_extended_diagnostics_) {
        stats["core_non_spike_packets_received"] = count_non_spike_packets_received_;
    }
    // 注意：stream 专用统计由 StreamWorkload 填充；这里保持为纯 SNN legacy host。
}

void SnnPESubComponent::forceEndGather() {
    if (!(gas_enable_ && gas_window_mode_ && gas_manual_window_drive_ && ensureMemoryReady_())) return;
    stdmem_ep_->sendGasCmd(GasOp::EndGather, /*ss*/0, /*slice*/0, /*tot*/1);
    if (window_read_debug_ && output_) {
        output_->verbose(CALL_INFO, 0, 0, "[diag-gas] 核心%d 手动触发 EndGather\n", core_id_);
    }
}

void SnnPESubComponent::orchestrateBeginGatherWindowSetup() {
    onStageBeginGatherCore_(curr_stage_seq_);
    if (impl_) impl_->markBeginGather(curr_stage_seq_);
}

void SnnPESubComponent::orchestratePrepareApplyWindow() {
    prepareEdgeWindowForApply_();
}

void SnnPESubComponent::orchestrateApplyWindowEntry() {
    onStageBeginApplyCore_(curr_stage_seq_);
    if (impl_) impl_->markBeginApply(curr_stage_seq_);
}

void SnnPESubComponent::orchestrateBeginApplyIssueFallback(bool strict_active) {
    issueFallbackReadsIfNeeded_(strict_active);
}

void SnnPESubComponent::orchestrateContinueIssueReads() {
    issueFromEdges_();
}

void SnnPESubComponent::orchestrateIssueFromEdgesDirect() {
    issueEdgeWeightFetches_();
}

void SnnPESubComponent::orchestrateBeginScatterSequence() {
    diag_spikes_stage_apply_ = 0;
    onStageEndApplyCore_(curr_stage_seq_);
    onStageBeginScatterCore_(curr_stage_seq_);
    clearFiredWindowCore_();
    if (impl_) impl_->markBeginScatter(curr_stage_seq_);
    spikes_generated_base_ = count_spikes_generated_;
    uint64_t spikes_emitted = applyAccumulatedWindowAndScatter_();
    if (spikes_emitted > 0) {
        if (auto* pe = parent_pe_cached_) pe->accumulateApplyScatterStats(0, 0, spikes_emitted, 0, 0, 0);
    }
}

void SnnPESubComponent::orchestrateEndScatterSequence() {
    uint64_t to_emit = window_spikes_all_ ? window_spikes_all_ : spikes_emitted_window_;
    if (to_emit == 0) {
        uint64_t delta = 0;
        if (count_spikes_generated_ >= spikes_generated_base_) delta = count_spikes_generated_ - spikes_generated_base_;
        if (delta > 0) to_emit = delta;
    }
    if (impl_) impl_->markEndScatter(curr_stage_seq_, to_emit);
    onStageEndScatterCore_(curr_stage_seq_, to_emit);
}

// deliverSpike 实现已拆分到 SnnPESubComponent_spike.cc（输入路径控制逻辑）

void SnnPESubComponent::resetMembraneState(float v_rest_value) {
    resetMembraneState_(v_rest_value);
    accReset_();
}

void SnnPESubComponent::setMemoryLink(SST::Link* link) {
    memory_link_ = link;
    
    // ★ 关键修正：直接使用提供的Link进行内存操作 ★
    if (memory_link_) {
        // output_->verbose(CALL_INFO, 2, 0, "🔗 核心%d设置内存连接成功\n", core_id_);
        memory_ready_ = true;  // 标记内存已准备就绪
    } else {
        output_->verbose(CALL_INFO, 2, 0, "🔗 核心%d设置内存连接失败 (link=nullptr)\n", core_id_);
        memory_ready_ = false;
    }
}

bool SnnPESubComponent::hasWork() const {
    if (workload_) return workload_->hasWork();
    return legacySnnHasWork();
}

double SnnPESubComponent::getUtilization() const {
    if (workload_) return workload_->getUtilization();
    return legacySnnGetUtilization();
}

void SnnPESubComponent::getStatistics(std::map<std::string, uint64_t>& stats) const {
    if (workload_) {
        workload_->getStatistics(stats);
        return;
    }
    legacySnnGetStatistics(stats);
}

void SnnPESubComponent::legacySnnOnNeuronFires(const std::vector<uint32_t>& neuron_indices, uint64_t /*now_cycle*/) {
    // Phase4-Task6.3：统计口径仍锚定在 CoreShell；workload 只负责 route/comm 事务。
    for (uint32_t neuron_idx : neuron_indices) {
        if (stat_neurons_fired_) stat_neurons_fired_->addData(1);
        if (stat_spikes_generated_) stat_spikes_generated_->addData(1);
        count_neurons_fired_++;
        count_spikes_generated_++;
        if (apply_acc_enable_ && gas_window_mode_) {
            window_spikes_all_++;
        }
        if (neuron_idx < fired_ever_.size() && fired_ever_[neuron_idx] == 0) {
            fired_ever_[neuron_idx] = 1;
            if (auto* pe = parent_pe_cached_) {
                pe->accumulateUniqueNeuronFired(1);
            }
        }
    }
}

void SnnPESubComponent::legacySnnOnGasScatterSpikesEmitted(uint32_t /*seq*/, uint64_t spikes_emitted) {
    // Phase4-Task6.4：scatter 事务由 workload=snn 执行；CoreShell 仅负责：
    // - 统计/聚合口径（用于 essential_summary / mesh_stats）
    // - EndScatter 的窗口 spikes hint
    spikes_emitted_window_ = spikes_emitted;
    if (spikes_emitted > 0) {
        if (stat_gas_scatter_spikes_emitted_total_) {
            stat_gas_scatter_spikes_emitted_total_->addData(spikes_emitted);
        }
        if (auto* pe = parent_pe_cached_) {
            // 目前 acc_updates/posts_touched 等细项统计未迁入 workload；保持历史口径为 0（仅保留 spikes_emitted）。
            pe->accumulateApplyScatterStats(0, 0, spikes_emitted, 0, 0, 0);
        }
    }
}

void SnnPESubComponent::handleNeuronFire_(uint32_t neuron_idx, float v_before, float v_after) {
    legacySnnOnNeuronFires(std::vector<uint32_t>{neuron_idx}, static_cast<uint64_t>(total_cycles_));
    output_->verbose(CALL_INFO, 3, 0, "🔥 核心%d神经元%d发放脉冲! v_before=%.3f -> v_after=%.3f\n",
                    core_id_, neuron_idx, v_before, v_after);
    // 发送职责已迁入 workload=snn（Phase4 Task6.3）。
    if (snn_comm_workload_) {
        snn_comm_workload_->emitNeuronFire(neuron_idx, static_cast<uint64_t>(total_cycles_));
    }
}

uint64_t SnnPESubComponent::routeAndSendOutputs_(const std::vector<FireEvent>& fired) {
    uint64_t emitted = 0;
    for (const auto& ev : fired) {
        handleNeuronFire_(ev.neuron_idx, ev.v_before, ev.v_after);
        emitted++;
    }
    return emitted;
}

void SnnPESubComponent::drainCoreOutputsAndRoute_(uint64_t now_cycle) {
    if (!compute_core_) return;
    compute_core_->endCycle(now_cycle);
    std::vector<FireEvent> fired;
    compute_core_->drainOutputs(fired, true);
    routeAndSendOutputs_(fired);
}

// === Learning writeback (called by compute core) ===
// applyLocalWeightUpdates_ 已拆分到 SnnPESubComponent_mem.cc

// processLocalSpike 实现已拆分到 SnnPESubComponent_spike.cc（输入路径控制逻辑）

// requestWeight / handleMemoryResponse 已拆分到 SnnPESubComponent_mem.cc（StandardMem 控制面）

void SnnPESubComponent::verifyDenseAccumulator_(uint32_t seq) {
    if (acc_ops_) {
        acc_ops_->verifyDense(seq);
    }
}

// === Helpers implementations ===
// prepareDenseRead_ / issueReadCommon_ 已拆分到 SnnPESubComponent_mem.cc

void SnnPESubComponent::initializeStatistics() {
    // output_->verbose(CALL_INFO, 2, 0, "📊 核心%d初始化统计收集\n", core_id_);
    
    stat_spikes_received_ = registerStatistic<uint64_t>("spikes_received");
    stat_spikes_generated_ = registerStatistic<uint64_t>("spikes_generated");
    stat_neurons_fired_ = registerStatistic<uint64_t>("neurons_fired");
    stat_memory_requests_ = registerStatistic<uint64_t>("memory_requests");
    stat_weight_cache_hits_ = registerStatistic<uint64_t>("weight_cache_hits");
    stat_weight_cache_misses_ = registerStatistic<uint64_t>("weight_cache_misses");
    stat_merged_reads_rows_ = registerStatistic<uint64_t>("merged_reads_rows");
    stat_merged_reads_cls_ = registerStatistic<uint64_t>("merged_reads_cls");
    stat_weights_verify_count_ = registerStatistic<uint64_t>("weights_verify_count");
    stat_weights_mismatch_count_ = registerStatistic<uint64_t>("weights_mismatch_count");
    stat_weights_verify_sum_ = registerStatistic<double>("weights_verify_sum");
    // 扩展统计
    stat_routes_entries_ = registerStatistic<uint64_t>("routes_entries");
    stat_fanout_per_spike_ = registerStatistic<uint64_t>("fanout_per_spike");
    stat_cache_evictions_ = registerStatistic<uint64_t>("cache_evictions");
    stat_pending_reqs_peak_ = registerStatistic<uint64_t>("pending_reqs_peak");
    stat_cycles_update_neuron_ = registerStatistic<uint64_t>("cycles_update_neuron");
    stat_synaptic_accesses_ = registerStatistic<uint64_t>("synaptic_accesses");
    stat_s1_bytes_read_ = registerStatistic<uint64_t>("scheme1_bytes_read");
    // 门控诊断：权重读请求发起次数
    stat_weight_read_requests_ = registerStatistic<uint64_t>("weight_read_requests");
    // GAS totals accumulated from GatherBufferIF via CustomResp
    stat_gas_unique_reads_total_ = registerStatistic<uint64_t>("gas_unique_reads_total");
    stat_gas_unique_bytes_total_ = registerStatistic<uint64_t>("gas_unique_bytes_total");
    // 边集合溢出计数（仅在容量保护触发时递增）
    stat_gas_edge_overflow_ = registerStatistic<uint64_t>("gas_edge_overflow");
    // Apply/Scatter端到端统计（Phase‑1）
    stat_gas_apply_acc_updates_total_ = registerStatistic<uint64_t>("gas_apply_acc_updates_total");
    stat_gas_acc_posts_touched_total_ = registerStatistic<uint64_t>("gas_acc_posts_touched_total");
    stat_gas_scatter_spikes_emitted_total_ = registerStatistic<uint64_t>("gas_scatter_spikes_emitted_total");
    stat_gas_acc_hwm_bytes_total_ = registerStatistic<uint64_t>("gas_acc_high_watermark_bytes_total");
    stat_gas_acc_spill_records_total_ = registerStatistic<uint64_t>("gas_acc_spill_records_total");
    stat_gas_acc_spilled_bytes_total_ = registerStatistic<uint64_t>("gas_acc_spilled_bytes_total");
    // GAS superstep durations（cycles@1GHz == ns）
    // 留空注册，默认不向 SST 统计输出；统一使用 stage_events_csv + 离线脚本聚合
    // Batch-A additions（若需要SST统计输出，可在后续版本开放）
    // stat_mem_read_latency_cycles_ = registerStatistic<uint64_t>("mem_read_latency_cycles");
    // stat_mem_read_latency_cycles_weights_ = registerStatistic<uint64_t>("mem_read_latency_cycles_weights");
    // stat_mem_read_latency_cycles_state_ = registerStatistic<uint64_t>("mem_read_latency_cycles_state");
    // 记录单次内存请求大小与发起时的未完成请求数（Mesh 汇总使用）
    stat_mem_req_size_bytes_ = registerStatistic<uint64_t>("mem_req_size_bytes");
    stat_mem_outstanding_at_issue_ = registerStatistic<uint64_t>("mem_outstanding_at_issue");

    // Phase6: stream workload stats (always registered; default no-op when workload_impl=snn)
    stat_stream_mem_writes_issued_total_ = registerStatistic<uint64_t>("stream_mem_writes_issued_total");
    stat_stream_mem_reads_issued_total_ = registerStatistic<uint64_t>("stream_mem_reads_issued_total");
    stat_stream_mem_bytes_written_total_ = registerStatistic<uint64_t>("stream_mem_bytes_written_total");
    stat_stream_mem_bytes_read_total_ = registerStatistic<uint64_t>("stream_mem_bytes_read_total");
    stat_stream_mem_verify_pass_total_ = registerStatistic<uint64_t>("stream_mem_verify_pass_total");
    stat_stream_mem_verify_fail_total_ = registerStatistic<uint64_t>("stream_mem_verify_fail_total");
    stat_stream_pkt_sent_total_ = registerStatistic<uint64_t>("stream_pkt_sent_total");
    stat_stream_pkt_recv_total_ = registerStatistic<uint64_t>("stream_pkt_recv_total");
    stat_stream_pkt_bad_crc_total_ = registerStatistic<uint64_t>("stream_pkt_bad_crc_total");
    stat_stream_pkt_bad_magic_total_ = registerStatistic<uint64_t>("stream_pkt_bad_magic_total");
    
    // Attach stats hooks to accumulator module now that they are registered.
    if (acc_ops_) {
        AccumulatorOpsConfig acc_cfg{};
        acc_cfg.num_neurons = num_neurons_;
        acc_cfg.dense_enable = acc_dense_enable_cfg_;
        acc_cfg.spill_enable = acc_spill_enable_cfg_;
        acc_cfg.high_watermark_bytes = acc_hwm_bytes_cfg_;
        acc_cfg.shadow_verify_enable = acc_shadow_verify_enable_cfg_;
        acc_cfg.window_read_debug = window_read_debug_;
        acc_cfg.core_id = core_id_;
        acc_cfg.verbose = verbose_;
        acc_cfg.out = output_;
        acc_cfg.updates_count = &acc_updates_count_;
        acc_cfg.posts_touched_count = &acc_posts_touched_count_;
        acc_cfg.spill_records_count = &acc_spill_records_count_;
        acc_cfg.spilled_bytes_sum = &acc_spilled_bytes_sum_;
        acc_cfg.hwm_bytes_max = &acc_hwm_bytes_max_;
        acc_cfg.stat_apply_updates_total = stat_gas_apply_acc_updates_total_;
        acc_cfg.stat_posts_touched_total = stat_gas_acc_posts_touched_total_;
        acc_cfg.stat_spill_records_total = stat_gas_acc_spill_records_total_;
        acc_cfg.stat_spilled_bytes_total = stat_gas_acc_spilled_bytes_total_;
        acc_cfg.stat_hwm_bytes_total = stat_gas_acc_hwm_bytes_total_;
        acc_ops_->configure(acc_cfg);
    }

    // output_->verbose(CALL_INFO, 2, 0, "✅ 核心%d统计收集初始化完成\n", core_id_);
}
// === BCSR 辅助实现 ===
// BCSR 读/缓存/诊断实现已拆分到 SnnPESubComponent_bcsr.cc
