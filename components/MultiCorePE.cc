// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// MultiCorePE.cc: 真正的多核脉冲神经网络处理单元实现文件
//

#include <sst/core/sst_config.h>
#include "MultiCorePE.h"
#include "noc/SnnNetworkAdapter.h"
#include "noc/OptimizedInternalRing.h"
#include "SnnNIC.h"
#include "GatingDecisionEvent.h"
#include "ICoreControlHooks.h"
#include "ICoreMemoryLink.h"
#include "IGlobalStepHooks.h"
#include "IGlobalStepCreditHooks.h"
#include "ILoaderReadyHooks.h"
#include "NocPacketEvent.h"
#include "NocPacketBatchEvent.h"
#include "GasStepBarrierEvent.h"
#include "LoaderDoneEvent.h"
#include "TassLfP0ReportEvent.h"
#include "TassNaiveWindowRequestEvent.h"
#include "TassNaiveResponseEvent.h"
#include "WorkloadConfig.h"
#include "SnnDLLogging.h"
#include "SnnCoreAPI.h"
#include "multicore/MultiCorePEConfig.h"
#include "workload_stats/WorkloadStatsRegistry.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <cstdlib>
#include <random>
#include <cstdlib>
#include <climits>
#include <cinttypes>
#include <cerrno>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>

using namespace SST;
using namespace SST::SnnDL;

#ifndef PE_LOG
#define PE_LOG(lvl, ...) SNNDL_LOGPTR(output_, (lvl), __VA_ARGS__)
#endif

// ===== MultiCorePE 主组件实现 =====

// P2: 环境变量前端化 – 不再使用TU级别的 getenv 缓存；改用构造期解析的成员 sentinel_enabled_

namespace {
inline uint8_t stepStageCodeFromName_(const std::string& ev) {
    if (ev == "BeginGather") return 1;
    if (ev == "BeginApply") return 2;
    if (ev == "EndApply") return 3;
    if (ev == "BeginScatter") return 4;
    if (ev == "EndScatter") return 5;
    return 0;
}
	inline const char* stepStageCodeName_(uint8_t code) {
	    switch (code) {
        case 1: return "BeginGather";
        case 2: return "BeginApply";
        case 3: return "EndApply";
        case 4: return "BeginScatter";
        case 5: return "EndScatter";
        default: return "None";
	    }
	}

	inline std::string defaultPeOutDir_(int node_id) {
	    char buf[32];
	    std::snprintf(buf, sizeof(buf), "pe%02d", node_id);
	    return std::string(buf);
	}

	inline bool ensureDirExists_(const std::string& dir) {
	    if (dir.empty() || dir == ".") return true;
	    struct stat st;
	    if (::stat(dir.c_str(), &st) == 0) {
	        return S_ISDIR(st.st_mode);
	    }
	    if (::mkdir(dir.c_str(), 0775) == 0) return true;
	    return (errno == EEXIST);
	}

inline uint64_t ceilDivU64_(uint64_t num, uint64_t den) {
    if (den == 0) return 0;
    return (num + den - 1ull) / den;
}

inline uint64_t packTassContributorKey_(uint32_t node_id, uint32_t core_id) {
    return (static_cast<uint64_t>(node_id) << 32) | static_cast<uint64_t>(core_id);
}

inline uint64_t packTassBlockEpochKey_(uint32_t block_origin_node, uint32_t window_seq) {
    return (static_cast<uint64_t>(window_seq) << 32) | static_cast<uint64_t>(block_origin_node);
}

inline void computeTassBlockInfo_(uint32_t mesh_rows,
                                  uint32_t mesh_cols,
                                  uint32_t block_h,
                                  uint32_t block_w,
                                  uint32_t node_id,
                                  uint32_t cores_per_pe,
                                  uint32_t& block_origin_node,
                                  uint32_t& expected_contributors) {
    mesh_rows = std::max<uint32_t>(1u, mesh_rows);
    mesh_cols = std::max<uint32_t>(1u, mesh_cols);
    block_h = std::max<uint32_t>(1u, block_h);
    block_w = std::max<uint32_t>(1u, block_w);
    cores_per_pe = std::max<uint32_t>(1u, cores_per_pe);
    const uint32_t pe_row = node_id / mesh_cols;
    const uint32_t pe_col = node_id % mesh_cols;
    const uint32_t block_row0 = (pe_row / block_h) * block_h;
    const uint32_t block_col0 = (pe_col / block_w) * block_w;
    const uint32_t block_rows = std::min<uint32_t>(block_h, mesh_rows > block_row0 ? (mesh_rows - block_row0) : 1u);
    const uint32_t block_cols = std::min<uint32_t>(block_w, mesh_cols > block_col0 ? (mesh_cols - block_col0) : 1u);
    block_origin_node = block_row0 * mesh_cols + block_col0;
    expected_contributors = std::max<uint32_t>(1u, block_rows * block_cols * cores_per_pe);
}

inline void computeTassBlockInfo_(const TassLfP0WindowReport& report,
                                  uint32_t& block_origin_node,
                                  uint32_t& expected_contributors) {
    computeTassBlockInfo_(report.mesh_rows,
                          report.mesh_cols,
                          report.block_h,
                          report.block_w,
                          report.node_id,
                          report.cores_per_pe,
                          block_origin_node,
                          expected_contributors);
}

inline void computeTassBlockInfo_(const TassNaiveWindowRequest& request,
                                  uint32_t& block_origin_node,
                                  uint32_t& expected_contributors) {
    computeTassBlockInfo_(request.mesh_rows,
                          request.mesh_cols,
                          request.block_h,
                          request.block_w,
                          request.source_node,
                          request.cores_per_pe,
                          block_origin_node,
                          expected_contributors);
}

inline float loadTassNaiveWeight_(const std::vector<uint8_t>& data, uint32_t byte_offset) {
    if (byte_offset + sizeof(float) > data.size()) return 0.0f;
    float value = 0.0f;
    std::memcpy(&value, data.data() + byte_offset, sizeof(float));
    return value;
}
} // namespace

MultiCorePE::MultiCorePE(ComponentId_t id, Params& params) : Component(id) {
    const MultiCorePEConfig cfg = parseMultiCorePEConfig(params);

    // 初始化输出对象
    output_ = new Output("MultiCorePE[@p:@l]: ", cfg.verbose_level, 0, Output::STDOUT);

    // 读取基础配置参数（已集中解析）
    num_cores_ = cfg.num_cores;
    neurons_per_core_ = cfg.neurons_per_core;
    total_neurons_ = num_cores_ * neurons_per_core_;
    neurons_per_pe_cfg_ = cfg.neurons_per_pe;
    node_id_ = cfg.node_id;
    total_nodes_ = cfg.total_nodes;
    global_neuron_base_ = cfg.global_neuron_base;
    base_addr_ = cfg.base_addr;
    sim_stop_ns_ = cfg.sim_stop_ns;
    // NoC e2e latency histogram range (cycles) for native multicast lab.
    noc_lat_hist_max_ = cfg.noc_lat_hist_max;
    noc_lat_spike_hist_.assign(static_cast<size_t>(noc_lat_hist_max_) + 2u, 0);
    noc_lat_spikekey_hist_.assign(static_cast<size_t>(noc_lat_hist_max_) + 2u, 0);
    verbose_ = cfg.verbose_level;
    clock_freq_ = cfg.clock_freq;
    // P2: 解析 sentinel 与步级诊断参数（未设置则回退环境变量）
    {
        sentinel_enabled_ = cfg.sentinel_enabled;
        progress_log_interval_ns_ = cfg.progress_log_interval_ns;
        progress_log_node_ = cfg.progress_log_node;
        step_diag_cap_cfg_ = cfg.step_diag_cap;
        step_diag_enable_cfg_ = cfg.step_diag_enable;
    }
    weights_file_ = cfg.weights_file;
    enable_numa_ = cfg.enable_numa;

    // ===== 全局ID布局口径（fail-fast）=====
    // 统一口径：neurons_per_pe 必须等于 num_cores*neurons_per_core（禁止用其它 stride/bytes 误填）
    const uint64_t derived_neurons_per_pe =
        static_cast<uint64_t>(num_cores_) * static_cast<uint64_t>(neurons_per_core_);
    if (neurons_per_pe_cfg_ == 0) {
        neurons_per_pe_cfg_ = static_cast<uint32_t>(derived_neurons_per_pe);
    } else if (static_cast<uint64_t>(neurons_per_pe_cfg_) != derived_neurons_per_pe) {
        output_->fatal(
            CALL_INFO, -1,
            "MultiCorePE fatal: neurons_per_pe 参数口径错误：cfg=%u 但期望 num_cores*neurons_per_core=%" PRIu64 "（请检查脚本/配置是否误把 per_core_stride/base_addr 等字节量传入）\n",
            neurons_per_pe_cfg_, derived_neurons_per_pe);
    }
    if (total_nodes_ <= 0) {
        output_->fatal(CALL_INFO, -1, "MultiCorePE fatal: total_nodes 必须 > 0，当前=%d\n", total_nodes_);
    }
    if (node_id_ < 0 || node_id_ >= total_nodes_) {
        output_->fatal(CALL_INFO, -1, "MultiCorePE fatal: node_id=%d 超出 [0,%d)\n", node_id_, total_nodes_);
    }
    global_layout_ = GlobalNeuronLayout(static_cast<uint32_t>(total_nodes_),
                                        static_cast<uint32_t>(num_cores_),
                                        static_cast<uint32_t>(neurons_per_core_));
    if (!global_layout_.valid()) {
        output_->fatal(CALL_INFO, -1,
            "MultiCorePE fatal: GlobalNeuronLayout 无效：total_nodes=%d num_cores=%d neurons_per_core=%d\n",
            total_nodes_, num_cores_, neurons_per_core_);
    }
    const uint64_t expected_base = global_layout_.globalBaseOfNode(static_cast<uint32_t>(node_id_));
    if (global_neuron_base_ != expected_base) {
        output_->fatal(
            CALL_INFO, -1,
            "MultiCorePE fatal: global_neuron_base(0x%" PRIx64 ") 与 node_id=%d 推导基址(0x%" PRIx64 ") 不一致（期望 global_neuron_base=node_id*neurons_per_pe）\n",
            global_neuron_base_, node_id_, expected_base);
    }
    
    // 神经元参数
    v_thresh_ = cfg.v_thresh;
    v_reset_ = cfg.v_reset;
    v_rest_ = cfg.v_rest;
    tau_mem_ = cfg.tau_mem;
    t_ref_ = cfg.t_ref;
    
    // 测试流量参数
    enable_test_traffic_ = cfg.enable_test_traffic;
    test_target_node_ = cfg.test_target_node;
    test_period_ = cfg.test_period;
    test_spikes_per_burst_ = cfg.test_spikes_per_burst;
    test_weight_ = cfg.test_weight;
    test_max_spikes_ = cfg.test_max_spikes;
    
    // 环形网络实现选择
    use_optimized_ring_ = cfg.use_optimized_ring;
    // Phase5：冻结 legacy InternalRing 分支（use_optimized_ring=0）
    // - NoC 子系统已以 OptimizedInternalRing 为唯一片上互连后端完成闭环
    // - legacy InternalRing 会引入维护成本与语义漂移风险，故在此明确禁用
    if (!use_optimized_ring_ && num_cores_ > 1) {
        output_->fatal(CALL_INFO, -1,
            "❌ 配置错误：use_optimized_ring=0 (legacy InternalRing) 已冻结/不再支持，请设置 use_optimized_ring=1\n");
    }
    // 输出控制：是否打印节点汇总
    print_node_summary_ = cfg.print_node_summary;
    primary_keepalive_ = cfg.primary_keepalive;
    manual_core_drive_enable_ = cfg.manual_core_drive_enable;
    manual_gas_gather_cycles_ = cfg.manual_gas_gather_cycles;
    
    // 权重验证参数
    verify_weights_ = cfg.verify_weights;
    weight_verify_samples_ = cfg.weight_verify_samples;
    expected_weight_value_ = cfg.expected_weight_value;
    verify_log_each_sample_ = cfg.verify_log_each_sample;
    
    // 权重回退参数
    use_event_weight_fallback_ = cfg.use_event_weight_fallback;
    enable_memory_weights_ = cfg.enable_memory_weights;
    write_weights_on_init_ = cfg.write_weights_on_init;

    // 时间窗口化统计参数（默认关闭）
    window_stats_enable_ = cfg.window_stats_enable;
    window_us_ = cfg.window_us;
    window_csv_ = cfg.window_csv;
    window_metrics_csv_ = cfg.window_metrics_csv;
    window_ns_ = window_us_ * 1000ULL; // 1us = 1000ns（组件时钟1GHz，tick≈1ns）
    diag_fire_log_ = cfg.diag_fire_log;
    exec_mode_ = cfg.exec_mode;
    std::vector<std::string> unknown_modules;
    workload_stats_modules_ =
        WorkloadStatsRegistry::buildModules(cfg.workload_impl, cfg.workload_stats_modules, &unknown_modules);
    if (!unknown_modules.empty() && output_) {
        std::ostringstream oss;
        for (size_t i = 0; i < unknown_modules.size(); ++i) {
            if (i) oss << ",";
            oss << unknown_modules[i];
        }
        output_->verbose(CALL_INFO, 1, 0, "[workload_stats] unknown modules ignored: %s\n", oss.str().c_str());
    }

    // Global Step/GAS barrier sync (Phase-step-sync)
    global_step_sync_enable_ = cfg.global_step_sync_enable;
    {
        std::string pol = cfg.global_step_done_policy;
        if (pol == "drain" || pol == "drain_based" || pol == "drainbased") {
            global_step_done_policy_ = GlobalStepDonePolicy::Drain;
        } else if (pol == "fixed" || pol == "fixed_cycles" || pol == "timer") {
            global_step_done_policy_ = GlobalStepDonePolicy::FixedCycles;
        } else if (pol == "quiescent" || pol == "quiet") {
            global_step_done_policy_ = GlobalStepDonePolicy::Quiescent;
        } else {
            global_step_done_policy_ = GlobalStepDonePolicy::EndScatter;
        }
    }
    global_step_quiescent_min_cycles_ = cfg.global_step_quiescent_min_cycles;
    global_step_drain_min_cycles_ = cfg.global_step_drain_min_cycles;
    global_step_fixed_cycles_ = cfg.global_step_fixed_cycles;
    // Step-limited fairness: avoid starting step injection while WeightLoader/BCSR metadata is not ready.
    // In naive_* (non-window) mode, issuing BCSR reads before loader/rowptr ready can enqueue millions of waiters.
    loader_done_key_ = cfg.loader_done_key;
    global_step_ready_delay_cycles_ = cfg.global_step_ready_delay_cycles;
    loader_done_timeout_cycles_ = params.find<uint64_t>("loader_done_timeout_cycles", 0);
    wait_for_loader_done_ = (!loader_done_key_.empty());
    loader_ready_latched_ = false;
    loader_ready_cycle_ = 0;
    if (wait_for_loader_done_) {
        loader_done_shared_ = std::make_unique<SST::Shared::SharedArray<int>>();
        loader_done_shared_->initialize(loader_done_key_, 1, 0);
    }
    tass_lf_p0_enable_ = params.find<int>("experimental_tass_lf_p0_enable", 0) != 0;
    naive_tass_enable_ = params.find<int>("experimental_naive_tass_enable", 0) != 0;
    tass_lf_p0_mesh_rows_ = std::max<uint32_t>(1u, params.find<uint32_t>("tass_lf_p0_mesh_rows", 1));
    tass_lf_p0_mesh_cols_ = std::max<uint32_t>(1u, params.find<uint32_t>("tass_lf_p0_mesh_cols", 1));
    tass_lf_p0_block_h_ = std::max<uint32_t>(1u, params.find<uint32_t>("tass_lf_p0_block_h", 2));
    tass_lf_p0_block_w_ = std::max<uint32_t>(1u, params.find<uint32_t>("tass_lf_p0_block_w", 2));
    output_->verbose(CALL_INFO, 1, 0,
                     "[tass-debug] node=%d naive_tass_enable=%d tass_lf_p0_enable=%d mesh=%ux%u block=%ux%u\n",
                     node_id_,
                     naive_tass_enable_ ? 1 : 0,
                     tass_lf_p0_enable_ ? 1 : 0,
                     tass_lf_p0_mesh_rows_,
                     tass_lf_p0_mesh_cols_,
                     tass_lf_p0_block_h_,
                     tass_lf_p0_block_w_);

    // Step-level random activation injection (Phase3-B): 下沉为独立子系统（MultiCorePE 仅转发 tick/阶段事件）
    {
        StepActivationSubsystem::Config step_cfg;
        step_cfg.enable = cfg.step_activation_enable;
        step_cfg.fraction = cfg.step_activation_fraction;
        step_cfg.fanout = cfg.step_activation_fanout;
        step_cfg.seed = cfg.step_activation_seed;
        step_cfg.period_cycles = cfg.step_activation_period_cycles;
        step_cfg.trigger_core = cfg.step_activation_trigger_core;
        step_cfg.reset_mem_each_step = cfg.step_reset_mem_each_step;
        step_cfg.event_weight = cfg.step_activation_event_weight;
        {
            const std::string& pat = cfg.step_activation_pre_pattern;
            if (pat == "clustered" || pat == "cluster" || pat == "block") {
                step_cfg.pre_pattern = StepActivationSubsystem::Config::PrePattern::Clustered;
            } else {
                step_cfg.pre_pattern = StepActivationSubsystem::Config::PrePattern::BernoulliUniform;
            }
        }
        step_cfg.pre_cluster_len = cfg.step_activation_pre_cluster_len;
        step_cfg.use_bcsr_routes = cfg.step_activation_use_bcsr_routes;
        step_cfg.bcsr_template = cfg.step_activation_bcsr_template;
        step_cfg.bcsr_rows_per_core =
            (cfg.step_activation_bcsr_rows_per_core > 0)
                ? cfg.step_activation_bcsr_rows_per_core
                : static_cast<uint32_t>(neurons_per_core_);
        step_cfg.bcsr_br = cfg.step_activation_bcsr_br;
        step_cfg.bcsr_bc = cfg.step_activation_bcsr_bc;
        step_cfg.bcsr_idx_bytes = cfg.step_activation_bcsr_idx_bytes;
        step_cfg.bcsr_val_bytes = cfg.step_activation_bcsr_val_bytes;
        step_cfg.bcsr_rowptr_offset = cfg.step_activation_bcsr_rowptr_offset;
        step_cfg.bcsr_colidx_offset = cfg.step_activation_bcsr_colidx_offset;
        step_cfg.bcsr_blockdata_offset = cfg.step_activation_bcsr_blockdata_offset;
        step_cfg.bcsr_blockids_offset = cfg.step_activation_bcsr_blockids_offset;
        step_cfg.bcsr_weight_epsilon = cfg.step_activation_bcsr_weight_epsilon;
        step_cfg.log_enable = cfg.step_activation_log_enable;
        step_cfg.build_local_only = cfg.step_activation_build_local_only;
        step_cfg.bcsr_align = cfg.step_activation_bcsr_align;

        // 通用 workload（例如 stream）下必须禁用 Step/Synapse 语义注入，否则会污染纯通信/纯内存负载。
        // 选择来源：优先 Params.workload_impl，其次环境变量 SNNDL_WORKLOAD_IMPL（保持脚本不改的兼容路径）。
        if (isNonSnnWorkloadKind(cfg.workload_kind)) {
            step_cfg.enable = false;
            step_cfg.fraction = 0.0;
            step_cfg.fanout = 0;
            step_cfg.period_cycles = 0;
            step_cfg.use_bcsr_routes = false;
            step_cfg.bcsr_template.clear();
        }

        step_activation_subsys_.configure(step_cfg);

        StepActivationSubsystem::Runtime step_rt;
        step_rt.log = output_;
        step_rt.node_id = node_id_;
        step_rt.total_nodes = total_nodes_;
        step_rt.global_neuron_base = global_neuron_base_;
        step_rt.num_cores = num_cores_;
        step_rt.neurons_per_core = neurons_per_core_;
        step_rt.neurons_per_pe_cfg = neurons_per_pe_cfg_;
        step_rt.layout = &global_layout_;
        step_rt.sentinel_enabled = sentinel_enabled_;
        step_rt.step_diag_cap_cfg = step_diag_cap_cfg_;
        step_rt.step_diag_enable_cfg = step_diag_enable_cfg_;
        step_rt.noc = &noc_subsys_;
        step_rt.reset_membranes = [this]() { resetAllCoreMembranes(); };

        step_activation_subsys_.bindRuntime(step_rt);
        step_activation_subsys_.initBcsrReachabilityIfEnabled();
    }

    // 外部端口 SpikeEvent 直注入（仅本地投递）：收敛为 Stimulus 子系统，MultiCorePE 仅装配/转发。
    {
        ExternalSpikeInputSubsystem::Runtime ex_rt;
        ex_rt.log = output_;
        ex_rt.node_id = node_id_;
        ex_rt.layout = &global_layout_;
        ex_rt.noc = &noc_subsys_;
        ex_rt.global_neuron_base = global_neuron_base_;
        ex_rt.num_cores = num_cores_;
        ex_rt.neurons_per_core = neurons_per_core_;
        ex_rt.total_neurons = total_neurons_;
        external_spike_input_subsys_.bindRuntime(ex_rt);
    }

    // Phase3-C：Spike 编解码与投递 glue 下沉为 synapse/route 子系统；MultiCorePE 仅装配。
    {
        SpikePacketBridge::Runtime brt;
        brt.log = output_;
        brt.node_id = node_id_;
        brt.num_cores = num_cores_;
        brt.layout = &global_layout_;
        brt.noc = &noc_subsys_;
        if (exec_mode_ == "naive_raw" && global_step_sync_enable_) {
            brt.active_step_seq = &global_step_active_seq_;
            brt.step_seq_offset = 1;
        }
        spike_packet_bridge_.bindRuntime(brt);
    }
    
    //     "🔧 多核PE配置: cores=%d, neurons_per_core=%d, total_neurons=%d, node_id=%d\n",
    //     num_cores_, neurons_per_core_, total_neurons_, node_id_);
    
    //     "🧠 神经元参数: v_thresh=%.3f, v_reset=%.3f, v_rest=%.3f, tau_mem=%.1fms, t_ref=%d\n",
    //     v_thresh_, v_reset_, v_rest_, tau_mem_, t_ref_);

    // 验证参数合理性
    if (num_cores_ <= 0 || num_cores_ > 64) {
        output_->fatal(CALL_INFO, -1, "❌ 错误: num_cores必须在1-64之间，当前值=%d\n", num_cores_);
    }
    // 放宽每核神经元上限，支持大规模单PE评估（例如 20核×50k/核 = 1M/PE）
    // 原上限为 1024，现放宽至 65536；如需更大规模，可视硬件内存与仿真需求再调高。
    if (neurons_per_core_ <= 0 || neurons_per_core_ > 65536) {
        output_->fatal(CALL_INFO, -1, "❌ 错误: neurons_per_core必须在1-65536之间，当前值=%d\n", neurons_per_core_);
    }

    // 初始化时钟计数器
    current_cycle_ = 0;
    test_cycle_counter_ = 0;
    test_spikes_sent_ = 0;
    
    // 初始化处理单元状态追踪
    unit_states_.resize(num_cores_);
    for (int i = 0; i < num_cores_; i++) {
        unit_states_[i].unit_id = i;
        unit_states_[i].neuron_id_start = i * neurons_per_core_;
        unit_states_[i].neuron_count = neurons_per_core_;
        unit_states_[i].is_active = false;
        unit_states_[i].spikes_processed = 0;
        unit_states_[i].neurons_fired = 0;
        unit_states_[i].utilization = 0.0;
    }
    
    // 初始化组件指针为空
    l2_cache_ = nullptr;
    memory_interface_ = nullptr;
    external_nic_ = nullptr;
    optimized_ring_ = nullptr;
    controller_ = nullptr;
    
    // 初始化端口指针为空
    external_spike_input_link_ = nullptr;
    external_spike_output_link_ = nullptr;
    mem_link_ = nullptr;
    
    
    // 初始化统计收集（必须在构造函数中）
    initializeStatistics();
    {
        StepActivationSubsystem::Stats st;
        st.invocations = stat_step_activation_invocations_;
        st.pre_selected = stat_step_activation_pre_selected_;
        st.spike_attempts = stat_step_activation_spike_attempts_;
        st.spikes_injected = stat_step_activation_spikes_injected_;
        st.route_hits = stat_step_activation_route_hits_;
        st.route_misses = stat_step_activation_route_misses_;
        st.local_drops = stat_step_activation_local_drops_;
        step_activation_subsys_.bindStats(st);
    }
    {
        NocSubsystem::Config noc_cfg;
        noc_cfg.log_enable = (verbose_ >= 5);
        noc_subsys_.configure(noc_cfg);

        NocSubsystem::Stats noc_st;
        noc_st.external_spikes_received = stat_external_spikes_received_;
        noc_st.external_spikes_sent = stat_external_spikes_sent_;
        noc_st.inter_core_messages = stat_inter_core_messages_;
        noc_subsys_.bindStats(noc_st);

        NocSubsystem::Runtime noc_rt;
        noc_rt.log = output_;
        noc_rt.node_id = node_id_;
        noc_rt.num_cores = num_cores_;
        noc_rt.nic = nullptr;  // init 后再注入
        noc_rt.optimized_ring = nullptr;  // init 后再注入
        noc_rt.external_spike_output_link = nullptr;  // init(phase0) link configured 后再注入
        if (exec_mode_ == "naive_raw" && global_step_sync_enable_) {
            noc_rt.active_step_seq = &global_step_active_seq_;
            noc_rt.step_seq_offset = 1;
        }
        noc_rt.deliver_to_endpoint = [this](int endpoint_id, NocPacketEvent* pkt) {
            deliverPacketToEndpoint_(endpoint_id, pkt);
        };
        noc_subsys_.bindRuntime(noc_rt);
    }
    if (naive_tass_enable_) {
        memory_interface_ = loadUserSubComponent<SST::Interfaces::StandardMem>(
            "memory_interface", ComponentInfo::SHARE_NONE,
            registerTimeBase("1ns"),
            new SST::Interfaces::StandardMem::Handler2<MultiCorePE, &MultiCorePE::handleTassNaiveMemoryResponse_>(this));
        if (!memory_interface_) {
            output_->fatal(CALL_INFO, -1,
                           "MultiCorePE fatal: experimental_naive_tass_enable=1 but memory_interface subcomponent is missing on node=%d\n",
                           node_id_);
        }
    }
    // 记录路径（若提供），用于派生输出目录
    stage_events_csv_path_ = cfg.stage_events_csv_path;
    stats_csv_path_ = cfg.stats_csv_path;
    
    // 关键修复：在构造函数中初始化网络接口，确保SST能在正确时机调用init()
    initializeNetworkInterface();
}

MultiCorePE::~MultiCorePE() {
    clearAllDeferredPackets_();

    // 清理SnnPE SubComponent核心（SST会自动管理SubComponent的生命周期）
    cores_.clear();
    
    // 清理内部组件
    delete optimized_ring_;
    delete controller_;
    // 避免析构次序竞态：不手动delete日志对象
    output_ = nullptr;
}

void MultiCorePE::init(unsigned int phase) {
    if (sentinel_enabled_ && output_) { output_->verbose(CALL_INFO, 2, 0, "[[sentinel-pe-init]] node=%d phase=%u enter\n", node_id_, phase); }

    // WeightLoader -> MultiCorePE loader_done bridge uses untimed data during init/setup.
    // We must actively drain recvUntimedData(); a timed handler will not be invoked.
    if (loader_done_link_) {
        SST::Event* ev = loader_done_link_->recvUntimedData();
        while (ev) {
            handleLoaderDoneEvent(ev);
            ev = loader_done_link_->recvUntimedData();
        }
    }
    if (phase == 0) {
        if (primary_keepalive_ || sim_stop_ns_ > 0) {
            registerAsPrimaryComponent();
            primaryComponentDoNotEndSim();
        }
        // 阶段0：初始化基础组件和端口
        
        // 配置时钟
        std::string clock_freq = clock_freq_;
        // 不需要单独的clock_handler_变量
        registerClock(clock_freq, new Clock::Handler2<MultiCorePE,&MultiCorePE::clockTick>(this));
        if (sentinel_enabled_ && output_) { output_->verbose(CALL_INFO, 2, 0, "[[sentinel-pe-init]] node=%d phase=0 clock-registered\n", node_id_); }
        
        
        // 初始化统计收集
        
        // 初始化端口连接
        external_spike_input_link_ = configureLink("external_spike_input", 
            new Event::Handler2<MultiCorePE,&MultiCorePE::handleExternalSpikeEvent>(this));
        external_spike_output_link_ = configureLink("external_spike_output");
        mem_link_ = configureLink("mem_link");
        loader_done_link_ = configureLink(
            "loader_done",
            new Event::Handler2<MultiCorePE, &MultiCorePE::handleLoaderDoneEvent>(this));
        tass_p0_out_link_ = configureLink("tass_p0_out");
        tass_p0_in_links_.clear();
        tass_p0_in_links_.reserve(static_cast<size_t>(std::max(0, total_nodes_)));
        tass_p0_rsp_out_links_.clear();
        tass_p0_rsp_out_links_.reserve(static_cast<size_t>(std::max(0, total_nodes_)));
        for (int src = 0; src < total_nodes_; ++src) {
            std::ostringstream port;
            port << "tass_p0_in" << src;
            SST::Link* link = configureLink(
                port.str(),
                new Event::Handler2<MultiCorePE, &MultiCorePE::handleTassLfP0ReportEvent>(this));
            tass_p0_in_links_.push_back(link);
        }
        for (int dst = 0; dst < total_nodes_; ++dst) {
            std::ostringstream port;
            port << "tass_p0_rsp_out" << dst;
            tass_p0_rsp_out_links_.push_back(configureLink(port.str()));
        }
        if (loader_done_link_) {
            SST::Event* ev = loader_done_link_->recvUntimedData();
            while (ev) {
                handleLoaderDoneEvent(ev);
                ev = loader_done_link_->recvUntimedData();
            }
        }
        if (global_step_sync_enable_) {
            gas_step_ctrl_link_ = configureLink(
                "gas_step_ctrl",
                new Event::Handler2<MultiCorePE, &MultiCorePE::handleGasStepCtrlEvent>(this));
            if (!gas_step_ctrl_link_) {
                output_->fatal(CALL_INFO, -1, "❌ 配置错误：global_step_sync_enable=1 但端口 gas_step_ctrl 未连接\n");
            }
        }
        if (sentinel_enabled_ && output_) { output_->verbose(CALL_INFO, 2, 0, "[[sentinel-pe-init]] node=%d phase=0 links-configured\n", node_id_); }
        
        
        // 初始化方向链路（用于端口代理机制）
        initializeDirectionLinks();
        if (sentinel_enabled_ && output_) { output_->verbose(CALL_INFO, 2, 0, "[[sentinel-pe-init]] node=%d phase=0 dir-links\n", node_id_); }
        
        // 初始化处理单元
        initializeProcessingUnits();
        if (sentinel_enabled_ && output_) { output_->verbose(CALL_INFO, 2, 0, "[[sentinel-pe-init]] node=%d phase=0 units-initialized\n", node_id_); }
        
        // 初始化内部互连
        initializeInternalRing();
        if (sentinel_enabled_ && output_) { output_->verbose(CALL_INFO, 2, 0, "[[sentinel-pe-init]] node=%d phase=0 ring-initialized\n", node_id_); }

        // Phase4-A1.2：NoC 子系统后端装配（NIC / ring / legacy link）
        {
            NocSubsystem::Runtime noc_rt;
            noc_rt.log = output_;
            noc_rt.node_id = node_id_;
            noc_rt.num_cores = num_cores_;
            noc_rt.nic = external_nic_;
            noc_rt.optimized_ring = optimized_ring_;
            noc_rt.external_spike_output_link = external_spike_output_link_;
            if (exec_mode_ == "naive_raw" && global_step_sync_enable_) {
                noc_rt.active_step_seq = &global_step_active_seq_;
                noc_rt.step_seq_offset = 1;
            }
            noc_rt.deliver_to_endpoint = [this](int endpoint_id, NocPacketEvent* pkt) {
                deliverPacketToEndpoint_(endpoint_id, pkt);
            };
            noc_subsys_.bindRuntime(noc_rt);
        }
        
        // 初始化多核控制器
        controller_ = new MultiCoreController(this, output_);
        if (sentinel_enabled_ && output_) { output_->verbose(CALL_INFO, 2, 0, "[[sentinel-pe-init]] node=%d phase=0 controller-created\n", node_id_); }
        

        // 将当前phase转发给所有子核心
        for (auto* core : cores_) {
            if (core) core->init(phase);
        }
        if (sentinel_enabled_ && output_) { output_->verbose(CALL_INFO, 2, 0, "[[sentinel-pe-init]] node=%d phase=0 cores-init-done\n", node_id_); }
        
        // 关键修复：转发init到网络接口
        if (external_nic_) {
            external_nic_->init(phase);
        }
        if (memory_interface_) {
            memory_interface_->init(phase);
        }
        if (sentinel_enabled_ && output_) { output_->verbose(CALL_INFO, 2, 0, "[[sentinel-pe-init]] node=%d phase=0 nic-init-done\n", node_id_); }
        // 标记 Step 注入就绪（保证 NIC 已完成 init）
        step_activation_subsys_.setInjectionReady(true);
    }
    else if (phase == 1) {
        if (sentinel_enabled_ && output_) { output_->verbose(CALL_INFO, 2, 0, "[[sentinel-pe-init]] node=%d phase=1 enter\n", node_id_); }
        // 阶段1：加载权重和配置子组件
        loadAndDistributeWeights();
        if (sentinel_enabled_ && output_) { output_->verbose(CALL_INFO, 2, 0, "[[sentinel-pe-init]] node=%d phase=1 weights-loaded\n", node_id_); }

        // 将当前phase转发给所有子核心
        for (auto* core : cores_) {
            if (core) core->init(phase);
        }
        if (sentinel_enabled_ && output_) { output_->verbose(CALL_INFO, 2, 0, "[[sentinel-pe-init]] node=%d phase=1 cores-init-done\n", node_id_); }
        
        // 转发init到网络接口
        if (external_nic_) {
            external_nic_->init(phase);
        }
        if (memory_interface_) {
            memory_interface_->init(phase);
        }
        if (sentinel_enabled_ && output_) { output_->verbose(CALL_INFO, 2, 0, "[[sentinel-pe-init]] node=%d phase=1 nic-init-done\n", node_id_); }
    }
    else {
        if (sentinel_enabled_ && output_) { output_->verbose(CALL_INFO, 2, 0, "[[sentinel-pe-init]] node=%d phase=%u forward-only\n", node_id_, phase); }
        // 其余phase同样转发
        for (auto* core : cores_) {
            if (core) core->init(phase);
        }
        
        // 转发init到网络接口
        if (external_nic_) {
            external_nic_->init(phase);
        }
        if (memory_interface_) {
            memory_interface_->init(phase);
        }
        if (sentinel_enabled_ && output_) { output_->verbose(CALL_INFO, 2, 0, "[[sentinel-pe-init]] node=%d phase=%u done\n", node_id_, phase); }
    }
}

void MultiCorePE::complete(unsigned int phase) {
    // 关键：转发 complete 给所有子核心与网络接口。
    // 若核心 memory 子组件内部使用 memHierarchy.standardInterface，则 complete() 是完成
    // init 握手/地址域传播的必要阶段；缺失会导致 getTargetDestination 找不到地址域并 fatal。
    for (auto* core : cores_) {
        if (core) core->complete(phase);
    }
    if (external_nic_) {
        external_nic_->complete(phase);
    }
    if (memory_interface_) {
        memory_interface_->complete(phase);
    }
}

void MultiCorePE::setup() {
    if (sentinel_enabled_ && output_) { output_->verbose(CALL_INFO, 2, 0, "[[sentinel-pe-setup]] node=%d enter\n", node_id_); }

    // Drain any remaining untimed loader_done events (e.g. strict_loader_done publishes in setup).
    if (loader_done_link_) {
        SST::Event* ev = loader_done_link_->recvUntimedData();
        while (ev) {
            handleLoaderDoneEvent(ev);
            ev = loader_done_link_->recvUntimedData();
        }
    }
    
    // 验证所有组件初始化完成
    if (cores_.size() != static_cast<size_t>(num_cores_)) {
        output_->fatal(CALL_INFO, -1, "❌ 错误: 核心数量不匹配，期望%d，实际%zu\n", 
                      num_cores_, cores_.size());
    }
    
    // 检查内部互连（Phase5：仅保留 OptimizedInternalRing）
    // 单核情况下不需要内部互连
    if (num_cores_ > 1 && !optimized_ring_) {
        output_->fatal(CALL_INFO, -1, "❌ 错误: 多核配置但内部互连未初始化\n");
    }
    // 调用子核心的setup
    for (auto* core : cores_) {
        if (core) core->setup();
    }
    if (sentinel_enabled_ && output_) { output_->verbose(CALL_INFO, 2, 0, "[[sentinel-pe-setup]] node=%d cores-setup\n", node_id_); }
    
    // 调用网络接口的setup
    if (external_nic_) {
        external_nic_->setup();
    }
    if (memory_interface_) {
        memory_interface_->setup();
    }
    if (sentinel_enabled_ && output_) { output_->verbose(CALL_INFO, 2, 0, "[[sentinel-pe-setup]] node=%d nic-setup\n", node_id_); }
    
    if (!controller_) {
        output_->fatal(CALL_INFO, -1, "❌ 错误: 多核控制器未初始化\n");
    }

    global_step_sync_ready_ = true;
    // Step barrier readiness:
    // - default: send PE_READY at setup (legacy behavior)
    // - if loader_done_key is provided: delay PE_READY until WeightLoader publishes done (prevents naive_* blowups)
    if (global_step_sync_enable_ && gas_step_ctrl_link_ && !global_step_ready_sent_) {
        if (!wait_for_loader_done_) {
            auto* ev = new GasStepBarrierEvent(GasStepBarrierOp::PeReady, /*seq*/0, static_cast<uint32_t>(node_id_));
            gas_step_ctrl_link_->send(ev);
            global_step_ready_sent_ = true;
        }
    }
    
    // 打印组件配置摘要
    
}

void MultiCorePE::finish() {
    // 更新最终统计信息
    updateStatistics();
    // 报告总仿真周期（单实例）
    if (stat_sim_cycles_total_) stat_sim_cycles_total_->addData(current_cycle_);

    // Experiment observability: if global step sync is enabled but we finish mid-step,
    // record it (do not fatal; matrix scripts may mark INCONCLUSIVE).
    if (global_step_sync_enable_ && global_step_active_seq_ != 0 && !global_step_done_sent_) {
        if (stat_compat_finish_incomplete_total_) stat_compat_finish_incomplete_total_->addData(1);
    }

    // Global Step barrier：若仿真结束时本 PE 尚未完成 active step，输出每核卡点快照（仅 debug/sentinel 模式）
    if (global_step_sync_enable_ && sentinel_enabled_ && output_ &&
        global_step_active_seq_ != 0 && !global_step_done_sent_) {
        size_t done_cnt = 0;
        for (auto v : global_step_done_cores_) if (v) ++done_cnt;
        output_->verbose(CALL_INFO, 2, 0,
            "[[sentinel-step-sync]] node=%d finish_incomplete active_seq=%u done=%zu/%d\n",
            node_id_, global_step_active_seq_, done_cnt, num_cores_);
        for (int i = 0; i < num_cores_; ++i) {
            const bool done = (i >= 0 &&
                               static_cast<size_t>(i) < global_step_done_cores_.size() &&
                               global_step_done_cores_[static_cast<size_t>(i)] != 0);
            if (done) continue;
            const size_t idx = static_cast<size_t>(i);
            const uint8_t code =
                (idx < global_step_last_stage_code_.size()) ? global_step_last_stage_code_[idx] : 0;
            const uint32_t seq =
                (idx < global_step_last_stage_seq_.size()) ? global_step_last_stage_seq_[idx] : 0;
            const uint64_t ts =
                (idx < global_step_last_stage_ts_ns_.size()) ? global_step_last_stage_ts_ns_[idx] : 0;
            const uint64_t spikes =
                (idx < global_step_last_stage_spikes_.size()) ? global_step_last_stage_spikes_[idx] : 0;
            output_->verbose(CALL_INFO, 2, 0,
                "[[sentinel-step-sync]] node=%d finish_core core=%d last=%s seq=%u ts_ns=%" PRIu64 " spikes=%" PRIu64 "\n",
                node_id_, i, stepStageCodeName_(code), seq, (uint64_t)ts, (uint64_t)spikes);
        }
    }
    auto resolve_out_dir = [this]() -> std::string {
        std::string ref = stage_events_csv_path_;
        if (ref.empty()) ref = stats_csv_path_;
        if (!ref.empty()) {
            auto pos = ref.find_last_of('/');
            return (pos == std::string::npos) ? std::string(".") : ref.substr(0, pos);
        }
        // No explicit output path: avoid multi-node overwrite by defaulting to per-node dir in CWD.
        std::string dir = defaultPeOutDir_(node_id_);
        if (!ensureDirExists_(dir)) return std::string(".");
        return dir;
    };

    // 输出 PE 级 per-window 发放聚合（与 stage_events 同目录）
    if (!window_spikes_pe_.empty()) {
        std::string dir = resolve_out_dir();
        // 文件名必须保持稳定：上游分析脚本默认查找 pe*/pe_window_spikes_db.csv
        std::string path = dir + "/pe_window_spikes_db.csv";
        std::ofstream fout(path);
        if (fout.good()) {
            fout << "seq,pe_spikes_emitted\n";
            std::vector<std::pair<uint32_t,uint64_t>> rows(window_spikes_pe_.begin(), window_spikes_pe_.end());
            std::sort(rows.begin(), rows.end(), [](auto&a, auto&b){return a.first<b.first;});
            for (auto &kv : rows) fout << kv.first << "," << kv.second << "\n";
            fout.close();
        }
    }
    // 输出 PE 级阶段事件CSV（与 pe_window_spikes_db.csv 同目录）
    if (!stage_marks_.empty()) {
        std::string dir = resolve_out_dir();
        // 文件名必须保持稳定：上游分析脚本默认查找 pe*/pe_stage_events_db.csv
        std::string pathA = dir + "/pe_stage_events_db.csv";
        std::ofstream fout(pathA);
        if (fout.good()) {
            fout << "seq,bg_ns,ga_ns,ea_ns,bs_ns,es_ns,gather_ns,apply_ns,scatter_ns,total_ns\n";
            std::vector<uint32_t> keys; keys.reserve(stage_marks_.size());
            for (auto &kv : stage_marks_) keys.push_back(kv.first);
            std::sort(keys.begin(), keys.end());
            for (auto seq : keys) {
                auto &m = stage_marks_[seq];
                uint64_t g=0,a=0,s=0,t=0;
                if (m.bg && m.ga && m.ga>=m.bg) g = m.ga - m.bg;
                if (m.ga && m.bs && m.bs>=m.ga) a = m.bs - m.ga;
                if (m.bs && m.es && m.es>=m.bs) s = m.es - m.bs;
                if (m.bg && m.es && m.es>=m.bg) t = m.es - m.bg;
                fout << seq << "," << m.bg << "," << m.ga << "," << m.ea << "," << m.bs << "," << m.es
                     << "," << g << "," << a << "," << s << "," << t << "\n";
            }
            fout.close();
        }
        // 兼容原 compute 工具：另写一份 stage_events_db_<sim>.csv，包含 spikes_emitted 列
        // 由于本组件无法直接得知仿真时长字符串，这里统一使用固定文件名，供上层脚本选择最新文件回退读取。
        std::string pathB = dir + "/stage_events_db.csv";
        std::ofstream fout2(pathB);
        if (fout2.good()) {
            fout2 << "seq,event,sim_time_ns,spikes_emitted\n";
            std::vector<uint32_t> keys; keys.reserve(stage_marks_.size());
            for (auto &kv : stage_marks_) keys.push_back(kv.first);
            std::sort(keys.begin(), keys.end());
            for (auto seq : keys) {
                auto &m = stage_marks_[seq];
                if (m.bg) fout2 << seq << ",BeginGather," << m.bg << ",0\n";
                if (m.ga) fout2 << seq << ",BeginApply," << m.ga << ",0\n";
                if (m.bs) fout2 << seq << ",BeginScatter," << m.bs << ",0\n";
                if (m.es) {
                    uint64_t spikes = 0;
                    auto it = window_spikes_pe_.find(seq);
                    if (it != window_spikes_pe_.end()) spikes = it->second;
                    fout2 << seq << ",EndScatter," << m.es << "," << spikes << "\n";
                }
            }
            fout2.close();
        }
    }

    // 节点结果摘要（可选）
    if (print_node_summary_) {
        uint64_t agg_spikes = 0;
        uint64_t agg_fired = 0;
        for (int i = 0; i < num_cores_; i++) {
            agg_spikes += unit_states_[i].spikes_processed;
            agg_fired  += unit_states_[i].neurons_fired;
        }
        PE_LOG(1, "NODE%d: 脉冲=%lu, 激发=%lu\n", node_id_, agg_spikes, agg_fired);
    }
    
    // 转发finish到所有子核心（确保子组件的收尾统计/摘要被打印与收集）
    for (auto* core : cores_) {
        if (core) core->finish();
    }

    // NoC e2e latency summary (cycles): printed once per node for native multicast experiments.
    {
        auto percentile = [](const auto& hist, double q) -> uint64_t {
            if (q <= 0.0) return 0;
            uint64_t total = 0;
            for (uint64_t c : hist) total += c;
            if (total == 0) return 0;
            const uint64_t need = static_cast<uint64_t>(std::ceil(q * static_cast<double>(total)));
            uint64_t acc = 0;
            for (uint64_t i = 0; i < hist.size(); ++i) {
                acc += hist[i];
                if (acc >= need) return i;
            }
            return static_cast<uint64_t>(hist.size() - 1);
        };

        const double spike_avg = (noc_lat_spike_cnt_ > 0) ? (double)noc_lat_spike_sum_ / (double)noc_lat_spike_cnt_ : 0.0;
        const double sk_avg = (noc_lat_spikekey_cnt_ > 0) ? (double)noc_lat_spikekey_sum_ / (double)noc_lat_spikekey_cnt_ : 0.0;
        const uint64_t spike_p50 = percentile(noc_lat_spike_hist_, 0.50);
        const uint64_t spike_p95 = percentile(noc_lat_spike_hist_, 0.95);
        const uint64_t spike_p99 = percentile(noc_lat_spike_hist_, 0.99);
        const uint64_t sk_p50 = percentile(noc_lat_spikekey_hist_, 0.50);
        const uint64_t sk_p95 = percentile(noc_lat_spikekey_hist_, 0.95);
        const uint64_t sk_p99 = percentile(noc_lat_spikekey_hist_, 0.99);
        const uint64_t spike_overflow = (!noc_lat_spike_hist_.empty()) ? noc_lat_spike_hist_.back() : 0;
        const uint64_t sk_overflow = (!noc_lat_spikekey_hist_.empty()) ? noc_lat_spikekey_hist_.back() : 0;

        // Noc end-to-end latency summary is a diagnostic; keep default runs quiet.
        output_->verbose(
            CALL_INFO, 1, 0,
            "[noc-lat] node=%d spike_cnt=%" PRIu64 " spike_lat_sum=%" PRIu64 " spike_lat_max=%" PRIu64
            " spike_avg=%.3f spike_p50=%" PRIu64 " spike_p95=%" PRIu64 " spike_p99=%" PRIu64
            " spikekey_cnt=%" PRIu64 " spikekey_lat_sum=%" PRIu64 " spikekey_lat_max=%" PRIu64
            " spikekey_avg=%.3f spikekey_p50=%" PRIu64 " spikekey_p95=%" PRIu64 " spikekey_p99=%" PRIu64
            " hist_max=%u spike_overflow=%" PRIu64 " spikekey_overflow=%" PRIu64 "\n",
            node_id_,
            noc_lat_spike_cnt_,
            noc_lat_spike_sum_,
            noc_lat_spike_max_,
            spike_avg,
            spike_p50,
            spike_p95,
            spike_p99,
            noc_lat_spikekey_cnt_,
            noc_lat_spikekey_sum_,
            noc_lat_spikekey_max_,
            sk_avg,
            sk_p50,
            sk_p95,
            sk_p99,
            noc_lat_hist_max_,
            spike_overflow,
            sk_overflow);
    }

    // 输出时间窗口化统计CSV（如已启用并指定路径）
    if (window_stats_enable_ && !window_csv_.empty()) {
        writeWindowCsv_();
    }

    // 调用网络接口的finish
    if (external_nic_) {
        external_nic_->finish();
    }
    // 注意：当使用 SST 引擎 stop-at 结束仿真时，finish() 期间再次触发 OKToEndSim
    // 会导致 Exit 事件被重复调度并在引擎侧触发双重释放；组件主控停止（sim_stop_ns_>0）
    // 已由 clockTick() 在阈值处触发 OKToEndSim。
}

void MultiCorePE::maybeInjectTestTraffic_(Cycle_t current_cycle) {
    // 0b. 测试注入：在首个有效周期从 core0 向 core1 注入一个跨核脉冲（仅当启用测试流量时）
    if (!(enable_test_traffic_ && !test_injected_ && num_cores_ > 1 && current_cycle == 5000)) return;

    // 构造一个从本PE core0 -> core1 的跨核脉冲（使用全局ID口径）
    const uint32_t src_global = static_cast<uint32_t>(global_neuron_base_);
    const uint32_t dst_global =
        static_cast<uint32_t>(static_cast<uint64_t>(global_neuron_base_) + static_cast<uint64_t>(neurons_per_core_));
    SpikeEvent* test_spike = new SpikeEvent(src_global, dst_global, static_cast<uint32_t>(node_id_), 0.5f, current_cycle_);
    int src_core = determineTargetUnit(static_cast<int>(src_global));
    int dst_core = determineTargetUnit(static_cast<int>(dst_global));
    if (src_core >= 0 && dst_core >= 0 && src_core != dst_core) {
        spike_packet_bridge_.sendAuto(test_spike);
        PE_LOG(1, "🧪 注入跨核脉冲: 核心%d->核心%d\n", src_core, dst_core);
        test_injected_ = true;
    } else {
        delete test_spike;
        test_injected_ = true;
    }
}

void MultiCorePE::driveCoresManually_(Cycle_t current_cycle) {
    // 2. SubComponent时钟由SST自动管理，无需手动调用tick
    // 若子组件未被SST调度（某些环境组合下可能发生），则回退为手动驱动一拍，确保窗口推进与队列消费
    if (!manual_core_drive_enable_) return;

    for (int i = 0; i < num_cores_; i++) {
        if (cores_[i] == nullptr) continue;
        if (auto* hooks = dynamic_cast<ICoreControlHooks*>(cores_[i])) {
            hooks->driveOneCycle();
        }
    }
}

void MultiCorePE::refreshProcessingUnitStates_() {
    static const std::map<std::string, uint64_t> kEmptyStats;
    // 更新处理单元状态统计（从SnnPE SubComponent获取实际数据）
    for (int i = 0; i < num_cores_; i++) {
        if (cores_[i] != nullptr) {
            std::map<std::string, uint64_t> core_stats;
            cores_[i]->getStatistics(core_stats);
            auto it_sp = core_stats.find("spikes_received");
            auto it_nf = core_stats.find("neurons_fired");
            uint64_t old_spikes = unit_states_[i].spikes_processed;
            uint64_t new_spikes = (it_sp != core_stats.end()) ? it_sp->second : 0;
            unit_states_[i].spikes_processed = new_spikes;
            unit_states_[i].neurons_fired = (it_nf != core_stats.end()) ? it_nf->second : 0;
            unit_states_[i].utilization = cores_[i]->getUtilization();
            unit_states_[i].is_active = cores_[i]->hasWork();
            for (auto& mod : workload_stats_modules_) {
                if (mod) mod->refreshCore(static_cast<size_t>(i), core_stats);
            }

            // 调试：跟踪统计数据变化 (已禁用避免过多输出)
            // if (new_spikes != old_spikes) {
            //     printf("DEBUG: 核心%d统计更新，节点%d - 旧值:%lu -> 新值:%lu (来自getStatistics)\n",
            //            i, node_id_, old_spikes, new_spikes);
            //     fflush(stdout);
            // }
            (void)old_spikes;
        } else {
            unit_states_[i].spikes_processed = 0;
            unit_states_[i].neurons_fired = 0;
            unit_states_[i].utilization = 0.0;
            unit_states_[i].is_active = false;
            for (auto& mod : workload_stats_modules_) {
                if (mod) mod->refreshCore(static_cast<size_t>(i), kEmptyStats);
            }
        }
    }
}

bool MultiCorePE::clockTick(Cycle_t current_cycle) {
    current_cycle_ = current_cycle;
    // 当启用组件主控停止时，到达阈值立刻OKToEndSim并停止本组件时钟
    if (sim_stop_ns_ > 0 && current_cycle_ >= sim_stop_ns_) {
        if (!ok_to_end_sent_) {
            primaryComponentOKToEndSim();
            ok_to_end_sent_ = true;
        }
        return false;
    }
    
    // 详细调试信息（仅在高详细度时输出）
    if (!first_tick_logged_) {
        if (sentinel_enabled_) {
            PE_LOG(1, "[[sentinel-pe-clock]] node=%d first_tick cyc=%" PRIu64 "\n", node_id_, (uint64_t)current_cycle_);
        }
    PE_LOG(2, "[diag-PE] clockTick start node=%d\n", node_id_);
        first_tick_logged_ = true;
    }

    // 进度心跳：默认关闭，仅在调试时开启，帮助定位仿真是否在推进/卡住
    if (progress_log_interval_ns_ > 0) {
        const bool node_ok = (progress_log_node_ < 0) || (progress_log_node_ == node_id_);
        if (node_ok && current_cycle_ > 0 && (static_cast<uint64_t>(current_cycle_) % progress_log_interval_ns_ == 0)) {
            PE_LOG(0, "[progress] node=%d cyc=%" PRIu64 " extq=%zu\n",
                   node_id_, (uint64_t)current_cycle_, noc_subsys_.incomingQueueSize());
        }
    }

    // Step barrier readiness gating (loader_done_key):
    // Avoid starting step-limited injections before WeightLoader/BCSR rowptr prefetch can proceed.
    if (global_step_sync_enable_ && global_step_sync_ready_ && gas_step_ctrl_link_ && !global_step_ready_sent_ && wait_for_loader_done_) {
        // Diagnostic (targeted): if step-limited hangs under MPI partitioning, the most common cause is that
        // WeightLoader and MultiCorePE are placed on different ranks, making SharedArray-based loader_done
        // gating invisible. Keep this extremely low-noise: only PE0 prints, rate-limited.
        static uint32_t diag_wait_prints = 0;
        const uint64_t now = static_cast<uint64_t>(current_cycle_);
        if (!loader_ready_latched_) {
            if (!loader_wait_started_) {
                loader_wait_started_ = true;
                loader_wait_start_cycle_ = now;
            }
            if (output_ && node_id_ == 0 && verbose_ >= 1 && (now % 10000u) == 0u && diag_wait_prints < 16u) {
                output_->verbose(
                    CALL_INFO, 1, 0,
                    "[[sentinel-step-sync]] node=%d waiting loader_done_key=%s shared_init=%d shared_ptr=%p\n",
                    node_id_,
                    loader_done_key_.c_str(),
                    (loader_done_shared_ ? 1 : 0),
                    (void*)loader_done_shared_.get());
                diag_wait_prints += 1;
            }
        }
        if (!loader_ready_latched_) {
            if (loader_done_shared_ && loader_done_shared_->size() > 0) {
                const int ready = loader_done_shared_->mutex_read(0);
                if (ready != 0) {
                    loader_ready_latched_ = true;
                    loader_ready_cycle_ = static_cast<uint64_t>(current_cycle_);
                    if (output_ && node_id_ == 0 && verbose_ >= 1) {
                        output_->verbose(
                            CALL_INFO, 1, 0,
                            "[[sentinel-step-sync]] node=%d loader_done latched (ready=%d) at cyc=%" PRIu64 "\n",
                            node_id_, ready, (uint64_t)loader_ready_cycle_);
                    }
                }
            }
        }
        if (!loader_ready_latched_ && loader_done_timeout_cycles_ > 0 && !loader_ready_forced_) {
            if (!loader_wait_started_) {
                loader_wait_started_ = true;
                loader_wait_start_cycle_ = now;
            }
            const uint64_t waited = (now >= loader_wait_start_cycle_) ? (now - loader_wait_start_cycle_) : 0;
            if (waited >= loader_done_timeout_cycles_) {
                auto* ev = new GasStepBarrierEvent(GasStepBarrierOp::PeReady, /*seq*/0, static_cast<uint32_t>(node_id_));
                gas_step_ctrl_link_->send(ev);
                global_step_ready_sent_ = true;
                loader_ready_forced_ = true;
                if (stat_loader_done_timeout_fallback_total_) stat_loader_done_timeout_fallback_total_->addData(1);
                if (output_) {
                    output_->verbose(
                        CALL_INFO, 0, 0,
                        "WARN: node=%d loader_done_key=%s timeout waited=%" PRIu64 " cycles, send PE_READY fallback\n",
                        node_id_, loader_done_key_.c_str(), (uint64_t)waited);
                }
            }
        }
        if (loader_ready_latched_) {
            const uint64_t since = (now >= loader_ready_cycle_) ? (now - loader_ready_cycle_) : 0;
            if (global_step_ready_delay_cycles_ == 0 || since >= global_step_ready_delay_cycles_) {
                auto* ev = new GasStepBarrierEvent(GasStepBarrierOp::PeReady, /*seq*/0, static_cast<uint32_t>(node_id_));
                gas_step_ctrl_link_->send(ev);
                global_step_ready_sent_ = true;
                if (output_ && node_id_ == 0 && verbose_ >= 1) {
                    output_->verbose(
                        CALL_INFO, 1, 0,
                        "[[sentinel-step-sync]] node=%d send PE_READY after loader_done (since=%" PRIu64 " delay=%" PRIu64 ")\n",
                        node_id_, (uint64_t)since, (uint64_t)global_step_ready_delay_cycles_);
                }
            }
        }
    }

    // Global Step barrier: 当收到 START_STEP(seq) 时，在时钟边界打开所有 core 的新窗口
    if (global_step_sync_enable_ && global_step_sync_ready_ && global_step_start_pending_) {
        const bool can_start = (global_step_active_seq_ == 0) || global_step_done_sent_;
        if (can_start) {
            beginGlobalStep_(global_step_pending_seq_);
            global_step_start_pending_ = false;
        }
    }
    
    // 0a. Step 注入调度（Phase3-B 下沉为 StepActivationSubsystem）
    step_activation_subsys_.tick(static_cast<uint64_t>(current_cycle_), getCurrentSimTimeNano());

    // 0b. 测试注入：在首个有效周期从 core0 向 core1 注入一个跨核脉冲（仅当启用测试流量时）
    maybeInjectTestTraffic_(current_cycle_);
    
    // 1. 处理外部脉冲队列（Phase4-A1.1：下沉至 NoC 子系统）
    noc_subsys_.drainIncomingQueue(static_cast<uint64_t>(current_cycle_));
    
    // 2. SubComponent时钟由SST自动管理，无需手动调用tick
    // 若子组件未被SST调度（某些环境组合下可能发生），则回退为手动驱动一拍，确保窗口推进与队列消费
    driveCoresManually_(current_cycle_);
    // 更新处理单元状态统计（从SnnPE SubComponent获取实际数据）
    refreshProcessingUnitStates_();
    
    // 3. 内部互连（ring）时钟滴答（Phase4-A1.1：由 NoC 子系统编排）
    noc_subsys_.tickRing(static_cast<uint64_t>(current_cycle));
    
    // 4. 多核控制器时钟滴答
    if (controller_) {
        controller_->tick();
        
        // 每100周期进行一次负载均衡检查
        if (current_cycle % 100 == 0) {
            checkLoadBalance();
        }
    }
    
    // 5. 生成测试流量
    if (enable_test_traffic_) {
        generateTestTraffic();
    }
    
    // 6. 更新统计信息（每1000周期一次）
    if (current_cycle % 1000 == 0) {
        updateStatistics();
    }

    // Global Step barrier（PE 侧 done policy）
    // - EndScatter policy: 在 notifyStageEvent() 处由 core 的 EndScatter(seq) 聚合触发。
    // - Drain/Quiescent/FixedCycles policy: 在此处基于“本 PE 是否仍有在途事务/是否静默足够久”触发。
    if (global_step_sync_enable_ &&
        gas_step_ctrl_link_ &&
        !global_step_done_sent_ &&
        global_step_active_seq_ != 0) {
        if (global_step_done_policy_ == GlobalStepDonePolicy::FixedCycles) {
            const uint64_t now = static_cast<uint64_t>(current_cycle_);
            const uint64_t since_begin = (now >= global_step_begin_cycle_) ? (now - global_step_begin_cycle_) : 0;
            if (since_begin >= global_step_fixed_cycles_) {
                auto* ev = new GasStepBarrierEvent(GasStepBarrierOp::PeDone,
                                                   global_step_active_seq_,
                                                   static_cast<uint32_t>(node_id_));
                // Step-level telemetry (best-effort): enables GlobalGasStepController criticality-aware control.
                // total: from BeginGather to PE_DONE send time (includes drain/quiescent wait for non-EndScatter policies)
                // apply: from BeginApply to BeginScatter (pure Apply duration, excludes drain)
                {
                    auto it = stage_marks_.find(global_step_active_seq_);
                    if (it != stage_marks_.end()) {
                        const auto& m = it->second;
                        const uint64_t now_ns = getCurrentSimTimeNano();
                        if (m.bg != 0 && now_ns >= m.bg) ev->step_total_ns = now_ns - m.bg;
                        if (m.ga != 0 && m.bs != 0 && m.bs >= m.ga) ev->step_apply_ns = m.bs - m.ga;
                    }
                }
                gas_step_ctrl_link_->send(ev);
                global_step_done_sent_ = true;
                if (stat_global_steps_done_total_) stat_global_steps_done_total_->addData(1);
                if (sentinel_enabled_ && output_ && global_step_active_seq_ == 1) {
                    output_->verbose(CALL_INFO, 2, 0,
                        "[[sentinel-step-sync]] node=%d send PE_DONE(fixed_cycles) seq=%u since_begin=%" PRIu64 "\n",
                        node_id_, global_step_active_seq_, (uint64_t)since_begin);
                }
            }
        } else if (global_step_done_policy_ == GlobalStepDonePolicy::Drain) {
            bool active = false;
            // 1) 必须等待本 step 的注入已发生，否则会在“同拍开始即完成”时误判。
            const bool injected = step_activation_subsys_.injectedForSeq(global_step_active_seq_);
            if (!injected) active = true;
            // 2) 显式检查 NoC/NIC/片上 ring 是否仍有在途包/背压排队。
            const bool noc_idle = noc_subsys_.isIdle();
            if (!noc_idle) active = true;

            // 3) step 语义等价：
            // - 对已经进入 BeginApply/EndApply/BeginScatter/EndScatter 的 core，必须等待该 core 到达 EndScatter(seq)。
            // - 对仅停在 BeginGather（或尚未产生阶段事件）的 core，回退为 hasWork()；否则“空 core 也必须 EndScatter”会把 step 永久卡住。
            bool uses_stage_events = false;
            bool all_end_scatter = false;
            int stage_progress_cores = 0;
            int stage_done_cores = 0;
            int stage_bg_only_cores = 0;
            int fallback_busy_cores = 0;
            if (global_step_last_stage_code_.size() == static_cast<size_t>(num_cores_) &&
                global_step_last_stage_seq_.size() == static_cast<size_t>(num_cores_)) {
                for (int i = 0; i < num_cores_; ++i) {
                    const size_t idx = static_cast<size_t>(i);
                    const bool have_stage =
                        (global_step_last_stage_seq_[idx] == global_step_active_seq_) &&
                        (global_step_last_stage_code_[idx] != 0);
                    const int stage_code = have_stage ? global_step_last_stage_code_[idx] : 0;

                    if (stage_code >= 2 /*BeginApply and later*/) {
                        uses_stage_events = true;
                        stage_progress_cores += 1;
                        if (stage_code == 5 /*EndScatter*/) {
                            stage_done_cores += 1;
                        } else {
                            active = true;
                        }
                        continue;
                    }

                    if (stage_code == 1 /*BeginGather*/) {
                        stage_bg_only_cores += 1;
                    }
                    if (cores_[i] && cores_[i]->hasWork()) {
                        fallback_busy_cores += 1;
                        active = true;
                    }
                }
                all_end_scatter = uses_stage_events && (stage_done_cores == stage_progress_cores);
            } else {
                // 非 window/GAS workload（或阶段镜像未初始化）：回退为 core 的 hasWork() 语义
                for (int i = 0; i < num_cores_; ++i) {
                    if (cores_[i] && cores_[i]->hasWork()) {
                        fallback_busy_cores += 1;
                        active = true;
                    }
                }
            }

            const bool hold_for_gather_completion =
                (!uses_stage_events && stage_bg_only_cores > 0 &&
                 global_step_last_stage_code_.size() == static_cast<size_t>(num_cores_) &&
                 global_step_last_stage_seq_.size() == static_cast<size_t>(num_cores_));
            if (hold_for_gather_completion) {
                // 仍停留在 Gather：必须等 workload 的 Gather->EndGather quiesce 规则自己推进，
                // 不能让 PE 级 Drain 先于空窗口/慢窗口的 BeginApply 收尾。
                active = true;
            }

            // Drain 卡点探针（仅 node0 / seq1 / sentinel 模式限量打印）：用于定位“为什么 step 不推进”。
            if (sentinel_enabled_ && output_ && node_id_ == 0 && global_step_active_seq_ == 1) {
                const uint64_t now = static_cast<uint64_t>(current_cycle_);
                if ((now % 10000u) == 0u && global_step_drain_diag_count_ < 64u) {
                    const size_t inq = noc_subsys_.incomingQueueSize();
                    const int ring_p = noc_subsys_.ringPendingMessageCount();
                    const size_t nic_p = noc_subsys_.nicPendingSendCount();
                    const uint64_t quiet_dbg = (now >= global_step_last_activity_cycle_)
                        ? (now - global_step_last_activity_cycle_)
                        : 0;
                    output_->verbose(CALL_INFO, 2, 0,
                        "[[sentinel-step-drain]] node=%d seq=%u injected=%d stage=%d all_es=%d progressed=%d done=%d bg_only=%d bg_hold=%d fallback_busy=%d noc_idle=%d inq=%zu ring=%d nic_pending=%zu quiet=%" PRIu64 " min=%" PRIu64 "\n",
                        node_id_,
                        global_step_active_seq_,
                        injected ? 1 : 0,
                        uses_stage_events ? 1 : 0,
                        all_end_scatter ? 1 : 0,
                        stage_progress_cores,
                        stage_done_cores,
                        stage_bg_only_cores,
                        hold_for_gather_completion ? 1 : 0,
                        fallback_busy_cores,
                        noc_idle ? 1 : 0,
                        inq,
                        ring_p,
                        nic_p,
                        (uint64_t)quiet_dbg,
                        (uint64_t)global_step_drain_min_cycles_);
                    global_step_drain_diag_count_ += 1;
                }
            }

            if (active) {
                global_step_last_activity_cycle_ = static_cast<uint64_t>(current_cycle_);
            } else {
                const uint64_t now = static_cast<uint64_t>(current_cycle_);
                const uint64_t last = global_step_last_activity_cycle_;
                const uint64_t quiet = (now >= last) ? (now - last) : 0;
                if (quiet >= global_step_drain_min_cycles_) {
                    auto* ev = new GasStepBarrierEvent(GasStepBarrierOp::PeDone,
                                                       global_step_active_seq_,
                                                       static_cast<uint32_t>(node_id_));
                    // Step-level telemetry (best-effort): enables GlobalGasStepController criticality-aware control.
                    {
                        auto it = stage_marks_.find(global_step_active_seq_);
                        if (it != stage_marks_.end()) {
                            const auto& m = it->second;
                            const uint64_t now_ns = getCurrentSimTimeNano();
                            if (m.bg != 0 && now_ns >= m.bg) ev->step_total_ns = now_ns - m.bg;
                            if (m.ga != 0 && m.bs != 0 && m.bs >= m.ga) ev->step_apply_ns = m.bs - m.ga;
                        }
                    }
                    gas_step_ctrl_link_->send(ev);
                    global_step_done_sent_ = true;
                    if (stat_global_steps_done_total_) stat_global_steps_done_total_->addData(1);
                    if (sentinel_enabled_ && output_ && global_step_active_seq_ == 1) {
                        output_->verbose(CALL_INFO, 2, 0,
                            "[[sentinel-step-sync]] node=%d send PE_DONE(drain) seq=%u quiet=%" PRIu64 " (min=%" PRIu64 ")\n",
                            node_id_, global_step_active_seq_, (uint64_t)quiet, (uint64_t)global_step_drain_min_cycles_);
                    }
                }
            }
        } else if (global_step_done_policy_ == GlobalStepDonePolicy::Quiescent) {
            bool active = false;
            if (!step_activation_subsys_.injectedForSeq(global_step_active_seq_)) active = true;
            if (!noc_subsys_.isIdle()) active = true;
            for (int i = 0; i < num_cores_; ++i) {
                if (cores_[i] && cores_[i]->hasWork()) {
                    active = true;
                    break;
                }
            }

            if (active) {
                global_step_last_activity_cycle_ = static_cast<uint64_t>(current_cycle_);
            } else {
                const uint64_t now = static_cast<uint64_t>(current_cycle_);
                const uint64_t last = global_step_last_activity_cycle_;
                const uint64_t quiet = (now >= last) ? (now - last) : 0;
                if (quiet >= global_step_quiescent_min_cycles_) {
                    auto* ev = new GasStepBarrierEvent(GasStepBarrierOp::PeDone,
                                                       global_step_active_seq_,
                                                       static_cast<uint32_t>(node_id_));
                    // Step-level telemetry (best-effort): enables GlobalGasStepController criticality-aware control.
                    {
                        auto it = stage_marks_.find(global_step_active_seq_);
                        if (it != stage_marks_.end()) {
                            const auto& m = it->second;
                            const uint64_t now_ns = getCurrentSimTimeNano();
                            if (m.bg != 0 && now_ns >= m.bg) ev->step_total_ns = now_ns - m.bg;
                            if (m.ga != 0 && m.bs != 0 && m.bs >= m.ga) ev->step_apply_ns = m.bs - m.ga;
                        }
                    }
                    gas_step_ctrl_link_->send(ev);
                    global_step_done_sent_ = true;
                    if (stat_global_steps_done_total_) stat_global_steps_done_total_->addData(1);
                    if (sentinel_enabled_ && output_ && global_step_active_seq_ == 1) {
                        output_->verbose(CALL_INFO, 2, 0,
                            "[[sentinel-step-sync]] node=%d send PE_DONE(quiescent) seq=%u quiet=%" PRIu64 "\n",
                            node_id_, global_step_active_seq_, (uint64_t)quiet);
                    }
                }
            }
        }
    }
    
    // 时钟事件处理，让外部组件有机会基于周期推进
    // 继续仿真
    return false;
}

void MultiCorePE::handleExternalSpikeEvent(SST::Event* ev) {
    if (!ev) return;
    // 外部端口事件解析：NocPacketEvent 走 NoC；SpikeEvent 走 Stimulus 的直注入（仅本地投递，语义冻结）。
    if (auto* spike = dynamic_cast<SpikeEvent*>(ev)) {
        handleExternalSpike(spike);
        return;
    }
    noc_subsys_.onExternalPortEvent(ev);
}

void MultiCorePE::handleLoaderDoneEvent(SST::Event* ev) {
    if (!ev) return;
    auto* msg = dynamic_cast<LoaderDoneEvent*>(ev);
    if (msg) {
        // Cross-rank bridge:
        // - SharedArray(loader_done_key) 在 MPI 多 rank 下不具备运行期一致性；
        // - 该事件作为权重就绪的权威信号：MultiCorePE 侧先 latch，再镜像写入本 rank 的 SharedArray
        //   以复用 core/workload 现有 ensureLoaderReady_ 逻辑（避免引入新接口/破坏兼容）。
        loader_ready_latched_ = true;
        // current_cycle_ 仅在 clockTick() 里更新；init/setup 阶段可能尚未进入时钟。
        loader_ready_cycle_ = first_tick_logged_ ? static_cast<uint64_t>(current_cycle_) : 0;
        // 注意：SharedArray 仅允许在 init 阶段写入；跨 rank 运行期镜像写入会触发 SST fatal。
        // 因此：通过 ILoaderReadyHooks 将“就绪”信号下发到各 core 的本地 latch，避免依赖 SharedArray 的一致性。
        for (auto* core : cores_) {
            if (!core) continue;
            if (auto* hook = dynamic_cast<ILoaderReadyHooks*>(core)) {
                hook->onLoaderReady();
            }
        }
        if (sentinel_enabled_ && output_ && verbose_ >= 1) {
            output_->verbose(CALL_INFO, 1, 0,
                             "[[sentinel-step-sync]] node=%d recv LoaderDoneEvent(src_node=%u) latch loader_ready\n",
                             node_id_, msg->src_node);
        }
    }
    delete ev;
}

void MultiCorePE::submitTassLfP0WindowReport(const TassLfP0WindowReport& report) {
    if (!tass_lf_p0_enable_) return;
    if (stat_gas_tass_lf_p0_reports_submit_total_) stat_gas_tass_lf_p0_reports_submit_total_->addData(1);
    uint32_t block_origin_node = 0;
    uint32_t expected_contributors = 1;
    computeTassBlockInfo_(report, block_origin_node, expected_contributors);
    if (block_origin_node == static_cast<uint32_t>(node_id_)) {
        reduceTassLfP0WindowReport_(report);
        return;
    }
    if (tass_p0_out_link_) {
        if (stat_gas_tass_lf_p0_reports_send_total_) stat_gas_tass_lf_p0_reports_send_total_->addData(1);
        tass_p0_out_link_->send(new TassLfP0ReportEvent(report));
    } else {
        if (stat_gas_tass_lf_p0_reports_drop_no_link_total_) stat_gas_tass_lf_p0_reports_drop_no_link_total_->addData(1);
    }
}

void MultiCorePE::submitTassNaiveWindowRequest(const TassNaiveWindowRequest& request) {
    if (!naive_tass_enable_) {
        if (!request.entries.empty()) {
            output_->fatal(CALL_INFO, -1,
                           "MultiCorePE fatal: received naive_tass window request while experimental_naive_tass_enable=0 "
                           "(node=%d source_node=%u source_core=%u entries=%zu window=%u)\n",
                           node_id_, request.source_node, request.source_core, request.entries.size(), request.window_seq);
        }
        return;
    }
    uint32_t block_origin_node = 0;
    uint32_t expected_contributors = 1;
    computeTassBlockInfo_(request, block_origin_node, expected_contributors);
    if (block_origin_node == static_cast<uint32_t>(node_id_)) {
        handleTassNaiveWindowRequest_(request);
        return;
    }
    if (!tass_p0_out_link_) {
        output_->fatal(CALL_INFO, -1,
                       "MultiCorePE fatal: naive_tass request missing tass_p0_out link on node=%d source_core=%u\n",
                       node_id_, request.source_core);
    }
    tass_p0_out_link_->send(new TassNaiveWindowRequestEvent(request));
}

void MultiCorePE::handleTassLfP0ReportEvent(SST::Event* ev) {
    if (!ev) return;
    if (auto* msg = dynamic_cast<TassLfP0ReportEvent*>(ev)) {
        if (stat_gas_tass_lf_p0_reports_recv_total_) stat_gas_tass_lf_p0_reports_recv_total_->addData(1);
        reduceTassLfP0WindowReport_(msg->report);
        delete ev;
        return;
    }
    if (auto* req = dynamic_cast<TassNaiveWindowRequestEvent*>(ev)) {
        handleTassNaiveWindowRequest_(req->request);
        delete ev;
        return;
    }
    if (auto* rsp = dynamic_cast<TassNaiveResponseEvent*>(ev)) {
        deliverTassNaiveResponsesLocal_(rsp->entries);
        delete ev;
        return;
    }
    delete ev;
}

void MultiCorePE::handleTassNaiveWindowRequest_(const TassNaiveWindowRequest& request) {
    if (!naive_tass_enable_) {
        if (!request.entries.empty()) {
            output_->fatal(CALL_INFO, -1,
                           "MultiCorePE fatal: TassNaiveWindowRequestEvent arrived on node with naive_tass disabled "
                           "(node=%d source_node=%u source_core=%u entries=%zu window=%u)\n",
                           node_id_, request.source_node, request.source_core, request.entries.size(), request.window_seq);
        }
        return;
    }
    if (request.entries.empty()) return;
    if (!memory_interface_) {
        output_->fatal(CALL_INFO, -1,
                       "MultiCorePE fatal: naive_tass request arrived without memory_interface on node=%d\n",
                       node_id_);
    }
    uint32_t block_origin_node = 0;
    uint32_t expected_contributors = 1;
    computeTassBlockInfo_(request, block_origin_node, expected_contributors);
    if (block_origin_node != static_cast<uint32_t>(node_id_)) {
        output_->fatal(CALL_INFO, -1,
                       "MultiCorePE fatal: naive_tass request routed to non-origin node=%d expected_origin=%u src_node=%u\n",
                       node_id_, block_origin_node, request.source_node);
    }
    const uint32_t line_size = std::max<uint32_t>(1u, request.line_size_bytes);
    for (const auto& entry : request.entries) {
        const uint64_t addr = base_addr_ + static_cast<uint64_t>(entry.widx) * sizeof(float);
        const uint64_t line_addr = (addr / static_cast<uint64_t>(line_size)) * static_cast<uint64_t>(line_size);
        const uint32_t byte_offset = static_cast<uint32_t>(addr - line_addr);
        auto& state = tass_naive_lines_[line_addr];
        if (state.window_seq != request.window_seq) {
            state = TassNaiveLineState{};
            state.window_seq = request.window_seq;
            state.line_size_bytes = line_size;
        }
        if (!state.inflight && !state.data.empty()) {
            TassNaiveResponseEntry rsp{};
            rsp.dst_core = request.source_core;
            rsp.window_seq = request.window_seq;
            rsp.retire_seq = entry.retire_seq;
            rsp.weight = loadTassNaiveWeight_(state.data, byte_offset);
            std::vector<TassNaiveResponseEntry> local_entries{};
            local_entries.push_back(rsp);
            dispatchTassNaiveResponses_(request.source_node, local_entries);
            continue;
        }
        TassNaiveLineWaiter waiter{};
        waiter.dst_node = request.source_node;
        waiter.dst_core = request.source_core;
        waiter.window_seq = request.window_seq;
        waiter.retire_seq = entry.retire_seq;
        waiter.byte_offset = byte_offset;
        state.waiters.push_back(waiter);
        if (state.inflight) continue;
        state.window_seq = request.window_seq;
        state.line_size_bytes = line_size;
        state.inflight = true;
        auto* rd = new SST::Interfaces::StandardMem::Read(line_addr, line_size);
        const auto req_id = rd->getID();
        tass_naive_req_to_line_[req_id] = line_addr;
        memory_interface_->send(rd);
    }
}

void MultiCorePE::handleTassNaiveMemoryResponse_(SST::Interfaces::StandardMem::Request* req) {
    if (!req) return;
    auto* rr = dynamic_cast<SST::Interfaces::StandardMem::ReadResp*>(req);
    if (!rr) {
        delete req;
        return;
    }
    const auto it = tass_naive_req_to_line_.find(req->getID());
    if (it == tass_naive_req_to_line_.end()) {
        output_->fatal(CALL_INFO, -1,
                       "MultiCorePE fatal: naive_tass memory response with unknown request id "
                       "(node=%d req_id=%" PRIu64 " inflight_req_count=%zu)\n",
                       node_id_, static_cast<uint64_t>(req->getID()), tass_naive_req_to_line_.size());
    }
    const uint64_t line_addr = it->second;
    tass_naive_req_to_line_.erase(it);
    auto line_it = tass_naive_lines_.find(line_addr);
    if (line_it == tass_naive_lines_.end()) {
        output_->fatal(CALL_INFO, -1,
                       "MultiCorePE fatal: naive_tass memory response line state missing "
                       "(node=%d line_addr=0x%" PRIx64 " req_id=%" PRIu64 ")\n",
                       node_id_, line_addr, static_cast<uint64_t>(req->getID()));
    }
    auto& state = line_it->second;
    state.inflight = false;
    state.data = rr->data;
    std::unordered_map<uint32_t, std::vector<TassNaiveResponseEntry>> by_node{};
    for (const auto& waiter : state.waiters) {
        TassNaiveResponseEntry entry{};
        entry.dst_core = waiter.dst_core;
        entry.window_seq = waiter.window_seq;
        entry.retire_seq = waiter.retire_seq;
        entry.weight = loadTassNaiveWeight_(state.data, waiter.byte_offset);
        by_node[waiter.dst_node].push_back(entry);
    }
    state.waiters.clear();
    for (auto& kv : by_node) {
        dispatchTassNaiveResponses_(kv.first, kv.second);
    }
    delete req;
}

void MultiCorePE::deliverTassNaiveResponsesLocal_(const std::vector<TassNaiveResponseEntry>& entries) {
    if (entries.empty()) return;
    std::unordered_map<uint32_t, std::vector<TassNaiveResponseEntry>> by_core{};
    for (const auto& entry : entries) {
        by_core[entry.dst_core].push_back(entry);
    }
    for (auto& kv : by_core) {
        const uint32_t core_id = kv.first;
        if (core_id >= static_cast<uint32_t>(cores_.size()) || !cores_[core_id]) {
            output_->fatal(CALL_INFO, -1,
                           "MultiCorePE fatal: naive_tass response target core out of range node=%d core=%u size=%zu\n",
                           node_id_, core_id, cores_.size());
        }
        auto* sink = dynamic_cast<ITassNaiveResponseSink*>(cores_[core_id]);
        if (!sink) {
            output_->fatal(CALL_INFO, -1,
                           "MultiCorePE fatal: core=%u on node=%d does not implement ITassNaiveResponseSink\n",
                           core_id, node_id_);
        }
        sink->onTassNaiveResponses(kv.second);
    }
}

void MultiCorePE::dispatchTassNaiveResponses_(uint32_t dst_node, std::vector<TassNaiveResponseEntry>& entries) {
    if (entries.empty()) return;
    if (dst_node == static_cast<uint32_t>(node_id_)) {
        deliverTassNaiveResponsesLocal_(entries);
        entries.clear();
        return;
    }
    if (dst_node >= tass_p0_rsp_out_links_.size() || !tass_p0_rsp_out_links_[dst_node]) {
        output_->fatal(CALL_INFO, -1,
                       "MultiCorePE fatal: naive_tass response link missing node=%d dst_node=%u entries=%zu\n",
                       node_id_, dst_node, entries.size());
    }
    tass_p0_rsp_out_links_[dst_node]->send(new TassNaiveResponseEvent(static_cast<uint32_t>(node_id_), std::move(entries)));
    entries.clear();
}

void MultiCorePE::reduceTassLfP0WindowReport_(const TassLfP0WindowReport& report) {
    if (!tass_lf_p0_enable_) return;
    if (stat_gas_tass_lf_p0_reports_reduce_total_) stat_gas_tass_lf_p0_reports_reduce_total_->addData(1);
    uint32_t block_origin_node = 0;
    uint32_t expected_contributors = 1;
    computeTassBlockInfo_(report, block_origin_node, expected_contributors);
    if (block_origin_node != static_cast<uint32_t>(node_id_)) {
        if (stat_gas_tass_lf_p0_reports_drop_non_origin_total_) stat_gas_tass_lf_p0_reports_drop_non_origin_total_->addData(1);
        return;
    }
    const uint64_t epoch_key = packTassBlockEpochKey_(block_origin_node, report.window_seq);
    const uint64_t contributor_key = packTassContributorKey_(report.node_id, report.core_id);
    auto& epoch = tass_lf_p0_pending_epochs_[epoch_key];
    if (epoch.expected_contributors == 0) {
        epoch.expected_contributors = expected_contributors;
        epoch.line_size_bytes = std::max<uint32_t>(1u, report.line_size_bytes);
    }
    const auto inserted = epoch.contributors.insert(contributor_key);
    if (!inserted.second) {
        return;
    }
    epoch.payload_bytes_total += report.payload_bytes_total;
    epoch.current_vlf_line_groups_total += report.current_vlf_line_groups_total;
    for (const auto& kv : report.pre_payload_entries) {
        epoch.pre_payload_bytes[kv.pre_global] += kv.payload_bytes;
        epoch.pre_contributors[kv.pre_global] += 1u;
    }
    if (epoch.contributors.size() < static_cast<size_t>(epoch.expected_contributors)) {
        return;
    }
    const uint64_t line_size = std::max<uint64_t>(1ull, static_cast<uint64_t>(epoch.line_size_bytes));
    uint64_t block_active_pres_total = static_cast<uint64_t>(epoch.pre_payload_bytes.size());
    uint64_t block_shared_pres_total = 0;
    uint64_t cross_core_joins_total = 0;
    uint64_t block_naive_line_count_total = 0;
    uint64_t response_fanout_total = 0;
    for (const auto& kv : epoch.pre_payload_bytes) {
        block_naive_line_count_total += ceilDivU64_(kv.second, line_size);
    }
    for (const auto& kv : epoch.pre_contributors) {
        const uint64_t fanout = static_cast<uint64_t>(kv.second);
        response_fanout_total += fanout;
        if (fanout > 1) {
            block_shared_pres_total += 1;
            cross_core_joins_total += (fanout - 1);
        }
    }
    if (stat_gas_tass_lf_p0_block_epochs_total_) stat_gas_tass_lf_p0_block_epochs_total_->addData(1);
    if (stat_gas_tass_lf_p0_block_active_pres_total_) stat_gas_tass_lf_p0_block_active_pres_total_->addData(block_active_pres_total);
    if (stat_gas_tass_lf_p0_block_shared_pres_total_) stat_gas_tass_lf_p0_block_shared_pres_total_->addData(block_shared_pres_total);
    if (stat_gas_tass_lf_p0_cross_core_joins_total_) stat_gas_tass_lf_p0_cross_core_joins_total_->addData(cross_core_joins_total);
    if (stat_gas_tass_lf_p0_payload_bytes_total_) stat_gas_tass_lf_p0_payload_bytes_total_->addData(epoch.payload_bytes_total);
    if (stat_gas_tass_lf_p0_current_vlf_line_groups_total_) stat_gas_tass_lf_p0_current_vlf_line_groups_total_->addData(epoch.current_vlf_line_groups_total);
    if (stat_gas_tass_lf_p0_block_naive_line_count_total_) stat_gas_tass_lf_p0_block_naive_line_count_total_->addData(block_naive_line_count_total);
    if (stat_gas_tass_lf_p0_block_fused_lb_line_count_total_) stat_gas_tass_lf_p0_block_fused_lb_line_count_total_->addData(ceilDivU64_(epoch.payload_bytes_total, line_size));
    if (stat_gas_tass_lf_p0_response_fanout_total_) stat_gas_tass_lf_p0_response_fanout_total_->addData(response_fanout_total);
    tass_lf_p0_pending_epochs_.erase(epoch_key);
}

void MultiCorePE::handleGasStepCtrlEvent(SST::Event* ev) {
    if (!ev) return;
    auto* msg = dynamic_cast<GasStepBarrierEvent*>(ev);
    if (!msg) {
        delete ev;
        return;
    }
    if (msg->operation() == GasStepBarrierOp::StartStep) {
        const uint32_t seq = msg->seq;
        const uint32_t active = global_step_active_seq_;
        const uint32_t pending = global_step_start_pending_ ? global_step_pending_seq_ : 0;
        const uint32_t last_seen = std::max(global_step_last_seen_seq_, std::max(active, pending));
        const uint32_t expected_next = (pending != 0) ? pending : ((active != 0) ? (active + 1u) : 0u);
        if (seq == 0) {
            global_step_start_stale_total_ += 1;
            if (output_) {
                output_->verbose(
                    CALL_INFO, 0, 0,
                    "WARN: node=%d recv START_STEP seq=0 ignored stale_total=%" PRIu64 "\n",
                    node_id_, (uint64_t)global_step_start_stale_total_);
            }
        } else if (seq == active || (pending != 0 && seq == pending)) {
            global_step_start_dup_total_ += 1;
            if (output_) {
                output_->verbose(
                    CALL_INFO, 0, 0,
                    "WARN: node=%d recv START_STEP dup seq=%" PRIu32 " active=%" PRIu32 " pending=%" PRIu32 " dup_total=%" PRIu64 "\n",
                    node_id_, seq, active, pending, (uint64_t)global_step_start_dup_total_);
            }
        } else if (last_seen != 0 && seq < last_seen) {
            global_step_start_stale_total_ += 1;
            if (output_) {
                output_->verbose(
                    CALL_INFO, 0, 0,
                    "WARN: node=%d recv START_STEP stale seq=%" PRIu32 " last_seen=%" PRIu32 " stale_total=%" PRIu64 "\n",
                    node_id_, seq, last_seen, (uint64_t)global_step_start_stale_total_);
            }
        } else {
            if (expected_next != 0 && seq > expected_next) {
                global_step_start_jump_total_ += 1;
                if (output_) {
                    output_->verbose(
                        CALL_INFO, 0, 0,
                        "WARN: node=%d recv START_STEP jump seq=%" PRIu32 " expected=%" PRIu32 " jump_total=%" PRIu64 "\n",
                        node_id_, seq, expected_next, (uint64_t)global_step_start_jump_total_);
                }
            }
            global_step_pending_seq_ = seq;
            global_step_pending_apply_bank_credit_target_ = msg->apply_bank_credit_target;
            global_step_start_pending_ = true;
            global_step_last_seen_seq_ = seq;
            if (sentinel_enabled_ && output_ && node_id_ == 0) {
                output_->verbose(CALL_INFO, 2, 0, "[[sentinel-step-sync]] node=%d recv START_STEP seq=%u\n", node_id_, seq);
            }
        }
    }
    delete msg;
}

void MultiCorePE::beginGlobalStep_(uint32_t seq) {
    tass_naive_lines_.clear();
    tass_naive_req_to_line_.clear();
    global_step_active_seq_ = seq;
    global_step_active_apply_bank_credit_target_ = global_step_pending_apply_bank_credit_target_;
    global_step_pending_apply_bank_credit_target_ = 0;
    global_step_done_sent_ = false;
    global_step_begin_cycle_ = static_cast<uint64_t>(current_cycle_);
    global_step_last_activity_cycle_ = static_cast<uint64_t>(current_cycle_);
    if (global_step_done_cores_.size() != static_cast<size_t>(num_cores_)) {
        global_step_done_cores_.assign(static_cast<size_t>(num_cores_), 0);
    } else {
        std::fill(global_step_done_cores_.begin(), global_step_done_cores_.end(), 0);
    }
    if (global_step_last_stage_code_.size() != static_cast<size_t>(num_cores_)) {
        global_step_last_stage_code_.assign(static_cast<size_t>(num_cores_), 0);
        global_step_last_stage_seq_.assign(static_cast<size_t>(num_cores_), 0);
        global_step_last_stage_ts_ns_.assign(static_cast<size_t>(num_cores_), 0);
        global_step_last_stage_spikes_.assign(static_cast<size_t>(num_cores_), 0);
    } else {
        std::fill(global_step_last_stage_code_.begin(), global_step_last_stage_code_.end(), 0);
        std::fill(global_step_last_stage_seq_.begin(), global_step_last_stage_seq_.end(), 0);
        std::fill(global_step_last_stage_ts_ns_.begin(), global_step_last_stage_ts_ns_.end(), 0);
        std::fill(global_step_last_stage_spikes_.begin(), global_step_last_stage_spikes_.end(), 0);
    }

    // 全局 step 同步：先让所有 core 进入 step（打开 Gather/窗口门控），再注入本 step 的 stimulus。
    // 否则注入可能发生在 core 仍处于 Idle 时，导致 recordEdge 被 gate 掉（Idle 默认不记录），
    // 进而出现 GAS 未读全权重/Apply 仅发出少量读请求（microbench 会被严重低估）。
    for (int i = 0; i < num_cores_; ++i) {
        auto* core = cores_[i];
        if (!core) {
            output_->fatal(CALL_INFO, -1, "GlobalStep fatal: core%d is null\n", i);
            return;
        }
        if (global_step_active_apply_bank_credit_target_ > 0) {
            if (auto* credit = dynamic_cast<IGlobalStepCreditHooks*>(core)) {
                credit->onGlobalStepApplyBankCredit(seq, global_step_active_apply_bank_credit_target_);
            }
        }
        auto* hook = dynamic_cast<IGlobalStepHooks*>(core);
        if (!hook) {
            output_->fatal(CALL_INFO, -1, "GlobalStep fatal: core%d does not implement IGlobalStepHooks\n", i);
            return;
        }
        if (sentinel_enabled_ && output_ && seq == 1) {
            output_->verbose(CALL_INFO, 2, 0, "[[sentinel-step-sync]] node=%d call onGlobalStepStart core=%d seq=%u\n", node_id_, i, seq);
        }
        hook->onGlobalStepStart(seq);
    }

    // naive_raw baseline：flush 上一 step 产生的“下一 step spike”，确保跨步传播但禁止步内级联。
    flushDeferredPacketsForSeq_(seq);

    // Step Random Activation：在全局 step 同步下，以 START_STEP(seq) 作为统一触发点（每 step 一次）。
    // 注意：必须在 core 进入 step 后再注入，确保所有触发的 spike 归属到本 step 的 Gather。
    step_activation_subsys_.onGlobalStepStart(seq, getCurrentSimTimeNano());

    if (sentinel_enabled_ && output_ && seq == 1) {
        output_->verbose(CALL_INFO, 2, 0, "[[sentinel-step-sync]] node=%d beginGlobalStep seq=%u cores=%d\n", node_id_, seq, num_cores_);
    }
}

void MultiCorePE::flushDeferredPacketsForSeq_(uint32_t seq) {
    if (deferred_packets_by_seq_.empty()) return;
    auto it = deferred_packets_by_seq_.find(seq);
    if (it == deferred_packets_by_seq_.end()) return;

    auto packets = std::move(it->second);
    deferred_packets_by_seq_.erase(it);

    for (auto& d : packets) {
        if (!d.pkt) continue;
        deliverPacketToEndpoint_(d.endpoint_id, d.pkt);
    }
}

void MultiCorePE::clearAllDeferredPackets_() {
    for (auto& kv : deferred_packets_by_seq_) {
        for (auto& d : kv.second) {
            delete d.pkt;
            d.pkt = nullptr;
        }
    }
    deferred_packets_by_seq_.clear();
}

void MultiCorePE::handleExternalSpike(SpikeEvent* spike) {
    external_spike_input_subsys_.onSpike(spike);
}

void MultiCorePE::sendExternalSpike(SpikeEvent* spike) {
    if (!spike) return;
    // Phase3-C：编解码/packet 化下沉到 SpikePacketBridge；NoC 仅做传输。
    spike_packet_bridge_.sendExternal(spike);
}

int MultiCorePE::determineTargetUnit(int neuron_id) const {
    // 使用global_neuron_base确定本节点管理的神经元范围
    int local_neuron_id = neuron_id - static_cast<int>(global_neuron_base_);
    
    if (local_neuron_id < 0 || local_neuron_id >= total_neurons_) {
        return -1;  // 非本MultiCorePE的神经元
    }
    
    int target_unit = local_neuron_id / neurons_per_core_;
    return (target_unit >= 0 && target_unit < num_cores_) ? target_unit : -1;
}

bool MultiCorePE::isLocalNeuron(int neuron_id) const {
    int start_id = static_cast<int>(global_neuron_base_);
    int end_id = start_id + total_neurons_;
    bool is_local = (neuron_id >= start_id && neuron_id < end_id);
    // printf("🔍 isLocalNeuron检查: 神经元%d, 范围[%d,%d), 节点%d, 结果:%s\n",
    //        neuron_id, start_id, end_id, node_id_, is_local ? "本地" : "非本地");
    // fflush(stdout);
    return is_local;
}

const ProcessingUnitState& MultiCorePE::getProcessingUnitState(int unit_id) const {
    static ProcessingUnitState empty_state;
    if (unit_id >= 0 && unit_id < num_cores_) {
        return unit_states_[unit_id];
    }
    return empty_state;
}

void MultiCorePE::getStatistics(std::map<std::string, uint64_t>& stats) const {
    stats["total_spikes_processed"] = stat_spikes_processed_->getCollectionCount();
    stats["inter_core_messages"] = stat_inter_core_messages_->getCollectionCount();
    stats["total_neurons_fired"] = stat_neurons_fired_->getCollectionCount();
    stats["external_spikes_sent"] = stat_external_spikes_sent_->getCollectionCount();
    stats["external_spikes_received"] = stat_external_spikes_received_->getCollectionCount();
    stats["current_cycle"] = current_cycle_;
}

void MultiCorePE::initializeStatistics() {
    
    stat_spikes_processed_ = registerStatistic<uint64_t>("total_spikes_processed");
    stat_inter_core_messages_ = registerStatistic<uint64_t>("inter_core_messages");
    stat_l2_hits_ = registerStatistic<uint64_t>("l2_cache_hits");
    stat_l2_misses_ = registerStatistic<uint64_t>("l2_cache_misses");
    stat_memory_requests_ = registerStatistic<uint64_t>("memory_requests");
    stat_avg_utilization_ = registerStatistic<double>("avg_core_utilization");
    stat_neurons_fired_ = registerStatistic<uint64_t>("total_neurons_fired");
    stat_unique_neurons_fired_total_ = registerStatistic<uint64_t>("unique_neurons_fired_total");
    stat_external_spikes_sent_ = registerStatistic<uint64_t>("external_spikes_sent");
    stat_external_spikes_received_ = registerStatistic<uint64_t>("external_spikes_received");
    // Experiment profile observability counters (do not change behavior)
    stat_compat_unexpected_stream_activity_total_ = registerStatistic<uint64_t>("compat_unexpected_stream_activity_total");
    stat_compat_naive_gas_stage_events_total_ = registerStatistic<uint64_t>("compat_naive_gas_stage_events_total");
    stat_compat_finish_incomplete_total_ = registerStatistic<uint64_t>("compat_finish_incomplete_total");
    stat_loader_done_timeout_fallback_total_ = registerStatistic<uint64_t>("loader_done_timeout_fallback_total");
    for (auto& mod : workload_stats_modules_) {
        if (!mod) continue;
        if (std::strcmp(mod->name(), "stream") == 0) {
            mod->bindUnexpectedActivityStat(stat_compat_unexpected_stream_activity_total_);
        }
        mod->initialize(*this, static_cast<size_t>(num_cores_));
    }
    // Batch-A: 注册组件级直方图统计（具体类型由Python侧设置为Histogram）
    stat_mem_read_latency_cycles_ = registerStatistic<uint64_t>("mem_read_latency_cycles");
    stat_mem_read_latency_cycles_weights_ = registerStatistic<uint64_t>("mem_read_latency_cycles_weights");
    stat_mem_read_latency_cycles_state_ = registerStatistic<uint64_t>("mem_read_latency_cycles_state");
    stat_mem_req_size_bytes_ = registerStatistic<uint64_t>("mem_req_size_bytes");
    stat_mem_outstanding_at_issue_ = registerStatistic<uint64_t>("mem_outstanding_at_issue");
    // GAS totals (accumulated from cores)
    stat_gas_unique_reads_total_ = registerStatistic<uint64_t>("gas_unique_reads_total");
    stat_gas_unique_bytes_total_ = registerStatistic<uint64_t>("gas_unique_bytes_total");
    stat_gas_rowwin_triggers_total_ = registerStatistic<uint64_t>("gas_row_window_triggers_total");
    stat_gas_rowwin_bytes_total_ = registerStatistic<uint64_t>("gas_row_window_bytes_total");
    stat_gas_total_bursts_ = registerStatistic<uint64_t>("gas_total_bursts");
    stat_gas_total_payload_bytes_ = registerStatistic<uint64_t>("gas_total_payload_bytes");
    stat_gas_unique_line_count_total_ = registerStatistic<uint64_t>("gas_unique_line_count_total");
    stat_gas_covered_line_count_total_ = registerStatistic<uint64_t>("gas_covered_line_count_total");
    stat_gas_overfetch_bytes_total_ = registerStatistic<uint64_t>("gas_overfetch_bytes_total");
    stat_gas_apply_bank_credit_effective_total_ = registerStatistic<uint64_t>("gas_apply_bank_credit_effective_total");
    stat_gas_cmd_cost_veto_total_ = registerStatistic<uint64_t>("gas_cmd_cost_veto_total");
    stat_gas_cmd_cost_veto_fine_gap_total_ = registerStatistic<uint64_t>("gas_cmd_cost_veto_fine_gap_total");
    stat_gas_cmd_cost_veto_row_window_total_ = registerStatistic<uint64_t>("gas_cmd_cost_veto_row_window_total");
    stat_gas_apply_acc_updates_total_ = registerStatistic<uint64_t>("gas_apply_acc_updates_total");
    stat_gas_acc_posts_touched_total_ = registerStatistic<uint64_t>("gas_acc_posts_touched_total");
    stat_gas_scatter_spikes_emitted_total_ = registerStatistic<uint64_t>("gas_scatter_spikes_emitted_total");
    stat_gas_acc_hwm_bytes_total_ = registerStatistic<uint64_t>("gas_acc_high_watermark_bytes_total");
    stat_gas_acc_spill_records_total_ = registerStatistic<uint64_t>("gas_acc_spill_records_total");
    stat_gas_acc_spilled_bytes_total_ = registerStatistic<uint64_t>("gas_acc_spilled_bytes_total");
    stat_gas_retire_global_hol_cycles_total_ = registerStatistic<uint64_t>("gas_retire_global_hol_cycles_total");
    stat_gas_retire_ready_but_blocked_edges_total_ = registerStatistic<uint64_t>("gas_retire_ready_but_blocked_edges_total");
    stat_gas_retire_per_post_progress_total_ = registerStatistic<uint64_t>("gas_retire_per_post_progress_total");
    stat_gas_tass_lf_p0_block_epochs_total_ = registerStatistic<uint64_t>("gas_tass_lf_p0_block_epochs_total");
    stat_gas_tass_lf_p0_block_active_pres_total_ = registerStatistic<uint64_t>("gas_tass_lf_p0_block_active_pres_total");
    stat_gas_tass_lf_p0_block_shared_pres_total_ = registerStatistic<uint64_t>("gas_tass_lf_p0_block_shared_pres_total");
    stat_gas_tass_lf_p0_cross_core_joins_total_ = registerStatistic<uint64_t>("gas_tass_lf_p0_cross_core_joins_total");
    stat_gas_tass_lf_p0_payload_bytes_total_ = registerStatistic<uint64_t>("gas_tass_lf_p0_payload_bytes_total");
    stat_gas_tass_lf_p0_current_vlf_line_groups_total_ = registerStatistic<uint64_t>("gas_tass_lf_p0_current_vlf_line_groups_total");
    stat_gas_tass_lf_p0_block_naive_line_count_total_ = registerStatistic<uint64_t>("gas_tass_lf_p0_block_naive_line_count_total");
    stat_gas_tass_lf_p0_block_fused_lb_line_count_total_ = registerStatistic<uint64_t>("gas_tass_lf_p0_block_fused_lb_line_count_total");
    stat_gas_tass_lf_p0_response_fanout_total_ = registerStatistic<uint64_t>("gas_tass_lf_p0_response_fanout_total");
    stat_gas_tass_lf_p0_reports_submit_total_ = registerStatistic<uint64_t>("gas_tass_lf_p0_reports_submit_total");
    stat_gas_tass_lf_p0_reports_send_total_ = registerStatistic<uint64_t>("gas_tass_lf_p0_reports_send_total");
    stat_gas_tass_lf_p0_reports_recv_total_ = registerStatistic<uint64_t>("gas_tass_lf_p0_reports_recv_total");
    stat_gas_tass_lf_p0_reports_reduce_total_ = registerStatistic<uint64_t>("gas_tass_lf_p0_reports_reduce_total");
    stat_gas_tass_lf_p0_reports_drop_no_link_total_ = registerStatistic<uint64_t>("gas_tass_lf_p0_reports_drop_no_link_total");
    stat_gas_tass_lf_p0_reports_drop_non_origin_total_ = registerStatistic<uint64_t>("gas_tass_lf_p0_reports_drop_non_origin_total");
    stat_gas_tass_lf_p0_reports_flushed_total_ = registerStatistic<uint64_t>("gas_tass_lf_p0_reports_flushed_total");
    stat_gas_tass_lf_p0_reports_nonzero_payload_total_ = registerStatistic<uint64_t>("gas_tass_lf_p0_reports_nonzero_payload_total");
    stat_gas_tass_lf_p0_reports_pre_entries_total_ = registerStatistic<uint64_t>("gas_tass_lf_p0_reports_pre_entries_total");
    stat_gas_tass_lf_p0_reports_via_callback_total_ = registerStatistic<uint64_t>("gas_tass_lf_p0_reports_via_callback_total");
    stat_gas_tass_lf_p0_reports_via_fallback_total_ = registerStatistic<uint64_t>("gas_tass_lf_p0_reports_via_fallback_total");
    stat_gcss_lookup_hit_total_ = registerStatistic<uint64_t>("gcss_lookup_hit_total");
    stat_gcss_lookup_miss_total_ = registerStatistic<uint64_t>("gcss_lookup_miss_total");
    stat_weight_read_dense_reqs_total_ = registerStatistic<uint64_t>("weight_read_dense_reqs_total");
    stat_weight_read_dense_bytes_total_ = registerStatistic<uint64_t>("weight_read_dense_bytes_total");
    stat_weight_read_rowptr_reqs_total_ = registerStatistic<uint64_t>("weight_read_rowptr_reqs_total");
    stat_weight_read_rowptr_bytes_total_ = registerStatistic<uint64_t>("weight_read_rowptr_bytes_total");
    stat_weight_read_colidx_reqs_total_ = registerStatistic<uint64_t>("weight_read_colidx_reqs_total");
    stat_weight_read_colidx_bytes_total_ = registerStatistic<uint64_t>("weight_read_colidx_bytes_total");
    stat_weight_read_blockdata_reqs_total_ = registerStatistic<uint64_t>("weight_read_blockdata_reqs_total");
    stat_weight_read_blockdata_bytes_total_ = registerStatistic<uint64_t>("weight_read_blockdata_bytes_total");
    stat_weight_read_gcss_reqs_total_ = registerStatistic<uint64_t>("weight_read_gcss_reqs_total");
    stat_weight_read_gcss_bytes_total_ = registerStatistic<uint64_t>("weight_read_gcss_bytes_total");
    stat_weight_idx_sram_reads_total_ = registerStatistic<uint64_t>("weight_idx_sram_reads_total");
    stat_weight_idx_sram_writes_total_ = registerStatistic<uint64_t>("weight_idx_sram_writes_total");
    stat_weight_idx_sram_bytes_read_total_ = registerStatistic<uint64_t>("weight_idx_sram_bytes_read_total");
    stat_weight_idx_sram_bytes_write_total_ = registerStatistic<uint64_t>("weight_idx_sram_bytes_write_total");
    stat_weight_idx_sram_bank_conflict_ticks_total_ = registerStatistic<uint64_t>("weight_idx_sram_bank_conflict_ticks_total");
    stat_weight_idx_sram_predicted_extra_cycles_total_ = registerStatistic<uint64_t>("weight_idx_sram_predicted_extra_cycles_total");
    stat_weight_idx_sram_resident_bytes_peak_ = registerStatistic<uint64_t>("weight_idx_sram_resident_bytes_peak");
    stat_weight_idx_lookup_total_ = registerStatistic<uint64_t>("weight_idx_lookup_total");
    stat_weight_idx_lookup_idx2_total_ = registerStatistic<uint64_t>("weight_idx_lookup_idx2_total");
    stat_weight_l0_sram_reads_total_ = registerStatistic<uint64_t>("weight_l0_sram_reads_total");
    stat_weight_l0_sram_writes_total_ = registerStatistic<uint64_t>("weight_l0_sram_writes_total");
    stat_weight_l0_sram_bytes_read_total_ = registerStatistic<uint64_t>("weight_l0_sram_bytes_read_total");
    stat_weight_l0_sram_bytes_write_total_ = registerStatistic<uint64_t>("weight_l0_sram_bytes_write_total");
    stat_weight_l0_sram_bank_conflict_ticks_total_ = registerStatistic<uint64_t>("weight_l0_sram_bank_conflict_ticks_total");
    stat_weight_l0_sram_predicted_extra_cycles_total_ = registerStatistic<uint64_t>("weight_l0_sram_predicted_extra_cycles_total");
    stat_weight_l0_sram_resident_bytes_peak_ = registerStatistic<uint64_t>("weight_l0_sram_resident_bytes_peak");
    stat_weight_l0_lookup_total_ = registerStatistic<uint64_t>("weight_l0_lookup_total");
    stat_weight_l0_hit_total_ = registerStatistic<uint64_t>("weight_l0_hit_total");
    stat_weight_l0_fill_total_ = registerStatistic<uint64_t>("weight_l0_fill_total");
    stat_weight_l0_evict_total_ = registerStatistic<uint64_t>("weight_l0_evict_total");
    stat_core_state_sram_reads_total_ = registerStatistic<uint64_t>("core_state_sram_reads_total");
    stat_core_state_sram_writes_total_ = registerStatistic<uint64_t>("core_state_sram_writes_total");
    stat_core_state_sram_bytes_read_total_ = registerStatistic<uint64_t>("core_state_sram_bytes_read_total");
    stat_core_state_sram_bytes_write_total_ = registerStatistic<uint64_t>("core_state_sram_bytes_write_total");
    stat_core_state_sram_bank_conflict_ticks_total_ = registerStatistic<uint64_t>("core_state_sram_bank_conflict_ticks_total");
    stat_core_state_sram_predicted_extra_cycles_total_ = registerStatistic<uint64_t>("core_state_sram_predicted_extra_cycles_total");
    stat_core_state_sram_resident_bytes_peak_ = registerStatistic<uint64_t>("core_state_sram_resident_bytes_peak");
    stat_gas_activity_f_ = registerStatistic<double>("gas_activity_f");
    stat_sim_cycles_total_ = registerStatistic<uint64_t>("sim_cycles_total");
    stat_global_steps_done_total_ = registerStatistic<uint64_t>("global_steps_done_total");
    stat_step_activation_invocations_ = registerStatistic<uint64_t>("step_activation_invocations");
    stat_step_activation_pre_selected_ = registerStatistic<uint64_t>("step_activation_pre_selected");
    stat_step_activation_spike_attempts_ = registerStatistic<uint64_t>("step_activation_spike_attempts");
    stat_step_activation_spikes_injected_ = registerStatistic<uint64_t>("step_activation_spikes_injected");
    stat_step_activation_route_hits_ = registerStatistic<uint64_t>("step_activation_route_hits");
    stat_step_activation_route_misses_ = registerStatistic<uint64_t>("step_activation_route_misses");
    stat_step_activation_local_drops_ = registerStatistic<uint64_t>("step_activation_local_drops");
    
}

void MultiCorePE::accumulateMemReadLatency(uint64_t latency_cycles, bool is_weight) {
    if (stat_mem_read_latency_cycles_) stat_mem_read_latency_cycles_->addData(latency_cycles);
    if (is_weight) {
        if (stat_mem_read_latency_cycles_weights_) stat_mem_read_latency_cycles_weights_->addData(latency_cycles);
    } else {
        if (stat_mem_read_latency_cycles_state_) stat_mem_read_latency_cycles_state_->addData(latency_cycles);
    }
    // 窗口化：按当前仿真时间(ns)聚合响应时延
    if (window_stats_enable_) {
        uint64_t now_ns = getCurrentSimTimeNano();
        WindowAgg& w = getOrCreateWindow_(now_ns);
        w.read_count += 1;
        w.read_latency_sum += latency_cycles;
    }
}

void MultiCorePE::accumulateIssueStats(uint64_t req_size_bytes, uint64_t inflight) {
    if (stat_mem_req_size_bytes_) stat_mem_req_size_bytes_->addData(req_size_bytes);
    if (stat_mem_outstanding_at_issue_) stat_mem_outstanding_at_issue_->addData(inflight);
    if (stat_memory_requests_) stat_memory_requests_->addData(1);
    // 窗口化：按当前仿真时间(ns)聚合发起侧指标
    if (window_stats_enable_) {
        uint64_t now_ns = getCurrentSimTimeNano();
        WindowAgg& w = getOrCreateWindow_(now_ns);
        w.issue_count += 1;
        w.req_size_sum += req_size_bytes;
        w.outstanding_sum += inflight;
    }
}

void MultiCorePE::accumulateGasStats(uint64_t unique_bytes, uint64_t unique_reads) {
    if (unique_reads && stat_gas_unique_reads_total_) stat_gas_unique_reads_total_->addData(unique_reads);
    if (unique_bytes && stat_gas_unique_bytes_total_) stat_gas_unique_bytes_total_->addData(unique_bytes);
}

void MultiCorePE::accumulateGasStatsExt(uint64_t unique_bytes, uint64_t unique_reads,
                                        uint64_t rowwin_triggers, uint64_t rowwin_bytes,
                                        uint64_t bursts, uint64_t payload_bytes,
                                        uint64_t window_inflight_peak,
                                        uint64_t window_buffer_max_bytes,
                                        uint64_t unique_line_count,
                                        uint64_t covered_line_count,
                                        uint64_t overfetch_bytes,
                                        uint64_t apply_bank_credit_effective,
                                        uint64_t cmd_cost_veto,
                                        uint64_t cmd_cost_veto_fine_gap,
                                        uint64_t cmd_cost_veto_row_window) {
    accumulateGasStats(unique_bytes, unique_reads);
    if (rowwin_triggers && stat_gas_rowwin_triggers_total_) stat_gas_rowwin_triggers_total_->addData(rowwin_triggers);
    if (rowwin_bytes && stat_gas_rowwin_bytes_total_) stat_gas_rowwin_bytes_total_->addData(rowwin_bytes);
    if (bursts && stat_gas_total_bursts_) stat_gas_total_bursts_->addData(bursts);
    if (payload_bytes && stat_gas_total_payload_bytes_) stat_gas_total_payload_bytes_->addData(payload_bytes);
    if (unique_line_count && stat_gas_unique_line_count_total_) stat_gas_unique_line_count_total_->addData(unique_line_count);
    if (covered_line_count && stat_gas_covered_line_count_total_) stat_gas_covered_line_count_total_->addData(covered_line_count);
    if (overfetch_bytes && stat_gas_overfetch_bytes_total_) stat_gas_overfetch_bytes_total_->addData(overfetch_bytes);
    if (apply_bank_credit_effective && stat_gas_apply_bank_credit_effective_total_) stat_gas_apply_bank_credit_effective_total_->addData(apply_bank_credit_effective);
    if (cmd_cost_veto && stat_gas_cmd_cost_veto_total_) stat_gas_cmd_cost_veto_total_->addData(cmd_cost_veto);
    if (cmd_cost_veto_fine_gap && stat_gas_cmd_cost_veto_fine_gap_total_) stat_gas_cmd_cost_veto_fine_gap_total_->addData(cmd_cost_veto_fine_gap);
    if (cmd_cost_veto_row_window && stat_gas_cmd_cost_veto_row_window_total_) stat_gas_cmd_cost_veto_row_window_total_->addData(cmd_cost_veto_row_window);
}

void MultiCorePE::accumulateActivityF(double f) {
    if (stat_gas_activity_f_) stat_gas_activity_f_->addData(f);
    if (window_stats_enable_) {
        uint64_t now_ns = getCurrentSimTimeNano();
        WindowAgg& w = getOrCreateWindow_(now_ns);
        w.activity_f_sum += f;
        w.activity_f_count += 1;
    }
}

void MultiCorePE::accumulateApplyScatterStats(uint64_t acc_updates, uint64_t posts_touched,
                                              uint64_t spikes_emitted, uint64_t hwm_bytes,
                                              uint64_t spill_records, uint64_t spilled_bytes) {
    if (acc_updates && stat_gas_apply_acc_updates_total_) stat_gas_apply_acc_updates_total_->addData(acc_updates);
    if (posts_touched && stat_gas_acc_posts_touched_total_) stat_gas_acc_posts_touched_total_->addData(posts_touched);
    if (spikes_emitted && stat_gas_scatter_spikes_emitted_total_) stat_gas_scatter_spikes_emitted_total_->addData(spikes_emitted);
    if (hwm_bytes && stat_gas_acc_hwm_bytes_total_) stat_gas_acc_hwm_bytes_total_->addData(hwm_bytes);
    if (spill_records && stat_gas_acc_spill_records_total_) stat_gas_acc_spill_records_total_->addData(spill_records);
    if (spilled_bytes && stat_gas_acc_spilled_bytes_total_) stat_gas_acc_spilled_bytes_total_->addData(spilled_bytes);
}

void MultiCorePE::accumulateSynapseReadStats(uint64_t gcss_lookup_hit_total,
                                             uint64_t gcss_lookup_miss_total,
                                             uint64_t dense_reqs_total,
                                             uint64_t dense_bytes_total,
                                             uint64_t rowptr_reqs_total,
                                             uint64_t rowptr_bytes_total,
                                             uint64_t colidx_reqs_total,
                                             uint64_t colidx_bytes_total,
                                             uint64_t blockdata_reqs_total,
                                             uint64_t blockdata_bytes_total,
                                             uint64_t gcss_reqs_total,
                                             uint64_t gcss_bytes_total,
                                             uint64_t weight_idx_sram_reads_total,
                                             uint64_t weight_idx_sram_writes_total,
                                             uint64_t weight_idx_sram_bytes_read_total,
                                             uint64_t weight_idx_sram_bytes_write_total,
                                             uint64_t weight_idx_sram_bank_conflict_ticks_total,
                                             uint64_t weight_idx_sram_predicted_extra_cycles_total,
                                             uint64_t weight_idx_sram_resident_bytes_peak,
                                             uint64_t weight_idx_lookup_total,
                                             uint64_t weight_idx_lookup_idx2_total,
                                             uint64_t weight_l0_sram_reads_total,
                                             uint64_t weight_l0_sram_writes_total,
                                             uint64_t weight_l0_sram_bytes_read_total,
                                             uint64_t weight_l0_sram_bytes_write_total,
                                             uint64_t weight_l0_sram_bank_conflict_ticks_total,
                                             uint64_t weight_l0_sram_predicted_extra_cycles_total,
                                             uint64_t weight_l0_sram_resident_bytes_peak,
                                             uint64_t weight_l0_lookup_total,
                                             uint64_t weight_l0_hit_total,
                                             uint64_t weight_l0_fill_total,
                                             uint64_t weight_l0_evict_total,
                                             uint64_t core_state_sram_reads_total,
                                             uint64_t core_state_sram_writes_total,
                                             uint64_t core_state_sram_bytes_read_total,
                                             uint64_t core_state_sram_bytes_write_total,
                                             uint64_t core_state_sram_bank_conflict_ticks_total,
                                             uint64_t core_state_sram_predicted_extra_cycles_total,
                                             uint64_t core_state_sram_resident_bytes_peak,
                                             uint64_t gas_retire_global_hol_cycles_total,
                                             uint64_t gas_retire_ready_but_blocked_edges_total,
                                             uint64_t gas_retire_per_post_progress_total,
                                             uint64_t gas_tass_lf_p0_block_epochs_total,
                                             uint64_t gas_tass_lf_p0_block_active_pres_total,
                                             uint64_t gas_tass_lf_p0_block_shared_pres_total,
                                             uint64_t gas_tass_lf_p0_cross_core_joins_total,
                                             uint64_t gas_tass_lf_p0_payload_bytes_total,
                                             uint64_t gas_tass_lf_p0_current_vlf_line_groups_total,
                                             uint64_t gas_tass_lf_p0_block_naive_line_count_total,
                                             uint64_t gas_tass_lf_p0_block_fused_lb_line_count_total,
                                             uint64_t gas_tass_lf_p0_response_fanout_total,
                                             uint64_t gas_tass_lf_p0_reports_flushed_total,
                                             uint64_t gas_tass_lf_p0_reports_nonzero_payload_total,
                                             uint64_t gas_tass_lf_p0_reports_pre_entries_total,
                                             uint64_t gas_tass_lf_p0_reports_via_callback_total,
                                             uint64_t gas_tass_lf_p0_reports_via_fallback_total) {
    if (stat_gcss_lookup_hit_total_) stat_gcss_lookup_hit_total_->addData(gcss_lookup_hit_total);
    if (stat_gcss_lookup_miss_total_) stat_gcss_lookup_miss_total_->addData(gcss_lookup_miss_total);
    if (stat_weight_read_dense_reqs_total_) stat_weight_read_dense_reqs_total_->addData(dense_reqs_total);
    if (stat_weight_read_dense_bytes_total_) stat_weight_read_dense_bytes_total_->addData(dense_bytes_total);
    if (stat_weight_read_rowptr_reqs_total_) stat_weight_read_rowptr_reqs_total_->addData(rowptr_reqs_total);
    if (stat_weight_read_rowptr_bytes_total_) stat_weight_read_rowptr_bytes_total_->addData(rowptr_bytes_total);
    if (stat_weight_read_colidx_reqs_total_) stat_weight_read_colidx_reqs_total_->addData(colidx_reqs_total);
    if (stat_weight_read_colidx_bytes_total_) stat_weight_read_colidx_bytes_total_->addData(colidx_bytes_total);
    if (stat_weight_read_blockdata_reqs_total_) stat_weight_read_blockdata_reqs_total_->addData(blockdata_reqs_total);
    if (stat_weight_read_blockdata_bytes_total_) stat_weight_read_blockdata_bytes_total_->addData(blockdata_bytes_total);
    if (stat_weight_read_gcss_reqs_total_) stat_weight_read_gcss_reqs_total_->addData(gcss_reqs_total);
    if (stat_weight_read_gcss_bytes_total_) stat_weight_read_gcss_bytes_total_->addData(gcss_bytes_total);
    if (stat_weight_idx_sram_reads_total_) stat_weight_idx_sram_reads_total_->addData(weight_idx_sram_reads_total);
    if (stat_weight_idx_sram_writes_total_) stat_weight_idx_sram_writes_total_->addData(weight_idx_sram_writes_total);
    if (stat_weight_idx_sram_bytes_read_total_) stat_weight_idx_sram_bytes_read_total_->addData(weight_idx_sram_bytes_read_total);
    if (stat_weight_idx_sram_bytes_write_total_) stat_weight_idx_sram_bytes_write_total_->addData(weight_idx_sram_bytes_write_total);
    if (stat_weight_idx_sram_bank_conflict_ticks_total_) stat_weight_idx_sram_bank_conflict_ticks_total_->addData(weight_idx_sram_bank_conflict_ticks_total);
    if (stat_weight_idx_sram_predicted_extra_cycles_total_) stat_weight_idx_sram_predicted_extra_cycles_total_->addData(weight_idx_sram_predicted_extra_cycles_total);
    if (stat_weight_idx_sram_resident_bytes_peak_) stat_weight_idx_sram_resident_bytes_peak_->addData(weight_idx_sram_resident_bytes_peak);
    if (stat_weight_idx_lookup_total_) stat_weight_idx_lookup_total_->addData(weight_idx_lookup_total);
    if (stat_weight_idx_lookup_idx2_total_) stat_weight_idx_lookup_idx2_total_->addData(weight_idx_lookup_idx2_total);
    if (stat_weight_l0_sram_reads_total_) stat_weight_l0_sram_reads_total_->addData(weight_l0_sram_reads_total);
    if (stat_weight_l0_sram_writes_total_) stat_weight_l0_sram_writes_total_->addData(weight_l0_sram_writes_total);
    if (stat_weight_l0_sram_bytes_read_total_) stat_weight_l0_sram_bytes_read_total_->addData(weight_l0_sram_bytes_read_total);
    if (stat_weight_l0_sram_bytes_write_total_) stat_weight_l0_sram_bytes_write_total_->addData(weight_l0_sram_bytes_write_total);
    if (stat_weight_l0_sram_bank_conflict_ticks_total_) stat_weight_l0_sram_bank_conflict_ticks_total_->addData(weight_l0_sram_bank_conflict_ticks_total);
    if (stat_weight_l0_sram_predicted_extra_cycles_total_) stat_weight_l0_sram_predicted_extra_cycles_total_->addData(weight_l0_sram_predicted_extra_cycles_total);
    if (stat_weight_l0_sram_resident_bytes_peak_) stat_weight_l0_sram_resident_bytes_peak_->addData(weight_l0_sram_resident_bytes_peak);
    if (stat_weight_l0_lookup_total_) stat_weight_l0_lookup_total_->addData(weight_l0_lookup_total);
    if (stat_weight_l0_hit_total_) stat_weight_l0_hit_total_->addData(weight_l0_hit_total);
    if (stat_weight_l0_fill_total_) stat_weight_l0_fill_total_->addData(weight_l0_fill_total);
    if (stat_weight_l0_evict_total_) stat_weight_l0_evict_total_->addData(weight_l0_evict_total);
    if (stat_core_state_sram_reads_total_) stat_core_state_sram_reads_total_->addData(core_state_sram_reads_total);
    if (stat_core_state_sram_writes_total_) stat_core_state_sram_writes_total_->addData(core_state_sram_writes_total);
    if (stat_core_state_sram_bytes_read_total_) stat_core_state_sram_bytes_read_total_->addData(core_state_sram_bytes_read_total);
    if (stat_core_state_sram_bytes_write_total_) stat_core_state_sram_bytes_write_total_->addData(core_state_sram_bytes_write_total);
    if (stat_core_state_sram_bank_conflict_ticks_total_) stat_core_state_sram_bank_conflict_ticks_total_->addData(core_state_sram_bank_conflict_ticks_total);
    if (stat_core_state_sram_predicted_extra_cycles_total_) stat_core_state_sram_predicted_extra_cycles_total_->addData(core_state_sram_predicted_extra_cycles_total);
    if (stat_core_state_sram_resident_bytes_peak_) stat_core_state_sram_resident_bytes_peak_->addData(core_state_sram_resident_bytes_peak);
    if (stat_gas_retire_global_hol_cycles_total_) stat_gas_retire_global_hol_cycles_total_->addData(gas_retire_global_hol_cycles_total);
    if (stat_gas_retire_ready_but_blocked_edges_total_) stat_gas_retire_ready_but_blocked_edges_total_->addData(gas_retire_ready_but_blocked_edges_total);
    if (stat_gas_retire_per_post_progress_total_) stat_gas_retire_per_post_progress_total_->addData(gas_retire_per_post_progress_total);
    if (stat_gas_tass_lf_p0_block_epochs_total_) stat_gas_tass_lf_p0_block_epochs_total_->addData(gas_tass_lf_p0_block_epochs_total);
    if (stat_gas_tass_lf_p0_block_active_pres_total_) stat_gas_tass_lf_p0_block_active_pres_total_->addData(gas_tass_lf_p0_block_active_pres_total);
    if (stat_gas_tass_lf_p0_block_shared_pres_total_) stat_gas_tass_lf_p0_block_shared_pres_total_->addData(gas_tass_lf_p0_block_shared_pres_total);
    if (stat_gas_tass_lf_p0_cross_core_joins_total_) stat_gas_tass_lf_p0_cross_core_joins_total_->addData(gas_tass_lf_p0_cross_core_joins_total);
    if (stat_gas_tass_lf_p0_payload_bytes_total_) stat_gas_tass_lf_p0_payload_bytes_total_->addData(gas_tass_lf_p0_payload_bytes_total);
    if (stat_gas_tass_lf_p0_current_vlf_line_groups_total_) stat_gas_tass_lf_p0_current_vlf_line_groups_total_->addData(gas_tass_lf_p0_current_vlf_line_groups_total);
    if (stat_gas_tass_lf_p0_block_naive_line_count_total_) stat_gas_tass_lf_p0_block_naive_line_count_total_->addData(gas_tass_lf_p0_block_naive_line_count_total);
    if (stat_gas_tass_lf_p0_block_fused_lb_line_count_total_) stat_gas_tass_lf_p0_block_fused_lb_line_count_total_->addData(gas_tass_lf_p0_block_fused_lb_line_count_total);
    if (stat_gas_tass_lf_p0_response_fanout_total_) stat_gas_tass_lf_p0_response_fanout_total_->addData(gas_tass_lf_p0_response_fanout_total);
    if (stat_gas_tass_lf_p0_reports_flushed_total_) stat_gas_tass_lf_p0_reports_flushed_total_->addData(gas_tass_lf_p0_reports_flushed_total);
    if (stat_gas_tass_lf_p0_reports_nonzero_payload_total_) stat_gas_tass_lf_p0_reports_nonzero_payload_total_->addData(gas_tass_lf_p0_reports_nonzero_payload_total);
    if (stat_gas_tass_lf_p0_reports_pre_entries_total_) stat_gas_tass_lf_p0_reports_pre_entries_total_->addData(gas_tass_lf_p0_reports_pre_entries_total);
    if (stat_gas_tass_lf_p0_reports_via_callback_total_) stat_gas_tass_lf_p0_reports_via_callback_total_->addData(gas_tass_lf_p0_reports_via_callback_total);
    if (stat_gas_tass_lf_p0_reports_via_fallback_total_) stat_gas_tass_lf_p0_reports_via_fallback_total_->addData(gas_tass_lf_p0_reports_via_fallback_total);
}

void MultiCorePE::notifyStageEvent(uint32_t seq, const std::string& event, uint64_t ts_ns,
                                   uint64_t spikes_emitted, int core_id) {
    auto &m = stage_marks_[seq];
    if (event == "BeginGather") {
        if (m.bg == 0 || ts_ns < m.bg) m.bg = ts_ns;
    } else if (event == "BeginApply") {
        // 多核窗口阶段并非严格同步：我们希望 PE 级阶段边界代表“所有核心都进入该阶段”的时刻。
        // 因此 BeginApply/BeginScatter 取 max（最慢核心），BeginGather 取 min（最早起点）。
        if (m.ga == 0 || ts_ns > m.ga) m.ga = ts_ns;
    } else if (event == "EndApply") {
        if (m.ea == 0 || ts_ns > m.ea) m.ea = ts_ns;
    } else if (event == "BeginScatter") {
        if (m.bs == 0 || ts_ns > m.bs) m.bs = ts_ns;
    } else if (event == "EndScatter") {
        if (m.es == 0 || ts_ns > m.es) m.es = ts_ns;
    }

    // Experiment observability: naive_raw must not execute GAS Apply/Scatter stages.
    // Note: BeginGather/EndScatter may be used as generic step boundary markers in some modes;
    // only flag stages unique to GAS/window pipeline to avoid false positives.
    if (exec_mode_ == "naive_raw" &&
        (event == "BeginApply" || event == "EndApply" || event == "BeginScatter")) {
        if (stat_compat_naive_gas_stage_events_total_) stat_compat_naive_gas_stage_events_total_->addData(1);
    }

    // Global Step 诊断：记录每核在 active_seq 内的最后阶段，用于 finish() 输出卡点
    if (global_step_sync_enable_ &&
        seq == global_step_active_seq_ &&
        core_id >= 0 &&
        core_id < num_cores_ &&
        global_step_last_stage_code_.size() == static_cast<size_t>(num_cores_)) {
        const size_t idx = static_cast<size_t>(core_id);
        global_step_last_stage_code_[idx] = stepStageCodeFromName_(event);
        global_step_last_stage_seq_[idx] = seq;
        global_step_last_stage_ts_ns_[idx] = ts_ns;
        global_step_last_stage_spikes_[idx] = spikes_emitted;
    }

    if (sentinel_enabled_ && output_ && seq == 1 &&
        global_step_sync_enable_ &&
        seq == global_step_active_seq_ &&
        (event == "BeginGather" || event == "BeginApply" || event == "EndScatter")) {
        output_->verbose(CALL_INFO, 2, 0,
            "[[sentinel-step-sync]] node=%d stage=%s core=%d seq=%u ts_ns=%" PRIu64 " spikes=%" PRIu64 "\n",
            node_id_, event.c_str(), core_id, seq, (uint64_t)ts_ns, (uint64_t)spikes_emitted);
    }

    // Step 注入事件转发（Phase3-B 下沉为 StepActivationSubsystem）
    if (event == "BeginGather") {
        step_activation_subsys_.onBeginGather(seq, ts_ns, core_id);
    } else if (event == "EndScatter") {
        step_activation_subsys_.onEndScatter(seq);
    }

    // Global Step barrier: 当本 PE 的所有 core 都完成 EndScatter(seq) 后，上报给控制器（EndScatter policy）
    if (global_step_sync_enable_ &&
        gas_step_ctrl_link_ &&
        !global_step_done_sent_ &&
        global_step_done_policy_ == GlobalStepDonePolicy::EndScatter &&
        event == "EndScatter" &&
        seq == global_step_active_seq_) {
        if (core_id >= 0 && core_id < num_cores_) {
            global_step_done_cores_[static_cast<size_t>(core_id)] = 1;
            if (sentinel_enabled_ && output_ && seq == 1) {
                size_t done_cnt = 0;
                for (auto v : global_step_done_cores_) if (v) ++done_cnt;
                output_->verbose(CALL_INFO, 2, 0,
                    "[[sentinel-step-sync]] node=%d mark EndScatter core=%d seq=%u active=%u done=%zu/%d\n",
                    node_id_, core_id, seq, global_step_active_seq_, done_cnt, num_cores_);
            }
            bool all_done = true;
            for (auto v : global_step_done_cores_) {
                if (!v) { all_done = false; break; }
            }
            if (all_done) {
                auto* ev = new GasStepBarrierEvent(GasStepBarrierOp::PeDone, seq, static_cast<uint32_t>(node_id_));
                // Step-level telemetry: reuse PE-level stage marks aggregated in notifyStageEvent().
                // This is used by GlobalGasStepController for criticality-aware global credit control.
                if (m.bg != 0 && m.es != 0 && m.es >= m.bg) {
                    ev->step_total_ns = m.es - m.bg;
                }
                if (m.ga != 0 && m.bs != 0 && m.bs >= m.ga) {
                    ev->step_apply_ns = m.bs - m.ga;
                }
                gas_step_ctrl_link_->send(ev);
                global_step_done_sent_ = true;
                if (stat_global_steps_done_total_) stat_global_steps_done_total_->addData(1);
                if (sentinel_enabled_ && output_ && seq == 1) {
                    output_->verbose(CALL_INFO, 2, 0, "[[sentinel-step-sync]] node=%d send PE_DONE seq=%u\n", node_id_, seq);
                }
            }
        }
    }
}

void MultiCorePE::mergeWindowMetricsFromCsv_() {
    if (!window_stats_enable_ || window_metrics_csv_.empty() || windows_.empty()) return;
    std::ifstream fin(window_metrics_csv_);
    if (!fin.good()) return;
    std::string line;
    if (!std::getline(fin, line)) return; // header
    std::vector<std::pair<uint64_t,uint64_t>> entries;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string field;
        std::vector<std::string> cols;
        while (std::getline(ss, field, ',')) cols.push_back(field);
        if (cols.size() < 5) continue;
        uint64_t inflight = 0;
        uint64_t buffer = 0;
        try { inflight = static_cast<uint64_t>(std::stoull(cols[3])); } catch (...) {}
        try { buffer = static_cast<uint64_t>(std::stoull(cols[4])); } catch (...) {}
        entries.emplace_back(buffer, inflight);
    }
    if (entries.empty()) return;
    const size_t total_rows = entries.size();
    for (size_t i = 0; i < entries.size(); ++i) {
        size_t idx = (total_rows > 0) ? (i * windows_.size()) / total_rows : 0;
        if (idx >= windows_.size()) idx = windows_.size() - 1;
        auto& w = windows_[idx];
        uint64_t buffer = entries[i].first;
        uint64_t inflight = entries[i].second;
        if (buffer > w.sb_peak_bytes) w.sb_peak_bytes = buffer;
        if (inflight > w.inflight_peak) w.inflight_peak = inflight;
    }
}

void MultiCorePE::writeWindowCsv_() {
    std::ofstream fout(window_csv_);
    if (!fout.good()) {
        PE_LOG(1, "⚠️ 无法写入窗口统计CSV: %s\n", window_csv_.c_str());
        return;
    }
    mergeWindowMetricsFromCsv_();

    fout << "window_start_us,window_end_us,read_count,avg_read_latency_cycles,issue_count,avg_req_size_bytes,avg_outstanding,avg_activity_f,sb_peak_bytes,inflight_peak" << '\n';
    for (const auto& w : windows_) {
        if (w.end_ns == 0) continue; // 未初始化
        double start_us = static_cast<double>(w.start_ns) / 1000.0;
        double end_us   = static_cast<double>(w.end_ns) / 1000.0;
        double avg_lat  = (w.read_count > 0) ? (static_cast<double>(w.read_latency_sum) / w.read_count) : 0.0;
        double avg_size = (w.issue_count > 0) ? (static_cast<double>(w.req_size_sum) / w.issue_count) : 0.0;
        double avg_out  = (w.issue_count > 0) ? (static_cast<double>(w.outstanding_sum) / w.issue_count) : 0.0;
        double avg_f   = (w.activity_f_count > 0) ? (w.activity_f_sum / (double)w.activity_f_count) : 0.0;
        fout << start_us << ',' << end_us << ','
             << w.read_count << ',' << avg_lat << ','
             << w.issue_count << ',' << avg_size << ',' << avg_out << ',' << avg_f << ','
             << w.sb_peak_bytes << ',' << w.inflight_peak << '\n';
    }
    fout.close();
}

void MultiCorePE::initializeProcessingUnits() {
    
    cores_.reserve(num_cores_);
    
    for (int i = 0; i < num_cores_; i++) {
        int neuron_id_start = node_id_ * total_neurons_ + i * neurons_per_core_;
        
        // 创建SnnPE SubComponent参数
        Params core_params;
        core_params.insert("core_id", std::to_string(i));
        // 默认口径：每个 core 只负责自身的 neurons_per_core（global_neuron_base 由 core 偏移决定）
        core_params.insert("num_neurons", std::to_string(neurons_per_core_));
        core_params.insert("neurons_per_pe", std::to_string(neurons_per_pe_cfg_));
        core_params.insert("total_cores", std::to_string(num_cores_));
        core_params.insert("total_nodes", std::to_string(total_nodes_));
        core_params.insert(
            "global_neuron_base",
            std::to_string(static_cast<uint64_t>(global_neuron_base_) + static_cast<uint64_t>(i) * static_cast<uint64_t>(neurons_per_core_)));
        core_params.insert("v_thresh", std::to_string(v_thresh_));
        core_params.insert("v_reset", std::to_string(v_reset_));
        core_params.insert("v_rest", std::to_string(v_rest_));
        core_params.insert("tau_mem", std::to_string(tau_mem_));
        core_params.insert("t_ref", std::to_string(t_ref_));
        core_params.insert("node_id", std::to_string(node_id_));
        // 若 Python 侧未提供 base_addr，则退回旧的简单映射；否则尊重传入值
        if (!core_params.contains("base_addr")) {
            core_params.insert("base_addr", std::to_string(neuron_id_start * 1000)); // 简单地址映射（仅兜底）
        }
        core_params.insert("verbose", std::to_string(verbose_));
        
        // 传递权重文件参数
        if (!weights_file_.empty()) {
            core_params.insert("weights_file", weights_file_);
        }
        
        // 传递权重验证参数
        core_params.insert("verify_weights", std::to_string(verify_weights_ ? 1 : 0));
        core_params.insert("weight_verify_samples", std::to_string(weight_verify_samples_));
        core_params.insert("expected_weight_value", std::to_string(expected_weight_value_));
        core_params.insert("verify_log_each_sample", std::to_string(verify_log_each_sample_ ? 1 : 0));
        
        // 传递权重回退参数 - 关键修复！
        core_params.insert("use_event_weight_fallback", std::to_string(use_event_weight_fallback_ ? 1 : 0));
        core_params.insert("enable_memory_weights", std::to_string(enable_memory_weights_ ? 1 : 0));
        core_params.insert("write_weights_on_init", std::to_string(write_weights_on_init_ ? 1 : 0));
        
        // 记录槽位可用性（Phase9：核心槽位 API 改为 CoreShellAPI）
        bool slot_api_ok = isSubComponentLoadableUsingAPI<CoreShellAPI>("core" + std::to_string(i));

        // 优先尝试通过用户在Python中配置的槽位加载
        CoreShellAPI* core = loadUserSubComponent<CoreShellAPI>(
            "core" + std::to_string(i), ComponentInfo::SHARE_NONE);
        if (core) {
        }

        if (!core) {
            // 如果用户未配置，则回退到匿名加载默认实现
            core = loadAnonymousSubComponent<CoreShellAPI>(
                "SnnDL.SnnPESubComponent", "core" + std::to_string(i), 0, ComponentInfo::SHARE_NONE, core_params);
            if (core) {
            } else {
                if (output_) {
                    output_->fatal(
                        CALL_INFO, -1,
                        "Phase9 fatal: failed to load core%d SubComponent.\n"
                        "  - slot=\"core%d\" api_ok(CoreShellAPI)=%d\n"
                        "  - attempted anonymous impl: \"SnnDL.SnnPESubComponent\"\n"
                        "请检查：SnnPESubComponent 的 ELI 注册父接口是否为 SnnCoreAPI/CoreShellAPI，以及 Python 侧是否覆写了 core%d 槽位。\n",
                        i, i, slot_api_ok ? 1 : 0, i);
                }
                // output_ == nullptr 时仍保持旧行为，避免 nullptr 解引用
                PE_LOG(1, "[core%d] 匿名加载失败（output_=nullptr）\n", i);
            }
        } else {
            // 若由用户配置，补充必要参数（若Python侧未给全量）
            // 这里不强制覆盖，参数以Python为准
        }
        
        if (core) {
            core->setParentInterface(this);
            // Phase4-A1.3：为 Control 注入 NoC 抽象接口（优先走 NocSubsystem，不依赖 MultiCorePE send/forward 细节）
            if (auto* hooks = dynamic_cast<ICoreControlHooks*>(core)) {
                hooks->setNocTransport(&noc_subsys_);
            }
            // 为每个核心配置内存Link（若用户在Python连接了对应端口则不为None）
            std::string port = "core" + std::to_string(i) + "_mem";
            Link* l = configureLink(port);
            if (l) {
                auto* ml = dynamic_cast<ICoreMemoryLink*>(core);
                if (!ml) {
                    output_->fatal(CALL_INFO, -1,
                                   "Phase9 fatal: core%d has a configured memory link (%s) but does not implement ICoreMemoryLink\n",
                                   i, port.c_str());
                }
                ml->setMemoryLink(l);
            }
            cores_.push_back(core);
        } else {
            cores_.push_back(nullptr);
        }
        
        PE_LOG(3, "   ✅ SnnPE核心%d: 神经元ID范围[%d, %d)\n",
                        i, neuron_id_start, neuron_id_start + neurons_per_core_);
    }

    // loader_done 事件可能在 init(phase0) 的 link 配置完成后、core 创建之前到达并被 drain，
    // 此时 handleLoaderDoneEvent() 无法下发到 core（cores_ 仍为空）。
    // 在 core 创建完成后，若 loader_ready 已被 latch，则补发一次本地就绪回调，确保 core 侧 latch 不丢失。
    if (loader_ready_latched_) {
        for (auto* core : cores_) {
            if (!core) continue;
            if (auto* hook = dynamic_cast<ILoaderReadyHooks*>(core)) {
                hook->onLoaderReady();
            }
        }
    }
    
    
    // 添加权重配置摘要
    // if (!weights_file_.empty()) {
    //     PE_LOG(1, "📋 节点%d权重配置摘要: %zu个核心使用权重文件 %s\n", 
    //                     node_id_, cores_.size(), weights_file_.c_str());
    // }
}

void MultiCorePE::initializeInternalRing() {
    // 单核情况下无需内部环形网络
    if (num_cores_ <= 1) {
        optimized_ring_ = nullptr;
        return;
    }
    
    // Phase5：仅保留 OptimizedInternalRing 作为片上互连后端（legacy InternalRing 已冻结/移除）
    if (!use_optimized_ring_) {
        output_->fatal(CALL_INFO, -1,
            "❌ 配置错误：use_optimized_ring=0 (legacy InternalRing) 已冻结/不再支持，请设置 use_optimized_ring=1\n");
    }

    // 使用新的 OptimizedInternalRing
    int num_vcs = 2;                // 每方向2个虚拟通道
    uint32_t credits_per_vc = 8;    // 每VC 8个信用
    optimized_ring_ = new OptimizedInternalRing(num_cores_, num_vcs, credits_per_vc, output_);
}

void MultiCorePE::loadAndDistributeWeights() {
    if (weights_file_.empty()) {
        PE_LOG(2, "⚠️ 未指定权重文件，使用默认权重\n");
        return;
    }

    // 当前架构下权重文件通过核心参数传递给各个 SnnPESubComponent，
    // 这里保持幂等 no-op，避免误导性的“未实现”行为。
    PE_LOG(2, "ℹ️ weights_file 已通过 core_params 下发至子核心，跳过PE级重复分发\n");
}

void MultiCorePE::updateStatistics() {
    // 收集处理单元统计信息
    uint64_t total_spikes = 0;
    uint64_t total_fired = 0;
    double total_utilization = 0.0;
    
    for (int i = 0; i < num_cores_; i++) {
        total_spikes += unit_states_[i].spikes_processed;
        total_fired += unit_states_[i].neurons_fired;
        total_utilization += unit_states_[i].utilization;
    }

    // 更新统计信息
    if (stat_spikes_processed_) stat_spikes_processed_->addData(total_spikes);
    if (stat_neurons_fired_) stat_neurons_fired_->addData(total_fired);
    if (stat_avg_utilization_) stat_avg_utilization_->addData(total_utilization / num_cores_);
    for (auto& mod : workload_stats_modules_) {
        if (mod) mod->emitDeltas();
    }

    if (diag_fire_log_ && output_) {
        std::ostringstream oss;
        oss << "[diag-pe-fire] node=" << node_id_
            << " cycle=" << current_cycle_
            << " total_fired=" << total_fired
            << " per_core=";
        for (int i = 0; i < num_cores_; ++i) {
            if (i) oss << ",";
            oss << i << ":" << unit_states_[i].neurons_fired;
        }
        output_->verbose(CALL_INFO, 1, 0, "%s\n", oss.str().c_str());
    }
    
    // 详细调试信息
    if (verbose_ >= 3 && current_cycle_ % 10000 == 0) {
        PE_LOG(3, "📊 周期%" PRIu64 "统计: 脉冲=%" PRIu64 ", 发放=%" PRIu64 ", 利用率=%.2f\n",
                        current_cycle_, total_spikes, total_fired, (total_utilization / num_cores_) * 100.0);
    }
}

void MultiCorePE::generateTestTraffic() {
    // 检查是否已达到最大测试脉冲数限制
    if (test_max_spikes_ > 0 && test_spikes_sent_ >= test_max_spikes_) {
        return;  // 已达到限制，停止生成测试流量
    }
    
    test_cycle_counter_++;
    
    if (test_cycle_counter_ >= static_cast<uint64_t>(test_period_)) {
        test_cycle_counter_ = 0;
        
        // 计算本次可发送的脉冲数
        int spikes_to_send = test_spikes_per_burst_;
        if (test_max_spikes_ > 0) {
            spikes_to_send = std::min(spikes_to_send, test_max_spikes_ - test_spikes_sent_);
        }
        
        if (spikes_to_send > 0) {
            PE_LOG(4, "🔥 生成测试流量: %d个脉冲 (已发送%d/%d)\n", 
                            spikes_to_send, test_spikes_sent_, test_max_spikes_);
            
            for (int i = 0; i < spikes_to_send; i++) {
                // 创建测试脉冲
                int src_neuron = node_id_ * total_neurons_ + (i % total_neurons_);
                int dst_neuron = test_target_node_ * total_neurons_ + (i % total_neurons_);

                // 使用配置的目标节点，避免被错误地回送到自身
                SpikeEvent* test_spike = new SpikeEvent(src_neuron, dst_neuron, static_cast<uint32_t>(test_target_node_),
                                                        test_weight_, current_cycle_);
                
                // 发送外部脉冲
                sendExternalSpike(test_spike);
                test_spikes_sent_++;
            }
        }
    }
}

void MultiCorePE::checkLoadBalance() {
    if (!controller_) return;
    
    // 计算负载差异
    double max_util = 0.0, min_util = 1.0;
    for (int i = 0; i < num_cores_; i++) {
        double util = unit_states_[i].utilization;
        max_util = std::max(max_util, util);
        min_util = std::min(min_util, util);
    }
    
    double load_imbalance = max_util - min_util;
    if (load_imbalance > 0.3) {  // 30%负载差异阈值
        //                 load_imbalance * 100.0, max_util * 100.0, min_util * 100.0);
        
        controller_->balanceLoad();
    }
}

// ===== MultiCoreController 实现 =====

MultiCoreController::MultiCoreController(MultiCorePE* parent, SST::Output* output)
    : parent_pe_(parent), output_(output) {
    
    // 初始化负载均衡状态
    core_utilization_history_.resize(parent_pe_->num_cores_, 0.0);
    core_work_count_.resize(parent_pe_->num_cores_, 0);
    
    // 初始化统计变量
    total_work_distributed_ = 0;
    load_imbalance_count_ = 0;
    load_balance_threshold_ = 0.2;  // 20%负载差异阈值
    
}

MultiCoreController::~MultiCoreController() {
}

void MultiCoreController::scheduleWork() {
    // 简单的轮询调度策略
    // 实际实现中可以根据负载情况进行智能调度
    
    static int next_core = 0;
    
    // 轮询分配工作到下一个核心
    next_core = (next_core + 1) % parent_pe_->num_cores_;
    core_work_count_[next_core]++;
    total_work_distributed_++;
    
    //                 next_core, total_work_distributed_);
}

void MultiCoreController::balanceLoad() {
    
    int most_loaded = findMostLoadedCore();
    int least_loaded = findLeastLoadedCore();
    
    if (most_loaded != least_loaded && most_loaded >= 0 && least_loaded >= 0) {
        double load_diff = core_utilization_history_[most_loaded] - core_utilization_history_[least_loaded];
        
        if (load_diff > load_balance_threshold_) {
            redistributeWork();
            load_imbalance_count_++;
            
            //                most_loaded, core_utilization_history_[most_loaded] * 100.0,
            //                least_loaded, core_utilization_history_[least_loaded] * 100.0);
        }
    }
}

void MultiCoreController::tick() {
    // 每个时钟周期更新性能计数器
    updatePerformanceCounters();
}

void MultiCoreController::updatePerformanceCounters() {
    // 更新每个核心的利用率历史
    for (int i = 0; i < parent_pe_->num_cores_; i++) {
        const auto& state = parent_pe_->getProcessingUnitState(i);
        
        // 使用指数移动平均更新利用率历史
        double alpha = 0.1;  // 平滑因子
        core_utilization_history_[i] = alpha * state.utilization + 
                                      (1.0 - alpha) * core_utilization_history_[i];
    }
}

double MultiCoreController::getCoreUtilization(int core_id) const {
    if (core_id >= 0 && core_id < parent_pe_->num_cores_) {
        return core_utilization_history_[core_id];
    }
    return 0.0;
}

double MultiCoreController::getOverallUtilization() const {
    if (parent_pe_->num_cores_ == 0) return 0.0;
    
    double total_util = 0.0;
    for (int i = 0; i < parent_pe_->num_cores_; i++) {
        total_util += core_utilization_history_[i];
    }
    
    return total_util / parent_pe_->num_cores_;
}

void MultiCoreController::redistributeWork() {
    // 简化的工作重分布策略
    // 实际实现中可能需要迁移脉冲队列或调整权重分布
    
    int most_loaded = findMostLoadedCore();
    int least_loaded = findLeastLoadedCore();
    
    if (most_loaded >= 0 && least_loaded >= 0 && most_loaded != least_loaded) {
        // 将一些工作从最繁忙的核心转移到最空闲的核心
        uint64_t work_to_transfer = core_work_count_[most_loaded] / 10;  // 转移10%的工作
        
        core_work_count_[most_loaded] -= work_to_transfer;
        core_work_count_[least_loaded] += work_to_transfer;
        
        //                 most_loaded, least_loaded, work_to_transfer);
    }
}

int MultiCoreController::findLeastLoadedCore() const {
    int least_loaded = 0;
    double min_utilization = core_utilization_history_[0];
    
    for (int i = 1; i < parent_pe_->num_cores_; i++) {
        if (core_utilization_history_[i] < min_utilization) {
            min_utilization = core_utilization_history_[i];
            least_loaded = i;
        }
    }
    
    return least_loaded;
}

int MultiCoreController::findMostLoadedCore() const {
    int most_loaded = 0;
    double max_utilization = core_utilization_history_[0];
    
    for (int i = 1; i < parent_pe_->num_cores_; i++) {
        if (core_utilization_history_[i] > max_utilization) {
            max_utilization = core_utilization_history_[i];
            most_loaded = i;
        }
    }
    
    return most_loaded;
}

// ===== SnnPEParentInterface 实现 =====

void MultiCorePE::sendSpike(SpikeEvent* event) {
    if (!event) return;

    PE_LOG(4, "📤 从SubComponent接收脉冲: 源神经元%d -> 目标神经元%d\n",
                    event->getSourceNeuron(), event->getDestinationNeuron());

    // Phase3-C：MultiCorePE 不直接做 Spike 编码；委托 SpikePacketBridge 生成 packet 并走 NoC。
    spike_packet_bridge_.sendAuto(event);
}

void MultiCorePE::requestMemoryAccess(uint64_t address, size_t size, 
                                    std::function<void(const void*)> callback) {
    (void)callback;
    output_->fatal(
        CALL_INFO, -1,
        "MultiCorePE fatal: requestMemoryAccess 已废弃/禁止使用（避免引入权重语义耦合）。"
        "请改走 services/memory/StandardMemAccess + services/synapse/weights 事务链路；addr=0x%" PRIx64 " size=%zu\n",
        static_cast<uint64_t>(address), size);
}

void MultiCorePE::deliverPacketToEndpoint_(int endpoint_id, NocPacketEvent* pkt) {
    if (!pkt) return;

    // step_seq 门控（naive_raw baseline）：禁用“步内级联”，将 future-step spike 暂存到对应 step 再投递。
    if (exec_mode_ == "naive_raw" &&
        global_step_sync_enable_ &&
        global_step_active_seq_ != 0 &&
        (pkt->packetKind() == NocPacketKind::Spike ||
         pkt->packetKind() == NocPacketKind::SpikeKey ||
         pkt->packetKind() == NocPacketKind::SpikeTileKey)) {
        const uint32_t target_seq = pkt->step_seq;
        if (target_seq != 0 && target_seq > global_step_active_seq_) {
            deferred_packets_by_seq_[target_seq].push_back(DeferredNocPacket{endpoint_id, pkt});
            return;
        }
    }

    // NoC e2e latency (cycles): measured at PE drain boundary (after mesh + local delivery).
    {
        const uint64_t send_ts = pkt->timestamp;
        const uint64_t now_ts = current_cycle_;
        const uint64_t lat = (now_ts >= send_ts) ? (now_ts - send_ts) : 0;
        const uint64_t bin = (lat <= static_cast<uint64_t>(noc_lat_hist_max_)) ? lat : (static_cast<uint64_t>(noc_lat_hist_max_) + 1u);
        if (pkt->packetKind() == NocPacketKind::Spike) {
            noc_lat_spike_cnt_ += 1;
            noc_lat_spike_sum_ += lat;
            noc_lat_spike_max_ = std::max<uint64_t>(noc_lat_spike_max_, lat);
            if (bin < noc_lat_spike_hist_.size()) noc_lat_spike_hist_[static_cast<size_t>(bin)] += 1;
        } else if (pkt->packetKind() == NocPacketKind::SpikeKey ||
                   pkt->packetKind() == NocPacketKind::SpikeTileKey) {
            noc_lat_spikekey_cnt_ += 1;
            noc_lat_spikekey_sum_ += lat;
            noc_lat_spikekey_max_ = std::max<uint64_t>(noc_lat_spikekey_max_, lat);
            if (bin < noc_lat_spikekey_hist_.size()) noc_lat_spikekey_hist_[static_cast<size_t>(bin)] += 1;
        }
    }
    if (pkt->packetKind() == NocPacketKind::Spike) {
        // Phase8 (strict): 平台层不再对 Spike packet 做任何 SpikeEvent 语义处理。
        // Spike 的编解码/处理必须由 core/workload 通过 deliverPacket() 完成；
        // 若 core 不接管，则直接 fail-fast，避免边界回退造成“看似能跑但语义漂移/非确定性”。
        if (endpoint_id < 0 || endpoint_id >= num_cores_) {
            output_->fatal(CALL_INFO, -1,
                           "Phase8(strict): invalid endpoint_id=%d for Spike packet (kind=%u) on node=%d\n",
                           endpoint_id, static_cast<unsigned>(pkt->kind), node_id_);
        }
        if (!cores_[endpoint_id]) {
            output_->fatal(CALL_INFO, -1,
                           "Phase8(strict): endpoint(core)=%d not configured for Spike packet (kind=%u) on node=%d\n",
                           endpoint_id, static_cast<unsigned>(pkt->kind), node_id_);
        }
        const bool taken = cores_[endpoint_id]->deliverPacket(pkt);
        if (!taken) {
            output_->fatal(CALL_INFO, -1,
                           "Phase8(strict): core=%d refused Spike packet (kind=%u); legacy deliverSpike fallback is disabled\n",
                           endpoint_id, static_cast<unsigned>(pkt->kind));
        }
        return; // packet owned by core
    }

    if (endpoint_id < 0 || endpoint_id >= num_cores_) {
        PE_LOG(2, "⚠️ 无效 endpoint_id=%d，丢弃 packet(kind=%u)\n",
               endpoint_id, static_cast<unsigned>(pkt->kind));
        delete pkt;
        return;
    }
    if (!cores_[endpoint_id]) {
        PE_LOG(2, "⚠️ endpoint(core)=%d 未配置，丢弃 packet(kind=%u)\n",
               endpoint_id, static_cast<unsigned>(pkt->kind));
        delete pkt;
        return;
    }

    const bool taken = cores_[endpoint_id]->deliverPacket(pkt);
    if (!taken) {
        PE_LOG(2, "⚠️ core=%d 未处理 packet(kind=%u)，已回收\n",
               endpoint_id, static_cast<unsigned>(pkt->kind));
        delete pkt;
    }
}

void MultiCorePE::resetAllCoreMembranes() {
    for (auto* core : cores_) {
        if (!core) continue;
        // Legacy-only hook: resetMembraneState is not part of CoreShellAPI.
        // It is used only by StepActivationSubsystem's optional reset_mem_each_step behavior.
        if (auto* snn = dynamic_cast<SnnCoreAPI*>(core)) {
            snn->resetMembraneState(v_rest_);
        }
    }
}


void MultiCorePE::initializeDirectionLinks() {
    
    // 配置方向链路，仅在实际连接时创建处理器
    north_link_ = configureLink("north", 
        new Event::Handler2<MultiCorePE,&MultiCorePE::handleNorthLinkEvent>(this));
    south_link_ = configureLink("south", 
        new Event::Handler2<MultiCorePE,&MultiCorePE::handleSouthLinkEvent>(this));
    east_link_ = configureLink("east", 
        new Event::Handler2<MultiCorePE,&MultiCorePE::handleEastLinkEvent>(this));
    west_link_ = configureLink("west", 
        new Event::Handler2<MultiCorePE,&MultiCorePE::handleWestLinkEvent>(this));
    // 重要：当 network_interface 子组件使用 SHARE_PORTS 时，它会占用/配置父组件的 "network" 端口。
    // 若父组件在此处再次 configureLink("network")，会造成端口重复绑定/回调歧义（同一端口被多处注册 handler）。
    // 因此：当已装配 external_nic_ 时，父组件不再直接配置 "network" 方向端口，避免连接冲突。
    if (!external_nic_) {
        network_link_ = configureLink("network",
            new Event::Handler2<MultiCorePE,&MultiCorePE::handleNetworkLinkEvent>(this));
    } else {
        network_link_ = nullptr;
    }
    
    // 统计活跃的方向链路
    int active_links = 0;
    if (north_link_) active_links++;
    if (south_link_) active_links++;
    if (east_link_) active_links++;
    if (west_link_) active_links++;
    if (network_link_) active_links++;
    
}

void MultiCorePE::initializeNetworkInterface() {
    
    // 尝试加载用户配置的网络接口
    // 关键修复：使用SHARE_PORTS允许网络接口暴露端口给hr_router
    external_nic_ = loadUserSubComponent<SnnInterface>(
        "network_interface", ComponentInfo::SHARE_PORTS);
    
    if (external_nic_) {
        
        // 配置网络接口的节点ID
        external_nic_->setNodeId(node_id_);
        
        // 设置通用接收回调：NoC packet 与控制面事件在此处分流
        external_nic_->setReceiveHandler([this](SST::Event* ev) {
            if (!ev) return;
            // Phase5‑5.4：batch unpack/NoC 输入收敛到 NocSubsystem；MultiCorePE 仅做事件分流与装配。
            if (dynamic_cast<NocPacketEvent*>(ev) || dynamic_cast<NocPacketBatchEvent*>(ev)) {
                this->noc_subsys_.onNicReceiveEvent(ev);
                return;
            }
            if (auto* gd = dynamic_cast<GatingDecisionEvent*>(ev)) {
                const uint32_t src_global =
                    static_cast<uint32_t>(gd->src_pe) * static_cast<uint32_t>(total_neurons_) + gd->src_row;
                std::vector<uint32_t> dpes = gd->dest_pes;
                for (auto* core : cores_) {
                    if (!core) continue;
                    auto* hooks = dynamic_cast<ICoreControlHooks*>(core);
                    if (!hooks) continue;
                    hooks->applyGatingDecision(src_global, dpes, current_cycle_, gd->ttl_cycles);
                }
                delete gd;
                return;
            }
            delete ev;
        });

        // 注意：SST框架会自动调用SubComponent的init()和setup()方法
        // 手动调用可能导致重复初始化和时序问题，因此移除
        
        //                 external_nic_->getNetworkStatus().c_str());
        
        // === 端口代理机制：将父组件的方向链路注入给SnnNetworkAdapter ===
        
        // 尝试将SnnInterface强制转换为SnnNetworkAdapter以访问链路注入接口
        auto* network_adapter = dynamic_cast<SnnNetworkAdapter*>(external_nic_);
        if (network_adapter) {
            // 注入各个方向的链路（如果存在）
            if (north_link_) {
                // network_adapter->injectDirectionLink("north", north_link_);
            }
            if (south_link_) {
                // network_adapter->injectDirectionLink("south", south_link_);
            }
            if (east_link_) {
                // network_adapter->injectDirectionLink("east", east_link_);
            }
            if (west_link_) {
                // network_adapter->injectDirectionLink("west", west_link_);
            }
            if (network_link_) {
                // network_adapter->injectDirectionLink("network", network_link_);
            }
            
        } else {
            // 非 SnnNetworkAdapter 场景（如 SnnNIC）：无需端口注入，外部NIC自带 network 端口
            if (external_nic_) {
            }
        }
    } else {
    }
}

// === 网络端口事件处理器实现 ===

void MultiCorePE::handleNorthLinkEvent(SST::Event* event) {
    PE_LOG(3, "📡 收到北向链路事件\n");
    if (!external_nic_) {
        PE_LOG(2, "⚠️ 网络接口未配置，无法处理north方向事件\n");
        delete event;
        return;
    }
    noc_subsys_.onDirectionalLinkEvent(event, "north");
}

void MultiCorePE::handleSouthLinkEvent(SST::Event* event) {
    PE_LOG(3, "📡 收到南向链路事件\n");
    if (!external_nic_) {
        PE_LOG(2, "⚠️ 网络接口未配置，无法处理south方向事件\n");
        delete event;
        return;
    }
    noc_subsys_.onDirectionalLinkEvent(event, "south");
}

void MultiCorePE::handleEastLinkEvent(SST::Event* event) {
    PE_LOG(3, "📡 收到东向链路事件\n");
    if (!external_nic_) {
        PE_LOG(2, "⚠️ 网络接口未配置，无法处理east方向事件\n");
        delete event;
        return;
    }
    noc_subsys_.onDirectionalLinkEvent(event, "east");
}

void MultiCorePE::handleWestLinkEvent(SST::Event* event) {
    PE_LOG(3, "📡 收到西向链路事件\n");
    if (!external_nic_) {
        PE_LOG(2, "⚠️ 网络接口未配置，无法处理west方向事件\n");
        delete event;
        return;
    }
    noc_subsys_.onDirectionalLinkEvent(event, "west");
}

void MultiCorePE::handleNetworkLinkEvent(SST::Event* event) {
    PE_LOG(3, "📡 收到通用网络链路事件\n");
    if (!external_nic_) {
        PE_LOG(2, "⚠️ 网络接口未配置，无法处理network方向事件\n");
        delete event;
        return;
    }
    noc_subsys_.onDirectionalLinkEvent(event, "network");
}
