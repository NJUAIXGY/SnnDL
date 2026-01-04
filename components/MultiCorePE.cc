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
#include "NocPacketEvent.h"
#include "NocPacketBatchEvent.h"
#include "GasStepBarrierEvent.h"
#include "WorkloadConfig.h"
#include "SnnCoreAPI.h"

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

using namespace SST;
using namespace SST::SnnDL;

// Lightweight logging helpers (file-local)
#ifndef SNNDL_LOGPTR
#define SNNDL_LOGPTR(ptr, lvl, ...) do { if (ptr) (ptr)->verbose(CALL_INFO, (lvl), 0, __VA_ARGS__); } while(0)
#endif
#ifndef PE_LOG
#define PE_LOG(lvl, ...) SNNDL_LOGPTR(output_, (lvl), __VA_ARGS__)
#endif

// ===== MultiCorePE 主组件实现 =====

// P2: 环境变量前端化 – 不再使用TU级别的 getenv 缓存；改用构造期解析的成员 sentinel_enabled_

MultiCorePE::MultiCorePE(ComponentId_t id, Params& params) : Component(id) {
    // 初始化输出对象
    int verbose_level = params.find<int>("verbose", 0);
    output_ = new Output("MultiCorePE[@p:@l]: ", verbose_level, 0, Output::STDOUT);
    
    
    // 读取基础配置参数
    num_cores_ = params.find<int>("num_cores", 4);
    neurons_per_core_ = params.find<int>("neurons_per_core", 64);
    total_neurons_ = num_cores_ * neurons_per_core_;
    neurons_per_pe_cfg_ = params.find<uint32_t>("neurons_per_pe", 0);
    node_id_ = params.find<int>("node_id", 0);
    total_nodes_ = params.find<int>("total_nodes", 1);
    global_neuron_base_ = params.find<uint64_t>("global_neuron_base", 0);
    sim_stop_ns_ = params.find<uint64_t>("sim_stop_ns", 0);
    verbose_ = verbose_level;
    // P2: 解析 sentinel 与步级诊断参数（未设置则回退环境变量）
    {
        // P2 Step3: 移除运行期 getenv 回退；仅由参数驱动（默认禁用）
        int sent_p = params.find<int>("sentinel_enable", 0);
        sentinel_enabled_ = (sent_p != 0);
        progress_log_interval_ns_ = params.find<uint64_t>("progress_log_interval_ns", 0);
        progress_log_node_ = params.find<int>("progress_log_node", -1);
        step_diag_cap_cfg_ = params.find<long>("step_diag_cap", 0);
        step_diag_enable_cfg_ = params.find<int>("step_diag_enable", 0);
    }
    weights_file_ = params.find<std::string>("weights_file", "");
    enable_numa_ = params.find<bool>("enable_numa", true);

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
    v_thresh_ = params.find<float>("v_thresh", 1.0f);
    v_reset_ = params.find<float>("v_reset", 0.0f);
    v_rest_ = params.find<float>("v_rest", 0.0f);
    tau_mem_ = params.find<float>("tau_mem", 20.0f);
    t_ref_ = params.find<int>("t_ref", 2);
    
    // 测试流量参数
    enable_test_traffic_ = params.find<bool>("enable_test_traffic", false);
    test_target_node_ = params.find<int>("test_target_node", 0);
    test_period_ = params.find<int>("test_period", 100);
    test_spikes_per_burst_ = params.find<int>("test_spikes_per_burst", 4);
    test_weight_ = params.find<float>("test_weight", 0.2f);
    test_max_spikes_ = params.find<int>("test_max_spikes", 10);
    
    // 环形网络实现选择
    use_optimized_ring_ = params.find<bool>("use_optimized_ring", true);
    // Phase5：冻结 legacy InternalRing 分支（use_optimized_ring=0）
    // - NoC 子系统已以 OptimizedInternalRing 为唯一片上互连后端完成闭环
    // - legacy InternalRing 会引入维护成本与语义漂移风险，故在此明确禁用
    if (!use_optimized_ring_ && num_cores_ > 1) {
        output_->fatal(CALL_INFO, -1,
            "❌ 配置错误：use_optimized_ring=0 (legacy InternalRing) 已冻结/不再支持，请设置 use_optimized_ring=1\n");
    }
    // 输出控制：是否打印节点汇总
    print_node_summary_ = params.find<bool>("print_node_summary", true);
    primary_keepalive_ = params.find<bool>("primary_keepalive", false);
    manual_core_drive_enable_ = params.find<bool>("manual_core_drive_enable", false);
    manual_gas_gather_cycles_ = params.find<uint64_t>("manual_gas_gather_cycles", 200);
    
    // 权重验证参数
    verify_weights_ = params.find<bool>("verify_weights", false);
    weight_verify_samples_ = params.find<uint32_t>("weight_verify_samples", 16);
    expected_weight_value_ = params.find<float>("expected_weight_value", 0.5f);
    verify_log_each_sample_ = params.find<bool>("verify_log_each_sample", false);
    
    // 权重回退参数
    use_event_weight_fallback_ = params.find<bool>("use_event_weight_fallback", false);
    enable_memory_weights_ = params.find<bool>("enable_memory_weights", true);
    write_weights_on_init_ = params.find<bool>("write_weights_on_init", true);

    // 时间窗口化统计参数（默认关闭）
    window_stats_enable_ = params.find<bool>("window_stats_enable", false);
    window_us_ = params.find<uint64_t>("window_us", 20);
    window_csv_ = params.find<std::string>("window_csv", "");
    window_metrics_csv_ = params.find<std::string>("window_metrics_csv", "");
    window_ns_ = window_us_ * 1000ULL; // 1us = 1000ns（组件时钟1GHz，tick≈1ns）
    diag_fire_log_ = params.find<bool>("diag_fire_log", false);

    // Global Step/GAS barrier sync (Phase-step-sync)
    global_step_sync_enable_ = params.find<bool>("global_step_sync_enable", false);

    // Step-level random activation injection (Phase3-B): 下沉为独立子系统（MultiCorePE 仅转发 tick/阶段事件）
    {
        StepActivationSubsystem::Config step_cfg;
        step_cfg.enable = params.find<bool>("step_activation_enable", false);
        step_cfg.fraction = params.find<double>("step_activation_fraction", 0.0);
        step_cfg.fanout = params.find<uint32_t>("step_activation_fanout", 0);
        step_cfg.seed = params.find<uint64_t>("step_activation_seed", 0xdecafbadULL);
        step_cfg.period_cycles = params.find<uint64_t>("step_activation_period_cycles", 0);
        step_cfg.trigger_core = params.find<int>("step_activation_trigger_core", 0);
        step_cfg.reset_mem_each_step = params.find<bool>("step_reset_mem_each_step", false);
        step_cfg.event_weight = params.find<double>("step_activation_event_weight", 0.0);
        step_cfg.use_bcsr_routes = params.find<bool>("step_activation_use_bcsr_routes", false);
        step_cfg.bcsr_template = params.find<std::string>("step_activation_bcsr_template", "");
        step_cfg.bcsr_rows_per_core = params.find<uint32_t>(
            "step_activation_bcsr_rows_per_core", static_cast<uint32_t>(neurons_per_core_));
        step_cfg.bcsr_br = params.find<uint32_t>("step_activation_bcsr_br", 16);
        step_cfg.bcsr_bc = params.find<uint32_t>("step_activation_bcsr_bc", 16);
        step_cfg.bcsr_idx_bytes = params.find<uint32_t>("step_activation_bcsr_idx_bytes", 2);
        step_cfg.bcsr_val_bytes = params.find<uint32_t>("step_activation_bcsr_val_bytes", 4);
        step_cfg.bcsr_rowptr_offset = params.find<uint64_t>("step_activation_bcsr_rowptr_offset", 0);
        step_cfg.bcsr_colidx_offset = params.find<uint64_t>("step_activation_bcsr_colidx_offset", 0);
        step_cfg.bcsr_blockdata_offset = params.find<uint64_t>("step_activation_bcsr_blockdata_offset", 0);
        step_cfg.bcsr_blockids_offset = params.find<uint64_t>("step_activation_bcsr_blockids_offset", 0);
        step_cfg.bcsr_weight_epsilon = params.find<double>("step_activation_bcsr_weight_epsilon", 0.0);
        step_cfg.log_enable = params.find<bool>("step_activation_log_enable", false);
        step_cfg.build_local_only = params.find<bool>("step_activation_build_local_only", true);
        step_cfg.bcsr_align = params.find<uint64_t>("step_activation_bcsr_align", 64);

        // 通用 workload（例如 stream）下必须禁用 Step/Synapse 语义注入，否则会污染纯通信/纯内存负载。
        // 选择来源：优先 Params.workload_impl，其次环境变量 SNNDL_WORKLOAD_IMPL（保持脚本不改的兼容路径）。
        {
            std::string w = params.find<std::string>("workload_impl", "");
            if (w.empty()) {
                if (const char* env = workloadImplFromEnvCached()) w = std::string(env);
            }
            std::transform(w.begin(), w.end(), w.begin(),
                           [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (w == "stream") {
                step_cfg.enable = false;
                step_cfg.fraction = 0.0;
                step_cfg.fanout = 0;
                step_cfg.period_cycles = 0;
                step_cfg.use_bcsr_routes = false;
                step_cfg.bcsr_template.clear();
            }
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
        unit_states_[i].stream_mem_writes_issued_total = 0;
        unit_states_[i].stream_mem_reads_issued_total = 0;
        unit_states_[i].stream_mem_bytes_written_total = 0;
        unit_states_[i].stream_mem_bytes_read_total = 0;
        unit_states_[i].stream_mem_verify_pass_total = 0;
        unit_states_[i].stream_mem_verify_fail_total = 0;
        unit_states_[i].stream_pkt_sent_total = 0;
        unit_states_[i].stream_pkt_recv_total = 0;
        unit_states_[i].stream_pkt_bad_crc_total = 0;
        unit_states_[i].stream_pkt_bad_magic_total = 0;
        unit_states_[i].utilization = 0.0;
    }
    stream_mem_verify_fail_last_.assign(static_cast<size_t>(num_cores_), 0);
    stream_mem_verify_pass_last_.assign(static_cast<size_t>(num_cores_), 0);
    stream_mem_writes_issued_last_.assign(static_cast<size_t>(num_cores_), 0);
    stream_mem_reads_issued_last_.assign(static_cast<size_t>(num_cores_), 0);
    stream_mem_bytes_written_last_.assign(static_cast<size_t>(num_cores_), 0);
    stream_mem_bytes_read_last_.assign(static_cast<size_t>(num_cores_), 0);
    stream_pkt_sent_last_.assign(static_cast<size_t>(num_cores_), 0);
    stream_pkt_recv_last_.assign(static_cast<size_t>(num_cores_), 0);
    stream_pkt_bad_crc_last_.assign(static_cast<size_t>(num_cores_), 0);
    stream_pkt_bad_magic_last_.assign(static_cast<size_t>(num_cores_), 0);
    
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
        noc_rt.deliver_to_endpoint = [this](int endpoint_id, NocPacketEvent* pkt) {
            deliverPacketToEndpoint_(endpoint_id, pkt);
        };
        noc_subsys_.bindRuntime(noc_rt);
    }
    // 记录路径（若提供），用于派生输出目录
    stage_events_csv_path_ = params.find<std::string>("stage_events_csv", "");
    stats_csv_path_ = params.find<std::string>("stats_csv", "");
    
    // 关键修复：在构造函数中初始化网络接口，确保SST能在正确时机调用init()
    initializeNetworkInterface();
}

MultiCorePE::~MultiCorePE() {
    
    // 清理SnnPE SubComponent核心（SST会自动管理SubComponent的生命周期）
    cores_.clear();
    
    // 清理内部组件
    delete optimized_ring_;
    delete controller_;
    // 避免析构次序竞态：不手动delete日志对象
    output_ = nullptr;
}

void MultiCorePE::init(unsigned int phase) {
    if (sentinel_enabled_ && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=%u enter\n", node_id_, phase); }
    if (phase == 0) {
        if (primary_keepalive_ || sim_stop_ns_ > 0) {
            registerAsPrimaryComponent();
            primaryComponentDoNotEndSim();
        }
        // 阶段0：初始化基础组件和端口
        
        // 配置时钟
        std::string clock_freq = "1GHz";  // 默认时钟频率
        // 不需要单独的clock_handler_变量
        registerClock(clock_freq, new Clock::Handler2<MultiCorePE,&MultiCorePE::clockTick>(this));
        if (sentinel_enabled_ && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=0 clock-registered\n", node_id_); }
        
        
        // 初始化统计收集
        
        // 初始化端口连接
        external_spike_input_link_ = configureLink("external_spike_input", 
            new Event::Handler2<MultiCorePE,&MultiCorePE::handleExternalSpikeEvent>(this));
        external_spike_output_link_ = configureLink("external_spike_output");
        mem_link_ = configureLink("mem_link");
        if (global_step_sync_enable_) {
            gas_step_ctrl_link_ = configureLink(
                "gas_step_ctrl",
                new Event::Handler2<MultiCorePE, &MultiCorePE::handleGasStepCtrlEvent>(this));
            if (!gas_step_ctrl_link_) {
                output_->fatal(CALL_INFO, -1, "❌ 配置错误：global_step_sync_enable=1 但端口 gas_step_ctrl 未连接\n");
            }
        }
        if (sentinel_enabled_ && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=0 links-configured\n", node_id_); }
        
        
        // 初始化方向链路（用于端口代理机制）
        initializeDirectionLinks();
        if (sentinel_enabled_ && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=0 dir-links\n", node_id_); }
        
        // 初始化处理单元
        initializeProcessingUnits();
        if (sentinel_enabled_ && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=0 units-initialized\n", node_id_); }
        
        // 初始化内部互连
        initializeInternalRing();
        if (sentinel_enabled_ && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=0 ring-initialized\n", node_id_); }

        // Phase4-A1.2：NoC 子系统后端装配（NIC / ring / legacy link）
        {
            NocSubsystem::Runtime noc_rt;
            noc_rt.log = output_;
            noc_rt.node_id = node_id_;
            noc_rt.num_cores = num_cores_;
            noc_rt.nic = external_nic_;
            noc_rt.optimized_ring = optimized_ring_;
            noc_rt.external_spike_output_link = external_spike_output_link_;
            noc_rt.deliver_to_endpoint = [this](int endpoint_id, NocPacketEvent* pkt) {
                deliverPacketToEndpoint_(endpoint_id, pkt);
            };
            noc_subsys_.bindRuntime(noc_rt);
        }
        
        // 初始化多核控制器
        controller_ = new MultiCoreController(this, output_);
        if (sentinel_enabled_ && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=0 controller-created\n", node_id_); }
        

        // 将当前phase转发给所有子核心
        for (auto* core : cores_) {
            if (core) core->init(phase);
        }
        if (sentinel_enabled_ && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=0 cores-init-done\n", node_id_); }
        
        // 关键修复：转发init到网络接口
        if (external_nic_) {
            external_nic_->init(phase);
        }
        if (sentinel_enabled_ && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=0 nic-init-done\n", node_id_); }
        // 标记 Step 注入就绪（保证 NIC 已完成 init）
        step_activation_subsys_.setInjectionReady(true);
    }
    else if (phase == 1) {
        if (sentinel_enabled_ && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=1 enter\n", node_id_); }
        // 阶段1：加载权重和配置子组件
        loadAndDistributeWeights();
        if (sentinel_enabled_ && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=1 weights-loaded\n", node_id_); }

        // 将当前phase转发给所有子核心
        for (auto* core : cores_) {
            if (core) core->init(phase);
        }
        if (sentinel_enabled_ && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=1 cores-init-done\n", node_id_); }
        
        // 转发init到网络接口
        if (external_nic_) {
            external_nic_->init(phase);
        }
        if (sentinel_enabled_ && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=1 nic-init-done\n", node_id_); }
    }
    else {
        if (sentinel_enabled_ && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=%u forward-only\n", node_id_, phase); }
        // 其余phase同样转发
        for (auto* core : cores_) {
            if (core) core->init(phase);
        }
        
        // 转发init到网络接口
        if (external_nic_) {
            external_nic_->init(phase);
        }
        if (sentinel_enabled_ && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=%u done\n", node_id_, phase); }
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
}

void MultiCorePE::setup() {
    if (sentinel_enabled_ && output_) { output_->output("[[sentinel-pe-setup]] node=%d enter\n", node_id_); }
    
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
    if (sentinel_enabled_ && output_) { output_->output("[[sentinel-pe-setup]] node=%d cores-setup\n", node_id_); }
    
    // 调用网络接口的setup
    if (external_nic_) {
        external_nic_->setup();
    }
    if (sentinel_enabled_ && output_) { output_->output("[[sentinel-pe-setup]] node=%d nic-setup\n", node_id_); }
    
    if (!controller_) {
        output_->fatal(CALL_INFO, -1, "❌ 错误: 多核控制器未初始化\n");
    }

    global_step_sync_ready_ = true;
    if (global_step_sync_enable_ && gas_step_ctrl_link_ && !global_step_ready_sent_) {
        auto* ev = new GasStepBarrierEvent(GasStepBarrierOp::PeReady, /*seq*/0, static_cast<uint32_t>(node_id_));
        gas_step_ctrl_link_->send(ev);
        global_step_ready_sent_ = true;
    }
    
    // 打印组件配置摘要
    
}

void MultiCorePE::finish() {
    // 更新最终统计信息
    updateStatistics();
    // 报告总仿真周期（单实例）
    if (stat_sim_cycles_total_) stat_sim_cycles_total_->addData(current_cycle_);
    // 输出 PE 级 per-window 发放聚合（与 stage_events 同目录）
    if (!window_spikes_pe_.empty()) {
        std::string ref = stage_events_csv_path_;
        if (ref.empty()) ref = stats_csv_path_;
        std::string dir = ".";
        if (!ref.empty()) {
            auto pos = ref.find_last_of('/');
            dir = (pos==std::string::npos) ? std::string(".") : ref.substr(0,pos);
        }
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
        std::string ref = stage_events_csv_path_;
        if (ref.empty()) ref = stats_csv_path_;
        std::string dir = ".";
        if (!ref.empty()) {
            auto pos = ref.find_last_of('/');
            dir = (pos==std::string::npos) ? std::string(".") : ref.substr(0,pos);
        }
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

    // Global Step barrier: 当收到 START_STEP(seq) 时，在时钟边界打开所有 core 的新窗口
    if (global_step_sync_enable_ && global_step_sync_ready_ && global_step_start_pending_) {
        beginGlobalStep_(global_step_pending_seq_);
        global_step_start_pending_ = false;
    }
    
    // 0a. Step 注入调度（Phase3-B 下沉为 StepActivationSubsystem）
    step_activation_subsys_.tick(static_cast<uint64_t>(current_cycle_));

    // 0b. 测试注入：在首个有效周期从 core0 向 core1 注入一个跨核脉冲（仅当启用测试流量时）
		    if (enable_test_traffic_ && !test_injected_ && num_cores_ > 1 && current_cycle_ == 5000) {
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
    
    // 1. 处理外部脉冲队列（Phase4-A1.1：下沉至 NoC 子系统）
    noc_subsys_.drainIncomingQueue(static_cast<uint64_t>(current_cycle_));
    
    // 2. SubComponent时钟由SST自动管理，无需手动调用tick
    // 若子组件未被SST调度（某些环境组合下可能发生），则回退为手动驱动一拍，确保窗口推进与队列消费
    if (manual_core_drive_enable_) {
        for (int i = 0; i < num_cores_; i++) {
            if (cores_[i] != nullptr) {
	                if (auto* hooks = dynamic_cast<ICoreControlHooks*>(cores_[i])) {
	                    hooks->driveOneCycle();
	                    if (manual_gas_gather_cycles_ > 0 && (current_cycle_ % manual_gas_gather_cycles_) == 0) {
	                        PE_LOG(2, "[diag-PE] forceEndGather: core=%d cyc=%" PRIu64 " period=%" PRIu64 "\n",
	                               i, (uint64_t)current_cycle_, (uint64_t)manual_gas_gather_cycles_);
	                        if (auto* snn = dynamic_cast<SnnCoreAPI*>(cores_[i])) {
	                            snn->forceEndGather();
	                        }
	                    }
	                }
	            }
	        }
	    }
    // 更新处理单元状态统计（从SnnPE SubComponent获取实际数据）
    for (int i = 0; i < num_cores_; i++) {
        if (cores_[i] != nullptr) {
            std::map<std::string, uint64_t> core_stats;
            cores_[i]->getStatistics(core_stats);
            auto it_sp = core_stats.find("spikes_received");
            auto it_nf = core_stats.find("neurons_fired");
            auto it_sm_w = core_stats.find("stream_mem_writes_issued_total");
            auto it_sm_r = core_stats.find("stream_mem_reads_issued_total");
            auto it_sm_bw = core_stats.find("stream_mem_bytes_written_total");
            auto it_sm_br = core_stats.find("stream_mem_bytes_read_total");
            auto it_sm_vp = core_stats.find("stream_mem_verify_pass_total");
            auto it_sm_vf = core_stats.find("stream_mem_verify_fail_total");
            auto it_pkt_s = core_stats.find("stream_pkt_sent_total");
            auto it_pkt_r = core_stats.find("stream_pkt_recv_total");
            auto it_pkt_bc = core_stats.find("stream_pkt_bad_crc_total");
            auto it_pkt_bm = core_stats.find("stream_pkt_bad_magic_total");
            uint64_t old_spikes = unit_states_[i].spikes_processed;
            uint64_t new_spikes = (it_sp != core_stats.end()) ? it_sp->second : 0;
            unit_states_[i].spikes_processed = new_spikes;
            unit_states_[i].neurons_fired = (it_nf != core_stats.end()) ? it_nf->second : 0;
            unit_states_[i].stream_mem_writes_issued_total = (it_sm_w != core_stats.end()) ? it_sm_w->second : 0;
            unit_states_[i].stream_mem_reads_issued_total = (it_sm_r != core_stats.end()) ? it_sm_r->second : 0;
            unit_states_[i].stream_mem_bytes_written_total = (it_sm_bw != core_stats.end()) ? it_sm_bw->second : 0;
            unit_states_[i].stream_mem_bytes_read_total = (it_sm_br != core_stats.end()) ? it_sm_br->second : 0;
            unit_states_[i].stream_mem_verify_pass_total = (it_sm_vp != core_stats.end()) ? it_sm_vp->second : 0;
            unit_states_[i].stream_mem_verify_fail_total = (it_sm_vf != core_stats.end()) ? it_sm_vf->second : 0;
            unit_states_[i].stream_pkt_sent_total = (it_pkt_s != core_stats.end()) ? it_pkt_s->second : 0;
            unit_states_[i].stream_pkt_recv_total = (it_pkt_r != core_stats.end()) ? it_pkt_r->second : 0;
            unit_states_[i].stream_pkt_bad_crc_total = (it_pkt_bc != core_stats.end()) ? it_pkt_bc->second : 0;
            unit_states_[i].stream_pkt_bad_magic_total = (it_pkt_bm != core_stats.end()) ? it_pkt_bm->second : 0;
            unit_states_[i].utilization = cores_[i]->getUtilization();
            unit_states_[i].is_active = cores_[i]->hasWork();
            
            // 调试：跟踪统计数据变化 (已禁用避免过多输出)
            // if (new_spikes != old_spikes) {
            //     printf("DEBUG: 核心%d统计更新，节点%d - 旧值:%lu -> 新值:%lu (来自getStatistics)\n", 
            //            i, node_id_, old_spikes, new_spikes);
            //     fflush(stdout);
            // }
        } else {
            unit_states_[i].spikes_processed = 0;
            unit_states_[i].neurons_fired = 0;
            unit_states_[i].utilization = 0.0;
            unit_states_[i].is_active = false;
        }
    }
    
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

void MultiCorePE::handleGasStepCtrlEvent(SST::Event* ev) {
    if (!ev) return;
    auto* msg = dynamic_cast<GasStepBarrierEvent*>(ev);
    if (!msg) {
        delete ev;
        return;
    }
    if (msg->operation() == GasStepBarrierOp::StartStep) {
        global_step_pending_seq_ = msg->seq;
        global_step_start_pending_ = true;
        if (sentinel_enabled_ && output_) {
            PE_LOG(1, "[[sentinel-step-sync]] node=%d recv START_STEP seq=%u\n", node_id_, msg->seq);
        }
    }
    delete msg;
}

void MultiCorePE::beginGlobalStep_(uint32_t seq) {
    global_step_active_seq_ = seq;
    global_step_done_sent_ = false;
    if (global_step_done_cores_.size() != static_cast<size_t>(num_cores_)) {
        global_step_done_cores_.assign(static_cast<size_t>(num_cores_), 0);
    } else {
        std::fill(global_step_done_cores_.begin(), global_step_done_cores_.end(), 0);
    }

    for (int i = 0; i < num_cores_; ++i) {
        auto* core = cores_[i];
        if (!core) {
            output_->fatal(CALL_INFO, -1, "GlobalStep fatal: core%d is null\n", i);
            return;
        }
        auto* hook = dynamic_cast<IGlobalStepHooks*>(core);
        if (!hook) {
            output_->fatal(CALL_INFO, -1, "GlobalStep fatal: core%d does not implement IGlobalStepHooks\n", i);
            return;
        }
        hook->onGlobalStepStart(seq);
    }
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
    stat_stream_mem_verify_fail_total_ = registerStatistic<uint64_t>("stream_mem_verify_fail_total");
    stat_stream_mem_writes_issued_total_ = registerStatistic<uint64_t>("stream_mem_writes_issued_total");
    stat_stream_mem_reads_issued_total_ = registerStatistic<uint64_t>("stream_mem_reads_issued_total");
    stat_stream_mem_bytes_written_total_ = registerStatistic<uint64_t>("stream_mem_bytes_written_total");
    stat_stream_mem_bytes_read_total_ = registerStatistic<uint64_t>("stream_mem_bytes_read_total");
    stat_stream_mem_verify_pass_total_ = registerStatistic<uint64_t>("stream_mem_verify_pass_total");
    stat_stream_pkt_sent_total_ = registerStatistic<uint64_t>("stream_pkt_sent_total");
    stat_stream_pkt_recv_total_ = registerStatistic<uint64_t>("stream_pkt_recv_total");
    stat_stream_pkt_bad_crc_total_ = registerStatistic<uint64_t>("stream_pkt_bad_crc_total");
    stat_stream_pkt_bad_magic_total_ = registerStatistic<uint64_t>("stream_pkt_bad_magic_total");
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
    stat_gas_apply_acc_updates_total_ = registerStatistic<uint64_t>("gas_apply_acc_updates_total");
    stat_gas_acc_posts_touched_total_ = registerStatistic<uint64_t>("gas_acc_posts_touched_total");
    stat_gas_scatter_spikes_emitted_total_ = registerStatistic<uint64_t>("gas_scatter_spikes_emitted_total");
    stat_gas_acc_hwm_bytes_total_ = registerStatistic<uint64_t>("gas_acc_high_watermark_bytes_total");
    stat_gas_acc_spill_records_total_ = registerStatistic<uint64_t>("gas_acc_spill_records_total");
    stat_gas_acc_spilled_bytes_total_ = registerStatistic<uint64_t>("gas_acc_spilled_bytes_total");
    stat_gas_activity_f_ = registerStatistic<double>("gas_activity_f");
    stat_sim_cycles_total_ = registerStatistic<uint64_t>("sim_cycles_total");
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
                                        uint64_t window_buffer_max_bytes) {
    if (unique_reads && stat_gas_unique_reads_total_) stat_gas_unique_reads_total_->addData(unique_reads);
    if (unique_bytes && stat_gas_unique_bytes_total_) stat_gas_unique_bytes_total_->addData(unique_bytes);
    if (rowwin_triggers && stat_gas_rowwin_triggers_total_) stat_gas_rowwin_triggers_total_->addData(rowwin_triggers);
    if (rowwin_bytes && stat_gas_rowwin_bytes_total_) stat_gas_rowwin_bytes_total_->addData(rowwin_bytes);
    if (bursts && stat_gas_total_bursts_) stat_gas_total_bursts_->addData(bursts);
    if (payload_bytes && stat_gas_total_payload_bytes_) stat_gas_total_payload_bytes_->addData(payload_bytes);
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
        if (spikes_emitted > 0) {
            // 同步记录到窗口发放聚合，便于 finish() 写 pe_window_spikes_db.csv
            window_spikes_pe_[seq] += spikes_emitted;
        }
    }

    // Step 注入事件转发（Phase3-B 下沉为 StepActivationSubsystem）
    if (event == "BeginGather") {
        step_activation_subsys_.onBeginGather(seq, ts_ns, core_id);
    } else if (event == "EndScatter") {
        step_activation_subsys_.onEndScatter(seq);
    }

    // Global Step barrier: 当本 PE 的所有 core 都完成 EndScatter(seq) 后，上报给控制器
    if (global_step_sync_enable_ &&
        gas_step_ctrl_link_ &&
        !global_step_done_sent_ &&
        event == "EndScatter" &&
        seq == global_step_active_seq_) {
        if (core_id >= 0 && core_id < num_cores_) {
            global_step_done_cores_[static_cast<size_t>(core_id)] = 1;
            bool all_done = true;
            for (auto v : global_step_done_cores_) {
                if (!v) { all_done = false; break; }
            }
            if (all_done) {
                auto* ev = new GasStepBarrierEvent(GasStepBarrierOp::PeDone, seq, static_cast<uint32_t>(node_id_));
                gas_step_ctrl_link_->send(ev);
                global_step_done_sent_ = true;
                if (sentinel_enabled_ && output_) {
                    PE_LOG(1, "[[sentinel-step-sync]] node=%d send PE_DONE seq=%u\n", node_id_, seq);
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
        // ★ 修正：每个核心需要能够接受整个PE的神经元范围，而不是只接受自己的4个神经元
        // 这样可以避免"无法映射的目标神经元"错误
        core_params.insert("num_neurons", std::to_string(num_cores_ * neurons_per_core_));
        core_params.insert("global_neuron_base", std::to_string(global_neuron_base_));
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
    
    
    // TODO: 实现权重加载和分布逻辑
    // 这里应该从文件加载权重并分发到各个处理单元
    
}

void MultiCorePE::updateStatistics() {
    // 收集处理单元统计信息
    uint64_t total_spikes = 0;
    uint64_t total_fired = 0;
    double total_utilization = 0.0;
    uint64_t stream_mem_verify_fail_delta = 0;
    uint64_t stream_mem_verify_pass_delta = 0;
    uint64_t stream_mem_writes_issued_delta = 0;
    uint64_t stream_mem_reads_issued_delta = 0;
    uint64_t stream_mem_bytes_written_delta = 0;
    uint64_t stream_mem_bytes_read_delta = 0;
    uint64_t stream_pkt_sent_delta = 0;
    uint64_t stream_pkt_recv_delta = 0;
    uint64_t stream_pkt_bad_crc_delta = 0;
    uint64_t stream_pkt_bad_magic_delta = 0;
    
    for (int i = 0; i < num_cores_; i++) {
        total_spikes += unit_states_[i].spikes_processed;
        total_fired += unit_states_[i].neurons_fired;
        total_utilization += unit_states_[i].utilization;

        if (static_cast<size_t>(i) < stream_mem_verify_fail_last_.size()) {
            const size_t idx = static_cast<size_t>(i);
            auto delta_of = [](uint64_t cur, uint64_t prev) -> uint64_t {
                return (cur >= prev) ? (cur - prev) : cur;
            };

            // stream mem verify
            {
                const uint64_t cur = unit_states_[i].stream_mem_verify_fail_total;
                const uint64_t prev = stream_mem_verify_fail_last_[idx];
                stream_mem_verify_fail_delta += delta_of(cur, prev);
                stream_mem_verify_fail_last_[idx] = cur;
            }
            {
                const uint64_t cur = unit_states_[i].stream_mem_verify_pass_total;
                const uint64_t prev = stream_mem_verify_pass_last_[idx];
                stream_mem_verify_pass_delta += delta_of(cur, prev);
                stream_mem_verify_pass_last_[idx] = cur;
            }

            // stream mem io
            {
                const uint64_t cur = unit_states_[i].stream_mem_writes_issued_total;
                const uint64_t prev = stream_mem_writes_issued_last_[idx];
                stream_mem_writes_issued_delta += delta_of(cur, prev);
                stream_mem_writes_issued_last_[idx] = cur;
            }
            {
                const uint64_t cur = unit_states_[i].stream_mem_reads_issued_total;
                const uint64_t prev = stream_mem_reads_issued_last_[idx];
                stream_mem_reads_issued_delta += delta_of(cur, prev);
                stream_mem_reads_issued_last_[idx] = cur;
            }
            {
                const uint64_t cur = unit_states_[i].stream_mem_bytes_written_total;
                const uint64_t prev = stream_mem_bytes_written_last_[idx];
                stream_mem_bytes_written_delta += delta_of(cur, prev);
                stream_mem_bytes_written_last_[idx] = cur;
            }
            {
                const uint64_t cur = unit_states_[i].stream_mem_bytes_read_total;
                const uint64_t prev = stream_mem_bytes_read_last_[idx];
                stream_mem_bytes_read_delta += delta_of(cur, prev);
                stream_mem_bytes_read_last_[idx] = cur;
            }

            // stream packets
            {
                const uint64_t cur = unit_states_[i].stream_pkt_sent_total;
                const uint64_t prev = stream_pkt_sent_last_[idx];
                stream_pkt_sent_delta += delta_of(cur, prev);
                stream_pkt_sent_last_[idx] = cur;
            }
            {
                const uint64_t cur = unit_states_[i].stream_pkt_recv_total;
                const uint64_t prev = stream_pkt_recv_last_[idx];
                stream_pkt_recv_delta += delta_of(cur, prev);
                stream_pkt_recv_last_[idx] = cur;
            }
            {
                const uint64_t cur = unit_states_[i].stream_pkt_bad_crc_total;
                const uint64_t prev = stream_pkt_bad_crc_last_[idx];
                stream_pkt_bad_crc_delta += delta_of(cur, prev);
                stream_pkt_bad_crc_last_[idx] = cur;
            }
            {
                const uint64_t cur = unit_states_[i].stream_pkt_bad_magic_total;
                const uint64_t prev = stream_pkt_bad_magic_last_[idx];
                stream_pkt_bad_magic_delta += delta_of(cur, prev);
                stream_pkt_bad_magic_last_[idx] = cur;
            }
        }
    }
    
    // 更新统计信息
    if (stat_spikes_processed_) stat_spikes_processed_->addData(total_spikes);
    stat_neurons_fired_->addData(total_fired);
    stat_avg_utilization_->addData(total_utilization / num_cores_);
    // Stream stats: always addData (even 0) so that CSV has stable keys for summary/DoD checks.
    if (stat_stream_mem_verify_fail_total_) stat_stream_mem_verify_fail_total_->addData(stream_mem_verify_fail_delta);
    if (stat_stream_mem_verify_pass_total_) stat_stream_mem_verify_pass_total_->addData(stream_mem_verify_pass_delta);
    if (stat_stream_mem_writes_issued_total_) stat_stream_mem_writes_issued_total_->addData(stream_mem_writes_issued_delta);
    if (stat_stream_mem_reads_issued_total_) stat_stream_mem_reads_issued_total_->addData(stream_mem_reads_issued_delta);
    if (stat_stream_mem_bytes_written_total_) stat_stream_mem_bytes_written_total_->addData(stream_mem_bytes_written_delta);
    if (stat_stream_mem_bytes_read_total_) stat_stream_mem_bytes_read_total_->addData(stream_mem_bytes_read_delta);
    if (stat_stream_pkt_sent_total_) stat_stream_pkt_sent_total_->addData(stream_pkt_sent_delta);
    if (stat_stream_pkt_recv_total_) stat_stream_pkt_recv_total_->addData(stream_pkt_recv_delta);
    if (stat_stream_pkt_bad_crc_total_) stat_stream_pkt_bad_crc_total_->addData(stream_pkt_bad_crc_delta);
    if (stat_stream_pkt_bad_magic_total_) stat_stream_pkt_bad_magic_total_->addData(stream_pkt_bad_magic_delta);

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

#if 0
// Phase3-B: StepActivationSubsystem 已接管 Step 注入与 BCSR 路由解析；
// 下面的旧实现仅保留作参考，避免中途回滚时丢失上下文。
std::string MultiCorePE::formatBcsrPath_(int pe, int core) const {
    if (step_activation_bcsr_template_.empty()) return std::string();
    std::string path = step_activation_bcsr_template_;
    // replace {pe[:width]}
    auto posp = path.find("{pe");
    if (posp != std::string::npos) {
        auto endp = path.find('}', posp);
        if (endp == std::string::npos) return std::string();
        int widthp = 0;
        auto colonp = path.find(':', posp);
        if (colonp != std::string::npos && colonp < endp) {
            auto spec_endp = path.find_first_of("diu", colonp);
            if (spec_endp != std::string::npos && spec_endp < endp) {
                std::string width_str = path.substr(colonp + 1, spec_endp - colonp - 1);
                widthp = std::atoi(width_str.c_str());
            }
        }
        std::ostringstream ossp;
        if (widthp > 0) {
            ossp << std::setfill('0') << std::setw(widthp);
        }
        ossp << pe;
        path.replace(posp, endp - posp + 1, ossp.str());
    }
    auto pos = path.find("{core");
    if (pos == std::string::npos) return path;
    auto end = path.find('}', pos);
    if (end == std::string::npos) return std::string();
    int width = 0;
    auto colon = path.find(':', pos);
    if (colon != std::string::npos && colon < end) {
        auto spec_end = path.find_first_of("diu", colon);
        if (spec_end != std::string::npos && spec_end < end) {
            std::string width_str = path.substr(colon + 1, spec_end - colon - 1);
            width = std::atoi(width_str.c_str());
        }
    }
    std::ostringstream oss;
    if (width > 0) {
        oss << std::setfill('0') << std::setw(width);
    }
    oss << core;
    path.replace(pos, end - pos + 1, oss.str());
    return path;
}

bool MultiCorePE::computeBcsrOffsets_(uint32_t n_block_rows, uint32_t total_blocks,
                                      uint64_t block_bytes,
                                      uint64_t& rowptr_offset, uint64_t& colidx_offset,
                                      uint64_t& blockdata_offset, uint64_t& blockids_offset) const {
    const uint64_t align = step_activation_bcsr_align_ ? step_activation_bcsr_align_ : 64;
    rowptr_offset = 0;
    colidx_offset = alignUp_(rowptr_offset + (uint64_t)(n_block_rows + 1) * sizeof(uint32_t), align);
    blockdata_offset = alignUp_(colidx_offset + (uint64_t)total_blocks * step_activation_bcsr_idx_bytes_, align);
    blockids_offset  = alignUp_(blockdata_offset + (uint64_t)total_blocks * block_bytes, align);
    return true;
}

bool MultiCorePE::checkBcsrOffsets_(uint64_t file_size, uint32_t n_block_rows,
                                    uint32_t total_blocks, uint64_t block_bytes,
                                    uint64_t& rowptr_offset, uint64_t& colidx_offset,
                                    uint64_t& blockdata_offset, uint64_t& blockids_offset) const {
    auto valid = [&](uint64_t off) { return off < file_size; };
    if (!valid(rowptr_offset) || !valid(colidx_offset) ||
        !valid(blockdata_offset) || !valid(blockids_offset)) {
        computeBcsrOffsets_(n_block_rows, total_blocks, block_bytes,
                            rowptr_offset, colidx_offset, blockdata_offset, blockids_offset);
    }
    if (rowptr_offset >= file_size) return false;
    if (colidx_offset >= file_size) return false;
    if (blockdata_offset >= file_size) return false;
    if (blockids_offset >= file_size) return false;
    const uint64_t need_rowptr = (uint64_t)(n_block_rows + 1) * sizeof(uint32_t);
    const uint64_t need_colidx = (uint64_t)total_blocks * step_activation_bcsr_idx_bytes_;
    const uint64_t need_block  = (uint64_t)total_blocks * block_bytes;
    if (rowptr_offset + need_rowptr > file_size) return false;
    if (colidx_offset + need_colidx > file_size) return false;
    if (blockdata_offset + need_block > file_size) return false;
    if (blockids_offset + need_block > file_size) return false;
    return true;
}

uint64_t MultiCorePE::alignUp_(uint64_t value, uint64_t align) const {
    if (align == 0) return value;
    uint64_t rem = value % align;
    return rem ? (value + align - rem) : value;
}

bool MultiCorePE::buildRoutesFromBcsrFile_(const std::string& path, uint32_t pe_id, uint32_t core_index) {
    const uint32_t rows_per_core = step_activation_bcsr_rows_per_core_;
    const uint32_t br = step_activation_bcsr_br_ ? step_activation_bcsr_br_ : 16;
    const uint32_t bc = step_activation_bcsr_bc_ ? step_activation_bcsr_bc_ : 16;
    const uint32_t n_block_rows = (rows_per_core + br - 1) / br;
    const size_t floats_per_block = static_cast<size_t>(br) * static_cast<size_t>(bc);
    const uint32_t neurons_per_pe = static_cast<uint32_t>(neurons_per_core_) * static_cast<uint32_t>(num_cores_);
    const uint32_t local_pre_begin = (neurons_per_pe > 0) ? static_cast<uint32_t>(node_id_) * neurons_per_pe : 0u;
    const uint32_t local_pre_end = local_pre_begin + neurons_per_pe;
    const uint32_t max_global = (total_nodes_ > 0 && neurons_per_pe > 0)
        ? static_cast<uint32_t>(total_nodes_) * neurons_per_pe
        : 0u;

    std::ifstream fin(path, std::ios::binary);
    if (!fin.good()) {
        output_->verbose(CALL_INFO, 0, 0, "⚠️ 无法读取BCSR文件: %s\n", path.c_str());
        return false;
    }

    // 文件大小与可用区间检查，防止越界读取
    fin.seekg(0, std::ios::end);
    const std::streamoff file_size = fin.tellg();
    fin.clear();
    fin.seekg(0, std::ios::beg);
    uint64_t rowptr_off = step_activation_bcsr_rowptr_offset_;
    uint64_t colidx_off = step_activation_bcsr_colidx_offset_;
    uint64_t blockdata_off = step_activation_bcsr_blockdata_offset_;
    uint64_t blockids_off = step_activation_bcsr_blockids_offset_;
    const uint64_t bytes_per_block_data = floats_per_block * sizeof(float);
    const uint64_t bytes_per_block_ids  = floats_per_block * sizeof(uint32_t);
    const uint64_t avail_rowptr_bytes = (rowptr_off < (uint64_t)file_size) ? ((uint64_t)file_size - rowptr_off) : 0ULL;
    const uint64_t avail_colidx_bytes = (colidx_off < blockdata_off && blockdata_off <= (uint64_t)file_size)
        ? (blockdata_off - colidx_off) : 0ULL;
    const uint64_t avail_blockdata_bytes = (blockdata_off < (uint64_t)file_size) ? ((uint64_t)file_size - blockdata_off) : 0ULL;
    const uint64_t avail_blockids_bytes  = (blockids_off  < (uint64_t)file_size) ? ((uint64_t)file_size - blockids_off)  : 0ULL;

    // 读取 rowptr（按可用长度截断）
    fin.seekg(step_activation_bcsr_rowptr_offset_, std::ios::beg);
    const uint64_t want_rowptr_bytes = (uint64_t)(n_block_rows + 1) * sizeof(uint32_t);
    const uint64_t take_rowptr_bytes = std::min<uint64_t>(want_rowptr_bytes, avail_rowptr_bytes);
    const uint32_t rowptr_elems = (uint32_t)(take_rowptr_bytes / sizeof(uint32_t));
    if (rowptr_elems < 2) {
        output_->verbose(CALL_INFO, 0, 0, "⚠️ rowptr区域不足: have=%" PRIu64 " need=%" PRIu64 " file=%lld\n",
                         (uint64_t)take_rowptr_bytes, (uint64_t)want_rowptr_bytes, (long long)file_size);
        return false;
    }
    std::vector<uint32_t> rowptr(rowptr_elems, 0);
    fin.read(reinterpret_cast<char*>(rowptr.data()), rowptr_elems * sizeof(uint32_t));
    if (!fin.good()) {
        output_->verbose(CALL_INFO, 0, 0, "⚠️ 读取rowptr失败: %s\n", path.c_str());
        return false;
    }
    const uint32_t total_blocks_rowptr = rowptr.back();
    const uint64_t max_blocks_colidx = (step_activation_bcsr_idx_bytes_ > 0)
        ? (avail_colidx_bytes / (uint64_t)step_activation_bcsr_idx_bytes_)
        : 0ULL;
    const uint64_t max_blocks_data = (bytes_per_block_data > 0) ? (avail_blockdata_bytes / bytes_per_block_data) : 0ULL;
    const uint64_t max_blocks_ids  = (bytes_per_block_ids  > 0) ? (avail_blockids_bytes  / bytes_per_block_ids ) : 0ULL;
    const uint64_t max_blocks_by_file = std::min(std::min(max_blocks_data, max_blocks_ids), max_blocks_colidx);
    const uint32_t total_blocks = (uint32_t) std::min<uint64_t>(total_blocks_rowptr, max_blocks_by_file);
    if (total_blocks == 0) {
        output_->verbose(CALL_INFO, 0, 0,
            "⚠️ total_blocks=0 (rowptr=%u, by_file=%" PRIu64 ") path=%s\n",
            total_blocks_rowptr, max_blocks_by_file, path.c_str());
        return false;
    }
    // 计算或校验 offset，确保与文件自洽；如配置非法则按对齐重算
    if (!checkBcsrOffsets_((uint64_t)file_size, n_block_rows, total_blocks,
                           bytes_per_block_data,
                           rowptr_off, colidx_off, blockdata_off, blockids_off)) {
        if (!step_activation_route_warned_) {
            output_->verbose(CALL_INFO, 0, 0,
                "⚠️ BCSR offsets/size mismatch: node=%d path=%s fsize=%lld blocks(rowptr)=%u br=%u bc=%u idxB=%u valB=%u (recomputed: rowptr=%llu colidx=%llu blockdata=%llu blockids=%llu)\n",
                node_id_, path.c_str(), (long long)file_size, total_blocks_rowptr, br, bc,
                step_activation_bcsr_idx_bytes_, step_activation_bcsr_val_bytes_,
                (unsigned long long)rowptr_off, (unsigned long long)colidx_off,
                (unsigned long long)blockdata_off, (unsigned long long)blockids_off);
        }
        return false;
    }

    std::vector<uint32_t> block_cols(total_blocks, 0);
    // 读取 colidx（按 total_blocks 截断）
    fin.seekg(step_activation_bcsr_colidx_offset_, std::ios::beg);
    if (step_activation_bcsr_idx_bytes_ == 2) {
        std::vector<uint16_t> tmp(total_blocks, 0);
        fin.read(reinterpret_cast<char*>(tmp.data()), tmp.size() * sizeof(uint16_t));
        if (!fin.good()) {
            output_->verbose(CALL_INFO, 0, 0, "⚠️ 读取colidx(2B)失败: %s blocks=%u\n", path.c_str(), total_blocks);
            return false;
        }
        for (uint32_t i = 0; i < total_blocks; ++i) block_cols[i] = tmp[i];
    } else {
        fin.read(reinterpret_cast<char*>(block_cols.data()), block_cols.size() * sizeof(uint32_t));
        if (!fin.good()) {
            output_->verbose(CALL_INFO, 0, 0, "⚠️ 读取colidx(4B)失败: %s blocks=%u\n", path.c_str(), total_blocks);
            return false;
        }
    }

    std::ifstream fdata(path, std::ios::binary);
    std::ifstream fids(path, std::ios::binary);
    fdata.seekg(step_activation_bcsr_blockdata_offset_, std::ios::beg);
    fids.seekg(step_activation_bcsr_blockids_offset_, std::ios::beg);
    std::vector<float> blockdata(floats_per_block, 0.0f);
    std::vector<uint32_t> blockids(floats_per_block, 0u);

    uint32_t block_index = 0;
    uint64_t skipped_blocks = 0;
    auto flush_skips = [&](uint64_t n) -> bool {
        if (n == 0) return true;
        const uint64_t data_skip = n * bytes_per_block_data;
        const uint64_t ids_skip  = n * bytes_per_block_ids;
        fdata.seekg(static_cast<std::streamoff>(data_skip), std::ios::cur);
        fids.seekg(static_cast<std::streamoff>(ids_skip), std::ios::cur);
        if (!fdata.good() || !fids.good()) {
            output_->verbose(CALL_INFO, 0, 0,
                "⚠️ BCSR seek skip failed: %s skip_blocks=%" PRIu64 "\n",
                path.c_str(), (uint64_t)n);
            return false;
        }
        return true;
    };
    for (uint32_t block_row = 0; block_row < n_block_rows; ++block_row) {
        if ((size_t)block_row + 1 >= rowptr.size()) break;
        uint32_t begin = rowptr[block_row];
        uint32_t end = rowptr[block_row + 1];
        if (begin >= total_blocks) break;
        if (end > total_blocks) end = total_blocks;
        for (uint32_t idx = begin; idx < end; ++idx, ++block_index) {
            if (block_index >= total_blocks) break;
            uint32_t block_col = block_cols[idx];
            // 关键优化：本节点只需要本地 pre 的出边；跳过不属于本节点 pre 区间的 block，
            // 避免对所有 block 读取 blockdata/blockids 再在内层过滤（可节省约 total_nodes 倍开销）。
            const uint64_t pre_block_begin = static_cast<uint64_t>(block_col) * static_cast<uint64_t>(bc);
            const uint64_t pre_block_end = pre_block_begin + static_cast<uint64_t>(bc);
            const bool overlap_local = (neurons_per_pe > 0) &&
                !(pre_block_end <= (uint64_t)local_pre_begin || pre_block_begin >= (uint64_t)local_pre_end);
            if (!overlap_local) {
                ++skipped_blocks;
                continue;
            }
            if (!flush_skips(skipped_blocks)) return false;
            skipped_blocks = 0;

            fdata.read(reinterpret_cast<char*>(blockdata.data()), blockdata.size() * sizeof(float));
            fids.read(reinterpret_cast<char*>(blockids.data()), blockids.size() * sizeof(uint32_t));
            if (!fdata.good() || !fids.good()) {
                output_->verbose(CALL_INFO, 0, 0,
                    "⚠️ 读取block数据失败: %s (block_index=%u/%u, fsize=%lld)\n",
                    path.c_str(), block_index, total_blocks, (long long)file_size);
                return false;
            }

            // 守卫 block_col * bc + cc 不越界
            if (block_col >= std::numeric_limits<uint32_t>::max() / bc) {
                continue;
            }
            const uint32_t pre_base = static_cast<uint32_t>(pre_block_begin);
            for (uint32_t rr = 0; rr < br; ++rr) {
                uint32_t post_local = block_row * br + rr;
                if (post_local >= rows_per_core) continue;
                const uint64_t post_global_64 =
                    static_cast<uint64_t>(pe_id) * static_cast<uint64_t>(neurons_per_pe) +
                    static_cast<uint64_t>(core_index) * static_cast<uint64_t>(rows_per_core) +
                    static_cast<uint64_t>(post_local);
                if (max_global > 0u && post_global_64 >= static_cast<uint64_t>(max_global)) {
                    continue;
                }
                const uint32_t post_global = static_cast<uint32_t>(post_global_64);
                for (uint32_t cc = 0; cc < bc; ++cc) {
                    size_t off = static_cast<size_t>(rr) * bc + cc;
                    // blockids 作为稀疏mask：0xFFFFFFFF 表示该位置无边；非mask值不代表 post_global
                    if (blockids[off] == 0xFFFFFFFFu) continue;
                    // 权重阈值过滤：避免把块内填充0当作有效边
                    float weight = blockdata[off];
                    if (std::fabs(weight) <= step_activation_bcsr_weight_epsilon_) continue;
                    uint32_t pre_global = pre_base + cc;
                    // 仅收集本节点 pre 的出边（跨 PE 的 pre 由其所属节点负责）
                    if (pre_global < local_pre_begin || pre_global >= local_pre_end) continue;
                    if (pre_global >= step_activation_routes_.size()) continue;
                    step_activation_routes_[pre_global].push_back(post_global);
                }
            }
        }
    }
    // 若末尾仍有跳过块，无需继续读取，但需保持逻辑自洽（仅用于调试/健壮性）
    // (不强制 seek 到文件尾，避免额外开销)
    // 诊断：一次性打印装载概览（仅在显式启用日志时）
    if (output_ && step_activation_log_enable_ && output_->getVerboseLevel() >= 1) {
        uint64_t edges = 0;
        for (auto &v : step_activation_routes_) edges += (uint64_t)v.size();
        output_->verbose(CALL_INFO, 1, 0,
            "[step-activation] BCSR reachability loaded: pe=%u core=%u rows=%u br=%u bc=%u total_blocks(rowptr)=%u used=%u edges=%llu\n",
            pe_id, core_index, rows_per_core, br, bc, total_blocks_rowptr, total_blocks,
            static_cast<unsigned long long>(edges));
    }
    return true;
}

bool MultiCorePE::loadBcsrReachability_() {
    if (output_ && step_activation_log_enable_) {
        output_->verbose(CALL_INFO, 0, 0,
            "[step-activation] node=%d loadBcsrReachability enable=%d use_bcsr=%d template=%s build_local_only=%d rows_per_core=%u br=%u bc=%u\n",
            node_id_, (int)step_activation_enable_, (int)step_activation_use_bcsr_routes_,
            step_activation_bcsr_template_.c_str(), (int)step_activation_build_local_only_,
            step_activation_bcsr_rows_per_core_, step_activation_bcsr_br_, step_activation_bcsr_bc_);
    }
    if (step_activation_bcsr_template_.empty()) {
        if (output_ && step_activation_log_enable_) {
            output_->verbose(CALL_INFO, 0, 0, "⚠️ 未提供 step_activation_bcsr_template，无法加载BCSR索引\n");
        }
        return false;
        }
        const uint64_t total_pre = static_cast<uint64_t>(global_neuron_base_) + static_cast<uint64_t>(total_neurons_);
        step_activation_routes_.assign(static_cast<size_t>(total_pre), {});
        bool success = true;
        // 遍历所有PE的权重文件，构建“本PE本地pre”的全局出边（可跨PE）
        int pe_begin = 0, pe_end = (total_nodes_ > 0 ? total_nodes_ : 1);
        if (step_activation_build_local_only_) {
            pe_begin = node_id_;
            pe_end = node_id_ + 1;
        }
        for (int pe = pe_begin; pe < pe_end; ++pe) {
            for (int core = 0; core < num_cores_; ++core) {
                std::string path = formatBcsrPath_(pe, core);
                if (path.empty()) { success = false; break; }
                if (!buildRoutesFromBcsrFile_(path, static_cast<uint32_t>(pe), static_cast<uint32_t>(core))) { success = false; break; }
            }
            if (!success) break;
        }
    if (success) {
        size_t with_routes = 0;
        for (const auto& vec : step_activation_routes_) {
            if (!vec.empty()) ++with_routes;
        }
        if (step_activation_log_enable_ && !step_activation_route_ack_logged_) {
            output_->verbose(CALL_INFO, 1, 0,
                "[step-activation] BCSR reachability loaded: pre_with_routes=%zu total_pre=%zu\n",
                with_routes, step_activation_routes_.size());
            // 每个节点仅打印一次远端/本地比例摘要，确认跨PE路由是否存在
            computeRouteRatios_();
            step_activation_route_ack_logged_ = true;
        }
        if (output_ && step_activation_log_enable_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[step-activation] node=%d loadBcsrReachability success route_vectors=%zu\n",
                node_id_, step_activation_routes_.size());
        }
        // 单 PE 场景：预先收集所有“拥有至少一条出边”的 pre_global，供注入阶段直接采样，
        // 避免在无出边的神经元上浪费激活尝试。
        step_activation_pre_with_routes_.clear();
        if (total_nodes_ == 1) {
            step_activation_pre_with_routes_.reserve(with_routes);
            for (uint32_t pre = 0; pre < step_activation_routes_.size(); ++pre) {
                if (!step_activation_routes_[pre].empty()) {
                    step_activation_pre_with_routes_.push_back(pre);
                }
            }
            if (output_ && step_activation_log_enable_) {
                output_->verbose(CALL_INFO, 1, 0,
                    "[step-activation] node=%d single-PE pre_with_routes_list size=%zu\n",
                    node_id_, step_activation_pre_with_routes_.size());
            }
        }
    } else {
        step_activation_pre_with_routes_.clear();
        if (step_activation_log_enable_ && !step_activation_route_warned_) {
            output_->verbose(CALL_INFO, 0, 0,
                "⚠️ step_activation BCSR route build failed, routes cleared; using fallback sampling\n");
            step_activation_route_warned_ = true;
        }
        step_activation_routes_.clear();
        if (output_ && step_activation_log_enable_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[step-activation] node=%d loadBcsrReachability FAILED\n",
                node_id_);
        }
    }
    return success;
}

void MultiCorePE::computeRouteRatios_() const {
    uint64_t local_edges = 0, remote_edges = 0, total_edges = 0;
    const uint32_t neurons_per_pe = static_cast<uint32_t>(neurons_per_core_) * static_cast<uint32_t>(num_cores_);
    if (neurons_per_pe > 0) {
        for (size_t pre = 0; pre < step_activation_routes_.size(); ++pre) {
            const auto& vec = step_activation_routes_[pre];
            total_edges += static_cast<uint64_t>(vec.size());
            for (auto post_global : vec) {
                uint32_t pe_of_post = static_cast<uint32_t>(post_global / neurons_per_pe);
                if (pe_of_post == static_cast<uint32_t>(node_id_)) ++local_edges;
                else ++remote_edges;
            }
        }
    }
    double local_ratio = (total_edges ? (double)local_edges / (double)total_edges : 0.0);
    double remote_ratio = (total_edges ? (double)remote_edges / (double)total_edges : 0.0);
    if (output_ && step_activation_log_enable_) {
        output_->verbose(CALL_INFO, 1, 0,
            "[step-activation] route_ratio: node=%d local_edges=%" PRIu64 " remote_edges=%" PRIu64 " total=%" PRIu64 " local_ratio=%.4f remote_ratio=%.4f\n",
            node_id_, local_edges, remote_edges, total_edges, local_ratio, remote_ratio);
    }
}

#endif // Phase3-B legacy Step activation helpers

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
    network_link_ = configureLink("network", 
        new Event::Handler2<MultiCorePE,&MultiCorePE::handleNetworkLinkEvent>(this));
    
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
