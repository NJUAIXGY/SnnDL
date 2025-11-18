// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// MultiCorePE.cc: 真正的多核脉冲神经网络处理单元实现文件
//

#include <sst/core/sst_config.h>
#include "MultiCorePE.h"
#include "SnnNetworkAdapter.h"
#include "OptimizedInternalRing.h"
#include "SnnNIC.h"
#include "GatingDecisionEvent.h"
#include "SnnPESubComponent.h"

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

namespace {
inline bool sentinelsEnabled() {
    static int cached = -1;
    if (cached == -1) {
        const char* env = std::getenv("SNNDL_SENTINEL_ENABLE");
        cached = (env && std::atoi(env) != 0) ? 1 : 0;
    }
    return cached == 1;
}
} // namespace

MultiCorePE::MultiCorePE(ComponentId_t id, Params& params) : Component(id) {
    // 初始化输出对象
    int verbose_level = params.find<int>("verbose", 0);
    output_ = new Output("MultiCorePE[@p:@l]: ", verbose_level, 0, Output::STDOUT);
    
    
    // 读取基础配置参数
    num_cores_ = params.find<int>("num_cores", 4);
    neurons_per_core_ = params.find<int>("neurons_per_core", 64);
    total_neurons_ = num_cores_ * neurons_per_core_;
    neurons_per_pe_cfg_ = params.find<uint32_t>("neurons_per_pe", 0);
    if (neurons_per_pe_cfg_ == 0) {
        neurons_per_pe_cfg_ = static_cast<uint32_t>(num_cores_) * static_cast<uint32_t>(neurons_per_core_);
    }
    node_id_ = params.find<int>("node_id", 0);
    total_nodes_ = params.find<int>("total_nodes", 1);
    global_neuron_base_ = params.find<uint64_t>("global_neuron_base", 0);
    sim_stop_ns_ = params.find<uint64_t>("sim_stop_ns", 0);
    verbose_ = verbose_level;
    weights_file_ = params.find<std::string>("weights_file", "");
    enable_numa_ = params.find<bool>("enable_numa", true);
    
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

    step_activation_enable_ = params.find<bool>("step_activation_enable", false);
    step_activation_fraction_ = params.find<double>("step_activation_fraction", 0.0);
    step_activation_fanout_ = params.find<uint32_t>("step_activation_fanout", 0);
    step_activation_seed_ = params.find<uint64_t>("step_activation_seed", 0xdecafbadULL);
    step_reset_mem_each_step_ = params.find<bool>("step_reset_mem_each_step", false);
    step_activation_event_weight_ = params.find<double>("step_activation_event_weight", 0.0);
    step_activation_use_bcsr_routes_ = params.find<bool>("step_activation_use_bcsr_routes", false);
    step_activation_bcsr_template_ = params.find<std::string>("step_activation_bcsr_template", "");
    step_activation_bcsr_rows_per_core_ = params.find<uint32_t>("step_activation_bcsr_rows_per_core", neurons_per_core_);
    step_activation_bcsr_br_ = params.find<uint32_t>("step_activation_bcsr_br", 16);
    step_activation_bcsr_bc_ = params.find<uint32_t>("step_activation_bcsr_bc", 16);
    step_activation_bcsr_idx_bytes_ = params.find<uint32_t>("step_activation_bcsr_idx_bytes", 2);
    step_activation_bcsr_val_bytes_ = params.find<uint32_t>("step_activation_bcsr_val_bytes", 4);
    step_activation_bcsr_rowptr_offset_ = params.find<uint64_t>("step_activation_bcsr_rowptr_offset", 0);
    step_activation_bcsr_colidx_offset_ = params.find<uint64_t>("step_activation_bcsr_colidx_offset", 0);
    step_activation_bcsr_blockdata_offset_ = params.find<uint64_t>("step_activation_bcsr_blockdata_offset", 0);
    step_activation_bcsr_blockids_offset_ = params.find<uint64_t>("step_activation_bcsr_blockids_offset", 0);
    step_activation_bcsr_weight_epsilon_ = params.find<double>("step_activation_bcsr_weight_epsilon", 0.0);
    // 仅本节点构建路由，降低IO与规避跨PE元数据不一致风险（可由脚本覆盖）
    step_activation_build_local_only_ = params.find<bool>("step_activation_build_local_only", true);
    
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

    if (step_activation_enable_ && step_activation_use_bcsr_routes_) {
    bool sentinel_on = sentinelsEnabled();
    if (sentinel_on && output_) output_->output("[[sentinel-pe-ctor]] node=%d building step BCSR reachability\n", node_id_);
        if (!loadBcsrReachability_()) {
            output_->verbose(CALL_INFO, 1, 0,
                "⚠️ step_activation_use_bcsr_routes=1 但 BCSR 索引加载失败，回退到均匀采样\n");
            step_activation_use_bcsr_routes_ = false;
        } else {
    if (sentinel_on && output_) output_->output("[[sentinel-pe-ctor]] node=%d step BCSR reachability built\n", node_id_);
        }
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
    internal_ring_ = nullptr;
    controller_ = nullptr;
    
    // 初始化端口指针为空
    external_spike_input_link_ = nullptr;
    external_spike_output_link_ = nullptr;
    mem_link_ = nullptr;
    
    
    // 初始化统计收集（必须在构造函数中）
    initializeStatistics();
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
    delete internal_ring_;
    delete controller_;
    // 避免析构次序竞态：不手动delete日志对象
    output_ = nullptr;
    
    // 清理外部脉冲队列
    while (!external_spike_queue_.empty()) {
        delete external_spike_queue_.front();
        external_spike_queue_.pop();
    }
    
    // 清理挂起的内存请求
    for (auto& pair : pending_memory_requests_) {
        delete pair.second;
    }
    pending_memory_requests_.clear();
}

void MultiCorePE::init(unsigned int phase) {
    if (sentinelsEnabled() && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=%u enter\n", node_id_, phase); }
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
        if (sentinelsEnabled() && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=0 clock-registered\n", node_id_); }
        
        
        // 初始化统计收集
        
        // 初始化端口连接
        external_spike_input_link_ = configureLink("external_spike_input", 
            new Event::Handler2<MultiCorePE,&MultiCorePE::handleExternalSpikeEvent>(this));
        external_spike_output_link_ = configureLink("external_spike_output");
        mem_link_ = configureLink("mem_link");
        if (sentinelsEnabled() && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=0 links-configured\n", node_id_); }
        
        
        // 初始化方向链路（用于端口代理机制）
        initializeDirectionLinks();
        if (sentinelsEnabled() && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=0 dir-links\n", node_id_); }
        
        // 初始化处理单元
        initializeProcessingUnits();
        if (sentinelsEnabled() && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=0 units-initialized\n", node_id_); }
        
        // 初始化内部互连
        initializeInternalRing();
        if (sentinelsEnabled() && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=0 ring-initialized\n", node_id_); }
        
        // 初始化多核控制器
        controller_ = new MultiCoreController(this, output_);
        if (sentinelsEnabled() && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=0 controller-created\n", node_id_); }
        

        // 将当前phase转发给所有子核心
        for (auto* core : cores_) {
            if (core) core->init(phase);
        }
        if (sentinelsEnabled() && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=0 cores-init-done\n", node_id_); }
        
        // 关键修复：转发init到网络接口
        if (external_nic_) {
            external_nic_->init(phase);
        }
        if (sentinelsEnabled() && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=0 nic-init-done\n", node_id_); }
        // 标记Step注入就绪（保证NIC已完成init）
        step_injection_ready_ = true;
    }
    else if (phase == 1) {
        if (sentinelsEnabled() && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=1 enter\n", node_id_); }
        // 阶段1：加载权重和配置子组件
        loadAndDistributeWeights();
        if (sentinelsEnabled() && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=1 weights-loaded\n", node_id_); }

        // 将当前phase转发给所有子核心
        for (auto* core : cores_) {
            if (core) core->init(phase);
        }
        if (sentinelsEnabled() && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=1 cores-init-done\n", node_id_); }
        
        // 转发init到网络接口
        if (external_nic_) {
            external_nic_->init(phase);
        }
        if (sentinelsEnabled() && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=1 nic-init-done\n", node_id_); }
    }
    else {
        if (sentinelsEnabled() && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=%u forward-only\n", node_id_, phase); }
        // 其余phase同样转发
        for (auto* core : cores_) {
            if (core) core->init(phase);
        }
        
        // 转发init到网络接口
        if (external_nic_) {
            external_nic_->init(phase);
        }
        if (sentinelsEnabled() && output_) { output_->output("[[sentinel-pe-init]] node=%d phase=%u done\n", node_id_, phase); }
    }
}

void MultiCorePE::setup() {
    if (sentinelsEnabled() && output_) { output_->output("[[sentinel-pe-setup]] node=%d enter\n", node_id_); }
    
    // 验证所有组件初始化完成
    if (cores_.size() != static_cast<size_t>(num_cores_)) {
        output_->fatal(CALL_INFO, -1, "❌ 错误: 核心数量不匹配，期望%d，实际%zu\n", 
                      num_cores_, cores_.size());
    }
    
    // 检查内部互连（新的优化版本或旧版本）
    // 单核情况下不需要内部互连
    if (num_cores_ > 1 && !optimized_ring_ && !internal_ring_) {
        output_->fatal(CALL_INFO, -1, "❌ 错误: 多核配置但内部互连未初始化\n");
    }
    // 调用子核心的setup
    for (auto* core : cores_) {
        if (core) core->setup();
    }
    if (sentinelsEnabled() && output_) { output_->output("[[sentinel-pe-setup]] node=%d cores-setup\n", node_id_); }
    
    // 调用网络接口的setup
    if (external_nic_) {
        external_nic_->setup();
    }
    if (sentinelsEnabled() && output_) { output_->output("[[sentinel-pe-setup]] node=%d nic-setup\n", node_id_); }
    
    if (!controller_) {
        output_->fatal(CALL_INFO, -1, "❌ 错误: 多核控制器未初始化\n");
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
        printf("NODE%d: 脉冲=%lu, 激发=%lu\n", node_id_, agg_spikes, agg_fired);
        fflush(stdout);
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
    if (primary_keepalive_) {
        primaryComponentOKToEndSim();
    }
}

bool MultiCorePE::clockTick(Cycle_t current_cycle) {
    current_cycle_ = current_cycle;
    // 当启用组件主控停止时，到达阈值立刻OKToEndSim并停止本组件时钟
    if (sim_stop_ns_ > 0 && current_cycle_ >= sim_stop_ns_) {
        if (primary_keepalive_ || sim_stop_ns_ > 0) {
            primaryComponentOKToEndSim();
        }
        return false;
    }
    
    // 详细调试信息（仅在高详细度时输出）
    static bool first_tick_logged = false;
    if (!first_tick_logged) {
        const char* sent_env = std::getenv("SNNDL_SENTINEL_ENABLE");
        if (sent_env && std::atoi(sent_env) != 0) {
            printf("[[sentinel-pe-clock]] node=%d first_tick cyc=%" PRIu64 "\n",
                   node_id_, (uint64_t)current_cycle_);
            fflush(stdout);
        }
    PE_LOG(2, "[diag-PE] clockTick start node=%d\n", node_id_);
        first_tick_logged = true;
    }
    
    // 0a. 若存在延迟的Step注入且已就绪，优先补发
    if (pending_step_inject_ && step_injection_ready_) {
        injectStepActivations(pending_step_seq_, current_cycle_);
        last_step_injection_seq_ = pending_step_seq_;
        pending_step_inject_ = false;
    }

    // 0b. 测试注入：在首个有效周期从 core0 向 core1 注入一个跨核脉冲（仅当启用测试流量时）
    if (enable_test_traffic_ && !test_injected_ && num_cores_ > 1 && current_cycle_ == 5000) {
        // 构造一个从全局神经元0 -> 全局神经元(neurons_per_core_) 的脉冲
        SpikeEvent* test_spike = new SpikeEvent(0, neurons_per_core_, 0, 0.5f, current_cycle_);
        int src_core = determineTargetUnit(test_spike->getSourceNeuron());
        int dst_core = determineTargetUnit(test_spike->getDestinationNeuron());
        if (src_core >=0 && dst_core >=0 && src_core != dst_core) {
            routeInternalSpike(src_core, dst_core, test_spike);
            PE_LOG(1, "🧪 注入跨核脉冲: 核心%d->核心%d\n", src_core, dst_core);
            test_injected_ = true;
        } else {
            delete test_spike;
            test_injected_ = true;
        }
    }
    
    // 1. 处理外部脉冲队列
    // Debug output disabled to prevent excessive logging
    while (!external_spike_queue_.empty()) {
        SpikeEvent* spike = external_spike_queue_.front();
        external_spike_queue_.pop();
        
        // Debug output removed to reduce log noise
        
        int target_unit = determineTargetUnit(spike->getDestinationNeuron());
        if (target_unit >= 0 && target_unit < num_cores_) {
            // 目标在本节点，直接投递给对应的处理单元
            // Debug output removed
            fflush(stdout);
            deliverSpikeToCore(target_unit, spike);
        } else {
            // 目标不在本节点，需要转发到其他节点
            if (external_nic_) {
                PE_LOG(3, "🔄 中继转发脉冲: 神经元%d -> 目标节点%d\n", 
                               spike->getDestinationNeuron(), spike->getDestinationNode());
                external_nic_->sendSpike(spike);
                // 不要删除spike，已经转交给网络适配器
            } else {
                PE_LOG(2, "⚠️ 无网络接口，丢弃跨节点脉冲: 神经元%d\n", 
                               spike->getDestinationNeuron());
                delete spike;
            }
        }
    }
    
    // 2. SubComponent时钟由SST自动管理，无需手动调用tick
    // 若子组件未被SST调度（某些环境组合下可能发生），则回退为手动驱动一拍，确保窗口推进与队列消费
    if (manual_core_drive_enable_) {
        for (int i = 0; i < num_cores_; i++) {
            if (cores_[i] != nullptr) {
                if (auto* sc = dynamic_cast<SnnPESubComponent*>(cores_[i])) {
                    sc->driveOneCycle();
                    if (manual_gas_gather_cycles_ > 0 && (current_cycle_ % manual_gas_gather_cycles_) == 0) {
                        PE_LOG(2, "[diag-PE] forceEndGather: core=%d cyc=%" PRIu64 " period=%" PRIu64 "\n",
                               i, (uint64_t)current_cycle_, (uint64_t)manual_gas_gather_cycles_);
                        sc->forceEndGather();
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
            uint64_t old_spikes = unit_states_[i].spikes_processed;
            uint64_t new_spikes = (it_sp != core_stats.end()) ? it_sp->second : 0;
            unit_states_[i].spikes_processed = new_spikes;
            unit_states_[i].neurons_fired = (it_nf != core_stats.end()) ? it_nf->second : 0;
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
    
    // 3. 内部互连时钟滴答
    if (optimized_ring_) {
        optimized_ring_->tick(current_cycle);
        
        // 处理跨核脉冲路由（使用新的优化环形网络）
        handleOptimizedCrossCoreRouting();
    } else if (internal_ring_) {
        internal_ring_->tick();
        
        // 处理跨核脉冲路由（旧版本兼容）
        handleCrossCoreRouting();
    }
    
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
    PE_LOG(1, "📥[MultiCorePE] handleExternalSpikeEvent被调用，节点ID=%d\n", node_id_);
    fflush(stdout);

    SpikeEvent* spike = dynamic_cast<SpikeEvent*>(ev);
    if (!spike) {
        PE_LOG(1, "⚠️ 接收到非SpikeEvent事件\n");
        delete ev;
        return;
    }
    
    // Debug output disabled to prevent excessive logging
    // printf("DEBUG: SpikeEvent转换成功，神经元%d -> 神经元%d\n", 
    //        spike->getSourceNeuron(), spike->getDestinationNeuron());
    fflush(stdout);
    
    // 检查跳数限制，防止无限循环
    if (spike->isExpired()) {
        PE_LOG(2, "⚠️ 脉冲达到最大跳数限制，丢弃: 源神经元%d -> 目标神经元%d\n",
                        spike->getSourceNeuron(), spike->getDestinationNeuron());
        delete spike;
        return;
    }
    
    spike->incrementHopCount();
    
    PE_LOG(3, "📨 接收外部脉冲: 源神经元%d -> 目标神经元%d, 权重%.3f, 跳数%d\n",
                    spike->getSourceNeuron(), spike->getDestinationNeuron(), spike->getWeight(), spike->getHopCount());
    
    stat_external_spikes_received_->addData(1);
    
    // 检查是否为本地节点（基于目标节点ID而不是神经元ID）
    uint32_t dest_node = spike->getDestinationNode();
    bool is_local = (dest_node == static_cast<uint32_t>(node_id_));

    // 调试输出：显示节点判断结果
    PE_LOG(2, "🔍 脉冲路由判断: 目标神经元=%d, 目标节点=%u, 本地节点=%d, 本地判断=%s\n",
                     spike->getDestinationNeuron(), dest_node, node_id_, is_local ? "本地" : "跨节点");
    // Debug output removed
    fflush(stdout);
    
    if (is_local) {
        // 本地脉冲，加入队列处理
        // Debug output removed
        fflush(stdout);
        external_spike_queue_.push(spike);
        PE_LOG(4, "✅ 本地脉冲已加入队列\n");
    } else {
        // 跨核（同一MultiCorePE内不同处理单元）或外部（非本PE）
        int target_unit = determineTargetUnit(spike->getDestinationNeuron());
        if (target_unit >= 0 && target_unit < num_cores_) {
            // 目标在本MultiCorePE内的其他处理单元，直接分发给目标处理单元
            SpikeEvent* cross_core_spike = new SpikeEvent(
                spike->getSourceNeuron(),
                spike->getDestinationNeuron(),
                spike->getDestinationNode(),
                spike->getWeight(),
                spike->getSpikeTime()
            );
            cross_core_spike->hop_count = spike->getHopCount();  // 传递跳数
            deliverSpikeToCore(target_unit, cross_core_spike);
            PE_LOG(4, "🔄 外部脉冲直接分发到核心%d\n", target_unit);
        } else {
            // 目标不在本MultiCorePE，视为外部转发（若配置了外部输出端口）
            PE_LOG(2, "🔍 准备转发跨节点脉冲: 神经元%d, 目标节点%d, external_nic_=%p, external_spike_output_link_=%p\n",
                           spike->getDestinationNeuron(), spike->getDestinationNode(),
                           (void*)external_nic_, (void*)external_spike_output_link_);
            if (external_nic_) {
                PE_LOG(2, "🌐 尝试通过SnnNIC发送跨节点脉冲: 神经元%d -> 目标节点%d\n",
                               spike->getDestinationNeuron(), spike->getDestinationNode());
                sendExternalSpike(spike);
                PE_LOG(2, "📤 外部转发脉冲到其他PE: 目标神经元%d, 跳数%d, 目标节点%d\n",
                                spike->getDestinationNeuron(), spike->getHopCount(), spike->getDestinationNode());
                return; // sendExternalSpike会接管事件
            } else {
                PE_LOG(1, "⚠️ 无法确定目标处理单元且无外部输出，丢弃: 神经元%d\n",
                                 spike->getDestinationNeuron());
            }
        }
        delete spike;
    }
}

void MultiCorePE::handleExternalSpike(SpikeEvent* spike) {
    if (!spike) return;
    
    PE_LOG(3, "🔄 处理外部脉冲: 目标神经元%d\n", spike->getDestinationNeuron());
    
    // 将脉冲加入外部队列，由时钟处理器处理
    external_spike_queue_.push(spike);
    stat_external_spikes_received_->addData(1);
}

void MultiCorePE::sendExternalSpike(SpikeEvent* spike) {
    if (!spike) return;

    // 自环防护：如果目标节点就是本节点，直接丢弃，避免外部回送循环
    int target_node = static_cast<int>(spike->getDestinationNode());
    if (target_node == node_id_) {
        PE_LOG(2, "⚠️ 试图向自身节点发送外部脉冲，丢弃: 源=%d 目标=%d 节点=%d\n",
                         spike->getSourceNeuron(), spike->getDestinationNeuron(), target_node);
        delete spike;
        return;
    }

    PE_LOG(3, "📤 发送外部脉冲: 源神经元%d -> 目标神经元%d, 跳数%d\n",
                     spike->getSourceNeuron(), spike->getDestinationNeuron(), spike->getHopCount());

    // 优先使用网络适配器，如果未配置则回退到传统链接
    static uint64_t s_nic_send_log_count = 0;
    if (external_nic_) {
        // 使用网络适配器发送脉冲（这将触发路由计算和统计收集）
        external_nic_->sendSpike(spike);
        PE_LOG(3, "🌐 通过网络适配器发送脉冲\n");
        // debug nic-send removed for production
    } else if (external_spike_output_link_) {
        // 回退到传统链接模式
        external_spike_output_link_->send(spike);
        PE_LOG(3, "🔗 通过传统链接发送脉冲\n");
    } else {
        PE_LOG(2, "⚠️ 没有可用的外部发送方式，丢弃脉冲\n");
        delete spike;
        return;
    }
    
    stat_external_spikes_sent_->addData(1);
}

void MultiCorePE::routeInternalSpike(int src_core, int dst_core, SpikeEvent* spike) {
    if (!spike) return;
    
    if (src_core < 0 || src_core >= num_cores_ || dst_core < 0 || dst_core >= num_cores_) {
        PE_LOG(1, "⚠️ 无效的核心ID: src=%d, dst=%d\n", src_core, dst_core);
        delete spike;
        return;
    }
    
    PE_LOG(4, "🔄 路由内部脉冲: 核心%d -> 核心%d, 神经元%d\n",
                    src_core, dst_core, spike->getDestinationNeuron());
    
    // 单核情况或同一核心内，直接递送
    if (num_cores_ <= 1 || src_core == dst_core) {
        deliverSpikeToCore(dst_core, spike);
        return;
    }
    
    // 创建内部消息
    RingMessage msg;
    msg.type = RingMessageType::SPIKE_MESSAGE;
    msg.src_unit = src_core;
    msg.dst_unit = dst_core;
    msg.timestamp = current_cycle_;
    msg.payload.spike_data = spike;
    
    bool sent_successfully = false;
    
    // 优先使用优化的环形网络
    if (optimized_ring_) {
        sent_successfully = optimized_ring_->sendMessage(src_core, dst_core, msg, 1); // 优先级1
        if (sent_successfully) {
            inter_core_messages_count_++;
            if (stat_inter_core_messages_) stat_inter_core_messages_->addData(1);
        }
    } 
    // 回退到旧的环形网络
    else if (internal_ring_) {
        sent_successfully = internal_ring_->sendMessage(msg);
        if (sent_successfully) {
            inter_core_messages_count_++;
            if (stat_inter_core_messages_) stat_inter_core_messages_->addData(1);
        }
    }
    
    if (!sent_successfully) {
        PE_LOG(2, "⚠️ 内部环形网络发送失败: 核心%d -> 核心%d\n", src_core, dst_core);
        delete spike;
    }
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

void MultiCorePE::notifyStageEvent(uint32_t seq, const std::string& event, uint64_t ts_ns, uint64_t spikes_emitted) {
    auto &m = stage_marks_[seq];
    if (event == "BeginGather") {
        if (m.bg == 0 || ts_ns < m.bg) m.bg = ts_ns;
    } else if (event == "BeginApply") {
        if (m.ga == 0 || ts_ns < m.ga) m.ga = ts_ns;
    } else if (event == "EndApply") {
        if (m.ea == 0 || ts_ns > m.ea) m.ea = ts_ns;
    } else if (event == "BeginScatter") {
        if (m.bs == 0 || ts_ns < m.bs) m.bs = ts_ns;
    } else if (event == "EndScatter") {
        if (m.es == 0 || ts_ns > m.es) m.es = ts_ns;
        if (spikes_emitted > 0) {
            // 同步记录到窗口发放聚合，便于 finish() 写 pe_window_spikes_db.csv
            window_spikes_pe_[seq] += spikes_emitted;
        }
    }

    if (step_activation_enable_ && event == "BeginGather") {
        if (last_step_injection_seq_ != seq) {
            if (step_injection_ready_) {
                injectStepActivations(seq, ts_ns);
                last_step_injection_seq_ = seq;
            } else {
                pending_step_inject_ = true;
                pending_step_seq_ = seq;
                pending_step_ts_ns_ = ts_ns;
            }
        }
    }
    if (step_reset_mem_each_step_ && event == "EndScatter") {
        if (last_step_reset_seq_ != seq) {
            resetAllCoreMembranes();
            last_step_reset_seq_ = seq;
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
        
        // 记录槽位可用性
        bool slot_api_ok = isSubComponentLoadableUsingAPI<SnnCoreAPI>("core" + std::to_string(i));

        // 优先尝试通过用户在Python中配置的槽位加载
        SnnCoreAPI* core = loadUserSubComponent<SnnCoreAPI>(
            "core" + std::to_string(i), ComponentInfo::SHARE_NONE);
        if (core) {
        }

        if (!core) {
            // 如果用户未配置，则回退到匿名加载默认实现
            core = loadAnonymousSubComponent<SnnCoreAPI>(
                "SnnDL.SnnPESubComponent", "core" + std::to_string(i), 0, ComponentInfo::SHARE_NONE, core_params);
            if (core) {
            } else {
                PE_LOG(1, "[core%d] 匿名加载失败\n", i);
            }
        } else {
            // 若由用户配置，补充必要参数（若Python侧未给全量）
            // 这里不强制覆盖，参数以Python为准
        }
        
        if (core) {
            core->setParentInterface(this);
            // 为每个核心配置内存Link（若用户在Python连接了对应端口则不为None）
            std::string port = "core" + std::to_string(i) + "_mem";
            Link* l = configureLink(port);
            if (l) core->setMemoryLink(l);
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
        internal_ring_ = nullptr;
        return;
    }
    
    // 检查是否使用优化版环形网络（默认使用）
    // 注意：此时我们在init阶段，需要存储参数以便后续使用
    bool use_optimized = use_optimized_ring_;
    
    if (use_optimized) {
        
        // 使用新的OptimizedInternalRing
        int num_vcs = 2;                // 每方向2个虚拟通道
        uint32_t credits_per_vc = 8;    // 每VC 8个信用
        
        optimized_ring_ = new OptimizedInternalRing(num_cores_, num_vcs, credits_per_vc, output_);
        internal_ring_ = nullptr;       // 不使用旧实现
        
                        // num_cores_, num_vcs, credits_per_vc);
    } else {
        
        // 使用原始InternalRing实现
        int latency_cycles = 1;  // 默认1周期延迟
        internal_ring_ = new InternalRing(num_cores_, latency_cycles, output_);
        optimized_ring_ = nullptr;  // 不使用新实现
        
                        // num_cores_, latency_cycles);
    }
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
    
    for (int i = 0; i < num_cores_; i++) {
        total_spikes += unit_states_[i].spikes_processed;
        total_fired += unit_states_[i].neurons_fired;
        total_utilization += unit_states_[i].utilization;
    }
    
    // 更新统计信息
    stat_neurons_fired_->addData(total_fired);
    stat_avg_utilization_->addData(total_utilization / num_cores_);
    
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

void MultiCorePE::handleCrossCoreRouting() {
    if (!internal_ring_) return;
    
    // 检查每个处理单元是否有跨核消息
    for (int i = 0; i < num_cores_; i++) {
        RingMessage msg;
        if (internal_ring_->receiveMessage(i, msg)) {
            if (msg.type == RingMessageType::SPIKE_MESSAGE && msg.payload.spike_data) {
                // 将脉冲传递给目标处理单元
                int target_unit = msg.dst_unit;
                if (target_unit >= 0 && target_unit < num_cores_) {
                    deliverSpikeToCore(target_unit, msg.payload.spike_data);
                    
                    PE_LOG(4, "🔄 跨核脉冲路由: 核心%d -> 核心%d\n", 
                                   msg.src_unit, msg.dst_unit);
                } else {
                    PE_LOG(2, "⚠️ 无效的目标单元: %d\n", target_unit);
                    delete msg.payload.spike_data;
                }
            }
        }
    }
}

void MultiCorePE::handleOptimizedCrossCoreRouting() {
    if (!optimized_ring_) return;
    
    // 检查每个处理单元是否有跨核消息（使用新的优化环形网络）
    for (int i = 0; i < num_cores_; i++) {
        RingMessage msg;
        while (optimized_ring_->receiveMessage(i, msg)) {
            if (msg.type == RingMessageType::SPIKE_MESSAGE && msg.payload.spike_data) {
                // 将脉冲传递给目标处理单元
                int target_unit = msg.dst_unit;
                if (target_unit >= 0 && target_unit < num_cores_) {
                    deliverSpikeToCore(target_unit, msg.payload.spike_data);
                    
                    // 增加跨核通信统计
                    inter_core_messages_count_++;
                    stat_inter_core_messages_->addData(1);
                    
                    PE_LOG(4, "🔄 优化跨核脉冲路由: 核心%d -> 核心%d\n", 
                                   msg.src_unit, msg.dst_unit);
                } else {
                    PE_LOG(2, "⚠️ 无效的目标单元: %d\n", target_unit);
                    delete msg.payload.spike_data;
                }
            } else {
                // 处理其他类型的消息（内存请求、控制消息等）
                PE_LOG(3, "🔄 处理非脉冲消息: 类型=%d\n", 
                               static_cast<int>(msg.type));
            }
        }
    }
    
    // 定期输出网络统计信息
    if (current_cycle_ % 5000 == 0 && verbose_ >= 2) {
        double avg_latency = optimized_ring_->getAverageLatency();
        double utilization = optimized_ring_->getNetworkUtilization();
        int pending_msgs = optimized_ring_->getPendingMessageCount();
        
        //                 current_cycle_, avg_latency, utilization * 100.0, pending_msgs);
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



// ===== InternalRing 实现 =====

InternalRing::InternalRing(int num_nodes, int latency_cycles, SST::Output* output)
    : num_nodes_(num_nodes), latency_cycles_(latency_cycles), output_(output) {
    
    // 初始化每个节点的输入输出队列
    node_input_queues_.resize(num_nodes_);
    node_output_queues_.resize(num_nodes_);
    
    // 初始化统计变量
    total_messages_routed_ = 0;
    total_latency_cycles_ = 0;
    
    //                 num_nodes_, latency_cycles_);
}

InternalRing::~InternalRing() {
    // 清理所有队列中的消息
    for (int i = 0; i < num_nodes_; i++) {
        while (!node_input_queues_[i].empty()) {
            RingMessage& msg = node_input_queues_[i].front();
            if (msg.type == RingMessageType::SPIKE_MESSAGE && msg.payload.spike_data) {
                delete msg.payload.spike_data;
            }
            node_input_queues_[i].pop();
        }
        
        while (!node_output_queues_[i].empty()) {
            RingMessage& msg = node_output_queues_[i].front();
            if (msg.type == RingMessageType::SPIKE_MESSAGE && msg.payload.spike_data) {
                delete msg.payload.spike_data;
            }
            node_output_queues_[i].pop();
        }
    }
    
    // 清理环形缓冲区
    while (!ring_buffer_.empty()) {
        RingMessage& msg = ring_buffer_.front();
        if (msg.type == RingMessageType::SPIKE_MESSAGE && msg.payload.spike_data) {
            delete msg.payload.spike_data;
        }
        ring_buffer_.pop();
    }
}

bool InternalRing::sendMessage(const RingMessage& msg) {
    if (msg.src_unit < 0 || msg.src_unit >= num_nodes_ || 
        msg.dst_unit < 0 || msg.dst_unit >= num_nodes_) {
                    //    msg.src_unit, msg.dst_unit);
        return false;
    }
    
    // 检查输出队列是否有空间
    if (node_output_queues_[msg.src_unit].size() >= 100) {  // 限制队列大小
        return false;
    }
    
    // 将消息加入源节点的输出队列
    node_output_queues_[msg.src_unit].push(msg);
    
                    // msg.src_unit, msg.dst_unit);
    
    return true;
}

bool InternalRing::receiveMessage(int node_id, RingMessage& msg) {
    if (node_id < 0 || node_id >= num_nodes_) {
        return false;
    }
    
    if (node_input_queues_[node_id].empty()) {
        return false;
    }
    
    msg = node_input_queues_[node_id].front();
    node_input_queues_[node_id].pop();
    
    
    return true;
}

void InternalRing::tick() {
    // 简化的环形网络实现：直接路由消息
    for (int src = 0; src < num_nodes_; src++) {
        while (!node_output_queues_[src].empty()) {
            RingMessage msg = node_output_queues_[src].front();
            node_output_queues_[src].pop();
            
            routeMessage(msg);
            total_messages_routed_++;
        }
    }
    
    // 处理环形缓冲区中的延迟消息
    std::queue<RingMessage> delayed_messages;
    while (!ring_buffer_.empty()) {
        RingMessage msg = ring_buffer_.front();
        ring_buffer_.pop();
        
        // 检查延迟是否满足
        uint64_t current_time = 0;  // 这里简化，实际应该获取当前时钟
        if (current_time - msg.timestamp >= static_cast<uint64_t>(latency_cycles_)) {
            // 延迟满足，发送到目标节点
            node_input_queues_[msg.dst_unit].push(msg);
            total_latency_cycles_ += (current_time - msg.timestamp);
        } else {
            // 延迟未满足，重新加入缓冲区
            delayed_messages.push(msg);
        }
    }
    
    // 将延迟消息重新加入缓冲区
    ring_buffer_ = delayed_messages;
}

bool InternalRing::hasTrafficForNode(int node_id) const {
    if (node_id < 0 || node_id >= num_nodes_) {
        return false;
    }
    return !node_input_queues_[node_id].empty();
}

int InternalRing::getPendingMessageCount() const {
    int total = ring_buffer_.size();
    for (int i = 0; i < num_nodes_; i++) {
        total += node_input_queues_[i].size() + node_output_queues_[i].size();
    }
    return total;
}

double InternalRing::getAverageLatency() const {
    if (total_messages_routed_ == 0) return 0.0;
    return static_cast<double>(total_latency_cycles_) / static_cast<double>(total_messages_routed_);
}

int InternalRing::getNextNode(int current_node) const {
    return (current_node + 1) % num_nodes_;
}

void InternalRing::routeMessage(const RingMessage& msg) {
    if (latency_cycles_ <= 0) {
        // 零延迟，直接发送
        node_input_queues_[msg.dst_unit].push(msg);
    } else {
        // 有延迟，加入环形缓冲区
        ring_buffer_.push(msg);
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

// ===== 内存响应处理 =====

void MultiCorePE::handleMemoryResponse(SST::Interfaces::StandardMem::Request* resp) {
    if (!resp) return;
    
    PE_LOG(4, "📨 收到内存响应: ID=%" PRIu64 "\n", 
                    resp->getID());
    
    // 查找对应的挂起请求
    auto it = pending_memory_requests_.find(resp->getID());
    if (it != pending_memory_requests_.end()) {
        SpikeEvent* original_spike = it->second;
        pending_memory_requests_.erase(it);
        
        // 处理原始脉冲事件
        if (original_spike) {
            handleExternalSpike(original_spike);
        }
        
        stat_memory_requests_->addData(1);
    } else {
        PE_LOG(2, "⚠️ 未找到对应的挂起内存请求: ID=%" PRIu64 "\n", resp->getID());
    }
    
    delete resp;
}

// ===== SnnPEParentInterface 实现 =====

void MultiCorePE::sendSpike(SpikeEvent* event) {
    if (!event) return;
    
    PE_LOG(4, "📤 从SubComponent接收脉冲: 源神经元%d -> 目标神经元%d\n",
                    event->getSourceNeuron(), event->getDestinationNeuron());
    
    int target_unit = determineTargetUnit(event->getDestinationNeuron());
    
    if (target_unit >= 0 && target_unit < num_cores_) {
        // 目标在本PE内，通过内部互连路由
        // 确定源核心（由于这是从SubComponent调用的，我们需要找到源核心）
        int src_core = determineTargetUnit(event->getSourceNeuron());
        if (src_core >= 0 && src_core < num_cores_) {
            routeInternalSpike(src_core, target_unit, event);
        } else {
            // 源不在本PE，直接递送给目标
            deliverSpikeToCore(target_unit, event);
        }
    } else {
        // 目标在其他PE，通过外部接口发送
        sendExternalSpike(event);
    }
}

void MultiCorePE::requestMemoryAccess(uint64_t address, size_t size, 
                                    std::function<void(const void*)> callback) {
    // TODO: 在Phase 2中实现内存访问
    PE_LOG(4, "📨 接收内存访问请求: 地址=0x%lx, 大小=%zu\n", address, size);
    
    // 暂时提供一个虚拟的响应
    static float dummy_data = 0.5f;
    if (callback) {
        callback(&dummy_data);
    }
}

void MultiCorePE::deliverSpikeToCore(int core_id, SpikeEvent* spike) {
    if (core_id < 0 || core_id >= num_cores_ || !spike) {
        // printf("DEBUG: deliverSpikeToCore失败 - 无效参数：core_id=%d, spike=%p，节点%d\n", core_id, (void*)spike, node_id_);
        // fflush(stdout);
        delete spike;
        return;
    }
    
    // 检查核心是否存在
    if (cores_[core_id] == nullptr) {
        // printf("DEBUG: deliverSpikeToCore失败 - 核心%d未配置，节点%d\n", core_id, node_id_);
        // fflush(stdout);
        PE_LOG(2, "⚠️ 核心%d未配置，丢弃脉冲\n", core_id);
        delete spike;
        return;
    }
    
    // 直接调用SnnPE SubComponent的接口
    cores_[core_id]->deliverSpike(spike);
    
    // 更新两种统计：SST统计对象和本地unit_states_
    stat_spikes_processed_->addData(1);
    unit_states_[core_id].spikes_processed++;
    
    // printf("DEBUG: deliverSpikeToCore完成 - 统计已更新：SST统计+本地unit_states_[%d]，节点%d\n", core_id, node_id_);
    // fflush(stdout);

    PE_LOG(4, "📨 向核心%d递送脉冲: 神经元%d\n", 
                    core_id, spike->getDestinationNeuron());
}

void MultiCorePE::injectStepActivations(uint32_t seq, uint64_t sim_time_ns) {
    if (!step_activation_enable_ || step_activation_fanout_ == 0 || total_neurons_ <= 0) {
        return;
    }
    double fraction = step_activation_fraction_;
    if (fraction <= 0.0) return;
    if (fraction > 1.0) fraction = 1.0;

    if (stat_step_activation_invocations_) {
        stat_step_activation_invocations_->addData(1);
    }

    const uint64_t local_total = static_cast<uint64_t>(total_neurons_);
    const uint64_t neurons_per_pe = static_cast<uint64_t>(neurons_per_pe_cfg_);
    const uint64_t max_global = (total_nodes_ > 0 && neurons_per_pe > 0)
        ? static_cast<uint64_t>(total_nodes_) * neurons_per_pe
        : 0ULL;
    static long diag_cap_cache = LONG_MIN;
    if (diag_cap_cache == LONG_MIN) {
        const char* env = std::getenv("STEP_ACTIVATION_DIAG_CAP");
        diag_cap_cache = env ? std::strtol(env, nullptr, 10) : -1;
    }
    const uint64_t diag_cap = (diag_cap_cache > 0) ? static_cast<uint64_t>(diag_cap_cache) : 0ULL;
    std::mt19937_64 rng(step_activation_seed_ ^ (static_cast<uint64_t>(seq) + (static_cast<uint64_t>(node_id_) << 32)));
    std::uniform_int_distribution<uint64_t> post_dist(0, local_total - 1);
    std::bernoulli_distribution pick(fraction);
    const bool activate_all = (fraction >= 0.999999);
    uint64_t spikes_injected = 0;
    uint64_t sources_selected = 0;
    uint64_t spike_attempts = 0;
    uint64_t route_hits = 0;
    uint64_t route_misses = 0;
    uint64_t local_drops = 0;
    bool diag_cap_hit = false;

    // 诊断：仅在 node_id_=0 且 seq 较小时，对路由表做一次全局采样，避免日志爆炸
    static bool route_diag_done = false;
    static int diag_enable_cache = INT_MIN;
    if (diag_enable_cache == INT_MIN) {
        const char* env = std::getenv("STEP_ACTIVATION_DIAG_ENABLE");
        diag_enable_cache = env ? std::atoi(env) : 0;
    }
    const bool step_diag_enabled = (diag_enable_cache != 0);

    if (step_diag_enabled && !route_diag_done && node_id_ == 0 && seq <= 1 && step_activation_use_bcsr_routes_) {
        uint64_t with_routes = 0;
        uint64_t max_routes = 0;
        uint64_t local_edges = 0;
        uint64_t remote_edges = 0;
        for (size_t i = 0; i < step_activation_routes_.size(); ++i) {
            const auto& v = step_activation_routes_[i];
            if (!v.empty()) {
                ++with_routes;
                if (v.size() > max_routes) max_routes = (uint64_t)v.size();
                for (auto post : v) {
                    uint32_t pe_of_post = (neurons_per_pe > 0) ? static_cast<uint32_t>(post / neurons_per_pe) : 0;
                    if (pe_of_post == (uint32_t)node_id_) ++local_edges; else ++remote_edges;
                }
            }
        }
        if (output_) {
            double denom_edges = (local_edges + remote_edges) ? (double)(local_edges + remote_edges) : 1.0;
            double local_ratio = local_edges / denom_edges;
            double remote_ratio = remote_edges / denom_edges;
            output_->verbose(CALL_INFO, 0, 0,
                "[step-activation-summary] node=%d routes_nonempty=%" PRIu64 " total_pre=%zu max_routes=%" PRIu64 " max_global=%" PRIu64 " local=%" PRIu64 " (%.2f) remote=%" PRIu64 " (%.2f)\n",
                node_id_, with_routes, step_activation_routes_.size(), max_routes, max_global,
                local_edges, local_ratio, remote_edges, remote_ratio);
        }
        route_diag_done = true;
    }

    for (int core = 0; core < num_cores_; ++core) {
        uint64_t base = global_neuron_base_ + static_cast<uint64_t>(core) * static_cast<uint64_t>(neurons_per_core_);
        for (int n = 0; n < neurons_per_core_; ++n) {
            if (!activate_all && !pick(rng)) continue;
            ++sources_selected;
            uint64_t pre_global_64 = base + static_cast<uint64_t>(n);
            if (max_global > 0 && pre_global_64 >= max_global) continue;
            uint32_t pre_global = static_cast<uint32_t>(pre_global_64);
            const bool use_routes = step_activation_use_bcsr_routes_ && pre_global < step_activation_routes_.size();
            const auto* routes = use_routes ? &step_activation_routes_[pre_global] : nullptr;
            // 诊断：采样少量带路由的 pre，观察路由规模
            static uint64_t route_sampled = 0;
            if (step_diag_enabled && use_routes && routes && !routes->empty() &&
                node_id_ == 0 && seq <= 1 && route_sampled < 16) {
                printf("[[step-diag-pre]] node=%d seq=%u pre_global=%u routes=%zu\n",
                       node_id_, seq, pre_global, routes->size());
                fflush(stdout);
                ++route_sampled;
            }
            for (uint32_t fan = 0; fan < step_activation_fanout_; ++fan) {
                ++spike_attempts;
                uint64_t post_global_64 = static_cast<uint64_t>(global_neuron_base_) + post_dist(rng);
                if (max_global > 0 && post_global_64 >= max_global) {
                    continue;
                }
                uint32_t post_global = static_cast<uint32_t>(post_global_64);
                if (use_routes) {
                    if (routes && !routes->empty()) {
                        std::uniform_int_distribution<size_t> route_pick(0, routes->size() - 1);
                        post_global = (*routes)[route_pick(rng)];
                        route_hits++;
                    } else {
                        route_misses++;
                    }
                }
                auto* spike = new SpikeEvent(pre_global, post_global, node_id_, 1.0f, sim_time_ns);
                int dst_core = determineTargetUnit(post_global);
                if (dst_core >= 0) {
                    // 目标在本PE
                    deliverSpikeToCore(dst_core, spike);
                    spikes_injected++;
                } else {
                    // 目标在其他PE：通过NIC注入
                    uint32_t neurons_per_pe32 = static_cast<uint32_t>(neurons_per_pe ? neurons_per_pe : 0ULL);
                    uint32_t dest_node = (neurons_per_pe32 > 0) ? (post_global / neurons_per_pe32) : 0u;
                    spike->setDestinationNode(dest_node);
                    sendExternalSpike(spike);
                    spikes_injected++;
                }

                if (diag_cap && spikes_injected >= diag_cap) {
                    diag_cap_hit = true;
                    break;
                }
            }
            if (diag_cap_hit) break;
        }
        if (diag_cap_hit) break;
    }

    if (stat_step_activation_pre_selected_ && sources_selected) {
        stat_step_activation_pre_selected_->addData(sources_selected);
    }
    if (stat_step_activation_spike_attempts_ && spike_attempts) {
        stat_step_activation_spike_attempts_->addData(spike_attempts);
    }
    if (stat_step_activation_spikes_injected_ && spikes_injected) {
        stat_step_activation_spikes_injected_->addData(spikes_injected);
    }
    if (stat_step_activation_route_hits_ && route_hits) {
        stat_step_activation_route_hits_->addData(route_hits);
    }
    if (stat_step_activation_route_misses_ && route_misses) {
        stat_step_activation_route_misses_->addData(route_misses);
    }
    if (stat_step_activation_local_drops_ && local_drops) {
        stat_step_activation_local_drops_->addData(local_drops);
    }

    if (output_ && output_->getVerboseLevel() >= 1) {
        PE_LOG(1,
            "[step-activation] seq=%u summary sources=%llu attempts=%llu spikes_ok=%llu route_hits=%llu route_miss=%llu local_drop=%llu fraction=%.4g fanout=%u use_routes=%d\n",
            seq,
            static_cast<unsigned long long>(sources_selected),
            static_cast<unsigned long long>(spike_attempts),
            static_cast<unsigned long long>(spikes_injected),
            static_cast<unsigned long long>(route_hits),
            static_cast<unsigned long long>(route_misses),
            static_cast<unsigned long long>(local_drops),
            fraction,
            step_activation_fanout_,
            step_activation_use_bcsr_routes_ ? 1 : 0);
    }

    if (step_diag_enabled && node_id_ == 0 && seq <= 1) {
        printf("[[step-diag-stats]] node=%d seq=%u sources=%llu attempts=%llu spikes=%llu hits=%llu miss=%llu cap=%" PRIu64 " cap_hit=%d\n",
               node_id_, seq,
               static_cast<unsigned long long>(sources_selected),
               static_cast<unsigned long long>(spike_attempts),
               static_cast<unsigned long long>(spikes_injected),
               static_cast<unsigned long long>(route_hits),
               static_cast<unsigned long long>(route_misses),
               diag_cap,
               diag_cap_hit ? 1 : 0);
        fflush(stdout);
    }
}

void MultiCorePE::resetAllCoreMembranes() {
    for (auto* core : cores_) {
        if (!core) continue;
        core->resetMembraneState(v_rest_);
    }
}

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

bool MultiCorePE::buildRoutesFromBcsrFile_(const std::string& path, uint32_t core_index) {
    const uint32_t rows_per_core = step_activation_bcsr_rows_per_core_;
    const uint32_t br = step_activation_bcsr_br_ ? step_activation_bcsr_br_ : 16;
    const uint32_t bc = step_activation_bcsr_bc_ ? step_activation_bcsr_bc_ : 16;
    const uint32_t n_block_rows = (rows_per_core + br - 1) / br;
    const size_t floats_per_block = static_cast<size_t>(br) * static_cast<size_t>(bc);

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
    const uint32_t post_base = global_neuron_base_ + core_index * neurons_per_core_;
    for (uint32_t block_row = 0; block_row < n_block_rows; ++block_row) {
        if ((size_t)block_row + 1 >= rowptr.size()) break;
        uint32_t begin = rowptr[block_row];
        uint32_t end = rowptr[block_row + 1];
        if (begin >= total_blocks) break;
        if (end > total_blocks) end = total_blocks;
        for (uint32_t idx = begin; idx < end; ++idx, ++block_index) {
            if (block_index >= total_blocks) break;
            uint32_t block_col = block_cols[idx];
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
            // 守卫 idx/idx_bytes 在合法范围
            if (block_col >= (uint32_t)(step_activation_bcsr_template_.empty() ? 0xFFFFFFFFu : 0xFFFFFFFFu)) {
                continue;
            }
            for (uint32_t rr = 0; rr < br; ++rr) {
                uint32_t post_local = block_row * br + rr;
                if (post_local >= rows_per_core) continue;
                for (uint32_t cc = 0; cc < bc; ++cc) {
                    size_t off = static_cast<size_t>(rr) * bc + cc;
                    uint32_t post_global = blockids[off];
                    // 路由健壮性守护：过滤无效/越界的post_global，避免后续注入到不存在的节点
                    if (post_global == 0xFFFFFFFFu) continue;
                    const uint32_t neurons_per_pe = static_cast<uint32_t>(neurons_per_core_) * static_cast<uint32_t>(num_cores_);
                    const uint32_t max_global = (total_nodes_ > 0 && neurons_per_pe > 0)
                        ? static_cast<uint32_t>(total_nodes_) * neurons_per_pe
                        : 0u;
                    if (max_global == 0u || post_global >= max_global) {
                        continue;
                    }
                    float weight = blockdata[off];
                    if (std::fabs(weight) <= step_activation_bcsr_weight_epsilon_) continue;
                    uint32_t pre_global = block_col * bc + cc;
                    // 仅收集“本PE本地pre”的出边，便于本节点注入；跨PE的pre由其所属节点在本地加载。
                    if ((pre_global / (neurons_per_pe ? neurons_per_pe : 1)) != static_cast<uint32_t>(node_id_)) continue;
                    if (pre_global >= step_activation_routes_.size()) continue;
                    step_activation_routes_[pre_global].push_back(post_global);
                }
            }
        }
    }
    // 诊断：一次性打印装载概览
    if (output_ && output_->getVerboseLevel() >= 1) {
        uint64_t edges = 0;
        for (auto &v : step_activation_routes_) edges += (uint64_t)v.size();
        output_->verbose(CALL_INFO, 1, 0,
            "[step-activation] BCSR reachability loaded: core=%u rows=%u br=%u bc=%u total_blocks(rowptr)=%u used=%u edges=%" PRIu64 "\n",
            core_index, rows_per_core, br, bc, total_blocks_rowptr, total_blocks, (unsigned long long)edges);
    }
    return true;
}

bool MultiCorePE::loadBcsrReachability_() {
    if (output_) {
        output_->verbose(CALL_INFO, 0, 0,
            "[step-activation] node=%d loadBcsrReachability enable=%d use_bcsr=%d template=%s build_local_only=%d rows_per_core=%u br=%u bc=%u\n",
            node_id_, (int)step_activation_enable_, (int)step_activation_use_bcsr_routes_,
            step_activation_bcsr_template_.c_str(), (int)step_activation_build_local_only_,
            step_activation_bcsr_rows_per_core_, step_activation_bcsr_br_, step_activation_bcsr_bc_);
    }
    if (step_activation_bcsr_template_.empty()) {
        output_->verbose(CALL_INFO, 0, 0, "⚠️ 未提供 step_activation_bcsr_template，无法加载BCSR索引\n");
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
                if (!buildRoutesFromBcsrFile_(path, core)) { success = false; break; }
            }
            if (!success) break;
        }
    if (success) {
        size_t with_routes = 0;
        for (const auto& vec : step_activation_routes_) {
            if (!vec.empty()) ++with_routes;
        }
        if (!step_activation_route_ack_logged_) {
            output_->verbose(CALL_INFO, 1, 0,
                "[step-activation] BCSR reachability loaded: pre_with_routes=%zu total_pre=%zu\n",
                with_routes, step_activation_routes_.size());
            // 每个节点仅打印一次远端/本地比例摘要，确认跨PE路由是否存在
            computeRouteRatios_();
            step_activation_route_ack_logged_ = true;
        }
        if (output_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[step-activation] node=%d loadBcsrReachability success route_vectors=%zu\n",
                node_id_, step_activation_routes_.size());
        }
    } else {
        if (!step_activation_route_warned_) {
            output_->verbose(CALL_INFO, 0, 0,
                "⚠️ step_activation BCSR route build failed, routes cleared; using fallback sampling\n");
            step_activation_route_warned_ = true;
        }
        step_activation_routes_.clear();
        if (output_) {
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
    if (output_) {
        output_->verbose(CALL_INFO, 1, 0,
            "[step-activation] route_ratio: node=%d local_edges=%" PRIu64 " remote_edges=%" PRIu64 " total=%" PRIu64 " local_ratio=%.4f remote_ratio=%.4f\n",
            node_id_, local_edges, remote_edges, total_edges, local_ratio, remote_ratio);
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
        
        // 设置脉冲处理回调
        external_nic_->setSpikeHandler([this](SpikeEvent* spike) {
            // 网络接口接收到脉冲时的处理
            this->handleExternalSpike(spike);
        });
        
        // 安装控制事件处理器（若底层NIC支持）
        if (auto* nic_impl = dynamic_cast<SnnNIC*>(external_nic_)) {
            nic_impl->setControlHandler([this](SST::Event* ev){
                auto* gd = dynamic_cast<GatingDecisionEvent*>(ev);
                if (!gd) return;
                // 计算源全局ID（同row映射）
                uint32_t src_global = gd->src_pe * total_neurons_ + gd->src_row;
                // 向所有核心广播（每核自行判断是否命中）
                std::vector<uint32_t> dpes = gd->dest_pes;
                for (auto* core : cores_) {
                    if (!core) continue;
                    auto* sub = dynamic_cast<SnnPESubComponent*>(core);
                    if (!sub) continue;
                    sub->applyGatingDecision(src_global, dpes, current_cycle_, gd->ttl_cycles);
                }
            });
        }

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
    forwardEventToNetworkAdapter(event, "north");
}

void MultiCorePE::handleSouthLinkEvent(SST::Event* event) {
    PE_LOG(3, "📡 收到南向链路事件\n");
    forwardEventToNetworkAdapter(event, "south");
}

void MultiCorePE::handleEastLinkEvent(SST::Event* event) {
    PE_LOG(3, "📡 收到东向链路事件\n");
    forwardEventToNetworkAdapter(event, "east");
}

void MultiCorePE::handleWestLinkEvent(SST::Event* event) {
    PE_LOG(3, "📡 收到西向链路事件\n");
    forwardEventToNetworkAdapter(event, "west");
}

void MultiCorePE::handleNetworkLinkEvent(SST::Event* event) {
    PE_LOG(3, "📡 收到通用网络链路事件\n");
    forwardEventToNetworkAdapter(event, "network");
}

void MultiCorePE::forwardEventToNetworkAdapter(SST::Event* event, const std::string& direction) {
    if (!external_nic_) {
        PE_LOG(2, "⚠️ 网络接口未配置，无法转发%s方向事件\n", direction.c_str());
        delete event;  // 清理事件内存
        return;
    }
    
    // 首先尝试将事件转换为SpikeEvent（直接脉冲事件）
    SpikeEvent* spike_event = dynamic_cast<SpikeEvent*>(event);
    if (spike_event) {
        PE_LOG(3, "🔄 转发%s方向的直接脉冲事件: 神经元%u\n", 
                        direction.c_str(), spike_event->getNeuronId());
        handleExternalSpike(spike_event);
        return;
    }
    
    // 尝试将事件转换为SpikeEventWrapper（SST网络传输的脉冲事件）
    SpikeEventWrapper* wrapper_event = dynamic_cast<SpikeEventWrapper*>(event);
    if (wrapper_event) {
        PE_LOG(3, "📦 收到%s方向的SpikeEventWrapper，开始解包\n", direction.c_str());
        
        // 从wrapper中提取SpikeEvent数据并创建新的SpikeEvent对象
        SpikeEvent* extracted_spike = extractSpikeFromWrapper(wrapper_event);
        if (extracted_spike) {
            PE_LOG(3, "✅ SpikeEventWrapper解包成功: 神经元%u -> 神经元%u\n", 
                            extracted_spike->getSourceNeuron(), extracted_spike->getDestinationNeuron());
            handleExternalSpike(extracted_spike);
        } else {
            PE_LOG(1, "❌ SpikeEventWrapper解包失败\n");
        }
        
        // 清理wrapper（SST会自动管理，但我们需要显式删除）
        delete wrapper_event;
        return;
    }
    
    // 如果都不是脉冲相关事件，记录并忽略
    PE_LOG(2, "⚠️ %s方向收到未知类型事件，忽略\n", direction.c_str());
    delete event;
}

SpikeEvent* MultiCorePE::extractSpikeFromWrapper(SpikeEventWrapper* wrapper) {
    if (!wrapper) {
        PE_LOG(1, "❌ extractSpikeFromWrapper: wrapper为空\n");
        return nullptr;
    }
    
    try {
        PE_LOG(3, "🔍 extractSpikeFromWrapper: 开始从wrapper提取SpikeEvent\n");
        
        // 从wrapper中获取原始的SpikeEvent
        SpikeEvent* original_spike = wrapper->getSpikeEvent();
        if (!original_spike) {
            PE_LOG(1, "❌ wrapper中的SpikeEvent为空\n");
            return nullptr;
        }
        
        // 创建一个新的SpikeEvent副本，避免内存管理冲突
        SpikeEvent* extracted_spike = new SpikeEvent(
            original_spike->getNeuronId(),
            original_spike->getDestinationNeuron(),
            original_spike->getDestinationNode(),
            original_spike->getWeight(),
            original_spike->getTimestamp()
        );
        
        // 复制hop_count属性（直接访问public字段）
        extracted_spike->hop_count = original_spike->hop_count;
        
        PE_LOG(3, "✅ extractSpikeFromWrapper成功: 神经元%u -> 神经元%u (节点%u)\n", 
                        extracted_spike->getSourceNeuron(), 
                        extracted_spike->getDestinationNeuron(), 
                        extracted_spike->getDestinationNode());
        
        return extracted_spike;
        
    } catch (const std::exception& e) {
        PE_LOG(1, "❌ extractSpikeFromWrapper异常: %s\n", e.what());
        return nullptr;
    } catch (...) {
        PE_LOG(1, "❌ extractSpikeFromWrapper未知异常\n");
        return nullptr;
    }
}
