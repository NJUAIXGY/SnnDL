// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// MultiCorePE.cc: 真正的多核脉冲神经网络处理单元实现文件
//

#include <sst/core/sst_config.h>
#include "MultiCorePE.h"
#include "SnnNetworkAdapter.h"
#include "MultiCorePERouterInterface.h"
#include "OptimizedInternalRing.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <functional>

using namespace SST;
using namespace SST::SnnDL;

// ===== MultiCorePE 主组件实现 =====

MultiCorePE::MultiCorePE(ComponentId_t id, Params& params) : Component(id) {
    // 初始化输出对象
    int verbose_level = params.find<int>("verbose", 0);
    output_ = new Output("MultiCorePE[@p:@l]: ", verbose_level, 0, Output::STDOUT);
    
    // output_->verbose(CALL_INFO, 1, 0, "🚀 初始化MultiCorePE组件 (ID: %" PRIu64 ")\n", id);
    
    // 读取基础配置参数
    num_cores_ = params.find<int>("num_cores", 4);
    neurons_per_core_ = params.find<int>("neurons_per_core", 64);
    total_neurons_ = num_cores_ * neurons_per_core_;
    node_id_ = params.find<int>("node_id", 0);
    global_neuron_base_ = params.find<uint64_t>("global_neuron_base", 0);
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
    
    // 权重验证参数
    verify_weights_ = params.find<bool>("verify_weights", false);
    weight_verify_samples_ = params.find<uint32_t>("weight_verify_samples", 16);
    expected_weight_value_ = params.find<float>("expected_weight_value", 0.5f);
    verify_log_each_sample_ = params.find<bool>("verify_log_each_sample", false);
    
    // 权重回退参数
    use_event_weight_fallback_ = params.find<bool>("use_event_weight_fallback", false);
    enable_memory_weights_ = params.find<bool>("enable_memory_weights", true);
    write_weights_on_init_ = params.find<bool>("write_weights_on_init", true);
    
    // output_->verbose(CALL_INFO, 2, 0, 
    //     "🔧 多核PE配置: cores=%d, neurons_per_core=%d, total_neurons=%d, node_id=%d\n",
    //     num_cores_, neurons_per_core_, total_neurons_, node_id_);
    
    // output_->verbose(CALL_INFO, 2, 0, 
    //     "🧠 神经元参数: v_thresh=%.3f, v_reset=%.3f, v_rest=%.3f, tau_mem=%.1fms, t_ref=%d\n",
    //     v_thresh_, v_reset_, v_rest_, tau_mem_, t_ref_);
    
    // 验证参数合理性
    if (num_cores_ <= 0 || num_cores_ > 64) {
        output_->fatal(CALL_INFO, -1, "❌ 错误: num_cores必须在1-64之间，当前值=%d\n", num_cores_);
    }
    if (neurons_per_core_ <= 0 || neurons_per_core_ > 1024) {
        output_->fatal(CALL_INFO, -1, "❌ 错误: neurons_per_core必须在1-1024之间，当前值=%d\n", neurons_per_core_);
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
    
    // output_->verbose(CALL_INFO, 1, 0, "✅ MultiCorePE基础初始化完成\n");

    // 初始化统计收集（必须在构造函数中）
    initializeStatistics();
    
    // 关键修复：在构造函数中初始化网络接口，确保SST能在正确时机调用init()
    initializeNetworkInterface();
}

MultiCorePE::~MultiCorePE() {
    // output_->verbose(CALL_INFO, 1, 0, "🗑️ 销毁MultiCorePE组件\n");
    
    // 清理SnnPE SubComponent核心（SST会自动管理SubComponent的生命周期）
    cores_.clear();
    
    // 清理内部组件
    delete optimized_ring_;
    delete internal_ring_;
    delete controller_;
    delete output_;
    
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
    // output_->verbose(CALL_INFO, 2, 0, "🔄 MultiCorePE初始化阶段 %d\n", phase);
    
    if (phase == 0) {
        // 阶段0：初始化基础组件和端口
        
        // 配置时钟
        std::string clock_freq = "1GHz";  // 默认时钟频率
        // 不需要单独的clock_handler_变量
        registerClock(clock_freq, new Clock::Handler2<MultiCorePE,&MultiCorePE::clockTick>(this));
        
        // output_->verbose(CALL_INFO, 2, 0, "⏰ 配置时钟频率: %s\n", clock_freq.c_str());
        
        // 初始化统计收集
        
        // 初始化端口连接
        external_spike_input_link_ = configureLink("external_spike_input", 
            new Event::Handler2<MultiCorePE,&MultiCorePE::handleExternalSpikeEvent>(this));
        external_spike_output_link_ = configureLink("external_spike_output");
        mem_link_ = configureLink("mem_link");
        
        // output_->verbose(CALL_INFO, 2, 0, "🔗 配置外部端口连接\n");
        
        // 初始化方向链路（用于端口代理机制）
        initializeDirectionLinks();
        
        // 初始化处理单元
        initializeProcessingUnits();
        
        // 初始化内部互连
        initializeInternalRing();
        
        // 初始化多核控制器
        controller_ = new MultiCoreController(this, output_);
        
        // output_->verbose(CALL_INFO, 1, 0, "✅ MultiCorePE阶段0初始化完成\n");

        // 将当前phase转发给所有子核心
        for (auto* core : cores_) {
            if (core) core->init(phase);
        }
        
        // 关键修复：转发init到网络接口
        if (external_nic_) {
            external_nic_->init(phase);
            // output_->verbose(CALL_INFO, 2, 0, "✅ 网络接口init(%u)完成\n", phase);
        }
    }
    else if (phase == 1) {
        // 阶段1：加载权重和配置子组件
        loadAndDistributeWeights();
        // output_->verbose(CALL_INFO, 1, 0, "✅ MultiCorePE阶段1初始化完成\n");

        // 将当前phase转发给所有子核心
        for (auto* core : cores_) {
            if (core) core->init(phase);
        }
        
        // 转发init到网络接口
        if (external_nic_) {
            external_nic_->init(phase);
            // output_->verbose(CALL_INFO, 2, 0, "✅ 网络接口init(%u)完成\n", phase);
        }
    }
    else {
        // 其余phase同样转发
        for (auto* core : cores_) {
            if (core) core->init(phase);
        }
        
        // 转发init到网络接口
        if (external_nic_) {
            external_nic_->init(phase);
            // output_->verbose(CALL_INFO, 2, 0, "✅ 网络接口init(%u)完成\n", phase);
        }
    }
}

void MultiCorePE::setup() {
    // output_->verbose(CALL_INFO, 1, 0, "🔧 MultiCorePE setup阶段\n");
    
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
    
    // 调用网络接口的setup
    if (external_nic_) {
        external_nic_->setup();
        // output_->verbose(CALL_INFO, 2, 0, "✅ 网络接口setup完成\n");
    }
    
    if (!controller_) {
        output_->fatal(CALL_INFO, -1, "❌ 错误: 多核控制器未初始化\n");
    }
    
    // 打印组件配置摘要
    // output_->verbose(CALL_INFO, 1, 0, "📊 MultiCorePE配置摘要:\n");
    // output_->verbose(CALL_INFO, 1, 0, "   - 处理单元数: %d\n", num_cores_);
    // output_->verbose(CALL_INFO, 1, 0, "   - 每核神经元数: %d\n", neurons_per_core_);
    // output_->verbose(CALL_INFO, 1, 0, "   - 总神经元数: %d\n", total_neurons_);
    // output_->verbose(CALL_INFO, 1, 0, "   - 节点ID: %d\n", node_id_);
    // output_->verbose(CALL_INFO, 1, 0, "   - NUMA优化: %s\n", enable_numa_ ? "启用" : "禁用");
    // output_->verbose(CALL_INFO, 1, 0, "   - 测试流量: %s\n", enable_test_traffic_ ? "启用" : "禁用");
    
    // output_->verbose(CALL_INFO, 1, 0, "✅ MultiCorePE setup完成\n");
}

void MultiCorePE::finish() {
    // 更新最终统计信息
    updateStatistics();
    
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
    
    // 调用网络接口的finish
    if (external_nic_) {
        external_nic_->finish();
    }
}

bool MultiCorePE::clockTick(Cycle_t current_cycle) {
    current_cycle_ = current_cycle;
    
    // 详细调试信息（仅在高详细度时输出）
    if (verbose_ >= 4 && current_cycle % 1000 == 0) {
        // output_->verbose(CALL_INFO, 4, 0, "⏰ MultiCorePE时钟周期 %" PRIu64 "\n", current_cycle);
    }
    
    // 0. 测试注入：在首个有效周期从 core0 向 core1 注入一个跨核脉冲（仅当启用测试流量时）
    if (enable_test_traffic_ && !test_injected_ && num_cores_ > 1 && current_cycle_ == 5000) {
        // 构造一个从全局神经元0 -> 全局神经元(neurons_per_core_) 的脉冲
        SpikeEvent* test_spike = new SpikeEvent(0, neurons_per_core_, 0, 0.5f, current_cycle_);
        int src_core = determineTargetUnit(test_spike->getSourceNeuron());
        int dst_core = determineTargetUnit(test_spike->getDestinationNeuron());
        if (src_core >=0 && dst_core >=0 && src_core != dst_core) {
            routeInternalSpike(src_core, dst_core, test_spike);
            output_->verbose(CALL_INFO, 1, 0, "🧪 注入跨核脉冲: 核心%d->核心%d\n", src_core, dst_core);
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
                output_->verbose(CALL_INFO, 3, 0, "🔄 中继转发脉冲: 神经元%d -> 目标节点%d\n", 
                               spike->getDestinationNeuron(), spike->getDestinationNode());
                external_nic_->sendSpike(spike);
                // 不要删除spike，已经转交给网络适配器
            } else {
                output_->verbose(CALL_INFO, 2, 0, "⚠️ 无网络接口，丢弃跨节点脉冲: 神经元%d\n", 
                               spike->getDestinationNeuron());
                delete spike;
            }
        }
    }
    
    // 2. SubComponent时钟由SST自动管理，无需手动调用tick
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
    // Debug output disabled to prevent excessive logging
    // printf("DEBUG: MultiCorePE::handleExternalSpikeEvent被调用，事件指针: %p, 节点ID: %d\n", (void*)ev, node_id_);
    fflush(stdout);
    
    SpikeEvent* spike = dynamic_cast<SpikeEvent*>(ev);
    if (!spike) {
        output_->verbose(CALL_INFO, 1, 0, "⚠️ 接收到非SpikeEvent事件\n");
        delete ev;
        return;
    }
    
    // Debug output disabled to prevent excessive logging
    // printf("DEBUG: SpikeEvent转换成功，神经元%d -> 神经元%d\n", 
    //        spike->getSourceNeuron(), spike->getDestinationNeuron());
    fflush(stdout);
    
    // 检查跳数限制，防止无限循环
    if (spike->isExpired()) {
        output_->verbose(CALL_INFO, 2, 0, "⚠️ 脉冲达到最大跳数限制，丢弃: 源神经元%d -> 目标神经元%d\n",
                        spike->getSourceNeuron(), spike->getDestinationNeuron());
        delete spike;
        return;
    }
    
    spike->incrementHopCount();
    
    output_->verbose(CALL_INFO, 3, 0, "📨 接收外部脉冲: 源神经元%d -> 目标神经元%d, 权重%.3f, 跳数%d\n",
                    spike->getSourceNeuron(), spike->getDestinationNeuron(), spike->getWeight(), spike->getHopCount());
    
    stat_external_spikes_received_->addData(1);
    
    // 检查是否为本地节点（基于目标节点ID而不是神经元ID）
    uint32_t dest_node = spike->getDestinationNode();
    bool is_local = (dest_node == static_cast<uint32_t>(node_id_));

    // 调试输出：显示节点判断结果
    output_->verbose(CALL_INFO, 2, 0, "🔍 脉冲路由判断: 目标神经元=%d, 目标节点=%u, 本地节点=%d, 本地判断=%s\n",
                     spike->getDestinationNeuron(), dest_node, node_id_, is_local ? "本地" : "跨节点");
    // Debug output removed
    fflush(stdout);
    
    if (is_local) {
        // 本地脉冲，加入队列处理
        // Debug output removed
        fflush(stdout);
        external_spike_queue_.push(spike);
        output_->verbose(CALL_INFO, 4, 0, "✅ 本地脉冲已加入队列\n");
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
            output_->verbose(CALL_INFO, 4, 0, "🔄 外部脉冲直接分发到核心%d\n", target_unit);
        } else {
            // 目标不在本MultiCorePE，视为外部转发（若配置了外部输出端口）
            output_->verbose(CALL_INFO, 2, 0, "🔍 准备转发跨节点脉冲: 神经元%d, 目标节点%d, external_nic_=%p, external_spike_output_link_=%p\n",
                           spike->getDestinationNeuron(), spike->getDestinationNode(),
                           (void*)external_nic_, (void*)external_spike_output_link_);
            if (external_nic_) {
                output_->verbose(CALL_INFO, 2, 0, "🌐 尝试通过SnnNIC发送跨节点脉冲: 神经元%d -> 目标节点%d\n",
                               spike->getDestinationNeuron(), spike->getDestinationNode());
                sendExternalSpike(spike);
                output_->verbose(CALL_INFO, 2, 0, "📤 外部转发脉冲到其他PE: 目标神经元%d, 跳数%d, 目标节点%d\n",
                                spike->getDestinationNeuron(), spike->getHopCount(), spike->getDestinationNode());
                return; // sendExternalSpike会接管事件
            } else {
                output_->verbose(CALL_INFO, 1, 0, "⚠️ 无法确定目标处理单元且无外部输出，丢弃: 神经元%d\n",
                                 spike->getDestinationNeuron());
            }
        }
        delete spike;
    }
}

void MultiCorePE::handleExternalSpike(SpikeEvent* spike) {
    if (!spike) return;
    
    output_->verbose(CALL_INFO, 3, 0, "🔄 处理外部脉冲: 目标神经元%d\n", spike->getDestinationNeuron());
    
    // 将脉冲加入外部队列，由时钟处理器处理
    external_spike_queue_.push(spike);
    stat_external_spikes_received_->addData(1);
}

void MultiCorePE::sendExternalSpike(SpikeEvent* spike) {
    if (!spike) return;

    // 自环防护：如果目标节点就是本节点，直接丢弃，避免外部回送循环
    int target_node = static_cast<int>(spike->getDestinationNode());
    if (target_node == node_id_) {
        output_->verbose(CALL_INFO, 2, 0, "⚠️ 试图向自身节点发送外部脉冲，丢弃: 源=%d 目标=%d 节点=%d\n",
                         spike->getSourceNeuron(), spike->getDestinationNeuron(), target_node);
        delete spike;
        return;
    }

    output_->verbose(CALL_INFO, 3, 0, "📤 发送外部脉冲: 源神经元%d -> 目标神经元%d, 跳数%d\n",
                     spike->getSourceNeuron(), spike->getDestinationNeuron(), spike->getHopCount());

    // 优先使用网络适配器，如果未配置则回退到传统链接
    if (external_nic_) {
        // 使用网络适配器发送脉冲（这将触发路由计算和统计收集）
        external_nic_->sendSpike(spike);
        output_->verbose(CALL_INFO, 3, 0, "🌐 通过网络适配器发送脉冲\n");
    } else if (external_spike_output_link_) {
        // 回退到传统链接模式
        external_spike_output_link_->send(spike);
        output_->verbose(CALL_INFO, 3, 0, "🔗 通过传统链接发送脉冲\n");
    } else {
        output_->verbose(CALL_INFO, 2, 0, "⚠️ 没有可用的外部发送方式，丢弃脉冲\n");
        delete spike;
        return;
    }
    
    stat_external_spikes_sent_->addData(1);
}

void MultiCorePE::routeInternalSpike(int src_core, int dst_core, SpikeEvent* spike) {
    if (!spike) return;
    
    if (src_core < 0 || src_core >= num_cores_ || dst_core < 0 || dst_core >= num_cores_) {
        output_->verbose(CALL_INFO, 1, 0, "⚠️ 无效的核心ID: src=%d, dst=%d\n", src_core, dst_core);
        delete spike;
        return;
    }
    
    output_->verbose(CALL_INFO, 4, 0, "🔄 路由内部脉冲: 核心%d -> 核心%d, 神经元%d\n",
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
        output_->verbose(CALL_INFO, 2, 0, "⚠️ 内部环形网络发送失败: 核心%d -> 核心%d\n", src_core, dst_core);
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
    // output_->verbose(CALL_INFO, 2, 0, "📊 初始化统计收集\n");
    
    stat_spikes_processed_ = registerStatistic<uint64_t>("total_spikes_processed");
    stat_inter_core_messages_ = registerStatistic<uint64_t>("inter_core_messages");
    stat_l2_hits_ = registerStatistic<uint64_t>("l2_cache_hits");
    stat_l2_misses_ = registerStatistic<uint64_t>("l2_cache_misses");
    stat_memory_requests_ = registerStatistic<uint64_t>("memory_requests");
    stat_avg_utilization_ = registerStatistic<double>("avg_core_utilization");
    stat_neurons_fired_ = registerStatistic<uint64_t>("total_neurons_fired");
    stat_external_spikes_sent_ = registerStatistic<uint64_t>("external_spikes_sent");
    stat_external_spikes_received_ = registerStatistic<uint64_t>("external_spikes_received");
    
    // output_->verbose(CALL_INFO, 2, 0, "✅ 统计收集初始化完成\n");
}

void MultiCorePE::initializeProcessingUnits() {
    // output_->verbose(CALL_INFO, 2, 0, "🔧 初始化%d个SnnPE SubComponent核心\n", num_cores_);
    
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
        core_params.insert("base_addr", std::to_string(neuron_id_start * 1000)); // 简单地址映射
        core_params.insert("verbose", std::to_string(verbose_));
        
        // 传递权重文件参数
        if (!weights_file_.empty()) {
            core_params.insert("weights_file", weights_file_);
            // output_->verbose(CALL_INFO, 2, 0, "[core%d] 配置权重文件: %s\n", i, weights_file_.c_str());
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
        // output_->verbose(CALL_INFO, 1, 0, "[core%d] 槽位可按 API 加载: %s\n", i, slot_api_ok ? "yes" : "no");

        // 优先尝试通过用户在Python中配置的槽位加载
        SnnCoreAPI* core = loadUserSubComponent<SnnCoreAPI>(
            "core" + std::to_string(i), ComponentInfo::SHARE_NONE);
        if (core) {
            // output_->verbose(CALL_INFO, 1, 0, "[core%d] 已通过用户槽位加载 SnnCoreAPI 实例\n", i);
        }

        if (!core) {
            // 如果用户未配置，则回退到匿名加载默认实现
            core = loadAnonymousSubComponent<SnnCoreAPI>(
                "SnnDL.SnnPESubComponent", "core" + std::to_string(i), 0, ComponentInfo::SHARE_NONE, core_params);
            if (core) {
                // output_->verbose(CALL_INFO, 1, 0, "[core%d] 匿名加载成功\n", i);
            } else {
                output_->verbose(CALL_INFO, 1, 0, "[core%d] 匿名加载失败\n", i);
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
            // output_->verbose(CALL_INFO, 1, 0, "[core%d] memory link = %s\n", i, l ? "connected" : "none");
            if (l) core->setMemoryLink(l);
            cores_.push_back(core);
        } else {
            cores_.push_back(nullptr);
            // output_->verbose(CALL_INFO, 1, 0, "⚠️ 无法加载SnnPE核心%d\n", i);
        }
        
        output_->verbose(CALL_INFO, 3, 0, "   ✅ SnnPE核心%d: 神经元ID范围[%d, %d)\n",
                        i, neuron_id_start, neuron_id_start + neurons_per_core_);
    }
    
    // output_->verbose(CALL_INFO, 2, 0, "✅ SnnPE SubComponent核心初始化完成（%zu个核心）\n", cores_.size());
    
    // 添加权重配置摘要
    // if (!weights_file_.empty()) {
    //     output_->verbose(CALL_INFO, 1, 0, "📋 节点%d权重配置摘要: %zu个核心使用权重文件 %s\n", 
    //                     node_id_, cores_.size(), weights_file_.c_str());
    // }
}

void MultiCorePE::initializeInternalRing() {
    // 单核情况下无需内部环形网络
    if (num_cores_ <= 1) {
        // output_->verbose(CALL_INFO, 2, 0, "🔗 单核配置，跳过内部环形互连初始化\n");
        optimized_ring_ = nullptr;
        internal_ring_ = nullptr;
        return;
    }
    
    // 检查是否使用优化版环形网络（默认使用）
    // 注意：此时我们在init阶段，需要存储参数以便后续使用
    bool use_optimized = use_optimized_ring_;
    
    if (use_optimized) {
        // output_->verbose(CALL_INFO, 2, 0, "🔗 初始化优化的内部环形互连\n");
        
        // 使用新的OptimizedInternalRing
        int num_vcs = 2;                // 每方向2个虚拟通道
        uint32_t credits_per_vc = 8;    // 每VC 8个信用
        
        optimized_ring_ = new OptimizedInternalRing(num_cores_, num_vcs, credits_per_vc, output_);
        internal_ring_ = nullptr;       // 不使用旧实现
        
        // output_->verbose(CALL_INFO, 2, 0, "✅ 优化环形互连初始化完成（%d节点，%d VCs，%d信用/VC）\n", 
                        // num_cores_, num_vcs, credits_per_vc);
    } else {
        // output_->verbose(CALL_INFO, 2, 0, "🔗 初始化原始内部环形互连（对比测试）\n");
        
        // 使用原始InternalRing实现
        int latency_cycles = 1;  // 默认1周期延迟
        internal_ring_ = new InternalRing(num_cores_, latency_cycles, output_);
        optimized_ring_ = nullptr;  // 不使用新实现
        
        // output_->verbose(CALL_INFO, 2, 0, "✅ 原始环形互连初始化完成（%d节点，%d周期延迟）\n", 
                        // num_cores_, latency_cycles);
    }
}

void MultiCorePE::loadAndDistributeWeights() {
    if (weights_file_.empty()) {
        output_->verbose(CALL_INFO, 2, 0, "⚠️ 未指定权重文件，使用默认权重\n");
        return;
    }
    
    // output_->verbose(CALL_INFO, 2, 0, "📥 加载权重文件: %s\n", weights_file_.c_str());
    
    // TODO: 实现权重加载和分布逻辑
    // 这里应该从文件加载权重并分发到各个处理单元
    
    // output_->verbose(CALL_INFO, 2, 0, "✅ 权重加载和分布完成\n");
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
        output_->verbose(CALL_INFO, 3, 0, "📊 周期%" PRIu64 "统计: 脉冲=%" PRIu64 ", 发放=%" PRIu64 ", 利用率=%.2f\n",
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
            output_->verbose(CALL_INFO, 4, 0, "🔥 生成测试流量: %d个脉冲 (已发送%d/%d)\n", 
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
                    
                    output_->verbose(CALL_INFO, 4, 0, "🔄 跨核脉冲路由: 核心%d -> 核心%d\n", 
                                   msg.src_unit, msg.dst_unit);
                } else {
                    output_->verbose(CALL_INFO, 2, 0, "⚠️ 无效的目标单元: %d\n", target_unit);
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
                    
                    output_->verbose(CALL_INFO, 4, 0, "🔄 优化跨核脉冲路由: 核心%d -> 核心%d\n", 
                                   msg.src_unit, msg.dst_unit);
                } else {
                    output_->verbose(CALL_INFO, 2, 0, "⚠️ 无效的目标单元: %d\n", target_unit);
                    delete msg.payload.spike_data;
                }
            } else {
                // 处理其他类型的消息（内存请求、控制消息等）
                output_->verbose(CALL_INFO, 3, 0, "🔄 处理非脉冲消息: 类型=%d\n", 
                               static_cast<int>(msg.type));
            }
        }
    }
    
    // 定期输出网络统计信息
    if (current_cycle_ % 5000 == 0 && verbose_ >= 2) {
        double avg_latency = optimized_ring_->getAverageLatency();
        double utilization = optimized_ring_->getNetworkUtilization();
        int pending_msgs = optimized_ring_->getPendingMessageCount();
        
        // output_->verbose(CALL_INFO, 2, 0, "📊 优化环形网络[周期%" PRIu64 "]: 平均延迟=%.2f, 利用率=%.2f%%, 待处理消息=%d\n",
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
        // output_->verbose(CALL_INFO, 3, 0, "⚖️ 检测到负载不均衡: %.2f (最大%.2f, 最小%.2f)\n",
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
    
    // output_->verbose(CALL_INFO, 2, 0, "🔗 内部环形网络初始化: %d个节点, %d周期延迟\n", 
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
        // output_->verbose(CALL_INFO, 1, 0, "⚠️ 内部环形网络: 无效的节点ID (src=%d, dst=%d)\n", 
                    //    msg.src_unit, msg.dst_unit);
        return false;
    }
    
    // 检查输出队列是否有空间
    if (node_output_queues_[msg.src_unit].size() >= 100) {  // 限制队列大小
        // output_->verbose(CALL_INFO, 2, 0, "⚠️ 内部环形网络: 节点%d输出队列已满\n", msg.src_unit);
        return false;
    }
    
    // 将消息加入源节点的输出队列
    node_output_queues_[msg.src_unit].push(msg);
    
    // output_->verbose(CALL_INFO, 4, 0, "📤 内部环形网络: 节点%d发送消息到节点%d\n", 
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
    
    // output_->verbose(CALL_INFO, 4, 0, "📨 内部环形网络: 节点%d接收消息\n", node_id);
    
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
    
    // output_->verbose(CALL_INFO, 2, 0, "⚖️ 多核控制器初始化: %d个核心\n", parent_pe_->num_cores_);
}

MultiCoreController::~MultiCoreController() {
    // output_->verbose(CALL_INFO, 2, 0, "🗑️ 销毁多核控制器\n");
}

void MultiCoreController::scheduleWork() {
    // 简单的轮询调度策略
    // 实际实现中可以根据负载情况进行智能调度
    
    static int next_core = 0;
    
    // 轮询分配工作到下一个核心
    next_core = (next_core + 1) % parent_pe_->num_cores_;
    core_work_count_[next_core]++;
    total_work_distributed_++;
    
    // output_->verbose(CALL_INFO, 5, 0, "📋 调度工作到核心%d (总工作量%" PRIu64 ")\n", 
    //                 next_core, total_work_distributed_);
}

void MultiCoreController::balanceLoad() {
    // output_->verbose(CALL_INFO, 3, 0, "⚖️ 执行负载均衡\n");
    
    int most_loaded = findMostLoadedCore();
    int least_loaded = findLeastLoadedCore();
    
    if (most_loaded != least_loaded && most_loaded >= 0 && least_loaded >= 0) {
        double load_diff = core_utilization_history_[most_loaded] - core_utilization_history_[least_loaded];
        
        if (load_diff > load_balance_threshold_) {
            redistributeWork();
            load_imbalance_count_++;
            
            // output_->verbose(CALL_INFO, 3, 0, "⚖️ 负载重分布: 核心%d(%.2f) -> 核心%d(%.2f)\n",
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
        
        // output_->verbose(CALL_INFO, 4, 0, "📋 工作重分布: 核心%d -> 核心%d (转移%" PRIu64 "个工作单元)\n",
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
    
    output_->verbose(CALL_INFO, 4, 0, "📨 收到内存响应: ID=%" PRIu64 "\n", 
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
        output_->verbose(CALL_INFO, 2, 0, "⚠️ 未找到对应的挂起内存请求: ID=%" PRIu64 "\n", resp->getID());
    }
    
    delete resp;
}

// ===== SnnPEParentInterface 实现 =====

void MultiCorePE::sendSpike(SpikeEvent* event) {
    if (!event) return;
    
    output_->verbose(CALL_INFO, 4, 0, "📤 从SubComponent接收脉冲: 源神经元%d -> 目标神经元%d\n",
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
    output_->verbose(CALL_INFO, 4, 0, "📨 接收内存访问请求: 地址=0x%lx, 大小=%zu\n", address, size);
    
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
        output_->verbose(CALL_INFO, 2, 0, "⚠️ 核心%d未配置，丢弃脉冲\n", core_id);
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
    
    output_->verbose(CALL_INFO, 4, 0, "📨 向核心%d递送脉冲: 神经元%d\n", 
                    core_id, spike->getDestinationNeuron());
}

void MultiCorePE::initializeDirectionLinks() {
    // output_->verbose(CALL_INFO, 2, 0, "🌐 初始化方向链路代理机制\n");
    
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
    
    // output_->verbose(CALL_INFO, 1, 0, "🔗 方向链路代理配置完成: %d个活跃链路\n", active_links);
}

void MultiCorePE::initializeNetworkInterface() {
    // output_->verbose(CALL_INFO, 2, 0, "🌐 初始化网络接口适配器\n");
    
    // 尝试加载用户配置的网络接口
    // 关键修复：使用SHARE_PORTS允许网络接口暴露端口给hr_router
    external_nic_ = loadUserSubComponent<SnnInterface>(
        "network_interface", ComponentInfo::SHARE_PORTS);
    
    if (external_nic_) {
        // output_->verbose(CALL_INFO, 1, 0, "✅ 通过用户配置成功加载网络接口适配器\n");
        
        // 配置网络接口的节点ID
        external_nic_->setNodeId(node_id_);
        
        // 设置脉冲处理回调
        external_nic_->setSpikeHandler([this](SpikeEvent* spike) {
            // 网络接口接收到脉冲时的处理
            this->handleExternalSpike(spike);
        });
        
        // 注意：SST框架会自动调用SubComponent的init()和setup()方法
        // 手动调用可能导致重复初始化和时序问题，因此移除
        // output_->verbose(CALL_INFO, 2, 0, "🔧 网络适配器将由SST框架自动初始化\n");
        
        // output_->verbose(CALL_INFO, 1, 0, "🔗 网络接口配置完成: %s\n", 
        //                 external_nic_->getNetworkStatus().c_str());
        
        // === 端口代理机制：将父组件的方向链路注入给SnnNetworkAdapter ===
        // output_->verbose(CALL_INFO, 2, 0, "🔗 开始注入方向链路到网络适配器\n");
        
        // 尝试将SnnInterface强制转换为SnnNetworkAdapter以访问链路注入接口
        auto* network_adapter = dynamic_cast<SnnNetworkAdapter*>(external_nic_);
        if (network_adapter) {
            // 注入各个方向的链路（如果存在）
            if (north_link_) {
                // network_adapter->injectDirectionLink("north", north_link_);
                // output_->verbose(CALL_INFO, 2, 0, "✅ 注入北向链路到网络适配器\n");
            }
            if (south_link_) {
                // network_adapter->injectDirectionLink("south", south_link_);
                // output_->verbose(CALL_INFO, 2, 0, "✅ 注入南向链路到网络适配器\n");
            }
            if (east_link_) {
                // network_adapter->injectDirectionLink("east", east_link_);
                // output_->verbose(CALL_INFO, 2, 0, "✅ 注入东向链路到网络适配器\n");
            }
            if (west_link_) {
                // network_adapter->injectDirectionLink("west", west_link_);
                // output_->verbose(CALL_INFO, 2, 0, "✅ 注入西向链路到网络适配器\n");
            }
            if (network_link_) {
                // network_adapter->injectDirectionLink("network", network_link_);
                // output_->verbose(CALL_INFO, 2, 0, "✅ 注入通用网络链路到网络适配器\n");
            }
            
            // output_->verbose(CALL_INFO, 1, 0, "🔄 端口代理机制配置完成\n");
        } else {
            // 检查是否是MultiCorePERouterInterface类型
            auto* router_interface = dynamic_cast<MultiCorePERouterInterface*>(external_nic_);
            if (router_interface) {
                output_->verbose(CALL_INFO, 1, 0, "🎯 MultiCorePERouterInterface模式：专用hr_router集成\n");
                output_->verbose(CALL_INFO, 2, 0, "✅ 无需端口注入，SubComponent自主管理network端口\n");
                output_->verbose(CALL_INFO, 2, 0, "🔗 可直接连接到hr_router：MultiCorePE.network → router.portX\n");
            } else {
                // 检查是否是SnnNIC类型
                // output_->verbose(CALL_INFO, 2, 0, "ℹ️ 网络接口不是SnnNetworkAdapter类型，检查其他类型\n");
                
                // 对于其他类型（如SnnNIC），我们不需要注入链路，它们有自己的network端口
                if (external_nic_) {
                    // output_->verbose(CALL_INFO, 1, 0, "🔗 其他网络接口模式：network端口可直接用于外部连接\n");
                    // output_->verbose(CALL_INFO, 1, 0, "💡 提示：可直接连接 MultiCorePE的SubComponent端口到外部路由器\n");
                }
            }
        }
    } else {
        // output_->verbose(CALL_INFO, 2, 0, "⚠️ 未配置网络接口适配器，将使用传统端口模式\n");
    }
}

// === 网络端口事件处理器实现 ===

void MultiCorePE::handleNorthLinkEvent(SST::Event* event) {
    output_->verbose(CALL_INFO, 3, 0, "📡 收到北向链路事件\n");
    forwardEventToNetworkAdapter(event, "north");
}

void MultiCorePE::handleSouthLinkEvent(SST::Event* event) {
    output_->verbose(CALL_INFO, 3, 0, "📡 收到南向链路事件\n");
    forwardEventToNetworkAdapter(event, "south");
}

void MultiCorePE::handleEastLinkEvent(SST::Event* event) {
    output_->verbose(CALL_INFO, 3, 0, "📡 收到东向链路事件\n");
    forwardEventToNetworkAdapter(event, "east");
}

void MultiCorePE::handleWestLinkEvent(SST::Event* event) {
    output_->verbose(CALL_INFO, 3, 0, "📡 收到西向链路事件\n");
    forwardEventToNetworkAdapter(event, "west");
}

void MultiCorePE::handleNetworkLinkEvent(SST::Event* event) {
    output_->verbose(CALL_INFO, 3, 0, "📡 收到通用网络链路事件\n");
    forwardEventToNetworkAdapter(event, "network");
}

void MultiCorePE::forwardEventToNetworkAdapter(SST::Event* event, const std::string& direction) {
    if (!external_nic_) {
        output_->verbose(CALL_INFO, 2, 0, "⚠️ 网络接口未配置，无法转发%s方向事件\n", direction.c_str());
        delete event;  // 清理事件内存
        return;
    }
    
    // 首先尝试将事件转换为SpikeEvent（直接脉冲事件）
    SpikeEvent* spike_event = dynamic_cast<SpikeEvent*>(event);
    if (spike_event) {
        output_->verbose(CALL_INFO, 3, 0, "🔄 转发%s方向的直接脉冲事件: 神经元%u\n", 
                        direction.c_str(), spike_event->getNeuronId());
        handleExternalSpike(spike_event);
        return;
    }
    
    // 尝试将事件转换为SpikeEventWrapper（SST网络传输的脉冲事件）
    SpikeEventWrapper* wrapper_event = dynamic_cast<SpikeEventWrapper*>(event);
    if (wrapper_event) {
        output_->verbose(CALL_INFO, 3, 0, "📦 收到%s方向的SpikeEventWrapper，开始解包\n", direction.c_str());
        
        // 从wrapper中提取SpikeEvent数据并创建新的SpikeEvent对象
        SpikeEvent* extracted_spike = extractSpikeFromWrapper(wrapper_event);
        if (extracted_spike) {
            output_->verbose(CALL_INFO, 3, 0, "✅ SpikeEventWrapper解包成功: 神经元%u -> 神经元%u\n", 
                            extracted_spike->getSourceNeuron(), extracted_spike->getDestinationNeuron());
            handleExternalSpike(extracted_spike);
        } else {
            output_->verbose(CALL_INFO, 1, 0, "❌ SpikeEventWrapper解包失败\n");
        }
        
        // 清理wrapper（SST会自动管理，但我们需要显式删除）
        delete wrapper_event;
        return;
    }
    
    // 如果都不是脉冲相关事件，记录并忽略
    output_->verbose(CALL_INFO, 2, 0, "⚠️ %s方向收到未知类型事件，忽略\n", direction.c_str());
    delete event;
}

SpikeEvent* MultiCorePE::extractSpikeFromWrapper(SpikeEventWrapper* wrapper) {
    if (!wrapper) {
        output_->verbose(CALL_INFO, 1, 0, "❌ extractSpikeFromWrapper: wrapper为空\n");
        return nullptr;
    }
    
    try {
        output_->verbose(CALL_INFO, 3, 0, "🔍 extractSpikeFromWrapper: 开始从wrapper提取SpikeEvent\n");
        
        // 从wrapper中获取原始的SpikeEvent
        SpikeEvent* original_spike = wrapper->getSpikeEvent();
        if (!original_spike) {
            output_->verbose(CALL_INFO, 1, 0, "❌ wrapper中的SpikeEvent为空\n");
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
        
        output_->verbose(CALL_INFO, 3, 0, "✅ extractSpikeFromWrapper成功: 神经元%u -> 神经元%u (节点%u)\n", 
                        extracted_spike->getSourceNeuron(), 
                        extracted_spike->getDestinationNeuron(), 
                        extracted_spike->getDestinationNode());
        
        return extracted_spike;
        
    } catch (const std::exception& e) {
        output_->verbose(CALL_INFO, 1, 0, "❌ extractSpikeFromWrapper异常: %s\n", e.what());
        return nullptr;
    } catch (...) {
        output_->verbose(CALL_INFO, 1, 0, "❌ extractSpikeFromWrapper未知异常\n");
        return nullptr;
    }
}
