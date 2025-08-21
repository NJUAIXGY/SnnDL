// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// SnnPE.cc: 单核脉冲神经网络处理单元实现文件
//

#include <sst/core/sst_config.h>
#include "SnnPE.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <cmath>

using namespace SST;
using namespace SST::SnnDL;

// ===== 构造函数 =====
SnnPE::SnnPE(ComponentId_t id, Params& params) : Component(id) {
    // 初始化输出对象
    int verbose_level = params.find<int>("verbose", 0);
    output = new Output("SnnPE[@p:@l]: ", verbose_level, 0, Output::STDOUT);
    
    output->verbose(CALL_INFO, 1, 0, "初始化SnnPE组件 (ID: %" PRIu64 ")\n", id);
    
    // 读取配置参数
    num_neurons = params.find<uint32_t>("num_neurons");
    if (num_neurons == 0) {
        output->fatal(CALL_INFO, -1, "错误: num_neurons参数是必需的且必须大于0\n");
    }
    
    // 网络配置
    node_id = params.find<uint32_t>("node_id", 0);
    
    v_thresh = params.find<float>("v_thresh", 1.0f);
    v_reset = params.find<float>("v_reset", 0.0f);
    v_rest = params.find<float>("v_rest", 0.0f);
    tau_mem = params.find<float>("tau_mem", 20.0f);
    t_ref = params.find<uint32_t>("t_ref", 2);
    
    // 内存相关参数（新增）
    base_addr = params.find<uint64_t>("base_addr", 0);
    weights_per_neuron = params.find<uint32_t>("weights_per_neuron", 0);
    
    output->verbose(CALL_INFO, 2, 0, 
        "神经元参数: num=%u, node_id=%u, v_thresh=%.3f, v_reset=%.3f, v_rest=%.3f, tau_mem=%.1fms, t_ref=%u\n",
        num_neurons, node_id, v_thresh, v_reset, v_rest, tau_mem, t_ref);
    
    output->verbose(CALL_INFO, 2, 0, 
        "内存参数: base_addr=0x%lx, weights_per_neuron=%u\n",
        base_addr, weights_per_neuron);
    
    // 预计算泄漏因子（将在setup()中根据实际时钟频率调整）
    leak_factor = exp(-1.0f / tau_mem);  // 临时值，setup()中会重新计算
    
    // 初始化神经元状态
    neurons.resize(num_neurons, NeuronState(v_rest));
    output->verbose(CALL_INFO, 2, 0, "初始化了%u个神经元状态\n", num_neurons);
    
    // 尝试加载SubComponent接口
    snn_interface = loadUserSubComponent<SnnInterface>("network_interface", ComponentInfo::SHARE_NONE);
    
    // 尝试加载嵌入式路由器（参考miranda.cpu模式）
    router = loadUserSubComponent<SST::Interfaces::SimpleNetwork>("router", ComponentInfo::SHARE_NONE, node_id);
    
    if (snn_interface && router) {
        use_interface_mode = true;
        use_embedded_router = true;
        output->verbose(CALL_INFO, 1, 0, "使用分布式SubComponent模式（接口+路由器）\n");
        
        // 配置接口
        snn_interface->setSpikeHandler(
            [this](SpikeEvent* spike) { this->handleInterfaceSpike(spike); }
        );
        
        // 配置路由器回调
        router->setNotifyOnReceive(
            new SST::Interfaces::SimpleNetwork::Handler2<SnnPE, &SnnPE::handleRouterRequest>(this)
        );
        router->setNotifyOnSend(
            new SST::Interfaces::SimpleNetwork::Handler2<SnnPE, &SnnPE::routerSpaceAvailable>(this)
        );
        
        // 传统链接设为nullptr
        spike_input_link = nullptr;
        spike_output_link = nullptr;
    } else if (snn_interface) {
        use_interface_mode = true;
        use_embedded_router = false;
        output->verbose(CALL_INFO, 1, 0, "使用SubComponent接口模式（无嵌入路由器）\n");
        
        // 配置接口
        snn_interface->setSpikeHandler(
            [this](SpikeEvent* spike) { this->handleInterfaceSpike(spike); }
        );
        
        // Phase 4.5: 混合模式 - 保持输入Link以支持SpikeSource，仅输出使用SubComponent
        spike_input_link = configureLink("spike_input", 
            new Event::Handler2<SnnPE,&SnnPE::handleSpikeEvent>(this));
        if (!spike_input_link) {
            output->verbose(CALL_INFO, 1, 0, "警告: 无法配置spike_input链接\n");
        } else {
            output->verbose(CALL_INFO, 2, 0, "混合模式: 配置了spike_input链接以支持SpikeSource\n");
        }
        
        // 输出使用SubComponent，传统输出链接设为nullptr
        spike_output_link = nullptr;
    } else {
        use_interface_mode = false;
        use_embedded_router = false;
        output->verbose(CALL_INFO, 1, 0, "使用传统Link模式\n");
        
        // 尝试配置传统链接和事件处理器（可选）
        spike_input_link = configureLink("spike_input", 
            new Event::Handler2<SnnPE,&SnnPE::handleSpikeEvent>(this));
        if (!spike_input_link) {
            output->verbose(CALL_INFO, 1, 0, "警告: 无法配置spike_input链接，将在纯内存模式下运行\n");
        } else {
            output->verbose(CALL_INFO, 2, 0, "成功配置spike_input链接\n");
        }
        
        spike_output_link = configureLink("spike_output");
        if (!spike_output_link) {
            output->verbose(CALL_INFO, 1, 0, "警告: 无法配置spike_output链接，将无法发送脉冲到其他组件\n");
        } else {
            output->verbose(CALL_INFO, 2, 0, "成功配置spike_output链接\n");
        }
        
        output->verbose(CALL_INFO, 2, 0, "配置了输入和输出链接（可选）\n");
    }
    
    // 配置StandardMem内存接口（新增）
    registerTimeBase("1ns");  // 注册时间基准
    memory_ = loadUserSubComponent<SST::Interfaces::StandardMem>("memory", 
        ComponentInfo::SHARE_NONE, 
        registerTimeBase("1ns"),  // 时间转换器
        new SST::Interfaces::StandardMem::Handler2<SnnPE, &SnnPE::handleMemResponse>(this));
    if (!memory_) {
        output->verbose(CALL_INFO, 1, 0, "警告: 无法加载memory SubComponent，将使用本地权重模式\n");
    } else {
        output->verbose(CALL_INFO, 2, 0, "成功配置StandardMem内存接口\n");
    }
    
    // 注册时钟处理器
    std::string clock_freq = params.find<std::string>("clock", "1GHz");
    registerClock(clock_freq, new Clock::Handler2<SnnPE,&SnnPE::clockTick>(this));
    output->verbose(CALL_INFO, 2, 0, "注册了时钟处理器，频率: %s\n", clock_freq.c_str());
    
    // 初始化统计计数器
    spikes_received_count = 0;
    spikes_generated_count = 0;
    neurons_fired_count = 0;
    synaptic_ops_count = 0;
    
    // 注册统计对象
    stat_spikes_received = registerStatistic<uint64_t>("spikes_received");
    stat_spikes_generated = registerStatistic<uint64_t>("spikes_generated");
    stat_neurons_fired = registerStatistic<uint64_t>("neurons_fired");
    stat_synaptic_ops = registerStatistic<uint64_t>("total_synaptic_ops");
    
    // 获取权重文件路径（在setup()中加载）
    weights_file_path = params.find<std::string>("weights_file", "");
    neuron_id_start = params.find<uint32_t>("neuron_id_start", 0);
    
    // 新增：二进制权重文件路径
    binary_weights_file_path = params.find<std::string>("binary_weights_file", "");
    
    // 测试流量配置
    enable_test_traffic = params.find<bool>("enable_test_traffic", false);
    test_target_node = params.find<uint32_t>("test_target_node", 0);
    test_period = params.find<uint32_t>("test_period", 100);
    test_spikes_per_burst = params.find<uint32_t>("test_spikes_per_burst", 4);
    test_weight = params.find<float>("test_weight", 0.2);

    if (!weights_file_path.empty()) {
        output->verbose(CALL_INFO, 1, 0, "将从文件加载权重: %s\n", weights_file_path.c_str());
        output->verbose(CALL_INFO, 1, 0, "本核心神经元ID范围: %u-%u\n", 
                       neuron_id_start, neuron_id_start + num_neurons - 1);
    } else if (!binary_weights_file_path.empty()) {
        output->verbose(CALL_INFO, 1, 0, "将从二进制文件加载权重: %s\n", binary_weights_file_path.c_str());
        output->verbose(CALL_INFO, 1, 0, "本核心神经元ID范围: %u-%u\n", 
                       neuron_id_start, neuron_id_start + num_neurons - 1);
    } else {
        output->verbose(CALL_INFO, 1, 0, "未指定权重文件，将使用空权重矩阵\n");
    }
    
    output->verbose(CALL_INFO, 1, 0, "SnnPE组件构造完成\n");
}

// ===== 析构函数 =====
SnnPE::~SnnPE() {
    if (output) {
        delete output;
    }
}

// ===== 生命周期方法 =====
void SnnPE::init(unsigned int phase) {
    output->verbose(CALL_INFO, 2, 0, "进入init阶段 %u\n", phase);
    
    // 初始化内存接口
    if (memory_) {
        memory_->init(phase);
        output->verbose(CALL_INFO, 2, 0, "内存接口初始化完成\n");
    }
    
    // 如果使用分布式SubComponent架构，则初始化
    if (use_interface_mode && snn_interface) {
        snn_interface->init(phase);
        output->verbose(CALL_INFO, 2, 0, "网络接口初始化完成\n");
        
        if (use_embedded_router && router) {
            router->init(phase);
            output->verbose(CALL_INFO, 2, 0, "嵌入式路由器初始化完成\n");
        }
    }
}

void SnnPE::setup() {
    output->verbose(CALL_INFO, 1, 0, "进入setup阶段，节点ID=%u\n", node_id);
    
    // 设置内存接口
    output->verbose(CALL_INFO, 1, 0, "检查内存接口，memory_=%p\n", memory_);
    if (memory_) {
        output->verbose(CALL_INFO, 1, 0, "开始内存接口setup\n");
        memory_->setup();
        output->verbose(CALL_INFO, 2, 0, "内存接口设置完成\n");
    }
    
    // 如果使用分布式SubComponent架构，则设置
    output->verbose(CALL_INFO, 1, 0, "检查网络接口，use_interface_mode=%d, snn_interface=%p\n", 
                   use_interface_mode, snn_interface);
    if (use_interface_mode && snn_interface) {
        output->verbose(CALL_INFO, 1, 0, "开始网络接口setup\n");
        snn_interface->setup();
        output->verbose(CALL_INFO, 1, 0, "网络接口设置完成\n");
        
        if (use_embedded_router && router) {
            output->verbose(CALL_INFO, 1, 0, "开始路由器setup\n");
            router->setup();
            output->verbose(CALL_INFO, 1, 0, "嵌入式路由器设置完成\n");
        }
    }
    
    // 重新计算泄漏因子（基于实际的时钟频率）
    output->verbose(CALL_INFO, 1, 0, "开始计算泄漏因子\n");
    float dt_ms = 1.0f;  // 1毫秒时间步长
    leak_factor = exp(-dt_ms / tau_mem);
    output->verbose(CALL_INFO, 2, 0, "重新计算泄漏因子: %.6f (dt=%.1fms, tau=%.1fms)\n", 
                   leak_factor, dt_ms, tau_mem);
    
    // 加载权重文件（仅在非内存模式下使用）
    output->verbose(CALL_INFO, 1, 0, "当前使用%s权重模式\n", 
                   (memory_ && weights_per_neuron > 0) ? "内存请求" : "本地CSR");
    
    // 如果没有配置内存模式，尝试加载本地权重作为降级方案
    if (!memory_ || weights_per_neuron == 0) {
        output->verbose(CALL_INFO, 1, 0, "内存模式未配置，加载文本权重文件作为降级方案\n");
        output->verbose(CALL_INFO, 1, 0, "权重文件路径: '%s'\n", weights_file_path.c_str());
        
        // 加载跨核权重文件
        if (!weights_file_path.empty()) {
            output->verbose(CALL_INFO, 1, 0, "开始加载权重文件: %s\n", weights_file_path.c_str());
            output->verbose(CALL_INFO, 1, 0, "🔥🔥🔥 ABOUT_TO_CALL_LOADWEIGHTS: 即将调用loadWeights函数\n");
            
            if (loadWeights(weights_file_path)) {
                output->verbose(CALL_INFO, 1, 0, "成功加载权重文件: %s\n", weights_file_path.c_str());
            } else {
                output->verbose(CALL_INFO, 1, 0, "权重文件加载失败，使用空权重矩阵\n");
                // 初始化空的CSR矩阵
                csr_row_ptr.resize(num_neurons + 1, 0);
            }
        } else {
            // 初始化空的CSR矩阵
            output->verbose(CALL_INFO, 1, 0, "未指定权重文件，初始化空CSR矩阵\n");
            csr_row_ptr.resize(num_neurons + 1, 0);
            output->verbose(CALL_INFO, 2, 0, "未指定权重文件，初始化了空的CSR权重矩阵\n");
        }
    } else {
        output->verbose(CALL_INFO, 1, 0, "使用内存模式，跳过本地权重加载\n");
        // 在内存模式下也需要初始化空的CSR矩阵，防止checkAndFireSpike访问时出现段错误
        csr_row_ptr.resize(num_neurons + 1, 0);
        output->verbose(CALL_INFO, 2, 0, "内存模式：初始化了空的CSR权重矩阵\n");
    }
    
    output->verbose(CALL_INFO, 1, 0, "setup完成，内存模式: %s，节点ID=%u\n", 
                   (memory_ && weights_per_neuron > 0) ? "已启用" : "降级到本地CSR", node_id);
}

void SnnPE::finish() {
    output->verbose(CALL_INFO, 1, 0, "进入finish阶段\n");
    
    // 如果使用分布式SubComponent架构，则完成
    if (use_interface_mode && snn_interface) {
        snn_interface->finish();
        output->verbose(CALL_INFO, 1, 0, "网络接口完成\n");
        
        if (use_embedded_router && router) {
            router->finish();
            output->verbose(CALL_INFO, 1, 0, "嵌入式路由器完成\n");
        }
    }
    
    // 输出最终统计信息
    output->output("=== SnnPE最终统计[节点%u] ===\n", node_id);
    output->output("接收脉冲数: %" PRIu64 "\n", spikes_received_count);
    output->output("生成脉冲数: %" PRIu64 "\n", spikes_generated_count);
    output->output("发放神经元数: %" PRIu64 "\n", neurons_fired_count);
    output->output("突触操作数: %" PRIu64 "\n", synaptic_ops_count);
    output->output("接口模式: %s\n", use_interface_mode ? "SubComponent" : "传统Link");
    output->output("路由模式: %s\n", use_embedded_router ? "嵌入式路由器" : "无路由器");
    
    // 更新统计对象
    stat_spikes_received->addData(spikes_received_count);
    stat_spikes_generated->addData(spikes_generated_count);
    stat_neurons_fired->addData(neurons_fired_count);
    stat_synaptic_ops->addData(synaptic_ops_count);
}

// ===== 时钟处理器 =====
bool SnnPE::clockTick(Cycle_t current_cycle) {
    // 对所有神经元应用泄漏和不应期更新
    for (uint32_t i = 0; i < num_neurons; i++) {
        // 处理不应期
        if (neurons[i].refractory_timer > 0) {
            neurons[i].refractory_timer--;
        } else {
            // 应用泄漏动态
            applyLeak(i);
        }
    }
    
    // Phase 3: 移除测试内存请求，改为完全由脉冲事件驱动
    // 现在SnnPE只通过handleSpikeEvent中的真实脉冲来触发内存访问
    // 这确保了完整的"脉冲 -> 内存读权重 -> 神经元计算"数据流    // 简单的测试脉冲生成（仅用于测试网络通信）
    // 周期性测试流量：通过SnnNIC向指定节点发包，验证网络与统计
    if (use_interface_mode && snn_interface && enable_test_traffic) {
        if (test_period > 0 && (current_cycle % test_period) == 0) {
            for (uint32_t i = 0; i < test_spikes_per_burst && i < num_neurons; i++) {
                // 选择一个本地神经元作为源，模拟其发放
                SpikeEvent* new_spike = new SpikeEvent(neuron_id_start + i);
                new_spike->setDestinationNode(test_target_node);
                new_spike->setDestinationNeuron(i % num_neurons);
                new_spike->setWeight(test_weight);
                output->verbose(CALL_INFO, 1, 0, "[测试流量] 周期=%" PRIu64 ": 节点%u -> 节点%u, 神经元%u, 权重=%.3f\n",
                               current_cycle, node_id, test_target_node, i % num_neurons, test_weight);
                snn_interface->sendSpike(new_spike);
            }
        }
    }
    
    // 每1000个周期输出一次调试信息
    if (current_cycle % 1000 == 0) {
        output->verbose(CALL_INFO, 3, 0, "时钟滴答: 周期%" PRIu64 "\n", current_cycle);
    }
    
    return false;  // 返回false表示继续仿真
}

// ===== 事件处理器 =====
void SnnPE::handleSpikeEvent(Event* ev) {
    SpikeEvent* spike_ev = dynamic_cast<SpikeEvent*>(ev);
    if (!spike_ev) {
        output->verbose(CALL_INFO, 1, 0, "警告: 接收到非SpikeEvent事件\n");
        delete ev;
        return;
    }
    
    uint32_t pre_syn_id = spike_ev->neuron_id;
    spikes_received_count++;
    
    output->verbose(CALL_INFO, 3, 0, "接收到脉冲事件: 神经元%u\n", pre_syn_id);
    
    // 检查是否为跨核脉冲（有目标神经元信息）
    if (spike_ev->getDestinationNeuron() != 0 || spike_ev->getDestinationNode() != 0) {
        // 这是跨核脉冲，直接施加到目标神经元
        uint32_t target_local_id = spike_ev->getDestinationNeuron();
        double weight = spike_ev->getWeight();
        
        printf("RECV_LINK: 核心%u通过Link接收跨核脉冲 - 源神经元%u -> 本地神经元%u, 权重=%.3f\n", 
               node_id, pre_syn_id, target_local_id, weight);
        
        // 检查目标神经元ID有效性
        if (target_local_id >= num_neurons) {
            printf("RECV_LINK: 错误 - 目标神经元ID %u 超出范围 (最大: %u)\n", 
                   target_local_id, num_neurons - 1);
            delete ev;
            return;
        }
        
        // 记录处理前的膜电位
        float old_v_mem = neurons[target_local_id].v_mem;
        
        // 检查目标神经元是否处于不应期
        if (neurons[target_local_id].refractory_timer == 0) {
            // 整合突触输入
            neurons[target_local_id].v_mem += weight;
            synaptic_ops_count++;
            
            printf("RECV_LINK: 核心%u处理成功 - 神经元%u: %.3f + %.3f = %.3f\n",
                   node_id, target_local_id, old_v_mem, (float)weight, neurons[target_local_id].v_mem);
            
            // 检查是否发放脉冲
            if (neurons[target_local_id].v_mem >= v_thresh) {
                printf("RECV_LINK: 核心%u神经元%u达到阈值，将发放脉冲！(%.3f >= %.3f)\n",
                       node_id, target_local_id, neurons[target_local_id].v_mem, v_thresh);
            }
            
            checkAndFireSpike(target_local_id);
        } else {
            printf("RECV_LINK: 核心%u神经元%u在不应期，忽略脉冲\n", node_id, target_local_id);
        }
        
        delete ev;
        return;
    }
    
    // 传统模式：将源神经元ID转换为本地ID处理
    // 检查神经元ID是否有效（转换为本地ID）
    if (pre_syn_id < neuron_id_start || pre_syn_id >= neuron_id_start + num_neurons) {
        output->verbose(CALL_INFO, 1, 0, "警告: 神经元ID %u 不属于本核心范围 %u-%u\n", 
                       pre_syn_id, neuron_id_start, neuron_id_start + num_neurons - 1);
        delete ev;
        return;
    }
    
    // 转换为本地神经元ID
    uint32_t local_pre_syn_id = pre_syn_id - neuron_id_start;
    
    // 新的内存请求模式
    if (memory_ && weights_per_neuron > 0) {
        // 计算需要的权重在内存中的地址
        uint64_t target_address = base_addr + (local_pre_syn_id * weights_per_neuron * sizeof(float));
        size_t request_size = weights_per_neuron * sizeof(float);
        
        output->verbose(CALL_INFO, 3, 0, "发送内存请求: 神经元%u, 地址=0x%lx, 大小=%zu\n", 
                       local_pre_syn_id, target_address, request_size);
        
        // 创建StandardMem Read请求
        SST::Interfaces::StandardMem::Read* req = new SST::Interfaces::StandardMem::Read(
            target_address, request_size);
        
        // 暂存原始脉冲事件
        PendingRequest pending_req(spike_ev);
        pending_requests[req->getID()] = pending_req;
        
        // 发送内存请求
        memory_->send(req);
        
        output->verbose(CALL_INFO, 3, 0, "内存请求已发送\n");
        // 不删除spike_ev，它被保存在pending_requests中
    } else {
        // 降级到传统模式（如果没有配置内存链接）
        output->verbose(CALL_INFO, 2, 0, "降级到传统CSR模式处理神经元%u的脉冲\n", pre_syn_id);
        
        // 检查是否有本地CSR矩阵
        if (csr_row_ptr.empty() || csr_row_ptr.size() <= local_pre_syn_id + 1) {
            output->verbose(CALL_INFO, 1, 0, "警告: 无本地权重矩阵，忽略脉冲\n");
            delete ev;
            return;
        }
        
        // 使用CSR格式查找所有突触后神经元
        uint64_t row_start = csr_row_ptr[local_pre_syn_id];
        uint64_t row_end = csr_row_ptr[local_pre_syn_id + 1];
        
        // 遍历所有连接
        for (uint64_t i = row_start; i < row_end; i++) {
            uint32_t global_post_syn_id = csr_col_indices[i];
            float weight = csr_weights[i];
            
            // 检查是否为本地连接
            if (global_post_syn_id >= neuron_id_start && global_post_syn_id < neuron_id_start + num_neurons) {
                // 本地连接：转换为本地ID并直接处理
                uint32_t local_post_syn_id = global_post_syn_id - neuron_id_start;
                
                // 检查突触后神经元是否处于不应期
                if (neurons[local_post_syn_id].refractory_timer == 0) {
                    // 整合突触输入
                    neurons[local_post_syn_id].v_mem += weight;
                    synaptic_ops_count++;
                    
                    output->verbose(CALL_INFO, 4, 0, "本地突触输入: %u -> %u (本地%u), 权重=%.3f, 新v_mem=%.3f\n",
                                   pre_syn_id, global_post_syn_id, local_post_syn_id, weight, neurons[local_post_syn_id].v_mem);
                    
                    // 检查是否发放脉冲
                    checkAndFireSpike(local_post_syn_id);
                }
            } else {
                // 跨核连接：创建脉冲事件发送给目标核心
                // 计算目标核心ID和本地神经元ID
                uint32_t dest_node_id = global_post_syn_id / 64;  // 假设每个核心64个神经元
                uint32_t dest_local_neuron = global_post_syn_id % 64;
                
                printf("CROSSCORE: 跨核连接处理 - 本地神经元%u -> 全局神经元%u (核心%u:神经元%u), 权重=%.3f\n",
                       local_pre_syn_id, global_post_syn_id, dest_node_id, dest_local_neuron, weight);
                
                // 创建跨核脉冲事件
                SpikeEvent* new_spike = new SpikeEvent(pre_syn_id);  // 源神经元全局ID
                new_spike->setDestinationNode(dest_node_id);
                new_spike->setDestinationNeuron(dest_local_neuron);
                new_spike->setWeight(weight);
                
                if (spike_output_link) {
                    // 使用传统Link发送
                    spike_output_link->send(new_spike);
                    printf("CROSSCORE: 跨核脉冲已发送: 源神经元%u -> 目标核心%u:神经元%u, 权重=%.3f\n",
                           pre_syn_id, dest_node_id, dest_local_neuron, weight);
                } else {
                    printf("CROSSCORE: 警告 - 无spike_output_link，跨核脉冲丢失\n");
                    delete new_spike;
                }
            }
        }
        
        // 清理事件
        delete ev;
    }
}

void SnnPE::handleInterfaceSpike(SpikeEvent* spike_event) {
    if (!spike_event) {
        printf("RECV_SPIKE: 核心%u接收到空的脉冲事件\n", node_id);
        return;
    }
    
    spikes_received_count++;
    stat_spikes_received->addData(1);
    
    uint32_t pre_syn_id = spike_event->getNeuronId();
    uint32_t dest_neuron = spike_event->getDestinationNeuron();
    uint32_t dest_node = spike_event->getDestinationNode();
    float weight = spike_event->getWeight();
    
    // printf("RECV_SPIKE: 核心%u接收到跨核脉冲 - 源神经元%u -> 目标核心%u:神经元%u, 权重=%.3f\n",
    //        node_id, pre_syn_id, dest_node, dest_neuron, weight);
    
    // 检查是否为本节点的脉冲
    if (dest_node != node_id) {
        printf("RECV_SPIKE: 错误 - 核心%u接收到发给核心%u的脉冲\n", node_id, dest_node);
        delete spike_event;
        return;
    }
    
    // 检查目标神经元索引
    if (dest_neuron >= num_neurons) {
        printf("RECV_SPIKE: 错误 - 目标神经元索引%u超出范围[0, %u)\n", dest_neuron, num_neurons);
        delete spike_event;
        return;
    }
    
    // 处理突触连接（简化版本：直接使用权重）
    if (weight != 0.0f) {
        // 记录接收前的膜电位
        float old_v_mem = neurons[dest_neuron].v_mem;
        
        // 检查目标神经元是否在不应期
        if (neurons[dest_neuron].refractory_timer == 0) {
            neurons[dest_neuron].v_mem += weight;
            synaptic_ops_count++;
            
            // printf("RECV_SPIKE: 核心%u处理脉冲成功 - 神经元%u: %.3f + %.3f = %.3f\n",
            //        node_id, dest_neuron, old_v_mem, weight, neurons[dest_neuron].v_mem);
            
            // 检查是否发放脉冲
            if (neurons[dest_neuron].v_mem >= v_thresh) {
                // printf("RECV_SPIKE: 核心%u神经元%u达到阈值，将发放脉冲！(%.3f >= %.3f)\n",
                //        node_id, dest_neuron, neurons[dest_neuron].v_mem, v_thresh);
            }
            
            checkAndFireSpike(dest_neuron);
        } else {
            printf("RECV_SPIKE: 核心%u神经元%u在不应期，忽略脉冲\n", node_id, dest_neuron);
        }
    }
    
    // 清理事件
    delete spike_event;
}

// ===== 私有辅助方法 =====
bool SnnPE::loadWeights(const std::string& file_path) {
    output->verbose(CALL_INFO, 1, 0, "🚨🚨🚨 LOADWEIGHTS_ENTRY: 新版本loadWeights函数被调用！文件: %s\n", file_path.c_str());
    output->verbose(CALL_INFO, 1, 0, "尝试打开权重文件: %s\n", file_path.c_str());
    std::ifstream file(file_path, std::ios::binary); // 以二进制模式打开
    if (!file.is_open()) {
        output->verbose(CALL_INFO, 1, 0, "无法打开权重文件: %s\n", file_path.c_str());
        return false;
    }
    
    output->verbose(CALL_INFO, 1, 0, "开始加载跨核权重文件: %s\n", file_path.c_str());
    output->verbose(CALL_INFO, 1, 0, "本核心神经元全局ID范围: %u-%u\n", 
                   neuron_id_start, neuron_id_start + num_neurons - 1);
    
    // 临时存储权重数据 - 基于本地神经元ID
    std::vector<std::vector<std::pair<uint32_t, float>>> temp_weights(num_neurons);
    
    // 读取文件头，获取总连接数和本地连接数
    uint32_t total_connections, local_connections;
    file.read(reinterpret_cast<char*>(&total_connections), sizeof(uint32_t));
    if (file.gcount() != sizeof(uint32_t)) {
        output->verbose(CALL_INFO, 1, 0, "错误: 无法读取总连接数，只读取了%ld字节\n", file.gcount());
        return false;
    }
    file.read(reinterpret_cast<char*>(&local_connections), sizeof(uint32_t));
    if (file.gcount() != sizeof(uint32_t)) {
        output->verbose(CALL_INFO, 1, 0, "错误: 无法读取本地连接数，只读取了%ld字节\n", file.gcount());
        return false;
    }
    
    output->verbose(CALL_INFO, 1, 0, "🔥 DEBUG: 新版本loadWeights正在运行！权重文件头: 总连接=%u, 本地连接=%u\n", total_connections, local_connections);
    
    uint32_t connections_loaded = 0;
    uint32_t cross_core_connections = 0;
    
    for (uint32_t i = 0; i < total_connections; ++i) {
        uint32_t global_pre_id, global_post_id;
        float weight;
        
        file.read(reinterpret_cast<char*>(&global_pre_id), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&global_post_id), sizeof(uint32_t));
        file.read(reinterpret_cast<char*>(&weight), sizeof(float));
        
        // 检查此连接的突触前神经元是否属于本核心
        if (global_pre_id >= neuron_id_start && global_pre_id < neuron_id_start + num_neurons) {
            // 属于本核心的突触前神经元，转换为本地ID
            uint32_t local_pre_id = global_pre_id - neuron_id_start;
            
            // 添加此连接（目标可以是任何核心的神经元）
            temp_weights[local_pre_id].push_back(std::make_pair(global_post_id, weight));
            connections_loaded++;
            
            if (global_post_id >= neuron_id_start && global_post_id < neuron_id_start + num_neurons) {
                local_connections++;
            } else {
                cross_core_connections++;
            }
        }
        // 如果突触前神经元不属于本核心，则忽略这条连接
    }
    
    file.close();
    output->verbose(CALL_INFO, 1, 0, "权重文件加载完成，共%u行，%u个连接属于本核心\n", total_connections, connections_loaded);
    output->verbose(CALL_INFO, 1, 0, "连接统计: %u个本地连接，%u个跨核连接\n", local_connections, cross_core_connections);
    
    // 构建CSR格式
    csr_row_ptr.clear();
    csr_col_indices.clear();
    csr_weights.clear();
    
    csr_row_ptr.resize(num_neurons + 1, 0);
    
    uint64_t nnz = 0;  // 非零元素计数
    for (uint32_t i = 0; i < num_neurons; i++) {
        csr_row_ptr[i] = nnz;
        
        for (size_t j = 0; j < temp_weights[i].size(); j++) {
            const auto& conn = temp_weights[i][j];
            uint32_t target_id = conn.first;
            float weight = conn.second;
            
            // 验证目标神经元ID的合理性
            if (target_id > 10000) {  // 不合理的大数值
                output->verbose(CALL_INFO, 1, 0, "错误: 检测到无效的目标神经元ID %u\n", target_id);
                continue;  // 跳过这个损坏的连接
            }
            
            csr_col_indices.push_back(target_id);  // 保存全局神经元ID
            csr_weights.push_back(weight);
            nnz++;
        }
    }
    csr_row_ptr[num_neurons] = nnz;
    
    output->verbose(CALL_INFO, 1, 0, "CSR格式构建完成，共%lu个突触连接\n", nnz);
    
    return true;
}

void SnnPE::applyLeak(uint32_t neuron_idx) {
    float& v_mem = neurons[neuron_idx].v_mem;
    // 指数泄漏: v(t+dt) = v_rest + (v(t) - v_rest) * exp(-dt/tau)
    v_mem = v_rest + (v_mem - v_rest) * leak_factor;
}

void SnnPE::checkAndFireSpike(uint32_t neuron_idx) {
    // 防止递归深度过大导致栈溢出
    static thread_local uint32_t recursion_depth = 0;
    const uint32_t MAX_RECURSION_DEPTH = 10;
    
    if (recursion_depth >= MAX_RECURSION_DEPTH) {
        output->verbose(CALL_INFO, 1, 0, "警告: 检测到递归深度过大（%u），中止神经元%u的脉冲处理\n", 
                       recursion_depth, neuron_idx);
        return;
    }
    
    output->verbose(CALL_INFO, 2, 0, "检查神经元%u发放: v_mem=%.6f, 阈值=%.6f\n",
                   neuron_idx, neurons[neuron_idx].v_mem, v_thresh);
    
    if (neurons[neuron_idx].v_mem >= v_thresh) {
        recursion_depth++;  // 增加递归计数
        
        // 发放脉冲
        output->verbose(CALL_INFO, 2, 0, "🔥 神经元%u发放脉冲! (v_mem=%.6f >= v_thresh=%.6f)\n",
                       neuron_idx, neurons[neuron_idx].v_mem, v_thresh);
        
        // 立即重置神经元状态，防止在递归中重复触发
        neurons[neuron_idx].v_mem = v_reset;
        neurons[neuron_idx].refractory_timer = t_ref;
        
        // 更新统计
        spikes_generated_count++;
        neurons_fired_count++;
        
        // DEBUG: 验证CSR访问的合法性
        if (neuron_idx >= csr_row_ptr.size() - 1) {
            output->verbose(CALL_INFO, 1, 0, "错误: 神经元索引%u超出CSR行指针范围（最大%zu）\n", 
                           neuron_idx, csr_row_ptr.size() - 1);
            recursion_depth--;
            return;
        }
        
        // 查找所有突触后连接
        uint64_t row_start = csr_row_ptr[neuron_idx];
        uint64_t row_end = csr_row_ptr[neuron_idx + 1];
        
        // DEBUG: 验证行边界的合法性
        output->verbose(CALL_INFO, 2, 0, "DEBUG: 神经元%u CSR访问 - 行边界[%lu, %lu), csr_col_indices.size()=%zu, csr_weights.size()=%zu\n", 
                       neuron_idx, row_start, row_end, csr_col_indices.size(), csr_weights.size());
        
        if (row_end > csr_col_indices.size() || row_end > csr_weights.size()) {
            output->verbose(CALL_INFO, 1, 0, "错误: 神经元%u的行边界[%lu, %lu)超出CSR数据范围\n", 
                           neuron_idx, row_start, row_end);
            recursion_depth--;
            return;
        }
        
        output->verbose(CALL_INFO, 4, 0, "处理神经元%u的%lu个输出连接\n", 
                       neuron_idx, row_end - row_start);
        
        // 遍历所有连接，发送脉冲到目标神经元
        for (uint64_t i = row_start; i < row_end; i++) {
            // 额外安全检查：确保索引在有效范围内
            if (i >= csr_col_indices.size() || i >= csr_weights.size()) {
                output->verbose(CALL_INFO, 1, 0, "CRITICAL: 索引%lu超出CSR数据范围（col_size=%zu, weights_size=%zu）\n", 
                               i, csr_col_indices.size(), csr_weights.size());
                break;  // 立即停止处理这个神经元
            }
            
            uint32_t global_target_neuron = csr_col_indices[i];
            float weight = csr_weights[i];
            
            // 验证目标神经元ID的合理性
            if (global_target_neuron > 1000) {  // 设置一个合理的上限
                output->verbose(CALL_INFO, 1, 0, "CRITICAL: 检测到损坏的目标神经元ID %u，中止神经元%u的脉冲处理\n", 
                               global_target_neuron, neuron_idx);
                break;  // 立即停止处理这个神经元
            }
            
            if (weight == 0.0f) continue;  // 跳过零权重连接
            
            // 计算目标核心ID和本地神经元ID（假设每个核心64个神经元）
            uint32_t dest_node_id = global_target_neuron / num_neurons;
            uint32_t local_target_neuron = global_target_neuron % num_neurons;
            
            output->verbose(CALL_INFO, 4, 0, "脉冲连接: 本地神经元%u (全局%u) -> 全局神经元%u (核心%u:神经元%u), 权重=%.3f\n",
                           neuron_idx, neuron_id_start + neuron_idx, global_target_neuron, dest_node_id, local_target_neuron, weight);
            
            // 检查是否为本地连接
            if (global_target_neuron >= neuron_id_start && global_target_neuron < neuron_id_start + num_neurons) {
                // 本地连接：直接处理
                uint32_t true_local_target = global_target_neuron - neuron_id_start;
                output->verbose(CALL_INFO, 4, 0, "本地连接: 神经元%u -> 神经元%u\n",
                               neuron_idx, true_local_target);
                
                // 检查目标神经元是否在不应期
                if (neurons[true_local_target].refractory_timer == 0) {
                    neurons[true_local_target].v_mem += weight;
                    synaptic_ops_count++;
                    
                    output->verbose(CALL_INFO, 5, 0, "本地突触更新: 神经元%u, 新v_mem=%.3f\n",
                                   true_local_target, neurons[true_local_target].v_mem);
                    
                    // 递归检查是否触发新的脉冲（现在有深度限制）
                    checkAndFireSpike(true_local_target);
                }
                
            } else {
                // 跨核连接：需要发送脉冲事件
                output->verbose(CALL_INFO, 3, 0, "跨核连接: 本地神经元%u -> 全局神经元%u (核心%u:神经元%u)\n",
                               neuron_idx, global_target_neuron, dest_node_id, local_target_neuron);
                
                // 创建脉冲事件
                SpikeEvent* new_spike = new SpikeEvent(neuron_id_start + neuron_idx);  // 使用全局神经元ID作为源
                new_spike->setDestinationNode(dest_node_id);
                new_spike->setDestinationNeuron(local_target_neuron);
                new_spike->setWeight(weight);
                
                if (use_interface_mode && snn_interface) {
                    // 通过网络接口发送
                    snn_interface->sendSpike(new_spike);
                    
                } else if (spike_output_link) {
                    // 使用传统Link发送
                    spike_output_link->send(new_spike);
                    
                } else {
                    // 没有输出通道，直接删除事件
                    delete new_spike;
                    output->verbose(CALL_INFO, 1, 0, "警告：神经元%u发放跨核脉冲但无输出通道到全局神经元%u\n", 
                                   neuron_idx, global_target_neuron);
                }
            }
        }
        
        recursion_depth--;  // 减少递归计数
    }
}

// ===== 分布式网络架构方法 =====

bool SnnPE::handleRouterRequest(int vn) {
    if (!router) return false;
    
    SST::Interfaces::SimpleNetwork::Request* req = router->recv(vn);
    while (req) {
        // 解析脉冲数据包
        SST::Event* payload = req->inspectPayload();
        if (payload) {
            SpikeEvent* spike_event = dynamic_cast<SpikeEvent*>(payload);
            
            if (spike_event && spike_event->getDestinationNode() == node_id) {
                // 本地处理脉冲
                processLocalSpike(spike_event);
            } else if (spike_event) {
                // 继续路由到其他节点
                routeSpike(spike_event, spike_event->getDestinationNode());
            }
        }
        
        delete req;
        req = router->recv(vn);
    }
    
    return true;
}

bool SnnPE::routerSpaceAvailable(int vn) {
    return router ? router->spaceToSend(vn, 8) : false;
}

bool SnnPE::initDistributedNetwork() {
    if (!use_embedded_router || !router || !snn_interface) {
        return false;
    }
    
    output->verbose(CALL_INFO, 1, 0, "初始化分布式网络架构\n");
    
    // 这里可以添加更多的网络拓扑设置
    return true;
}

void SnnPE::routeSpike(SpikeEvent* spike_event, uint32_t target_node) {
    if (!router) {
        output->verbose(CALL_INFO, 1, 0, "警告：无路由器，无法路由脉冲到节点%u\n", target_node);
        return;
    }
    
    if (router->spaceToSend(0, 8)) {
        // 创建脉冲事件副本作为负载
        SpikeEvent* payload = new SpikeEvent(*spike_event);
        
        // 创建网络请求
        SST::Interfaces::SimpleNetwork::Request* req = 
            new SST::Interfaces::SimpleNetwork::Request(target_node, node_id, sizeof(SpikeEvent) * 8, true, true, payload);
        
        // 发送
        router->send(req, 0);
        
        output->verbose(CALL_INFO, 3, 0, "路由脉冲：节点%u -> 节点%u\n", node_id, target_node);
    } else {
        output->verbose(CALL_INFO, 1, 0, "警告：路由器缓冲区满，丢弃脉冲到节点%u\n", target_node);
    }
}

void SnnPE::processLocalSpike(SpikeEvent* spike_event) {
    // 处理到达本节点的脉冲
    uint32_t target_neuron = spike_event->getDestinationNeuron();
    
    if (target_neuron < num_neurons) {
        // 应用突触权重
        float weight = spike_event->getWeight();
        neurons[target_neuron].v_mem += weight;
        
        synaptic_ops_count++;
        spikes_received_count++;
        
        output->verbose(CALL_INFO, 3, 0, "处理本地脉冲：神经元%u，权重=%.3f，新膜电位=%.3f\n",
                       target_neuron, weight, neurons[target_neuron].v_mem);
        
        // 检查是否发放脉冲
        checkAndFireSpike(target_neuron);
    }
}

// ===== 内存响应处理器 (使用StandardMem接口) =====
void SnnPE::handleMemResponse(SST::Interfaces::StandardMem::Request *req) {
    output->verbose(CALL_INFO, 3, 0, "接收到内存响应\n");
    
    // 确保这是一个ReadResp
    SST::Interfaces::StandardMem::ReadResp* readResp = 
        dynamic_cast<SST::Interfaces::StandardMem::ReadResp*>(req);
    if (!readResp) {
        output->verbose(CALL_INFO, 1, 0, "警告: 接收到非ReadResp响应\n");
        delete req;
        return;
    }
    
    // 查找与此响应对应的原始请求
    auto it = pending_requests.find(req->getID());
    if (it == pending_requests.end()) {
        output->verbose(CALL_INFO, 1, 0, "警告: 收到未知的内存响应\n");
        delete req;
        return;
    }
    
    PendingRequest pending_req = it->second;
    SpikeEvent* original_spike = pending_req.original_spike;
    uint32_t pre_syn_id = original_spike->neuron_id;
    
    output->verbose(CALL_INFO, 3, 0, "恢复处理神经元%u的脉冲\n", pre_syn_id);
    
    // 从响应中提取权重数据
    std::vector<uint8_t>& data = readResp->data;
    if (data.size() < weights_per_neuron * sizeof(float)) {
        output->verbose(CALL_INFO, 1, 0, "警告: 内存响应数据不足，期望%zu字节，实际%zu字节\n",
                       weights_per_neuron * sizeof(float), data.size());
        delete original_spike;
        pending_requests.erase(it);
        delete req;
        return;
    }
    
    float* weights = reinterpret_cast<float*>(data.data());
    
    // 添加权重数据调试信息
    output->verbose(CALL_INFO, 2, 0, "解析权重数据: %zu字节，%u个权重\n", data.size(), weights_per_neuron);
    
    // 权重数据已从内存正确加载，无需hack
    
    for (uint32_t i = 0; i < weights_per_neuron && i < 8; ++i) {
        output->verbose(CALL_INFO, 2, 0, "权重[%u] = %.6f\n", i, weights[i]);
    }
    
    // 使用获取的权重恢复计算
    for (uint32_t i = 0; i < weights_per_neuron; ++i) {
        uint32_t post_syn_id = i;  // 简化映射：权重i连接到本地神经元i
        
        if (post_syn_id < num_neurons && neurons[post_syn_id].refractory_timer == 0) {
            float weight = weights[i];
            float old_v_mem = neurons[post_syn_id].v_mem;
            
            neurons[post_syn_id].v_mem += weight;
            synaptic_ops_count++;
            
            output->verbose(CALL_INFO, 2, 0, "内存突触输入: %u -> %u, 权重=%.6f, v_mem: %.6f -> %.6f, 阈值=%.6f\n",
                           pre_syn_id, post_syn_id, weight, old_v_mem, neurons[post_syn_id].v_mem, v_thresh);
            
            // 检查是否发放脉冲
            checkAndFireSpike(post_syn_id);
        }
    }
    
    output->verbose(CALL_INFO, 3, 0, "完成处理神经元%u的脉冲（内存模式）\n", pre_syn_id);
    
    // 清理
    delete original_spike;
    pending_requests.erase(it);
    delete req;
}
