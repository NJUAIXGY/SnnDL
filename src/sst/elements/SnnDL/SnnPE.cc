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
    printf("DEBUG: SnnPE constructor called\n");
    fflush(stdout);
    
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
    
    // 获取权重文件路径
    weights_file_path = params.find<std::string>("weights_file", "");
    
    output->verbose(CALL_INFO, 2, 0, 
        "神经元参数: num=%u, node_id=%u, v_thresh=%.3f, v_reset=%.3f, v_rest=%.3f, tau_mem=%.1fms, t_ref=%u, weights_file=%s\n",
        num_neurons, node_id, v_thresh, v_reset, v_rest, tau_mem, t_ref, weights_file_path.c_str());
    
    // 预计算泄漏因子（将在setup()中根据实际时钟频率调整）
    leak_factor = exp(-1.0f / tau_mem);  // 临时值，setup()中会重新计算
    
    // 初始化神经元状态
    neurons.resize(num_neurons, NeuronState(v_rest));
    output->verbose(CALL_INFO, 2, 0, "初始化了%u个神经元状态\n", num_neurons);
    
    // 尝试加载SubComponent接口（待修复）
    // SST::Params empty_params;
    // snn_interface = loadUserSubComponent<SnnInterface>("interface", ComponentInfo::SHARE_NONE, empty_params);
    
    // if (snn_interface) {
    if (false) {  // 暂时禁用SubComponent模式
        use_interface_mode = true;
        output->verbose(CALL_INFO, 1, 0, "使用SubComponent接口模式\n");
        
        // 配置接口
        // snn_interface->setSpikeHandler(
        //     [this](SpikeEvent* spike) { this->handleInterfaceSpike(spike); }
        // );
        
        // 传统链接设为nullptr
        spike_input_link = nullptr;
        spike_output_link = nullptr;
    } else {
        use_interface_mode = false;
        output->verbose(CALL_INFO, 1, 0, "使用传统Link模式\n");
        
        // 配置传统链接和事件处理器
        spike_input_link = configureLink("spike_input", 
            new Event::Handler2<SnnPE,&SnnPE::handleSpikeEvent>(this));
        if (!spike_input_link) {
            output->fatal(CALL_INFO, -1, "错误: 无法配置spike_input链接\n");
        }
        
        spike_output_link = configureLink("spike_output");
        if (!spike_output_link) {
            output->verbose(CALL_INFO, 1, 0, "警告: 无法配置spike_output链接，将无法发送脉冲到其他组件\n");
        } else {
            output->verbose(CALL_INFO, 2, 0, "成功配置spike_output链接\n");
        }
        
        output->verbose(CALL_INFO, 2, 0, "配置了输入和输出链接\n");
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
    std::string weights_file = params.find<std::string>("weights_file", "");
    if (!weights_file.empty()) {
        output->verbose(CALL_INFO, 1, 0, "将从文件加载权重: %s\n", weights_file.c_str());
    } else {
        output->verbose(CALL_INFO, 1, 0, "未指定权重文件，将使用空权重矩阵\n");
    }
    
    output->verbose(CALL_INFO, 1, 0, "SnnPE组件构造完成\n");
    printf("DEBUG: SnnPE constructor completed successfully\n");
    fflush(stdout);
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
    
    // 如果使用SubComponent接口，则初始化接口
    // 暂时禁用SubComponent检查
    if (false) { // use_interface_mode && snn_interface
        // snn_interface->init(phase);
    }
}

void SnnPE::setup() {
    output->verbose(CALL_INFO, 1, 0, "进入setup阶段\n");
    
    printf("DEBUG: SnnPE setup() called, weights_file_path='%s'\n", weights_file_path.c_str());
    fflush(stdout);
    
    // 如果使用SubComponent接口，则设置接口
    // 暂时禁用SubComponent检查
    if (false) { // use_interface_mode && snn_interface
        // snn_interface->setup();
        output->verbose(CALL_INFO, 1, 0, "SubComponent接口设置完成\n");
    }
    
    // 重新计算泄漏因子（基于实际的时钟频率）
    // 注意：这里我们假设时钟频率已知，实际实现中可能需要从时钟对象获取
    // 简化处理：使用默认的1ms时间步长
    float dt_ms = 1.0f;  // 1毫秒时间步长
    leak_factor = exp(-dt_ms / tau_mem);
    output->verbose(CALL_INFO, 2, 0, "重新计算泄漏因子: %.6f (dt=%.1fms, tau=%.1fms)\n", 
                   leak_factor, dt_ms, tau_mem);
    
    // 加载权重文件
    printf("DEBUG: Loading weights file: '%s'\n", weights_file_path.c_str());
    fflush(stdout);
    
    output->verbose(CALL_INFO, 1, 0, "权重文件参数: '%s'\n", weights_file_path.c_str());
    
    if (!weights_file_path.empty()) {
        if (loadWeights(weights_file_path)) {
            output->verbose(CALL_INFO, 1, 0, "成功加载权重文件: %s\n", weights_file_path.c_str());
            printf("DEBUG: 权重加载成功，连接数: %zu\n", csr_weights.size());
        } else {
            output->verbose(CALL_INFO, 1, 0, "权重文件加载失败: %s，使用空权重矩阵\n", weights_file_path.c_str());
            printf("DEBUG: 权重加载失败，使用空权重矩阵\n");
        }
    } else {
        printf("DEBUG: 未指定权重文件，使用空权重矩阵\n");
        output->verbose(CALL_INFO, 1, 0, "未指定权重文件，使用空权重矩阵\n");
    }
    
    // 如果没有权重文件，初始化空的CSR矩阵
    if (csr_row_ptr.empty()) {
        csr_row_ptr.resize(num_neurons + 1, 0);
        output->verbose(CALL_INFO, 2, 0, "初始化了空的CSR权重矩阵\n");
    }
    
    output->verbose(CALL_INFO, 1, 0, "setup完成，权重矩阵大小: %zu个连接\n", csr_weights.size());
}

void SnnPE::finish() {
    output->verbose(CALL_INFO, 1, 0, "进入finish阶段\n");
    
    // 如果使用SubComponent接口，则完成接口（暂时注释）
    // if (use_interface_mode && snn_interface) {
    //     snn_interface->finish();
    //     output->verbose(CALL_INFO, 1, 0, "SubComponent接口完成\n");
    // }
    
    // 输出最终统计信息
    output->output("=== SnnPE最终统计[节点%u] ===\n", node_id);
    output->output("接收脉冲数: %" PRIu64 "\n", spikes_received_count);
    output->output("生成脉冲数: %" PRIu64 "\n", spikes_generated_count);
    output->output("发放神经元数: %" PRIu64 "\n", neurons_fired_count);
    output->output("突触操作数: %" PRIu64 "\n", synaptic_ops_count);
    output->output("接口模式: %s\n", use_interface_mode ? "SubComponent" : "传统Link");
    
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
    
    printf("DEBUG: 接收到脉冲 - 神经元%u，权重矩阵大小: %zu\n", pre_syn_id, csr_weights.size());
    output->verbose(CALL_INFO, 3, 0, "接收到脉冲事件: 神经元%u\n", pre_syn_id);
    
    // 检查神经元ID是否有效
    if (pre_syn_id >= num_neurons) {
        output->verbose(CALL_INFO, 1, 0, "警告: 无效的神经元ID %u (最大: %u)\n", 
                       pre_syn_id, num_neurons - 1);
        delete ev;
        return;
    }
    
    output->verbose(CALL_INFO, 2, 0, "处理神经元%u的脉冲，CSR行范围: [%lu, %lu)\n", 
                   pre_syn_id, csr_row_ptr[pre_syn_id], csr_row_ptr[pre_syn_id + 1]);
    
    // 使用CSR格式查找所有突触后神经元
    uint64_t row_start = csr_row_ptr[pre_syn_id];
    uint64_t row_end = csr_row_ptr[pre_syn_id + 1];
    
    printf("DEBUG: 神经元%u的连接范围: [%lu, %lu)，连接数: %lu\n", 
           pre_syn_id, row_start, row_end, row_end - row_start);
    
    // 遍历所有连接
    for (uint64_t i = row_start; i < row_end; i++) {
        uint32_t post_syn_id = csr_col_indices[i];
        float weight = csr_weights[i];
        
        output->verbose(CALL_INFO, 3, 0, "突触连接: %u -> %u，权重: %.2f\n", 
                       pre_syn_id, post_syn_id, weight);
        
        // 检查突触后神经元是否处于不应期
        if (neurons[post_syn_id].refractory_timer == 0) {
            // 整合突触输入
            neurons[post_syn_id].v_mem += weight;
            synaptic_ops_count++;
            
            output->verbose(CALL_INFO, 4, 0, "突触输入: %u -> %u, 权重=%.3f, 新v_mem=%.3f\n",
                           pre_syn_id, post_syn_id, weight, neurons[post_syn_id].v_mem);
            
            // 检查是否发放脉冲
            checkAndFireSpike(post_syn_id);
        }
    }
    
    // 清理事件
    delete ev;
}

void SnnPE::handleInterfaceSpike(SpikeEvent* spike_event) {
    if (!spike_event) {
        output->verbose(CALL_INFO, 1, 0, "接收到空的脉冲事件\n");
        return;
    }
    
    spikes_received_count++;
    stat_spikes_received->addData(1);
    
    uint32_t pre_syn_id = spike_event->getNeuronId();
    uint32_t dest_neuron = spike_event->getDestinationNeuron();
    uint32_t dest_node = spike_event->getDestinationNode();
    
    output->verbose(CALL_INFO, 3, 0, "接收到接口脉冲：神经元%u -> 节点%u:神经元%u\n",
                   pre_syn_id, dest_node, dest_neuron);
    
    // 检查是否为本节点的脉冲
    if (dest_node != node_id) {
        output->verbose(CALL_INFO, 1, 0, "警告：接收到非本节点的脉冲（目标节点%u，本节点%u）\n",
                       dest_node, node_id);
        delete spike_event;
        return;
    }
    
    // 检查目标神经元索引
    if (dest_neuron >= num_neurons) {
        output->verbose(CALL_INFO, 1, 0, "警告：目标神经元索引%u超出范围[0, %u)\n",
                       dest_neuron, num_neurons);
        delete spike_event;
        return;
    }
    
    // 处理突触连接（简化版本：直接使用权重）
    float weight = spike_event->getWeight();
    if (weight != 0.0f) {
        // 检查目标神经元是否在不应期
        if (neurons[dest_neuron].refractory_timer == 0) {
            neurons[dest_neuron].v_mem += weight;
            synaptic_ops_count++;
            
            output->verbose(CALL_INFO, 4, 0, "接口突触输入: %u -> %u, 权重=%.3f, 新v_mem=%.3f\n",
                           pre_syn_id, dest_neuron, weight, neurons[dest_neuron].v_mem);
            
            // 检查是否发放脉冲
            checkAndFireSpike(dest_neuron);
        }
    }
    
    // 清理事件
    delete spike_event;
}

// ===== 私有辅助方法 =====
bool SnnPE::loadWeights(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        output->verbose(CALL_INFO, 1, 0, "无法打开权重文件: %s\n", file_path.c_str());
        return false;
    }
    
    output->verbose(CALL_INFO, 1, 0, "开始加载权重文件: %s\n", file_path.c_str());
    
    // 临时存储权重数据
    std::vector<std::vector<std::pair<uint32_t, float>>> temp_weights(num_neurons);
    
    std::string line;
    uint32_t line_count = 0;
    uint32_t connections_loaded = 0;
    
    while (std::getline(file, line)) {
        line_count++;
        if (line.empty() || line[0] == '#') continue;  // 跳过空行和注释
        
        std::istringstream iss(line);
        uint32_t pre_id, post_id;
        float weight;
        
        if (!(iss >> pre_id >> post_id >> weight)) {
            output->verbose(CALL_INFO, 1, 0, "权重文件第%u行格式错误: %s\n", line_count, line.c_str());
            continue;
        }
        
        if (pre_id >= num_neurons || post_id >= num_neurons) {
            output->verbose(CALL_INFO, 1, 0, "权重文件第%u行神经元ID超出范围: pre=%u, post=%u, max=%u\n", 
                           line_count, pre_id, post_id, num_neurons);
            continue;
        }
        
        temp_weights[pre_id].push_back(std::make_pair(post_id, weight));
        connections_loaded++;
        output->verbose(CALL_INFO, 2, 0, "加载突触连接: %u -> %u, 权重: %.2f\n", pre_id, post_id, weight);
    }
    
    file.close();
    output->verbose(CALL_INFO, 1, 0, "权重文件加载完成，共%u行，%u个连接\n", line_count, connections_loaded);
    
    // 构建CSR格式
    csr_row_ptr.clear();
    csr_col_indices.clear();
    csr_weights.clear();
    
    csr_row_ptr.resize(num_neurons + 1, 0);
    
    uint64_t nnz = 0;  // 非零元素计数
    for (uint32_t i = 0; i < num_neurons; i++) {
        csr_row_ptr[i] = nnz;
        for (const auto& conn : temp_weights[i]) {
            csr_col_indices.push_back(conn.first);
            csr_weights.push_back(conn.second);
            nnz++;
        }
        if (!temp_weights[i].empty()) {
            output->verbose(CALL_INFO, 2, 0, "神经元%u有%zu个输出连接\n", i, temp_weights[i].size());
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
    if (neurons[neuron_idx].v_mem >= v_thresh) {
        // 发放脉冲
        SpikeEvent* new_spike = new SpikeEvent(neuron_idx);
        
        // 暂时禁用SubComponent检查
        if (false) { // use_interface_mode && snn_interface
            // 使用SubComponent接口发送
            // snn_interface->sendSpike(new_spike);
        } else if (spike_output_link) {
            // 使用传统Link发送
            spike_output_link->send(new_spike);
        } else {
            // 没有输出通道，直接删除事件
            delete new_spike;
            output->verbose(CALL_INFO, 1, 0, "警告：神经元%u发放脉冲但无输出通道\n", neuron_idx);
        }
        
        // 更新统计
        spikes_generated_count++;
        neurons_fired_count++;
        
        output->verbose(CALL_INFO, 3, 0, "神经元%u发放脉冲 (v_mem=%.3f >= v_thresh=%.3f)\n",
                       neuron_idx, neurons[neuron_idx].v_mem, v_thresh);
        
        // 重置神经元状态
        neurons[neuron_idx].v_mem = v_reset;
        neurons[neuron_idx].refractory_timer = t_ref;
    }
}
