// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// SpikeSource.cc: 脉冲数据源组件实现文件
//

#include <sst/core/sst_config.h>
#include "SpikeSource.h"

#include <fstream>
#include <sstream>
#include <iostream>
#include <algorithm>

using namespace SST;
using namespace SST::SnnDL;

// ===== 构造函数 =====
SpikeSource::SpikeSource(ComponentId_t id, Params& params) : Component(id) {
    
    // 初始化输出对象
    int verbose_level = params.find<int>("verbose", 0);
    output = new Output("SpikeSource[@p:@l]: ", verbose_level, 0, Output::STDOUT);
    
    output->verbose(CALL_INFO, 1, 0, "初始化SpikeSource组件 (ID: %" PRIu64 ")\n", id);
    
    // 读取配置参数
    dataset_path = params.find<std::string>("dataset_path", "");
    if (dataset_path.empty()) {
        output->fatal(CALL_INFO, -1, "错误: dataset_path参数是必需的\n");
    }

    dataset_format = params.find<std::string>("dataset_format", "TEXT");
    time_scale = params.find<float>("time_scale", 1.0f);
    neuron_offset = params.find<uint32_t>("neuron_offset", 0);
    max_events = params.find<uint32_t>("max_events", 0);
    neurons_per_core = params.find<uint32_t>("neurons_per_core", 4);  // 添加neurons_per_core参数
    
    output->verbose(CALL_INFO, 2, 0,
        "数据集参数: path=%s, format=%s, time_scale=%.3f, offset=%u, max_events=%u, neurons_per_core=%u\n",
        dataset_path.c_str(), dataset_format.c_str(), time_scale, neuron_offset, max_events, neurons_per_core);
    
    // 配置输出链接
    spike_output_link = configureLink("spike_output");
    if (!spike_output_link) {
        output->verbose(CALL_INFO, 1, 0, "警告: 无法配置spike_output链接，将在运行时跳过事件发送\n");
    } else {
        output->verbose(CALL_INFO, 2, 0, "配置了输出链接\n");
    }
    
    // 注册时钟处理器（使用较高频率以确保精确的事件时序）
    std::string clock_freq = params.find<std::string>("clock", "1MHz");
    registerClock(clock_freq, new Clock::Handler2<SpikeSource,&SpikeSource::clockTick>(this));
    output->verbose(CALL_INFO, 2, 0, "注册了时钟处理器，频率: %s\n", clock_freq.c_str());
    
    // 初始化状态变量
    current_sim_time = 0;
    data_loaded = false;
    finished_sending = false;
    
    // 初始化统计计数器
    events_loaded_count = 0;
    events_sent_count = 0;
    
    // 注册统计对象
    stat_events_loaded = registerStatistic<uint64_t>("events_loaded");
    stat_events_sent = registerStatistic<uint64_t>("events_sent");
    
    output->verbose(CALL_INFO, 1, 0, "SpikeSource组件构造完成\n");
}

// ===== 析构函数 =====
SpikeSource::~SpikeSource() {
    if (output) {
        delete output;
    }
}

// ===== 生命周期方法 =====
void SpikeSource::init(unsigned int phase) {
    output->verbose(CALL_INFO, 2, 0, "进入init阶段 %u\n", phase);
}

void SpikeSource::setup() {
    output->verbose(CALL_INFO, 1, 0, "进入setup阶段\n");
    
    // 加载数据集
    if (loadDataset()) {
        data_loaded = true;
        output->verbose(CALL_INFO, 1, 0, "数据集加载成功，共%zu个事件\n", spike_queue.size());
    } else {
        output->fatal(CALL_INFO, -1, "数据集加载失败\n");
    }
}

void SpikeSource::finish() {
    output->verbose(CALL_INFO, 1, 0, "进入finish阶段\n");
    
    // 输出最终统计信息
    output->output("=== SpikeSource最终统计 ===\n");
    output->output("加载事件数: %" PRIu64 "\n", events_loaded_count);
    output->output("发送事件数: %" PRIu64 "\n", events_sent_count);
    
    // 更新统计对象
    stat_events_loaded->addData(events_loaded_count);
    stat_events_sent->addData(events_sent_count);
}

// ===== 时钟处理器 =====
bool SpikeSource::clockTick(Cycle_t current_cycle) {
    if (!data_loaded || finished_sending) {
        return false;
    }
    
    // 更新当前仿真时间（1MHz时钟，每个周期1微秒）
    // 数据文件中的时间戳是微秒，所以我们需要将周期数转换为微秒
    // 1MHz时钟意味着每个周期 = 1微秒
    current_sim_time = current_cycle;
    
    // 调试输出：检查时间匹配问题
    if (current_cycle <= 20) {  // 只在前20个周期输出调试信息
        printf("DEBUG: SpikeSource 周期: %lu, 当前时间: %lu, 队列大小: %zu", 
               current_cycle, current_sim_time, spike_queue.size());
        if (!spike_queue.empty()) {
            printf(", 下一个事件时间: %lu", spike_queue.top().timestamp);
        }
        printf("\n");
        fflush(stdout);
    }
    
    // 发送所有到期的脉冲事件
    while (!spike_queue.empty() && spike_queue.top().timestamp <= current_sim_time) {
        const SpikeData& spike_data = spike_queue.top();
        
        // 创建并发送脉冲事件 - 自动计算目标节点ID  
        // ★ 修正：每个PE有16个神经元，需要除以16而不是4
        uint32_t neurons_per_pe = neurons_per_core * 4;  // 4 cores per PE × 4 neurons per core = 16 neurons per PE
        uint32_t dest_node_id = spike_data.neuron_id / neurons_per_pe;
        SpikeEvent* spike_event = new SpikeEvent(spike_data.neuron_id, spike_data.neuron_id, dest_node_id, 1.0, spike_data.timestamp);
        
        // 检查链接是否有效
        if (spike_output_link) {
            // printf("DEBUG: 尝试发送脉冲事件 - 神经元: %u, 时间戳: %lu, Link指针: %p\n", 
            //        spike_data.neuron_id, spike_data.timestamp, (void*)spike_output_link);
            // fflush(stdout);
            
            spike_output_link->send(spike_event);
            events_sent_count++;
            
            output->verbose(CALL_INFO, 4, 0, "发送脉冲: 神经元%u, 时间%" PRIu64 "\n",
                           spike_data.neuron_id, spike_data.timestamp);
        } else {
            output->verbose(CALL_INFO, 2, 0, "警告: 脉冲输出链接为空，丢弃事件: 神经元%u, 时间%" PRIu64 "\n",
                           spike_data.neuron_id, spike_data.timestamp);
            delete spike_event;  // 清理事件内存
        }
        
        spike_queue.pop();
    }
    
    // 检查是否完成发送
    if (spike_queue.empty()) {
        finished_sending = true;
        output->verbose(CALL_INFO, 1, 0, "所有脉冲事件已发送完毕\n");
    }
    
    return false;  // 继续仿真
}

// ===== 私有辅助方法 =====
bool SpikeSource::loadDataset() {
    output->verbose(CALL_INFO, 1, 0, "开始加载数据集: %s (格式: %s)\n", 
                   dataset_path.c_str(), dataset_format.c_str());
    
    if (dataset_format == "TEXT") {
        return loadTextFormat(dataset_path);
    } else if (dataset_format == "NMNIST_AER") {
        return loadNMNISTFormat(dataset_path);
    } else if (dataset_format == "SHD_HDF5") {
        return loadSHDFormat(dataset_path);
    } else {
        output->verbose(CALL_INFO, 1, 0, "不支持的数据集格式: %s\n", dataset_format.c_str());
        return false;
    }
}

bool SpikeSource::loadTextFormat(const std::string& file_path) {
    std::ifstream file(file_path);
    if (!file.is_open()) {
        output->verbose(CALL_INFO, 1, 0, "无法打开文件: %s\n", file_path.c_str());
        return false;
    }
    
    std::string line;
    uint32_t line_count = 0;
    uint32_t events_count = 0;
    
    while (std::getline(file, line) && (max_events == 0 || events_count < max_events)) {
        line_count++;
        if (line.empty() || line[0] == '#') continue;  // 跳过空行和注释
        
        std::istringstream iss(line);
        uint32_t neuron_id;
        uint64_t timestamp;
        
        if (!(iss >> neuron_id >> timestamp)) {
            output->verbose(CALL_INFO, 1, 0, "文件第%u行格式错误\n", line_count);
            continue;
        }
        
        // 检查神经元ID过滤条件
        bool should_load_event = false;
        
        if (neuron_offset == 0) {
            // neuron_offset=0表示加载所有神经元的事件，不进行过滤
            should_load_event = true;
        } else {
            // 使用偏移量进行神经元范围过滤（原有逻辑）
            uint32_t neurons_per_node = 2;  // 根据配置可能需要调整
            uint32_t start_neuron = neuron_offset;
            uint32_t end_neuron = start_neuron + neurons_per_node;
            should_load_event = (neuron_id >= start_neuron && neuron_id < end_neuron);
        }
        
        if (should_load_event) {
            // 时间戳已经是微秒，直接使用
            uint64_t adjusted_timestamp = timestamp;

            spike_queue.push(SpikeData(neuron_id, adjusted_timestamp));  // 保持原始神经元ID
            events_count++;
            events_loaded_count++;
            
            // printf("DEBUG: 加载脉冲事件 - 神经元: %u, 时间戳: %lu\n", 
            //        neuron_id, adjusted_timestamp);
        }
    }
    
    file.close();
    
    output->verbose(CALL_INFO, 1, 0, "TEXT格式加载完成: %u个事件\n", events_count);
    return true;
}

bool SpikeSource::loadNMNISTFormat(const std::string& file_path) {
    // N-MNIST AER格式的实现
    // 这里提供一个简化的实现框架，实际需要根据具体的二进制格式来解析
    
    output->verbose(CALL_INFO, 1, 0, "N-MNIST AER格式暂未完全实现\n");
    output->verbose(CALL_INFO, 1, 0, "建议使用TEXT格式或实现完整的AER解析器\n");
    
    // 简化实现：假设文件已转换为文本格式 (x, y, timestamp, polarity)
    std::ifstream file(file_path);
    if (!file.is_open()) {
        return false;
    }
    
    std::string line;
    uint32_t events_count = 0;
    
    while (std::getline(file, line) && (max_events == 0 || events_count < max_events)) {
        if (line.empty() || line[0] == '#') continue;
        
        std::istringstream iss(line);
        uint32_t x, y, polarity;
        uint64_t timestamp;
        
        if (!(iss >> x >> y >> timestamp >> polarity)) {
            continue;
        }
        
        // 将(x,y)坐标转换为神经元ID (假设28x28图像)
        uint32_t neuron_id = y * 28 + x + neuron_offset;
        uint64_t adjusted_timestamp = convertToSimTime(timestamp);
        
        spike_queue.push(SpikeData(neuron_id, adjusted_timestamp));
        events_count++;
        events_loaded_count++;
    }
    
    file.close();
    
    output->verbose(CALL_INFO, 1, 0, "N-MNIST格式加载完成: %u个事件\n", events_count);
    return true;
}

bool SpikeSource::loadSHDFormat(const std::string& /* file_path */) {
    // SHD HDF5格式的实现
    // 需要链接HDF5库，这里提供一个占位实现
    
    output->verbose(CALL_INFO, 1, 0, "SHD HDF5格式需要HDF5库支持\n");
    output->verbose(CALL_INFO, 1, 0, "请安装HDF5开发库并实现相应的解析代码\n");
    
    // 占位实现：返回false表示未实现
    return false;
}

uint64_t SpikeSource::convertToSimTime(uint64_t data_timestamp) {
    // 应用时间缩放因子
    return static_cast<uint64_t>(data_timestamp * time_scale);
}
