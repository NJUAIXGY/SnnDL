// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// SnnNIC.cc: SNN网络接口控制器实现文件
//

#include "SnnNIC.h"
#include <sst/core/serialization/serialize.h>

using namespace SST;
using namespace SST::SnnDL;
using namespace SST::Interfaces;

SnnNIC::SnnNIC(ComponentId_t id, Params& params)
    : SnnInterface(id, params),
      output(nullptr),
      network(nullptr),
      spike_handler(nullptr),
      spikes_sent_count(0),
      spikes_received_count(0),
      packets_sent_count(0),
      packets_received_count(0)
{
    // 获取参数
    node_id = params.find<uint32_t>("node_id", 0);
    link_bw = params.find<std::string>("link_bw", "40GiB/s");
    input_buf_size = params.find<std::string>("input_buf_size", "1KiB");
    output_buf_size = params.find<std::string>("output_buf_size", "1KiB");
    
    int verbose = params.find<int>("verbose", 0);
    
    // 初始化日志输出
    output = new Output("SnnNIC[@p:@l]: ", verbose, 0, Output::STDOUT);
    
    output->verbose(CALL_INFO, 1, 0, "初始化SnnNIC组件，节点ID=%u\n", node_id);
    
    // 配置SimpleNetwork接口
    Params net_params;
    net_params.insert("port_name", "network");
    net_params.insert("link_bw", link_bw);
    net_params.insert("input_buf_size", input_buf_size);
    net_params.insert("output_buf_size", output_buf_size);
    
    // 创建SimpleNetwork接口
    network = loadUserSubComponent<SimpleNetwork>("network", ComponentInfo::SHARE_NONE, net_params, this);
    
    if (!network) {
        output->fatal(CALL_INFO, -1, "无法创建SimpleNetwork接口\n");
    }
    
    output->verbose(CALL_INFO, 1, 0, "网络接口创建成功，带宽=%s\n", link_bw.c_str());
    
    // 注册统计信息
    stat_spikes_sent = registerStatistic<uint64_t>("spikes_sent");
    stat_spikes_received = registerStatistic<uint64_t>("spikes_received");
    stat_packets_sent = registerStatistic<uint64_t>("packets_sent");
    stat_packets_received = registerStatistic<uint64_t>("packets_received");
    
    output->verbose(CALL_INFO, 1, 0, "SnnNIC初始化完成\n");
}

SnnNIC::~SnnNIC()
{
    if (output) {
        delete output;
        output = nullptr;
    }
}

void SnnNIC::setSpikeHandler(SpikeHandler handler)
{
    spike_handler = handler;
    output->verbose(CALL_INFO, 2, 0, "设置脉冲处理器\n");
}

void SnnNIC::sendSpike(SpikeEvent* spike_event)
{
    if (!spike_event || !network) {
        output->verbose(CALL_INFO, 1, 0, "发送脉冲失败：参数无效\n");
        return;
    }
    
    uint32_t dest_node = spike_event->getDestinationNode();
    
    // 检查是否为本地消息
    if (dest_node == node_id) {
        output->verbose(CALL_INFO, 3, 0, "本地脉冲直接传递：神经元%u -> 神经元%u\n",
                       spike_event->getNeuronId(), spike_event->getDestinationNeuron());
        
        // 直接调用本地处理器
        if (spike_handler) {
            spike_handler(spike_event);
        }
        return;
    }
    
    // 创建网络请求
    SimpleNetwork::Request* req = createNetworkRequest(spike_event, dest_node);
    if (!req) {
        output->verbose(CALL_INFO, 1, 0, "创建网络请求失败\n");
        return;
    }
    
    // 发送网络请求
    bool success = network->send(req, 0); // 使用VN 0
    
    if (success) {
        spikes_sent_count++;
        packets_sent_count++;
        stat_spikes_sent->addData(1);
        stat_packets_sent->addData(1);
        
        output->verbose(CALL_INFO, 3, 0, "发送脉冲：节点%u -> 节点%u，神经元%u\n",
                       node_id, dest_node, spike_event->getNeuronId());
    } else {
        output->verbose(CALL_INFO, 1, 0, "网络发送失败，添加到待发送队列\n");
        pending_spikes.push(spike_event);
        delete req; // 清理请求对象
    }
}

void SnnNIC::setNodeId(uint32_t new_node_id)
{
    node_id = new_node_id;
    output->verbose(CALL_INFO, 1, 0, "更新节点ID为%u\n", node_id);
}

uint32_t SnnNIC::getNodeId() const
{
    return node_id;
}

std::string SnnNIC::getNetworkStatus() const
{
    std::stringstream ss;
    ss << "SnnNIC状态[节点" << node_id << "]: ";
    ss << "发送脉冲=" << spikes_sent_count;
    ss << ", 接收脉冲=" << spikes_received_count;
    ss << ", 发送包=" << packets_sent_count;
    ss << ", 接收包=" << packets_received_count;
    ss << ", 待发送=" << pending_spikes.size();
    return ss.str();
}

bool SnnNIC::handle(int vn, SimpleNetwork::Request* req)
{
    if (!req) {
        output->verbose(CALL_INFO, 1, 0, "接收到空网络请求\n");
        return false;
    }
    
    packets_received_count++;
    stat_packets_received->addData(1);
    
    output->verbose(CALL_INFO, 3, 0, "接收网络数据包：VN=%d，来源=%d，目标=%d\n",
                   vn, req->src, req->dest);
    
    // 解包SpikeEvent
    SpikeEvent* spike_event = extractSpikeEvent(req);
    
    if (spike_event) {
        spikes_received_count++;
        stat_spikes_received->addData(1);
        
        output->verbose(CALL_INFO, 3, 0, "解包脉冲事件：神经元%u -> 神经元%u\n",
                       spike_event->getNeuronId(), spike_event->getDestinationNeuron());
        
        // 调用脉冲处理器
        if (spike_handler) {
            spike_handler(spike_event);
        } else {
            output->verbose(CALL_INFO, 1, 0, "警告：未设置脉冲处理器，丢弃脉冲\n");
            delete spike_event;
        }
    } else {
        output->verbose(CALL_INFO, 1, 0, "解包脉冲事件失败\n");
    }
    
    // 清理网络请求
    delete req;
    
    return true; // 表示成功处理
}

void SnnNIC::init(unsigned int phase)
{
    output->verbose(CALL_INFO, 1, 0, "初始化阶段%u\n", phase);
    
    if (network) {
        network->init(phase);
    }
}

void SnnNIC::setup()
{
    output->verbose(CALL_INFO, 1, 0, "设置阶段\n");
    
    if (network) {
        network->setup();
    }
    
    output->verbose(CALL_INFO, 1, 0, "SnnNIC设置完成，节点ID=%u\n", node_id);
}

void SnnNIC::finish()
{
    output->verbose(CALL_INFO, 1, 0, "完成阶段\n");
    
    // 输出最终统计信息
    output->output("SnnNIC[节点%u]最终统计：\n", node_id);
    output->output("  发送脉冲: %lu\n", spikes_sent_count);
    output->output("  接收脉冲: %lu\n", spikes_received_count);
    output->output("  发送包: %lu\n", packets_sent_count);
    output->output("  接收包: %lu\n", packets_received_count);
    output->output("  待发送队列: %zu\n", pending_spikes.size());
    
    // 清理待发送队列
    while (!pending_spikes.empty()) {
        delete pending_spikes.front();
        pending_spikes.pop();
    }
    
    if (network) {
        network->finish();
    }
}

SimpleNetwork::Request* SnnNIC::createNetworkRequest(SpikeEvent* spike_event, uint32_t dest_node)
{
    if (!spike_event) {
        return nullptr;
    }
    
    // 计算数据包大小（固定大小的脉冲事件）
    size_t packet_size = sizeof(uint32_t) * 4 + sizeof(double) * 2; // 基本字段大小
    
    // 创建网络请求
    SimpleNetwork::Request* req = new SimpleNetwork::Request();
    req->src = node_id;
    req->dest = dest_node;
    req->size_in_bits = packet_size * 8;
    req->head = true;
    req->tail = true;
    req->vn = 0;
    
    // 序列化SpikeEvent到payload
    // 这里使用简单的内存拷贝方法
    // 在实际应用中，可能需要更复杂的序列化机制
    req->inspectPayload = nullptr;
    
    // 将SpikeEvent的关键信息编码到payload
    // 注意：这是一个简化的实现，实际中需要更完善的序列化
    struct SpikePayload {
        uint32_t neuron_id;
        uint32_t dest_neuron;
        uint32_t dest_node;
        uint32_t spike_time_cycles;
        double weight;
        double timestamp;
    };
    
    SpikePayload* payload = new SpikePayload();
    payload->neuron_id = spike_event->getNeuronId();
    payload->dest_neuron = spike_event->getDestinationNeuron();
    payload->dest_node = spike_event->getDestinationNode();
    payload->spike_time_cycles = spike_event->getSpikeTime();
    payload->weight = spike_event->getWeight();
    payload->timestamp = spike_event->getTimestamp();
    
    req->givePayload(payload);
    
    output->verbose(CALL_INFO, 4, 0, "创建网络请求：大小=%zu字节\n", packet_size);
    
    return req;
}

SpikeEvent* SnnNIC::extractSpikeEvent(SimpleNetwork::Request* req)
{
    if (!req || !req->inspectPayload) {
        return nullptr;
    }
    
    // 从payload中提取SpikeEvent信息
    struct SpikePayload {
        uint32_t neuron_id;
        uint32_t dest_neuron;
        uint32_t dest_node;
        uint32_t spike_time_cycles;
        double weight;
        double timestamp;
    };
    
    SpikePayload* payload = static_cast<SpikePayload*>(req->inspectPayload);
    
    // 重建SpikeEvent对象
    SpikeEvent* spike_event = new SpikeEvent(
        payload->neuron_id,
        payload->dest_neuron,
        payload->dest_node,
        payload->spike_time_cycles,
        payload->weight
    );
    
    spike_event->setTimestamp(payload->timestamp);
    
    output->verbose(CALL_INFO, 4, 0, "解包SpikeEvent：神经元%u -> 神经元%u\n",
                   payload->neuron_id, payload->dest_neuron);
    
    return spike_event;
}
