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
      direct_link(nullptr),
      spike_handler(nullptr),
      spikes_sent_count(0),
      spikes_received_count(0),
      packets_sent_count(0),
      packets_received_count(0),
      use_direct_link(false)
{
    // 获取参数
    node_id = params.find<uint32_t>("node_id", 0);
    link_bw = params.find<std::string>("link_bw", "40GiB/s");
    input_buf_size = params.find<std::string>("input_buf_size", "1KiB");
    output_buf_size = params.find<std::string>("output_buf_size", "1KiB");
    use_direct_link = params.find<bool>("use_direct_link", true);  // 默认使用直接Link模式
    
    int verbose = params.find<int>("verbose", 0);
    
    // 初始化日志输出
    output = new Output("SnnNIC[@p:@l]: ", verbose, 0, Output::STDOUT);
    
    // output->verbose(CALL_INFO, 1, 0, "初始化SnnNIC组件，节点ID=%u，直接链接模式=%s\n", 
    //                 node_id, use_direct_link ? "是" : "否");
    
    if (use_direct_link) {
        // 使用直接Link模式
        direct_link = configureLink("network", 
            new Event::Handler2<SnnNIC,&SnnNIC::handleDirectSpikeEvent>(this));
        
        if (direct_link) {
            // output->verbose(CALL_INFO, 1, 0, "直接Link网络接口创建成功\n");
        } else {
            // output->verbose(CALL_INFO, 1, 0, "警告：直接Link创建失败\n");
        }
    } else {
        // 使用SimpleNetwork模式 - 参考MemNIC的成功实现
        // output->verbose(CALL_INFO, 1, 0, "尝试加载网络接口...\n");
        
        // 首先尝试加载用户定义的网络接口 (推荐方式)
        network = loadUserSubComponent<SimpleNetwork>("linkcontrol", ComponentInfo::SHARE_NONE, 1);
        
        if (!network) {
            // 如果没有用户定义的接口，创建默认的merlin.linkcontrol
            // output->verbose(CALL_INFO, 1, 0, "未找到用户定义的linkcontrol，创建默认merlin.linkcontrol\n");
            
            Params net_params;
            net_params.insert("port_name", params.find<std::string>("port_name", "network"));
            net_params.insert("link_bw", link_bw);
            net_params.insert("input_buf_size", input_buf_size);
            net_params.insert("output_buf_size", output_buf_size);
            net_params.insert("num_vns", "2");  // 与路由器保持一致的虚拟网络数
            
            // 添加PortControl协议调试参数
            net_params.insert("job_id", "0");
            // ★ 修正：job_size应该是网络中的总节点数，而不是1
            uint32_t total_nodes = params.find<uint32_t>("total_nodes", 16);  // 从外部参数获取或默认16
            net_params.insert("job_size", std::to_string(total_nodes));
            net_params.insert("logical_nid", std::to_string(node_id));
            
            // output->verbose(CALL_INFO, 1, 0, "🔧 LinkControl参数: port_name=%s, job_id=0, job_size=%u, logical_nid=%u\n",
            //                params.find<std::string>("port_name", "network").c_str(), total_nodes, node_id);
            
            // 使用与MemNIC相同的标志和参数
            network = loadAnonymousSubComponent<SimpleNetwork>("merlin.linkcontrol", "linkcontrol", 0, 
                ComponentInfo::SHARE_PORTS | ComponentInfo::INSERT_STATS, net_params, 1);
        }
        
        if (network) {
            // output->verbose(CALL_INFO, 1, 0, "网络接口创建成功，设置回调处理器\n");
            
            // 设置网络回调处理器 - 只设置接收回调，发送回调是可选的
            network->setNotifyOnReceive(new SimpleNetwork::Handler2<SnnNIC,&SnnNIC::handleIncoming>(this));
            // 启用发送可用回调以便处理待发送队列
            network->setNotifyOnSend(new SimpleNetwork::Handler2<SnnNIC,&SnnNIC::spaceAvailable>(this));
        } else {
            // output->fatal(CALL_INFO, -1, "错误：无法创建网络接口，网络通信将不可用\n");
        }
    }
    
    // output->verbose(CALL_INFO, 1, 0, "SnnNIC初始化完成\n");
    
    // 注册统计信息
    stat_spikes_sent = registerStatistic<uint64_t>("spikes_sent");
    stat_spikes_received = registerStatistic<uint64_t>("spikes_received");
    stat_packets_sent = registerStatistic<uint64_t>("packets_sent");
    stat_packets_received = registerStatistic<uint64_t>("packets_received");
    
    // output->verbose(CALL_INFO, 1, 0, "SnnNIC初始化完成\n");
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
    if (!spike_event) {
        output->verbose(CALL_INFO, 1, 0, "发送脉冲失败：参数无效\n");
        return;
    }
    
    uint32_t dest_node = spike_event->getDestinationNode();
    
    // 检查是否为本地消息：只有当神经元ID和节点ID都相同时才是本地消息
    uint32_t source_neuron = spike_event->getNeuronId();
    uint32_t dest_neuron = spike_event->getDestinationNeuron();
    if (dest_node == node_id && source_neuron == dest_neuron) {
        output->verbose(CALL_INFO, 3, 0, "本地脉冲直接传递：神经元%u -> 神经元%u (同节点同神经元)\n",
                       source_neuron, dest_neuron);

        // 直接调用本地处理器
        if (spike_handler) {
            spike_handler(spike_event);
        }
        return;
    }
    
    if (use_direct_link && direct_link) {
        // 使用直接Link模式发送脉冲
        // output->verbose(CALL_INFO, 3, 0, "通过直接Link发送脉冲：节点%u -> 节点%u，神经元%u\n",
        //                node_id, dest_node, spike_event->getNeuronId());
        
        // 创建包装的SpikeEvent用于网络传输
        SpikeEvent* network_spike = new SpikeEvent(*spike_event);  // 复制构造
        
        // 直接通过Link发送
        direct_link->send(network_spike);
        
        spikes_sent_count++;
        packets_sent_count++;
        stat_spikes_sent->addData(1);
        stat_packets_sent->addData(1);
        
        // output->verbose(CALL_INFO, 3, 0, "直接Link发送成功\n");
        
    } else if (!use_direct_link && network) {
        // 使用SimpleNetwork模式发送脉冲
        
        // 创建网络请求
        SimpleNetwork::Request* req = createNetworkRequest(spike_event, dest_node);
        if (!req) {
            output->verbose(CALL_INFO, 1, 0, "创建网络请求失败\n");
            return;
        }
        
        // 双重检查：确保网络接口仍然有效
        if (!network) {
            output->verbose(CALL_INFO, 1, 0, "警告：网络接口在发送过程中变为空\n");
            delete req;
            return;
        }
        
        // 按照MemNIC模式：先检查空间，再发送
        if (network->spaceToSend(0, req->size_in_bits) && network->send(req, 0)) {
            // 发送成功
            spikes_sent_count++;
            packets_sent_count++;
            stat_spikes_sent->addData(1);
            stat_packets_sent->addData(1);
            
            output->verbose(CALL_INFO, 1, 0, "发送脉冲成功：节点%u -> 节点%u，神经元%u (vn=0)\n",
                           node_id, dest_node, spike_event->getNeuronId());
        } else {
            // 发送失败 - 添加到待发送队列，稍后重试
            output->verbose(CALL_INFO, 1, 0, "网络发送失败（空间不足），添加到待发送队列 (vn=0)\n");
            pending_spikes.push(spike_event);
            delete req; // 清理请求对象，因为没有发送成功
        }
    } else {
        output->verbose(CALL_INFO, 1, 0, "发送脉冲失败：无可用网络接口\n");
    }
}

void SnnNIC::setNodeId(uint32_t new_node_id)
{
    node_id = new_node_id;
    // output->verbose(CALL_INFO, 1, 0, "更新节点ID为%u\n", node_id);
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

bool SnnNIC::handleIncoming(int vn)
{
    SimpleNetwork::Request* req = network->recv(vn);
    if (!req) {
        return true; // 继续处理
    }
    
    packets_received_count++;  // 更新内部计数器
    stat_packets_received->addData(1);
    
    output->verbose(CALL_INFO, 3, 0, "接收网络数据包：VN=%d，来源=%ld，目标=%ld\n",
                   vn, req->src, req->dest);
    
    // 提取并处理脉冲事件
    SpikeEvent* spike_event = extractSpikeEvent(req);
    if (spike_event && spike_handler) {
        output->verbose(CALL_INFO, 4, 0, "提取到脉冲事件：源神经元=%u，目标神经元=%u\n",
                       spike_event->neuron_id, spike_event->getDestinationNeuron());
        
        stat_spikes_received->addData(1);
        spike_handler(spike_event);
    }
    
    delete req;
    return true; // 继续处理更多数据包
}

bool SnnNIC::spaceAvailable(int vn)
{
    output->verbose(CALL_INFO, 5, 0, "网络发送空间可用：VN=%d\n", vn);
    
    // 检查网络接口是否有效
    if (!network) {
        output->verbose(CALL_INFO, 1, 0, "警告：网络接口为空，跳过待发送队列处理\n");
        return true;
    }
    
    // 处理待发送队列中的脉冲（如果有的话）
    while (!pending_spikes.empty() && network->spaceToSend(vn, 1)) {
        SpikeEvent* spike = pending_spikes.front();
        pending_spikes.pop();
        
        // 获取目标节点ID
        uint32_t dest_node = spike->getDestinationNode();
        SimpleNetwork::Request* req = createNetworkRequest(spike, dest_node);
        
        // 使用相同的双重检查模式
        if (req && network->spaceToSend(vn, req->size_in_bits) && network->send(req, vn)) {
            output->verbose(CALL_INFO, 4, 0, "发送延迟的脉冲事件成功：节点%u -> 节点%u\n", node_id, dest_node);
            stat_spikes_sent->addData(1);
            stat_packets_sent->addData(1);
        } else {
            // 如果仍然无法发送，重新加入队列
            pending_spikes.push(spike);
            if (req) delete req; // 清理请求对象
            break;
        }
    }
    
    return true; // 继续处理
}

void SnnNIC::init(unsigned int phase)
{
    // output->verbose(CALL_INFO, 1, 0, "🔄 SnnNIC[节点%u] 初始化阶段%u开始\n", node_id, phase);
    
    if (!use_direct_link && network) {
        // output->verbose(CALL_INFO, 1, 0, "🔧 调用LinkControl.init(%u)\n", phase);
        
        try {
            // 只调用网络接口的init，不进行复杂的初始化数据交换
            network->init(phase);
            // output->verbose(CALL_INFO, 1, 0, "✅ LinkControl.init(%u)成功完成\n", phase);
        } catch (const std::exception& e) {
            output->verbose(CALL_INFO, 0, 0, "❌ LinkControl.init(%u)异常: %s\n", phase, e.what());
            throw;
        }
        
        // output->verbose(CALL_INFO, 1, 0, "✅ SnnNIC[节点%u] 网络接口初始化完成，阶段%u\n", node_id, phase);
    } else {
        output->verbose(CALL_INFO, 2, 0, "⏭️ SnnNIC[节点%u] 跳过网络接口初始化 (direct_link=%s)\n", 
                       node_id, use_direct_link ? "true" : "false");
    }
}

void SnnNIC::setup()
{
    // output->verbose(CALL_INFO, 1, 0, "🔧 SnnNIC[节点%u] 设置阶段开始\n", node_id);
    
    if (!use_direct_link && network) {
        // output->verbose(CALL_INFO, 1, 0, "🔧 调用LinkControl.setup()\n");
        
        try {
            network->setup();
            // output->verbose(CALL_INFO, 1, 0, "✅ LinkControl.setup()成功完成\n");
        } catch (const std::exception& e) {
            output->verbose(CALL_INFO, 0, 0, "❌ LinkControl.setup()异常: %s\n", e.what());
            throw;
        }
    }
    
    // output->verbose(CALL_INFO, 1, 0, "✅ SnnNIC[节点%u] 设置完成，模式=%s\n", 
    //                 node_id, use_direct_link ? "直接Link" : "SimpleNetwork");
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
    output->output("  网络模式: %s\n", use_direct_link ? "直接Link" : "SimpleNetwork");
    
    // 清理待发送队列
    while (!pending_spikes.empty()) {
        delete pending_spikes.front();
        pending_spikes.pop();
    }
    
    if (!use_direct_link && network) {
        network->finish();
    }
}

SimpleNetwork::Request* SnnNIC::createNetworkRequest(SpikeEvent* spike_event, uint32_t dest_node)
{
    if (!spike_event) {
        return nullptr;
    }
    
    // 创建可序列化的载荷事件
    class SpikePayload : public SST::Event {
    public:
        uint32_t src_neuron_id;
        uint32_t dest_neuron_id;
        uint64_t timestamp;
        float weight;
        
        SpikePayload() : SST::Event(), src_neuron_id(0), dest_neuron_id(0), timestamp(0), weight(0.0f) {}
        
        SpikePayload(const SpikeEvent* spike) : SST::Event()
        {
            src_neuron_id = spike->neuron_id;
            dest_neuron_id = spike->getDestinationNeuron();
            timestamp = spike->timestamp;
            weight = spike->getWeight();
        }
        
        void serialize_order(SST::Core::Serialization::serializer& ser) override {
            Event::serialize_order(ser);
            SST_SER(src_neuron_id);
            SST_SER(dest_neuron_id);
            SST_SER(timestamp);
            SST_SER(weight);
        }
        
        ImplementSerializable(SpikePayload)
    };
    
    // 创建网络请求
    SimpleNetwork::Request* req = new SimpleNetwork::Request();
    req->dest = dest_node;
    req->src = node_id;
    req->vn = 0; // 使用虚拟网络0
    req->size_in_bits = sizeof(SpikePayload) * 8;
    req->head = true;
    req->tail = true;
    req->allow_adaptive = true;
    
    // 创建载荷并赋给请求
    SpikePayload* payload = new SpikePayload(spike_event);
    req->givePayload(payload);
    
    output->verbose(CALL_INFO, 4, 0, "创建网络请求：源=%ld，目标=%ld，大小=%zu bits\n",
                   req->src, req->dest, req->size_in_bits);
    
    return req;
}

SpikeEvent* SnnNIC::extractSpikeEvent(SimpleNetwork::Request* req)
{
    if (!req || !req->inspectPayload()) {
        return nullptr;
    }
    
    // 从payload中提取SpikeEvent信息
    class SpikePayload : public SST::Event {
    public:
        uint32_t src_neuron_id;
        uint32_t dest_neuron_id;
        uint64_t timestamp;
        float weight;
        
        SpikePayload() : SST::Event(), src_neuron_id(0), dest_neuron_id(0), timestamp(0), weight(0.0f) {}
        
        void serialize_order(SST::Core::Serialization::serializer& ser) override {
            Event::serialize_order(ser);
            SST_SER(src_neuron_id);
            SST_SER(dest_neuron_id);
            SST_SER(timestamp);
            SST_SER(weight);
        }
        
        ImplementSerializable(SpikePayload)
    };
    
    SpikePayload* payload = static_cast<SpikePayload*>(req->inspectPayload());
    
    // 重建SpikeEvent对象
    SpikeEvent* spike_event = new SpikeEvent();
    spike_event->neuron_id = payload->src_neuron_id;
    spike_event->setDestinationNeuron(payload->dest_neuron_id);
    spike_event->timestamp = payload->timestamp;
    spike_event->setWeight(payload->weight);
    // 设置目标节点，确保接收端能够正确识别本地投递
    spike_event->setDestinationNode(static_cast<uint32_t>(req->dest));
    
    output->verbose(CALL_INFO, 4, 0, "解包SpikeEvent：神经元%u -> 神经元%u\n",
                   payload->src_neuron_id, payload->dest_neuron_id);
    
    return spike_event;
}

void SnnNIC::handleDirectSpikeEvent(SST::Event* event)
{
    if (!event) {
        return;
    }
    
    // 直接转换为SpikeEvent
    SpikeEvent* spike_event = static_cast<SpikeEvent*>(event);
    
    output->verbose(CALL_INFO, 3, 0, "接收直接Link脉冲：源神经元=%u，目标神经元=%u\n",
                   spike_event->neuron_id, spike_event->getDestinationNeuron());
    
    if (spike_handler) {
        spikes_received_count++;
        packets_received_count++;
        stat_spikes_received->addData(1);
        stat_packets_received->addData(1);
        
        // 调用脉冲处理器
        spike_handler(spike_event);
    } else {
        output->verbose(CALL_INFO, 1, 0, "警告：未设置脉冲处理器，丢弃接收的脉冲\n");
        delete spike_event;
    }
}
