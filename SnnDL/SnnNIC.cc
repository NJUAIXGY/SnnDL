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
    
    // 新增：虚拟通道与批处理参数（默认保持兼容）
    virtual_channels_ = params.find<uint32_t>("virtual_channels", 2);
    vn_spike_data_ = params.find<uint32_t>("vn_spike_data", 0);
    vn_batch_data_ = params.find<uint32_t>("vn_batch_data", 1);
    enable_batching_ = params.find<bool>("enable_batching", false);
    batch_size_local_ = params.find<uint32_t>("batch_size_local", 8);
    batch_size_remote_ = params.find<uint32_t>("batch_size_remote", 32);
    batch_flush_window_ns_ = params.find<uint64_t>("batch_flush_window", 1000);
    total_nodes_ = params.find<uint32_t>("total_nodes", 16);
    
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
            net_params.insert("num_vns", std::to_string(virtual_channels_));  // 与路由器保持一致的虚拟网络数
            
            // 添加PortControl协议调试参数
            net_params.insert("job_id", "0");
            // ★ 修正：job_size应该是网络中的总节点数，而不是1
            net_params.insert("job_size", std::to_string(total_nodes_));
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
    stat_batches_sent = registerStatistic<uint64_t>("batches_sent");
    
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
        if (enable_batching_) {
            // 纳入批处理并按阈值尝试刷新
            tryBatchSpike(spike_event);
            uint32_t threshold = isNeighborNode(dest_node) ? batch_size_local_ : batch_size_remote_;
            auto it = batch_buckets_.find(dest_node);
            if (it != batch_buckets_.end() && it->second.size() >= threshold) {
                flushBatchToNode(dest_node);
            }
        } else {
            // 立即发送单条
            SimpleNetwork::Request* req = createNetworkRequest(spike_event, dest_node);
            if (!req) {
                output->verbose(CALL_INFO, 1, 0, "创建网络请求失败\n");
                return;
            }
            req->vn = static_cast<int>(vn_spike_data_);
            if (network->spaceToSend(req->vn, req->size_in_bits) && network->send(req, req->vn)) {
                spikes_sent_count++;
                packets_sent_count++;
                stat_spikes_sent->addData(1);
                stat_packets_sent->addData(1);
                output->verbose(CALL_INFO, 1, 0, "发送脉冲成功：节点%u -> 节点%u，神经元%u (vn=%u)\n",
                                node_id, dest_node, spike_event->getNeuronId(), vn_spike_data_);
            } else {
                output->verbose(CALL_INFO, 1, 0, "网络发送失败（空间不足），添加到待发送队列 (vn=%u)\n", vn_spike_data_);
                pending_spikes.push(spike_event);
                delete req;
            }
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
    
    // 批处理：仅当达到阈值时在回调中刷新，避免过早将批次拆成单条
    if (enable_batching_) {
        for (auto& kv : batch_buckets_) {
            uint32_t dest = kv.first;
            const auto& vec = kv.second;
            uint32_t threshold = isNeighborNode(dest) ? batch_size_local_ : batch_size_remote_;
            if (vec.size() >= threshold) {
                flushBatchToNode(dest);
            }
        }
    }

    // 处理待发送队列中的脉冲（如果有的话）
    while (!pending_spikes.empty() && network->spaceToSend(vn, 1)) {
        SpikeEvent* spike = pending_spikes.front();
        pending_spikes.pop();
        
        // 获取目标节点ID
        uint32_t dest_node = spike->getDestinationNode();
        SimpleNetwork::Request* req = createNetworkRequest(spike, dest_node);
        if (req) req->vn = vn; // 使用当前可用的vn
        // 使用相同的双重检查模式
        if (req && network->spaceToSend(vn, req->size_in_bits) && network->send(req, vn)) {
            output->verbose(CALL_INFO, 4, 0, "发送延迟的脉冲事件成功：节点%u -> 节点%u (vn=%d)\n", node_id, dest_node, vn);
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
    
    // 注册批处理时间窗时钟（若开启批处理且窗口>0）
    if (enable_batching_ && batch_flush_window_ns_ > 0) {
        std::string period = std::to_string(batch_flush_window_ns_) + "ns";
        // 使用Handler2避免弃用警告
        registerClock(period, new SST::Clock::Handler2<SnnNIC, &SnnNIC::flushClockTick>(this));
    }

    // output->verbose(CALL_INFO, 1, 0, "✅ SnnNIC[节点%u] 设置完成，模式=%s\n", 
    //                 node_id, use_direct_link ? "直接Link" : "SimpleNetwork");
}

void SnnNIC::finish()
{
    output->verbose(CALL_INFO, 1, 0, "完成阶段\n");
    // 在完成阶段刷新剩余批次，确保不丢数据
    if (enable_batching_) {
        flushAllBatches();
    }
    
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
    
    // 批量载荷类型
    class SpikeBatchPayload : public SST::Event {
    public:
        struct PackedSpike {
            uint32_t src_neuron_id;
            uint32_t dest_neuron_id;
            uint64_t timestamp;
            float weight;
            void serialize_order(SST::Core::Serialization::serializer& ser) {
                SST_SER(src_neuron_id);
                SST_SER(dest_neuron_id);
                SST_SER(timestamp);
                SST_SER(weight);
            }
        };
        std::vector<PackedSpike> batch;
        uint32_t source_node;
        uint32_t dest_node;
        uint64_t batch_timestamp;
        void serialize_order(SST::Core::Serialization::serializer& ser) override {
            Event::serialize_order(ser);
            SST_SER(batch);
            SST_SER(source_node);
            SST_SER(dest_node);
            SST_SER(batch_timestamp);
        }
        ImplementSerializable(SpikeBatchPayload)
    };

    // 注意：批量载荷的处理在 extractSpikeEvent() 中实现
    
    // 创建网络请求（单条）
    SimpleNetwork::Request* req = new SimpleNetwork::Request();
    req->dest = dest_node;
    req->src = node_id;
    // 默认使用配置的脉冲数据VN，调用方可覆盖
    req->vn = static_cast<int>(vn_spike_data_);
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
    
    // 从payload中提取SpikeEvent信息（单条或批量）
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
    // 重建单条SpikeEvent对象
    SpikeEvent* spike_event = new SpikeEvent();
    spike_event->neuron_id = payload->src_neuron_id;
    spike_event->setDestinationNeuron(payload->dest_neuron_id);
    spike_event->timestamp = payload->timestamp;
    spike_event->setWeight(payload->weight);
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

// ======== 批处理辅助实现 ========

void SnnNIC::tryBatchSpike(SpikeEvent* spike)
{
    if (!spike) return;
    uint32_t dest = spike->getDestinationNode();
    auto& vec = batch_buckets_[dest];
    vec.push_back(spike);
    // earliest ts 记录
    auto ts = spike->timestamp;
    auto it = batch_earliest_ts_.find(dest);
    if (it == batch_earliest_ts_.end() || ts < it->second) {
        batch_earliest_ts_[dest] = ts;
    }
}

void SnnNIC::flushBatchToNode(uint32_t dest_node)
{
    auto it = batch_buckets_.find(dest_node);
    if (it == batch_buckets_.end() || it->second.empty()) return;
    if (!network) return;

    // 定义本地批量载荷事件，与extract中的类型保持一致
    class SpikeBatchPayload : public SST::Event {
    public:
        struct PackedSpike {
            uint32_t src_neuron_id;
            uint32_t dest_neuron_id;
            uint64_t timestamp;
            float weight;
            void serialize_order(SST::Core::Serialization::serializer& ser) {
                SST_SER(src_neuron_id);
                SST_SER(dest_neuron_id);
                SST_SER(timestamp);
                SST_SER(weight);
            }
        };
        std::vector<PackedSpike> batch;
        uint32_t source_node;
        uint32_t dest_node;
        uint64_t batch_timestamp;
        void serialize_order(SST::Core::Serialization::serializer& ser) override {
            Event::serialize_order(ser);
            SST_SER(batch);
            SST_SER(source_node);
            SST_SER(dest_node);
            SST_SER(batch_timestamp);
        }
        ImplementSerializable(SpikeBatchPayload)
    };

    // 填充批量数据
    auto* payload = new SpikeBatchPayload();
    payload->batch.reserve(it->second.size());
    for (auto* s : it->second) {
        SpikeBatchPayload::PackedSpike ps{ s->neuron_id, s->getDestinationNeuron(), s->timestamp, s->getWeight() };
        payload->batch.push_back(ps);
    }
    payload->source_node = node_id;
    payload->dest_node = dest_node;
    payload->batch_timestamp = batch_earliest_ts_[dest_node];

    // 创建请求
    SimpleNetwork::Request* req = new SimpleNetwork::Request();
    req->dest = dest_node;
    req->src = node_id;
    req->vn = static_cast<int>(vn_batch_data_);
    // 大致估算消息大小
    size_t per = sizeof(uint32_t)*2 + sizeof(uint64_t) + sizeof(float);
    size_t bytes = sizeof(uint32_t)*3 + sizeof(uint64_t) + per * payload->batch.size();
    req->size_in_bits = bytes * 8;
    req->head = true;
    req->tail = true;
    req->allow_adaptive = true;
    req->givePayload(payload);

    if (network->spaceToSend(req->vn, req->size_in_bits) && network->send(req, req->vn)) {
        packets_sent_count++;
        stat_packets_sent->addData(1);
        if (stat_spikes_sent) stat_spikes_sent->addData(static_cast<uint64_t>(payload->batch.size()));
        if (stat_batches_sent) stat_batches_sent->addData(1);
        output->verbose(CALL_INFO, 2, 0, "批量发送：%zu 条 -> 节点%u (vn=%u)\n", payload->batch.size(), dest_node, vn_batch_data_);
        it->second.clear();
        batch_earliest_ts_.erase(dest_node);
    } else {
        // 保持桶以便后续重试
        delete req;
    }
}

void SnnNIC::flushAllBatches()
{
    std::vector<uint32_t> keys;
    keys.reserve(batch_buckets_.size());
    for (auto& kv : batch_buckets_) if (!kv.second.empty()) keys.push_back(kv.first);
    for (auto d : keys) flushBatchToNode(d);
}

bool SnnNIC::isNeighborNode(uint32_t dest_node) const
{
    return manhattanDistance(node_id, dest_node) <= 1;
}

uint32_t SnnNIC::manhattanDistance(uint32_t src_node, uint32_t dst_node) const
{
    if (total_nodes_ == 0) return 0;
    // 估算方阵维度
    uint32_t side = 1;
    while (side * side < total_nodes_) ++side;
    uint32_t sx = src_node % side, sy = src_node / side;
    uint32_t dx = dst_node % side, dy = dst_node / side;
    int dxm = (int)sx - (int)dx;
    int dym = (int)sy - (int)dy;
    return (uint32_t)((dxm<0?-dxm:dxm) + (dym<0?-dym:dym));
}

bool SnnNIC::flushClockTick(SST::Cycle_t /*currentCycle*/)
{
    if (enable_batching_) {
        flushAllBatches();
    }
    return false; // 持续触发
}
