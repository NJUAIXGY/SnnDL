// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// SnnNIC.cc: SNN网络接口控制器实现文件
//

#include "SnnNIC.h"
#include <sst/core/serialization/serialize.h>
#include <sstream>

using namespace SST;
using namespace SST::SnnDL;
using namespace SST::Interfaces;

// Lightweight logging helpers (file-local)
#ifndef SNNDL_LOGPTR
#define SNNDL_LOGPTR(ptr, lvl, ...) do { if (ptr) (ptr)->verbose(CALL_INFO, (lvl), 0, __VA_ARGS__); } while(0)
#endif
#ifndef NIC_LOG
// Usage: NIC_LOG(level, "fmt...", args...)
// Note: Do NOT pass CALL_INFO here; SNNDL_LOGPTR injects it.
#define NIC_LOG(lvl, ...) SNNDL_LOGPTR(output, (lvl), __VA_ARGS__)
#endif

// Unify network payload types to avoid repeated class definitions in functions
namespace {
class NetSpikePayload : public SST::Event {
public:
    uint32_t src_neuron_id{};
    uint32_t dest_neuron_id{};
    uint64_t timestamp{};
    double   weight{};
    NetSpikePayload() : SST::Event() {}
    explicit NetSpikePayload(const SST::SnnDL::SpikeEvent* spike) : SST::Event()
    {
        src_neuron_id  = spike->neuron_id;
        dest_neuron_id = spike->getDestinationNeuron();
        timestamp      = spike->timestamp;
        weight         = spike->getWeight();
    }
    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(src_neuron_id);
        SST_SER(dest_neuron_id);
        SST_SER(timestamp);
        SST_SER(weight);
    }
    ImplementSerializable(NetSpikePayload)
};

class NetSpikeBatchPayload : public SST::Event {
public:
    struct PackedSpike {
        uint32_t src_neuron_id{};
        uint32_t dest_neuron_id{};
        uint64_t timestamp{};
        double   weight{};
        void serialize_order(SST::Core::Serialization::serializer& ser) {
            SST_SER(src_neuron_id);
            SST_SER(dest_neuron_id);
            SST_SER(timestamp);
            SST_SER(weight);
        }
    };
    std::vector<PackedSpike> batch;
    uint32_t source_node{};
    uint32_t dest_node{};
    uint64_t batch_timestamp{};
    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(batch);
        SST_SER(source_node);
        SST_SER(dest_node);
        SST_SER(batch_timestamp);
    }
    ImplementSerializable(NetSpikeBatchPayload)
};

class NetInterRankBatchPayload : public SST::Event {
public:
    struct PackedSpike {
        uint32_t src_neuron_id{};
        uint32_t dest_neuron_id{};   // 目标神经元
        uint32_t final_dest_node{};  // 最终目标节点（rank内展开）
        uint64_t timestamp{};
        double   weight{};
        void serialize_order(SST::Core::Serialization::serializer& ser) {
            SST_SER(src_neuron_id);
            SST_SER(dest_neuron_id);
            SST_SER(final_dest_node);
            SST_SER(timestamp);
            SST_SER(weight);
        }
    };
    std::vector<PackedSpike> batch;  // 属于同一目标rank的一批脉冲
    uint32_t source_node{};
    uint32_t gateway_node{};         // 目标rank的网关节点（本消息的目的地）
    uint64_t batch_timestamp{};
    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(batch);
        SST_SER(source_node);
        SST_SER(gateway_node);
        SST_SER(batch_timestamp);
    }
    ImplementSerializable(NetInterRankBatchPayload)
};
} // anonymous namespace

// ---- 调试辅助：打印门控状态快照 ----
void SnnNIC::logGatingSnapshot(const char* ctx) const {
    if (!output) return;
    int isInit = (network != nullptr) ? (int)network->isNetworkInitialized() : 0;
    std::stringstream vns;
    vns << "[";
    for (size_t i = 0; i < vn_ready_.size(); ++i) {
        vns << (vn_ready_[i] ? '1' : '0');
        if (i + 1 < vn_ready_.size()) vns << ',';
    }
    vns << "]";
    NIC_LOG(1, "GATING(%s): ready=%d, isInit=%d, link_ready=%d, eff_vns=%u, vn_ready=%s\n",
            ctx ? ctx : "?", (int)network_ready_, isInit, (int)link_ready_, effective_num_vns_, vns.str().c_str());
}

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
    use_direct_link = params.find<bool>("use_direct_link", false);  // 默认禁用直连，统一走SimpleNetwork
    // 禁用direct_link实现，统一走SimpleNetwork，避免脚本依赖非标准直连模式
    if (use_direct_link) {
        NIC_LOG(1, "direct_link 模式已禁用，回退到 SimpleNetwork");
        use_direct_link = false;
    }
    
    // 新增：虚拟通道与批处理参数（默认保持兼容）
    // 将默认虚拟通道数改为1以与常见路由配置对齐，避免VN不一致导致的崩溃
    virtual_channels_ = params.find<uint32_t>("virtual_channels", 1);
    network_num_vns_ = params.find<uint32_t>("network_num_vns", 0);
    auto_vn_fallback_ = params.find<bool>("auto_vn_fallback", true);
    effective_num_vns_ = (network_num_vns_ > 0) ? std::min(virtual_channels_, network_num_vns_) : virtual_channels_;
    // 降低默认日志级别，避免常规运行下产生噪声
    NIC_LOG(2, "VN配置: virtual_channels=%u, network_num_vns=%u, effective_num_vns=%u\n",
                 virtual_channels_, network_num_vns_, effective_num_vns_);
    vn_spike_data_ = params.find<uint32_t>("vn_spike_data", 0);
    vn_batch_data_ = params.find<uint32_t>("vn_batch_data", 1);
    // 强制单VN运行，移除VN相关不稳定因素
    virtual_channels_ = 1;
    network_num_vns_ = 1;
    auto_vn_fallback_ = false;
    effective_num_vns_ = 1;
    vn_spike_data_ = 0;
    vn_batch_data_ = 0;
    NIC_LOG(2, "VN配置: 已强制单VN (vn=0)\n");
    enable_batching_ = params.find<bool>("enable_batching", false);
    flush_on_credit_ = params.find<bool>("flush_on_credit", true);
    probe_vn_on_setup_ = params.find<bool>("probe_vn_on_setup", false);
    batch_size_local_ = params.find<uint32_t>("batch_size_local", 8);
    batch_size_remote_ = params.find<uint32_t>("batch_size_remote", 32);
    batch_flush_window_ns_ = params.find<uint64_t>("batch_flush_window", 1000);
    total_nodes_ = params.find<uint32_t>("total_nodes", 16);
    // 控制VN（用于门控等控制事件）
    vn_control_ = params.find<uint32_t>("vn_control", 1);

    // 跨Rank代理聚合参数（默认关闭）
    enable_inter_rank_batching_ = params.find<bool>("enable_inter_rank_batching", false);
    inter_rank_batch_window_ns_ = params.find<uint64_t>("inter_rank_batch_window", 0);
    nodes_per_rank_ = params.find<uint32_t>("nodes_per_rank", 0);
    // 强制禁用跨Rank聚合
    enable_inter_rank_batching_ = false;
    inter_rank_batch_window_ns_ = 0;
    nodes_per_rank_ = 0;
    
    int verbose = params.find<int>("verbose", 0);
    
    // 初始化日志输出
    output = new Output("SnnNIC[@p:@l]: ", verbose, 0, Output::STDOUT);
    
    //                 node_id, use_direct_link ? "是" : "否");
    // 初始化每个VN的就绪标志
    vn_ready_.assign(effective_num_vns_, false);
    if (use_direct_link) {
        // 使用直接Link模式
        direct_link = configureLink("network", 
            new Event::Handler2<SnnNIC,&SnnNIC::handleDirectSpikeEvent>(this));
        
        if (direct_link) {
        } else {
        }
    } else {
        // 使用SimpleNetwork模式 - 参考MemNIC的成功实现
        
        // 首先尝试加载用户定义的网络接口 (推荐方式)
        // 将请求的VN数量传递给LinkControl构造函数（否则默认为1会导致VN>0崩溃）
        network = loadUserSubComponent<SimpleNetwork>("linkcontrol", ComponentInfo::SHARE_NONE, static_cast<int>(effective_num_vns_));
        
        if (!network) {
            // 如果没有用户定义的接口，创建默认的merlin.linkcontrol
            
            Params net_params;
            net_params.insert("port_name", params.find<std::string>("port_name", "network"));
            net_params.insert("link_bw", link_bw);
            net_params.insert("input_buf_size", input_buf_size);
            net_params.insert("output_buf_size", output_buf_size);
            // 强制单VN
            net_params.insert("num_vns", std::to_string(1));
            // 明确指定VN重映射为恒等映射，避免部分环境下握手不一致
            // 不显式设置 vn_remap，让端点与路由通过初始化协议协商映射，避免解析格式差异
            // 降低默认日志级别
            NIC_LOG(2, "LinkControl参数: num_vns=%u, port_name=%s\n", effective_num_vns_,
                         net_params.find<std::string>("port_name", "network").c_str());
            
            // 添加PortControl协议调试参数
            net_params.insert("job_id", "0");
            // ★ 修正：job_size应该是网络中的总节点数，而不是1
            net_params.insert("job_size", std::to_string(total_nodes_));
            net_params.insert("logical_nid", std::to_string(node_id));
            
            //                params.find<std::string>("port_name", "network").c_str(), total_nodes, node_id);
            
            // 使用与MemNIC相同的标志和参数
            network = loadAnonymousSubComponent<SimpleNetwork>("merlin.linkcontrol", "linkcontrol", 0, 
                ComponentInfo::SHARE_PORTS | ComponentInfo::INSERT_STATS, net_params, static_cast<int>(effective_num_vns_));
        }
        
        if (network) {
            
            // 设置网络回调处理器 - 只设置接收回调，发送回调是可选的
            network->setNotifyOnReceive(new SimpleNetwork::Handler2<SnnNIC,&SnnNIC::handleIncoming>(this));
            // 启用发送可用回调以便处理待发送队列
            network->setNotifyOnSend(new SimpleNetwork::Handler2<SnnNIC,&SnnNIC::spaceAvailable>(this));
        } else {
            // output->fatal(CALL_INFO, -1, "错误：无法创建网络接口，网络通信将不可用\n");
        }
    }
    
    
    // 注册统计信息
    stat_spikes_sent = registerStatistic<uint64_t>("spikes_sent");
    stat_spikes_received = registerStatistic<uint64_t>("spikes_received");
    stat_packets_sent = registerStatistic<uint64_t>("packets_sent");
    stat_packets_received = registerStatistic<uint64_t>("packets_received");
    stat_batches_sent = registerStatistic<uint64_t>("batches_sent");
    stat_ir_batches_sent = registerStatistic<uint64_t>("inter_rank_batches_sent");
    
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
    NIC_LOG(2, "设置脉冲处理器\n");
}

void SnnNIC::sendSpike(SpikeEvent* spike_event)
{
    if (!spike_event) {
        NIC_LOG(1, "发送脉冲失败：参数无效\n");
        return;
    }
    
    uint32_t dest_node = spike_event->getDestinationNode();
    
    // 检查是否为本地消息：只有当神经元ID和节点ID都相同时才是本地消息
    uint32_t source_neuron = spike_event->getNeuronId();
    uint32_t dest_neuron = spike_event->getDestinationNeuron();
    if (dest_node == node_id && source_neuron == dest_neuron) {
        NIC_LOG(3, "本地脉冲直接传递：神经元%u -> 神经元%u (同节点同神经元)\n",
                       source_neuron, dest_neuron);

        // 直接调用本地处理器
        if (spike_handler) {
            spike_handler(spike_event);
        }
        return;
    }
    
    if (use_direct_link && direct_link) {
        // 使用直接Link模式发送脉冲
        //                node_id, dest_node, spike_event->getNeuronId());
        
        // 创建包装的SpikeEvent用于网络传输
        SpikeEvent* network_spike = new SpikeEvent(*spike_event);  // 复制构造
        
        // 直接通过Link发送
        direct_link->send(network_spike);
        
        spikes_sent_count++;
        packets_sent_count++;
        stat_spikes_sent->addData(1);
        stat_packets_sent->addData(1);
        
        
    } else if (!use_direct_link && network) {
        // 跨Rank代理聚合（可选）：依据简化映射 nodes_per_rank_
        if (enable_inter_rank_batching_ && nodes_per_rank_ > 0) {
            uint32_t dest_rank = computeRankForNode(dest_node);
            uint32_t self_rank = computeRankForNode(node_id);
            if (dest_rank != self_rank) {
                tryInterRankBatch(spike_event);
                // 窗口驱动或credit回调将刷新，此处直接返回避免重复进入普通路径
                return;
            }
        }

        // 若网络尚未完成初始化或链路未就绪，先缓存，等待初始化完成后再发送
        if (!network_ready_ || !network->isNetworkInitialized()) {
            if (enable_batching_) {
                tryBatchSpike(spike_event);
            } else {
                pending_spikes.push(spike_event);
            }
            NIC_LOG(2, "网络未就绪(ready=%d, isInit=%d)，暂存脉冲(目的节点=%u)\n",
                   (int)network_ready_, network ? (int)network->isNetworkInitialized() : 0, dest_node);
            if (!gating_logged_send_) {
                gating_logged_send_ = true;
                logGatingSnapshot("sendSpike");
            }
            return;
        }
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
            // 立即发送单条（单VN=0）
            SimpleNetwork::Request* req = createNetworkRequest(spike_event, dest_node);
            if (!req) {
                NIC_LOG(1, "创建网络请求失败\n");
                return;
            }
            req->vn = 0;
            bool sent = false;
            if (network->spaceToSend(0, req->size_in_bits)) {
                sent = network->send(req, 0);
            }
            if (sent) {
                spikes_sent_count++;
                packets_sent_count++;
                stat_spikes_sent->addData(1);
                stat_packets_sent->addData(1);
                NIC_LOG(1, "发送脉冲成功：节点%u -> 节点%u，神经元%u (vn=%u)\n",
                                node_id, dest_node, spike_event->getNeuronId(), 0u);
            } else {
                NIC_LOG(1, "网络发送失败（空间不足），添加到待发送队列 (vn=0)\n");
                pending_spikes.push(spike_event);
                delete req;
            }
        }
    } else {
        NIC_LOG(1, "发送脉冲失败：无可用网络接口\n");
    }
}

void SnnNIC::setNodeId(uint32_t new_node_id)
{
    node_id = new_node_id;
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
    
    NIC_LOG(3, "接收网络数据包：VN=%d，来源=%ld，目标=%ld\n",
                   vn, req->src, req->dest);
    // 判断payload类型
    SST::Event* any = req->inspectPayload();
    if (any) {
        // 控制事件（门控等）
        if (auto* gd = dynamic_cast<GatingDecisionEvent*>(any)) {
            if (control_handler_) control_handler_(gd);
        }
        // 跨Rank代理批量：在网关节点展开并分发
        else if (auto* ir = dynamic_cast<NetInterRankBatchPayload*>(any)) {
            NIC_LOG(2, "IR-Recv: gateway=%u, batch=%zu\n", node_id, ir->batch.size());
            for (auto& ps : ir->batch) {
                // 在目标rank内部转发：若到我则直接投递，否则按普通网络路径转发
                auto* spike_event = new SpikeEvent(ps.src_neuron_id, ps.dest_neuron_id,
                                                  static_cast<uint32_t>(ps.final_dest_node), (float)ps.weight, ps.timestamp);
                if (ps.final_dest_node == node_id) {
                    // 本地递交
                    if (spike_handler) {
                        stat_spikes_received->addData(1);
                        spike_handler(spike_event);
                    } else {
                        delete spike_event;
                    }
                } else {
                    // rank内部再转发
                    sendSpike(spike_event);
                }
            }
        }
        // 批量脉冲
        else if (auto* batch = dynamic_cast<NetSpikeBatchPayload*>(any)) {
            for (auto& ps : batch->batch) {
                auto* spike_event = new SpikeEvent(ps.src_neuron_id, ps.dest_neuron_id,
                                                  static_cast<uint32_t>(req->dest), (float)ps.weight, ps.timestamp);
                if (spike_handler) {
                    stat_spikes_received->addData(1);
                    spike_handler(spike_event);
                } else {
                    delete spike_event;
                }
            }
        }
        // 单条脉冲
        else {
            // 提取并处理脉冲事件
            SpikeEvent* spike_event = extractSpikeEvent(req);
            if (spike_event && spike_handler) {
                NIC_LOG(4, "提取到脉冲事件：源神经元=%u，目标神经元=%u\n",
                               spike_event->neuron_id, spike_event->getDestinationNeuron());
                stat_spikes_received->addData(1);
                spike_handler(spike_event);
            }
        }
    }
    
    delete req;
    return true; // 继续处理更多数据包
}

bool SnnNIC::spaceAvailable(int vn)
{
    NIC_LOG(5, "网络发送空间可用：VN=%d\n", vn);
    // 标记链路与对应VN已就绪（端口握手允许发送）
    if (vn >= 0 && vn < (int)vn_ready_.size()) vn_ready_[vn] = true;
    if (!link_ready_) {
        link_ready_ = true;
        NIC_LOG(3, "LinkControl握手完成: link_ready=1, VN=%d 标记就绪\n", vn);
    }
    
    // 检查网络接口是否有效
    if (!network_ready_ || !network || !network->isNetworkInitialized() || !link_ready_) {
        NIC_LOG(2, "回调时网络未就绪，跳过待发送队列: ready=%d, isInit=%d, link_ready=%d\n",
               (int)network_ready_, network ? (int)network->isNetworkInitialized() : 0, (int)link_ready_);
        return true;
    }
    
    // 批处理：根据开关决定是否在credit回调中刷新（默认开；便于调试时可关闭，仅依赖时钟窗口触发）
    if (enable_batching_ && flush_on_credit_) {
        for (auto& kv : batch_buckets_) {
            uint32_t dest = kv.first;
            const auto& vec = kv.second;
            uint32_t threshold = isNeighborNode(dest) ? batch_size_local_ : batch_size_remote_;
            if (vec.size() >= threshold) {
                flushBatchToNode(dest);
            }
        }
    }

    // 跨Rank代理批量：回调时触发刷新（按窗口或有数据即刷）
    if (enable_inter_rank_batching_ && nodes_per_rank_ > 0) {
        // 简化：在credit回调时尝试全部刷新（窗口限制由时钟+门控保证）
        flushInterRankAll();
    }

    // 处理待发送队列中的脉冲（如果有的话）
    while (!pending_spikes.empty() && network->spaceToSend(vn, 1)) {
        SpikeEvent* spike = pending_spikes.front();
        pending_spikes.pop();
        
        // 获取目标节点ID
        uint32_t dest_node = spike->getDestinationNode();
        SimpleNetwork::Request* req = createNetworkRequest(spike, dest_node);
        if (req) {
            req->vn = vn; // 使用当前可用的vn
            if (vn == 1) {
                NIC_LOG(0, "ℹ️ 使用VN1发送延迟脉冲：节点%u -> 节点%u\n", node_id, dest_node);
            }
        }
        // 使用相同的双重检查模式
        if (req && network->spaceToSend(vn, req->size_in_bits) && network->send(req, vn)) {
            NIC_LOG(4, "发送延迟的脉冲事件成功：节点%u -> 节点%u (vn=%d)\n", node_id, dest_node, vn);
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
    
    if (!use_direct_link && network) {
        
        try {
            // 只调用网络接口的init，不进行复杂的初始化数据交换
            network->init(phase);
        } catch (const std::exception& e) {
            NIC_LOG(0, "❌ LinkControl.init(%u)异常: %s\n", phase, e.what());
            throw;
        }
        
    } else {
        NIC_LOG(2, "⏭️ SnnNIC[节点%u] 跳过网络接口初始化 (direct_link=%s)\n", 
                       node_id, use_direct_link ? "true" : "false");
    }

    // 标记初始化阶段是否完成（通常在phase>=2后，VN映射已协商完成）
    if (phase >= 2) init_done_ = true;
    NIC_LOG(4, "init阶段=%u: isNetworkInitialized=%d, init_done_=%d\n",
            phase, network ? (int)network->isNetworkInitialized() : 0, (int)init_done_);
}

void SnnNIC::setup()
{
    
    if (!use_direct_link && network) {
        
        try {
            network->setup();
        } catch (const std::exception& e) {
            NIC_LOG(0, "❌ LinkControl.setup()异常: %s\n", e.what());
            throw;
        }
    }
    
    // 注册批处理时间窗时钟（若开启批处理且窗口>0）
    if (enable_batching_ && batch_flush_window_ns_ > 0) {
        std::string period = std::to_string(batch_flush_window_ns_) + "ns";
        // 使用Handler2避免弃用警告
        registerClock(period, new SST::Clock::Handler2<SnnNIC, &SnnNIC::flushClockTick>(this));
    }

    // 可选：在setup阶段主动探测各VN的可用性，预热vn_ready_
    if (!use_direct_link && network && probe_vn_on_setup_) {
        for (int v = 0; v < (int)effective_num_vns_; ++v) {
            bool can = network->spaceToSend(v, 1);
            NIC_LOG(2, "VN探测(spaceToSend): vn=%d can=%d\n", v, (int)can);
        }
    }

    //                 node_id, use_direct_link ? "直接Link" : "SimpleNetwork");
    network_ready_ = true;
    NIC_LOG(3, "setup完成: network_ready=1, isNetworkInitialized=%d, eff_vns=%u\n",
            network ? (int)network->isNetworkInitialized() : 0, effective_num_vns_);
}

void SnnNIC::finish()
{
    NIC_LOG(1, "完成阶段\n");
    // 注意：SST在complete/finish阶段禁止再调用send/recv。
    // 因此此处不再做批次刷新，必须依赖运行时窗口刷出。
    // 若确需在末尾确保发送，请将窗口调小或在更早阶段触发刷新。
    
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
    
    // 注意：批量载荷的处理在 extractSpikeEvent() 中实现
    
    // 创建网络请求（单条）
    SimpleNetwork::Request* req = new SimpleNetwork::Request();
    req->dest = dest_node;
    req->src = node_id;
    // 默认使用配置的脉冲数据VN，调用方可覆盖
    req->vn = static_cast<int>(vn_spike_data_);
    req->size_in_bits = sizeof(NetSpikePayload) * 8;
    req->head = true;
    req->tail = true;
    req->allow_adaptive = true;
    
    // 创建载荷并赋给请求
    auto* payload = new NetSpikePayload(spike_event);
    req->givePayload(payload);
    
    NIC_LOG(4, "创建网络请求：源=%ld，目标=%ld，大小=%zu bits\n",
                   req->src, req->dest, req->size_in_bits);
    
    return req;
}

SpikeEvent* SnnNIC::extractSpikeEvent(SimpleNetwork::Request* req)
{
    if (!req || !req->inspectPayload()) {
        return nullptr;
    }
    
    // 从payload中提取SpikeEvent信息
    NetSpikePayload* payload = static_cast<NetSpikePayload*>(req->inspectPayload());
    // 重建单条SpikeEvent对象
    SpikeEvent* spike_event = new SpikeEvent();
    spike_event->neuron_id = payload->src_neuron_id;
    spike_event->setDestinationNeuron(payload->dest_neuron_id);
    spike_event->timestamp = payload->timestamp;
    spike_event->setWeight(payload->weight);
    spike_event->setDestinationNode(static_cast<uint32_t>(req->dest));
    NIC_LOG(4, "解包SpikeEvent：神经元%u -> 神经元%u\n",
                    payload->src_neuron_id, payload->dest_neuron_id);
    return spike_event;
}

void SnnNIC::handleDirectSpikeEvent(SST::Event* event)
{
    if (!event) {
        return;
    }
    // 控制事件优先
    if (auto* gd = dynamic_cast<GatingDecisionEvent*>(event)) {
        if (control_handler_) control_handler_(gd);
        else delete gd;
        return;
    }
    // 直接转换为SpikeEvent
    if (auto* spike_event = dynamic_cast<SpikeEvent*>(event)) {
        NIC_LOG(3, "接收直接Link脉冲：源神经元=%u，目标神经元=%u\n",
                       spike_event->neuron_id, spike_event->getDestinationNeuron());
        if (spike_handler) {
            spikes_received_count++;
            packets_received_count++;
            stat_spikes_received->addData(1);
            stat_packets_received->addData(1);
            spike_handler(spike_event);
        } else { delete spike_event; }
        return;
    }
    // 未知事件
    delete event;
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

void SnnNIC::tryInterRankBatch(SpikeEvent* spike)
{
    if (!spike) return;
    uint32_t dest_node = spike->getDestinationNode();
    uint32_t dest_rank = computeRankForNode(dest_node);
    auto& vec = ir_buckets_[dest_rank];
    vec.push_back(spike);
    auto ts = spike->timestamp;
    auto it = ir_earliest_ts_.find(dest_rank);
    if (it == ir_earliest_ts_.end() || ts < it->second) ir_earliest_ts_[dest_rank] = ts;
}

void SnnNIC::flushBatchToNode(uint32_t dest_node)
{
    // 若网络未完成初始化，延后批量刷新
    if (!network || !network->isNetworkInitialized()) {
        if (!gating_logged_flush_) { gating_logged_flush_ = true; logGatingSnapshot("flushBatchToNode(isInit=0)"); }
        return;
    }

    auto it = batch_buckets_.find(dest_node);
    if (it == batch_buckets_.end() || it->second.empty()) return;
    if (!network_ready_ || !network || !network->isNetworkInitialized()) {
        if (!gating_logged_flush_) { gating_logged_flush_ = true; logGatingSnapshot("flushBatchToNode(gating)"); }
        return;
    }

    // 填充批量数据
    auto* payload = new NetSpikeBatchPayload();
    payload->batch.reserve(it->second.size());
    for (auto* s : it->second) {
        NetSpikeBatchPayload::PackedSpike ps{ s->neuron_id, s->getDestinationNeuron(), s->timestamp, s->getWeight() };
        payload->batch.push_back(ps);
    }
    payload->source_node = node_id;
    payload->dest_node = dest_node;
    payload->batch_timestamp = batch_earliest_ts_[dest_node];

    // 创建请求（单VN=0）
    SimpleNetwork::Request* req = new SimpleNetwork::Request();
    req->dest = dest_node;
    req->src = node_id;
    req->vn = 0;
    // 大致估算消息大小
    size_t per = sizeof(uint32_t)*2 + sizeof(uint64_t) + sizeof(float);
    size_t bytes = sizeof(uint32_t)*3 + sizeof(uint64_t) + per * payload->batch.size();
    req->size_in_bits = bytes * 8;
    req->head = true;
    req->tail = true;
    req->allow_adaptive = true;
    req->givePayload(payload);

    NIC_LOG(1, "flushBatchToNode: 即将发送 批量=%zu, vn=0, eff_vns=%u, size_bits=%zu\n",
                 payload->batch.size(), effective_num_vns_, req->size_in_bits);
    bool sent = false;
    if (network->spaceToSend(0, req->size_in_bits)) {
        sent = network->send(req, 0);
    }
    if (sent) {
        packets_sent_count++;
        stat_packets_sent->addData(1);
        if (stat_spikes_sent) stat_spikes_sent->addData(static_cast<uint64_t>(payload->batch.size()));
        if (stat_batches_sent) stat_batches_sent->addData(1);
        NIC_LOG(2, "批量发送：%zu 条 -> 节点%u (vn=0)\n", payload->batch.size(), dest_node);
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

void SnnNIC::flushInterRankTo(uint32_t dest_rank)
{
    if (!network || !network->isNetworkInitialized()) return;
    if (!enable_inter_rank_batching_ || nodes_per_rank_ == 0) return;
    auto it = ir_buckets_.find(dest_rank);
    if (it == ir_buckets_.end() || it->second.empty()) return;
    if (!network_ready_) return;

    // 选取网关节点（该rank的首个节点）
    uint32_t gateway = computeGatewayNodeForRank(dest_rank);
    // 构造代理批量载荷
    auto* payload = new NetInterRankBatchPayload();
    payload->batch.reserve(it->second.size());
    for (auto* s : it->second) {
        NetInterRankBatchPayload::PackedSpike ps{ s->neuron_id, s->getDestinationNeuron(), s->getDestinationNode(), s->timestamp, s->getWeight() };
        payload->batch.push_back(ps);
    }
    payload->source_node = node_id;
    payload->gateway_node = gateway;
    payload->batch_timestamp = ir_earliest_ts_[dest_rank];

    // 创建请求并发送到目标rank网关节点
    SimpleNetwork::Request* req = new SimpleNetwork::Request();
    req->dest = gateway;
    req->src = node_id;
    int vn_use = static_cast<int>(vn_batch_data_);
    if (vn_use >= static_cast<int>(effective_num_vns_)) vn_use = 0;
    req->vn = vn_use;
    // 粗略估算大小
    size_t per = sizeof(uint32_t)*3 + sizeof(uint64_t) + sizeof(double);
    size_t bytes = sizeof(uint32_t)*2 + sizeof(uint64_t) + per * payload->batch.size();
    req->size_in_bits = bytes * 8;
    req->head = true;
    req->tail = true;
    req->allow_adaptive = true;
    req->givePayload(payload);

    NIC_LOG(1, "IR-Flush: rank=%u -> gateway=%u, batch=%zu, vn=%d\n", dest_rank, gateway, payload->batch.size(), req->vn);
    bool sent_ir = false;
    if (network->spaceToSend(req->vn, req->size_in_bits)) {
        if (req->vn == 1) {
            NIC_LOG(0, "ℹ️ 使用VN1发送跨Rank批量：网关=%u，条数=%zu\n", gateway, payload->batch.size());
        }
        sent_ir = network->send(req, req->vn);
    } else if (auto_vn_fallback_ && req->vn != 0 && network->spaceToSend(0, req->size_in_bits)) {
        NIC_LOG(0, "⚠️ 自动降级VN: IR batch 使用VN=%d→0 (space不足/未就绪)\n", req->vn);
        req->vn = 0;
        sent_ir = network->send(req, req->vn);
    }
    if (sent_ir) {
        stat_packets_sent->addData(1);
        if (stat_spikes_sent) stat_spikes_sent->addData((uint64_t)payload->batch.size());
        if (stat_ir_batches_sent) stat_ir_batches_sent->addData(1);
        it->second.clear();
        ir_earliest_ts_.erase(dest_rank);
    } else {
        delete req;
    }
}

void SnnNIC::flushInterRankAll()
{
    std::vector<uint32_t> keys;
    keys.reserve(ir_buckets_.size());
    for (auto& kv : ir_buckets_) if (!kv.second.empty()) keys.push_back(kv.first);
    for (auto r : keys) flushInterRankTo(r);
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
    if (enable_inter_rank_batching_ && inter_rank_batch_window_ns_ > 0 && nodes_per_rank_ > 0) {
        flushInterRankAll();
    }
    return false; // 持续触发
}
bool SnnNIC::sendControl(SST::Event* ev, uint32_t dest_node)
{
    if (!network || !ev) return false;
    SimpleNetwork::Request* req = new SimpleNetwork::Request();
    req->dest = dest_node;
    req->src = node_id;
    int vn_use = static_cast<int>(vn_control_);
    if (vn_use >= static_cast<int>(effective_num_vns_)) {
        if (!vn_guard_warned_) { NIC_LOG(0, "⚠️ VN越界: vn_control=%d, virtual_channels=%u, 回退VN=0\n", vn_use, virtual_channels_); vn_guard_warned_ = true; }
        vn_use = 0;
    }
    if (auto_vn_fallback_ && vn_use != 0 && effective_num_vns_ > 1) {
        NIC_LOG(0, "⚠️ 自动降级VN: control 使用VN=%d→0 (multi-VN不稳定)\n", vn_use);
        vn_use = 0;
    }
    req->vn = vn_use;
    req->size_in_bits = 256; // 近似大小；SimpleNetwork不强制精确
    req->head = true;
    req->tail = true;
    req->allow_adaptive = true;
    req->givePayload(ev);
    if (network->spaceToSend(req->vn, req->size_in_bits) && network->send(req, req->vn)) {
        stat_packets_sent->addData(1);
        return true;
    }
    delete req;
    return false;
}
