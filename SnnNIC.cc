// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// SnnNIC.cc: SNN网络接口控制器实现文件
//

#include "SnnNIC.h"
#include <sst/core/serialization/serialize.h>
#include <sstream>
#include <fstream>

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

// Static member definitions
std::mutex SnnNIC::s_spike_csv_mutex_;
std::unordered_set<std::string> SnnNIC::s_spike_csv_files_;

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
    // 构造期哨兵（默认静默，仅当 SNNDL_SENTINEL_ENABLE 打开时打印）
    do {
        const char* _sent = std::getenv("SNNDL_SENTINEL_ENABLE");
        if (_sent && std::atoi(_sent) != 0) {
            NIC_LOG(0, "[[sentinel-nic-ctor]] enter\n");
        }
    } while(0);
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
    const char* sent_env = std::getenv("SNNDL_SENTINEL_ENABLE");
    if (sent_env && std::atoi(sent_env) != 0 && verbose == 0) verbose = 1;
    
    // 初始化日志输出
    output = new Output("SnnNIC[@p:@l]: ", verbose, 0, Output::STDOUT);
    
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
        // 首先尝试加载用户定义的网络接口
        network = loadUserSubComponent<SimpleNetwork>("linkcontrol", ComponentInfo::SHARE_NONE, static_cast<int>(effective_num_vns_));
        if (!network) {
            // 创建默认的 linkcontrol
            Params net_params;
            net_params.insert("port_name", params.find<std::string>("port_name", "network"));
            net_params.insert("link_bw", link_bw);
            net_params.insert("input_buf_size", input_buf_size);
            net_params.insert("output_buf_size", output_buf_size);
            // 强制单VN
            net_params.insert("num_vns", std::to_string(1));
            // 添加PortControl协议调试参数
            net_params.insert("job_id", "0");
            // job_size=总节点数
            net_params.insert("job_size", std::to_string(total_nodes_));
            net_params.insert("logical_nid", std::to_string(node_id));
            
            network = loadAnonymousSubComponent<SimpleNetwork>("merlin.linkcontrol", "linkcontrol", 0, 
                ComponentInfo::SHARE_PORTS | ComponentInfo::INSERT_STATS, net_params, static_cast<int>(effective_num_vns_));
        }
        
        if (network) {
            // 设置网络回调处理器
            network->setNotifyOnReceive(new SimpleNetwork::Handler2<SnnNIC,&SnnNIC::handleIncoming>(this));
            network->setNotifyOnSend(new SimpleNetwork::Handler2<SnnNIC,&SnnNIC::spaceAvailable>(this));
        }
    }
    
    // 注册统计信息
    stat_spikes_sent = registerStatistic<uint64_t>("spikes_sent");
    stat_spikes_received = registerStatistic<uint64_t>("spikes_received");
    stat_packets_sent = registerStatistic<uint64_t>("packets_sent");
    stat_packets_received = registerStatistic<uint64_t>("packets_received");
    stat_batches_sent = registerStatistic<uint64_t>("batches_sent");
    stat_ir_batches_sent = registerStatistic<uint64_t>("inter_rank_batches_sent");
    // P1：注册消息时延统计（纳秒）
    stat_msg_latency_ns = registerStatistic<uint64_t>("msg_latency_ns");

    // Tier 1
    stat_spikes_local_core = registerStatistic<uint64_t>("spikes_local_core");
    stat_spikes_neighbor_node = registerStatistic<uint64_t>("spikes_neighbor_node");
    stat_spikes_remote_node = registerStatistic<uint64_t>("spikes_remote_node");

    // 批处理效率
    stat_batch_total_spikes = registerStatistic<uint64_t>("batch_total_spikes");
    stat_batch_flush_timeout = registerStatistic<uint64_t>("batch_flush_timeout");
    stat_batch_flush_full = registerStatistic<uint64_t>("batch_flush_full");

    // 流量类型
    stat_payload_bytes_sent = registerStatistic<uint64_t>("payload_bytes_sent");
    stat_total_bytes_sent = registerStatistic<uint64_t>("total_bytes_sent");

    // hop count
    stat_hop_count_sum = registerStatistic<uint64_t>("hop_count_sum");
    stat_hop_count_max = registerStatistic<uint64_t>("hop_count_max");
    stat_hop_count_min = registerStatistic<uint64_t>("hop_count_min");

    // 批大小聚合
    stat_batch_size_sum = registerStatistic<uint64_t>("batch_size_sum");
    stat_batch_size_max = registerStatistic<uint64_t>("batch_size_max");
    stat_batch_size_min = registerStatistic<uint64_t>("batch_size_min");
}

SnnNIC::~SnnNIC()
{
    // 让进程退出阶段由运行时回收日志对象，避免潜在的析构次序竞态
    output = nullptr;
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
    // 直接Link模式已禁用，统一走network
    if (use_direct_link && direct_link) {
        SpikeEvent* network_spike = new SpikeEvent(*spike_event);
        direct_link->send(network_spike);
        spikes_sent_count++;
        packets_sent_count++;
        stat_spikes_sent->addData(1);
        stat_packets_sent->addData(1);
        return;
    }
    // 使用网络接口
    SimpleNetwork::Request* req = createNetworkRequest(spike_event, dest_node);
    if (!req) return;
    int vn_use = static_cast<int>(vn_spike_data_);
    if (vn_use >= static_cast<int>(effective_num_vns_)) vn_use = 0;
    req->vn = vn_use;
    bool sent = false;
    bool canSpace = false;
    if (network) {
        canSpace = network->spaceToSend(req->vn, req->size_in_bits);
        if (canSpace) {
            sent = network->send(req, req->vn);
        }
    }
    if (sent) {
        stat_packets_sent->addData(1);
        stat_spikes_sent->addData(1);
        spikes_sent_count++;
        // 成功发送后，NIC 接管了该脉冲的生命周期，释放本地对象
        delete spike_event;
        // debug print removed for production
    } else {
        // 无空间，进入待发送队列
        pending_spikes.push(spike_event);
        delete req;
        // debug print removed for production
    }
}

void SnnNIC::setNodeId(uint32_t id)
{
    node_id = id;
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
                auto* spike_event = new SpikeEvent(ps.src_neuron_id, ps.dest_neuron_id,
                                                  static_cast<uint32_t>(ps.final_dest_node), (float)ps.weight, ps.timestamp);
                if (ps.final_dest_node == node_id) {
                    if (spike_handler) {
                        spikes_received_count++;
                        stat_spikes_received->addData(1);
                        spike_handler(spike_event);
                    } else {
                        delete spike_event;
                    }
                } else {
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
                    spikes_received_count++;
                    stat_spikes_received->addData(1);
                    spike_handler(spike_event);
                } else {
                    delete spike_event;
                }
            }
        }
        // 单条脉冲
        else {
            SpikeEvent* spike_event = extractSpikeEvent(req);
            if (spike_event && spike_handler) {
                NIC_LOG(4, "提取到脉冲事件：源神经元=%u，目标神经元=%u\n",
                               spike_event->neuron_id, spike_event->getDestinationNeuron());
                spikes_received_count++;
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
    if (vn >= 0 && vn < (int)vn_ready_.size()) vn_ready_[vn] = true;
    if (!link_ready_) {
        link_ready_ = true;
        NIC_LOG(3, "LinkControl握手完成: link_ready=1, VN=%d 标记就绪\n", vn);
    }
    if (!network_ready_ || !network || !network->isNetworkInitialized() || !link_ready_) {
        NIC_LOG(2, "回调时网络未就绪，跳过待发送队列: ready=%d, isInit=%d, link_ready=%d\n",
               (int)network_ready_, network ? (int)network->isNetworkInitialized() : 0, (int)link_ready_);
        return true;
    }
    // 批处理刷新
    if (enable_batching_ && flush_on_credit_) {
        for (auto& kv : batch_buckets_) {
            uint32_t dest = kv.first;
            const auto& vec = kv.second;
            uint32_t threshold = isNeighborNode(dest) ? batch_size_local_ : batch_size_remote_;
            if (vec.size() >= threshold) flushBatchToNode(dest);
        }
    }
    // 处理待发送队列（仅在成功发送后出队；失败则保留以便后续重试）
    while (!pending_spikes.empty() && network->spaceToSend(vn, 1)) {
        SpikeEvent* spike = pending_spikes.front();
        uint32_t dest_node = spike->getDestinationNode();
        SimpleNetwork::Request* req = createNetworkRequest(spike, dest_node);
        if (req) {
            req->vn = vn;
            if (network->send(req, vn)) {
                stat_packets_sent->addData(1);
                stat_spikes_sent->addData(1);
                spikes_sent_count++;
                pending_spikes.pop();
                delete spike;
            } else {
                delete req;
                // 无法发送则退出，等待下一次credit/tick再重试，保留队首
                break;
            }
        }
    }
    return true;
}

void SnnNIC::finish()
{
    // 安全收尾：避免在finish阶段进行复杂I/O或对象释放，交由进程退出回收
    // 注：不打印“最终统计”，减少退出期竞态干扰
    // 注：不主动清空 pending_spikes，防止潜在外部持有者与析构次序问题
    if (!use_direct_link && network) network->finish();
}

SimpleNetwork::Request* SnnNIC::createNetworkRequest(SpikeEvent* spike_event, uint32_t dest_node)
{
    if (!spike_event) return nullptr;
    SimpleNetwork::Request* req = new SimpleNetwork::Request();
    req->dest = dest_node;
    req->src = node_id;
    req->vn = static_cast<int>(vn_spike_data_);
    req->size_in_bits = sizeof(NetSpikePayload) * 8;
    req->head = true;
    req->tail = true;
    req->allow_adaptive = true;
    auto* payload = new NetSpikePayload(spike_event);
    req->givePayload(payload);
    NIC_LOG(4, "创建网络请求：源=%ld，目标=%ld，大小=%zu bits\n",
                   req->src, req->dest, req->size_in_bits);
    return req;
}

SpikeEvent* SnnNIC::extractSpikeEvent(SimpleNetwork::Request* req)
{
    if (!req || !req->inspectPayload()) return nullptr;
    NetSpikePayload* payload = static_cast<NetSpikePayload*>(req->inspectPayload());
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
    if (!event) return;
    if (auto* gd = dynamic_cast<GatingDecisionEvent*>(event)) {
        if (control_handler_) control_handler_(gd);
        else delete gd;
        return;
    }
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
    delete event;
}

// === SST lifecycle ===
void SnnNIC::init(unsigned int phase)
{
    if (output) {
        const char* _sent = std::getenv("SNNDL_SENTINEL_ENABLE");
        if (_sent && std::atoi(_sent) != 0) output->output("[[sentinel-nic-init]] node=%u phase=%u enter\n", node_id, phase);
    }
    if (!use_direct_link && network) {
        network->init(phase);
    }
    // 可选：注册周期性tick用于批处理窗口刷新
    if (phase == 0 && (enable_batching_ || enable_inter_rank_batching_)) {
        // 轻量tick；网络未就绪时仅返回false不做事
        registerClock("1GHz", new Clock::Handler<SnnNIC>(this, &SnnNIC::flushClockTick));
    }
    if (phase >= 1) init_done_ = true;
    if (output) {
        const char* _sent = std::getenv("SNNDL_SENTINEL_ENABLE");
        if (_sent && std::atoi(_sent) != 0) output->output("[[sentinel-nic-init]] node=%u phase=%u done\n", node_id, phase);
    }
}

void SnnNIC::setup()
{
    if (output) {
        const char* _sent = std::getenv("SNNDL_SENTINEL_ENABLE");
        if (_sent && std::atoi(_sent) != 0) output->output("[[sentinel-nic-setup]] node=%u enter\n", node_id);
    }
    if (!use_direct_link && network) {
        network->setup();
    }
    network_ready_ = true;
    if (output) {
        output->verbose(CALL_INFO, 0, 0,
            "[nic-config] node=%u mode=%s link_bw=%s in_buf=%s out_buf=%s vn=%u\n",
            node_id,
            use_direct_link ? "direct" : "SimpleNetwork",
            link_bw.c_str(), input_buf_size.c_str(), output_buf_size.c_str(),
            effective_num_vns_);
    }
    if (output) {
        const char* _sent = std::getenv("SNNDL_SENTINEL_ENABLE");
        if (_sent && std::atoi(_sent) != 0) output->output("[[sentinel-nic-setup]] node=%u done\n", node_id);
    }
}

bool SnnNIC::flushClockTick(SST::Cycle_t /*currentCycle*/)
{
    // 批处理窗口刷新（轻量）
    if (enable_batching_) {
        // 简化：遍历所有目的节点刷新
        for (auto& kv : batch_buckets_) {
            if (!kv.second.empty()) {
                flushBatchToNode(kv.first);
            }
        }
    }
    if (enable_inter_rank_batching_ && nodes_per_rank_ > 0) {
        flushInterRankAll();
    }
    return false; // 不要求持续tick
}

uint32_t SnnNIC::computeDestNode(uint32_t /*dest_neuron*/) const
{
    // 当前SpikeEvent已包含目标节点，若需映射请在调用前设置
    return 0;
}

bool SnnNIC::sendControl(SST::Event* ev, uint32_t dest_node)
{
    if (!network || !ev) return false;
    SimpleNetwork::Request* req = new SimpleNetwork::Request();
    req->dest = dest_node;
    req->src = node_id;
    int vn_use = static_cast<int>(vn_control_);
    if (vn_use >= static_cast<int>(effective_num_vns_)) vn_use = 0;
    req->vn = vn_use;
    req->size_in_bits = 256;
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

void SnnNIC::logSpikeMessage_(const char* kind,
                          uint32_t src_node,
                          uint32_t dst_node,
                          uint64_t /*logical_step*/,
                          uint64_t /*start_ns*/,
                          uint32_t spike_count,
                          uint32_t payload_bytes)
{
    if (spike_trace_csv_.empty()) return;
    std::lock_guard<std::mutex> guard(s_spike_csv_mutex_);
    std::ofstream f(spike_trace_csv_, std::ios::app);
    if (!f.good()) return;
    static constexpr const char* header =
        "kind,src_tile,dst_tile,spike_count,payload_bytes\n";
    if (s_spike_csv_files_.insert(spike_trace_csv_).second) {
        f << header;
    }
    f << (kind ? kind : "spike") << ',' << src_node << ',' << dst_node << ','
      << spike_count << ',' << payload_bytes << '\n';
}
