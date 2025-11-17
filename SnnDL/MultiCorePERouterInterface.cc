// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// MultiCorePERouterInterface.cc: MultiCorePE专用的hr_router接口实现
//

#include "MultiCorePERouterInterface.h"
#include <cassert>
#include <cstdarg>

using namespace SST;
using namespace SST::SnnDL;

MultiCorePERouterInterface::MultiCorePERouterInterface(ComponentId_t id, Params& params)
    : SnnInterface(id, params)
    , node_id_(0)
    , verbose_(0)
    , port_name_("network")
    , router_(nullptr)
    , spike_handler_(nullptr)
    , output_(nullptr)
{
    // 解析参数
    node_id_ = params.find<uint32_t>("node_id", 0);
    verbose_ = params.find<uint32_t>("verbose", 0);
    port_name_ = params.find<std::string>("port_name", "network");
    link_bw_ = params.find<std::string>("link_bw", "40GiB/s");
    input_buf_size_ = params.find<std::string>("input_buf_size", "2KiB");
    output_buf_size_ = params.find<std::string>("output_buf_size", "2KiB");
    
    // 初始化输出
    output_ = new Output("MultiCorePERouterInterface[@p:@l]: ", verbose_, 0, Output::STDOUT);
    
    debugPrint(1, "🚀 MultiCorePERouterInterface构造: 节点%u, 端口=%s", 
               node_id_, port_name_.c_str());
    
    // 初始化SimpleNetwork接口
    initializeSimpleNetwork();
    
    // 注册统计项
    stat_spikes_sent_ = registerStatistic<uint64_t>("spikes_sent");
    stat_spikes_received_ = registerStatistic<uint64_t>("spikes_received");
    stat_bytes_sent_ = registerStatistic<uint64_t>("bytes_sent");
    stat_bytes_received_ = registerStatistic<uint64_t>("bytes_received");
    stat_packets_sent_ = registerStatistic<uint64_t>("packets_sent");
    stat_packets_received_ = registerStatistic<uint64_t>("packets_received");
    stat_send_buffer_occupancy_ = registerStatistic<double>("send_buffer_occupancy");
    stat_recv_buffer_occupancy_ = registerStatistic<double>("recv_buffer_occupancy");
    
    debugPrint(2, "📊 统计项注册完成");
}

MultiCorePERouterInterface::~MultiCorePERouterInterface() {
    // 清理发送队列
    while (!send_queue_.empty()) {
        delete send_queue_.front();
        send_queue_.pop();
    }
    
    if (output_) {
        delete output_;
    }
    
    debugPrint(1, "🔚 MultiCorePERouterInterface析构完成");
}

void MultiCorePERouterInterface::initializeSimpleNetwork() {
    debugPrint(2, "🔗 开始初始化SimpleNetwork接口");
    
    // 准备LinkControl参数
    Params net_params;
    net_params.insert("link_bw", link_bw_);
    net_params.insert("input_buf_size", input_buf_size_);
    net_params.insert("output_buf_size", output_buf_size_);
    net_params.insert("port_name", port_name_);
    
    // LinkControl端点配置
    net_params.insert("job_id", "0");
    net_params.insert("job_size", "1");
    net_params.insert("logical_nid", std::to_string(node_id_));
    
    debugPrint(3, "📋 LinkControl参数: bw=%s, buf=%s/%s, nid=%u", 
               link_bw_.c_str(), input_buf_size_.c_str(), output_buf_size_.c_str(), node_id_);
    
    try {
        // 创建LinkControl SubComponent，使用SHARE_PORTS确保端口暴露
        router_ = loadAnonymousSubComponent<SST::Interfaces::SimpleNetwork>(
            "merlin.linkcontrol", 
            "linkcontrol", 
            0, 
            ComponentInfo::SHARE_PORTS | ComponentInfo::INSERT_STATS, 
            net_params,
            1  // num_vns
        );
        
        if (router_) {
            debugPrint(1, "✅ LinkControl SubComponent创建成功");
            
            // 设置接收处理函数
            router_->setNotifyOnReceive(
                new SST::Interfaces::SimpleNetwork::Handler2<MultiCorePERouterInterface, &MultiCorePERouterInterface::handleNetworkEvent>(
                    this
                )
            );
            
            debugPrint(2, "✅ 网络事件处理器注册完成");
        } else {
            debugPrint(0, "❌ LinkControl SubComponent创建失败");
            throw std::runtime_error("无法创建LinkControl SubComponent");
        }
        
    } catch (const std::exception& e) {
        debugPrint(0, "❌ SimpleNetwork初始化异常: %s", e.what());
        throw;
    }
}

void MultiCorePERouterInterface::init(unsigned int phase) {
    debugPrint(3, "🔄 init阶段%u开始", phase);
    
    if (router_) {
        router_->init(phase);
        debugPrint(3, "✅ LinkControl.init(%u)完成", phase);
    }
}

void MultiCorePERouterInterface::setup() {
    debugPrint(2, "⚙️ setup开始");
    
    if (router_) {
        router_->setup();
        debugPrint(2, "✅ LinkControl.setup()完成");
    }
    
    debugPrint(1, "🎯 MultiCorePERouterInterface setup完成: 节点%u就绪", node_id_);
}

void MultiCorePERouterInterface::finish() {
    debugPrint(2, "🏁 finish开始");
    
    if (router_) {
        router_->finish();
        debugPrint(2, "✅ LinkControl.finish()完成");
    }
    
    // 输出最终统计
    debugPrint(1, "📊 最终统计: MultiCorePERouterInterface finish完成");
}

void MultiCorePERouterInterface::setSpikeHandler(SpikeHandler handler) {
    spike_handler_ = handler;
    debugPrint(2, "🎯 脉冲处理器设置完成");
}

void MultiCorePERouterInterface::sendSpike(SpikeEvent* spike_event) {
    if (!spike_event) {
        debugPrint(1, "⚠️ 忽略空脉冲事件");
        return;
    }
    
    if (!router_) {
        debugPrint(1, "⚠️ 路由器未初始化，丢弃脉冲");
        delete spike_event;
        return;
    }
    
    debugPrint(4, "📤 发送脉冲: src=%u, dst=%u, target_node=%u", 
               spike_event->getSourceNeuron(), spike_event->getDestinationNeuron(), spike_event->getDestinationNode());
    
    // 转换为网络请求
    auto* req = convertSpikeToRequest(spike_event);
    if (!req) {
        debugPrint(1, "❌ 脉冲转换网络请求失败");
        delete spike_event;
        return;
    }
    
    // 发送请求
    bool sent = router_->send(req, 0);  // 使用VN 0
    if (sent) {
        stat_spikes_sent_->addData(1);
        stat_packets_sent_->addData(1);
        stat_bytes_sent_->addData(req->size_in_bits / 8);
        
        debugPrint(4, "✅ 脉冲发送成功: 目标节点%u", spike_event->getDestinationNode());
    } else {
        debugPrint(2, "⏳ 发送缓冲区满，加入队列");
        send_queue_.push(spike_event);
        delete req;  // 发送失败，清理请求
    }
    
    // 注意：spike_event由convertSpikeToRequest内部管理
}

bool MultiCorePERouterInterface::handleNetworkEvent(int vn) {
    // 接收网络请求
    SST::Interfaces::SimpleNetwork::Request* req = router_->recv(vn);
    if (!req) {
        return true; // 继续处理
    }
    
    debugPrint(4, "📥 接收网络请求: src=%u, dst=%u, size=%u", 
               req->src, req->dest, req->size_in_bits / 8);
    
    // 转换为脉冲事件
    SpikeEvent* spike_event = convertRequestToSpike(req);
    if (!spike_event) {
        debugPrint(1, "❌ 网络请求转换脉冲失败");
        delete req;
        return true;
    }
    
    // 更新统计
    stat_spikes_received_->addData(1);
    stat_packets_received_->addData(1);
    stat_bytes_received_->addData(req->size_in_bits / 8);
    
    debugPrint(4, "🎯 脉冲接收: src=%u, dst=%u, weight=%.3f", 
               spike_event->getSourceNeuron(), spike_event->getDestinationNeuron(), spike_event->getWeight());
    
    // 转发给父组件处理
    if (spike_handler_) {
        spike_handler_(spike_event);
    } else {
        debugPrint(1, "⚠️ 未设置脉冲处理器，丢弃脉冲");
        delete spike_event;
    }
    
    delete req;
    return true;
}

void MultiCorePERouterInterface::processSendQueue() {
    if (!router_ || send_queue_.empty()) {
        return;
    }
    
    // 尝试发送队列中的脉冲
    while (!send_queue_.empty()) {
        SpikeEvent* spike_event = send_queue_.front();
        
        auto* req = convertSpikeToRequest(spike_event);
        if (!req) {
            send_queue_.pop();
            delete spike_event;
            continue;
        }
        
        bool sent = router_->send(req, 0);
        if (sent) {
            send_queue_.pop();
            stat_spikes_sent_->addData(1);
            stat_packets_sent_->addData(1);
            stat_bytes_sent_->addData(req->size_in_bits / 8);
            
            debugPrint(4, "✅ 队列脉冲发送成功");
        } else {
            delete req;
            break;  // 缓冲区仍满，等待下次
        }
    }
}

SST::Interfaces::SimpleNetwork::Request* 
MultiCorePERouterInterface::convertSpikeToRequest(SpikeEvent* spike_event) {
    if (!spike_event) return nullptr;
    
    // 创建SpikeEventWrapper作为载荷
    SpikeEventWrapper* wrapper = new SpikeEventWrapper(spike_event);
    
    // 创建网络请求
    auto* req = new SST::Interfaces::SimpleNetwork::Request();
    req->src = node_id_;
    req->dest = spike_event->getDestinationNode();
    req->size_in_bits = sizeof(SpikeEventWrapper) * 8;
    req->vn = 0;
    req->head = true;
    req->tail = true;
    req->allow_adaptive = true;
    req->givePayload(wrapper);
    
    debugPrint(5, "🔄 脉冲转换请求: %u→%u, size=%lu", 
               req->src, req->dest, sizeof(SpikeEventWrapper));
    
    return req;
}

SpikeEvent* 
MultiCorePERouterInterface::convertRequestToSpike(SST::Interfaces::SimpleNetwork::Request* request) {
    if (!request || !request->inspectPayload()) {
        return nullptr;
    }
    
    // 从载荷中提取SpikeEventWrapper
    SpikeEventWrapper* wrapper = static_cast<SpikeEventWrapper*>(request->inspectPayload());
    if (!wrapper) {
        debugPrint(1, "❌ 无效的载荷类型");
        return nullptr;
    }
    
    // 从wrapper中提取原始SpikeEvent
    SpikeEvent* original_spike = wrapper->getSpikeEvent();
    if (!original_spike) {
        debugPrint(1, "❌ wrapper中没有SpikeEvent数据");
        delete wrapper;
        return nullptr;
    }
    
    // 创建新的SpikeEvent
    SpikeEvent* spike_event = new SpikeEvent(
        original_spike->getSourceNeuron(),
        original_spike->getDestinationNeuron(),
        original_spike->getDestinationNode(),
        original_spike->getWeight(),
        original_spike->getTimestamp()
    );
    
    debugPrint(5, "🔄 请求转换脉冲: src=%u, dst=%u, weight=%.3f", 
               spike_event->getSourceNeuron(), spike_event->getDestinationNeuron(), spike_event->getWeight());
    
    delete wrapper;  // 清理载荷
    return spike_event;
}

void MultiCorePERouterInterface::setNodeId(uint32_t node_id) {
    node_id_ = node_id;
    debugPrint(2, "🆔 节点ID设置为: %u", node_id_);
}

uint32_t MultiCorePERouterInterface::getNodeId() const {
    return node_id_;
}

std::string MultiCorePERouterInterface::getNetworkStatus() const {
    std::string status = "MultiCorePERouterInterface[节点" + std::to_string(node_id_) + "]";
    status += " 状态: ";
    status += (router_ ? "就绪" : "未初始化");
    status += ", 发送队列: " + std::to_string(send_queue_.size());
    return status;
}

void MultiCorePERouterInterface::updateBufferStats() {
    if (!router_) return;
    
    // 简单的缓冲区占用率估算
    double send_occupancy = static_cast<double>(send_queue_.size()) / 100.0; // 假设最大队列100
    // Note: These stats are initialized in constructor but may not be available in this scope
    // stat_send_buffer_occupancy_->addData(std::min(send_occupancy, 1.0));
    
    // 接收缓冲区占用率由router内部管理
    // stat_recv_buffer_occupancy_->addData(0.0);
}

void MultiCorePERouterInterface::debugPrint(uint32_t level, const char* format, ...) {
    if (level <= verbose_ && output_) {
        va_list args;
        va_start(args, format);
        
        char buffer[1024];
        vsnprintf(buffer, sizeof(buffer), format, args);
        
        output_->verbose(CALL_INFO, level, 0, "%s\n", buffer);
        
        va_end(args);
    }
}