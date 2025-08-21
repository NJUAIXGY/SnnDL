// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// OptimizedInternalRing.cc: 优化的内部环形网络实现
//

#include <sst/core/sst_config.h>
#include "OptimizedInternalRing.h"
#include "SpikeEvent.h"

#include <algorithm>
#include <cmath>
#include <limits>

using namespace SST;
using namespace SST::SnnDL;

// ===== RingNode 实现 =====

void RingNode::initializeVCs(int num_vcs_per_direction, uint32_t credits_per_vc) {
    // 初始化顺时针虚拟通道
    cw_vcs.clear();
    cw_vcs.reserve(num_vcs_per_direction);
    for (int i = 0; i < num_vcs_per_direction; i++) {
        cw_vcs.emplace_back(i, i, credits_per_vc);  // VC ID = priority for simplicity
    }
    
    // 初始化逆时针虚拟通道
    ccw_vcs.clear();
    ccw_vcs.reserve(num_vcs_per_direction);
    for (int i = 0; i < num_vcs_per_direction; i++) {
        ccw_vcs.emplace_back(i, i, credits_per_vc);
    }
    
    // 初始化本地虚拟通道
    local_vcs.clear();
    local_vcs.reserve(num_vcs_per_direction);
    for (int i = 0; i < num_vcs_per_direction; i++) {
        local_vcs.emplace_back(i, i, credits_per_vc);
    }
}

VirtualChannel* RingNode::selectOutputVC(RouteDirection direction, int priority) {
    std::vector<VirtualChannel>* vcs = nullptr;
    
    switch (direction) {
        case RouteDirection::CLOCKWISE:
            vcs = &cw_vcs;
            break;
        case RouteDirection::COUNTER_CLOCKWISE:
            vcs = &ccw_vcs;
            break;
        case RouteDirection::LOCAL:
            vcs = &local_vcs;
            break;
        default:
            return nullptr;
    }
    
    // 首先尝试找到匹配优先级且有空间的VC
    for (auto& vc : *vcs) {
        if (vc.priority == priority && vc.hasSpace()) {
            return &vc;
        }
    }
    
    // 如果没有匹配的，找到任何有空间的VC
    for (auto& vc : *vcs) {
        if (vc.hasSpace()) {
            return &vc;
        }
    }
    
    return nullptr;  // 没有可用的VC
}

bool RingNode::canAcceptMessage(RouteDirection direction, int priority) const {
    const std::vector<VirtualChannel>* vcs = nullptr;
    
    switch (direction) {
        case RouteDirection::CLOCKWISE:
            vcs = &cw_vcs;
            break;
        case RouteDirection::COUNTER_CLOCKWISE:
            vcs = &ccw_vcs;
            break;
        case RouteDirection::LOCAL:
            vcs = &local_vcs;
            break;
        default:
            return false;
    }
    
    // 检查是否有任何VC可以接受这个优先级的消息
    for (const auto& vc : *vcs) {
        if (vc.priority <= priority && vc.hasSpace()) {
            return true;
        }
    }
    
    return false;
}

// ===== OptimizedInternalRing 实现 =====

OptimizedInternalRing::OptimizedInternalRing(int num_nodes, int num_vcs, 
                                           uint32_t credits_per_vc, SST::Output* output)
    : num_nodes_(num_nodes), num_vcs_(num_vcs), credits_per_vc_(credits_per_vc), 
      output_(output), last_stats_cycle_(0) {
    
    if (output_) {
        // output_->verbose(CALL_INFO, 1, 0, "🔗 初始化优化的内部环形网络: %d节点, %d VCs, %d信用/VC\n",
        //                 num_nodes_, num_vcs_, credits_per_vc_);
    }
    
    // 验证参数
    if (num_nodes_ < 2) {
        if (output_) {
            output_->fatal(CALL_INFO, -1, "❌ 环形网络至少需要2个节点，当前: %d\n", num_nodes_);
        }
    }
    
    if (num_vcs_ < 1) {
        if (output_) {
            output_->fatal(CALL_INFO, -1, "❌ 至少需要1个虚拟通道，当前: %d\n", num_vcs_);
        }
    }
    
    // 初始化节点
    nodes_.reserve(num_nodes_);
    for (int i = 0; i < num_nodes_; i++) {
        nodes_.emplace_back(std::make_unique<RingNode>(i));
        nodes_[i]->initializeVCs(num_vcs_, credits_per_vc_);
    }
    
    // 初始化环形拓扑
    initializeTopology();
    
    // 初始化路由缓存
    route_cache_.resize(num_nodes_ * num_nodes_, 0);
    
    if (output_) {
        // output_->verbose(CALL_INFO, 1, 0, "✅ 优化的环形网络初始化完成\n");
    }
}

OptimizedInternalRing::~OptimizedInternalRing() {
    if (output_) {
        // output_->verbose(CALL_INFO, 1, 0, "🗑️ 销毁优化的环形网络\n");
        
        // 输出最终统计信息
        // output_->verbose(CALL_INFO, 1, 0, "📊 最终统计: 总消息数=%" PRIu64 ", 平均延迟=%.2f周期\n",
        //                 total_messages_routed_.load(), getAverageLatency());
    }
    
    // 清理所有节点（unique_ptr会自动清理）
    nodes_.clear();
}

void OptimizedInternalRing::initializeTopology() {
    if (num_nodes_ < 2) return;
    
    // 构建双向环形拓扑
    for (int i = 0; i < num_nodes_; i++) {
        RingNode* current = nodes_[i].get();
        
        // 顺时针连接
        current->next_cw = nodes_[(i + 1) % num_nodes_].get();
        current->prev_cw = nodes_[(i + num_nodes_ - 1) % num_nodes_].get();
        
        // 逆时针连接（反向）
        current->next_ccw = nodes_[(i + num_nodes_ - 1) % num_nodes_].get();
        current->prev_ccw = nodes_[(i + 1) % num_nodes_].get();
    }
    
    if (output_) {
        // output_->verbose(CALL_INFO, 2, 0, "🔗 环形拓扑初始化完成: %d节点双向环\n", num_nodes_);
    }
    
    // 验证拓扑完整性
    if (!verifyTopology()) {
        if (output_) {
            output_->fatal(CALL_INFO, -1, "❌ 环形拓扑验证失败\n");
        }
    }
}

bool OptimizedInternalRing::sendMessage(int src_node, int dst_node, const RingMessage& message, int priority) {
    // 参数验证
    if (src_node < 0 || src_node >= num_nodes_ || dst_node < 0 || dst_node >= num_nodes_) {
        if (output_) {
            output_->verbose(CALL_INFO, 1, 0, "⚠️ 无效的节点ID: src=%d, dst=%d\n", src_node, dst_node);
        }
        return false;
    }
    
    RingNode* src = getNode(src_node);
    if (!src) return false;
    
    // 如果是本地消息，直接放入弹出队列
    if (src_node == dst_node) {
        src->ejection_queue.push(message);
        src->messages_ejected++;
        return true;
    }
    
    // 选择路由方向
    RouteDirection route_dir = selectRoute(src_node, dst_node);
    if (route_dir == RouteDirection::INVALID) {
        if (output_) {
            output_->verbose(CALL_INFO, 1, 0, "⚠️ 无法路由消息: src=%d, dst=%d\n", src_node, dst_node);
        }
        return false;
    }
    
    // 选择虚拟通道
    VirtualChannel* vc = src->selectOutputVC(route_dir, priority);
    if (!vc) {
        if (output_) {
            output_->verbose(CALL_INFO, 3, 0, "⚠️ 节点%d无可用VC，方向=%d，优先级=%d\n", 
                           src_node, static_cast<int>(route_dir), priority);
        }
        return false;  // 背压：没有可用的VC
    }
    
    // 创建带有路由信息的消息副本
    RingMessage routed_msg = message;
    routed_msg.src_unit = src_node;
    routed_msg.dst_unit = dst_node;
    routed_msg.timestamp = total_cycles_.load();
    
    // 将消息加入VC缓冲区
    vc->buffer.push(routed_msg);
    vc->consumeCredit();
    vc->state = VCState::ACTIVE;
    vc->last_activity_cycle = total_cycles_.load();
    
    // 更新统计
    src->messages_injected++;
    
    if (output_) {
        output_->verbose(CALL_INFO, 4, 0, "📤 消息注入: 节点%d->%d, VC%d, 方向=%d\n",
                        src_node, dst_node, vc->vc_id, static_cast<int>(route_dir));
    }
    
    return true;
}

bool OptimizedInternalRing::receiveMessage(int node_id, RingMessage& message) {
    RingNode* node = getNode(node_id);
    if (!node || node->ejection_queue.empty()) {
        return false;
    }
    
    message = node->ejection_queue.front();
    node->ejection_queue.pop();
    
    // 更新延迟统计
    uint64_t current_cycle = total_cycles_.load();
    if (current_cycle >= message.timestamp) {
        uint64_t latency = current_cycle - message.timestamp;
        node->total_latency_cycles += latency;
        total_latency_cycles_.fetch_add(latency);
    }
    
    if (output_) {
        output_->verbose(CALL_INFO, 4, 0, "📥 消息弹出: 节点%d, 延迟=%" PRIu64 "周期\n",
                        node_id, current_cycle - message.timestamp);
    }
    
    return true;
}

void OptimizedInternalRing::tick(uint64_t current_cycle) {
    total_cycles_.store(current_cycle);
    
    // 处理每个节点的路由
    for (int i = 0; i < num_nodes_; i++) {
        processNodeRouting(i, current_cycle);
    }
    
    // 定期更新统计信息
    if (current_cycle - last_stats_cycle_ >= 1000) {
        updateStatistics(current_cycle);
        last_stats_cycle_ = current_cycle;
    }
}

RouteDirection OptimizedInternalRing::selectRoute(int src, int dst) const {
    if (src == dst) {
        return RouteDirection::LOCAL;
    }
    
    // 检查缓存
    uint64_t cache_key = generateRouteCacheKey(src, dst);
    auto cache_it = route_lookup_cache_.find(cache_key);
    if (cache_it != route_lookup_cache_.end()) {
        return cache_it->second;
    }
    
    // 计算两个方向的跳数
    int cw_hops = calculateHops(src, dst, RouteDirection::CLOCKWISE);
    int ccw_hops = calculateHops(src, dst, RouteDirection::COUNTER_CLOCKWISE);
    
    // 选择跳数更少的方向，相等时优先选择顺时针
    RouteDirection selected = (cw_hops <= ccw_hops) ? RouteDirection::CLOCKWISE : RouteDirection::COUNTER_CLOCKWISE;
    
    // 缓存路由决策
    route_lookup_cache_[cache_key] = selected;
    
    return selected;
}

int OptimizedInternalRing::calculateHops(int src, int dst, RouteDirection direction) const {
    if (src == dst) return 0;
    
    int hops = 0;
    
    if (direction == RouteDirection::CLOCKWISE) {
        if (dst > src) {
            hops = dst - src;
        } else {
            hops = num_nodes_ - src + dst;
        }
    } else if (direction == RouteDirection::COUNTER_CLOCKWISE) {
        if (src > dst) {
            hops = src - dst;
        } else {
            hops = num_nodes_ - dst + src;
        }
    }
    
    return hops;
}

void OptimizedInternalRing::processNodeRouting(int node_id, uint64_t current_cycle) {
    RingNode* node = getNode(node_id);
    if (!node) return;
    
    // 处理各个方向的虚拟通道
    processDirectionVCs(node, RouteDirection::CLOCKWISE, current_cycle);
    processDirectionVCs(node, RouteDirection::COUNTER_CLOCKWISE, current_cycle);
    
    // 处理注入队列
    processInjectionQueue(node, current_cycle);
}

void OptimizedInternalRing::processDirectionVCs(RingNode* node, RouteDirection direction, uint64_t current_cycle) {
    std::vector<VirtualChannel>& vcs = getVCs(node, direction);
    
    // VC仲裁：选择一个VC进行服务
    int selected_vc = vcArbitration(vcs);
    if (selected_vc == -1) return;  // 没有可服务的VC
    
    VirtualChannel& vc = vcs[selected_vc];
    if (!vc.hasData()) return;
    
    RingMessage& msg = vc.buffer.front();
    
    // 检查是否到达目标
    if (msg.dst_unit == node->node_id) {
        // 消息到达目标，弹出到本地
        node->ejection_queue.push(msg);
        vc.buffer.pop();
        vc.returnCredit();
        node->messages_ejected++;
        
        if (output_) {
            output_->verbose(CALL_INFO, 4, 0, "🎯 消息到达目标: 节点%d\n", node->node_id);
        }
        return;
    }
    
    // 需要继续转发，确定下一跳方向
    RouteDirection next_dir = selectRoute(node->node_id, msg.dst_unit);
    if (next_dir == RouteDirection::INVALID) {
        // 路由失败，丢弃消息
        vc.buffer.pop();
        vc.returnCredit();
        if (output_) {
            output_->verbose(CALL_INFO, 1, 0, "⚠️ 路由失败，丢弃消息: 当前节点%d, 目标%d\n", 
                           node->node_id, msg.dst_unit);
        }
        return;
    }
    
    // 转发消息
    if (forwardMessage(node, msg, next_dir)) {
        vc.buffer.pop();
        vc.returnCredit();
        node->messages_forwarded++;
        
        if (output_) {
            output_->verbose(CALL_INFO, 4, 0, "🔄 消息转发: 节点%d, 方向=%d\n", 
                           node->node_id, static_cast<int>(next_dir));
        }
    }
    // 如果转发失败，消息保留在当前VC中等待下个周期
}

bool OptimizedInternalRing::forwardMessage(RingNode* node, const RingMessage& message, RouteDirection direction) {
    RingNode* next_node = nullptr;
    
    switch (direction) {
        case RouteDirection::CLOCKWISE:
            next_node = node->next_cw;
            break;
        case RouteDirection::COUNTER_CLOCKWISE:
            next_node = node->next_ccw;
            break;
        case RouteDirection::LOCAL:
            // 本地消息，直接加入弹出队列
            node->ejection_queue.push(message);
            return true;
        default:
            return false;
    }
    
    if (!next_node) return false;
    
    // 检查下一个节点是否能接受消息
    if (!next_node->canAcceptMessage(direction, message.priority)) {
        return false;  // 背压：下一节点无法接受
    }
    
    // 选择下一节点的虚拟通道
    VirtualChannel* next_vc = next_node->selectOutputVC(direction, message.priority);
    if (!next_vc) {
        return false;  // 没有可用的VC
    }
    
    // 转发消息
    next_vc->buffer.push(message);
    next_vc->consumeCredit();
    next_vc->state = VCState::ACTIVE;
    next_vc->last_activity_cycle = total_cycles_.load();
    
    return true;
}

int OptimizedInternalRing::vcArbitration(const std::vector<VirtualChannel>& vcs) const {
    // 轮询仲裁：优先服务优先级高且有数据的VC
    int best_vc = -1;
    int highest_priority = std::numeric_limits<int>::max();
    
    for (size_t i = 0; i < vcs.size(); i++) {
        const VirtualChannel& vc = vcs[i];
        if (vc.hasData() && vc.priority < highest_priority) {
            highest_priority = vc.priority;
            best_vc = static_cast<int>(i);
        }
    }
    
    return best_vc;
}

std::vector<VirtualChannel>& OptimizedInternalRing::getVCs(RingNode* node, RouteDirection direction) {
    switch (direction) {
        case RouteDirection::CLOCKWISE:
            return node->cw_vcs;
        case RouteDirection::COUNTER_CLOCKWISE:
            return node->ccw_vcs;
        case RouteDirection::LOCAL:
            return node->local_vcs;
        default:
            // 返回一个静态的空vector作为fallback
            static std::vector<VirtualChannel> empty_vcs;
            return empty_vcs;
    }
}

const std::vector<VirtualChannel>& OptimizedInternalRing::getVCs(const RingNode* node, RouteDirection direction) const {
    switch (direction) {
        case RouteDirection::CLOCKWISE:
            return node->cw_vcs;
        case RouteDirection::COUNTER_CLOCKWISE:
            return node->ccw_vcs;
        case RouteDirection::LOCAL:
            return node->local_vcs;
        default:
            static const std::vector<VirtualChannel> empty_vcs;
            return empty_vcs;
    }
}

bool OptimizedInternalRing::hasTrafficForNode(int node_id) const {
    const RingNode* node = getNode(node_id);
    if (!node) return false;
    
    return !node->ejection_queue.empty();
}

int OptimizedInternalRing::getPendingMessageCount() const {
    int total = 0;
    
    for (const auto& node_ptr : nodes_) {
        const RingNode* node = node_ptr.get();
        
        // 计算所有VC中的消息
        for (const auto& vc : node->cw_vcs) {
            total += static_cast<int>(vc.buffer.size());
        }
        for (const auto& vc : node->ccw_vcs) {
            total += static_cast<int>(vc.buffer.size());
        }
        for (const auto& vc : node->local_vcs) {
            total += static_cast<int>(vc.buffer.size());
        }
        
        // 计算注入和弹出队列
        total += static_cast<int>(node->injection_queue.size());
        total += static_cast<int>(node->ejection_queue.size());
    }
    
    return total;
}

double OptimizedInternalRing::getAverageLatency() const {
    uint64_t total_msgs = total_messages_routed_.load();
    uint64_t total_lat = total_latency_cycles_.load();
    
    if (total_msgs == 0) return 0.0;
    return static_cast<double>(total_lat) / static_cast<double>(total_msgs);
}

double OptimizedInternalRing::getNetworkUtilization() const {
    uint64_t total_cycles = total_cycles_.load();
    if (total_cycles == 0) return 0.0;
    
    // 计算所有VC的使用率
    uint64_t active_vc_cycles = 0;
    uint64_t total_vc_capacity = num_nodes_ * num_vcs_ * 2;  // 2个方向
    
    for (const auto& node_ptr : nodes_) {
        const RingNode* node = node_ptr.get();
        
        for (const auto& vc : node->cw_vcs) {
            if (vc.state == VCState::ACTIVE) {
                active_vc_cycles += (total_cycles - vc.last_activity_cycle);
            }
        }
        for (const auto& vc : node->ccw_vcs) {
            if (vc.state == VCState::ACTIVE) {
                active_vc_cycles += (total_cycles - vc.last_activity_cycle);
            }
        }
    }
    
    return static_cast<double>(active_vc_cycles) / static_cast<double>(total_vc_capacity * total_cycles);
}

void OptimizedInternalRing::updateStatistics(uint64_t current_cycle) {
    // 更新全局统计
    uint64_t total_routed = 0;
    for (const auto& node_ptr : nodes_) {
        total_routed += node_ptr->messages_forwarded + node_ptr->messages_ejected;
    }
    total_messages_routed_.store(total_routed);
    
    if (output_ && current_cycle % 10000 == 0) {
        // output_->verbose(CALL_INFO, 2, 0, "📊 环形网络统计[周期%" PRIu64 "]: 消息=%" PRIu64 ", 延迟=%.2f, 利用率=%.2f%%\n",
        //                 current_cycle, total_routed, getAverageLatency(), getNetworkUtilization() * 100.0);
    }
}

bool OptimizedInternalRing::verifyTopology() const {
    for (int i = 0; i < num_nodes_; i++) {
        const RingNode* node = nodes_[i].get();
        
        // 检查顺时针连接
        if (!node->next_cw || !node->prev_cw) {
            if (output_) {
                output_->verbose(CALL_INFO, 1, 0, "❌ 节点%d顺时针连接不完整\n", i);
            }
            return false;
        }
        
        // 检查逆时针连接
        if (!node->next_ccw || !node->prev_ccw) {
            if (output_) {
                output_->verbose(CALL_INFO, 1, 0, "❌ 节点%d逆时针连接不完整\n", i);
            }
            return false;
        }
        
        // 检查连接的一致性
        if (node->next_cw->prev_cw != node || node->prev_cw->next_cw != node) {
            if (output_) {
                output_->verbose(CALL_INFO, 1, 0, "❌ 节点%d顺时针连接不一致\n", i);
            }
            return false;
        }
    }
    
    return true;
}

void OptimizedInternalRing::processInjectionQueue(RingNode* node, uint64_t current_cycle) {
    // 这个方法为将来扩展预留，当前版本中注入通过sendMessage直接处理
    // 可以在这里实现更复杂的注入控制逻辑
    (void)node;        // 避免未使用参数警告
    (void)current_cycle;
}

bool OptimizedInternalRing::detectDeadlock() const {
    // 简单的死锁检测：检查是否所有VC都被阻塞且有数据
    for (const auto& node_ptr : nodes_) {
        const RingNode* node = node_ptr.get();
        
        // 检查顺时针VC
        for (const auto& vc : node->cw_vcs) {
            if (vc.hasData() && !vc.hasSpace()) {
                // 找到一个被阻塞的VC，检查其他节点是否也被阻塞
                bool all_blocked = true;
                for (const auto& other_node_ptr : nodes_) {
                    const RingNode* other_node = other_node_ptr.get();
                    for (const auto& other_vc : other_node->cw_vcs) {
                        if (!other_vc.hasData() || other_vc.hasSpace()) {
                            all_blocked = false;
                            break;
                        }
                    }
                    if (!all_blocked) break;
                }
                
                if (all_blocked) {
                    if (output_) {
                        output_->verbose(CALL_INFO, 1, 0, "⚠️ 检测到潜在死锁\n");
                    }
                    return true;
                }
            }
        }
    }
    
    return false;
}

void OptimizedInternalRing::printNetworkState() const {
    if (!output_) return;
    
    // output_->verbose(CALL_INFO, 1, 0, "=== 环形网络状态 ===\n");
    // for (int i = 0; i < num_nodes_; i++) {
    //     const RingNode* node = nodes_[i].get();
    //     output_->verbose(CALL_INFO, 1, 0, "节点%d: 注入=%" PRIu64 ", 弹出=%" PRIu64 ", 转发=%" PRIu64 "\n",
    //                     i, node->messages_injected, node->messages_ejected, node->messages_forwarded);
        
    //     // 显示VC状态
    //     for (size_t j = 0; j < node->cw_vcs.size(); j++) {
    //         const VirtualChannel& vc = node->cw_vcs[j];
    //         output_->verbose(CALL_INFO, 1, 0, "  CW_VC%zu: 缓冲=%zu, 信用=%u\n",
    //                        j, vc.buffer.size(), vc.credits);
    //     }
    // }
    // output_->verbose(CALL_INFO, 1, 0, "==================\n");
}

// 添加缺失的方法实现
bool OptimizedInternalRing::checkCredit(int node_id, RouteDirection direction, int vc_id) const {
    const RingNode* node = getNode(node_id);
    if (!node) return false;
    
    const std::vector<VirtualChannel>& vcs = getVCs(node, direction);
    if (vc_id < 0 || vc_id >= static_cast<int>(vcs.size())) return false;
    
    return vcs[vc_id].credits > 0;
}

void OptimizedInternalRing::updateCredit(int node_id, RouteDirection direction, int vc_id, bool increment) {
    RingNode* node = getNode(node_id);
    if (!node) return;
    
    std::vector<VirtualChannel>& vcs = getVCs(node, direction);
    if (vc_id < 0 || vc_id >= static_cast<int>(vcs.size())) return;
    
    if (increment) {
        vcs[vc_id].returnCredit();
    } else {
        vcs[vc_id].consumeCredit();
    }
}

void OptimizedInternalRing::getNodeStatistics(int node_id, uint64_t& injected, uint64_t& ejected, 
                                             uint64_t& forwarded, double& avg_latency) const {
    const RingNode* node = getNode(node_id);
    if (!node) {
        injected = ejected = forwarded = 0;
        avg_latency = 0.0;
        return;
    }
    
    injected = node->messages_injected;
    ejected = node->messages_ejected;
    forwarded = node->messages_forwarded;
    
    if (ejected > 0) {
        avg_latency = static_cast<double>(node->total_latency_cycles) / static_cast<double>(ejected);
    } else {
        avg_latency = 0.0;
    }
}

double OptimizedInternalRing::getVCUtilization(int node_id, RouteDirection direction, int vc_id) const {
    const RingNode* node = getNode(node_id);
    if (!node) return 0.0;
    
    const std::vector<VirtualChannel>& vcs = getVCs(node, direction);
    if (vc_id < 0 || vc_id >= static_cast<int>(vcs.size())) return 0.0;
    
    const VirtualChannel& vc = vcs[vc_id];
    uint64_t total_cycles = total_cycles_.load();
    
    if (total_cycles == 0) return 0.0;
    
    // 计算VC的使用率（基于活动时间）
    uint64_t active_cycles = (vc.state == VCState::ACTIVE) ? 
                            (total_cycles - vc.last_activity_cycle) : 0;
    
    return static_cast<double>(active_cycles) / static_cast<double>(total_cycles);
}