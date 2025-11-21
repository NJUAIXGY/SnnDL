// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// MPITypes.cc: SST MPI类型实现
// 基于SST框架的MPI仿真支持，不直接依赖OpenMPI
//

#include "MPITypes.h"
#include "SpikeEvent.h"
#include <iostream>
#include <sst/core/params.h>

namespace SST {
namespace SnnDL {

// ===== MPITypes 静态成员初始化 =====
MPITypes::MPIConfig MPITypes::config_;

bool MPITypes::initializeMPIConfig(SST::Params& params) {
    config_.enabled = params.find<bool>("enable_mpi", false);
    config_.rank = params.find<int>("mpi_rank", 0);
    config_.size = params.find<int>("mpi_size", 1);
    config_.comm_pattern = params.find<std::string>("mpi_comm_strategy", "eager");
    config_.buffer_size = params.find<size_t>("mpi_buffer_size", 65536);
    
    #ifdef SNNDL_ENABLE_DEBUG_LOG
    if (config_.enabled) {
        std::cout << "MPITypes: 配置已启用 - rank=" << config_.rank 
                  << ", size=" << config_.size 
                  << ", pattern=" << config_.comm_pattern << std::endl;
    }
    #endif
    
    return true;
}

// ===== SSTMPICommHelper 实现 =====

SSTMPICommHelper::SSTMPICommHelper(SST::Component* parent) 
    : parent_component_(parent), initialized_(false) {
}

SSTMPICommHelper::~SSTMPICommHelper() {
}

bool SSTMPICommHelper::initialize(const MPITypes::MPIConfig& config) {
    config_ = config;
    
    if (!config_.enabled) {
        return true; // 如果MPI未启用，返回成功但不初始化
    }
    
    // 为每个其他rank创建通信链路
    node_links_.resize(config_.size);
    for (int i = 0; i < config_.size; ++i) {
        if (i != config_.rank) {
            std::string link_name = "mpi_link_" + std::to_string(i);
            // 注意：configureLink是protected方法，需要在组件内部调用
            // 这里先创建占位符，实际的Link配置需要在MPIMultiCorePE中完成
            node_links_[i] = nullptr;  // 待后续配置
            
            // 暂时跳过Link检查，实际配置在组件内部完成
        }
    }
    
    initialized_ = true;
    #ifdef SNNDL_ENABLE_DEBUG_LOG
    std::cout << "SSTMPICommHelper: 初始化成功 - rank=" << config_.rank << std::endl;
    #endif
    return true;
}

bool SSTMPICommHelper::sendSpikeEvent(SpikeEvent* spike, int dest_rank, int tag) {
    if (!initialized_ || !config_.enabled || dest_rank >= config_.size || dest_rank == config_.rank) {
        return false;
    }
    
    // 通过SST Link发送事件
    if (node_links_[dest_rank]) {
        node_links_[dest_rank]->send(spike);
        SSTMPIPerformanceMonitor::recordMessageSent(sizeof(SpikeEvent));
        return true;
    }
    
    return false;
}

bool SSTMPICommHelper::broadcastSpikeEvent(SpikeEvent* spike) {
    if (!initialized_ || !config_.enabled) {
        return false;
    }
    
    bool success = true;
    for (int i = 0; i < config_.size; ++i) {
        if (i != config_.rank) {
            success &= sendSpikeEvent(spike, i, MPITypes::SPIKE_TAG);
        }
    }
    
    return success;
}

void SSTMPICommHelper::handleIncomingMessage(SST::Event* ev) {
    if (!ev) return;
    
    // 处理接收到的脉冲事件
    SpikeEvent* spike = dynamic_cast<SpikeEvent*>(ev);
    if (spike) {
        SSTMPIPerformanceMonitor::recordMessageReceived(sizeof(SpikeEvent));
        // 这里应该将脉冲传递给父组件处理
        // parent_component_->handleRemoteSpike(spike);
    }
}

bool SSTMPICommHelper::barrier() {
    if (!initialized_ || !config_.enabled) {
        return true; // 单节点模式下直接返回成功
    }
    
    // 简单的屏障实现：发送同步消息给所有其他节点
    // 实际实现需要更复杂的协调机制
    SSTMPIPerformanceMonitor::recordBarrierCall(0.001); // 假设1ms延迟
    return true;
}

// ===== SSTMPIPerformanceMonitor 实现 =====

SSTMPIPerformanceMonitor::MPIStats SSTMPIPerformanceMonitor::stats_;

void SSTMPIPerformanceMonitor::recordMessageSent(size_t bytes) {
    stats_.messages_sent++;
    stats_.bytes_sent += bytes;
}

void SSTMPIPerformanceMonitor::recordMessageReceived(size_t bytes) {
    stats_.messages_received++;
    stats_.bytes_received += bytes;
}

void SSTMPIPerformanceMonitor::recordCommunicationTime(double time) {
    stats_.total_comm_time += time;
}

void SSTMPIPerformanceMonitor::recordBarrierCall(double time) {
    stats_.barrier_calls++;
    stats_.total_barrier_time += time;
}

void SSTMPIPerformanceMonitor::printStats() {
    #ifdef SNNDL_ENABLE_DEBUG_LOG
    std::cout << "=== MPI性能统计 ===" << std::endl;
    std::cout << "发送消息数: " << stats_.messages_sent << std::endl;
    std::cout << "接收消息数: " << stats_.messages_received << std::endl;
    std::cout << "发送字节数: " << stats_.bytes_sent << std::endl;
    std::cout << "接收字节数: " << stats_.bytes_received << std::endl;
    std::cout << "通信总时间: " << stats_.total_comm_time << "s" << std::endl;
    std::cout << "屏障调用次数: " << stats_.barrier_calls << std::endl;
    std::cout << "屏障总时间: " << stats_.total_barrier_time << "s" << std::endl;
    #endif
}

} // namespace SnnDL
} // namespace SST
