// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// MPITypes.h: MPI数据类型定义和通信工具
// 注意：这是新增的MPI扩展文件，不修改任何现有SnnDL代码
//

#ifndef _MPI_TYPES_H
#define _MPI_TYPES_H

// 使用SST内置的MPI支持，而不是直接包含mpi.h
// 我们将通过SST的API接口实现MPI功能
#include <sst/core/sst_types.h>
#include <sst/core/component.h>

#include <vector>
#include <memory>
#include <functional>
#include <chrono>

#include "SpikeEvent.h"

namespace SST {
namespace SnnDL {

// 前置声明
class SpikeEvent;

/**
 * @brief SST MPI相关类型和常量定义
 * 
 * 基于SST框架的MPI支持，不直接使用OpenMPI API
 */
class MPITypes {
public:
    // 消息类型标签定义
    static constexpr int SPIKE_TAG = 1001;
    static constexpr int CONTROL_TAG = 1002;
    static constexpr int HEARTBEAT_TAG = 1003;
    static constexpr int DATA_TAG = 1004;
    static constexpr int BARRIER_TAG = 1005;
    
    // SST MPI仿真配置
    struct MPIConfig {
        int rank = 0;
        int size = 1;
        bool enabled = false;
        std::string comm_pattern = "eager";  // eager, rendezvous
        size_t buffer_size = 65536;
    };
    
    /**
     * @brief 初始化MPI配置（基于SST参数）
     */
    static bool initializeMPIConfig(SST::Params& params);
    
    /**
     * @brief 获取当前MPI配置
     */
    static const MPIConfig& getMPIConfig() { return config_; }
    
    /**
     * @brief 检查MPI是否启用
     */
    static bool isMPIEnabled() { return config_.enabled; }

private:
    static MPIConfig config_;
};

/**
 * @brief SST MPI通信助手类
 * 
 * 通过SST Links实现跨节点通信，而非直接MPI调用
 */
class SSTMPICommHelper {
public:
    SSTMPICommHelper(SST::Component* parent);
    ~SSTMPICommHelper();
    
    /**
     * @brief 初始化通信链路
     */
    bool initialize(const MPITypes::MPIConfig& config);
    
    /**
     * @brief 发送脉冲事件到指定rank
     */
    bool sendSpikeEvent(SpikeEvent* spike, int dest_rank, int tag = MPITypes::SPIKE_TAG);
    
    /**
     * @brief 广播脉冲事件到所有节点
     */
    bool broadcastSpikeEvent(SpikeEvent* spike);
    
    /**
     * @brief 处理接收到的跨节点消息
     */
    void handleIncomingMessage(SST::Event* ev);
    
    /**
     * @brief 同步屏障（通过SST事件实现）
     */
    bool barrier();

private:
    SST::Component* parent_component_;
    std::vector<SST::Link*> node_links_;  // 到其他节点的链路
    bool initialized_;
    MPITypes::MPIConfig config_;
};

/**
 * @brief SST MPI性能监控工具
 */
class SSTMPIPerformanceMonitor {
public:
    struct MPIStats {
        uint64_t messages_sent = 0;
        uint64_t messages_received = 0;
        uint64_t bytes_sent = 0;
        uint64_t bytes_received = 0;
        double total_comm_time = 0.0;
        uint64_t barrier_calls = 0;
        double total_barrier_time = 0.0;
        
        void reset() {
            messages_sent = messages_received = 0;
            bytes_sent = bytes_received = 0;
            total_comm_time = total_barrier_time = 0.0;
            barrier_calls = 0;
        }
    };
    
    static void recordMessageSent(size_t bytes);
    static void recordMessageReceived(size_t bytes);
    static void recordCommunicationTime(double time);
    static void recordBarrierCall(double time);
    
    static const MPIStats& getStats() { return stats_; }
    static void resetStats() { stats_.reset(); }
    static void printStats();

private:
    static MPIStats stats_;
};

} // namespace SnnDL
} // namespace SST

#endif // _MPI_TYPES_H