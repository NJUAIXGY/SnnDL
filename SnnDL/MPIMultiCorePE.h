// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// MPIMultiCorePE.h: MPI增强的多核处理单元头文件
// 
// 重要：这个组件通过继承扩展现有MultiCorePE功能，不修改任何原有代码
// 确保完全的向后兼容性
//

#ifndef _MPI_MULTICORE_PE_H
#define _MPI_MULTICORE_PE_H

// 包含原有MultiCorePE头文件
#include "MultiCorePE.h"
#include "MPITypes.h"

// 使用SST内置MPI支持，不直接依赖OpenMPI

#include <memory>
#include <unordered_map>
#include <thread>
#include <mutex>
#include <atomic>

namespace SST {
namespace SnnDL {

/**
 * @brief MPI增强的多核处理单元
 * 
 * 这个类通过继承MultiCorePE来扩展MPI功能，而不修改原有实现。
 * 所有原有的MultiCorePE功能都完全保留，同时添加跨节点通信能力。
 * 
 * 兼容性保证：
 * 1. 如果系统没有MPI支持，这个组件会退化为标准MultiCorePE
 * 2. 所有原有的参数和接口都完全兼容
 * 3. 现有的测试和配置文件无需修改
 */
class MPIMultiCorePE : public MultiCorePE {
public:
    // ===== ELI注册信息 =====
    SST_ELI_REGISTER_COMPONENT(
        MPIMultiCorePE,
        "SnnDL",
        "MPIMultiCorePE",
        SST_ELI_ELEMENT_VERSION(1,0,0),
        "MPI增强的多核脉冲神经网络处理单元（兼容MultiCorePE）",
        COMPONENT_CATEGORY_PROCESSOR
    )
    
    // 继承所有父类参数，并添加MPI相关参数
    SST_ELI_DOCUMENT_PARAMS(
        // 所有MultiCorePE的参数都会自动继承
        {"enable_mpi",       "是否启用MPI功能 (0=关闭, 1=启用)", "0"},
        {"mpi_data_parallel", "是否启用MPI数据并行", "1"},
        {"mpi_model_parallel", "是否启用MPI模型并行", "0"},
        {"mpi_comm_strategy", "MPI通信策略 (eager|rendezvous|adaptive)", "eager"},
        {"mpi_buffer_size",  "MPI通信缓冲区大小 (bytes)", "65536"},
        {"mpi_async_threshold", "异步通信阈值 (消息数)", "10"},
        {"mpi_heartbeat_interval", "心跳间隔 (毫秒)", "100"},
        {"mpi_verbose",      "MPI调试输出级别", "0"}
    )
    
    // 继承所有父类的子组件槽位
    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS(
        // MultiCorePE的所有子组件会自动继承
        {"mpi_comm_manager", "MPI通信管理器", "SST::SnnDL::MPICommManager"}
    )
    
    // 无新增端口：复用父类端口；所有跨节点通信经由SnnNIC/merlin网络实现
    
    // 明确声明所有统计：父类统计 + MPI扩展统计
    SST_ELI_DOCUMENT_STATISTICS(
        // === 继承自MultiCorePE的基础统计 ===
        {"total_spikes_processed", "处理的脉冲总数", "spikes", 1},
        {"inter_core_messages", "核间消息数量", "messages", 1},
        {"l2_cache_hits", "L2缓存命中数", "hits", 1},
        {"l2_cache_misses", "L2缓存缺失数", "misses", 1},
        {"memory_requests", "内存请求数", "requests", 1},
        {"avg_core_utilization", "平均核心利用率", "percentage", 1},
        {"total_neurons_fired", "总神经元发放数", "neurons", 1},
        {"external_spikes_sent", "发送的外部脉冲数", "spikes", 1},
        {"external_spikes_received", "接收的外部脉冲数", "spikes", 1},
        // === MPI扩展统计 ===
        {"mpi_messages_sent",     "MPI发送消息数", "messages", 1},
        {"mpi_messages_received", "MPI接收消息数", "messages", 1},
        {"mpi_bytes_sent",        "MPI发送字节数", "bytes", 1},
        {"mpi_bytes_received",    "MPI接收字节数", "bytes", 1},
        {"mpi_comm_time",         "MPI通信总时间", "seconds", 1},
        {"mpi_sync_time",         "MPI同步等待时间", "seconds", 1},
        {"mpi_load_imbalance",    "MPI负载不均衡度", "ratio", 1}
    )

public:
    /**
     * @brief 构造函数
     * 
     * 调用父类MultiCorePE的构造函数，然后添加MPI相关初始化
     */
    MPIMultiCorePE(ComponentId_t id, Params& params);
    
    /**
     * @brief 析构函数
     */
    ~MPIMultiCorePE();
    
    // ===== 重写父类的生命周期方法 =====
    
    /**
     * @brief 初始化阶段
     * 
     * 先调用父类的init，然后进行MPI相关初始化
     */
    void init(unsigned int phase) override;
    
    /**
     * @brief 设置阶段
     * 
     * 先调用父类的setup，然后进行MPI相关设置
     */
    void setup() override;
    
    /**
     * @brief 完成阶段
     * 
     * 进行MPI相关清理，然后调用父类的finish
     */
    void finish() override;

    // ===== MPI扩展功能接口 =====
    
    /**
     * @brief 检查MPI是否启用和可用
     */
    bool isMPIEnabled() const { return mpi_enabled_ && mpi_available_; }
    
    /**
     * @brief 获取MPI rank信息
     */
    int getMPIRank() const { return mpi_rank_; }
    int getMPISize() const { return mpi_size_; }
    
    /**
     * @brief 发送跨节点脉冲事件
     */
    void sendInterNodeSpike(SpikeEvent* spike, int target_rank);
    
    /**
     * @brief 广播脉冲事件到所有节点
     */
    void broadcastSpike(SpikeEvent* spike);
    
    /**
     * @brief MPI全局同步
     */
    void mpiBarrier();
    
    /**
     * @brief 获取MPI性能统计
     */
    void getMPIStats(std::map<std::string, uint64_t>& stats) const;

protected:
    // ===== MPI相关配置参数 =====
    bool mpi_enabled_;              // MPI功能是否启用
    bool mpi_available_;            // MPI是否实际可用
    bool mpi_data_parallel_;        // 数据并行模式
    bool mpi_model_parallel_;       // 模型并行模式
    std::string mpi_comm_strategy_; // 通信策略
    size_t mpi_buffer_size_;        // 缓冲区大小
    int mpi_async_threshold_;       // 异步通信阈值
    int mpi_heartbeat_interval_;    // 心跳间隔
    int mpi_verbose_;               // MPI调试级别

    // ===== SST MPI仿真状态 =====
    MPITypes::MPIConfig mpi_config_;     // MPI配置
    int mpi_rank_;                       // 当前rank
    int mpi_size_;                       // 总rank数
    
    // ===== MPI工具对象 =====
    std::unique_ptr<SSTMPICommHelper> sst_mpi_comm_helper_;
    
    // ===== MPI统计信息 =====
    Statistic<uint64_t>* stat_mpi_messages_sent_;
    Statistic<uint64_t>* stat_mpi_messages_received_;
    Statistic<uint64_t>* stat_mpi_bytes_sent_;
    Statistic<uint64_t>* stat_mpi_bytes_received_;
    Statistic<double>* stat_mpi_comm_time_;
    Statistic<double>* stat_mpi_sync_time_;
    Statistic<double>* stat_mpi_load_imbalance_;
    
private:
    /**
     * @brief 初始化SST MPI仿真环境
     */
    bool initializeSSTMPI();
    
    /**
     * @brief 清理SST MPI环境
     */
    void finalizeSSTMPI();
    
    /**
     * @brief 初始化MPI统计收集
     */
    void initializeMPIStatistics();
};

/**
 * @brief MPI通信管理器子组件
 * 
 * 可选的子组件，用于更高级的MPI通信管理
 */
class MPICommManager : public SST::SubComponent {
public:
    SST_ELI_REGISTER_SUBCOMPONENT_API(SST::SnnDL::MPICommManager)
    
    SST_ELI_REGISTER_SUBCOMPONENT(
        MPICommManager,
        "SnnDL",
        "MPICommManager",
        SST_ELI_ELEMENT_VERSION(1,0,0),
        "MPI通信管理器子组件",
        SST::SnnDL::MPICommManager
    )
    
    SST_ELI_DOCUMENT_PARAMS(
        {"strategy", "通信策略", "default"}
    )

public:
    MPICommManager(ComponentId_t id, Params& params);
    virtual ~MPICommManager() = default;
    
    virtual void init(unsigned int phase) {}
    virtual void setup() {}
    virtual void finish() {}

protected:
    std::string strategy_;
};

} // namespace SnnDL
} // namespace SST

#endif // _MPI_MULTICORE_PE_H
