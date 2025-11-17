// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// MPIMultiCorePE.cc: SST MPI增强的多核处理单元实现
// 基于SST框架MPI仿真，不直接依赖OpenMPI
//

#include "MPIMultiCorePE.h"
#include <iostream>
#include <sstream>

using namespace SST;
using namespace SST::SnnDL;

// ===== MPIMultiCorePE 主要实现 =====

MPIMultiCorePE::MPIMultiCorePE(ComponentId_t id, Params& params) 
    : MultiCorePE(id, params)  // 重要：首先调用父类构造函数
    , mpi_enabled_(false)
    , mpi_available_(false)
    , mpi_data_parallel_(true)
    , mpi_model_parallel_(false)
    , mpi_comm_strategy_("eager")
    , mpi_buffer_size_(65536)
    , mpi_async_threshold_(10)
    , mpi_heartbeat_interval_(100)
    , mpi_verbose_(0)
    , mpi_rank_(0)
    , mpi_size_(1)
    , stat_mpi_messages_sent_(nullptr)
    , stat_mpi_messages_received_(nullptr)
    , stat_mpi_bytes_sent_(nullptr)
    , stat_mpi_bytes_received_(nullptr)
    , stat_mpi_comm_time_(nullptr)
    , stat_mpi_sync_time_(nullptr)
    , stat_mpi_load_imbalance_(nullptr)
{
    // 读取MPI相关参数（这些参数是新增的，不影响原有参数）
    mpi_enabled_ = params.find<bool>("enable_mpi", false);
    mpi_data_parallel_ = params.find<bool>("mpi_data_parallel", true);
    mpi_model_parallel_ = params.find<bool>("mpi_model_parallel", false);
    mpi_comm_strategy_ = params.find<std::string>("mpi_comm_strategy", "eager");
    mpi_buffer_size_ = params.find<size_t>("mpi_buffer_size", 65536);
    mpi_async_threshold_ = params.find<int>("mpi_async_threshold", 10);
    mpi_heartbeat_interval_ = params.find<int>("mpi_heartbeat_interval", 100);
    mpi_verbose_ = params.find<int>("mpi_verbose", 0);

    // 初始化MPI配置
    if (mpi_enabled_) {
        MPITypes::initializeMPIConfig(params);
        mpi_config_ = MPITypes::getMPIConfig();
        mpi_rank_ = mpi_config_.rank;
        mpi_size_ = mpi_config_.size;
        mpi_available_ = true;
        
        if (mpi_verbose_ > 0) {
            std::cout << "MPIMultiCorePE: MPI enabled - rank=" << mpi_rank_ 
                      << ", size=" << mpi_size_ << std::endl;
        }
    } else {
        if (mpi_verbose_ > 0) {
            std::cout << "MPIMultiCorePE: MPI disabled, running in single-node mode" << std::endl;
        }
    }
    
    // 输出组件信息
    if (mpi_verbose_ > 1) {
        std::cout << "MPIMultiCorePE constructed with:" << std::endl;
        std::cout << "  enable_mpi: " << mpi_enabled_ << std::endl;
        std::cout << "  mpi_data_parallel: " << mpi_data_parallel_ << std::endl;
        std::cout << "  mpi_model_parallel: " << mpi_model_parallel_ << std::endl;
        std::cout << "  mpi_comm_strategy: " << mpi_comm_strategy_ << std::endl;
        std::cout << "  mpi_buffer_size: " << mpi_buffer_size_ << std::endl;
    }
}

MPIMultiCorePE::~MPIMultiCorePE() {
    finalizeSSTMPI();
}

void MPIMultiCorePE::init(unsigned int phase) {
    // 首先调用父类的init
    MultiCorePE::init(phase);
    
    // 然后初始化MPI相关功能
    if (mpi_enabled_ && phase == 0) {
        initializeSSTMPI();
    }
}

void MPIMultiCorePE::setup() {
    // 首先调用父类的setup
    MultiCorePE::setup();
    
    // MPI相关设置
    if (mpi_enabled_) {
        initializeMPIStatistics();
    }
}

void MPIMultiCorePE::finish() {
    // MPI相关清理
    if (mpi_enabled_) {
        finalizeSSTMPI();
    }
    
    // 最后调用父类的finish
    MultiCorePE::finish();
}

// ===== MPI扩展功能实现 =====

void MPIMultiCorePE::sendInterNodeSpike(SpikeEvent* spike, int target_rank) {
    if (!spike) return;
    if (!isMPIEnabled()) {
        // 在未启用MPI时不进行跨节点发送，保持与原行为一致
        return;
    }

    // 将“rank”视为网络节点ID，通过已存在的外部网络路径发送
    spike->setDestinationNode(static_cast<uint32_t>(target_rank));
    this->sendExternalSpike(spike);

    // 发送侧MPI统计（近似按事件大小计）
    if (stat_mpi_messages_sent_) stat_mpi_messages_sent_->addData(1);
    if (stat_mpi_bytes_sent_)    stat_mpi_bytes_sent_->addData(static_cast<uint64_t>(sizeof(SpikeEvent)));
    SSTMPIPerformanceMonitor::recordMessageSent(sizeof(SpikeEvent));
}

void MPIMultiCorePE::broadcastSpike(SpikeEvent* spike) {
    if (!spike) return;
    if (!isMPIEnabled()) return;

    // 向所有其他rank进行广播（通过SnnNIC/merlin路径）
    for (int r = 0; r < mpi_size_; ++r) {
        if (r == mpi_rank_) continue;
        // 为每个目标创建独立事件副本，避免共享同一指针
        auto* copy = new SpikeEvent(*spike);
        copy->setDestinationNode(static_cast<uint32_t>(r));
        this->sendExternalSpike(copy);
        if (stat_mpi_messages_sent_) stat_mpi_messages_sent_->addData(1);
        if (stat_mpi_bytes_sent_)    stat_mpi_bytes_sent_->addData(static_cast<uint64_t>(sizeof(SpikeEvent)));
        SSTMPIPerformanceMonitor::recordMessageSent(sizeof(SpikeEvent));
    }
}

void MPIMultiCorePE::mpiBarrier() {
    if (!isMPIEnabled() || !sst_mpi_comm_helper_) {
        return;
    }
    
    sst_mpi_comm_helper_->barrier();
}

void MPIMultiCorePE::getMPIStats(std::map<std::string, uint64_t>& stats) const {
    if (!isMPIEnabled()) {
        return;
    }
    
    const auto& mpi_stats = SSTMPIPerformanceMonitor::getStats();
    stats["mpi_messages_sent"] = mpi_stats.messages_sent;
    stats["mpi_messages_received"] = mpi_stats.messages_received;
    stats["mpi_bytes_sent"] = mpi_stats.bytes_sent;
    stats["mpi_bytes_received"] = mpi_stats.bytes_received;
    stats["mpi_barrier_calls"] = mpi_stats.barrier_calls;
}

// ===== 私有方法实现 =====

bool MPIMultiCorePE::initializeSSTMPI() {
    if (!mpi_enabled_) {
        return true;
    }
    
    // 创建MPI通信助手
    sst_mpi_comm_helper_ = std::make_unique<SSTMPICommHelper>(this);
    if (!sst_mpi_comm_helper_->initialize(mpi_config_)) {
        std::cerr << "Failed to initialize SST MPI communication helper" << std::endl;
        return false;
    }
    
    if (mpi_verbose_ > 0) {
        std::cout << "SST MPI initialized successfully for rank " << mpi_rank_ << std::endl;
    }
    
    return true;
}

void MPIMultiCorePE::finalizeSSTMPI() {
    if (sst_mpi_comm_helper_) {
        sst_mpi_comm_helper_.reset();
    }
    
    if (mpi_verbose_ > 0 && mpi_enabled_) {
        std::cout << "SST MPI finalized for rank " << mpi_rank_ << std::endl;
        SSTMPIPerformanceMonitor::printStats();
    }
}

void MPIMultiCorePE::initializeMPIStatistics() {
    if (!mpi_enabled_) {
        return;
    }
    
    // 注册MPI相关的统计信息
    stat_mpi_messages_sent_ = registerStatistic<uint64_t>("mpi_messages_sent");
    stat_mpi_messages_received_ = registerStatistic<uint64_t>("mpi_messages_received");
    stat_mpi_bytes_sent_ = registerStatistic<uint64_t>("mpi_bytes_sent");
    stat_mpi_bytes_received_ = registerStatistic<uint64_t>("mpi_bytes_received");
    stat_mpi_comm_time_ = registerStatistic<double>("mpi_comm_time");
    stat_mpi_sync_time_ = registerStatistic<double>("mpi_sync_time");
    stat_mpi_load_imbalance_ = registerStatistic<double>("mpi_load_imbalance");
    
    if (mpi_verbose_ > 1) {
        std::cout << "MPI statistics initialized for rank " << mpi_rank_ << std::endl;
    }
}

// ===== MPICommManager 子组件实现 =====

MPICommManager::MPICommManager(ComponentId_t id, Params& params) 
    : SubComponent(id) {
    strategy_ = params.find<std::string>("strategy", "default");
}
