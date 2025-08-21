// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// MultiCorePE.h: 真正的多核脉冲神经网络处理单元头文件
//

#ifndef _MULTICOREPE_H
#define _MULTICOREPE_H

#include <sst/core/component.h>
#include <sst/core/subcomponent.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <sst/core/params.h>
#include <sst/core/event.h>
#include <sst/core/clock.h>
#include <sst/core/interfaces/simpleNetwork.h>
#include <sst/core/interfaces/stdMem.h>

#include <vector>
#include <string>
#include <cstdint>
#include <map>
#include <memory>
#include <queue>
#include <unordered_map>

#include "SpikeEvent.h"
#include "SpikeEventWrapper.h"
#include "SnnInterface.h"
#include "SnnPEParentInterface.h"
#include "SnnCoreAPI.h"
#include "OptimizedInternalRing.h"

namespace SST {
namespace SnnDL {

// 前置声明
class InternalRing;
class MultiCoreController;
class SnnNetworkAdapter;
class MultiCorePERouterInterface;

// RingMessage和RingMessageType现在定义在OptimizedInternalRing.h中

/**
 * @brief 处理单元状态
 */
struct ProcessingUnitState {
    int unit_id;
    int neuron_id_start;
    int neuron_count;
    bool is_active;
    uint64_t spikes_processed;
    uint64_t neurons_fired;
    double utilization;
    
    ProcessingUnitState() : unit_id(-1), neuron_id_start(0), neuron_count(0), 
                           is_active(false), spikes_processed(0), neurons_fired(0), utilization(0.0) {}
};

/**
 * @brief 真正的多核处理单元组件
 * 
 * 集成多个ProcessingUnit、共享L2缓存、内部互连网络
 */
class MultiCorePE : public SST::Component, public SnnPEParentInterface {
public:
    // ELI注册信息
    SST_ELI_REGISTER_COMPONENT(
        MultiCorePE,                    // 类名
        "SnnDL",                       // 库名  
        "MultiCorePE",                 // 组件名
        SST_ELI_ELEMENT_VERSION(1,0,0), // 版本
        "真正的多核脉冲神经网络处理单元",  // 描述
        COMPONENT_CATEGORY_PROCESSOR    // 类别
    )

    // 参数文档
    SST_ELI_DOCUMENT_PARAMS(
        {"clock",            "多核PE内部时钟频率", "1GHz"},
        {"num_cores",        "处理单元数量", "4"},
        {"neurons_per_core", "每个处理单元的神经元数量", "64"},
        {"l2_cache_size",    "共享L2缓存大小", "256KB"},
        {"l2_associativity", "L2缓存关联度", "8"},
        {"l2_cache_line_size", "L2缓存行大小", "64B"},
        {"internal_ring_latency", "内部环形网络延迟", "1ns"},
        {"verbose",          "日志详细级别", "0"},
        {"node_id",          "网络节点ID", "0"},
        {"base_addr",        "全局内存基地址", "0"},
        {"weights_file",     "权重文件路径", ""},
        {"enable_numa",      "启用NUMA优化", "1"},
        {"v_thresh",         "触发脉冲的膜电位阈值", "1.0"},
        {"v_reset",          "脉冲发放后膜电位重置值", "0.0"},
        {"v_rest",           "静息膜电位", "0.0"},
        {"tau_mem",          "膜电位泄漏时间常数(ms)", "20.0"},
        {"t_ref",            "不应期时长(时钟周期)", "2"},
        {"enable_test_traffic", "是否启用网络测试流量", "0"},
        {"test_target_node", "测试流量的目标节点ID", "0"},
        {"test_period",      "测试流量发送周期(周期数)", "100"},
        {"test_spikes_per_burst", "每次周期性发送的测试脉冲数量", "4"},
        {"test_weight",      "测试脉冲权重", "0.2"},
        {"use_optimized_ring", "使用优化的环形网络实现(1)或原始实现(0)", "1"}
    )

    // 子组件槽位文档
    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS(
        {"core0", "SnnPE计算核心0", "SST::SnnDL::SnnCoreAPI"},
        {"core1", "SnnPE计算核心1", "SST::SnnDL::SnnCoreAPI"},
        {"core2", "SnnPE计算核心2", "SST::SnnDL::SnnCoreAPI"},
        {"core3", "SnnPE计算核心3", "SST::SnnDL::SnnCoreAPI"},
        {"core4", "SnnPE计算核心4", "SST::SnnDL::SnnCoreAPI"},
        {"core5", "SnnPE计算核心5", "SST::SnnDL::SnnCoreAPI"},
        {"core6", "SnnPE计算核心6", "SST::SnnDL::SnnCoreAPI"},
        {"core7", "SnnPE计算核心7", "SST::SnnDL::SnnCoreAPI"},
        {"l2_cache", "共享L2缓存", "SST::MemHierarchy::Cache"},
        {"memory_interface", "内存接口", "SST::Interfaces::StandardMem"},
        {"external_nic", "外部网络接口", "SST::SnnDL::SnnInterface"}
    )

    // 端口文档 
    SST_ELI_DOCUMENT_PORTS(
        {"external_spike_input",  "外部脉冲输入端口", {"SnnDL.SpikeEvent"}},
        {"external_spike_output", "外部脉冲输出端口", {"SnnDL.SpikeEvent"}},
        {"network", "网络连接端口（用于direct_link模式）", {"SnnDL.SpikeEvent", "SimpleNetwork"}},
        {"north", "北向网络连接端口（网格拓扑）", {"SnnDL.SpikeEvent"}},
        {"south", "南向网络连接端口（网格拓扑）", {"SnnDL.SpikeEvent"}},
        {"east", "东向网络连接端口（网格拓扑）", {"SnnDL.SpikeEvent"}},
        {"west", "西向网络连接端口（网格拓扑）", {"SnnDL.SpikeEvent"}},
        {"mem_link", "内存层次结构连接端口（保留，通用用途）", {"memHierarchy.MemEventBase"}},
        {"core0_mem", "核心0的内存连接端口", {"memHierarchy.MemEventBase"}},
        {"core1_mem", "核心1的内存连接端口", {"memHierarchy.MemEventBase"}},
        {"core2_mem", "核心2的内存连接端口", {"memHierarchy.MemEventBase"}},
        {"core3_mem", "核心3的内存连接端口", {"memHierarchy.MemEventBase"}},
        {"core4_mem", "核心4的内存连接端口", {"memHierarchy.MemEventBase"}},
        {"core5_mem", "核心5的内存连接端口", {"memHierarchy.MemEventBase"}},
        {"core6_mem", "核心6的内存连接端口", {"memHierarchy.MemEventBase"}},
        {"core7_mem", "核心7的内存连接端口", {"memHierarchy.MemEventBase"}}
    )

    // 统计信息文档
    SST_ELI_DOCUMENT_STATISTICS(
        {"total_spikes_processed", "处理的脉冲总数", "spikes", 1},
        {"inter_core_messages", "核间消息数量", "messages", 1}, 
        {"l2_cache_hits", "L2缓存命中数", "hits", 1},
        {"l2_cache_misses", "L2缓存缺失数", "misses", 1},
        {"memory_requests", "内存请求数", "requests", 1},
        {"avg_core_utilization", "平均核心利用率", "percentage", 1},
        {"total_neurons_fired", "总神经元发放数", "neurons", 1},
        {"external_spikes_sent", "发送的外部脉冲数", "spikes", 1},
        {"external_spikes_received", "接收的外部脉冲数", "spikes", 1}
    )

    /**
     * @brief 构造函数
     */
    MultiCorePE(SST::ComponentId_t id, SST::Params& params);

    /**
     * @brief 析构函数  
     */
    ~MultiCorePE();

    // SST组件生命周期方法
    void init(unsigned int phase) override;
    void setup() override;
    void finish() override;

    // ===== 公共接口方法 =====
    
    /**
     * @brief 处理来自其他MultiCorePE的外部脉冲
     */
    void handleExternalSpike(SpikeEvent* spike);
    
    /**
     * @brief 发送脉冲到其他MultiCorePE  
     */
    void sendExternalSpike(SpikeEvent* spike);
    
    /**
     * @brief 获取处理单元状态信息
     */
    const ProcessingUnitState& getProcessingUnitState(int unit_id) const;
    
    /**
     * @brief 获取多核PE统计信息
     */
    void getStatistics(std::map<std::string, uint64_t>& stats) const;

    // ===== SnnPEParentInterface 实现 =====
    
    /**
     * @brief 向父级组件发送脉冲（从SnnPE SubComponent调用）
     */
    void sendSpike(SpikeEvent* event) override;
    
    /**
     * @brief 向父级组件请求内存访问（从SnnPE SubComponent调用）
     */
    void requestMemoryAccess(uint64_t address, size_t size, 
                           std::function<void(const void*)> callback) override;
    
    /**
     * @brief 获取当前仿真周期
     */
    uint64_t getCurrentCycle() const override { return current_cycle_; }
    
    /**
     * @brief 获取本PE的节点ID
     */
    int getNodeId() const override { return node_id_; }
    
    /**
     * @brief 获取本PE管理的神经元总数
     */
    int getTotalNeurons() const override { return total_neurons_; }

    // 友元类声明
    friend class InternalRing;
    friend class MultiCoreController;

private:
    // ===== 配置参数 =====
    
    int num_cores_;
    int neurons_per_core_;
    int total_neurons_;
    int node_id_;
    uint64_t global_neuron_base_;
    int verbose_;
    std::string weights_file_;
    bool enable_numa_;
    bool enable_test_traffic_;
    
    // 神经元参数
    float v_thresh_;
    float v_reset_;
    float v_rest_;
    float tau_mem_;
    int t_ref_;
    
    // 测试流量参数
    int test_target_node_;
    int test_period_;
    int test_spikes_per_burst_;
    float test_weight_;
    int test_max_spikes_;  // 最大测试脉冲数限制
    
    // 环形网络实现选择
    bool use_optimized_ring_;
    
    // 权重验证参数
    bool verify_weights_;
    uint32_t weight_verify_samples_;
    float expected_weight_value_;
    bool verify_log_each_sample_;
    
    // ===== 组件对象 =====
    
    // 时钟和输出
    SST::Output* output_;
    
    // 统计变量
    Statistic<uint64_t>* stat_spikes_processed_;
    Statistic<uint64_t>* stat_inter_core_messages_;
    Statistic<uint64_t>* stat_l2_hits_;
    Statistic<uint64_t>* stat_l2_misses_;
    Statistic<uint64_t>* stat_memory_requests_;
    Statistic<double>* stat_avg_utilization_;
    Statistic<uint64_t>* stat_neurons_fired_;
    Statistic<uint64_t>* stat_external_spikes_sent_;
    Statistic<uint64_t>* stat_external_spikes_received_;

    // 本地统计：仅在环形跨核投递成功时累加
    uint64_t inter_core_messages_count_ = 0;

    // 测试注入一次跨核脉冲
    bool test_injected_ = false;
    
    // 子组件
    std::vector<SnnCoreAPI*> cores_;
    SST::Interfaces::StandardMem* l2_cache_;
    SST::Interfaces::StandardMem* memory_interface_;
    SST::SnnDL::SnnInterface* external_nic_;
    
    // 内部架构组件
    OptimizedInternalRing* optimized_ring_;
    InternalRing* internal_ring_;  // 保留兼容性
    MultiCoreController* controller_;
    
    // 处理单元状态跟踪
    std::vector<ProcessingUnitState> unit_states_;
    
    // 外部端口
    SST::Link* external_spike_input_link_;
    SST::Link* external_spike_output_link_;
    SST::Link* mem_link_;
    
    // 网络方向端口（用于端口代理机制）
    SST::Link* north_link_;
    SST::Link* south_link_;
    SST::Link* east_link_;
    SST::Link* west_link_;
    SST::Link* network_link_;
    
    // 内部数据结构
    std::queue<SpikeEvent*> external_spike_queue_;
    std::unordered_map<uint64_t, SpikeEvent*> pending_memory_requests_;
    
    // 时钟计数器和测试流量
    uint64_t current_cycle_;
    uint64_t test_cycle_counter_;
    int test_spikes_sent_;  // 已发送的测试脉冲计数
    
    // ===== 核心方法 =====
    
    /**
     * @brief 时钟滴答处理器
     */
    bool clockTick(SST::Cycle_t current_cycle);
    
    /**
     * @brief 处理外部脉冲事件（从Link接收）
     */
    void handleExternalSpikeEvent(SST::Event* ev);
    
    /**
     * @brief 从SpikeEventWrapper中提取SpikeEvent数据
     */
    SpikeEvent* extractSpikeFromWrapper(SpikeEventWrapper* wrapper);
    
    /**
     * @brief 处理内部脉冲路由
     */
    void routeInternalSpike(int src_core, int dst_core, SpikeEvent* spike);
    
    /**
     * @brief 处理内存响应
     */
    void handleMemoryResponse(SST::Interfaces::StandardMem::Request* resp);
    
    /**
     * @brief 加载和分布权重
     */
    void loadAndDistributeWeights();
    
    /**
     * @brief 初始化SnnPE SubComponent核心
     */
    void initializeProcessingUnits();
    
    /**
     * @brief 初始化网络接口适配器
     */
    void initializeNetworkInterface();
    
    /**
     * @brief 初始化方向链路代理机制
     */
    void initializeDirectionLinks();
    
    /**
     * @brief 向指定核心递送脉冲
     */
    void deliverSpikeToCore(int core_id, SpikeEvent* spike);
    
    // === 网络端口事件处理器 ===
    
    /**
     * @brief 处理北向链路事件
     */
    void handleNorthLinkEvent(SST::Event* event);
    
    /**
     * @brief 处理南向链路事件
     */
    void handleSouthLinkEvent(SST::Event* event);
    
    /**
     * @brief 处理东向链路事件
     */
    void handleEastLinkEvent(SST::Event* event);
    
    /**
     * @brief 处理西向链路事件
     */
    void handleWestLinkEvent(SST::Event* event);
    
    /**
     * @brief 处理通用网络链路事件
     */
    void handleNetworkLinkEvent(SST::Event* event);
    
    /**
     * @brief 转发事件给SnnNetworkAdapter
     */
    void forwardEventToNetworkAdapter(SST::Event* event, const std::string& direction);
    
    /**
     * @brief 初始化内部互连
     */
    void initializeInternalRing();
    
    /**
     * @brief 初始化统计收集
     */
    void initializeStatistics();
    
    /**
     * @brief 更新统计信息
     */
    void updateStatistics();
    
    /**
     * @brief 生成测试流量
     */
    void generateTestTraffic();
    
    /**
     * @brief 处理跨核脉冲路由
     */
    void handleCrossCoreRouting();
    
    /**
     * @brief 处理优化的跨核脉冲路由
     */
    void handleOptimizedCrossCoreRouting();
    
    /**
     * @brief 负载均衡检查
     */
    void checkLoadBalance();
    
    /**
     * @brief 确定脉冲的目标处理单元
     */
    int determineTargetUnit(int neuron_id) const;
    
    /**
     * @brief 确定神经元ID是否属于本MultiCorePE
     */
    bool isLocalNeuron(int neuron_id) const;
};



/**
 * @brief 内部环形互连（简化版）
 */
class InternalRing {
public:
    InternalRing(int num_nodes, int latency_cycles, SST::Output* output);
    ~InternalRing();
    
    // 消息传递接口
    bool sendMessage(const RingMessage& msg);
    bool receiveMessage(int node_id, RingMessage& msg);
    
    // 路由和仲裁
    void tick();
    bool hasTrafficForNode(int node_id) const;
    int getPendingMessageCount() const;
    
    // 统计信息
    uint64_t getTotalMessagesRouted() const { return total_messages_routed_; }
    double getAverageLatency() const;

private:
    int num_nodes_;
    int latency_cycles_;
    SST::Output* output_;
    
    // 环形缓冲区
    std::vector<std::queue<RingMessage>> node_input_queues_;
    std::vector<std::queue<RingMessage>> node_output_queues_;
    std::queue<RingMessage> ring_buffer_;
    
    // 统计信息
    uint64_t total_messages_routed_;
    uint64_t total_latency_cycles_;
    
    // 内部方法
    int getNextNode(int current_node) const;
    void routeMessage(const RingMessage& msg);
};

/**
 * @brief 多核控制器
 */
class MultiCoreController {
public:
    MultiCoreController(MultiCorePE* parent, SST::Output* output);
    ~MultiCoreController();
    
    // 调度和负载均衡
    void scheduleWork();
    void balanceLoad();
    void tick();
    
    // 性能监控
    void updatePerformanceCounters();
    double getCoreUtilization(int core_id) const;
    double getOverallUtilization() const;
    
    // 统计信息
    uint64_t getTotalWorkDistributed() const { return total_work_distributed_; }
    int getLoadImbalanceCount() const { return load_imbalance_count_; }

private:
    MultiCorePE* parent_pe_;
    SST::Output* output_;
    
    // 负载均衡状态
    std::vector<double> core_utilization_history_;
    std::vector<uint64_t> core_work_count_;
    
    // 统计信息
    uint64_t total_work_distributed_;
    int load_imbalance_count_;
    double load_balance_threshold_;
    
    // 内部方法
    void redistributeWork();
    int findLeastLoadedCore() const;
    int findMostLoadedCore() const;
};

} // namespace SnnDL
} // namespace SST

#endif // _MULTICOREPE_H
