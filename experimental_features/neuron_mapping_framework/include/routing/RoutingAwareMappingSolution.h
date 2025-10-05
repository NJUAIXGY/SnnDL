#ifndef NEURON_MAPPING_ROUTING_AWARE_MAPPING_SOLUTION_H
#define NEURON_MAPPING_ROUTING_AWARE_MAPPING_SOLUTION_H

#include "core/MappingSolution.h"
#include "routing/AddressEvent.h"
#include "routing/RoutingTable.h"
#include "routing/MulticastGroup.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <string>

namespace NeuronMapping {

// 前向声明
class RoutingTableGenerator;
class MulticastGroupManager;

/**
 * @brief 路由信息结构
 */
struct RoutingInfo {
    AddressEvent::GlobalNeuronId global_neuron_id = AddressEvent::INVALID_GLOBAL_ID;
    uint32_t routing_key = 0;                      // 路由键
    uint32_t routing_mask = 0;                     // 路由掩码
    std::vector<RouteDirection> output_ports;      // 输出端口列表
    uint16_t multicast_group_id = 0;               // 多播组ID（0表示非多播）
    float routing_cost = 0.0f;                     // 路由成本
    uint32_t hop_count = 0;                        // 跳数
    bool is_local_route = true;                    // 是否为本地路由
    
    RoutingInfo() = default;
    RoutingInfo(AddressEvent::GlobalNeuronId gid, uint32_t key) 
        : global_neuron_id(gid), routing_key(key) {}
    
    std::string toString() const;
};

/**
 * @brief 路由感知的映射解决方案
 * 
 * 扩展基础MappingSolution，集成SpiNNaker风格的路由信息，
 * 支持全局神经元ID管理、路由表生成和多播优化。
 */
class RoutingAwareMappingSolution : public MappingSolution {
public:
    RoutingAwareMappingSolution() = default;
    explicit RoutingAwareMappingSolution(uint32_t num_neurons);
    virtual ~RoutingAwareMappingSolution() = default;
    
    // === 全局神经元ID管理 ===
    
    /**
     * @brief 分配全局神经元ID
     * @param local_neuron_id 本地神经元ID
     * @param pe_id PE ID
     * @param core_id 核心ID
     * @return 全局神经元ID
     */
    AddressEvent::GlobalNeuronId assignGlobalNeuronId(NeuronId local_neuron_id, PEId pe_id, uint32_t core_id = 0);
    
    /**
     * @brief 获取神经元的全局ID
     * @param local_neuron_id 本地神经元ID
     * @return 全局ID，未分配返回INVALID_GLOBAL_ID
     */
    AddressEvent::GlobalNeuronId getGlobalNeuronId(NeuronId local_neuron_id) const;
    
    /**
     * @brief 从全局ID获取本地信息
     * @param global_id 全局神经元ID
     * @return {local_id, pe_id, core_id}，失败返回无效值
     */
    std::tuple<NeuronId, PEId, uint32_t> getLocalInfoFromGlobalId(AddressEvent::GlobalNeuronId global_id) const;
    
    /**
     * @brief 批量分配全局ID
     * @param network 神经网络
     * @return 分配的ID数量
     */
    size_t batchAssignGlobalIds(const NeuralNetwork& network);
    
    /**
     * @brief 验证全局ID分配
     * @return 验证错误列表
     */
    std::vector<std::string> validateGlobalIdAssignment() const;
    
    // === 路由信息管理 ===
    
    /**
     * @brief 添加神经元路由信息
     * @param neuron_id 神经元ID
     * @param routing_info 路由信息
     * @return 是否成功添加
     */
    bool addRoutingInfo(NeuronId neuron_id, const RoutingInfo& routing_info);
    
    /**
     * @brief 获取神经元路由信息
     * @param neuron_id 神经元ID
     * @return 路由信息指针，不存在返回nullptr
     */
    const RoutingInfo* getRoutingInfo(NeuronId neuron_id) const;
    
    /**
     * @brief 更新路由信息
     * @param neuron_id 神经元ID
     * @param routing_info 新路由信息
     * @return 是否成功更新
     */
    bool updateRoutingInfo(NeuronId neuron_id, const RoutingInfo& routing_info);
    
    /**
     * @brief 移除路由信息
     * @param neuron_id 神经元ID
     * @return 是否成功移除
     */
    bool removeRoutingInfo(NeuronId neuron_id);
    
    /**
     * @brief 获取所有路由信息
     * @return 神经元ID到路由信息的映射
     */
    const std::unordered_map<NeuronId, RoutingInfo>& getAllRoutingInfo() const {
        return neuron_routing_info_;
    }
    
    // === 路由表集成 ===
    
    /**
     * @brief 生成分布式路由表
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param generator 路由表生成器
     * @return PE到路由表的映射
     */
    std::unordered_map<PEId, std::unique_ptr<RoutingTable>> generateRoutingTables(
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        RoutingTableGenerator& generator) const;
    
    /**
     * @brief 设置路由表
     * @param pe_id PE ID
     * @param table 路由表
     * @return 是否成功设置
     */
    bool setRoutingTable(PEId pe_id, std::unique_ptr<RoutingTable> table);
    
    /**
     * @brief 获取PE的路由表
     * @param pe_id PE ID
     * @return 路由表指针，不存在返回nullptr
     */
    RoutingTable* getRoutingTable(PEId pe_id);
    const RoutingTable* getRoutingTable(PEId pe_id) const;
    
    /**
     * @brief 获取所有路由表
     * @return PE到路由表的映射
     */
    const std::unordered_map<PEId, std::unique_ptr<RoutingTable>>& getAllRoutingTables() const {
        return pe_routing_tables_;
    }
    
    /**
     * @brief 同步路由表和映射信息
     * @return 同步的表项数量
     */
    size_t synchronizeRoutingTables();
    
    // === 多播组管理 ===
    
    /**
     * @brief 设置多播组管理器
     * @param manager 多播组管理器
     */
    void setMulticastGroupManager(std::shared_ptr<MulticastGroupManager> manager);
    
    /**
     * @brief 获取多播组管理器
     * @return 管理器指针
     */
    std::shared_ptr<MulticastGroupManager> getMulticastGroupManager() const {
        return multicast_manager_;
    }
    
    /**
     * @brief 创建多播组
     * @param group_id 组ID
     * @param member_neurons 成员神经元
     * @return 是否成功创建
     */
    bool createMulticastGroup(uint16_t group_id, const std::vector<NeuronId>& member_neurons);
    
    /**
     * @brief 获取神经元所属的多播组
     * @param neuron_id 神经元ID
     * @return 组ID列表
     */
    std::vector<uint16_t> getNeuronMulticastGroups(NeuronId neuron_id) const;
    
    /**
     * @brief 自动检测和创建多播组
     * @param network 神经网络
     * @param min_group_size 最小组大小
     * @return 创建的组数量
     */
    size_t autoCreateMulticastGroups(const NeuralNetwork& network, size_t min_group_size = 3);
    
    // === 路由性能分析 ===
    
    /**
     * @brief 计算路由性能指标
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @return 路由性能指标
     */
    std::unordered_map<std::string, float> calculateRoutingPerformance(
        const NeuralNetwork& network,
        const HardwareTopology& topology) const;
    
    /**
     * @brief 分析通信模式
     * @param network 神经网络
     * @return 通信模式统计
     */
    std::unordered_map<std::string, uint32_t> analyzeCommunicationPatterns(
        const NeuralNetwork& network) const;
    
    /**
     * @brief 计算总路由成本
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param include_multicast 是否包含多播成本
     * @return 总路由成本
     */
    float calculateTotalRoutingCost(const NeuralNetwork& network,
                                   const HardwareTopology& topology,
                                   bool include_multicast = true) const;
    
    /**
     * @brief 识别路由热点
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @return PE的路由负载排序
     */
    std::vector<std::pair<PEId, float>> identifyRoutingHotspots(
        const NeuralNetwork& network,
        const HardwareTopology& topology) const;
    
    /**
     * @brief 分析跨PE通信
     * @param network 神经网络
     * @return 跨PE通信统计
     */
    std::unordered_map<std::string, float> analyzeInterPECommunication(
        const NeuralNetwork& network) const;
    
    // === 路由优化 ===
    
    /**
     * @brief 优化神经元放置以减少路由成本
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param max_iterations 最大迭代次数
     * @return 优化统计信息
     */
    std::unordered_map<std::string, float> optimizeForRouting(
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        uint32_t max_iterations = 100);
    
    /**
     * @brief 重新映射高通信开销的神经元
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param cost_threshold 成本阈值
     * @return 重新映射的神经元数量
     */
    size_t remapHighCostNeurons(const NeuralNetwork& network,
                               const HardwareTopology& topology,
                               float cost_threshold = 2.0f);
    
    /**
     * @brief 平衡路由负载
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param balance_threshold 平衡阈值
     * @return 负载平衡改善程度
     */
    float balanceRoutingLoad(const NeuralNetwork& network,
                           const HardwareTopology& topology,
                           float balance_threshold = 0.8f);
    
    // === 路由验证和调试 ===
    
    /**
     * @brief 验证路由一致性
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @return 验证错误列表
     */
    std::vector<std::string> validateRoutingConsistency(
        const NeuralNetwork& network,
        const HardwareTopology& topology) const;
    
    /**
     * @brief 模拟数据包路由
     * @param source_neuron 源神经元
     * @param target_neuron 目标神经元
     * @param topology 硬件拓扑
     * @return 路由路径和统计信息
     */
    std::pair<std::vector<PEId>, std::unordered_map<std::string, float>> simulatePacketRouting(
        NeuronId source_neuron,
        NeuronId target_neuron,
        const HardwareTopology& topology) const;
    
    /**
     * @brief 调试路由问题
     * @param neuron_id 神经元ID
     * @param issue_type 问题类型
     * @return 调试信息
     */
    std::string debugRoutingIssue(NeuronId neuron_id, const std::string& issue_type) const;
    
    /**
     * @brief 生成路由调试报告
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param detailed 是否详细报告
     * @return 报告字符串
     */
    std::string generateRoutingDebugReport(const NeuralNetwork& network,
                                         const HardwareTopology& topology,
                                         bool detailed = false) const;
    
    // === 序列化和导出 ===
    
    /**
     * @brief 导出路由配置
     * @param format 格式（"json", "xml", "binary"）
     * @return 导出数据
     */
    std::string exportRoutingConfiguration(const std::string& format = "json") const;
    
    /**
     * @brief 导入路由配置
     * @param config_data 配置数据
     * @param format 格式
     * @return 是否成功导入
     */
    bool importRoutingConfiguration(const std::string& config_data, const std::string& format = "json");
    
    /**
     * @brief 导出为SpiNNaker兼容格式
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @return SpiNNaker格式的配置数据
     */
    std::string exportToSpiNNakerFormat(const NeuralNetwork& network,
                                       const HardwareTopology& topology) const;
    
    // === 统计和监控 ===
    
    /**
     * @brief 获取路由统计信息
     * @return 统计信息映射
     */
    std::unordered_map<std::string, uint64_t> getRoutingStatistics() const;
    
    /**
     * @brief 重置路由统计
     */
    void resetRoutingStatistics();
    
    /**
     * @brief 启用路由监控
     * @param enable 是否启用
     * @param monitoring_interval 监控间隔（毫秒）
     */
    void enableRoutingMonitoring(bool enable, uint32_t monitoring_interval = 1000);
    
    /**
     * @brief 收集路由性能数据
     * @param duration_ms 收集持续时间（毫秒）
     * @return 性能数据
     */
    std::unordered_map<std::string, std::vector<float>> collectRoutingPerformanceData(
        uint32_t duration_ms = 5000) const;
    
    // === 扩展的映射操作（重写基类方法以支持路由） ===
    
    /**
     * @brief 路由感知的神经元分配
     * @param neuron_id 神经元ID
     * @param pe_id PE ID
     * @param core_id 核心ID
     * @return 是否成功分配
     */
    bool assignNeuronWithRouting(NeuronId neuron_id, PEId pe_id, uint32_t core_id = 0);
    
    /**
     * @brief 路由感知的神经元移动
     * @param neuron_id 神经元ID
     * @param target_pe_id 目标PE ID
     * @param core_id 核心ID
     * @return 是否成功移动
     */
    bool moveNeuronWithRouting(NeuronId neuron_id, PEId target_pe_id, uint32_t core_id = 0);
    
    /**
     * @brief 重新计算所有路由信息
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @return 更新的路由信息数量
     */
    size_t recalculateAllRoutingInfo(const NeuralNetwork& network, const HardwareTopology& topology);
    
    // === 克隆和比较 ===
    
    /**
     * @brief 克隆路由感知映射解决方案
     * @return 深拷贝的解决方案
     */
    std::unique_ptr<RoutingAwareMappingSolution> cloneWithRouting() const;
    
    /**
     * @brief 比较路由性能
     * @param other 另一个解决方案
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @return 比较结果（负值表示当前解更好）
     */
    float compareRoutingPerformance(const RoutingAwareMappingSolution& other,
                                   const NeuralNetwork& network,
                                   const HardwareTopology& topology) const;

private:
    // 全局神经元ID管理
    std::unordered_map<NeuronId, AddressEvent::GlobalNeuronId> local_to_global_id_;
    std::unordered_map<AddressEvent::GlobalNeuronId, NeuronId> global_to_local_id_;
    AddressEvent::GlobalNeuronId next_global_id_ = 1;
    
    // 路由信息存储
    std::unordered_map<NeuronId, RoutingInfo> neuron_routing_info_;
    std::unordered_map<PEId, std::unique_ptr<RoutingTable>> pe_routing_tables_;
    
    // 多播组管理
    std::shared_ptr<MulticastGroupManager> multicast_manager_;
    std::unordered_map<NeuronId, std::vector<uint16_t>> neuron_to_multicast_groups_;
    
    // 路由统计
    mutable std::unordered_map<std::string, uint64_t> routing_statistics_;
    bool routing_monitoring_enabled_ = false;
    uint32_t monitoring_interval_ms_ = 1000;
    mutable std::chrono::steady_clock::time_point last_monitoring_time_;
    
    // 缓存的性能数据
    mutable std::unordered_map<std::string, float> cached_routing_performance_;
    mutable bool routing_performance_valid_ = false;
    
    // 内部辅助方法
    void invalidateRoutingCache() { routing_performance_valid_ = false; }
    void updateRoutingStatistic(const std::string& metric, uint64_t value) const;
    RoutingInfo generateRoutingInfoForNeuron(NeuronId neuron_id, 
                                           const NeuralNetwork& network,
                                           const HardwareTopology& topology) const;
    uint32_t generateRoutingKey(AddressEvent::GlobalNeuronId global_id) const;
    std::vector<RouteDirection> calculateOutputPorts(NeuronId neuron_id,
                                                    const NeuralNetwork& network,
                                                    const HardwareTopology& topology) const;
    float calculateNeuronRoutingCost(NeuronId neuron_id,
                                   const NeuralNetwork& network,
                                   const HardwareTopology& topology) const;
    bool isRoutingOptimal(NeuronId neuron_id,
                         const NeuralNetwork& network,
                         const HardwareTopology& topology) const;
    void updateRoutingInfoAfterMove(NeuronId neuron_id, PEId old_pe, PEId new_pe,
                                   const NeuralNetwork& network,
                                   const HardwareTopology& topology);
    void synchronizeMulticastGroups(const NeuralNetwork& network);
    
    // 性能优化辅助方法
    std::vector<NeuronId> identifyHighCostNeurons(const NeuralNetwork& network,
                                                 const HardwareTopology& topology,
                                                 float cost_threshold) const;
    PEId findBestPEForNeuron(NeuronId neuron_id,
                           const NeuralNetwork& network,
                           const HardwareTopology& topology,
                           const std::unordered_set<PEId>& excluded_pes = {}) const;
    float evaluateRoutingCostReduction(NeuronId neuron_id, PEId target_pe,
                                     const NeuralNetwork& network,
                                     const HardwareTopology& topology) const;
    
    // 验证辅助方法
    bool validateGlobalIdUniqueness() const;
    bool validateRoutingTableConsistency() const;
    std::vector<std::string> findRoutingInconsistencies(const NeuralNetwork& network,
                                                       const HardwareTopology& topology) const;
};

/**
 * @brief 路由感知映射解决方案工厂
 */
class RoutingAwareMappingSolutionFactory {
public:
    /**
     * @brief 从基础映射解决方案创建路由感知版本
     * @param base_solution 基础解决方案
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @return 路由感知解决方案
     */
    static std::unique_ptr<RoutingAwareMappingSolution> createFromBaseSolution(
        const MappingSolution& base_solution,
        const NeuralNetwork& network,
        const HardwareTopology& topology);
    
    /**
     * @brief 创建优化的路由感知解决方案
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param optimization_level 优化级别 (1-3)
     * @return 路由感知解决方案
     */
    static std::unique_ptr<RoutingAwareMappingSolution> createOptimizedSolution(
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        uint8_t optimization_level = 2);
    
    /**
     * @brief 创建多播优化的解决方案
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param enable_adaptive_groups 启用自适应组
     * @return 路由感知解决方案
     */
    static std::unique_ptr<RoutingAwareMappingSolution> createMulticastOptimizedSolution(
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        bool enable_adaptive_groups = true);

private:
    RoutingAwareMappingSolutionFactory() = default;
};

} // namespace NeuronMapping

#endif // NEURON_MAPPING_ROUTING_AWARE_MAPPING_SOLUTION_H