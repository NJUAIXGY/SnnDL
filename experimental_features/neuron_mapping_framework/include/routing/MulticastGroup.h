#ifndef NEURON_MAPPING_MULTICAST_GROUP_H
#define NEURON_MAPPING_MULTICAST_GROUP_H

#include "AddressEvent.h"
#include "core/Types.h"
#include "core/NeuralNetwork.h"
#include "core/HardwareTopology.h"
#include "core/MappingSolution.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <string>
#include <chrono>

namespace NeuronMapping {

/**
 * @brief 多播组类型枚举
 */
enum class MulticastGroupType {
    STATIC,         // 静态多播组（编译时确定）
    DYNAMIC,        // 动态多播组（运行时变化）
    ADAPTIVE,       // 自适应多播组（根据流量调整）
    TEMPORAL        // 时域多播组（时间相关）
};

/**
 * @brief 多播传输模式
 */
enum class MulticastMode {
    RELIABLE,       // 可靠传输（确保所有成员接收）
    BEST_EFFORT,    // 尽力传输（不保证接收）
    PRIORITY,       // 优先级传输（重要成员优先）
    SELECTIVE       // 选择性传输（部分成员接收）
};

/**
 * @brief 多播组统计信息
 */
struct MulticastGroupStats {
    uint64_t packets_sent = 0;              // 发送的数据包数
    uint64_t packets_delivered = 0;         // 成功传递的数据包数
    uint64_t packets_dropped = 0;           // 丢弃的数据包数
    uint64_t total_transmissions = 0;       // 总传输次数
    float delivery_rate = 0.0f;             // 传递率
    float average_latency = 0.0f;           // 平均延迟（微秒）
    float bandwidth_usage = 0.0f;           // 带宽使用率
    uint32_t tree_depth = 0;                // 多播树深度
    uint32_t tree_width = 0;                // 多播树宽度
    std::chrono::steady_clock::time_point last_activity; // 最后活动时间
    
    void reset();
    std::string toString() const;
};

/**
 * @brief 增强的多播组
 * 
 * 扩展基础MulticastGroup，添加更多管理功能和优化特性。
 */
class EnhancedMulticastGroup {
public:
    EnhancedMulticastGroup() = default;
    explicit EnhancedMulticastGroup(uint16_t group_id);
    EnhancedMulticastGroup(uint16_t group_id, const std::vector<AddressEvent::GlobalNeuronId>& members);
    virtual ~EnhancedMulticastGroup() = default;
    
    // === 基本属性访问 ===
    
    uint16_t getGroupId() const { return group_id_; }
    void setGroupId(uint16_t id) { group_id_ = id; }
    
    MulticastGroupType getGroupType() const { return group_type_; }
    void setGroupType(MulticastGroupType type) { group_type_ = type; }
    
    MulticastMode getMulticastMode() const { return multicast_mode_; }
    void setMulticastMode(MulticastMode mode) { multicast_mode_ = mode; }
    
    bool isActive() const { return active_; }
    void setActive(bool active) { active_ = active; }
    
    float getPriority() const { return priority_; }
    void setPriority(float priority) { priority_ = priority; }
    
    // === 成员管理 ===
    
    /**
     * @brief 添加成员神经元
     * @param neuron_id 神经元全局ID
     * @param weight 成员权重
     * @return 是否成功添加
     */
    bool addMember(AddressEvent::GlobalNeuronId neuron_id, float weight = 1.0f);
    
    /**
     * @brief 批量添加成员
     * @param neuron_ids 神经元ID列表
     * @param weights 权重列表（可选）
     * @return 成功添加的数量
     */
    size_t addMembers(const std::vector<AddressEvent::GlobalNeuronId>& neuron_ids,
                     const std::vector<float>& weights = {});
    
    /**
     * @brief 移除成员神经元
     * @param neuron_id 神经元全局ID
     * @return 是否成功移除
     */
    bool removeMember(AddressEvent::GlobalNeuronId neuron_id);
    
    /**
     * @brief 检查是否包含指定神经元
     * @param neuron_id 神经元全局ID
     * @return 是否包含
     */
    bool containsMember(AddressEvent::GlobalNeuronId neuron_id) const;
    
    /**
     * @brief 获取所有成员
     * @return 成员神经元ID列表
     */
    std::vector<AddressEvent::GlobalNeuronId> getMembers() const;
    
    /**
     * @brief 获取成员数量
     * @return 成员数量
     */
    size_t getMemberCount() const { return members_.size(); }
    
    /**
     * @brief 获取成员权重
     * @param neuron_id 神经元全局ID
     * @return 权重值，不存在返回0
     */
    float getMemberWeight(AddressEvent::GlobalNeuronId neuron_id) const;
    
    /**
     * @brief 设置成员权重
     * @param neuron_id 神经元全局ID
     * @param weight 新权重
     * @return 是否成功设置
     */
    bool setMemberWeight(AddressEvent::GlobalNeuronId neuron_id, float weight);
    
    // === PE管理 ===
    
    /**
     * @brief 更新目标PE列表
     * @param mapping 映射解决方案
     * @return 更新的PE数量
     */
    size_t updateTargetPEs(const MappingSolution& mapping);
    
    /**
     * @brief 获取目标PE列表
     * @return PE ID列表
     */
    std::vector<PEId> getTargetPEs() const { return target_pes_; }
    
    /**
     * @brief 获取PE的神经元数量
     * @param pe_id PE标识
     * @return 该PE上的组成员数量
     */
    size_t getNeuronCountOnPE(PEId pe_id) const;
    
    /**
     * @brief 获取PE分布信息
     * @return PE到神经元数量的映射
     */
    std::unordered_map<PEId, size_t> getPEDistribution() const;
    
    // === 多播树管理 ===
    
    /**
     * @brief 构建多播树
     * @param topology 硬件拓扑
     * @param root_pe 根PE（可选，自动选择最优根）
     * @return 是否成功构建
     */
    bool buildMulticastTree(const HardwareTopology& topology, PEId root_pe = INVALID_PE_ID);
    
    /**
     * @brief 获取多播树根节点
     * @return 根PE ID
     */
    PEId getTreeRoot() const { return tree_root_; }
    
    /**
     * @brief 获取多播树路由
     * @return PE到路由方向的映射
     */
    const std::unordered_map<PEId, std::vector<RouteDirection>>& getTreeRoutes() const {
        return tree_routes_;
    }
    
    /**
     * @brief 优化多播树
     * @param topology 硬件拓扑
     * @param optimization_target 优化目标（"latency", "bandwidth", "balance"）
     * @return 优化统计信息
     */
    std::unordered_map<std::string, float> optimizeTree(
        const HardwareTopology& topology,
        const std::string& optimization_target = "latency");
    
    /**
     * @brief 验证多播树有效性
     * @param topology 硬件拓扑
     * @return 验证错误列表
     */
    std::vector<std::string> validateTree(const HardwareTopology& topology) const;
    
    // === 流量控制和QoS ===
    
    /**
     * @brief 设置QoS参数
     * @param max_latency 最大延迟（微秒）
     * @param min_bandwidth 最小带宽要求
     * @param reliability 可靠性要求 (0-1)
     */
    void setQoSRequirements(float max_latency, float min_bandwidth, float reliability);
    
    /**
     * @brief 获取QoS参数
     * @return QoS参数映射
     */
    std::unordered_map<std::string, float> getQoSRequirements() const;
    
    /**
     * @brief 应用流量控制
     * @param max_rate 最大传输速率（包/秒）
     * @param burst_size 突发大小
     */
    void applyFlowControl(uint32_t max_rate, uint32_t burst_size);
    
    /**
     * @brief 检查QoS满足情况
     * @return QoS满足度 (0-1)
     */
    float checkQoSSatisfaction() const;
    
    // === 动态适应 ===
    
    /**
     * @brief 根据流量模式调整组配置
     * @param traffic_pattern 流量模式数据
     * @return 调整的参数数量
     */
    size_t adaptToTrafficPattern(const std::unordered_map<std::string, float>& traffic_pattern);
    
    /**
     * @brief 动态添加/移除成员
     * @param candidate_neurons 候选神经元
     * @param traffic_threshold 流量阈值
     * @return 成员变化数量
     */
    int32_t dynamicMembershipUpdate(
        const std::vector<AddressEvent::GlobalNeuronId>& candidate_neurons,
        float traffic_threshold);
    
    /**
     * @brief 自适应树重构
     * @param topology 硬件拓扑
     * @param performance_metrics 性能指标
     * @return 是否重构了树
     */
    bool adaptiveTreeReconstruction(
        const HardwareTopology& topology,
        const std::unordered_map<std::string, float>& performance_metrics);
    
    // === 统计和监控 ===
    
    /**
     * @brief 更新统计信息
     * @param packets_sent 发送的数据包数
     * @param packets_delivered 传递的数据包数
     * @param latency 延迟（微秒）
     */
    void updateStatistics(uint32_t packets_sent, uint32_t packets_delivered, float latency);
    
    /**
     * @brief 获取统计信息
     * @return 统计信息
     */
    MulticastGroupStats getStatistics() const { return stats_; }
    
    /**
     * @brief 重置统计信息
     */
    void resetStatistics() { stats_.reset(); }
    
    /**
     * @brief 生成性能报告
     * @return 性能报告字符串
     */
    std::string generatePerformanceReport() const;
    
    // === 序列化和持久化 ===
    
    /**
     * @brief 序列化组信息
     * @param format 格式（"binary", "json", "xml"）
     * @return 序列化数据
     */
    std::vector<uint8_t> serialize(const std::string& format = "binary") const;
    
    /**
     * @brief 反序列化组信息
     * @param data 序列化数据
     * @param format 格式
     * @return 是否成功
     */
    bool deserialize(const std::vector<uint8_t>& data, const std::string& format = "binary");
    
    /**
     * @brief 导出组配置
     * @param format 格式（"json", "xml", "yaml"）
     * @return 配置字符串
     */
    std::string exportConfig(const std::string& format = "json") const;
    
    /**
     * @brief 导入组配置
     * @param config_str 配置字符串
     * @param format 格式
     * @return 是否成功导入
     */
    bool importConfig(const std::string& config_str, const std::string& format = "json");
    
    // === 工具方法 ===
    
    /**
     * @brief 克隆多播组
     * @return 组的深拷贝
     */
    std::unique_ptr<EnhancedMulticastGroup> clone() const;
    
    /**
     * @brief 合并另一个多播组
     * @param other 另一个组
     * @param conflict_resolution 冲突解决策略
     * @return 合并的成员数量
     */
    size_t merge(const EnhancedMulticastGroup& other, const std::string& conflict_resolution = "union");
    
    /**
     * @brief 分割多播组
     * @param max_members_per_group 每组最大成员数
     * @return 分割后的组列表
     */
    std::vector<std::unique_ptr<EnhancedMulticastGroup>> split(size_t max_members_per_group) const;
    
    /**
     * @brief 获取组详细信息
     * @return 详细信息字符串
     */
    std::string getDetailedInfo() const;

private:
    // 基本属性
    uint16_t group_id_ = 0;
    MulticastGroupType group_type_ = MulticastGroupType::STATIC;
    MulticastMode multicast_mode_ = MulticastMode::RELIABLE;
    bool active_ = true;
    float priority_ = 1.0f;
    
    // 成员管理
    std::unordered_map<AddressEvent::GlobalNeuronId, float> members_; // 神经元ID到权重的映射
    std::vector<PEId> target_pes_;                                    // 目标PE列表
    std::unordered_map<PEId, std::vector<AddressEvent::GlobalNeuronId>> pe_to_neurons_; // PE到神经元的映射
    
    // 多播树
    PEId tree_root_ = INVALID_PE_ID;
    std::unordered_map<PEId, std::vector<RouteDirection>> tree_routes_;
    uint32_t tree_version_ = 0;                                       // 树版本号
    
    // QoS和流量控制
    float max_latency_ = 1000.0f;       // 最大延迟（微秒）
    float min_bandwidth_ = 0.0f;        // 最小带宽要求
    float reliability_requirement_ = 0.9f; // 可靠性要求
    uint32_t max_rate_ = 0;             // 最大传输速率（0表示无限制）
    uint32_t burst_size_ = 0;           // 突发大小
    
    // 统计信息
    MulticastGroupStats stats_;
    
    // 动态适应参数
    float adaptation_threshold_ = 0.1f;  // 适应阈值
    uint32_t adaptation_window_ = 1000;  // 适应窗口（毫秒）
    std::chrono::steady_clock::time_point last_adaptation_;
    
    // 辅助方法
    void updatePEToNeuronsMapping(const MappingSolution& mapping);
    PEId selectOptimalTreeRoot(const HardwareTopology& topology) const;
    std::vector<PEId> computeMinimalSpanningTree(const HardwareTopology& topology, PEId root) const;
    float calculateTreeCost(const HardwareTopology& topology, PEId root) const;
    void validateMemberConsistency() const;
};

/**
 * @brief 多播组管理器
 * 
 * 管理系统中的所有多播组，提供组间协调和全局优化功能。
 */
class MulticastGroupManager {
public:
    MulticastGroupManager() = default;
    virtual ~MulticastGroupManager() = default;
    
    // === 组管理 ===
    
    /**
     * @brief 创建多播组
     * @param group_id 组ID
     * @param members 初始成员列表
     * @param group_type 组类型
     * @return 是否成功创建
     */
    bool createGroup(uint16_t group_id,
                    const std::vector<AddressEvent::GlobalNeuronId>& members,
                    MulticastGroupType group_type = MulticastGroupType::STATIC);
    
    /**
     * @brief 删除多播组
     * @param group_id 组ID
     * @return 是否成功删除
     */
    bool removeGroup(uint16_t group_id);
    
    /**
     * @brief 获取多播组
     * @param group_id 组ID
     * @return 组指针，不存在返回nullptr
     */
    EnhancedMulticastGroup* getGroup(uint16_t group_id);
    const EnhancedMulticastGroup* getGroup(uint16_t group_id) const;
    
    /**
     * @brief 获取所有组
     * @return 组ID到组的映射
     */
    const std::unordered_map<uint16_t, std::unique_ptr<EnhancedMulticastGroup>>& getAllGroups() const {
        return groups_;
    }
    
    /**
     * @brief 获取组数量
     * @return 总组数
     */
    size_t getGroupCount() const { return groups_.size(); }
    
    // === 自动组检测和创建 ===
    
    /**
     * @brief 自动检测潜在多播组
     * @param network 神经网络
     * @param mapping 映射解决方案
     * @param min_group_size 最小组大小
     * @param similarity_threshold 相似度阈值
     * @return 检测到的组数量
     */
    size_t autoDetectGroups(const NeuralNetwork& network,
                           const MappingSolution& mapping,
                           size_t min_group_size = 3,
                           float similarity_threshold = 0.7f);
    
    /**
     * @brief 基于连接模式创建组
     * @param network 神经网络
     * @param mapping 映射解决方案
     * @param pattern_type 模式类型（"fan-out", "convergence", "bidirectional"）
     * @return 创建的组数量
     */
    size_t createGroupsByPattern(const NeuralNetwork& network,
                                const MappingSolution& mapping,
                                const std::string& pattern_type);
    
    /**
     * @brief 基于拓扑距离创建组
     * @param network 神经网络
     * @param mapping 映射解决方案
     * @param topology 硬件拓扑
     * @param max_distance 最大拓扑距离
     * @return 创建的组数量
     */
    size_t createGroupsByTopologyDistance(const NeuralNetwork& network,
                                         const MappingSolution& mapping,
                                         const HardwareTopology& topology,
                                         uint32_t max_distance);
    
    // === 全局优化 ===
    
    /**
     * @brief 优化所有多播组
     * @param topology 硬件拓扑
     * @param optimization_goals 优化目标列表
     * @return 优化统计信息
     */
    std::unordered_map<std::string, float> optimizeAllGroups(
        const HardwareTopology& topology,
        const std::vector<std::string>& optimization_goals = {"latency", "bandwidth"});
    
    /**
     * @brief 解决组间路由冲突
     * @param topology 硬件拓扑
     * @return 解决的冲突数量
     */
    size_t resolveRoutingConflicts(const HardwareTopology& topology);
    
    /**
     * @brief 负载均衡组分布
     * @param topology 硬件拓扑
     * @param rebalance_threshold 重平衡阈值
     * @return 重新分配的组数量
     */
    size_t rebalanceGroupDistribution(const HardwareTopology& topology, float rebalance_threshold = 0.8f);
    
    /**
     * @brief 合并相似的多播组
     * @param similarity_threshold 相似度阈值
     * @return 合并的组数量
     */
    size_t mergeSimilarGroups(float similarity_threshold = 0.8f);
    
    // === 动态管理 ===
    
    /**
     * @brief 动态调整组成员
     * @param network 神经网络
     * @param mapping 映射解决方案
     * @param traffic_data 流量数据
     * @return 调整的组数量
     */
    size_t dynamicMembershipAdjustment(
        const NeuralNetwork& network,
        const MappingSolution& mapping,
        const std::unordered_map<AddressEvent::GlobalNeuronId, float>& traffic_data);
    
    /**
     * @brief 自适应组重构
     * @param topology 硬件拓扑
     * @param performance_threshold 性能阈值
     * @return 重构的组数量
     */
    size_t adaptiveGroupReconstruction(const HardwareTopology& topology, float performance_threshold = 0.7f);
    
    /**
     * @brief 处理网络变化
     * @param network_changes 网络变化描述
     * @param mapping 新的映射解决方案
     * @return 受影响的组数量
     */
    size_t handleNetworkChanges(const std::vector<std::string>& network_changes,
                               const MappingSolution& mapping);
    
    // === 监控和分析 ===
    
    /**
     * @brief 收集全局统计信息
     * @return 全局统计信息
     */
    std::unordered_map<std::string, uint64_t> collectGlobalStatistics() const;
    
    /**
     * @brief 分析组性能
     * @param analysis_type 分析类型（"efficiency", "utilization", "conflicts"）
     * @return 分析结果
     */
    std::unordered_map<uint16_t, float> analyzeGroupPerformance(const std::string& analysis_type) const;
    
    /**
     * @brief 检测性能瓶颈
     * @param topology 硬件拓扑
     * @return 瓶颈组ID和严重程度
     */
    std::vector<std::pair<uint16_t, float>> detectPerformanceBottlenecks(
        const HardwareTopology& topology) const;
    
    /**
     * @brief 生成全局报告
     * @param include_details 是否包含详细信息
     * @return 报告字符串
     */
    std::string generateGlobalReport(bool include_details = false) const;
    
    // === 配置和调优 ===
    
    /**
     * @brief 设置全局优化参数
     * @param parameters 参数映射
     */
    void setGlobalOptimizationParameters(const std::unordered_map<std::string, float>& parameters);
    
    /**
     * @brief 启用/禁用自动优化
     * @param enable 是否启用
     * @param optimization_interval 优化间隔（毫秒）
     */
    void enableAutoOptimization(bool enable, uint32_t optimization_interval = 10000);
    
    /**
     * @brief 设置组创建策略
     * @param strategy 策略名称
     * @param parameters 策略参数
     */
    void setGroupCreationStrategy(const std::string& strategy,
                                 const std::unordered_map<std::string, float>& parameters);
    
    // === 故障处理 ===
    
    /**
     * @brief 处理PE故障
     * @param failed_pe_id 故障PE ID
     * @param topology 硬件拓扑
     * @return 受影响的组数量
     */
    size_t handlePEFailure(PEId failed_pe_id, const HardwareTopology& topology);
    
    /**
     * @brief 恢复故障组
     * @param group_id 组ID
     * @param topology 硬件拓扑
     * @param recovery_strategy 恢复策略
     * @return 是否成功恢复
     */
    bool recoverFailedGroup(uint16_t group_id, const HardwareTopology& topology,
                           const std::string& recovery_strategy = "rebuild");

private:
    std::unordered_map<uint16_t, std::unique_ptr<EnhancedMulticastGroup>> groups_;
    
    // 全局优化参数
    std::unordered_map<std::string, float> global_optimization_params_;
    bool auto_optimization_enabled_ = false;
    uint32_t optimization_interval_ms_ = 10000;
    std::chrono::steady_clock::time_point last_optimization_;
    
    // 组创建策略
    std::string group_creation_strategy_ = "auto_detect";
    std::unordered_map<std::string, float> creation_strategy_params_;
    
    // 故障恢复
    std::unordered_set<uint16_t> failed_groups_;
    std::unordered_map<uint16_t, std::chrono::steady_clock::time_point> failure_timestamps_;
    
    // 辅助方法
    uint16_t generateUniqueGroupId();
    float calculateGroupSimilarity(const EnhancedMulticastGroup& group1, 
                                  const EnhancedMulticastGroup& group2) const;
    std::vector<AddressEvent::GlobalNeuronId> findCommonTargets(
        const std::vector<AddressEvent::GlobalNeuronId>& neurons,
        const NeuralNetwork& network) const;
    bool hasRoutingConflict(const EnhancedMulticastGroup& group1, 
                           const EnhancedMulticastGroup& group2,
                           const HardwareTopology& topology) const;
    void performPeriodicOptimization(const HardwareTopology& topology);
    void redistributeFailedGroupMembers(uint16_t failed_group_id);
};

} // namespace NeuronMapping

#endif // NEURON_MAPPING_MULTICAST_GROUP_H