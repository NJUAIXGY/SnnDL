#ifndef NEURON_MAPPING_ROUTING_AWARE_GRAPH_PARTITIONING_STRATEGY_H
#define NEURON_MAPPING_ROUTING_AWARE_GRAPH_PARTITIONING_STRATEGY_H

#include "strategies/GraphPartitioningStrategy.h"
#include "routing/RoutingAwareMappingSolution.h"
#include "routing/RoutingTableGenerator.h"
#include "routing/MulticastGroup.h"
#include <vector>
#include <unordered_map>
#include <memory>

namespace neuron_mapping {

/**
 * @brief 路由感知的图分割策略
 * 
 * 扩展基础GraphPartitioningStrategy，在分割过程中考虑路由成本、
 * 多播优化和通信模式，生成路由友好的神经元映射。
 */
class RoutingAwareGraphPartitioningStrategy : public GraphPartitioningStrategy {
public:
    /**
     * @brief 路由优化目标枚举
     */
    enum class RoutingOptimizationTarget {
        MINIMIZE_HOPS,          // 最小化跳数
        MINIMIZE_CONGESTION,    // 最小化拥塞
        MAXIMIZE_MULTICAST,     // 最大化多播效率
        BALANCED_LOAD,          // 平衡路由负载
        MINIMIZE_LATENCY,       // 最小化延迟
        HYBRID                  // 混合优化
    };
    
    /**
     * @brief 路由感知配置
     */
    struct RoutingAwareConfig {
        RoutingOptimizationTarget optimization_target = RoutingOptimizationTarget::MINIMIZE_HOPS;
        float routing_weight = 0.4f;               // 路由成本权重
        float communication_weight = 0.3f;         // 通信成本权重
        float load_balance_weight = 0.3f;          // 负载均衡权重
        bool enable_multicast_optimization = true; // 启用多播优化
        bool enable_routing_prediction = true;     // 启用路由预测
        uint32_t max_hop_distance = 5;            // 最大跳数距离
        float congestion_threshold = 0.8f;        // 拥塞阈值
        uint32_t min_multicast_group_size = 3;    // 最小多播组大小
        bool adaptive_partitioning = true;        // 自适应分割
        
        RoutingAwareConfig() = default;
    };

public:
    /**
     * @brief 构造函数
     * @param algorithm 基础分割算法
     * @param routing_config 路由感知配置
     * @param seed 随机种子
     */
    explicit RoutingAwareGraphPartitioningStrategy(
        PartitioningAlgorithm algorithm = PartitioningAlgorithm::MULTILEVEL,
        const RoutingAwareConfig& routing_config = RoutingAwareConfig(),
        uint32_t seed = 42);

    virtual ~RoutingAwareGraphPartitioningStrategy() = default;

    // 重写基类接口以支持路由感知
    std::unique_ptr<NeuronMapping::MappingSolution> mapNetwork(
        const NeuronMapping::NeuralNetwork& network,
        const NeuronMapping::HardwareTopology& topology,
        const NeuronMapping::MappingConfig& config) override;

    std::string getName() const override { return "RoutingAwareGraphPartitioning"; }
    std::string getDescription() const override;

    // === 路由感知特有方法 ===
    
    /**
     * @brief 生成路由感知的映射解决方案
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param config 映射配置
     * @return 路由感知映射解决方案
     */
    std::unique_ptr<NeuronMapping::RoutingAwareMappingSolution> mapNetworkWithRouting(
        const NeuronMapping::NeuralNetwork& network,
        const NeuronMapping::HardwareTopology& topology,
        const NeuronMapping::MappingConfig& config);
    
    /**
     * @brief 设置路由感知配置
     * @param config 配置参数
     */
    void setRoutingConfig(const RoutingAwareConfig& config) { routing_config_ = config; }
    
    /**
     * @brief 获取路由感知配置
     * @return 配置参数
     */
    const RoutingAwareConfig& getRoutingConfig() const { return routing_config_; }
    
    /**
     * @brief 设置路由表生成器
     * @param generator 生成器实例
     */
    void setRoutingTableGenerator(std::shared_ptr<NeuronMapping::RoutingTableGenerator> generator) {
        routing_generator_ = generator;
    }
    
    /**
     * @brief 设置多播组管理器
     * @param manager 管理器实例
     */
    void setMulticastGroupManager(std::shared_ptr<NeuronMapping::MulticastGroupManager> manager) {
        multicast_manager_ = manager;
    }

private:
    // 路由感知配置
    RoutingAwareConfig routing_config_;
    
    // 路由组件
    std::shared_ptr<NeuronMapping::RoutingTableGenerator> routing_generator_;
    std::shared_ptr<NeuronMapping::MulticastGroupManager> multicast_manager_;
    
    // 路由感知的图数据结构
    struct RoutingAwareEdge : public Edge {
        float routing_cost;           // 路由成本
        uint32_t hop_count;          // 跳数
        bool is_multicast_candidate; // 是否为多播候选
        float communication_volume;  // 通信量
        
        RoutingAwareEdge(NeuronMapping::NeuronId s, NeuronMapping::NeuronId t, float w)
            : Edge(s, t, w), routing_cost(0.0f), hop_count(0), 
              is_multicast_candidate(false), communication_volume(0.0f) {}
    };
    
    struct RoutingAwareVertex : public Vertex {
        NeuronMapping::PEId preferred_pe;    // 首选PE
        float routing_load;                  // 路由负载
        std::vector<uint16_t> multicast_groups; // 所属多播组
        float communication_centrality;     // 通信中心性
        
        explicit RoutingAwareVertex(NeuronMapping::NeuronId vertex_id, float w = 1.0f)
            : Vertex(vertex_id, w), preferred_pe(NeuronMapping::INVALID_PE_ID),
              routing_load(0.0f), communication_centrality(0.0f) {}
    };
    
    struct RoutingAwarePartition : public Partition {
        float total_routing_cost;           // 总路由成本
        float internal_communication_ratio; // 内部通信比例
        std::vector<uint16_t> multicast_groups; // 包含的多播组
        float congestion_level;             // 拥塞级别
        uint32_t max_hop_count;            // 最大跳数
        
        RoutingAwarePartition() : Partition(), total_routing_cost(0.0f),
                                internal_communication_ratio(0.0f),
                                congestion_level(0.0f), max_hop_count(0) {}
    };
    
    // 路由感知的图分割算法
    std::vector<RoutingAwarePartition> routingAwarePartitionGraph(
        const NeuronMapping::NeuralNetwork& network,
        const NeuronMapping::HardwareTopology& topology,
        const NeuronMapping::MappingConfig& config);
    
    // 路由成本计算
    void calculateRoutingCosts(
        const NeuronMapping::NeuralNetwork& network,
        const NeuronMapping::HardwareTopology& topology);
    
    /**
     * @brief 计算边的路由成本
     * @param edge 边
     * @param topology 硬件拓扑
     * @return 路由成本
     */
    float calculateEdgeRoutingCost(
        const RoutingAwareEdge& edge,
        const NeuronMapping::HardwareTopology& topology) const;
    
    /**
     * @brief 估算神经元间通信量
     * @param network 神经网络
     * @param source 源神经元
     * @param target 目标神经元
     * @return 通信量估算
     */
    float estimateCommunicationVolume(
        const NeuronMapping::NeuralNetwork& network,
        NeuronMapping::NeuronId source,
        NeuronMapping::NeuronId target) const;
    
    // 多播优化
    /**
     * @brief 检测多播模式
     * @param network 神经网络
     * @return 多播候选组
     */
    std::vector<std::vector<NeuronMapping::NeuronId>> detectMulticastPatterns(
        const NeuronMapping::NeuralNetwork& network);
    
    /**
     * @brief 优化多播组分配
     * @param partitions 分区列表
     * @param multicast_patterns 多播模式
     * @param topology 硬件拓扑
     */
    void optimizeMulticastGroupAssignment(
        std::vector<RoutingAwarePartition>& partitions,
        const std::vector<std::vector<NeuronMapping::NeuronId>>& multicast_patterns,
        const NeuronMapping::HardwareTopology& topology);
    
    // 路由感知的分区质量评估
    /**
     * @brief 计算分区的路由质量
     * @param partitions 分区列表
     * @param topology 硬件拓扑
     * @return 质量分数
     */
    float calculateRoutingAwarePartitionQuality(
        const std::vector<RoutingAwarePartition>& partitions,
        const NeuronMapping::HardwareTopology& topology) const;
    
    /**
     * @brief 计算分区间通信成本
     * @param partition1 分区1
     * @param partition2 分区2
     * @param topology 硬件拓扑
     * @return 通信成本
     */
    float calculateInterPartitionCommunicationCost(
        const RoutingAwarePartition& partition1,
        const RoutingAwarePartition& partition2,
        const NeuronMapping::HardwareTopology& topology) const;
    
    // 拥塞感知分割
    /**
     * @brief 应用拥塞感知优化
     * @param partitions 分区列表
     * @param topology 硬件拓扑
     */
    void applyCongestionAwareOptimization(
        std::vector<RoutingAwarePartition>& partitions,
        const NeuronMapping::HardwareTopology& topology);
    
    /**
     * @brief 估算PE拥塞级别
     * @param pe_id PE标识
     * @param partition 分区
     * @param topology 硬件拓扑
     * @return 拥塞级别 [0,1]
     */
    float estimatePECongestionLevel(
        NeuronMapping::PEId pe_id,
        const RoutingAwarePartition& partition,
        const NeuronMapping::HardwareTopology& topology) const;
    
    // 自适应分割
    /**
     * @brief 自适应调整分割参数
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param current_quality 当前质量
     */
    void adaptiveParameterAdjustment(
        const NeuronMapping::NeuralNetwork& network,
        const NeuronMapping::HardwareTopology& topology,
        float current_quality);
    
    /**
     * @brief 分析网络通信特征
     * @param network 神经网络
     * @return 通信特征统计
     */
    std::unordered_map<std::string, float> analyzeNetworkCommunicationCharacteristics(
        const NeuronMapping::NeuralNetwork& network) const;
    
    // 路由预测
    /**
     * @brief 预测路由性能
     * @param partitions 分区列表
     * @param topology 硬件拓扑
     * @return 预测的性能指标
     */
    std::unordered_map<std::string, float> predictRoutingPerformance(
        const std::vector<RoutingAwarePartition>& partitions,
        const NeuronMapping::HardwareTopology& topology) const;
    
    /**
     * @brief 基于预测调整分割策略
     * @param predicted_performance 预测性能
     * @param target_performance 目标性能
     */
    void adjustStrategyBasedOnPrediction(
        const std::unordered_map<std::string, float>& predicted_performance,
        const std::unordered_map<std::string, float>& target_performance);
    
    // 专门的分割算法（路由感知版本）
    /**
     * @brief 路由感知的多级分割
     * @param num_partitions 分区数
     * @param topology 硬件拓扑
     * @return 分区列表
     */
    std::vector<RoutingAwarePartition> routingAwareMultilevelPartitioning(
        uint32_t num_partitions,
        const NeuronMapping::HardwareTopology& topology);
    
    /**
     * @brief 路由感知的谱分割
     * @param num_partitions 分区数
     * @param topology 硬件拓扑
     * @return 分区列表
     */
    std::vector<RoutingAwarePartition> routingAwareSpectralPartitioning(
        uint32_t num_partitions,
        const NeuronMapping::HardwareTopology& topology);
    
    /**
     * @brief 路由感知的递归二分
     * @param num_partitions 分区数
     * @param topology 硬件拓扑
     * @return 分区列表
     */
    std::vector<RoutingAwarePartition> routingAwareRecursiveBisection(
        uint32_t num_partitions,
        const NeuronMapping::HardwareTopology& topology);
    
    // 后处理优化
    /**
     * @brief 后处理路由优化
     * @param partitions 分区列表
     * @param topology 硬件拓扑
     */
    void postProcessRoutingOptimization(
        std::vector<RoutingAwarePartition>& partitions,
        const NeuronMapping::HardwareTopology& topology);
    
    /**
     * @brief 边界神经元重分配
     * @param partitions 分区列表
     * @param topology 硬件拓扑
     * @return 重分配的神经元数量
     */
    size_t boundaryNeuronReassignment(
        std::vector<RoutingAwarePartition>& partitions,
        const NeuronMapping::HardwareTopology& topology);
    
    /**
     * @brief 多播组一致性调整
     * @param partitions 分区列表
     * @param multicast_patterns 多播模式
     */
    void multicastGroupConsistencyAdjustment(
        std::vector<RoutingAwarePartition>& partitions,
        const std::vector<std::vector<NeuronMapping::NeuronId>>& multicast_patterns);
    
    // 工具方法
    /**
     * @brief 构建路由感知图
     * @param network 神经网络
     * @param topology 硬件拓扑
     */
    void buildRoutingAwareGraph(
        const NeuronMapping::NeuralNetwork& network,
        const NeuronMapping::HardwareTopology& topology);
    
    /**
     * @brief 将基础分区转换为路由感知分区
     * @param base_partitions 基础分区
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @return 路由感知分区
     */
    std::vector<RoutingAwarePartition> convertToRoutingAwarePartitions(
        const std::vector<Partition>& base_partitions,
        const NeuronMapping::NeuralNetwork& network,
        const NeuronMapping::HardwareTopology& topology);
    
    /**
     * @brief 生成最终的路由感知映射解决方案
     * @param partitions 路由感知分区
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @return 路由感知映射解决方案
     */
    std::unique_ptr<NeuronMapping::RoutingAwareMappingSolution> generateFinalSolution(
        const std::vector<RoutingAwarePartition>& partitions,
        const NeuronMapping::NeuralNetwork& network,
        const NeuronMapping::HardwareTopology& topology);
    
    // 调试和统计
    /**
     * @brief 生成路由感知分割统计
     * @param partitions 分区列表
     * @param topology 硬件拓扑
     * @return 统计信息
     */
    std::unordered_map<std::string, float> generateRoutingAwareStatistics(
        const std::vector<RoutingAwarePartition>& partitions,
        const NeuronMapping::HardwareTopology& topology) const;
    
    /**
     * @brief 验证路由感知分割结果
     * @param partitions 分区列表
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @return 验证错误列表
     */
    std::vector<std::string> validateRoutingAwarePartitioning(
        const std::vector<RoutingAwarePartition>& partitions,
        const NeuronMapping::NeuralNetwork& network,
        const NeuronMapping::HardwareTopology& topology) const;

private:
    // 路由感知图数据（扩展基类）
    std::vector<RoutingAwareVertex> routing_vertices_;
    std::vector<RoutingAwareEdge> routing_edges_;
    std::unordered_map<NeuronMapping::NeuronId, size_t> routing_vertex_map_;
    
    // 多播模式缓存
    std::vector<std::vector<NeuronMapping::NeuronId>> cached_multicast_patterns_;
    bool multicast_patterns_valid_ = false;
    
    // 通信特征缓存
    std::unordered_map<std::string, float> cached_communication_characteristics_;
    bool communication_characteristics_valid_ = false;
    
    // 性能预测模型参数
    struct PredictionModel {
        float hop_cost_coefficient = 1.0f;
        float congestion_penalty = 2.0f;
        float multicast_benefit = 0.8f;
        float load_imbalance_penalty = 1.5f;
    } prediction_model_;
    
    // 自适应参数
    float adaptive_learning_rate_ = 0.1f;
    std::unordered_map<std::string, float> adaptive_parameters_;
    
    void invalidateCaches() {
        multicast_patterns_valid_ = false;
        communication_characteristics_valid_ = false;
    }
};

/**
 * @brief 路由感知图分割策略工厂
 */
class RoutingAwareGraphPartitioningStrategyFactory {
public:
    /**
     * @brief 创建标准路由感知策略
     * @param optimization_target 优化目标
     * @return 策略实例
     */
    static std::unique_ptr<RoutingAwareGraphPartitioningStrategy> createStandardStrategy(
        RoutingAwareGraphPartitioningStrategy::RoutingOptimizationTarget optimization_target =
            RoutingAwareGraphPartitioningStrategy::RoutingOptimizationTarget::MINIMIZE_HOPS);
    
    /**
     * @brief 创建多播优化策略
     * @param min_group_size 最小多播组大小
     * @return 策略实例
     */
    static std::unique_ptr<RoutingAwareGraphPartitioningStrategy> createMulticastOptimizedStrategy(
        uint32_t min_group_size = 3);
    
    /**
     * @brief 创建低延迟策略
     * @param max_hop_distance 最大跳数
     * @return 策略实例
     */
    static std::unique_ptr<RoutingAwareGraphPartitioningStrategy> createLowLatencyStrategy(
        uint32_t max_hop_distance = 3);
    
    /**
     * @brief 创建负载均衡策略
     * @param congestion_threshold 拥塞阈值
     * @return 策略实例
     */
    static std::unique_ptr<RoutingAwareGraphPartitioningStrategy> createLoadBalancedStrategy(
        float congestion_threshold = 0.7f);
    
    /**
     * @brief 创建自定义策略
     * @param algorithm 基础算法
     * @param routing_config 路由配置
     * @return 策略实例
     */
    static std::unique_ptr<RoutingAwareGraphPartitioningStrategy> createCustomStrategy(
        RoutingAwareGraphPartitioningStrategy::PartitioningAlgorithm algorithm,
        const RoutingAwareGraphPartitioningStrategy::RoutingAwareConfig& routing_config);

private:
    RoutingAwareGraphPartitioningStrategyFactory() = default;
};

} // namespace neuron_mapping

#endif // NEURON_MAPPING_ROUTING_AWARE_GRAPH_PARTITIONING_STRATEGY_H