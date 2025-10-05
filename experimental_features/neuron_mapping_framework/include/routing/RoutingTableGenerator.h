#ifndef NEURON_MAPPING_ROUTING_TABLE_GENERATOR_H
#define NEURON_MAPPING_ROUTING_TABLE_GENERATOR_H

#include "RoutingTable.h"
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

namespace NeuronMapping {

/**
 * @brief 路由生成策略枚举
 */
enum class RoutingStrategy {
    SHORTEST_PATH,          // 最短路径路由
    LOAD_BALANCED,         // 负载均衡路由
    MINIMUM_CONGESTION,    // 最小拥塞路由
    MULTICAST_OPTIMIZED,   // 多播优化路由
    FAULT_TOLERANT,        // 容错路由
    HYBRID                 // 混合策略
};

/**
 * @brief 路由压缩方法枚举
 */
enum class CompressionMethod {
    NONE,                  // 无压缩
    PREFIX_AGGREGATION,    // 前缀聚合
    ROUTE_MERGING,         // 路由合并
    MULTICAST_TREES,       // 多播树压缩
    HIERARCHICAL          // 层次化压缩
};

/**
 * @brief 路由生成配置
 */
struct RoutingGenerationConfig {
    RoutingStrategy strategy = RoutingStrategy::SHORTEST_PATH;
    CompressionMethod compression = CompressionMethod::PREFIX_AGGREGATION;
    bool enable_multicast = true;           // 启用多播优化
    bool enable_compression = true;         // 启用路由压缩
    bool enable_fault_tolerance = false;   // 启用容错路由
    float load_balance_factor = 0.7f;      // 负载均衡因子
    uint32_t max_path_length = 10;         // 最大路径长度
    uint32_t max_routes_per_entry = 4;     // 每个表项的最大路由数
    float compression_ratio_target = 0.5f; // 目标压缩比
    uint8_t multicast_tree_depth = 4;      // 多播树深度限制
    
    RoutingGenerationConfig() = default;
};

/**
 * @brief 路由生成统计信息
 */
struct RoutingGenerationStats {
    size_t total_neurons = 0;              // 总神经元数
    size_t total_connections = 0;          // 总连接数
    size_t total_routes_generated = 0;     // 生成的路由数
    size_t routing_table_entries = 0;      // 路由表项数
    size_t multicast_groups = 0;           // 多播组数
    float average_path_length = 0.0f;      // 平均路径长度
    float compression_ratio = 0.0f;        // 压缩比
    float generation_time_ms = 0.0f;       // 生成时间（毫秒）
    size_t memory_usage_bytes = 0;         // 内存使用量
    
    void reset() {
        total_neurons = 0;
        total_connections = 0;
        total_routes_generated = 0;
        routing_table_entries = 0;
        multicast_groups = 0;
        average_path_length = 0.0f;
        compression_ratio = 0.0f;
        generation_time_ms = 0.0f;
        memory_usage_bytes = 0;
    }
    
    std::string toString() const;
};

/**
 * @brief 路由表生成器
 * 
 * 基于SpiNNaker编译原理，将神经网络映射解决方案转换为
 * 分布式路由表，支持多种路由策略和优化方法。
 */
class RoutingTableGenerator {
public:
    RoutingTableGenerator() = default;
    explicit RoutingTableGenerator(const RoutingGenerationConfig& config);
    virtual ~RoutingTableGenerator() = default;
    
    // === 主要生成接口 ===
    
    /**
     * @brief 生成分布式路由表
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param mapping 映射解决方案
     * @param config 生成配置
     * @return PE到路由表的映射
     */
    std::unordered_map<PEId, std::unique_ptr<RoutingTable>> generateRoutingTables(
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingSolution& mapping,
        const RoutingGenerationConfig& config = RoutingGenerationConfig());
    
    /**
     * @brief 生成单个PE的路由表
     * @param pe_id PE标识
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param mapping 映射解决方案
     * @param config 生成配置
     * @return 路由表
     */
    std::unique_ptr<RoutingTable> generatePERoutingTable(
        PEId pe_id,
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingSolution& mapping,
        const RoutingGenerationConfig& config = RoutingGenerationConfig());
    
    // === 路由策略实现 ===
    
    /**
     * @brief 最短路径路由生成
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param mapping 映射解决方案
     * @return 路由表映射
     */
    std::unordered_map<PEId, std::unique_ptr<RoutingTable>> generateShortestPathRouting(
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingSolution& mapping);
    
    /**
     * @brief 负载均衡路由生成
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param mapping 映射解决方案
     * @param load_factor 负载均衡因子
     * @return 路由表映射
     */
    std::unordered_map<PEId, std::unique_ptr<RoutingTable>> generateLoadBalancedRouting(
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingSolution& mapping,
        float load_factor = 0.7f);
    
    /**
     * @brief 多播优化路由生成
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param mapping 映射解决方案
     * @return 路由表映射
     */
    std::unordered_map<PEId, std::unique_ptr<RoutingTable>> generateMulticastOptimizedRouting(
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingSolution& mapping);
    
    /**
     * @brief 容错路由生成
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param mapping 映射解决方案
     * @param redundancy_level 冗余级别
     * @return 路由表映射
     */
    std::unordered_map<PEId, std::unique_ptr<RoutingTable>> generateFaultTolerantRouting(
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingSolution& mapping,
        uint8_t redundancy_level = 2);
    
    // === 路由压缩和优化 ===
    
    /**
     * @brief 压缩路由表
     * @param routing_tables 路由表映射
     * @param method 压缩方法
     * @param target_ratio 目标压缩比
     * @return 压缩统计信息
     */
    std::unordered_map<std::string, float> compressRoutingTables(
        std::unordered_map<PEId, std::unique_ptr<RoutingTable>>& routing_tables,
        CompressionMethod method = CompressionMethod::PREFIX_AGGREGATION,
        float target_ratio = 0.5f);
    
    /**
     * @brief 前缀聚合压缩
     * @param table 路由表
     * @return 压缩的表项数量
     */
    size_t performPrefixAggregation(RoutingTable& table);
    
    /**
     * @brief 路由合并压缩
     * @param table 路由表
     * @return 合并的表项数量
     */
    size_t performRouteMerging(RoutingTable& table);
    
    /**
     * @brief 多播树压缩
     * @param routing_tables 路由表映射
     * @param topology 硬件拓扑
     * @return 压缩的路由数量
     */
    size_t performMulticastTreeCompression(
        std::unordered_map<PEId, std::unique_ptr<RoutingTable>>& routing_tables,
        const HardwareTopology& topology);
    
    // === 多播路由生成 ===
    
    /**
     * @brief 检测多播组
     * @param network 神经网络
     * @param mapping 映射解决方案
     * @param min_group_size 最小组大小
     * @return 多播组列表
     */
    std::vector<MulticastGroup> detectMulticastGroups(
        const NeuralNetwork& network,
        const MappingSolution& mapping,
        size_t min_group_size = 3);
    
    /**
     * @brief 构建多播树
     * @param group 多播组
     * @param topology 硬件拓扑
     * @param root_pe 根PE
     * @return 多播树路由
     */
    std::unordered_map<PEId, std::vector<RouteDirection>> buildMulticastTree(
        const MulticastGroup& group,
        const HardwareTopology& topology,
        PEId root_pe);
    
    /**
     * @brief 优化多播树
     * @param multicast_trees 多播树映射
     * @param topology 硬件拓扑
     * @return 优化统计信息
     */
    std::unordered_map<std::string, float> optimizeMulticastTrees(
        std::unordered_map<uint16_t, std::unordered_map<PEId, std::vector<RouteDirection>>>& multicast_trees,
        const HardwareTopology& topology);
    
    // === 全局神经元ID管理 ===
    
    /**
     * @brief 分配全局神经元ID
     * @param network 神经网络
     * @param mapping 映射解决方案
     * @return 本地ID到全局ID的映射
     */
    std::unordered_map<NeuronId, AddressEvent::GlobalNeuronId> assignGlobalNeuronIds(
        const NeuralNetwork& network,
        const MappingSolution& mapping);
    
    /**
     * @brief 生成神经元ID分配表
     * @param network 神经网络
     * @param mapping 映射解决方案
     * @param pe_id PE标识
     * @return PE内的神经元ID映射
     */
    std::unordered_map<NeuronId, AddressEvent::GlobalNeuronId> generateNeuronIdMapping(
        const NeuralNetwork& network,
        const MappingSolution& mapping,
        PEId pe_id);
    
    /**
     * @brief 验证全局ID分配
     * @param global_id_mapping 全局ID映射
     * @param network 神经网络
     * @return 验证错误列表
     */
    std::vector<std::string> validateGlobalIdAssignment(
        const std::unordered_map<NeuronId, AddressEvent::GlobalNeuronId>& global_id_mapping,
        const NeuralNetwork& network);
    
    // === 路径计算和分析 ===
    
    /**
     * @brief 计算神经元间路径
     * @param source_neuron 源神经元
     * @param target_neuron 目标神经元
     * @param mapping 映射解决方案
     * @param topology 硬件拓扑
     * @return 路径上的PE列表
     */
    std::vector<PEId> calculateNeuronPath(
        NeuronId source_neuron,
        NeuronId target_neuron,
        const MappingSolution& mapping,
        const HardwareTopology& topology);
    
    /**
     * @brief 分析路由路径长度分布
     * @param network 神经网络
     * @param mapping 映射解决方案
     * @param topology 硬件拓扑
     * @return 路径长度统计
     */
    std::unordered_map<uint32_t, uint32_t> analyzePathLengthDistribution(
        const NeuralNetwork& network,
        const MappingSolution& mapping,
        const HardwareTopology& topology);
    
    /**
     * @brief 识别路由瓶颈
     * @param network 神经网络
     * @param mapping 映射解决方案
     * @param topology 硬件拓扑
     * @return 瓶颈PE和严重程度
     */
    std::vector<std::pair<PEId, float>> identifyRoutingBottlenecks(
        const NeuralNetwork& network,
        const MappingSolution& mapping,
        const HardwareTopology& topology);
    
    // === 统计和监控 ===
    
    /**
     * @brief 获取生成统计信息
     * @return 统计信息
     */
    RoutingGenerationStats getGenerationStatistics() const { return stats_; }
    
    /**
     * @brief 重置统计信息
     */
    void resetStatistics() { stats_.reset(); }
    
    /**
     * @brief 分析路由表质量
     * @param routing_tables 路由表映射
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @return 质量指标
     */
    std::unordered_map<std::string, float> analyzeRoutingQuality(
        const std::unordered_map<PEId, std::unique_ptr<RoutingTable>>& routing_tables,
        const NeuralNetwork& network,
        const HardwareTopology& topology);
    
    /**
     * @brief 验证路由表正确性
     * @param routing_tables 路由表映射
     * @param network 神经网络
     * @param mapping 映射解决方案
     * @param topology 硬件拓扑
     * @return 验证错误列表
     */
    std::vector<std::string> validateRoutingTables(
        const std::unordered_map<PEId, std::unique_ptr<RoutingTable>>& routing_tables,
        const NeuralNetwork& network,
        const MappingSolution& mapping,
        const HardwareTopology& topology);
    
    // === 配置和调优 ===
    
    /**
     * @brief 设置生成配置
     * @param config 配置参数
     */
    void setConfig(const RoutingGenerationConfig& config) { config_ = config; }
    
    /**
     * @brief 获取生成配置
     * @return 配置参数
     */
    const RoutingGenerationConfig& getConfig() const { return config_; }
    
    /**
     * @brief 自动调优生成参数
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param mapping 映射解决方案
     * @return 优化后的配置
     */
    RoutingGenerationConfig autoTuneParameters(
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingSolution& mapping);
    
    // === 调试和可视化 ===
    
    /**
     * @brief 生成路由表报告
     * @param routing_tables 路由表映射
     * @param format 报告格式 ("text", "json", "html")
     * @return 报告内容
     */
    std::string generateRoutingReport(
        const std::unordered_map<PEId, std::unique_ptr<RoutingTable>>& routing_tables,
        const std::string& format = "text");
    
    /**
     * @brief 导出路由可视化数据
     * @param routing_tables 路由表映射
     * @param topology 硬件拓扑
     * @param format 数据格式 ("graphviz", "json", "csv")
     * @return 可视化数据
     */
    std::string exportVisualizationData(
        const std::unordered_map<PEId, std::unique_ptr<RoutingTable>>& routing_tables,
        const HardwareTopology& topology,
        const std::string& format = "graphviz");
    
    /**
     * @brief 调试单个连接的路由
     * @param source_neuron 源神经元
     * @param target_neuron 目标神经元
     * @param routing_tables 路由表映射
     * @param mapping 映射解决方案
     * @param topology 硬件拓扑
     * @return 调试信息
     */
    std::string debugConnectionRouting(
        NeuronId source_neuron,
        NeuronId target_neuron,
        const std::unordered_map<PEId, std::unique_ptr<RoutingTable>>& routing_tables,
        const MappingSolution& mapping,
        const HardwareTopology& topology);

    /**
     * @brief 生成并导出路由相关产物（公共API）
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param mapping 映射方案
     * @param out_dir 输出目录
     * @param config 生成配置（可选）
     * @return 是否成功
     */
    bool exportArtifacts(
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingSolution& mapping,
        const std::string& out_dir,
        const RoutingGenerationConfig& config = RoutingGenerationConfig());

private:
    RoutingGenerationConfig config_;
    RoutingGenerationStats stats_;
    
    // 内部数据结构
    std::unordered_map<NeuronId, AddressEvent::GlobalNeuronId> global_id_mapping_;
    std::unordered_map<uint32_t, MulticastGroup> detected_multicast_groups_;
    std::unordered_map<PEId, std::vector<PEId>> topology_paths_cache_;
    
    // 辅助方法
    void updateStatistics(const std::string& metric, float value);
    RouteDirection getRouteDirection(PEId from_pe, PEId to_pe, const HardwareTopology& topology);
    std::vector<PEId> findShortestPath(PEId source, PEId target, const HardwareTopology& topology);
    std::vector<std::vector<PEId>> findKShortestPaths(PEId source, PEId target, 
                                                     const HardwareTopology& topology, uint8_t k);
    
    // 路由表项生成
    void generateUnicastRoutes(
        RoutingTable& table,
        const NeuralNetwork& network,
        const MappingSolution& mapping,
        const HardwareTopology& topology,
        PEId pe_id);
    
    void generateMulticastRoutes(
        RoutingTable& table,
        const std::vector<MulticastGroup>& multicast_groups,
        const HardwareTopology& topology,
        PEId pe_id);
    
    // 压缩算法实现
    bool canMergeRoutingEntries(const RoutingEntry& entry1, const RoutingEntry& entry2);
    RoutingEntry mergeRoutingEntries(const RoutingEntry& entry1, const RoutingEntry& entry2);
    uint32_t findCommonPrefix(uint32_t key1, uint32_t key2, uint32_t mask1, uint32_t mask2);
    
    // 多播优化
    PEId findOptimalMulticastRoot(const MulticastGroup& group, const HardwareTopology& topology);
    std::vector<PEId> computeMulticastSpanningTree(const std::vector<PEId>& target_pes, 
                                                  PEId root_pe, const HardwareTopology& topology);
    
    // 负载均衡
    float calculatePELoad(PEId pe_id, const NeuralNetwork& network, const MappingSolution& mapping);
    std::vector<PEId> selectAlternativePaths(PEId source, PEId target, 
                                           const HardwareTopology& topology, uint8_t num_paths);
    
    // 验证和调试
    bool validateRouteConnectivity(const RoutingTable& table, const HardwareTopology& topology);
    void logRoutingDecision(const std::string& decision, const std::vector<std::string>& parameters);
    
    // 缓存管理
    void clearPathCache() { topology_paths_cache_.clear(); }
    const std::vector<PEId>* getCachedPath(PEId source, PEId target);
    void cachePath(PEId source, PEId target, const std::vector<PEId>& path);
};

/**
 * @brief 路由表生成器工厂
 * 
 * 用于创建不同类型和配置的路由表生成器。
 */
class RoutingTableGeneratorFactory {
public:
    /**
     * @brief 创建标准路由表生成器
     * @param strategy 路由策略
     * @return 生成器实例
     */
    static std::unique_ptr<RoutingTableGenerator> createStandardGenerator(
        RoutingStrategy strategy = RoutingStrategy::SHORTEST_PATH);
    
    /**
     * @brief 创建优化的路由表生成器
     * @param enable_multicast 启用多播优化
     * @param enable_compression 启用压缩
     * @return 生成器实例
     */
    static std::unique_ptr<RoutingTableGenerator> createOptimizedGenerator(
        bool enable_multicast = true,
        bool enable_compression = true);
    
    /**
     * @brief 创建容错路由表生成器
     * @param redundancy_level 冗余级别
     * @return 生成器实例
     */
    static std::unique_ptr<RoutingTableGenerator> createFaultTolerantGenerator(
        uint8_t redundancy_level = 2);
    
    /**
     * @brief 创建自定义路由表生成器
     * @param config 自定义配置
     * @return 生成器实例
     */
    static std::unique_ptr<RoutingTableGenerator> createCustomGenerator(
        const RoutingGenerationConfig& config);
    
    /**
     * @brief 获取可用的路由策略列表
     * @return 策略名称列表
     */
    static std::vector<std::string> getAvailableStrategies();
    
    /**
     * @brief 获取可用的压缩方法列表
     * @return 压缩方法名称列表
     */
    static std::vector<std::string> getAvailableCompressionMethods();

private:
    RoutingTableGeneratorFactory() = default;
};

} // namespace NeuronMapping

#endif // NEURON_MAPPING_ROUTING_TABLE_GENERATOR_H
