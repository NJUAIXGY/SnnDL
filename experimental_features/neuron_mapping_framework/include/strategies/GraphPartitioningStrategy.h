#ifndef NEURON_MAPPING_FRAMEWORK_STRATEGIES_GRAPH_PARTITIONING_STRATEGY_H
#define NEURON_MAPPING_FRAMEWORK_STRATEGIES_GRAPH_PARTITIONING_STRATEGY_H

#include "MappingStrategy.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <random>

namespace neuron_mapping {

/**
 * @brief 图分割映射策略
 * 
 * 使用图分割算法将神经网络划分为多个子图，并将每个子图映射到不同的PE上。
 * 支持多种图分割算法：谱分割、Kernighan-Lin算法、多级分割等。
 */
class GraphPartitioningStrategy : public MappingStrategy {
public:
    /**
     * @brief 图分割算法类型
     */
    enum class PartitioningAlgorithm {
        SPECTRAL,           // 谱分割
        KERNIGHAN_LIN,     // Kernighan-Lin算法
        MULTILEVEL,        // 多级分割
        RECURSIVE_BISECTION, // 递归二分
        COMMUNITY_DETECTION  // 社区检测
    };

    /**
     * @brief 分割质量度量
     */
    enum class QualityMetric {
        EDGE_CUT,          // 边切割数
        COMMUNICATION_VOLUME, // 通信量
        NORMALIZED_CUT,    // 归一化切割
        CONDUCTANCE        // 电导率
    };

    /**
     * @brief 构造函数
     * @param algorithm 分割算法类型
     * @param quality_metric 质量度量类型
     * @param seed 随机种子
     */
    explicit GraphPartitioningStrategy(
        PartitioningAlgorithm algorithm = PartitioningAlgorithm::MULTILEVEL,
        QualityMetric quality_metric = QualityMetric::EDGE_CUT,
        uint32_t seed = 42);

    virtual ~GraphPartitioningStrategy() = default;

    // 实现基类接口
    std::unique_ptr<NeuronMapping::MappingSolution> mapNetwork(
        const NeuronMapping::NeuralNetwork& network,
        const NeuronMapping::HardwareTopology& topology,
        const NeuronMapping::MappingConfig& config) override;

    std::string getName() const override { return "GraphPartitioning"; }
    std::string getDescription() const override;

    // 参数设置接口
    void setAlgorithm(PartitioningAlgorithm algorithm) { algorithm_ = algorithm; }
    void setQualityMetric(QualityMetric metric) { quality_metric_ = metric; }
    void setBalanceThreshold(double threshold) { balance_threshold_ = threshold; }
    void setMaxIterations(uint32_t iterations) { max_iterations_ = iterations; }

private:
    // 图数据结构
    struct Edge {
        NeuronMapping::NeuronId source;
        NeuronMapping::NeuronId target;
        float weight;
        
        Edge(NeuronMapping::NeuronId s, NeuronMapping::NeuronId t, float w)
            : source(s), target(t), weight(w) {}
    };

    struct Vertex {
        NeuronMapping::NeuronId id;
        float weight;
        std::vector<size_t> edges; // 邻接边的索引
        
        explicit Vertex(NeuronMapping::NeuronId vertex_id, float w = 1.0f)
            : id(vertex_id), weight(w) {}
    };

    struct Partition {
        std::vector<NeuronMapping::NeuronId> vertices;
        float total_weight;
        NeuronMapping::PEId pe_id;
        
        Partition() : total_weight(0.0f), pe_id(NeuronMapping::INVALID_PE_ID) {}
    };

    // 算法配置
    PartitioningAlgorithm algorithm_;
    QualityMetric quality_metric_;
    double balance_threshold_;
    uint32_t max_iterations_;
    uint32_t seed_;
    std::mt19937 rng_;

    // 图数据
    std::vector<Vertex> vertices_;
    std::vector<Edge> edges_;
    std::unordered_map<NeuronMapping::NeuronId, size_t> vertex_map_;

    // 核心算法实现
    std::vector<Partition> partitionGraph(
        const NeuronMapping::NeuralNetwork& network,
        const NeuronMapping::HardwareTopology& topology,
        const NeuronMapping::MappingConfig& config);

    // 不同分割算法
    std::vector<Partition> spectralPartitioning(
        uint32_t num_partitions,
        const NeuronMapping::HardwareTopology& topology);
    
    std::vector<Partition> kernighanLinPartitioning(
        uint32_t num_partitions,
        const NeuronMapping::HardwareTopology& topology);
    
    std::vector<Partition> multilevelPartitioning(
        uint32_t num_partitions,
        const NeuronMapping::HardwareTopology& topology);
    
    std::vector<Partition> recursiveBisection(
        uint32_t num_partitions,
        const NeuronMapping::HardwareTopology& topology);
    
    std::vector<Partition> communityDetection(
        uint32_t num_partitions,
        const NeuronMapping::HardwareTopology& topology);

    // 辅助方法
    void buildGraphFromNetwork(const NeuronMapping::NeuralNetwork& network);
    float calculatePartitionQuality(const std::vector<Partition>& partitions);
    bool isBalanced(const std::vector<Partition>& partitions, float threshold) const;
    void improvePartitions(std::vector<Partition>& partitions);
    void assignPartitionsToPEs(
        std::vector<Partition>& partitions,
        const NeuronMapping::HardwareTopology& topology);

    // Kernighan-Lin算法辅助
    float calculateSwapGain(
        size_t vertex1, size_t vertex2,
        const std::vector<Partition>& partitions);
    void performSwap(
        size_t vertex1, size_t vertex2,
        std::vector<Partition>& partitions);

    // 谱分割辅助
    std::vector<std::vector<float>> computeLaplacianMatrix();
    std::vector<float> computeFiedlerVector(
        const std::vector<std::vector<float>>& laplacian);
    std::vector<Partition> bisectByFiedlerVector(
        const std::vector<float>& fiedler_vector);

    // 多级分割辅助
    struct CoarseGraph {
        std::vector<Vertex> vertices;
        std::vector<Edge> edges;
        std::vector<std::vector<size_t>> mapping; // 粗化到精细的映射
    };

    CoarseGraph coarsenGraph(int target_size);
    std::vector<Partition> refinePartitions(
        const std::vector<Partition>& coarse_partitions,
        const CoarseGraph& coarse_graph);

    // 社区检测辅助
    float calculateModularity(const std::vector<Partition>& partitions);
    std::vector<Partition> louvainAlgorithm();

    // 工具方法
    std::vector<size_t> getNeighbors(size_t vertex_idx) const;
    float getEdgeWeight(size_t vertex1, size_t vertex2) const;
    void normalizePartitionWeights(std::vector<Partition>& partitions);
    void printPartitionStatistics(const std::vector<Partition>& partitions) const;
};

} // namespace neuron_mapping

#endif // NEURON_MAPPING_FRAMEWORK_STRATEGIES_GRAPH_PARTITIONING_STRATEGY_H