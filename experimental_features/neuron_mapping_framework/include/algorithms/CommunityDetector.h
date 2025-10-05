#ifndef NEURON_MAPPING_COMMUNITY_DETECTOR_H
#define NEURON_MAPPING_COMMUNITY_DETECTOR_H

#include "core/Types.h"
#include "core/NeuralNetwork.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <random>

namespace neuron_mapping {

using namespace NeuronMapping;

/**
 * @brief 社区结构表示
 */
struct Community {
    std::vector<NeuronId> neurons;     // 社区内的神经元
    float internal_weight;             // 社区内部连接权重
    float external_weight;             // 社区外部连接权重
    float modularity_contribution;    // 对模块度的贡献
    uint32_t community_id;            // 社区ID
    
    Community() : internal_weight(0.0f), external_weight(0.0f), 
                  modularity_contribution(0.0f), community_id(0) {}
    
    float getDensity() const {
        if (neurons.size() <= 1) return 0.0f;
        float max_possible_edges = neurons.size() * (neurons.size() - 1) / 2.0f;
        return max_possible_edges > 0 ? internal_weight / max_possible_edges : 0.0f;
    }
    
    float getConductance() const {
        float total_weight = internal_weight + external_weight;
        return total_weight > 0 ? external_weight / total_weight : 0.0f;
    }
};

/**
 * @brief 社区检测结果
 */
struct CommunityStructure {
    std::vector<Community> communities;              // 检测到的社区
    std::unordered_map<NeuronId, uint32_t> neuron_to_community;  // 神经元到社区的映射
    float modularity;                               // 整体模块度
    uint32_t num_communities;                       // 社区数量
    std::vector<std::vector<float>> community_adjacency;  // 社区间邻接矩阵
    
    CommunityStructure() : modularity(0.0f), num_communities(0) {}
    
    /**
     * @brief 获取神经元所属的社区ID
     */
    uint32_t getCommunityId(NeuronId neuron_id) const {
        auto it = neuron_to_community.find(neuron_id);
        return it != neuron_to_community.end() ? it->second : INVALID_COMMUNITY_ID;
    }
    
    /**
     * @brief 获取社区统计信息
     */
    std::vector<float> getCommunityDensities() const {
        std::vector<float> densities;
        for (const auto& community : communities) {
            densities.push_back(community.getDensity());
        }
        return densities;
    }
    
    static const uint32_t INVALID_COMMUNITY_ID = UINT32_MAX;
};

/**
 * @brief 社区检测算法类
 * 
 * 实现多种社区检测算法，用于分析神经网络的社区结构。
 */
class CommunityDetector {
public:
    /**
     * @brief 社区检测算法类型
     */
    enum class Algorithm {
        LOUVAIN,           // Louvain算法
        LEIDEN,            // Leiden算法
        LABEL_PROPAGATION, // 标签传播算法
        MODULARITY_MAXIMIZATION,  // 模块度最大化
        SPECTRAL_CLUSTERING,      // 谱聚类
        INFOMAP                   // Infomap算法
    };
    
    /**
     * @brief 社区质量度量类型
     */
    enum class QualityMetric {
        MODULARITY,        // 模块度
        CONDUCTANCE,       // 电导率
        COVERAGE,          // 覆盖率
        SILHOUETTE,        // 轮廓系数
        NORMALIZED_CUT     // 归一化切割
    };

    /**
     * @brief 构造函数
     * @param algorithm 检测算法
     * @param seed 随机种子
     */
    explicit CommunityDetector(Algorithm algorithm = Algorithm::LOUVAIN, uint32_t seed = 42);
    
    virtual ~CommunityDetector() = default;

    /**
     * @brief 检测网络中的社区结构
     * @param network 神经网络
     * @return 社区结构
     */
    std::unique_ptr<CommunityStructure> detectCommunities(const NeuralNetwork& network);

    /**
     * @brief 评估社区结构质量
     * @param structure 社区结构
     * @param network 神经网络
     * @param metric 质量度量
     * @return 质量分数
     */
    float evaluateCommunityQuality(const CommunityStructure& structure,
                                  const NeuralNetwork& network,
                                  QualityMetric metric = QualityMetric::MODULARITY);

    /**
     * @brief 合并相似的社区
     * @param structure 社区结构
     * @param network 神经网络
     * @param threshold 合并阈值
     * @return 合并后的社区结构
     */
    std::unique_ptr<CommunityStructure> mergeSimilarCommunities(
        const CommunityStructure& structure,
        const NeuralNetwork& network,
        float threshold = 0.1f);

    /**
     * @brief 分割过大的社区
     * @param structure 社区结构
     * @param network 神经网络
     * @param max_community_size 最大社区大小
     * @return 分割后的社区结构
     */
    std::unique_ptr<CommunityStructure> splitLargeCommunities(
        const CommunityStructure& structure,
        const NeuralNetwork& network,
        uint32_t max_community_size);

    // 参数设置
    void setAlgorithm(Algorithm algorithm) { algorithm_ = algorithm; }
    void setResolution(float resolution) { resolution_ = resolution; }
    void setMaxIterations(uint32_t max_iterations) { max_iterations_ = max_iterations; }
    void setConvergenceThreshold(float threshold) { convergence_threshold_ = threshold; }

    // 获取算法信息
    Algorithm getAlgorithm() const { return algorithm_; }
    std::string getAlgorithmName() const;
    std::string getDescription() const;

private:
    Algorithm algorithm_;
    float resolution_;          // 分辨率参数
    uint32_t max_iterations_;   // 最大迭代次数
    float convergence_threshold_; // 收敛阈值
    uint32_t seed_;             // 随机种子
    std::mt19937 rng_;          // 随机数生成器

    // 图数据结构
    struct GraphEdge {
        NeuronId source;
        NeuronId target;
        float weight;
        
        GraphEdge(NeuronId s, NeuronId t, float w) : source(s), target(t), weight(w) {}
    };

    std::vector<GraphEdge> edges_;
    std::unordered_map<NeuronId, std::vector<size_t>> adjacency_list_;
    float total_edge_weight_;

    // 核心算法实现
    std::unique_ptr<CommunityStructure> louvainAlgorithm(const NeuralNetwork& network);
    std::unique_ptr<CommunityStructure> leidenAlgorithm(const NeuralNetwork& network);
    std::unique_ptr<CommunityStructure> labelPropagationAlgorithm(const NeuralNetwork& network);
    std::unique_ptr<CommunityStructure> modularityMaximization(const NeuralNetwork& network);
    std::unique_ptr<CommunityStructure> spectralClustering(const NeuralNetwork& network);
    std::unique_ptr<CommunityStructure> infomapAlgorithm(const NeuralNetwork& network);

    // Louvain算法辅助方法
    float calculateModularityGain(NeuronId neuron, uint32_t target_community,
                                 const std::vector<uint32_t>& node_communities,
                                 const std::vector<float>& community_weights,
                                 const NeuralNetwork& network);
    
    void moveNeuronToCommunity(NeuronId neuron, uint32_t new_community,
                              std::vector<uint32_t>& node_communities,
                              std::vector<float>& community_weights);

    // 标签传播算法辅助方法
    std::vector<uint32_t> initializeLabels(const std::vector<NeuronId>& neurons);
    void propagateLabels(std::vector<uint32_t>& labels, const NeuralNetwork& network);
    uint32_t getMostFrequentLabel(NeuronId neuron, const std::vector<uint32_t>& labels,
                                 const NeuralNetwork& network);

    // 谱聚类辅助方法
    std::vector<std::vector<float>> computeNormalizedLaplacian(const NeuralNetwork& network);
    std::vector<std::vector<float>> computeEigenvectors(const std::vector<std::vector<float>>& matrix,
                                                       uint32_t num_clusters);
    std::vector<uint32_t> kMeansClustering(const std::vector<std::vector<float>>& features,
                                          uint32_t num_clusters);

    // 工具方法
    void buildGraphStructure(const NeuralNetwork& network);
    float calculateModularity(const std::vector<uint32_t>& communities, const NeuralNetwork& network);
    float calculateConductance(const Community& community, const NeuralNetwork& network);
    float calculateCoverage(const CommunityStructure& structure, const NeuralNetwork& network);
    float calculateSilhouetteScore(const CommunityStructure& structure, const NeuralNetwork& network);
    
    std::unique_ptr<CommunityStructure> convertToCommunityStructure(
        const std::vector<uint32_t>& node_communities,
        const NeuralNetwork& network);
    
    void updateCommunityStatistics(CommunityStructure& structure, const NeuralNetwork& network);
    
    std::vector<NeuronId> getNeighbors(NeuronId neuron, const NeuralNetwork& network);
    float getEdgeWeight(NeuronId source, NeuronId target, const NeuralNetwork& network);
    
    // 调试和统计
    void printCommunityStatistics(const CommunityStructure& structure) const;
    void logAlgorithmProgress(uint32_t iteration, float current_modularity) const;
};

} // namespace neuron_mapping

#endif // NEURON_MAPPING_COMMUNITY_DETECTOR_H