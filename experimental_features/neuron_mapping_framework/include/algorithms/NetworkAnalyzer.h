#ifndef NEURON_MAPPING_NETWORK_ANALYZER_H
#define NEURON_MAPPING_NETWORK_ANALYZER_H

#include "core/Types.h"
#include "core/NeuralNetwork.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <random>
#include <iomanip>

namespace NeuronMapping {

/**
 * @brief 网络分析结果结构
 */
struct NetworkAnalysis {
    // 基本属性
    uint32_t node_count = 0;
    uint32_t edge_count = 0;
    float density = 0.0f;
    float average_degree = 0.0f;
    
    // 连通性
    uint32_t connected_components = 0;
    float clustering_coefficient = 0.0f;
    float average_path_length = 0.0f;
    uint32_t diameter = 0;
    
    // 中心性度量
    std::unordered_map<NeuronId, float> betweenness_centrality;
    std::unordered_map<NeuronId, float> closeness_centrality;
    std::unordered_map<NeuronId, float> degree_centrality;
    std::unordered_map<NeuronId, float> eigenvector_centrality;
    
    // 度分布
    std::unordered_map<uint32_t, uint32_t> degree_distribution;
    float degree_assortativity = 0.0f;
    
    // 模块性和社区结构
    std::unordered_map<NeuronId, uint32_t> community_assignment;
    float modularity = 0.0f;
    uint32_t num_communities = 0;
    
    // 层次结构
    std::unordered_map<NeuronId, uint32_t> layer_assignment;
    uint32_t num_layers = 0;
    bool is_feedforward = false;
    
    // 小世界属性
    bool is_small_world = false;
    float small_world_sigma = 0.0f;
    
    // 无标度属性
    bool is_scale_free = false;
    float power_law_exponent = 0.0f;
    float power_law_goodness_of_fit = 0.0f;
    
    // 重置所有值
    void reset();
    
    // 生成分析报告
    std::string generateReport() const;
};

/**
 * @brief 神经网络分析器类
 * 
 * 提供神经网络拓扑结构的分析功能，包括连通性、中心性、社区结构等。
 */
class NetworkAnalyzer {
public:
    NetworkAnalyzer() = default;
    virtual ~NetworkAnalyzer() = default;
    
    // === 主分析接口 ===
    
    /**
     * @brief 执行完整的网络分析
     * @param network 神经网络
     * @param include_centrality 是否计算中心性度量
     * @param include_community 是否进行社区检测
     * @param include_advanced 是否进行高级分析
     * @return 网络分析结果
     */
    NetworkAnalysis analyzeNetwork(const NeuralNetwork& network,
                                  bool include_centrality = true,
                                  bool include_community = true,
                                  bool include_advanced = false) const;
    
    // === 基本属性分析 ===
    
    /**
     * @brief 计算网络基本属性
     * @param network 神经网络
     * @return 基本属性结果
     */
    NetworkAnalysis calculateBasicProperties(const NeuralNetwork& network) const;
    
    /**
     * @brief 计算度分布
     * @param network 神经网络
     * @return 度分布映射
     */
    std::unordered_map<uint32_t, uint32_t> calculateDegreeDistribution(const NeuralNetwork& network) const;
    
    /**
     * @brief 计算网络密度
     * @param network 神经网络
     * @return 网络密度 [0, 1]
     */
    float calculateDensity(const NeuralNetwork& network) const;
    
    /**
     * @brief 计算平均度
     * @param network 神经网络
     * @return 平均度
     */
    float calculateAverageDegree(const NeuralNetwork& network) const;
    
    /**
     * @brief 计算度同配性
     * @param network 神经网络
     * @return 度同配性系数
     */
    float calculateDegreeAssortativity(const NeuralNetwork& network) const;
    
    // === 连通性分析 ===
    
    /**
     * @brief 找到所有连通分量
     * @param network 神经网络
     * @return 连通分量列表，每个分量包含神经元ID集合
     */
    std::vector<std::unordered_set<NeuronId>> findConnectedComponents(const NeuralNetwork& network) const;
    
    /**
     * @brief 计算最大连通分量
     * @param network 神经网络
     * @return 最大连通分量的神经元ID集合
     */
    std::unordered_set<NeuronId> getLargestConnectedComponent(const NeuralNetwork& network) const;
    
    /**
     * @brief 计算聚集系数
     * @param network 神经网络
     * @param neuron_id 神经元ID（可选，计算全局聚集系数时忽略）
     * @return 聚集系数
     */
    float calculateClusteringCoefficient(const NeuralNetwork& network, 
                                       NeuronId neuron_id = INVALID_NEURON_ID) const;
    
    /**
     * @brief 计算平均路径长度
     * @param network 神经网络
     * @param sample_size 采样大小（0表示计算所有路径）
     * @return 平均路径长度
     */
    float calculateAveragePathLength(const NeuralNetwork& network, uint32_t sample_size = 1000) const;
    
    /**
     * @brief 计算网络直径
     * @param network 神经网络
     * @return 网络直径
     */
    uint32_t calculateDiameter(const NeuralNetwork& network) const;
    
    // === 中心性度量 ===
    
    /**
     * @brief 计算度中心性
     * @param network 神经网络
     * @param normalized 是否归一化
     * @return 度中心性映射
     */
    std::unordered_map<NeuronId, float> calculateDegreeCentrality(const NeuralNetwork& network,
                                                                 bool normalized = true) const;
    
    /**
     * @brief 计算接近中心性
     * @param network 神经网络
     * @param normalized 是否归一化
     * @return 接近中心性映射
     */
    std::unordered_map<NeuronId, float> calculateClosenessCentrality(const NeuralNetwork& network,
                                                                    bool normalized = true) const;
    
    /**
     * @brief 计算介数中心性
     * @param network 神经网络
     * @param sample_size 采样大小（0表示精确计算）
     * @return 介数中心性映射
     */
    std::unordered_map<NeuronId, float> calculateBetweennessCentrality(const NeuralNetwork& network,
                                                                      uint32_t sample_size = 0) const;
    
    /**
     * @brief 计算特征向量中心性
     * @param network 神经网络
     * @param max_iterations 最大迭代次数
     * @param tolerance 收敛容差
     * @return 特征向量中心性映射
     */
    std::unordered_map<NeuronId, float> calculateEigenvectorCentrality(const NeuralNetwork& network,
                                                                      uint32_t max_iterations = 100,
                                                                      float tolerance = 1e-6f) const;
    
    // === 社区检测 ===
    
    /**
     * @brief 使用Louvain算法进行社区检测
     * @param network 神经网络
     * @param resolution 分辨率参数
     * @return 社区分配映射
     */
    std::unordered_map<NeuronId, uint32_t> detectCommunitiesLouvain(const NeuralNetwork& network,
                                                                   float resolution = 1.0f) const;
    
    /**
     * @brief 使用标签传播算法进行社区检测
     * @param network 神经网络
     * @param max_iterations 最大迭代次数
     * @return 社区分配映射
     */
    std::unordered_map<NeuronId, uint32_t> detectCommunitiesLabelPropagation(const NeuralNetwork& network,
                                                                             uint32_t max_iterations = 100) const;
    
    /**
     * @brief 计算模块度
     * @param network 神经网络
     * @param communities 社区分配
     * @return 模块度值
     */
    float calculateModularity(const NeuralNetwork& network,
                             const std::unordered_map<NeuronId, uint32_t>& communities) const;
    
    /**
     * @brief 层次聚类社区检测
     * @param network 神经网络
     * @param method 聚类方法（"single", "complete", "average"）
     * @param num_clusters 目标簇数（0表示自动确定）
     * @return 社区分配映射
     */
    std::unordered_map<NeuronId, uint32_t> detectCommunitiesHierarchical(const NeuralNetwork& network,
                                                                         const std::string& method = "complete",
                                                                         uint32_t num_clusters = 0) const;
    
    // === 层次结构分析 ===
    
    /**
     * @brief 检测神经网络层次结构
     * @param network 神经网络
     * @return 层次分配映射
     */
    std::unordered_map<NeuronId, uint32_t> detectLayerStructure(const NeuralNetwork& network) const;
    
    /**
     * @brief 检查是否为前馈网络
     * @param network 神经网络
     * @return 是否为前馈网络
     */
    bool isFeedforwardNetwork(const NeuralNetwork& network) const;
    
    /**
     * @brief 找到网络中的循环
     * @param network 神经网络
     * @return 循环列表，每个循环包含神经元ID序列
     */
    std::vector<std::vector<NeuronId>> findCycles(const NeuralNetwork& network) const;
    
    // === 网络模型识别 ===
    
    /**
     * @brief 检查是否为小世界网络
     * @param network 神经网络
     * @param num_random_samples 随机网络样本数
     * @return 是否为小世界网络和sigma值
     */
    std::pair<bool, float> isSmallWorld(const NeuralNetwork& network, uint32_t num_random_samples = 10) const;
    
    /**
     * @brief 检查是否为无标度网络
     * @param network 神经网络
     * @param min_degree 最小度阈值
     * @return 是否为无标度网络、幂律指数和拟合优度
     */
    std::tuple<bool, float, float> isScaleFree(const NeuralNetwork& network, uint32_t min_degree = 1) const;
    
    /**
     * @brief 检查是否符合随机图模型
     * @param network 神经网络
     * @return 是否符合随机图模型
     */
    bool isRandomGraph(const NeuralNetwork& network) const;
    
    // === 路径和距离计算 ===
    
    /**
     * @brief 计算最短路径
     * @param network 神经网络
     * @param source 源神经元ID
     * @param target 目标神经元ID
     * @return 最短路径（神经元ID序列），空表示无路径
     */
    std::vector<NeuronId> findShortestPath(const NeuralNetwork& network,
                                          NeuronId source, NeuronId target) const;
    
    /**
     * @brief 计算所有对最短距离
     * @param network 神经网络
     * @return 距离矩阵
     */
    std::unordered_map<NeuronId, std::unordered_map<NeuronId, uint32_t>>
    calculateAllPairsShortestDistances(const NeuralNetwork& network) const;
    
    /**
     * @brief 计算从源点的单源最短距离
     * @param network 神经网络
     * @param source 源神经元ID
     * @return 距离映射
     */
    std::unordered_map<NeuronId, uint32_t> calculateSingleSourceDistances(const NeuralNetwork& network,
                                                                          NeuronId source) const;
    
    // === 网络比较 ===
    
    /**
     * @brief 比较两个网络的结构相似性
     * @param network1 网络1
     * @param network2 网络2
     * @return 相似性分数 [0, 1]
     */
    float compareNetworks(const NeuralNetwork& network1, const NeuralNetwork& network2) const;
    
    /**
     * @brief 计算网络的结构指纹
     * @param network 神经网络
     * @return 结构指纹向量
     */
    std::vector<float> calculateStructuralFingerprint(const NeuralNetwork& network) const;
    
    // === 网络生成和随机化 ===
    
    /**
     * @brief 生成度序列保持的随机化网络
     * @param network 原始网络
     * @param num_swaps 边交换次数
     * @return 随机化网络
     */
    std::unique_ptr<NeuralNetwork> generateRandomizedNetwork(const NeuralNetwork& network,
                                                            uint32_t num_swaps = 1000) const;
    
    /**
     * @brief 生成配置模型随机图
     * @param degree_sequence 度序列
     * @return 随机图
     */
    std::unique_ptr<NeuralNetwork> generateConfigurationModel(const std::vector<uint32_t>& degree_sequence) const;

private:
    // === 辅助方法 ===
    
    // BFS相关
    std::unordered_map<NeuronId, uint32_t> breadthFirstSearch(const NeuralNetwork& network,
                                                             NeuronId start) const;
    
    // DFS相关
    void depthFirstSearch(const NeuralNetwork& network, NeuronId current,
                         std::unordered_set<NeuronId>& visited,
                         std::vector<NeuronId>& component) const;
    
    // 拓扑排序
    std::vector<NeuronId> topologicalSort(const NeuralNetwork& network) const;
    
    // 强连通分量
    std::vector<std::vector<NeuronId>> findStronglyConnectedComponents(const NeuralNetwork& network) const;
    
    // 幂律拟合
    std::pair<float, float> fitPowerLaw(const std::vector<uint32_t>& data, uint32_t min_value) const;
    
    // 统计检验
    float kolmogorovSmirnovTest(const std::vector<float>& data1, const std::vector<float>& data2) const;
    
    // 网络指标计算辅助
    float calculateLocalClusteringCoefficient(const NeuralNetwork& network, NeuronId neuron_id) const;
    std::vector<NeuronId> getNeighbors(const NeuralNetwork& network, NeuronId neuron_id) const;
    uint32_t countTriangles(const NeuralNetwork& network, NeuronId neuron_id) const;
    
    // 社区检测辅助
    float calculateModularityGain(const NeuralNetwork& network,
                                 const std::unordered_map<NeuronId, uint32_t>& communities,
                                 NeuronId neuron, uint32_t old_community, uint32_t new_community) const;
    
    // 随机数生成
    mutable std::random_device rd_;
    mutable std::mt19937 rng_{rd_()};
};

} // namespace NeuronMapping

#endif // NEURON_MAPPING_NETWORK_ANALYZER_H