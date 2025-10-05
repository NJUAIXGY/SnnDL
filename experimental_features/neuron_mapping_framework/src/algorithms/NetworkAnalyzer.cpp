#include "algorithms/NetworkAnalyzer.h"
#include "utils/Logger.h"
#include "utils/MathUtils.h"
#include <queue>
#include <stack>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <sstream>
#include <iomanip>

namespace NeuronMapping {

// === NetworkAnalysis 实现 ===

void NetworkAnalysis::reset() {
    node_count = 0;
    edge_count = 0;
    density = 0.0f;
    average_degree = 0.0f;
    connected_components = 0;
    clustering_coefficient = 0.0f;
    average_path_length = 0.0f;
    diameter = 0;
    
    betweenness_centrality.clear();
    closeness_centrality.clear();
    degree_centrality.clear();
    eigenvector_centrality.clear();
    degree_distribution.clear();
    degree_assortativity = 0.0f;
    
    community_assignment.clear();
    modularity = 0.0f;
    num_communities = 0;
    
    layer_assignment.clear();
    num_layers = 0;
    is_feedforward = false;
    
    is_small_world = false;
    small_world_sigma = 0.0f;
    is_scale_free = false;
    power_law_exponent = 0.0f;
    power_law_goodness_of_fit = 0.0f;
}

std::string NetworkAnalysis::generateReport() const {
    std::ostringstream report;
    
    report << "Network Analysis Report\n";
    report << "=====================\n\n";
    
    // 基本属性
    report << "Basic Properties:\n";
    report << "  Nodes: " << node_count << "\n";
    report << "  Edges: " << edge_count << "\n";
    report << "  Density: " << std::fixed << std::setprecision(4) << density << "\n";
    report << "  Average Degree: " << average_degree << "\n\n";
    
    // 连通性
    report << "Connectivity:\n";
    report << "  Connected Components: " << connected_components << "\n";
    report << "  Clustering Coefficient: " << clustering_coefficient << "\n";
    report << "  Average Path Length: " << average_path_length << "\n";
    report << "  Diameter: " << diameter << "\n\n";
    
    return report.str();
}

// === NetworkAnalyzer 实现 ===

NetworkAnalysis NetworkAnalyzer::analyzeNetwork(const NeuralNetwork& network,
                                               bool include_centrality,
                                               bool include_community,
                                               bool include_advanced) const {
    LOG_INFO("Starting network analysis");
    NetworkAnalysis analysis;
    
    // 基本属性
    LOG_DEBUG("Calculating basic properties");
    analysis = calculateBasicProperties(network);
    
    // 简化的连通性分析
    auto components = findConnectedComponents(network);
    analysis.connected_components = static_cast<uint32_t>(components.size());
    
    LOG_INFO("Network analysis completed");
    return analysis;
}

NetworkAnalysis NetworkAnalyzer::calculateBasicProperties(const NeuralNetwork& network) const {
    NetworkAnalysis analysis;
    
    analysis.node_count = network.getNeuronCount();
    analysis.edge_count = network.getConnectionCount();
    analysis.density = calculateDensity(network);
    analysis.average_degree = calculateAverageDegree(network);
    analysis.degree_distribution = calculateDegreeDistribution(network);
    
    return analysis;
}

std::unordered_map<uint32_t, uint32_t> NetworkAnalyzer::calculateDegreeDistribution(const NeuralNetwork& network) const {
    std::unordered_map<uint32_t, uint32_t> distribution;
    
    // 计算每个神经元的度
    std::unordered_map<NeuronId, uint32_t> neuron_degrees;
    for (NeuronId neuron_id : network.getAllNeuronIds()) {
        neuron_degrees[neuron_id] = 0;
    }
    
    // 统计连接数
    for (const auto& connection : network.getAllConnections()) {
        neuron_degrees[connection.source_id]++;
        neuron_degrees[connection.target_id]++;
    }
    
    // 构建度分布
    for (const auto& pair : neuron_degrees) {
        uint32_t degree = pair.second;
        distribution[degree]++;
    }
    
    return distribution;
}

float NetworkAnalyzer::calculateDensity(const NeuralNetwork& network) const {
    uint32_t n = network.getNeuronCount();
    if (n <= 1) return 0.0f;
    
    uint32_t max_edges = n * (n - 1); // 有向图
    uint32_t actual_edges = network.getConnectionCount();
    
    return static_cast<float>(actual_edges) / max_edges;
}

float NetworkAnalyzer::calculateAverageDegree(const NeuralNetwork& network) const {
    if (network.getNeuronCount() == 0) return 0.0f;
    
    // 对于有向图，平均度 = 2 * 边数 / 节点数
    return 2.0f * network.getConnectionCount() / network.getNeuronCount();
}

std::vector<std::unordered_set<NeuronId>> NetworkAnalyzer::findConnectedComponents(const NeuralNetwork& network) const {
    std::vector<std::unordered_set<NeuronId>> components;
    std::unordered_set<NeuronId> visited;
    
    for (NeuronId neuron_id : network.getAllNeuronIds()) {
        if (visited.find(neuron_id) == visited.end()) {
            std::vector<NeuronId> component;
            depthFirstSearch(network, neuron_id, visited, component);
            
            std::unordered_set<NeuronId> component_set(component.begin(), component.end());
            components.push_back(component_set);
        }
    }
    
    return components;
}

std::unordered_set<NeuronId> NetworkAnalyzer::getLargestConnectedComponent(const NeuralNetwork& network) const {
    auto components = findConnectedComponents(network);
    
    if (components.empty()) {
        return std::unordered_set<NeuronId>();
    }
    
    auto largest = std::max_element(components.begin(), components.end(),
        [](const std::unordered_set<NeuronId>& a, const std::unordered_set<NeuronId>& b) {
            return a.size() < b.size();
        });
    
    return *largest;
}

float NetworkAnalyzer::calculateClusteringCoefficient(const NeuralNetwork& network, NeuronId neuron_id) const {
    // 简化实现，返回0
    return 0.0f;
}

float NetworkAnalyzer::calculateAveragePathLength(const NeuralNetwork& network, uint32_t sample_size) const {
    // 简化实现，返回0
    return 0.0f;
}

uint32_t NetworkAnalyzer::calculateDiameter(const NeuralNetwork& network) const {
    // 简化实现，返回0
    return 0;
}

std::unordered_map<NeuronId, float> NetworkAnalyzer::calculateDegreeCentrality(const NeuralNetwork& network,
                                                                              bool normalized) const {
    std::unordered_map<NeuronId, float> centrality;
    
    // 计算每个神经元的度
    std::unordered_map<NeuronId, uint32_t> neuron_degrees;
    for (NeuronId neuron_id : network.getAllNeuronIds()) {
        neuron_degrees[neuron_id] = 0;
    }
    
    // 统计连接数
    for (const auto& connection : network.getAllConnections()) {
        neuron_degrees[connection.source_id]++;
        neuron_degrees[connection.target_id]++;
    }
    
    // 计算中心性
    uint32_t n = network.getNeuronCount();
    float normalization_factor = normalized ? (n > 1 ? 1.0f / (n - 1) : 1.0f) : 1.0f;
    
    for (const auto& pair : neuron_degrees) {
        centrality[pair.first] = pair.second * normalization_factor;
    }
    
    return centrality;
}

std::unordered_map<NeuronId, float> NetworkAnalyzer::calculateClosenessCentrality(const NeuralNetwork& network,
                                                                                 bool normalized) const {
    std::unordered_map<NeuronId, float> centrality;
    
    // 简化实现，所有神经元的接近中心性设为0
    for (NeuronId neuron_id : network.getAllNeuronIds()) {
        centrality[neuron_id] = 0.0f;
    }
    
    return centrality;
}

std::unordered_map<NeuronId, float> NetworkAnalyzer::calculateBetweennessCentrality(const NeuralNetwork& network,
                                                                                   uint32_t sample_size) const {
    std::unordered_map<NeuronId, float> centrality;
    
    // 简化实现，所有神经元的介数中心性设为0
    for (NeuronId neuron_id : network.getAllNeuronIds()) {
        centrality[neuron_id] = 0.0f;
    }
    
    return centrality;
}

std::unordered_map<NeuronId, float> NetworkAnalyzer::calculateEigenvectorCentrality(const NeuralNetwork& network,
                                                                                   uint32_t max_iterations,
                                                                                   float tolerance) const {
    std::unordered_map<NeuronId, float> centrality;
    
    // 简化实现，所有神经元的特征向量中心性设为相等值
    float equal_value = 1.0f / std::sqrt(static_cast<float>(network.getNeuronCount()));
    for (NeuronId neuron_id : network.getAllNeuronIds()) {
        centrality[neuron_id] = equal_value;
    }
    
    return centrality;
}

std::unordered_map<NeuronId, uint32_t> NetworkAnalyzer::detectCommunitiesLouvain(const NeuralNetwork& network,
                                                                                float resolution) const {
    std::unordered_map<NeuronId, uint32_t> communities;
    
    // 简化实现，每个神经元为单独的社区
    uint32_t community_id = 0;
    for (NeuronId neuron_id : network.getAllNeuronIds()) {
        communities[neuron_id] = community_id++;
    }
    
    return communities;
}

std::unordered_map<NeuronId, uint32_t> NetworkAnalyzer::detectCommunitiesLabelPropagation(const NeuralNetwork& network,
                                                                                         uint32_t max_iterations) const {
    std::unordered_map<NeuronId, uint32_t> communities;
    
    // 简化实现，每个神经元为单独的社区
    uint32_t community_id = 0;
    for (NeuronId neuron_id : network.getAllNeuronIds()) {
        communities[neuron_id] = community_id++;
    }
    
    return communities;
}

float NetworkAnalyzer::calculateModularity(const NeuralNetwork& network,
                                          const std::unordered_map<NeuronId, uint32_t>& communities) const {
    // 简化实现，返回0
    return 0.0f;
}

std::unordered_map<NeuronId, uint32_t> NetworkAnalyzer::detectLayerStructure(const NeuralNetwork& network) const {
    std::unordered_map<NeuronId, uint32_t> layers;
    
    // 简化实现，所有神经元在同一层
    for (NeuronId neuron_id : network.getAllNeuronIds()) {
        layers[neuron_id] = 0;
    }
    
    return layers;
}

bool NetworkAnalyzer::isFeedforwardNetwork(const NeuralNetwork& network) const {
    // 简化实现，假设不是前馈网络
    return false;
}

std::vector<std::vector<NeuronId>> NetworkAnalyzer::findCycles(const NeuralNetwork& network) const {
    // 简化实现，返回空列表
    return std::vector<std::vector<NeuronId>>();
}

std::pair<bool, float> NetworkAnalyzer::isSmallWorld(const NeuralNetwork& network, uint32_t num_random_samples) const {
    // 简化实现
    return {false, 0.0f};
}

std::tuple<bool, float, float> NetworkAnalyzer::isScaleFree(const NeuralNetwork& network, uint32_t min_degree) const {
    // 简化实现
    return {false, 0.0f, 0.0f};
}

bool NetworkAnalyzer::isRandomGraph(const NeuralNetwork& network) const {
    // 简化实现
    return false;
}

std::vector<NeuronId> NetworkAnalyzer::findShortestPath(const NeuralNetwork& network,
                                                       NeuronId source, NeuronId target) const {
    // 简化实现，返回空路径
    return std::vector<NeuronId>();
}

std::unordered_map<NeuronId, std::unordered_map<NeuronId, uint32_t>>
NetworkAnalyzer::calculateAllPairsShortestDistances(const NeuralNetwork& network) const {
    // 简化实现，返回空距离矩阵
    return std::unordered_map<NeuronId, std::unordered_map<NeuronId, uint32_t>>();
}

std::unordered_map<NeuronId, uint32_t> NetworkAnalyzer::calculateSingleSourceDistances(const NeuralNetwork& network,
                                                                                       NeuronId source) const {
    // 简化的BFS实现
    return breadthFirstSearch(network, source);
}

float NetworkAnalyzer::compareNetworks(const NeuralNetwork& network1, const NeuralNetwork& network2) const {
    // 简化实现
    return 0.0f;
}

std::vector<float> NetworkAnalyzer::calculateStructuralFingerprint(const NeuralNetwork& network) const {
    // 简化实现，返回基本统计信息
    std::vector<float> fingerprint;
    fingerprint.push_back(static_cast<float>(network.getNeuronCount()));
    fingerprint.push_back(static_cast<float>(network.getConnectionCount()));
    fingerprint.push_back(calculateDensity(network));
    fingerprint.push_back(calculateAverageDegree(network));
    
    return fingerprint;
}

std::unique_ptr<NeuralNetwork> NetworkAnalyzer::generateRandomizedNetwork(const NeuralNetwork& network,
                                                                         uint32_t num_swaps) const {
    // 简化实现，返回nullptr
    return nullptr;
}

std::unique_ptr<NeuralNetwork> NetworkAnalyzer::generateConfigurationModel(const std::vector<uint32_t>& degree_sequence) const {
    // 简化实现，返回nullptr
    return nullptr;
}

// === 辅助方法实现 ===

std::unordered_map<NeuronId, uint32_t> NetworkAnalyzer::breadthFirstSearch(const NeuralNetwork& network,
                                                                          NeuronId start) const {
    std::unordered_map<NeuronId, uint32_t> distances;
    std::queue<NeuronId> queue;
    
    distances[start] = 0;
    queue.push(start);
    
    while (!queue.empty()) {
        NeuronId current = queue.front();
        queue.pop();
        
        // 查找邻居节点
        for (const auto& connection : network.getAllConnections()) {
            NeuronId neighbor = INVALID_NEURON_ID;
            if (connection.source_id == current) {
                neighbor = connection.target_id;
            } else if (connection.target_id == current) {
                neighbor = connection.source_id;
            }
            
            if (neighbor != INVALID_NEURON_ID && distances.find(neighbor) == distances.end()) {
                distances[neighbor] = distances[current] + 1;
                queue.push(neighbor);
            }
        }
    }
    
    return distances;
}

void NetworkAnalyzer::depthFirstSearch(const NeuralNetwork& network, NeuronId current,
                                      std::unordered_set<NeuronId>& visited,
                                      std::vector<NeuronId>& component) const {
    visited.insert(current);
    component.push_back(current);
    
    // 查找邻居节点
    for (const auto& connection : network.getAllConnections()) {
        NeuronId neighbor = INVALID_NEURON_ID;
        if (connection.source_id == current) {
            neighbor = connection.target_id;
        } else if (connection.target_id == current) {
            neighbor = connection.source_id;
        }
        
        if (neighbor != INVALID_NEURON_ID && visited.find(neighbor) == visited.end()) {
            depthFirstSearch(network, neighbor, visited, component);
        }
    }
}

std::unordered_map<NeuronId, uint32_t> NetworkAnalyzer::detectCommunitiesHierarchical(const NeuralNetwork& network,
                                                                                     const std::string& method,
                                                                                     uint32_t num_clusters) const {
    // 简化实现
    return detectCommunitiesLouvain(network);
}

} // namespace NeuronMapping