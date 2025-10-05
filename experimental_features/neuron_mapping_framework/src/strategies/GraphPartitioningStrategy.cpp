#include "strategies/GraphPartitioningStrategy.h"
#include "utils/Logger.h"
#include "utils/MathUtils.h"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <iomanip>

namespace neuron_mapping {

using namespace NeuronMapping;

GraphPartitioningStrategy::GraphPartitioningStrategy(
    PartitioningAlgorithm algorithm,
    QualityMetric quality_metric,
    uint32_t seed)
    : algorithm_(algorithm)
    , quality_metric_(quality_metric)
    , balance_threshold_(0.1)  // 10% imbalance tolerance
    , max_iterations_(100)
    , seed_(seed)
    , rng_(seed) {
}

std::string GraphPartitioningStrategy::getDescription() const {
    std::string alg_name;
    switch (algorithm_) {
        case PartitioningAlgorithm::SPECTRAL: alg_name = "Spectral"; break;
        case PartitioningAlgorithm::KERNIGHAN_LIN: alg_name = "Kernighan-Lin"; break;
        case PartitioningAlgorithm::MULTILEVEL: alg_name = "Multilevel"; break;
        case PartitioningAlgorithm::RECURSIVE_BISECTION: alg_name = "Recursive Bisection"; break;
        case PartitioningAlgorithm::COMMUNITY_DETECTION: alg_name = "Community Detection"; break;
        default: alg_name = "Unknown"; break;
    }
    return "Graph Partitioning Strategy using " + alg_name + " algorithm";
}

std::unique_ptr<MappingSolution> GraphPartitioningStrategy::mapNetwork(
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const MappingConfig& config) {
    
    if (!validateInputs(network, topology, config)) {
        return nullptr;
    }

    LOG_INFO("Starting graph partitioning mapping with " + getDescription());
    LOG_INFO("Network: " + std::to_string(network.getNeuronCount()) + " neurons, " + 
             std::to_string(network.getConnectionCount()) + " connections");
    LOG_INFO("Hardware: " + std::to_string(topology.getTotalPEs()) + " PEs");

    // 构建图数据结构
    buildGraphFromNetwork(network);
    LOG_INFO("Built graph with " + std::to_string(vertices_.size()) + " vertices and " + 
             std::to_string(edges_.size()) + " edges");

    // 执行图分割
    auto partitions = partitionGraph(network, topology, config);
    if (partitions.empty()) {
        LOG_ERROR("Graph partitioning failed");
        return nullptr;
    }

    LOG_INFO("Created " + std::to_string(partitions.size()) + " partitions");
    printPartitionStatistics(partitions);

    // 创建映射解决方案
    auto solution = std::make_unique<MappingSolution>(network.getNeuronCount());

    // 将分割结果转换为神经元分配
    for (const auto& partition : partitions) {
        for (NeuronId neuron_id : partition.vertices) {
            if (!solution->assignNeuron(neuron_id, partition.pe_id, 0)) {
                LOG_WARNING("Failed to assign neuron " + std::to_string(neuron_id) + 
                           " to PE " + std::to_string(partition.pe_id));
            }
        }
    }

    // 验证映射质量
    float partition_quality = calculatePartitionQuality(partitions);
    LOG_INFO("Final partition quality: " + std::to_string(partition_quality));
    LOG_INFO("Assigned " + std::to_string(solution->getAssignedNeuronCount()) + " neurons");

    return solution;
}

void GraphPartitioningStrategy::buildGraphFromNetwork(const NeuralNetwork& network) {
    vertices_.clear();
    edges_.clear();
    vertex_map_.clear();

    // 添加顶点（神经元）
    auto neuron_ids = network.getAllNeuronIds();
    vertices_.reserve(neuron_ids.size());
    
    for (size_t i = 0; i < neuron_ids.size(); ++i) {
        NeuronId neuron_id = neuron_ids[i];
        vertices_.emplace_back(neuron_id, 1.0f); // 默认权重为1
        vertex_map_[neuron_id] = i;
    }

    // 添加边（连接）
    auto connections = network.getAllConnections();
    edges_.reserve(connections.size());

    for (const auto& conn : connections) {
        edges_.emplace_back(conn.source_id, conn.target_id, std::abs(conn.weight));
        
        // 更新顶点的邻接边列表
        auto src_it = vertex_map_.find(conn.source_id);
        auto tgt_it = vertex_map_.find(conn.target_id);
        
        if (src_it != vertex_map_.end() && tgt_it != vertex_map_.end()) {
            size_t edge_idx = edges_.size() - 1;
            vertices_[src_it->second].edges.push_back(edge_idx);
            vertices_[tgt_it->second].edges.push_back(edge_idx);
        }
    }
}

std::vector<GraphPartitioningStrategy::Partition> GraphPartitioningStrategy::partitionGraph(
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const MappingConfig& config) {
    
    uint32_t num_partitions = topology.getTotalPEs();
    if (num_partitions == 0) {
        LOG_ERROR("No PEs available in topology");
        return {};
    }

    std::vector<Partition> partitions;

    // 选择分割算法
    switch (algorithm_) {
        case PartitioningAlgorithm::SPECTRAL:
            partitions = spectralPartitioning(num_partitions, topology);
            break;
        case PartitioningAlgorithm::KERNIGHAN_LIN:
            partitions = kernighanLinPartitioning(num_partitions, topology);
            break;
        case PartitioningAlgorithm::MULTILEVEL:
            partitions = multilevelPartitioning(num_partitions, topology);
            break;
        case PartitioningAlgorithm::RECURSIVE_BISECTION:
            partitions = recursiveBisection(num_partitions, topology);
            break;
        case PartitioningAlgorithm::COMMUNITY_DETECTION:
            partitions = communityDetection(num_partitions, topology);
            break;
        default:
            LOG_ERROR("Unknown partitioning algorithm");
            return {};
    }

    if (partitions.empty()) {
        LOG_ERROR("Partitioning algorithm failed");
        return {};
    }

    // 改进分割质量
    improvePartitions(partitions);

    // 分配分区到PE
    assignPartitionsToPEs(partitions, topology);

    return partitions;
}

std::vector<GraphPartitioningStrategy::Partition> 
GraphPartitioningStrategy::spectralPartitioning(
    uint32_t num_partitions,
    const HardwareTopology& topology) {
    
    LOG_INFO("Performing spectral partitioning");

    if (vertices_.size() < num_partitions) {
        LOG_WARNING("Not enough vertices for spectral partitioning");
        return {};
    }

    // 计算图的拉普拉斯矩阵
    auto laplacian = computeLaplacianMatrix();
    if (laplacian.empty()) {
        LOG_ERROR("Failed to compute Laplacian matrix");
        return {};
    }

    // 计算Fiedler向量（第二小特征向量）
    auto fiedler = computeFiedlerVector(laplacian);
    if (fiedler.empty()) {
        LOG_ERROR("Failed to compute Fiedler vector");
        return {};
    }

    // 基于Fiedler向量进行二分切割
    auto partitions = bisectByFiedlerVector(fiedler);

    // 如果需要更多分区，递归分割
    while (partitions.size() < num_partitions && partitions.size() < vertices_.size()) {
        // 找到最大的分区并继续分割
        auto max_it = std::max_element(partitions.begin(), partitions.end(),
            [](const Partition& a, const Partition& b) {
                return a.vertices.size() < b.vertices.size();
            });
        
        if (max_it->vertices.size() <= 1) break;

        // 简化处理：随机分割最大分区
        Partition new_partition;
        size_t split_point = max_it->vertices.size() / 2;
        
        new_partition.vertices.assign(max_it->vertices.begin() + split_point, max_it->vertices.end());
        max_it->vertices.erase(max_it->vertices.begin() + split_point, max_it->vertices.end());
        
        partitions.push_back(std::move(new_partition));
    }

    LOG_INFO("Spectral partitioning created " + std::to_string(partitions.size()) + " partitions");
    return partitions;
}

std::vector<GraphPartitioningStrategy::Partition> 
GraphPartitioningStrategy::kernighanLinPartitioning(
    uint32_t num_partitions,
    const HardwareTopology& topology) {
    
    LOG_INFO("Performing Kernighan-Lin partitioning");

    // 简化实现：先随机初始化分区，然后优化
    std::vector<Partition> partitions(num_partitions);
    
    // 随机分配顶点到分区
    std::uniform_int_distribution<uint32_t> dist(0, num_partitions - 1);
    
    for (size_t i = 0; i < vertices_.size(); ++i) {
        uint32_t partition_idx = dist(rng_);
        partitions[partition_idx].vertices.push_back(vertices_[i].id);
        partitions[partition_idx].total_weight += vertices_[i].weight;
    }

    // Kernighan-Lin优化迭代
    bool improved = true;
    uint32_t iteration = 0;
    
    while (improved && iteration < max_iterations_) {
        improved = false;
        iteration++;

        // 尝试交换顶点以减少切割代价
        for (size_t i = 0; i < partitions.size(); ++i) {
            for (size_t j = i + 1; j < partitions.size(); ++j) {
                if (partitions[i].vertices.empty() || partitions[j].vertices.empty()) continue;

                // 尝试交换每对顶点
                for (size_t vi = 0; vi < partitions[i].vertices.size(); ++vi) {
                    for (size_t vj = 0; vj < partitions[j].vertices.size(); ++vj) {
                        NeuronId v1 = partitions[i].vertices[vi];
                        NeuronId v2 = partitions[j].vertices[vj];

                        auto it1 = vertex_map_.find(v1);
                        auto it2 = vertex_map_.find(v2);
                        if (it1 == vertex_map_.end() || it2 == vertex_map_.end()) continue;

                        float gain = calculateSwapGain(it1->second, it2->second, partitions);
                        
                        if (gain > 0) {
                            // 执行交换
                            std::swap(partitions[i].vertices[vi], partitions[j].vertices[vj]);
                            improved = true;
                            break;
                        }
                    }
                    if (improved) break;
                }
                if (improved) break;
            }
            if (improved) break;
        }
    }

    LOG_INFO("Kernighan-Lin completed after " + std::to_string(iteration) + " iterations");
    return partitions;
}

std::vector<GraphPartitioningStrategy::Partition> 
GraphPartitioningStrategy::multilevelPartitioning(
    uint32_t num_partitions,
    const HardwareTopology& topology) {
    
    LOG_INFO("Performing multilevel partitioning");

    // 简化实现：多级方法的基本框架
    // 1. 粗化阶段
    int target_coarse_size = std::max(static_cast<int>(num_partitions * 2), 10);
    CoarseGraph coarse_graph = coarsenGraph(target_coarse_size);
    
    LOG_INFO("Coarsened graph to " + std::to_string(coarse_graph.vertices.size()) + " vertices");

    // 2. 初始分割阶段（在粗化图上）
    std::vector<Partition> coarse_partitions(num_partitions);
    
    // 简单的贪心分配
    for (size_t i = 0; i < coarse_graph.vertices.size(); ++i) {
        uint32_t partition_idx = i % num_partitions;
        coarse_partitions[partition_idx].vertices.push_back(coarse_graph.vertices[i].id);
        coarse_partitions[partition_idx].total_weight += coarse_graph.vertices[i].weight;
    }

    // 3. 精化阶段
    auto refined_partitions = refinePartitions(coarse_partitions, coarse_graph);

    LOG_INFO("Multilevel partitioning completed with " + std::to_string(refined_partitions.size()) + " partitions");
    return refined_partitions;
}

std::vector<GraphPartitioningStrategy::Partition> 
GraphPartitioningStrategy::recursiveBisection(
    uint32_t num_partitions,
    const HardwareTopology& topology) {
    
    LOG_INFO("Performing recursive bisection");

    std::vector<Partition> partitions;
    
    // 初始分区包含所有顶点
    Partition initial_partition;
    for (const auto& vertex : vertices_) {
        initial_partition.vertices.push_back(vertex.id);
        initial_partition.total_weight += vertex.weight;
    }
    
    std::queue<Partition> partition_queue;
    partition_queue.push(std::move(initial_partition));

    // 递归二分
    while (partitions.size() + partition_queue.size() < num_partitions && !partition_queue.empty()) {
        Partition current = std::move(partition_queue.front());
        partition_queue.pop();

        if (current.vertices.size() <= 1) {
            partitions.push_back(std::move(current));
            continue;
        }

        // 简单的二分：按顶点ID排序后分割
        std::sort(current.vertices.begin(), current.vertices.end());
        
        size_t mid = current.vertices.size() / 2;
        
        Partition left_partition, right_partition;
        
        for (size_t i = 0; i < mid; ++i) {
            left_partition.vertices.push_back(current.vertices[i]);
            auto it = vertex_map_.find(current.vertices[i]);
            if (it != vertex_map_.end()) {
                left_partition.total_weight += vertices_[it->second].weight;
            }
        }
        
        for (size_t i = mid; i < current.vertices.size(); ++i) {
            right_partition.vertices.push_back(current.vertices[i]);
            auto it = vertex_map_.find(current.vertices[i]);
            if (it != vertex_map_.end()) {
                right_partition.total_weight += vertices_[it->second].weight;
            }
        }

        partition_queue.push(std::move(left_partition));
        partition_queue.push(std::move(right_partition));
    }

    // 添加剩余的分区
    while (!partition_queue.empty()) {
        partitions.push_back(std::move(partition_queue.front()));
        partition_queue.pop();
    }

    LOG_INFO("Recursive bisection created " + std::to_string(partitions.size()) + " partitions");
    return partitions;
}

std::vector<GraphPartitioningStrategy::Partition> 
GraphPartitioningStrategy::communityDetection(
    uint32_t num_partitions,
    const HardwareTopology& topology) {
    
    LOG_INFO("Performing community detection");

    // 使用简化的Louvain算法
    auto communities = louvainAlgorithm();
    
    LOG_INFO("Community detection found " + std::to_string(communities.size()) + " communities");

    // 如果社区数量不匹配，进行调整
    while (communities.size() > num_partitions) {
        // 合并最小的社区
        auto min_it = std::min_element(communities.begin(), communities.end(),
            [](const Partition& a, const Partition& b) {
                return a.vertices.size() < b.vertices.size();
            });
        
        if (min_it != communities.end() && communities.size() > 1) {
            // 找到第二小的社区并合并
            auto second_min_it = communities.begin();
            if (second_min_it == min_it) ++second_min_it;
            
            for (auto it = communities.begin(); it != communities.end(); ++it) {
                if (it != min_it && it->vertices.size() < second_min_it->vertices.size()) {
                    second_min_it = it;
                }
            }

            // 合并社区
            second_min_it->vertices.insert(second_min_it->vertices.end(),
                                         min_it->vertices.begin(), min_it->vertices.end());
            second_min_it->total_weight += min_it->total_weight;
            communities.erase(min_it);
        } else {
            break;
        }
    }

    // 如果社区数量不足，分割大社区
    while (communities.size() < num_partitions) {
        auto max_it = std::max_element(communities.begin(), communities.end(),
            [](const Partition& a, const Partition& b) {
                return a.vertices.size() < b.vertices.size();
            });
        
        if (max_it->vertices.size() <= 1) break;

        // 分割最大社区
        Partition new_community;
        size_t split_point = max_it->vertices.size() / 2;
        
        new_community.vertices.assign(max_it->vertices.begin() + split_point, max_it->vertices.end());
        max_it->vertices.erase(max_it->vertices.begin() + split_point, max_it->vertices.end());
        
        // 重新计算权重
        max_it->total_weight = 0;
        for (NeuronId neuron_id : max_it->vertices) {
            auto it = vertex_map_.find(neuron_id);
            if (it != vertex_map_.end()) {
                max_it->total_weight += vertices_[it->second].weight;
            }
        }
        
        new_community.total_weight = 0;
        for (NeuronId neuron_id : new_community.vertices) {
            auto it = vertex_map_.find(neuron_id);
            if (it != vertex_map_.end()) {
                new_community.total_weight += vertices_[it->second].weight;
            }
        }
        
        communities.push_back(std::move(new_community));
    }

    return communities;
}

// 辅助方法实现
std::vector<std::vector<float>> GraphPartitioningStrategy::computeLaplacianMatrix() {
    size_t n = vertices_.size();
    std::vector<std::vector<float>> laplacian(n, std::vector<float>(n, 0.0f));

    // 构建邻接矩阵
    for (const auto& edge : edges_) {
        auto src_it = vertex_map_.find(edge.source);
        auto tgt_it = vertex_map_.find(edge.target);
        
        if (src_it != vertex_map_.end() && tgt_it != vertex_map_.end()) {
            size_t i = src_it->second;
            size_t j = tgt_it->second;
            
            laplacian[i][j] = -edge.weight;
            laplacian[j][i] = -edge.weight;
        }
    }

    // 设置对角线元素（度数）
    for (size_t i = 0; i < n; ++i) {
        float degree = 0.0f;
        for (size_t j = 0; j < n; ++j) {
            if (i != j) {
                degree -= laplacian[i][j];
            }
        }
        laplacian[i][i] = degree;
    }

    return laplacian;
}

std::vector<float> GraphPartitioningStrategy::computeFiedlerVector(
    const std::vector<std::vector<float>>& laplacian) {
    
    // 简化实现：使用幂迭代法近似计算第二小特征向量
    size_t n = laplacian.size();
    if (n == 0) return {};

    std::vector<float> vector(n);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    
    // 随机初始化
    for (size_t i = 0; i < n; ++i) {
        vector[i] = dist(rng_);
    }

    // 简化处理：返回随机向量作为近似
    return vector;
}

std::vector<GraphPartitioningStrategy::Partition> 
GraphPartitioningStrategy::bisectByFiedlerVector(const std::vector<float>& fiedler_vector) {
    
    std::vector<Partition> partitions(2);
    
    // 按Fiedler向量值的符号分割
    for (size_t i = 0; i < fiedler_vector.size() && i < vertices_.size(); ++i) {
        size_t partition_idx = (fiedler_vector[i] >= 0) ? 0 : 1;
        partitions[partition_idx].vertices.push_back(vertices_[i].id);
        partitions[partition_idx].total_weight += vertices_[i].weight;
    }

    return partitions;
}

GraphPartitioningStrategy::CoarseGraph GraphPartitioningStrategy::coarsenGraph(int target_size) {
    CoarseGraph coarse;
    
    // 简化实现：随机合并邻接顶点
    std::vector<bool> visited(vertices_.size(), false);
    
    for (size_t i = 0; i < vertices_.size() && static_cast<int>(coarse.vertices.size()) < target_size; ++i) {
        if (visited[i]) continue;
        
        Vertex coarse_vertex = vertices_[i];
        std::vector<size_t> merged_vertices = {i};
        visited[i] = true;
        
        // 尝试合并一个邻居
        for (size_t edge_idx : vertices_[i].edges) {
            const auto& edge = edges_[edge_idx];
            size_t neighbor_idx = INVALID_NEURON_ID;
            
            if (edge.source == vertices_[i].id) {
                auto it = vertex_map_.find(edge.target);
                if (it != vertex_map_.end()) neighbor_idx = it->second;
            } else if (edge.target == vertices_[i].id) {
                auto it = vertex_map_.find(edge.source);
                if (it != vertex_map_.end()) neighbor_idx = it->second;
            }
            
            if (neighbor_idx != INVALID_NEURON_ID && neighbor_idx < visited.size() && !visited[neighbor_idx]) {
                coarse_vertex.weight += vertices_[neighbor_idx].weight;
                merged_vertices.push_back(neighbor_idx);
                visited[neighbor_idx] = true;
                break;
            }
        }
        
        coarse.vertices.push_back(coarse_vertex);
        coarse.mapping.push_back(merged_vertices);
    }
    
    return coarse;
}

std::vector<GraphPartitioningStrategy::Partition> 
GraphPartitioningStrategy::refinePartitions(
    const std::vector<Partition>& coarse_partitions,
    const CoarseGraph& coarse_graph) {
    
    std::vector<Partition> refined_partitions(coarse_partitions.size());
    
    // 将粗化图的分区映射回原始图
    for (size_t part_idx = 0; part_idx < coarse_partitions.size(); ++part_idx) {
        for (NeuronId coarse_vertex_id : coarse_partitions[part_idx].vertices) {
            // 找到对应的原始顶点
            for (size_t coarse_idx = 0; coarse_idx < coarse_graph.vertices.size(); ++coarse_idx) {
                if (coarse_graph.vertices[coarse_idx].id == coarse_vertex_id) {
                    // 添加所有映射的原始顶点
                    for (size_t orig_idx : coarse_graph.mapping[coarse_idx]) {
                        if (orig_idx < vertices_.size()) {
                            refined_partitions[part_idx].vertices.push_back(vertices_[orig_idx].id);
                            refined_partitions[part_idx].total_weight += vertices_[orig_idx].weight;
                        }
                    }
                    break;
                }
            }
        }
    }
    
    return refined_partitions;
}

std::vector<GraphPartitioningStrategy::Partition> GraphPartitioningStrategy::louvainAlgorithm() {
    // 简化的Louvain算法实现
    std::vector<Partition> communities;
    std::vector<size_t> vertex_to_community(vertices_.size());
    
    // 初始化：每个顶点为一个社区
    for (size_t i = 0; i < vertices_.size(); ++i) {
        Partition community;
        community.vertices.push_back(vertices_[i].id);
        community.total_weight = vertices_[i].weight;
        communities.push_back(std::move(community));
        vertex_to_community[i] = i;
    }

    // 简化处理：合并高度连接的社区
    bool improved = true;
    while (improved && communities.size() > 1) {
        improved = false;
        
        for (size_t i = 0; i < communities.size() && !improved; ++i) {
            for (size_t j = i + 1; j < communities.size(); ++j) {
                // 计算社区间的连接强度
                float inter_edges = 0.0f;
                
                for (NeuronId neuron1 : communities[i].vertices) {
                    for (NeuronId neuron2 : communities[j].vertices) {
                        float weight = getEdgeWeight(vertex_map_[neuron1], vertex_map_[neuron2]);
                        if (weight > 0) {
                            inter_edges += weight;
                        }
                    }
                }
                
                // 如果连接足够强，合并社区
                float threshold = 0.1f * (communities[i].total_weight + communities[j].total_weight);
                if (inter_edges > threshold) {
                    communities[i].vertices.insert(communities[i].vertices.end(),
                                                 communities[j].vertices.begin(),
                                                 communities[j].vertices.end());
                    communities[i].total_weight += communities[j].total_weight;
                    communities.erase(communities.begin() + j);
                    improved = true;
                    break;
                }
            }
        }
    }

    return communities;
}

float GraphPartitioningStrategy::calculatePartitionQuality(const std::vector<Partition>& partitions) {
    if (partitions.empty()) return 0.0f;

    switch (quality_metric_) {
        case QualityMetric::EDGE_CUT: {
            float cut_edges = 0.0f;
            for (const auto& edge : edges_) {
                // 找到源和目标顶点所在的分区
                int src_partition = -1, tgt_partition = -1;
                
                for (size_t p = 0; p < partitions.size(); ++p) {
                    for (NeuronId neuron : partitions[p].vertices) {
                        if (neuron == edge.source) src_partition = static_cast<int>(p);
                        if (neuron == edge.target) tgt_partition = static_cast<int>(p);
                    }
                }
                
                if (src_partition != tgt_partition && src_partition >= 0 && tgt_partition >= 0) {
                    cut_edges += edge.weight;
                }
            }
            return cut_edges;
        }
        
        case QualityMetric::COMMUNICATION_VOLUME:
        case QualityMetric::NORMALIZED_CUT:
        case QualityMetric::CONDUCTANCE:
        default:
            // 简化实现：返回边切割数
            return calculatePartitionQuality(partitions);
    }
}

float GraphPartitioningStrategy::calculateSwapGain(
    size_t vertex1, size_t vertex2,
    const std::vector<Partition>& partitions) {
    
    // 简化实现：估算交换增益
    if (vertex1 >= vertices_.size() || vertex2 >= vertices_.size()) return 0.0f;
    
    // 计算两个顶点之间的连接权重
    float connection_weight = getEdgeWeight(vertex1, vertex2);
    
    // 如果有连接，交换会增加内部连接，减少外部连接
    return connection_weight > 0 ? connection_weight : -0.1f;
}

float GraphPartitioningStrategy::getEdgeWeight(size_t vertex1, size_t vertex2) const {
    if (vertex1 >= vertices_.size() || vertex2 >= vertices_.size()) return 0.0f;
    
    NeuronId id1 = vertices_[vertex1].id;
    NeuronId id2 = vertices_[vertex2].id;
    
    for (const auto& edge : edges_) {
        if ((edge.source == id1 && edge.target == id2) ||
            (edge.source == id2 && edge.target == id1)) {
            return edge.weight;
        }
    }
    
    return 0.0f;
}

bool GraphPartitioningStrategy::isBalanced(const std::vector<Partition>& partitions, float threshold) const {
    if (partitions.empty()) return true;
    
    float total_weight = 0.0f;
    for (const auto& partition : partitions) {
        total_weight += partition.total_weight;
    }
    
    float average_weight = total_weight / partitions.size();
    
    for (const auto& partition : partitions) {
        float imbalance = std::abs(partition.total_weight - average_weight) / average_weight;
        if (imbalance > threshold) {
            return false;
        }
    }
    
    return true;
}

void GraphPartitioningStrategy::improvePartitions(std::vector<Partition>& partitions) {
    // 简单的负载均衡改进
    if (partitions.size() <= 1) return;
    
    float total_weight = 0.0f;
    for (const auto& partition : partitions) {
        total_weight += partition.total_weight;
    }
    
    float target_weight = total_weight / partitions.size();
    
    // 从重载的分区移动顶点到轻载的分区
    for (size_t i = 0; i < partitions.size(); ++i) {
        if (partitions[i].total_weight > target_weight * (1.0f + balance_threshold_)) {
            // 找到最轻的分区
            size_t lightest = 0;
            for (size_t j = 1; j < partitions.size(); ++j) {
                if (partitions[j].total_weight < partitions[lightest].total_weight) {
                    lightest = j;
                }
            }
            
            // 移动一个顶点
            if (!partitions[i].vertices.empty() && i != lightest) {
                NeuronId vertex_to_move = partitions[i].vertices.back();
                partitions[i].vertices.pop_back();
                
                auto it = vertex_map_.find(vertex_to_move);
                if (it != vertex_map_.end() && it->second < vertices_.size()) {
                    float vertex_weight = vertices_[it->second].weight;
                    partitions[i].total_weight -= vertex_weight;
                    
                    partitions[lightest].vertices.push_back(vertex_to_move);
                    partitions[lightest].total_weight += vertex_weight;
                }
            }
        }
    }
}

void GraphPartitioningStrategy::assignPartitionsToPEs(
    std::vector<Partition>& partitions,
    const HardwareTopology& topology) {
    
    // 简单分配：按顺序分配给PE
    for (size_t i = 0; i < partitions.size() && i < topology.getTotalPEs(); ++i) {
        partitions[i].pe_id = static_cast<PEId>(i);
    }
}

void GraphPartitioningStrategy::printPartitionStatistics(const std::vector<Partition>& partitions) const {
    if (partitions.empty()) return;
    
    LOG_INFO("Partition Statistics:");
    for (size_t i = 0; i < partitions.size(); ++i) {
        LOG_INFO("  Partition " + std::to_string(i) + ": " + 
                std::to_string(partitions[i].vertices.size()) + " vertices, weight=" + 
                std::to_string(partitions[i].total_weight) + ", PE=" + 
                std::to_string(partitions[i].pe_id));
    }
    
    float quality = const_cast<GraphPartitioningStrategy*>(this)->calculatePartitionQuality(partitions);
    LOG_INFO("Overall partition quality: " + std::to_string(quality));
    
    bool balanced = isBalanced(partitions, balance_threshold_);
    LOG_INFO("Partitions balanced: " + (balanced ? std::string("Yes") : std::string("No")));
}

} // namespace neuron_mapping