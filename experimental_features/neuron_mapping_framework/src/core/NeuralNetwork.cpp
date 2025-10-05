#include "core/NeuralNetwork.h"
#include <algorithm>
#include <queue>
#include <unordered_set>
#include <stack>
#include <numeric>
#include <cmath>
#include <random>

namespace NeuronMapping {

// === 神经元管理实现 ===

bool NeuralNetwork::addNeuron(const NeuronProperties& properties) {
    if (neurons_.find(properties.id) != neurons_.end()) {
        return false; // 神经元已存在
    }
    
    neurons_[properties.id] = properties;
    statistics_valid_ = false; // 无效化缓存的统计信息
    return true;
}

uint32_t NeuralNetwork::addNeurons(const std::vector<NeuronProperties>& neurons) {
    uint32_t added_count = 0;
    for (const auto& neuron : neurons) {
        if (addNeuron(neuron)) {
            added_count++;
        }
    }
    return added_count;
}

bool NeuralNetwork::removeNeuron(NeuronId neuron_id) {
    auto it = neurons_.find(neuron_id);
    if (it == neurons_.end()) {
        return false; // 神经元不存在
    }
    
    // 移除所有相关连接
    auto connections_to_remove = std::vector<size_t>();
    for (size_t i = 0; i < connections_.size(); ++i) {
        const auto& conn = connections_[i];
        if (conn.source_id == neuron_id || conn.target_id == neuron_id) {
            connections_to_remove.push_back(i);
        }
    }
    
    // 从后向前删除连接，避免索引变化
    for (auto it = connections_to_remove.rbegin(); it != connections_to_remove.rend(); ++it) {
        connections_.erase(connections_.begin() + *it);
    }
    
    // 移除神经元
    neurons_.erase(it);
    
    // 重建索引
    updateConnectionIndices();
    statistics_valid_ = false;
    
    return true;
}

bool NeuralNetwork::hasNeuron(NeuronId neuron_id) const {
    return neurons_.find(neuron_id) != neurons_.end();
}

const NeuronProperties* NeuralNetwork::getNeuron(NeuronId neuron_id) const {
    auto it = neurons_.find(neuron_id);
    return (it != neurons_.end()) ? &it->second : nullptr;
}

bool NeuralNetwork::updateNeuron(const NeuronProperties& properties) {
    auto it = neurons_.find(properties.id);
    if (it == neurons_.end()) {
        return false;
    }
    
    it->second = properties;
    statistics_valid_ = false;
    return true;
}

std::vector<NeuronId> NeuralNetwork::getAllNeuronIds() const {
    std::vector<NeuronId> ids;
    ids.reserve(neurons_.size());
    
    for (const auto& [id, properties] : neurons_) {
        ids.push_back(id);
    }
    
    return ids;
}

// === 连接管理实现 ===

bool NeuralNetwork::addConnection(const Connection& connection) {
    // 检查神经元是否存在
    if (!hasNeuron(connection.source_id) || !hasNeuron(connection.target_id)) {
        return false;
    }
    
    // 检查连接是否已存在
    uint64_t key = makeConnectionKey(connection.source_id, connection.target_id);
    if (connection_index_.find(key) != connection_index_.end()) {
        return false; // 连接已存在
    }
    
    // 添加连接
    size_t connection_idx = connections_.size();
    connections_.push_back(connection);
    
    // 更新索引
    addConnectionToIndices(connection_idx);
    statistics_valid_ = false;
    
    return true;
}

uint32_t NeuralNetwork::addConnections(const std::vector<Connection>& connections) {
    uint32_t added_count = 0;
    for (const auto& conn : connections) {
        if (addConnection(conn)) {
            added_count++;
        }
    }
    return added_count;
}

bool NeuralNetwork::removeConnection(NeuronId source_id, NeuronId target_id) {
    uint64_t key = makeConnectionKey(source_id, target_id);
    auto it = connection_index_.find(key);
    
    if (it == connection_index_.end()) {
        return false; // 连接不存在
    }
    
    size_t connection_idx = it->second;
    
    // 从索引中移除
    removeConnectionFromIndices(connection_idx);
    
    // 从连接数组中移除
    connections_.erase(connections_.begin() + connection_idx);
    
    // 重建索引（因为索引可能发生变化）
    updateConnectionIndices();
    statistics_valid_ = false;
    
    return true;
}

bool NeuralNetwork::hasConnection(NeuronId source_id, NeuronId target_id) const {
    uint64_t key = makeConnectionKey(source_id, target_id);
    return connection_index_.find(key) != connection_index_.end();
}

const Connection* NeuralNetwork::getConnection(NeuronId source_id, NeuronId target_id) const {
    uint64_t key = makeConnectionKey(source_id, target_id);
    auto it = connection_index_.find(key);
    
    if (it == connection_index_.end()) {
        return nullptr;
    }
    
    return &connections_[it->second];
}

bool NeuralNetwork::updateConnectionWeight(NeuronId source_id, NeuronId target_id, Weight new_weight) {
    uint64_t key = makeConnectionKey(source_id, target_id);
    auto it = connection_index_.find(key);
    
    if (it == connection_index_.end()) {
        return false;
    }
    
    connections_[it->second].weight = new_weight;
    statistics_valid_ = false;
    return true;
}

std::vector<Connection> NeuralNetwork::getAllConnections() const {
    return connections_;
}

// === 网络拓扑查询实现 ===

std::vector<Connection> NeuralNetwork::getIncomingConnections(NeuronId neuron_id) const {
    std::vector<Connection> incoming;
    
    auto it = incoming_connections_.find(neuron_id);
    if (it != incoming_connections_.end()) {
        for (size_t idx : it->second) {
            incoming.push_back(connections_[idx]);
        }
    }
    
    return incoming;
}

std::vector<Connection> NeuralNetwork::getOutgoingConnections(NeuronId neuron_id) const {
    std::vector<Connection> outgoing;
    
    auto it = outgoing_connections_.find(neuron_id);
    if (it != outgoing_connections_.end()) {
        for (size_t idx : it->second) {
            outgoing.push_back(connections_[idx]);
        }
    }
    
    return outgoing;
}

std::vector<NeuronId> NeuralNetwork::getPredecessors(NeuronId neuron_id) const {
    std::vector<NeuronId> predecessors;
    
    auto it = incoming_connections_.find(neuron_id);
    if (it != incoming_connections_.end()) {
        for (size_t idx : it->second) {
            predecessors.push_back(connections_[idx].source_id);
        }
    }
    
    return predecessors;
}

std::vector<NeuronId> NeuralNetwork::getSuccessors(NeuronId neuron_id) const {
    std::vector<NeuronId> successors;
    
    auto it = outgoing_connections_.find(neuron_id);
    if (it != outgoing_connections_.end()) {
        for (size_t idx : it->second) {
            successors.push_back(connections_[idx].target_id);
        }
    }
    
    return successors;
}

uint32_t NeuralNetwork::getInDegree(NeuronId neuron_id) const {
    auto it = incoming_connections_.find(neuron_id);
    return (it != incoming_connections_.end()) ? static_cast<uint32_t>(it->second.size()) : 0;
}

uint32_t NeuralNetwork::getOutDegree(NeuronId neuron_id) const {
    auto it = outgoing_connections_.find(neuron_id);
    return (it != outgoing_connections_.end()) ? static_cast<uint32_t>(it->second.size()) : 0;
}

// === 网络分析实现 ===

NetworkStatistics NeuralNetwork::calculateStatistics() const {
    if (statistics_valid_) {
        return cached_statistics_;
    }
    
    NetworkStatistics stats;
    stats.total_neurons = static_cast<uint32_t>(neurons_.size());
    stats.total_connections = static_cast<uint32_t>(connections_.size());
    
    if (stats.total_neurons == 0) {
        cached_statistics_ = stats;
        statistics_valid_ = true;
        return stats;
    }
    
    // 计算度分布
    std::vector<uint32_t> in_degrees, out_degrees;
    double weight_sum = 0.0;
    double weight_sum_sq = 0.0;
    
    for (const auto& [neuron_id, properties] : neurons_) {
        uint32_t in_degree = getInDegree(neuron_id);
        uint32_t out_degree = getOutDegree(neuron_id);
        
        in_degrees.push_back(in_degree);
        out_degrees.push_back(out_degree);
        
        stats.max_in_degree = std::max(stats.max_in_degree, in_degree);
        stats.max_out_degree = std::max(stats.max_out_degree, out_degree);
    }
    
    // 计算平均度数
    stats.average_in_degree = static_cast<float>(
        std::accumulate(in_degrees.begin(), in_degrees.end(), 0.0) / in_degrees.size());
    stats.average_out_degree = static_cast<float>(
        std::accumulate(out_degrees.begin(), out_degrees.end(), 0.0) / out_degrees.size());
    
    // 计算权重统计
    for (const auto& conn : connections_) {
        weight_sum += std::abs(conn.weight);
        weight_sum_sq += conn.weight * conn.weight;
    }
    
    if (!connections_.empty()) {
        stats.average_weight = static_cast<float>(weight_sum / connections_.size());
        stats.weight_variance = static_cast<float>(
            weight_sum_sq / connections_.size() - stats.average_weight * stats.average_weight);
    }
    
    // 计算连接密度
    uint64_t max_possible_connections = static_cast<uint64_t>(stats.total_neurons) * (stats.total_neurons - 1);
    stats.connection_density = (max_possible_connections > 0) ? 
        static_cast<float>(stats.total_connections) / max_possible_connections : 0.0f;
    
    // 缓存结果
    cached_statistics_ = stats;
    statistics_valid_ = true;
    
    return stats;
}

std::unordered_map<NeuronId, uint32_t> NeuralNetwork::findStronglyConnectedComponents() const {
    std::unordered_map<NeuronId, uint32_t> component_map;
    std::unordered_map<NeuronId, int32_t> indices;
    std::unordered_map<NeuronId, int32_t> lowlinks;
    std::unordered_set<NeuronId> on_stack;
    std::stack<NeuronId> stack;
    
    int32_t index = 0;
    uint32_t component_id = 0;
    
    // Tarjan算法的递归实现（使用迭代栈避免栈溢出）
    std::function<void(NeuronId)> strongConnect = [&](NeuronId v) {
        indices[v] = index;
        lowlinks[v] = index;
        index++;
        stack.push(v);
        on_stack.insert(v);
        
        // 遍历所有后继节点
        auto successors = getSuccessors(v);
        for (NeuronId w : successors) {
            if (indices.find(w) == indices.end()) {
                // 后继节点尚未被访问，递归访问
                strongConnect(w);
                lowlinks[v] = std::min(lowlinks[v], lowlinks[w]);
            } else if (on_stack.count(w)) {
                // 后继节点在栈中，说明是当前强连通分量的一部分
                lowlinks[v] = std::min(lowlinks[v], indices[w]);
            }
        }
        
        // 如果v是强连通分量的根节点
        if (lowlinks[v] == indices[v]) {
            NeuronId w;
            do {
                w = stack.top();
                stack.pop();
                on_stack.erase(w);
                component_map[w] = component_id;
            } while (w != v);
            component_id++;
        }
    };
    
    // 对所有未访问的节点运行强连通分量检测
    for (const auto& [neuron_id, properties] : neurons_) {
        if (indices.find(neuron_id) == indices.end()) {
            strongConnect(neuron_id);
        }
    }
    
    return component_map;
}

std::vector<NeuronId> NeuralNetwork::topologicalSort() const {
    std::vector<NeuronId> result;
    std::unordered_map<NeuronId, uint32_t> in_degree_count;
    std::queue<NeuronId> zero_in_degree_queue;
    
    // 初始化入度计数
    for (const auto& [neuron_id, properties] : neurons_) {
        in_degree_count[neuron_id] = getInDegree(neuron_id);
        if (in_degree_count[neuron_id] == 0) {
            zero_in_degree_queue.push(neuron_id);
        }
    }
    
    // Kahn算法
    while (!zero_in_degree_queue.empty()) {
        NeuronId current = zero_in_degree_queue.front();
        zero_in_degree_queue.pop();
        result.push_back(current);
        
        // 处理所有后继节点
        auto successors = getSuccessors(current);
        for (NeuronId successor : successors) {
            in_degree_count[successor]--;
            if (in_degree_count[successor] == 0) {
                zero_in_degree_queue.push(successor);
            }
        }
    }
    
    // 检查是否存在环
    if (result.size() != neurons_.size()) {
        return {}; // 存在环，返回空数组
    }
    
    return result;
}

bool NeuralNetwork::isDAG() const {
    return !topologicalSort().empty();
}

std::unordered_map<NeuronId, uint32_t> NeuralNetwork::identifyLayers() const {
    std::unordered_map<NeuronId, uint32_t> layers;
    auto topo_order = topologicalSort();
    
    if (topo_order.empty()) {
        return {}; // 不是DAG，无法分层
    }
    
    // 初始化所有神经元的层级为0
    for (const auto& [neuron_id, properties] : neurons_) {
        layers[neuron_id] = 0;
    }
    
    // 按拓扑序遍历，计算每个神经元的层级
    for (NeuronId neuron_id : topo_order) {
        auto predecessors = getPredecessors(neuron_id);
        uint32_t max_predecessor_layer = 0;
        
        for (NeuronId pred : predecessors) {
            max_predecessor_layer = std::max(max_predecessor_layer, layers[pred]);
        }
        
        if (!predecessors.empty()) {
            layers[neuron_id] = max_predecessor_layer + 1;
        }
    }
    
    return layers;
}

int32_t NeuralNetwork::shortestPathLength(NeuronId source, NeuronId target) const {
    if (source == target) return 0;
    if (!hasNeuron(source) || !hasNeuron(target)) return -1;
    
    // BFS算法
    std::queue<NeuronId> queue;
    std::unordered_map<NeuronId, int32_t> distances;
    
    queue.push(source);
    distances[source] = 0;
    
    while (!queue.empty()) {
        NeuronId current = queue.front();
        queue.pop();
        
        if (current == target) {
            return distances[current];
        }
        
        auto successors = getSuccessors(current);
        for (NeuronId successor : successors) {
            if (distances.find(successor) == distances.end()) {
                distances[successor] = distances[current] + 1;
                queue.push(successor);
            }
        }
    }
    
    return -1; // 不可达
}

// === 网络变换实现 ===

std::unique_ptr<NeuralNetwork> NeuralNetwork::createSubnetwork(const std::vector<NeuronId>& neuron_subset) const {
    auto subnetwork = std::make_unique<NeuralNetwork>();
    
    std::unordered_set<NeuronId> neuron_set(neuron_subset.begin(), neuron_subset.end());
    
    // 添加神经元
    for (NeuronId neuron_id : neuron_subset) {
        auto neuron = getNeuron(neuron_id);
        if (neuron) {
            subnetwork->addNeuron(*neuron);
        }
    }
    
    // 添加子网内的连接
    for (const auto& conn : connections_) {
        if (neuron_set.count(conn.source_id) && neuron_set.count(conn.target_id)) {
            subnetwork->addConnection(conn);
        }
    }
    
    return subnetwork;
}

uint32_t NeuralNetwork::pruneConnections(Weight weight_threshold) {
    uint32_t removed_count = 0;
    
    auto it = connections_.begin();
    while (it != connections_.end()) {
        if (std::abs(it->weight) < weight_threshold) {
            it = connections_.erase(it);
            removed_count++;
        } else {
            ++it;
        }
    }
    
    updateConnectionIndices();
    statistics_valid_ = false;
    
    return removed_count;
}

void NeuralNetwork::randomizeWeights(Weight min_weight, Weight max_weight, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<Weight> dist(min_weight, max_weight);
    
    for (auto& conn : connections_) {
        conn.weight = dist(rng);
    }
    
    statistics_valid_ = false;
}

// === 数据导入导出实现 ===

bool NeuralNetwork::loadFromAdjacencyMatrix(const std::vector<std::vector<Weight>>& adjacency_matrix) {
    clear();
    
    if (adjacency_matrix.empty()) return false;
    
    uint32_t n = static_cast<uint32_t>(adjacency_matrix.size());
    
    // 检查矩阵是否为方阵
    for (const auto& row : adjacency_matrix) {
        if (row.size() != n) return false;
    }
    
    // 添加神经元
    for (uint32_t i = 0; i < n; ++i) {
        addNeuron(NeuronProperties(i));
    }
    
    // 添加连接
    for (uint32_t i = 0; i < n; ++i) {
        for (uint32_t j = 0; j < n; ++j) {
            if (std::abs(adjacency_matrix[i][j]) > 1e-6) { // 非零权重
                Connection conn(i, j, adjacency_matrix[i][j]);
                addConnection(conn);
            }
        }
    }
    
    return true;
}

bool NeuralNetwork::loadFromSparseFormat(const std::vector<Connection>& connections,
                                        const std::vector<NeuronProperties>& neurons) {
    clear();
    
    // 添加神经元
    for (const auto& neuron : neurons) {
        if (!addNeuron(neuron)) {
            clear();
            return false;
        }
    }
    
    // 添加连接
    for (const auto& conn : connections) {
        if (!addConnection(conn)) {
            clear();
            return false;
        }
    }
    
    return true;
}

std::vector<std::vector<Weight>> NeuralNetwork::exportToAdjacencyMatrix() const {
    if (neurons_.empty()) {
        return {};
    }
    
    auto neuron_ids = getAllNeuronIds();
    std::sort(neuron_ids.begin(), neuron_ids.end());
    
    size_t n = neuron_ids.size();
    std::vector<std::vector<Weight>> matrix(n, std::vector<Weight>(n, 0.0f));
    
    // 创建ID到索引的映射
    std::unordered_map<NeuronId, size_t> id_to_index;
    for (size_t i = 0; i < n; ++i) {
        id_to_index[neuron_ids[i]] = i;
    }
    
    // 填充权重矩阵
    for (const auto& conn : connections_) {
        auto pre_it = id_to_index.find(conn.source_id);
        auto post_it = id_to_index.find(conn.target_id);
        
        if (pre_it != id_to_index.end() && post_it != id_to_index.end()) {
            matrix[pre_it->second][post_it->second] = conn.weight;
        }
    }
    
    return matrix;
}

void NeuralNetwork::clear() {
    neurons_.clear();
    connections_.clear();
    connection_index_.clear();
    incoming_connections_.clear();
    outgoing_connections_.clear();
    statistics_valid_ = false;
}

std::vector<std::string> NeuralNetwork::validateConsistency() const {
    std::vector<std::string> errors;
    
    // 检查连接是否引用了存在的神经元
    for (const auto& conn : connections_) {
        if (!hasNeuron(conn.source_id)) {
            errors.push_back("Connection references non-existent pre-neuron: " + std::to_string(conn.source_id));
        }
        if (!hasNeuron(conn.target_id)) {
            errors.push_back("Connection references non-existent post-neuron: " + std::to_string(conn.target_id));
        }
    }
    
    // 检查索引一致性
    if (connection_index_.size() != connections_.size()) {
        errors.push_back("Connection index size mismatch");
    }
    
    return errors;
}

// === 辅助方法实现 ===

void NeuralNetwork::updateConnectionIndices() {
    connection_index_.clear();
    incoming_connections_.clear();
    outgoing_connections_.clear();
    
    for (size_t i = 0; i < connections_.size(); ++i) {
        addConnectionToIndices(i);
    }
}

void NeuralNetwork::addConnectionToIndices(size_t connection_idx) {
    const auto& conn = connections_[connection_idx];
    uint64_t key = makeConnectionKey(conn.source_id, conn.target_id);
    
    connection_index_[key] = connection_idx;
    incoming_connections_[conn.target_id].push_back(connection_idx);
    outgoing_connections_[conn.source_id].push_back(connection_idx);
}

void NeuralNetwork::removeConnectionFromIndices(size_t connection_idx) {
    const auto& conn = connections_[connection_idx];
    uint64_t key = makeConnectionKey(conn.source_id, conn.target_id);
    
    connection_index_.erase(key);
    
    // 从输入连接索引中移除
    auto& incoming = incoming_connections_[conn.target_id];
    incoming.erase(std::remove(incoming.begin(), incoming.end(), connection_idx), incoming.end());
    
    // 从输出连接索引中移除
    auto& outgoing = outgoing_connections_[conn.source_id];
    outgoing.erase(std::remove(outgoing.begin(), outgoing.end(), connection_idx), outgoing.end());
}

} // namespace NeuronMapping