#include "core/HardwareTopology.h"
#include <algorithm>
#include <queue>
#include <limits>
#include <cmath>
#include <unordered_set>

namespace NeuronMapping {

// === PE管理实现 ===

bool HardwareTopology::addPE(const ProcessingElement& pe) {
    if (pes_.find(pe.id) != pes_.end()) {
        return false; // PE已存在
    }
    
    pes_[pe.id] = pe;
    invalidateCache();
    return true;
}

uint32_t HardwareTopology::addPEs(const std::vector<ProcessingElement>& pes) {
    uint32_t added_count = 0;
    for (const auto& pe : pes) {
        if (addPE(pe)) {
            added_count++;
        }
    }
    return added_count;
}

bool HardwareTopology::removePE(PEId pe_id) {
    auto it = pes_.find(pe_id);
    if (it == pes_.end()) {
        return false; // PE不存在
    }
    
    // 移除所有相关链路
    auto links_to_remove = std::vector<size_t>();
    for (size_t i = 0; i < links_.size(); ++i) {
        const auto& link = links_[i];
        if (link.pe1 == pe_id || link.pe2 == pe_id) {
            links_to_remove.push_back(i);
        }
    }
    
    // 从后向前删除链路
    for (auto it = links_to_remove.rbegin(); it != links_to_remove.rend(); ++it) {
        links_.erase(links_.begin() + *it);
    }
    
    // 移除PE
    pes_.erase(it);
    
    // 重建索引
    updateLinkIndices();
    buildAdjacencyList();
    invalidateCache();
    
    return true;
}

bool HardwareTopology::hasPE(PEId pe_id) const {
    return pes_.find(pe_id) != pes_.end();
}

const ProcessingElement* HardwareTopology::getPE(PEId pe_id) const {
    auto it = pes_.find(pe_id);
    return (it != pes_.end()) ? &it->second : nullptr;
}

bool HardwareTopology::updatePE(const ProcessingElement& pe) {
    auto it = pes_.find(pe.id);
    if (it == pes_.end()) {
        return false;
    }
    
    it->second = pe;
    return true;
}

std::vector<PEId> HardwareTopology::getAllPEIds() const {
    std::vector<PEId> ids;
    ids.reserve(pes_.size());
    
    for (const auto& [id, pe] : pes_) {
        ids.push_back(id);
    }
    
    return ids;
}

// === 网络链路管理实现 ===

bool HardwareTopology::addLink(const NetworkLink& link) {
    // 检查PE是否存在
    if (!hasPE(link.pe1) || !hasPE(link.pe2)) {
        return false;
    }
    
    // 检查链路是否已存在
    uint64_t key = makeLinkKey(link.pe1, link.pe2);
    if (link_index_.find(key) != link_index_.end()) {
        return false; // 链路已存在
    }
    
    // 添加链路
    links_.push_back(link);
    addLinkToIndices(link);
    invalidateCache();
    
    return true;
}

uint32_t HardwareTopology::addLinks(const std::vector<NetworkLink>& links) {
    uint32_t added_count = 0;
    for (const auto& link : links) {
        if (addLink(link)) {
            added_count++;
        }
    }
    return added_count;
}

bool HardwareTopology::removeLink(PEId pe1, PEId pe2) {
    uint64_t key = makeLinkKey(pe1, pe2);
    auto it = link_index_.find(key);
    
    if (it == link_index_.end()) {
        return false; // 链路不存在
    }
    
    size_t link_idx = it->second;
    
    // 从索引中移除
    removeLinkFromIndices(pe1, pe2);
    
    // 从链路数组中移除
    links_.erase(links_.begin() + link_idx);
    
    // 重建索引
    updateLinkIndices();
    buildAdjacencyList();
    invalidateCache();
    
    return true;
}

bool HardwareTopology::hasLink(PEId pe1, PEId pe2) const {
    uint64_t key = makeLinkKey(pe1, pe2);
    return link_index_.find(key) != link_index_.end();
}

const NetworkLink* HardwareTopology::getLink(PEId pe1, PEId pe2) const {
    uint64_t key = makeLinkKey(pe1, pe2);
    auto it = link_index_.find(key);
    
    if (it == link_index_.end()) {
        return nullptr;
    }
    
    return &links_[it->second];
}

std::vector<NetworkLink> HardwareTopology::getAllLinks() const {
    return links_;
}

// === 拓扑查询实现 ===

std::vector<PEId> HardwareTopology::getNeighbors(PEId pe_id) const {
    auto it = adjacency_list_.find(pe_id);
    return (it != adjacency_list_.end()) ? it->second : std::vector<PEId>();
}

uint32_t HardwareTopology::getDegree(PEId pe_id) const {
    auto neighbors = getNeighbors(pe_id);
    return static_cast<uint32_t>(neighbors.size());
}

int32_t HardwareTopology::getDistance(PEId pe1, PEId pe2) const {
    if (pe1 == pe2) return 0;
    if (!hasPE(pe1) || !hasPE(pe2)) return -1;
    
    // 确保距离矩阵已计算
    if (!distance_matrix_valid_) {
        computeDistanceMatrix();
    }
    
    // 找到PE在距离矩阵中的索引
    auto all_pe_ids = getAllPEIds();
    std::sort(all_pe_ids.begin(), all_pe_ids.end());
    
    auto it1 = std::find(all_pe_ids.begin(), all_pe_ids.end(), pe1);
    auto it2 = std::find(all_pe_ids.begin(), all_pe_ids.end(), pe2);
    
    if (it1 == all_pe_ids.end() || it2 == all_pe_ids.end()) {
        return -1;
    }
    
    size_t idx1 = std::distance(all_pe_ids.begin(), it1);
    size_t idx2 = std::distance(all_pe_ids.begin(), it2);
    
    return distance_matrix_[idx1][idx2];
}

std::vector<PEId> HardwareTopology::getPath(PEId pe1, PEId pe2) const {
    if (pe1 == pe2) return {pe1};
    if (!hasPE(pe1) || !hasPE(pe2)) return {};
    
    // BFS寻找最短路径
    std::queue<PEId> queue;
    std::unordered_map<PEId, PEId> parent;
    std::unordered_set<PEId> visited;
    
    queue.push(pe1);
    visited.insert(pe1);
    parent[pe1] = pe1;
    
    while (!queue.empty()) {
        PEId current = queue.front();
        queue.pop();
        
        if (current == pe2) {
            // 重构路径
            std::vector<PEId> path;
            PEId node = pe2;
            while (node != pe1) {
                path.push_back(node);
                node = parent[node];
            }
            path.push_back(pe1);
            std::reverse(path.begin(), path.end());
            return path;
        }
        
        auto neighbors = getNeighbors(current);
        for (PEId neighbor : neighbors) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                parent[neighbor] = current;
                queue.push(neighbor);
            }
        }
    }
    
    return {}; // 不可达
}

const std::vector<std::vector<int32_t>>& HardwareTopology::getDistanceMatrix() const {
    if (!distance_matrix_valid_) {
        computeDistanceMatrix();
    }
    return distance_matrix_;
}

float HardwareTopology::calculateCommunicationCost(PEId pe1, PEId pe2, float traffic_volume) const {
    if (pe1 == pe2) return 0.0f; // 本地通信无成本
    
    int32_t distance = getDistance(pe1, pe2);
    if (distance < 0) return std::numeric_limits<float>::infinity(); // 不可达
    
    // 基础通信成本 = 流量 × 距离 × 延迟因子
    float base_cost = traffic_volume * distance;
    
    // 考虑网络拥塞和带宽限制
    auto path = getPath(pe1, pe2);
    float total_latency = 0.0f;
    float min_bandwidth = std::numeric_limits<float>::max();
    
    for (size_t i = 0; i < path.size() - 1; ++i) {
        auto link = getLink(path[i], path[i + 1]);
        if (link) {
            total_latency += link->latency * link->congestion_factor;
            min_bandwidth = std::min(min_bandwidth, link->bandwidth);
        }
    }
    
    // 带宽限制成本
    float bandwidth_cost = (min_bandwidth > 0) ? traffic_volume / min_bandwidth : 0.0f;
    
    return base_cost + total_latency + bandwidth_cost;
}

// === 拓扑生成实现 ===

bool HardwareTopology::createMesh2D(uint32_t rows, uint32_t cols, const ProcessingElement& pe_config) {
    clear();
    topology_type_ = "mesh2d";
    
    // 创建PEs
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            ProcessingElement pe = pe_config;
            pe.id = r * cols + c;
            
            if (!addPE(pe)) {
                clear();
                return false;
            }
        }
    }
    
    // 创建链路
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t c = 0; c < cols; ++c) {
            PEId current_id = r * cols + c;
            
            // 向右连接
            if (c < cols - 1) {
                PEId right_id = r * cols + (c + 1);
                addLink(NetworkLink(current_id, right_id));
            }
            
            // 向下连接
            if (r < rows - 1) {
                PEId down_id = (r + 1) * cols + c;
                addLink(NetworkLink(current_id, down_id));
            }
        }
    }
    
    buildAdjacencyList();
    return true;
}

bool HardwareTopology::createMesh3D(uint32_t x_dim, uint32_t y_dim, uint32_t z_dim, 
                                   const ProcessingElement& pe_config) {
    clear();
    topology_type_ = "mesh3d";
    
    // 创建PEs
    for (uint32_t x = 0; x < x_dim; ++x) {
        for (uint32_t y = 0; y < y_dim; ++y) {
            for (uint32_t z = 0; z < z_dim; ++z) {
                ProcessingElement pe = pe_config;
                pe.id = x * (y_dim * z_dim) + y * z_dim + z;
                
                if (!addPE(pe)) {
                    clear();
                    return false;
                }
            }
        }
    }
    
    // 创建链路
    for (uint32_t x = 0; x < x_dim; ++x) {
        for (uint32_t y = 0; y < y_dim; ++y) {
            for (uint32_t z = 0; z < z_dim; ++z) {
                PEId current_id = x * (y_dim * z_dim) + y * z_dim + z;
                
                // X方向连接
                if (x < x_dim - 1) {
                    PEId next_x = (x + 1) * (y_dim * z_dim) + y * z_dim + z;
                    addLink(NetworkLink(current_id, next_x));
                }
                
                // Y方向连接
                if (y < y_dim - 1) {
                    PEId next_y = x * (y_dim * z_dim) + (y + 1) * z_dim + z;
                    addLink(NetworkLink(current_id, next_y));
                }
                
                // Z方向连接
                if (z < z_dim - 1) {
                    PEId next_z = x * (y_dim * z_dim) + y * z_dim + (z + 1);
                    addLink(NetworkLink(current_id, next_z));
                }
            }
        }
    }
    
    buildAdjacencyList();
    return true;
}

bool HardwareTopology::createTorus2D(uint32_t rows, uint32_t cols, const ProcessingElement& pe_config) {
    // 先创建2D Mesh
    if (!createMesh2D(rows, cols, pe_config)) {
        return false;
    }
    
    topology_type_ = "torus2d";
    
    // 添加环形连接
    for (uint32_t r = 0; r < rows; ++r) {
        // 水平环形连接
        PEId left_end = r * cols + 0;
        PEId right_end = r * cols + (cols - 1);
        addLink(NetworkLink(left_end, right_end));
    }
    
    for (uint32_t c = 0; c < cols; ++c) {
        // 垂直环形连接
        PEId top_end = 0 * cols + c;
        PEId bottom_end = (rows - 1) * cols + c;
        addLink(NetworkLink(top_end, bottom_end));
    }
    
    buildAdjacencyList();
    return true;
}

bool HardwareTopology::createTree(uint32_t levels, uint32_t branching_factor, 
                                 const ProcessingElement& pe_config) {
    clear();
    topology_type_ = "tree";
    
    if (levels == 0 || branching_factor == 0) return false;
    
    // 计算总节点数
    uint32_t total_nodes = 0;
    for (uint32_t l = 0; l < levels; ++l) {
        total_nodes += static_cast<uint32_t>(std::pow(branching_factor, l));
    }
    
    // 创建PEs
    for (uint32_t i = 0; i < total_nodes; ++i) {
        ProcessingElement pe = pe_config;
        pe.id = i;
        
        if (!addPE(pe)) {
            clear();
            return false;
        }
    }
    
    // 创建树形连接
    PEId current_id = 0;
    for (uint32_t level = 0; level < levels - 1; ++level) {
        uint32_t level_nodes = static_cast<uint32_t>(std::pow(branching_factor, level));
        
        for (uint32_t node = 0; node < level_nodes; ++node) {
            PEId parent_id = current_id + node;
            PEId first_child = current_id + level_nodes;
            
            for (uint32_t child = 0; child < branching_factor; ++child) {
                PEId child_id = first_child + node * branching_factor + child;
                addLink(NetworkLink(parent_id, child_id));
            }
        }
        current_id += level_nodes;
    }
    
    buildAdjacencyList();
    return true;
}

bool HardwareTopology::createFullyConnected(uint32_t num_pes, const ProcessingElement& pe_config) {
    clear();
    topology_type_ = "fully_connected";
    
    // 创建PEs
    for (uint32_t i = 0; i < num_pes; ++i) {
        ProcessingElement pe = pe_config;
        pe.id = i;
        
        if (!addPE(pe)) {
            clear();
            return false;
        }
    }
    
    // 创建全连接
    for (uint32_t i = 0; i < num_pes; ++i) {
        for (uint32_t j = i + 1; j < num_pes; ++j) {
            addLink(NetworkLink(i, j));
        }
    }
    
    buildAdjacencyList();
    return true;
}

bool HardwareTopology::createRing(uint32_t num_pes, const ProcessingElement& pe_config) {
    clear();
    topology_type_ = "ring";
    
    if (num_pes < 3) return false; // 至少需要3个节点形成环
    
    // 创建PEs
    for (uint32_t i = 0; i < num_pes; ++i) {
        ProcessingElement pe = pe_config;
        pe.id = i;
        
        if (!addPE(pe)) {
            clear();
            return false;
        }
    }
    
    // 创建环形连接
    for (uint32_t i = 0; i < num_pes; ++i) {
        PEId next_id = (i + 1) % num_pes;
        addLink(NetworkLink(i, next_id));
    }
    
    buildAdjacencyList();
    return true;
}

// === 容量管理实现 ===

uint32_t HardwareTopology::getTotalNeuronCapacity() const {
    uint32_t total = 0;
    for (const auto& [pe_id, pe] : pes_) {
        total += pe.max_neurons;
    }
    return total;
}

uint64_t HardwareTopology::getTotalMemoryCapacity() const {
    uint64_t total = 0;
    for (const auto& [pe_id, pe] : pes_) {
        total += pe.memory_capacity;
    }
    return total;
}

uint32_t HardwareTopology::getRemainingNeuronCapacity(PEId pe_id, 
                    const std::unordered_map<PEId, std::vector<NeuronId>>& current_assignment) const {
    auto pe = getPE(pe_id);
    if (!pe) return 0;
    
    auto it = current_assignment.find(pe_id);
    uint32_t used = (it != current_assignment.end()) ? static_cast<uint32_t>(it->second.size()) : 0;
    
    return (pe->max_neurons > used) ? (pe->max_neurons - used) : 0;
}

uint64_t HardwareTopology::getRemainingMemoryCapacity(PEId pe_id, uint64_t current_memory_usage) const {
    auto pe = getPE(pe_id);
    if (!pe) return 0;
    
    return (pe->memory_capacity > current_memory_usage) ? 
           (pe->memory_capacity - current_memory_usage) : 0;
}

bool HardwareTopology::canAccommodateNeurons(PEId pe_id, uint32_t neuron_count,
                        const std::unordered_map<PEId, std::vector<NeuronId>>& current_assignment) const {
    return getRemainingNeuronCapacity(pe_id, current_assignment) >= neuron_count;
}

// === 拓扑分析实现 ===

uint32_t HardwareTopology::calculateNetworkDiameter() const {
    if (!distance_matrix_valid_) {
        computeDistanceMatrix();
    }
    
    uint32_t diameter = 0;
    for (const auto& row : distance_matrix_) {
        for (int32_t distance : row) {
            if (distance > 0 && distance != std::numeric_limits<int32_t>::max()) {
                diameter = std::max(diameter, static_cast<uint32_t>(distance));
            }
        }
    }
    
    return diameter;
}

float HardwareTopology::calculateAveragePathLength() const {
    if (!distance_matrix_valid_) {
        computeDistanceMatrix();
    }
    
    float total_distance = 0.0f;
    uint32_t path_count = 0;
    
    for (const auto& row : distance_matrix_) {
        for (int32_t distance : row) {
            if (distance > 0 && distance != std::numeric_limits<int32_t>::max()) {
                total_distance += distance;
                path_count++;
            }
        }
    }
    
    return (path_count > 0) ? (total_distance / path_count) : 0.0f;
}

float HardwareTopology::calculateTotalBandwidth() const {
    float total = 0.0f;
    for (const auto& link : links_) {
        total += link.bandwidth;
    }
    return total;
}

bool HardwareTopology::isConnected() const {
    if (pes_.empty()) return true;
    
    // BFS检查连通性
    std::unordered_set<PEId> visited;
    std::queue<PEId> queue;
    
    PEId start = pes_.begin()->first;
    queue.push(start);
    visited.insert(start);
    
    while (!queue.empty()) {
        PEId current = queue.front();
        queue.pop();
        
        auto neighbors = getNeighbors(current);
        for (PEId neighbor : neighbors) {
            if (visited.find(neighbor) == visited.end()) {
                visited.insert(neighbor);
                queue.push(neighbor);
            }
        }
    }
    
    return visited.size() == pes_.size();
}

std::vector<NetworkLink> HardwareTopology::findBottleneckLinks() const {
    std::vector<NetworkLink> bottlenecks;
    
    if (links_.empty()) return bottlenecks;
    
    // 找到最小带宽
    float min_bandwidth = std::numeric_limits<float>::max();
    for (const auto& link : links_) {
        min_bandwidth = std::min(min_bandwidth, link.bandwidth);
    }
    
    // 找到所有最小带宽链路
    for (const auto& link : links_) {
        if (std::abs(link.bandwidth - min_bandwidth) < 1e-6f) {
            bottlenecks.push_back(link);
        }
    }
    
    return bottlenecks;
}

// === 负载均衡分析实现 ===

std::unordered_map<PEId, float> HardwareTopology::calculatePELoads(
    const std::unordered_map<PEId, std::vector<NeuronId>>& assignment,
    const std::unordered_map<NeuronId, float>& neuron_loads) const {
    
    std::unordered_map<PEId, float> pe_loads;
    
    // 初始化所有PE的负载为0
    for (const auto& [pe_id, pe] : pes_) {
        pe_loads[pe_id] = 0.0f;
    }
    
    // 计算每个PE的总负载
    for (const auto& [pe_id, neurons] : assignment) {
        float total_load = 0.0f;
        for (NeuronId neuron_id : neurons) {
            auto it = neuron_loads.find(neuron_id);
            if (it != neuron_loads.end()) {
                total_load += it->second;
            } else {
                total_load += 1.0f; // 默认负载
            }
        }
        pe_loads[pe_id] = total_load;
    }
    
    return pe_loads;
}

PEId HardwareTopology::findMostLoadedPE(const std::unordered_map<PEId, float>& pe_loads) const {
    PEId most_loaded = INVALID_PE_ID;
    float max_load = -1.0f;
    
    for (const auto& [pe_id, load] : pe_loads) {
        if (load > max_load) {
            max_load = load;
            most_loaded = pe_id;
        }
    }
    
    return most_loaded;
}

PEId HardwareTopology::findLeastLoadedPE(const std::unordered_map<PEId, float>& pe_loads) const {
    PEId least_loaded = INVALID_PE_ID;
    float min_load = std::numeric_limits<float>::max();
    
    for (const auto& [pe_id, load] : pe_loads) {
        if (load < min_load) {
            min_load = load;
            least_loaded = pe_id;
        }
    }
    
    return least_loaded;
}

// === 序列化实现 ===

void HardwareTopology::clear() {
    pes_.clear();
    links_.clear();
    link_index_.clear();
    adjacency_list_.clear();
    invalidateCache();
    topology_type_ = "custom";
}

std::vector<std::string> HardwareTopology::validateTopology() const {
    std::vector<std::string> errors;
    
    // 检查链路是否引用了存在的PE
    for (const auto& link : links_) {
        if (!hasPE(link.pe1)) {
            errors.push_back("Link references non-existent PE1: " + std::to_string(link.pe1));
        }
        if (!hasPE(link.pe2)) {
            errors.push_back("Link references non-existent PE2: " + std::to_string(link.pe2));
        }
    }
    
    // 检查索引一致性
    if (link_index_.size() != links_.size()) {
        errors.push_back("Link index size mismatch");
    }
    
    return errors;
}

std::unique_ptr<HardwareTopology> HardwareTopology::clone() const {
    auto cloned = std::make_unique<HardwareTopology>();
    
    cloned->pes_ = pes_;
    cloned->links_ = links_;
    cloned->topology_type_ = topology_type_;
    
    cloned->updateLinkIndices();
    cloned->buildAdjacencyList();
    
    return cloned;
}

// === 辅助方法实现 ===

void HardwareTopology::updateLinkIndices() {
    link_index_.clear();
    
    for (size_t i = 0; i < links_.size(); ++i) {
        uint64_t key = makeLinkKey(links_[i].pe1, links_[i].pe2);
        link_index_[key] = i;
    }
}

void HardwareTopology::addLinkToIndices(const NetworkLink& link) {
    uint64_t key = makeLinkKey(link.pe1, link.pe2);
    link_index_[key] = links_.size() - 1;
}

void HardwareTopology::removeLinkFromIndices(PEId pe1, PEId pe2) {
    uint64_t key = makeLinkKey(pe1, pe2);
    link_index_.erase(key);
}

void HardwareTopology::buildAdjacencyList() {
    adjacency_list_.clear();
    
    // 初始化邻接表
    for (const auto& [pe_id, pe] : pes_) {
        adjacency_list_[pe_id] = std::vector<PEId>();
    }
    
    // 填充邻接表
    for (const auto& link : links_) {
        adjacency_list_[link.pe1].push_back(link.pe2);
        adjacency_list_[link.pe2].push_back(link.pe1); // 无向图
    }
}

void HardwareTopology::computeDistanceMatrix() const {
    auto all_pe_ids = getAllPEIds();
    std::sort(all_pe_ids.begin(), all_pe_ids.end());
    
    size_t n = all_pe_ids.size();
    distance_matrix_.assign(n, std::vector<int32_t>(n, std::numeric_limits<int32_t>::max()));
    
    // 初始化对角线为0
    for (size_t i = 0; i < n; ++i) {
        distance_matrix_[i][i] = 0;
    }
    
    // 创建PE ID到索引的映射
    std::unordered_map<PEId, size_t> pe_to_index;
    for (size_t i = 0; i < n; ++i) {
        pe_to_index[all_pe_ids[i]] = i;
    }
    
    // 初始化直接连接的距离
    for (const auto& link : links_) {
        auto it1 = pe_to_index.find(link.pe1);
        auto it2 = pe_to_index.find(link.pe2);
        
        if (it1 != pe_to_index.end() && it2 != pe_to_index.end()) {
            distance_matrix_[it1->second][it2->second] = 1;
            distance_matrix_[it2->second][it1->second] = 1; // 无向图
        }
    }
    
    // Floyd-Warshall算法
    floydWarshall();
    
    distance_matrix_valid_ = true;
}

void HardwareTopology::invalidateCache() {
    distance_matrix_valid_ = false;
    distance_matrix_.clear();
}

void HardwareTopology::floydWarshall() const {
    size_t n = distance_matrix_.size();
    
    for (size_t k = 0; k < n; ++k) {
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (distance_matrix_[i][k] != std::numeric_limits<int32_t>::max() &&
                    distance_matrix_[k][j] != std::numeric_limits<int32_t>::max()) {
                    
                    int32_t new_distance = distance_matrix_[i][k] + distance_matrix_[k][j];
                    if (new_distance < distance_matrix_[i][j]) {
                        distance_matrix_[i][j] = new_distance;
                    }
                }
            }
        }
    }
}

uint32_t HardwareTopology::getPECapacity(PEId pe_id) const {
    auto it = pes_.find(pe_id);
    if (it != pes_.end()) {
        return it->second.max_neurons;
    }
    return 0;
}

uint64_t HardwareTopology::getPEMemoryCapacity(PEId pe_id) const {
    auto it = pes_.find(pe_id);
    if (it != pes_.end()) {
        return it->second.memory_capacity;
    }
    return 0;
}

} // namespace NeuronMapping