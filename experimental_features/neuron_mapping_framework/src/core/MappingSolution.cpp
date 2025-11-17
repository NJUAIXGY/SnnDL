#include "core/MappingSolution.h"
#include "utils/Logger.h"
#include "utils/MathUtils.h"
#include <algorithm>
#include <random>
#include <cmath>
#include <sstream>
#include <limits>

namespace NeuronMapping {

// === 构造函数 ===

MappingSolution::MappingSolution(uint32_t num_neurons) {
    // 预留空间以提高性能
    neuron_to_assignment_.reserve(num_neurons);
}

// === 映射分配管理 ===

bool MappingSolution::assignNeuron(NeuronId neuron_id, PEId pe_id, uint32_t core_id) {
    // 检查神经元是否已经分配
    if (neuron_to_assignment_.find(neuron_id) != neuron_to_assignment_.end()) {
        return false;
    }
    
    // 创建分配信息
    Assignment assignment;
    assignment.neuron_id = neuron_id;
    assignment.pe_id = pe_id;
    assignment.core_id = core_id;
    assignment.timestamp = std::chrono::steady_clock::now();
    
    // 添加到映射
    neuron_to_assignment_[neuron_id] = assignment;
    addNeuronToPE(neuron_id, pe_id);
    
    // 清除缓存的性能指标
    clearCachedMetrics();
    
    return true;
}

bool MappingSolution::moveNeuron(NeuronId neuron_id, PEId target_pe_id, uint32_t core_id) {
    // 先取消当前分配
    if (!unassignNeuron(neuron_id)) {
        return false;
    }
    
    // 重新分配到目标PE
    return assignNeuron(neuron_id, target_pe_id, core_id);
}

uint32_t MappingSolution::assignNeurons(const std::vector<Assignment>& assignments) {
    uint32_t successful_assignments = 0;
    
    for (const auto& assignment : assignments) {
        if (assignNeuron(assignment.neuron_id, assignment.pe_id, assignment.core_id)) {
            successful_assignments++;
        }
    }
    
    return successful_assignments;
}

bool MappingSolution::unassignNeuron(NeuronId neuron_id) {
    auto it = neuron_to_assignment_.find(neuron_id);
    if (it == neuron_to_assignment_.end()) {
        return false;
    }
    
    PEId pe_id = it->second.pe_id;
    removeNeuronFromPE(neuron_id, pe_id);
    neuron_to_assignment_.erase(it);
    
    // 清除缓存的性能指标
    clearCachedMetrics();
    
    return true;
}

bool MappingSolution::reassignNeuron(NeuronId neuron_id, PEId new_pe_id, uint32_t new_core_id) {
    auto it = neuron_to_assignment_.find(neuron_id);
    if (it == neuron_to_assignment_.end()) {
        return false;
    }
    
    PEId old_pe_id = it->second.pe_id;
    
    // 更新分配信息
    it->second.pe_id = new_pe_id;
    it->second.core_id = new_core_id;
    it->second.timestamp = std::chrono::steady_clock::now();
    
    // 更新PE映射
    updatePEMapping(neuron_id, old_pe_id, new_pe_id);
    
    // 清除缓存的性能指标
    clearCachedMetrics();
    
    return true;
}

bool MappingSolution::swapNeuronAssignments(NeuronId neuron1, NeuronId neuron2) {
    auto it1 = neuron_to_assignment_.find(neuron1);
    auto it2 = neuron_to_assignment_.find(neuron2);
    
    if (it1 == neuron_to_assignment_.end() || it2 == neuron_to_assignment_.end()) {
        return false;
    }
    
    PEId pe1 = it1->second.pe_id;
    PEId pe2 = it2->second.pe_id;
    uint32_t core1 = it1->second.core_id;
    uint32_t core2 = it2->second.core_id;
    
    // 交换分配
    it1->second.pe_id = pe2;
    it1->second.core_id = core2;
    it1->second.timestamp = std::chrono::steady_clock::now();
    
    it2->second.pe_id = pe1;
    it2->second.core_id = core1;
    it2->second.timestamp = std::chrono::steady_clock::now();
    
    // 更新PE映射
    updatePEMapping(neuron1, pe1, pe2);
    updatePEMapping(neuron2, pe2, pe1);
    
    // 清除缓存的性能指标
    clearCachedMetrics();
    
    return true;
}

// === 映射查询 ===

PEId MappingSolution::getNeuronPE(NeuronId neuron_id) const {
    auto it = neuron_to_assignment_.find(neuron_id);
    if (it != neuron_to_assignment_.end()) {
        return it->second.pe_id;
    }
    return INVALID_PE_ID;
}

uint32_t MappingSolution::getNeuronCore(NeuronId neuron_id) const {
    auto it = neuron_to_assignment_.find(neuron_id);
    if (it != neuron_to_assignment_.end()) {
        return it->second.core_id;
    }
    return 0;
}

const Assignment* MappingSolution::getNeuronAssignment(NeuronId neuron_id) const {
    auto it = neuron_to_assignment_.find(neuron_id);
    if (it != neuron_to_assignment_.end()) {
        return &it->second;
    }
    return nullptr;
}

bool MappingSolution::isNeuronAssigned(NeuronId neuron_id) const {
    return neuron_to_assignment_.find(neuron_id) != neuron_to_assignment_.end();
}

std::vector<NeuronId> MappingSolution::getPENeurons(PEId pe_id) const {
    auto it = pe_to_neurons_.find(pe_id);
    if (it != pe_to_neurons_.end()) {
        return it->second;
    }
    return std::vector<NeuronId>();
}

uint32_t MappingSolution::getPENeuronCount(PEId pe_id) const {
    auto it = pe_to_neurons_.find(pe_id);
    if (it != pe_to_neurons_.end()) {
        return static_cast<uint32_t>(it->second.size());
    }
    return 0;
}

std::vector<Assignment> MappingSolution::getAllAssignments() const {
    std::vector<Assignment> assignments;
    assignments.reserve(neuron_to_assignment_.size());
    
    for (const auto& pair : neuron_to_assignment_) {
        assignments.push_back(pair.second);
    }
    
    return assignments;
}

// === 容量管理 ===

bool MappingSolution::canPEAccommodate(PEId pe_id, uint32_t additional_neurons, 
                                      const HardwareTopology& topology) const {
    uint32_t current_neurons = getPENeuronCount(pe_id);
    uint32_t pe_capacity = topology.getPECapacity(pe_id);
    
    return (current_neurons + additional_neurons) <= pe_capacity;
}

uint32_t MappingSolution::getPERemainingCapacity(PEId pe_id, const HardwareTopology& topology) const {
    uint32_t current_neurons = getPENeuronCount(pe_id);
    uint32_t pe_capacity = topology.getPECapacity(pe_id);
    
    return (current_neurons <= pe_capacity) ? (pe_capacity - current_neurons) : 0;
}

float MappingSolution::calculatePELoad(PEId pe_id, const NeuralNetwork& network) const {
    auto neurons = getPENeurons(pe_id);
    float total_load = 0.0f;
    
    for (NeuronId neuron_id : neurons) {
        if (network.hasNeuron(neuron_id)) {
            const auto* neuron = network.getNeuron(neuron_id);
            if (neuron) {
                // 计算神经元负载：基于计算负载
                total_load += neuron->computational_load;
            }
        }
    }
    
    return total_load;
}

std::unordered_map<PEId, float> MappingSolution::calculateAllPELoads(const NeuralNetwork& network) const {
    std::unordered_map<PEId, float> pe_loads;
    
    for (const auto& pe_neurons_pair : pe_to_neurons_) {
        PEId pe_id = pe_neurons_pair.first;
        pe_loads[pe_id] = calculatePELoad(pe_id, network);
    }
    
    return pe_loads;
}

// === 性能评估 ===

PerformanceMetrics MappingSolution::evaluatePerformance(const NeuralNetwork& network,
                                                       const HardwareTopology& topology,
                                                       const MappingConfig& config) const {
    PerformanceMetrics metrics;
    
    // 计算通信成本
    metrics.communication_cost = calculateCommunicationCost(network, topology);
    
    // 计算负载不均衡
    metrics.load_imbalance = calculateLoadImbalance(network);
    
    // 计算内存使用情况
    auto memory_usage = calculateMemoryUsage(network, topology);
    metrics.average_memory_usage = std::get<0>(memory_usage);
    metrics.peak_memory_usage = std::get<1>(memory_usage);
    metrics.memory_utilization = std::get<2>(memory_usage);
    
    // 计算跨PE通信比例
    metrics.inter_pe_communication_ratio = calculateInterPECommunicationRatio(network);
    
    // 计算平均通信距离
    metrics.average_communication_distance = calculateAverageCommunicationDistance(network, topology);
    
    // 计算目标函数值
    metrics.objective_value = computeObjectiveFunction(network, topology, config);
    
    // 计算其他指标
    metrics.pe_utilization = static_cast<float>(getUsedPECount()) / topology.getTotalPEs();
    metrics.neuron_coverage = static_cast<float>(getAssignedNeuronCount()) / network.getNeuronCount();
    
    return metrics;
}

float MappingSolution::calculateCommunicationCost(const NeuralNetwork& network,
                                                 const HardwareTopology& topology) const {
    float total_cost = 0.0f;
    
    // 遍历所有连接
    for (const auto& connection : network.getAllConnections()) {
        NeuronId src_neuron = connection.source_id;
        NeuronId dst_neuron = connection.target_id;
        
        PEId src_pe = getNeuronPE(src_neuron);
        PEId dst_pe = getNeuronPE(dst_neuron);
        
        // 只计算跨PE通信成本
        if (src_pe != INVALID_PE_ID && dst_pe != INVALID_PE_ID && src_pe != dst_pe) {
            float distance = topology.getDistance(src_pe, dst_pe);
            float weight = connection.weight;
            total_cost += distance * std::abs(weight);
        }
    }
    
    return total_cost;
}

float MappingSolution::calculateLoadImbalance(const NeuralNetwork& network) const {
    if (pe_to_neurons_.empty()) {
        return 0.0f;
    }
    
    std::vector<float> pe_loads;
    pe_loads.reserve(pe_to_neurons_.size());
    
    float total_load = 0.0f;
    for (const auto& pe_neurons_pair : pe_to_neurons_) {
        float load = calculatePELoad(pe_neurons_pair.first, network);
        pe_loads.push_back(load);
        total_load += load;
    }
    
    if (total_load <= 0.0f) {
        return 0.0f;
    }
    
    float average_load = total_load / pe_loads.size();
    float variance = 0.0f;
    
    for (float load : pe_loads) {
        float deviation = load - average_load;
        variance += deviation * deviation;
    }
    
    variance /= pe_loads.size();
    float std_deviation = std::sqrt(variance);
    
    // 负载不均衡系数：标准差 / 平均值
    return (average_load > 0.0f) ? (std_deviation / average_load) : 0.0f;
}

std::tuple<float, float, float> MappingSolution::calculateMemoryUsage(const NeuralNetwork& network,
                                                                     const HardwareTopology& topology) const {
    float total_memory = 0.0f;
    float peak_memory = 0.0f;
    float total_capacity = 0.0f;
    
    for (const auto& pe_neurons_pair : pe_to_neurons_) {
        PEId pe_id = pe_neurons_pair.first;
        const auto& neurons = pe_neurons_pair.second;
        
        float pe_memory = 0.0f;
        for (NeuronId neuron_id : neurons) {
            if (network.hasNeuron(neuron_id)) {
                const auto* neuron = network.getNeuron(neuron_id);
                if (neuron) {
                    pe_memory += neuron->memory_requirement;
                }
            }
        }
        
        total_memory += pe_memory;
        peak_memory = std::max(peak_memory, pe_memory);
        total_capacity += topology.getPEMemoryCapacity(pe_id);
    }
    
    float average_memory = pe_to_neurons_.empty() ? 0.0f : (total_memory / pe_to_neurons_.size());
    float memory_utilization = (total_capacity > 0.0f) ? (total_memory / total_capacity) : 0.0f;
    
    return std::make_tuple(average_memory, peak_memory, memory_utilization);
}

float MappingSolution::calculateInterPECommunicationRatio(const NeuralNetwork& network) const {
    auto connection_counts = countLocalAndRemoteConnections(network);
    uint32_t local_connections = connection_counts.first;
    uint32_t remote_connections = connection_counts.second;
    
    uint32_t total_connections = local_connections + remote_connections;
    if (total_connections == 0) {
        return 0.0f;
    }
    
    return static_cast<float>(remote_connections) / total_connections;
}

float MappingSolution::calculateAverageCommunicationDistance(const NeuralNetwork& network,
                                                           const HardwareTopology& topology) const {
    float total_distance = 0.0f;
    uint32_t remote_connections = 0;
    
    // 遍历所有连接
    for (const auto& connection : network.getAllConnections()) {
        NeuronId src_neuron = connection.source_id;
        NeuronId dst_neuron = connection.target_id;
        
        PEId src_pe = getNeuronPE(src_neuron);
        PEId dst_pe = getNeuronPE(dst_neuron);
        
        // 只计算跨PE通信距离
        if (src_pe != INVALID_PE_ID && dst_pe != INVALID_PE_ID && src_pe != dst_pe) {
            float distance = topology.getDistance(src_pe, dst_pe);
            total_distance += distance;
            remote_connections++;
        }
    }
    
    return (remote_connections > 0) ? (total_distance / remote_connections) : 0.0f;
}

// === 映射优化 ===

bool MappingSolution::randomInitialize(uint32_t num_neurons, const HardwareTopology& topology, uint32_t seed) {
    clear();
    
    std::mt19937 rng(seed == 0 ? std::random_device{}() : seed);
    std::uniform_int_distribution<PEId> pe_dist(0, topology.getTotalPEs() - 1);
    
    for (uint32_t neuron_id = 0; neuron_id < num_neurons; ++neuron_id) {
        // 随机选择PE，确保容量约束
        std::vector<PEId> valid_pes;
        for (PEId pe_id = 0; pe_id < topology.getTotalPEs(); ++pe_id) {
            if (canPEAccommodate(pe_id, 1, topology)) {
                valid_pes.push_back(pe_id);
            }
        }
        
        if (valid_pes.empty()) {
            // 无可用PE，初始化失败
            return false;
        }
        
        std::uniform_int_distribution<size_t> valid_pe_dist(0, valid_pes.size() - 1);
        PEId selected_pe = valid_pes[valid_pe_dist(rng)];
        
        if (!assignNeuron(neuron_id, selected_pe)) {
            return false;
        }
    }
    
    return true;
}

bool MappingSolution::greedyInitialize(const NeuralNetwork& network, const HardwareTopology& topology) {
    clear();
    
    // 获取按连接数排序的神经元列表
    std::vector<NeuronId> neurons = network.getAllNeuronIds();
    std::sort(neurons.begin(), neurons.end(), [&network](NeuronId a, NeuronId b) {
        const auto* neuron_a = network.getNeuron(a);
        const auto* neuron_b = network.getNeuron(b);
        if (!neuron_a || !neuron_b) return false;
        return neuron_a->computational_load > neuron_b->computational_load;
    });
    
    for (NeuronId neuron_id : neurons) {
        PEId best_pe = INVALID_PE_ID;
        float best_cost = std::numeric_limits<float>::max();
        
        // 评估每个可用PE的成本
        for (PEId pe_id = 0; pe_id < topology.getTotalPEs(); ++pe_id) {
            if (!canPEAccommodate(pe_id, 1, topology)) {
                continue;
            }
            
            // 计算分配到此PE的通信成本
            float cost = 0.0f;
            
            // 遍历所有连接，计算与已分配神经元的通信成本
            for (const auto& connection : network.getAllConnections()) {
                if (connection.source_id == neuron_id) {
                    // 输出连接
                    PEId dst_pe = getNeuronPE(connection.target_id);
                    if (dst_pe != INVALID_PE_ID && dst_pe != pe_id) {
                        cost += topology.getDistance(pe_id, dst_pe) * std::abs(connection.weight);
                    }
                } else if (connection.target_id == neuron_id) {
                    // 输入连接
                    PEId src_pe = getNeuronPE(connection.source_id);
                    if (src_pe != INVALID_PE_ID && src_pe != pe_id) {
                        cost += topology.getDistance(src_pe, pe_id) * std::abs(connection.weight);
                    }
                }
            }
            
            if (cost < best_cost) {
                best_cost = cost;
                best_pe = pe_id;
            }
        }
        
        if (best_pe == INVALID_PE_ID) {
            // 无可用PE
            return false;
        }
        
        if (!assignNeuron(neuron_id, best_pe)) {
            return false;
        }
    }
    
    return true;
}

std::unique_ptr<MappingSolution> MappingSolution::generateNeighborSolution(const HardwareTopology& topology,
                                                                          uint32_t max_moves) const {
    auto neighbor = clone();
    
    if (neuron_to_assignment_.empty()) {
        return neighbor;
    }
    
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<size_t> neuron_dist(0, neuron_to_assignment_.size() - 1);
    std::uniform_int_distribution<PEId> pe_dist(0, topology.getTotalPEs() - 1);
    
    uint32_t moves = std::min(max_moves, static_cast<uint32_t>(neuron_to_assignment_.size()));
    
    for (uint32_t move = 0; move < moves; ++move) {
        // 随机选择神经元
        auto it = neuron_to_assignment_.begin();
        std::advance(it, neuron_dist(rng));
        NeuronId neuron_id = it->first;
        PEId current_pe = it->second.pe_id;
        
        // 随机选择新的PE
        PEId new_pe = pe_dist(rng);
        
        // 确保新PE不同且有容量
        if (new_pe != current_pe && neighbor->canPEAccommodate(new_pe, 1, topology)) {
            neighbor->reassignNeuron(neuron_id, new_pe);
        }
    }
    
    return neighbor;
}

float MappingSolution::localOptimize(const NeuralNetwork& network,
                                   const HardwareTopology& topology,
                                   const MappingConfig& config,
                                   uint32_t max_iterations) {
    float initial_score = calculateFitnessScore(network, topology, config);
    float current_score = initial_score;
    
    for (uint32_t iteration = 0; iteration < max_iterations; ++iteration) {
        auto neighbor = generateNeighborSolution(topology, 1);
        float neighbor_score = neighbor->calculateFitnessScore(network, topology, config);
        
        if (neighbor_score > current_score) {
            // 接受更好的解
            *this = *neighbor;
            current_score = neighbor_score;
        }
    }
    
    return current_score - initial_score;
}

// === 约束验证 ===

std::vector<std::string> MappingSolution::validateConstraints(const HardwareTopology& topology) const {
    std::vector<std::string> violations;
    
    // 检查容量约束
    for (const auto& pe_neurons_pair : pe_to_neurons_) {
        PEId pe_id = pe_neurons_pair.first;
        uint32_t neuron_count = static_cast<uint32_t>(pe_neurons_pair.second.size());
        uint32_t capacity = topology.getPECapacity(pe_id);
        
        if (neuron_count > capacity) {
            std::ostringstream oss;
            oss << "PE " << pe_id << " capacity violation: " << neuron_count << " > " << capacity;
            violations.push_back(oss.str());
        }
    }
    
    // 检查映射一致性
    auto consistency_errors = validateConsistency();
    violations.insert(violations.end(), consistency_errors.begin(), consistency_errors.end());
    
    return violations;
}

bool MappingSolution::checkCapacityConstraints(const HardwareTopology& topology) const {
    for (const auto& pe_neurons_pair : pe_to_neurons_) {
        PEId pe_id = pe_neurons_pair.first;
        uint32_t neuron_count = static_cast<uint32_t>(pe_neurons_pair.second.size());
        uint32_t capacity = topology.getPECapacity(pe_id);
        
        if (neuron_count > capacity) {
            return false;
        }
    }
    return true;
}

bool MappingSolution::repairConstraintViolations(const HardwareTopology& topology) {
    // 修复容量约束违反
    for (const auto& pe_neurons_pair : pe_to_neurons_) {
        PEId pe_id = pe_neurons_pair.first;
        auto neurons = pe_neurons_pair.second; // 复制以避免迭代器失效
        uint32_t capacity = topology.getPECapacity(pe_id);
        
        if (neurons.size() > capacity) {
            // 需要迁移多余的神经元
            uint32_t excess = static_cast<uint32_t>(neurons.size()) - capacity;
            
            for (uint32_t i = 0; i < excess; ++i) {
                NeuronId neuron_to_move = neurons[capacity + i];
                
                // 寻找有容量的PE
                bool moved = false;
                for (PEId target_pe = 0; target_pe < topology.getTotalPEs(); ++target_pe) {
                    if (target_pe != pe_id && canPEAccommodate(target_pe, 1, topology)) {
                        reassignNeuron(neuron_to_move, target_pe);
                        moved = true;
                        break;
                    }
                }
                
                if (!moved) {
                    // 无法修复
                    return false;
                }
            }
        }
    }
    
    return true;
}

// === 解决方案比较 ===

int MappingSolution::compare(const MappingSolution& other,
                           const NeuralNetwork& network,
                           const HardwareTopology& topology,
                           const MappingConfig& config) const {
    float this_score = calculateFitnessScore(network, topology, config);
    float other_score = other.calculateFitnessScore(network, topology, config);
    
    if (this_score > other_score) return -1;
    if (this_score < other_score) return 1;
    return 0;
}

float MappingSolution::calculateFitnessScore(const NeuralNetwork& network,
                                           const HardwareTopology& topology,
                                           const MappingConfig& config) const {
    // 检查是否有缓存的有效指标
    if (hasValidCachedMetrics()) {
        return -cached_metrics_.objective_value; // 负值因为我们希望最小化目标函数
    }
    
    // 计算各项指标
    float comm_cost = calculateCommunicationCost(network, topology);
    float load_imbalance = calculateLoadImbalance(network);
    float inter_pe_ratio = calculateInterPECommunicationRatio(network);
    
    // 加权综合分数（负值，因为我们要最大化适应度）
    float score = -(config.communication_weight * comm_cost +
                   config.load_balance_weight * load_imbalance +
                   config.inter_pe_comm_weight * inter_pe_ratio);
    
    return score;
}

// === 数据管理 ===

void MappingSolution::clear() {
    neuron_to_assignment_.clear();
    pe_to_neurons_.clear();
    clearCachedMetrics();
}

std::unique_ptr<MappingSolution> MappingSolution::clone() const {
    auto copy = std::make_unique<MappingSolution>();
    copy->neuron_to_assignment_ = neuron_to_assignment_;
    copy->pe_to_neurons_ = pe_to_neurons_;
    copy->cached_metrics_ = cached_metrics_;
    copy->metrics_valid_ = metrics_valid_;
    return copy;
}

uint32_t MappingSolution::mergePartialMapping(const MappingSolution& other, 
                                             const std::vector<NeuronId>& neuron_subset) {
    uint32_t merged_count = 0;
    
    for (NeuronId neuron_id : neuron_subset) {
        const Assignment* assignment = other.getNeuronAssignment(neuron_id);
        if (assignment != nullptr) {
            if (assignNeuron(assignment->neuron_id, assignment->pe_id, assignment->core_id)) {
                merged_count++;
            }
        }
    }
    
    return merged_count;
}

std::string MappingSolution::getStatistics() const {
    std::ostringstream stats;
    stats << "Mapping Statistics:\n";
    stats << "  Assigned neurons: " << getAssignedNeuronCount() << "\n";
    stats << "  Used PEs: " << getUsedPECount() << "\n";
    
    if (!pe_to_neurons_.empty()) {
        uint32_t min_neurons = UINT32_MAX;
        uint32_t max_neurons = 0;
        float avg_neurons = 0.0f;
        
        for (const auto& pe_neurons_pair : pe_to_neurons_) {
            uint32_t count = static_cast<uint32_t>(pe_neurons_pair.second.size());
            min_neurons = std::min(min_neurons, count);
            max_neurons = std::max(max_neurons, count);
            avg_neurons += count;
        }
        
        avg_neurons /= pe_to_neurons_.size();
        
        stats << "  Neurons per PE: min=" << min_neurons 
              << ", max=" << max_neurons 
              << ", avg=" << avg_neurons << "\n";
    }
    
    if (hasValidCachedMetrics()) {
        stats << "  Communication cost: " << cached_metrics_.communication_cost << "\n";
        stats << "  Load imbalance: " << cached_metrics_.load_imbalance << "\n";
        stats << "  Inter-PE comm ratio: " << cached_metrics_.inter_pe_communication_ratio << "\n";
    }
    
    return stats.str();
}

std::vector<std::string> MappingSolution::validateConsistency() const {
    std::vector<std::string> errors;
    
    // 验证neuron_to_assignment_和pe_to_neurons_的一致性
    for (const auto& neuron_assignment : neuron_to_assignment_) {
        NeuronId neuron_id = neuron_assignment.first;
        PEId pe_id = neuron_assignment.second.pe_id;
        
        // 检查PE映射中是否包含此神经元
        auto pe_it = pe_to_neurons_.find(pe_id);
        if (pe_it == pe_to_neurons_.end()) {
            std::ostringstream oss;
            oss << "Neuron " << neuron_id << " assigned to PE " << pe_id 
                << ", but PE not found in pe_to_neurons_";
            errors.push_back(oss.str());
        } else {
            const auto& neurons = pe_it->second;
            if (std::find(neurons.begin(), neurons.end(), neuron_id) == neurons.end()) {
                std::ostringstream oss;
                oss << "Neuron " << neuron_id << " assigned to PE " << pe_id 
                    << ", but not found in PE's neuron list";
                errors.push_back(oss.str());
            }
        }
    }
    
    // 反向验证
    for (const auto& pe_neurons : pe_to_neurons_) {
        PEId pe_id = pe_neurons.first;
        for (NeuronId neuron_id : pe_neurons.second) {
            auto neuron_it = neuron_to_assignment_.find(neuron_id);
            if (neuron_it == neuron_to_assignment_.end()) {
                std::ostringstream oss;
                oss << "PE " << pe_id << " contains neuron " << neuron_id 
                    << ", but neuron not found in assignment map";
                errors.push_back(oss.str());
            } else if (neuron_it->second.pe_id != pe_id) {
                std::ostringstream oss;
                oss << "PE " << pe_id << " contains neuron " << neuron_id 
                    << ", but neuron is assigned to PE " << neuron_it->second.pe_id;
                errors.push_back(oss.str());
            }
        }
    }
    
    return errors;
}

// === 辅助方法 ===

void MappingSolution::addNeuronToPE(NeuronId neuron_id, PEId pe_id) {
    pe_to_neurons_[pe_id].push_back(neuron_id);
}

void MappingSolution::removeNeuronFromPE(NeuronId neuron_id, PEId pe_id) {
    auto pe_it = pe_to_neurons_.find(pe_id);
    if (pe_it != pe_to_neurons_.end()) {
        auto& neurons = pe_it->second;
        neurons.erase(std::remove(neurons.begin(), neurons.end(), neuron_id), neurons.end());
        
        // 如果PE没有神经元了，移除此PE
        if (neurons.empty()) {
            pe_to_neurons_.erase(pe_it);
        }
    }
}

void MappingSolution::updatePEMapping(NeuronId neuron_id, PEId old_pe_id, PEId new_pe_id) {
    removeNeuronFromPE(neuron_id, old_pe_id);
    addNeuronToPE(neuron_id, new_pe_id);
}

// === 私有方法 ===

float MappingSolution::computeObjectiveFunction(const NeuralNetwork& network,
                                              const HardwareTopology& topology,
                                              const MappingConfig& config) const {
    float comm_cost = calculateCommunicationCost(network, topology);
    float load_imbalance = calculateLoadImbalance(network);
    float inter_pe_ratio = calculateInterPECommunicationRatio(network);
    
    return config.communication_weight * comm_cost +
           config.load_balance_weight * load_imbalance +
           config.inter_pe_comm_weight * inter_pe_ratio;
}

std::pair<uint32_t, uint32_t> MappingSolution::countLocalAndRemoteConnections(const NeuralNetwork& network) const {
    uint32_t local_connections = 0;
    uint32_t remote_connections = 0;
    
    for (const auto& connection : network.getAllConnections()) {
        PEId src_pe = getNeuronPE(connection.source_id);
        PEId dst_pe = getNeuronPE(connection.target_id);
        
        if (src_pe != INVALID_PE_ID && dst_pe != INVALID_PE_ID) {
            if (src_pe == dst_pe) {
                local_connections++;
            } else {
                remote_connections++;
            }
        }
    }
    
    return std::make_pair(local_connections, remote_connections);
}

std::vector<float> MappingSolution::calculatePELoadDistribution(const NeuralNetwork& network) const {
    std::vector<float> loads;
    loads.reserve(pe_to_neurons_.size());
    
    for (const auto& pe_neurons_pair : pe_to_neurons_) {
        float load = calculatePELoad(pe_neurons_pair.first, network);
        loads.push_back(load);
    }
    
    return loads;
}

} // namespace NeuronMapping