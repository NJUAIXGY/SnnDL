#include "evaluators/CommunicationCostEvaluator.h"
#include "utils/MathUtils.h"
#include <algorithm>
#include <cmath>

namespace neuron_mapping {

CommunicationCostEvaluator::CommunicationCostEvaluator(const CommunicationWeights& weights)
    : weights_(weights) {
    LOG_INFO("CommunicationCostEvaluator initialized");
}

PerformanceMetrics CommunicationCostEvaluator::evaluateBasic(
    const MappingSolution& solution,
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const MappingConfig& config) {
    
    PerformanceMetrics metrics;
    
    if (!validateInputs(solution, network, topology)) {
        LOG_ERROR("Input validation failed for communication cost evaluation");
        return metrics;
    }

    auto connections = analyzeConnections(solution, network, topology);
    
    // 计算总通信成本
    metrics.communication_cost = calculateTotalCommunicationCost(solution, network, topology);
    
    // 计算通信比例
    metrics.inter_pe_communication_ratio = calculateCommunicationRatio(connections);
    
    // 计算平均通信距离
    metrics.average_communication_distance = calculateAverageDistance(connections);
    
    return metrics;
}

DetailedMetrics CommunicationCostEvaluator::evaluateDetailed(
    const MappingSolution& solution,
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const MappingConfig& config) {
    
    DetailedMetrics metrics;
    
    if (!validateInputs(solution, network, topology)) {
        LOG_ERROR("Input validation failed for detailed communication cost evaluation");
        return metrics;
    }

    auto connections = analyzeConnections(solution, network, topology);
    
    // 通信成本统计
    metrics.total_communication_cost = 0.0f;
    metrics.intra_pe_communication_cost = 0.0f;
    metrics.inter_pe_communication_cost = 0.0f;
    metrics.total_connections = connections.size();
    metrics.inter_pe_connections = 0;
    
    float max_distance = 0.0f;
    float total_distance = 0.0f;
    uint32_t distance_count = 0;
    
    for (const auto& conn : connections) {
        metrics.total_communication_cost += conn.communication_cost;
        
        if (conn.is_inter_pe) {
            metrics.inter_pe_communication_cost += conn.communication_cost;
            metrics.inter_pe_connections++;
            total_distance += conn.distance;
            distance_count++;
            max_distance = std::max(max_distance, conn.distance);
        } else {
            metrics.intra_pe_communication_cost += conn.communication_cost;
        }
    }
    
    // 计算平均和最大距离
    metrics.average_communication_distance = distance_count > 0 ? 
        total_distance / distance_count : 0.0f;
    metrics.max_communication_distance = max_distance;
    
    LOG_INFO("Communication cost evaluation completed:");
    LOG_INFO("  Total cost: " + std::to_string(metrics.total_communication_cost));
    LOG_INFO("  Inter-PE connections: " + std::to_string(metrics.inter_pe_connections) + 
             "/" + std::to_string(metrics.total_connections));
    LOG_INFO("  Average distance: " + std::to_string(metrics.average_communication_distance));
    
    return metrics;
}

std::string CommunicationCostEvaluator::getName() const {
    return "CommunicationCostEvaluator";
}

std::string CommunicationCostEvaluator::getDescription() const {
    return "Evaluates communication cost based on connection weights and PE distances";
}

void CommunicationCostEvaluator::setWeights(const CommunicationWeights& weights) {
    weights_ = weights;
}

const CommunicationWeights& 
CommunicationCostEvaluator::getWeights() const {
    return weights_;
}

float CommunicationCostEvaluator::calculateTotalCommunicationCost(
    const MappingSolution& solution,
    const NeuralNetwork& network,
    const HardwareTopology& topology) {
    
    float total_cost = 0.0f;
    auto connections = network.getAllConnections();
    
    for (const auto& conn : connections) {
        PEId source_pe = solution.getNeuronPE(conn.source_id);
        PEId target_pe = solution.getNeuronPE(conn.target_id);
        
        if (source_pe == INVALID_PE_ID || target_pe == INVALID_PE_ID) {
            continue;
        }
        
        float cost = calculateConnectionCost(conn, source_pe, target_pe, topology);
        total_cost += cost;
    }
    
    return total_cost;
}

std::vector<CommunicationCostEvaluator::ConnectionInfo> 
CommunicationCostEvaluator::analyzeConnections(
    const MappingSolution& solution,
    const NeuralNetwork& network,
    const HardwareTopology& topology) {
    
    std::vector<ConnectionInfo> connection_infos;
    auto connections = network.getAllConnections();
    
    for (const auto& conn : connections) {
        PEId source_pe = solution.getNeuronPE(conn.source_id);
        PEId target_pe = solution.getNeuronPE(conn.target_id);
        
        if (source_pe == INVALID_PE_ID || target_pe == INVALID_PE_ID) {
            continue;
        }
        
        ConnectionInfo info;
        info.source_neuron = conn.source_id;
        info.target_neuron = conn.target_id;
        info.source_pe = source_pe;
        info.target_pe = target_pe;
        info.weight = conn.weight;
        info.distance = calculateDistance(source_pe, target_pe, topology);
        info.is_inter_pe = (source_pe != target_pe);
        info.communication_cost = calculateConnectionCost(conn, source_pe, target_pe, topology);
        
        connection_infos.push_back(info);
    }
    
    return connection_infos;
}

float CommunicationCostEvaluator::calculateConnectionCost(
    const Connection& conn,
    PEId source_pe,
    PEId target_pe,
    const HardwareTopology& topology) {
    
    float weight_factor = std::abs(conn.weight);
    
    if (source_pe == target_pe) {
        // PE内通信
        return weight_factor * weights_.intra_pe_weight;
    } else {
        // PE间通信
        float distance = calculateDistance(source_pe, target_pe, topology);
        float distance_cost = distance * weights_.distance_weight;
        float inter_pe_cost = weights_.inter_pe_weight;
        
        return weight_factor * (inter_pe_cost + distance_cost);
    }
}

float CommunicationCostEvaluator::calculateDistance(
    PEId pe1, 
    PEId pe2, 
    const HardwareTopology& topology) {
    
    if (pe1 == pe2) {
        return 0.0f;
    }
    
    // 简化距离计算 - 在实际实现中应该调用topology的距离计算方法
    // 这里使用PE ID差值作为简单的距离度量
    return std::abs(static_cast<float>(pe1) - static_cast<float>(pe2));
}

float CommunicationCostEvaluator::calculateCommunicationRatio(
    const std::vector<ConnectionInfo>& connections) {
    
    if (connections.empty()) {
        return 0.0f;
    }
    
    uint32_t inter_pe_count = 0;
    for (const auto& conn : connections) {
        if (conn.is_inter_pe) {
            inter_pe_count++;
        }
    }
    
    return static_cast<float>(inter_pe_count) / connections.size();
}

float CommunicationCostEvaluator::calculateAverageDistance(
    const std::vector<ConnectionInfo>& connections) {
    
    float total_distance = 0.0f;
    uint32_t count = 0;
    
    for (const auto& conn : connections) {
        if (conn.is_inter_pe) {
            total_distance += conn.distance;
            count++;
        }
    }
    
    return count > 0 ? total_distance / count : 0.0f;
}

float CommunicationCostEvaluator::calculateMaxDistance(
    const std::vector<ConnectionInfo>& connections) {
    
    float max_distance = 0.0f;
    
    for (const auto& conn : connections) {
        if (conn.is_inter_pe) {
            max_distance = std::max(max_distance, conn.distance);
        }
    }
    
    return max_distance;
}

}