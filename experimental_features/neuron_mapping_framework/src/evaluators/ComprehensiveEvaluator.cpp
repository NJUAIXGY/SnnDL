#include "evaluators/ComprehensiveEvaluator.h"
#include <algorithm>

namespace neuron_mapping {

ComprehensiveEvaluator::ComprehensiveEvaluator(const ComprehensiveWeights& weights,
                                             const CommunicationWeights& comm_weights,
                                             const LoadWeights& load_weights)
    : weights_(weights) {
    comm_evaluator_ = std::make_unique<CommunicationCostEvaluator>(comm_weights);
    load_evaluator_ = std::make_unique<LoadBalanceEvaluator>(LoadMetric::COMBINED, load_weights);
    
    LOG_INFO("ComprehensiveEvaluator initialized");
}

PerformanceMetrics ComprehensiveEvaluator::evaluateBasic(
    const MappingSolution& solution,
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const MappingConfig& config) {
    
    if (!validateInputs(solution, network, topology)) {
        LOG_ERROR("Input validation failed for comprehensive evaluation");
        return PerformanceMetrics();
    }

    // 获取通信成本指标
    auto comm_metrics = comm_evaluator_->evaluateBasic(solution, network, topology, config);
    
    // 获取负载均衡指标
    auto load_metrics = load_evaluator_->evaluateBasic(solution, network, topology, config);
    
    // 合并指标
    PerformanceMetrics metrics;
    metrics.communication_cost = comm_metrics.communication_cost;
    metrics.inter_pe_communication_ratio = comm_metrics.inter_pe_communication_ratio;
    metrics.average_communication_distance = comm_metrics.average_communication_distance;
    
    metrics.load_imbalance = load_metrics.load_imbalance;
    metrics.load_imbalance_factor = load_metrics.load_imbalance_factor;
    metrics.max_min_load_ratio = load_metrics.max_min_load_ratio;
    metrics.load_variance = load_metrics.load_variance;
    
    // 计算综合评分
    metrics.overall_score = calculateOverallScore(metrics);
    metrics.objective_value = metrics.communication_cost * weights_.communication_weight +
                             metrics.load_imbalance * weights_.load_balance_weight;
    
    // 计算PE利用率
    metrics.pe_utilization = static_cast<float>(solution.getAssignedNeuronCount()) / 
                           (topology.getTotalPEs() * std::max(1u, solution.getAssignedNeuronCount() / topology.getTotalPEs()));
    
    // 计算神经元覆盖率
    metrics.neuron_coverage = network.getNeuronCount() > 0 ? 
        static_cast<float>(solution.getAssignedNeuronCount()) / network.getNeuronCount() : 0.0f;
    
    return metrics;
}

DetailedMetrics ComprehensiveEvaluator::evaluateDetailed(
    const MappingSolution& solution,
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const MappingConfig& config) {
    
    if (!validateInputs(solution, network, topology)) {
        LOG_ERROR("Input validation failed for detailed comprehensive evaluation");
        return DetailedMetrics();
    }

    // 获取详细的通信成本指标
    auto comm_detailed = comm_evaluator_->evaluateDetailed(solution, network, topology, config);
    
    // 获取详细的负载均衡指标
    auto load_detailed = load_evaluator_->evaluateDetailed(solution, network, topology, config);
    
    // 创建综合详细指标
    DetailedMetrics metrics;
    
    // 复制通信指标
    metrics.total_communication_cost = comm_detailed.total_communication_cost;
    metrics.intra_pe_communication_cost = comm_detailed.intra_pe_communication_cost;
    metrics.inter_pe_communication_cost = comm_detailed.inter_pe_communication_cost;
    metrics.average_communication_distance = comm_detailed.average_communication_distance;
    metrics.max_communication_distance = comm_detailed.max_communication_distance;
    metrics.total_connections = comm_detailed.total_connections;
    metrics.inter_pe_connections = comm_detailed.inter_pe_connections;
    
    // 复制负载均衡指标
    metrics.max_pe_load = load_detailed.max_pe_load;
    metrics.min_pe_load = load_detailed.min_pe_load;
    metrics.average_pe_load = load_detailed.average_pe_load;
    metrics.load_standard_deviation = load_detailed.load_standard_deviation;
    metrics.load_coefficient_of_variation = load_detailed.load_coefficient_of_variation;
    metrics.pe_loads = load_detailed.pe_loads;
    
    // 复制内存使用指标
    metrics.max_pe_memory_usage = load_detailed.max_pe_memory_usage;
    metrics.min_pe_memory_usage = load_detailed.min_pe_memory_usage;
    metrics.average_pe_memory_usage = load_detailed.average_pe_memory_usage;
    metrics.memory_standard_deviation = load_detailed.memory_standard_deviation;
    metrics.pe_memory_usage = load_detailed.pe_memory_usage;
    
    // 复制连通性分析
    metrics.isolated_pes = load_detailed.isolated_pes;
    metrics.utilized_pes = load_detailed.utilized_pes;
    metrics.pe_utilization_ratio = load_detailed.pe_utilization_ratio;
    
    LOG_INFO("Comprehensive detailed evaluation completed:");
    LOG_INFO("  Communication cost: " + std::to_string(metrics.total_communication_cost));
    LOG_INFO("  Load balance (std dev): " + std::to_string(metrics.load_standard_deviation));
    LOG_INFO("  PE utilization: " + std::to_string(metrics.pe_utilization_ratio * 100.0f) + "%");
    
    return metrics;
}

std::string ComprehensiveEvaluator::getName() const {
    return "ComprehensiveEvaluator";
}

std::string ComprehensiveEvaluator::getDescription() const {
    return "Comprehensive evaluator combining communication cost and load balance analysis";
}

void ComprehensiveEvaluator::setWeights(const ComprehensiveWeights& weights) {
    weights_ = weights;
}

const ComprehensiveWeights& ComprehensiveEvaluator::getWeights() const {
    return weights_;
}

CommunicationCostEvaluator& ComprehensiveEvaluator::getCommunicationEvaluator() {
    return *comm_evaluator_;
}

LoadBalanceEvaluator& ComprehensiveEvaluator::getLoadBalanceEvaluator() {
    return *load_evaluator_;
}

float ComprehensiveEvaluator::calculateOverallScore(const PerformanceMetrics& metrics) {
    // 标准化各项指标（较低的值表示更好的性能）
    float normalized_comm_cost = std::min(1.0f, metrics.communication_cost / 1000.0f);
    float normalized_load_imbalance = std::min(1.0f, metrics.load_imbalance);
    float normalized_inter_pe_ratio = metrics.inter_pe_communication_ratio;
    
    // 计算综合评分（0-1之间，越低越好）
    float score = normalized_comm_cost * weights_.communication_weight +
                  normalized_load_imbalance * weights_.load_balance_weight +
                  normalized_inter_pe_ratio * weights_.utilization_weight;
    
    return score;
}

float ComprehensiveEvaluator::calculateObjectiveValue(const DetailedMetrics& detailed_metrics) {
    return detailed_metrics.total_communication_cost * weights_.communication_weight +
           detailed_metrics.load_standard_deviation * weights_.load_balance_weight +
           detailed_metrics.average_pe_memory_usage * weights_.memory_weight;
}

}