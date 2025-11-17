#include "evaluators/LoadBalanceEvaluator.h"
#include "utils/MathUtils.h"
#include <algorithm>
#include <cmath>
#include <numeric>

namespace neuron_mapping {

LoadBalanceEvaluator::LoadBalanceEvaluator(LoadMetric metric, const LoadWeights& weights)
    : load_metric_(metric), load_weights_(weights) {
    LOG_INFO("LoadBalanceEvaluator initialized with metric: " + 
             std::to_string(static_cast<int>(metric)));
}

PerformanceMetrics LoadBalanceEvaluator::evaluateBasic(
    const MappingSolution& solution,
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const MappingConfig& config) {
    
    PerformanceMetrics metrics;
    
    if (!validateInputs(solution, network, topology)) {
        LOG_ERROR("Input validation failed for load balance evaluation");
        return metrics;
    }

    auto pe_loads = calculatePELoads(solution, network, topology);
    
    std::vector<float> loads;
    loads.reserve(pe_loads.size());
    for (const auto& pe_load : pe_loads) {
        loads.push_back(pe_load.total_load);
    }
    
    // 计算负载不均衡指标
    metrics.load_imbalance = calculateLoadImbalance(loads);
    metrics.load_imbalance_factor = metrics.load_imbalance;  // 别名
    metrics.max_min_load_ratio = calculateMaxMinRatio(loads);
    metrics.load_variance = calculateLoadVariance(loads);
    
    return metrics;
}

DetailedMetrics LoadBalanceEvaluator::evaluateDetailed(
    const MappingSolution& solution,
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const MappingConfig& config) {
    
    DetailedMetrics metrics;
    
    if (!validateInputs(solution, network, topology)) {
        LOG_ERROR("Input validation failed for detailed load balance evaluation");
        return metrics;
    }

    auto pe_loads = calculatePELoads(solution, network, topology);
    
    // 收集负载数据
    std::vector<float> loads;
    loads.reserve(pe_loads.size());
    for (const auto& pe_load : pe_loads) {
        loads.push_back(pe_load.total_load);
        metrics.pe_loads.push_back(pe_load.total_load);
    }
    
    // 计算基本统计量
    if (!loads.empty()) {
        metrics.max_pe_load = *std::max_element(loads.begin(), loads.end());
        metrics.min_pe_load = *std::min_element(loads.begin(), loads.end());
        metrics.average_pe_load = std::accumulate(loads.begin(), loads.end(), 0.0f) / loads.size();
        metrics.load_standard_deviation = calculateStandardDeviation(loads);
        metrics.load_coefficient_of_variation = calculateCoefficientOfVariation(loads);
    }
    
    // 收集内存使用数据
    for (const auto& pe_load : pe_loads) {
        metrics.pe_memory_usage.push_back(pe_load.memory_usage);
    }
    
    if (!metrics.pe_memory_usage.empty()) {
        metrics.max_pe_memory_usage = *std::max_element(metrics.pe_memory_usage.begin(), 
                                                       metrics.pe_memory_usage.end());
        metrics.min_pe_memory_usage = *std::min_element(metrics.pe_memory_usage.begin(), 
                                                       metrics.pe_memory_usage.end());
        metrics.average_pe_memory_usage = std::accumulate(metrics.pe_memory_usage.begin(), 
                                                         metrics.pe_memory_usage.end(), 0.0f) / 
                                         metrics.pe_memory_usage.size();
        metrics.memory_standard_deviation = calculateStandardDeviation(metrics.pe_memory_usage);
    }
    
    // 计算PE利用情况
    metrics.utilized_pes = countUtilizedPEs(pe_loads);
    metrics.isolated_pes = countIsolatedPEs(pe_loads);
    metrics.pe_utilization_ratio = static_cast<float>(metrics.utilized_pes) / topology.getTotalPEs();
    
    LOG_INFO("Load balance evaluation completed:");
    LOG_INFO("  Utilized PEs: " + std::to_string(metrics.utilized_pes) + 
             "/" + std::to_string(topology.getTotalPEs()));
    LOG_INFO("  Average load: " + std::to_string(metrics.average_pe_load));
    LOG_INFO("  Load std dev: " + std::to_string(metrics.load_standard_deviation));
    
    return metrics;
}

std::string LoadBalanceEvaluator::getName() const {
    return "LoadBalanceEvaluator";
}

std::string LoadBalanceEvaluator::getDescription() const {
    return "Evaluates load balance across processing elements using various metrics";
}

void LoadBalanceEvaluator::setLoadMetric(LoadMetric metric) {
    load_metric_ = metric;
}

void LoadBalanceEvaluator::setLoadWeights(const LoadWeights& weights) {
    load_weights_ = weights;
}

LoadMetric LoadBalanceEvaluator::getLoadMetric() const {
    return load_metric_;
}

const LoadWeights& LoadBalanceEvaluator::getLoadWeights() const {
    return load_weights_;
}

std::vector<LoadBalanceEvaluator::PELoadInfo> LoadBalanceEvaluator::calculatePELoads(
    const MappingSolution& solution,
    const NeuralNetwork& network,
    const HardwareTopology& topology) {
    
    std::vector<PELoadInfo> pe_loads(topology.getTotalPEs());
    
    // 初始化PE信息
    for (PEId pe_id = 0; pe_id < topology.getTotalPEs(); ++pe_id) {
        pe_loads[pe_id].pe_id = pe_id;
        pe_loads[pe_id].neuron_count = solution.getPENeuronCount(pe_id);
    }
    
    // 计算连接数和计算负载
    auto connections = network.getAllConnections();
    for (const auto& conn : connections) {
        PEId source_pe = solution.getNeuronPE(conn.source_id);
        PEId target_pe = solution.getNeuronPE(conn.target_id);
        
        if (source_pe != INVALID_PE_ID) {
            pe_loads[source_pe].outgoing_connections++;
            pe_loads[source_pe].computational_load += std::abs(conn.weight) * 0.1f;
        }
        
        if (target_pe != INVALID_PE_ID) {
            pe_loads[target_pe].incoming_connections++;
            pe_loads[target_pe].computational_load += std::abs(conn.weight) * 0.1f;
        }
    }
    
    // 估算内存使用
    for (auto& pe_load : pe_loads) {
        pe_load.memory_usage = pe_load.neuron_count * 1024.0f +  // 每个神经元1KB
                              (pe_load.incoming_connections + pe_load.outgoing_connections) * 64.0f; // 每个连接64B
        
        // 计算容量利用率
        uint32_t capacity = topology.getPECapacity(pe_load.pe_id);
        pe_load.capacity_utilization = capacity > 0 ? 
            static_cast<float>(pe_load.neuron_count) / capacity : 0.0f;
    }
    
    // 根据负载度量类型计算总负载
    for (auto& pe_load : pe_loads) {
        switch (load_metric_) {
            case LoadMetric::NEURON_COUNT:
                pe_load.total_load = pe_load.neuron_count * load_weights_.neuron_count_weight;
                break;
            case LoadMetric::COMPUTATIONAL_LOAD:
                pe_load.total_load = pe_load.computational_load * load_weights_.computational_weight;
                break;
            case LoadMetric::MEMORY_USAGE:
                pe_load.total_load = pe_load.memory_usage * load_weights_.memory_weight;
                break;
            case LoadMetric::COMBINED:
            default:
                pe_load.total_load = pe_load.neuron_count * load_weights_.neuron_count_weight +
                                   pe_load.computational_load * load_weights_.computational_weight +
                                   pe_load.memory_usage * load_weights_.memory_weight +
                                   (pe_load.incoming_connections + pe_load.outgoing_connections) * 
                                   load_weights_.connection_weight;
                break;
        }
    }
    
    return pe_loads;
}

float LoadBalanceEvaluator::calculateLoadImbalance(const std::vector<float>& loads) {
    if (loads.empty()) {
        return 0.0f;
    }
    
    float max_load = *std::max_element(loads.begin(), loads.end());
    float min_load = *std::min_element(loads.begin(), loads.end());
    float avg_load = std::accumulate(loads.begin(), loads.end(), 0.0f) / loads.size();
    
    if (avg_load <= 0.0f) {
        return 0.0f;
    }
    
    // 使用标准差与平均值的比值作为不均衡度量
    float std_dev = calculateStandardDeviation(loads);
    return std_dev / avg_load;
}

float LoadBalanceEvaluator::calculateLoadVariance(const std::vector<float>& loads) {
    if (loads.size() <= 1) {
        return 0.0f;
    }
    
    float mean = std::accumulate(loads.begin(), loads.end(), 0.0f) / loads.size();
    float variance = 0.0f;
    
    for (float load : loads) {
        float diff = load - mean;
        variance += diff * diff;
    }
    
    return variance / (loads.size() - 1);
}

float LoadBalanceEvaluator::calculateMaxMinRatio(const std::vector<float>& loads) {
    if (loads.empty()) {
        return 1.0f;
    }
    
    float max_load = *std::max_element(loads.begin(), loads.end());
    float min_load = *std::min_element(loads.begin(), loads.end());
    
    return min_load > 0.0f ? max_load / min_load : 
           (max_load > 0.0f ? std::numeric_limits<float>::max() : 1.0f);
}

float LoadBalanceEvaluator::calculateStandardDeviation(const std::vector<float>& loads) {
    return std::sqrt(calculateLoadVariance(loads));
}

float LoadBalanceEvaluator::calculateCoefficientOfVariation(const std::vector<float>& loads) {
    if (loads.empty()) {
        return 0.0f;
    }
    
    float mean = std::accumulate(loads.begin(), loads.end(), 0.0f) / loads.size();
    if (mean <= 0.0f) {
        return 0.0f;
    }
    
    float std_dev = calculateStandardDeviation(loads);
    return std_dev / mean;
}

uint32_t LoadBalanceEvaluator::countUtilizedPEs(const std::vector<PELoadInfo>& pe_loads) {
    uint32_t count = 0;
    for (const auto& pe_load : pe_loads) {
        if (pe_load.neuron_count > 0) {
            count++;
        }
    }
    return count;
}

uint32_t LoadBalanceEvaluator::countIsolatedPEs(const std::vector<PELoadInfo>& pe_loads) {
    uint32_t count = 0;
    for (const auto& pe_load : pe_loads) {
        if (pe_load.neuron_count == 0) {
            count++;
        }
    }
    return count;
}

}