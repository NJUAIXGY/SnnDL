#ifndef NEURON_MAPPING_FRAMEWORK_EVALUATORS_PERFORMANCE_EVALUATOR_H
#define NEURON_MAPPING_FRAMEWORK_EVALUATORS_PERFORMANCE_EVALUATOR_H

#include "core/Types.h"
#include "core/NeuralNetwork.h"
#include "core/HardwareTopology.h"
#include "core/MappingSolution.h"
#include "utils/Logger.h"
#include <memory>
#include <vector>
#include <map>

namespace neuron_mapping {

using namespace NeuronMapping;

struct DetailedMetrics {
    // 通信成本细节
    float total_communication_cost = 0.0f;
    float intra_pe_communication_cost = 0.0f;
    float inter_pe_communication_cost = 0.0f;
    float average_communication_distance = 0.0f;
    float max_communication_distance = 0.0f;
    uint32_t total_connections = 0;
    uint32_t inter_pe_connections = 0;
    
    // 负载均衡细节
    float max_pe_load = 0.0f;
    float min_pe_load = 0.0f;
    float average_pe_load = 0.0f;
    float load_standard_deviation = 0.0f;
    float load_coefficient_of_variation = 0.0f;
    std::vector<float> pe_loads;
    
    // 内存使用细节
    float max_pe_memory_usage = 0.0f;
    float min_pe_memory_usage = 0.0f;
    float average_pe_memory_usage = 0.0f;
    float memory_standard_deviation = 0.0f;
    std::vector<float> pe_memory_usage;
    
    // 连通性分析
    uint32_t isolated_pes = 0;
    uint32_t utilized_pes = 0;
    float pe_utilization_ratio = 0.0f;
    
    void clear() {
        total_communication_cost = 0.0f;
        intra_pe_communication_cost = 0.0f;
        inter_pe_communication_cost = 0.0f;
        average_communication_distance = 0.0f;
        max_communication_distance = 0.0f;
        total_connections = 0;
        inter_pe_connections = 0;
        max_pe_load = 0.0f;
        min_pe_load = 0.0f;
        average_pe_load = 0.0f;
        load_standard_deviation = 0.0f;
        load_coefficient_of_variation = 0.0f;
        pe_loads.clear();
        max_pe_memory_usage = 0.0f;
        min_pe_memory_usage = 0.0f;
        average_pe_memory_usage = 0.0f;
        memory_standard_deviation = 0.0f;
        pe_memory_usage.clear();
        isolated_pes = 0;
        utilized_pes = 0;
        pe_utilization_ratio = 0.0f;
    }
};

class PerformanceEvaluator {
public:
    PerformanceEvaluator() = default;
    virtual ~PerformanceEvaluator() = default;

    virtual PerformanceMetrics evaluateBasic(
        const MappingSolution& solution,
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config
    ) = 0;

    virtual DetailedMetrics evaluateDetailed(
        const MappingSolution& solution,
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config
    ) = 0;

    virtual std::string getName() const = 0;
    virtual std::string getDescription() const = 0;

    PerformanceEvaluator(const PerformanceEvaluator&) = delete;
    PerformanceEvaluator& operator=(const PerformanceEvaluator&) = delete;
    PerformanceEvaluator(PerformanceEvaluator&&) = default;
    PerformanceEvaluator& operator=(PerformanceEvaluator&&) = default;

protected:
    bool validateInputs(const MappingSolution& solution, 
                       const NeuralNetwork& network,
                       const HardwareTopology& topology) const {
        if (solution.getAssignedNeuronCount() == 0) {
            LOG_ERROR("Empty mapping solution provided for evaluation");
            return false;
        }

        if (network.getNeuronCount() == 0) {
            LOG_ERROR("Empty neural network provided for evaluation");
            return false;
        }

        if (topology.getTotalPEs() == 0) {
            LOG_ERROR("Hardware topology has no processing elements");
            return false;
        }

        return true;
    }
};

}

#endif