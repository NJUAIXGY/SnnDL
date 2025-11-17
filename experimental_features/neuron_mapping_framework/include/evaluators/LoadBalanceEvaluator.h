#ifndef NEURON_MAPPING_FRAMEWORK_EVALUATORS_LOAD_BALANCE_EVALUATOR_H
#define NEURON_MAPPING_FRAMEWORK_EVALUATORS_LOAD_BALANCE_EVALUATOR_H

#include "evaluators/PerformanceEvaluator.h"

namespace neuron_mapping {

using namespace NeuronMapping;

enum class LoadMetric {
    NEURON_COUNT,           // 基于神经元数量
    COMPUTATIONAL_LOAD,     // 基于计算负载
    MEMORY_USAGE,          // 基于内存使用
    COMBINED               // 综合度量
};

struct LoadWeights {
    float neuron_count_weight = 1.0f;
    float computational_weight = 2.0f;
    float memory_weight = 1.5f;
    float connection_weight = 1.0f;
};

class LoadBalanceEvaluator : public PerformanceEvaluator {
public:

    explicit LoadBalanceEvaluator(LoadMetric metric = LoadMetric::COMBINED,
                                 const LoadWeights& weights = LoadWeights());
    ~LoadBalanceEvaluator() override = default;

    PerformanceMetrics evaluateBasic(
        const MappingSolution& solution,
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config
    ) override;

    DetailedMetrics evaluateDetailed(
        const MappingSolution& solution,
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config
    ) override;

    std::string getName() const override;
    std::string getDescription() const override;

    void setLoadMetric(LoadMetric metric);
    void setLoadWeights(const LoadWeights& weights);
    LoadMetric getLoadMetric() const;
    const LoadWeights& getLoadWeights() const;

private:
    LoadMetric load_metric_;
    LoadWeights load_weights_;

    struct PELoadInfo {
        PEId pe_id;
        uint32_t neuron_count = 0;
        float computational_load = 0.0f;
        float memory_usage = 0.0f;
        uint32_t incoming_connections = 0;
        uint32_t outgoing_connections = 0;
        float total_load = 0.0f;
        float capacity_utilization = 0.0f;
    };

    std::vector<PELoadInfo> calculatePELoads(
        const MappingSolution& solution,
        const NeuralNetwork& network,
        const HardwareTopology& topology
    );

    float calculateNeuronCountLoad(
        const MappingSolution& solution,
        const HardwareTopology& topology
    );

    float calculateComputationalLoad(
        const MappingSolution& solution,
        const NeuralNetwork& network,
        const HardwareTopology& topology
    );

    float calculateMemoryLoad(
        const MappingSolution& solution,
        const NeuralNetwork& network,
        const HardwareTopology& topology
    );

    float calculateCombinedLoad(
        const std::vector<PELoadInfo>& pe_loads
    );

    float calculateLoadImbalance(
        const std::vector<float>& loads
    );

    float calculateLoadVariance(
        const std::vector<float>& loads
    );

    float calculateMaxMinRatio(
        const std::vector<float>& loads
    );

    float calculateStandardDeviation(
        const std::vector<float>& loads
    );

    float calculateCoefficientOfVariation(
        const std::vector<float>& loads
    );

    uint32_t countUtilizedPEs(
        const std::vector<PELoadInfo>& pe_loads
    );

    uint32_t countIsolatedPEs(
        const std::vector<PELoadInfo>& pe_loads
    );
};

}

#endif