#ifndef NEURON_MAPPING_FRAMEWORK_EVALUATORS_COMMUNICATION_COST_EVALUATOR_H
#define NEURON_MAPPING_FRAMEWORK_EVALUATORS_COMMUNICATION_COST_EVALUATOR_H

#include "evaluators/PerformanceEvaluator.h"
#include <unordered_map>

namespace neuron_mapping {

using namespace NeuronMapping;

struct CommunicationWeights {
    float intra_pe_weight = 1.0f;       // PE内通信权重
    float inter_pe_weight = 10.0f;      // PE间通信权重
    float distance_weight = 5.0f;       // 距离权重
    float bandwidth_weight = 1.0f;      // 带宽权重
    float latency_weight = 1.0f;        // 延迟权重
};

class CommunicationCostEvaluator : public PerformanceEvaluator {
public:

    explicit CommunicationCostEvaluator(const CommunicationWeights& weights = CommunicationWeights());
    ~CommunicationCostEvaluator() override = default;

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

    void setWeights(const CommunicationWeights& weights);
    const CommunicationWeights& getWeights() const;

private:
    CommunicationWeights weights_;

    struct ConnectionInfo {
        NeuronId source_neuron;
        NeuronId target_neuron;
        PEId source_pe;
        PEId target_pe;
        float weight;
        float distance;
        bool is_inter_pe;
        float communication_cost;
    };

    float calculateTotalCommunicationCost(
        const MappingSolution& solution,
        const NeuralNetwork& network,
        const HardwareTopology& topology
    );

    std::vector<ConnectionInfo> analyzeConnections(
        const MappingSolution& solution,
        const NeuralNetwork& network,
        const HardwareTopology& topology
    );

    float calculateConnectionCost(
        const Connection& conn,
        PEId source_pe,
        PEId target_pe,
        const HardwareTopology& topology
    );

    float calculateDistance(
        PEId pe1, 
        PEId pe2, 
        const HardwareTopology& topology
    );

    float calculateCommunicationRatio(
        const std::vector<ConnectionInfo>& connections
    );

    float calculateAverageDistance(
        const std::vector<ConnectionInfo>& connections
    );

    float calculateMaxDistance(
        const std::vector<ConnectionInfo>& connections
    );
};

}

#endif