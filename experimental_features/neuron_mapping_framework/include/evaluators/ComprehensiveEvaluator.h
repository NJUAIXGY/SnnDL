#ifndef NEURON_MAPPING_FRAMEWORK_EVALUATORS_COMPREHENSIVE_EVALUATOR_H
#define NEURON_MAPPING_FRAMEWORK_EVALUATORS_COMPREHENSIVE_EVALUATOR_H

#include "evaluators/PerformanceEvaluator.h"
#include "evaluators/CommunicationCostEvaluator.h"
#include "evaluators/LoadBalanceEvaluator.h"
#include <memory>

namespace neuron_mapping {

using namespace NeuronMapping;

struct ComprehensiveWeights {
    float communication_weight = 0.6f;
    float load_balance_weight = 0.4f;
    float memory_weight = 0.2f;
    float utilization_weight = 0.1f;
};

class ComprehensiveEvaluator : public PerformanceEvaluator {
public:
    explicit ComprehensiveEvaluator(const ComprehensiveWeights& weights = ComprehensiveWeights(),
                                   const CommunicationWeights& comm_weights = CommunicationWeights(),
                                   const LoadWeights& load_weights = LoadWeights());
    ~ComprehensiveEvaluator() override = default;

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

    void setWeights(const ComprehensiveWeights& weights);
    const ComprehensiveWeights& getWeights() const;

    CommunicationCostEvaluator& getCommunicationEvaluator();
    LoadBalanceEvaluator& getLoadBalanceEvaluator();

private:
    ComprehensiveWeights weights_;
    std::unique_ptr<CommunicationCostEvaluator> comm_evaluator_;
    std::unique_ptr<LoadBalanceEvaluator> load_evaluator_;

    float calculateOverallScore(const PerformanceMetrics& metrics);
    float calculateObjectiveValue(const DetailedMetrics& detailed_metrics);
};

}

#endif