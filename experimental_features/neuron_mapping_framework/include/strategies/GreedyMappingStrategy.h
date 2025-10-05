#ifndef NEURON_MAPPING_FRAMEWORK_STRATEGIES_GREEDY_MAPPING_STRATEGY_H
#define NEURON_MAPPING_FRAMEWORK_STRATEGIES_GREEDY_MAPPING_STRATEGY_H

#include "strategies/MappingStrategy.h"
#include <vector>

namespace neuron_mapping {

using namespace NeuronMapping;

class GreedyMappingStrategy : public MappingStrategy {
public:
    GreedyMappingStrategy() = default;
    ~GreedyMappingStrategy() override = default;

    std::unique_ptr<MappingSolution> mapNetwork(
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config
    ) override;

    std::string getName() const override;
    std::string getDescription() const override;

private:
    struct NeuronScore {
        NeuronId neuron_id;
        double score;
        size_t connection_count;
        
        bool operator<(const NeuronScore& other) const {
            return score > other.score;
        }
    };

    std::vector<NeuronScore> calculateNeuronScores(const NeuralNetwork& network);
    
    PEId findBestPE(
        NeuronId neuron_id,
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingSolution& current_solution
    );

    double calculatePlacementCost(
        NeuronId neuron_id,
        PEId pe_id,
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingSolution& current_solution
    );
};

}

#endif