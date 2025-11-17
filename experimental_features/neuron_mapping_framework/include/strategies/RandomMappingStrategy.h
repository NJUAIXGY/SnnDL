#ifndef NEURON_MAPPING_FRAMEWORK_STRATEGIES_RANDOM_MAPPING_STRATEGY_H
#define NEURON_MAPPING_FRAMEWORK_STRATEGIES_RANDOM_MAPPING_STRATEGY_H

#include "strategies/MappingStrategy.h"
#include <random>

namespace neuron_mapping {

using namespace NeuronMapping;

class RandomMappingStrategy : public MappingStrategy {
public:
    explicit RandomMappingStrategy(uint32_t seed = std::random_device{}());
    ~RandomMappingStrategy() override = default;

    std::unique_ptr<MappingSolution> mapNetwork(
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config
    ) override;

    std::string getName() const override;
    std::string getDescription() const override;

    void setSeed(uint32_t seed);

private:
    std::mt19937 rng_;
    uint32_t seed_;

    void assignNeuronsRandomly(
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        MappingSolution& solution
    );
};

}

#endif