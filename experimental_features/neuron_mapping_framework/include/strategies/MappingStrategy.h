#ifndef NEURON_MAPPING_FRAMEWORK_STRATEGIES_MAPPING_STRATEGY_H
#define NEURON_MAPPING_FRAMEWORK_STRATEGIES_MAPPING_STRATEGY_H

#include "core/Types.h"
#include "core/NeuralNetwork.h"
#include "core/HardwareTopology.h"
#include "core/MappingSolution.h"
#include "utils/Logger.h"
#include <memory>

namespace neuron_mapping {

using namespace NeuronMapping;

class MappingStrategy {
public:
    MappingStrategy() = default;
    virtual ~MappingStrategy() = default;

    virtual std::unique_ptr<MappingSolution> mapNetwork(
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config
    ) = 0;

    virtual std::string getName() const = 0;
    virtual std::string getDescription() const = 0;

    MappingStrategy(const MappingStrategy&) = delete;
    MappingStrategy& operator=(const MappingStrategy&) = delete;
    MappingStrategy(MappingStrategy&&) = default;
    MappingStrategy& operator=(MappingStrategy&&) = default;

protected:
    bool validateInputs(const NeuralNetwork& network, 
                       const HardwareTopology& topology,
                       const MappingConfig& config) const {
        if (network.getNeuronCount() == 0) {
            LOG_ERROR("Empty neural network provided for mapping");
            return false;
        }

        if (topology.getTotalPEs() == 0) {
            LOG_ERROR("Hardware topology has no processing elements");
            return false;
        }

        uint32_t total_capacity = 0;
        for (PEId pe_id = 0; pe_id < topology.getTotalPEs(); ++pe_id) {
            total_capacity += topology.getPECapacity(pe_id);
        }

        if (total_capacity < network.getNeuronCount()) {
            LOG_ERROR("Insufficient hardware capacity for neural network mapping");
            return false;
        }

        return true;
    }
};

}

#endif