#ifndef NEURON_MAPPING_FRAMEWORK_OPTIMIZERS_MAPPING_OPTIMIZER_H
#define NEURON_MAPPING_FRAMEWORK_OPTIMIZERS_MAPPING_OPTIMIZER_H

#include "core/Types.h"
#include "core/NeuralNetwork.h"
#include "core/HardwareTopology.h"
#include "core/MappingSolution.h"
#include "utils/Logger.h"
#include <memory>
#include <functional>

namespace neuron_mapping {

using namespace NeuronMapping;

struct OptimizationConfig {
    uint32_t max_iterations = 1000;
    double convergence_threshold = 1e-6;
    double initial_temperature = 100.0;
    double cooling_rate = 0.95;
    uint32_t plateau_limit = 50;
    bool enable_logging = true;
    
    std::function<double(const MappingSolution&)> objective_function = nullptr;
};

class MappingOptimizer {
public:
    MappingOptimizer() = default;
    virtual ~MappingOptimizer() = default;

    virtual std::unique_ptr<MappingSolution> optimize(
        std::unique_ptr<MappingSolution> initial_solution,
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const OptimizationConfig& config
    ) = 0;

    virtual std::string getName() const = 0;
    virtual std::string getDescription() const = 0;

    MappingOptimizer(const MappingOptimizer&) = delete;
    MappingOptimizer& operator=(const MappingOptimizer&) = delete;
    MappingOptimizer(MappingOptimizer&&) = default;
    MappingOptimizer& operator=(MappingOptimizer&&) = default;

protected:
    bool validateInputs(const MappingSolution& solution, 
                       const NeuralNetwork& network,
                       const HardwareTopology& topology,
                       const OptimizationConfig& config) const {
        if (solution.getAssignedNeuronCount() == 0) {
            LOG_ERROR("Empty mapping solution provided for optimization");
            return false;
        }

        if (network.getNeuronCount() == 0) {
            LOG_ERROR("Empty neural network provided for optimization");
            return false;
        }

        if (topology.getTotalPEs() == 0) {
            LOG_ERROR("Hardware topology has no processing elements");
            return false;
        }

        if (config.max_iterations == 0) {
            LOG_ERROR("Invalid optimization configuration: max_iterations must be > 0");
            return false;
        }

        return true;
    }

    double evaluateObjective(const MappingSolution& solution,
                           const NeuralNetwork& network,
                           const HardwareTopology& topology,
                           const OptimizationConfig& config) const {
        if (config.objective_function) {
            return config.objective_function(solution);
        }
        
        MappingConfig default_config;
        auto metrics = solution.evaluatePerformance(network, topology, default_config);
        return metrics.communication_cost + metrics.load_imbalance * 0.5;
    }
};

}

#endif