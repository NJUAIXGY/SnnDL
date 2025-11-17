#include "optimizers/LocalSearchOptimizer.h"
#include "utils/MathUtils.h"
#include <algorithm>
#include <limits>

namespace neuron_mapping {

LocalSearchOptimizer::LocalSearchOptimizer(NeighborType neighbor_type, uint32_t seed)
    : neighbor_type_(neighbor_type), seed_(seed), rng_(seed) {
    LOG_INFO("LocalSearchOptimizer initialized with neighbor type: " + 
             std::to_string(static_cast<int>(neighbor_type)) + ", seed: " + std::to_string(seed));
}

std::unique_ptr<MappingSolution> LocalSearchOptimizer::optimize(
    std::unique_ptr<MappingSolution> initial_solution,
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const OptimizationConfig& config) {
    
    if (!initial_solution) {
        LOG_ERROR("Null initial solution provided to LocalSearchOptimizer");
        return nullptr;
    }

    if (!validateInputs(*initial_solution, network, topology, config)) {
        LOG_ERROR("Input validation failed for local search optimization");
        return nullptr;
    }

    auto best_solution = std::make_unique<MappingSolution>(*initial_solution);
    auto current_solution = std::make_unique<MappingSolution>(*initial_solution);

    OptimizationStats stats;
    stats.initial_objective = evaluateObjective(*current_solution, network, topology, config);
    stats.best_objective = stats.initial_objective;

    LOG_INFO("Starting local search optimization with initial objective: " + 
             std::to_string(stats.initial_objective));

    for (uint32_t iter = 0; iter < config.max_iterations; ++iter) {
        stats.iterations = iter + 1;

        auto neighbor_solution = std::make_unique<MappingSolution>(*current_solution);

        NeighborType selected_type = neighbor_type_;
        if (neighbor_type_ == NeighborType::MIXED) {
            std::uniform_int_distribution<int> type_dist(0, 3);
            selected_type = static_cast<NeighborType>(type_dist(rng_));
        }

        bool neighbor_generated = generateNeighbor(*neighbor_solution, network, topology, selected_type);
        if (!neighbor_generated) {
            stats.plateau_count++;
            continue;
        }

        double neighbor_objective = evaluateObjective(*neighbor_solution, network, topology, config);

        if (neighbor_objective < stats.best_objective) {
            best_solution = std::make_unique<MappingSolution>(*neighbor_solution);
            current_solution = std::move(neighbor_solution);
            stats.best_objective = neighbor_objective;
            stats.improvements++;
            stats.plateau_count = 0;

            if (config.enable_logging && stats.improvements % 10 == 0) {
                LOG_INFO("Improvement " + std::to_string(stats.improvements) + 
                        " at iteration " + std::to_string(iter) + 
                        ", objective: " + std::to_string(neighbor_objective));
            }
        } else {
            current_solution = std::move(neighbor_solution);
            stats.plateau_count++;
        }

        if (stats.plateau_count >= config.plateau_limit) {
            LOG_INFO("Optimization stopped due to plateau limit at iteration " + std::to_string(iter));
            break;
        }

        double improvement_ratio = std::abs(stats.best_objective - stats.initial_objective) / 
                                  std::max(std::abs(stats.initial_objective), 1e-10);
        if (improvement_ratio < config.convergence_threshold) {
            LOG_INFO("Optimization converged at iteration " + std::to_string(iter));
            break;
        }
    }

    logProgress(stats, config);
    return best_solution;
}

std::string LocalSearchOptimizer::getName() const {
    return "LocalSearchOptimizer";
}

std::string LocalSearchOptimizer::getDescription() const {
    return "Local search optimizer using various neighborhood operations";
}

void LocalSearchOptimizer::setNeighborType(NeighborType type) {
    neighbor_type_ = type;
}

void LocalSearchOptimizer::setSeed(uint32_t seed) {
    seed_ = seed;
    rng_.seed(seed);
}

bool LocalSearchOptimizer::generateNeighbor(MappingSolution& solution,
                                           const NeuralNetwork& network,
                                           const HardwareTopology& topology,
                                           NeighborType type) {
    switch (type) {
        case NeighborType::RANDOM_SWAP:
            return performRandomSwap(solution, topology);
        case NeighborType::RANDOM_MOVE:
            return performRandomMove(solution, topology);
        case NeighborType::GREEDY_SWAP:
            return performGreedySwap(solution, network, topology);
        case NeighborType::GREEDY_MOVE:
            return performGreedyMove(solution, network, topology);
        default:
            return performRandomSwap(solution, topology);
    }
}

bool LocalSearchOptimizer::performRandomSwap(MappingSolution& solution,
                                           const HardwareTopology& topology) {
    auto assignments = solution.getAllAssignments();
    if (assignments.size() < 2) {
        return false;
    }

    std::uniform_int_distribution<size_t> dist(0, assignments.size() - 1);
    size_t idx1 = dist(rng_);
    size_t idx2 = dist(rng_);

    if (idx1 == idx2) {
        idx2 = (idx2 + 1) % assignments.size();
    }

    NeuronId neuron1 = assignments[idx1].neuron_id;
    NeuronId neuron2 = assignments[idx2].neuron_id;

    return solution.swapNeuronAssignments(neuron1, neuron2);
}

bool LocalSearchOptimizer::performRandomMove(MappingSolution& solution,
                                           const HardwareTopology& topology) {
    auto assignments = solution.getAllAssignments();
    if (assignments.empty()) {
        return false;
    }

    std::uniform_int_distribution<size_t> neuron_dist(0, assignments.size() - 1);
    std::uniform_int_distribution<PEId> pe_dist(0, topology.getTotalPEs() - 1);

    size_t neuron_idx = neuron_dist(rng_);
    NeuronId neuron_id = assignments[neuron_idx].neuron_id;
    PEId current_pe = assignments[neuron_idx].pe_id;

    PEId target_pe;
    int attempts = 0;
    do {
        target_pe = pe_dist(rng_);
        attempts++;
    } while (target_pe == current_pe && attempts < 10);

    if (target_pe == current_pe || !isValidMove(neuron_id, target_pe, solution, topology)) {
        return false;
    }

    return solution.reassignNeuron(neuron_id, target_pe, 0);
}

bool LocalSearchOptimizer::performGreedySwap(MappingSolution& solution,
                                           const NeuralNetwork& network,
                                           const HardwareTopology& topology) {
    return performRandomSwap(solution, topology);
}

bool LocalSearchOptimizer::performGreedyMove(MappingSolution& solution,
                                           const NeuralNetwork& network,
                                           const HardwareTopology& topology) {
    return performRandomMove(solution, topology);
}

std::pair<NeuronId, NeuronId> LocalSearchOptimizer::findBestSwapPair(
    const MappingSolution& solution,
    const NeuralNetwork& network,
    const OptimizationConfig& config) {
    return std::make_pair(INVALID_NEURON_ID, INVALID_NEURON_ID);
}

NeuronId LocalSearchOptimizer::findBestMoveNeuron(const MappingSolution& solution,
                                                 const NeuralNetwork& network,
                                                 const HardwareTopology& topology,
                                                 const OptimizationConfig& config) {
    return INVALID_NEURON_ID;
}

PEId LocalSearchOptimizer::findBestTargetPE(NeuronId neuron_id,
                                           const MappingSolution& solution,
                                           const NeuralNetwork& network,
                                           const HardwareTopology& topology,
                                           const OptimizationConfig& config) {
    return INVALID_PE_ID;
}

bool LocalSearchOptimizer::isValidMove(NeuronId neuron_id, 
                                      PEId target_pe,
                                      const MappingSolution& solution,
                                      const HardwareTopology& topology) const {
    if (target_pe >= topology.getTotalPEs()) {
        return false;
    }

    uint32_t target_pe_load = solution.getPENeuronCount(target_pe);
    uint32_t target_pe_capacity = topology.getPECapacity(target_pe);

    PEId current_pe = solution.getNeuronPE(neuron_id);
    if (current_pe == target_pe) {
        return false;
    }

    if (current_pe != INVALID_PE_ID && current_pe == target_pe) {
        return true;
    }

    return target_pe_load < target_pe_capacity;
}

void LocalSearchOptimizer::logProgress(const OptimizationStats& stats,
                                     const OptimizationConfig& config) const {
    if (!config.enable_logging) {
        return;
    }

    double improvement_percentage = 0.0;
    if (stats.initial_objective > 0) {
        improvement_percentage = ((stats.initial_objective - stats.best_objective) / 
                                 stats.initial_objective) * 100.0;
    }

    LOG_INFO("Local search optimization completed:");
    LOG_INFO("  Total iterations: " + std::to_string(stats.iterations));
    LOG_INFO("  Improvements found: " + std::to_string(stats.improvements));
    LOG_INFO("  Initial objective: " + std::to_string(stats.initial_objective));
    LOG_INFO("  Final objective: " + std::to_string(stats.best_objective));
    LOG_INFO("  Improvement: " + std::to_string(improvement_percentage) + "%");
}

}