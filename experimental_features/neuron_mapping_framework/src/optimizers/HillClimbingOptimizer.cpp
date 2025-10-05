#include "optimizers/HillClimbingOptimizer.h"
#include <algorithm>
#include <limits>

namespace neuron_mapping {

HillClimbingOptimizer::HillClimbingOptimizer(MoveType move_type, uint32_t seed)
    : move_type_(move_type), seed_(seed), rng_(seed) {
    LOG_INFO("HillClimbingOptimizer initialized with move type: " + 
             std::to_string(static_cast<int>(move_type)) + ", seed: " + std::to_string(seed));
}

std::unique_ptr<MappingSolution> HillClimbingOptimizer::optimize(
    std::unique_ptr<MappingSolution> initial_solution,
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const OptimizationConfig& config) {
    
    if (!initial_solution) {
        LOG_ERROR("Null initial solution provided to HillClimbingOptimizer");
        return nullptr;
    }

    if (!validateInputs(*initial_solution, network, topology, config)) {
        LOG_ERROR("Input validation failed for hill climbing optimization");
        return nullptr;
    }

    auto current_solution = std::make_unique<MappingSolution>(*initial_solution);
    double current_objective = evaluateObjective(*current_solution, network, topology, config);
    double best_objective = current_objective;

    LOG_INFO("Starting hill climbing optimization with initial objective: " + 
             std::to_string(current_objective));

    uint32_t improvements = 0;
    uint32_t restarts = 0;
    uint32_t plateau_count = 0;

    for (uint32_t iter = 0; iter < config.max_iterations; ++iter) {
        auto moves = generateAllMoves(*current_solution, network, topology, config);
        
        if (moves.empty()) {
            plateau_count++;
            if (plateau_count >= config.plateau_limit) {
                if (move_type_ == MoveType::RANDOM_RESTART && restarts < 5) {
                    if (performRandomRestart(*current_solution, topology)) {
                        current_objective = evaluateObjective(*current_solution, network, topology, config);
                        plateau_count = 0;
                        restarts++;
                        LOG_INFO("Random restart " + std::to_string(restarts) + 
                                " at iteration " + std::to_string(iter));
                        continue;
                    }
                }
                LOG_INFO("Hill climbing stopped due to plateau at iteration " + std::to_string(iter));
                break;
            }
            continue;
        }

        std::sort(moves.begin(), moves.end(), 
                 [](const Move& a, const Move& b) { return a.improvement > b.improvement; });

        bool improvement_found = false;
        
        if (move_type_ == MoveType::BEST_IMPROVEMENT) {
            if (moves[0].improvement > config.convergence_threshold) {
                applyMove(*current_solution, moves[0]);
                current_objective -= moves[0].improvement;
                improvement_found = true;
                improvements++;
            }
        } else if (move_type_ == MoveType::FIRST_IMPROVEMENT) {
            for (const auto& move : moves) {
                if (move.improvement > config.convergence_threshold) {
                    applyMove(*current_solution, move);
                    current_objective -= move.improvement;
                    improvement_found = true;
                    improvements++;
                    break;
                }
            }
        }

        if (improvement_found) {
            plateau_count = 0;
            if (current_objective < best_objective) {
                best_objective = current_objective;
            }
            
            if (config.enable_logging && improvements % 10 == 0) {
                LOG_INFO("Improvement " + std::to_string(improvements) + 
                        " at iteration " + std::to_string(iter) + 
                        ", objective: " + std::to_string(current_objective));
            }
        } else {
            plateau_count++;
        }
    }

    LOG_INFO("Hill climbing optimization completed:");
    LOG_INFO("  Total improvements: " + std::to_string(improvements));
    LOG_INFO("  Random restarts: " + std::to_string(restarts));
    LOG_INFO("  Final objective: " + std::to_string(current_objective));

    return current_solution;
}

std::string HillClimbingOptimizer::getName() const {
    return "HillClimbingOptimizer";
}

std::string HillClimbingOptimizer::getDescription() const {
    return "Hill climbing optimizer with best/first improvement and random restart options";
}

void HillClimbingOptimizer::setMoveType(MoveType type) {
    move_type_ = type;
}

void HillClimbingOptimizer::setSeed(uint32_t seed) {
    seed_ = seed;
    rng_.seed(seed);
}

std::vector<HillClimbingOptimizer::Move> HillClimbingOptimizer::generateAllMoves(
    const MappingSolution& solution,
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const OptimizationConfig& config) {
    
    std::vector<Move> all_moves;
    
    auto swap_moves = generateSwapMoves(solution, network, topology, config);
    all_moves.insert(all_moves.end(), swap_moves.begin(), swap_moves.end());
    
    auto relocate_moves = generateRelocateMoves(solution, network, topology, config);
    all_moves.insert(all_moves.end(), relocate_moves.begin(), relocate_moves.end());
    
    return all_moves;
}

std::vector<HillClimbingOptimizer::Move> HillClimbingOptimizer::generateSwapMoves(
    const MappingSolution& solution,
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const OptimizationConfig& config) {
    
    std::vector<Move> moves;
    auto assignments = solution.getAllAssignments();
    
    size_t max_swaps = std::min(static_cast<size_t>(100), assignments.size() * assignments.size() / 4);
    std::uniform_int_distribution<size_t> dist(0, assignments.size() - 1);
    
    for (size_t i = 0; i < max_swaps && moves.size() < 50; ++i) {
        size_t idx1 = dist(rng_);
        size_t idx2 = dist(rng_);
        
        if (idx1 == idx2) continue;
        
        NeuronId neuron1 = assignments[idx1].neuron_id;
        NeuronId neuron2 = assignments[idx2].neuron_id;
        
        Move move = Move::createSwap(neuron1, neuron2, 0.0);
        move.improvement = evaluateMove(solution, move, network, topology, config);
        
        if (move.improvement > 0) {
            moves.push_back(move);
        }
    }
    
    return moves;
}

std::vector<HillClimbingOptimizer::Move> HillClimbingOptimizer::generateRelocateMoves(
    const MappingSolution& solution,
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const OptimizationConfig& config) {
    
    std::vector<Move> moves;
    auto assignments = solution.getAllAssignments();
    
    size_t max_relocations = std::min(static_cast<size_t>(100), 
                                     assignments.size() * topology.getTotalPEs() / 4);
    std::uniform_int_distribution<size_t> neuron_dist(0, assignments.size() - 1);
    std::uniform_int_distribution<PEId> pe_dist(0, topology.getTotalPEs() - 1);
    
    for (size_t i = 0; i < max_relocations && moves.size() < 50; ++i) {
        size_t neuron_idx = neuron_dist(rng_);
        NeuronId neuron_id = assignments[neuron_idx].neuron_id;
        PEId current_pe = assignments[neuron_idx].pe_id;
        PEId target_pe = pe_dist(rng_);
        
        if (target_pe == current_pe) continue;
        if (solution.getPENeuronCount(target_pe) >= topology.getPECapacity(target_pe)) continue;
        
        Move move = Move::createRelocate(neuron_id, target_pe, 0.0);
        move.improvement = evaluateMove(solution, move, network, topology, config);
        
        if (move.improvement > 0) {
            moves.push_back(move);
        }
    }
    
    return moves;
}

bool HillClimbingOptimizer::applyMove(MappingSolution& solution, const Move& move) {
    if (move.type == Move::SWAP) {
        return solution.swapNeuronAssignments(move.neuron1, move.neuron2);
    } else if (move.type == Move::RELOCATE) {
        return solution.reassignNeuron(move.neuron1, move.target_pe, 0);
    }
    return false;
}

double HillClimbingOptimizer::evaluateMove(const MappingSolution& solution,
                                          const Move& move,
                                          const NeuralNetwork& network,
                                          const HardwareTopology& topology,
                                          const OptimizationConfig& config) {
    auto temp_solution = solution;
    double original_objective = evaluateObjective(temp_solution, network, topology, config);
    
    if (!applyMove(temp_solution, move)) {
        return 0.0;
    }
    
    double new_objective = evaluateObjective(temp_solution, network, topology, config);
    return original_objective - new_objective;
}

bool HillClimbingOptimizer::performRandomRestart(MappingSolution& solution,
                                                const HardwareTopology& topology,
                                                double restart_probability) {
    auto assignments = solution.getAllAssignments();
    if (assignments.empty()) return false;
    
    std::uniform_real_distribution<double> prob_dist(0.0, 1.0);
    std::uniform_int_distribution<PEId> pe_dist(0, topology.getTotalPEs() - 1);
    
    bool changed = false;
    for (const auto& assignment : assignments) {
        if (prob_dist(rng_) < restart_probability) {
            PEId new_pe = pe_dist(rng_);
            if (new_pe != assignment.pe_id && 
                solution.getPENeuronCount(new_pe) < topology.getPECapacity(new_pe)) {
                if (solution.reassignNeuron(assignment.neuron_id, new_pe, 0)) {
                    changed = true;
                }
            }
        }
    }
    
    return changed;
}

}