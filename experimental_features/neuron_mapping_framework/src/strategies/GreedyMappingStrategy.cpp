#include "strategies/GreedyMappingStrategy.h"
#include "utils/Logger.h"
#include <algorithm>
#include <limits>

namespace neuron_mapping {

std::unique_ptr<MappingSolution> GreedyMappingStrategy::mapNetwork(
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const MappingConfig& config) {
    
    if (!validateInputs(network, topology, config)) {
        LOG_ERROR("Input validation failed for greedy mapping");
        return nullptr;
    }

    auto solution = std::make_unique<MappingSolution>(network.getNeuronCount());
    
    auto neuron_scores = calculateNeuronScores(network);
    std::sort(neuron_scores.begin(), neuron_scores.end());
    
    for (const auto& neuron_score : neuron_scores) {
        PEId best_pe = findBestPE(neuron_score.neuron_id, network, topology, *solution);
        
        if (best_pe != INVALID_PE_ID) {
            uint32_t core_id = 0;  // 使用默认核心ID
            
            if (!solution->assignNeuron(neuron_score.neuron_id, best_pe, core_id)) {
                LOG_WARNING("Failed to assign neuron " + std::to_string(neuron_score.neuron_id) + 
                           " to PE " + std::to_string(best_pe));
            }
        } else {
            LOG_ERROR("No suitable PE found for neuron " + std::to_string(neuron_score.neuron_id));
        }
    }
    
    LOG_INFO("Greedy mapping completed. Mapped " + 
             std::to_string(network.getNeuronCount()) + " neurons to " +
             std::to_string(topology.getTotalPEs()) + " PEs");
    
    return solution;
}

std::string GreedyMappingStrategy::getName() const {
    return "GreedyMapping";
}

std::string GreedyMappingStrategy::getDescription() const {
    return "Greedy mapping strategy that assigns neurons to minimize communication cost";
}

std::vector<GreedyMappingStrategy::NeuronScore> 
GreedyMappingStrategy::calculateNeuronScores(const NeuralNetwork& network) {
    std::vector<NeuronScore> scores;
    
    for (NeuronId neuron_id : network.getAllNeuronIds()) {
        NeuronScore score;
        score.neuron_id = neuron_id;
        
        auto incoming = network.getIncomingConnections(neuron_id);
        auto outgoing = network.getOutgoingConnections(neuron_id);
        
        score.connection_count = incoming.size() + outgoing.size();
        
        double weight_sum = 0.0;
        for (const Connection& conn : incoming) {
            weight_sum += std::abs(conn.weight);
        }
        for (const Connection& conn : outgoing) {
            weight_sum += std::abs(conn.weight);
        }
        
        score.score = weight_sum * score.connection_count;
        scores.push_back(score);
    }
    
    return scores;
}

PEId GreedyMappingStrategy::findBestPE(
    NeuronId neuron_id,
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const MappingSolution& current_solution) {
    
    PEId best_pe = INVALID_PE_ID;
    double best_cost = std::numeric_limits<double>::max();
    
    for (PEId pe_id = 0; pe_id < topology.getTotalPEs(); ++pe_id) {
        if (current_solution.getPENeuronCount(pe_id) >= topology.getPECapacity(pe_id)) {
            continue;
        }
        
        double cost = calculatePlacementCost(neuron_id, pe_id, network, topology, current_solution);
        
        if (cost < best_cost) {
            best_cost = cost;
            best_pe = pe_id;
        }
    }
    
    return best_pe;
}

double GreedyMappingStrategy::calculatePlacementCost(
    NeuronId neuron_id,
    PEId pe_id,
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const MappingSolution& current_solution) {
    
    double total_cost = 0.0;
    
    auto incoming = network.getIncomingConnections(neuron_id);
    auto outgoing = network.getOutgoingConnections(neuron_id);
    
    for (const Connection& conn : incoming) {
        auto source_assignment = current_solution.getNeuronAssignment(conn.source_id);
        if (source_assignment) {
            double distance = 1.0;  // 简化距离计算
            double weight = std::abs(conn.weight);
            total_cost += distance * weight;
        }
    }
    
    for (const Connection& conn : outgoing) {
        auto target_assignment = current_solution.getNeuronAssignment(conn.target_id);
        if (target_assignment) {
            double distance = 1.0;  // 简化距离计算
            double weight = std::abs(conn.weight);
            total_cost += distance * weight;
        }
    }
    
    double load_penalty = static_cast<double>(current_solution.getPENeuronCount(pe_id)) / 
                         topology.getPECapacity(pe_id);
    total_cost += load_penalty * 10.0;
    
    return total_cost;
}

}