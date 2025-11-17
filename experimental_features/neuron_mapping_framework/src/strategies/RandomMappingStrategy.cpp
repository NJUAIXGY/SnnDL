#include "strategies/RandomMappingStrategy.h"
#include "utils/Logger.h"
#include <algorithm>
#include <vector>

namespace neuron_mapping {

RandomMappingStrategy::RandomMappingStrategy(uint32_t seed)
    : seed_(seed), rng_(seed) {
    LOG_INFO("RandomMappingStrategy initialized with seed: " + std::to_string(seed));
}

std::unique_ptr<MappingSolution> RandomMappingStrategy::mapNetwork(
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const MappingConfig& config) {
    
    if (!validateInputs(network, topology, config)) {
        LOG_ERROR("Input validation failed for random mapping");
        return nullptr;
    }

    auto solution = std::make_unique<MappingSolution>(network.getNeuronCount());
    
    assignNeuronsRandomly(network, topology, *solution);
    
    LOG_INFO("Random mapping completed. Mapped " + 
             std::to_string(network.getNeuronCount()) + " neurons to " +
             std::to_string(topology.getTotalPEs()) + " PEs");
    
    return solution;
}

std::string RandomMappingStrategy::getName() const {
    return "RandomMapping";
}

std::string RandomMappingStrategy::getDescription() const {
    return "Random mapping strategy that assigns neurons to PEs randomly while respecting capacity constraints";
}

void RandomMappingStrategy::setSeed(uint32_t seed) {
    seed_ = seed;
    rng_.seed(seed);
    LOG_INFO("Random mapping seed updated to: " + std::to_string(seed));
}

void RandomMappingStrategy::assignNeuronsRandomly(
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    MappingSolution& solution) {
    
    std::vector<NeuronId> neurons = network.getAllNeuronIds();
    std::shuffle(neurons.begin(), neurons.end(), rng_);
    
    std::vector<uint32_t> pe_loads(topology.getTotalPEs(), 0);
    
    for (NeuronId neuron_id : neurons) {
        std::vector<PEId> available_pes;
        
        for (PEId pe_id = 0; pe_id < topology.getTotalPEs(); ++pe_id) {
            if (pe_loads[pe_id] < topology.getPECapacity(pe_id)) {
                available_pes.push_back(pe_id);
            }
        }
        
        if (available_pes.empty()) {
            LOG_ERROR("No available PE for neuron " + std::to_string(neuron_id));
            continue;
        }
        
        std::uniform_int_distribution<size_t> dist(0, available_pes.size() - 1);
        PEId selected_pe = available_pes[dist(rng_)];
        
        uint32_t core_id = 0;  // 使用默认核心ID
        
        if (solution.assignNeuron(neuron_id, selected_pe, core_id)) {
            pe_loads[selected_pe]++;
        } else {
            LOG_WARNING("Failed to assign neuron " + std::to_string(neuron_id) + 
                       " to PE " + std::to_string(selected_pe));
        }
    }
}

}