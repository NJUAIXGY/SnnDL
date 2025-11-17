#ifndef NEURON_MAPPING_FRAMEWORK_OPTIMIZERS_LOCAL_SEARCH_OPTIMIZER_H
#define NEURON_MAPPING_FRAMEWORK_OPTIMIZERS_LOCAL_SEARCH_OPTIMIZER_H

#include "optimizers/MappingOptimizer.h"
#include <random>
#include <vector>

namespace neuron_mapping {

using namespace NeuronMapping;

class LocalSearchOptimizer : public MappingOptimizer {
public:
    enum class NeighborType {
        RANDOM_SWAP,      // 随机交换两个神经元
        RANDOM_MOVE,      // 随机移动一个神经元
        GREEDY_SWAP,      // 贪心交换
        GREEDY_MOVE,      // 贪心移动
        MIXED             // 混合策略
    };

    explicit LocalSearchOptimizer(NeighborType neighbor_type = NeighborType::MIXED,
                                 uint32_t seed = std::random_device{}());
    ~LocalSearchOptimizer() override = default;

    std::unique_ptr<MappingSolution> optimize(
        std::unique_ptr<MappingSolution> initial_solution,
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const OptimizationConfig& config
    ) override;

    std::string getName() const override;
    std::string getDescription() const override;

    void setNeighborType(NeighborType type);
    void setSeed(uint32_t seed);

private:
    NeighborType neighbor_type_;
    std::mt19937 rng_;
    uint32_t seed_;

    struct OptimizationStats {
        uint32_t iterations = 0;
        uint32_t improvements = 0;
        uint32_t plateau_count = 0;
        double best_objective = std::numeric_limits<double>::max();
        double initial_objective = 0.0;
    };

    bool generateNeighbor(MappingSolution& solution,
                         const NeuralNetwork& network,
                         const HardwareTopology& topology,
                         NeighborType type);

    bool performRandomSwap(MappingSolution& solution,
                          const HardwareTopology& topology);

    bool performRandomMove(MappingSolution& solution,
                          const HardwareTopology& topology);

    bool performGreedySwap(MappingSolution& solution,
                          const NeuralNetwork& network,
                          const HardwareTopology& topology);

    bool performGreedyMove(MappingSolution& solution,
                          const NeuralNetwork& network,
                          const HardwareTopology& topology);

    std::pair<NeuronId, NeuronId> findBestSwapPair(const MappingSolution& solution,
                                                   const NeuralNetwork& network,
                                                   const OptimizationConfig& config);

    NeuronId findBestMoveNeuron(const MappingSolution& solution,
                               const NeuralNetwork& network,
                               const HardwareTopology& topology,
                               const OptimizationConfig& config);

    PEId findBestTargetPE(NeuronId neuron_id,
                         const MappingSolution& solution,
                         const NeuralNetwork& network,
                         const HardwareTopology& topology,
                         const OptimizationConfig& config);

    bool isValidMove(NeuronId neuron_id, 
                    PEId target_pe,
                    const MappingSolution& solution,
                    const HardwareTopology& topology) const;

    void logProgress(const OptimizationStats& stats,
                    const OptimizationConfig& config) const;
};

}

#endif