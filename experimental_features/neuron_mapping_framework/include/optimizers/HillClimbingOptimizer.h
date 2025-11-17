#ifndef NEURON_MAPPING_FRAMEWORK_OPTIMIZERS_HILL_CLIMBING_OPTIMIZER_H
#define NEURON_MAPPING_FRAMEWORK_OPTIMIZERS_HILL_CLIMBING_OPTIMIZER_H

#include "optimizers/MappingOptimizer.h"
#include <random>

namespace neuron_mapping {

using namespace NeuronMapping;

class HillClimbingOptimizer : public MappingOptimizer {
public:
    enum class MoveType {
        BEST_IMPROVEMENT,    // 选择最佳改进
        FIRST_IMPROVEMENT,   // 选择第一个改进
        RANDOM_RESTART       // 随机重启爬山
    };

    explicit HillClimbingOptimizer(MoveType move_type = MoveType::BEST_IMPROVEMENT,
                                  uint32_t seed = std::random_device{}());
    ~HillClimbingOptimizer() override = default;

    std::unique_ptr<MappingSolution> optimize(
        std::unique_ptr<MappingSolution> initial_solution,
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const OptimizationConfig& config
    ) override;

    std::string getName() const override;
    std::string getDescription() const override;

    void setMoveType(MoveType type);
    void setSeed(uint32_t seed);

private:
    MoveType move_type_;
    std::mt19937 rng_;
    uint32_t seed_;

    struct Move {
        enum Type { SWAP, RELOCATE } type;
        NeuronId neuron1;
        NeuronId neuron2;  // Only used for SWAP
        PEId target_pe;    // Only used for RELOCATE
        double improvement;
        
        // Constructor for SWAP moves
        static Move createSwap(NeuronId n1, NeuronId n2, double imp) {
            Move move;
            move.type = SWAP;
            move.neuron1 = n1;
            move.neuron2 = n2;
            move.target_pe = INVALID_PE_ID;
            move.improvement = imp;
            return move;
        }
        
        // Constructor for RELOCATE moves
        static Move createRelocate(NeuronId n, PEId pe, double imp) {
            Move move;
            move.type = RELOCATE;
            move.neuron1 = n;
            move.neuron2 = INVALID_NEURON_ID;
            move.target_pe = pe;
            move.improvement = imp;
            return move;
        }
        
        Move() : type(SWAP), neuron1(INVALID_NEURON_ID), neuron2(INVALID_NEURON_ID), 
                target_pe(INVALID_PE_ID), improvement(0.0) {}
    };

    std::vector<Move> generateAllMoves(const MappingSolution& solution,
                                      const NeuralNetwork& network,
                                      const HardwareTopology& topology,
                                      const OptimizationConfig& config);

    std::vector<Move> generateSwapMoves(const MappingSolution& solution,
                                       const NeuralNetwork& network,
                                       const HardwareTopology& topology,
                                       const OptimizationConfig& config);

    std::vector<Move> generateRelocateMoves(const MappingSolution& solution,
                                           const NeuralNetwork& network,
                                           const HardwareTopology& topology,
                                           const OptimizationConfig& config);

    bool applyMove(MappingSolution& solution, const Move& move);

    double evaluateMove(const MappingSolution& solution,
                       const Move& move,
                       const NeuralNetwork& network,
                       const HardwareTopology& topology,
                       const OptimizationConfig& config);

    bool performRandomRestart(MappingSolution& solution,
                             const HardwareTopology& topology,
                             double restart_probability = 0.1);
};

}

#endif