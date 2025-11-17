#include "optimizers/SimulatedAnnealingOptimizer.h"
#include "core/NeuralNetwork.h"
#include "core/HardwareTopology.h"
#include <iostream>
#include <cassert>

using namespace NeuronMapping;
using namespace neuron_mapping;

void testSimulatedAnnealingBasic() {
    std::cout << "=== Testing Simulated Annealing Basic Functionality ===\n";
    
    // 创建测试网络
    NeuralNetwork network;
    const int num_neurons = 16;
    
    // 添加神经元
    for (NeuronId i = 0; i < num_neurons; ++i) {
        NeuronProperties props(i);
        assert(network.addNeuron(props));
    }
    
    // 创建连接
    for (NeuronId i = 0; i < num_neurons - 1; ++i) {
        Connection conn(i, i + 1, 1.0f + i * 0.1f);
        assert(network.addConnection(conn));
    }
    
    // 添加一些跨连接
    for (NeuronId i = 0; i < num_neurons; i += 4) {
        Connection conn(i, (i + 8) % num_neurons, 0.5f);
        assert(network.addConnection(conn));
    }
    
    std::cout << "Created network: " << network.getNeuronCount() 
              << " neurons, " << network.getConnectionCount() << " connections\n";
    
    // 创建硬件拓扑
    HardwareTopology topology;
    ProcessingElement pe_config;
    pe_config.max_neurons = 5;
    assert(topology.createMesh2D(2, 2, pe_config));
    
    std::cout << "Created topology: " << topology.getTotalPEs() << " PEs\n";
    
    // 创建初始映射解决方案
    auto initial_solution = std::make_unique<MappingSolution>(network.getNeuronCount());
    initial_solution->greedyInitialize(network, topology);
    
    std::cout << "Initial solution assigned neurons: " 
              << initial_solution->getAssignedNeuronCount() << "\n";
    
    // 创建优化器
    SimulatedAnnealingOptimizer optimizer(
        100.0,  // 初始温度
        0.01,   // 终止温度
        5000,   // 最大迭代次数
        SimulatedAnnealingOptimizer::CoolingSchedule::EXPONENTIAL,
        SimulatedAnnealingOptimizer::NeighborhoodStrategy::SINGLE_SWAP
    );
    
    std::cout << "Optimizer: " << optimizer.getName() << "\n";
    std::cout << "Description: " << optimizer.getDescription() << "\n";
    
    OptimizationConfig config;
    config.max_iterations = 5000;
    config.convergence_threshold = 1e-6;
    
    // 执行优化
    auto optimized_solution = optimizer.optimize(
        std::move(initial_solution), network, topology, config);
    
    if (optimized_solution) {
        std::cout << "✓ Optimization successful!\n";
        std::cout << "  Final assigned neurons: " 
                  << optimized_solution->getAssignedNeuronCount() << "\n";
        std::cout << "  Iterations performed: " 
                  << optimizer.getIterationsPerformed() << "\n";
        std::cout << "  Final temperature: " 
                  << optimizer.getFinalTemperature() << "\n";
        std::cout << "  Acceptance ratio: " 
                  << optimizer.getAcceptanceRatio() * 100.0 << "%\n";
        std::cout << "  Best cost: " << optimizer.getBestCost() << "\n";
        
        // 验证结果
        assert(optimized_solution->getAssignedNeuronCount() <= network.getNeuronCount());
        
    } else {
        std::cout << "✗ Optimization failed!\n";
        assert(false);
    }
    
    std::cout << "\n=== Simulated Annealing Basic Test Completed ===\n";
}

void testDifferentCoolingSchedules() {
    std::cout << "\n=== Testing Different Cooling Schedules ===\n";
    
    // 创建简单网络
    NeuralNetwork network;
    const int num_neurons = 12;
    
    for (NeuronId i = 0; i < num_neurons; ++i) {
        NeuronProperties props(i);
        network.addNeuron(props);
    }
    
    // 环形连接
    for (NeuronId i = 0; i < num_neurons; ++i) {
        Connection conn(i, (i + 1) % num_neurons, 1.0f);
        network.addConnection(conn);
    }
    
    HardwareTopology topology;
    ProcessingElement pe_config;
    pe_config.max_neurons = 4;
    topology.createMesh2D(2, 2, pe_config);
    
    OptimizationConfig config;
    config.max_iterations = 2000;
    
    std::vector<SimulatedAnnealingOptimizer::CoolingSchedule> schedules = {
        SimulatedAnnealingOptimizer::CoolingSchedule::LINEAR,
        SimulatedAnnealingOptimizer::CoolingSchedule::EXPONENTIAL,
        SimulatedAnnealingOptimizer::CoolingSchedule::LOGARITHMIC,
        SimulatedAnnealingOptimizer::CoolingSchedule::ADAPTIVE
    };
    
    std::vector<std::string> schedule_names = {
        "Linear", "Exponential", "Logarithmic", "Adaptive"
    };
    
    for (size_t i = 0; i < schedules.size(); ++i) {
        std::cout << "\n--- Testing " << schedule_names[i] << " Cooling ---\n";
        
        // 创建初始解
        auto initial_solution = std::make_unique<MappingSolution>(network.getNeuronCount());
        initial_solution->greedyInitialize(network, topology);
        
        // 创建优化器
        SimulatedAnnealingOptimizer optimizer(50.0, 0.01, 2000, schedules[i]);
        
        auto result = optimizer.optimize(std::move(initial_solution), network, topology, config);
        
        if (result) {
            std::cout << "✓ " << schedule_names[i] << " cooling completed\n";
            std::cout << "  Iterations: " << optimizer.getIterationsPerformed() << "\n";
            std::cout << "  Final cost: " << optimizer.getBestCost() << "\n";
            std::cout << "  Acceptance ratio: " << optimizer.getAcceptanceRatio() * 100.0 << "%\n";
        } else {
            std::cout << "✗ " << schedule_names[i] << " cooling failed\n";
        }
    }
    
    std::cout << "\n=== Cooling Schedules Test Completed ===\n";
}

void testDifferentNeighborhoodStrategies() {
    std::cout << "\n=== Testing Different Neighborhood Strategies ===\n";
    
    // 创建测试网络
    NeuralNetwork network;
    const int num_neurons = 20;
    
    for (NeuronId i = 0; i < num_neurons; ++i) {
        NeuronProperties props(i);
        network.addNeuron(props);
    }
    
    // 创建随机连接
    std::random_device rd;
    std::mt19937 rng(rd());
    std::uniform_int_distribution<int> dist(0, num_neurons - 1);
    
    for (int i = 0; i < 30; ++i) {
        NeuronId src = dist(rng);
        NeuronId tgt = dist(rng);
        if (src != tgt) {
            Connection conn(src, tgt, 1.0f);
            network.addConnection(conn);
        }
    }
    
    HardwareTopology topology;
    ProcessingElement pe_config;
    pe_config.max_neurons = 6;
    topology.createMesh2D(2, 2, pe_config);
    
    OptimizationConfig config;
    config.max_iterations = 3000;
    
    std::vector<SimulatedAnnealingOptimizer::NeighborhoodStrategy> strategies = {
        SimulatedAnnealingOptimizer::NeighborhoodStrategy::SINGLE_SWAP,
        SimulatedAnnealingOptimizer::NeighborhoodStrategy::DOUBLE_SWAP,
        SimulatedAnnealingOptimizer::NeighborhoodStrategy::BLOCK_MOVE,
        SimulatedAnnealingOptimizer::NeighborhoodStrategy::RANDOM_RESTART
    };
    
    std::vector<std::string> strategy_names = {
        "Single Swap", "Double Swap", "Block Move", "Random Restart"
    };
    
    for (size_t i = 0; i < strategies.size(); ++i) {
        std::cout << "\n--- Testing " << strategy_names[i] << " Strategy ---\n";
        
        auto initial_solution = std::make_unique<MappingSolution>(network.getNeuronCount());
        initial_solution->greedyInitialize(network, topology);
        
        SimulatedAnnealingOptimizer optimizer(
            80.0, 0.01, 3000, 
            SimulatedAnnealingOptimizer::CoolingSchedule::EXPONENTIAL,
            strategies[i]
        );
        
        auto result = optimizer.optimize(std::move(initial_solution), network, topology, config);
        
        if (result) {
            std::cout << "✓ " << strategy_names[i] << " strategy completed\n";
            std::cout << "  Best cost: " << optimizer.getBestCost() << "\n";
            std::cout << "  Acceptance ratio: " << optimizer.getAcceptanceRatio() * 100.0 << "%\n";
        } else {
            std::cout << "✗ " << strategy_names[i] << " strategy failed\n";
        }
    }
    
    std::cout << "\n=== Neighborhood Strategies Test Completed ===\n";
}

int main() {
    try {
        testSimulatedAnnealingBasic();
        testDifferentCoolingSchedules();
        testDifferentNeighborhoodStrategies();
        
        std::cout << "\n✅ All Simulated Annealing tests completed successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n❌ Test failed with unknown exception!\n";
        return 1;
    }
}