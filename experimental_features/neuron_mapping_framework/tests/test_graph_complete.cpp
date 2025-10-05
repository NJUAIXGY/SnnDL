#include "strategies/GraphPartitioningStrategy.h"
#include "core/NeuralNetwork.h"
#include "core/HardwareTopology.h"
#include <iostream>
#include <cassert>

using namespace NeuronMapping;
using namespace neuron_mapping;

void testCompleteGraphPartitioning() {
    std::cout << "=== Testing Complete Graph Partitioning ===\n";
    
    // 创建神经网络
    NeuralNetwork network;
    const int num_neurons = 12;
    
    // 添加神经元
    for (NeuronId i = 0; i < num_neurons; ++i) {
        NeuronProperties props(i);
        assert(network.addNeuron(props));
    }
    
    // 创建环形连接 + 一些跨连接
    for (NeuronId i = 0; i < num_neurons; ++i) {
        Connection conn(i, (i + 1) % num_neurons, 1.0f + i * 0.1f);
        assert(network.addConnection(conn));
    }
    
    // 添加一些长距离连接
    for (NeuronId i = 0; i < num_neurons; i += 3) {
        Connection conn(i, (i + 6) % num_neurons, 0.5f);
        assert(network.addConnection(conn));
    }
    
    std::cout << "Created network: " << network.getNeuronCount() 
              << " neurons, " << network.getConnectionCount() << " connections\n";
    
    // 创建硬件拓扑
    HardwareTopology topology;
    ProcessingElement pe_config;
    pe_config.max_neurons = 4;
    assert(topology.createMesh2D(2, 2, pe_config));
    
    std::cout << "Created topology: " << topology.getTotalPEs() << " PEs\n";
    
    MappingConfig config;
    config.strategy = "graph_partitioning";
    config.max_iterations = 50;
    
    // 测试所有分割算法
    std::vector<GraphPartitioningStrategy::PartitioningAlgorithm> algorithms = {
        GraphPartitioningStrategy::PartitioningAlgorithm::SPECTRAL,
        GraphPartitioningStrategy::PartitioningAlgorithm::KERNIGHAN_LIN,
        GraphPartitioningStrategy::PartitioningAlgorithm::MULTILEVEL,
        GraphPartitioningStrategy::PartitioningAlgorithm::RECURSIVE_BISECTION,
        GraphPartitioningStrategy::PartitioningAlgorithm::COMMUNITY_DETECTION
    };
    
    std::vector<std::string> algorithm_names = {
        "Spectral", "Kernighan-Lin", "Multilevel", "Recursive Bisection", "Community Detection"
    };
    
    for (size_t i = 0; i < algorithms.size(); ++i) {
        std::cout << "\n--- Testing " << algorithm_names[i] << " Algorithm ---\n";
        
        GraphPartitioningStrategy strategy(algorithms[i]);
        
        std::cout << "Strategy: " << strategy.getName() << "\n";
        std::cout << "Description: " << strategy.getDescription() << "\n";
        
        auto solution = strategy.mapNetwork(network, topology, config);
        
        if (solution) {
            std::cout << "✓ Mapping successful!\n";
            std::cout << "  Assigned neurons: " << solution->getAssignedNeuronCount() << "\n";
            
            // 验证映射结果
            assert(solution->getAssignedNeuronCount() <= network.getNeuronCount());
            
            // 检查PE分配
            std::vector<int> pe_counts(topology.getTotalPEs(), 0);
            for (NeuronId neuron_id = 0; neuron_id < network.getNeuronCount(); ++neuron_id) {
                PEId pe_id = solution->getNeuronPE(neuron_id);
                if (pe_id != INVALID_PE_ID && pe_id < topology.getTotalPEs()) {
                    pe_counts[pe_id]++;
                }
            }
            
            std::cout << "  PE allocation: ";
            for (size_t pe = 0; pe < pe_counts.size(); ++pe) {
                std::cout << "PE" << pe << ":" << pe_counts[pe] << " ";
            }
            std::cout << "\n";
            
            // 计算性能指标
            auto metrics = solution->evaluatePerformance(network, topology, config);
            std::cout << "  Communication cost: " << metrics.communication_cost << "\n";
            std::cout << "  Load imbalance: " << metrics.load_imbalance << "\n";
            
        } else {
            std::cout << "✗ Mapping failed!\n";
        }
    }
    
    std::cout << "\n=== Complete Graph Partitioning Tests Completed ===\n";
}

void testPartitioningParameters() {
    std::cout << "\n=== Testing Partitioning Parameters ===\n";
    
    // 创建较大网络
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
    
    for (int i = 0; i < 40; ++i) {
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
    
    MappingConfig config;
    config.max_iterations = 100;
    
    // 测试参数设置
    GraphPartitioningStrategy strategy(GraphPartitioningStrategy::PartitioningAlgorithm::MULTILEVEL);
    
    // 设置不同参数
    strategy.setBalanceThreshold(0.2);
    strategy.setMaxIterations(200);
    strategy.setQualityMetric(GraphPartitioningStrategy::QualityMetric::EDGE_CUT);
    
    auto solution = strategy.mapNetwork(network, topology, config);
    
    if (solution) {
        std::cout << "✓ Parameter configuration test passed\n";
        std::cout << "  Final assignment count: " << solution->getAssignedNeuronCount() << "\n";
    } else {
        std::cout << "✗ Parameter configuration test failed\n";
    }
}

int main() {
    try {
        testCompleteGraphPartitioning();
        testPartitioningParameters();
        
        std::cout << "\n✅ All complete graph partitioning tests completed successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n❌ Test failed with unknown exception!\n";
        return 1;
    }
}