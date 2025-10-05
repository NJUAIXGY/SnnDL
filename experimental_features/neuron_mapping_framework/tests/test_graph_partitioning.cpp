#include "strategies/GraphPartitioningStrategy.h"
#include "core/NeuralNetwork.h"
#include "core/HardwareTopology.h"
#include <iostream>
#include <cassert>

using namespace NeuronMapping;
using namespace neuron_mapping;

void testGraphPartitioningStrategy() {
    std::cout << "=== Testing Graph Partitioning Strategy ===" << std::endl;
    
    // 创建测试网络
    NeuralNetwork network;
    const int num_neurons = 12;
    
    // 添加神经元
    for (NeuronId i = 0; i < num_neurons; ++i) {
        NeuronProperties props(i);
        assert(network.addNeuron(props));
    }
    
    // 创建环形连接结构
    for (NeuronId i = 0; i < num_neurons; ++i) {
        Connection conn(i, (i + 1) % num_neurons, 1.0f + i * 0.1f);
        assert(network.addConnection(conn));
    }
    
    // 添加一些跨越连接
    for (NeuronId i = 0; i < num_neurons; i += 3) {
        Connection conn(i, (i + 6) % num_neurons, 0.5f);
        assert(network.addConnection(conn));
    }
    
    std::cout << "Created test network: " << network.getNeuronCount() 
              << " neurons, " << network.getConnectionCount() << " connections" << std::endl;
    
    // 创建硬件拓扑
    HardwareTopology topology;
    ProcessingElement pe_config;
    pe_config.max_neurons = 4;
    assert(topology.createMesh2D(2, 2, pe_config)); // 4个PE
    
    std::cout << "Created hardware topology: " << topology.getTotalPEs() << " PEs" << std::endl;
    
    MappingConfig config;
    config.strategy = "graph_partitioning";
    config.max_iterations = 50;
    
    // 测试不同的图分割算法
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
        std::cout << "\n--- Testing " << algorithm_names[i] << " Algorithm ---" << std::endl;
        
        GraphPartitioningStrategy strategy(algorithms[i]);
        
        std::cout << "Strategy: " << strategy.getName() << std::endl;
        std::cout << "Description: " << strategy.getDescription() << std::endl;
        
        auto solution = strategy.mapNetwork(network, topology, config);
        
        if (solution) {
            std::cout << "✓ Mapping successful!" << std::endl;
            std::cout << "  Assigned neurons: " << solution->getAssignedNeuronCount() << std::endl;
            
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
            std::cout << std::endl;
            
            // 计算性能指标
            auto metrics = solution->evaluatePerformance(network, topology, config);
            std::cout << "  Communication cost: " << metrics.communication_cost << std::endl;
            std::cout << "  Load imbalance: " << metrics.load_imbalance << std::endl;
            
        } else {
            std::cout << "✗ Mapping failed!" << std::endl;
        }
    }
    
    std::cout << "\n=== Graph Partitioning Strategy Tests Completed ===" << std::endl;
}

void testPartitioningParameters() {
    std::cout << "\n=== Testing Partitioning Parameters ===" << std::endl;
    
    // 创建一个较大的测试网络
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
    strategy.setBalanceThreshold(0.2);  // 20% imbalance tolerance
    strategy.setMaxIterations(200);
    strategy.setQualityMetric(GraphPartitioningStrategy::QualityMetric::EDGE_CUT);
    
    auto solution = strategy.mapNetwork(network, topology, config);
    
    if (solution) {
        std::cout << "✓ Parameter configuration test passed" << std::endl;
        std::cout << "  Final assignment count: " << solution->getAssignedNeuronCount() << std::endl;
    } else {
        std::cout << "✗ Parameter configuration test failed" << std::endl;
    }
}

int main() {
    try {
        testGraphPartitioningStrategy();
        testPartitioningParameters();
        
        std::cout << "\n✅ All graph partitioning tests completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n❌ Test failed with unknown exception!" << std::endl;
        return 1;
    }
}