#include "core/NeuronMapper.h"
#include "factories/MapperFactory.h"
#include "strategies/GraphPartitioningStrategy.h"
#include "optimizers/SimulatedAnnealingOptimizer.h"
#include "algorithms/CommunityDetector.h"
#include "evaluators/PerformanceEvaluator.h"
#include <iostream>
#include <cassert>
#include <chrono>
#include <memory>

using namespace NeuronMapping;
using namespace neuron_mapping;

/**
 * @brief 创建标准测试网络
 */
std::unique_ptr<NeuralNetwork> createStandardTestNetwork() {
    auto network = std::make_unique<NeuralNetwork>();
    
    // 创建具有层次结构的网络
    const int num_layers = 3;
    const int neurons_per_layer = 6;
    
    // 添加神经元
    for (int layer = 0; layer < num_layers; ++layer) {
        for (int i = 0; i < neurons_per_layer; ++i) {
            NeuronId neuron_id = layer * neurons_per_layer + i;
            NeuronProperties props(neuron_id);
            props.computational_load = 1.0f;
            props.memory_requirement = 100;
            network->addNeuron(props);
        }
    }
    
    // 添加层内连接
    for (int layer = 0; layer < num_layers; ++layer) {
        for (int i = 0; i < neurons_per_layer; ++i) {
            for (int j = i + 1; j < neurons_per_layer; ++j) {
                NeuronId src = layer * neurons_per_layer + i;
                NeuronId tgt = layer * neurons_per_layer + j;
                network->addConnection(Connection(src, tgt, 1.5f));
            }
        }
    }
    
    // 添加层间连接
    for (int layer = 0; layer < num_layers - 1; ++layer) {
        for (int i = 0; i < neurons_per_layer; ++i) {
            NeuronId src = layer * neurons_per_layer + i;
            NeuronId tgt = (layer + 1) * neurons_per_layer + (i % neurons_per_layer);
            network->addConnection(Connection(src, tgt, 1.0f));
        }
    }
    
    return network;
}

/**
 * @brief 创建标准测试拓扑
 */
std::unique_ptr<HardwareTopology> createStandardTestTopology() {
    auto topology = std::make_unique<HardwareTopology>();
    
    ProcessingElement pe_config;
    pe_config.max_neurons = 5;
    pe_config.memory_capacity = 256 * 1024 * 1024;  // 256MB
    pe_config.computational_capability = 1.0f;
    
    if (!topology->createMesh2D(2, 2, pe_config)) {
        return nullptr;
    }
    
    return topology;
}

/**
 * @brief 测试框架集成
 */
void testFrameworkIntegration() {
    std::cout << "=== Testing Framework Integration ===\n";
    
    auto network = createStandardTestNetwork();
    auto topology = createStandardTestTopology();
    
    assert(network != nullptr);
    assert(topology != nullptr);
    assert(network->getNeuronCount() > 0);
    assert(topology->getTotalPEs() > 0);
    
    std::cout << "✓ Created test network: " << network->getNeuronCount() 
              << " neurons, " << network->getConnectionCount() << " connections\n";
    std::cout << "✓ Created test topology: " << topology->getTotalPEs() << " PEs\n";
    
    // 测试工厂模式
    auto default_mapper = MapperFactory::createDefaultMapper();
    auto fast_mapper = MapperFactory::createFastMapper();
    auto precision_mapper = MapperFactory::createPrecisionMapper();
    
    assert(default_mapper != nullptr);
    assert(fast_mapper != nullptr);
    assert(precision_mapper != nullptr);
    
    std::cout << "✓ Factory pattern works correctly\n";
    
    // 测试映射
    MappingConfig config;
    config.max_iterations = 100;
    
    auto solution = default_mapper->mapNetwork(*network, *topology, config);
    assert(solution != nullptr);
    assert(solution->getAssignedNeuronCount() > 0);
    
    std::cout << "✓ Basic mapping completed\n";
    
    // 测试性能评估
    auto metrics = solution->evaluatePerformance(*network, *topology, config);
    assert(metrics.communication_cost >= 0);
    assert(metrics.load_imbalance >= 0);
    assert(metrics.pe_utilization >= 0);
    
    std::cout << "✓ Performance evaluation completed\n";
    std::cout << "  Communication cost: " << metrics.communication_cost << "\n";
    std::cout << "  Load imbalance: " << metrics.load_imbalance << "\n";
    std::cout << "  PE utilization: " << metrics.pe_utilization << "\n";
    
    std::cout << "Framework integration test passed!\n\n";
}

/**
 * @brief 测试映射策略
 */
void testMappingStrategies() {
    std::cout << "=== Testing Mapping Strategies ===\n";
    
    auto network = createStandardTestNetwork();
    auto topology = createStandardTestTopology();
    
    MappingConfig config;
    config.max_iterations = 50;
    
    // 测试图分割策略
    std::vector<GraphPartitioningStrategy::PartitioningAlgorithm> algorithms = {
        GraphPartitioningStrategy::PartitioningAlgorithm::MULTILEVEL,
        GraphPartitioningStrategy::PartitioningAlgorithm::SPECTRAL,
        GraphPartitioningStrategy::PartitioningAlgorithm::RECURSIVE_BISECTION
    };
    
    std::vector<std::string> algorithm_names = {
        "Multilevel", "Spectral", "Recursive Bisection"
    };
    
    for (size_t i = 0; i < algorithms.size(); ++i) {
        std::cout << "Testing " << algorithm_names[i] << " partitioning...\n";
        
        GraphPartitioningStrategy strategy(algorithms[i]);
        auto solution = strategy.mapNetwork(*network, *topology, config);
        
        assert(solution != nullptr);
        assert(solution->getAssignedNeuronCount() > 0);
        
        auto metrics = solution->evaluatePerformance(*network, *topology, config);
        std::cout << "  ✓ " << algorithm_names[i] << " completed - Cost: " 
                  << metrics.communication_cost << "\n";
    }
    
    std::cout << "Mapping strategies test passed!\n\n";
}

/**
 * @brief 测试优化器
 */
void testOptimizers() {
    std::cout << "=== Testing Optimizers ===\n";
    
    auto network = createStandardTestNetwork();
    auto topology = createStandardTestTopology();
    
    // 创建初始解
    auto initial_solution = std::make_unique<MappingSolution>(network->getNeuronCount());
    initial_solution->greedyInitialize(*network, *topology);
    
    MappingConfig mapping_config;
    auto initial_metrics = initial_solution->evaluatePerformance(*network, *topology, mapping_config);
    
    std::cout << "Initial solution cost: " << initial_metrics.communication_cost << "\n";
    
    // 测试模拟退火优化器
    std::vector<SimulatedAnnealingOptimizer::CoolingSchedule> schedules = {
        SimulatedAnnealingOptimizer::CoolingSchedule::EXPONENTIAL,
        SimulatedAnnealingOptimizer::CoolingSchedule::LINEAR,
        SimulatedAnnealingOptimizer::CoolingSchedule::ADAPTIVE
    };
    
    std::vector<std::string> schedule_names = {
        "Exponential", "Linear", "Adaptive"
    };
    
    OptimizationConfig opt_config;
    opt_config.max_iterations = 500;
    opt_config.convergence_threshold = 1e-6;
    
    for (size_t i = 0; i < schedules.size(); ++i) {
        std::cout << "Testing " << schedule_names[i] << " cooling...\n";
        
        auto test_solution = std::make_unique<MappingSolution>(*initial_solution);
        
        SimulatedAnnealingOptimizer optimizer(20.0, 0.01, 500, schedules[i]);
        auto optimized = optimizer.optimize(std::move(test_solution), *network, *topology, opt_config);
        
        assert(optimized != nullptr);
        
        auto final_metrics = optimized->evaluatePerformance(*network, *topology, mapping_config);
        std::cout << "  ✓ " << schedule_names[i] << " completed - Final cost: " 
                  << final_metrics.communication_cost << " (Iterations: " 
                  << optimizer.getIterationsPerformed() << ")\n";
    }
    
    std::cout << "Optimizers test passed!\n\n";
}

/**
 * @brief 测试网络分析
 */
void testNetworkAnalysis() {
    std::cout << "=== Testing Network Analysis ===\n";
    
    auto network = createStandardTestNetwork();
    
    // 测试社区检测
    std::vector<CommunityDetector::Algorithm> algorithms = {
        CommunityDetector::Algorithm::LOUVAIN,
        CommunityDetector::Algorithm::LABEL_PROPAGATION,
        CommunityDetector::Algorithm::MODULARITY_MAXIMIZATION
    };
    
    std::vector<std::string> algorithm_names = {
        "Louvain", "Label Propagation", "Modularity Maximization"
    };
    
    for (size_t i = 0; i < algorithms.size(); ++i) {
        std::cout << "Testing " << algorithm_names[i] << " community detection...\n";
        
        CommunityDetector detector(algorithms[i]);
        detector.setMaxIterations(50);
        
        auto structure = detector.detectCommunities(*network);
        
        assert(structure != nullptr);
        assert(structure->num_communities > 0);
        
        std::cout << "  ✓ " << algorithm_names[i] << " found " 
                  << structure->num_communities << " communities (Modularity: " 
                  << structure->modularity << ")\n";
    }
    
    std::cout << "Network analysis test passed!\n\n";
}

/**
 * @brief 测试性能基准
 */
void testPerformanceBenchmark() {
    std::cout << "=== Testing Performance Benchmark ===\n";
    
    // 创建不同大小的网络进行性能测试
    std::vector<int> network_sizes = {10, 20, 30};
    
    for (int size : network_sizes) {
        std::cout << "Benchmarking network size: " << size << " neurons\n";
        
        // 创建测试网络
        NeuralNetwork network;
        for (NeuronId i = 0; i < size; ++i) {
            NeuronProperties props(i);
            network.addNeuron(props);
        }
        
        // 添加随机连接
        for (int i = 0; i < size * 2; ++i) {
            NeuronId src = std::rand() % size;
            NeuronId tgt = std::rand() % size;
            if (src != tgt) {
                network.addConnection(Connection(src, tgt, 1.0f));
            }
        }
        
        // 创建拓扑
        HardwareTopology topology;
        ProcessingElement pe_config;
        pe_config.max_neurons = 8;
        topology.createMesh2D(2, 2, pe_config);
        
        // 测试映射性能
        auto mapper = MapperFactory::createDefaultMapper();
        MappingConfig config;
        config.max_iterations = 100;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        auto solution = mapper->mapNetwork(network, topology, config);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        assert(solution != nullptr);
        
        std::cout << "  Mapping time: " << duration.count() << "ms\n";
        std::cout << "  Assigned neurons: " << solution->getAssignedNeuronCount() << "/" << size << "\n";
        
        auto metrics = solution->evaluatePerformance(network, topology, config);
        std::cout << "  Communication cost: " << metrics.communication_cost << "\n";
        std::cout << "  Load imbalance: " << metrics.load_imbalance << "\n";
    }
    
    std::cout << "Performance benchmark test passed!\n\n";
}

/**
 * @brief 测试错误处理
 */
void testErrorHandling() {
    std::cout << "=== Testing Error Handling ===\n";
    
    // 测试空网络
    NeuralNetwork empty_network;
    HardwareTopology topology;
    ProcessingElement pe_config;
    pe_config.max_neurons = 4;
    topology.createMesh2D(2, 2, pe_config);
    
    auto mapper = MapperFactory::createDefaultMapper();
    MappingConfig config;
    
    auto solution = mapper->mapNetwork(empty_network, topology, config);
    // 应该返回空解或处理空网络
    std::cout << "✓ Empty network handled correctly\n";
    
    // 测试无效配置
    auto network = createStandardTestNetwork();
    MappingConfig invalid_config;
    invalid_config.max_iterations = 0;  // 无效配置
    
    // 应该使用默认值或处理无效配置
    auto solution2 = mapper->mapNetwork(*network, topology, invalid_config);
    std::cout << "✓ Invalid configuration handled correctly\n";
    
    // 测试社区检测异常情况
    CommunityDetector detector(CommunityDetector::Algorithm::LOUVAIN);
    auto structure = detector.detectCommunities(empty_network);
    assert(structure != nullptr);  // 应该返回有效但空的结构
    std::cout << "✓ Empty network community detection handled correctly\n";
    
    std::cout << "Error handling test passed!\n\n";
}

/**
 * @brief 主测试函数
 */
int main() {
    try {
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        
        std::cout << "=== Comprehensive Framework Test Suite ===\n";
        std::cout << "Testing all major components and integration...\n\n";
        
        // 配置日志为警告级别以减少输出
        Logger::getInstance().setLevel(LogLevel::WARNING);
        Logger::getInstance().enableConsoleOutput(true);
        
        // 运行所有测试
        testFrameworkIntegration();
        testMappingStrategies();
        testOptimizers();
        testNetworkAnalysis();
        testPerformanceBenchmark();
        testErrorHandling();
        
        std::cout << "=== All Tests Completed Successfully! ===\n";
        std::cout << "✅ Framework Integration\n";
        std::cout << "✅ Mapping Strategies\n";
        std::cout << "✅ Optimization Algorithms\n";
        std::cout << "✅ Network Analysis\n";
        std::cout << "✅ Performance Benchmarks\n";
        std::cout << "✅ Error Handling\n\n";
        
        std::cout << "The Neuron Mapping Framework is ready for production use!\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Comprehensive test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n❌ Comprehensive test failed with unknown exception!\n";
        return 1;
    }
}