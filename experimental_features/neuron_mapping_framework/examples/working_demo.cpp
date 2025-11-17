#include "strategies/GraphPartitioningStrategy.h"
#include "optimizers/SimulatedAnnealingOptimizer.h"
#include "algorithms/CommunityDetector.h"
#include "utils/Logger.h"
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace NeuronMapping;
using namespace neuron_mapping;

/**
 * @brief 创建示例神经网络
 */
std::unique_ptr<NeuralNetwork> createExampleNetwork() {
    auto network = std::make_unique<NeuralNetwork>();
    
    const int layers = 3;
    const int neurons_per_layer = 6;
    
    std::cout << "Creating layered neural network:\n";
    std::cout << "  Layers: " << layers << "\n";
    std::cout << "  Neurons per layer: " << neurons_per_layer << "\n";
    
    // 添加神经元
    for (int layer = 0; layer < layers; ++layer) {
        for (int i = 0; i < neurons_per_layer; ++i) {
            NeuronId neuron_id = layer * neurons_per_layer + i;
            NeuronProperties props(neuron_id);
            props.computational_load = 1.0f + (layer * 0.5f);
            props.memory_requirement = 100 + (layer * 50);
            network->addNeuron(props);
        }
    }
    
    // 添加层内连接（密集连接）
    for (int layer = 0; layer < layers; ++layer) {
        for (int i = 0; i < neurons_per_layer; ++i) {
            for (int j = i + 1; j < neurons_per_layer; ++j) {
                NeuronId src = layer * neurons_per_layer + i;
                NeuronId tgt = layer * neurons_per_layer + j;
                float weight = 1.5f + static_cast<float>(std::rand()) / RAND_MAX * 0.5f;
                network->addConnection(Connection(src, tgt, weight));
            }
        }
    }
    
    // 添加层间连接（稀疏连接）
    for (int layer = 0; layer < layers - 1; ++layer) {
        for (int i = 0; i < neurons_per_layer; ++i) {
            int connections_per_neuron = 2;
            for (int c = 0; c < connections_per_neuron; ++c) {
                NeuronId src = layer * neurons_per_layer + i;
                NeuronId tgt = (layer + 1) * neurons_per_layer + (i + c) % neurons_per_layer;
                float weight = 0.8f + static_cast<float>(std::rand()) / RAND_MAX * 0.4f;
                network->addConnection(Connection(src, tgt, weight));
            }
        }
    }
    
    std::cout << "  Total neurons: " << network->getNeuronCount() << "\n";
    std::cout << "  Total connections: " << network->getConnectionCount() << "\n";
    
    return network;
}

/**
 * @brief 创建示例硬件拓扑
 */
std::unique_ptr<HardwareTopology> createExampleTopology() {
    auto topology = std::make_unique<HardwareTopology>();
    
    const int grid_width = 3;
    const int grid_height = 3;
    const int neurons_per_pe = 4;
    
    std::cout << "\nCreating mesh hardware topology:\n";
    std::cout << "  Grid size: " << grid_width << "x" << grid_height << "\n";
    std::cout << "  Max neurons per PE: " << neurons_per_pe << "\n";
    
    ProcessingElement pe_config;
    pe_config.max_neurons = neurons_per_pe;
    pe_config.memory_capacity = 512 * 1024 * 1024;  // 512MB
    pe_config.computational_capability = 1.0f;
    
    if (!topology->createMesh2D(grid_width, grid_height, pe_config)) {
        std::cerr << "Failed to create hardware topology!\n";
        return nullptr;
    }
    
    std::cout << "  Total PEs: " << topology->getTotalPEs() << "\n";
    
    return topology;
}

/**
 * @brief 演示社区检测功能
 */
void demonstrateCommunityDetection(const NeuralNetwork& network) {
    std::cout << "\n=== Community Detection Analysis ===\n";
    
    CommunityDetector detector(CommunityDetector::Algorithm::LOUVAIN);
    detector.setResolution(1.0f);
    detector.setMaxIterations(100);
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto structure = detector.detectCommunities(network);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (structure) {
        std::cout << "Community detection completed in " << duration.count() << "ms\n";
        std::cout << "Results:\n";
        std::cout << "  Communities found: " << structure->num_communities << "\n";
        std::cout << "  Modularity: " << std::fixed << std::setprecision(4) << structure->modularity << "\n";
        
        // 分析社区大小分布
        std::vector<size_t> community_sizes;
        for (const auto& community : structure->communities) {
            community_sizes.push_back(community.neurons.size());
        }
        
        std::sort(community_sizes.rbegin(), community_sizes.rend());
        std::cout << "  Community sizes: ";
        for (size_t i = 0; i < std::min(size_t(5), community_sizes.size()); ++i) {
            std::cout << community_sizes[i] << " ";
        }
        if (community_sizes.size() > 5) {
            std::cout << "...";
        }
        std::cout << "\n";
        
        // 质量度量
        float coverage = detector.evaluateCommunityQuality(*structure, network, 
                                                         CommunityDetector::QualityMetric::COVERAGE);
        std::cout << "  Coverage: " << std::fixed << std::setprecision(4) << coverage << "\n";
    }
}

/**
 * @brief 演示图分割映射策略
 */
void demonstrateMappingStrategies(const NeuralNetwork& network, const HardwareTopology& topology) {
    std::cout << "\n=== Graph Partitioning Mapping ===\n";
    
    MappingConfig config;
    config.strategy = "graph_partitioning";
    config.max_iterations = 1000;
    
    std::vector<GraphPartitioningStrategy::PartitioningAlgorithm> algorithms = {
        GraphPartitioningStrategy::PartitioningAlgorithm::MULTILEVEL,
        GraphPartitioningStrategy::PartitioningAlgorithm::SPECTRAL,
        GraphPartitioningStrategy::PartitioningAlgorithm::RECURSIVE_BISECTION
    };
    
    std::vector<std::string> algorithm_names = {
        "Multilevel", "Spectral", "Recursive Bisection"
    };
    
    std::cout << std::setw(20) << "Algorithm" << std::setw(12) << "Time(ms)" 
              << std::setw(15) << "Comm Cost" << std::setw(15) << "Load Imbal" << "\n";
    std::cout << std::string(62, '-') << "\n";
    
    for (size_t i = 0; i < algorithms.size(); ++i) {
        std::cout << "\nTesting " << algorithm_names[i] << " algorithm...\n";
        
        GraphPartitioningStrategy strategy(algorithms[i]);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        auto solution = strategy.mapNetwork(network, topology, config);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        if (solution) {
            auto metrics = solution->evaluatePerformance(network, topology, config);
            std::cout << std::setw(20) << algorithm_names[i] 
                      << std::setw(12) << std::fixed << std::setprecision(0) << duration.count()
                      << std::setw(15) << std::fixed << std::setprecision(2) << metrics.communication_cost
                      << std::setw(15) << std::fixed << std::setprecision(4) << metrics.load_imbalance
                      << "\n";
        } else {
            std::cout << std::setw(20) << algorithm_names[i] 
                      << std::setw(12) << "FAILED"
                      << std::setw(15) << "N/A"
                      << std::setw(15) << "N/A" << "\n";
        }
    }
}

/**
 * @brief 演示模拟退火优化
 */
void demonstrateOptimization(const NeuralNetwork& network, const HardwareTopology& topology) {
    std::cout << "\n=== Simulated Annealing Optimization ===\n";
    
    // 创建初始解
    auto initial_solution = std::make_unique<MappingSolution>(network.getNeuronCount());
    initial_solution->greedyInitialize(network, topology);
    
    MappingConfig config;
    auto initial_metrics = initial_solution->evaluatePerformance(network, topology, config);
    
    std::cout << "Initial solution performance:\n";
    std::cout << "  Communication cost: " << std::fixed << std::setprecision(2) 
              << initial_metrics.communication_cost << "\n";
    std::cout << "  Load imbalance: " << std::fixed << std::setprecision(4) 
              << initial_metrics.load_imbalance << "\n";
    
    // 创建优化器
    SimulatedAnnealingOptimizer optimizer(
        20.0,    // 初始温度
        0.01,    // 终止温度
        1000,    // 最大迭代次数
        SimulatedAnnealingOptimizer::CoolingSchedule::EXPONENTIAL,
        SimulatedAnnealingOptimizer::NeighborhoodStrategy::SINGLE_SWAP
    );
    
    OptimizationConfig opt_config;
    opt_config.max_iterations = 1000;
    opt_config.convergence_threshold = 1e-6;
    
    std::cout << "\nRunning optimization...\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    auto optimized_solution = optimizer.optimize(
        std::move(initial_solution), network, topology, opt_config);
    auto end_time = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (optimized_solution) {
        auto final_metrics = optimized_solution->evaluatePerformance(network, topology, config);
        
        std::cout << "\nOptimization completed in " << duration.count() << "ms\n";
        std::cout << "Optimizer statistics:\n";
        std::cout << "  Iterations performed: " << optimizer.getIterationsPerformed() << "\n";
        std::cout << "  Final temperature: " << std::fixed << std::setprecision(4) 
                  << optimizer.getFinalTemperature() << "\n";
        std::cout << "  Acceptance ratio: " << std::fixed << std::setprecision(2) 
                  << optimizer.getAcceptanceRatio() * 100.0 << "%\n";
        
        std::cout << "\nFinal solution performance:\n";
        std::cout << "  Communication cost: " << std::fixed << std::setprecision(2) 
                  << final_metrics.communication_cost << "\n";
        std::cout << "  Load imbalance: " << std::fixed << std::setprecision(4) 
                  << final_metrics.load_imbalance << "\n";
        
        // 计算改进程度
        float comm_improvement = ((initial_metrics.communication_cost - final_metrics.communication_cost) 
                                 / initial_metrics.communication_cost) * 100.0f;
        float balance_improvement = ((initial_metrics.load_imbalance - final_metrics.load_imbalance) 
                                   / initial_metrics.load_imbalance) * 100.0f;
        
        std::cout << "\nImprovement:\n";
        std::cout << "  Communication cost: " << std::fixed << std::setprecision(1) 
                  << comm_improvement << "%\n";
        std::cout << "  Load balance: " << std::fixed << std::setprecision(1) 
                  << balance_improvement << "%\n";
    }
}

/**
 * @brief 主演示函数
 */
int main() {
    try {
        std::srand(static_cast<unsigned>(std::time(nullptr)));
        
        std::cout << "=== Neuron Mapping Framework Working Demo ===\n";
        std::cout << "This demo showcases the implemented functionality of the framework.\n\n";
        
        // 配置日志
        Logger::getInstance().setLevel(LogLevel::INFO);
        Logger::getInstance().enableConsoleOutput(true);
        Logger::getInstance().enableTimestamp(false);
        
        // 创建示例数据
        auto network = createExampleNetwork();
        auto topology = createExampleTopology();
        
        if (!network || !topology) {
            std::cerr << "Failed to create example data!\n";
            return 1;
        }
        
        // 演示各个功能模块
        demonstrateCommunityDetection(*network);
        demonstrateMappingStrategies(*network, *topology);
        demonstrateOptimization(*network, *topology);
        
        std::cout << "\n=== Demo Completed Successfully ===\n";
        std::cout << "The framework provides:\n";
        std::cout << "• Graph partitioning strategies (Multilevel, Spectral, Recursive Bisection)\n";
        std::cout << "• Advanced optimization algorithms (Simulated Annealing)\n";
        std::cout << "• Network analysis tools (Community Detection)\n";
        std::cout << "• Comprehensive performance evaluation\n";
        std::cout << "• Flexible configuration options\n\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Demo failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n❌ Demo failed with unknown exception!\n";
        return 1;
    }
}