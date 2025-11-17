#include "core/NeuronMapper.h"
#include "factories/MapperFactory.h"
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
 * 构建一个具有明确社区结构的神经网络用于演示
 */
std::unique_ptr<NeuralNetwork> createExampleNetwork() {
    auto network = std::make_unique<NeuralNetwork>();
    
    // 创建具有层次结构的神经网络
    const int layers = 3;
    const int neurons_per_layer = 8;
    
    std::cout << "Creating layered neural network:\n";
    std::cout << "  Layers: " << layers << "\n";
    std::cout << "  Neurons per layer: " << neurons_per_layer << "\n";
    
    // 添加神经元
    for (int layer = 0; layer < layers; ++layer) {
        for (int i = 0; i < neurons_per_layer; ++i) {
            NeuronId neuron_id = layer * neurons_per_layer + i;
            NeuronProperties props(neuron_id);
            props.computational_load = 1.0f + (layer * 0.5f);  // 深层神经元计算成本更高
            props.memory_requirement = 100 + (layer * 50);   // 深层神经元内存需求更大
            network->addNeuron(props);
        }
    }
    
    // 添加层内连接（密集连接）
    for (int layer = 0; layer < layers; ++layer) {
        for (int i = 0; i < neurons_per_layer; ++i) {
            for (int j = i + 1; j < neurons_per_layer; ++j) {
                NeuronId src = layer * neurons_per_layer + i;
                NeuronId tgt = layer * neurons_per_layer + j;
                
                // 层内连接权重较高
                float weight = 1.5f + static_cast<float>(std::rand()) / RAND_MAX * 0.5f;
                network->addConnection(Connection(src, tgt, weight));
            }
        }
    }
    
    // 添加层间连接（稀疏连接）
    for (int layer = 0; layer < layers - 1; ++layer) {
        for (int i = 0; i < neurons_per_layer; ++i) {
            // 每个神经元连接到下一层的几个神经元
            int connections_per_neuron = 3;
            for (int c = 0; c < connections_per_neuron; ++c) {
                NeuronId src = layer * neurons_per_layer + i;
                NeuronId tgt = (layer + 1) * neurons_per_layer + (i + c) % neurons_per_layer;
                
                // 层间连接权重中等
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
 * 构建一个网格状的硬件拓扑用于映射
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
 * @brief 演示不同映射策略的性能比较
 */
void demonstrateMappingStrategies(const NeuralNetwork& network, const HardwareTopology& topology) {
    std::cout << "\n=== Mapping Strategy Comparison ===\n";
    
    MappingConfig config;
    config.strategy = "comparison";
    config.max_iterations = 1000;
    config.optimization_level = 2;
    
    // 测试不同的映射策略
    std::vector<std::string> strategy_names = {"Random", "Greedy", "Graph Partitioning"};
    std::vector<std::unique_ptr<MappingSolution>> solutions;
    std::vector<float> execution_times;
    
    // 随机映射
    {
        std::cout << "\nTesting Random Mapping...\n";
        auto mapper = MapperFactory::createFastMapper();
        
        auto start_time = std::chrono::high_resolution_clock::now();
        auto solution = mapper->mapNetwork(network, topology, config);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        execution_times.push_back(static_cast<float>(duration.count()));
        
        if (solution) {
            auto metrics = solution->evaluatePerformance(network, topology, config);
            std::cout << "  Execution time: " << duration.count() << "ms\n";
            std::cout << "  Communication cost: " << std::fixed << std::setprecision(2) 
                      << metrics.communication_cost << "\n";
            std::cout << "  Load imbalance: " << std::fixed << std::setprecision(4) 
                      << metrics.load_imbalance << "\n";
            std::cout << "  PE utilization: " << std::fixed << std::setprecision(4) 
                      << metrics.pe_utilization << "\n";
            
            solutions.push_back(std::move(solution));
        }
    }
    
    // 贪心映射
    {
        std::cout << "\nTesting Greedy Mapping...\n";
        auto mapper = MapperFactory::createDefaultMapper();
        
        auto start_time = std::chrono::high_resolution_clock::now();
        auto solution = mapper->mapNetwork(network, topology, config);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        execution_times.push_back(static_cast<float>(duration.count()));
        
        if (solution) {
            auto metrics = solution->evaluatePerformance(network, topology, config);
            std::cout << "  Execution time: " << duration.count() << "ms\n";
            std::cout << "  Communication cost: " << std::fixed << std::setprecision(2) 
                      << metrics.communication_cost << "\n";
            std::cout << "  Load imbalance: " << std::fixed << std::setprecision(4) 
                      << metrics.load_imbalance << "\n";
            std::cout << "  PE utilization: " << std::fixed << std::setprecision(4) 
                      << metrics.pe_utilization << "\n";
            
            solutions.push_back(std::move(solution));
        }
    }
    
    // 图分割映射
    {
        std::cout << "\nTesting Graph Partitioning Mapping...\n";
        
        GraphPartitioningStrategy strategy(GraphPartitioningStrategy::PartitioningAlgorithm::MULTILEVEL);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        auto solution = strategy.mapNetwork(network, topology, config);
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        execution_times.push_back(static_cast<float>(duration.count()));
        
        if (solution) {
            auto metrics = solution->evaluatePerformance(network, topology, config);
            std::cout << "  Execution time: " << duration.count() << "ms\n";
            std::cout << "  Communication cost: " << std::fixed << std::setprecision(2) 
                      << metrics.communication_cost << "\n";
            std::cout << "  Load imbalance: " << std::fixed << std::setprecision(4) 
                      << metrics.load_imbalance << "\n";
            std::cout << "  PE utilization: " << std::fixed << std::setprecision(4) 
                      << metrics.pe_utilization << "\n";
            
            solutions.push_back(std::move(solution));
        }
    }
    
    // 比较结果
    if (solutions.size() >= 2) {
        std::cout << "\n=== Performance Comparison Summary ===\n";
        std::cout << std::setw(20) << "Strategy" << std::setw(12) << "Time(ms)" 
                  << std::setw(15) << "Comm Cost" << std::setw(15) << "Load Imbal" << "\n";
        std::cout << std::string(62, '-') << "\n";
        
        for (size_t i = 0; i < solutions.size() && i < strategy_names.size(); ++i) {
            auto metrics = solutions[i]->evaluatePerformance(network, topology, config);
            std::cout << std::setw(20) << strategy_names[i] 
                      << std::setw(12) << std::fixed << std::setprecision(0) << execution_times[i]
                      << std::setw(15) << std::fixed << std::setprecision(2) << metrics.communication_cost
                      << std::setw(15) << std::fixed << std::setprecision(4) << metrics.load_imbalance
                      << "\n";
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
        50.0,    // 初始温度
        0.01,    // 终止温度
        2000,    // 最大迭代次数
        SimulatedAnnealingOptimizer::CoolingSchedule::EXPONENTIAL,
        SimulatedAnnealingOptimizer::NeighborhoodStrategy::SINGLE_SWAP
    );
    
    OptimizationConfig opt_config;
    opt_config.max_iterations = 2000;
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
        
        std::cout << "=== Neuron Mapping Framework Comprehensive Demo ===\n";
        std::cout << "This demo showcases the complete functionality of the framework.\n\n";
        
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
        std::cout << "• Multiple mapping strategies (Random, Greedy, Graph Partitioning)\n";
        std::cout << "• Advanced optimization algorithms (Simulated Annealing)\n";
        std::cout << "• Network analysis tools (Community Detection)\n";
        std::cout << "• Comprehensive performance evaluation\n";
        std::cout << "• Flexible configuration options\n\n";
        
        std::cout << "For more advanced usage, see the examples/ directory.\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Demo failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n❌ Demo failed with unknown exception!\n";
        return 1;
    }
}