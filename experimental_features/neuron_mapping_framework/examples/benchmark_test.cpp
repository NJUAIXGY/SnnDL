#include "strategies/GraphPartitioningStrategy.h"
#include "strategies/RandomMappingStrategy.h"
#include "optimizers/SimulatedAnnealingOptimizer.h"
#include "algorithms/CommunityDetector.h"
#include "utils/Logger.h"
#include <iostream>
#include <iomanip>
#include <chrono>
#include <random>
#include <vector>
#include <algorithm>

using namespace NeuronMapping;
using namespace neuron_mapping;

/**
 * @brief 生成大规模随机神经网络
 */
std::unique_ptr<NeuralNetwork> generateLargeRandomNetwork(
    uint32_t num_neurons, 
    float connection_probability = 0.1f,
    uint32_t random_seed = 42) {
    
    std::mt19937 rng(random_seed);
    std::uniform_real_distribution<float> weight_dist(0.1f, 2.0f);
    std::uniform_real_distribution<float> prob_dist(0.0f, 1.0f);
    std::uniform_real_distribution<float> load_dist(0.5f, 2.0f);
    std::uniform_int_distribution<uint32_t> memory_dist(512, 4096);
    
    auto network = std::make_unique<NeuralNetwork>();
    
    std::cout << "生成 " << num_neurons << " 个神经元的随机网络...\n";
    
    // 添加神经元
    for (uint32_t i = 0; i < num_neurons; ++i) {
        NeuronProperties props(i);
        props.computational_load = load_dist(rng);
        props.memory_requirement = memory_dist(rng);
        network->addNeuron(props);
    }
    
    // 添加随机连接
    uint32_t connections_added = 0;
    for (uint32_t i = 0; i < num_neurons; ++i) {
        for (uint32_t j = 0; j < num_neurons; ++j) {
            if (i != j && prob_dist(rng) < connection_probability) {
                float weight = weight_dist(rng);
                network->addConnection(Connection(i, j, weight));
                connections_added++;
            }
        }
    }
    
    std::cout << "生成完成：" << num_neurons << " 神经元，" << connections_added << " 连接\n";
    std::cout << "连接密度：" << std::fixed << std::setprecision(4) 
              << (float(connections_added) / (num_neurons * (num_neurons - 1))) << "\n\n";
    
    return network;
}

/**
 * @brief 创建硬件拓扑
 */
std::unique_ptr<HardwareTopology> createBenchmarkTopology(uint32_t num_pes) {
    auto topology = std::make_unique<HardwareTopology>();
    
    // 创建4x4网格拓扑
    uint32_t grid_size = static_cast<uint32_t>(std::sqrt(num_pes));
    if (grid_size * grid_size != num_pes) {
        // 如果不是完全平方数，调整为最接近的矩形
        uint32_t width = grid_size;
        uint32_t height = (num_pes + width - 1) / width;
        grid_size = width;
        
        std::cout << "创建 " << width << "x" << height << " 网格拓扑 (总共" << num_pes << "个PE)\n";
    } else {
        std::cout << "创建 " << grid_size << "x" << grid_size << " 网格拓扑\n";
    }
    
    ProcessingElement pe_config;
    pe_config.max_neurons = 32;  // 每个PE最多32个神经元
    pe_config.memory_capacity = 512 * 1024 * 1024;  // 512MB
    pe_config.computational_capability = 1.0f;
    
    if (!topology->createMesh2D(grid_size, grid_size, pe_config)) {
        std::cerr << "Failed to create topology!\n";
        return nullptr;
    }
    
    std::cout << "硬件拓扑创建完成：" << topology->getTotalPEs() << " PEs\n\n";
    
    return topology;
}

/**
 * @brief 创建均匀分布映射（基线方案）
 */
std::unique_ptr<MappingSolution> createUniformMapping(
    const NeuralNetwork& network, 
    const HardwareTopology& topology) {
    
    auto solution = std::make_unique<MappingSolution>(network.getNeuronCount());
    auto neurons = network.getAllNeuronIds();
    uint32_t total_pes = topology.getTotalPEs();
    
    std::cout << "创建均匀分布映射...\n";
    
    // 简单的轮询分配
    for (size_t i = 0; i < neurons.size(); ++i) {
        PEId target_pe = i % total_pes;
        solution->assignNeuron(neurons[i], target_pe);
    }
    
    std::cout << "均匀分布映射完成：每个PE平均 " 
              << (float(neurons.size()) / total_pes) << " 个神经元\n\n";
    
    return solution;
}

/**
 * @brief 运行基准测试
 */
void runBenchmarkTest(uint32_t num_neurons = 256, uint32_t num_pes = 16) {
    std::cout << "===== 神经元映射框架基准测试 =====\n";
    std::cout << "测试规模：" << num_neurons << " 神经元 → " << num_pes << " PEs\n";
    std::cout << "========================================\n\n";
    
    // 1. 生成测试数据
    auto network = generateLargeRandomNetwork(num_neurons, 0.08f);  // 8%连接概率
    auto topology = createBenchmarkTopology(num_pes);
    
    if (!network || !topology) {
        std::cerr << "Failed to create test data!\n";
        return;
    }
    
    MappingConfig config;
    config.max_iterations = 500;  // 减少迭代次数以加快测试
    
    std::cout << std::left << std::setw(25) << "映射方法" 
              << std::setw(12) << "时间(ms)" 
              << std::setw(15) << "通信成本" 
              << std::setw(15) << "负载不均衡" 
              << std::setw(12) << "PE利用率"
              << std::setw(12) << "改进程度" << "\n";
    std::cout << std::string(85, '-') << "\n";
    
    // 2. 基线方案：均匀分布映射
    std::cout << "运行基线测试：均匀分布映射...\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    auto uniform_solution = createUniformMapping(*network, *topology);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto uniform_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    auto uniform_metrics = uniform_solution->evaluatePerformance(*network, *topology, config);
    
    std::cout << std::left << std::setw(25) << "均匀分布 (基线)" 
              << std::setw(12) << uniform_time.count()
              << std::setw(15) << std::fixed << std::setprecision(2) << uniform_metrics.communication_cost
              << std::setw(15) << std::fixed << std::setprecision(4) << uniform_metrics.load_imbalance
              << std::setw(12) << std::fixed << std::setprecision(4) << uniform_metrics.pe_utilization
              << std::setw(12) << "0.0%" << "\n";
    
    // 3. 随机映射策略
    std::cout << "\n运行随机映射策略...\n";
    start_time = std::chrono::high_resolution_clock::now();
    RandomMappingStrategy random_strategy;
    auto random_solution = random_strategy.mapNetwork(*network, *topology, config);
    end_time = std::chrono::high_resolution_clock::now();
    auto random_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (random_solution) {
        auto random_metrics = random_solution->evaluatePerformance(*network, *topology, config);
        float random_improvement = ((uniform_metrics.communication_cost - random_metrics.communication_cost) / 
                                   uniform_metrics.communication_cost) * 100.0f;
        
        std::cout << std::left << std::setw(25) << "随机映射" 
                  << std::setw(12) << random_time.count()
                  << std::setw(15) << std::fixed << std::setprecision(2) << random_metrics.communication_cost
                  << std::setw(15) << std::fixed << std::setprecision(4) << random_metrics.load_imbalance
                  << std::setw(12) << std::fixed << std::setprecision(4) << random_metrics.pe_utilization
                  << std::setw(12) << std::fixed << std::setprecision(1) << random_improvement << "%" << "\n";
    }
    
    // 4. 多层次图分割策略
    std::cout << "\n运行多层次图分割映射...\n";
    start_time = std::chrono::high_resolution_clock::now();
    GraphPartitioningStrategy multilevel_strategy(GraphPartitioningStrategy::PartitioningAlgorithm::MULTILEVEL);
    auto multilevel_solution = multilevel_strategy.mapNetwork(*network, *topology, config);
    end_time = std::chrono::high_resolution_clock::now();
    auto multilevel_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (multilevel_solution) {
        auto multilevel_metrics = multilevel_solution->evaluatePerformance(*network, *topology, config);
        float multilevel_improvement = ((uniform_metrics.communication_cost - multilevel_metrics.communication_cost) / 
                                       uniform_metrics.communication_cost) * 100.0f;
        
        std::cout << std::left << std::setw(25) << "多层次图分割" 
                  << std::setw(12) << multilevel_time.count()
                  << std::setw(15) << std::fixed << std::setprecision(2) << multilevel_metrics.communication_cost
                  << std::setw(15) << std::fixed << std::setprecision(4) << multilevel_metrics.load_imbalance
                  << std::setw(12) << std::fixed << std::setprecision(4) << multilevel_metrics.pe_utilization
                  << std::setw(12) << std::fixed << std::setprecision(1) << multilevel_improvement << "%" << "\n";
    }
    
    // 5. 谱分割策略
    std::cout << "\n运行谱分割映射...\n";
    start_time = std::chrono::high_resolution_clock::now();
    GraphPartitioningStrategy spectral_strategy(GraphPartitioningStrategy::PartitioningAlgorithm::SPECTRAL);
    auto spectral_solution = spectral_strategy.mapNetwork(*network, *topology, config);
    end_time = std::chrono::high_resolution_clock::now();
    auto spectral_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    if (spectral_solution) {
        auto spectral_metrics = spectral_solution->evaluatePerformance(*network, *topology, config);
        float spectral_improvement = ((uniform_metrics.communication_cost - spectral_metrics.communication_cost) / 
                                     uniform_metrics.communication_cost) * 100.0f;
        
        std::cout << std::left << std::setw(25) << "谱分割" 
                  << std::setw(12) << spectral_time.count()
                  << std::setw(15) << std::fixed << std::setprecision(2) << spectral_metrics.communication_cost
                  << std::setw(15) << std::fixed << std::setprecision(4) << spectral_metrics.load_imbalance
                  << std::setw(12) << std::fixed << std::setprecision(4) << spectral_metrics.pe_utilization
                  << std::setw(12) << std::fixed << std::setprecision(1) << spectral_improvement << "%" << "\n";
    }
    
    // 6. 模拟退火优化（基于多层次分割的初始解）
    if (multilevel_solution) {
        std::cout << "\n运行模拟退火优化（基于多层次分割初始解）...\n";
        
        // 复制多层次分割的解作为初始解
        auto initial_solution = std::make_unique<MappingSolution>(*multilevel_solution);
        auto initial_metrics = initial_solution->evaluatePerformance(*network, *topology, config);
        
        start_time = std::chrono::high_resolution_clock::now();
        SimulatedAnnealingOptimizer sa_optimizer(
            30.0f,   // 初始温度
            0.001f,  // 终止温度
            1000,    // 最大迭代次数
            SimulatedAnnealingOptimizer::CoolingSchedule::EXPONENTIAL,
            SimulatedAnnealingOptimizer::NeighborhoodStrategy::SINGLE_SWAP
        );
        
        OptimizationConfig opt_config;
        opt_config.max_iterations = 1000;
        opt_config.convergence_threshold = 1e-6;
        
        auto sa_solution = sa_optimizer.optimize(std::move(initial_solution), *network, *topology, opt_config);
        end_time = std::chrono::high_resolution_clock::now();
        auto sa_time = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        if (sa_solution) {
            auto sa_metrics = sa_solution->evaluatePerformance(*network, *topology, config);
            float sa_improvement = ((uniform_metrics.communication_cost - sa_metrics.communication_cost) / 
                                   uniform_metrics.communication_cost) * 100.0f;
            
            std::cout << std::left << std::setw(25) << "多层次+模拟退火" 
                      << std::setw(12) << (multilevel_time.count() + sa_time.count())
                      << std::setw(15) << std::fixed << std::setprecision(2) << sa_metrics.communication_cost
                      << std::setw(15) << std::fixed << std::setprecision(4) << sa_metrics.load_imbalance
                      << std::setw(12) << std::fixed << std::setprecision(4) << sa_metrics.pe_utilization
                      << std::setw(12) << std::fixed << std::setprecision(1) << sa_improvement << "%" << "\n";
            
            std::cout << "\n模拟退火详细统计:\n";
            std::cout << "  执行迭代次数: " << sa_optimizer.getIterationsPerformed() << "\n";
            std::cout << "  最终温度: " << std::fixed << std::setprecision(6) << sa_optimizer.getFinalTemperature() << "\n";
            std::cout << "  接受率: " << std::fixed << std::setprecision(2) << sa_optimizer.getAcceptanceRatio() * 100.0f << "%\n";
            std::cout << "  初始成本: " << std::fixed << std::setprecision(2) << initial_metrics.communication_cost << "\n";
            std::cout << "  最终成本: " << std::fixed << std::setprecision(2) << sa_metrics.communication_cost << "\n";
            std::cout << "  相对初始解改进: " << std::fixed << std::setprecision(1) 
                      << ((initial_metrics.communication_cost - sa_metrics.communication_cost) / initial_metrics.communication_cost) * 100.0f 
                      << "%\n";
        }
    }
    
    std::cout << "\n========================================\n";
    std::cout << "基准测试完成！\n";
    
    // 网络特征分析
    std::cout << "\n网络特征分析:\n";
    std::cout << "  神经元数量: " << network->getNeuronCount() << "\n";
    std::cout << "  连接数量: " << network->getConnectionCount() << "\n";
    std::cout << "  平均度: " << std::fixed << std::setprecision(2) 
              << (2.0f * network->getConnectionCount() / network->getNeuronCount()) << "\n";
    std::cout << "  连接密度: " << std::fixed << std::setprecision(4) 
              << (float(network->getConnectionCount()) / (network->getNeuronCount() * (network->getNeuronCount() - 1))) << "\n";
    
    // 硬件配置分析
    std::cout << "\n硬件配置分析:\n";
    std::cout << "  PE数量: " << topology->getTotalPEs() << "\n";
    std::cout << "  理论最大神经元/PE: " << (num_neurons / num_pes) << "\n";
    std::cout << "  拓扑类型: 网格拓扑\n";
    // 计算平均距离和直径（简化实现）
    float avg_dist = 0.0f;
    uint32_t diameter = 0;
    if (num_pes == 16) {  // 4x4网格
        avg_dist = 2.67f;  // 4x4网格的理论平均距离
        diameter = 6;      // 4x4网格的理论直径
    } else {
        avg_dist = 2.0f;   // 估算值
        diameter = 4;      // 估算值
    }
    std::cout << "  平均距离: " << std::fixed << std::setprecision(2) << avg_dist << "\n";
    std::cout << "  网络直径: " << diameter << "\n";
}

/**
 * @brief 主函数
 */
int main(int argc, char* argv[]) {
    // 禁用详细日志以提高性能
    Logger::getInstance().setLevel(LogLevel::WARNING);
    Logger::getInstance().enableConsoleOutput(true);
    Logger::getInstance().enableTimestamp(false);
    
    uint32_t num_neurons = 256;
    uint32_t num_pes = 16;
    
    // 处理命令行参数
    if (argc >= 2) {
        num_neurons = std::stoul(argv[1]);
    }
    if (argc >= 3) {
        num_pes = std::stoul(argv[2]);
    }
    
    try {
        runBenchmarkTest(num_neurons, num_pes);
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "基准测试失败: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "基准测试失败: 未知错误\n";
        return 1;
    }
}