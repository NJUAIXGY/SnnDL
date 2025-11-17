/**
 * @file basic_usage_example.cpp
 * @brief 神经元映射框架基本使用示例
 * 
 * 本示例展示了如何使用神经元映射框架：
 * 1. 创建神经网络
 * 2. 定义硬件拓扑
 * 3. 执行映射
 * 4. 评估结果
 */

#include "../include/core/NeuralNetwork.h"
#include "../include/core/HardwareTopology.h"
#include "../include/core/MappingSolution.h"
#include "../include/core/NeuronMapper.h"
#include "../include/utils/Logger.h"
#include <iostream>
#include <memory>

using namespace NeuronMapping;

/**
 * @brief 创建一个简单的4层前馈神经网络
 * @return 神经网络实例
 */
std::unique_ptr<NeuralNetwork> createSimpleNetwork() {
    auto network = std::make_unique<NeuralNetwork>();
    
    // 添加神经元：输入层(4) -> 隐藏层1(8) -> 隐藏层2(4) -> 输出层(2)
    std::vector<NeuronProperties> neurons;
    
    // 输入层神经元 (ID: 0-3)
    for (uint32_t i = 0; i < 4; ++i) {
        NeuronProperties neuron(i);
        neuron.computational_load = 0.5f;  // 输入层计算负载较低
        neuron.memory_requirement = 1024;   // 1KB
        neuron.neuron_type = "input";
        neurons.push_back(neuron);
    }
    
    // 隐藏层1神经元 (ID: 4-11)
    for (uint32_t i = 4; i < 12; ++i) {
        NeuronProperties neuron(i);
        neuron.computational_load = 1.0f;   // 标准计算负载
        neuron.memory_requirement = 2048;   // 2KB
        neuron.neuron_type = "hidden";
        neurons.push_back(neuron);
    }
    
    // 隐藏层2神经元 (ID: 12-15)
    for (uint32_t i = 12; i < 16; ++i) {
        NeuronProperties neuron(i);
        neuron.computational_load = 1.0f;
        neuron.memory_requirement = 2048;
        neuron.neuron_type = "hidden";
        neurons.push_back(neuron);
    }
    
    // 输出层神经元 (ID: 16-17)
    for (uint32_t i = 16; i < 18; ++i) {
        NeuronProperties neuron(i);
        neuron.computational_load = 0.8f;
        neuron.memory_requirement = 1536;   // 1.5KB
        neuron.neuron_type = "output";
        neurons.push_back(neuron);
    }
    
    network->addNeurons(neurons);
    
    // 添加连接：全连接的层间连接
    std::vector<Connection> connections;
    
    // 输入层 -> 隐藏层1
    for (uint32_t i = 0; i < 4; ++i) {
        for (uint32_t j = 4; j < 12; ++j) {
            Connection conn(i, j, 0.5f + (i + j) * 0.1f);  // 权重在0.5-1.2之间
            conn.spike_frequency = 10.0f;  // 10Hz
            connections.push_back(conn);
        }
    }
    
    // 隐藏层1 -> 隐藏层2
    for (uint32_t i = 4; i < 12; ++i) {
        for (uint32_t j = 12; j < 16; ++j) {
            Connection conn(i, j, 0.3f + (i * j) * 0.05f);
            conn.spike_frequency = 15.0f;  // 15Hz
            connections.push_back(conn);
        }
    }
    
    // 隐藏层2 -> 输出层
    for (uint32_t i = 12; i < 16; ++i) {
        for (uint32_t j = 16; j < 18; ++j) {
            Connection conn(i, j, 0.8f + (i - j) * 0.1f);
            conn.spike_frequency = 20.0f;  // 20Hz
            connections.push_back(conn);
        }
    }
    
    network->addConnections(connections);
    
    LOG_INFO_F("Created neural network with " << network->getNeuronCount() 
               << " neurons and " << network->getConnectionCount() << " connections");
    
    return network;
}

/**
 * @brief 创建4x4 Mesh硬件拓扑
 * @return 硬件拓扑实例
 */
std::unique_ptr<HardwareTopology> create4x4MeshTopology() {
    auto topology = std::make_unique<HardwareTopology>();
    
    // 创建4x4 Mesh拓扑，每个PE可以容纳16个神经元，64MB内存
    ProcessingElement pe_template;
    pe_template.max_neurons = 16;
    pe_template.memory_capacity = 64 * 1024 * 1024;  // 64MB
    pe_template.computational_capability = 1.0f;
    pe_template.pe_type = "standard";
    
    bool success = topology->createMesh2D(4, 4, pe_template);
    
    if (success) {
        LOG_INFO_F("Created 4x4 Mesh topology with " << topology->getPECount() 
                   << " PEs, total capacity: " << topology->getTotalNeuronCapacity() << " neurons");
    } else {
        LOG_ERROR("Failed to create 4x4 Mesh topology");
    }
    
    return topology;
}

/**
 * @brief 打印映射结果
 * @param mapping 映射解决方案
 * @param network 神经网络
 * @param topology 硬件拓扑
 */
void printMappingResults(const MappingSolution& mapping,
                        const NeuralNetwork& network,
                        const HardwareTopology& topology) {
    
    std::cout << "\n=== Mapping Results ===" << std::endl;
    
    // 打印每个PE的分配情况
    for (PEId pe_id = 0; pe_id < 16; ++pe_id) {
        auto neurons = mapping.getPENeurons(pe_id);
        if (!neurons.empty()) {
            std::cout << "PE " << pe_id << ": " << neurons.size() << " neurons [";
            for (size_t i = 0; i < neurons.size(); ++i) {
                std::cout << neurons[i];
                if (i < neurons.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        }
    }
    
    // 计算和打印性能指标
    MappingConfig config;  // 使用默认配置
    auto metrics = mapping.evaluatePerformance(network, topology, config);
    
    std::cout << "\n=== Performance Metrics ===" << std::endl;
    std::cout << "Communication Cost: " << metrics.communication_cost << std::endl;
    std::cout << "Inter-PE Communication Ratio: " << metrics.inter_pe_communication_ratio * 100 << "%" << std::endl;
    std::cout << "Average Communication Distance: " << metrics.average_communication_distance << " hops" << std::endl;
    std::cout << "Load Imbalance Factor: " << metrics.load_imbalance_factor << std::endl;
    std::cout << "Max/Min Load Ratio: " << metrics.max_min_load_ratio << std::endl;
    std::cout << "Memory Utilization: " << metrics.memory_utilization * 100 << "%" << std::endl;
    std::cout << "Overall Score: " << metrics.overall_score << std::endl;
}

/**
 * @brief 演示不同的映射策略
 * @param network 神经网络
 * @param topology 硬件拓扑
 */
void demonstrateMappingStrategies(const NeuralNetwork& network,
                                 const HardwareTopology& topology) {
    
    std::cout << "\n=== Demonstrating Different Mapping Strategies ===" << std::endl;
    
    auto mapper = MapperFactory::createDefaultMapper();
    
    // 测试不同的配置
    std::vector<std::pair<std::string, MappingConfig>> configs = {
        {"Balanced", MappingConfig()},  // 默认平衡配置
        {"Communication-Optimized", []() {
            MappingConfig config;
            config.communication_weight = 0.8f;
            config.load_balance_weight = 0.15f;
            config.memory_weight = 0.05f;
            return config;
        }()},
        {"Load-Balance-Optimized", []() {
            MappingConfig config;
            config.communication_weight = 0.3f;
            config.load_balance_weight = 0.6f;
            config.memory_weight = 0.1f;
            return config;
        }()}
    };
    
    for (const auto& [strategy_name, config] : configs) {
        std::cout << "\n--- " << strategy_name << " Strategy ---" << std::endl;
        
        try {
            auto mapping = mapper->mapNetwork(network, topology, config);
            
            if (mapping) {
                auto metrics = mapping->evaluatePerformance(network, topology, config);
                
                std::cout << "Communication Cost: " << metrics.communication_cost << std::endl;
                std::cout << "Load Imbalance: " << metrics.load_imbalance_factor << std::endl;
                std::cout << "Memory Utilization: " << metrics.memory_utilization * 100 << "%" << std::endl;
                std::cout << "Overall Score: " << metrics.overall_score << std::endl;
                
                // 验证约束
                auto violations = mapping->validateConstraints(topology);
                if (violations.empty()) {
                    std::cout << "✓ All constraints satisfied" << std::endl;
                } else {
                    std::cout << "⚠ Constraint violations: " << violations.size() << std::endl;
                    for (const auto& violation : violations) {
                        std::cout << "  - " << violation << std::endl;
                    }
                }
                
            } else {
                std::cout << "❌ Mapping failed" << std::endl;
            }
            
        } catch (const std::exception& e) {
            std::cout << "❌ Exception: " << e.what() << std::endl;
        }
    }
}

/**
 * @brief 主函数
 */
int main() {
    try {
        // 初始化日志系统
        Logger::getInstance().setLevel(LogLevel::INFO);
        Logger::getInstance().enableConsoleOutput(true);
        Logger::getInstance().enableTimestamp(true);
        
        LOG_INFO("Starting Neuron Mapping Framework Example");
        
        // 1. 创建神经网络
        LOG_INFO("Step 1: Creating neural network...");
        auto network = createSimpleNetwork();
        
        // 打印网络统计信息
        auto stats = network->calculateStatistics();
        LOG_INFO_F("Network statistics - Neurons: " << stats.total_neurons 
                  << ", Connections: " << stats.total_connections
                  << ", Density: " << stats.connection_density);
        
        // 2. 创建硬件拓扑
        LOG_INFO("Step 2: Creating hardware topology...");
        auto topology = create4x4MeshTopology();
        
        // 检查容量是否足够
        if (topology->getTotalNeuronCapacity() < network->getNeuronCount()) {
            LOG_ERROR_F("Insufficient capacity! Network has " << network->getNeuronCount() 
                       << " neurons but topology only supports " << topology->getTotalNeuronCapacity());
            return -1;
        }
        
        // 3. 创建映射器并执行映射
        LOG_INFO("Step 3: Creating mapper and executing mapping...");
        auto mapper = MapperFactory::createDefaultMapper();
        
        // 设置进度回调
        mapper->setProgressCallback([](uint32_t current, uint32_t max, float current_cost, float best_cost) {
            if (current % 100 == 0) {  // 每100次迭代打印一次
                std::cout << "Progress: " << current << "/" << max 
                         << ", Current Cost: " << current_cost 
                         << ", Best Cost: " << best_cost << std::endl;
            }
        });
        
        // 配置映射参数
        MappingConfig config;
        config.strategy = "hybrid";
        config.optimizer = "simulated_annealing";
        config.max_iterations = 1000;
        config.enable_verbose_logging = true;
        
        // 执行映射
        auto mapping = mapper->mapNetwork(*network, *topology, config);
        
        if (!mapping) {
            LOG_ERROR("Mapping failed!");
            return -1;
        }
        
        LOG_INFO("Mapping completed successfully!");
        
        // 4. 评估和打印结果
        LOG_INFO("Step 4: Evaluating mapping results...");
        printMappingResults(*mapping, *network, *topology);
        
        // 5. 演示不同策略
        demonstrateMappingStrategies(*network, *topology);
        
        // 6. 生成详细报告
        LOG_INFO("Step 5: Generating detailed report...");
        std::string report = mapper->generateMappingReport(*mapping, *network, *topology);
        std::cout << "\n=== Detailed Mapping Report ===" << std::endl;
        std::cout << report << std::endl;
        
        LOG_INFO("Example completed successfully!");
        
    } catch (const std::exception& e) {
        LOG_CRITICAL_F("Unhandled exception: " << e.what());
        return -1;
    }
    
    return 0;
}

// 编译命令示例:
// g++ -std=c++17 -I../include -o basic_usage_example basic_usage_example.cpp \
//     ../src/core/*.cpp ../src/utils/*.cpp ../src/algorithms/*.cpp ../src/strategies/*.cpp \
//     -pthread -lm

// 运行示例:
// ./basic_usage_example