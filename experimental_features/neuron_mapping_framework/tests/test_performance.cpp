#include "core/NeuralNetwork.h"
#include "core/HardwareTopology.h"
#include "core/NeuronMapper.h"
#include "factories/MapperFactory.h"
#include "utils/Logger.h"
#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>

using namespace NeuronMapping;
using namespace neuron_mapping;

class PerformanceTimer {
private:
    std::chrono::high_resolution_clock::time_point start_time;
    
public:
    void start() {
        start_time = std::chrono::high_resolution_clock::now();
    }
    
    double stop() {
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        return duration.count() / 1000.0; // 返回毫秒
    }
};

// 创建随机网络的工具函数
std::unique_ptr<NeuralNetwork> createRandomNetwork(int num_neurons, double connectivity = 0.1, int seed = 42) {
    auto network = std::make_unique<NeuralNetwork>();
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> weight_dist(0.1f, 2.0f);
    
    // 添加神经元
    for (NeuronId i = 0; i < num_neurons; ++i) {
        NeuronProperties props(i);
        props.neuron_type = (i < num_neurons * 0.2) ? NeuronType::INPUT :
                           (i > num_neurons * 0.8) ? NeuronType::OUTPUT : NeuronType::HIDDEN;
        network->addNeuron(props);
    }
    
    // 添加随机连接
    std::uniform_int_distribution<int> neuron_dist(0, num_neurons - 1);
    int target_connections = static_cast<int>(num_neurons * num_neurons * connectivity);
    int added_connections = 0;
    
    while (added_connections < target_connections) {
        NeuronId source = neuron_dist(rng);
        NeuronId target = neuron_dist(rng);
        
        if (source != target) {
            Connection conn(source, target, weight_dist(rng));
            if (network->addConnection(conn)) {
                added_connections++;
            }
        }
    }
    
    return network;
}

// 网络规模性能测试
void testNetworkScaling() {
    std::cout << "\n=== Network Scaling Performance Test ===" << std::endl;
    std::cout << std::setw(12) << "Neurons" << std::setw(15) << "Connections" 
              << std::setw(15) << "Map Time (ms)" << std::setw(15) << "Eval Time (ms)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    std::vector<int> neuron_counts = {10, 25, 50, 100, 200, 500};
    
    for (int num_neurons : neuron_counts) {
        // 创建测试网络
        auto network = createRandomNetwork(num_neurons, 0.15);
        
        // 创建适当大小的硬件拓扑
        HardwareTopology topology;
        int grid_size = std::max(2, static_cast<int>(std::sqrt(num_neurons / 5.0)) + 1);
        topology.createMeshTopology(grid_size, grid_size, num_neurons / (grid_size * grid_size) + 1);
        
        // 创建映射器和配置
        auto mapper = NeuronMapping::MapperFactory::createDefaultMapper();
        MappingConfig config;
        config.strategy = "random";
        config.max_iterations = 100;
        
        // 测试映射性能
        PerformanceTimer timer;
        timer.start();
        auto solution = mapper->mapNetwork(*network, topology, config);
        double map_time = timer.stop();
        
        // 测试评估性能
        timer.start();
        auto metrics = mapper->evaluateMapping(*solution, *network, topology, config);
        double eval_time = timer.stop();
        
        std::cout << std::setw(12) << num_neurons 
                  << std::setw(15) << network->getConnectionCount()
                  << std::setw(15) << std::fixed << std::setprecision(2) << map_time
                  << std::setw(15) << std::fixed << std::setprecision(2) << eval_time << std::endl;
    }
}

// 不同拓扑性能测试
void testTopologyPerformance() {
    std::cout << "\n=== Hardware Topology Performance Test ===" << std::endl;
    std::cout << std::setw(15) << "Topology" << std::setw(12) << "PEs" 
              << std::setw(15) << "Create (ms)" << std::setw(15) << "Distance (ms)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    struct TopologyConfig {
        std::string name;
        std::function<bool(HardwareTopology&)> creator;
    };
    
    std::vector<TopologyConfig> topologies = {
        {"2x2 Mesh", [](HardwareTopology& topo) { return topo.createMeshTopology(2, 2, 10); }},
        {"4x4 Mesh", [](HardwareTopology& topo) { return topo.createMeshTopology(4, 4, 10); }},
        {"6x6 Mesh", [](HardwareTopology& topo) { return topo.createMeshTopology(6, 6, 10); }},
        {"8x8 Mesh", [](HardwareTopology& topo) { return topo.createMeshTopology(8, 8, 10); }},
        {"3x3 Torus", [](HardwareTopology& topo) { return topo.createTorusTopology(3, 3, 10); }},
        {"5x5 Torus", [](HardwareTopology& topo) { return topo.createTorusTopology(5, 5, 10); }},
    };
    
    for (const auto& topo_config : topologies) {
        HardwareTopology topology;
        
        // 测试拓扑创建时间
        PerformanceTimer timer;
        timer.start();
        bool created = topo_config.creator(topology);
        double create_time = timer.stop();
        
        if (!created) {
            std::cout << std::setw(15) << topo_config.name << " - Creation failed" << std::endl;
            continue;
        }
        
        // 测试距离计算时间
        timer.start();
        int num_distance_calls = std::min(1000, topology.getTotalPEs() * topology.getTotalPEs());
        for (int i = 0; i < num_distance_calls; ++i) {
            PEId pe1 = i % topology.getTotalPEs();
            PEId pe2 = (i + 1) % topology.getTotalPEs();
            topology.calculateDistance(pe1, pe2);
        }
        double distance_time = timer.stop();
        
        std::cout << std::setw(15) << topo_config.name 
                  << std::setw(12) << topology.getTotalPEs()
                  << std::setw(15) << std::fixed << std::setprecision(2) << create_time
                  << std::setw(15) << std::fixed << std::setprecision(2) << distance_time << std::endl;
    }
}

// 映射策略性能比较
void testStrategyPerformance() {
    std::cout << "\n=== Mapping Strategy Performance Test ===" << std::endl;
    std::cout << std::setw(20) << "Strategy" << std::setw(15) << "Map Time (ms)" 
              << std::setw(15) << "Comm Cost" << std::setw(15) << "Load Imbalance" << std::endl;
    std::cout << std::string(70, '-') << std::endl;
    
    // 创建中等规模的测试网络
    auto network = createRandomNetwork(100, 0.12);
    
    HardwareTopology topology;
    topology.createMeshTopology(5, 5, 8);
    
    MappingConfig config;
    config.max_iterations = 200;
    
    struct StrategyConfig {
        std::string name;
        std::function<std::unique_ptr<INeuronMapper>()> creator;
    };
    
    std::vector<StrategyConfig> strategies = {
        {"Default", []() { return NeuronMapping::MapperFactory::createDefaultMapper(); }},
        {"Fast", []() { return NeuronMapping::MapperFactory::createFastMapper(); }},
        {"Precision", []() { return NeuronMapping::MapperFactory::createPrecisionMapper(); }},
        {"Custom Random", []() { return NeuronMapping::MapperFactory::createCustomMapper("RandomMapping", "LocalSearch"); }},
        {"Custom Greedy", []() { return NeuronMapping::MapperFactory::createCustomMapper("GreedyMapping", "HillClimbing"); }},
    };
    
    for (const auto& strategy_config : strategies) {
        try {
            auto mapper = strategy_config.creator();
            config.strategy = "random"; // 设置默认策略
            
            PerformanceTimer timer;
            timer.start();
            auto solution = mapper->mapNetwork(*network, topology, config);
            double map_time = timer.stop();
            
            if (solution) {
                auto metrics = mapper->evaluateMapping(*solution, *network, topology, config);
                
                std::cout << std::setw(20) << strategy_config.name
                          << std::setw(15) << std::fixed << std::setprecision(2) << map_time
                          << std::setw(15) << std::fixed << std::setprecision(3) << metrics.communication_cost
                          << std::setw(15) << std::fixed << std::setprecision(3) << metrics.load_imbalance << std::endl;
            } else {
                std::cout << std::setw(20) << strategy_config.name << " - Failed to create solution" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cout << std::setw(20) << strategy_config.name << " - Exception: " << e.what() << std::endl;
        }
    }
}

// 内存使用性能测试
void testMemoryPerformance() {
    std::cout << "\n=== Memory Performance Test ===" << std::endl;
    std::cout << "Testing memory allocation patterns for different network sizes..." << std::endl;
    
    std::vector<int> sizes = {50, 100, 200, 500, 1000};
    
    for (int size : sizes) {
        std::cout << "\nTesting network size: " << size << " neurons" << std::endl;
        
        try {
            // 测试网络创建和销毁的内存模式
            for (int iteration = 0; iteration < 3; ++iteration) {
                auto network = createRandomNetwork(size, 0.1);
                
                HardwareTopology topology;
                int grid_size = std::max(3, static_cast<int>(std::sqrt(size / 10.0)));
                topology.createMeshTopology(grid_size, grid_size, size / (grid_size * grid_size) + 1);
                
                auto mapper = NeuronMapping::MapperFactory::createDefaultMapper();
                MappingConfig config;
                config.strategy = "random";
                config.max_iterations = 50;
                
                auto solution = mapper->mapNetwork(*network, topology, config);
                
                if (solution) {
                    auto metrics = mapper->evaluateMapping(*solution, *network, topology, config);
                    std::cout << "  Iteration " << iteration + 1 << ": " 
                              << network->getNeuronCount() << " neurons, "
                              << network->getConnectionCount() << " connections, "
                              << solution->getAssignedNeuronCount() << " assigned" << std::endl;
                }
            }
        } catch (const std::exception& e) {
            std::cout << "  Exception at size " << size << ": " << e.what() << std::endl;
        }
    }
}

// 并发性能测试（模拟）
void testConcurrentPerformance() {
    std::cout << "\n=== Concurrent Mapping Performance Test ===" << std::endl;
    std::cout << "Testing multiple mapping operations..." << std::endl;
    
    // 创建多个不同的网络
    std::vector<std::unique_ptr<NeuralNetwork>> networks;
    for (int i = 0; i < 5; ++i) {
        networks.push_back(createRandomNetwork(50 + i * 10, 0.1 + i * 0.02, 42 + i));
    }
    
    HardwareTopology topology;
    topology.createMeshTopology(4, 4, 10);
    
    MappingConfig config;
    config.strategy = "random";
    config.max_iterations = 100;
    
    auto mapper = NeuronMapping::MapperFactory::createDefaultMapper();
    
    // 顺序执行映射
    PerformanceTimer timer;
    timer.start();
    
    std::vector<std::unique_ptr<MappingSolution>> solutions;
    for (const auto& network : networks) {
        auto solution = mapper->mapNetwork(*network, topology, config);
        solutions.push_back(std::move(solution));
    }
    
    double sequential_time = timer.stop();
    
    std::cout << "Sequential mapping of " << networks.size() << " networks: " 
              << std::fixed << std::setprecision(2) << sequential_time << " ms" << std::endl;
    
    // 验证所有解决方案都已创建
    int successful_mappings = 0;
    for (const auto& solution : solutions) {
        if (solution && solution->getAssignedNeuronCount() > 0) {
            successful_mappings++;
        }
    }
    
    std::cout << "Successful mappings: " << successful_mappings << "/" << networks.size() << std::endl;
    std::cout << "Average time per mapping: " << sequential_time / networks.size() << " ms" << std::endl;
}

int main() {
    std::cout << "=== Neuron Mapping Framework Performance Tests ===" << std::endl;
    
    // 设置日志级别为警告，减少输出噪音
    Logger::getInstance().setLogLevel(LogLevel::WARNING);
    
    // 运行所有性能测试
    testNetworkScaling();
    testTopologyPerformance();
    testStrategyPerformance();
    testMemoryPerformance();
    testConcurrentPerformance();
    
    std::cout << "\n=== Performance Tests Completed ===" << std::endl;
    
    return 0;
}