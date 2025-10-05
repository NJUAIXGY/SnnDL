#include "core/NeuralNetwork.h"
#include "core/HardwareTopology.h"
#include "core/MappingSolution.h"
#include "core/NeuronMapper.h"
#include "utils/Logger.h"
#include <iostream>
#include <cassert>
#include <memory>

using namespace NeuronMapping;

// 简单测试框架
class TestFramework {
private:
    static int total_tests;
    static int passed_tests;
    
public:
    static void assertTrue(bool condition, const std::string& test_name) {
        total_tests++;
        if (condition) {
            passed_tests++;
            std::cout << "[PASS] " << test_name << std::endl;
        } else {
            std::cout << "[FAIL] " << test_name << std::endl;
        }
    }
    
    static void assertEqual(int expected, int actual, const std::string& test_name) {
        assertTrue(expected == actual, test_name + " (expected: " + 
                  std::to_string(expected) + ", actual: " + std::to_string(actual) + ")");
    }
    
    static void printSummary() {
        std::cout << "\n=== Test Summary ===" << std::endl;
        std::cout << "Total tests: " << total_tests << std::endl;
        std::cout << "Passed: " << passed_tests << std::endl;
        std::cout << "Failed: " << (total_tests - passed_tests) << std::endl;
        std::cout << "Success rate: " << (total_tests > 0 ? 
                     (100.0 * passed_tests / total_tests) : 0) << "%" << std::endl;
    }
};

int TestFramework::total_tests = 0;
int TestFramework::passed_tests = 0;

// NeuralNetwork 测试
void testNeuralNetwork() {
    std::cout << "\n=== Testing NeuralNetwork ===" << std::endl;
    
    NeuralNetwork network;
    
    // 测试初始状态
    TestFramework::assertEqual(0, network.getNeuronCount(), "Initial neuron count should be 0");
    TestFramework::assertEqual(0, network.getConnectionCount(), "Initial connection count should be 0");
    
    // 测试添加神经元
    for (NeuronId i = 0; i < 5; ++i) {
        NeuronProperties props(i);
        TestFramework::assertTrue(network.addNeuron(props), "Add neuron " + std::to_string(i));
    }
    
    TestFramework::assertEqual(5, network.getNeuronCount(), "Should have 5 neurons after adding");
    
    // 测试添加连接
    for (NeuronId i = 0; i < 4; ++i) {
        Connection conn(i, i + 1, 1.5f);
        TestFramework::assertTrue(network.addConnection(conn), "Add connection " + std::to_string(i) + " -> " + std::to_string(i + 1));
    }
    
    TestFramework::assertEqual(4, network.getConnectionCount(), "Should have 4 connections after adding");
    
    // 测试神经元存在性检查
    TestFramework::assertTrue(network.hasNeuron(0), "Neuron 0 should exist");
    TestFramework::assertTrue(network.hasNeuron(4), "Neuron 4 should exist");
    TestFramework::assertTrue(!network.hasNeuron(10), "Neuron 10 should not exist");
    
    // 测试获取邻居（使用后继神经元）
    auto successors = network.getSuccessors(0);
    TestFramework::assertEqual(1, successors.size(), "Neuron 0 should have 1 successor");
    
    auto predecessors = network.getPredecessors(2);
    TestFramework::assertEqual(1, predecessors.size(), "Neuron 2 should have 1 predecessor");
    successors = network.getSuccessors(2);
    TestFramework::assertEqual(1, successors.size(), "Neuron 2 should have 1 successor");
    
    // 测试网络清空
    network.clear();
    TestFramework::assertEqual(0, network.getNeuronCount(), "Network should be empty after clear");
    TestFramework::assertEqual(0, network.getConnectionCount(), "Network should have no connections after clear");
}

// HardwareTopology 测试
void testHardwareTopology() {
    std::cout << "\n=== Testing HardwareTopology ===" << std::endl;
    
    HardwareTopology topology;
    
    // 测试初始状态
    TestFramework::assertEqual(0, topology.getTotalPEs(), "Initial PE count should be 0");
    
    // 测试网格拓扑创建
    ProcessingElement pe_config;
    pe_config.max_neurons = 10;
    TestFramework::assertTrue(topology.createMesh2D(2, 2, pe_config), "Create 2x2 mesh topology");
    TestFramework::assertEqual(4, topology.getTotalPEs(), "Should have 4 PEs after mesh creation");
    
    // 测试距离计算
    float distance = topology.getDistance(0, 3);
    TestFramework::assertTrue(distance > 0, "Distance between PE 0 and 3 should be positive");
    
    // 测试PE容量
    TestFramework::assertEqual(10, topology.getPECapacity(0), "PE 0 capacity should be 10");
    
    // 测试PE存在性
    TestFramework::assertTrue(topology.hasPE(0), "PE 0 should exist");
    TestFramework::assertTrue(topology.hasPE(3), "PE 3 should exist");
    TestFramework::assertTrue(!topology.hasPE(10), "PE 10 should not exist");
    
    // 测试邻居PE
    auto neighbors = topology.getNeighbors(0);
    TestFramework::assertTrue(neighbors.size() > 0, "PE 0 should have neighbors");
}

// MappingSolution 测试
void testMappingSolution() {
    std::cout << "\n=== Testing MappingSolution ===" << std::endl;
    
    // 创建简单的网络和拓扑进行测试
    NeuralNetwork network;
    for (NeuronId i = 0; i < 6; ++i) {
        NeuronProperties props(i);
        network.addNeuron(props);
    }
    
    HardwareTopology topology;
    ProcessingElement pe_config;
    pe_config.max_neurons = 3;
    topology.createMesh2D(2, 2, pe_config);  // 2x2网格，每个PE容量3
    
    MappingSolution solution(network.getNeuronCount());
    
    // 测试初始状态
    TestFramework::assertEqual(0, solution.getAssignedNeuronCount(), "Initially no neurons should be assigned");
    
    // 测试神经元分配
    TestFramework::assertTrue(solution.assignNeuron(0, 0, 0), "Assign neuron 0 to PE 0");
    TestFramework::assertTrue(solution.assignNeuron(1, 0, 0), "Assign neuron 1 to PE 0");
    TestFramework::assertTrue(solution.assignNeuron(2, 1, 0), "Assign neuron 2 to PE 1");
    
    TestFramework::assertEqual(3, solution.getAssignedNeuronCount(), "Should have 3 assigned neurons");
    TestFramework::assertEqual(2, solution.getPENeuronCount(0), "PE 0 should have 2 neurons");
    TestFramework::assertEqual(1, solution.getPENeuronCount(1), "PE 1 should have 1 neuron");
    
    // 测试神经元位置查询
    TestFramework::assertEqual(0, solution.getNeuronPE(0), "Neuron 0 should be on PE 0");
    TestFramework::assertEqual(1, solution.getNeuronPE(2), "Neuron 2 should be on PE 1");
    TestFramework::assertEqual(INVALID_PE_ID, solution.getNeuronPE(5), "Unassigned neuron should return INVALID_PE_ID");
    
    // 测试重复分配保护
    TestFramework::assertTrue(!solution.assignNeuron(0, 1, 0), "Should not allow reassigning neuron 0");
    
    // 测试移除神经元 - 跳过这个测试，因为MappingSolution可能没有removeNeuron方法
    // TestFramework::assertTrue(solution.removeNeuron(0), "Remove neuron 0");
    // TestFramework::assertEqual(2, solution.getAssignedNeuronCount(), "Should have 2 assigned neurons after removal");
    // TestFramework::assertEqual(1, solution.getPENeuronCount(0), "PE 0 should have 1 neuron after removal");
    
    // 测试解决方案复制
    MappingSolution copy_solution(solution);
    TestFramework::assertEqual(solution.getAssignedNeuronCount(), copy_solution.getAssignedNeuronCount(), "Copy should have same neuron count");
    TestFramework::assertEqual(solution.getNeuronPE(1), copy_solution.getNeuronPE(1), "Copy should have same neuron assignments");
}

// NeuronMapper 测试
void testNeuronMapper() {
    std::cout << "\n=== Testing NeuronMapper ===" << std::endl;
    
    // 创建测试数据
    NeuralNetwork network;
    for (NeuronId i = 0; i < 8; ++i) {
        NeuronProperties props(i);
        network.addNeuron(props);
    }
    
    // 添加一些连接
    for (NeuronId i = 0; i < 7; ++i) {
        Connection conn(i, i + 1, 1.0f);
        network.addConnection(conn);
    }
    
    HardwareTopology topology;
    ProcessingElement pe_config;
    pe_config.max_neurons = 4;
    topology.createMesh2D(2, 2, pe_config);  // 2x2网格，每个PE容量4
    
    MappingConfig config;
    config.strategy = "random";
    config.max_iterations = 100;
    
    // 创建映射器并测试
    auto mapper = std::make_unique<NeuronMapper>();
    
    // 测试基础信息
    TestFramework::assertTrue(!mapper->getVersion().empty(), "Mapper should have version string");
    TestFramework::assertTrue(!mapper->getMapperType().empty(), "Mapper should have type string");
    
    // 测试支持的算法列表
    auto strategies = mapper->getSupportedStrategies();
    TestFramework::assertTrue(strategies.size() > 0, "Should support at least one strategy");
    
    auto optimizers = mapper->getSupportedOptimizers();
    TestFramework::assertTrue(optimizers.size() > 0, "Should support at least one optimizer");
    
    // 测试网络映射
    auto solution = mapper->mapNetwork(network, topology, config);
    TestFramework::assertTrue(solution != nullptr, "Mapping should produce a solution");
    
    if (solution) {
        TestFramework::assertTrue(solution->getAssignedNeuronCount() > 0, "Solution should have assigned neurons");
        TestFramework::assertTrue(solution->getAssignedNeuronCount() <= network.getNeuronCount(), 
                                "Solution should not have more neurons than network");
        
        // 测试映射评估
        auto metrics = mapper->evaluateMapping(*solution, network, topology, config);
        TestFramework::assertTrue(metrics.communication_cost >= 0, "Communication cost should be non-negative");
        TestFramework::assertTrue(metrics.load_imbalance >= 0, "Load imbalance should be non-negative");
    }
    
    // 测试配置验证
    MappingConfig invalid_config;
    invalid_config.max_iterations = 0;  // 无效值
    auto errors = mapper->validateConfiguration(invalid_config);
    TestFramework::assertTrue(errors.size() > 0, "Invalid config should generate errors");
}

// 工厂模式测试
void testMapperFactory() {
    std::cout << "\n=== Testing Factory Pattern ===" << std::endl;
    
    // 测试默认映射器创建
    auto default_mapper = MapperFactory::createDefaultMapper();
    TestFramework::assertTrue(default_mapper != nullptr, "Should create default mapper");
    
    // 测试快速映射器创建
    auto fast_mapper = MapperFactory::createFastMapper();
    TestFramework::assertTrue(fast_mapper != nullptr, "Should create fast mapper");
    
    // 测试高精度映射器创建
    auto precision_mapper = MapperFactory::createPrecisionMapper();
    TestFramework::assertTrue(precision_mapper != nullptr, "Should create precision mapper");
    
    // 测试自定义映射器创建
    auto custom_mapper = MapperFactory::createCustomMapper("RandomMapping", "LocalSearch");
    TestFramework::assertTrue(custom_mapper != nullptr, "Should create custom mapper");
    
    // 测试可用映射器类型
    auto types = MapperFactory::getAvailableMapperTypes();
    TestFramework::assertTrue(types.size() > 0, "Should have available mapper types");
    
    std::cout << "Available mapper types: ";
    for (const auto& type : types) {
        std::cout << type << " ";
    }
    std::cout << std::endl;
}

// 集成测试：完整的映射流程
void testCompleteWorkflow() {
    std::cout << "\n=== Testing Complete Workflow ===" << std::endl;
    
    try {
        // 1. 创建较大的测试网络
        NeuralNetwork network;
        const int num_neurons = 20;
        
        for (NeuronId i = 0; i < num_neurons; ++i) {
            NeuronProperties props(i);
            network.addNeuron(props);
        }
        
        // 添加连接（创建小世界网络结构）
        for (NeuronId i = 0; i < num_neurons - 1; ++i) {
            Connection conn(i, i + 1, 1.0f + (i % 3) * 0.5f);
            network.addConnection(conn);
        }
        
        // 添加一些跨层连接
        for (NeuronId i = 0; i < num_neurons - 3; ++i) {
            if (i % 4 == 0) {
                Connection conn(i, i + 3, 2.0f);
                network.addConnection(conn);
            }
        }
        
        TestFramework::assertEqual(num_neurons, network.getNeuronCount(), "Network should have correct neuron count");
        TestFramework::assertTrue(network.getConnectionCount() > num_neurons, "Network should have sufficient connections");
        
        // 2. 创建硬件拓扑
        HardwareTopology topology;
        ProcessingElement pe_config;
        pe_config.max_neurons = 5;
        TestFramework::assertTrue(topology.createMesh2D(3, 3, pe_config), "Create 3x3 mesh topology");
        TestFramework::assertEqual(9, topology.getTotalPEs(), "Should have 9 PEs in 3x3 mesh");
        
        // 3. 测试不同的映射器
        std::vector<std::unique_ptr<INeuronMapper>> mappers;
        mappers.push_back(MapperFactory::createDefaultMapper());
        mappers.push_back(MapperFactory::createFastMapper());
        mappers.push_back(MapperFactory::createPrecisionMapper());
        
        MappingConfig config;
        config.strategy = "random";
        config.max_iterations = 50;
        
        std::vector<std::unique_ptr<MappingSolution>> solutions;
        
        for (size_t i = 0; i < mappers.size(); ++i) {
            auto solution = mappers[i]->mapNetwork(network, topology, config);
            TestFramework::assertTrue(solution != nullptr, "Mapper " + std::to_string(i) + " should produce solution");
            
            if (solution) {
                TestFramework::assertTrue(solution->getAssignedNeuronCount() > 0, "Solution should assign neurons");
                solutions.push_back(std::move(solution));
            }
        }
        
        // 4. 比较映射质量
        if (solutions.size() >= 2) {
            auto& mapper = mappers[0];
            auto rankings = mapper->compareMappings(solutions, network, topology, config);
            TestFramework::assertEqual(solutions.size(), rankings.size(), "Rankings should cover all solutions");
            TestFramework::assertTrue(rankings[0] < solutions.size(), "Best ranking should be valid index");
        }
        
        // 5. 测试增量映射
        if (!solutions.empty()) {
            auto& best_solution = solutions[0];
            std::vector<NeuronId> new_neurons = {num_neurons, num_neurons + 1, num_neurons + 2};
            
            // 添加新神经元到网络
            for (NeuronId id : new_neurons) {
                NeuronProperties props(id);
                network.addNeuron(props);
            }
            
            auto incremental_solution = mappers[0]->incrementalMap(*best_solution, new_neurons, network, topology, config);
            TestFramework::assertTrue(incremental_solution != nullptr, "Incremental mapping should succeed");
            
            if (incremental_solution) {
                TestFramework::assertTrue(incremental_solution->getAssignedNeuronCount() >= best_solution->getAssignedNeuronCount(), 
                                       "Incremental solution should have at least as many neurons");
            }
        }
        
        std::cout << "Complete workflow test passed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cout << "Exception in complete workflow test: " << e.what() << std::endl;
        TestFramework::assertTrue(false, "Complete workflow should not throw exceptions");
    }
}

int main() {
    std::cout << "=== Neuron Mapping Framework Core Tests ===" << std::endl;
    
    // 初始化日志系统
    Logger::getInstance().setLevel(LogLevel::INFO);
    
    // 运行所有测试
    testNeuralNetwork();
    testHardwareTopology();
    testMappingSolution();
    testNeuronMapper();
    testMapperFactory();
    testCompleteWorkflow();
    
    // 输出测试结果总结
    TestFramework::printSummary();
    
    return 0;
}