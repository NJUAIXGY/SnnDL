#include "factories/MapperFactory.h"
#include "core/NeuralNetwork.h"
#include "core/HardwareTopology.h"
#include <iostream>
#include <cassert>

using namespace NeuronMapping;

void testFactoryPattern() {
    std::cout << "=== Testing Factory Pattern ===" << std::endl;
    
    // 创建测试数据
    NeuralNetwork network;
    for (NeuronId i = 0; i < 8; ++i) {
        NeuronProperties props(i);
        assert(network.addNeuron(props));
    }
    
    for (NeuronId i = 0; i < 7; ++i) {
        Connection conn(i, i + 1, 1.0f);
        assert(network.addConnection(conn));
    }
    
    HardwareTopology topology;
    ProcessingElement pe_config;
    pe_config.max_neurons = 4;
    assert(topology.createMesh2D(2, 2, pe_config));
    
    MappingConfig config;
    config.strategy = "random";
    config.max_iterations = 50;
    
    std::cout << "Created test network: " << network.getNeuronCount() 
              << " neurons, " << network.getConnectionCount() << " connections" << std::endl;
    std::cout << "Created test topology: " << topology.getTotalPEs() << " PEs" << std::endl;
    
    // 1. 测试默认映射器创建
    std::cout << "\n1. Testing Default Mapper..." << std::endl;
    auto default_mapper = MapperFactory::createDefaultMapper();
    assert(default_mapper != nullptr);
    std::cout << "  ✓ Default mapper created: " << default_mapper->getMapperType() << std::endl;
    std::cout << "  ✓ Version: " << default_mapper->getVersion() << std::endl;
    
    auto solution = default_mapper->mapNetwork(network, topology, config);
    assert(solution != nullptr);
    std::cout << "  ✓ Mapping successful: " << solution->getAssignedNeuronCount() << " neurons assigned" << std::endl;
    
    // 2. 测试快速映射器创建
    std::cout << "\n2. Testing Fast Mapper..." << std::endl;
    auto fast_mapper = MapperFactory::createFastMapper();
    assert(fast_mapper != nullptr);
    std::cout << "  ✓ Fast mapper created: " << fast_mapper->getMapperType() << std::endl;
    
    auto fast_solution = fast_mapper->mapNetwork(network, topology, config);
    assert(fast_solution != nullptr);
    std::cout << "  ✓ Fast mapping successful: " << fast_solution->getAssignedNeuronCount() << " neurons assigned" << std::endl;
    
    // 3. 测试高精度映射器创建
    std::cout << "\n3. Testing Precision Mapper..." << std::endl;
    auto precision_mapper = MapperFactory::createPrecisionMapper();
    assert(precision_mapper != nullptr);
    std::cout << "  ✓ Precision mapper created: " << precision_mapper->getMapperType() << std::endl;
    
    auto precision_solution = precision_mapper->mapNetwork(network, topology, config);
    assert(precision_solution != nullptr);
    std::cout << "  ✓ Precision mapping successful: " << precision_solution->getAssignedNeuronCount() << " neurons assigned" << std::endl;
    
    // 4. 测试自定义映射器创建
    std::cout << "\n4. Testing Custom Mapper..." << std::endl;
    auto custom_mapper = MapperFactory::createCustomMapper("RandomMapping", "LocalSearch");
    assert(custom_mapper != nullptr);
    std::cout << "  ✓ Custom mapper created" << std::endl;
    
    auto strategies = custom_mapper->getSupportedStrategies();
    std::cout << "  ✓ Supported strategies: ";
    for (const auto& strategy : strategies) {
        std::cout << strategy << " ";
    }
    std::cout << std::endl;
    
    auto optimizers = custom_mapper->getSupportedOptimizers();
    std::cout << "  ✓ Supported optimizers: ";
    for (const auto& optimizer : optimizers) {
        std::cout << optimizer << " ";
    }
    std::cout << std::endl;
    
    auto custom_solution = custom_mapper->mapNetwork(network, topology, config);
    assert(custom_solution != nullptr);
    std::cout << "  ✓ Custom mapping successful: " << custom_solution->getAssignedNeuronCount() << " neurons assigned" << std::endl;
    
    // 5. 测试可用映射器类型
    std::cout << "\n5. Testing Available Mapper Types..." << std::endl;
    auto types = MapperFactory::getAvailableMapperTypes();
    assert(types.size() > 0);
    std::cout << "  ✓ Available mapper types (" << types.size() << "): ";
    for (const auto& type : types) {
        std::cout << type << " ";
    }
    std::cout << std::endl;
    
    // 6. 测试映射质量比较
    std::cout << "\n6. Testing Mapping Quality Comparison..." << std::endl;
    std::vector<std::unique_ptr<MappingSolution>> solutions;
    solutions.push_back(std::move(solution));
    solutions.push_back(std::move(fast_solution));
    solutions.push_back(std::move(precision_solution));
    
    if (solutions.size() >= 2) {
        auto rankings = default_mapper->compareMappings(solutions, network, topology, config);
        assert(rankings.size() == solutions.size());
        std::cout << "  ✓ Solution rankings: ";
        for (size_t rank : rankings) {
            std::cout << rank << " ";
        }
        std::cout << "(0=best)" << std::endl;
    }
    
    std::cout << "\n=== All Factory Tests Passed! ===" << std::endl;
}

int main() {
    try {
        testFactoryPattern();
        std::cout << "\n✅ All factory tests completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n❌ Test failed with unknown exception!" << std::endl;
        return 1;
    }
}