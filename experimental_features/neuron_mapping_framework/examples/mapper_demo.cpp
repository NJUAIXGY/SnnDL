#include "factories/MapperFactory.h"
#include "core/NeuralNetwork.h"
#include "core/HardwareTopology.h"
#include "utils/Logger.h"
#include <iostream>

using namespace neuron_mapping;
using namespace NeuronMapping;

int main() {
    std::cout << "=== Neuron Mapper Factory Demo ===" << std::endl;
    
    // 创建一个简单的神经网络
    NeuralNetwork network;
    
    // 添加一些神经元
    for (NeuronId i = 0; i < 10; ++i) {
        NeuronProperties props(i);
        network.addNeuron(props);
    }
    
    // 添加一些连接
    for (NeuronId i = 0; i < 9; ++i) {
        Connection conn(i, i + 1, 1.0f);
        network.addConnection(conn);
    }
    
    std::cout << "Created neural network with " << network.getNeuronCount() 
              << " neurons and " << network.getConnectionCount() << " connections" << std::endl;
    
    // 创建硬件拓扑
    HardwareTopology topology;
    
    // 添加4个PE
    for (PEId pe_id = 0; pe_id < 4; ++pe_id) {
        ProcessingElement pe;
        pe.pe_id = pe_id;
        pe.capacity = 5;  // 每个PE可以容纳5个神经元
        topology.addPE(pe);
    }
    
    std::cout << "Created hardware topology with " << topology.getTotalPEs() << " PEs" << std::endl;
    
    // 创建映射配置
    MappingConfig config;
    config.strategy = "random";
    config.max_iterations = 100;
    
    std::cout << "\n=== Testing Different Mapper Types ===" << std::endl;
    
    try {
        // 1. 默认映射器
        std::cout << "\n1. Creating Default Mapper..." << std::endl;
        auto default_mapper = NeuronMapperFactory::createDefaultMapper();
        if (default_mapper) {
            std::cout << "   Mapper Type: " << default_mapper->getMapperType() << std::endl;
            std::cout << "   Version: " << default_mapper->getVersion() << std::endl;
            
            auto solution = default_mapper->mapNetwork(network, topology, config);
            if (solution) {
                std::cout << "   Mapping successful! Assigned " 
                         << solution->getAssignedNeuronCount() << " neurons" << std::endl;
            }
        }
        
        // 2. 快速映射器
        std::cout << "\n2. Creating Fast Mapper..." << std::endl;
        auto fast_mapper = NeuronMapperFactory::createFastMapper();
        if (fast_mapper) {
            std::cout << "   Mapper Info: " 
                     << NeuronMapperFactory::getMapperInfo(MapperType::FAST) << std::endl;
        }
        
        // 3. 精度映射器
        std::cout << "\n3. Creating Precision Mapper..." << std::endl;
        auto precision_mapper = NeuronMapperFactory::createPrecisionMapper();
        if (precision_mapper) {
            std::cout << "   Mapper Info: " 
                     << NeuronMapperFactory::getMapperInfo(MapperType::PRECISION) << std::endl;
        }
        
        // 4. 自定义映射器
        std::cout << "\n4. Creating Custom Mapper..." << std::endl;
        auto custom_mapper = NeuronMapperFactory::createCustomMapper(
            "GreedyMapping", "HillClimbing", "Comprehensive");
        
        if (custom_mapper) {
            std::cout << "   Custom mapper created successfully" << std::endl;
            std::cout << "   Supported strategies: ";
            auto strategies = custom_mapper->getSupportedStrategies();
            for (const auto& strategy : strategies) {
                std::cout << strategy << " ";
            }
            std::cout << std::endl;
        }
        
        // 5. 组件工厂测试
        std::cout << "\n5. Testing Component Factory..." << std::endl;
        std::cout << "   Available Strategies: ";
        auto strategies = ComponentFactory::getAvailableStrategies();
        for (const auto& strategy : strategies) {
            std::cout << strategy << " ";
        }
        std::cout << std::endl;
        
        std::cout << "   Available Optimizers: ";
        auto optimizers = ComponentFactory::getAvailableOptimizers();
        for (const auto& optimizer : optimizers) {
            std::cout << optimizer << " ";
        }
        std::cout << std::endl;
        
        std::cout << "   Available Evaluators: ";
        auto evaluators = ComponentFactory::getAvailableEvaluators();
        for (const auto& evaluator : evaluators) {
            std::cout << evaluator << " ";
        }
        std::cout << std::endl;
        
        // 6. 配置验证测试
        std::cout << "\n6. Testing Configuration Validation..." << std::endl;
        MapperConfiguration test_config;
        test_config.max_optimization_iterations = 0;  // 无效值
        test_config.convergence_threshold = -1.0;     // 无效值
        
        auto errors = NeuronMapperFactory::validateConfiguration(test_config);
        if (!errors.empty()) {
            std::cout << "   Configuration errors found:" << std::endl;
            for (const auto& error : errors) {
                std::cout << "     - " << error << std::endl;
            }
        }
        
        std::cout << "\n=== Demo completed successfully! ===" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error during demo: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}