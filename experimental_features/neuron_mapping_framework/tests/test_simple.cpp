#include "core/NeuralNetwork.h"
#include "core/HardwareTopology.h"
#include "core/MappingSolution.h"
#include <iostream>
#include <cassert>

using namespace NeuronMapping;

void testBasicFunctionality() {
    std::cout << "=== Testing Basic Framework Functionality ===" << std::endl;
    
    // 1. 测试NeuralNetwork基本功能
    std::cout << "Testing NeuralNetwork..." << std::endl;
    NeuralNetwork network;
    
    // 添加一些神经元
    for (NeuronId i = 0; i < 5; ++i) {
        NeuronProperties props(i);
        assert(network.addNeuron(props));
    }
    assert(network.getNeuronCount() == 5);
    std::cout << "  ✓ Added " << network.getNeuronCount() << " neurons" << std::endl;
    
    // 添加一些连接
    for (NeuronId i = 0; i < 4; ++i) {
        Connection conn(i, i + 1, 1.0f);
        assert(network.addConnection(conn));
    }
    assert(network.getConnectionCount() == 4);
    std::cout << "  ✓ Added " << network.getConnectionCount() << " connections" << std::endl;
    
    // 2. 测试HardwareTopology基本功能
    std::cout << "Testing HardwareTopology..." << std::endl;
    HardwareTopology topology;
    
    ProcessingElement pe_config;
    pe_config.max_neurons = 10;
    
    assert(topology.createMesh2D(2, 2, pe_config));
    assert(topology.getTotalPEs() == 4);
    std::cout << "  ✓ Created 2x2 mesh topology with " << topology.getTotalPEs() << " PEs" << std::endl;
    
    // 3. 测试MappingSolution基本功能
    std::cout << "Testing MappingSolution..." << std::endl;
    MappingSolution solution(network.getNeuronCount());
    
    // 分配一些神经元
    assert(solution.assignNeuron(0, 0, 0));
    assert(solution.assignNeuron(1, 0, 0));
    assert(solution.assignNeuron(2, 1, 0));
    assert(solution.getAssignedNeuronCount() == 3);
    std::cout << "  ✓ Assigned " << solution.getAssignedNeuronCount() << " neurons to PEs" << std::endl;
    
    // 验证分配
    assert(solution.getNeuronPE(0) == 0);
    assert(solution.getNeuronPE(1) == 0);
    assert(solution.getNeuronPE(2) == 1);
    std::cout << "  ✓ Neuron assignments verified" << std::endl;
    
    // 4. 测试性能度量计算
    std::cout << "Testing performance evaluation..." << std::endl;
    MappingConfig config;
    auto metrics = solution.evaluatePerformance(network, topology, config);
    
    std::cout << "  ✓ Communication cost: " << metrics.communication_cost << std::endl;
    std::cout << "  ✓ Load imbalance: " << metrics.load_imbalance << std::endl;
    std::cout << "  ✓ PE utilization: " << metrics.pe_utilization << std::endl;
    
    std::cout << "\n=== All Basic Tests Passed! ===" << std::endl;
}

int main() {
    try {
        testBasicFunctionality();
        std::cout << "\n✅ All tests completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n❌ Test failed with unknown exception!" << std::endl;
        return 1;
    }
}