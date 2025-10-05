#include "core/NeuralNetwork.h"
#include "core/HardwareTopology.h"
#include "strategies/GraphPartitioningStrategy.h"
#include "routing/RoutingTableGenerator.h"
#include <fstream>
#include <iostream>
#include <filesystem>

using namespace NeuronMapping;

static void write_file(const std::string& path, const std::string& data) {
    std::ofstream ofs(path, std::ios::out | std::ios::trunc);
    ofs << data;
}

int main() {
    // 1) 构造一个小网络（32神经元，4层，每层8个，层间全连接）
    NeuralNetwork net;
    const uint32_t num_neurons = 32;
    for (uint32_t i = 0; i < num_neurons; ++i) {
        NeuronProperties np(i);
        net.addNeuron(np);
    }
    const uint32_t layers = 4, per_layer = num_neurons / layers;
    for (uint32_t L = 0; L + 1 < layers; ++L) {
        for (uint32_t i = 0; i < per_layer; ++i) {
            for (uint32_t j = 0; j < per_layer; ++j) {
                net.addConnection(Connection(L * per_layer + i, (L + 1) * per_layer + j, 0.1f));
            }
        }
    }

    // 2) 构造拓扑（4x2 mesh，8 PEs）
    HardwareTopology topo;
    ProcessingElement pe_cfg; pe_cfg.max_neurons = 8; pe_cfg.memory_capacity = 1024 * 1024;
    topo.createMesh2D(2, 4, pe_cfg);

    // 3) 基于图分割生成映射
    neuron_mapping::GraphPartitioningStrategy partitioner;
    MappingConfig cfg; // 使用默认配置
    auto solution = partitioner.mapNetwork(net, topo, cfg);

    // 4) 生成路由表（启用多播+前缀聚合）
    RoutingGenerationConfig rcfg;
    rcfg.enable_multicast = true;
    rcfg.enable_compression = true;
    rcfg.compression = CompressionMethod::PREFIX_AGGREGATION;
    RoutingTableGenerator gen(rcfg);
    // 5) 使用公共API导出产物
    gen.exportArtifacts(net, topo, *solution, "export", rcfg);

    std::cout << "Artifacts written to ./export\n";
    return 0;
}
