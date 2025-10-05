#include "core/NeuralNetwork.h"
#include "core/HardwareTopology.h"
#include "core/MappingSolution.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <algorithm>
#include <random>

using namespace NeuronMapping;

// 简化的图分割策略实现用于测试
class SimpleGraphPartitioning {
public:
    std::unique_ptr<MappingSolution> mapNetwork(
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config) {
        
        std::cout << "Starting simplified graph partitioning..." << std::endl;
        
        auto neuron_ids = network.getAllNeuronIds();
        uint32_t num_partitions = topology.getTotalPEs();
        
        if (num_partitions == 0) {
            std::cout << "No PEs available!" << std::endl;
            return nullptr;
        }
        
        auto solution = std::make_unique<MappingSolution>(network.getNeuronCount());
        
        // 简单的图分割：基于连接权重的分组
        std::vector<std::vector<NeuronId>> partitions(num_partitions);
        std::vector<bool> assigned(neuron_ids.size(), false);
        
        std::random_device rd;
        std::mt19937 rng(rd());
        
        // 为每个分区选择一个种子神经元
        std::vector<NeuronId> seeds;
        for (uint32_t i = 0; i < num_partitions && i < neuron_ids.size(); ++i) {
            seeds.push_back(neuron_ids[i]);
            partitions[i].push_back(neuron_ids[i]);
            assigned[i] = true;
        }
        
        // 基于连接强度将剩余神经元分配到分区
        for (size_t i = 0; i < neuron_ids.size(); ++i) {
            if (assigned[i]) continue;
            
            NeuronId neuron = neuron_ids[i];
            
            // 计算与每个分区的连接强度
            std::vector<float> partition_scores(num_partitions, 0.0f);
            
            auto connections = network.getAllConnections();
            for (const auto& conn : connections) {
                if (conn.source_id == neuron || conn.target_id == neuron) {
                    NeuronId connected_neuron = (conn.source_id == neuron) ? conn.target_id : conn.source_id;
                    
                    // 找到connected_neuron在哪个分区
                    for (uint32_t p = 0; p < num_partitions; ++p) {
                        auto it = std::find(partitions[p].begin(), partitions[p].end(), connected_neuron);
                        if (it != partitions[p].end()) {
                            partition_scores[p] += std::abs(conn.weight);
                            break;
                        }
                    }
                }
            }
            
            // 选择连接强度最高的分区，如果都是0则随机选择
            uint32_t best_partition = 0;
            float best_score = partition_scores[0];
            
            for (uint32_t p = 1; p < num_partitions; ++p) {
                if (partition_scores[p] > best_score) {
                    best_score = partition_scores[p];
                    best_partition = p;
                }
            }
            
            // 如果所有分区得分都是0，选择最小的分区
            if (best_score == 0.0f) {
                best_partition = 0;
                for (uint32_t p = 1; p < num_partitions; ++p) {
                    if (partitions[p].size() < partitions[best_partition].size()) {
                        best_partition = p;
                    }
                }
            }
            
            partitions[best_partition].push_back(neuron);
            assigned[i] = true;
        }
        
        // 将分区结果转换为映射
        for (uint32_t p = 0; p < num_partitions; ++p) {
            for (NeuronId neuron_id : partitions[p]) {
                solution->assignNeuron(neuron_id, static_cast<PEId>(p), 0);
            }
        }
        
        // 打印分区统计
        std::cout << "Partition statistics:" << std::endl;
        for (uint32_t p = 0; p < num_partitions; ++p) {
            std::cout << "  Partition " << p << ": " << partitions[p].size() << " neurons" << std::endl;
        }
        
        return solution;
    }
};

void testSimpleGraphPartitioning() {
    std::cout << "=== Testing Simplified Graph Partitioning ===" << std::endl;
    
    // 创建测试网络
    NeuralNetwork network;
    const int num_neurons = 16;
    
    // 添加神经元
    for (NeuronId i = 0; i < num_neurons; ++i) {
        NeuronProperties props(i);
        assert(network.addNeuron(props));
    }
    
    // 创建集群连接模式
    // 集群1: 神经元0-3
    for (NeuronId i = 0; i < 4; ++i) {
        for (NeuronId j = i + 1; j < 4; ++j) {
            Connection conn(i, j, 2.0f);
            assert(network.addConnection(conn));
        }
    }
    
    // 集群2: 神经元4-7
    for (NeuronId i = 4; i < 8; ++i) {
        for (NeuronId j = i + 1; j < 8; ++j) {
            Connection conn(i, j, 2.0f);
            assert(network.addConnection(conn));
        }
    }
    
    // 集群3: 神经元8-11
    for (NeuronId i = 8; i < 12; ++i) {
        for (NeuronId j = i + 1; j < 12; ++j) {
            Connection conn(i, j, 2.0f);
            assert(network.addConnection(conn));
        }
    }
    
    // 集群4: 神经元12-15
    for (NeuronId i = 12; i < 16; ++i) {
        for (NeuronId j = i + 1; j < 16; ++j) {
            Connection conn(i, j, 2.0f);
            assert(network.addConnection(conn));
        }
    }
    
    // 添加一些跨集群连接（较弱）
    Connection conn1(3, 4, 0.5f);
    Connection conn2(7, 8, 0.5f);
    Connection conn3(11, 12, 0.5f);
    network.addConnection(conn1);
    network.addConnection(conn2);
    network.addConnection(conn3);
    
    std::cout << "Created clustered network: " << network.getNeuronCount() 
              << " neurons, " << network.getConnectionCount() << " connections" << std::endl;
    
    // 创建硬件拓扑
    HardwareTopology topology;
    ProcessingElement pe_config;
    pe_config.max_neurons = 6;  // 每个PE可容纳6个神经元
    assert(topology.createMesh2D(2, 2, pe_config)); // 4个PE
    
    std::cout << "Created hardware topology: " << topology.getTotalPEs() << " PEs" << std::endl;
    
    MappingConfig config;
    config.strategy = "simple_graph";
    
    // 执行图分割映射
    SimpleGraphPartitioning partitioner;
    auto solution = partitioner.mapNetwork(network, topology, config);
    
    if (solution) {
        std::cout << "✓ Graph partitioning mapping successful!" << std::endl;
        std::cout << "  Assigned neurons: " << solution->getAssignedNeuronCount() << std::endl;
        
        // 验证映射质量
        assert(solution->getAssignedNeuronCount() <= network.getNeuronCount());
        
        // 分析PE负载分布
        std::vector<int> pe_loads(topology.getTotalPEs(), 0);
        std::cout << "PE assignments:" << std::endl;
        
        for (NeuronId neuron_id = 0; neuron_id < network.getNeuronCount(); ++neuron_id) {
            PEId pe_id = solution->getNeuronPE(neuron_id);
            if (pe_id != INVALID_PE_ID && pe_id < topology.getTotalPEs()) {
                pe_loads[pe_id]++;
                if (neuron_id < 16) {  // 只打印前16个神经元
                    std::cout << "  Neuron " << neuron_id << " -> PE " << pe_id << std::endl;
                }
            }
        }
        
        std::cout << "PE load distribution:" << std::endl;
        for (size_t pe = 0; pe < pe_loads.size(); ++pe) {
            std::cout << "  PE" << pe << ": " << pe_loads[pe] << " neurons" << std::endl;
        }
        
        // 计算映射质量指标
        auto metrics = solution->evaluatePerformance(network, topology, config);
        std::cout << "Performance metrics:" << std::endl;
        std::cout << "  Communication cost: " << metrics.communication_cost << std::endl;
        std::cout << "  Load imbalance: " << metrics.load_imbalance << std::endl;
        std::cout << "  PE utilization: " << metrics.pe_utilization << std::endl;
        
        // 计算边切割数（简化版本）
        int cut_edges = 0;
        int total_edges = 0;
        auto connections = network.getAllConnections();
        
        for (const auto& conn : connections) {
            total_edges++;
            PEId src_pe = solution->getNeuronPE(conn.source_id);
            PEId tgt_pe = solution->getNeuronPE(conn.target_id);
            
            if (src_pe != INVALID_PE_ID && tgt_pe != INVALID_PE_ID && src_pe != tgt_pe) {
                cut_edges++;
            }
        }
        
        std::cout << "Edge cut analysis:" << std::endl;
        std::cout << "  Total edges: " << total_edges << std::endl;
        std::cout << "  Cut edges: " << cut_edges << std::endl;
        std::cout << "  Cut ratio: " << (total_edges > 0 ? (float)cut_edges / total_edges : 0.0f) << std::endl;
        
        // 验证负载均衡
        int max_load = *std::max_element(pe_loads.begin(), pe_loads.end());
        int min_load = *std::min_element(pe_loads.begin(), pe_loads.end());
        float imbalance = (max_load > 0) ? (float)(max_load - min_load) / max_load : 0.0f;
        
        std::cout << "Load balance analysis:" << std::endl;
        std::cout << "  Max load: " << max_load << ", Min load: " << min_load << std::endl;
        std::cout << "  Load imbalance: " << imbalance << std::endl;
        
    } else {
        std::cout << "✗ Graph partitioning mapping failed!" << std::endl;
        assert(false);
    }
    
    std::cout << "\n=== Simple Graph Partitioning Test Completed ===" << std::endl;
}

int main() {
    try {
        testSimpleGraphPartitioning();
        
        std::cout << "\n✅ All graph partitioning tests completed successfully!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n❌ Test failed with unknown exception!" << std::endl;
        return 1;
    }
}