#include "algorithms/CommunityDetector.h"
#include "core/NeuralNetwork.h"
#include <iostream>
#include <cassert>

using namespace NeuronMapping;
using namespace neuron_mapping;

void testBasicCommunityDetection() {
    std::cout << "=== Testing Basic Community Detection ===\n";
    
    // 创建具有明显社区结构的测试网络
    NeuralNetwork network;
    const int num_neurons = 12;
    
    // 添加神经元
    for (NeuronId i = 0; i < num_neurons; ++i) {
        NeuronProperties props(i);
        assert(network.addNeuron(props));
    }
    
    // 创建两个密集连接的社区
    // 社区1: 神经元0-5
    for (NeuronId i = 0; i < 6; ++i) {
        for (NeuronId j = i + 1; j < 6; ++j) {
            Connection conn(i, j, 2.0f);  // 社区内强连接
            assert(network.addConnection(conn));
        }
    }
    
    // 社区2: 神经元6-11
    for (NeuronId i = 6; i < 12; ++i) {
        for (NeuronId j = i + 1; j < 12; ++j) {
            Connection conn(i, j, 2.0f);  // 社区内强连接
            assert(network.addConnection(conn));
        }
    }
    
    // 社区间的弱连接
    Connection inter_conn1(2, 8, 0.3f);
    Connection inter_conn2(4, 9, 0.3f);
    network.addConnection(inter_conn1);
    network.addConnection(inter_conn2);
    
    std::cout << "Created network with clear community structure: " 
              << network.getNeuronCount() << " neurons, " 
              << network.getConnectionCount() << " connections\n";
    
    // 创建社区检测器
    CommunityDetector detector(CommunityDetector::Algorithm::LOUVAIN);
    
    std::cout << "Detector: " << detector.getAlgorithmName() << "\n";
    std::cout << "Description: " << detector.getDescription() << "\n";
    
    // 执行社区检测
    auto structure = detector.detectCommunities(network);
    
    if (structure) {
        std::cout << "✓ Community detection successful!\n";
        std::cout << "  Number of communities: " << structure->num_communities << "\n";
        std::cout << "  Modularity: " << structure->modularity << "\n";
        
        // 验证结果
        assert(structure->num_communities > 0);
        assert(structure->communities.size() == structure->num_communities);
        
        // 检查社区分配
        std::cout << "Community assignments:\n";
        for (const auto& community : structure->communities) {
            std::cout << "  Community " << community.community_id << ": ";
            for (NeuronId neuron : community.neurons) {
                std::cout << neuron << " ";
            }
            std::cout << "| Density=" << community.getDensity() 
                      << ", Conductance=" << community.getConductance() << "\n";
        }
        
        // 评估社区质量
        float modularity_score = detector.evaluateCommunityQuality(
            *structure, network, CommunityDetector::QualityMetric::MODULARITY);
        std::cout << "  Community quality (modularity): " << modularity_score << "\n";
        
    } else {
        std::cout << "✗ Community detection failed!\n";
        assert(false);
    }
    
    std::cout << "\n=== Basic Community Detection Test Completed ===\n";
}

void testDifferentAlgorithms() {
    std::cout << "\n=== Testing Different Community Detection Algorithms ===\n";
    
    // 创建简单网络
    NeuralNetwork network;
    const int num_neurons = 16;
    
    for (NeuronId i = 0; i < num_neurons; ++i) {
        NeuronProperties props(i);
        network.addNeuron(props);
    }
    
    // 创建环形网络
    for (NeuronId i = 0; i < num_neurons; ++i) {
        Connection conn(i, (i + 1) % num_neurons, 1.0f);
        network.addConnection(conn);
    }
    
    // 添加一些跨越连接形成社区
    for (NeuronId i = 0; i < num_neurons; i += 4) {
        Connection conn1(i, (i + 2) % num_neurons, 1.5f);
        Connection conn2(i + 1, (i + 3) % num_neurons, 1.5f);
        network.addConnection(conn1);
        network.addConnection(conn2);
    }
    
    std::cout << "Created ring network: " << network.getNeuronCount() 
              << " neurons, " << network.getConnectionCount() << " connections\n";
    
    std::vector<CommunityDetector::Algorithm> algorithms = {
        CommunityDetector::Algorithm::LOUVAIN,
        CommunityDetector::Algorithm::LABEL_PROPAGATION,
        CommunityDetector::Algorithm::MODULARITY_MAXIMIZATION,
        CommunityDetector::Algorithm::SPECTRAL_CLUSTERING
    };
    
    std::vector<std::string> algorithm_names = {
        "Louvain", "Label Propagation", "Modularity Maximization", "Spectral Clustering"
    };
    
    for (size_t i = 0; i < algorithms.size(); ++i) {
        std::cout << "\n--- Testing " << algorithm_names[i] << " Algorithm ---\n";
        
        CommunityDetector detector(algorithms[i]);
        detector.setMaxIterations(50);
        
        auto structure = detector.detectCommunities(network);
        
        if (structure) {
            std::cout << "✓ " << algorithm_names[i] << " completed\n";
            std::cout << "  Communities found: " << structure->num_communities << "\n";
            std::cout << "  Modularity: " << structure->modularity << "\n";
            
            // 计算社区大小分布
            std::vector<size_t> community_sizes;
            for (const auto& community : structure->communities) {
                community_sizes.push_back(community.neurons.size());
            }
            
            std::cout << "  Community sizes: ";
            for (size_t size : community_sizes) {
                std::cout << size << " ";
            }
            std::cout << "\n";
            
        } else {
            std::cout << "✗ " << algorithm_names[i] << " failed\n";
        }
    }
    
    std::cout << "\n=== Algorithm Comparison Test Completed ===\n";
}

void testCommunityPostProcessing() {
    std::cout << "\n=== Testing Community Post-Processing ===\n";
    
    // 创建具有多个小社区的网络
    NeuralNetwork network;
    const int num_neurons = 20;
    
    for (NeuronId i = 0; i < num_neurons; ++i) {
        NeuronProperties props(i);
        network.addNeuron(props);
    }
    
    // 创建多个小社区
    // 社区1: 0-3
    for (NeuronId i = 0; i < 4; ++i) {
        for (NeuronId j = i + 1; j < 4; ++j) {
            Connection conn(i, j, 1.5f);
            network.addConnection(conn);
        }
    }
    
    // 社区2: 4-7
    for (NeuronId i = 4; i < 8; ++i) {
        for (NeuronId j = i + 1; j < 8; ++j) {
            Connection conn(i, j, 1.5f);
            network.addConnection(conn);
        }
    }
    
    // 社区3: 8-11 (较大社区)
    for (NeuronId i = 8; i < 12; ++i) {
        for (NeuronId j = i + 1; j < 12; ++j) {
            Connection conn(i, j, 1.5f);
            network.addConnection(conn);
        }
    }
    
    // 社区4: 12-19 (很大社区)
    for (NeuronId i = 12; i < 20; ++i) {
        for (NeuronId j = i + 1; j < 20; ++j) {
            Connection conn(i, j, 1.5f);
            network.addConnection(conn);
        }
    }
    
    // 社区间连接
    network.addConnection(Connection(3, 4, 0.2f));
    network.addConnection(Connection(7, 8, 0.2f));
    network.addConnection(Connection(11, 12, 0.2f));
    
    std::cout << "Created network with varied community sizes\n";
    
    CommunityDetector detector(CommunityDetector::Algorithm::LOUVAIN);
    auto original_structure = detector.detectCommunities(network);
    
    if (original_structure) {
        std::cout << "Original structure: " << original_structure->num_communities 
                  << " communities\n";
        
        // 测试社区合并
        std::cout << "\nTesting community merging...\n";
        auto merged_structure = detector.mergeSimilarCommunities(
            *original_structure, network, 0.1f);
        
        if (merged_structure) {
            std::cout << "✓ Community merging completed\n";
            std::cout << "  Communities after merging: " 
                      << merged_structure->num_communities << "\n";
        }
        
        // 测试社区分割
        std::cout << "\nTesting community splitting...\n";
        auto split_structure = detector.splitLargeCommunities(
            *original_structure, network, 6);  // 最大社区大小为6
        
        if (split_structure) {
            std::cout << "✓ Community splitting completed\n";
            std::cout << "  Communities after splitting: " 
                      << split_structure->num_communities << "\n";
        }
        
    } else {
        std::cout << "✗ Initial community detection failed\n";
    }
    
    std::cout << "\n=== Community Post-Processing Test Completed ===\n";
}

void testQualityMetrics() {
    std::cout << "\n=== Testing Community Quality Metrics ===\n";
    
    // 创建简单测试网络
    NeuralNetwork network;
    for (NeuronId i = 0; i < 8; ++i) {
        NeuronProperties props(i);
        network.addNeuron(props);
    }
    
    // 创建两个清晰的社区
    for (NeuronId i = 0; i < 4; ++i) {
        for (NeuronId j = i + 1; j < 4; ++j) {
            network.addConnection(Connection(i, j, 1.0f));
        }
    }
    
    for (NeuronId i = 4; i < 8; ++i) {
        for (NeuronId j = i + 1; j < 8; ++j) {
            network.addConnection(Connection(i, j, 1.0f));
        }
    }
    
    // 社区间弱连接
    network.addConnection(Connection(1, 6, 0.2f));
    
    CommunityDetector detector(CommunityDetector::Algorithm::LOUVAIN);
    auto structure = detector.detectCommunities(network);
    
    if (structure) {
        std::cout << "Testing different quality metrics:\n";
        
        std::vector<CommunityDetector::QualityMetric> metrics = {
            CommunityDetector::QualityMetric::MODULARITY,
            CommunityDetector::QualityMetric::COVERAGE,
            CommunityDetector::QualityMetric::SILHOUETTE
        };
        
        std::vector<std::string> metric_names = {
            "Modularity", "Coverage", "Silhouette Score"
        };
        
        for (size_t i = 0; i < metrics.size(); ++i) {
            float score = detector.evaluateCommunityQuality(*structure, network, metrics[i]);
            std::cout << "  " << metric_names[i] << ": " << score << "\n";
        }
        
        std::cout << "✓ Quality metrics evaluation completed\n";
    }
    
    std::cout << "\n=== Quality Metrics Test Completed ===\n";
}

int main() {
    try {
        testBasicCommunityDetection();
        testDifferentAlgorithms();
        testCommunityPostProcessing();
        testQualityMetrics();
        
        std::cout << "\n✅ All Community Detection tests completed successfully!\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "\n❌ Test failed with unknown exception!\n";
        return 1;
    }
}