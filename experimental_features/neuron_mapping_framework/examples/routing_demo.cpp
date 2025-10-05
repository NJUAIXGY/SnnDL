#include "core/NeuralNetwork.h"
#include "core/HardwareTopology.h"
#include "strategies/GraphPartitioningStrategy.h"
#include "routing/RoutingAwareMappingSolution.h"
#include "routing/RoutingTableGenerator.h"
#include "routing/SpikeRouter.h"
#include "routing/MulticastGroup.h"
#include <iostream>
#include <memory>
#include <chrono>
#include <iomanip>

using namespace NeuronMapping;

/**
 * @brief 演示SpiNNaker风格路由系统的完整功能
 */
class RoutingSystemDemo {
public:
    RoutingSystemDemo() = default;
    
    void runCompleteDemo() {
        std::cout << "=== SpiNNaker风格路由系统演示 ===\n\n";
        
        // 1. 创建网络和硬件拓扑
        auto network = createDemoNetwork();
        auto topology = createDemoTopology();
        
        std::cout << "1. 网络配置:\n";
        std::cout << "   - 神经元数量: " << network->getNeuronCount() << "\n";
        std::cout << "   - 连接数量: " << network->getConnectionCount() << "\n";
        std::cout << "   - PE数量: " << topology->getTotalPEs() << "\n\n";
        
        // 2. 创建路由感知映射解决方案
        auto solution = createRoutingAwareSolution(*network, *topology);
        
        // 3. 演示全局神经元ID管理
        demonstrateGlobalIdManagement(*solution, *network);
        
        // 4. 演示路由表生成
        demonstrateRoutingTableGeneration(*solution, *network, *topology);
        
        // 5. 演示多播组管理
        demonstrateMulticastGroups(*solution, *network);
        
        // 6. 演示路由性能分析
        demonstrateRoutingPerformanceAnalysis(*solution, *network, *topology);
        
        // 7. 演示路由优化
        demonstrateRoutingOptimization(*solution, *network, *topology);
        
        // 8. 演示数据包路由模拟
        demonstratePacketRoutingSimulation(*solution, *network, *topology);
        
        // 9. 性能基准测试
        runPerformanceBenchmark(*network, *topology);
        
        std::cout << "\n=== 路由系统演示完成 ===\n";
    }

private:
    std::unique_ptr<NeuralNetwork> createDemoNetwork() {
        auto network = std::make_unique<NeuralNetwork>();
        
        // 创建32个神经元的小型网络
        const uint32_t num_neurons = 32;
        const uint32_t num_layers = 4;
        const uint32_t neurons_per_layer = num_neurons / num_layers;
        
        for (uint32_t i = 0; i < num_neurons; ++i) {
            network->addNeuron(i, 0.8f);  // 添加LIF神经元，阈值0.8
        }
        
        // 创建层间全连接
        for (uint32_t layer = 0; layer < num_layers - 1; ++layer) {
            for (uint32_t i = 0; i < neurons_per_layer; ++i) {
                for (uint32_t j = 0; j < neurons_per_layer; ++j) {
                    uint32_t src = layer * neurons_per_layer + i;
                    uint32_t dst = (layer + 1) * neurons_per_layer + j;
                    float weight = 0.1f + static_cast<float>(rand()) / RAND_MAX * 0.3f;
                    network->addConnection(src, dst, weight, 1.0f);  // 延迟1.0ms
                }
            }
        }
        
        // 添加一些跨层连接（稀疏）
        for (uint32_t i = 0; i < num_neurons; ++i) {
            if (rand() % 4 == 0) {  // 25%概率
                uint32_t target = rand() % num_neurons;
                if (target != i) {
                    float weight = 0.05f + static_cast<float>(rand()) / RAND_MAX * 0.1f;
                    network->addConnection(i, target, weight, 2.0f);
                }
            }
        }
        
        return network;
    }
    
    std::unique_ptr<HardwareTopology> createDemoTopology() {
        auto topology = std::make_unique<HardwareTopology>();
        
        // 创建4x2网格拓扑（8个PE）
        const uint32_t grid_width = 4;
        const uint32_t grid_height = 2;
        
        for (uint32_t y = 0; y < grid_height; ++y) {
            for (uint32_t x = 0; x < grid_width; ++x) {
                uint32_t pe_id = y * grid_width + x;
                topology->addPE(pe_id, 8, 1024, 100.0f);  // 8神经元容量，1024KB内存，100MHz
                
                // 添加网格连接
                if (x < grid_width - 1) {
                    topology->addConnection(pe_id, pe_id + 1, 10.0f, 1000.0f);  // 东
                }
                if (y < grid_height - 1) {
                    topology->addConnection(pe_id, pe_id + grid_width, 10.0f, 1000.0f);  // 南
                }
            }
        }
        
        return topology;
    }
    
    std::unique_ptr<RoutingAwareMappingSolution> createRoutingAwareSolution(
            const NeuralNetwork& network, const HardwareTopology& topology) {
        std::cout << "2. 创建路由感知映射解决方案...\n";
        
        auto solution = std::make_unique<RoutingAwareMappingSolution>(network.getNeuronCount());
        
        // 使用图分割进行初始映射
        GraphPartitioningStrategy partitioner;
        MappingConfig config;
        config.optimization_target = OptimizationTarget::COMMUNICATION_COST;
        config.enable_load_balancing = true;
        
        auto base_solution = partitioner.generateMapping(network, topology, config);
        
        // 复制基础映射到路由感知解决方案
        for (const auto& assignment : base_solution->getAllAssignments()) {
            solution->assignNeuron(assignment.neuron_id, assignment.pe_id, assignment.core_id);
        }
        
        // 批量分配全局ID
        size_t assigned_ids = solution->batchAssignGlobalIds(network);
        std::cout << "   - 分配全局ID: " << assigned_ids << " 个\n";
        
        // 创建多播组管理器
        auto multicast_manager = std::make_shared<MulticastGroupManager>();
        solution->setMulticastGroupManager(multicast_manager);
        
        // 自动创建多播组
        size_t groups_created = solution->autoCreateMulticastGroups(network, 3);
        std::cout << "   - 创建多播组: " << groups_created << " 个\n\n";
        
        return solution;
    }
    
    void demonstrateGlobalIdManagement(const RoutingAwareMappingSolution& solution,
                                     const NeuralNetwork& network) {
        std::cout << "3. 全局神经元ID管理演示:\n";
        
        // 显示前10个神经元的全局ID
        for (uint32_t i = 0; i < std::min(10u, network.getNeuronCount()); ++i) {
            auto global_id = solution.getGlobalNeuronId(i);
            auto [local_id, pe_id, core_id] = solution.getLocalInfoFromGlobalId(global_id);
            
            std::cout << "   神经元 " << i << " -> 全局ID: 0x" 
                      << std::hex << global_id << std::dec
                      << " (PE" << pe_id << ", Core" << core_id << ")\n";
        }
        
        // 验证全局ID分配
        auto errors = solution.validateGlobalIdAssignment();
        if (errors.empty()) {
            std::cout << "   ✓ 全局ID分配验证通过\n\n";
        } else {
            std::cout << "   ✗ 发现 " << errors.size() << " 个全局ID错误\n\n";
        }
    }
    
    void demonstrateRoutingTableGeneration(RoutingAwareMappingSolution& solution,
                                         const NeuralNetwork& network,
                                         const HardwareTopology& topology) {
        std::cout << "4. 路由表生成演示:\n";
        
        RoutingTableGenerator generator;
        
        // 生成分布式路由表
        auto routing_tables = solution.generateRoutingTables(network, topology, generator);
        
        std::cout << "   - 生成路由表: " << routing_tables.size() << " 个PE\n";
        
        // 显示第一个PE的路由表示例
        if (!routing_tables.empty()) {
            auto pe_id = routing_tables.begin()->first;
            auto& table = routing_tables.begin()->second;
            
            std::cout << "   - PE" << pe_id << " 路由表 (" << table->getEntryCount() << " 条目):\n";
            
            auto entries = table->getAllEntries();
            for (size_t i = 0; i < std::min(5lu, entries.size()); ++i) {
                const auto& entry = entries[i];
                std::cout << "     Key: 0x" << std::hex << entry.key 
                          << ", Mask: 0x" << entry.mask << std::dec
                          << ", Routes: " << entry.routes.size() << "\n";
            }
            
            // 设置路由表到解决方案
            for (auto& [pid, table_ptr] : routing_tables) {
                solution.setRoutingTable(pid, std::move(table_ptr));
            }
        }
        
        std::cout << "\n";
    }
    
    void demonstrateMulticastGroups(const RoutingAwareMappingSolution& solution,
                                  const NeuralNetwork& network) {
        std::cout << "5. 多播组管理演示:\n";
        
        auto manager = solution.getMulticastGroupManager();
        if (!manager) {
            std::cout << "   多播组管理器未初始化\n\n";
            return;
        }
        
        auto groups = manager->getAllGroups();
        std::cout << "   - 多播组数量: " << groups.size() << "\n";
        
        // 显示多播组详情
        for (const auto& [group_id, group] : groups) {
            auto members = group->getMembers();
            auto destinations = group->getDestinations();
            
            std::cout << "   - 组 " << group_id << ": " 
                      << members.size() << " 成员, "
                      << destinations.size() << " 目标\n";
                      
            // 显示组的性能指标
            auto metrics = group->getPerformanceMetrics();
            std::cout << "     性能: 延迟=" << std::fixed << std::setprecision(2) 
                      << metrics.average_latency << "ms, "
                      << "带宽=" << metrics.bandwidth_usage << "MB/s\n";
        }
        
        std::cout << "\n";
    }
    
    void demonstrateRoutingPerformanceAnalysis(const RoutingAwareMappingSolution& solution,
                                             const NeuralNetwork& network,
                                             const HardwareTopology& topology) {
        std::cout << "6. 路由性能分析演示:\n";
        
        // 计算路由性能指标
        auto routing_perf = solution.calculateRoutingPerformance(network, topology);
        
        std::cout << "   路由性能指标:\n";
        for (const auto& [metric, value] : routing_perf) {
            std::cout << "   - " << metric << ": " << std::fixed << std::setprecision(3) 
                      << value << "\n";
        }
        
        // 分析通信模式
        auto comm_patterns = solution.analyzeCommunicationPatterns(network);
        std::cout << "\n   通信模式分析:\n";
        for (const auto& [pattern, count] : comm_patterns) {
            std::cout << "   - " << pattern << ": " << count << "\n";
        }
        
        // 识别路由热点
        auto hotspots = solution.identifyRoutingHotspots(network, topology);
        std::cout << "\n   路由热点 (前3个PE):\n";
        for (size_t i = 0; i < std::min(3lu, hotspots.size()); ++i) {
            std::cout << "   - PE" << hotspots[i].first 
                      << ": 负载 " << std::fixed << std::setprecision(2) 
                      << hotspots[i].second << "\n";
        }
        
        std::cout << "\n";
    }
    
    void demonstrateRoutingOptimization(RoutingAwareMappingSolution& solution,
                                      const NeuralNetwork& network,
                                      const HardwareTopology& topology) {
        std::cout << "7. 路由优化演示:\n";
        
        // 优化前的成本
        float cost_before = solution.calculateTotalRoutingCost(network, topology);
        std::cout << "   优化前路由成本: " << std::fixed << std::setprecision(3) 
                  << cost_before << "\n";
        
        // 执行路由优化
        auto opt_stats = solution.optimizeForRouting(network, topology, 50);
        
        // 优化后的成本
        float cost_after = solution.calculateTotalRoutingCost(network, topology);
        std::cout << "   优化后路由成本: " << std::fixed << std::setprecision(3) 
                  << cost_after << "\n";
        
        float improvement = ((cost_before - cost_after) / cost_before) * 100.0f;
        std::cout << "   路由成本改善: " << std::fixed << std::setprecision(1) 
                  << improvement << "%\n";
        
        // 显示优化统计
        std::cout << "\n   优化统计:\n";
        for (const auto& [metric, value] : opt_stats) {
            std::cout << "   - " << metric << ": " << std::fixed << std::setprecision(3) 
                      << value << "\n";
        }
        
        std::cout << "\n";
    }
    
    void demonstratePacketRoutingSimulation(const RoutingAwareMappingSolution& solution,
                                          const NeuralNetwork& network,
                                          const HardwareTopology& topology) {
        std::cout << "8. 数据包路由模拟演示:\n";
        
        // 模拟几个神经元之间的通信
        std::vector<std::pair<uint32_t, uint32_t>> test_pairs = {
            {0, 15}, {5, 25}, {10, 30}, {2, 18}
        };
        
        for (const auto& [src, dst] : test_pairs) {
            if (src < network.getNeuronCount() && dst < network.getNeuronCount()) {
                auto [path, stats] = solution.simulatePacketRouting(src, dst, topology);
                
                std::cout << "   神经元 " << src << " -> " << dst << ":\n";
                std::cout << "     路径长度: " << path.size() << " 跳\n";
                std::cout << "     预估延迟: " << std::fixed << std::setprecision(2) 
                          << stats.at("total_latency") << "ms\n";
                std::cout << "     路径: ";
                for (size_t i = 0; i < path.size(); ++i) {
                    std::cout << "PE" << path[i];
                    if (i < path.size() - 1) std::cout << " -> ";
                }
                std::cout << "\n\n";
            }
        }
    }
    
    void runPerformanceBenchmark(const NeuralNetwork& network, 
                               const HardwareTopology& topology) {
        std::cout << "9. 性能基准测试:\n";
        
        // 测试不同映射策略的路由性能
        std::vector<std::string> strategies = {"uniform", "graph_partition", "routing_aware"};
        
        for (const auto& strategy : strategies) {
            auto start_time = std::chrono::high_resolution_clock::now();
            
            std::unique_ptr<RoutingAwareMappingSolution> solution;
            
            if (strategy == "uniform") {
                solution = createUniformMappingSolution(network, topology);
            } else if (strategy == "graph_partition") {
                solution = createGraphPartitionSolution(network, topology);
            } else {
                solution = RoutingAwareMappingSolutionFactory::createOptimizedSolution(
                    network, topology, 2);
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time).count();
            
            float routing_cost = solution->calculateTotalRoutingCost(network, topology);
            auto routing_perf = solution->calculateRoutingPerformance(network, topology);
            
            std::cout << "   " << strategy << "策略:\n";
            std::cout << "     - 生成时间: " << duration << "ms\n";
            std::cout << "     - 路由成本: " << std::fixed << std::setprecision(3) 
                      << routing_cost << "\n";
            std::cout << "     - 平均跳数: " << std::fixed << std::setprecision(2) 
                      << routing_perf.at("average_hop_count") << "\n";
            std::cout << "     - 多播效率: " << std::fixed << std::setprecision(2) 
                      << routing_perf.at("multicast_efficiency") * 100.0f << "%\n\n";
        }
    }
    
    std::unique_ptr<RoutingAwareMappingSolution> createUniformMappingSolution(
            const NeuralNetwork& network, const HardwareTopology& topology) {
        auto solution = std::make_unique<RoutingAwareMappingSolution>(network.getNeuronCount());
        
        // 均匀分配神经元到PE
        auto pe_ids = topology.getAllPEIds();
        for (uint32_t i = 0; i < network.getNeuronCount(); ++i) {
            PEId pe_id = pe_ids[i % pe_ids.size()];
            solution->assignNeuron(i, pe_id);
        }
        
        solution->batchAssignGlobalIds(network);
        
        auto manager = std::make_shared<MulticastGroupManager>();
        solution->setMulticastGroupManager(manager);
        solution->autoCreateMulticastGroups(network, 3);
        
        return solution;
    }
    
    std::unique_ptr<RoutingAwareMappingSolution> createGraphPartitionSolution(
            const NeuralNetwork& network, const HardwareTopology& topology) {
        auto solution = std::make_unique<RoutingAwareMappingSolution>(network.getNeuronCount());
        
        // 使用图分割
        GraphPartitioningStrategy partitioner;
        MappingConfig config;
        config.optimization_target = OptimizationTarget::COMMUNICATION_COST;
        
        auto base_solution = partitioner.generateMapping(network, topology, config);
        
        for (const auto& assignment : base_solution->getAllAssignments()) {
            solution->assignNeuron(assignment.neuron_id, assignment.pe_id, assignment.core_id);
        }
        
        solution->batchAssignGlobalIds(network);
        
        auto manager = std::make_shared<MulticastGroupManager>();
        solution->setMulticastGroupManager(manager);
        solution->autoCreateMulticastGroups(network, 3);
        
        return solution;
    }
};

/**
 * @brief 路由性能测试套件
 */
class RoutingPerformanceTestSuite {
public:
    void runAllTests() {
        std::cout << "\n=== 路由性能测试套件 ===\n\n";
        
        testAddressEventEncoding();
        testRoutingTableLookup();
        testMulticastOptimization();
        testRoutingAwarePartitioning();
        testScalabilityPerformance();
        
        std::cout << "=== 路由测试套件完成 ===\n";
    }

private:
    void testAddressEventEncoding() {
        std::cout << "测试1: AddressEvent编码性能\n";
        
        const uint32_t num_tests = 100000;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (uint32_t i = 0; i < num_tests; ++i) {
            auto global_id = AddressEvent::encodeGlobalNeuronId(i % 256, i % 16, i % 4);
            auto [pe, core, local] = AddressEvent::decodeGlobalNeuronId(global_id);
            
            // 验证编码/解码正确性
            if (pe != (i % 256) || core != (i % 16) || local != (i % 4)) {
                std::cout << "   ✗ 编码错误在测试 " << i << "\n";
                return;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count();
        
        std::cout << "   - " << num_tests << " 次编码/解码: " 
                  << duration << "μs (平均 " << std::fixed << std::setprecision(2)
                  << static_cast<double>(duration) / num_tests << "μs/次)\n";
        std::cout << "   ✓ 编码正确性验证通过\n\n";
    }
    
    void testRoutingTableLookup() {
        std::cout << "测试2: 路由表查找性能\n";
        
        RoutingTable table;
        
        // 添加测试路由条目
        for (uint32_t i = 0; i < 1000; ++i) {
            RoutingEntry entry;
            entry.key = i << 16;  // 高16位作为神经元ID
            entry.mask = 0xFFFF0000;
            entry.routes.push_back(static_cast<RouteDirection>(i % 6));
            table.addEntry(entry);
        }
        
        std::cout << "   - 路由表大小: " << table.getEntryCount() << " 条目\n";
        
        // 性能测试
        const uint32_t num_lookups = 100000;
        uint32_t hits = 0;
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (uint32_t i = 0; i < num_lookups; ++i) {
            uint32_t test_key = (i % 1000) << 16 | (i % 256);
            auto routes = table.lookup(test_key);
            if (!routes.empty()) {
                hits++;
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
            end_time - start_time).count();
        
        std::cout << "   - " << num_lookups << " 次查找: " 
                  << duration << "μs (平均 " << std::fixed << std::setprecision(3)
                  << static_cast<double>(duration) / num_lookups << "μs/次)\n";
        std::cout << "   - 命中率: " << std::fixed << std::setprecision(1)
                  << (static_cast<float>(hits) / num_lookups) * 100.0f << "%\n\n";
    }
    
    void testMulticastOptimization() {
        std::cout << "测试3: 多播优化性能\n";
        
        MulticastGroupManager manager;
        
        // 创建测试网络
        auto network = createTestNetwork(64);
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // 自动检测多播组
        auto groups = manager.autoDetectGroups(*network, 4);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - start_time).count();
        
        std::cout << "   - 检测到多播组: " << groups.size() << "\n";
        std::cout << "   - 检测时间: " << duration << "ms\n";
        
        // 计算多播效率
        float total_reduction = 0.0f;
        for (const auto& group : groups) {
            auto members = group->getMembers();
            auto destinations = group->getDestinations();
            
            // 计算流量减少（粗略估计）
            float unicast_traffic = members.size() * destinations.size();
            float multicast_traffic = members.size() + destinations.size();
            float reduction = (unicast_traffic - multicast_traffic) / unicast_traffic;
            total_reduction += reduction;
        }
        
        if (!groups.empty()) {
            float avg_reduction = total_reduction / groups.size();
            std::cout << "   - 平均流量减少: " << std::fixed << std::setprecision(1)
                      << avg_reduction * 100.0f << "%\n";
        }
        
        std::cout << "\n";
    }
    
    void testRoutingAwarePartitioning() {
        std::cout << "测试4: 路由感知分割性能\n";
        
        auto network = createTestNetwork(128);
        auto topology = createTestTopology();
        
        // 标准图分割
        auto start_time = std::chrono::high_resolution_clock::now();
        
        GraphPartitioningStrategy standard_partitioner;
        MappingConfig config;
        config.optimization_target = OptimizationTarget::COMMUNICATION_COST;
        
        auto standard_solution = standard_partitioner.generateMapping(*network, *topology, config);
        
        auto mid_time = std::chrono::high_resolution_clock::now();
        
        // 路由感知分割
        auto routing_solution = RoutingAwareMappingSolutionFactory::createOptimizedSolution(
            *network, *topology, 2);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        
        auto standard_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            mid_time - start_time).count();
        auto routing_duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            end_time - mid_time).count();
        
        // 创建标准解决方案的路由版本进行比较
        auto standard_routing = RoutingAwareMappingSolutionFactory::createFromBaseSolution(
            *standard_solution, *network, *topology);
        
        float standard_cost = standard_routing->calculateTotalRoutingCost(*network, *topology);
        float routing_cost = routing_solution->calculateTotalRoutingCost(*network, *topology);
        
        std::cout << "   - 标准分割时间: " << standard_duration << "ms\n";
        std::cout << "   - 路由感知分割时间: " << routing_duration << "ms\n";
        std::cout << "   - 标准分割路由成本: " << std::fixed << std::setprecision(3) 
                  << standard_cost << "\n";
        std::cout << "   - 路由感知分割成本: " << std::fixed << std::setprecision(3) 
                  << routing_cost << "\n";
        
        float improvement = ((standard_cost - routing_cost) / standard_cost) * 100.0f;
        std::cout << "   - 路由成本改善: " << std::fixed << std::setprecision(1) 
                  << improvement << "%\n\n";
    }
    
    void testScalabilityPerformance() {
        std::cout << "测试5: 可扩展性性能\n";
        
        std::vector<uint32_t> network_sizes = {64, 128, 256, 512};
        
        for (uint32_t size : network_sizes) {
            auto network = createTestNetwork(size);
            auto topology = createScalableTopology(size / 8);  // 每8个神经元一个PE
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            auto solution = RoutingAwareMappingSolutionFactory::createOptimizedSolution(
                *network, *topology, 1);  // 快速优化
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time).count();
            
            float routing_cost = solution->calculateTotalRoutingCost(*network, *topology);
            auto stats = solution->getRoutingStatistics();
            
            std::cout << "   网络规模 " << size << " 神经元:\n";
            std::cout << "     - 映射时间: " << duration << "ms\n";
            std::cout << "     - 路由成本: " << std::fixed << std::setprecision(3) 
                      << routing_cost << "\n";
            std::cout << "     - 全局ID数量: " << stats.at("global_ids_assigned") << "\n";
            std::cout << "     - 路由表条目: " << stats.at("routing_table_entries") << "\n\n";
        }
    }
    
    std::unique_ptr<NeuralNetwork> createTestNetwork(uint32_t num_neurons) {
        auto network = std::make_unique<NeuralNetwork>();
        
        // 创建神经元
        for (uint32_t i = 0; i < num_neurons; ++i) {
            network->addNeuron(i, 0.8f);
        }
        
        // 创建随机连接
        for (uint32_t i = 0; i < num_neurons; ++i) {
            uint32_t num_connections = 3 + rand() % 8;  // 3-10个连接
            for (uint32_t j = 0; j < num_connections; ++j) {
                uint32_t target = rand() % num_neurons;
                if (target != i) {
                    float weight = 0.1f + static_cast<float>(rand()) / RAND_MAX * 0.4f;
                    float delay = 1.0f + static_cast<float>(rand()) / RAND_MAX * 3.0f;
                    network->addConnection(i, target, weight, delay);
                }
            }
        }
        
        return network;
    }
    
    std::unique_ptr<HardwareTopology> createTestTopology() {
        auto topology = std::make_unique<HardwareTopology>();
        
        // 4x4网格
        const uint32_t grid_size = 4;
        for (uint32_t y = 0; y < grid_size; ++y) {
            for (uint32_t x = 0; x < grid_size; ++x) {
                uint32_t pe_id = y * grid_size + x;
                topology->addPE(pe_id, 16, 2048, 200.0f);
                
                // 网格连接
                if (x < grid_size - 1) {
                    topology->addConnection(pe_id, pe_id + 1, 5.0f, 2000.0f);
                }
                if (y < grid_size - 1) {
                    topology->addConnection(pe_id, pe_id + grid_size, 5.0f, 2000.0f);
                }
            }
        }
        
        return topology;
    }
    
    std::unique_ptr<HardwareTopology> createScalableTopology(uint32_t num_pes) {
        auto topology = std::make_unique<HardwareTopology>();
        
        // 创建近似正方形的网格
        uint32_t grid_width = static_cast<uint32_t>(std::sqrt(num_pes)) + 1;
        uint32_t grid_height = (num_pes + grid_width - 1) / grid_width;
        
        for (uint32_t i = 0; i < num_pes; ++i) {
            uint32_t x = i % grid_width;
            uint32_t y = i / grid_width;
            
            topology->addPE(i, 16, 2048, 200.0f);
            
            // 连接到相邻PE
            if (x < grid_width - 1 && i + 1 < num_pes) {
                topology->addConnection(i, i + 1, 5.0f, 2000.0f);
            }
            if (y < grid_height - 1 && i + grid_width < num_pes) {
                topology->addConnection(i, i + grid_width, 5.0f, 2000.0f);
            }
        }
        
        return topology;
    }
};

int main() {
    std::cout << "SpiNNaker风格神经元映射框架路由系统演示\n";
    std::cout << "========================================\n\n";
    
    try {
        // 设置随机种子以获得可重现的结果
        srand(42);
        
        // 运行主要演示
        RoutingSystemDemo demo;
        demo.runCompleteDemo();
        
        // 运行性能测试
        RoutingPerformanceTestSuite test_suite;
        test_suite.runAllTests();
        
    } catch (const std::exception& e) {
        std::cerr << "演示过程中发生错误: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}