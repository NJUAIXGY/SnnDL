#include "routing/AddressEvent.h"
#include "routing/RoutingTable.h"
#include "routing/SpikeRouter.h"
#include "routing/RoutingTableGenerator.h"
#include "routing/MulticastGroup.h"
#include "routing/RoutingAwareMappingSolution.h"
#include "routing/RoutingAwareGraphPartitioningStrategy.h"
#include "strategies/GraphPartitioningStrategy.h"
#include "core/NeuralNetwork.h"
#include "core/HardwareTopology.h"
#include <iostream>
#include <cassert>
#include <vector>
#include <memory>
#include <chrono>

using namespace NeuronMapping;

/**
 * @brief 路由系统测试类
 */
class RoutingSystemTests {
public:
    void runAllTests() {
        std::cout << "=== 路由系统单元测试 ===\n\n";
        
        test_counter_ = 0;
        passed_tests_ = 0;
        
        testAddressEventOperations();
        testRoutingTableOperations();
        testSpikeRouterFunctionality();
        testRoutingTableGeneration();
        testMulticastGroupManagement();
        testRoutingAwareMappingSolution();
        testRoutingAwareGraphPartitioning();
        testPerformanceAndScalability();
        testErrorHandlingAndEdgeCases();
        testIntegrationScenarios();
        
        printTestSummary();
    }

private:
    int test_counter_ = 0;
    int passed_tests_ = 0;
    
    void runTest(const std::string& test_name, std::function<bool()> test_func) {
        test_counter_++;
        std::cout << "测试 " << test_counter_ << ": " << test_name << " ... ";
        
        try {
            bool result = test_func();
            if (result) {
                std::cout << "✓ 通过\n";
                passed_tests_++;
            } else {
                std::cout << "✗ 失败\n";
            }
        } catch (const std::exception& e) {
            std::cout << "✗ 异常: " << e.what() << "\n";
        }
    }
    
    void testAddressEventOperations() {
        std::cout << "1. AddressEvent 测试组\n";
        
        runTest("全局神经元ID编码/解码", [this]() {
            // 测试边界值
            auto global_id = AddressEvent::encodeGlobalNeuronId(255, 15, 255);
            auto [pe_id, core_id, local_id] = AddressEvent::decodeGlobalNeuronId(global_id);
            
            return pe_id == 255 && core_id == 15 && local_id == 255;
        });
        
        runTest("AddressEvent创建和序列化", [this]() {
            AddressEvent event(123, 456, 1000, 1.5f);
            
            auto serialized = event.serialize();
            AddressEvent deserialized;
            deserialized.deserialize(serialized);
            
            return deserialized.source_neuron_id == 123 &&
                   deserialized.target_neuron_id == 456 &&
                   deserialized.timestamp == 1000 &&
                   std::abs(deserialized.payload - 1.5f) < 0.001f;
        });
        
        runTest("AddressEvent批处理", [this]() {
            AddressEventBatch batch;
            
            for (uint32_t i = 0; i < 10; ++i) {
                batch.addEvent(AddressEvent(i, i + 100, i * 1000, 1.0f));
            }
            
            auto events = batch.getEvents();
            batch.sortByTimestamp();
            
            return events.size() == 10 && batch.isEmpty() == false;
        });
        
        runTest("多播事件处理", [this]() {
            AddressEvent multicast_event(0, 0, 1000, 1.0f);
            multicast_event.setMulticast(true);
            multicast_event.addMulticastTarget(100);
            multicast_event.addMulticastTarget(200);
            multicast_event.addMulticastTarget(300);
            
            auto targets = multicast_event.getMulticastTargets();
            return targets.size() == 3 && multicast_event.isMulticast();
        });
        
        std::cout << "\n";
    }
    
    void testRoutingTableOperations() {
        std::cout << "2. RoutingTable 测试组\n";
        
        runTest("路由表条目添加和查找", [this]() {
            RoutingTable table;
            
            RoutingEntry entry;
            entry.key = 0x12340000;
            entry.mask = 0xFFFF0000;
            entry.routes.push_back(RouteDirection::EAST);
            entry.routes.push_back(RouteDirection::NORTH);
            
            table.addEntry(entry);
            
            auto routes = table.lookup(0x12345678);
            return routes.size() == 2 && 
                   routes[0] == RouteDirection::EAST &&
                   routes[1] == RouteDirection::NORTH;
        });
        
        runTest("路由表优化和压缩", [this]() {
            RoutingTable table;
            
            // 添加可以合并的条目
            for (uint32_t i = 0; i < 16; ++i) {
                RoutingEntry entry;
                entry.key = (0x1000 + i) << 16;
                entry.mask = 0xFFFF0000;
                entry.routes.push_back(RouteDirection::EAST);
                table.addEntry(entry);
            }
            
            size_t before_optimization = table.getEntryCount();
            table.optimize();
            size_t after_optimization = table.getEntryCount();
            
            return after_optimization < before_optimization;
        });
        
        runTest("路由表统计和性能分析", [this]() {
            RoutingTable table;
            
            // 添加测试条目
            for (uint32_t i = 0; i < 100; ++i) {
                RoutingEntry entry;
                entry.key = i << 16;
                entry.mask = 0xFFFF0000;
                entry.routes.push_back(static_cast<RouteDirection>(i % 6));
                table.addEntry(entry);
            }
            
            auto stats = table.getStatistics();
            return stats.total_entries == 100 && 
                   stats.memory_usage > 0 &&
                   stats.average_routes_per_entry > 0;
        });
        
        runTest("分布式路由表管理", [this]() {
            DistributedRoutingTableManager manager;
            
            for (uint32_t pe = 0; pe < 4; ++pe) {
                auto table = std::make_unique<RoutingTable>();
                
                RoutingEntry entry;
                entry.key = pe << 24;
                entry.mask = 0xFF000000;
                entry.routes.push_back(RouteDirection::LOCAL);
                table->addEntry(entry);
                
                manager.setRoutingTable(pe, std::move(table));
            }
            
            auto pe0_table = manager.getRoutingTable(0);
            auto pe3_table = manager.getRoutingTable(3);
            
            return pe0_table != nullptr && pe3_table != nullptr &&
                   manager.getAllPEs().size() == 4;
        });
        
        std::cout << "\n";
    }
    
    void testSpikeRouterFunctionality() {
        std::cout << "3. SpikeRouter 测试组\n";
        
        runTest("路由器初始化和配置", [this]() {
            RouterConfig config;
            config.max_buffer_size = 1024;
            config.enable_flow_control = true;
            config.enable_multicast = true;
            
            SpikeRouter router(0, config);
            
            return router.getPEId() == 0 && 
                   router.isFlowControlEnabled() &&
                   router.isMulticastEnabled();
        });
        
        runTest("数据包路由处理", [this]() {
            RouterConfig config;
            SpikeRouter router(0, config);
            
            // 设置简单路由表
            auto table = std::make_unique<RoutingTable>();
            RoutingEntry entry;
            entry.key = 0x12340000;
            entry.mask = 0xFFFF0000;
            entry.routes.push_back(RouteDirection::EAST);
            table->addEntry(entry);
            
            router.setRoutingTable(std::move(table));
            
            // 创建测试数据包
            AddressEvent packet(123, 456, 1000, 1.0f);
            packet.source_neuron_id = 0x12345678;
            
            auto routes = router.routePacket(packet);
            return !routes.empty() && routes[0] == RouteDirection::EAST;
        });
        
        runTest("流量控制和拥塞管理", [this]() {
            RouterConfig config;
            config.max_buffer_size = 10;
            config.enable_flow_control = true;
            config.congestion_threshold = 0.8f;
            
            SpikeRouter router(0, config);
            
            // 模拟拥塞
            for (int i = 0; i < 12; ++i) {
                AddressEvent packet(i, i + 100, i * 1000, 1.0f);
                router.bufferPacket(packet, RouteDirection::EAST);
            }
            
            auto stats = router.getStatistics();
            return stats.congestion_events > 0 && stats.dropped_packets > 0;
        });
        
        runTest("多播路由处理", [this]() {
            RouterConfig config;
            config.enable_multicast = true;
            SpikeRouter router(0, config);
            
            // 创建多播数据包
            AddressEvent multicast_packet(0, 0, 1000, 1.0f);
            multicast_packet.setMulticast(true);
            multicast_packet.addMulticastTarget(100);
            multicast_packet.addMulticastTarget(200);
            
            auto replicated = router.replicateMulticastPacket(multicast_packet);
            return replicated.size() == 2;
        });
        
        std::cout << "\n";
    }
    
    void testRoutingTableGeneration() {
        std::cout << "4. RoutingTableGenerator 测试组\n";
        
        runTest("基本路由表生成", [this]() {
            auto network = createTestNetwork(16);
            auto topology = createTestTopology();
            auto solution = createTestMappingSolution(*network, *topology);
            
            RoutingTableGenerator generator;
            auto tables = generator.generateDistributedTables(*solution, *network, *topology);
            
            return !tables.empty() && tables.size() <= topology->getTotalPEs();
        });
        
        runTest("最短路径路由策略", [this]() {
            auto network = createTestNetwork(8);
            auto topology = createTestTopology();
            auto solution = createTestMappingSolution(*network, *topology);
            
            RoutingTableGenerator generator;
            RoutingTableConfig config;
            config.strategy = RoutingStrategy::SHORTEST_PATH;
            
            auto tables = generator.generateDistributedTables(*solution, *network, *topology, config);
            
            // 验证生成的路由表
            bool has_valid_routes = false;
            for (const auto& [pe_id, table] : tables) {
                if (table->getEntryCount() > 0) {
                    has_valid_routes = true;
                    break;
                }
            }
            
            return has_valid_routes;
        });
        
        runTest("负载均衡路由策略", [this]() {
            auto network = createTestNetwork(16);
            auto topology = createTestTopology();
            auto solution = createTestMappingSolution(*network, *topology);
            
            RoutingTableGenerator generator;
            RoutingTableConfig config;
            config.strategy = RoutingStrategy::LOAD_BALANCED;
            config.enable_adaptive_routing = true;
            
            auto tables = generator.generateDistributedTables(*solution, *network, *topology, config);
            
            return !tables.empty();
        });
        
        runTest("路由表压缩算法", [this]() {
            auto network = createTestNetwork(32);
            auto topology = createTestTopology();
            auto solution = createTestMappingSolution(*network, *topology);
            
            RoutingTableGenerator generator;
            RoutingTableConfig config;
            config.enable_compression = true;
            config.compression_method = CompressionMethod::PREFIX_AGGREGATION;
            
            auto tables = generator.generateDistributedTables(*solution, *network, *topology, config);
            
            // 检查压缩效果
            size_t total_entries = 0;
            for (const auto& [pe_id, table] : tables) {
                total_entries += table->getEntryCount();
            }
            
            return total_entries > 0;
        });
        
        std::cout << "\n";
    }
    
    void testMulticastGroupManagement() {
        std::cout << "5. MulticastGroup 测试组\n";
        
        runTest("多播组创建和管理", [this]() {
            EnhancedMulticastGroup group(1, 100);
            
            group.addMember(1);
            group.addMember(2);
            group.addMember(3);
            
            group.addDestination(101);
            group.addDestination(102);
            
            auto members = group.getMembers();
            auto destinations = group.getDestinations();
            
            return members.size() == 3 && destinations.size() == 2;
        });
        
        runTest("多播组自动检测", [this]() {
            auto network = createTestNetwork(16);
            
            MulticastGroupManager manager;
            auto groups = manager.autoDetectGroups(*network, 3);
            
            return !groups.empty();
        });
        
        runTest("多播组QoS管理", [this]() {
            EnhancedMulticastGroup group(1, 100);
            
            MulticastQoS qos;
            qos.priority = 5;
            qos.max_latency = 10.0f;
            qos.min_bandwidth = 100.0f;
            
            group.setQoS(qos);
            auto retrieved_qos = group.getQoS();
            
            return retrieved_qos.priority == 5 && 
                   retrieved_qos.max_latency == 10.0f;
        });
        
        runTest("多播组性能优化", [this]() {
            EnhancedMulticastGroup group(1, 100);
            
            // 添加成员和目标
            for (uint32_t i = 0; i < 10; ++i) {
                group.addMember(i);
                group.addDestination(i + 100);
            }
            
            // 优化多播树
            group.optimizeMulticastTree();
            
            auto metrics = group.getPerformanceMetrics();
            return metrics.tree_efficiency > 0.0f;
        });
        
        std::cout << "\n";
    }
    
    void testRoutingAwareMappingSolution() {
        std::cout << "6. RoutingAwareMappingSolution 测试组\n";
        
        runTest("全局神经元ID分配", [this]() {
            auto network = createTestNetwork(16);
            auto topology = createTestTopology();
            
            RoutingAwareMappingSolution solution(network->getNeuronCount());
            
            // 分配神经元到PE
            for (uint32_t i = 0; i < network->getNeuronCount(); ++i) {
                solution.assignNeuron(i, i % topology->getTotalPEs());
            }
            
            size_t assigned = solution.batchAssignGlobalIds(*network);
            
            return assigned == network->getNeuronCount();
        });
        
        runTest("路由信息管理", [this]() {
            auto network = createTestNetwork(8);
            auto topology = createTestTopology();
            
            RoutingAwareMappingSolution solution(network->getNeuronCount());
            
            // 添加路由信息
            for (uint32_t i = 0; i < network->getNeuronCount(); ++i) {
                solution.assignNeuron(i, i % topology->getTotalPEs());
                
                RoutingInfo info;
                info.global_neuron_id = i + 1000;
                info.routing_key = i << 16;
                info.routing_cost = static_cast<float>(i);
                
                solution.addRoutingInfo(i, info);
            }
            
            auto info = solution.getRoutingInfo(0);
            return info != nullptr && info->routing_key == 0;
        });
        
        runTest("路由性能分析", [this]() {
            auto network = createTestNetwork(16);
            auto topology = createTestTopology();
            auto solution = createTestRoutingAwareSolution(*network, *topology);
            
            auto performance = solution->calculateRoutingPerformance(*network, *topology);
            auto comm_patterns = solution->analyzeCommunicationPatterns(*network);
            
            return !performance.empty() && !comm_patterns.empty();
        });
        
        runTest("路由优化", [this]() {
            auto network = createTestNetwork(16);
            auto topology = createTestTopology();
            auto solution = createTestRoutingAwareSolution(*network, *topology);
            
            float cost_before = solution->calculateTotalRoutingCost(*network, *topology);
            auto opt_stats = solution->optimizeForRouting(*network, *topology, 10);
            float cost_after = solution->calculateTotalRoutingCost(*network, *topology);
            
            return cost_after <= cost_before;  // 成本应该减少或保持不变
        });
        
        std::cout << "\n";
    }
    
    void testRoutingAwareGraphPartitioning() {
        std::cout << "7. RoutingAwareGraphPartitioning 测试组\n";
        
        runTest("路由感知分割基本功能", [this]() {
            auto network = createTestNetwork(16);
            auto topology = createTestTopology();
            
            RoutingAwareGraphPartitioningStrategy partitioner;
            MappingConfig config;
            config.optimization_target = OptimizationTarget::COMMUNICATION_COST;
            
            auto solution = partitioner.generateMapping(*network, *topology, config);
            
            return solution != nullptr && 
                   solution->getAssignedNeuronCount() == network->getNeuronCount();
        });
        
        runTest("多播模式优化", [this]() {
            auto network = createTestNetwork(16);
            auto topology = createTestTopology();
            
            RoutingAwareGraphPartitioningStrategy partitioner;
            RoutingAwareConfig routing_config;
            routing_config.optimization_target = RoutingOptimizationTarget::MULTICAST_EFFICIENCY;
            routing_config.enable_multicast_detection = true;
            
            MappingConfig config;
            config.custom_params["routing_config"] = reinterpret_cast<void*>(&routing_config);
            
            auto solution = partitioner.generateMapping(*network, *topology, config);
            
            return solution != nullptr;
        });
        
        runTest("拥塞感知分割", [this]() {
            auto network = createTestNetwork(32);
            auto topology = createTestTopology();
            
            RoutingAwareGraphPartitioningStrategy partitioner;
            RoutingAwareConfig routing_config;
            routing_config.optimization_target = RoutingOptimizationTarget::CONGESTION_MINIMIZATION;
            routing_config.enable_congestion_aware_partitioning = true;
            
            MappingConfig config;
            config.custom_params["routing_config"] = reinterpret_cast<void*>(&routing_config);
            
            auto solution = partitioner.generateMapping(*network, *topology, config);
            
            return solution != nullptr;
        });
        
        std::cout << "\n";
    }
    
    void testPerformanceAndScalability() {
        std::cout << "8. 性能和可扩展性测试组\n";
        
        runTest("大规模网络映射性能", [this]() {
            auto network = createTestNetwork(256);
            auto topology = createScalableTopology(32);
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            auto solution = RoutingAwareMappingSolutionFactory::createOptimizedSolution(
                *network, *topology, 1);
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time).count();
            
            return duration < 5000 && solution != nullptr;  // 应在5秒内完成
        });
        
        runTest("路由表查找性能", [this]() {
            RoutingTable table;
            
            // 添加大量路由条目
            for (uint32_t i = 0; i < 10000; ++i) {
                RoutingEntry entry;
                entry.key = i << 16;
                entry.mask = 0xFFFF0000;
                entry.routes.push_back(static_cast<RouteDirection>(i % 6));
                table.addEntry(entry);
            }
            
            auto start_time = std::chrono::high_resolution_clock::now();
            
            // 执行大量查找
            for (uint32_t i = 0; i < 100000; ++i) {
                table.lookup((i % 10000) << 16);
            }
            
            auto end_time = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
                end_time - start_time).count();
            
            return duration < 1000;  // 应在1秒内完成10万次查找
        });
        
        runTest("内存使用效率", [this]() {
            auto network = createTestNetwork(128);
            auto topology = createTestTopology();
            auto solution = createTestRoutingAwareSolution(*network, *topology);
            
            // 生成路由表
            RoutingTableGenerator generator;
            auto tables = solution->generateRoutingTables(*network, *topology, generator);
            
            // 计算总内存使用
            size_t total_memory = 0;
            for (const auto& [pe_id, table] : tables) {
                auto stats = table->getStatistics();
                total_memory += stats.memory_usage;
            }
            
            // 内存使用应该合理（小于10MB）
            return total_memory < 10 * 1024 * 1024;
        });
        
        std::cout << "\n";
    }
    
    void testErrorHandlingAndEdgeCases() {
        std::cout << "9. 错误处理和边界情况测试组\n";
        
        runTest("无效参数处理", [this]() {
            RoutingAwareMappingSolution solution(10);
            
            // 测试无效神经元ID
            bool result1 = !solution.assignNeuron(1000, 0);
            
            // 测试无效PE ID
            bool result2 = !solution.assignNeuron(0, 1000);
            
            // 测试重复分配
            solution.assignNeuron(0, 0);
            bool result3 = !solution.assignNeuron(0, 1);  // 应该失败，因为已经分配
            
            return result1 && result2 && result3;
        });
        
        runTest("空网络处理", [this]() {
            NeuralNetwork empty_network;
            auto topology = createTestTopology();
            
            RoutingAwareMappingSolution solution(0);
            auto performance = solution.calculateRoutingPerformance(empty_network, *topology);
            
            return performance.empty() || performance.begin()->second == 0.0f;
        });
        
        runTest("路由表一致性验证", [this]() {
            auto network = createTestNetwork(16);
            auto topology = createTestTopology();
            auto solution = createTestRoutingAwareSolution(*network, *topology);
            
            auto errors = solution->validateRoutingConsistency(*network, *topology);
            
            return errors.empty();  // 应该没有一致性错误
        });
        
        runTest("多播组冲突处理", [this]() {
            MulticastGroupManager manager;
            
            // 创建重叠的多播组
            auto group1 = std::make_shared<EnhancedMulticastGroup>(1, 100);
            group1->addMember(1);
            group1->addMember(2);
            
            auto group2 = std::make_shared<EnhancedMulticastGroup>(2, 200);
            group2->addMember(2);  // 重叠成员
            group2->addMember(3);
            
            manager.addGroup(group1);
            manager.addGroup(group2);
            
            auto conflicts = manager.findGroupConflicts();
            
            return !conflicts.empty();  // 应该检测到冲突
        });
        
        std::cout << "\n";
    }
    
    void testIntegrationScenarios() {
        std::cout << "10. 集成场景测试组\n";
        
        runTest("端到端路由流程", [this]() {
            // 创建完整的测试场景
            auto network = createTestNetwork(32);
            auto topology = createTestTopology();
            
            // 1. 创建路由感知解决方案
            auto solution = RoutingAwareMappingSolutionFactory::createOptimizedSolution(
                *network, *topology, 2);
            
            // 2. 生成路由表
            RoutingTableGenerator generator;
            auto tables = solution->generateRoutingTables(*network, *topology, generator);
            
            // 3. 创建路由器
            RouterConfig config;
            SpikeRouter router(0, config);
            if (!tables.empty()) {
                auto table_copy = std::make_unique<RoutingTable>(*tables.begin()->second);
                router.setRoutingTable(std::move(table_copy));
            }
            
            // 4. 模拟数据包路由
            AddressEvent packet(0, 15, 1000, 1.0f);
            auto routes = router.routePacket(packet);
            
            return !routes.empty();
        });
        
        runTest("多层网络路由", [this]() {
            auto network = createLayeredNetwork(64, 4);  // 64神经元，4层
            auto topology = createTestTopology();
            
            auto solution = RoutingAwareMappingSolutionFactory::createOptimizedSolution(
                *network, *topology, 2);
            
            auto performance = solution->calculateRoutingPerformance(*network, *topology);
            
            return !performance.empty();
        });
        
        runTest("动态路由调整", [this]() {
            auto network = createTestNetwork(16);
            auto topology = createTestTopology();
            auto solution = createTestRoutingAwareSolution(*network, *topology);
            
            float cost_before = solution->calculateTotalRoutingCost(*network, *topology);
            
            // 移动一些神经元
            solution->moveNeuronWithRouting(0, 1);
            solution->moveNeuronWithRouting(5, 2);
            
            // 重新计算路由信息
            size_t updated = solution->recalculateAllRoutingInfo(*network, *topology);
            
            float cost_after = solution->calculateTotalRoutingCost(*network, *topology);
            
            return updated > 0;  // 应该有路由信息被更新
        });
        
        std::cout << "\n";
    }
    
    void printTestSummary() {
        std::cout << "=== 测试总结 ===\n";
        std::cout << "总测试数: " << test_counter_ << "\n";
        std::cout << "通过测试: " << passed_tests_ << "\n";
        std::cout << "失败测试: " << (test_counter_ - passed_tests_) << "\n";
        std::cout << "成功率: " << std::fixed << std::setprecision(1) 
                  << (static_cast<float>(passed_tests_) / test_counter_) * 100.0f << "%\n\n";
        
        if (passed_tests_ == test_counter_) {
            std::cout << "🎉 所有测试都通过了！\n";
        } else {
            std::cout << "⚠️  有测试失败，请检查实现。\n";
        }
    }
    
    // 辅助方法
    std::unique_ptr<NeuralNetwork> createTestNetwork(uint32_t num_neurons) {
        auto network = std::make_unique<NeuralNetwork>();
        
        for (uint32_t i = 0; i < num_neurons; ++i) {
            network->addNeuron(i, 0.8f);
        }
        
        // 创建随机连接
        for (uint32_t i = 0; i < num_neurons; ++i) {
            uint32_t num_connections = 2 + (i % 5);
            for (uint32_t j = 0; j < num_connections; ++j) {
                uint32_t target = (i + j + 1) % num_neurons;
                network->addConnection(i, target, 0.5f, 1.0f);
            }
        }
        
        return network;
    }
    
    std::unique_ptr<NeuralNetwork> createLayeredNetwork(uint32_t num_neurons, uint32_t num_layers) {
        auto network = std::make_unique<NeuralNetwork>();
        
        uint32_t neurons_per_layer = num_neurons / num_layers;
        
        for (uint32_t i = 0; i < num_neurons; ++i) {
            network->addNeuron(i, 0.8f);
        }
        
        // 层间连接
        for (uint32_t layer = 0; layer < num_layers - 1; ++layer) {
            for (uint32_t i = 0; i < neurons_per_layer; ++i) {
                for (uint32_t j = 0; j < neurons_per_layer; ++j) {
                    uint32_t src = layer * neurons_per_layer + i;
                    uint32_t dst = (layer + 1) * neurons_per_layer + j;
                    if (src < num_neurons && dst < num_neurons) {
                        network->addConnection(src, dst, 0.3f, 1.0f);
                    }
                }
            }
        }
        
        return network;
    }
    
    std::unique_ptr<HardwareTopology> createTestTopology() {
        auto topology = std::make_unique<HardwareTopology>();
        
        // 2x2网格
        for (uint32_t i = 0; i < 4; ++i) {
            topology->addPE(i, 16, 1024, 100.0f);
        }
        
        // 连接
        topology->addConnection(0, 1, 5.0f, 1000.0f);  // PE0 -> PE1
        topology->addConnection(1, 3, 5.0f, 1000.0f);  // PE1 -> PE3
        topology->addConnection(0, 2, 5.0f, 1000.0f);  // PE0 -> PE2
        topology->addConnection(2, 3, 5.0f, 1000.0f);  // PE2 -> PE3
        
        return topology;
    }
    
    std::unique_ptr<HardwareTopology> createScalableTopology(uint32_t num_pes) {
        auto topology = std::make_unique<HardwareTopology>();
        
        for (uint32_t i = 0; i < num_pes; ++i) {
            topology->addPE(i, 16, 1024, 100.0f);
            
            // 连接到下一个PE（环形）
            if (i < num_pes - 1) {
                topology->addConnection(i, i + 1, 5.0f, 1000.0f);
            } else {
                topology->addConnection(i, 0, 5.0f, 1000.0f);
            }
        }
        
        return topology;
    }
    
    std::unique_ptr<MappingSolution> createTestMappingSolution(
            const NeuralNetwork& network, const HardwareTopology& topology) {
        auto solution = std::make_unique<MappingSolution>(network.getNeuronCount());
        
        // 简单轮询分配
        auto pe_ids = topology.getAllPEIds();
        for (uint32_t i = 0; i < network.getNeuronCount(); ++i) {
            PEId pe_id = pe_ids[i % pe_ids.size()];
            solution->assignNeuron(i, pe_id);
        }
        
        return solution;
    }
    
    std::unique_ptr<RoutingAwareMappingSolution> createTestRoutingAwareSolution(
            const NeuralNetwork& network, const HardwareTopology& topology) {
        auto solution = std::make_unique<RoutingAwareMappingSolution>(network.getNeuronCount());
        
        // 分配神经元
        auto pe_ids = topology.getAllPEIds();
        for (uint32_t i = 0; i < network.getNeuronCount(); ++i) {
            PEId pe_id = pe_ids[i % pe_ids.size()];
            solution->assignNeuron(i, pe_id);
        }
        
        // 分配全局ID
        solution->batchAssignGlobalIds(network);
        
        // 设置多播管理器
        auto manager = std::make_shared<MulticastGroupManager>();
        solution->setMulticastGroupManager(manager);
        
        return solution;
    }
};

int main() {
    std::cout << "路由系统测试套件\n";
    std::cout << "===============\n\n";
    
    try {
        RoutingSystemTests tests;
        tests.runAllTests();
    } catch (const std::exception& e) {
        std::cerr << "测试过程中发生异常: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}