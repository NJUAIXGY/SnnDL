# 神经元映射框架技术文档

## 目录

1. [架构概述](#架构概述)
2. [核心组件详解](#核心组件详解)
3. [算法实现原理](#算法实现原理)
4. [性能优化策略](#性能优化策略)
5. [扩展开发指南](#扩展开发指南)
6. [API参考](#api参考)

## 架构概述

### 总体架构

神经元映射框架采用分层架构设计，具有高度的模块化和可扩展性：

```
┌─────────────────────────────────────────┐
│              用户接口层                    │
│  NeuronMapper | MapperFactory           │
├─────────────────────────────────────────┤
│              策略层                      │
│  Random | Greedy | GraphPartitioning    │
├─────────────────────────────────────────┤
│              优化器层                     │
│  SimulatedAnnealing | LocalSearch       │
├─────────────────────────────────────────┤
│              评估器层                     │
│  Performance | Communication | Load     │
├─────────────────────────────────────────┤
│              算法层                      │
│  CommunityDetector | NetworkAnalyzer    │
├─────────────────────────────────────────┤
│              核心数据层                   │
│  NeuralNetwork | HardwareTopology       │
└─────────────────────────────────────────┘
```

### 设计原则

1. **分离关注点**：每个组件专注于特定功能，减少耦合
2. **策略模式**：支持算法的动态选择和替换
3. **工厂模式**：统一对象创建，简化配置管理
4. **RAII原则**：自动资源管理，避免内存泄漏
5. **接口导向**：通过抽象接口实现松耦合

## 核心组件详解

### 1. 神经网络表示 (NeuralNetwork)

#### 数据结构设计

```cpp
class NeuralNetwork {
private:
    std::unordered_map<NeuronId, NeuronProperties> neurons_;
    std::vector<Connection> connections_;
    std::unordered_map<NeuronId, std::vector<size_t>> in_connections_;
    std::unordered_map<NeuronId, std::vector<size_t>> out_connections_;
    NetworkStatistics stats_;
};
```

#### 关键特性

- **稀疏表示**：使用邻接表优化内存使用
- **快速查询**：O(1)时间复杂度的神经元访问
- **统计缓存**：预计算网络统计信息，避免重复计算
- **增量更新**：支持动态添加/删除神经元和连接

#### 性能优化

```cpp
// 批量添加连接，减少重复操作
void addConnections(const std::vector<Connection>& connections) {
    connections_.reserve(connections_.size() + connections.size());
    for (const auto& conn : connections) {
        addConnection(conn);
    }
    updateStatistics();  // 批量更新统计信息
}
```

### 2. 硬件拓扑 (HardwareTopology)

#### 拓扑类型支持

1. **网格拓扑 (Mesh)**
   - 2D/3D网格结构
   - 支持torus环绕连接
   - 高效的Manhattan距离计算

2. **树形拓扑 (Tree)**
   - 二叉树/k-ary树
   - 层次化通信模式
   - 适合分层神经网络

3. **自定义拓扑**
   - 用户自定义连接模式
   - 灵活的距离函数

#### 距离计算优化

```cpp
// 使用缓存避免重复计算
class HardwareTopology {
private:
    mutable std::unordered_map<std::pair<PEId, PEId>, uint32_t> distance_cache_;
    
public:
    uint32_t getDistance(PEId pe1, PEId pe2) const {
        auto key = std::make_pair(std::min(pe1, pe2), std::max(pe1, pe2));
        auto it = distance_cache_.find(key);
        if (it != distance_cache_.end()) {
            return it->second;
        }
        
        uint32_t dist = calculateDistance(pe1, pe2);
        distance_cache_[key] = dist;
        return dist;
    }
};
```

### 3. 映射解 (MappingSolution)

#### 数据结构

```cpp
class MappingSolution {
private:
    std::vector<PEId> neuron_to_pe_;          // 神经元->PE映射
    std::vector<uint32_t> neuron_to_core_;    // 神经元->核心映射
    std::vector<std::set<NeuronId>> pe_assignments_; // PE->神经元列表
    PerformanceMetrics cached_metrics_;        // 缓存的性能指标
    bool metrics_valid_;                      // 指标有效性标记
};
```

#### 性能评估

性能评估采用多指标综合评价：

```cpp
PerformanceMetrics MappingSolution::evaluatePerformance(
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const MappingConfig& config) {
    
    if (metrics_valid_) {
        return cached_metrics_;  // 返回缓存结果
    }
    
    PerformanceMetrics metrics;
    
    // 1. 计算通信成本
    metrics.communication_cost = calculateCommunicationCost(network, topology);
    
    // 2. 计算负载均衡
    metrics.load_imbalance = calculateLoadImbalance(network);
    
    // 3. 计算内存使用
    metrics.memory_usage = calculateMemoryUsage(network);
    
    // 4. 计算综合适应度
    metrics.fitness = config.communication_weight * metrics.communication_cost +
                     config.load_balance_weight * metrics.load_imbalance +
                     config.memory_weight * metrics.memory_usage;
    
    cached_metrics_ = metrics;
    metrics_valid_ = true;
    
    return metrics;
}
```

## 算法实现原理

### 1. 图分割映射算法

#### 多层次分割 (Multilevel Partitioning)

算法分为三个主要阶段：

1. **粗化阶段 (Coarsening)**
```cpp
NeuralNetwork GraphPartitioningStrategy::coarsenGraph(const NeuralNetwork& graph) {
    NeuralNetwork coarsened;
    std::unordered_map<NeuronId, NeuronId> vertex_mapping;
    
    // 使用重边匹配算法
    auto matching = findHeavyEdgeMatching(graph);
    
    // 合并匹配的顶点
    for (const auto& [v1, v2] : matching) {
        NeuronId new_vertex = coarsened.addMergedNeuron(
            graph.getNeuron(v1), graph.getNeuron(v2));
        vertex_mapping[v1] = vertex_mapping[v2] = new_vertex;
    }
    
    // 重构边权重
    updateEdgeWeights(coarsened, graph, vertex_mapping);
    
    return coarsened;
}
```

2. **初始分割阶段 (Initial Partitioning)**
```cpp
std::vector<Partition> GraphPartitioningStrategy::initialPartition(
    const NeuralNetwork& coarse_graph, uint32_t num_partitions) {
    
    // 使用贪心算法进行初始分割
    std::vector<Partition> partitions(num_partitions);
    auto vertices = coarse_graph.getAllNeuronIds();
    
    // 按权重排序
    std::sort(vertices.begin(), vertices.end(), 
        [&](NeuronId a, NeuronId b) {
            return coarse_graph.getNeuron(a).computational_load > 
                   coarse_graph.getNeuron(b).computational_load;
        });
    
    // 轮询分配到负载最小的分区
    for (NeuronId vertex : vertices) {
        auto min_partition = std::min_element(partitions.begin(), partitions.end(),
            [](const Partition& a, const Partition& b) {
                return a.total_load < b.total_load;
            });
        min_partition->addVertex(vertex);
    }
    
    return partitions;
}
```

3. **细化阶段 (Refinement)**
```cpp
void GraphPartitioningStrategy::refinePartition(
    std::vector<Partition>& partitions, 
    const NeuralNetwork& graph) {
    
    bool improved = true;
    while (improved) {
        improved = false;
        
        // Kernighan-Lin风格的改进
        for (auto& partition : partitions) {
            for (NeuronId vertex : partition.getBoundaryVertices()) {
                auto best_move = findBestMove(vertex, partitions, graph);
                if (best_move.gain > 0) {
                    executeMove(vertex, best_move.target_partition, partitions);
                    improved = true;
                }
            }
        }
    }
}
```

#### 谱分割 (Spectral Partitioning)

基于拉普拉斯矩阵的特征向量：

```cpp
std::vector<uint32_t> GraphPartitioningStrategy::spectralPartitioning(
    const NeuralNetwork& network, uint32_t num_partitions) {
    
    // 1. 构建拉普拉斯矩阵
    auto laplacian = buildLaplacianMatrix(network);
    
    // 2. 计算特征向量（简化实现）
    auto eigenvectors = computeEigenvectors(laplacian, num_partitions);
    
    // 3. K-means聚类
    return performKMeansClustering(eigenvectors, num_partitions);
}
```

### 2. 模拟退火优化

#### 算法核心流程

```cpp
std::unique_ptr<MappingSolution> SimulatedAnnealingOptimizer::optimize(
    std::unique_ptr<MappingSolution> initial_solution,
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const OptimizationConfig& config) {
    
    auto current_solution = std::move(initial_solution);
    auto best_solution = std::make_unique<MappingSolution>(*current_solution);
    
    float current_cost = evaluateObjective(*current_solution, network, topology);
    float best_cost = current_cost;
    
    float temperature = initial_temperature_;
    uint32_t iteration = 0;
    
    while (temperature > final_temperature_ && iteration < config.max_iterations) {
        // 生成邻域解
        auto neighbor = generateNeighbor(*current_solution, network, topology);
        if (!neighbor) {
            continue;
        }
        
        float neighbor_cost = evaluateObjective(*neighbor, network, topology);
        float delta = neighbor_cost - current_cost;
        
        // 接受准则
        bool accept = false;
        if (delta < 0) {
            accept = true;  // 更优解直接接受
        } else {
            float probability = std::exp(-delta / temperature);
            accept = (rng_uniform_(rng_) < probability);
        }
        
        if (accept) {
            current_solution = std::move(neighbor);
            current_cost = neighbor_cost;
            moves_accepted_++;
            
            // 更新最优解
            if (neighbor_cost < best_cost) {
                best_solution = std::make_unique<MappingSolution>(*current_solution);
                best_cost = neighbor_cost;
            }
        }
        
        // 更新温度
        temperature = updateTemperature(temperature, iteration);
        iteration++;
    }
    
    return best_solution;
}
```

#### 冷却策略

```cpp
float SimulatedAnnealingOptimizer::updateTemperature(float current_temp, uint32_t iteration) {
    switch (cooling_schedule_) {
        case CoolingSchedule::LINEAR:
            return initial_temperature_ * (1.0f - float(iteration) / max_iterations_);
            
        case CoolingSchedule::EXPONENTIAL:
            return current_temp * cooling_rate_;
            
        case CoolingSchedule::LOGARITHMIC:
            return initial_temperature_ / std::log(iteration + 2);
            
        case CoolingSchedule::ADAPTIVE:
            // 根据接受率动态调整
            float acceptance_rate = float(moves_accepted_) / (iteration + 1);
            if (acceptance_rate > 0.6f) {
                return current_temp * 0.95f;  // 降温更快
            } else if (acceptance_rate < 0.2f) {
                return current_temp * 0.99f;  // 降温更慢
            }
            return current_temp * cooling_rate_;
            
        default:
            return current_temp * cooling_rate_;
    }
}
```

### 3. 社区检测算法

#### Louvain算法

```cpp
std::unique_ptr<CommunityStructure> CommunityDetector::louvainAlgorithm(
    const NeuralNetwork& network) {
    
    auto neurons = network.getAllNeuronIds();
    std::vector<uint32_t> communities(neurons.size());
    
    // 初始化：每个节点为一个社区
    for (size_t i = 0; i < neurons.size(); ++i) {
        communities[i] = i;
    }
    
    bool improvement = true;
    while (improvement) {
        improvement = false;
        
        // 第一阶段：节点移动
        for (size_t i = 0; i < neurons.size(); ++i) {
            NeuronId neuron = neurons[i];
            uint32_t current_community = communities[i];
            
            // 寻找最佳邻居社区
            float best_gain = 0.0f;
            uint32_t best_community = current_community;
            
            for (auto neighbor_community : getNeighborCommunities(neuron, communities, network)) {
                float gain = calculateModularityGain(neuron, neighbor_community, communities, network);
                if (gain > best_gain) {
                    best_gain = gain;
                    best_community = neighbor_community;
                }
            }
            
            // 移动到最佳社区
            if (best_community != current_community) {
                communities[i] = best_community;
                improvement = true;
            }
        }
        
        // 第二阶段：社区合并（简化版本）
        if (!improvement) {
            improvement = mergeSmallCommunities(communities, network);
        }
    }
    
    return convertToCommunityStructure(communities, network);
}
```

## 性能优化策略

### 1. 内存优化

#### 稀疏数据结构

```cpp
class SparseConnectionMatrix {
private:
    // CSR格式存储
    std::vector<float> values_;        // 非零值
    std::vector<size_t> column_indices_; // 列索引
    std::vector<size_t> row_pointers_;   // 行指针
    
public:
    float getConnection(NeuronId src, NeuronId tgt) const {
        size_t start = row_pointers_[src];
        size_t end = row_pointers_[src + 1];
        
        auto it = std::lower_bound(
            column_indices_.begin() + start,
            column_indices_.begin() + end,
            tgt
        );
        
        if (it != column_indices_.begin() + end && *it == tgt) {
            return values_[it - column_indices_.begin()];
        }
        return 0.0f;
    }
};
```

#### 对象池模式

```cpp
template<typename T>
class ObjectPool {
private:
    std::vector<std::unique_ptr<T>> pool_;
    std::queue<T*> available_;
    
public:
    T* acquire() {
        if (available_.empty()) {
            pool_.emplace_back(std::make_unique<T>());
            return pool_.back().get();
        }
        
        T* obj = available_.front();
        available_.pop();
        return obj;
    }
    
    void release(T* obj) {
        obj->reset();  // 重置对象状态
        available_.push(obj);
    }
};
```

### 2. 计算优化

#### 并行计算

```cpp
void GraphPartitioningStrategy::parallelPartitioning(
    const NeuralNetwork& network,
    std::vector<Partition>& partitions) {
    
    const uint32_t num_threads = std::thread::hardware_concurrency();
    std::vector<std::thread> workers;
    std::mutex partition_mutex;
    
    auto process_vertices = [&](size_t start, size_t end) {
        for (size_t i = start; i < end; ++i) {
            auto best_partition = findBestPartition(vertices[i], partitions, network);
            
            std::lock_guard<std::mutex> lock(partition_mutex);
            partitions[best_partition].addVertex(vertices[i]);
        }
    };
    
    // 分配工作负载
    auto vertices = network.getAllNeuronIds();
    size_t chunk_size = vertices.size() / num_threads;
    
    for (uint32_t t = 0; t < num_threads; ++t) {
        size_t start = t * chunk_size;
        size_t end = (t == num_threads - 1) ? vertices.size() : (t + 1) * chunk_size;
        workers.emplace_back(process_vertices, start, end);
    }
    
    // 等待所有线程完成
    for (auto& worker : workers) {
        worker.join();
    }
}
```

#### 缓存友好的数据访问

```cpp
// 按内存布局顺序访问数据
void optimizedEvaluation(const MappingSolution& solution) {
    // 预排序以提高缓存命中率
    auto neurons = solution.getAssignedNeurons();
    std::sort(neurons.begin(), neurons.end(), 
        [&](NeuronId a, NeuronId b) {
            return solution.getPE(a) < solution.getPE(b);
        });
    
    // 按PE分组处理
    PEId current_pe = INVALID_PE_ID;
    for (NeuronId neuron : neurons) {
        PEId pe = solution.getPE(neuron);
        if (pe != current_pe) {
            // 切换到新PE时，预加载相关数据
            prefetchPEData(pe);
            current_pe = pe;
        }
        processNeuron(neuron);
    }
}
```

### 3. 算法优化

#### 早停策略

```cpp
bool checkConvergence(const std::vector<float>& cost_history, 
                     float threshold, size_t window_size) {
    if (cost_history.size() < window_size) return false;
    
    // 计算最近window_size次迭代的改进幅度
    float recent_improvement = 0.0f;
    for (size_t i = cost_history.size() - window_size; i < cost_history.size() - 1; ++i) {
        recent_improvement += std::abs(cost_history[i] - cost_history[i + 1]);
    }
    
    return (recent_improvement / window_size) < threshold;
}
```

#### 自适应参数调整

```cpp
void SimulatedAnnealingOptimizer::adaptiveParameterTuning(uint32_t iteration) {
    float acceptance_rate = float(moves_accepted_) / iteration;
    
    // 根据接受率调整邻域大小
    if (acceptance_rate > 0.8f) {
        // 接受率过高，增大邻域探索范围
        neighborhood_size_ = std::min(neighborhood_size_ * 1.1f, max_neighborhood_size_);
    } else if (acceptance_rate < 0.2f) {
        // 接受率过低，减小邻域探索范围
        neighborhood_size_ = std::max(neighborhood_size_ * 0.9f, min_neighborhood_size_);
    }
    
    // 动态调整温度衰减率
    if (iteration % 100 == 0) {
        float convergence_rate = calculateConvergenceRate();
        if (convergence_rate < 0.01f) {
            cooling_rate_ *= 1.05f;  // 加快冷却
        } else if (convergence_rate > 0.1f) {
            cooling_rate_ *= 0.95f;  // 减慢冷却
        }
    }
}
```

## 扩展开发指南

### 1. 添加新的映射策略

```cpp
class CustomMappingStrategy : public MappingStrategy {
public:
    std::unique_ptr<MappingSolution> mapNetwork(
        const NeuralNetwork& network,
        const HardwareTopology& topology, 
        const MappingConfig& config) override {
        
        // 1. 预处理分析
        auto analysis = analyzeNetworkStructure(network);
        
        // 2. 生成初始映射
        auto solution = generateInitialMapping(network, topology, analysis);
        
        // 3. 应用策略特定的优化
        applyCustomOptimization(*solution, network, topology);
        
        return solution;
    }
    
private:
    NetworkAnalysis analyzeNetworkStructure(const NeuralNetwork& network) {
        NetworkAnalysis analysis;
        
        // 分析网络特征（层次结构、连接密度等）
        analysis.layer_structure = detectLayerStructure(network);
        analysis.connectivity_patterns = analyzeConnectivityPatterns(network);
        analysis.critical_paths = findCriticalPaths(network);
        
        return analysis;
    }
};
```

### 2. 自定义优化器

```cpp
class GeneticAlgorithmOptimizer : public MappingOptimizer {
private:
    uint32_t population_size_;
    float mutation_rate_;
    float crossover_rate_;
    
public:
    std::unique_ptr<MappingSolution> optimize(
        std::unique_ptr<MappingSolution> initial_solution,
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const OptimizationConfig& config) override {
        
        // 1. 初始化种群
        auto population = initializePopulation(*initial_solution, population_size_);
        
        for (uint32_t generation = 0; generation < config.max_iterations; ++generation) {
            // 2. 评估适应度
            evaluatePopulation(population, network, topology);
            
            // 3. 选择操作
            auto parents = selectParents(population);
            
            // 4. 交叉操作
            auto offspring = performCrossover(parents);
            
            // 5. 变异操作
            performMutation(offspring, network, topology);
            
            // 6. 环境选择
            population = environmentalSelection(population, offspring);
            
            // 7. 检查收敛
            if (checkConvergence(population, config)) {
                break;
            }
        }
        
        return std::make_unique<MappingSolution>(*getBestIndividual(population));
    }
};
```

### 3. 自定义评估指标

```cpp
class CustomEvaluator : public PerformanceEvaluator {
public:
    PerformanceMetrics evaluateBasic(
        const MappingSolution& solution,
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config) override {
        
        PerformanceMetrics metrics = PerformanceEvaluator::evaluateBasic(
            solution, network, topology, config);
        
        // 添加自定义指标
        metrics.custom_metric_1 = calculatePowerConsumption(solution, topology);
        metrics.custom_metric_2 = calculateThermalDistribution(solution, topology);
        metrics.custom_metric_3 = calculateFaultTolerance(solution, network);
        
        return metrics;
    }
    
private:
    float calculatePowerConsumption(const MappingSolution& solution, 
                                   const HardwareTopology& topology) {
        float total_power = 0.0f;
        
        for (PEId pe = 0; pe < topology.getTotalPEs(); ++pe) {
            auto assigned_neurons = solution.getAssignedNeurons(pe);
            float pe_load = calculatePELoad(assigned_neurons);
            
            // 功耗模型：P = α * load² + β * load + γ
            total_power += power_alpha_ * pe_load * pe_load + 
                          power_beta_ * pe_load + 
                          power_gamma_;
        }
        
        return total_power;
    }
};
```

## API参考

### 核心接口

#### NeuralNetwork

```cpp
class NeuralNetwork {
public:
    // 神经元管理
    bool addNeuron(const NeuronProperties& neuron);
    bool removeNeuron(NeuronId id);
    NeuronProperties getNeuron(NeuronId id) const;
    std::vector<NeuronId> getAllNeuronIds() const;
    
    // 连接管理
    bool addConnection(const Connection& connection);
    bool removeConnection(NeuronId src, NeuronId tgt);
    std::vector<Connection> getAllConnections() const;
    std::vector<Connection> getIncomingConnections(NeuronId neuron) const;
    std::vector<Connection> getOutgoingConnections(NeuronId neuron) const;
    
    // 统计信息
    uint32_t getNeuronCount() const;
    uint32_t getConnectionCount() const;
    NetworkStatistics getStatistics() const;
    
    // 查询操作
    bool hasNeuron(NeuronId id) const;
    bool hasConnection(NeuronId src, NeuronId tgt) const;
    float getConnectionWeight(NeuronId src, NeuronId tgt) const;
};
```

#### HardwareTopology

```cpp
class HardwareTopology {
public:
    // 拓扑创建
    bool createMesh2D(uint32_t width, uint32_t height, const ProcessingElement& pe_template);
    bool createMesh3D(uint32_t width, uint32_t height, uint32_t depth, const ProcessingElement& pe_template);
    bool createTorus2D(uint32_t width, uint32_t height, const ProcessingElement& pe_template);
    bool createTree(uint32_t levels, uint32_t branching_factor, const ProcessingElement& pe_template);
    
    // 查询操作
    uint32_t getTotalPEs() const;
    ProcessingElement getPE(PEId id) const;
    std::vector<PEId> getAllPEIds() const;
    
    // 距离计算
    uint32_t getDistance(PEId pe1, PEId pe2) const;
    std::vector<PEId> getShortestPath(PEId src, PEId tgt) const;
    std::vector<PEId> getNeighbors(PEId pe) const;
    
    // 拓扑分析
    TopologyType getTopologyType() const;
    TopologyStatistics getStatistics() const;
    float getAverageDistance() const;
    uint32_t getDiameter() const;
};
```

#### MappingSolution

```cpp
class MappingSolution {
public:
    // 构造函数
    explicit MappingSolution(uint32_t num_neurons);
    MappingSolution(const MappingSolution& other);
    
    // 映射操作
    bool assignNeuron(NeuronId neuron, PEId pe, uint32_t core = 0);
    bool unassignNeuron(NeuronId neuron);
    bool moveNeuron(NeuronId neuron, PEId target_pe, uint32_t core = 0);
    
    // 查询操作
    PEId getPE(NeuronId neuron) const;
    uint32_t getCore(NeuronId neuron) const;
    std::vector<NeuronId> getAssignedNeurons(PEId pe) const;
    std::vector<NeuronId> getUnassignedNeurons() const;
    
    // 统计信息
    uint32_t getAssignedNeuronCount() const;
    uint32_t getUnassignedNeuronCount() const;
    uint32_t getUsedPECount() const;
    
    // 性能评估
    PerformanceMetrics evaluatePerformance(
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config);
    
    // 初始化方法
    void randomInitialize(const NeuralNetwork& network, const HardwareTopology& topology);
    void greedyInitialize(const NeuralNetwork& network, const HardwareTopology& topology);
    
    // 验证操作
    bool isValid(const NeuralNetwork& network, const HardwareTopology& topology) const;
    std::vector<std::string> getValidationErrors(const NeuralNetwork& network, const HardwareTopology& topology) const;
};
```

### 配置参数

#### MappingConfig

```cpp
struct MappingConfig {
    // 算法选择
    std::string strategy = "hybrid";              // "layered", "partition", "hybrid", "adaptive"
    std::string optimizer = "simulated_annealing"; // 优化算法
    
    // 目标函数权重
    float communication_weight = 0.6f;            // 通信成本权重
    float load_balance_weight = 0.3f;             // 负载均衡权重
    float memory_weight = 0.1f;                   // 内存使用权重
    
    // 约束参数
    float max_load_imbalance = 1.5f;              // 最大负载不均衡
    float max_memory_usage = 0.9f;                // 最大内存使用率
    
    // 优化参数
    uint32_t max_iterations = 10000;              // 最大迭代次数
    float convergence_threshold = 0.001f;         // 收敛阈值
    uint32_t random_seed = 12345;                 // 随机种子
    
    // 性能参数
    bool enable_parallel_processing = true;       // 并行处理
    uint32_t num_threads = 4;                     // 线程数
    bool enable_verbose_logging = false;          // 详细日志
};
```

#### OptimizationConfig

```cpp
struct OptimizationConfig {
    uint32_t max_iterations = 1000;              // 最大迭代次数
    float convergence_threshold = 1e-6f;         // 收敛阈值
    uint32_t max_stagnation_iterations = 100;    // 最大停滞迭代数
    bool enable_adaptive_parameters = true;      // 自适应参数调整
    
    // 早停策略
    bool enable_early_stopping = true;
    uint32_t early_stopping_patience = 50;
    float early_stopping_threshold = 1e-8f;
    
    // 重启策略  
    bool enable_random_restart = false;
    uint32_t restart_interval = 500;
    uint32_t max_restarts = 3;
    
    // 调试选项
    bool track_optimization_progress = false;
    uint32_t progress_report_interval = 100;
    std::string log_file_path = "";
};
```

---

<function_calls>
<invoke name="TodoWrite">
<parameter name="todos">[{"content": "\u521b\u5efa\u795e\u7ecf\u5143\u6620\u5c04\u6838\u5fc3\u63a5\u53e3\u548c\u6570\u636e\u7ed3\u6784", "status": "completed", "id": "1"}, {"content": "\u5b9e\u73b0\u57fa\u7840\u6570\u636e\u7ed3\u6784 - NeuralNetwork\u7c7b\u7684\u6838\u5fc3\u65b9\u6cd5", "status": "completed", "id": "2"}, {"content": "\u5b9e\u73b0HardwareTopology\u7c7b - \u62d3\u6251\u521b\u5efa\u548c\u8ddd\u79bb\u8ba1\u7b97", "status": "completed", "id": "3"}, {"content": "\u5b9e\u73b0MappingSolution\u7c7b - \u6620\u5c04\u7ba1\u7406\u548c\u6027\u80fd\u8bc4\u4f30", "status": "completed", "id": "4"}, {"content": "\u5b9e\u73b0\u5de5\u5177\u7c7b - Logger\u548cMathUtils\u7684\u6838\u5fc3\u529f\u80fd", "status": "completed", "id": "5"}, {"content": "\u5b9e\u73b0\u7f51\u7edc\u5206\u6790\u5668\u63a5\u53e3\u548c\u57fa\u7840\u7b97\u6cd5", "status": "completed", "id": "6"}, {"content": "\u68c0\u67e5\u5b9e\u73b0\u8bed\u6cd5\u9519\u8bef\u5e76\u4fee\u590d", "status": "completed", "id": "6.5"}, {"content": "\u5b9e\u73b0\u7b80\u5355\u6620\u5c04\u7b56\u7565 - \u968f\u673a\u6620\u5c04\u548c\u8d2a\u5fc3\u6620\u5c04", "status": "completed", "id": "7"}, {"content": "\u5b9e\u73b0\u57fa\u7840\u4f18\u5316\u5668 - \u5c40\u90e8\u641c\u7d22\u7b97\u6cd5", "status": "completed", "id": "8"}, {"content": "\u5b9e\u73b0\u6027\u80fd\u8bc4\u4f30\u5668 - \u901a\u4fe1\u6210\u672c\u548c\u8d1f\u8f7d\u5747\u8861\u8ba1\u7b97", "status": "completed", "id": "9"}, {"content": "\u5b9e\u73b0NeuronMapper\u4e3b\u7c7b\u548c\u5de5\u5382\u6a21\u5f0f", "status": "completed", "id": "10"}, {"content": "\u521b\u5efa\u57fa\u7840\u6d4b\u8bd5\u7528\u4f8b\u9a8c\u8bc1\u6838\u5fc3\u529f\u80fd", "status": "completed", "id": "11"}, {"content": "\u5b9e\u73b0\u9ad8\u7ea7\u6620\u5c04\u7b56\u7565 - \u56fe\u5206\u5272\u7b97\u6cd5", "status": "completed", "id": "12"}, {"content": "\u4fee\u590dLogger\u94fe\u63a5\u95ee\u9898\uff0c\u5b8c\u5584\u65e5\u5fd7\u7cfb\u7edf", "status": "completed", "id": "12.5"}, {"content": "\u5b9e\u73b0\u6a21\u62df\u9000\u706b\u4f18\u5316\u5668", "status": "completed", "id": "13"}, {"content": "\u6dfb\u52a0\u66f4\u591a\u7f51\u7edc\u5206\u6790\u529f\u80fd - \u793e\u533a\u68c0\u6d4b\u7b97\u6cd5", "status": "completed", "id": "14"}, {"content": "\u521b\u5efa\u53ef\u7f16\u8bd1\u8fd0\u884c\u7684\u7efc\u5408\u6d4b\u8bd5\u548c\u6f14\u793a\u7a0b\u5e8f", "status": "completed", "id": "15"}, {"content": "\u7f16\u5199\u9879\u76eeREADME\u548c\u6280\u672f\u6587\u6863", "status": "completed", "id": "16"}]