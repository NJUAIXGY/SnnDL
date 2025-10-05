#ifndef NEURON_MAPPING_MAPPING_SOLUTION_H
#define NEURON_MAPPING_MAPPING_SOLUTION_H

#include "Types.h"
#include "NeuralNetwork.h"
#include "HardwareTopology.h"
#include <vector>
#include <unordered_map>
#include <memory>

namespace NeuronMapping {

/**
 * @brief 神经元映射解决方案类
 * 
 * 表示神经元到PE的映射分配方案，包括映射关系、性能指标和相关元数据。
 * 提供映射操作、性能评估和解决方案比较功能。
 */
class MappingSolution {
public:
    MappingSolution() = default;
    explicit MappingSolution(uint32_t num_neurons);
    virtual ~MappingSolution() = default;
    
    // === 映射分配管理 ===
    
    /**
     * @brief 分配神经元到PE
     * @param neuron_id 神经元ID
     * @param pe_id PE ID
     * @param core_id 核心ID（可选）
     * @return 是否成功分配
     */
    bool assignNeuron(NeuronId neuron_id, PEId pe_id, uint32_t core_id = 0);
    
    /**
     * @brief 移动神经元到新的PE
     * @param neuron_id 神经元ID
     * @param target_pe_id 目标PE ID
     * @param core_id 核心ID（可选）
     * @return 移动成功返回true
     */
    bool moveNeuron(NeuronId neuron_id, PEId target_pe_id, uint32_t core_id = 0);
    
    /**
     * @brief 批量分配神经元
     * @param assignments 分配列表
     * @return 成功分配的数量
     */
    uint32_t assignNeurons(const std::vector<Assignment>& assignments);
    
    /**
     * @brief 取消神经元分配
     * @param neuron_id 神经元ID
     * @return 是否成功取消分配
     */
    bool unassignNeuron(NeuronId neuron_id);
    
    /**
     * @brief 重新分配神经元
     * @param neuron_id 神经元ID
     * @param new_pe_id 新的PE ID
     * @param new_core_id 新的核心ID（可选）
     * @return 是否成功重新分配
     */
    bool reassignNeuron(NeuronId neuron_id, PEId new_pe_id, uint32_t new_core_id = 0);
    
    /**
     * @brief 交换两个神经元的分配
     * @param neuron1 神经元1 ID
     * @param neuron2 神经元2 ID
     * @return 是否成功交换
     */
    bool swapNeuronAssignments(NeuronId neuron1, NeuronId neuron2);
    
    // === 映射查询 ===
    
    /**
     * @brief 获取神经元所分配的PE
     * @param neuron_id 神经元ID
     * @return PE ID，未分配返回INVALID_PE_ID
     */
    PEId getNeuronPE(NeuronId neuron_id) const;
    
    /**
     * @brief 获取神经元所分配的核心
     * @param neuron_id 神经元ID
     * @return 核心ID
     */
    uint32_t getNeuronCore(NeuronId neuron_id) const;
    
    /**
     * @brief 获取神经元的分配信息
     * @param neuron_id 神经元ID
     * @return 分配信息指针，未分配返回nullptr
     */
    const Assignment* getNeuronAssignment(NeuronId neuron_id) const;
    
    /**
     * @brief 检查神经元是否已分配
     * @param neuron_id 神经元ID
     * @return 是否已分配
     */
    bool isNeuronAssigned(NeuronId neuron_id) const;
    
    /**
     * @brief 获取PE上分配的所有神经元
     * @param pe_id PE ID
     * @return 神经元ID列表
     */
    std::vector<NeuronId> getPENeurons(PEId pe_id) const;
    
    /**
     * @brief 获取PE上分配的神经元数量
     * @param pe_id PE ID
     * @return 神经元数量
     */
    uint32_t getPENeuronCount(PEId pe_id) const;
    
    /**
     * @brief 获取所有分配信息
     * @return 分配信息列表
     */
    std::vector<Assignment> getAllAssignments() const;
    
    /**
     * @brief 获取已分配的神经元数量
     * @return 已分配神经元数
     */
    uint32_t getAssignedNeuronCount() const { return static_cast<uint32_t>(neuron_to_assignment_.size()); }
    
    /**
     * @brief 获取使用的PE数量
     * @return 使用的PE数
     */
    uint32_t getUsedPECount() const { return static_cast<uint32_t>(pe_to_neurons_.size()); }
    
    // === 容量管理 ===
    
    /**
     * @brief 检查PE是否还能容纳更多神经元
     * @param pe_id PE ID
     * @param additional_neurons 额外神经元数量
     * @param topology 硬件拓扑
     * @return 是否可以容纳
     */
    bool canPEAccommodate(PEId pe_id, uint32_t additional_neurons, 
                         const HardwareTopology& topology) const;
    
    /**
     * @brief 获取PE的剩余容量
     * @param pe_id PE ID
     * @param topology 硬件拓扑
     * @return 剩余神经元容量
     */
    uint32_t getPERemainingCapacity(PEId pe_id, const HardwareTopology& topology) const;
    
    /**
     * @brief 计算PE的负载
     * @param pe_id PE ID
     * @param network 神经网络
     * @return PE负载
     */
    float calculatePELoad(PEId pe_id, const NeuralNetwork& network) const;
    
    /**
     * @brief 计算所有PE的负载
     * @param network 神经网络
     * @return PE负载映射
     */
    std::unordered_map<PEId, float> calculateAllPELoads(const NeuralNetwork& network) const;
    
    // === 性能评估 ===
    
    /**
     * @brief 评估映射性能
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param config 映射配置
     * @return 性能指标
     */
    PerformanceMetrics evaluatePerformance(const NeuralNetwork& network,
                                          const HardwareTopology& topology,
                                          const MappingConfig& config) const;
    
    /**
     * @brief 计算通信成本
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @return 总通信成本
     */
    float calculateCommunicationCost(const NeuralNetwork& network,
                                   const HardwareTopology& topology) const;
    
    /**
     * @brief 计算负载不均衡
     * @param network 神经网络
     * @return 负载不均衡因子
     */
    float calculateLoadImbalance(const NeuralNetwork& network) const;
    
    /**
     * @brief 计算内存使用情况
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @return 内存使用指标
     */
    std::tuple<float, float, float> calculateMemoryUsage(const NeuralNetwork& network,
                                                        const HardwareTopology& topology) const;
    
    /**
     * @brief 计算跨PE通信比例
     * @param network 神经网络
     * @return 跨PE通信比例 [0, 1]
     */
    float calculateInterPECommunicationRatio(const NeuralNetwork& network) const;
    
    /**
     * @brief 计算平均通信距离
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @return 平均通信距离（跳数）
     */
    float calculateAverageCommunicationDistance(const NeuralNetwork& network,
                                              const HardwareTopology& topology) const;
    
    // === 映射优化 ===
    
    /**
     * @brief 随机初始化映射
     * @param num_neurons 神经元数量
     * @param topology 硬件拓扑
     * @param seed 随机种子
     * @return 是否成功初始化
     */
    bool randomInitialize(uint32_t num_neurons, const HardwareTopology& topology, uint32_t seed = 0);
    
    /**
     * @brief 贪心初始化映射
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @return 是否成功初始化
     */
    bool greedyInitialize(const NeuralNetwork& network, const HardwareTopology& topology);
    
    /**
     * @brief 生成邻居解（用于局部搜索）
     * @param topology 硬件拓扑
     * @param max_moves 最大移动数
     * @return 邻居解
     */
    std::unique_ptr<MappingSolution> generateNeighborSolution(const HardwareTopology& topology,
                                                             uint32_t max_moves = 1) const;
    
    /**
     * @brief 局部优化
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param config 映射配置
     * @param max_iterations 最大迭代次数
     * @return 优化后的性能提升
     */
    float localOptimize(const NeuralNetwork& network,
                       const HardwareTopology& topology,
                       const MappingConfig& config,
                       uint32_t max_iterations = 100);
    
    // === 约束验证 ===
    
    /**
     * @brief 验证映射约束
     * @param topology 硬件拓扑
     * @return 约束违反列表
     */
    std::vector<std::string> validateConstraints(const HardwareTopology& topology) const;
    
    /**
     * @brief 检查容量约束
     * @param topology 硬件拓扑
     * @return 是否满足容量约束
     */
    bool checkCapacityConstraints(const HardwareTopology& topology) const;
    
    /**
     * @brief 修复约束违反
     * @param topology 硬件拓扑
     * @return 是否成功修复
     */
    bool repairConstraintViolations(const HardwareTopology& topology);
    
    // === 解决方案比较 ===
    
    /**
     * @brief 比较两个映射解决方案
     * @param other 另一个解决方案
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param config 映射配置
     * @return 比较结果：< 0表示当前解更好，> 0表示other更好，= 0表示相等
     */
    int compare(const MappingSolution& other,
               const NeuralNetwork& network,
               const HardwareTopology& topology,
               const MappingConfig& config) const;
    
    /**
     * @brief 计算解决方案的适应度分数
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param config 映射配置
     * @return 适应度分数（越高越好）
     */
    float calculateFitnessScore(const NeuralNetwork& network,
                               const HardwareTopology& topology,
                               const MappingConfig& config) const;
    
    // === 数据管理 ===
    
    /**
     * @brief 清空映射
     */
    void clear();
    
    /**
     * @brief 复制映射解决方案
     * @return 深拷贝的解决方案
     */
    std::unique_ptr<MappingSolution> clone() const;
    
    /**
     * @brief 合并另一个解决方案的部分映射
     * @param other 另一个解决方案
     * @param neuron_subset 要合并的神经元子集
     * @return 合并的映射数量
     */
    uint32_t mergePartialMapping(const MappingSolution& other, 
                                const std::vector<NeuronId>& neuron_subset);
    
    /**
     * @brief 获取映射统计信息
     * @return 映射统计字符串
     */
    std::string getStatistics() const;
    
    /**
     * @brief 验证映射一致性
     * @return 错误信息列表，空列表表示无错误
     */
    std::vector<std::string> validateConsistency() const;
    
    // === 缓存的性能指标 ===
    
    /**
     * @brief 获取缓存的性能指标
     * @return 性能指标
     */
    const PerformanceMetrics& getCachedMetrics() const { return cached_metrics_; }
    
    /**
     * @brief 设置缓存的性能指标
     * @param metrics 性能指标
     */
    void setCachedMetrics(const PerformanceMetrics& metrics) { 
        cached_metrics_ = metrics; 
        metrics_valid_ = true;
    }
    
    /**
     * @brief 清除缓存的性能指标
     */
    void clearCachedMetrics() { 
        cached_metrics_.reset();
        metrics_valid_ = false; 
    }
    
    /**
     * @brief 检查缓存的性能指标是否有效
     * @return 是否有效
     */
    bool hasValidCachedMetrics() const { return metrics_valid_; }

protected:
    // 内部数据结构
    std::unordered_map<NeuronId, Assignment> neuron_to_assignment_; // 神经元到分配的映射
    std::unordered_map<PEId, std::vector<NeuronId>> pe_to_neurons_; // PE到神经元列表的映射
    
    // 缓存的性能指标
    PerformanceMetrics cached_metrics_;
    bool metrics_valid_ = false;
    
    // 辅助方法
    void addNeuronToPE(NeuronId neuron_id, PEId pe_id);
    void removeNeuronFromPE(NeuronId neuron_id, PEId pe_id);
    void updatePEMapping(NeuronId neuron_id, PEId old_pe_id, PEId new_pe_id);
    
private:
    // 内部性能计算辅助函数
    float computeObjectiveFunction(const NeuralNetwork& network,
                                  const HardwareTopology& topology,
                                  const MappingConfig& config) const;
    
    std::pair<uint32_t, uint32_t> countLocalAndRemoteConnections(const NeuralNetwork& network) const;
    
    std::vector<float> calculatePELoadDistribution(const NeuralNetwork& network) const;
};

} // namespace NeuronMapping

#endif // NEURON_MAPPING_MAPPING_SOLUTION_H