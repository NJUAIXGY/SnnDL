#ifndef NEURON_MAPPING_NEURAL_NETWORK_H
#define NEURON_MAPPING_NEURAL_NETWORK_H

#include "Types.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <functional>

namespace NeuronMapping {

/**
 * @brief 神经网络表示类
 * 
 * 提供神经网络的完整表示，包括神经元、连接和网络拓扑信息。
 * 支持稀疏和稠密两种存储格式，优化大规模网络的存储和访问。
 */
class NeuralNetwork {
public:
    NeuralNetwork() = default;
    virtual ~NeuralNetwork() = default;
    
    // === 神经元管理 ===
    
    /**
     * @brief 添加神经元
     * @param properties 神经元属性
     * @return 是否成功添加
     */
    bool addNeuron(const NeuronProperties& properties);
    
    /**
     * @brief 批量添加神经元
     * @param neurons 神经元列表
     * @return 成功添加的数量
     */
    uint32_t addNeurons(const std::vector<NeuronProperties>& neurons);
    
    /**
     * @brief 移除神经元
     * @param neuron_id 神经元ID
     * @return 是否成功移除
     */
    bool removeNeuron(NeuronId neuron_id);
    
    /**
     * @brief 检查神经元是否存在
     * @param neuron_id 神经元ID
     * @return 是否存在
     */
    bool hasNeuron(NeuronId neuron_id) const;
    
    /**
     * @brief 获取神经元属性
     * @param neuron_id 神经元ID
     * @return 神经元属性指针，不存在返回nullptr
     */
    const NeuronProperties* getNeuron(NeuronId neuron_id) const;
    
    /**
     * @brief 更新神经元属性
     * @param properties 新的神经元属性
     * @return 是否成功更新
     */
    bool updateNeuron(const NeuronProperties& properties);
    
    /**
     * @brief 获取所有神经元ID
     * @return 神经元ID列表
     */
    std::vector<NeuronId> getAllNeuronIds() const;
    
    /**
     * @brief 获取神经元数量
     * @return 神经元总数
     */
    uint32_t getNeuronCount() const { return static_cast<uint32_t>(neurons_.size()); }
    
    // === 连接管理 ===
    
    /**
     * @brief 添加连接
     * @param connection 连接信息
     * @return 是否成功添加
     */
    bool addConnection(const Connection& connection);
    
    /**
     * @brief 批量添加连接
     * @param connections 连接列表
     * @return 成功添加的数量
     */
    uint32_t addConnections(const std::vector<Connection>& connections);
    
    /**
     * @brief 移除连接
     * @param pre_neuron 前神经元ID
     * @param post_neuron 后神经元ID
     * @return 是否成功移除
     */
    bool removeConnection(NeuronId pre_neuron, NeuronId post_neuron);
    
    /**
     * @brief 检查连接是否存在
     * @param pre_neuron 前神经元ID
     * @param post_neuron 后神经元ID
     * @return 是否存在连接
     */
    bool hasConnection(NeuronId pre_neuron, NeuronId post_neuron) const;
    
    /**
     * @brief 获取连接信息
     * @param pre_neuron 前神经元ID
     * @param post_neuron 后神经元ID
     * @return 连接信息指针，不存在返回nullptr
     */
    const Connection* getConnection(NeuronId pre_neuron, NeuronId post_neuron) const;
    
    /**
     * @brief 更新连接权重
     * @param pre_neuron 前神经元ID
     * @param post_neuron 后神经元ID
     * @param new_weight 新权重
     * @return 是否成功更新
     */
    bool updateConnectionWeight(NeuronId pre_neuron, NeuronId post_neuron, Weight new_weight);
    
    /**
     * @brief 获取所有连接
     * @return 连接列表
     */
    std::vector<Connection> getAllConnections() const;
    
    /**
     * @brief 获取连接数量
     * @return 连接总数
     */
    uint32_t getConnectionCount() const { return static_cast<uint32_t>(connections_.size()); }
    
    // === 网络拓扑查询 ===
    
    /**
     * @brief 获取神经元的输入连接
     * @param neuron_id 神经元ID
     * @return 输入连接列表
     */
    std::vector<Connection> getIncomingConnections(NeuronId neuron_id) const;
    
    /**
     * @brief 获取神经元的输出连接
     * @param neuron_id 神经元ID
     * @return 输出连接列表
     */
    std::vector<Connection> getOutgoingConnections(NeuronId neuron_id) const;
    
    /**
     * @brief 获取神经元的前驱神经元
     * @param neuron_id 神经元ID
     * @return 前驱神经元ID列表
     */
    std::vector<NeuronId> getPredecessors(NeuronId neuron_id) const;
    
    /**
     * @brief 获取神经元的后继神经元
     * @param neuron_id 神经元ID
     * @return 后继神经元ID列表
     */
    std::vector<NeuronId> getSuccessors(NeuronId neuron_id) const;
    
    /**
     * @brief 获取神经元的入度
     * @param neuron_id 神经元ID
     * @return 入度（输入连接数）
     */
    uint32_t getInDegree(NeuronId neuron_id) const;
    
    /**
     * @brief 获取神经元的出度
     * @param neuron_id 神经元ID
     * @return 出度（输出连接数）
     */
    uint32_t getOutDegree(NeuronId neuron_id) const;
    
    /**
     * @brief 获取神经元的总度数
     * @param neuron_id 神经元ID
     * @return 总度数
     */
    uint32_t getDegree(NeuronId neuron_id) const {
        return getInDegree(neuron_id) + getOutDegree(neuron_id);
    }
    
    // === 网络分析 ===
    
    /**
     * @brief 计算网络统计信息
     * @return 网络统计数据
     */
    NetworkStatistics calculateStatistics() const;
    
    /**
     * @brief 检测强连通分量
     * @return 每个神经元所属的连通分量ID
     */
    std::unordered_map<NeuronId, uint32_t> findStronglyConnectedComponents() const;
    
    /**
     * @brief 拓扑排序（仅适用于DAG）
     * @return 拓扑排序后的神经元列表，如果有环返回空
     */
    std::vector<NeuronId> topologicalSort() const;
    
    /**
     * @brief 检测网络是否为DAG（有向无环图）
     * @return 是否为DAG
     */
    bool isDAG() const;
    
    /**
     * @brief 识别网络层次结构
     * @return 每个神经元所属的层次
     */
    std::unordered_map<NeuronId, uint32_t> identifyLayers() const;
    
    /**
     * @brief 计算两个神经元之间的最短路径长度
     * @param source 源神经元
     * @param target 目标神经元
     * @return 最短路径长度，-1表示不可达
     */
    int32_t shortestPathLength(NeuronId source, NeuronId target) const;
    
    // === 网络变换 ===
    
    /**
     * @brief 创建子网络
     * @param neuron_subset 神经元子集
     * @return 子网络
     */
    std::unique_ptr<NeuralNetwork> createSubnetwork(const std::vector<NeuronId>& neuron_subset) const;
    
    /**
     * @brief 网络剪枝
     * @param weight_threshold 权重阈值，小于此值的连接被移除
     * @return 被移除的连接数
     */
    uint32_t pruneConnections(Weight weight_threshold);
    
    /**
     * @brief 随机化连接权重
     * @param min_weight 最小权重
     * @param max_weight 最大权重
     * @param seed 随机种子
     */
    void randomizeWeights(Weight min_weight, Weight max_weight, uint32_t seed = 0);
    
    // === 数据导入导出 ===
    
    /**
     * @brief 从邻接矩阵加载网络
     * @param adjacency_matrix 邻接矩阵（稠密格式）
     * @return 是否成功加载
     */
    bool loadFromAdjacencyMatrix(const std::vector<std::vector<Weight>>& adjacency_matrix);
    
    /**
     * @brief 从稀疏格式加载网络
     * @param connections 连接列表
     * @param neurons 神经元列表
     * @return 是否成功加载
     */
    bool loadFromSparseFormat(const std::vector<Connection>& connections,
                             const std::vector<NeuronProperties>& neurons);
    
    /**
     * @brief 导出为邻接矩阵
     * @return 邻接矩阵
     */
    std::vector<std::vector<Weight>> exportToAdjacencyMatrix() const;
    
    /**
     * @brief 清空网络
     */
    void clear();
    
    /**
     * @brief 验证网络一致性
     * @return 错误信息列表，空列表表示无错误
     */
    std::vector<std::string> validateConsistency() const;
    
    // === 迭代器支持 ===
    
    /**
     * @brief 神经元迭代器
     */
    using neuron_iterator = std::unordered_map<NeuronId, NeuronProperties>::const_iterator;
    neuron_iterator begin_neurons() const { return neurons_.begin(); }
    neuron_iterator end_neurons() const { return neurons_.end(); }
    
    /**
     * @brief 连接迭代器
     */
    using connection_iterator = std::vector<Connection>::const_iterator;
    connection_iterator begin_connections() const { return connections_.begin(); }
    connection_iterator end_connections() const { return connections_.end(); }

protected:
    // 内部数据结构
    std::unordered_map<NeuronId, NeuronProperties> neurons_;
    std::vector<Connection> connections_;
    
    // 快速查找索引
    std::unordered_map<uint64_t, size_t> connection_index_; // 连接快速查找
    std::unordered_map<NeuronId, std::vector<size_t>> incoming_connections_; // 输入连接索引
    std::unordered_map<NeuronId, std::vector<size_t>> outgoing_connections_; // 输出连接索引
    
    // 辅助方法
    uint64_t makeConnectionKey(NeuronId pre, NeuronId post) const {
        return (static_cast<uint64_t>(pre) << 32) | static_cast<uint64_t>(post);
    }
    
    void updateConnectionIndices();
    void addConnectionToIndices(size_t connection_idx);
    void removeConnectionFromIndices(size_t connection_idx);
    
private:
    mutable NetworkStatistics cached_statistics_;
    mutable bool statistics_valid_ = false;
};

} // namespace NeuronMapping

#endif // NEURON_MAPPING_NEURAL_NETWORK_H