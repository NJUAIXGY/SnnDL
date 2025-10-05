#ifndef NEURON_MAPPING_HARDWARE_TOPOLOGY_H
#define NEURON_MAPPING_HARDWARE_TOPOLOGY_H

#include "Types.h"
#include <vector>
#include <unordered_map>
#include <memory>
#include <functional>

namespace NeuronMapping {

/**
 * @brief 硬件拓扑结构类
 * 
 * 描述目标硬件平台的PE拓扑、网络连接和性能特征。
 * 支持各种拓扑结构：Mesh、Torus、Tree、自定义拓扑等。
 */
class HardwareTopology {
public:
    HardwareTopology() = default;
    virtual ~HardwareTopology() = default;
    
    // === PE管理 ===
    
    /**
     * @brief 添加处理单元
     * @param pe 处理单元信息
     * @return 是否成功添加
     */
    bool addPE(const ProcessingElement& pe);
    
    /**
     * @brief 批量添加PE
     * @param pes PE列表
     * @return 成功添加的数量
     */
    uint32_t addPEs(const std::vector<ProcessingElement>& pes);
    
    /**
     * @brief 移除PE
     * @param pe_id PE ID
     * @return 是否成功移除
     */
    bool removePE(PEId pe_id);
    
    /**
     * @brief 检查PE是否存在
     * @param pe_id PE ID
     * @return 是否存在
     */
    bool hasPE(PEId pe_id) const;
    
    /**
     * @brief 获取PE信息
     * @param pe_id PE ID
     * @return PE信息指针，不存在返回nullptr
     */
    const ProcessingElement* getPE(PEId pe_id) const;
    
    /**
     * @brief 更新PE信息
     * @param pe 新的PE信息
     * @return 是否成功更新
     */
    bool updatePE(const ProcessingElement& pe);
    
    /**
     * @brief 获取所有PE ID
     * @return PE ID列表
     */
    std::vector<PEId> getAllPEIds() const;
    
    /**
     * @brief 获取PE数量
     * @return PE总数
     */
    uint32_t getPECount() const { return static_cast<uint32_t>(pes_.size()); }
    
    /**
     * @brief 获取总PE数量 (别名)
     * @return PE总数
     */
    uint32_t getTotalPEs() const { return getPECount(); }
    
    /**
     * @brief 获取指定PE的神经元容量
     * @param pe_id PE ID
     * @return 神经元容量
     */
    uint32_t getPECapacity(PEId pe_id) const;
    
    /**
     * @brief 获取指定PE的内存容量
     * @param pe_id PE ID
     * @return 内存容量
     */
    uint64_t getPEMemoryCapacity(PEId pe_id) const;
    
    // === 网络链路管理 ===
    
    /**
     * @brief 添加网络链路
     * @param link 网络链路信息
     * @return 是否成功添加
     */
    bool addLink(const NetworkLink& link);
    
    /**
     * @brief 批量添加链路
     * @param links 链路列表
     * @return 成功添加的数量
     */
    uint32_t addLinks(const std::vector<NetworkLink>& links);
    
    /**
     * @brief 移除链路
     * @param pe1 PE1 ID
     * @param pe2 PE2 ID
     * @return 是否成功移除
     */
    bool removeLink(PEId pe1, PEId pe2);
    
    /**
     * @brief 检查链路是否存在
     * @param pe1 PE1 ID
     * @param pe2 PE2 ID
     * @return 是否存在链路
     */
    bool hasLink(PEId pe1, PEId pe2) const;
    
    /**
     * @brief 获取链路信息
     * @param pe1 PE1 ID
     * @param pe2 PE2 ID
     * @return 链路信息指针，不存在返回nullptr
     */
    const NetworkLink* getLink(PEId pe1, PEId pe2) const;
    
    /**
     * @brief 获取所有链路
     * @return 链路列表
     */
    std::vector<NetworkLink> getAllLinks() const;
    
    /**
     * @brief 获取链路数量
     * @return 链路总数
     */
    uint32_t getLinkCount() const { return static_cast<uint32_t>(links_.size()); }
    
    // === 拓扑查询 ===
    
    /**
     * @brief 获取PE的邻居
     * @param pe_id PE ID
     * @return 邻居PE ID列表
     */
    std::vector<PEId> getNeighbors(PEId pe_id) const;
    
    /**
     * @brief 获取PE的度数（邻居数量）
     * @param pe_id PE ID
     * @return 度数
     */
    uint32_t getDegree(PEId pe_id) const;
    
    /**
     * @brief 计算两个PE之间的距离
     * @param pe1 PE1 ID
     * @param pe2 PE2 ID
     * @return 距离（跳数），-1表示不可达
     */
    int32_t getDistance(PEId pe1, PEId pe2) const;
    
    /**
     * @brief 获取两个PE之间的路径
     * @param pe1 源PE
     * @param pe2 目标PE
     * @return 路径上的PE列表，空表示不可达
     */
    std::vector<PEId> getPath(PEId pe1, PEId pe2) const;
    
    /**
     * @brief 获取距离矩阵
     * @return PE间距离矩阵
     */
    const std::vector<std::vector<int32_t>>& getDistanceMatrix() const;
    
    /**
     * @brief 计算通信成本
     * @param pe1 源PE
     * @param pe2 目标PE
     * @param traffic_volume 通信量
     * @return 通信成本
     */
    float calculateCommunicationCost(PEId pe1, PEId pe2, float traffic_volume) const;
    
    // === 拓扑生成 ===
    
    /**
     * @brief 创建2D Mesh拓扑
     * @param rows 行数
     * @param cols 列数
     * @param pe_config 默认PE配置
     * @return 是否成功创建
     */
    bool createMesh2D(uint32_t rows, uint32_t cols, const ProcessingElement& pe_config = ProcessingElement());
    
    /**
     * @brief 创建3D Mesh拓扑
     * @param x_dim X维度
     * @param y_dim Y维度
     * @param z_dim Z维度
     * @param pe_config 默认PE配置
     * @return 是否成功创建
     */
    bool createMesh3D(uint32_t x_dim, uint32_t y_dim, uint32_t z_dim, 
                      const ProcessingElement& pe_config = ProcessingElement());
    
    /**
     * @brief 创建Torus拓扑
     * @param rows 行数
     * @param cols 列数
     * @param pe_config 默认PE配置
     * @return 是否成功创建
     */
    bool createTorus2D(uint32_t rows, uint32_t cols, const ProcessingElement& pe_config = ProcessingElement());
    
    /**
     * @brief 创建树拓扑
     * @param levels 层数
     * @param branching_factor 分支因子
     * @param pe_config 默认PE配置
     * @return 是否成功创建
     */
    bool createTree(uint32_t levels, uint32_t branching_factor, 
                    const ProcessingElement& pe_config = ProcessingElement());
    
    /**
     * @brief 创建全连接拓扑
     * @param num_pes PE数量
     * @param pe_config 默认PE配置
     * @return 是否成功创建
     */
    bool createFullyConnected(uint32_t num_pes, const ProcessingElement& pe_config = ProcessingElement());
    
    /**
     * @brief 创建环形拓扑
     * @param num_pes PE数量
     * @param pe_config 默认PE配置
     * @return 是否成功创建
     */
    bool createRing(uint32_t num_pes, const ProcessingElement& pe_config = ProcessingElement());
    
    // === 容量管理 ===
    
    /**
     * @brief 获取总神经元容量
     * @return 所有PE的神经元容量总和
     */
    uint32_t getTotalNeuronCapacity() const;
    
    /**
     * @brief 获取总内存容量
     * @return 所有PE的内存容量总和
     */
    uint64_t getTotalMemoryCapacity() const;
    
    /**
     * @brief 获取PE的剩余神经元容量
     * @param pe_id PE ID
     * @param current_assignment 当前分配情况
     * @return 剩余容量
     */
    uint32_t getRemainingNeuronCapacity(PEId pe_id, 
                                       const std::unordered_map<PEId, std::vector<NeuronId>>& current_assignment) const;
    
    /**
     * @brief 获取PE的剩余内存容量
     * @param pe_id PE ID
     * @param current_memory_usage PE当前内存使用量
     * @return 剩余内存容量
     */
    uint64_t getRemainingMemoryCapacity(PEId pe_id, uint64_t current_memory_usage) const;
    
    /**
     * @brief 检查PE是否可以容纳指定数量的神经元
     * @param pe_id PE ID
     * @param neuron_count 神经元数量
     * @param current_assignment 当前分配情况
     * @return 是否可以容纳
     */
    bool canAccommodateNeurons(PEId pe_id, uint32_t neuron_count,
                              const std::unordered_map<PEId, std::vector<NeuronId>>& current_assignment) const;
    
    // === 拓扑分析 ===
    
    /**
     * @brief 获取拓扑类型
     * @return 拓扑类型字符串
     */
    std::string getTopologyType() const { return topology_type_; }
    
    /**
     * @brief 设置拓扑类型
     * @param type 拓扑类型
     */
    void setTopologyType(const std::string& type) { topology_type_ = type; }
    
    /**
     * @brief 计算网络直径
     * @return 网络直径（最大最短路径）
     */
    uint32_t calculateNetworkDiameter() const;
    
    /**
     * @brief 计算平均路径长度
     * @return 平均路径长度
     */
    float calculateAveragePathLength() const;
    
    /**
     * @brief 计算网络带宽
     * @return 总带宽
     */
    float calculateTotalBandwidth() const;
    
    /**
     * @brief 检查网络连通性
     * @return 是否连通
     */
    bool isConnected() const;
    
    /**
     * @brief 找出网络瓶颈链路
     * @return 瓶颈链路列表
     */
    std::vector<NetworkLink> findBottleneckLinks() const;
    
    // === 负载均衡分析 ===
    
    /**
     * @brief 计算PE负载分布
     * @param assignment 当前神经元分配
     * @param neuron_loads 神经元负载
     * @return PE负载映射
     */
    std::unordered_map<PEId, float> calculatePELoads(
        const std::unordered_map<PEId, std::vector<NeuronId>>& assignment,
        const std::unordered_map<NeuronId, float>& neuron_loads) const;
    
    /**
     * @brief 找出负载最高的PE
     * @param pe_loads PE负载映射
     * @return 负载最高的PE ID
     */
    PEId findMostLoadedPE(const std::unordered_map<PEId, float>& pe_loads) const;
    
    /**
     * @brief 找出负载最低的PE
     * @param pe_loads PE负载映射
     * @return 负载最低的PE ID
     */
    PEId findLeastLoadedPE(const std::unordered_map<PEId, float>& pe_loads) const;
    
    // === 序列化 ===
    
    /**
     * @brief 清空拓扑
     */
    void clear();
    
    /**
     * @brief 验证拓扑一致性
     * @return 错误信息列表，空列表表示无错误
     */
    std::vector<std::string> validateTopology() const;
    
    /**
     * @brief 克隆拓扑
     * @return 拓扑的深拷贝
     */
    std::unique_ptr<HardwareTopology> clone() const;

protected:
    // 内部数据结构
    std::unordered_map<PEId, ProcessingElement> pes_;
    std::vector<NetworkLink> links_;
    std::string topology_type_ = "custom";
    
    // 快速查找索引
    std::unordered_map<uint64_t, size_t> link_index_; // 链路快速查找
    std::unordered_map<PEId, std::vector<PEId>> adjacency_list_; // 邻接表
    
    // 缓存的计算结果
    mutable std::vector<std::vector<int32_t>> distance_matrix_;
    mutable bool distance_matrix_valid_ = false;
    
    // 辅助方法
    uint64_t makeLinkKey(PEId pe1, PEId pe2) const {
        if (pe1 > pe2) std::swap(pe1, pe2); // 确保顺序一致
        return (static_cast<uint64_t>(pe1) << 32) | static_cast<uint64_t>(pe2);
    }
    
    void updateLinkIndices();
    void addLinkToIndices(const NetworkLink& link);
    void removeLinkFromIndices(PEId pe1, PEId pe2);
    void buildAdjacencyList();
    void computeDistanceMatrix() const;
    void invalidateCache();

private:
    // Floyd-Warshall算法计算所有对最短路径
    void floydWarshall() const;
};

} // namespace NeuronMapping

#endif // NEURON_MAPPING_HARDWARE_TOPOLOGY_H