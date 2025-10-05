#ifndef NEURON_MAPPING_ROUTING_TABLE_H
#define NEURON_MAPPING_ROUTING_TABLE_H

#include "AddressEvent.h"
#include "core/Types.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <string>

namespace NeuronMapping {

/**
 * @brief 路由方向枚举
 */
enum class RouteDirection : uint8_t {
    LOCAL = 0,      // 本地PE
    NORTH = 1,      // 北方
    SOUTH = 2,      // 南方  
    EAST = 3,       // 东方
    WEST = 4,       // 西方
    UP = 5,         // 上方（3D拓扑）
    DOWN = 6,       // 下方（3D拓扑）
    BROADCAST = 7   // 广播
};

/**
 * @brief 路由表项
 * 
 * 基于SpiNNaker的三元组路由表项设计：(key, mask, route)
 * 支持最长前缀匹配和多路径路由。
 */
struct RoutingEntry {
    uint32_t key;                              // 路由键
    uint32_t mask;                             // 路由掩码
    std::vector<RouteDirection> routes;        // 输出路由方向列表
    uint16_t priority = 0;                     // 路由优先级
    uint32_t packet_count = 0;                 // 数据包计数
    float bandwidth_weight = 1.0f;             // 带宽权重
    bool enabled = true;                       // 是否启用
    
    RoutingEntry() = default;
    RoutingEntry(uint32_t k, uint32_t m, const std::vector<RouteDirection>& r)
        : key(k), mask(m), routes(r) {}
    
    // 匹配检查
    bool matches(uint32_t packet_key) const {
        return enabled && ((packet_key & mask) == (key & mask));
    }
    
    // 获取掩码位数
    uint8_t getMaskBits() const;
    
    // 字符串表示
    std::string toString() const;
};

/**
 * @brief 多播组表项
 */
struct MulticastGroup {
    uint16_t group_id;                         // 组ID
    std::vector<AddressEvent::GlobalNeuronId> members; // 成员神经元列表
    std::vector<PEId> target_pes;              // 目标PE列表
    std::vector<RouteDirection> routes;        // 路由方向
    uint32_t packet_count = 0;                 // 组播包计数
    float traffic_rate = 0.0f;                 // 流量速率
    bool active = true;                        // 是否活跃
    
    MulticastGroup() = default;
    MulticastGroup(uint16_t id, const std::vector<AddressEvent::GlobalNeuronId>& neurons)
        : group_id(id), members(neurons) {}
        
    bool containsNeuron(AddressEvent::GlobalNeuronId neuron_id) const;
    void addMember(AddressEvent::GlobalNeuronId neuron_id);
    void removeMember(AddressEvent::GlobalNeuronId neuron_id);
    std::string toString() const;
};

/**
 * @brief 路由表类
 * 
 * 实现SpiNNaker风格的路由表，支持快速硬件查找、
 * 路由压缩和多播路由功能。
 */
class RoutingTable {
public:
    RoutingTable() = default;
    explicit RoutingTable(PEId pe_id);
    virtual ~RoutingTable() = default;
    
    // === 基本路由表操作 ===
    
    /**
     * @brief 添加路由表项
     * @param entry 路由表项
     * @return 是否成功添加
     */
    bool addEntry(const RoutingEntry& entry);
    
    /**
     * @brief 添加单播路由
     * @param key 路由键
     * @param mask 路由掩码
     * @param direction 路由方向
     * @param priority 优先级
     * @return 是否成功添加
     */
    bool addUnicastRoute(uint32_t key, uint32_t mask, RouteDirection direction, uint16_t priority = 0);
    
    /**
     * @brief 添加多播路由
     * @param key 路由键
     * @param mask 路由掩码
     * @param directions 路由方向列表
     * @param priority 优先级
     * @return 是否成功添加
     */
    bool addMulticastRoute(uint32_t key, uint32_t mask, 
                          const std::vector<RouteDirection>& directions, uint16_t priority = 0);
    
    /**
     * @brief 移除路由表项
     * @param key 路由键
     * @param mask 路由掩码
     * @return 是否成功移除
     */
    bool removeEntry(uint32_t key, uint32_t mask);
    
    /**
     * @brief 清空路由表
     */
    void clear();
    
    // === 路由查找 ===
    
    /**
     * @brief 查找路由（SpiNNaker风格CAM查找）
     * @param packet_key 数据包键值
     * @return 匹配的路由方向列表
     */
    std::vector<RouteDirection> lookupRoute(uint32_t packet_key) const;
    
    /**
     * @brief 查找最佳匹配路由表项
     * @param packet_key 数据包键值
     * @return 路由表项指针，无匹配返回nullptr
     */
    const RoutingEntry* findBestMatch(uint32_t packet_key) const;
    
    /**
     * @brief 查找所有匹配的路由表项
     * @param packet_key 数据包键值
     * @return 匹配的路由表项列表
     */
    std::vector<const RoutingEntry*> findAllMatches(uint32_t packet_key) const;
    
    /**
     * @brief 检查路由是否存在
     * @param packet_key 数据包键值
     * @return 是否存在路由
     */
    bool hasRoute(uint32_t packet_key) const;
    
    // === 多播组管理 ===
    
    /**
     * @brief 添加多播组
     * @param group 多播组
     * @return 是否成功添加
     */
    bool addMulticastGroup(const MulticastGroup& group);
    
    /**
     * @brief 移除多播组
     * @param group_id 组ID
     * @return 是否成功移除
     */
    bool removeMulticastGroup(uint16_t group_id);
    
    /**
     * @brief 查找多播组
     * @param group_id 组ID
     * @return 多播组指针，不存在返回nullptr
     */
    const MulticastGroup* findMulticastGroup(uint16_t group_id) const;
    
    /**
     * @brief 获取所有多播组
     * @return 多播组列表
     */
    std::vector<MulticastGroup> getAllMulticastGroups() const;
    
    // === 路由表优化 ===
    
    /**
     * @brief 压缩路由表（合并相似表项）
     * @return 压缩后减少的表项数量
     */
    size_t compressTable();
    
    /**
     * @brief 优化路由表项顺序（按匹配频率排序）
     */
    void optimizeEntryOrder();
    
    /**
     * @brief 验证路由表一致性
     * @return 错误列表，空列表表示无错误
     */
    std::vector<std::string> validateTable() const;
    
    /**
     * @brief 移除未使用的路由表项
     * @param min_packet_count 最小数据包计数阈值
     * @return 移除的表项数量
     */
    size_t removeUnusedEntries(uint32_t min_packet_count = 0);
    
    // === 统计和监控 ===
    
    /**
     * @brief 获取路由表大小
     * @return 表项数量
     */
    size_t getTableSize() const { return entries_.size(); }
    
    /**
     * @brief 获取多播组数量
     * @return 多播组数量
     */
    size_t getMulticastGroupCount() const { return multicast_groups_.size(); }
    
    /**
     * @brief 获取路由表利用率
     * @return 利用率统计
     */
    std::unordered_map<std::string, float> getUtilizationStats() const;
    
    /**
     * @brief 获取路由统计信息
     * @return 统计信息字符串
     */
    std::string getStatistics() const;
    
    /**
     * @brief 重置统计计数器
     */
    void resetStatistics();
    
    // === 序列化和导入导出 ===
    
    /**
     * @brief 导出路由表
     * @param format 格式（"json", "csv", "binary"）
     * @return 导出数据
     */
    std::string exportTable(const std::string& format = "json") const;
    
    /**
     * @brief 导入路由表
     * @param data 路由表数据
     * @param format 格式
     * @return 是否成功导入
     */
    bool importTable(const std::string& data, const std::string& format = "json");
    
    /**
     * @brief 序列化为二进制格式
     * @return 二进制数据
     */
    std::vector<uint8_t> serializeBinary() const;
    
    /**
     * @brief 从二进制格式反序列化
     * @param data 二进制数据
     * @return 是否成功
     */
    bool deserializeBinary(const std::vector<uint8_t>& data);
    
    // === 调试和诊断 ===
    
    /**
     * @brief 打印路由表（调试用）
     * @param max_entries 最大打印条目数
     */
    void printTable(size_t max_entries = 20) const;
    
    /**
     * @brief 获取路由表详细信息
     * @return 详细信息字符串
     */
    std::string getDetailedInfo() const;
    
    /**
     * @brief 验证特定路由
     * @param packet_key 数据包键值
     * @return 路由验证结果
     */
    std::string validateRoute(uint32_t packet_key) const;

private:
    PEId pe_id_ = INVALID_PE_ID;               // 所属PE的ID
    std::vector<RoutingEntry> entries_;        // 路由表项列表
    std::unordered_map<uint16_t, MulticastGroup> multicast_groups_; // 多播组映射
    
    // 查找索引（用于优化）
    mutable std::unordered_map<uint32_t, std::vector<size_t>> key_index_;
    mutable bool index_valid_ = false;
    
    // 统计信息
    mutable uint64_t total_lookups_ = 0;
    mutable uint64_t successful_lookups_ = 0;
    mutable uint64_t cache_hits_ = 0;
    
    // 缓存最近查找结果
    struct CacheEntry {
        uint32_t key;
        std::vector<RouteDirection> routes;
        uint64_t timestamp;
    };
    mutable std::vector<CacheEntry> lookup_cache_;
    static constexpr size_t CACHE_SIZE = 64;
    
    // 内部辅助方法
    void buildIndex() const;
    void invalidateIndex() { index_valid_ = false; }
    void updateLookupStats(bool success, bool cache_hit = false) const;
    void updateCacheEntry(uint32_t key, const std::vector<RouteDirection>& routes) const;
    const CacheEntry* findCacheEntry(uint32_t key) const;
    
    // 路由表项排序比较函数
    struct EntryComparator {
        bool operator()(const RoutingEntry& a, const RoutingEntry& b) const;
    };
    
    // 表项合并辅助方法
    bool canMergeEntries(const RoutingEntry& a, const RoutingEntry& b) const;
    RoutingEntry mergeEntries(const RoutingEntry& a, const RoutingEntry& b) const;
};

/**
 * @brief 分布式路由表管理器
 * 
 * 管理整个系统中所有PE的路由表，提供全局路由视图和优化功能。
 */
class DistributedRoutingTableManager {
public:
    DistributedRoutingTableManager() = default;
    virtual ~DistributedRoutingTableManager() = default;
    
    // === PE路由表管理 ===
    
    /**
     * @brief 添加PE路由表
     * @param pe_id PE标识
     * @param table 路由表
     * @return 是否成功添加
     */
    bool addPERoutingTable(PEId pe_id, std::unique_ptr<RoutingTable> table);
    
    /**
     * @brief 获取PE路由表
     * @param pe_id PE标识
     * @return 路由表指针，不存在返回nullptr
     */
    RoutingTable* getPERoutingTable(PEId pe_id);
    const RoutingTable* getPERoutingTable(PEId pe_id) const;
    
    /**
     * @brief 移除PE路由表
     * @param pe_id PE标识
     * @return 是否成功移除
     */
    bool removePERoutingTable(PEId pe_id);
    
    /**
     * @brief 获取所有PE的路由表
     * @return PE ID到路由表的映射
     */
    const std::unordered_map<PEId, std::unique_ptr<RoutingTable>>& getAllRoutingTables() const {
        return pe_routing_tables_;
    }
    
    // === 全局路由操作 ===
    
    /**
     * @brief 全局路由查找
     * @param source_pe 源PE
     * @param packet_key 数据包键值
     * @return 下一跳PE列表
     */
    std::vector<PEId> globalRouteLookup(PEId source_pe, uint32_t packet_key) const;
    
    /**
     * @brief 计算端到端路由路径
     * @param source_pe 源PE
     * @param target_pe 目标PE
     * @return 路由路径（PE序列）
     */
    std::vector<PEId> calculateRoutePath(PEId source_pe, PEId target_pe) const;
    
    /**
     * @brief 验证全局路由一致性
     * @return 不一致问题列表
     */
    std::vector<std::string> validateGlobalConsistency() const;
    
    // === 路由表同步和分发 ===
    
    /**
     * @brief 同步所有路由表
     * @return 是否成功同步
     */
    bool synchronizeAllTables();
    
    /**
     * @brief 分发路由表更新
     * @param pe_id PE标识
     * @param updates 更新的表项
     * @return 分发的PE数量
     */
    size_t distributeTableUpdates(PEId pe_id, const std::vector<RoutingEntry>& updates);
    
    /**
     * @brief 广播路由表变更
     * @param change_type 变更类型
     * @param affected_keys 影响的键值列表
     */
    void broadcastTableChanges(const std::string& change_type, 
                              const std::vector<uint32_t>& affected_keys);
    
    // === 性能监控和优化 ===
    
    /**
     * @brief 收集全局路由统计
     * @return 统计信息映射
     */
    std::unordered_map<std::string, uint64_t> collectGlobalStatistics() const;
    
    /**
     * @brief 分析路由瓶颈
     * @return 瓶颈PE列表和严重程度
     */
    std::vector<std::pair<PEId, float>> analyzeRoutingBottlenecks() const;
    
    /**
     * @brief 优化全局路由表
     * @param optimization_level 优化级别 (1-3)
     * @return 优化统计信息
     */
    std::unordered_map<std::string, float> optimizeGlobalRouting(uint8_t optimization_level = 2);
    
    /**
     * @brief 负载均衡路由重分配
     * @param threshold 负载阈值
     * @return 重分配的路由数量
     */
    size_t rebalanceRouting(float threshold = 0.8f);
    
    // === 容错和恢复 ===
    
    /**
     * @brief 检测故障PE并重路由
     * @param failed_pe_id 故障PE
     * @return 影响的路由数量
     */
    size_t handlePEFailure(PEId failed_pe_id);
    
    /**
     * @brief 重建受损路由表
     * @param pe_id PE标识
     * @param backup_source 备份源类型
     * @return 是否成功重建
     */
    bool rebuildRoutingTable(PEId pe_id, const std::string& backup_source = "neighbors");
    
    // === 工具方法 ===
    
    /**
     * @brief 获取总表项数量
     * @return 所有PE路由表项的总数
     */
    size_t getTotalEntryCount() const;
    
    /**
     * @brief 获取总多播组数量
     * @return 所有PE多播组的总数
     */
    size_t getTotalMulticastGroupCount() const;
    
    /**
     * @brief 获取平均表项数量
     * @return 每个PE的平均路由表项数
     */
    float getAverageEntryCountPerPE() const;
    
    /**
     * @brief 生成全局路由报告
     * @return 详细报告字符串
     */
    std::string generateGlobalReport() const;

private:
    std::unordered_map<PEId, std::unique_ptr<RoutingTable>> pe_routing_tables_;
    
    // 全局路由缓存
    mutable std::unordered_map<uint64_t, std::vector<PEId>> global_route_cache_;
    mutable bool global_cache_valid_ = false;
    
    // 故障检测和恢复
    std::unordered_set<PEId> failed_pes_;
    std::unordered_map<PEId, std::chrono::steady_clock::time_point> failure_timestamps_;
    
    // 辅助方法
    uint64_t makeRouteKey(PEId source_pe, uint32_t packet_key) const {
        return (static_cast<uint64_t>(source_pe) << 32) | packet_key;
    }
    
    void invalidateGlobalCache() { global_cache_valid_ = false; }
    std::vector<PEId> findAlternativePaths(PEId source_pe, PEId failed_pe, PEId target_pe) const;
    void redistributeFailedRoutes(PEId failed_pe);
};

} // namespace NeuronMapping

#endif // NEURON_MAPPING_ROUTING_TABLE_H