#ifndef NEURON_MAPPING_SPIKE_ROUTER_H
#define NEURON_MAPPING_SPIKE_ROUTER_H

#include "AddressEvent.h"
#include "RoutingTable.h"
#include "core/Types.h"
#include "core/HardwareTopology.h"
#include <vector>
#include <unordered_map>
#include <queue>
#include <memory>
#include <functional>
#include <chrono>
#include <atomic>
#include <mutex>

namespace NeuronMapping {

/**
 * @brief 路由器端口定义
 */
struct RouterPort {
    RouteDirection direction;           // 端口方向
    PEId connected_pe = INVALID_PE_ID; // 连接的PE（如果是本地端口）
    uint32_t port_id;                  // 端口ID
    float bandwidth = 1.0f;            // 带宽权重
    uint32_t buffer_size = 1024;       // 缓冲区大小
    uint32_t packet_count = 0;         // 数据包计数
    bool enabled = true;               // 是否启用
    
    RouterPort() = default;
    RouterPort(RouteDirection dir, uint32_t id) : direction(dir), port_id(id) {}
};

/**
 * @brief 路由器状态统计
 */
struct RouterStatistics {
    uint64_t total_packets_received = 0;     // 接收的总数据包数
    uint64_t total_packets_forwarded = 0;    // 转发的总数据包数
    uint64_t total_packets_dropped = 0;      // 丢弃的总数据包数
    uint64_t multicast_packets = 0;          // 多播数据包数
    uint64_t unicast_packets = 0;            // 单播数据包数
    uint64_t control_packets = 0;            // 控制数据包数
    uint64_t routing_table_hits = 0;         // 路由表命中次数
    uint64_t routing_table_misses = 0;       // 路由表未命中次数
    uint64_t buffer_overflows = 0;           // 缓冲区溢出次数
    float average_latency = 0.0f;            // 平均延迟（微秒）
    float throughput = 0.0f;                 // 吞吐量（包/秒）
    
    void reset() {
        total_packets_received = 0;
        total_packets_forwarded = 0;
        total_packets_dropped = 0;
        multicast_packets = 0;
        unicast_packets = 0;
        control_packets = 0;
        routing_table_hits = 0;
        routing_table_misses = 0;
        buffer_overflows = 0;
        average_latency = 0.0f;
        throughput = 0.0f;
    }
    
    std::string toString() const;
};

/**
 * @brief 脉冲路由器
 * 
 * 基于SpiNNaker架构设计的硬件路由器，支持：
 * - CAM（内容寻址存储器）风格的路由查找
 * - 多播和单播路由
 * - 硬件加速的数据包转发
 * - 流量控制和拥塞管理
 */
class SpikeRouter {
public:
    // 路由器配置结构
    struct RouterConfig {
        PEId router_id = INVALID_PE_ID;        // 路由器ID
        size_t max_routing_entries = 1024;     // 最大路由表项数
        size_t input_buffer_size = 256;        // 输入缓冲区大小
        size_t output_buffer_size = 256;       // 输出缓冲区大小
        uint32_t max_packet_rate = 10000;      // 最大数据包速率（包/秒）
        uint32_t routing_latency_ns = 100;     // 路由延迟（纳秒）
        bool enable_multicast = true;          // 启用多播
        bool enable_flow_control = true;       // 启用流量控制
        bool enable_statistics = true;         // 启用统计
        uint8_t priority_levels = 4;           // 优先级级数
        
        RouterConfig() = default;
    };
    
    // 数据包处理回调函数类型
    using PacketHandler = std::function<void(const AddressEvent&, RouteDirection)>;
    using ErrorHandler = std::function<void(const std::string&, const AddressEvent&)>;

public:
    /**
     * @brief 构造函数
     * @param config 路由器配置
     */
    explicit SpikeRouter(const RouterConfig& config);
    
    /**
     * @brief 析构函数
     */
    virtual ~SpikeRouter();
    
    // === 初始化和配置 ===
    
    /**
     * @brief 初始化路由器
     * @param topology 硬件拓扑
     * @return 是否成功初始化
     */
    bool initialize(const HardwareTopology& topology);
    
    /**
     * @brief 设置路由表
     * @param table 路由表
     */
    void setRoutingTable(std::unique_ptr<RoutingTable> table);
    
    /**
     * @brief 获取路由表
     * @return 路由表指针
     */
    RoutingTable* getRoutingTable() { return routing_table_.get(); }
    const RoutingTable* getRoutingTable() const { return routing_table_.get(); }
    
    /**
     * @brief 配置端口
     * @param port 端口配置
     * @return 是否成功配置
     */
    bool configurePort(const RouterPort& port);
    
    /**
     * @brief 获取端口配置
     * @param direction 端口方向
     * @return 端口配置指针，不存在返回nullptr
     */
    const RouterPort* getPort(RouteDirection direction) const;
    
    // === 数据包路由 ===
    
    /**
     * @brief 路由单个数据包（主要接口）
     * @param packet 输入数据包
     * @param input_port 输入端口方向
     * @return 是否成功路由
     */
    bool routePacket(const AddressEvent& packet, RouteDirection input_port = RouteDirection::LOCAL);
    
    /**
     * @brief 批量路由数据包
     * @param packets 数据包批次
     * @param input_port 输入端口方向
     * @return 成功路由的数据包数量
     */
    size_t routePacketBatch(const AddressEventBatch& packets, 
                           RouteDirection input_port = RouteDirection::LOCAL);
    
    /**
     * @brief 路由多播数据包
     * @param packet 多播数据包
     * @param input_port 输入端口方向
     * @return 转发的端口数量
     */
    size_t routeMulticastPacket(const AddressEvent& packet, 
                               RouteDirection input_port = RouteDirection::LOCAL);
    
    // === 硬件模拟的CAM查找 ===
    
    /**
     * @brief CAM风格路由查找
     * @param routing_key 路由键
     * @return 匹配的输出端口列表
     */
    std::vector<RouteDirection> camLookup(uint32_t routing_key) const;
    
    /**
     * @brief 并行CAM查找（模拟硬件并行性）
     * @param routing_keys 路由键列表
     * @return 每个键对应的输出端口列表
     */
    std::vector<std::vector<RouteDirection>> parallelCamLookup(
        const std::vector<uint32_t>& routing_keys) const;
    
    /**
     * @brief 优先级CAM查找
     * @param routing_key 路由键
     * @param min_priority 最小优先级
     * @return 满足优先级要求的输出端口列表
     */
    std::vector<RouteDirection> priorityCamLookup(uint32_t routing_key, uint16_t min_priority) const;
    
    // === 流量控制和拥塞管理 ===
    
    /**
     * @brief 检查端口拥塞状态
     * @param direction 端口方向
     * @return 拥塞级别 (0-1, 0表示无拥塞)
     */
    float getPortCongestion(RouteDirection direction) const;
    
    /**
     * @brief 启用/禁用流量控制
     * @param enable 是否启用
     */
    void enableFlowControl(bool enable) { flow_control_enabled_ = enable; }
    
    /**
     * @brief 设置背压阈值
     * @param threshold 背压阈值 (0-1)
     */
    void setBackpressureThreshold(float threshold) { backpressure_threshold_ = threshold; }
    
    /**
     * @brief 应用流量整形
     * @param direction 端口方向
     * @param max_rate 最大速率（包/秒）
     */
    void applyTrafficShaping(RouteDirection direction, uint32_t max_rate);
    
    // === 统计和监控 ===
    
    /**
     * @brief 获取路由器统计信息
     * @return 统计信息
     */
    RouterStatistics getStatistics() const;
    
    /**
     * @brief 重置统计信息
     */
    void resetStatistics();
    
    /**
     * @brief 获取端口统计信息
     * @param direction 端口方向
     * @return 端口统计信息
     */
    std::unordered_map<std::string, uint64_t> getPortStatistics(RouteDirection direction) const;
    
    /**
     * @brief 启用/禁用统计收集
     * @param enable 是否启用
     */
    void enableStatistics(bool enable) { statistics_enabled_ = enable; }
    
    // === 回调和事件处理 ===
    
    /**
     * @brief 设置数据包处理回调
     * @param handler 处理回调函数
     */
    void setPacketHandler(PacketHandler handler) { packet_handler_ = handler; }
    
    /**
     * @brief 设置错误处理回调
     * @param handler 错误处理回调函数
     */
    void setErrorHandler(ErrorHandler handler) { error_handler_ = handler; }
    
    /**
     * @brief 触发数据包处理事件
     * @param packet 数据包
     * @param output_port 输出端口
     */
    void triggerPacketEvent(const AddressEvent& packet, RouteDirection output_port);
    
    /**
     * @brief 触发错误事件
     * @param error_message 错误消息
     * @param packet 相关数据包
     */
    void triggerErrorEvent(const std::string& error_message, const AddressEvent& packet);
    
    // === 动态路由表管理 ===
    
    /**
     * @brief 添加路由表项
     * @param entry 路由表项
     * @return 是否成功添加
     */
    bool addRoutingEntry(const RoutingEntry& entry);
    
    /**
     * @brief 移除路由表项
     * @param key 路由键
     * @param mask 路由掩码
     * @return 是否成功移除
     */
    bool removeRoutingEntry(uint32_t key, uint32_t mask);
    
    /**
     * @brief 更新路由表项
     * @param key 路由键
     * @param mask 路由掩码
     * @param new_routes 新路由方向列表
     * @return 是否成功更新
     */
    bool updateRoutingEntry(uint32_t key, uint32_t mask, 
                           const std::vector<RouteDirection>& new_routes);
    
    /**
     * @brief 批量更新路由表
     * @param entries 路由表项列表
     * @return 成功更新的表项数量
     */
    size_t batchUpdateRoutingTable(const std::vector<RoutingEntry>& entries);
    
    // === 故障检测和恢复 ===
    
    /**
     * @brief 检测端口故障
     * @param direction 端口方向
     * @return 是否检测到故障
     */
    bool detectPortFailure(RouteDirection direction);
    
    /**
     * @brief 标记端口故障
     * @param direction 端口方向
     * @param failed 是否故障
     */
    void markPortFailure(RouteDirection direction, bool failed);
    
    /**
     * @brief 重路由故障端口的流量
     * @param failed_direction 故障端口方向
     * @return 重路由的数据包数量
     */
    size_t rerouteFailedTraffic(RouteDirection failed_direction);
    
    /**
     * @brief 自动故障恢复
     * @param enable 是否启用自动恢复
     */
    void enableAutoFailureRecovery(bool enable) { auto_recovery_enabled_ = enable; }
    
    // === 调试和诊断 ===
    
    /**
     * @brief 获取路由器状态
     * @return 状态字符串
     */
    std::string getStatus() const;
    
    /**
     * @brief 验证路由器配置
     * @return 配置错误列表
     */
    std::vector<std::string> validateConfiguration() const;
    
    /**
     * @brief 生成诊断报告
     * @return 诊断报告字符串
     */
    std::string generateDiagnosticReport() const;
    
    /**
     * @brief 启用调试模式
     * @param enable 是否启用
     * @param detail_level 详细级别 (1-3)
     */
    void enableDebugMode(bool enable, uint8_t detail_level = 1);
    
    // === 性能优化 ===
    
    /**
     * @brief 优化路由表缓存
     */
    void optimizeRoutingCache();
    
    /**
     * @brief 预加载路由表项
     * @param high_frequency_keys 高频访问的键列表
     */
    void preloadRoutingEntries(const std::vector<uint32_t>& high_frequency_keys);
    
    /**
     * @brief 启用/禁用路由预测
     * @param enable 是否启用
     */
    void enableRoutePrediction(bool enable) { route_prediction_enabled_ = enable; }
    
    // === 工具方法 ===
    
    /**
     * @brief 获取路由器ID
     * @return 路由器ID
     */
    PEId getRouterId() const { return config_.router_id; }
    
    /**
     * @brief 检查路由器是否活跃
     * @return 是否活跃
     */
    bool isActive() const { return active_; }
    
    /**
     * @brief 启动路由器
     */
    void start() { active_ = true; }
    
    /**
     * @brief 停止路由器
     */
    void stop() { active_ = false; }
    
    /**
     * @brief 重启路由器
     * @return 是否成功重启
     */
    bool restart();

private:
    // 配置和状态
    RouterConfig config_;
    bool active_ = false;
    std::atomic<bool> statistics_enabled_{true};
    std::atomic<bool> flow_control_enabled_{true};
    std::atomic<bool> auto_recovery_enabled_{false};
    std::atomic<bool> route_prediction_enabled_{false};
    std::atomic<float> backpressure_threshold_{0.8f};
    
    // 路由表和端口
    std::unique_ptr<RoutingTable> routing_table_;
    std::unordered_map<RouteDirection, RouterPort> ports_;
    std::unordered_set<RouteDirection> failed_ports_;
    
    // 统计信息
    mutable std::mutex statistics_mutex_;
    RouterStatistics statistics_;
    std::unordered_map<RouteDirection, std::unordered_map<std::string, uint64_t>> port_statistics_;
    
    // 缓冲区管理
    struct PacketBuffer {
        std::queue<AddressEvent> packets;
        size_t max_size;
        mutable std::mutex mutex;
        
        explicit PacketBuffer(size_t size) : max_size(size) {}
        bool push(const AddressEvent& packet);
        bool pop(AddressEvent& packet);
        size_t size() const;
        bool full() const;
    };
    
    std::unordered_map<RouteDirection, std::unique_ptr<PacketBuffer>> input_buffers_;
    std::unordered_map<RouteDirection, std::unique_ptr<PacketBuffer>> output_buffers_;
    
    // 路由缓存
    struct RouteCacheEntry {
        uint32_t key;
        std::vector<RouteDirection> routes;
        std::chrono::steady_clock::time_point timestamp;
        uint32_t hit_count;
    };
    mutable std::unordered_map<uint32_t, RouteCacheEntry> route_cache_;
    mutable std::mutex cache_mutex_;
    static constexpr size_t MAX_CACHE_SIZE = 256;
    
    // 流量整形
    struct TrafficShaper {
        uint32_t max_rate;
        uint32_t tokens;
        std::chrono::steady_clock::time_point last_update;
        
        explicit TrafficShaper(uint32_t rate) : max_rate(rate), tokens(rate) {
            last_update = std::chrono::steady_clock::now();
        }
        
        bool allowPacket();
        void updateTokens();
    };
    std::unordered_map<RouteDirection, std::unique_ptr<TrafficShaper>> traffic_shapers_;
    
    // 回调函数
    PacketHandler packet_handler_;
    ErrorHandler error_handler_;
    
    // 调试和日志
    mutable std::mutex debug_mutex_;
    bool debug_enabled_ = false;
    uint8_t debug_level_ = 1;
    std::vector<std::string> debug_log_;
    static constexpr size_t MAX_DEBUG_LOG_SIZE = 1000;
    
    // 内部辅助方法
    void initializeBuffers();
    void initializePorts(const HardwareTopology& topology);
    bool validatePacket(const AddressEvent& packet) const;
    void updateStatistics(const AddressEvent& packet, bool success, 
                         std::chrono::nanoseconds processing_time);
    void updatePortStatistics(RouteDirection direction, const std::string& metric, uint64_t value);
    bool checkFlowControl(RouteDirection output_port) const;
    void applyBackpressure(RouteDirection port);
    void logDebugMessage(const std::string& message) const;
    
    // 路由缓存管理
    void updateRouteCache(uint32_t key, const std::vector<RouteDirection>& routes) const;
    const std::vector<RouteDirection>* findCachedRoute(uint32_t key) const;
    void cleanupRouteCache() const;
    
    // 故障检测辅助
    bool isPortHealthy(RouteDirection direction) const;
    void performHealthCheck();
    std::vector<RouteDirection> findAlternativeRoutes(RouteDirection failed_port, 
                                                     uint32_t routing_key) const;
};

/**
 * @brief 路由器网络管理器
 * 
 * 管理整个系统中的所有路由器，提供网络级别的路由服务。
 */
class RouterNetworkManager {
public:
    RouterNetworkManager() = default;
    virtual ~RouterNetworkManager() = default;
    
    // === 路由器管理 ===
    
    /**
     * @brief 添加路由器
     * @param router_id 路由器ID
     * @param router 路由器实例
     * @return 是否成功添加
     */
    bool addRouter(PEId router_id, std::unique_ptr<SpikeRouter> router);
    
    /**
     * @brief 获取路由器
     * @param router_id 路由器ID
     * @return 路由器指针
     */
    SpikeRouter* getRouter(PEId router_id);
    const SpikeRouter* getRouter(PEId router_id) const;
    
    /**
     * @brief 移除路由器
     * @param router_id 路由器ID
     * @return 是否成功移除
     */
    bool removeRouter(PEId router_id);
    
    // === 网络级路由 ===
    
    /**
     * @brief 网络级数据包路由
     * @param packet 数据包
     * @param source_router 源路由器ID
     * @return 路由跳数
     */
    size_t routePacketThroughNetwork(const AddressEvent& packet, PEId source_router);
    
    /**
     * @brief 计算网络路径
     * @param source 源路由器
     * @param target 目标路由器
     * @return 路径上的路由器ID列表
     */
    std::vector<PEId> calculateNetworkPath(PEId source, PEId target) const;
    
    /**
     * @brief 广播数据包
     * @param packet 数据包
     * @param source_router 源路由器ID
     * @param exclude_routers 排除的路由器ID列表
     * @return 接收到数据包的路由器数量
     */
    size_t broadcastPacket(const AddressEvent& packet, PEId source_router,
                          const std::vector<PEId>& exclude_routers = {});
    
    // === 网络监控和统计 ===
    
    /**
     * @brief 获取网络统计信息
     * @return 网络级统计信息
     */
    std::unordered_map<std::string, uint64_t> getNetworkStatistics() const;
    
    /**
     * @brief 检测网络拥塞
     * @return 拥塞路由器列表和严重程度
     */
    std::vector<std::pair<PEId, float>> detectNetworkCongestion() const;
    
    /**
     * @brief 分析网络性能
     * @return 性能指标映射
     */
    std::unordered_map<std::string, float> analyzeNetworkPerformance() const;
    
    // === 网络优化 ===
    
    /**
     * @brief 优化网络路由
     * @param optimization_strategy 优化策略
     * @return 优化结果统计
     */
    std::unordered_map<std::string, float> optimizeNetworkRouting(
        const std::string& optimization_strategy = "load_balance");
    
    /**
     * @brief 动态负载均衡
     * @param rebalance_threshold 重平衡阈值
     * @return 重新路由的连接数量
     */
    size_t dynamicLoadBalancing(float rebalance_threshold = 0.7f);
    
    // === 故障恢复 ===
    
    /**
     * @brief 处理路由器故障
     * @param failed_router_id 故障路由器ID
     * @return 影响的连接数量
     */
    size_t handleRouterFailure(PEId failed_router_id);
    
    /**
     * @brief 网络自愈
     * @return 是否成功恢复网络连通性
     */
    bool performNetworkSelfHealing();

private:
    std::unordered_map<PEId, std::unique_ptr<SpikeRouter>> routers_;
    std::unordered_map<PEId, std::vector<PEId>> network_topology_; // 路由器连接关系
    std::unordered_set<PEId> failed_routers_;
    
    // 网络级缓存
    mutable std::unordered_map<uint64_t, std::vector<PEId>> path_cache_;
    mutable bool path_cache_valid_ = false;
    
    // 辅助方法
    void buildNetworkTopology();
    void invalidatePathCache() { path_cache_valid_ = false; }
    std::vector<PEId> findShortestPath(PEId source, PEId target) const;
    void redistributeTraffic(PEId failed_router);
};

} // namespace NeuronMapping

#endif // NEURON_MAPPING_SPIKE_ROUTER_H