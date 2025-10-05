#ifndef NEURON_MAPPING_ADDRESS_EVENT_H
#define NEURON_MAPPING_ADDRESS_EVENT_H

#include "core/Types.h"
#include <cstdint>
#include <vector>
#include <chrono>
#include <algorithm>

namespace NeuronMapping {

/**
 * @brief 地址事件表示（AER）协议的脉冲事件
 * 
 * 基于SpiNNaker架构的设计理念，使用32位全局ID标识神经元，
 * 支持高效的硬件路由和多播传输。
 */
class AddressEvent {
public:
    // 32位神经元全局ID（AER协议核心）
    using GlobalNeuronId = uint32_t;
    using Timestamp = uint64_t;  // 微秒时间戳
    using Priority = uint8_t;
    
    // 事件类型枚举
    enum class EventType : uint8_t {
        SPIKE = 0,          // 标准脉冲事件
        MULTICAST = 1,      // 多播脉冲事件
        CONTROL = 2,        // 控制消息
        HEARTBEAT = 3       // 心跳信号
    };
    
    // 优先级常量
    static constexpr Priority PRIORITY_LOW = 0;
    static constexpr Priority PRIORITY_NORMAL = 1;
    static constexpr Priority PRIORITY_HIGH = 2;
    static constexpr Priority PRIORITY_CRITICAL = 3;
    
    // 无效ID常量
    static constexpr GlobalNeuronId INVALID_GLOBAL_ID = UINT32_MAX;

public:
    /**
     * @brief 默认构造函数
     */
    AddressEvent() = default;
    
    /**
     * @brief 构造脉冲事件
     * @param global_id 全局神经元ID
     * @param timestamp 时间戳（微秒）
     * @param weight 脉冲权重
     * @param priority 优先级
     */
    AddressEvent(GlobalNeuronId global_id, 
                Timestamp timestamp, 
                float weight = 1.0f,
                Priority priority = PRIORITY_NORMAL);
    
    /**
     * @brief 构造带类型的事件
     * @param global_id 全局神经元ID
     * @param timestamp 时间戳
     * @param event_type 事件类型
     * @param weight 权重
     * @param priority 优先级
     */
    AddressEvent(GlobalNeuronId global_id,
                Timestamp timestamp,
                EventType event_type,
                float weight = 1.0f,
                Priority priority = PRIORITY_NORMAL);
    
    // === 访问器方法 ===
    
    GlobalNeuronId getGlobalId() const { return global_id_; }
    void setGlobalId(GlobalNeuronId id) { global_id_ = id; }
    
    Timestamp getTimestamp() const { return timestamp_; }
    void setTimestamp(Timestamp ts) { timestamp_ = ts; }
    
    float getWeight() const { return weight_; }
    void setWeight(float w) { weight_ = w; }
    
    Priority getPriority() const { return priority_; }
    void setPriority(Priority p) { priority_ = p; }
    
    EventType getEventType() const { return event_type_; }
    void setEventType(EventType type) { event_type_ = type; }
    
    uint32_t getSequenceNumber() const { return sequence_number_; }
    void setSequenceNumber(uint32_t seq) { sequence_number_ = seq; }
    
    // === 路由相关方法 ===
    
    /**
     * @brief 获取路由键（SpiNNaker风格）
     * @return 32位路由键，用于硬件路由查找
     */
    uint32_t getRoutingKey() const { return global_id_; }
    
    /**
     * @brief 生成掩码（用于路由表匹配）
     * @param mask_bits 掩码位数
     * @return 路由掩码
     */
    uint32_t generateRoutingMask(uint8_t mask_bits) const;
    
    /**
     * @brief 检查是否匹配路由规则
     * @param key 路由键
     * @param mask 路由掩码
     * @return 是否匹配
     */
    bool matchesRoute(uint32_t key, uint32_t mask) const;
    
    // === 多播支持 ===
    
    /**
     * @brief 检查是否为多播事件
     * @return 是否为多播
     */
    bool isMulticast() const { return event_type_ == EventType::MULTICAST; }
    
    /**
     * @brief 设置多播组ID
     * @param group_id 多播组标识
     */
    void setMulticastGroup(uint16_t group_id);
    
    /**
     * @brief 获取多播组ID
     * @return 多播组ID，如果不是多播事件返回0
     */
    uint16_t getMulticastGroup() const;
    
    // === 批量事件支持 ===
    
    /**
     * @brief 检查是否可以与另一个事件批量处理
     * @param other 另一个事件
     * @param time_window 时间窗口（微秒）
     * @return 是否可以批量处理
     */
    bool canBatchWith(const AddressEvent& other, uint64_t time_window = 1000) const;
    
    /**
     * @brief 合并另一个事件（用于批量处理）
     * @param other 要合并的事件
     * @return 是否成功合并
     */
    bool mergeWith(const AddressEvent& other);
    
    // === 序列化支持 ===
    
    /**
     * @brief 序列化为字节数组
     * @return 序列化数据
     */
    std::vector<uint8_t> serialize() const;
    
    /**
     * @brief 从字节数组反序列化
     * @param data 序列化数据
     * @return 是否成功反序列化
     */
    bool deserialize(const std::vector<uint8_t>& data);
    
    /**
     * @brief 获取序列化大小
     * @return 字节数
     */
    static size_t getSerializedSize() { return SERIALIZED_SIZE; }
    
    // === 比较和排序 ===
    
    bool operator==(const AddressEvent& other) const;
    bool operator!=(const AddressEvent& other) const { return !(*this == other); }
    bool operator<(const AddressEvent& other) const;
    
    // === 工具方法 ===
    
    /**
     * @brief 检查事件有效性
     * @return 是否有效
     */
    bool isValid() const;
    
    /**
     * @brief 获取事件描述字符串
     * @return 描述字符串
     */
    std::string toString() const;
    
    /**
     * @brief 重置事件到默认状态
     */
    void reset();
    
    /**
     * @brief 克隆事件
     * @return 事件副本
     */
    AddressEvent clone() const;
    
    // === 静态工具方法 ===
    
    /**
     * @brief 从本地神经元ID生成全局ID
     * @param local_id 本地神经元ID
     * @param pe_id PE标识
     * @param core_id 核心标识
     * @return 全局神经元ID
     */
    static GlobalNeuronId generateGlobalId(NeuronId local_id, PEId pe_id, uint16_t core_id = 0);
    
    /**
     * @brief 从全局ID提取本地信息
     * @param global_id 全局神经元ID
     * @return {local_id, pe_id, core_id}
     */
    static std::tuple<NeuronId, PEId, uint16_t> extractLocalInfo(GlobalNeuronId global_id);
    
    /**
     * @brief 获取当前时间戳
     * @return 微秒时间戳
     */
    static Timestamp getCurrentTimestamp();
    
    /**
     * @brief 创建控制事件
     * @param command 控制命令
     * @param timestamp 时间戳
     * @return 控制事件
     */
    static AddressEvent createControlEvent(uint32_t command, Timestamp timestamp);
    
    /**
     * @brief 创建心跳事件
     * @param pe_id PE标识
     * @param timestamp 时间戳
     * @return 心跳事件
     */
    static AddressEvent createHeartbeatEvent(PEId pe_id, Timestamp timestamp);

private:
    // 核心数据成员
    GlobalNeuronId global_id_ = INVALID_GLOBAL_ID;  // 32位全局神经元ID
    Timestamp timestamp_ = 0;                       // 时间戳（微秒）
    float weight_ = 1.0f;                          // 脉冲权重
    EventType event_type_ = EventType::SPIKE;       // 事件类型
    Priority priority_ = PRIORITY_NORMAL;           // 优先级
    uint32_t sequence_number_ = 0;                  // 序列号
    
    // 辅助字段
    uint16_t flags_ = 0;                           // 标志位
    uint16_t reserved_ = 0;                        // 保留字段
    
    // 序列化常量
    static constexpr size_t SERIALIZED_SIZE = 24;  // 序列化后的字节数
    
    // 全局ID编码常量
    static constexpr uint8_t LOCAL_ID_BITS = 16;   // 本地ID位数
    static constexpr uint8_t PE_ID_BITS = 12;      // PE ID位数
    static constexpr uint8_t CORE_ID_BITS = 4;     // 核心ID位数
    
    static constexpr uint32_t LOCAL_ID_MASK = (1U << LOCAL_ID_BITS) - 1;
    static constexpr uint32_t PE_ID_MASK = (1U << PE_ID_BITS) - 1;
    static constexpr uint32_t CORE_ID_MASK = (1U << CORE_ID_BITS) - 1;
    
    // 内部辅助方法
    void validateFields() const;
    uint32_t calculateChecksum() const;
};

/**
 * @brief 地址事件批量容器
 * 
 * 用于高效处理批量脉冲事件，支持时间窗口合并和优先级排序。
 */
class AddressEventBatch {
public:
    using EventList = std::vector<AddressEvent>;
    using Iterator = EventList::iterator;
    using ConstIterator = EventList::const_iterator;
    
    AddressEventBatch() = default;
    explicit AddressEventBatch(size_t reserve_size);
    
    // === 事件管理 ===
    
    void addEvent(const AddressEvent& event);
    void addEvent(AddressEvent&& event);
    void addEvents(const std::vector<AddressEvent>& events);
    
    void removeEvent(size_t index);
    void clear();
    
    // === 访问方法 ===
    
    size_t size() const { return events_.size(); }
    bool empty() const { return events_.empty(); }
    
    const AddressEvent& operator[](size_t index) const { return events_[index]; }
    AddressEvent& operator[](size_t index) { return events_[index]; }
    
    const AddressEvent& at(size_t index) const { return events_.at(index); }
    AddressEvent& at(size_t index) { return events_.at(index); }
    
    // === 迭代器支持 ===
    
    Iterator begin() { return events_.begin(); }
    Iterator end() { return events_.end(); }
    ConstIterator begin() const { return events_.begin(); }
    ConstIterator end() const { return events_.end(); }
    ConstIterator cbegin() const { return events_.cbegin(); }
    ConstIterator cend() const { return events_.cend(); }
    
    // === 批量操作 ===
    
    /**
     * @brief 按时间戳排序事件
     */
    void sortByTimestamp();
    
    /**
     * @brief 按优先级排序事件
     */
    void sortByPriority();
    
    /**
     * @brief 合并相似事件
     * @param time_window 时间窗口（微秒）
     * @return 合并的事件数量
     */
    size_t mergeEvents(uint64_t time_window = 1000);
    
    /**
     * @brief 过滤事件
     * @param predicate 过滤条件
     */
    template<typename Predicate>
    void filterEvents(Predicate pred) {
        events_.erase(std::remove_if(events_.begin(), events_.end(), 
                                   [&pred](const AddressEvent& e) { return !pred(e); }),
                     events_.end());
    }
    
    /**
     * @brief 分割批量事件
     * @param max_batch_size 最大批量大小
     * @return 分割后的批量事件列表
     */
    std::vector<AddressEventBatch> split(size_t max_batch_size) const;
    
    /**
     * @brief 序列化批量事件
     * @return 序列化数据
     */
    std::vector<uint8_t> serialize() const;
    
    /**
     * @brief 反序列化批量事件
     * @param data 序列化数据
     * @return 是否成功
     */
    bool deserialize(const std::vector<uint8_t>& data);
    
    /**
     * @brief 获取事件统计信息
     * @return 统计信息字符串
     */
    std::string getStatistics() const;

private:
    EventList events_;
    mutable size_t cached_checksum_ = 0;
    mutable bool checksum_valid_ = false;
    
    void invalidateChecksum() { checksum_valid_ = false; }
    size_t calculateChecksum() const;
};

} // namespace NeuronMapping

#endif // NEURON_MAPPING_ADDRESS_EVENT_H