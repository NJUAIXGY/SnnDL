#ifndef NEURON_MAPPING_TYPES_H
#define NEURON_MAPPING_TYPES_H

#include <cstdint>
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <unordered_map>
#include <chrono>

namespace NeuronMapping {

// 基本类型定义
using NeuronId = uint32_t;
using PEId = uint32_t;
using Weight = float;
using Distance = float;
using Load = float;

// 无效ID常量
constexpr NeuronId INVALID_NEURON_ID = UINT32_MAX;
constexpr PEId INVALID_PE_ID = UINT32_MAX;

// 神经元属性结构
struct NeuronProperties {
    NeuronId id;
    Load computational_load = 1.0f;      // 计算负载
    uint64_t memory_requirement = 0;      // 内存需求 (bytes)
    std::string neuron_type = "default";  // 神经元类型
    std::map<std::string, float> custom_properties; // 自定义属性
    
    NeuronProperties() = default;
    NeuronProperties(NeuronId neuron_id) : id(neuron_id) {}
};

// 连接信息结构
struct Connection {
    NeuronId source_id;      // 源神经元ID (别名: pre_neuron)
    NeuronId target_id;      // 目标神经元ID (别名: post_neuron)
    Weight weight;
    float spike_frequency = 1.0f;         // 脉冲频率估计
    float delay = 0.0f;                   // 传输延迟
    bool enabled = true;                  // 连接是否启用
    std::string connection_type = "excitatory"; // 连接类型
    
    Connection() = default;
    Connection(NeuronId src, NeuronId tgt, Weight w) 
        : source_id(src), target_id(tgt), weight(w) {}
    
    // 向后兼容的getter方法
    NeuronId& pre_neuron() { return source_id; }
    const NeuronId& pre_neuron() const { return source_id; }
    NeuronId& post_neuron() { return target_id; }
    const NeuronId& post_neuron() const { return target_id; }
    
    // 便利方法
    bool isExcitatory() const { return connection_type == "excitatory" && weight > 0.0f; }
    bool isInhibitory() const { return connection_type == "inhibitory" || weight < 0.0f; }
    Weight getEffectiveWeight() const { return enabled ? weight : 0.0f; }
};

// PE（处理单元）信息结构
struct ProcessingElement {
    PEId id;
    uint32_t max_neurons = 16;            // 最大神经元容量
    uint64_t memory_capacity = 64 * 1024 * 1024; // 内存容量 (64MB)
    float computational_capability = 1.0f; // 计算能力
    std::vector<PEId> neighbors;          // 相邻PE列表
    std::string pe_type = "standard";     // PE类型
    std::map<std::string, float> custom_attributes; // 自定义属性
    
    ProcessingElement() = default;
    ProcessingElement(PEId pe_id) : id(pe_id) {}
};

// 网络链路信息
struct NetworkLink {
    PEId pe1, pe2;
    float bandwidth = 1.0f;               // 带宽 (相对值)
    float latency = 1.0f;                 // 延迟 (相对值)
    float congestion_factor = 1.0f;       // 拥塞因子
    
    NetworkLink() = default;
    NetworkLink(PEId p1, PEId p2) : pe1(p1), pe2(p2) {}
};

// 映射分配信息
struct Assignment {
    NeuronId neuron_id;
    PEId pe_id;
    uint32_t core_id = 0;                 // 核心ID（如果PE有多核）
    float assignment_confidence = 1.0f;   // 分配置信度
    std::chrono::steady_clock::time_point timestamp; // 分配时间戳
    
    Assignment() = default;
    Assignment(NeuronId nid, PEId pid) : neuron_id(nid), pe_id(pid) {}
};

// 性能指标结构
struct PerformanceMetrics {
    // 通信指标
    float communication_cost = 0.0f;      // 总通信成本
    float inter_pe_communication_ratio = 0.0f; // 跨PE通信比例
    float average_communication_distance = 0.0f; // 平均通信距离
    
    // 负载均衡指标
    float load_imbalance = 0.0f;          // 负载不均衡（别名: load_imbalance_factor）
    float load_imbalance_factor = 0.0f;   // 负载不均衡因子
    float max_min_load_ratio = 1.0f;      // 最大最小负载比
    float load_variance = 0.0f;           // 负载方差
    
    // 内存使用指标
    float memory_utilization = 0.0f;      // 平均内存利用率
    float memory_imbalance = 0.0f;        // 内存不均衡
    float max_memory_usage = 0.0f;        // 最大内存使用率
    float average_memory_usage = 0.0f;    // 平均内存使用
    float peak_memory_usage = 0.0f;       // 峰值内存使用
    
    // 综合指标
    float overall_score = 0.0f;           // 综合评分
    float objective_value = 0.0f;         // 目标函数值
    float pe_utilization = 0.0f;          // PE利用率
    float neuron_coverage = 0.0f;         // 神经元覆盖率
    
    // 重置所有指标
    void reset() {
        communication_cost = 0.0f;
        inter_pe_communication_ratio = 0.0f;
        average_communication_distance = 0.0f;
        load_imbalance = 0.0f;
        load_imbalance_factor = 0.0f;
        max_min_load_ratio = 1.0f;
        load_variance = 0.0f;
        memory_utilization = 0.0f;
        memory_imbalance = 0.0f;
        max_memory_usage = 0.0f;
        average_memory_usage = 0.0f;
        peak_memory_usage = 0.0f;
        overall_score = 0.0f;
        objective_value = 0.0f;
        pe_utilization = 0.0f;
        neuron_coverage = 0.0f;
    }
};

// 映射配置结构
struct MappingConfig {
    // 算法选择
    std::string strategy = "hybrid";      // "layered", "partition", "hybrid", "adaptive"
    std::string optimizer = "simulated_annealing"; // 优化算法
    
    // 目标函数权重
    float communication_weight = 0.6f;
    float load_balance_weight = 0.3f;
    float memory_weight = 0.1f;
    float inter_pe_comm_weight = 0.1f;    // 跨PE通信权重
    
    // 约束参数
    float max_load_imbalance = 1.5f;      // 最大负载不均衡
    float max_memory_usage = 0.9f;        // 最大内存使用率
    
    // 优化参数
    uint32_t max_iterations = 10000;
    float convergence_threshold = 0.001f;
    uint32_t random_seed = 12345;
    
    // 性能参数
    bool enable_parallel_processing = true;
    uint32_t num_threads = 4;
    bool enable_verbose_logging = false;
    
    MappingConfig() = default;
};

// 网络统计信息
struct NetworkStatistics {
    uint32_t total_neurons = 0;
    uint32_t total_connections = 0;
    uint32_t total_pes = 0;
    
    float average_in_degree = 0.0f;
    float average_out_degree = 0.0f;
    float connection_density = 0.0f;
    float average_weight = 0.0f;
    float weight_variance = 0.0f;
    
    uint32_t max_in_degree = 0;
    uint32_t max_out_degree = 0;
    float clustering_coefficient = 0.0f;
    float small_world_index = 0.0f;
    
    void reset() {
        total_neurons = 0;
        total_connections = 0;
        total_pes = 0;
        average_in_degree = 0.0f;
        average_out_degree = 0.0f;
        connection_density = 0.0f;
        average_weight = 0.0f;
        weight_variance = 0.0f;
        max_in_degree = 0;
        max_out_degree = 0;
        clustering_coefficient = 0.0f;
        small_world_index = 0.0f;
    }
};

// 错误代码枚举
enum class MappingResult {
    SUCCESS = 0,
    INVALID_INPUT,
    INSUFFICIENT_CAPACITY,
    ALGORITHM_FAILURE,
    CONVERGENCE_FAILURE,
    MEMORY_ERROR,
    CONFIGURATION_ERROR
};

// 日志级别
enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3,
    CRITICAL = 4
};

} // namespace NeuronMapping

#endif // NEURON_MAPPING_TYPES_H