#ifndef NEURON_MAPPING_NEURON_MAPPER_H
#define NEURON_MAPPING_NEURON_MAPPER_H

#include "Types.h"
#include "NeuralNetwork.h"
#include "HardwareTopology.h"
#include "MappingSolution.h"
#include "MappingInterfaces.h"
#include <memory>
#include <vector>
#include <functional>

namespace NeuronMapping {

/**
 * @brief 神经元映射器接口
 * 
 * 主要的映射器接口，协调各个组件完成神经元到PE的映射任务。
 * 支持多种映射策略、优化算法和评估方法。
 */
class INeuronMapper {
public:
    virtual ~INeuronMapper() = default;
    
    // === 核心映射方法 ===
    
    /**
     * @brief 执行神经元映射
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param config 映射配置
     * @return 映射解决方案
     */
    virtual std::unique_ptr<MappingSolution> mapNetwork(
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config) = 0;
    
    /**
     * @brief 增量映射（在现有映射基础上添加新神经元）
     * @param current_mapping 当前映射
     * @param new_neurons 新增神经元
     * @param network 完整网络（包含新神经元）
     * @param topology 硬件拓扑
     * @param config 映射配置
     * @return 更新后的映射解决方案
     */
    virtual std::unique_ptr<MappingSolution> incrementalMap(
        const MappingSolution& current_mapping,
        const std::vector<NeuronId>& new_neurons,
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config) = 0;
    
    /**
     * @brief 重映射（优化现有映射）
     * @param current_mapping 当前映射
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param config 映射配置
     * @return 优化后的映射解决方案
     */
    virtual std::unique_ptr<MappingSolution> remapNetwork(
        const MappingSolution& current_mapping,
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config) = 0;
    
    // === 组件管理 ===
    
    /**
     * @brief 设置映射策略
     * @param strategy 映射策略
     */
    virtual void setMappingStrategy(std::unique_ptr<IMappingStrategy> strategy) = 0;
    
    /**
     * @brief 设置网络分析器
     * @param analyzer 网络分析器
     */
    virtual void setNetworkAnalyzer(std::unique_ptr<INetworkAnalyzer> analyzer) = 0;
    
    /**
     * @brief 设置优化器
     * @param optimizer 优化器
     */
    virtual void setOptimizer(std::unique_ptr<IOptimizer> optimizer) = 0;
    
    /**
     * @brief 设置评估器
     * @param evaluator 评估器
     */
    virtual void setEvaluator(std::unique_ptr<IEvaluator> evaluator) = 0;
    
    // === 性能分析 ===
    
    /**
     * @brief 评估映射质量
     * @param mapping 映射解决方案
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param config 映射配置
     * @return 性能指标
     */
    virtual PerformanceMetrics evaluateMapping(
        const MappingSolution& mapping,
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config) = 0;
    
    /**
     * @brief 比较多个映射解决方案
     * @param mappings 映射解决方案列表
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param config 映射配置
     * @return 排序后的映射索引（按性能从好到差）
     */
    virtual std::vector<size_t> compareMappings(
        const std::vector<std::unique_ptr<MappingSolution>>& mappings,
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config) = 0;
    
    // === 信息查询 ===
    
    /**
     * @brief 获取支持的映射策略列表
     * @return 策略名称列表
     */
    virtual std::vector<std::string> getSupportedStrategies() const = 0;
    
    /**
     * @brief 获取支持的优化算法列表
     * @return 优化算法名称列表
     */
    virtual std::vector<std::string> getSupportedOptimizers() const = 0;
    
    /**
     * @brief 获取映射器版本信息
     * @return 版本字符串
     */
    virtual std::string getVersion() const = 0;
    
    /**
     * @brief 获取映射器类型
     * @return 类型字符串
     */
    virtual std::string getMapperType() const = 0;
};

/**
 * @brief 默认神经元映射器实现
 * 
 * 提供完整的映射器功能实现，整合各个组件完成映射任务。
 */
class NeuronMapper : public INeuronMapper {
public:
    NeuronMapper();
    virtual ~NeuronMapper();
    
    // 实现INeuronMapper接口
    std::unique_ptr<MappingSolution> mapNetwork(
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config) override;
    
    std::unique_ptr<MappingSolution> incrementalMap(
        const MappingSolution& current_mapping,
        const std::vector<NeuronId>& new_neurons,
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config) override;
    
    std::unique_ptr<MappingSolution> remapNetwork(
        const MappingSolution& current_mapping,
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config) override;
    
    void setMappingStrategy(std::unique_ptr<IMappingStrategy> strategy) override;
    void setNetworkAnalyzer(std::unique_ptr<INetworkAnalyzer> analyzer) override;
    void setOptimizer(std::unique_ptr<IOptimizer> optimizer) override;
    void setEvaluator(std::unique_ptr<IEvaluator> evaluator) override;
    
    PerformanceMetrics evaluateMapping(
        const MappingSolution& mapping,
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config) override;
    
    std::vector<size_t> compareMappings(
        const std::vector<std::unique_ptr<MappingSolution>>& mappings,
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config) override;
    
    std::vector<std::string> getSupportedStrategies() const override;
    std::vector<std::string> getSupportedOptimizers() const override;
    std::string getVersion() const override { return "1.0.0"; }
    std::string getMapperType() const override { return "DefaultNeuronMapper"; }
    
    // === 扩展功能 ===
    
    /**
     * @brief 批量映射多个网络
     * @param networks 神经网络列表
     * @param topology 硬件拓扑
     * @param config 映射配置
     * @return 映射解决方案列表
     */
    std::vector<std::unique_ptr<MappingSolution>> mapMultipleNetworks(
        const std::vector<std::reference_wrapper<const NeuralNetwork>>& networks,
        const HardwareTopology& topology,
        const MappingConfig& config);
    
    /**
     * @brief 自适应映射（根据网络特征自动选择策略）
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param config 映射配置
     * @return 映射解决方案
     */
    std::unique_ptr<MappingSolution> adaptiveMap(
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config);
    
    /**
     * @brief 多目标优化映射
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @param config 映射配置
     * @param objectives 目标函数权重
     * @return Pareto前沿解决方案集合
     */
    std::vector<std::unique_ptr<MappingSolution>> multiObjectiveMap(
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config,
        const std::vector<float>& objectives);
    
    // === 回调和监听 ===
    
    /**
     * @brief 映射进度回调函数类型
     * @param current_iteration 当前迭代
     * @param max_iterations 最大迭代数
     * @param current_cost 当前成本
     * @param best_cost 最佳成本
     */
    using ProgressCallback = std::function<void(uint32_t, uint32_t, float, float)>;
    
    /**
     * @brief 设置进度回调
     * @param callback 回调函数
     */
    void setProgressCallback(ProgressCallback callback) { progress_callback_ = callback; }
    
    /**
     * @brief 映射事件回调函数类型
     * @param event_type 事件类型
     * @param message 事件消息
     */
    using EventCallback = std::function<void(const std::string&, const std::string&)>;
    
    /**
     * @brief 设置事件回调
     * @param callback 回调函数
     */
    void setEventCallback(EventCallback callback) { event_callback_ = callback; }
    
    // === 调试和诊断 ===
    
    /**
     * @brief 启用详细日志
     * @param enable 是否启用
     */
    void enableVerboseLogging(bool enable) { verbose_logging_ = enable; }
    
    /**
     * @brief 获取最后的映射统计信息
     * @return 统计信息
     */
    std::string getLastMappingStatistics() const { return last_mapping_stats_; }
    
    /**
     * @brief 生成映射报告
     * @param mapping 映射解决方案
     * @param network 神经网络
     * @param topology 硬件拓扑
     * @return 详细报告字符串
     */
    std::string generateMappingReport(
        const MappingSolution& mapping,
        const NeuralNetwork& network,
        const HardwareTopology& topology) const;
    
    /**
     * @brief 验证映射器配置
     * @param config 映射配置
     * @return 配置错误列表，空列表表示配置正确
     */
    std::vector<std::string> validateConfiguration(const MappingConfig& config) const;

private:
    // 内部组件
    std::unique_ptr<IMappingStrategy> strategy_;
    std::unique_ptr<INetworkAnalyzer> analyzer_;
    std::unique_ptr<IOptimizer> optimizer_;
    std::unique_ptr<IEvaluator> evaluator_;
    
    // 回调函数
    ProgressCallback progress_callback_;
    EventCallback event_callback_;
    
    // 配置和状态
    bool verbose_logging_ = false;
    std::string last_mapping_stats_;
    
    // 内部辅助方法
    std::unique_ptr<MappingSolution> executeMapping(
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config,
        const MappingSolution* initial_mapping = nullptr);
    
    std::string selectBestStrategy(
        const NeuralNetwork& network,
        const HardwareTopology& topology) const;
    
    void initializeDefaultComponents();
    void logEvent(const std::string& event_type, const std::string& message) const;
    void updateProgress(uint32_t current, uint32_t max, float current_cost, float best_cost) const;
};

/**
 * @brief 映射器工厂类
 * 
 * 用于创建不同类型的映射器实例。
 */
class MapperFactory {
public:
    /**
     * @brief 创建默认映射器
     * @return 映射器实例
     */
    static std::unique_ptr<INeuronMapper> createDefaultMapper();
    
    /**
     * @brief 创建快速映射器（适用于快速原型）
     * @return 映射器实例
     */
    static std::unique_ptr<INeuronMapper> createFastMapper();
    
    /**
     * @brief 创建高精度映射器（适用于性能优化）
     * @return 映射器实例
     */
    static std::unique_ptr<INeuronMapper> createPrecisionMapper();
    
    /**
     * @brief 创建自定义映射器
     * @param strategy_name 策略名称
     * @param optimizer_name 优化器名称
     * @return 映射器实例
     */
    static std::unique_ptr<INeuronMapper> createCustomMapper(
        const std::string& strategy_name,
        const std::string& optimizer_name);
    
    /**
     * @brief 获取可用的映射器类型
     * @return 映射器类型列表
     */
    static std::vector<std::string> getAvailableMapperTypes();
    
private:
    MapperFactory() = default;
};

} // namespace NeuronMapping

#endif // NEURON_MAPPING_NEURON_MAPPER_H