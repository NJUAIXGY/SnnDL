#ifndef NEURON_MAPPING_FRAMEWORK_FACTORIES_MAPPER_FACTORY_H
#define NEURON_MAPPING_FRAMEWORK_FACTORIES_MAPPER_FACTORY_H

#include "core/NeuronMapper.h"
#include "strategies/MappingStrategy.h"
#include "optimizers/MappingOptimizer.h"
#include "evaluators/PerformanceEvaluator.h"
#include "evaluators/CommunicationCostEvaluator.h"
#include "evaluators/LoadBalanceEvaluator.h"
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>

namespace neuron_mapping {

using namespace NeuronMapping;

enum class MapperType {
    DEFAULT,
    FAST,
    PRECISION,
    CUSTOM
};

enum class StrategyType {
    RANDOM,
    GREEDY,
    CUSTOM
};

enum class OptimizerType {
    LOCAL_SEARCH,
    HILL_CLIMBING,
    NONE
};

enum class EvaluatorType {
    COMMUNICATION_COST,
    LOAD_BALANCE,
    COMPREHENSIVE
};

struct MapperConfiguration {
    MapperType mapper_type = MapperType::DEFAULT;
    StrategyType strategy_type = StrategyType::RANDOM;
    OptimizerType optimizer_type = OptimizerType::LOCAL_SEARCH;
    EvaluatorType evaluator_type = EvaluatorType::COMPREHENSIVE;
    
    // 策略参数
    uint32_t random_seed = 12345;
    
    // 优化器参数
    uint32_t max_optimization_iterations = 1000;
    double convergence_threshold = 1e-6;
    
    // 评估器参数
    CommunicationWeights comm_weights;
    LoadWeights load_weights;
    
    // 调试参数
    bool enable_verbose_logging = false;
    bool enable_progress_callback = false;
};

class ComponentFactory {
public:
    // 策略工厂方法
    static std::unique_ptr<MappingStrategy> createStrategy(
        StrategyType type, 
        const MapperConfiguration& config = MapperConfiguration());
    
    // 优化器工厂方法
    static std::unique_ptr<MappingOptimizer> createOptimizer(
        OptimizerType type,
        const MapperConfiguration& config = MapperConfiguration());
    
    // 评估器工厂方法
    static std::unique_ptr<PerformanceEvaluator> createEvaluator(
        EvaluatorType type,
        const MapperConfiguration& config = MapperConfiguration());
    
    // 获取可用组件列表
    static std::vector<std::string> getAvailableStrategies();
    static std::vector<std::string> getAvailableOptimizers();
    static std::vector<std::string> getAvailableEvaluators();
    
private:
    ComponentFactory() = default;
};

class NeuronMapperFactory {
public:
    // 主要工厂方法
    static std::unique_ptr<INeuronMapper> createMapper(
        const MapperConfiguration& config = MapperConfiguration());
    
    static std::unique_ptr<INeuronMapper> createMapper(
        MapperType type);
    
    // 便利方法
    static std::unique_ptr<INeuronMapper> createDefaultMapper();
    static std::unique_ptr<INeuronMapper> createFastMapper();
    static std::unique_ptr<INeuronMapper> createPrecisionMapper();
    
    // 自定义映射器创建
    static std::unique_ptr<INeuronMapper> createCustomMapper(
        const std::string& strategy_name,
        const std::string& optimizer_name = "LocalSearch",
        const std::string& evaluator_name = "Comprehensive");
    
    // 从配置文件创建
    static std::unique_ptr<INeuronMapper> createFromConfig(
        const std::string& config_file_path);
    
    // 工厂注册系统
    using MapperCreator = std::function<std::unique_ptr<INeuronMapper>(const MapperConfiguration&)>;
    
    static void registerMapper(const std::string& name, MapperCreator creator);
    static void unregisterMapper(const std::string& name);
    static std::vector<std::string> getRegisteredMappers();
    
    // 配置验证
    static std::vector<std::string> validateConfiguration(const MapperConfiguration& config);
    
    // 帮助信息
    static std::string getMapperInfo(MapperType type);
    static std::string getComponentInfo(const std::string& component_name);
    
private:
    static std::unordered_map<std::string, MapperCreator> registered_mappers_;
    
    NeuronMapperFactory() = default;
    
    static void initializeDefaultMappers();
    static MapperConfiguration getDefaultConfig(MapperType type);
};

}

#endif