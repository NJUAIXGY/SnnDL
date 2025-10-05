#include "factories/MapperFactory.h"
#include "strategies/RandomMappingStrategy.h"
#include "strategies/GreedyMappingStrategy.h"
#include "optimizers/LocalSearchOptimizer.h"
#include "optimizers/HillClimbingOptimizer.h"
#include "evaluators/CommunicationCostEvaluator.h"
#include "evaluators/LoadBalanceEvaluator.h"
#include "evaluators/ComprehensiveEvaluator.h"
#include "utils/Logger.h"
#include <fstream>

namespace neuron_mapping {

// 静态成员初始化
std::unordered_map<std::string, NeuronMapperFactory::MapperCreator> 
    NeuronMapperFactory::registered_mappers_;

// ComponentFactory 实现
std::unique_ptr<MappingStrategy> ComponentFactory::createStrategy(
    StrategyType type, 
    const MapperConfiguration& config) {
    
    switch (type) {
        case StrategyType::RANDOM:
            return std::make_unique<RandomMappingStrategy>(config.random_seed);
        
        case StrategyType::GREEDY:
            return std::make_unique<GreedyMappingStrategy>();
        
        case StrategyType::CUSTOM:
        default:
            LOG_WARNING("Unknown strategy type, falling back to RandomMappingStrategy");
            return std::make_unique<RandomMappingStrategy>(config.random_seed);
    }
}

std::unique_ptr<MappingOptimizer> ComponentFactory::createOptimizer(
    OptimizerType type,
    const MapperConfiguration& config) {
    
    switch (type) {
        case OptimizerType::LOCAL_SEARCH:
            return std::make_unique<LocalSearchOptimizer>(
                LocalSearchOptimizer::NeighborType::MIXED, config.random_seed);
        
        case OptimizerType::HILL_CLIMBING:
            return std::make_unique<HillClimbingOptimizer>(
                HillClimbingOptimizer::MoveType::BEST_IMPROVEMENT, config.random_seed);
        
        case OptimizerType::NONE:
        default:
            return nullptr;
    }
}

std::unique_ptr<PerformanceEvaluator> ComponentFactory::createEvaluator(
    EvaluatorType type,
    const MapperConfiguration& config) {
    
    switch (type) {
        case EvaluatorType::COMMUNICATION_COST:
            return std::make_unique<CommunicationCostEvaluator>(config.comm_weights);
        
        case EvaluatorType::LOAD_BALANCE:
            return std::make_unique<LoadBalanceEvaluator>(LoadMetric::COMBINED, config.load_weights);
        
        case EvaluatorType::COMPREHENSIVE:
        default:
            return std::make_unique<ComprehensiveEvaluator>(
                ComprehensiveWeights(), config.comm_weights, config.load_weights);
    }
}

std::vector<std::string> ComponentFactory::getAvailableStrategies() {
    return {"RandomMapping", "GreedyMapping"};
}

std::vector<std::string> ComponentFactory::getAvailableOptimizers() {
    return {"LocalSearch", "HillClimbing", "None"};
}

std::vector<std::string> ComponentFactory::getAvailableEvaluators() {
    return {"CommunicationCost", "LoadBalance", "Comprehensive"};
}

// NeuronMapperFactory 实现
std::unique_ptr<INeuronMapper> NeuronMapperFactory::createMapper(
    const MapperConfiguration& config) {
    
    auto mapper = std::make_unique<NeuronMapping::NeuronMapper>();
    
    // 启用详细日志
    mapper->enableVerboseLogging(config.enable_verbose_logging);
    
    // 创建并设置策略
    auto strategy = ComponentFactory::createStrategy(config.strategy_type, config);
    if (strategy) {
        // 需要适配器将新接口转换为旧接口
        LOG_INFO("Strategy created successfully: " + strategy->getName());
    }
    
    // 创建并设置优化器
    auto optimizer = ComponentFactory::createOptimizer(config.optimizer_type, config);
    if (optimizer) {
        LOG_INFO("Optimizer created successfully: " + optimizer->getName());
    }
    
    // 创建并设置评估器
    auto evaluator = ComponentFactory::createEvaluator(config.evaluator_type, config);
    if (evaluator) {
        LOG_INFO("Evaluator created successfully: " + evaluator->getName());
    }
    
    return mapper;
}

std::unique_ptr<INeuronMapper> NeuronMapperFactory::createMapper(MapperType type) {
    MapperConfiguration config = getDefaultConfig(type);
    return createMapper(config);
}

std::unique_ptr<INeuronMapper> NeuronMapperFactory::createDefaultMapper() {
    return createMapper(MapperType::DEFAULT);
}

std::unique_ptr<INeuronMapper> NeuronMapperFactory::createFastMapper() {
    return createMapper(MapperType::FAST);
}

std::unique_ptr<INeuronMapper> NeuronMapperFactory::createPrecisionMapper() {
    return createMapper(MapperType::PRECISION);
}

std::unique_ptr<INeuronMapper> NeuronMapperFactory::createCustomMapper(
    const std::string& strategy_name,
    const std::string& optimizer_name,
    const std::string& evaluator_name) {
    
    MapperConfiguration config;
    
    // 根据名称设置策略类型
    if (strategy_name == "RandomMapping") {
        config.strategy_type = StrategyType::RANDOM;
    } else if (strategy_name == "GreedyMapping") {
        config.strategy_type = StrategyType::GREEDY;
    } else {
        LOG_WARNING("Unknown strategy name: " + strategy_name + ", using Random");
        config.strategy_type = StrategyType::RANDOM;
    }
    
    // 根据名称设置优化器类型
    if (optimizer_name == "LocalSearch") {
        config.optimizer_type = OptimizerType::LOCAL_SEARCH;
    } else if (optimizer_name == "HillClimbing") {
        config.optimizer_type = OptimizerType::HILL_CLIMBING;
    } else if (optimizer_name == "None") {
        config.optimizer_type = OptimizerType::NONE;
    } else {
        LOG_WARNING("Unknown optimizer name: " + optimizer_name + ", using LocalSearch");
        config.optimizer_type = OptimizerType::LOCAL_SEARCH;
    }
    
    // 根据名称设置评估器类型
    if (evaluator_name == "CommunicationCost") {
        config.evaluator_type = EvaluatorType::COMMUNICATION_COST;
    } else if (evaluator_name == "LoadBalance") {
        config.evaluator_type = EvaluatorType::LOAD_BALANCE;
    } else if (evaluator_name == "Comprehensive") {
        config.evaluator_type = EvaluatorType::COMPREHENSIVE;
    } else {
        LOG_WARNING("Unknown evaluator name: " + evaluator_name + ", using Comprehensive");
        config.evaluator_type = EvaluatorType::COMPREHENSIVE;
    }
    
    return createMapper(config);
}

std::unique_ptr<INeuronMapper> NeuronMapperFactory::createFromConfig(
    const std::string& config_file_path) {
    
    // 简化实现：返回默认映射器
    LOG_WARNING("Config file loading not implemented, returning default mapper");
    return createDefaultMapper();
}

void NeuronMapperFactory::registerMapper(const std::string& name, MapperCreator creator) {
    registered_mappers_[name] = creator;
    LOG_INFO("Mapper registered: " + name);
}

void NeuronMapperFactory::unregisterMapper(const std::string& name) {
    registered_mappers_.erase(name);
    LOG_INFO("Mapper unregistered: " + name);
}

std::vector<std::string> NeuronMapperFactory::getRegisteredMappers() {
    std::vector<std::string> names;
    names.reserve(registered_mappers_.size());
    
    for (const auto& pair : registered_mappers_) {
        names.push_back(pair.first);
    }
    
    return names;
}

std::vector<std::string> NeuronMapperFactory::validateConfiguration(
    const MapperConfiguration& config) {
    
    std::vector<std::string> errors;
    
    if (config.max_optimization_iterations == 0) {
        errors.push_back("max_optimization_iterations must be greater than 0");
    }
    
    if (config.convergence_threshold <= 0.0) {
        errors.push_back("convergence_threshold must be positive");
    }
    
    if (config.comm_weights.intra_pe_weight < 0 || config.comm_weights.inter_pe_weight < 0) {
        errors.push_back("Communication weights must be non-negative");
    }
    
    if (config.load_weights.neuron_count_weight < 0 || config.load_weights.computational_weight < 0) {
        errors.push_back("Load weights must be non-negative");
    }
    
    return errors;
}

std::string NeuronMapperFactory::getMapperInfo(MapperType type) {
    switch (type) {
        case MapperType::DEFAULT:
            return "Default mapper with balanced performance and accuracy";
        case MapperType::FAST:
            return "Fast mapper optimized for quick results";
        case MapperType::PRECISION:
            return "Precision mapper optimized for solution quality";
        case MapperType::CUSTOM:
            return "Custom mapper with user-defined configuration";
        default:
            return "Unknown mapper type";
    }
}

std::string NeuronMapperFactory::getComponentInfo(const std::string& component_name) {
    if (component_name == "RandomMapping") {
        return "Random mapping strategy for baseline performance";
    } else if (component_name == "GreedyMapping") {
        return "Greedy mapping strategy for communication cost optimization";
    } else if (component_name == "LocalSearch") {
        return "Local search optimizer using neighborhood operations";
    } else if (component_name == "HillClimbing") {
        return "Hill climbing optimizer with best/first improvement";
    } else if (component_name == "Comprehensive") {
        return "Comprehensive evaluator combining multiple metrics";
    }
    
    return "Component information not available";
}

void NeuronMapperFactory::initializeDefaultMappers() {
    // 注册默认映射器
    registerMapper("default", [](const MapperConfiguration& config) {
        return createMapper(config);
    });
    
    registerMapper("fast", [](const MapperConfiguration& config) {
        auto fast_config = config;
        fast_config.strategy_type = StrategyType::RANDOM;
        fast_config.optimizer_type = OptimizerType::NONE;
        return createMapper(fast_config);
    });
    
    registerMapper("precision", [](const MapperConfiguration& config) {
        auto precision_config = config;
        precision_config.strategy_type = StrategyType::GREEDY;
        precision_config.optimizer_type = OptimizerType::HILL_CLIMBING;
        precision_config.max_optimization_iterations = 5000;
        return createMapper(precision_config);
    });
}

MapperConfiguration NeuronMapperFactory::getDefaultConfig(MapperType type) {
    MapperConfiguration config;
    
    switch (type) {
        case MapperType::FAST:
            config.strategy_type = StrategyType::RANDOM;
            config.optimizer_type = OptimizerType::NONE;
            config.max_optimization_iterations = 100;
            break;
            
        case MapperType::PRECISION:
            config.strategy_type = StrategyType::GREEDY;
            config.optimizer_type = OptimizerType::HILL_CLIMBING;
            config.max_optimization_iterations = 5000;
            config.convergence_threshold = 1e-8;
            break;
            
        case MapperType::DEFAULT:
        case MapperType::CUSTOM:
        default:
            config.strategy_type = StrategyType::RANDOM;
            config.optimizer_type = OptimizerType::LOCAL_SEARCH;
            config.max_optimization_iterations = 1000;
            config.convergence_threshold = 1e-6;
            break;
    }
    
    return config;
}

}