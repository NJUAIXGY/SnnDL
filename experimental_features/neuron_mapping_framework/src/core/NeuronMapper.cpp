#include "core/NeuronMapper.h"
#include "strategies/RandomMappingStrategy.h"
#include "strategies/GreedyMappingStrategy.h"
#include "optimizers/LocalSearchOptimizer.h"
#include "optimizers/HillClimbingOptimizer.h"
#include "evaluators/ComprehensiveEvaluator.h"
#include "utils/Logger.h"
#include <algorithm>
#include <sstream>

namespace NeuronMapping {

// 适配器类：将新的组件接口适配到旧的接口
namespace {

class MappingStrategyAdapter : public IMappingStrategy {
private:
    std::unique_ptr<neuron_mapping::MappingStrategy> strategy_;
public:
    explicit MappingStrategyAdapter(std::unique_ptr<neuron_mapping::MappingStrategy> strategy)
        : strategy_(std::move(strategy)) {}
    
    std::unique_ptr<MappingSolution> mapNetwork(
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config) override {
        if (strategy_) {
            return strategy_->mapNetwork(network, topology, config);
        }
        return nullptr;
    }
};

class EvaluatorAdapter : public IEvaluator {
private:
    std::unique_ptr<neuron_mapping::PerformanceEvaluator> evaluator_;
public:
    explicit EvaluatorAdapter(std::unique_ptr<neuron_mapping::PerformanceEvaluator> evaluator)
        : evaluator_(std::move(evaluator)) {}
    
    PerformanceMetrics evaluate(
        const MappingSolution& mapping,
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config) override {
        if (evaluator_) {
            return evaluator_->evaluateBasic(mapping, network, topology, config);
        }
        return PerformanceMetrics();
    }
};

}

NeuronMapper::NeuronMapper() {
    initializeDefaultComponents();
}

NeuronMapper::~NeuronMapper() = default;

std::unique_ptr<MappingSolution> NeuronMapper::mapNetwork(
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const MappingConfig& config) {
    
    logEvent("MAPPING_START", "Starting network mapping");
    
    auto result = executeMapping(network, topology, config);
    
    if (result) {
        auto metrics = evaluateMapping(*result, network, topology, config);
        
        std::ostringstream stats;
        stats << "Mapping completed - Communication cost: " << metrics.communication_cost
              << ", Load imbalance: " << metrics.load_imbalance
              << ", PE utilization: " << metrics.pe_utilization;
        last_mapping_stats_ = stats.str();
        
        logEvent("MAPPING_COMPLETE", last_mapping_stats_);
    } else {
        logEvent("MAPPING_ERROR", "Failed to create mapping");
    }
    
    return result;
}

std::unique_ptr<MappingSolution> NeuronMapper::incrementalMap(
    const MappingSolution& current_mapping,
    const std::vector<NeuronId>& new_neurons,
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const MappingConfig& config) {
    
    logEvent("INCREMENTAL_MAPPING_START", "Starting incremental mapping for " + 
             std::to_string(new_neurons.size()) + " neurons");
    
    // 简化实现：从当前映射开始，为新神经元分配位置
    auto solution = std::make_unique<MappingSolution>(current_mapping);
    
    // 为每个新神经元找到最佳位置
    for (NeuronId neuron_id : new_neurons) {
        PEId best_pe = INVALID_PE_ID;
        float best_cost = std::numeric_limits<float>::max();
        
        for (PEId pe_id = 0; pe_id < topology.getTotalPEs(); ++pe_id) {
            if (solution->getPENeuronCount(pe_id) < topology.getPECapacity(pe_id)) {
                // 简单的负载均衡策略
                float cost = solution->getPENeuronCount(pe_id);
                if (cost < best_cost) {
                    best_cost = cost;
                    best_pe = pe_id;
                }
            }
        }
        
        if (best_pe != INVALID_PE_ID) {
            solution->assignNeuron(neuron_id, best_pe, 0);
        }
    }
    
    logEvent("INCREMENTAL_MAPPING_COMPLETE", "Incremental mapping completed");
    return solution;
}

std::unique_ptr<MappingSolution> NeuronMapper::remapNetwork(
    const MappingSolution& current_mapping,
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const MappingConfig& config) {
    
    logEvent("REMAPPING_START", "Starting network remapping");
    
    auto result = executeMapping(network, topology, config, &current_mapping);
    
    if (result && optimizer_) {
        logEvent("OPTIMIZATION_START", "Starting mapping optimization");
        
        neuron_mapping::OptimizationConfig opt_config;
        opt_config.max_iterations = config.max_iterations;
        opt_config.convergence_threshold = 1e-6;
        opt_config.enable_logging = verbose_logging_;
        
        // 创建适配的优化器
        auto local_optimizer = std::make_unique<neuron_mapping::LocalSearchOptimizer>();
        result = local_optimizer->optimize(std::move(result), network, topology, opt_config);
        
        logEvent("OPTIMIZATION_COMPLETE", "Mapping optimization completed");
    }
    
    logEvent("REMAPPING_COMPLETE", "Network remapping completed");
    return result;
}

void NeuronMapper::setMappingStrategy(std::unique_ptr<IMappingStrategy> strategy) {
    strategy_ = std::move(strategy);
    logEvent("COMPONENT_UPDATE", "Mapping strategy updated");
}

void NeuronMapper::setNetworkAnalyzer(std::unique_ptr<INetworkAnalyzer> analyzer) {
    analyzer_ = std::move(analyzer);
    logEvent("COMPONENT_UPDATE", "Network analyzer updated");
}

void NeuronMapper::setOptimizer(std::unique_ptr<IOptimizer> optimizer) {
    optimizer_ = std::move(optimizer);
    logEvent("COMPONENT_UPDATE", "Optimizer updated");
}

void NeuronMapper::setEvaluator(std::unique_ptr<IEvaluator> evaluator) {
    evaluator_ = std::move(evaluator);
    logEvent("COMPONENT_UPDATE", "Evaluator updated");
}

PerformanceMetrics NeuronMapper::evaluateMapping(
    const MappingSolution& mapping,
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const MappingConfig& config) {
    
    if (evaluator_) {
        return evaluator_->evaluate(mapping, network, topology, config);
    }
    
    // 默认评估器
    auto default_evaluator = std::make_unique<neuron_mapping::ComprehensiveEvaluator>();
    return default_evaluator->evaluateBasic(mapping, network, topology, config);
}

std::vector<size_t> NeuronMapper::compareMappings(
    const std::vector<std::unique_ptr<MappingSolution>>& mappings,
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const MappingConfig& config) {
    
    std::vector<std::pair<size_t, float>> indexed_scores;
    
    for (size_t i = 0; i < mappings.size(); ++i) {
        auto metrics = evaluateMapping(*mappings[i], network, topology, config);
        float score = metrics.communication_cost + metrics.load_imbalance * 0.5f;
        indexed_scores.emplace_back(i, score);
    }
    
    // 按分数排序（越低越好）
    std::sort(indexed_scores.begin(), indexed_scores.end(),
              [](const auto& a, const auto& b) { return a.second < b.second; });
    
    std::vector<size_t> result;
    result.reserve(indexed_scores.size());
    for (const auto& pair : indexed_scores) {
        result.push_back(pair.first);
    }
    
    return result;
}

std::vector<std::string> NeuronMapper::getSupportedStrategies() const {
    return {"RandomMapping", "GreedyMapping"};
}

std::vector<std::string> NeuronMapper::getSupportedOptimizers() const {
    return {"LocalSearch", "HillClimbing"};
}

std::vector<std::unique_ptr<MappingSolution>> NeuronMapper::mapMultipleNetworks(
    const std::vector<std::reference_wrapper<const NeuralNetwork>>& networks,
    const HardwareTopology& topology,
    const MappingConfig& config) {
    
    std::vector<std::unique_ptr<MappingSolution>> results;
    results.reserve(networks.size());
    
    for (const auto& network_ref : networks) {
        auto solution = mapNetwork(network_ref.get(), topology, config);
        results.push_back(std::move(solution));
    }
    
    return results;
}

std::unique_ptr<MappingSolution> NeuronMapper::adaptiveMap(
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const MappingConfig& config) {
    
    // 根据网络特征选择策略
    std::string best_strategy = selectBestStrategy(network, topology);
    
    logEvent("ADAPTIVE_MAPPING", "Selected strategy: " + best_strategy);
    
    // 临时设置策略并执行映射
    auto original_strategy = std::move(strategy_);
    
    if (best_strategy == "GreedyMapping") {
        auto greedy_strategy = std::make_unique<neuron_mapping::GreedyMappingStrategy>();
        strategy_ = std::make_unique<MappingStrategyAdapter>(std::move(greedy_strategy));
    } else {
        auto random_strategy = std::make_unique<neuron_mapping::RandomMappingStrategy>();
        strategy_ = std::make_unique<MappingStrategyAdapter>(std::move(random_strategy));
    }
    
    auto result = executeMapping(network, topology, config);
    
    // 恢复原始策略
    strategy_ = std::move(original_strategy);
    
    return result;
}

std::vector<std::unique_ptr<MappingSolution>> NeuronMapper::multiObjectiveMap(
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const MappingConfig& config,
    const std::vector<float>& objectives) {
    
    std::vector<std::unique_ptr<MappingSolution>> pareto_solutions;
    
    // 简化实现：生成多个不同权重的解
    for (float comm_weight = 0.1f; comm_weight <= 1.0f; comm_weight += 0.3f) {
        auto solution = mapNetwork(network, topology, config);
        if (solution) {
            pareto_solutions.push_back(std::move(solution));
        }
    }
    
    return pareto_solutions;
}

std::string NeuronMapper::generateMappingReport(
    const MappingSolution& mapping,
    const NeuralNetwork& network,
    const HardwareTopology& topology) const {
    
    std::ostringstream report;
    report << "=== Neuron Mapping Report ===\n";
    report << "Network: " << network.getNeuronCount() << " neurons, " 
           << network.getConnectionCount() << " connections\n";
    report << "Hardware: " << topology.getTotalPEs() << " PEs\n";
    report << "Assigned neurons: " << mapping.getAssignedNeuronCount() << "\n";
    
    MappingConfig default_config;
    auto metrics = const_cast<NeuronMapper*>(this)->evaluateMapping(mapping, network, topology, default_config);
    
    report << "Performance Metrics:\n";
    report << "  Communication cost: " << metrics.communication_cost << "\n";
    report << "  Load imbalance: " << metrics.load_imbalance << "\n";
    report << "  PE utilization: " << metrics.pe_utilization << "\n";
    report << "  Inter-PE communication ratio: " << metrics.inter_pe_communication_ratio << "\n";
    
    return report.str();
}

std::vector<std::string> NeuronMapper::validateConfiguration(const MappingConfig& config) const {
    std::vector<std::string> errors;
    
    if (config.max_iterations == 0) {
        errors.push_back("max_iterations must be greater than 0");
    }
    
    if (config.strategy.empty()) {
        errors.push_back("strategy cannot be empty");
    }
    
    return errors;
}

std::unique_ptr<MappingSolution> NeuronMapper::executeMapping(
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const MappingConfig& config,
    const MappingSolution* initial_mapping) {
    
    if (!strategy_) {
        logEvent("ERROR", "No mapping strategy available");
        return nullptr;
    }
    
    updateProgress(0, config.max_iterations, 0.0f, 0.0f);
    
    auto solution = strategy_->mapNetwork(network, topology, config);
    
    updateProgress(config.max_iterations, config.max_iterations, 
                  solution ? 1.0f : std::numeric_limits<float>::max(),
                  solution ? 1.0f : std::numeric_limits<float>::max());
    
    return solution;
}

std::string NeuronMapper::selectBestStrategy(
    const NeuralNetwork& network,
    const HardwareTopology& topology) const {
    
    // 简单的策略选择逻辑
    if (network.getNeuronCount() > 1000) {
        return "RandomMapping";  // 大网络使用随机映射
    } else {
        return "GreedyMapping";  // 小网络使用贪心映射
    }
}

void NeuronMapper::initializeDefaultComponents() {
    // 设置默认策略
    auto default_strategy = std::make_unique<neuron_mapping::RandomMappingStrategy>();
    strategy_ = std::make_unique<MappingStrategyAdapter>(std::move(default_strategy));
    
    // 设置默认评估器
    auto default_evaluator = std::make_unique<neuron_mapping::ComprehensiveEvaluator>();
    evaluator_ = std::make_unique<EvaluatorAdapter>(std::move(default_evaluator));
    
    logEvent("INITIALIZATION", "Default components initialized");
}

void NeuronMapper::logEvent(const std::string& event_type, const std::string& message) const {
    if (verbose_logging_) {
        LOG_INFO("[" + event_type + "] " + message);
    }
    
    if (event_callback_) {
        event_callback_(event_type, message);
    }
}

void NeuronMapper::updateProgress(uint32_t current, uint32_t max, float current_cost, float best_cost) const {
    if (progress_callback_) {
        progress_callback_(current, max, current_cost, best_cost);
    }
}

// MapperFactory 实现
std::unique_ptr<INeuronMapper> MapperFactory::createDefaultMapper() {
    return std::make_unique<NeuronMapper>();
}

std::unique_ptr<INeuronMapper> MapperFactory::createFastMapper() {
    auto mapper = std::make_unique<NeuronMapper>();
    
    // 配置快速映射器
    auto fast_strategy = std::make_unique<neuron_mapping::RandomMappingStrategy>();
    mapper->setMappingStrategy(std::make_unique<MappingStrategyAdapter>(std::move(fast_strategy)));
    
    return mapper;
}

std::unique_ptr<INeuronMapper> MapperFactory::createPrecisionMapper() {
    auto mapper = std::make_unique<NeuronMapper>();
    
    // 配置高精度映射器
    auto precision_strategy = std::make_unique<neuron_mapping::GreedyMappingStrategy>();
    mapper->setMappingStrategy(std::make_unique<MappingStrategyAdapter>(std::move(precision_strategy)));
    
    return mapper;
}

std::unique_ptr<INeuronMapper> MapperFactory::createCustomMapper(
    const std::string& strategy_name,
    const std::string& optimizer_name) {
    
    auto mapper = std::make_unique<NeuronMapper>();
    
    // 根据名称创建策略
    if (strategy_name == "RandomMapping") {
        auto strategy = std::make_unique<neuron_mapping::RandomMappingStrategy>();
        mapper->setMappingStrategy(std::make_unique<MappingStrategyAdapter>(std::move(strategy)));
    } else if (strategy_name == "GreedyMapping") {
        auto strategy = std::make_unique<neuron_mapping::GreedyMappingStrategy>();
        mapper->setMappingStrategy(std::make_unique<MappingStrategyAdapter>(std::move(strategy)));
    }
    
    return mapper;
}

std::vector<std::string> MapperFactory::getAvailableMapperTypes() {
    return {"DefaultMapper", "FastMapper", "PrecisionMapper", "CustomMapper"};
}

}