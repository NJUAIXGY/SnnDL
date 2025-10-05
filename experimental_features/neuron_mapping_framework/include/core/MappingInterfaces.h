#ifndef NEURON_MAPPING_FRAMEWORK_CORE_MAPPING_INTERFACES_H
#define NEURON_MAPPING_FRAMEWORK_CORE_MAPPING_INTERFACES_H

#include "core/Types.h"
#include "core/NeuralNetwork.h"
#include "core/HardwareTopology.h"
#include "core/MappingSolution.h"
#include <memory>

namespace NeuronMapping {

using namespace NeuronMapping;

// 映射策略接口
class IMappingStrategy {
public:
    virtual ~IMappingStrategy() = default;
    
    virtual std::unique_ptr<MappingSolution> mapNetwork(
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config
    ) = 0;
};

// 网络分析器接口
class INetworkAnalyzer {
public:
    virtual ~INetworkAnalyzer() = default;
    
    virtual void analyzeNetwork(const NeuralNetwork& network) = 0;
    virtual std::string getAnalysisReport() const = 0;
};

// 优化器接口
class IOptimizer {
public:
    virtual ~IOptimizer() = default;
    
    virtual std::unique_ptr<MappingSolution> optimize(
        std::unique_ptr<MappingSolution> initial_solution,
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config
    ) = 0;
};

// 评估器接口
class IEvaluator {
public:
    virtual ~IEvaluator() = default;
    
    virtual PerformanceMetrics evaluate(
        const MappingSolution& mapping,
        const NeuralNetwork& network,
        const HardwareTopology& topology,
        const MappingConfig& config
    ) = 0;
};

}

#endif