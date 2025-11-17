#ifndef NEURON_MAPPING_SIMULATED_ANNEALING_OPTIMIZER_H
#define NEURON_MAPPING_SIMULATED_ANNEALING_OPTIMIZER_H

#include "MappingOptimizer.h"
#include <random>
#include <functional>

namespace neuron_mapping {

/**
 * @brief 模拟退火优化器
 * 
 * 使用模拟退火算法优化神经元映射方案。
 * 通过温度控制接受较差解的概率，逐渐降低温度以收敛到局部最优解。
 */
class SimulatedAnnealingOptimizer : public MappingOptimizer {
public:
    /**
     * @brief 温度调度策略
     */
    enum class CoolingSchedule {
        LINEAR,         // 线性降温
        EXPONENTIAL,    // 指数降温
        LOGARITHMIC,    // 对数降温
        ADAPTIVE        // 自适应降温
    };

    /**
     * @brief 邻域生成策略
     */
    enum class NeighborhoodStrategy {
        SINGLE_SWAP,    // 单个神经元交换
        DOUBLE_SWAP,    // 两个神经元交换
        BLOCK_MOVE,     // 块移动
        RANDOM_RESTART  // 随机重启
    };

    /**
     * @brief 构造函数
     * @param initial_temperature 初始温度
     * @param final_temperature 终止温度
     * @param max_iterations 最大迭代次数
     * @param cooling_schedule 降温策略
     * @param neighborhood_strategy 邻域策略
     * @param seed 随机种子
     */
    explicit SimulatedAnnealingOptimizer(
        double initial_temperature = 100.0,
        double final_temperature = 0.01,
        uint32_t max_iterations = 10000,
        CoolingSchedule cooling_schedule = CoolingSchedule::EXPONENTIAL,
        NeighborhoodStrategy neighborhood_strategy = NeighborhoodStrategy::SINGLE_SWAP,
        uint32_t seed = 42);

    virtual ~SimulatedAnnealingOptimizer() = default;

    // 实现基类接口
    std::unique_ptr<NeuronMapping::MappingSolution> optimize(
        std::unique_ptr<NeuronMapping::MappingSolution> initial_solution,
        const NeuronMapping::NeuralNetwork& network,
        const NeuronMapping::HardwareTopology& topology,
        const OptimizationConfig& config) override;

    std::string getName() const override { return "SimulatedAnnealing"; }
    std::string getDescription() const override;

    // 参数设置接口
    void setTemperatureRange(double initial, double final) {
        initial_temperature_ = initial;
        final_temperature_ = final;
    }
    
    void setMaxIterations(uint32_t max_iter) { max_iterations_ = max_iter; }
    void setCoolingSchedule(CoolingSchedule schedule) { cooling_schedule_ = schedule; }
    void setNeighborhoodStrategy(NeighborhoodStrategy strategy) { neighborhood_strategy_ = strategy; }
    void setAcceptanceRatio(double target_ratio) { target_acceptance_ratio_ = target_ratio; }
    
    // 统计信息获取
    uint32_t getIterationsPerformed() const { return iterations_performed_; }
    double getFinalTemperature() const { return current_temperature_; }
    double getAcceptanceRatio() const { return acceptance_ratio_; }
    double getBestCost() const { return best_cost_; }

private:
    // 算法参数
    double initial_temperature_;
    double final_temperature_;
    uint32_t max_iterations_;
    CoolingSchedule cooling_schedule_;
    NeighborhoodStrategy neighborhood_strategy_;
    double target_acceptance_ratio_;
    
    // 运行时状态
    double current_temperature_;
    uint32_t iterations_performed_;
    double acceptance_ratio_;
    double best_cost_;
    uint32_t accepted_moves_;
    uint32_t total_moves_;
    
    // 随机数生成
    uint32_t seed_;
    std::mt19937 rng_;
    std::uniform_real_distribution<double> uniform_dist_;

    // 核心算法
    std::unique_ptr<NeuronMapping::MappingSolution> performSimulatedAnnealing(
        std::unique_ptr<NeuronMapping::MappingSolution> solution,
        const NeuronMapping::NeuralNetwork& network,
        const NeuronMapping::HardwareTopology& topology,
        const OptimizationConfig& config);

    // 邻域生成
    std::unique_ptr<NeuronMapping::MappingSolution> generateNeighbor(
        const NeuronMapping::MappingSolution& current_solution,
        const NeuronMapping::NeuralNetwork& network,
        const NeuronMapping::HardwareTopology& topology);

    std::unique_ptr<NeuronMapping::MappingSolution> singleSwapNeighbor(
        const NeuronMapping::MappingSolution& solution,
        const NeuronMapping::NeuralNetwork& network,
        const NeuronMapping::HardwareTopology& topology);

    std::unique_ptr<NeuronMapping::MappingSolution> doubleSwapNeighbor(
        const NeuronMapping::MappingSolution& solution,
        const NeuronMapping::NeuralNetwork& network,
        const NeuronMapping::HardwareTopology& topology);

    std::unique_ptr<NeuronMapping::MappingSolution> blockMoveNeighbor(
        const NeuronMapping::MappingSolution& solution,
        const NeuronMapping::NeuralNetwork& network,
        const NeuronMapping::HardwareTopology& topology);

    std::unique_ptr<NeuronMapping::MappingSolution> randomRestartNeighbor(
        const NeuronMapping::MappingSolution& solution,
        const NeuronMapping::NeuralNetwork& network,
        const NeuronMapping::HardwareTopology& topology);

    // 温度控制
    void updateTemperature(uint32_t iteration);
    double calculateTemperature(uint32_t iteration) const;

    // 接受准则
    bool acceptSolution(double current_cost, double neighbor_cost) const;
    double calculateAcceptanceProbability(double current_cost, double neighbor_cost) const;

    // 自适应参数调整
    void adaptParameters(uint32_t iteration);
    void updateAcceptanceRatio();

    // 收敛检测
    bool hasConverged(uint32_t iteration, double cost_improvement) const;

    // 统计和调试
    void logProgress(uint32_t iteration, double current_cost, double best_cost) const;
    void printFinalStatistics() const;

    // 工具方法
    std::vector<NeuronMapping::NeuronId> getRandomNeuronSubset(
        const NeuronMapping::NeuralNetwork& network, 
        size_t subset_size) const;

    NeuronMapping::PEId findBestTargetPE(
        NeuronMapping::NeuronId neuron_id,
        const NeuronMapping::MappingSolution& solution,
        const NeuronMapping::NeuralNetwork& network,
        const NeuronMapping::HardwareTopology& topology) const;
};

} // namespace neuron_mapping

#endif // NEURON_MAPPING_SIMULATED_ANNEALING_OPTIMIZER_H