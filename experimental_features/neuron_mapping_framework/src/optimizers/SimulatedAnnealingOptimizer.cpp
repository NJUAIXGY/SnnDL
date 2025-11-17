#include "optimizers/SimulatedAnnealingOptimizer.h"
#include "utils/Logger.h"
#include "utils/MathUtils.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace neuron_mapping {

using namespace NeuronMapping;

SimulatedAnnealingOptimizer::SimulatedAnnealingOptimizer(
    double initial_temperature,
    double final_temperature,
    uint32_t max_iterations,
    CoolingSchedule cooling_schedule,
    NeighborhoodStrategy neighborhood_strategy,
    uint32_t seed)
    : initial_temperature_(initial_temperature)
    , final_temperature_(final_temperature)
    , max_iterations_(max_iterations)
    , cooling_schedule_(cooling_schedule)
    , neighborhood_strategy_(neighborhood_strategy)
    , target_acceptance_ratio_(0.5)  // 目标接受率50%
    , current_temperature_(initial_temperature)
    , iterations_performed_(0)
    , acceptance_ratio_(0.0)
    , best_cost_(std::numeric_limits<double>::max())
    , accepted_moves_(0)
    , total_moves_(0)
    , seed_(seed)
    , rng_(seed)
    , uniform_dist_(0.0, 1.0) {
}

std::string SimulatedAnnealingOptimizer::getDescription() const {
    std::string schedule_name;
    switch (cooling_schedule_) {
        case CoolingSchedule::LINEAR: schedule_name = "Linear"; break;
        case CoolingSchedule::EXPONENTIAL: schedule_name = "Exponential"; break;
        case CoolingSchedule::LOGARITHMIC: schedule_name = "Logarithmic"; break;
        case CoolingSchedule::ADAPTIVE: schedule_name = "Adaptive"; break;
        default: schedule_name = "Unknown"; break;
    }

    std::string neighborhood_name;
    switch (neighborhood_strategy_) {
        case NeighborhoodStrategy::SINGLE_SWAP: neighborhood_name = "Single Swap"; break;
        case NeighborhoodStrategy::DOUBLE_SWAP: neighborhood_name = "Double Swap"; break;
        case NeighborhoodStrategy::BLOCK_MOVE: neighborhood_name = "Block Move"; break;
        case NeighborhoodStrategy::RANDOM_RESTART: neighborhood_name = "Random Restart"; break;
        default: neighborhood_name = "Unknown"; break;
    }

    return "Simulated Annealing Optimizer with " + schedule_name + 
           " cooling and " + neighborhood_name + " neighborhood";
}

std::unique_ptr<MappingSolution> SimulatedAnnealingOptimizer::optimize(
    std::unique_ptr<MappingSolution> initial_solution,
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const OptimizationConfig& config) {

    if (!initial_solution) {
        LOG_ERROR("SimulatedAnnealing: No initial solution provided");
        return nullptr;
    }

    LOG_INFO("Starting Simulated Annealing optimization");
    LOG_INFO("Initial temperature: " + std::to_string(initial_temperature_));
    LOG_INFO("Final temperature: " + std::to_string(final_temperature_));
    LOG_INFO("Max iterations: " + std::to_string(max_iterations_));

    // 重置状态
    current_temperature_ = initial_temperature_;
    iterations_performed_ = 0;
    accepted_moves_ = 0;
    total_moves_ = 0;
    best_cost_ = evaluateObjective(*initial_solution, network, topology, config);

    // 执行模拟退火算法
    auto optimized_solution = performSimulatedAnnealing(
        std::move(initial_solution), network, topology, config);

    printFinalStatistics();
    return optimized_solution;
}

std::unique_ptr<MappingSolution> SimulatedAnnealingOptimizer::performSimulatedAnnealing(
    std::unique_ptr<MappingSolution> solution,
    const NeuralNetwork& network,
    const HardwareTopology& topology,
    const OptimizationConfig& config) {

    auto current_solution = std::move(solution);
    auto best_solution = std::make_unique<MappingSolution>(*current_solution);
    
    double current_cost = evaluateObjective(*current_solution, network, topology, config);
    best_cost_ = current_cost;

    LOG_INFO("Initial solution cost: " + std::to_string(current_cost));

    for (uint32_t iteration = 0; iteration < max_iterations_; ++iteration) {
        iterations_performed_ = iteration + 1;

        // 生成邻域解
        auto neighbor_solution = generateNeighbor(*current_solution, network, topology);
        if (!neighbor_solution) {
            continue;
        }

        double neighbor_cost = evaluateObjective(*neighbor_solution, network, topology, config);
        total_moves_++;

        // 决定是否接受新解
        bool accept = acceptSolution(current_cost, neighbor_cost);
        
        if (accept) {
            current_solution = std::move(neighbor_solution);
            current_cost = neighbor_cost;
            accepted_moves_++;

            // 更新最佳解
            if (neighbor_cost < best_cost_) {
                best_solution = std::make_unique<MappingSolution>(*current_solution);
                best_cost_ = neighbor_cost;
                LOG_DEBUG("New best solution found: cost = " + std::to_string(best_cost_));
            }
        }

        // 更新温度
        updateTemperature(iteration);

        // 自适应参数调整
        if ((iteration + 1) % 100 == 0) {
            adaptParameters(iteration);
        }

        // 定期输出进度
        if ((iteration + 1) % 1000 == 0) {
            logProgress(iteration, current_cost, best_cost_);
        }

        // 检查收敛
        double improvement = (current_cost - best_cost_) / best_cost_;
        if (hasConverged(iteration, improvement)) {
            LOG_INFO("Converged at iteration " + std::to_string(iteration + 1));
            break;
        }

        // 温度过低时停止
        if (current_temperature_ < final_temperature_) {
            LOG_INFO("Temperature reached minimum at iteration " + std::to_string(iteration + 1));
            break;
        }
    }

    updateAcceptanceRatio();
    return best_solution;
}

std::unique_ptr<MappingSolution> SimulatedAnnealingOptimizer::generateNeighbor(
    const MappingSolution& current_solution,
    const NeuralNetwork& network,
    const HardwareTopology& topology) {

    switch (neighborhood_strategy_) {
        case NeighborhoodStrategy::SINGLE_SWAP:
            return singleSwapNeighbor(current_solution, network, topology);
        case NeighborhoodStrategy::DOUBLE_SWAP:
            return doubleSwapNeighbor(current_solution, network, topology);
        case NeighborhoodStrategy::BLOCK_MOVE:
            return blockMoveNeighbor(current_solution, network, topology);
        case NeighborhoodStrategy::RANDOM_RESTART:
            return randomRestartNeighbor(current_solution, network, topology);
        default:
            return singleSwapNeighbor(current_solution, network, topology);
    }
}

std::unique_ptr<MappingSolution> SimulatedAnnealingOptimizer::singleSwapNeighbor(
    const MappingSolution& solution,
    const NeuralNetwork& network,
    const HardwareTopology& topology) {

    auto neighbor = std::make_unique<MappingSolution>(solution);
    
    // 随机选择一个神经元
    auto neuron_ids = network.getAllNeuronIds();
    if (neuron_ids.empty()) return nullptr;
    
    std::uniform_int_distribution<size_t> neuron_dist(0, neuron_ids.size() - 1);
    NeuronId selected_neuron = neuron_ids[neuron_dist(rng_)];
    
    // 找到当前PE
    PEId current_pe = solution.getNeuronPE(selected_neuron);
    if (current_pe == INVALID_PE_ID) return nullptr;
    
    // 随机选择目标PE
    uint32_t total_pes = topology.getTotalPEs();
    std::uniform_int_distribution<PEId> pe_dist(0, total_pes - 1);
    
    PEId target_pe;
    int attempts = 0;
    do {
        target_pe = pe_dist(rng_);
        attempts++;
    } while (target_pe == current_pe && attempts < 10);
    
    if (target_pe == current_pe) return nullptr;
    
    // 执行移动
    if (neighbor->moveNeuron(selected_neuron, target_pe)) {
        return neighbor;
    }
    
    return nullptr;
}

std::unique_ptr<MappingSolution> SimulatedAnnealingOptimizer::doubleSwapNeighbor(
    const MappingSolution& solution,
    const NeuralNetwork& network,
    const HardwareTopology& topology) {

    auto neighbor = std::make_unique<MappingSolution>(solution);
    
    auto neuron_ids = network.getAllNeuronIds();
    if (neuron_ids.size() < 2) return nullptr;
    
    // 随机选择两个不同的神经元
    std::uniform_int_distribution<size_t> neuron_dist(0, neuron_ids.size() - 1);
    
    size_t idx1 = neuron_dist(rng_);
    size_t idx2;
    int attempts = 0;
    do {
        idx2 = neuron_dist(rng_);
        attempts++;
    } while (idx1 == idx2 && attempts < 10);
    
    if (idx1 == idx2) return nullptr;
    
    NeuronId neuron1 = neuron_ids[idx1];
    NeuronId neuron2 = neuron_ids[idx2];
    
    PEId pe1 = solution.getNeuronPE(neuron1);
    PEId pe2 = solution.getNeuronPE(neuron2);
    
    if (pe1 == INVALID_PE_ID || pe2 == INVALID_PE_ID || pe1 == pe2) {
        return nullptr;
    }
    
    // 交换两个神经元的PE分配
    if (neighbor->moveNeuron(neuron1, pe2) && neighbor->moveNeuron(neuron2, pe1)) {
        return neighbor;
    }
    
    return nullptr;
}

std::unique_ptr<MappingSolution> SimulatedAnnealingOptimizer::blockMoveNeighbor(
    const MappingSolution& solution,
    const NeuralNetwork& network,
    const HardwareTopology& topology) {

    auto neighbor = std::make_unique<MappingSolution>(solution);
    
    // 选择一个随机的神经元块（2-5个神经元）
    std::uniform_int_distribution<size_t> block_size_dist(2, 5);
    size_t block_size = std::min(static_cast<size_t>(block_size_dist(rng_)), 
                                 static_cast<size_t>(network.getNeuronCount() / 2));
    
    auto selected_neurons = getRandomNeuronSubset(network, block_size);
    if (selected_neurons.empty()) return nullptr;
    
    // 选择目标PE
    uint32_t total_pes = topology.getTotalPEs();
    std::uniform_int_distribution<PEId> pe_dist(0, total_pes - 1);
    PEId target_pe = pe_dist(rng_);
    
    // 尝试将所有选中的神经元移动到目标PE
    bool success = true;
    for (NeuronId neuron_id : selected_neurons) {
        if (solution.getNeuronPE(neuron_id) != target_pe) {
            if (!neighbor->moveNeuron(neuron_id, target_pe)) {
                success = false;
                break;
            }
        }
    }
    
    return success ? std::move(neighbor) : nullptr;
}

std::unique_ptr<MappingSolution> SimulatedAnnealingOptimizer::randomRestartNeighbor(
    const MappingSolution& solution,
    const NeuralNetwork& network,
    const HardwareTopology& topology) {

    auto neighbor = std::make_unique<MappingSolution>(solution);
    
    // 随机重新分配一定比例的神经元
    auto neuron_ids = network.getAllNeuronIds();
    std::uniform_real_distribution<double> ratio_dist(0.1, 0.3);  // 10%-30%的神经元
    size_t num_to_reassign = static_cast<size_t>(neuron_ids.size() * ratio_dist(rng_));
    
    if (num_to_reassign == 0) num_to_reassign = 1;
    
    auto selected_neurons = getRandomNeuronSubset(network, num_to_reassign);
    
    uint32_t total_pes = topology.getTotalPEs();
    std::uniform_int_distribution<PEId> pe_dist(0, total_pes - 1);
    
    // 随机重新分配选中的神经元
    for (NeuronId neuron_id : selected_neurons) {
        PEId new_pe = pe_dist(rng_);
        neighbor->moveNeuron(neuron_id, new_pe);
    }
    
    return neighbor;
}

void SimulatedAnnealingOptimizer::updateTemperature(uint32_t iteration) {
    current_temperature_ = calculateTemperature(iteration);
}

double SimulatedAnnealingOptimizer::calculateTemperature(uint32_t iteration) const {
    double progress = static_cast<double>(iteration) / max_iterations_;
    
    switch (cooling_schedule_) {
        case CoolingSchedule::LINEAR:
            return initial_temperature_ * (1.0 - progress);
        
        case CoolingSchedule::EXPONENTIAL: {
            double alpha = std::log(final_temperature_ / initial_temperature_) / max_iterations_;
            return initial_temperature_ * std::exp(alpha * iteration);
        }
        
        case CoolingSchedule::LOGARITHMIC:
            return initial_temperature_ / std::log(2.0 + iteration);
        
        case CoolingSchedule::ADAPTIVE: {
            // 基于接受率调整降温速度
            double target_ratio = target_acceptance_ratio_;
            double current_ratio = (total_moves_ > 0) ? static_cast<double>(accepted_moves_) / total_moves_ : 0.5;
            double adjustment = (current_ratio > target_ratio) ? 0.95 : 1.05;  // 接受率高则降温快，低则降温慢
            
            double base_temp = initial_temperature_ * std::pow(final_temperature_ / initial_temperature_, progress);
            return base_temp * adjustment;
        }
        
        default:
            return initial_temperature_ * std::exp(-3.0 * progress);
    }
}

bool SimulatedAnnealingOptimizer::acceptSolution(double current_cost, double neighbor_cost) const {
    if (neighbor_cost <= current_cost) {
        return true;  // 总是接受更好的解
    }
    
    double probability = calculateAcceptanceProbability(current_cost, neighbor_cost);
    return const_cast<std::uniform_real_distribution<double>&>(uniform_dist_)(const_cast<std::mt19937&>(rng_)) < probability;
}

double SimulatedAnnealingOptimizer::calculateAcceptanceProbability(double current_cost, double neighbor_cost) const {
    if (current_temperature_ <= 0.0 || neighbor_cost <= current_cost) {
        return 1.0;
    }
    
    double delta = neighbor_cost - current_cost;
    return std::exp(-delta / current_temperature_);
}

void SimulatedAnnealingOptimizer::adaptParameters(uint32_t iteration) {
    updateAcceptanceRatio();
    
    // 如果接受率过低，可能需要调整参数
    if (acceptance_ratio_ < 0.1 && cooling_schedule_ != CoolingSchedule::ADAPTIVE) {
        // 适度提高温度
        current_temperature_ *= 1.1;
        LOG_DEBUG("Adjusted temperature up due to low acceptance ratio: " + 
                 std::to_string(acceptance_ratio_));
    }
    
    // 如果接受率过高，可能收敛太慢
    if (acceptance_ratio_ > 0.9) {
        current_temperature_ *= 0.9;
        LOG_DEBUG("Adjusted temperature down due to high acceptance ratio: " + 
                 std::to_string(acceptance_ratio_));
    }
}

void SimulatedAnnealingOptimizer::updateAcceptanceRatio() {
    acceptance_ratio_ = (total_moves_ > 0) ? static_cast<double>(accepted_moves_) / total_moves_ : 0.0;
}

bool SimulatedAnnealingOptimizer::hasConverged(uint32_t iteration, double cost_improvement) const {
    // 如果连续多次迭代没有显著改进
    static uint32_t stagnant_iterations = 0;
    static double last_best_cost = std::numeric_limits<double>::max();
    
    if (std::abs(best_cost_ - last_best_cost) < 1e-6) {
        stagnant_iterations++;
    } else {
        stagnant_iterations = 0;
        last_best_cost = best_cost_;
    }
    
    // 如果停滞超过max_iterations的10%，认为收敛
    return stagnant_iterations > max_iterations_ * 0.1;
}

void SimulatedAnnealingOptimizer::logProgress(uint32_t iteration, double current_cost, double best_cost) const {
    const_cast<SimulatedAnnealingOptimizer*>(this)->updateAcceptanceRatio();
    LOG_INFO("Iteration " + std::to_string(iteration + 1) + 
             " | T=" + std::to_string(current_temperature_) +
             " | Current=" + std::to_string(current_cost) + 
             " | Best=" + std::to_string(best_cost) +
             " | Accept=" + std::to_string(acceptance_ratio_ * 100.0) + "%");
}

void SimulatedAnnealingOptimizer::printFinalStatistics() const {
    LOG_INFO("=== Simulated Annealing Final Statistics ===");
    LOG_INFO("Total iterations: " + std::to_string(iterations_performed_));
    LOG_INFO("Final temperature: " + std::to_string(current_temperature_));
    LOG_INFO("Best cost achieved: " + std::to_string(best_cost_));
    LOG_INFO("Total moves attempted: " + std::to_string(total_moves_));
    LOG_INFO("Moves accepted: " + std::to_string(accepted_moves_));
    LOG_INFO("Final acceptance ratio: " + std::to_string(acceptance_ratio_ * 100.0) + "%");
}


std::vector<NeuronId> SimulatedAnnealingOptimizer::getRandomNeuronSubset(
    const NeuralNetwork& network, 
    size_t subset_size) const {
    
    auto all_neurons = network.getAllNeuronIds();
    if (subset_size >= all_neurons.size()) {
        return all_neurons;
    }
    
    std::vector<NeuronId> subset;
    subset.reserve(subset_size);
    
    // 使用洗牌算法选择随机子集
    std::vector<NeuronId> shuffled = all_neurons;
    std::shuffle(shuffled.begin(), shuffled.end(), const_cast<std::mt19937&>(rng_));
    
    for (size_t i = 0; i < subset_size; ++i) {
        subset.push_back(shuffled[i]);
    }
    
    return subset;
}

PEId SimulatedAnnealingOptimizer::findBestTargetPE(
    NeuronId neuron_id,
    const MappingSolution& solution,
    const NeuralNetwork& network,
    const HardwareTopology& topology) const {

    uint32_t total_pes = topology.getTotalPEs();
    PEId best_pe = 0;
    double best_cost = std::numeric_limits<double>::max();
    
    for (PEId pe_id = 0; pe_id < total_pes; ++pe_id) {
        // 计算移动到此PE的成本
        double cost = 0.0;
        
        // 计算与其他神经元的通信成本
        auto connections = network.getAllConnections();
        for (const auto& conn : connections) {
            if (conn.source_id == neuron_id || conn.target_id == neuron_id) {
                NeuronId other_neuron = (conn.source_id == neuron_id) ? conn.target_id : conn.source_id;
                PEId other_pe = solution.getNeuronPE(other_neuron);
                
                if (other_pe != INVALID_PE_ID) {
                    double distance = topology.getDistance(pe_id, other_pe);
                    cost += std::abs(conn.weight) * distance;
                }
            }
        }
        
        if (cost < best_cost) {
            best_cost = cost;
            best_pe = pe_id;
        }
    }
    
    return best_pe;
}

} // namespace neuron_mapping