#ifndef NEURON_MAPPING_MATH_UTILS_H
#define NEURON_MAPPING_MATH_UTILS_H

#include "../core/Types.h"
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>

namespace NeuronMapping {
namespace MathUtils {

// === 统计函数 ===

/**
 * @brief 计算向量的平均值
 * @param values 数值向量
 * @return 平均值
 */
template<typename T>
double mean(const std::vector<T>& values) {
    if (values.empty()) return 0.0;
    return std::accumulate(values.begin(), values.end(), 0.0) / values.size();
}

/**
 * @brief 计算向量的方差
 * @param values 数值向量
 * @return 方差
 */
template<typename T>
double variance(const std::vector<T>& values) {
    if (values.size() < 2) return 0.0;
    double mean_val = mean(values);
    double sum_sq_diff = 0.0;
    for (const auto& val : values) {
        double diff = val - mean_val;
        sum_sq_diff += diff * diff;
    }
    return sum_sq_diff / (values.size() - 1);
}

/**
 * @brief 计算向量的标准差
 * @param values 数值向量
 * @return 标准差
 */
template<typename T>
double standardDeviation(const std::vector<T>& values) {
    return std::sqrt(variance(values));
}

/**
 * @brief 计算向量的变异系数
 * @param values 数值向量
 * @return 变异系数（标准差/平均值）
 */
template<typename T>
double coefficientOfVariation(const std::vector<T>& values) {
    double mean_val = mean(values);
    if (std::abs(mean_val) < 1e-10) return 0.0;
    return standardDeviation(values) / mean_val;
}

/**
 * @brief 找到向量的最小值和最大值
 * @param values 数值向量
 * @return {最小值, 最大值}
 */
template<typename T>
std::pair<T, T> minMax(const std::vector<T>& values) {
    if (values.empty()) return {T{}, T{}};
    auto [min_it, max_it] = std::minmax_element(values.begin(), values.end());
    return {*min_it, *max_it};
}

/**
 * @brief 计算百分位数
 * @param values 数值向量（会被排序）
 * @param percentile 百分位（0-100）
 * @return 百分位数值
 */
template<typename T>
double percentile(std::vector<T> values, double percentile) {
    if (values.empty()) return 0.0;
    
    std::sort(values.begin(), values.end());
    double index = percentile / 100.0 * (values.size() - 1);
    
    if (index <= 0) return static_cast<double>(values[0]);
    if (index >= values.size() - 1) return static_cast<double>(values.back());
    
    size_t lower = static_cast<size_t>(std::floor(index));
    size_t upper = static_cast<size_t>(std::ceil(index));
    double weight = index - lower;
    
    return (1.0 - weight) * values[lower] + weight * values[upper];
}

// === 距离函数 ===

/**
 * @brief 计算欧几里得距离
 * @param p1 点1坐标
 * @param p2 点2坐标
 * @return 欧几里得距离
 */
double euclideanDistance(const std::vector<double>& p1, const std::vector<double>& p2);

/**
 * @brief 计算曼哈顿距离
 * @param p1 点1坐标
 * @param p2 点2坐标
 * @return 曼哈顿距离
 */
double manhattanDistance(const std::vector<double>& p1, const std::vector<double>& p2);

/**
 * @brief 计算2D网格中两点的曼哈顿距离
 * @param x1, y1 点1坐标
 * @param x2, y2 点2坐标
 * @return 曼哈顿距离
 */
int32_t manhattanDistance2D(int32_t x1, int32_t y1, int32_t x2, int32_t y2);

/**
 * @brief 计算3D网格中两点的曼哈顿距离
 * @param x1, y1, z1 点1坐标
 * @param x2, y2, z2 点2坐标
 * @return 曼哈顿距离
 */
int32_t manhattanDistance3D(int32_t x1, int32_t y1, int32_t z1,
                           int32_t x2, int32_t y2, int32_t z2);

// === 随机数生成 ===

/**
 * @brief 线程安全的随机数生成器
 */
class RandomGenerator {
public:
    /**
     * @brief 构造函数
     * @param seed 随机种子，0表示使用随机种子
     */
    explicit RandomGenerator(uint32_t seed = 0);
    
    /**
     * @brief 设置随机种子
     * @param seed 随机种子
     */
    void setSeed(uint32_t seed);
    
    /**
     * @brief 生成[0, 1)范围内的随机浮点数
     * @return 随机浮点数
     */
    double uniform();
    
    /**
     * @brief 生成[min, max)范围内的随机浮点数
     * @param min 最小值
     * @param max 最大值
     * @return 随机浮点数
     */
    double uniform(double min, double max);
    
    /**
     * @brief 生成[min, max]范围内的随机整数
     * @param min 最小值
     * @param max 最大值
     * @return 随机整数
     */
    int32_t uniformInt(int32_t min, int32_t max);
    
    /**
     * @brief 生成高斯分布的随机数
     * @param mean 均值
     * @param stddev 标准差
     * @return 高斯分布随机数
     */
    double gaussian(double mean = 0.0, double stddev = 1.0);
    
    /**
     * @brief 生成指数分布的随机数
     * @param lambda 参数
     * @return 指数分布随机数
     */
    double exponential(double lambda = 1.0);
    
    /**
     * @brief 从集合中随机选择元素
     * @param container 容器
     * @return 随机选择的元素
     */
    template<typename Container>
    typename Container::value_type choice(const Container& container) {
        if (container.empty()) return typename Container::value_type{};
        auto it = container.begin();
        std::advance(it, uniformInt(0, static_cast<int32_t>(container.size()) - 1));
        return *it;
    }
    
    /**
     * @brief 随机打乱容器
     * @param container 容器
     */
    template<typename Container>
    void shuffle(Container& container) {
        std::shuffle(container.begin(), container.end(), generator_);
    }
    
    /**
     * @brief 生成随机排列
     * @param n 元素数量
     * @return 随机排列的索引向量
     */
    std::vector<size_t> permutation(size_t n);
    
    /**
     * @brief 按概率选择（轮盘赌选择）
     * @param probabilities 概率向量
     * @return 选中的索引
     */
    size_t rouletteWheel(const std::vector<double>& probabilities);

private:
    std::mt19937 generator_;
    std::uniform_real_distribution<double> uniform_dist_;
    std::normal_distribution<double> normal_dist_;
    std::exponential_distribution<double> exp_dist_;
};

// === 矩阵运算 ===

/**
 * @brief 稀疏矩阵类（CSR格式）
 */
class SparseMatrix {
public:
    SparseMatrix() = default;
    SparseMatrix(size_t rows, size_t cols);
    
    /**
     * @brief 设置矩阵元素
     * @param row 行索引
     * @param col 列索引
     * @param value 值
     */
    void set(size_t row, size_t col, double value);
    
    /**
     * @brief 获取矩阵元素
     * @param row 行索引
     * @param col 列索引
     * @return 元素值
     */
    double get(size_t row, size_t col) const;
    
    /**
     * @brief 矩阵向量乘法
     * @param vec 输入向量
     * @return 结果向量
     */
    std::vector<double> multiply(const std::vector<double>& vec) const;
    
    /**
     * @brief 获取行数
     * @return 行数
     */
    size_t rows() const { return num_rows_; }
    
    /**
     * @brief 获取列数
     * @return 列数
     */
    size_t cols() const { return num_cols_; }
    
    /**
     * @brief 获取非零元素数量
     * @return 非零元素数
     */
    size_t nonZeros() const { return values_.size(); }
    
    /**
     * @brief 清空矩阵
     */
    void clear();
    
    /**
     * @brief 压缩存储（删除零元素）
     */
    void compress();

private:
    size_t num_rows_ = 0;
    size_t num_cols_ = 0;
    std::vector<double> values_;        // 非零值
    std::vector<size_t> col_indices_;   // 列索引
    std::vector<size_t> row_pointers_;  // 行指针
    
    void finalize();
    bool is_finalized_ = false;
};

// === 优化算法辅助函数 ===

/**
 * @brief 模拟退火接受概率
 * @param old_cost 旧成本
 * @param new_cost 新成本
 * @param temperature 当前温度
 * @return 接受概率
 */
double simulatedAnnealingAcceptanceProbability(double old_cost, double new_cost, double temperature);

/**
 * @brief 线性冷却调度
 * @param initial_temp 初始温度
 * @param final_temp 最终温度
 * @param current_iteration 当前迭代
 * @param max_iterations 最大迭代数
 * @return 当前温度
 */
double linearCoolingSchedule(double initial_temp, double final_temp,
                           uint32_t current_iteration, uint32_t max_iterations);

/**
 * @brief 指数冷却调度
 * @param initial_temp 初始温度
 * @param cooling_rate 冷却率（0-1）
 * @param current_iteration 当前迭代
 * @return 当前温度
 */
double exponentialCoolingSchedule(double initial_temp, double cooling_rate, uint32_t current_iteration);

/**
 * @brief 对数冷却调度
 * @param initial_temp 初始温度
 * @param current_iteration 当前迭代
 * @return 当前温度
 */
double logarithmicCoolingSchedule(double initial_temp, uint32_t current_iteration);

// === 图算法辅助函数 ===

/**
 * @brief 计算图的聚类系数
 * @param adjacency_matrix 邻接矩阵
 * @return 平均聚类系数
 */
double calculateClusteringCoefficient(const std::vector<std::vector<bool>>& adjacency_matrix);

/**
 * @brief 计算两个集合的Jaccard相似度
 * @param set1 集合1
 * @param set2 集合2
 * @return Jaccard相似度
 */
template<typename T>
double jaccardSimilarity(const std::vector<T>& set1, const std::vector<T>& set2) {
    std::vector<T> intersection;
    std::vector<T> union_set;
    
    std::set_intersection(set1.begin(), set1.end(), set2.begin(), set2.end(),
                         std::back_inserter(intersection));
    std::set_union(set1.begin(), set1.end(), set2.begin(), set2.end(),
                  std::back_inserter(union_set));
    
    if (union_set.empty()) return 1.0; // 两个空集合认为完全相似
    return static_cast<double>(intersection.size()) / union_set.size();
}

// === 数值稳定性函数 ===

/**
 * @brief 安全的除法操作
 * @param numerator 分子
 * @param denominator 分母
 * @param default_value 分母为0时的默认值
 * @return 除法结果
 */
double safeDivide(double numerator, double denominator, double default_value = 0.0);

/**
 * @brief 检查浮点数是否接近
 * @param a 数值a
 * @param b 数值b
 * @param tolerance 容差
 * @return 是否接近
 */
bool isClose(double a, double b, double tolerance = 1e-9);

/**
 * @brief 将值限制在指定范围内
 * @param value 输入值
 * @param min_val 最小值
 * @param max_val 最大值
 * @return 限制后的值
 */
template<typename T>
T clamp(T value, T min_val, T max_val) {
    return std::max(min_val, std::min(max_val, value));
}

/**
 * @brief 线性插值
 * @param a 起始值
 * @param b 结束值
 * @param t 插值参数（0-1）
 * @return 插值结果
 */
double lerp(double a, double b, double t);

} // namespace MathUtils
} // namespace NeuronMapping

#endif // NEURON_MAPPING_MATH_UTILS_H