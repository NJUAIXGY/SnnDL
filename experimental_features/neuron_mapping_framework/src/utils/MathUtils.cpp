#include "utils/MathUtils.h"
#include "utils/Logger.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>
#include <set>

namespace NeuronMapping {
namespace MathUtils {

// === 统计函数 ===

float mean(const std::vector<float>& values) {
    if (values.empty()) {
        LOG_WARNING("Calculating mean of empty vector");
        return 0.0f;
    }
    
    float sum = std::accumulate(values.begin(), values.end(), 0.0f);
    return sum / values.size();
}

float median(std::vector<float> values) {
    if (values.empty()) {
        LOG_WARNING("Calculating median of empty vector");
        return 0.0f;
    }
    
    std::sort(values.begin(), values.end());
    size_t size = values.size();
    
    if (size % 2 == 0) {
        return (values[size/2 - 1] + values[size/2]) / 2.0f;
    } else {
        return values[size/2];
    }
}

float standardDeviation(const std::vector<float>& values) {
    if (values.size() <= 1) {
        return 0.0f;
    }
    
    float m = mean(values);
    float variance = 0.0f;
    
    for (float value : values) {
        float diff = value - m;
        variance += diff * diff;
    }
    
    variance /= (values.size() - 1);
    return std::sqrt(variance);
}

float variance(const std::vector<float>& values) {
    if (values.size() <= 1) {
        return 0.0f;
    }
    
    float m = mean(values);
    float var = 0.0f;
    
    for (float value : values) {
        float diff = value - m;
        var += diff * diff;
    }
    
    return var / (values.size() - 1);
}

std::pair<float, float> minMax(const std::vector<float>& values) {
    if (values.empty()) {
        LOG_WARNING("Finding min/max of empty vector");
        return {0.0f, 0.0f};
    }
    
    auto result = std::minmax_element(values.begin(), values.end());
    return {*result.first, *result.second};
}

float percentile(std::vector<float> values, float p) {
    if (values.empty()) {
        LOG_WARNING("Calculating percentile of empty vector");
        return 0.0f;
    }
    
    if (p < 0.0f || p > 1.0f) {
        LOG_ERROR("Invalid percentile value: " + std::to_string(p));
        return 0.0f;
    }
    
    std::sort(values.begin(), values.end());
    
    if (p == 0.0f) return values.front();
    if (p == 1.0f) return values.back();
    
    float index = p * (values.size() - 1);
    size_t lower = static_cast<size_t>(std::floor(index));
    size_t upper = static_cast<size_t>(std::ceil(index));
    
    if (lower == upper) {
        return values[lower];
    }
    
    float weight = index - lower;
    return values[lower] * (1.0f - weight) + values[upper] * weight;
}

float covariance(const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size() || x.empty()) {
        LOG_ERROR("Covariance calculation: vectors must have same non-zero size");
        return 0.0f;
    }
    
    float mean_x = mean(x);
    float mean_y = mean(y);
    
    float cov = 0.0f;
    for (size_t i = 0; i < x.size(); ++i) {
        cov += (x[i] - mean_x) * (y[i] - mean_y);
    }
    
    return cov / (x.size() - 1);
}

float correlation(const std::vector<float>& x, const std::vector<float>& y) {
    float cov = covariance(x, y);
    float std_x = standardDeviation(x);
    float std_y = standardDeviation(y);
    
    if (std_x == 0.0f || std_y == 0.0f) {
        LOG_WARNING("Correlation calculation: zero standard deviation");
        return 0.0f;
    }
    
    return cov / (std_x * std_y);
}

// === 向量操作 ===

std::vector<float> normalize(const std::vector<float>& values, float min_val, float max_val) {
    if (values.empty()) {
        return values;
    }
    
    auto minmax = minMax(values);
    float range = minmax.second - minmax.first;
    
    if (range == 0.0f) {
        LOG_WARNING("Normalizing vector with zero range");
        return std::vector<float>(values.size(), (min_val + max_val) / 2.0f);
    }
    
    std::vector<float> normalized;
    normalized.reserve(values.size());
    
    float target_range = max_val - min_val;
    
    for (float value : values) {
        float normalized_value = min_val + ((value - minmax.first) / range) * target_range;
        normalized.push_back(normalized_value);
    }
    
    return normalized;
}

std::vector<float> standardize(const std::vector<float>& values) {
    if (values.empty()) {
        return values;
    }
    
    float m = mean(values);
    float std_dev = standardDeviation(values);
    
    if (std_dev == 0.0f) {
        LOG_WARNING("Standardizing vector with zero standard deviation");
        return std::vector<float>(values.size(), 0.0f);
    }
    
    std::vector<float> standardized;
    standardized.reserve(values.size());
    
    for (float value : values) {
        standardized.push_back((value - m) / std_dev);
    }
    
    return standardized;
}

float dotProduct(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        LOG_ERROR("Dot product: vectors must have same size");
        return 0.0f;
    }
    
    float result = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    
    return result;
}

float euclideanDistance(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        LOG_ERROR("Euclidean distance: vectors must have same size");
        return 0.0f;
    }
    
    float sum_squared = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        float diff = a[i] - b[i];
        sum_squared += diff * diff;
    }
    
    return std::sqrt(sum_squared);
}

float manhattanDistance(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        LOG_ERROR("Manhattan distance: vectors must have same size");
        return 0.0f;
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        sum += std::abs(a[i] - b[i]);
    }
    
    return sum;
}

float cosineSimilarity(const std::vector<float>& a, const std::vector<float>& b) {
    if (a.size() != b.size()) {
        LOG_ERROR("Cosine similarity: vectors must have same size");
        return 0.0f;
    }
    
    float dot = dotProduct(a, b);
    float norm_a = std::sqrt(dotProduct(a, a));
    float norm_b = std::sqrt(dotProduct(b, b));
    
    if (norm_a == 0.0f || norm_b == 0.0f) {
        LOG_WARNING("Cosine similarity: zero vector norm");
        return 0.0f;
    }
    
    return dot / (norm_a * norm_b);
}

// === 数值工具 ===

bool isClose(float a, float b, float tolerance) {
    return std::abs(a - b) <= tolerance;
}

bool isZero(float value, float tolerance) {
    return std::abs(value) <= tolerance;
}

float clamp(float value, float min_val, float max_val) {
    return std::max(min_val, std::min(value, max_val));
}

int sign(float value) {
    if (value > 0.0f) return 1;
    if (value < 0.0f) return -1;
    return 0;
}

float lerp(float a, float b, float t) {
    return a + t * (b - a);
}

float smoothstep(float edge0, float edge1, float x) {
    float t = clamp((x - edge0) / (edge1 - edge0), 0.0f, 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

float sigmoid(float x) {
    return 1.0f / (1.0f + std::exp(-x));
}

float relu(float x) {
    return std::max(0.0f, x);
}

float tanh_activation(float x) {
    return std::tanh(x);
}

// === 随机数生成 ===

namespace Random {
    static thread_local std::mt19937 generator(std::random_device{}());
    static thread_local bool seeded = false;
}

void setSeed(uint32_t seed) {
    Random::generator.seed(seed);
    Random::seeded = true;
}

float uniform(float min_val, float max_val) {
    std::uniform_real_distribution<float> dist(min_val, max_val);
    return dist(Random::generator);
}

int uniformInt(int min_val, int max_val) {
    std::uniform_int_distribution<int> dist(min_val, max_val);
    return dist(Random::generator);
}

float normal(float mean, float std_dev) {
    std::normal_distribution<float> dist(mean, std_dev);
    return dist(Random::generator);
}

float exponential(float lambda) {
    std::exponential_distribution<float> dist(lambda);
    return dist(Random::generator);
}

std::vector<float> uniformVector(size_t size, float min_val, float max_val) {
    std::vector<float> result;
    result.reserve(size);
    
    for (size_t i = 0; i < size; ++i) {
        result.push_back(uniform(min_val, max_val));
    }
    
    return result;
}

std::vector<float> normalVector(size_t size, float mean, float std_dev) {
    std::vector<float> result;
    result.reserve(size);
    
    for (size_t i = 0; i < size; ++i) {
        result.push_back(normal(mean, std_dev));
    }
    
    return result;
}

std::vector<int> randomPermutation(int n) {
    std::vector<int> result(n);
    std::iota(result.begin(), result.end(), 0);
    std::shuffle(result.begin(), result.end(), Random::generator);
    return result;
}

std::vector<int> randomSample(int population_size, int sample_size) {
    if (sample_size > population_size) {
        LOG_ERROR("Sample size cannot exceed population size");
        return std::vector<int>();
    }
    
    std::vector<int> population(population_size);
    std::iota(population.begin(), population.end(), 0);
    
    std::shuffle(population.begin(), population.end(), Random::generator);
    
    std::vector<int> sample(population.begin(), population.begin() + sample_size);
    return sample;
}

// === 优化工具 ===

std::pair<float, size_t> findMinimum(const std::vector<float>& values) {
    if (values.empty()) {
        LOG_WARNING("Finding minimum of empty vector");
        return {0.0f, 0};
    }
    
    auto it = std::min_element(values.begin(), values.end());
    return {*it, static_cast<size_t>(std::distance(values.begin(), it))};
}

std::pair<float, size_t> findMaximum(const std::vector<float>& values) {
    if (values.empty()) {
        LOG_WARNING("Finding maximum of empty vector");
        return {0.0f, 0};
    }
    
    auto it = std::max_element(values.begin(), values.end());
    return {*it, static_cast<size_t>(std::distance(values.begin(), it))};
}

std::vector<size_t> argsort(const std::vector<float>& values, bool ascending) {
    std::vector<size_t> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);
    
    if (ascending) {
        std::sort(indices.begin(), indices.end(),
                  [&values](size_t i, size_t j) { return values[i] < values[j]; });
    } else {
        std::sort(indices.begin(), indices.end(),
                  [&values](size_t i, size_t j) { return values[i] > values[j]; });
    }
    
    return indices;
}

std::vector<size_t> topK(const std::vector<float>& values, size_t k, bool largest) {
    if (k > values.size()) {
        k = values.size();
    }
    
    auto indices = argsort(values, !largest);
    indices.resize(k);
    return indices;
}

float weightedSum(const std::vector<float>& values, const std::vector<float>& weights) {
    if (values.size() != weights.size()) {
        LOG_ERROR("Weighted sum: values and weights must have same size");
        return 0.0f;
    }
    
    float sum = 0.0f;
    for (size_t i = 0; i < values.size(); ++i) {
        sum += values[i] * weights[i];
    }
    
    return sum;
}

float weightedAverage(const std::vector<float>& values, const std::vector<float>& weights) {
    if (values.size() != weights.size()) {
        LOG_ERROR("Weighted average: values and weights must have same size");
        return 0.0f;
    }
    
    float weighted_sum = weightedSum(values, weights);
    float weight_sum = std::accumulate(weights.begin(), weights.end(), 0.0f);
    
    if (weight_sum == 0.0f) {
        LOG_WARNING("Weighted average: zero weight sum");
        return 0.0f;
    }
    
    return weighted_sum / weight_sum;
}

// === 集合操作 ===

std::vector<int> setIntersection(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> sorted_a = a;
    std::vector<int> sorted_b = b;
    
    std::sort(sorted_a.begin(), sorted_a.end());
    std::sort(sorted_b.begin(), sorted_b.end());
    
    std::vector<int> result;
    std::set_intersection(sorted_a.begin(), sorted_a.end(),
                          sorted_b.begin(), sorted_b.end(),
                          std::back_inserter(result));
    
    return result;
}

std::vector<int> setUnion(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> sorted_a = a;
    std::vector<int> sorted_b = b;
    
    std::sort(sorted_a.begin(), sorted_a.end());
    std::sort(sorted_b.begin(), sorted_b.end());
    
    std::vector<int> result;
    std::set_union(sorted_a.begin(), sorted_a.end(),
                   sorted_b.begin(), sorted_b.end(),
                   std::back_inserter(result));
    
    return result;
}

std::vector<int> setDifference(const std::vector<int>& a, const std::vector<int>& b) {
    std::vector<int> sorted_a = a;
    std::vector<int> sorted_b = b;
    
    std::sort(sorted_a.begin(), sorted_a.end());
    std::sort(sorted_b.begin(), sorted_b.end());
    
    std::vector<int> result;
    std::set_difference(sorted_a.begin(), sorted_a.end(),
                        sorted_b.begin(), sorted_b.end(),
                        std::back_inserter(result));
    
    return result;
}

bool isSubset(const std::vector<int>& subset, const std::vector<int>& superset) {
    std::vector<int> sorted_subset = subset;
    std::vector<int> sorted_superset = superset;
    
    std::sort(sorted_subset.begin(), sorted_subset.end());
    std::sort(sorted_superset.begin(), sorted_superset.end());
    
    return std::includes(sorted_superset.begin(), sorted_superset.end(),
                         sorted_subset.begin(), sorted_subset.end());
}

float jaccardSimilarity(const std::vector<int>& a, const std::vector<int>& b) {
    auto intersection = setIntersection(a, b);
    auto union_set = setUnion(a, b);
    
    if (union_set.empty()) {
        return (a.empty() && b.empty()) ? 1.0f : 0.0f;
    }
    
    return static_cast<float>(intersection.size()) / union_set.size();
}

} // namespace MathUtils
} // namespace NeuronMapping