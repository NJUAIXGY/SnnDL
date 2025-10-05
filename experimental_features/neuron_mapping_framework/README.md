# 神经元映射框架 (Neuron Mapping Framework)

一个高效的神经网络到硬件映射优化框架，专为脉冲神经网络设计，支持多种映射策略和优化算法。

## 🌟 主要特性

- **多种映射策略**：支持随机映射、贪心映射、图分割映射等
- **高级优化算法**：模拟退火、局部搜索、爬山算法
- **网络分析工具**：社区检测、网络拓扑分析
- **性能评估**：通信成本、负载均衡、内存使用评估
- **硬件拓扑支持**：网格、环状等多种硬件拓扑结构
- **灵活配置**：支持多种参数配置和优化目标

## 🚀 快速开始

### 编译项目
```bash
# 编译所有组件
make all

# 测试编译（检查语法错误）
make test-compile
```

### 运行演示
```bash
# 运行工作演示程序
make run-demo

# 运行综合测试
make run-test

# 运行社区检测测试
make run-community-test
```

## 📁 项目结构

```
neuron_mapping_framework/
├── include/                # 头文件
│   ├── core/              # 核心数据结构
│   ├── algorithms/        # 算法实现
│   ├── strategies/        # 映射策略
│   ├── optimizers/        # 优化器
│   ├── evaluators/        # 评估器
│   └── utils/             # 工具类
├── src/                   # 源代码实现
├── examples/              # 示例程序
├── tests/                 # 测试程序
└── Makefile              # 编译配置
```

## 🔧 核心组件

### 映射策略
- **随机映射**：快速生成初始映射解
- **贪心映射**：基于局部最优的映射策略
- **图分割映射**：多层次、谱聚类、递归二分等高级算法

### 优化算法
- **模拟退火**：支持多种冷却策略和邻域策略
- **局部搜索**：随机交换、贪心移动等邻域操作
- **爬山算法**：简单高效的局部优化

### 网络分析
- **社区检测**：Louvain、标签传播、模块度最大化等
- **网络统计**：度分布、聚类系数、路径长度等
- **拓扑分析**：小世界、无标度网络检测

## 📊 使用示例

```cpp
#include "strategies/GraphPartitioningStrategy.h"
#include "optimizers/SimulatedAnnealingOptimizer.h"

// 创建神经网络
auto network = std::make_unique<NeuralNetwork>();
// ... 添加神经元和连接

// 创建硬件拓扑
auto topology = std::make_unique<HardwareTopology>();
topology->createMesh2D(4, 4, pe_config);

// 图分割映射
GraphPartitioningStrategy strategy(
    GraphPartitioningStrategy::PartitioningAlgorithm::MULTILEVEL);
auto solution = strategy.mapNetwork(*network, *topology, config);

// 模拟退火优化
SimulatedAnnealingOptimizer optimizer(50.0, 0.01, 1000);
auto optimized = optimizer.optimize(
    std::move(solution), *network, *topology, opt_config);
```

## 🎯 性能指标

框架支持多种性能评估指标：
- **通信成本**：神经元间通信开销
- **负载均衡**：处理器负载分布均匀程度
- **内存使用**：内存资源利用率
- **处理器利用率**：硬件资源使用效率

## 🧪 测试覆盖

- ✅ 基础数据结构测试
- ✅ 映射策略算法验证
- ✅ 优化器性能测试
- ✅ 网络分析功能测试
- ✅ 错误处理和边界条件
- ✅ 性能基准测试

## 📖 技术文档

详细技术文档请参考 [技术文档](./docs/technical_documentation.md)

## 🛠️ 开发环境

- **编译器**：g++ 支持 C++17
- **构建系统**：Make
- **依赖**：STL 标准库，无外部依赖

## 📈 版本历史

- **v1.0.0** - 完成核心功能实现
  - 基础映射策略和优化算法
  - 图分割和社区检测
  - 性能评估和测试框架

## 🤝 贡献指南

1. 保持代码风格一致
2. 添加适当的测试用例
3. 更新相关文档
4. 确保编译无错误和警告

## 📄 许可证

本项目采用 MIT 许可证。

## 🔗 相关链接

- [SST 仿真框架](https://github.com/sstsimulator/sst-core)
- [脉冲神经网络资源](http://snntorch.com/)

---

**开发团队**：神经计算实验室  
**最后更新**：2025年8月24日