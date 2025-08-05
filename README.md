# SnnDL - Spiking Neural Network Deep Learning Framework for SST

[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![SST Version](https://img.shields.io/badge/SST-15.0.0+-orange.svg)](https://sst-simulator.org)

SnnDL是一个基于SST (Structural Simulation Toolkit) 框架的高性能脉冲神经网络(SNN)仿真库，专为大规模分布式神经形态计算设计。

## 🚀 特性

### 核心功能
- **高性能SNN仿真**: 基于SST框架的并行仿真引擎
- **分布式网络通信**: Miranda风格的SubComponent接口架构
- **网络感知路由**: 扩展的SpikeEvent支持跨节点神经元通信
- **插件化网络接口**: 可配置的网络适配器(SnnNIC)
- **多格式数据支持**: TEXT、NMNIST-AER、SHD-HDF5等数据格式

### 架构优势
- **模块化设计**: 组件化架构，易于扩展和定制
- **高可扩展性**: 支持从单核到大规模集群的部署
- **标准化接口**: 符合SST Element开发规范
- **统计监控**: 内置性能统计和网络分析工具

## 📋 系统要求

### 必需依赖
- **SST Core**: >= 15.0.0
- **SST Elements**: >= 15.0.0
- **编译器**: GCC >= 7.0 或 Clang >= 5.0 (C++17支持)
- **操作系统**: Linux (推荐Ubuntu 20.04+)

### 可选依赖
- **MPI**: 用于分布式仿真
- **HDF5**: 用于HDF5格式数据支持
- **Python**: >= 3.6 (用于配置脚本)

## 🛠️ 安装指南

### 1. 安装SST框架

```bash
# 下载SST Core和Elements
wget https://github.com/sstsimulator/sst-core/releases/download/v15.0.0/sst-core-15.0.0.tar.gz
wget https://github.com/sstsimulator/sst-elements/releases/download/v15.0.0/sst-elements-15.0.0.tar.gz

# 编译安装SST Core
tar -xzf sst-core-15.0.0.tar.gz
cd sst-core-15.0.0
./configure --prefix=/path/to/sst-install
make -j$(nproc)
make install

# 编译安装SST Elements
cd ../
tar -xzf sst-elements-15.0.0.tar.gz
cd sst-elements-15.0.0
./configure --prefix=/path/to/sst-install --with-sst-core=/path/to/sst-install
make -j$(nproc)
make install
```

### 2. 编译SnnDL

```bash
# 克隆仓库
git clone https://github.com/YOUR_USERNAME/SnnDL.git
cd SnnDL

# 将SnnDL目录复制到SST Elements源码中
cp -r src/sst/elements/SnnDL /path/to/sst-elements-15.0.0/src/sst/elements/

# 重新编译SST Elements
cd /path/to/sst-elements-15.0.0
make -j$(nproc)
make install
```

### 3. 验证安装

```bash
# 检查SnnDL组件是否注册成功
sst-info SnnDL

# 运行示例测试
cd SnnDL/tests
sst test_minimal_snn.py --stop-at=5ms
```

## 📖 快速开始

### 基本示例

```python
#!/usr/bin/env python3
import sst

# 创建脉冲源
spike_source = sst.Component("SpikeSource", "SnnDL.SpikeSource")
spike_source.addParams({
    "dataset_path": "spike_data.txt",
    "dataset_format": "TEXT",
    "time_scale": 1.0,
    "max_events": 1000
})

# 创建SNN处理元素
snn_pe = sst.Component("SnnPE", "SnnDL.SnnPE")
snn_pe.addParams({
    "num_neurons": 100,
    "v_thresh": 1.0,
    "v_reset": 0.0,
    "tau_mem": 20.0,
    "clock": "1GHz"
})

# 连接组件
link = sst.Link("spike_link")
link.connect((spike_source, "spike_output", "1ns"), 
             (snn_pe, "spike_input", "1ns"))
```

### 数据格式

**TEXT格式** (spike_data.txt):
```
# 神经元ID 时间戳(微秒)
0 1000
1 1500
2 2000
```

## 🏗️ 架构设计

### 组件层次结构

```
SnnDL
├── 核心组件
│   ├── SnnPE          # 脉冲神经网络处理元素
│   └── SpikeSource    # 脉冲数据源
├── 网络接口
│   ├── SnnInterface   # 抽象网络接口基类
│   └── SnnNIC         # SimpleNetwork适配器
└── 事件系统
    └── SpikeEvent     # 网络感知的脉冲事件
```

### 扩展的SpikeEvent

```cpp
class SpikeEvent : public SST::Event {
public:
    uint32_t neuron_id;      // 源神经元ID
    uint32_t dest_node;      // 目标节点ID
    uint32_t dest_neuron;    // 目标神经元ID
    uint32_t spike_time;     // 脉冲时间戳
    float weight;            // 突触权重
    double timestamp;        // 仿真时间戳
};
```

## 🧪 测试和验证

### 单元测试

```bash
cd tests/
sst test_minimal_snn.py --stop-at=5ms      # 基础功能测试
sst test_network_spikes.py --stop-at=5ms   # 网络通信测试
```

### 性能基准测试

```bash
cd examples/
sst benchmark_large_network.py --stop-at=100ms
```

## 📊 性能特征

### 基准测试结果

| 配置 | 神经元数量 | 仿真时间 | 吞吐量 |
|------|------------|----------|--------|
| 单核 | 1,000 | 10ms | ~100K spikes/s |
| 4核 | 10,000 | 100ms | ~1M spikes/s |
| 集群 | 100,000 | 1s | ~10M spikes/s |

### 内存使用

- 基础SpikeEvent: ~48字节
- SnnPE状态: ~8字节/神经元
- 网络缓冲: 可配置(默认1KiB)

## 🤝 贡献指南

### 开发环境设置

```bash
# Fork并克隆仓库
git clone https://github.com/YOUR_USERNAME/SnnDL.git
cd SnnDL

# 创建开发分支
git checkout -b feature/your-feature-name

# 进行开发...

# 运行测试
make test

# 提交更改
git add .
git commit -m "Add: your feature description"
git push origin feature/your-feature-name
```

### 代码规范

- 遵循SST编码标准
- 使用C++17特性
- 添加单元测试
- 更新文档

## 📄 许可证

本项目基于MIT许可证开源 - 详见 [LICENSE](LICENSE) 文件。

## 🙏 致谢

- [SST团队](https://sst-simulator.org) 提供的优秀仿真框架
- 脉冲神经网络研究社区的宝贵贡献
- 所有测试和反馈的贡献者

## 📞 联系方式

- **Issues**: [GitHub Issues](https://github.com/YOUR_USERNAME/SnnDL/issues)
- **讨论**: [GitHub Discussions](https://github.com/YOUR_USERNAME/SnnDL/discussions)
- **邮件**: your-email@domain.com

---

**SnnDL** - 让脉冲神经网络仿真更高效、更可扩展！
