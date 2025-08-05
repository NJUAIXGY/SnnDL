# SnnDL - Spiking Neural Network Deep Learning 元素库

## 概述

SnnDL是一个用于SST（Structural Simulation Toolkit）的元素库，专门设计用于模拟脉冲神经网络（Spiking Neural Networks, SNN）。该库实现了基于LIF（Leaky Integrate-and-Fire）模型的单核处理单元。

## 核心组件

### 1. SnnPE - 脉冲神经网络处理单元

**功能特性：**
- 基于LIF神经元模型的单核处理单元  
- 事件驱动和时钟驱动的混合处理模式
- CSR（压缩稀疏行）格式的突触权重存储
- 完整的神经元状态管理

**配置参数：**
- `clock`: PE时钟频率（默认: "1GHz"）
- `num_neurons`: 神经元总数（必需）
- `v_thresh`: 发放阈值（默认: 1.0）
- `tau_mem`: 膜时间常数ms（默认: 20.0）
- `weights_file`: 权重文件路径
- `verbose`: 日志级别（默认: 0）

**端口：**
- `spike_input`: 接收输入脉冲事件
- `spike_output`: 发送输出脉冲事件

### 2. SpikeSource - 脉冲数据源

**功能特性：**
- 多格式数据加载器（TEXT/N-MNIST/SHD）
- 精确时序脉冲生成
- 事件驱动数据流

**配置参数：**
- `dataset_path`: 数据集文件路径（必需）
- `dataset_format`: 数据格式（默认: "TEXT"）
- `time_scale`: 时间缩放因子（默认: 1.0）
- `verbose`: 日志级别（默认: 0）

**端口：**
- `spike_output`: 发送脉冲事件

### 3. SpikeEvent - 脉冲事件

轻量级事件结构，用于组件间脉冲传递，支持序列化。

## 数据格式

**权重文件格式：**
```
# input_neuron output_neuron weight
0 1 0.5
1 2 0.3
```

**TEXT格式脉冲数据：**
```
# neuron_id timestamp(microseconds) 
0 1000
1 2000
```

## 编译要求

- SST框架 v15.0.0+
- C++14编译器
- Autotools构建系统

## 使用示例

```python
import sst

# 创建脉冲源
source = sst.Component("source", "SnnDL.SpikeSource")
source.addParams({
    "dataset_path": "spikes.txt",
    "dataset_format": "TEXT"
})

# 创建处理单元
pe = sst.Component("pe", "SnnDL.SnnPE") 
pe.addParams({
    "num_neurons": 10,
    "weights_file": "weights.txt"
})

# 连接组件
link = sst.Link("spike_link")
link.connect((source, "spike_output", "1ns"),
             (pe, "spike_input", "1ns"))
```

## 版权信息

Copyright 2025 SST Contributors
基于SST框架开源许可证发布
- `spike_output`: 发送输出脉冲事件

### 2. SpikeSource - 脉冲数据源

**功能特性：**
- 支持多种神经形态数据集格式
- 精确的时序控制和事件调度
- 可配置的时间缩放和神经元ID映射
- 事件数量限制支持

**配置参数：**
- `dataset_path`: 数据集文件路径（必需）
- `dataset_format`: 数据格式 ("TEXT"|"NMNIST_AER"|"SHD_HDF5")
- `time_scale`: 时间缩放因子（默认: 1.0）
- `neuron_offset`: 神经元ID偏移（默认: 0）
- `max_events`: 最大事件数限制（默认: 0，无限制）
- `verbose`: 日志详细级别（默认: 0）

**端口：**
- `spike_output`: 发送脉冲事件的输出端口

### 3. SpikeEvent - 脉冲事件类

**功能特性：**
- 轻量级的脉冲事件表示
- 支持SST并行仿真的序列化
- 包含神经元ID和时间戳信息

## 数据格式

### TEXT格式
简单的文本格式，每行包含：
```
neuron_id timestamp
```

示例：
```
0 1000
1 1500
0 2000
2 2200
```

### N-MNIST AER格式
地址事件表示格式，适用于N-MNIST数据集。
格式：x, y, timestamp, polarity

### SHD HDF5格式
Spiking Heidelberg Digits数据集的HDF5格式（需要HDF5库支持）。

## 权重文件格式

突触权重文件采用简单的文本格式：
```
pre_neuron_id post_neuron_id weight
```

示例：
```
0 1 0.5
0 2 0.3
1 2 0.8
1 3 0.2
```

## 使用示例

### Python配置脚本

```python
import sst

# 创建脉冲数据源
spike_source = sst.Component("spike_source", "SnnDL.SpikeSource")
spike_source.addParams({
    "dataset_path": "/path/to/dataset.txt",
    "dataset_format": "TEXT",
    "time_scale": "1.0",
    "verbose": "1"
})

# 创建SNN处理单元
snn_pe = sst.Component("snn_pe", "SnnDL.SnnPE")
snn_pe.addParams({
    "clock": "1GHz",
    "num_neurons": "100",
    "v_thresh": "1.0",
    "v_reset": "0.0",
    "v_rest": "0.0",
    "tau_mem": "20.0",
    "t_ref": "2",
    "weights_file": "/path/to/weights.txt",
    "verbose": "1"
})

# 连接组件
link = sst.Link("spike_link")
link.connect(
    (spike_source, "spike_output", "100ps"),
    (snn_pe, "spike_input", "100ps")
)

# 配置统计输出
sst.setStatisticOutput("sst.statOutputCSV", {"filepath": "snn_stats.csv"})
sst.enableAllStatisticsForComponentType("SnnDL.SnnPE")
sst.enableAllStatisticsForComponentType("SnnDL.SpikeSource")

# 运行仿真
sst.setProgramOption("stop-at", "1ms")
```

## 编译和安装

1. 确保SST核心和SST元素库已正确安装
2. 在SnnDL目录中运行：
```bash
make
make install
```

3. 或者在sst-elements根目录重新构建：
```bash
./configure --prefix=/path/to/sst/install
make
make install
```

## 扩展和开发

### 添加新的神经元模型
1. 修改`NeuronState`结构体添加新的状态变量
2. 在`SnnPE::clockTick()`中实现新的动力学方程
3. 更新相关的配置参数

### 添加新的数据格式
1. 在`SpikeSource`中添加新的加载函数
2. 更新`dataset_format`参数的文档
3. 实现相应的解析逻辑

### 性能优化
- 考虑使用SIMD指令优化神经元更新
- 实现更高效的稀疏矩阵格式
- 添加GPU加速支持

## 许可证

该项目遵循SST的许可证条款。

## 贡献

欢迎提交问题报告和改进建议。在提交代码之前，请确保：
1. 代码符合SST的编码规范
2. 添加适当的测试用例
3. 更新相关文档

## 联系方式

如有问题或建议，请通过SST社区渠道联系。
