# SnnDL - Spiking Neural Network Simulation Library for SST

## 1. Overview

SnnDL is a component library for the Structural Simulation Toolkit (SST), designed for simulating large-scale Spiking Neural Networks (SNNs), particularly those implemented on Network-on-Chip (NoC) architectures.

It provides the fundamental building blocks for SNN simulations, including processing elements (PEs), network interfaces (NICs), and spike data sources. This library enables research into SNN models, learning algorithms, and the underlying hardware architectures.

## 2. Core Components

The library consists of three main components:

### a. `SnnPE` (Spiking Neural Network Processing Element)

The `SnnPE` is the core computational unit, simulating a cluster of neurons.

- **Functionality**:
    - Implements the Leaky Integrate-and-Fire (LIF) neuron model.
    - Operates in a hybrid event-driven and clock-driven mode.
    - Stores synaptic weights in a Compressed Sparse Row (CSR) format for efficiency.
    - Manages the complete state of each neuron (membrane potential, refractory period, etc.).
- **Configuration Parameters**:
    - `clock`: Clock frequency of the PE (e.g., "1GHz").
    - `num_neurons`: **(Required)** Total number of neurons in this PE.
    - `weights_file`: **(Required)** Path to the synaptic weight file.
    - `v_thresh`: Firing threshold voltage (Default: `1.0`).
    - `tau_mem`: Membrane time constant in ms (Default: `20.0`).
    - `verbose`: Verbosity level for logging (Default: `0`).
- **Ports**:
    - `spike_input`: Receives incoming `SpikeEvent`s from the `SnnNIC`.
    - `spike_output`: Sends outgoing `SpikeEvent`s to the `SnnNIC`.

### b. `SnnNIC` (Spiking Neural Network Network Interface Card)

The `SnnNIC` acts as the bridge between a `SnnPE` and the simulated network (e.g., a mesh or torus built with `merlin`).

- **Functionality**:
    - Forwards incoming spikes from the `SnnPE` to the network.
    - Receives spikes from the network and delivers them to the connected `SnnPE`.
    - Translates between SST `SpikeEvent`s and network packets.
- **Ports**:
    - `spike_input`: Connects to the `SnnPE`'s `spike_output` port.
    - `spike_output`: Connects to the `SnnPE`'s `spike_input` port.
    - `port_network`: Connects to a network component (e.g., `merlin.hr_router`).

### c. `SpikeSource`

This component reads spike data from a file and injects it into the simulation at precise times.

- **Functionality**:
    - Acts as a data loader and spike generator.
    - Supports multiple input formats (e.g., TEXT, N-MNIST).
    - Generates precisely timed `SpikeEvent`s based on the input data.
- **Configuration Parameters**:
    - `dataset_path`: **(Required)** Path to the spike data file.
    - `dataset_format`: Format of the dataset (Default: `"TEXT"`).
    - `time_scale`: Scaling factor for timestamps in the data (Default: `1.0`).
    - `verbose`: Verbosity level for logging (Default: `0`).
- **Ports**:
    - `spike_output`: Sends `SpikeEvent`s to a target component (typically an `SnnNIC` or `SnnPE`).

## 3. Event Structure: `SpikeEvent`

A lightweight, serializable event used to represent a single spike. It contains the ID of the neuron that fired.

## 4. Workflow

A typical simulation workflow is as follows:
1.  `SpikeSource` reads spike data and sends `SpikeEvent`s to a target `SnnPE` (via its `SnnNIC`).
2.  The `SnnNIC` receives the event and passes it to the `SnnPE`.
3.  The `SnnPE` processes the spike, updating the membrane potential of connected neurons.
4.  If any neuron's potential exceeds `v_thresh`, it fires and generates a new `SpikeEvent`.
5.  The `SnnPE` sends the new `SpikeEvent` to its `SnnNIC`.
6.  The `SnnNIC` packetizes the event and routes it through the network to the destination `SnnNIC`, which then delivers it to the final `SnnPE`.

## 5. Data Formats

### Weight File (CSR-like)
A text file where each line represents a synaptic connection.
```
# from_neuron to_neuron weight
0 10 0.75
0 12 0.5
1 15 0.9
```

### Spike Input File (`TEXT` format)
A text file where each line represents a spike event.
```
# neuron_id timestamp(microseconds)
0 1000
5 1250
8 3000
```

## 6. Python API Usage Example

Here is an example of setting up a simple network with one `SpikeSource` and one `SnnPE`.

```python
import sst
from sst.merlin import *

# Define the network topology
topo = topoTorus()
topo.setShape([2, 2, 1]) # Example 2x2 torus
topo.setWidth([1, 1, 0])
topo.setLocalPortCount(1)

# Set up the network endpoint
netif = buildSimpleNetworkInterface("networkIF", "firefly", 1, 1)
netif.addParam("link_bw", "10GB/s")

# Create a Spike Source
source = sst.Component("source_0", "SnnDL.SpikeSource")
source.addParams({
    "dataset_path": "input_spikes.txt",
    "dataset_format": "TEXT"
})

# Create an SNN Processing Element and its NIC
pe_0 = sst.Component("pe_0", "SnnDL.SnnPE")
pe_0.addParams({
    "num_neurons": 100,
    "weights_file": "weights.txt"
})

nic_0 = sst.Component("nic_0", "SnnDL.SnnNIC")

# Connect the components
# Source -> NIC -> PE
link_in = sst.Link("spike_input_link_0")
link_in.connect(
    (source, "spike_output", "1ns"),
    (nic_0, "spike_input", "1ns")
)

link_pe = sst.Link("pe_link_0")
link_pe.connect(
    (nic_0, "spike_output", "1ns"),
    (pe_0, "spike_input", "1ns")
)
link_pe.connect(
    (pe_0, "spike_output", "1ns"),
    (nic_0, "spike_input", "1ns")
)

# Connect the NIC to the network router
router_0 = sst.Component("router_0", "merlin.hr_router")
router_0.addParams(topo.getRouterParams(0))
router_0.setSubComponent("networkIF", netif)

net_link = sst.Link("network_link_0")
net_link.connect(
    (nic_0, "port_network", "2ns"),
    (router_0, "port0", "2ns")
)
```

## 7. Build and Installation

This component is part of the `sst-elements` library and is compiled along with it. Ensure that `SnnDL` is included in the build configuration when running `configure` for `sst-elements`.

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
