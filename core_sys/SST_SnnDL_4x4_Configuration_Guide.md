# SST-SnnDL 4x4网格系统配置完整指南

## 目录
1. [系统概述](#系统概述)
2. [核心配置参数](#核心配置参数)
3. [组件配置详解](#组件配置详解)
4. [网络配置](#网络配置)
5. [数据文件配置](#数据文件配置)
6. [内存系统配置](#内存系统配置)
7. [统计和监控](#统计和监控)
8. [故障排除](#故障排除)
9. [最佳实践](#最佳实践)

## 系统概述

### 架构描述
4x4网格SST-SnnDL系统包含以下核心组件：
- **16个MultiCorePE节点**：每个PE包含4个SnnPESubComponent核心
- **16个SpikeSource组件**：提供外部脉冲输入
- **16个SnnNIC网络接口**：处理跨节点通信
- **16个Merlin路由器**：实现4x4网格拓扑
- **WeightLoader**：预加载权重到全局内存
- **分层内存系统**：每个PE有独立的L1缓存和内存控制器

### 系统拓扑
```
PE0---PE1---PE2---PE3
 |     |     |     |
PE4---PE5---PE6---PE7
 |     |     |     |
PE8---PE9---PE10--PE11
 |     |     |     |
PE12--PE13--PE14--PE15
```

## 核心配置参数

### 全局参数
```python
# === 基础配置 ===
MESH_SIZE = 4                    # 网格大小 (4x4)
NUM_CORES_PER_PE = 4            # 每个PE的核心数
NEURONS_PER_CORE = 4            # 每个核心的神经元数
NEURONS_PER_PE = 16             # 每个PE的神经元数 (4*4)
TOTAL_NODES = 16                # 总节点数 (4*4)
SIMULATION_TIME = "50us"        # 仿真时间（关键：必须足够长）

# === 内存配置 ===
BASE_WEIGHT_ADDR = 0x10000000   # 权重基址
PER_NODE_STRIDE = 32768         # 内存步长 (32KB per node)

# === 网络配置 ===
NETWORK_BANDWIDTH = "40GiB/s"   # 网络带宽
BUFFER_SIZE = "8KiB"            # 缓冲区大小
```

### 关键配置要点
1. **SIMULATION_TIME**：必须大于脉冲数据的最大时间戳
2. **PER_NODE_STRIDE**：确保各节点内存地址不重叠
3. **NEURONS_PER_PE**：必须与权重矩阵大小匹配

## 组件配置详解

### 1. MultiCorePE配置

```python
# 每个PE的基本配置
node_params = {
    "verbose": 2,                           # 日志级别 (0-4)
    "num_cores": NUM_CORES_PER_PE,         # 核心数：4
    "neurons_per_core": NEURONS_PER_CORE,  # 每核神经元数：4
    "total_neurons": TOTAL_NODES * NEURONS_PER_PE,  # 全局神经元总数：256
    "node_id": i,                          # 节点ID (0-15)
    "global_neuron_base": i * NEURONS_PER_PE,  # 全局神经元基址
    
    # === 测试和功能开关 ===
    "enable_test_traffic": 1,              # 启用测试流量
    "enable_memory_weights": 1,            # 启用内存权重读取
    "write_weights_on_init": 1,            # 初始化时写入权重
    "enable_weight_fetch": 1,              # 启用从内存获取权重
    
    # === 神经元模型参数 ===
    "v_thresh": 0.1,                       # 神经元阈值
    "v_rest": 0.0,                         # 静息电位
    "v_reset": 0.0,                        # 重置电位
    
    # === 权重回退机制 ===
    "use_event_weight_fallback": 1,        # 启用事件权重回退
    "event_weight_fallback": 0.5,          # 回退权重值
    
    # === 权重验证参数 ===
    "verify_weights": 1,                   # 启用权重验证
    "weight_verify_samples": 8,            # 验证样本数
    "expected_weight_value": 1.0,          # 期望权重值
    "verify_log_each_sample": 1,           # 记录每个验证样本
    "memory_warmup_cycles": 100,           # 内存预热周期
    
    # === 文件和地址配置 ===
    "weights_file": f"datasets/4x4_weights_node_{i}.bin",
    "base_addr": BASE_WEIGHT_ADDR + i * PER_NODE_STRIDE,
}
```

#### 关键参数说明：
- **global_neuron_base**：节点i管理神经元[i*16, (i+1)*16-1]
- **base_addr**：各节点权重的内存地址，必须不重叠
- **enable_weight_fetch**：核心开关，必须为1才能读取权重

### 2. SnnNIC网络接口配置

```python
nic_params = {
    "node_id": str(i),                     # 节点ID
    "link_bw": NETWORK_BANDWIDTH,          # 链路带宽
    "input_buf_size": BUFFER_SIZE,         # 输入缓冲区大小
    "output_buf_size": BUFFER_SIZE,        # 输出缓冲区大小
    "use_direct_link": "false",            # 使用标准网络模式
    "port_name": "network",                # 端口名称
    "verbose": 1,                          # 日志级别
    "total_nodes": TOTAL_NODES,            # ★关键★ 总节点数用于job_size
}
```

#### 关键配置点：
- **total_nodes**：必须设置，用于merlin网络的job_size配置
- **use_direct_link**：设为false使用merlin网络路由
- **port_name**：必须与路由器连接端口名匹配

### 3. SpikeSource配置

```python
spike_source_params = {
    "verbose": 2,                          # 日志级别
    "dataset_path": spike_data_files[i],   # 脉冲数据文件路径
    "neurons_per_core": NEURONS_PER_CORE,  # 每核神经元数（用于目标计算）
    "start_time_us": 1.0 + (i % 4) * 0.5, # 错开启动时间
    "loop_dataset": 1,                     # 循环数据集
    "source_id": i,                        # 源ID
}
```

#### 关键参数说明：
- **neurons_per_core**：用于正确计算目标节点ID
- **start_time_us**：错开启动时间避免同时发送

### 4. Merlin路由器配置

```python
router_params = {
    "id": i,                               # 路由器ID
    "num_ports": 5,                        # 端口数：4方向+1本地
    "link_bw": NETWORK_BANDWIDTH,          # 链路带宽
    "flit_size": "8B",                     # 数据片大小
    "xbar_bw": NETWORK_BANDWIDTH,          # 交换机带宽
    "input_latency": "10ns",               # 输入延迟
    "output_latency": "10ns",              # 输出延迟
    "input_buf_size": "4KiB",              # 输入缓冲区
    "output_buf_size": "4KiB",             # 输出缓冲区
    "num_vns": 1,                          # 虚拟网络数
    "xbar_arb": "merlin.xbar_arb_lru",     # 仲裁算法
    "debug": 0,                            # 调试级别
    "verbose": 0,                          # 日志级别
}

# 拓扑配置
topo_params = {
    "shape": f"{MESH_SIZE}x{MESH_SIZE}",   # 网格形状：4x4
    "width": "1x1",                        # 网格宽度
    "local_ports": "1",                    # 本地端口数
}
```

## 网络配置

### 连接拓扑

#### 1. NIC到路由器连接
```python
# 每个NIC连接到对应路由器的本地端口(port4)
for i in range(TOTAL_NODES):
    nic_router_link = sst.Link(f"nic_{i}_to_router_{i}")
    nic_router_link.connect(
        (nics[i], "network", "5ns"),       # NIC的network端口
        (routers[i], "port4", "5ns")       # 路由器的本地端口
    )
```

#### 2. 路由器间连接（4x4网格）
```python
# 水平连接 (East-West)
for y in range(MESH_SIZE):
    for x in range(MESH_SIZE - 1):
        node_id = y * MESH_SIZE + x
        east_node_id = y * MESH_SIZE + (x + 1)
        
        link = sst.Link(f"router_east_{node_id}_to_{east_node_id}")
        link.connect(
            (routers[node_id], "port0", "5ns"),      # East port
            (routers[east_node_id], "port1", "5ns")  # West port
        )

# 垂直连接 (North-South)
for x in range(MESH_SIZE):
    for y in range(MESH_SIZE - 1):
        node_id = y * MESH_SIZE + x
        south_node_id = (y + 1) * MESH_SIZE + x
        
        link = sst.Link(f"router_south_{node_id}_to_{south_node_id}")
        link.connect(
            (routers[node_id], "port2", "5ns"),       # South port
            (routers[south_node_id], "port3", "5ns")  # North port
        )
```

### 端口映射
- **port0**: East (东)
- **port1**: West (西) 
- **port2**: South (南)
- **port3**: North (北)
- **port4**: Local (本地PE)

## 数据文件配置

### 脉冲数据文件格式
```
# 神经元ID 时间戳(us)
0 2
0 3
0 4
1 2
1 3
...
```

#### 数据生成函数
```python
def create_cross_node_spike_data(filename, source_node_id, target_neurons):
    """
    创建跨节点脉冲数据
    
    Args:
        filename: 输出文件名
        source_node_id: 源节点ID  
        target_neurons: 目标神经元ID列表
    """
    with open(filename, 'w') as f:
        f.write("# 神经元ID 时间戳(us)\n")
        for i, neuron_id in enumerate(target_neurons):
            timestamp = 2 + i  # 从2us开始
            # 每个神经元发送多个脉冲增加激活概率
            for offset in [0, 1, 2, 5, 8, 10]:
                f.write(f"{neuron_id} {timestamp + offset}\n")
```

### 权重文件格式
```python
def create_weight_file(node_id, neurons_per_pe, total_neurons):
    """
    创建二进制权重文件
    
    权重矩阵大小: neurons_per_pe × total_neurons
    对于4x4网格: 16 × 256 = 4096个权重/节点
    """
    weight_file = f"datasets/4x4_weights_node_{node_id}.bin"
    weights = [1.0] * (neurons_per_pe * total_neurons)
    
    with open(weight_file, 'wb') as f:
        for w in weights:
            f.write(struct.pack('f', w))  # 32位浮点数
```

## 内存系统配置

### 全局内存控制器
```python
global_memory_controller = sst.Component("global_memory_controller", "memHierarchy.MemController")
global_memory_controller.addParams({
    "clock": "1GHz",                       # 时钟频率
    "backing": "malloc",                   # 后端存储
    "backend": "memHierarchy.simpleMem",   # 内存后端
    "backend.access_time": "100ns",        # 访问时间
    "backend.mem_size": "1GiB",           # 内存大小
    "addr_range_start": "0",              # 地址范围起始
    "addr_range_end": "1073741823"        # 地址范围结束
})
```

### WeightLoader配置
```python
weight_loader_params = {
    "verbose": 2,                          # 日志级别
    "base_addr_start": BASE_WEIGHT_ADDR,   # 基址
    "per_core_stride": PER_NODE_STRIDE,    # 步长
    "num_cores": TOTAL_NODES,              # 核心数
    "neurons_per_core": NEURONS_PER_CORE,  # 每核神经元数
    "total_neurons": TOTAL_NODES * NEURONS_PER_PE,  # 总神经元数
    "weight_format": "bin",                # 权重格式
    "per_core_files": 1,                   # 每核心一个文件
    "file_template": "datasets/4x4_weights_node_{core}.bin",  # 文件模板
    "fill_value": 0.0,                     # 填充值
    "validate_length": 1,                  # 验证长度
    "row_major": 1                         # 行主序
}
```

### L1缓存配置
```python
# 为每个PE的每个核心创建L1缓存
l1_cache_params = {
    "cache_frequency": "2GHz",             # 缓存频率
    "cache_size": "4KiB",                  # 缓存大小
    "associativity": "4",                  # 关联度
    "cache_line_size": "64",               # 缓存行大小
    "access_latency_cycles": "2",          # 访问延迟
    "L1": "1",                            # L1标识
    "coherence_protocol": "none",          # 一致性协议
    "debug": "0",                         # 调试级别
    "verbose": "0"                        # 日志级别
}
```

### 内存控制器配置
```python
# 每个核心的内存控制器
mem_ctrl_params = {
    "clock": "2GHz",                       # 时钟频率
    "backing": "malloc",                   # 后端存储
    "backend": "memHierarchy.simpleMem",   # 内存后端
    "backend.access_time": "30ns",         # 访问时间
    "backend.mem_size": "8MiB",           # 内存大小
    "addr_range_start": "0",              # 地址范围
    "addr_range_end": "8388607"           # 地址范围
}
```

## 统计和监控

### 统计配置
```python
# 全局统计设置
sst.setStatisticLoadLevel(5)                          # 统计级别
sst.enableAllStatisticsForComponentType("SnnDL.MultiCorePE")
sst.enableAllStatisticsForComponentType("SnnDL.SpikeSource") 
sst.enableAllStatisticsForComponentType("merlin.hr_router")

# 为PE启用特定统计
for node in nodes:
    node.enableStatistics([
        "total_external_spikes_received",     # 外部脉冲接收
        "total_internal_spikes_processed",    # 内部脉冲处理
        "total_network_spikes_sent",          # 网络脉冲发送
        "total_neuron_activations",           # 神经元激活
        "memory_accesses"                     # 内存访问
    ])

# 为路由器启用统计
for router in routers:
    router.enableStatistics([
        "router.packet_count",                # 包计数
        "router.network_load"                 # 网络负载
    ])
```

### 关键统计指标
- **SpikeSource统计**：加载事件数、发送事件数
- **SnnNIC统计**：发送/接收脉冲数、发送/接收包数
- **MultiCorePE统计**：神经元激活、内存访问、脉冲处理
- **路由器统计**：包传输、网络延迟、缓冲区使用

## 故障排除

### 常见问题及解决方案

#### 1. 包发送但不接收
**症状**：SnnNIC显示发送包>0，接收包=0
**原因**：Merlin网络job_size配置错误
**解决**：确保SnnNIC配置中包含正确的total_nodes参数

#### 2. 神经元映射错误
**症状**：出现"无法映射的目标神经元"错误
**原因**：核心的神经元范围配置过小
**解决**：设置num_neurons = num_cores * neurons_per_core

#### 3. 脉冲不发送
**症状**：SpikeSource加载事件但发送事件=0
**原因**：仿真时间短于脉冲时间戳
**解决**：增加SIMULATION_TIME到足够长

#### 4. 权重读取失败
**症状**：weight_verify失败或memory_link=no
**原因**：内存连接未正确建立
**解决**：检查L1缓存连接和内存控制器配置

### 调试技巧

#### 1. 启用详细日志
```python
# 在组件参数中设置
"verbose": 2,  # 0=静默, 1=基本, 2=详细, 3=调试, 4=全部
```

#### 2. 检查统计输出
```bash
# 查看特定组件统计
grep -A10 "SnnNIC.*最终统计" output.log
grep -A5 "SpikeSource.*最终统计" output.log
```

#### 3. 验证网络连接
```bash
# 检查路由器包传输
grep "router.*packet_count" output.log
```

## 最佳实践

### 1. 参数设置建议
- **仿真时间**：设为脉冲数据最大时间戳的2倍以上
- **网络带宽**：根据预期流量设置，避免网络拥塞
- **缓冲区大小**：设置足够大避免包丢失
- **内存地址**：确保各节点地址范围不重叠

### 2. 性能优化
- **权重预加载**：使用WeightLoader在仿真前加载权重
- **错开启动**：SpikeSource使用不同start_time_us避免同步
- **内存层次**：使用L1缓存提高内存访问性能

### 3. 扩展指南
```python
# 扩展到更大网格 (如8x8)
MESH_SIZE = 8
TOTAL_NODES = 64
# 相应调整内存地址、文件数量等
```

### 4. 配置验证清单
- [ ] SIMULATION_TIME足够长
- [ ] total_nodes参数正确传递给SnnNIC
- [ ] 权重文件大小匹配：neurons_per_pe × total_neurons
- [ ] 脉冲数据时间戳在仿真时间范围内
- [ ] 各节点内存地址不重叠
- [ ] 网络连接正确建立
- [ ] 统计功能正确启用

## 配置模板

### 完整配置示例
```python
#!/usr/bin/env python3
import sst
import os
import struct

# === 全局配置 ===
MESH_SIZE = 4
NUM_CORES_PER_PE = 4
NEURONS_PER_CORE = 4
NEURONS_PER_PE = NUM_CORES_PER_PE * NEURONS_PER_CORE
TOTAL_NODES = MESH_SIZE * MESH_SIZE
SIMULATION_TIME = "50us"

BASE_WEIGHT_ADDR = 0x10000000
PER_NODE_STRIDE = 32768

NETWORK_BANDWIDTH = "40GiB/s"
BUFFER_SIZE = "8KiB"

# === 数据文件准备 ===
# [数据文件创建代码]

# === 全局内存 ===
# [全局内存控制器配置]

# === WeightLoader ===
# [WeightLoader配置]

# === 路由器网络 ===
# [路由器创建和配置]

# === PE节点 ===
# [MultiCorePE配置循环]

# === SpikeSource ===
# [SpikeSource配置循环]

# === 网络连接 ===
# [NIC-路由器连接]
# [路由器间连接]

# === 统计配置 ===
# [统计启用]

# === 仿真启动 ===
sst.setProgramOption("timebase", "1ps")
sst.setProgramOption("stop-at", SIMULATION_TIME)
```

---

## 版本信息
- **文档版本**：1.0
- **适用SST版本**：15.0.0
- **最后更新**：2025-01-22
- **测试配置**：4x4网格，16节点，256神经元

此文档基于成功运行的test_corrected_4x4.py配置编写，包含了所有关键参数和配置要点。