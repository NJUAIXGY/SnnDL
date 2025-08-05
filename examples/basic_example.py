#!/usr/bin/env python3
"""
SnnDL基本使用示例
展示如何使用SnnDL组件创建简单的脉冲神经网络仿真
"""

import sst

print("=== SnnDL基本使用示例 ===")
print("创建简单的脉冲神经网络仿真")

# 1. 创建脉冲数据源
spike_source = sst.Component("SpikeSource", "SnnDL.SpikeSource")
spike_source.addParams({
    "dataset_path": "example_spikes.txt",
    "dataset_format": "TEXT",
    "time_scale": 1.0,
    "neuron_offset": 0,
    "max_events": 50,
    "verbose": 1
})

# 2. 创建SNN处理元素
snn_pe = sst.Component("ExamplePE", "SnnDL.SnnPE")
snn_pe.addParams({
    "num_neurons": 20,
    "v_thresh": 1.0,
    "v_reset": 0.0,
    "v_rest": 0.0,
    "tau_mem": 20.0,
    "t_ref": 2,
    "clock": "1GHz",
    "verbose": 1
})

# 3. 连接组件
link = sst.Link("spike_link")
link.connect((spike_source, "spike_output", "1ns"), 
             (snn_pe, "spike_input", "1ns"))

# 4. 配置统计收集
sst.setStatisticLoadLevel(1)
sst.setStatisticOutput("sst.statOutputCSV")
sst.enableAllStatisticsForComponentType("SnnDL.SnnPE")
sst.enableAllStatisticsForComponentType("SnnDL.SpikeSource")

print("组件配置完成")
print("运行命令: sst basic_example.py --stop-at=10ms")
print("输出文件: StatisticOutput.csv")
