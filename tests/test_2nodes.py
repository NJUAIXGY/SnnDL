#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
2节点SnnDL测试 - 修复版本
测试扩展的SpikeEvent在节点间传播
"""

import sst

# 仿真设置
sst.setProgramOption("timebase", "1ns")
SIMULATION_CYCLES = 200000  # 200us

# ========== 创建脉冲源组件 ==========
spike_source = sst.Component("SpikeSource0", "SnnDL.SpikeSource")
spike_source.addParams({
    "dataset_path": "simple_test.txt",
    "start_time": "0ns",
    "repeat_interval": "1000us"
})

# ========== 创建第一个SNN处理单元 ==========
snn_pe0 = sst.Component("SnnPE0", "SnnDL.SnnPE") 
snn_pe0.addParams({
    "node_id": 0,
    "num_neurons": 50,
    "weights_file": "simple_weights.txt",
    "clock": "1GHz",
    "dt_ms": 0.001,
    "v_thresh": -60.0,
    "v_reset": -70.0,
    "v_rest": -65.0,
    "tau_mem": 20.0,
    "t_ref": 1
})

# ========== 创建第二个SNN处理单元 ==========
snn_pe1 = sst.Component("SnnPE1", "SnnDL.SnnPE") 
snn_pe1.addParams({
    "node_id": 1,
    "num_neurons": 50,
    "weights_file": "simple_weights.txt",
    "clock": "1GHz",
    "dt_ms": 0.001,
    "v_thresh": -60.0,
    "v_reset": -70.0,
    "v_rest": -65.0,
    "tau_mem": 20.0,
    "t_ref": 1
})

# ========== 配置连接 ==========
# 脉冲源到PE0
link_source_pe0 = sst.Link("source_to_pe0")
link_source_pe0.connect(
    (spike_source, "spike_output", "1ns"),
    (snn_pe0, "spike_input", "1ns")
)

# PE0到PE1 - 这是关键连接
link_pe0_pe1 = sst.Link("pe0_to_pe1")
link_pe0_pe1.connect(
    (snn_pe0, "spike_output", "1ns"),
    (snn_pe1, "spike_input", "1ns")
)

# ========== 统计配置 ==========
sst.enableAllStatisticsForComponentType("SnnDL.SnnPE")
sst.enableAllStatisticsForComponentType("SnnDL.SpikeSource")
sst.setStatisticLoadLevel(7)
sst.setStatisticOutput("sst.statOutputCSV")
sst.setStatisticOutputOptions({
    "filepath": "test_2nodes_stats.csv",
    "separator": ","
})

# 设置仿真停止条件
sst.setProgramOption("stop-at", "{}ns".format(SIMULATION_CYCLES))

print("配置完成：简化的2节点SnnDL测试")
print("测试内容：")
print("  - SpikeSource -> SnnPE0 -> SnnPE1")
print("  - 验证扩展后的SpikeEvent传播")
print("  - 传统Link模式，无SubComponent接口")
print("仿真时长: {}ns".format(SIMULATION_CYCLES))
print("统计输出: test_2nodes_stats.csv")
