#!/usr/bin/env python3
"""
带网络路由的SnnDL测试配置
测试扩展的SpikeEvent功能，包含网络路由字段
"""

import sst

print("配置完成：网络路由SnnDL测试")
print("测试目标：验证扩展SpikeEvent的网络路由功能")
print("拓扑结构：Source -> PE0 -> PE1 (通过网络路由)")

# === 创建组件 ===

# 脉冲源
spike_source = sst.Component("SpikeSource", "SnnDL.SpikeSource")
spike_source.addParams({
    "dataset_path": "/home/anarchy/SST/test_spikes.txt",
    "dataset_format": "TEXT",
    "time_scale": 1.0,
    "neuron_offset": 0,
    "max_events": 10
})

# 第一个SNN处理元素
pe0 = sst.Component("PE0", "SnnDL.SnnPE")
pe0.addParams({
    "num_neurons": 5,
    "v_thresh": 1.0,
    "v_reset": 0.0,
    "v_rest": 0.0,
    "tau_mem": 20.0,
    "t_ref": 2,
    "clock": "1GHz"
})

# 第二个SNN处理元素
pe1 = sst.Component("PE1", "SnnDL.SnnPE")
pe1.addParams({
    "num_neurons": 5,
    "v_thresh": 1.0,
    "v_reset": 0.0,
    "v_rest": 0.0,
    "tau_mem": 20.0,
    "t_ref": 2,
    "clock": "1GHz"
})

# === 配置链接 ===

# Source -> PE0
link_source_pe0 = sst.Link("link_source_pe0")
link_source_pe0.connect((spike_source, "spike_output", "1ns"), 
                        (pe0, "spike_input", "1ns"))

# PE0 -> PE1 (测试网络路由)
link_pe0_pe1 = sst.Link("link_pe0_pe1")
link_pe0_pe1.connect((pe0, "spike_output", "1ns"),
                     (pe1, "spike_input", "1ns"))

print("数据文件：/home/anarchy/SST/test_spikes.txt")
print("链接配置：Source -> PE0 -> PE1")
print("网络测试：验证dest_node, dest_neuron, weight字段")
