#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试扩展的SpikeEvent基本功能

这个配置测试我们扩展后的SpikeEvent类，
验证新增的网络路由字段是否正常工作。
"""

import sst

# 创建一个简单的测试
def create_minimal_test():
    """创建最小的SnnDL测试配置"""
    
    # 仅创建一个SnnPE节点（不使用SubComponent接口）
    snnpe = sst.Component("snnpe_test", "SnnDL.SnnPE")
    snnpe.addParams({
        "clock": "1GHz",
        "num_neurons": 10,
        "v_thresh": 1.0,
        "v_reset": 0.0,
        "v_rest": 0.0,
        "tau_mem": 20.0,
        "t_ref": 2,
        "node_id": 0,
        "weights_file": "",
        "verbose": 3
    })
    
    # 创建脉冲源
    spike_source = sst.Component("spike_source", "SnnDL.SpikeSource")
    spike_source.addParams({
        "dataset_path": "/home/anarchy/SST/test_spikes.txt",
        "dataset_format": "TEXT",
        "time_scale": 1.0,
        "neuron_offset": 0,
        "max_events": 5,
        "verbose": 3
    })
    
    # 连接组件
    link = sst.Link("test_link")
    link.connect(
        (spike_source, "spike_output", "1ns"),
        (snnpe, "spike_input", "1ns")
    )
    
    # 启用统计
    snnpe.enableAllStatistics({"type": "sst.AccumulatorStatistic"})
    spike_source.enableAllStatistics({"type": "sst.AccumulatorStatistic"})

# 执行测试
create_minimal_test()

print("配置完成：最小SnnDL测试")
print("测试目标：验证扩展后的SpikeEvent类功能")
print("组件：SpikeSource -> SnnPE")
print("数据文件：/home/anarchy/SST/test_spikes.txt")
