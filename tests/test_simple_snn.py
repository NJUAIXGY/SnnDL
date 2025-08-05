#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化的SnnDL测试配置

测试扩展后的SpikeEvent类的基本功能，
不使用复杂的SubComponent接口，先验证核心功能。
"""

import sst

# === 基本参数 ===
def create_simple_test():
    """创建简单的2节点测试"""
    
    # 节点0：SnnPE + SpikeSource
    snnpe0 = sst.Component("snnpe0", "SnnDL.SnnPE")
    snnpe0.addParams({
        "clock": "1GHz",
        "num_neurons": 50,
        "v_thresh": 1.0,
        "v_reset": 0.0,
        "v_rest": 0.0, 
        "tau_mem": 20.0,
        "t_ref": 2,
        "node_id": 0,
        "weights_file": "",
        "verbose": 2
    })
    
    # 节点1：SnnPE
    snnpe1 = sst.Component("snnpe1", "SnnDL.SnnPE")
    snnpe1.addParams({
        "clock": "1GHz",
        "num_neurons": 50, 
        "v_thresh": 1.0,
        "v_reset": 0.0,
        "v_rest": 0.0,
        "tau_mem": 20.0,
        "t_ref": 2,
        "node_id": 1,
        "weights_file": "",
        "verbose": 2
    })
    
    # 脉冲源
    spike_source = sst.Component("spike_source", "SnnDL.SpikeSource")
    spike_source.addParams({
        "dataset_path": "/home/anarchy/SST/node_0_input.txt",
        "dataset_format": "TEXT",
        "time_scale": 1.0,
        "neuron_offset": 0,
        "max_events": 20,
        "verbose": 2
    })
    
    # 传统链接连接
    link0 = sst.Link("source_to_pe0")
    link0.connect(
        (spike_source, "spike_output", "1ns"),
        (snnpe0, "spike_input", "1ns")
    )
    
    link1 = sst.Link("pe0_to_pe1") 
    link1.connect(
        (snnpe0, "spike_output", "1ns"),
        (snnpe1, "spike_input", "1ns")
    )
    
    # 启用统计
    snnpe0.enableAllStatistics({"type": "sst.AccumulatorStatistic"})
    snnpe1.enableAllStatistics({"type": "sst.AccumulatorStatistic"})
    spike_source.enableAllStatistics({"type": "sst.AccumulatorStatistic"})

# 执行测试
create_simple_test()

print("配置完成：简化的2节点SnnDL测试")
print("测试内容：")
print("  - SpikeSource -> SnnPE0 -> SnnPE1")
print("  - 验证扩展后的SpikeEvent传播")
print("  - 传统Link模式，无SubComponent接口")
