#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Miranda风格的SnnDL网络配置示例

这个配置演示了如何使用SubComponent接口创建一个具有
SnnPE节点的2x2网格网络，类似于miranda的CPU连接方式。
"""

import sst
import sst.merlin

# === 基本参数 ===
network_config = {
    "topology": "merlin.torus",
    "torus:shape": "2x2",
    "link_bw": "40GB/s",
    "xbar_bw": "40GB/s",
    "input_latency": "10ns",
    "output_latency": "10ns",
    "input_buf_size": "1KiB",
    "output_buf_size": "1KiB",
    "flit_size": "8B",
    "network_inspectors": ""
}

# SnnPE配置
snnpe_config = {
    "clock": "1GHz",
    "num_neurons": 100,
    "v_thresh": 1.0,
    "v_reset": 0.0,
    "v_rest": 0.0,
    "tau_mem": 20.0,
    "t_ref": 2,
    "verbose": 2
}

# === 创建2x2网格的SnnPE节点 ===
def create_snn_node(node_id, x, y):
    """创建一个SnnPE节点及其网络接口"""
    
    # 创建SnnPE组件
    snnpe = sst.Component(f"snnpe_{node_id}", "SnnDL.SnnPE")
    
    # 配置SnnPE参数
    config = snnpe_config.copy()
    config["node_id"] = node_id
    config["weights_file"] = ""  # 暂时不使用权重文件
    
    for key, value in config.items():
        snnpe.addParams({key: value})
    
    # 创建SnnNIC子组件作为网络接口
    snn_nic = snnpe.setSubComponent("interface", "SnnDL.SnnNIC")
    snn_nic.addParams({
        "node_id": node_id,
        "link_bw": "40GiB/s",
        "input_buf_size": "1KiB", 
        "output_buf_size": "1KiB",
        "verbose": 2
    })
    
    # 创建网络路由器
    router = sst.Component(f"router_{node_id}", "merlin.hr_router")
    router.addParams(network_config)
    router.addParam("id", node_id)
    
    # 连接SnnNIC到路由器
    link = sst.Link(f"link_snn_{node_id}")
    link.connect(
        (snn_nic, "network", "1ns"),
        (router, "port0", "1ns")
    )
    
    return snnpe, router

# === 创建网络拓扑 ===
nodes = []
routers = []

# 创建2x2网格的节点
for y in range(2):
    for x in range(2):
        node_id = y * 2 + x
        snnpe, router = create_snn_node(node_id, x, y)
        nodes.append(snnpe)
        routers.append(router)

# 连接路由器形成2x2 torus拓扑
# 水平连接
for y in range(2):
    for x in range(2):
        curr_id = y * 2 + x
        next_x = (x + 1) % 2
        next_id = y * 2 + next_x
        
        link_h = sst.Link(f"link_h_{curr_id}_{next_id}")
        link_h.connect(
            (routers[curr_id], f"port{1+x}", "1ns"),
            (routers[next_id], f"port{1+(x+1)%2}", "1ns") 
        )

# 垂直连接
for x in range(2):
    for y in range(2):
        curr_id = y * 2 + x
        next_y = (y + 1) % 2
        next_id = next_y * 2 + x
        
        link_v = sst.Link(f"link_v_{curr_id}_{next_id}")
        link_v.connect(
            (routers[curr_id], f"port{3+y}", "1ns"),
            (routers[next_id], f"port{3+(y+1)%2}", "1ns")
        )

# === 添加脉冲源（用于测试） ===
spike_source = sst.Component("spike_source", "SnnDL.SpikeSource")
spike_source.addParams({
    "spikes_to_send": 10,
    "spike_delay": "1ms",
    "target_neuron_ids": "0,25,50,75",  # 发送到不同节点的神经元
    "verbose": 2
})

# 连接脉冲源到第一个SnnPE（传统方式）
spike_link = sst.Link("spike_input_link")
spike_link.connect(
    (spike_source, "spike_output", "1ns"),
    (nodes[0], "spike_input", "1ns")
)

# === 仿真配置 ===
sst.setStatisticLoadLevel(7)

# 为所有SnnPE节点启用统计
for i, node in enumerate(nodes):
    node.enableAllStatistics({"type": "sst.AccumulatorStatistic"})

# 仿真参数
print("配置完成：2x2 SnnDL网络，使用SubComponent接口")
print("节点配置：")
for i in range(4):
    print(f"  节点{i}: SnnPE + SnnNIC + Router")
print("网络拓扑：2x2 torus")
print("测试：脉冲源 -> 节点0，观察网络传播")
