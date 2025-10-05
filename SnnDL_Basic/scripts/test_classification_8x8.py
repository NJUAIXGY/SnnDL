#!/usr/bin/env python3

import sst
import os
import struct


# === 提前配置统计输出（必须在创建任何组件之前）===
sst.setStatisticLoadLevel(5)
sst.setStatisticOutput("sst.statOutputCSV")
sst.setStatisticOutputOptions({
    "filepath": "./complex_classification_stats_8x8.csv",
    "separator": ","
})
sst.enableAllStatisticsForComponentType("SnnDL.MultiCorePE")
sst.enableAllStatisticsForComponentType("SnnDL.SpikeSource")
sst.enableAllStatisticsForComponentType("merlin.hr_router")
sst.enableAllStatisticsForComponentType("SnnDL.SnnPESubComponent")

# === 8x8分层网络分类任务配置 ===

# === 网络架构配置 ===
MESH_SIZE = 8  # 8x8网格
NUM_CORES_PER_PE = 4  # 每个PE的core数：4
NEURONS_PER_CORE = 4   # 每个core的神经元数：4
NEURONS_PER_PE = NUM_CORES_PER_PE * NEURONS_PER_CORE  # 每个PE的神经元数：16
TOTAL_NODES = MESH_SIZE * MESH_SIZE  # 64个节点
SIMULATION_TIME = "2000us"   # 拉长仿真时间以获得稳定统计

# 网络分层定义 (5层架构)
INPUT_LAYER = list(range(0, 8))      # PE 0-7: 输入层 (8个PE)
HIDDEN_LAYER_1 = list(range(8, 24))   # PE 8-23: 隐藏层1 (16个PE)
HIDDEN_LAYER_2 = list(range(24, 40))  # PE 24-39: 隐藏层2 (16个PE)
HIDDEN_LAYER_3 = list(range(40, 56))  # PE 40-55: 隐藏层3 (16个PE)
OUTPUT_LAYER = list(range(56, 64))   # PE 56-63: 输出层 (8个PE)

# 复杂分类任务配置 (扩展到8类)
NUM_CLASSES = 8  # 8类分类
CLASS_A_FREQ = 40   # 类别A: 低频规律脉冲 (40Hz)
CLASS_B_FREQ = 60   # 类别B: 低中频突发脉冲 (60Hz)
CLASS_C_FREQ = 80   # 类别C: 中频突发脉冲 (80Hz)
CLASS_D_FREQ = 100  # 类别D: 中高频混合模式 (100Hz)
CLASS_E_FREQ = 120  # 类别E: 高频混合模式 (120Hz)
CLASS_F_FREQ = 150  # 类别F: 高频稀疏脉冲 (150Hz)
CLASS_G_FREQ = 180  # 类别G: 超高频模式 (180Hz)
CLASS_H_FREQ = 200  # 类别H: 极高频稀疏脉冲 (200Hz)

# 权重内存布局
BASE_WEIGHT_ADDR = 0x10000000
PER_NODE_STRIDE = 65536  # 增加内存步长以容纳更大权重矩阵

# 网络参数 (扩展带宽以支持更大网络)
NETWORK_BANDWIDTH = "80GiB/s"  # 双倍带宽
BUFFER_SIZE = "16KiB"  # 双倍缓冲区大小

print(f"🧠 8x8分层神经网络分类任务: {MESH_SIZE}x{MESH_SIZE} = {TOTAL_NODES}个节点")
print(f"📊 网络架构:")
print(f"  输入层 (PE 0-7): {len(INPUT_LAYER)}个PE")
print(f"  隐藏层1 (PE 8-23): {len(HIDDEN_LAYER_1)}个PE")
print(f"  隐藏层2 (PE 24-39): {len(HIDDEN_LAYER_2)}个PE")
print(f"  隐藏层3 (PE 40-55): {len(HIDDEN_LAYER_3)}个PE")
print(f"  输出层 (PE 56-63): {len(OUTPUT_LAYER)}个PE")
print(f"🎯 复杂分类任务: 8类频率识别 (40Hz-200Hz)")
print(f"⚙️ 权重加载模式: 启用内存权重，使用分层设计的权重文件")

# === 数据文件路径配置 ===
weights_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights_8x8")

# 脉冲数据文件将从预先生成的文件中加载

# 权重将完全从预先生成的二进制文件中加载

# === 加载预先生成的脉冲数据文件 ===
spike_data_files = []
spike_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "spike_data_8x8")

# 加载输入层(PE 0-7)的8类预生成数据
class_names = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
frequencies = [CLASS_A_FREQ, CLASS_B_FREQ, CLASS_C_FREQ, CLASS_D_FREQ, 
               CLASS_E_FREQ, CLASS_F_FREQ, CLASS_G_FREQ, CLASS_H_FREQ]

for pe_id in INPUT_LAYER:
    class_name = class_names[pe_id]
    
    spike_file = os.path.join(spike_dir, f"complex_input_pe_{pe_id}_class_{class_name}.txt")
    
    # 验证文件存在
    if not os.path.exists(spike_file):
        print(f"❌ 错误: 脉冲数据文件不存在: {spike_file}")
        print(f"请先运行: python3 scripts/generate_spike_data_8x8.py")
        exit(1)
    
    spike_data_files.append(spike_file)
    
    # 读取文件统计信息
    with open(spike_file, 'r') as f:
        lines = [line for line in f.readlines() if not line.startswith('#')]
    
    freq = frequencies[pe_id]
    print(f"  ✅ 加载PE{pe_id}: 类别{class_name} ({freq}Hz), {len(lines)}个脉冲事件")

print(f"\n🔗 使用预先生成的权重文件:")
print(f"  权重文件将完全从 {weights_dir} 目录中加载")
print(f"  每个PE有独立的权重文件: classification_weights_pe_{{0-63}}.bin")

# === 为每个PE创建独立的内存系统 ===
pe_memory_controllers = []
pe_weight_loaders = []
pe_memory_buses = []  # 存储内存总线以便后续连接L1缓存

for pe_id in range(TOTAL_NODES):
    # 为每个PE创建独立的内存控制器
    mem_controller = sst.Component(f"pe_{pe_id}_memory_controller", "memHierarchy.MemController")
    mem_controller.addParams({
        "clock": "1GHz",
        "backing": "malloc",
        "addr_range_start": "0",
        "addr_range_end": "134217727"  # 128MB地址范围
    })
    # 后端子组件（与4x4保持一致）
    mem_backend = mem_controller.setSubComponent("backend", "memHierarchy.simpleMem")
    mem_backend.addParams({
        "access_time": "100ns",
        "mem_size": "128MiB"
    })
    pe_memory_controllers.append(mem_controller)
    
    # 为每个PE创建内存总线，支持多个L1缓存连接
    mem_bus = sst.Component(f"pe_{pe_id}_memory_bus", "memHierarchy.Bus")
    mem_bus.addParams({
        "bus_frequency": "1GHz",
        "debug": "0",
        "verbose": "0"
    })
    pe_memory_buses.append(mem_bus)
    
    # 连接内存总线到内存控制器
    bus_to_mem_link = sst.Link(f"pe_{pe_id}_bus_to_mem")
    bus_to_mem_link.connect(
        (mem_bus, "lowlink0", "5ns"),
        (mem_controller, "highlink", "5ns")
    )
    
    # 为每个PE创建独立的WeightLoader
    weight_loader = sst.Component(f"pe_{pe_id}_weight_loader", "SnnDL.WeightLoader")
    weight_loader.addParams({
        "verbose": 0,
        "base_addr_start": 0x0,  # 每个PE从地址0开始
        "per_core_stride": PER_NODE_STRIDE,
        "num_cores": 1,  # 每个WeightLoader只管理一个PE
        "neurons_per_core": NEURONS_PER_CORE,
        "num_cores": NUM_CORES_PER_PE,
        # 行=post_local(16)，列=pre_global(1024)
        "rows_per_core": NEURONS_PER_PE,
        "cols_per_core": TOTAL_NODES * NEURONS_PER_PE,
        "total_neurons": NEURONS_PER_PE,  # 只管理当前PE的神经元
        "weight_format": "bin",
        "per_core_files": 1,
        "file_template": os.path.join(weights_dir, f"classification_weights_pe_{pe_id}.bin"),
        "fill_value": 0.0,
        "validate_length": 1,
        "row_major": 1
    })
    
    # 连接WeightLoader到内存总线
    weight_loader_mem = weight_loader.setSubComponent("memory", "memHierarchy.standardInterface")
    weight_loader_mem.addParams({"port": "lowlink"})
    
    weight_loader_link = sst.Link(f"pe_{pe_id}_weight_loader_to_bus")
    weight_loader_link.connect(
        (weight_loader_mem, "lowlink", "5ns"),
        (mem_bus, "highlink4", "5ns")  # WeightLoader使用highlink4
    )
    
    pe_weight_loaders.append(weight_loader)

print(f"✅ 创建{len(pe_memory_controllers)}个独立PE内存控制器和{len(pe_memory_buses)}个内存总线")

# === 创建网络路由器 ===
routers = []
for i in range(TOTAL_NODES):
    router = sst.Component(f"router_{i}", "merlin.hr_router")
    router.addParams({
        "id": i,
        "num_ports": 5,  # 4个方向端口 + 1个本地端口
        "link_bw": NETWORK_BANDWIDTH,
        "flit_size": "8B",
        "xbar_bw": NETWORK_BANDWIDTH,
        "input_latency": "10ns",
        "output_latency": "10ns",
        "input_buf_size": BUFFER_SIZE,
        "output_buf_size": BUFFER_SIZE,
        "num_vns": 1,
        "xbar_arb": "merlin.xbar_arb_lru",
        "debug": 0,
        "verbose": 0,
        "network_inspectors": "",
    })

    # 配置mesh拓扑
    topo = router.setSubComponent("topology", "merlin.mesh")
    topo.addParams({
        "shape": f"{MESH_SIZE}x{MESH_SIZE}",
        "width": "1x1",
        "local_ports": "1",
    })

    routers.append(router)

print(f"✅ 创建{len(routers)}个路由器完成")

# === 创建PE节点 ===
nodes = []
nics = []

for i in range(TOTAL_NODES):
    node = sst.Component(f"multicore_pe_{i}", "SnnDL.MultiCorePE")
    
    # 根据层类型调整神经元参数
    layer_name = "输入层" if i in INPUT_LAYER else \
                 "隐藏层1" if i in HIDDEN_LAYER_1 else \
                 "隐藏层2" if i in HIDDEN_LAYER_2 else \
                 "隐藏层3" if i in HIDDEN_LAYER_3 else "输出层"
    
    # 分层优化神经元阈值
    if i in INPUT_LAYER:
        v_thresh = 0.02  # 输入层低阈值
    elif i in HIDDEN_LAYER_1:
        v_thresh = 0.02  # 隐藏层1低阈值
    elif i in HIDDEN_LAYER_2:
        v_thresh = 0.022 # 隐藏层2稍高阈值
    elif i in HIDDEN_LAYER_3:
        v_thresh = 0.024 # 隐藏层3更高阈值
    else:  # OUTPUT_LAYER
        v_thresh = 0.026 # 输出层最高阈值，提高选择性
    
    node_params = {
        "verbose": 0,
        "num_cores": NUM_CORES_PER_PE,
        "neurons_per_core": NEURONS_PER_CORE,
        "num_cores": NUM_CORES_PER_PE,
        "total_neurons": TOTAL_NODES * NEURONS_PER_PE,
        "node_id": i,
        "global_neuron_base": i * NEURONS_PER_PE,
        "enable_test_traffic": 0,
        "test_target_node": 63,  # 测试流量重定向到最后一个PE
        "test_period": 2000,  # 增大测试流量间隔
        "test_spikes_per_burst": 1,
        "test_max_spikes": 10,
        "loop_dataset": 0,
        "enable_memory_weights": 1,
        "write_weights_on_init": 1,
        "weights_file": os.path.join(weights_dir, f"classification_weights_pe_{i}.bin"),
        "v_thresh": v_thresh,  # 分层调整阈值
        "v_rest": 0.0,
        "v_reset": 0.0,
        "use_event_weight_fallback": 1,
        "event_weight_fallback": 0.1,
        "verify_weights": 0,
        "weight_verify_samples": 2,
        "expected_weight_value": 0.0,
        "verify_log_each_sample": 0,
        "memory_warmup_cycles": 200,
        "enable_weight_fetch": 1,
        "memory_weight_priority": 1,
        "debug_weight_loading": 0,
        "debug_memory_accesses": 0,
        "verbose_weight_fetch": 0,
        "enable_weight_fetch": 1,
        "memory_warmup_cycles": 100
    }

    # 每个PE使用本地地址空间
    weight_addr = 0x0
    node_params["base_addr"] = weight_addr

    node.addParams(node_params)

    # 创建SnnNIC网络接口
    nic = node.setSubComponent("network_interface", "SnnDL.SnnNIC")
    nic.addParams({
        "node_id": str(i),
        "link_bw": NETWORK_BANDWIDTH,
        "input_buf_size": BUFFER_SIZE,
        "output_buf_size": BUFFER_SIZE,
        "use_direct_link": "false",
        "port_name": "network",
        "verbose": 0,
        "total_nodes": TOTAL_NODES,
    })

    # 为每个核心创建SnnPESubComponent
    for core_idx in range(NUM_CORES_PER_PE):
        core_subcomponent = node.setSubComponent(f"core{core_idx}", "SnnDL.SnnPESubComponent")
        core_subcomponent.addParams({
            "core_id": core_idx,
            "total_cores": NUM_CORES_PER_PE,
            "global_neuron_base": i * NEURONS_PER_PE,
            "num_neurons": NUM_CORES_PER_PE * NEURONS_PER_CORE,
            "v_thresh": v_thresh,
            "v_reset": 0.0,
            "v_rest": 0.0,
            "tau_mem": 20.0,
            "t_ref": 2,
            "base_addr": weight_addr,
            "node_id": i,
            "verbose": 0,
            "enable_weight_fetch": 1,
            "write_weights_on_init": 1,
            "memory_warmup_cycles": 100,
            "init_default_weight": 0.5,
            "max_outstanding_requests": 16,
            "max_cache_entries": 4096,
            "use_event_weight_fallback": 1 if i in INPUT_LAYER else 0,
            "merge_read_cacheline": 1,
            "merge_read_row": 0,
            # 全网读取：行=post_local(0..15), 列=pre_global(0..1023)
            "weights_cols": TOTAL_NODES * NEURONS_PER_PE,
            "index_mode": "post_row_pre_col",
            # 按权重构建路由（禁用层掩码，允许同PE）
            "routing_mode": "weight_driven",
            "weights_template": os.path.join(weights_dir, "classification_weights_pe_{pe}.bin"),
            "total_nodes": TOTAL_NODES,
            "routing_epsilon": 0.6,
            "routing_topk_per_pe": 2,
            "routing_topk": 12,
            "route_exclude_self_pe": 0,
            "route_layers_mask": "",
            "route_filter_warn": 1,
            "line_size_bytes": 64,
            "enable_detailed_map_log": 0,
            "verify_weights": 0,
            "weight_verify_samples": 2,
            "expected_weight_value": 0.0,
            "verify_epsilon": 1e-4,
            "verify_log_each_sample": 0
        })
        
        # 配置内存子组件
        core_memory = core_subcomponent.setSubComponent("memory", "memHierarchy.standardInterface")
        
        # 创建每个核心的L1缓存
        core_l1_cache = sst.Component(f"pe_{i}_core{core_idx}_l1", "memHierarchy.Cache")
        core_l1_cache.addParams({
            "cache_frequency": "2GHz",
            "cache_size": "8KiB",  # 增加缓存大小
            "associativity": "4",
            "cache_line_size": "64",
            "access_latency_cycles": "2",
            "L1": "1",
            "coherence_protocol": "none",
            "debug": "0",
            "verbose": "0"
        })

        # 连接核心的StandardMem接口到L1缓存
        core_mem_link = sst.Link(f"pe_{i}_core{core_idx}_mem")
        core_mem_link.connect(
            (core_memory, "lowlink", "1ns"),
            (core_l1_cache, "highlink", "1ns")
        )

        # 连接核心的L1缓存到当前PE的内存总线
        core_l1_to_bus_link = sst.Link(f"pe_{i}_core{core_idx}_l1_to_pe_bus")
        core_l1_to_bus_link.connect(
            (core_l1_cache, "lowlink", "5ns"),
            (pe_memory_buses[i], f"highlink{core_idx}", "5ns")
        )

    nodes.append(node)
    nics.append(nic)

    print(f"  PE{i} ({layer_name}): 阈值={v_thresh}, 权重地址=0x{weight_addr:x}")

print(f"✅ 创建{len(nodes)}个分层PE节点完成")

# === 创建SpikeSource组件（仅连接到输入层）===
spike_sources = []
for i, pe_id in enumerate(INPUT_LAYER):  # 只为输入层PE创建SpikeSource
    spike_source = sst.Component(f"spike_source_{pe_id}", "SnnDL.SpikeSource")
    spike_source.addParams({
        "verbose": 0,
        "dataset_path": spike_data_files[i],
        "neurons_per_core": NEURONS_PER_CORE,
        "num_cores": NUM_CORES_PER_PE,
        "start_time_us": 2.0 + pe_id * 0.5,  # 错开启动时间
        "loop_dataset": 0,
        "source_id": pe_id
    })
    spike_sources.append((spike_source, pe_id))

print(f"✅ 创建{len(spike_sources)}个SpikeSource（仅连接输入层PE 0-7）")

# === 网络连接 ===
# NIC连接到路由器
for i in range(TOTAL_NODES):
    nic_router_link = sst.Link(f"nic_{i}_to_router_{i}")
    nic_router_link.connect(
        (nics[i], "network", "5ns"),
        (routers[i], "port4", "5ns")
    )

# 建立路由器间连接（8x8 mesh）
connection_count = 0
mesh_size = MESH_SIZE

# 水平连接 (East-West)
for y in range(mesh_size):
    for x in range(mesh_size - 1):
        node_id = y * mesh_size + x
        east_node_id = y * mesh_size + (x + 1)

        router_east_link = sst.Link(f"router_east_{node_id}_to_{east_node_id}")
        router_east_link.connect(
            (routers[node_id], "port0", "5ns"),
            (routers[east_node_id], "port1", "5ns")
        )
        connection_count += 1

# 垂直连接 (North-South)
for x in range(mesh_size):
    for y in range(mesh_size - 1):
        node_id = y * mesh_size + x
        south_node_id = (y + 1) * mesh_size + x

        router_south_link = sst.Link(f"router_south_{node_id}_to_{south_node_id}")
        router_south_link.connect(
            (routers[node_id], "port2", "5ns"),
            (routers[south_node_id], "port3", "5ns")
        )
        connection_count += 1

print(f"✅ 完成{len(nics)}个NIC连接和{connection_count}个路由器连接")

# 连接SpikeSource到输入层PE
for spike_source, pe_id in spike_sources:
    spike_link = sst.Link(f"spike_source_{pe_id}_to_pe_{pe_id}")
    spike_link.connect(
        (spike_source, "spike_output", "5ns"),
        (nodes[pe_id], "external_spike_input", "5ns")
    )

print(f"✅ 完成{len(spike_sources)}个SpikeSource到输入层连接")

# === 统计信息收集 ===
for i, node in enumerate(nodes):
    node.enableStatistics([
        "total_external_spikes_received",
        "total_internal_spikes_processed",
        "total_network_spikes_sent",
        "total_neuron_activations",
        "memory_accesses"
    ])

for i, router in enumerate(routers):
    router.enableStatistics([
        "router.packet_count",
        "router.network_load"
    ])

# === 配置仿真 ===
sst.setProgramOption("timebase", "1ps")
sst.setProgramOption("stop-at", SIMULATION_TIME)

# 统计输出已在文件头部配置

print(f"\n🎯 8x8分层分类网络目标:")
print(f"📊 网络架构: 输入层(8) -> 隐藏层1(16) -> 隐藏层2(16) -> 隐藏层3(16) -> 输出层(8)")
print(f"🔥 输入模式: 8类频率识别 (40Hz-200Hz)")
print(f"🌐 权重连接: 优化分层连接，权重强度递增")
print(f"🧠 神经元阈值: 分层递增阈值(0.02-0.026)")
print(f"📈 期望结果: 8个输出层PE分别激活对应8个类别")

print(f"\n🚀 运行8x8分层分类网络...")
print(f"📡 架构: 8个SpikeSource -> 输入层 -> 3个隐藏层 -> 输出层")
print(f"🎯 目标: 验证8x8分层网络的扩展分类能力")
