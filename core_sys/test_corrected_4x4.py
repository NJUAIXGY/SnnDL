#!/usr/bin/env python3

import sst
import os
import struct

# === 扩展到4x4网格的正确配置 ===

# === 配置参数 ===
MESH_SIZE = 4  # 扩展到4x4网格
NUM_CORES_PER_PE = 4  # 每个PE的core数：4
NEURONS_PER_CORE = 4   # 每个core的神经元数：4
NEURONS_PER_PE = NUM_CORES_PER_PE * NEURONS_PER_CORE  # 每个PE的神经元数：16
TOTAL_NODES = MESH_SIZE * MESH_SIZE  # 16个节点
SIMULATION_TIME = "50us"  # 足够长的仿真时间来处理所有脉冲事件

# 权重内存布局
BASE_WEIGHT_ADDR = 0x10000000
PER_NODE_STRIDE = 32768  # 内存步长以容纳1024个权重（8KB）

# 网络参数
NETWORK_BANDWIDTH = "40GiB/s"
BUFFER_SIZE = "8KiB"

print(f"大规模配置: {MESH_SIZE}x{MESH_SIZE} = {TOTAL_NODES}个节点（{MESH_SIZE}x{MESH_SIZE}网格）")

# === 数据文件创建 ===
test_dir = "/home/anarchy/SST/snnDL_core_system_v2/datasets"
os.makedirs(test_dir, exist_ok=True)

def create_spike_data_file(filename, neuron_id, start_time_us=2.0, duration_us=8.0, rate_hz=100):
    spikes = []
    import random
    random.seed(42 + neuron_id)

    current_time = start_time_us
    end_time = start_time_us + duration_us
    interval_us = 1000000.0 / rate_hz

    while current_time < end_time:
        jitter = random.uniform(-0.2, 0.2) * interval_us
        actual_interval = interval_us + jitter
        current_time += actual_interval
        if current_time < end_time:
            spikes.append(current_time)

    if len(spikes) < 3:
        spikes = [start_time_us + 1.0, start_time_us + 3.0, start_time_us + 5.0, start_time_us + 7.0]
        spikes = [t for t in spikes if t < start_time_us + duration_us]

    with open(filename, 'w') as f:
        f.write("# 神经元ID 时间戳(us)\n")
        for spike_time in spikes:
            timestamp_us = int(spike_time)
            # 每个神经元发送更多脉冲，增加激活机会
            for offset in [0, 1, 2, 5, 8, 10]:
                f.write(f"{neuron_id} {timestamp_us + offset}\n")

    return len(spikes) * 6

def create_cross_node_spike_data(filename, source_node_id, target_neurons):
    with open(filename, 'w') as f:
        f.write("# 神经元ID 时间戳(us)\n")
        for i, neuron_id in enumerate(target_neurons):
            timestamp = 2 + i
            # 每个神经元发送多个脉冲
            for offset in [0, 1, 2, 5, 8, 10]:
                f.write(f"{neuron_id} {timestamp + offset}\n")
    return len(target_neurons) * 6

# 创建脉冲数据文件（为4x4网格的16个PE创建16个SpikeSource）
spike_data_files = []

# 为每个PE创建对应的SpikeSource数据文件
for pe_id in range(TOTAL_NODES):
    # 每个SpikeSource发送到对应PE范围内的神经元 + 一些跨PE的神经元
    start_neuron = pe_id * NEURONS_PER_PE
    end_neuron = (pe_id + 1) * NEURONS_PER_PE - 1
    local_neurons = list(range(start_neuron, end_neuron + 1))

    # 添加跨PE的神经元来激活网络通信（修正索引范围）
    cross_pe_neurons = []
    # 连接到右边PE的神经元（考虑边界）
    if (pe_id + 1) % MESH_SIZE != 0:  # 不在行末尾
        right_pe = pe_id + 1
        right_start = right_pe * NEURONS_PER_PE
        cross_pe_neurons.extend([right_start, right_start + 1])

    # 连接到下边PE的神经元（考虑边界）
    if pe_id + MESH_SIZE < TOTAL_NODES:  # 不在底部行
        down_pe = pe_id + MESH_SIZE
        down_start = down_pe * NEURONS_PER_PE
        cross_pe_neurons.extend([down_start, down_start + 1])

    # 连接到对角PE的神经元（考虑边界）
    if (pe_id + 1) % MESH_SIZE != 0 and pe_id + MESH_SIZE + 1 < TOTAL_NODES:
        diag_pe = pe_id + MESH_SIZE + 1
        diag_start = diag_pe * NEURONS_PER_PE
        cross_pe_neurons.extend([diag_start])

    target_neurons = local_neurons + cross_pe_neurons

    spike_file = os.path.join(test_dir, f"4x4_spike_data_source_{pe_id}.txt")
    spike_count = create_cross_node_spike_data(spike_file, pe_id, target_neurons)
    spike_data_files.append(spike_file)
    print(f"  SpikeSource{pe_id}: 本地[{start_neuron}-{end_neuron}] + 跨PE神经元, {spike_count}个脉冲")

# 创建权重文件（为16个PE创建权重文件）
for node_id in range(TOTAL_NODES):
    weight_file = os.path.join(test_dir, f"4x4_weights_node_{node_id}.bin")

    # 创建权重矩阵，使用更高的权重值以便神经元激活
    weights = [1.0] * (NEURONS_PER_PE * (TOTAL_NODES * NEURONS_PER_PE))

    with open(weight_file, 'wb') as f:
        for w in weights:
            f.write(struct.pack('f', w))

    print(f"  节点{node_id}: {NEURONS_PER_PE}x{TOTAL_NODES * NEURONS_PER_PE}权重矩阵")

# === 全局内存系统 ===
global_memory_controller = sst.Component("global_memory_controller", "memHierarchy.MemController")
global_memory_controller.addParams({
    "clock": "1GHz",
    "backing": "malloc",
    "backend": "memHierarchy.simpleMem",
    "backend.access_time": "100ns",
    "backend.mem_size": "1GiB",
    "addr_range_start": "0",
    "addr_range_end": "1073741823"
})

# WeightLoader配置
weight_loader = sst.Component("weight_loader", "SnnDL.WeightLoader")
weight_loader.addParams({
    "verbose": 2,  # 增加详细日志
    "base_addr_start": BASE_WEIGHT_ADDR,
    "per_core_stride": PER_NODE_STRIDE,
    "num_cores": TOTAL_NODES,
    "neurons_per_core": NEURONS_PER_CORE,
    "total_neurons": TOTAL_NODES * NEURONS_PER_PE,
    "weight_format": "bin",
    "per_core_files": 1,
    "file_template": os.path.join(test_dir, "4x4_weights_node_{core}.bin"),
    "fill_value": 0.0,
    "validate_length": 1,
    "row_major": 1
})

weight_loader_mem = weight_loader.setSubComponent("memory", "memHierarchy.standardInterface")
weight_loader_mem.addParams({"port": "lowlink"})

weight_loader_link = sst.Link("weight_loader_to_global_mem")
weight_loader_link.connect(
    (weight_loader_mem, "lowlink", "5ns"),
    (global_memory_controller, "highlink", "5ns")
)

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
        "input_latency": "10ns",  # 增加延迟以提高稳定性
        "output_latency": "10ns", # 增加延迟以提高稳定性
        "input_buf_size": "4KiB",  # 增加缓冲区大小
        "output_buf_size": "4KiB", # 增加缓冲区大小
        "num_vns": 1,             # 单虚拟网络
        "xbar_arb": "merlin.xbar_arb_lru",   # 使用LRU仲裁器提高稳定性
        "debug": 0,
        "verbose": 0,
        "network_inspectors": "",
    })

    # 配置mesh拓扑 - 修正配置
    topo = router.setSubComponent("topology", "merlin.mesh")
    topo.addParams({
        "shape": f"{MESH_SIZE}x{MESH_SIZE}",
        "width": "1x1",           # 添加width参数
        "local_ports": "1",      # 1个本地端口用于连接PE
    })

    routers.append(router)

print(f"✅ 创建{len(routers)}个路由器完成")

# === 创建PE节点（扩展到16个）===
nodes = []
nics = []

for i in range(TOTAL_NODES):
    node = sst.Component(f"multicore_pe_{i}", "SnnDL.MultiCorePE")

    node_params = {
        "verbose": 2,
        "num_cores": NUM_CORES_PER_PE,
        "neurons_per_core": NEURONS_PER_CORE,
        "total_neurons": TOTAL_NODES * NEURONS_PER_PE,
        "node_id": i,
        "global_neuron_base": i * NEURONS_PER_PE,
        "enable_test_traffic": 1,
        "enable_memory_weights": 1,
        "write_weights_on_init": 1,
        "weights_file": os.path.join(test_dir, f"4x4_weights_node_{i}.bin"),
        # StandardMem接口通过子组件配置
        "v_thresh": 0.1,  # 降低神经元阈值
        "v_rest": 0.0,
        "v_reset": 0.0,
        "use_event_weight_fallback": 1,  # 启用事件权重回退
        "event_weight_fallback": 0.5,    # 设置事件权重回退值
        # 权重验证参数
        "verify_weights": 1,             # 启用权重验证
        "weight_verify_samples": 8,      # 验证8个权重样本
        "expected_weight_value": 1.0,    # 期望的权重值（与权重文件中的值匹配）
        "verify_log_each_sample": 1,     # 记录每个验证样本
        "memory_warmup_cycles": 100,     # 减少暖机时间
        "enable_weight_fetch": 1         # ★ 关键：启用从内存获取权重 ★
    }

    # 计算权重地址（确保地址不重叠）
    weight_addr = BASE_WEIGHT_ADDR + i * PER_NODE_STRIDE
    node_params["base_addr"] = weight_addr

    node.addParams(node_params)

    # 创建SnnNIC网络接口
    nic = node.setSubComponent("network_interface", "SnnDL.SnnNIC")
    nic.addParams({
        "node_id": str(i),
        "link_bw": NETWORK_BANDWIDTH,
        "input_buf_size": BUFFER_SIZE,
        "output_buf_size": BUFFER_SIZE,
        "use_direct_link": "false",  # 使用标准网络模式
        "port_name": "network",
        "verbose": 1,  # 标准日志级别
        "total_nodes": TOTAL_NODES,  # ★ 添加total_nodes参数用于修正job_size
    })

    # 为每个核心创建L1缓存和内存控制器
    for core_idx in range(NUM_CORES_PER_PE):
        # 创建内存控制器
        mem_ctrl = sst.Component(f"pe_{i}_core{core_idx}_mem_ctrl", "memHierarchy.MemController")
        mem_ctrl.addParams({
            "clock": "2GHz",
            "backing": "malloc",
            "backend": "memHierarchy.simpleMem",
            "backend.access_time": "30ns",
            "backend.mem_size": "8MiB",
            "addr_range_start": "0",
            "addr_range_end": "8388607"
        })

        # ★ 关键修正：创建L1缓存，使用正确的memHierarchy配置
        l1_cache = sst.Component(f"pe_{i}_core{core_idx}_l1", "memHierarchy.Cache")
        l1_cache.addParams({
            "cache_frequency": "2GHz",
            "cache_size": "4KiB",
            "associativity": "4",
            "cache_line_size": "64",
            "access_latency_cycles": "2",
            "L1": "1",
            "coherence_protocol": "none",
            "debug": "0",
            "verbose": "0"
        })

        # ★ 关键修正：直接连接到MultiCorePE的核心内存端口 ★
        # MultiCorePE中使用的端口名格式：core0_mem, core1_mem, core2_mem, core3_mem
        core_mem_link = sst.Link(f"pe_{i}_core{core_idx}_mem")
        core_mem_link.connect(
            (node, f"core{core_idx}_mem", "1ns"),
            (l1_cache, "highlink", "1ns")
        )

        # 连接L1缓存到内存控制器
        l1_to_mem_link = sst.Link(f"pe_{i}_core{core_idx}_l1_to_mem")
        l1_to_mem_link.connect(
            (l1_cache, "lowlink", "5ns"),
            (mem_ctrl, "highlink", "5ns")
        )

    # 只在所有核心配置完成后添加一次node和nic
    nodes.append(node)
    nics.append(nic)

    print(f"  节点{i}: 外部SpikeSource模式，权重地址0x{weight_addr:x}")

print(f"✅ 创建{len(nodes)}个节点和{len(nics)}个NIC完成（{MESH_SIZE}x{MESH_SIZE}网格）")

# === 创建SpikeSource组件（扩展到16个）===
spike_sources = []
for source_id in range(TOTAL_NODES):
    spike_source = sst.Component(f"spike_source_{source_id}", "SnnDL.SpikeSource")
    spike_source.addParams({
        "verbose": 2,
        "dataset_path": spike_data_files[source_id],
        "neurons_per_core": NEURONS_PER_CORE,
        "start_time_us": 1.0 + (source_id % 4) * 0.5,  # 错开启动时间
        "loop_dataset": 1,
        "source_id": source_id
    })
    spike_sources.append(spike_source)

print(f"✅ 创建{len(spike_sources)}个SpikeSource完成（{MESH_SIZE}x{MESH_SIZE}网格）")

# === 内存系统配置 ===
# WeightLoader连接到全局内存控制器（已配置）
# 每个PE的L1缓存直接连接到各自的内存控制器（已在PE循环中配置）

# === 网络连接 ===
# NIC连接到路由器（本地端口）
for i in range(TOTAL_NODES):
    nic_router_link = sst.Link(f"nic_{i}_to_router_{i}")
    nic_router_link.connect(
        (nics[i], "network", "5ns"),
        (routers[i], "port4", "5ns")  # 本地端口
    )

# 建立路由器间连接（4x4 mesh）
connection_count = 0
mesh_size = MESH_SIZE

# 水平连接 (East-West)
for y in range(mesh_size):
    for x in range(mesh_size - 1):
        node_id = y * mesh_size + x
        east_node_id = y * mesh_size + (x + 1)

        router_east_link = sst.Link(f"router_east_{node_id}_to_{east_node_id}")
        router_east_link.connect(
            (routers[node_id], "port0", "5ns"),      # East port
            (routers[east_node_id], "port1", "5ns")  # West port
        )
        connection_count += 1

# 垂直连接 (North-South)
for x in range(mesh_size):
    for y in range(mesh_size - 1):
        node_id = y * mesh_size + x
        south_node_id = (y + 1) * mesh_size + x

        router_south_link = sst.Link(f"router_south_{node_id}_to_{south_node_id}")
        router_south_link.connect(
            (routers[node_id], "port2", "5ns"),       # South port
            (routers[south_node_id], "port3", "5ns")  # North port
        )
        connection_count += 1

print(f"✅ 完成{len(nics)}个NIC连接和{connection_count}个路由器连接")

# 连接多个SpikeSource到对应PE，测试跨核通信
for i in range(TOTAL_NODES):
    spike_link = sst.Link(f"spike_source_{i}_to_pe_{i}")
    spike_link.connect(
        (spike_sources[i], "spike_output", "5ns"),
        (nodes[i], "external_spike_input", "5ns")
    )

# 完成路由器网络连接配置
print(f"✅ 完成路由器间网络连接：{connection_count}个连接")

# === 添加统计信息收集 ===
# 为MultiCorePE添加统计输出
for i, node in enumerate(nodes):
    node.enableStatistics([
        "total_external_spikes_received",
        "total_internal_spikes_processed",
        "total_network_spikes_sent",
        "total_neuron_activations",
        "memory_accesses"
    ])

# 为路由器添加统计输出
for i, router in enumerate(routers):
    router.enableStatistics([
        "router.packet_count",
        "router.network_load"
    ])

# === 配置仿真 ===
sst.setProgramOption("timebase", "1ps")
sst.setProgramOption("stop-at", SIMULATION_TIME)

print(f"📊 启用统计信息收集：{len(nodes)}个PE + {len(routers)}个路由器")
print(f"🎯 跨核通信测试：{TOTAL_NODES}个SpikeSource -> {TOTAL_NODES}个PE")

# 启用统计
sst.setStatisticLoadLevel(5)
sst.enableAllStatisticsForComponentType("SnnDL.MultiCorePE")
sst.enableAllStatisticsForComponentType("SnnDL.SpikeSource")
sst.enableAllStatisticsForComponentType("merlin.hr_router")

print(f"\n🎯 {MESH_SIZE}x{MESH_SIZE}网格目标:")
print(f"1. ✅ WeightLoader预加载权重矩阵到内存")
print(f"2. 🔥 {TOTAL_NODES}个SpikeSource直接连接到对应PE的external_spike_input端口")
print(f"3. 🌐 {TOTAL_NODES}个PE间通过路由器网络通信")
print(f"4. 🧠 PE从内存读取权重处理脉冲")
print(f"5. 📊 验证4cores分层存储和跨PE权重连接")
print(f"6. 🎯 验证{MESH_SIZE}x{MESH_SIZE}网格架构的正确性")

print(f"\n🚀 运行{MESH_SIZE}x{MESH_SIZE}网格系统...")
print(f"📡 架构: {TOTAL_NODES}个SpikeSource→{TOTAL_NODES}个PE + {len(routers)}个路由器网络 + 4cores分层存储")
print(f"🎯 目标: 验证{MESH_SIZE}x{MESH_SIZE}网格的高性能分布式SNN架构")