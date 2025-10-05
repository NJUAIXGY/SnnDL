#!/usr/bin/env python3

import sst
import os

# 8x8 分层网络分类任务（在 4x4 基础上扩展，保持最小改动与稳定性）

# 统计输出（必须在创建任何组件之前）
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

# 网络架构
MESH_SIZE = 8  # 8x8 网格
NUM_CORES_PER_PE = 4
NEURONS_PER_CORE = 4
NEURONS_PER_PE = NUM_CORES_PER_PE * NEURONS_PER_CORE  # 16
TOTAL_NODES = MESH_SIZE * MESH_SIZE  # 64
SIMULATION_TIME = "2000us"

# 分层：按 16 节点一个层展开（与 4x4 兼容的直观扩展）
INPUT_LAYER = list(range(0, 16))
HIDDEN_LAYER_1 = list(range(16, 32))
HIDDEN_LAYER_2 = list(range(32, 48))
OUTPUT_LAYER = list(range(48, 64))

# 驱动脉冲仅对前 4 个 PE（保留原 4 类输入，避免缺数据文件）
INPUT_DRIVER_PES = [0, 1, 2, 3]

# 分类任务参数（与 4x4 相同）
NUM_CLASSES = 4
CLASS_A_FREQ = 40
CLASS_B_FREQ = 80
CLASS_C_FREQ = 120
CLASS_D_FREQ = 200

# 权重与内存
BASE_WEIGHT_ADDR = 0x0
PER_NODE_STRIDE = 65536  # 16(rows)*1024(cols)*4B = 65536B，按 8x8 列宽调整
NETWORK_BANDWIDTH = "40GiB/s"
BUFFER_SIZE = "8KiB"

print(f"🧠 扩展分层网络: {MESH_SIZE}x{MESH_SIZE} = {TOTAL_NODES} 个节点")

weights_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "weights")
spike_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "spike_data")

# 加载仅 4 个输入驱动文件（与现有数据对齐）
spike_data_files = []
class_names = ['A', 'B', 'C', 'D']
freqs = [CLASS_A_FREQ, CLASS_B_FREQ, CLASS_C_FREQ, CLASS_D_FREQ]
for pe_id in INPUT_DRIVER_PES:
    spike_file = os.path.join(spike_dir, f"complex_input_pe_{pe_id}_class_{class_names[pe_id]}.txt")
    if not os.path.exists(spike_file):
        print(f"❌ 错误: 脉冲数据文件不存在: {spike_file}")
        print(f"请先运行: python3 scripts/generate_spike_data.py")
        raise SystemExit(1)
    spike_data_files.append(spike_file)
    with open(spike_file, 'r') as f:
        lines = [line for line in f.readlines() if not line.startswith('#')]
    print(f"  ✅ 加载PE{pe_id}: 类别{class_names[pe_id]} ({freqs[pe_id]}Hz), {len(lines)}个脉冲事件")

# 为每个 PE 创建内存控制器 + 总线 + 权重加载器
pe_memory_controllers = []
pe_weight_loaders = []
pe_memory_buses = []

for pe_id in range(TOTAL_NODES):
    mem_controller = sst.Component(f"pe_{pe_id}_memory_controller", "memHierarchy.MemController")
    mem_controller.addParams({
        "clock": "1GHz",
        "backing": "malloc",
        "addr_range_start": "0",
        "addr_range_end": "67108863"  # 64MB
    })
    mem_backend = mem_controller.setSubComponent("backend", "memHierarchy.simpleMem")
    mem_backend.addParams({"access_time": "100ns", "mem_size": "64MiB"})
    pe_memory_controllers.append(mem_controller)

    mem_bus = sst.Component(f"pe_{pe_id}_memory_bus", "memHierarchy.Bus")
    mem_bus.addParams({"bus_frequency": "1GHz", "debug": "0", "verbose": "0"})
    pe_memory_buses.append(mem_bus)

    bus_to_mem_link = sst.Link(f"pe_{pe_id}_bus_to_mem")
    bus_to_mem_link.connect((mem_bus, "lowlink0", "5ns"), (mem_controller, "highlink", "5ns"))

    weight_loader = sst.Component(f"pe_{pe_id}_weight_loader", "SnnDL.WeightLoader")
    weight_loader.addParams({
        "verbose": 0,
        "base_addr_start": BASE_WEIGHT_ADDR,
        "per_core_stride": PER_NODE_STRIDE,
        "num_cores": 1,
        "neurons_per_core": NEURONS_PER_CORE,
        "rows_per_core": NEURONS_PER_PE,
        "cols_per_core": TOTAL_NODES * NEURONS_PER_PE,
        "total_neurons": NEURONS_PER_PE,
        "weight_format": "bin",
        "per_core_files": 1,
        "file_template": os.path.join(weights_dir, f"classification_weights_pe_{pe_id}.bin"),
        "fill_value": 0.0,
        "validate_length": 0,   # 8x8 下多数PE无文件，避免噪声
        "row_major": 1
    })
    weight_loader_mem = weight_loader.setSubComponent("memory", "memHierarchy.standardInterface")
    weight_loader_mem.addParams({"port": "lowlink"})
    weight_loader_link = sst.Link(f"pe_{pe_id}_weight_loader_to_bus")
    weight_loader_link.connect((weight_loader_mem, "lowlink", "5ns"), (mem_bus, "highlink4", "5ns"))
    pe_weight_loaders.append(weight_loader)

print(f"✅ 创建{len(pe_memory_controllers)}个PE内存子系统")

# 创建 8x8 路由器
routers = []
for i in range(TOTAL_NODES):
    router = sst.Component(f"router_{i}", "merlin.hr_router")
    router.addParams({
        "id": i,
        "num_ports": 5,
        "link_bw": NETWORK_BANDWIDTH,
        "flit_size": "8B",
        "xbar_bw": NETWORK_BANDWIDTH,
        "input_latency": "10ns",
        "output_latency": "10ns",
        "input_buf_size": "4KiB",
        "output_buf_size": "4KiB",
        "num_vns": 1,
        "xbar_arb": "merlin.xbar_arb_lru",
        "debug": 0,
        "verbose": 0,
    })
    topo = router.setSubComponent("topology", "merlin.mesh")
    topo.addParams({"shape": f"{MESH_SIZE}x{MESH_SIZE}", "width": "1x1", "local_ports": "1"})
    routers.append(router)

print(f"✅ 创建{len(routers)}个路由器完成")

# 创建 64 个 PE 节点
nodes = []
nics = []
for i in range(TOTAL_NODES):
    node = sst.Component(f"multicore_pe_{i}", "SnnDL.MultiCorePE")

    # 分层阈值（扩展规则）
    if i in INPUT_LAYER:
        v_thresh = 0.02
    elif i in HIDDEN_LAYER_1:
        v_thresh = 0.03
    elif i in HIDDEN_LAYER_2:
        v_thresh = 0.03
    else:
        v_thresh = 0.035

    print_node_summary_val = 1 if i in OUTPUT_LAYER else 0

    node_params = {
        "verbose": 0,
        "print_node_summary": print_node_summary_val,
        "num_cores": NUM_CORES_PER_PE,
        "neurons_per_core": NEURONS_PER_CORE,
        "total_neurons": TOTAL_NODES * NEURONS_PER_PE,
        "node_id": i,
        "global_neuron_base": i * NEURONS_PER_PE,
        "enable_test_traffic": 0,
        "loop_dataset": 0,
        "enable_memory_weights": 1,
        "write_weights_on_init": 1,
        "weights_file": os.path.join(weights_dir, f"classification_weights_pe_{i}.bin"),
        "v_thresh": v_thresh,
        "v_rest": 0.0,
        "v_reset": 0.0,
        "use_event_weight_fallback": 0,
        "verify_weights": 0,              # 扩展规模下关闭冗余验证
        "memory_warmup_cycles": 100,
        "enable_weight_fetch": 1,
        "memory_weight_priority": 1,
    }
    node_params["base_addr"] = BASE_WEIGHT_ADDR
    node.addParams(node_params)

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

    # 4 个核心子组件
    for core_idx in range(NUM_CORES_PER_PE):
        core = node.setSubComponent(f"core{core_idx}", "SnnDL.SnnPESubComponent")
        core.addParams({
            "core_id": core_idx,
            "total_cores": NUM_CORES_PER_PE,
            "global_neuron_base": i * NEURONS_PER_PE,
            "num_neurons": NUM_CORES_PER_PE * NEURONS_PER_CORE,
            "v_thresh": v_thresh,
            "v_reset": 0.0,
            "v_rest": 0.0,
            "tau_mem": 20.0,
            "t_ref": 2,
            "base_addr": BASE_WEIGHT_ADDR,
            "node_id": i,
            "verbose": 0,
            "enable_weight_fetch": 1,
            "write_weights_on_init": 1,
            "memory_warmup_cycles": 100,
            "init_default_weight": 0.5,
            "max_outstanding_requests": 16,
            "max_cache_entries": 4096,
            "use_event_weight_fallback": 0,
            "merge_read_cacheline": 1,
            "merge_read_row": 0,
            # 全网读取：行=post_local(0..15), 列=pre_global(0..{TOTAL_NODES*16-1})
            "weights_cols": TOTAL_NODES * NEURONS_PER_PE,
            "index_mode": "post_row_pre_col",
            "line_size_bytes": 64,
            # 扫描文件构建路由（保持与4x4一致），禁用层掩码以规避v2固定层划分
            "routing_mode": "weight_driven",
            "weights_template": os.path.join(weights_dir, "classification_weights_pe_{pe}.bin"),
            "total_nodes": TOTAL_NODES,
            "routing_epsilon": 0.6,
            "routing_topk_per_pe": 2,
            "routing_topk": 12,
            "route_exclude_self_pe": 0,
            "route_layers_mask": "",
            "route_filter_warn": 1,
            # 关闭文件比对，避免64个PE下的缺失告警
            "verify_against_file": 0,
            "verify_weights": 0,
        })

        # memory 子组件 + L1 缓存
        core_mem = core.setSubComponent("memory", "memHierarchy.standardInterface")
        core_l1 = sst.Component(f"pe_{i}_core{core_idx}_l1", "memHierarchy.Cache")
        core_l1.addParams({
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
        link_core_mem = sst.Link(f"pe_{i}_core{core_idx}_mem")
        link_core_mem.connect((core_mem, "lowlink", "1ns"), (core_l1, "highlink", "1ns"))
        link_l1_bus = sst.Link(f"pe_{i}_core{core_idx}_l1_to_pe_bus")
        link_l1_bus.connect((core_l1, "lowlink", "5ns"), (pe_memory_buses[i], f"highlink{core_idx}", "5ns"))

    nodes.append(node)
    nics.append(nic)

print(f"✅ 创建{len(nodes)}个PE节点完成")

# SpikeSource（仅连接到前4个输入PE）
spike_sources = []
for idx, pe_id in enumerate(INPUT_DRIVER_PES):
    spike_source = sst.Component(f"spike_source_{pe_id}", "SnnDL.SpikeSource")
    spike_source.addParams({
        "verbose": 0,
        "dataset_path": spike_data_files[idx],
        "neurons_per_core": NEURONS_PER_CORE,
        "start_time_us": 2.0 + pe_id * 0.5,
        "loop_dataset": 0,
        "source_id": pe_id
    })
    spike_sources.append((spike_source, pe_id))

print(f"✅ 创建{len(spike_sources)}个SpikeSource（驱动PE {INPUT_DRIVER_PES}）")

# 网络连接
for i in range(TOTAL_NODES):
    link_nic_router = sst.Link(f"nic_{i}_to_router_{i}")
    link_nic_router.connect((nics[i], "network", "5ns"), (routers[i], "port4", "5ns"))

connection_count = 0
for y in range(MESH_SIZE):
    for x in range(MESH_SIZE - 1):
        a = y * MESH_SIZE + x
        b = y * MESH_SIZE + (x + 1)
        link = sst.Link(f"router_east_{a}_to_{b}")
        link.connect((routers[a], "port0", "5ns"), (routers[b], "port1", "5ns"))
        connection_count += 1
for x in range(MESH_SIZE):
    for y in range(MESH_SIZE - 1):
        a = y * MESH_SIZE + x
        b = (y + 1) * MESH_SIZE + x
        link = sst.Link(f"router_south_{a}_to_{b}")
        link.connect((routers[a], "port2", "5ns"), (routers[b], "port3", "5ns"))
        connection_count += 1

for spike_source, pe_id in spike_sources:
    link = sst.Link(f"spike_source_{pe_id}_to_pe_{pe_id}")
    link.connect((spike_source, "spike_output", "5ns"), (nodes[pe_id], "external_spike_input", "5ns"))

print(f"✅ 完成{len(nics)}个NIC连接和{connection_count}个路由器连接")

# 统计
for node in nodes:
    node.enableStatistics([
        "total_external_spikes_received",
        "total_internal_spikes_processed",
        "total_network_spikes_sent",
        "total_neuron_activations",
        "memory_accesses"
    ])
for router in routers:
    router.enableStatistics(["router.packet_count", "router.network_load"])

# 仿真配置
sst.setProgramOption("timebase", "1ps")
sst.setProgramOption("stop-at", SIMULATION_TIME)

print("\n🚀 运行 8x8 分层分类网络（4 输入驱动，权重内存加载，按权重驱动扇出）...")

