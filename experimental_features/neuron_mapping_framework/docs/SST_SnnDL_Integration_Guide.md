# SnnDL × 神经元映射框架 集成详细指南（预处理式 + 可选Merlin插件）

本文面向使用 SnnDL + Merlin 网络的项目，提供“离线映射、核心分配、路由生成、产物导出、SnnDL脚本生成与（可选）Merlin路由器集成”的极为详细的对接说明。目标是在不修改 SnnDL 核心组件的前提下，以最小侵入方式获得高效的多播/单播映射与路由；并在需要时进一步把路由表下沉到 merlin.hr_router 中，实现更硬件化的 CAM 多播。

---

## 1. 总览与基本概念

- 预处理式工作流（推荐，零侵入 SnnDL）：
  1) 构建 `NeuralNetwork`（或从现有SnnDL网络提取）
  2) 构建 `HardwareTopology`（如 8×8 PE）
  3) 图分割映射 → 生成 `MappingSolution`（神经元→PE）
  4) 按层次结构分配 `core_id`（每PE 8 核，每核 ≤128 神经元）
  5) 路由生成（多播树 + 连续LSB前缀聚合）
  6) 导出产物（JSON/CSV/报告）
  7) 依据 `mapping.json` 分片权重并生成优化后的 SnnDL Python 脚本
  8) 使用 SNNDL 运行新的配置

- 层次结构（当前默认）：
  - 拓扑：8×8 PEs
  - 每个 PE：8 个核心（cores）
  - 每核心：最多支持 128 个神经元
  - AER 32位键编码：`local_id:16 | pe_id:12 | core_id:4`（local_id 为 per-core 0..127 的索引）

- 多播/单播：
  - 自动探测扇出≥3的源，构建近似Steiner的多播树；其余走最短路单播
  - 路由表项采用 (key, mask, routes[]) 三元组；routes 为方向集合（LOCAL/N/E/S/W/UP/DOWN）
  - 连续LSB前缀聚合：仅在标准前缀语义下合并（保守，避免误匹配）

---

## 2. 目录与构建

- 核心目录
  - `experimental_features/neuron_mapping_framework/include`：公共头文件（core/、routing/ 等）
  - `experimental_features/neuron_mapping_framework/src`：实现源码
  - `experimental_features/neuron_mapping_framework/tools`：工具（示例导出、小规模/大规模实验）
  - `experimental_features/neuron_mapping_framework/tests`：测试
  - `experimental_features/neuron_mapping_framework/docs`：文档（本文件、技术说明等）

- 构建与自检
```bash
cd experimental_features/neuron_mapping_framework
make test-compile       # 编译所有源码
make run-export-test    # 小型导出自测（export_test/ 产物）
make run-large          # 大规模实验（65536/64）仅输出汇总（export_large/）
```

---

## 3. 公共 API 与工具

### 3.1 RoutingTableGenerator::exportArtifacts（公共API）
- 位置：`include/routing/RoutingTableGenerator.h`
- 功能：生成路由表并导出产物，用于 SnnDL 预处理集成
- 签名：
```cpp
bool exportArtifacts(
  const NeuralNetwork& network,
  const HardwareTopology& topology,
  const MappingSolution& mapping,
  const std::string& out_dir,
  const RoutingGenerationConfig& config = RoutingGenerationConfig());
```
- 产物（写入 `out_dir/`）：
  - `mapping.json`：神经元→{pe_id, core_id}
  - `routing_tables.json`：每PE路由表（JSON，体量大时不建议大规模导出）
  - `routing_stats.csv`：PE表项统计（`pe_id,entries`）
  - `report.json`：总体指标（见 §4）

### 3.2 小工具
- `tools/routing_exporter`：小型演示导出工具（export/）
- `tools/routing_experiment_large`：大规模实验（export_large/），仅输出汇总（避免超大JSON）

---

## 4. 数据格式（集成必读）

### 4.1 mapping.json（核心对接）
- 结构：
```json
{
  "assignments": [
    {"neuron_id": <uint>, "pe_id": <uint>, "core_id": <uint>},
    ...
  ]
}
```
- 说明：
  - `neuron_id`：全局神经元ID（0..N-1）
  - `pe_id`：目标PE（0..PEs-1）
  - `core_id`：该PE内核心编号（0..7）。每核心 ≤128 个神经元

### 4.2 routing_tables.json（可选，诊断/可视化）
- 结构：
```json
{
  "pe_<id>": {
    "entries": [
      {"key": <uint32>, "mask": <uint32>, "priority": <uint16>, "routes": [<int> ...]},
      ...
    ]
  },
  ...
}
```
- 说明：
  - `routes` 中 0=LOCAL, 1=NORTH, 2=SOUTH, 3=EAST, 4=WEST, 5=UP, 6=DOWN
  - CAM 匹配规则：`(key & mask) == (entry.key & mask)`；按最长前缀/priority 选择

### 4.3 routing_stats.csv
```
pe_id,entries
0,16
1,12
...
```

### 4.4 report.json（总览）
- 字段：
  - `total_pes`、`total_entries`、`average_path_length`、`multicast_groups`
  - `compression_ratio`：压缩比（前缀聚合后 / 前）
  - `avg_entries_per_pe`、`max_entries_per_pe`

---

## 5. SnnDL 预处理式集成（推荐）

### 5.1 总体思路
- 离线执行映射/路由 → 导出 `mapping.json`
- 依据 `mapping.json` 分片权重，生成每PE（或每核）权重文件
- 生成优化后的 SnnDL Python 脚本：为每个 SnnPE 组件设置：
  - `num_neurons`（=该PE的神经元数量）
  - `node_id`（=PE id）
  - `weights_file`（指向分片权重文件）
- SnnDL/merlin.hr_router 保持原有最短路拓扑；多播暂在 NIC 层（NetworkAdapter）拆单播（若需要）

### 5.2 分片权重脚本（示例伪代码）
```python
# 假设原始权重为稀疏CSR或二进制矩阵，需要按 mapping.json 的 assignments 切分
import json
from pathlib import Path

def load_mapping(path):
    m = json.loads(Path(path).read_text())
    per_pe = {}
    for a in m["assignments"]:
        per_pe.setdefault(a["pe_id"], []).append(a["neuron_id"])
    return per_pe

def slice_weights(original_weights, per_pe, out_dir):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    for pe, ids in per_pe.items():
        ids_sorted = sorted(ids)
        # TODO: 从 original_weights 提取 ids_sorted 子矩阵/索引
        #       保存为 out / f"pe_{pe}.bin"
        pass

per_pe = load_mapping("export/mapping.json")
slice_weights("weights.bin", per_pe, "weights_sliced")
```

### 5.3 生成 SnnDL 配置脚本（示例伪代码）
```python
import json
import sst

mapping = json.load(open("export/mapping.json"))
per_pe = {}
for a in mapping["assignments"]:
    per_pe.setdefault(a["pe_id"], []).append(a["neuron_id"])

# 生成 8x8 节点
for pe in range(64):
    comp = sst.Component(f"snnpe_{pe}", "SnnDL.SnnPE")
    comp.addParams({
        "num_neurons": len(per_pe.get(pe, [])),
        "node_id": pe,
        "weights_file": f"weights_sliced/pe_{pe}.bin"
    })
    # TODO: 接入 NIC / linkcontrol / hr_router 拓扑（沿用你现有脚本）
```

### 5.4 NIC 层“多播拆单播”（可选）
- 若 NIC 侧可以接入 `mapping.json`，则在发送脉冲时将多播拆为多个目的 PE 的单播包；Merlin hr_router 做最短路转发
- 优点：零改 Merlin；代价：复制在 NIC 端完成

---

## 6. Merlin hr_router 集成（可选加强）

### 6.1 方案B1：拓扑插件（推荐）
- 新增 `merlin.topology.aer_static`：
  - 载入 `{router_id: entries[]}` 的路由表文件（我们可导出 `aer_router_tables.json`）
  - 提供接口：给定包头中携带的 `key` 字段，返回“多端口”集合（需要拓扑或 hr_router 支持一次返回多个 outport）
- SnnDL Python：
```python
hr = sst.Component("rtr_12", "merlin.hr_router")
hr.addParams({
  "topology": "merlin.topology.aer_static",
  "route_file": "aer_router_tables.json"
})
```
- 导出适配：
  - 将我们的 `RouteDirection` 映射为 hr_router 的端口号；LOCAL → 本地端口/NIC
  - `aer_router_tables.json` 样例：
```json
{
  "router_12": {
    "entries": [
      {"key": 305419896, "mask": 4294967295, "ports": [1,3]},
      {"key": 305419904, "mask": 4294967280, "ports": [0]}
    ]
  }
}
```
- 封包：在 NIC 封包时把 32 位 `key` 写入包头（拓扑插件据此匹配）

### 6.2 方案B2：hr_router 路由算法插件
- 实现 `routing_alg = "aer_table"` 的派生或插件，在 hr_router 内加载表，自己复制转发
- 工作量 > B1；需核对 Merlin 插件扩展能力（ELI 注册等）

---

## 7. 性能与参数建议
- 多播树启用：`RoutingGenerationConfig.enable_multicast = true`
- 前缀聚合启用：`enable_compression = true; compression = PREFIX_AGGREGATION`
- 局部对齐重编码（已内置）：按 routes 模式分组 + 2^t 对齐块，提升聚合命中
- 大规模时仅导出汇总：使用 `routing_experiment_large`，避免巨型 JSON
- 验证建议：
  - `report.json` 中 `average_path_length` 合理（8×8 Mesh 通常 ~4–8）
  - `total_entries`、`avg/max_entries_per_pe` 规模适中；与 `routing_stats.csv` 一致

---

## 8. 常见问题与排查
- 压缩比仍为1：
  - 随机网络 routes 模式分散，难以形成成对聚合 → 可提高分组粒度或上提压缩
- 表项偏多：
  - 扇出极大时，多播树可显著减少；必要时提高聚合位数或采用更强策略
- SnnDL 侧如何核对 core：
  - 当前 SnnDL 参数不暴露 core；我们在 AER 键与导出里维护 core_id，主要供路由器/诊断使用

---

## 9. 一键指令与验证流程（总结）
- 小规模验证：
```bash
cd experimental_features/neuron_mapping_framework
make run-export-test            # export_test/ 下四个产物
```
- 大规模统计（65536/64）：
```bash
cd neuron_mapping_framework
make run-large                  # export_large/ 下 report.json、stats、timings
```
- 你的 SnnDL 集成：
  1) 用项目网络生成/提取 NeuralNetwork；调用 exportArtifacts 导出 mapping.json
  2) 分片权重到 per-PE 文件
  3) 生成 optimized_config.py（每 SnnPE 设置 num_neurons/node_id/weights_file）
  4) sst optimized_config.py

---

## 10. 后续演进（可选）
- 路由感知分割：在 refine 阶段加入路由成本（跳数/热点），减少跨PE扇出
- 多播树“上提”压缩：直接减少重复子树上的条目（不改键）
- Merlin 插件：落地 `aer_static`，实现路由器级别多播复制
- 报告扩展：per-core 表项、热点链路 Top-K、端口带宽估算等

---

如需，我可以提供：
- generate_optimized_sst.py 的参考实现（读取 mapping.json，分片你的权重，生成 SnnDL 脚本）
- aer_router_tables.json 的导出适配与 merlin.topology.aer_static 的C++骨架
- 针对你的实际网络/权重格式的抽取与分片脚本
