# SnnDL（SST Element：Spiking Neural Network / Deep Learning）

本目录是 `sst-elements` 中的 **SnnDL 元素库源码**，用于在 SST 中模拟多核 Processing Element（PE）上的工作负载：默认提供 SNN（Spike/GAS/BCSR/Step）链路，同时也支持以 **packet-first** 方式加载非 SNN workload（例如通信+内存 streaming 校验）。

核心设计目标：**平台核（NoC/Mem/CoreShell）通用可复用；业务语义（SNN/Step/GAS/BCSR/Stream）作为可插拔 workload/子系统加载；边界清晰、可回归、fail-fast。**

推荐总览入口（先看这个更容易读懂整体）：
- `docs/SNNDL_HIERARCHY_AND_WORKFLOW.md`

---

## 快速构建与安装（修改生效必须 install）

> 说明：SnnDL 属于 `sst-elements` 的一部分，通常在已 configure 的工作区内增量编译安装。

```bash
cd "sst_workspace/sst-elements/src/sst/elements/SnnDL"
make -j4
make install
```

---

## 快速运行回归（4×4 mesh 模板）

推荐使用项目内 mesh 模板驱动（详细见 `sst_dram_si/docs/mesh_template_guide_20251122-000250.md`）：

```bash
cd "sst_dram_si"
export MESH_SIM_TIME="100us"
./tools/run_mesh_with_time.sh
```

可选：切换到 `workload=stream`（不涉及 Spike/GAS/权重/BCSR，仅做 packet-first 通信 + 内存 read-after-write 校验）：

```bash
cd "sst_dram_si"
export SNNDL_WORKLOAD_IMPL="stream"
export MESH_SIM_TIME="100us"
./tools/run_mesh_with_time.sh
```

可选：切换到 `workload=traffic`（不建模动力学；用于通信/多播（SpikeKey）链路验证）：

```bash
cd "sst_dram_si"
export SNNDL_WORKLOAD_IMPL="traffic"
export MESH_SIM_TIME="100us"
./tools/run_mesh_with_time.sh
```

输出目录通常位于：
- `sst_dram_si/outputs_large/paper2/dram_mesh_4x4/YYYYMMDD-HHMMSS/`
  - `essential_summary_mesh.json`：关键指标摘要（用于 100us 回归判定）。

### 内存建模口径（默认 cacheline 语义）

SnnDL 的默认内存建模语义与 `memHierarchy` 保持一致：**以 cacheline（例如 64B）作为系统层事务/流量的基本单位**。

- 论文/报告中的“DRAM traffic”主口径建议以 `memHierarchy MemController requests_received_*`（L2 traffic）为准；
- `memory_bytes/memory_requests` 表示上层发起的逻辑请求（L1 logical request），用于解释合并/去重形态，不应直接等价为 off-chip 流量；
- 若切换到 row-streaming/DMA 假设，必须显式标注并单列结果（不得与 cacheline 模式混算）。

---

## Native Multicast（SpikeKey / blocked multicast）

SnnDL 支持一条“原生多播（native multicast）”路径：以 `SpikeKey` 包承载“块内多播目标集合（core mask）”，并采用“两阶段路由”：

- **INTER（块间单播）**：每个目标 block 只发一个包，单播到该 block 的 `ingress_node`。
- **INTRA（块内多播）**：包到达 ingress 后在 block 内按树扩散，并按 `core_mask[cell]` 精确投递到目标 core。

关键约束（当前实现口径）：
- `multicast_block_w * multicast_block_h <= 64`（固定 payload 上限）
- `cores_per_pe <= 32`（core mask 使用 32-bit 位图）

推荐入口（端到端验证 + 统计落盘）：
- `experimental_features/native_multicast_lab/`（实验脚本与 runner）

---

## 目录结构（按边界划分）

> 说明：以“源码子目录”为准（忽略 `.deps/.libs` 等构建产物目录）。每个源码子目录都应有自己的 README，优先以子目录 README 为准。

```
SnnDL/
├── api/            # 跨层稳定接口（窄抽象）
├── events/         # 事件与数据载体（Spike/Gating 等）
├── components/     # SST 组件装配壳（ELI 注册对象）
│   ├── gather/     # GatherBufferIF 构造期参数解析收敛（不新增运行期组件）
│   ├── multicore/  # MultiCorePE 构造期参数解析收敛（不新增运行期组件）
├── control/        # 通用 CoreShell（只做装配/分发/统计汇聚；业务逻辑在 workload）
├── compute/        # 可替换 compute core（神经动力学/学习/验证）
├── services/       # 可复用事务子系统（按子域拆分）
│   ├── noc/        # NoC 传输域（send/recv/forward/本地投递）
│   ├── memory/     # 纯内存访问域（地址→字节块）
│   ├── synapse/    # 突触语义域（weights/route/gas 事务闭环）
│   ├── stimulus/   # Stimulus 域（Step 注入/外部刺激）
│   ├── workload/   # Workload 插件域（snn/stream 等）
│   │   ├── layout/ # Workload neuron layout 口径归一化（num_neurons/neurons_per_pe/base）
│   └── legacy/     # 历史遗留/参考实现（默认不进主链路）
├── docs/           # 设计与阶段性方案文档
└── tests/          # include 自检等轻量测试
```

---

## 关键边界（控制 / 计算 / 路由 / 内存 / 权重）

- `components/`：SST 对接与装配壳（端口、Link、Clock、Stat、生命周期）；尽量不写算法事务。
- `control/`：通用 CoreShell（时钟驱动、packet 递送、统计汇聚）；**不包含 SNN 业务状态机**。
- `services/workload/`：workload 插件（例如 `snn`/`stream`）；承载业务状态机与事务编排。
- `services/workload/traffic`：通信/多播验证用 workload；不建模动力学，但可复用 `synapse/route` 的 fanout/multicast 事务。
- `compute/`：神经动力学与学习等计算逻辑（`ISnnComputeCore`）；**不直接触碰 StandardMem/NoC**（通过 `IWeightReader`/workload 注入）。
- `services/noc/`：纯传输（send/recv/forward/本地投递）；**不做 fanout/权重语义**。
- `services/memory/`：纯地址/字节访问；**不出现权重/突触/路由语义**。
- `services/synapse/`：权重语义/路由与 fanout/GAS 辅助（weights/route/gas 事务闭环）。
- `services/stimulus/`：注入时基与选源（Step 等刺激）；通过 NoC/Route 完成投递与外发。

---

## 开发提示

- 修改 C++ 后必须执行 `make install` 才会影响实际 `sst` 运行加载的元素库。
- 若需要调整构建清单：优先改 `Makefile.am`，并同步保持 `Makefile.in` 与之匹配（`make` 会通过 `config.status` 重生成 `Makefile`）。
- 回归建议：每个阶段至少跑一次 `MESH_SIM_TIME="100us"`，并对比 `essential_summary_mesh.json` 的关键字段（避免非确定性回归）。

---

## 对比实验输出（GAS vs naive）

mesh 模板支持 `MESH_EXEC_MODE=gas|naive_raw|naive_opt` 的对比实验（统一 step-limited 口径建议使用 `MESH_MAX_STEPS=4`），输出目录为：

- `sst_dram_si/outputs_large/paper2/dram_mesh_4x4_exec_mode_compare/`

详见该目录的 README：`sst_dram_si/outputs_large/paper2/dram_mesh_4x4_exec_mode_compare/README.md`
