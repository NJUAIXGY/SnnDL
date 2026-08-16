# SnnDL v5 2D SNN 芯片周期级建模路线图（P0-P8 历史路线）

> - 状态：历史路线文档；P0-P8 已完成当前闭环，P8 correctness/scale gate 已通过。后续科研基准和负载覆盖进入 P9。
> - 日期：2026-08-04
> - 适用代码：`remote/snndl_spec/`、`remote/snndl_system/`、
>   `remote/sst_dram_si/`、`remote/tools/` 与本 SnnDL 子模块
> - 语义基础：`2026-08-03-snndl-nextgen-2d-non-gas-architecture.md`
> - 明确排除：GAS、3D NoC/route、旧通用 workload、未经校准的硬件结论

## 1. 文档定位

本文保留 P0-P7 的原始设计和验收计划，作为历史决策记录；它不是当前能力清单。
当前状态以父仓库的 `docs/SNNDL_V5_TECHNICAL_ARCHITECTURE.md`、
`TECH_PROGRESS.md` 和 P8 `acceptance.json` 为准。

本路线图定义 SnnDL 从 v4 功能模型演进为 v5 周期、容量、争用和能耗可解释模型的完整工程路径。它不替代 v4 的同步 timestep 语义规范，而是在其上增加真实资源约束。

两个版本的职责固定如下：

| 版本 | 权威职责 | 不承担的职责 |
|---|---|---|
| v4 | LIF、BCSR 图语义、timestep 隔离、counted-drain、状态/脉冲哈希和 Python/C++ 等价 oracle | 完整周期、存储容量、flit 传输和功耗结论 |
| v5 | SST 原生周期资源、显式 SRAM、L1/L2/DRAM、DMA、Merlin NoC、硬件同步与统计证据 | mapping 优化、GAS、3D 和未声明的兼容降级 |

本路线图使用新的 P0-P7 编号。它们是 v5 的工程阶段，不对应旧文档中已经完成的 v4 Phase 0-7。

## 2. 执行摘要

### 2.1 当前判断

当前系统已经能证明：

1. 2D SNN timestep 可以在 SST 中完整开始、封闭、排空和提交。
2. logical tx/rx、memory request/response 和 synapse create/retire 可以精确闭合。
3. NoC 队列拥塞时数据不会被静默删除。
4. 每个 Core 私有 L1、每个 PE 共享 L2、全芯片共享 DRAM 的 SST 对象图已经连通。
5. SimpleMem 和 Ramulator2 后端都能完成真实 SST 运行。

当前系统仍不能可靠回答：

1. 一个 spike 在 Core 内经历多少周期以及阻塞在哪一级。
2. rowptr、colidx、state、delta 和 route table 占用多少 SRAM、发生多少 bank conflict。
3. `link_bw`、flit size、crossbar 和 credit 如何影响 NoC 延迟。
4. DMA 预装载、容量不足和 spill 如何影响 timestep。
5. Core、SRAM、cache、NoC 和 DRAM 各自贡献多少能耗与面积。

因此，v5 的主线不是增加更多 neuron model，而是先让现有 LIF+BCSR+2D Mesh 闭环成为可计时、可阻塞、可归因和可校准的芯片模型。

### 2.2 总体阶段

| 阶段 | 主题 | 达成后可以回答的问题 |
|---|---|---|
| P0 | 契约、地址、统计、构建边界 | v5 将建模什么，所有真值在哪里，怎样验收 |
| P1 | Core 周期流水 | 计算本身需要多少周期，哪个执行资源饱和 |
| P2 | 显式 SRAM | state/delta/index 的容量、端口和 bank 争用代价 |
| P3 | Weight hierarchy 与 DMA | L1/L2/DRAM、预装载和 spill 的流量与延迟 |
| P4 | Merlin unicast NoC | flit、带宽、buffer、crossbar 和路由争用代价 |
| P5 | Multicast 与硬件同步 | 通信压缩收益和真实 timestep 控制开销 |
| P6 | Artifact、NIR、workload 与规模化 | 外部模型如何稳定进入芯片并形成可复现实验 |
| P7 | 功耗、面积与校准 | 每 timestep、spike、synop 的能耗和面积来源 |

P0-P4 是最小可信周期模型。P5-P7 完成后，系统才具备完整架构评估能力。

## 3. 当前基线与证据

### 3.1 已验证运行

截至 2026-08-04，选定的有效证据包括：

| 运行 | 规模 | 结果 | 结论边界 |
|---|---|---|---|
| `snndl_runs/sst_fixed_scale_20260804/` | 4x4 PE，4 Core/PE，64 neuron/Core，4096 neuron，8 timestep | 所有 invariant 通过 | neuron 数量扩展通过，但每步活动和 edge 数仍较小 |
| `snndl_runs/sst_final_stress_v4_20260804/` | 4x4 PE，2 Core/PE，8 neuron/Core，768 task/step | 1511 次 backpressure、零 drop，2 timestep 通过 | 验证拥塞保包，不代表 flit 级 NoC |
| `snndl_runs/sst_post_install_ramulator2_20260804/` | 2 PE，1 Core/PE，4 neuron/Core | Ramulator2 两次 read 完成 | 只证明后端连通，不是 DRAM 校准 |

旧失败目录仍是历史证据，不能用日期通配符把所有运行描述为成功。

### 3.2 当前实现矩阵

| 子系统 | 已建模 | 近似或缺失 | v5 处理阶段 |
|---|---|---|---|
| timestep 语义 | Start/Seal/Commit、held spike、counted-drain | 中央协调器不计硬件成本 | P5 |
| LIF | leak、threshold、reset、refractory | 一次函数提交全部 neuron | P1/P2 |
| delta | 确定性累加 | C++ vector，无端口、bank、容量 | P1/P2 |
| BCSR | edge 顺序、digest、value address | descriptor 是逐 edge 文本；rowptr/colidx 未计时 | P0/P2/P3 |
| Core | 多 Core 逻辑实例 | `SnnCoreTile` 不是 SST Component，无独立时钟/链路 | P1 |
| L1/L2 | SST `memHierarchy.Cache` 对象存在 | 参数硬编码，只服务 value read | P0/P3 |
| DRAM | SST MemController + SimpleMem/Ramulator2 | 地址图像很小，缺少系统性压力与校准 | P3/P7 |
| local storage | PE-local value map 和固定 ready cycle | 无真实容量、bank、端口、替换、DMA | P2/P3 |
| NoC | 2D XY、有限队列、每跳延迟、背压保包 | 直接 SST Link；带宽字符串不约束传输 | P4 |
| multicast | 按目标 PE 合包的逻辑路径 | 不是 router tree replication | P5 |
| 统计 | per-step JSON 计数 | `MeshPE2D`/Coordinator 的 SST Statistic 为零 | P0-P7 |
| 功耗/面积 | Ramulator2 可打印部分 DRAM energy | 无全芯片 action/area contract | P7 |

### 3.3 当前关键技术债

1. `processIngress_()` 在一次 tick 中清空整个 ingress queue。
2. `commitTimestep_()` 在一个事件回调内提交所有 Core 和 neuron。
3. 每个 edge 可以立即发出 StandardMem request，没有 issue width 或 request queue 容量。
4. descriptor 在启动期直接解析路由和期望权重，运行时只有 value read 经过内存。
5. `getCurrentSimTimeNano()` 被局部当作 cycle，未区分多时钟域。
6. L1 16 KiB、L2 128 KiB、关联度、MSHR 和频率写死在 Python builder。
7. `noc_link_bw` 只进入参数和摘要，不控制序列化。
8. `BankedSramModel` 是 observe-only，不能把预测周期事后加到真实完成时间。
9. 活跃 `PeDmaScheduler` 仍含 Gather/Apply/Scatter 阶段语义。
10. canonical `libSnnDL.la` 仍加载不属于 v4 必需闭包的 Comm/Local 库。

## 4. 目标、非目标与声明边界

### 4.1 P0-P7 总目标

1. 建立一个所有功能运行均由 SST 组件驱动的 2D SNN 芯片模型。
2. 保留 v4 作为功能 oracle，v5 的性能结果不得来自 Python 执行器。
3. 将 algorithm timestep、组件 cycle 和 SST sim time 明确分离。
4. 每项可见延迟都能归因到有限资源、链路或同步协议。
5. 每个存储对象拥有唯一内容真值、地址空间、容量和访问路径。
6. 每个队列有限且有明确 backpressure，不允许 drop 或无限隐式缓冲。
7. 所有性能参数进入 effective config；环境变量不能暗中改变架构。
8. 支持功能、周期、规模和能耗四级独立验收。
9. mapping 与模拟保持项目边界；SnnDL 只消费确定 artifact。
10. 形成可复现实验所需的 spec、artifact、版本、统计和校准证据链。

### 4.2 永久非目标

- GAS 或 Gather/Apply/Scatter 阶段架构。
- 3D NoC、3D route 或 3D memory placement。
- 把 Tensor、Stream、Traffic、RISC-V workload 重新装入 SNN 核心闭包。
- 在 P0-P7 内优先支持 STDP、在线训练或大量 neuron model。
- 用 fixed delay、静默窗口或 post-processing 修正真实资源阻塞。
- 把 Loihi 2、SpiNNaker2 的公开参数直接当作 SnnDL 校准值。
- 用同一运行中的两个工具重复计算并相加同一 DRAM 能耗。

### 4.3 声明等级

| 等级 | 必须完成 | 允许声明 |
|---|---|---|
| F：功能正确 | v4/v5 state、spike、delivery、task hash 等价 | 算法语义正确 |
| T：周期可信 | P1-P4 解析测试、有限资源与统计闭合 | 周期、吞吐和瓶颈 |
| S：规模可信 | P6 多规模、多 rank、完整 artifact | 规模趋势和 workload 比较 |
| E：能耗可信 | P7 action count、校准和误差边界 | 能耗、面积和效率 |

未达到对应等级时，报告必须使用“功能验证”“连通性 smoke”或“未校准估计”等准确措辞。

## 5. 外部依据与本项目取舍

### 5.1 SST 原生组件优先

SST `memHierarchy` 已提供 Cache、Scratchpad、Bus、MemController 和 StandardMem 接口；Merlin 已提供 input/output queued `hr_router`、flit、link/xbar bandwidth、buffer、仲裁和 SimpleNetwork endpoint 接口。SnnDL 应只实现 SNN 特有的数据面与 SST 尚缺失的局部 bank timing，不重新实现通用 cache、DRAM controller 或 router。

本地 SST 15.0.0 的实际 `sst-info` 是实现权威。在线文档可能已经对应 SST 16，任何新增参数必须先在本地安装中核实。

### 5.2 神经形态芯片参考边界

- Loihi 2 说明可编程 neuron pipeline、本地 neural memory、spike message NoC 和异步 core 是合理建模维度。
- SpiNNaker2 说明 local SRAM、DMA、off-chip DRAM、NoC 和 multicast 可以组成另一种有效实现。
- SnnDL v5 不是两者的复制品。其基线是同步 timestep、专用 SNN pipeline、显式局部 SRAM、cache-backed weights 和 2D Mesh。

### 5.3 模型与 benchmark 接口

- NIR 用于表达神经动力学与事件图，不负责 placement、地址或硬件容量。
- NeuroBench 用于组织算法级与系统级 workload/metric，不替代周期模型验证。
- Accelergy 用于片上 component/action 的面积和能耗估计。
- Ramulator2 用于 DRAM controller/device 周期；DRAMPower 可作为替代或交叉校准，不能重复计入总能耗。

## 6. v5 分层模型

v5 必须把以下五层分开：

```text
Model semantics
  LIF equations, timestep ownership, deterministic contribution order
            |
Artifact contract
  graph, placement, route, BCSR arrays, stimulus, digests, address map
            |
Architecture instance
  Core widths, SRAM geometry, cache hierarchy, DMA, NoC, synchronization
            |
SST execution
  Components, SubComponents, Links, clocks, events, statistics
            |
Evidence
  effective config, raw stats, timestep trace, validation, energy/area report
```

依赖只能向下：

1. 模型语义不知道 Merlin、Ramulator2 或具体 cache。
2. artifact 不包含运行时策略或统计结果。
3. architecture 不重新解释 graph 内容。
4. SST builder 不从环境变量改写已解析的结构字段。
5. summary 只派生已有原始统计，不成为新的功能真值。

## 7. 目标芯片结构

```text
NIR / dataset / synthetic source
              |
      external mapper/compiler
              |
  manifest + placement + route + BCSR + stimulus
              |
+----------------------------- SnnDL v5 chip ------------------------------+
|                                                                          |
|  Validation mode: out-of-band TimestepCoordinator                        |
|  Performance mode: timed reduction/broadcast Sync Tree on control VN     |
|                                                                          |
|     +---------------- Chip memory fabric -----------------------------+   |
|     | MemController -> SimpleMem or Ramulator2 -> chip-shared DRAM    |   |
|     +--------------------------^--------------------------------------+   |
|                                |                                          |
|  +----------------------------- PE[x,y] ------------------------------+   |
|  | Merlin hr_router <-> PeNetworkEndpoint                             |   |
|  |                         |                                           |   |
|  |            route SRAM / local dispatch queues                       |   |
|  |                         |                                           |   |
|  |   Core[0] ... Core[N-1] | PE DMA | shared noncoherent L2           |   |
|  |      | StandardMem weight path              |                       |   |
|  |      +-> private noncoherent L1 ------------+----------------DRAM   |   |
|  |                                                                      |   |
|  |   Each Core:                                                        |   |
|  |   IngressQ -> RowLookup -> SynapseLanes -> DeterministicRetireQ     |   |
|  |       |          |               |                  |                |   |
|  |   index SRAM   weight path      delta SRAM       NeuronPipeline     |   |
|  |                                                   |                  |   |
|  |                    state SRAM <-------------------+                  |   |
|  |                                                   |                  |   |
|  |                                               HeldSpikeQ             |   |
|  +----------------------------------------------------------------------+   |
+----------------------------------------------------------------------------+
```

### 7.1 Core 不再是 PE 内普通对象

`SnnCoreV5` 应成为独立 SST Component，原因如下：

1. 每个 Core 需要独立 clock handler、SST statistics 和有限队列。
2. Core 数量不应受 `core_memory0..7` 静态 slot 限制。
3. 每个 Core 的 L1、state SRAM、delta SRAM 和 index SRAM 可以独立连接。
4. SST partition/rank 可以把 Core/PE 作为显式图节点处理。
5. PE 只保留 NIC、route、local dispatch、DMA 和共享 L2 资源。

### 7.2 Cache 与 SRAM 的职责

| 资源 | 类型 | 内容 | 基线策略 |
|---|---|---|---|
| Core state store | 显式 SRAM | membrane、refractory、model state | 固定驻留，容量不足 fail |
| Core delta store | 显式 SRAM | 当前 timestep 增量、touched metadata | 固定驻留，容量不足 fail |
| Core index store | 显式 SRAM | rowptr、常驻 colidx 子集 | P2 固定驻留；P3 可 DMA refill |
| PE route store | 显式 SRAM | destination PE/Core 与 multicast metadata | 固定驻留，容量不足 fail |
| optional weight SPM | 显式 SRAM | DMA 预装载的 weight tile/shard | P3 显式启用 |
| Core L1 | SST Cache | DRAM-backed index/value cachelines | private、noncoherent、有限 MSHR |
| PE L2 | SST Cache | PE Core 共享的 backing data | shared、noncoherent、noninclusive |
| chip DRAM | SST MemController backend | 完整 BCSR backing image | SimpleMem 或 Ramulator2 |

推理基线中的 weights 在运行期不可修改，state/delta 不进入 cache hierarchy。因此 v5 默认使用 `coherence_protocol=none`，避免把通用 CPU coherence 流量误认为 SNN 芯片成本。未来在线学习必须另立写一致性契约。

## 8. 时间与执行语义

### 8.1 三种时间

| 时间 | 类型 | 权威用途 |
|---|---|---|
| `timestep_id` | `uint64_t` | SNN 算法状态版本和正确性 |
| domain cycle | 每组件本地计数器 | Core/SRAM/NoC/memory 资源调度 |
| SST sim time | `SimTime_t` 和带单位时间 | 跨时钟域事件与端到端延迟 |

禁止在 v5 摘要中使用无域的裸 `cycles`。字段必须写成 `core_cycles`、`noc_cycles`、`memory_cycles` 或 `elapsed_ns`。

### 8.2 同步 timestep 不变量

v5 继承并强化 v4 不变量：

1. 输入、route task、memory request、retire entry、delta 和输出都携带 `timestep_id`。
2. timestep `k` 只读取 `state[k]`，只写 `delta[k]`。
3. 每个 neuron 在 `k` 恰好提交一次。
4. `k` 产生的 spike 只能在 `Start(k+1)` 后注入 NoC。
5. memory response 顺序不能改变 state/spike 结果。
6. queue full 只能 backpressure，不能 drop、覆盖或隐式扩容。
7. `SealIngress(k)` 后的新 `k` 数据事件必须 fail-fast。
8. 任何旧 timestep 事件必须 fail-fast。
9. CommitReady 前所有功能 token 和所有资源队列必须归零。
10. watchdog 只能终止并报告 snapshot，不能强制完成 timestep。

### 8.3 Core 周期流水

建议基线流水如下：

| Stage | 输入 | 有限资源 | 输出 |
|---|---|---|---|
| S0 Admit | spike packet/local delivery | ingress entries、accept width | admitted spike |
| S1 Row lookup | admitted spike | row lookup lanes、index ports | row bounds |
| S2 Expand | row bounds | edge lanes、colidx ports | synapse task |
| S3 Weight fetch | synapse task | request queue、L1/MSHR/DMA | ready weight |
| S4 Retire/accumulate | ready weight | retire entries、accumulator lanes、delta ports | committed delta |
| S5 Neuron update | state+delta | neuron lanes、state/delta ports | next state、held spike |

每个 stage 的 latency、width 和 queue depth 必须独立配置。一个 tick 内的处理次数不得由容器大小隐式决定。

## 9. Artifact 与地址契约

### 9.1 Artifact 文件集合

```text
artifact/
  manifest.json
  descriptor.bin
  placement.bin
  route.bin
  rowptr.bin
  colidx.bin
  weights.bin
  stimulus.bin
```

`manifest.json` 是 provenance 和结构权威；运行时数组内容来自二进制文件。`descriptor.bin` 保存 C++ setup 所需的固定头、region 和 digest，不复制 edge/weight 真值。

### 9.2 Manifest 必需字段

```text
artifact_format_version
generator_name / generator_version
graph_digest
model_digest
placement_digest
route_digest
bcsr_digest
stimulus_digest
total_neurons / total_edges
mesh_rows / mesh_cols / cores_per_pe / neurons_per_core
orientation = pre_to_post
index_width_bytes / value_width_bytes
regions[]
source_provenance
```

每个 `regions[]` 条目至少包含：

```text
name
address_space
owner_scope
owner_id
base
size_bytes
alignment_bytes
element_bytes
file
file_offset
sha256
backing_region (optional)
```

### 9.3 地址空间

v5 不把所有局部资源伪装成一个扁平物理地址。请求使用强类型 `(AddressSpaceId, owner, offset)`：

| AddressSpaceId | Owner | 典型内容 |
|---|---|---|
| `CoreState` | PE/Core | neuron state |
| `CoreDelta` | PE/Core | timestep delta |
| `CoreIndex` | PE/Core | rowptr、colidx |
| `PeRoute` | PE | route/group table |
| `PeWeightSpm` | PE 或 Core | optional weight residency |
| `ChipDram` | chip | BCSR backing image |

只有 `ChipDram` 地址进入 L1/L2/DRAM。DMA descriptor 显式声明 source space、destination space、owner、地址和字节数。

### 9.4 单一真值规则

1. route 和 BCSR 必须由同一 graph/placement 生成。
2. C++ 不得从另一个 edge 文本重新生成 route。
3. task 不得携带一个“期望 weight”作为运行时计算真值。
4. Python oracle 必须读取相同 artifact，不读取原始 edge 列表的第二份副本。
5. setup 可以验证 digest 和 region，但不能用 descriptor 内容绕过计时数据访问。
6. 任一 digest、shape、stride、orientation 或 file size 不一致都必须在 setup 失败。

## 10. v5 Spec 总体契约

### 10.1 设计规则

1. `schema_version=5` 使用独立 resolver，不经过 v1-v4 兼容映射。
2. `model=snn_mesh_2d_cycle` 是唯一初始模型名。
3. 性能关键字段必须显式出现；P0 不引入会隐藏参数的 profile 展开。
4. unknown field、unknown enum 和错误单位直接拒绝。
5. GAS、3D 和 legacy stage 字段在任意层级出现都报 deprecated error。
6. 环境变量只允许设置输出目录、日志等级和 watchdog，不允许改硬件结构。
7. `effective_config.json` 必须完整展开，并保留 user spec 和 artifact digest。
8. `run_class=performance` 禁止无限队列、observe-only timing 和未校准隐式默认值。

### 10.2 建议 Spec 骨架

以下数值只用于展示字段，不是校准默认值：

```json
{
  "schema_version": 5,
  "model": "snn_mesh_2d_cycle",
  "run": {
    "class": "functional",
    "watchdog": "10ms"
  },
  "platform": {
    "mesh": {"rows": 4, "cols": 4},
    "pe": {"cores_per_pe": 4}
  },
  "execution": {
    "semantics": "synchronous_timestep",
    "completion": "counted_drain",
    "start_timestep": 0,
    "max_timesteps": 8,
    "sync_mode": "oracle_out_of_band"
  },
  "clocks": {
    "core": "1GHz",
    "local_storage": "1GHz",
    "noc": "1GHz",
    "memory_fabric": "1GHz"
  },
  "neuron": {
    "model": "lif",
    "neurons_per_core": 256,
    "dt_ms": 1.0,
    "tau_mem_ms": 20.0,
    "threshold": 1.0,
    "reset": 0.0,
    "refractory_timesteps": 1,
    "state_bytes": 8
  },
  "core": {
    "ingress": {"queue_entries": 64, "accept_per_cycle": 1},
    "row_lookup": {"lanes": 1, "latency_cycles": 1},
    "synapse": {"lanes": 4, "latency_cycles": 1, "task_queue_entries": 128},
    "weight_requests": {"queue_entries": 64, "issue_per_cycle": 2},
    "retire": {"entries": 128, "retire_per_cycle": 4},
    "accumulator": {"updates_per_cycle": 4},
    "neuron_pipeline": {"lanes": 4, "latency_cycles": 3},
    "held_spike_queue_entries": 256
  },
  "storage": {
    "state_sram": {
      "size": "8KiB", "banks": 4, "ports_per_bank": 1,
      "interleave": "8B", "read_latency_cycles": 1, "write_latency_cycles": 1
    },
    "delta_sram": {
      "size": "4KiB", "banks": 4, "ports_per_bank": 1,
      "interleave": "4B", "read_latency_cycles": 1, "write_latency_cycles": 1
    },
    "index_sram": {
      "size": "16KiB", "banks": 4, "ports_per_bank": 1,
      "interleave": "4B", "read_latency_cycles": 1, "write_latency_cycles": 1
    },
    "route_sram": {
      "size": "32KiB", "banks": 4, "ports_per_bank": 1,
      "interleave": "8B", "read_latency_cycles": 1, "write_latency_cycles": 1
    },
    "weight_spm": {"enabled": false, "size": "0B"},
    "overflow_policy": "fail"
  },
  "cache": {
    "l1": {
      "size": "16KiB", "associativity": 2, "line_size": "64B",
      "banks": 2, "access_latency_cycles": 2,
      "max_requests_per_cycle": 2, "mshr_entries": 16,
      "request_link_width": "16B", "coherence_protocol": "none"
    },
    "l2": {
      "size": "128KiB", "associativity": 8, "line_size": "64B",
      "banks": 8, "access_latency_cycles": 8,
      "max_requests_per_cycle": 4, "mshr_entries": 32,
      "request_link_width": "32B", "coherence_protocol": "none",
      "cache_type": "noninclusive"
    }
  },
  "dma": {
    "enabled": false,
    "channels": 1,
    "queue_entries": 16,
    "max_outstanding": 8,
    "burst_bytes": 64,
    "bytes_per_cycle": 16,
    "setup_cycles": 1
  },
  "noc": {
    "implementation": "merlin.hr_router",
    "topology": "mesh",
    "routing": "xy",
    "link_bw": "40GiB/s",
    "flit_size": "16B",
    "xbar_bw": "80GiB/s",
    "input_latency": "1ns",
    "output_latency": "1ns",
    "input_buf_size": "1KiB",
    "output_buf_size": "1KiB",
    "num_vns": 2,
    "data_vn": 0,
    "control_vn": 1,
    "packet_header_bytes": 24,
    "multicast_mode": "source_replication"
  },
  "memory": {
    "scope": "chip_shared",
    "controller_clock": "1GHz",
    "fabric_link_latency": "1ns",
    "backend": {
      "type": "simple",
      "mem_size": "1GiB",
      "access_time": "100ns",
      "request_width": "64B",
      "max_requests_per_cycle": 1
    }
  },
  "artifact": {
    "manifest": "artifact/manifest.json",
    "expected_graph_digest": "..."
  },
  "statistics": {
    "load_level": 2,
    "output": "sst_stats.csv",
    "timestep_trace": "timestep_trace.jsonl"
  }
}
```

### 10.3 字段所有权

| 字段组 | Owner | 禁止读取者 |
|---|---|---|
| neuron equations | Domain neuron model | NoC、DMA、cache builder |
| Core widths/queues | `SnnCoreV5` | artifact generator |
| SRAM geometry | storage builder/backend | neuron equation |
| cache/memory | memHierarchy builder | route codec |
| NoC geometry | Merlin builder/endpoint | synapse compute |
| sync mode | synchronization builder | weight store |
| artifact paths/digests | artifact loader | architecture heuristics |

## 11. Statistics 与证据契约

### 11.1 原始统计权威

v5 的原始性能权威是 SST Statistics 和明确的 step trace。Coordinator 写出的聚合 JSON 不能是唯一数据源。

输出最小集合：

```text
effective_config.json
run_manifest.json
artifact_manifest.json
sst_stats.csv
timestep_trace.jsonl
essential_summary_v5.json
validation.log
energy_area/              # P7 才要求
```

### 11.2 命名空间

| Namespace | 关键统计 |
|---|---|
| `timestep.*` | started、sealed、committed、elapsed_ns、sync_wait_ns |
| `core.ingress.*` | accepted、occupancy、full_cycles、stalls |
| `core.synapse.*` | rows、tasks、issued、retired、lane_busy_cycles |
| `core.retire.*` | occupancy、head_blocked_cycles、reorder_distance |
| `core.neuron.*` | evaluated、fired、lane_busy_cycles、commit_cycles |
| `storage.state.*` | reads、writes、bytes、bank_conflicts、stall_cycles |
| `storage.delta.*` | reads、writes、updates、bank_conflicts、stall_cycles |
| `storage.index.*` | rowptr_reads、colidx_reads、miss/refill、stall_cycles |
| `storage.route.*` | lookups、fanout_entries、bank_conflicts、stall_cycles |
| `cache.l1/l2.*` | SST cache events、MSHR occupancy、bank conflicts |
| `dma.*` | descriptors、bursts、bytes、queue occupancy、stall reasons |
| `noc.*` | logical deliveries、packets、flits、bits、hops、latency、stalls |
| `sync.*` | control packets/bytes、reduce/broadcast latency、barrier wait |
| `dram.*` | requests、bytes、latency、row hit/miss/conflict、energy source |
| `energy.*` | action counts、dynamic/leakage energy、provenance |

### 11.3 类型与单位

- Count：`uint64_t`，单位必须是 events、requests、packets、flits、bytes 或 operations。
- Occupancy：同时报告 sample sum、sample count 和 peak，不能只报告最终深度。
- Latency：原始值用 ns 或 SST timebase；histogram 明确 bin。
- Cycle：字段名带 clock domain。
- Energy：统一 pJ；area 统一 `um^2` 或 `mm^2`，summary 明确转换。
- Ratio：summary 派生，raw stat 不保存已经舍入的百分比。

### 11.4 正确性 invariant

每个 timestep 必须满足：

```text
logical_tx == logical_rx
synapse_tasks_created == synapse_tasks_retired
memory_requests_issued == memory_responses_completed
all tracker tokens == 0 before CommitReady
all finite resource queues == 0 before CommitReady
neuron_commit_count == participating_neurons
held_spikes_created[k+1] == held_spikes_released[k+1]
queue_drops == 0
stale/future/post_seal events == 0
```

### 11.5 性能派生规则

1. `timestep_elapsed_ns` 从 Start 到最后 CommitDone。
2. `synop` 定义为一个被 retire 并写入 delta 的逻辑突触贡献，不等于 cache request。
3. logical delivery、physical packet、flit 和 byte 必须分别报告。
4. payload bytes 与 protocol/header bytes 必须分别报告。
5. 并行 stage 的 busy/stall cycles 可以重叠，不能简单相加为 wall time。
6. bottleneck 判断使用 occupancy、backpressure 和关键路径时间，不使用事后猜测。
7. 功能 oracle 的运行时间不进入 v5 性能指标。

## 12. 验证方法

### 12.1 四层验证

| 层 | 方法 | 失败含义 |
|---|---|---|
| Contract | schema、artifact、address、build boundary | 配置或所有权不闭合 |
| Functional | Python oracle/v4/v5 hash 与计数对照 | 算法语义回归 |
| Analytic timing | 无争用微测试的精确周期公式 | 资源时序实现错误 |
| Stress/scaling | 有限容量、拥塞、多 rank、长 timestep | 背压、死锁或可扩展性问题 |

### 12.2 标准微测试

1. 1 neuron，无输入：只测 neuron scan。
2. 1 pre -> 1 post：一次 route、row lookup、weight、delta。
3. 多 pre -> 1 post：retire 顺序与 delta bank 热点。
4. 1 pre -> 多 post：row expansion 与 synapse lane 饱和。
5. 跨 Core：PE local dispatch。
6. 单跳跨 PE：NoC 单包解析周期。
7. 多跳跨 PE：XY hop/flit 延迟。
8. hotspot fan-in：router output、L2 和 delta 冲突。
9. recurrent A -> B -> A：无步内级联。
10. cold/warm weights：L1/L2/DRAM 层级。
11. SRAM fit/non-fit：fail 与显式 DMA/spill。
12. multicast fanout：logical 等价与 physical 压缩。

### 12.3 扰动测试

- 反转或随机化 memory response 顺序。
- 改变 Core、SRAM、NoC 和 memory clock 比例。
- 将所有 queue 调到 1，验证反压链。
- 改变 flit size、link bandwidth、xbar bandwidth。
- 强制相同 bank 与均匀 bank 地址。
- 使用 1/2/4 SST rank，比较最终 hash 和全局计数。
- 在允许的每个入口注入长延迟，不允许功能结果变化。

### 12.4 规模不是单一 neuron 数

规模实验至少同时记录以下五个轴：

```text
total_neurons
total_edges
active_spikes_per_timestep
fanout_distribution
number_of_timesteps
```

只增加未活动 neuron 不能证明 synapse、NoC 或 DRAM 扩展能力。

## 13. 阶段依赖关系

```text
P0 Contract
 |\
 | +------> P2 Explicit SRAM ----+
 v                                |
P1 Core Pipeline -----------------+--> P3 Weight/DMA --+
 |                                                     |
 +-------------------------------> P4 Merlin NoC ------+--> P5 Multicast/Sync
                                                               |
                                                               v
                                                        P6 Artifact/Scale
                                                               |
                                                               v
                                                        P7 Energy/Area
```

阶段隔离规则：

1. P1 使用可解析的理想 memory/loopback transport，隔离 compute timing。
2. P2 固定 compute width，隔离 SRAM geometry。
3. P3 固定 NoC 模式，隔离 memory hierarchy 和 DMA。
4. P4 使用已验证的 memory fixture，隔离 NoC。
5. P5 才同时打开通信扩展与硬件同步。
6. P6 才引入真实 workload 和 MPI scale。
7. P7 不改变功能和周期，只消费 action count 并校准。

## 14. 全阶段共同工程规则

P0-P7 不是八条可以独立堆叠的 feature branch。每个阶段都必须遵守同一组工程规则，否则后续统计即使看起来完整，也不能形成可信的架构结论。

### 14.1 版本与兼容策略

1. v4 在整个迁移期间保持可运行，并作为功能 oracle；不得把 v5 的周期参数回填到 v4。
2. v5 使用独立的 `schema_version=5`、组件名、artifact version 和摘要版本，不在 v4 resolver 中增加兼容分支。
3. P0-P3 的 v5 结果只能标记为 `functional` 或 `development`；P4 通过后才允许 `timing`。
4. 新 v5 C++ 源统一放在 `v5/` 下。它只能依赖 SST、`libSnnDLNextCore` 中的纯语义类型和 `v5/api/`，不能 include 旧 Comm、Local、Research 或 archive。
5. `SnnDL.MeshPE2D` 与 `SnnDL.TimestepCoordinator` 保留为 v4 ELI；v5 使用 `SnnDL.SnnCoreV5`、`SnnDL.PeEndpointV5` 等不同名称。
6. 不提供把 v4 参数静默翻译成 v5 参数的兼容层。迁移必须通过显式 spec 转换工具并输出差异。

### 14.2 每阶段 Definition of Done

每个阶段只有同时满足下列条件才算完成：

- **契约完成**：spec、ELI 参数、artifact、统计字段和单位有唯一文档。
- **实现完成**：正常路径、队列满、容量不足、非法事件和 teardown 均有定义。
- **功能完成**：与相同 artifact 的 v4 oracle 比较 state/spike/task digest。
- **解析完成**：至少一个无争用微测试能用手算公式得到精确周期。
- **压力完成**：至少一个有限队列或热点测试真实触发 stall，且零 drop。
- **证据完成**：保存 effective config、版本、raw stats、验证日志和失败原因。
- **边界完成**：构建检查证明没有 GAS、3D、旧 workload 或隐式 fallback 进入 v5。
- **文档完成**：本路线图状态和外层 `TECH_PROGRESS.md` 只追加更新。

任何一项未满足，阶段状态只能写 `partial`，不能用“主路径能跑”代替完成。

### 14.3 禁止的通用捷径

- 用容器一次性遍历模拟任意宽度硬件。
- 先无限接收，再在摘要中估算 queue stall。
- 把 `latency_cycles` 事后加到完成时间，而资源请求没有真的等待。
- 把 Python oracle 的 wall time 或 loop count 当作 SST 周期。
- 从 spec、descriptor、edge list 和 memory image 中选择“方便的一份”作为真值。
- 容量不足时自动扩大 SRAM/cache、切换 backend 或回退到本地 map。
- 用 retry 丢失原请求顺序、重复 token，或把 overflow 当 drop。
- 为了通过测试而关闭统计、缩短输入或减少实际活动边，却仍使用原规模标签。

## 15. P0：契约冻结、旧边界清理与可执行地基

### 15.1 目标与范围

P0 不实现周期级 Core，也不产生 v5 性能数字。它的任务是把后续实现的输入、输出、依赖和验收方式一次封闭，确保 P1 不再边写流水边修改语义。

P0 完成后必须能够回答：

1. 哪个文件定义 v5 spec，unknown/deprecated 字段怎样失败。
2. graph、placement、route、BCSR、stimulus 和地址分别由谁生成、谁读取。
3. C++ 与 Python 如何确认正在解释同一版 contract。
4. 哪些统计是 raw authority，单位和 reset scope 是什么。
5. canonical `SnnDL` ELI 会加载哪些库，哪些旧库只能显式加载。
6. v4 的哪些结果被正式冻结为回归 oracle。
7. P1 开始前需要执行哪些手工验收命令。

### 15.2 P0.0：冻结并复验 v4 基线

**动作**

1. 记录外层仓库、`sst-elements` 和 SnnDL 三层 commit ID、dirty 状态、编译器版本、Python 版本、SST 版本和安装前缀。
2. 以新的只写运行目录重新执行 minimal、scale、stress 和 Ramulator2 四个 v4 fixture；不得覆盖 2026-08-04 现有证据。
3. 从结果中抽取功能字段，而不是冻结整个含绝对路径和时间戳的 run directory。
4. 明确旧 `totals.cycles` 只属于 v4 近似计数，不作为 v5 timing golden。

**冻结字段**

```text
schema/model
graph/route/descriptor digest
topology tuple
per-timestep state_hash/spike_hash
run-level state_digest/spike_digest
logical tx/rx
memory request/response
synapse create/retire
fired/neurons committed
all invariants and zero-error diagnostics
```

**退出条件**：四个 fixture 均由真实 SST executor 返回 0；scale 仍为 4096 neuron；stress 真实出现 backpressure 且 drop=0；Ramulator2 的 request/response 和后端统计非零。

### 15.3 P0.1：确定文档权威顺序

规范冲突时按以下顺序裁决：

1. 本路线图中“强制契约”和已经签字的 P0 ADR。
2. `2026-08-03-snndl-nextgen-2d-non-gas-architecture.md` 的 v4 功能语义。
3. `SNNDL_V4_STORAGE_CONTRACT.md` 的 v4 兼容行为。
4. 当前实现和旧 README。
5. `archive/` 内文档仅用于历史解释，不具有设计权威。

P0 新增 `docs/adr/0001-v5-contract-boundary.md`，只记录已经批准的选择：同步 timestep、2D、无 GAS、外部 mapping、显式 SRAM、noncoherent cache、Merlin、单一 artifact 真值。未决定项必须进入第 27 节的 decision log，不能偷偷成为默认值。

### 15.4 P0.2：建立严格 v5 spec

**实现边界**

- `snndl_spec/v5.py` 独立解析 v5，不调用 v1-v4 resolver。
- `tools/snndl_spec_cli.py` 只根据顶层 version 分发，不能猜测 model。
- 所有结构字段使用白名单；boolean 不接受 `0/1`，integer 不接受浮点截断。
- time、frequency、bandwidth 和 bytes 必须带单位；内部解析为规范 SI 值，同时保留原 literal。
- cross-field validation 检查地址对齐、容量、VN 编号、clock domain、mesh/placement shape 和 artifact digest。
- `GAS`、`Gather`、`Apply`、`Scatter`、3D、legacy workload 字段即使嵌套在未知对象中也给出明确 deprecated error。
- P0 的 `mesh_minimal_v5.json` 仅用于 contract validation；在 P1 组件存在前，runner 必须明确报告 `runtime_not_implemented`，不能转调 v4 并伪装 v5。

**P0 必须冻结的字段组**

| 字段组 | P0 冻结内容 | 可延后内容 |
|---|---|---|
| identity | schema、model、run class | 无 |
| topology | mesh、Core/PE、neuron/Core | partition policy 到 P6 |
| clocks | Core、storage、NoC、memory domain | DVFS 到路线图外 |
| core | stage width、latency、queue depth | 具体校准值到 P7 |
| storage | region、size、bank、port、latency、overflow | replacement/spill 到 P3 |
| cache | L1/L2 geometry、MSHR、noncoherent policy | prefetch policy 到 P3 |
| DMA | channels、queue、outstanding、burst、bandwidth | schedule 到 P3 |
| NoC | Merlin 参数、VN、header、multicast mode | native tree policy 到 P5 |
| memory | backend、clock、capacity、config | calibrated DRAM config 到 P7 |
| artifact | manifest、expected digest | NIR provenance extension 到 P6 |
| statistics | load level、raw output、step trace | energy output 到 P7 |

### 15.5 P0.3：冻结 artifact、binary header 与地址契约

P0 交付四类机器可检查契约：

1. `spec_contract.json`：字段、类型、单位和 enum。
2. `artifact_contract.json`：manifest、region 和 digest 规则。
3. `statistics_contract.json`：统计名、类型、单位、scope 和 introduced phase。
4. `provenance_contract.json`：run manifest 的版本和文件哈希。

每个 binary artifact 使用同一固定前缀：

```text
magic[8]
format_version:u16
header_bytes:u16
element_width:u16
flags:u16
element_count:u64
payload_bytes:u64
owner_scope:u32
owner_id:u32
payload_sha256[32]
```

多字节字段固定 little-endian；float weight 初始只允许 IEEE-754 binary32。P0 必须提供越界、截断、错误 endian 标记、错误 digest、重叠 region、未对齐 base 和 shape 不一致的负测试。

地址计算必须来自 region：

```text
typed_address = (space, owner_id, byte_offset)
dram_physical = region.base + byte_offset
```

只有 manifest 中声明 `backing_region=ChipDram:*` 的 region 可以产生 DRAM 地址。C++ 不得再用 `edge.ordinal * memory_bytes` 隐式构造完整地址。

### 15.6 P0.4：冻结 statistics ABI

`statistics_contract.json` 每个条目至少包含：

```text
name
value_type
unit
scope = component | timestep | run
aggregation = sum | max | histogram | sampled_average
reset = never | timestep_start
introduced_phase
required_for_run_class[]
description
```

P0 生成 `v5/api/StatisticNames.h`，Python summary loader 和 C++ 均不得手写第二套字符串。生成器输出必须稳定排序并嵌入 contract SHA-256；测试比较重新生成结果，禁止 drift。

P0 只验证统计注册表和 summary 空壳，不制造非零周期数据。`run.class=functional` 可以缺少 P4/P7 统计；`run.class=performance` 必须因阶段尚未完成而拒绝运行。

### 15.7 P0.5：归档带旧阶段语义的 DMA 簇

以下活动文件整体移动到 `archive/legacy_gas/` 对应目录，保留历史内容但不进入活动 manifest：

```text
api/IDmaSchedulerProvider.h
api/IDmaTaggedAccess.h
platform/memory/DmaMemAccessProxy.h
platform/memory/DmaMemAccessProxy.cc
platform/memory/PeDmaScheduler.h
platform/memory/PeDmaScheduler.cc
tests/test_pe_dma_scheduler.cc
```

同时删除活动 `test-dma-scheduler` target、README/API 索引和生成 Makefile 对它们的引用。P0 的边界检查应在活动源中拒绝 `PeDmaScheduler`、`stage_budget_permille` 和 `Stage::{Gather,Apply,Scatter}`。

P3 的新 DMA 不从这些类继承。它只接受 byte-copy descriptor，并用 SST `StandardMem::MoveData` 或等价的 Scratchpad Get/Put 路径完成显式搬运。

### 15.8 P0.6：建立 canonical library/ELI 依赖闭包

P0 采用以下构建边界：

```text
libSnnDLNextCore.la       pure v4/v5 semantic primitives
libSnnDLV5Contracts.la    v5 address/artifact/stat types; no SST Component
libSnnDLPlatform2D.la     frozen v4 MeshPE2D + Coordinator
libSnnDL.la               canonical ELI aggregate, P0 only exposes v4 platform

libSnnDLComm.la           explicit legacy/extension load only
libSnnDLLocal.la          explicit extension load only, no old DMA cluster
libSnnDLResearch.la       explicit 3D research load only; never a v5 dependency
```

`libSnnDL.la` 在 P0 从 `LIBADD` 中移除无关的 Core、Comm、Local 和 Registry 聚合依赖，只保留 v4 平台及其真实传递依赖。`libSnnDLV5Contracts.la` 因不含 ELI constructor，不需要塞入 aggregate；P1 的 v5 component library 才加入 canonical aggregate。

边界验收同时检查：

- source manifest include 规则；
- `readelf -d`/`ldd` 的 `DT_NEEDED`；
- `sst-info SnnDL` 只出现 canonical v4/v5 ELI；
- `sst-info SnnDLResearch` 仍可显式找到历史 3D research，但 canonical 输出中不得出现；
- v5 source 和 generated dependency file 都不引用 archive。

### 15.9 P0.7：建立精简 v4 golden fixtures

版本库只保存小而稳定的 oracle，不保存整套 transient SST 输出。建议固定四类 fixture：

| Fixture | 目的 | 必须比较 |
|---|---|---|
| `minimal_recurrent` | A->B->A、held spike | 每步 state/spike hash |
| `response_reorder` | memory 回包反序 | 与 forward 完全等价 |
| `fanout_multicore` | PE/Core mapping 与逻辑投递 | tx/rx/task/commit |
| `zero_edge_scan` | 无突触 neuron commit | commit count 和 state |

每个 fixture 目录只含输入 spec、artifact manifest/小二进制和 `expected_v4.json`。expected 文件带 oracle code commit 和 contract digest。4096-neuron、stress、Ramulator2 运行保留为外部 evidence pointer，不作为逐字 golden。

### 15.10 P0.8：建立 effective config 与 run provenance

`run_manifest.json` 至少记录：

- 三层 Git commit 与 dirty boolean；dirty 时保存相关路径的 diff digest，不复制未提交源码。
- SST Core/elements 版本、compiler、Python、host、rank count。
- user spec path/SHA-256、effective config SHA-256、artifact manifest SHA-256。
- SnnDL ELI library realpath 和 shared-object SHA-256。
- exact command、start/end UTC、exit status、run class、claim level。
- raw output 清单及每个文件的 SHA-256。

`effective_config.json` 只能由严格 resolver 生成，写入后 runner 以只读方式传递。任何 builder-side normalization 都必须回到 resolver；builder 不得再覆写 neuron 数、cache 大小或 backend。

### 15.11 P0.9：手工验收套件

按用户既定边界，P0 **不建立 repository-wide CI 门禁**。新增 `tools/verify_snndl_v5_p0.sh` 作为人工调用的、fail-fast 的验收脚本；它不接入现有 CI 配置。

脚本顺序固定为：contract generation drift -> Python unit -> v4 oracle -> C++ boundary/unit -> serial build/install -> ELI -> 四个真实 v4 SST run -> evidence audit。失败时保留运行目录并返回非零，不自动重试。

### 15.12 P0.10：评审与签字

P0 结束前做一次只读 review，逐项确认：

1. v5 schema 是否仍有隐式默认的性能关键参数。
2. artifact 是否存在第二份 route/weight 内容真值。
3. C++ 是否能绕过 typed address 直接按 edge 构造地址。
4. canonical ELI 是否加载旧 DMA、旧 multicast、3D 或 archive。
5. golden 是否只冻结功能而没有冻结 v4 假周期。
6. `performance` run class 是否在 P1-P4 未完成时被正确拒绝。
7. 所有 P0 命令是否在 clean build 和当前 SST 15.0.0 上重复通过。

评审结论写入 `docs/reviews/2026-xx-xx-v5-p0-signoff.md`；只有结论为 `accepted` 才进入 P1。

### 15.13 P0 精确文件清单

#### 外层 `remote/` 新增

```text
snndl_spec/v5.py
snndl_spec/contracts/v5/spec_contract.json
snndl_spec/contracts/v5/artifact_contract.json
snndl_spec/contracts/v5/statistics_contract.json
snndl_spec/contracts/v5/provenance_contract.json
snndl_system/v5_artifact.py
snndl_system/v5_evidence.py
snndl_system/test_v5_artifact.py
snndl_system/test_v5_evidence.py
snndl_system/testdata/v4_golden/{minimal_recurrent,response_reorder,fanout_multicore,zero_edge_scan}/...
tools/export_snndl_v5_contracts.py
tools/specs/mesh_minimal_v5.json
tools/test_snndl_v5_spec.py
tools/verify_snndl_v5_p0.sh
```

#### 外层 `remote/` 修改

```text
tools/snndl_spec_cli.py
tools/test_snndl_spec_cli.py
snndl_spec/__init__.py
snndl_system/__init__.py
.gitignore
TECH_PROGRESS.md                 append only
```

P0 不修改 `tools/run_snndl_with_time.sh` 的执行分发；v5 runtime 在 P1 有真实 SST component 后再接入。

#### SnnDL 子模块新增

```text
v5/api/V5Types.h
v5/api/AddressSpace.h
v5/api/ArtifactContract.h
v5/api/StatisticNames.h          generated snapshot
v5/contracts/README.md
build/v5_sources.am
tools/check_v5_boundaries.py
tests/test_v5_address_contract.cc
tests/test_v5_statistics_contract.cc
docs/adr/0001-v5-contract-boundary.md
docs/reviews/                    sign-off 时新增具体文件
```

#### SnnDL 子模块修改或归档

```text
Makefile.am
Makefile.in                      由 Automake 生成
Makefile                         由 configure/Automake 生成，不手改逻辑
build/extension_sources.am
tools/check_build_boundaries.py
api/README.md
platform/memory/README.md
tests/README.md
docs/README.md
docs/plans/README.md
archive/legacy_gas/README.md
```

归档列表严格使用第 15.7 节七个文件。实现前再次 `rg` 引用；若发现同簇活动文件，先加入迁移清单，不留下 forwarding header。

### 15.14 P0 工作包、依赖和检查点

| 顺序 | 工作包 | 前置 | 主要产物 | 检查点 |
|---:|---|---|---|---|
| 1 | P0.0 baseline refresh | 无 | 新 evidence + golden candidate | G0 v4 baseline |
| 2 | P0.1 authority/ADR | P0.0 | authority index + ADR | 文档批准 |
| 3 | P0.2 strict spec | P0.1 | resolver、contract、negative tests | G1 schema |
| 4 | P0.3 artifact/address | P0.2 | manifest/binary/typed address | G2 artifact |
| 5 | P0.4 statistics ABI | P0.2 | registry、generated header | G3 stats ABI |
| 6 | P0.5 archive old DMA | P0.1 | legacy move + manifest cleanup | boundary pass |
| 7 | P0.6 library/ELI closure | P0.4/P0.5 | v5 contract lib + slim aggregate | clean build/ELI |
| 8 | P0.7 golden fixtures | P0.0/P0.3 | curated oracle set | deterministic replay |
| 9 | P0.8 provenance | P0.2-P0.4 | run/evidence writer | manifest self-audit |
| 10 | P0.9 manual verification | 全部 | one-command verifier | G4 P0 acceptance |
| 11 | P0.10 sign-off | G4 | review record | P1 unlock |

任务并行只允许发生在无共同文件的 P0.3 与 P0.4；Makefile、contract version 和 documentation authority 必须串行合并。一个检查点失败时不得跳到后续阶段。

### 15.15 P0 手工验收命令

从外层 `remote/`：

```bash
./sst --version
python3 tools/snndl_spec_cli.py validate tools/specs/mesh_minimal_v4.json
python3 tools/snndl_spec_cli.py validate tools/specs/mesh_minimal_v5.json
python3 -m unittest snndl_system.test_synchronous_mesh
python3 -m unittest tools.test_snndl_v5_spec
python3 -m unittest snndl_system.test_v5_artifact snndl_system.test_v5_evidence
python3 tools/export_snndl_v5_contracts.py --check
```

从 SnnDL 子模块：

```bash
make -j1
make check-boundaries check-v5-boundaries
make test-timestep-core test-bcsr-source-contract
make test-v5-address-contract test-v5-statistics-contract
make install
```

ELI 和共享库检查：

```bash
sst_install_mpi/bin/sst-info SnnDL
sst_install_mpi/bin/sst-info SnnDLResearch
readelf -d sst_install_mpi/lib/sst-elements-library/libSnnDL.so
```

四个真实 v4 保护性回归：

```bash
SNNDL_V4_RUN_DIR=/tmp/snndl_p0_minimal \
  bash tools/run_snndl_with_time.sh --spec tools/specs/mesh_minimal_v4.json
SNNDL_V4_RUN_DIR=/tmp/snndl_p0_scale \
  bash tools/run_snndl_with_time.sh --spec tools/specs/mesh_scale_v4.json
SNNDL_V4_RUN_DIR=/tmp/snndl_p0_stress \
  bash tools/run_snndl_with_time.sh --spec tools/specs/mesh_stress_v4.json
SNNDL_V4_RUN_DIR=/tmp/snndl_p0_ramulator2 \
  bash tools/run_snndl_with_time.sh --spec tools/specs/mesh_ramulator2_v4.json
```

最终统一入口：

```bash
SNNDL_V5_P0_RUN_ROOT=/tmp/snndl_v5_p0_verify \
  bash tools/verify_snndl_v5_p0.sh
```

### 15.16 P0 交付物与退出标准

**交付物**

- 一份已批准 ADR 和四份 machine-readable contract。
- 一个严格 v5 resolver 与有效/无效 fixture。
- typed address、artifact header 和 statistics name 的跨语言测试。
- 已归档的旧 DMA 簇和不含它的活动构建。
- 可审计的 canonical ELI closure。
- 四个精简 v4 golden 和新的全套 v4 运行证据。
- 一个手工 P0 验收脚本和 sign-off 模板。

**全部退出标准**

1. v4 四个保护 fixture 全部通过且功能 digest 无漂移。
2. v5 valid spec resolve 后再 resolve 字节等价；所有 negative spec 精确失败。
3. artifact round-trip、corruption 和 region validation 全部通过。
4. generated statistics header 无 drift；名称/单位/scope 唯一。
5. active source/build/ELI 中无旧 DMA 簇、GAS 和 3D v5 dependency。
6. serial build/install、boundary、ELI 和 shared-object dependency 检查通过。
7. `performance` v5 run 在 P4 前 fail-closed，绝不回退 v4。
8. sign-off 文档列出的 blocker 为零。

### 15.17 P0 风险与禁止简化

| 风险 | 预防措施 |
|---|---|
| schema 过早固化错误参数 | 冻结字段语义，不冻结未经校准的数值；数值标记 provenance |
| Python/C++ contract 漂移 | 机器可读 authority + generated header + digest handshake |
| 归档 DMA 破坏旧显式库 | canonical v4 与 opt-in extension 分别检查，变更写 migration note |
| golden 被环境路径污染 | 只保存规范化功能字段和 digest |
| aggregate 瘦身后 ELI constructor 未加载 | `sst-info` 与 `DT_NEEDED` 双检查，不依赖链接器偶然行为 |
| P0 被扩张成 P1 | 明确禁止增加周期 Core 或宣称 v5 可运行 |

禁止为了保留旧调用者而留下 `PeDmaScheduler` forwarding header；禁止让 v5 resolver 接受缺字段后套用“reasonable defaults”；禁止把 P0 contract-only run 写成 timing evidence。

### 15.18 建议提交拆分与三层版本顺序

本路线图只规划提交，本轮不执行 Git 写操作。实施 P0 时按以下顺序形成可审阅提交：

1. SnnDL：`docs(snndl): define v5 cycle-model roadmap`
2. SnnDL：`chore(snndl): archive stage-coupled dma scheduler`
3. SnnDL：`feat(snndl): add v5 contract and build boundaries`
4. `sst-elements`：`chore(sst-elements): advance SnnDL for v5 contracts`
5. 外层：`feat(snndl): add strict v5 spec and artifact contracts`
6. 外层：`test(snndl): freeze v4 goldens and add p0 verification`
7. 外层：`docs(snndl): record v5 p0 acceptance`
8. 外层：更新 `sst-elements` pointer，并单独提交，避免与大量既有脏文件混合。

每次提交前只 stage 对应路径；SnnDL、`sst-elements`、外层仓库分别检查 status 和 diff。不得在外层一次提交中吞入当前无关研究改动。

## 16. P1：独立 SST Core 与周期流水

### 16.1 目标与范围

P1 把 `SnnCoreTile` 的一次性函数调用改造成独立 `SnnDL.SnnCoreV5` SST Component。P1 只回答计算流水的周期问题，使用解析可控的 ideal row/weight/local transport；真实 SRAM、cache/DRAM 和 Merlin 分别留给 P2-P4。

### 16.2 组件与文件

```text
v5/core/SnnCoreV5.{h,cc}
v5/core/CorePipeline.{h,cc}
v5/core/DeterministicRetireQueue.{h,cc}
v5/core/LifNeuronOp.{h,cc}
v5/events/CoreEvents.h
v5/test/IdealSynapseSource.{h,cc}
tests/v5/test_core_pipeline.cc
tests/v5/test_core_timing.py
build/v5_core_sources.am
```

`LifNeuronOp` 是纯函数；`CorePipeline` 拥有有限 stage/queue；`SnnCoreV5` 只负责 SST clock、ports、statistics 和生命周期。现有 `NextGenNeuronEngine` 作为行为 oracle，不直接承担 v5 宽度或时序。

### 16.3 端口与事件

- `control`：Start、Seal、Commit/Abort，携带 timestep。
- `spike_in`：本 Core 目标 spike，具备 link-level backpressure/ack。
- `spike_out`：提交后 held spike，仅在下一 Start 释放。
- `row_provider`：P1 ideal，P2/P3 替换为地址化 index/weight path。
- `status`：精确 token、egress closed、commit ready。

事件至少带 `timestep_id`、`source_neuron`、稳定 sequence、artifact row ID；不得携带期望 weight 值。

### 16.4 Spec 字段

P1 激活 `clocks.core` 与 `core.*`：每级 latency、width、queue entries、retire entries、accumulator width、neuron lanes 和 held-spike entries。所有 latency 可为 0 仅在 contract 明确允许的 combinational test mode；performance mode 最少 1 cycle。

### 16.5 周期规则

1. 每个 clock edge 先完成上周期 ready 项，再仲裁新 issue，避免同拍穿透多级。
2. 每级最多处理 `width` 项；queue full 向上游传递 backpressure。
3. memory/row response 可以乱序 ready，但 retire 依据 `(post_neuron, source_event_seq, edge_ordinal)` 稳定提交。
4. neuron scan 按 lane 数分批；不能一次提交所有 neuron。
5. CommitReady 只有在 pipeline、retire、delta 和 held-output transfer 均满足协议时发送。
6. 不同 clock domain 的事件只用 SST time 交接，不比较裸 local cycle。

### 16.6 统计

P1 必须产生 `core.ingress`、`core.row_lookup`、`core.synapse`、`core.retire`、`core.accumulator`、`core.neuron` 和 `core.held_spike` 的 accepted/issued/completed、occupancy、busy、full/stall cycle；另报 `timestep.core_elapsed_ns`。

### 16.7 测试与规模

- 1 Core、1 row、1 edge 的逐拍 trace 与手算完全一致。
- 分别把每级 width 设为 1/2/4，验证吞吐拐点。
- queue entries=1，制造整条反压链并保持 drop=0。
- 反转 response 顺序，最终 hash 不变而 retire wait 可变化。
- 64/256/1024 neuron scan，周期随 `ceil(neurons/lanes)` 变化。
- 1 PE 内 1/2/4/8 个独立 Core，消除 v4 静态 StandardMem slot 限制。

### 16.8 交付与退出条件

交付独立 Core ELI、ideal provider、解析 timing tests 和 v5 minimal SST runner。退出时 minimal fixture 与 v4 功能 digest 等价；至少三种 pipeline bottleneck 能由对应 stall stat 定位；改变无关 stage 参数不应改变目标微测周期。

### 16.9 风险与禁止捷径

主要风险是 SST callback 内无界 drain、同一 tick 多级穿透和错误的乱序浮点累加。禁止用 `while (!queue.empty())` 清空 stage；禁止把 ideal provider 延迟称为 memory timing；禁止把 P1 结果标记为完整芯片周期。

## 17. P2：显式 SRAM、容量、bank 与端口

### 17.1 目标与范围

P2 将 state、delta、index 和 route 从 C++ vector/map 变为真实请求驱动的局部存储。优先使用 SST `memHierarchy.Scratchpad` 的容量、backing、Get/Put 和请求协议；仅为 SST 15.0.0 缺少的 bank/port timing 增加 SnnDL backend，不修改 vendored memHierarchy。

### 17.2 组件与拓扑

每 Core 配置独立 state、delta、index Scratchpad；每 PE 配置 route Scratchpad。`SnnCoreV5` 通过 StandardMem/明确的 scratch interface 发起访问。推荐 backend：

```text
SnnCoreV5
  -> memHierarchy.standardInterface
  -> memHierarchy.Scratchpad
       -> memHierarchy.scratchpadBackendConvertor
            -> SnnDL.BankedScratchBackendV5
```

新文件集中在 `v5/storage/`，包括 bank mapper、有限 request queue、backend 和 region binder。旧 `BankedSramModel` 只可作为公式参考，不能以 observe-only 模式进入 v5。

### 17.3 Spec 字段

激活每个 SRAM 的 size、banks、ports/bank、interleave、request width、read/write latency、frontend queue、response/cycle、backing 和 overflow policy。P2 只允许 `overflow_policy=fail`；refill/spill 在 P3 才开放。

### 17.4 正确性与时序规则

1. setup 根据 manifest region 计算实际 resident bytes，超过 capacity 立即失败。
2. 访问先进入有限 queue，按 address 选 bank；每 bank 每 cycle 最多服务 port 数。
3. latency 从真正获得 bank service 的周期开始，不从 enqueue 开始伪算。
4. state read 与 write 的顺序保护 timestep snapshot；delta reset 不得一次性清 vector。
5. bank conflict 必须延迟 response，`predicted_extra_cycles` 不得作为完成机制。
6. backing=`none` 只能用于不检查数据的 timing microtest；功能/性能运行要求真实 backing。

### 17.5 统计

每个 region 报 reads/writes/bytes、request/response、queue occupancy、bank busy、bank conflict、port stall、capacity、resident peak 和 latency histogram。统计中区分 state、delta、rowptr、colidx、route，不能合成一个 `sram_accesses`。

### 17.6 测试与规模

- 同 bank 与均匀 bank 的解析周期对照。
- read/write 同 bank、单/双端口、不同 interleave。
- 刚好 fit、一字节超限、region overlap 和错误 owner。
- 256/1024 neuron 的 state scan；fan-in 对 delta bank 的热点。
- route table 多 Core fanout，验证 PE-shared route SRAM 仲裁。
- 与 P1 ideal storage 对比：功能等价，周期差等于真实排队与服务时间。

### 17.7 交付与退出条件

交付 SST-native Scratchpad 图、banked backend、typed region binding 和容量报告。至少一个 microtest 的额外周期精确等于 bank service rounds；所有 overflow fail-closed；功能 digest 与 v4/P1 一致。

### 17.8 风险与禁止捷径

风险是双重容量、地址 alias、Scratchpad response queue 与 backend queue 重复计 stall。每一级必须定义唯一 ownership。禁止同时保留 vector 作为功能真值；禁止“统计访问 Scratchpad、实际读 vector”；禁止用 GPU warp broadcast 语义作为默认 SNN SRAM 语义，除非另立配置和实验。

## 18. P3：Weight cache hierarchy、DRAM 与干净 DMA

### 18.1 目标与范围

P3 建立用户要求的完整层级：每 Core 私有 noncoherent L1、每 PE Core 共享 noncoherent L2、全芯片共享一个 DRAM controller/backend，并支持显式 weight/index preload、refill 和 spill。

### 18.2 SST 对象图

```text
SnnCoreV5 StandardMem
  -> private memHierarchy.Cache L1
  -> PE-local memHierarchy.Bus
  -> shared memHierarchy.Cache L2
  -> chip memory fabric
  -> memHierarchy.MemController
  -> simpleMem | ramulator2
```

state/delta/route 永不进入 cache。rowptr/colidx 是否 cache-backed 由 manifest residency 明确；weight backing image 始终在 ChipDram。P3 小规模可使用显式 Bus；进入 P6 前必须用可扩展的 memHierarchy MemNIC/fabric 替代全芯片单总线，并保持同一单 DRAM owner。

### 18.3 新 DMA 设计

`SnnDmaEngineV5` 是 PE 级 SST Component，只理解：

```text
descriptor_id, timestep_id(optional)
src_space/src_owner/src_addr
dst_space/dst_owner/dst_addr
bytes, burst_bytes, completion_token
```

它使用 `StandardMem::MoveData` 驱动 Scratchpad Get/Put，拥有有限 channel、descriptor queue、outstanding 和 byte/cycle。它不知道 neuron、GAS stage、priority class 或 workload。调度基线为 FIFO；其它 policy 必须作为 P3 单独可比较配置。

### 18.4 Spec 字段

激活 cache L1/L2 和 DMA 全部字段，补充 memory fabric、write policy、prefetch/residency plan、cold/warm start 和 address mapping。P3 基线固定 `coherence_protocol=none`、read-only weights、no writeback learning。

### 18.5 请求所有权与背压

- Core task 在成功分配 request queue/retire entry 后才能发出 read。
- StandardMem request ID 映射到唯一 task/timestep；response 未知或重复立即 fatal。
- L1 miss、MSHR full、Bus/L2/DRAM backpressure 必须由 SST 原生路径体现。
- DMA 和 demand request 在共享 L2/DRAM 处真实竞争；不在 summary 中人为叠加延迟。
- preload 未完成时 Start 由 token 阻塞；watchdog 只报告队列快照。

### 18.6 统计

除 SST Cache/MemController 原始统计外，增加 task-to-request 映射、demand/DMA bytes、cold/warm miss、MSHR full、Bus/fabric stall、preload elapsed、resident/evicted/spilled bytes。DRAM read/write/row hit/miss/conflict 直接引用 backend 统计并记录来源。

### 18.7 测试与规模

1. working set < L1、介于 L1/L2、> L2 三点容量曲线。
2. cold 与 warm 两次相同 timestep；功能相同，cache events 不同。
3. 两 Core 共享 line，验证共享 L2 命中而无 coherence traffic。
4. DMA preload on/off、不同 burst/channel/outstanding。
5. demand 与 DMA 冲突，验证公平性和无 starvation。
6. SimpleMem 解析 smoke 与 Ramulator2 row-local/row-conflict trace。
7. 1/4/16 PE 同时访问单 DRAM，验证 controller queue 压力。

### 18.8 交付与退出条件

交付完整 cache/DRAM 图、clean DMA、preload plan 和 memory evidence。退出要求所有 request/response/MoveData token 闭合；三层容量拐点可由原始 cache 统计解释；SimpleMem 与 Ramulator2 功能 digest 一致；切换 backend 不改变地址或 graph 内容。

### 18.9 风险与禁止捷径

风险包括 Bus 成为非目标瓶颈、cache 默认 coherence 引入伪流量、DMA 与 demand 双重读取和 Ramulator2 config 不匹配。禁止使用当前 PE-local weight map 作为 cache hit；禁止从 descriptor 直接取 weight；禁止在 Ramulator2 后端外再加固定 DRAM delay；禁止把两个 DRAM 能耗源相加。

### 18.10 P3.5 集成签收（2026-08-05）

P3.5 已把 P1/P2 Core 与 P3 memory hierarchy 合成真实 SST 闭环。权重镜像在
init 阶段写入 ChipDram，`SnnDmaEngineV5` 仅接受
`ChipDram <-> PeWeightSpm` typed descriptor；DMA 完成 token 到达前，driver
不会发送 `Start`。row request 随后通过 StandardMem timed read 从已预取 SPM
读取 16-byte edge record，解码后交给 `SnnCoreV5`，state/delta/index/route
继续使用 P2 typed SRAM。

SimpleMem、Ramulator2、2 PE x 2 Core、queued row、非法 address-space 负例以及
完整 P1/P3 回归均已通过。签收文档为
`docs/reviews/2026-08-05-p35-integration-signoff.md`，权威运行证据为
`snndl_runs/p35_v5_acceptance_20260805/acceptance.json`。P3.5 仍是
development timing evidence；Merlin NoC 完成前不开放 performance claim。

## 19. P4：Merlin 2D unicast NoC 与最小可信周期闭环

### 19.1 目标与范围

P4 用 SST Merlin `hr_router`、`merlin.mesh` topology 和 `SST::Interfaces::SimpleNetwork` 替换 v4 的方向 Link/手写 packet queue。P4 只实现 unicast/source-replication baseline，不实现 router-native multicast。

P4 通过后，P1-P4 共同构成最小可信周期模型，首次允许 `run.class=performance` 和 claim level T。

### 19.2 组件与对象图

```text
SnnCoreV5
  <-> PeEndpointV5
       <-> merlin.linkcontrol (SimpleNetwork)
            <-> merlin.hr_router mesh
```

`PeEndpointV5` 负责 logical delivery 与 packet codec，不自行实现 hop queue。每个 PE 一个 endpoint；PE 内 Core dispatch 使用独立有限 local queue/arbiter，不伪装成网络 hop。

### 19.3 Packet contract

header 至少包含 format version、timestep、source neuron、source PE/Core、event sequence、destination PE/Core 和 payload count。header/payload bytes 进入真实 packet size；SimpleNetwork request size 与统计一致。P4 一个物理包对应一个目标 PE，PE 内可含多个 Core delivery entry。

### 19.4 Spec 字段

激活 `noc.implementation/topology/routing/link_bw/flit_size/xbar_bw/input_latency/output_latency/input_buf_size/output_buf_size/num_vns` 和 endpoint queue。data/control VN 分离；P4 control VN 只可承载验证控制，不把 out-of-band Coordinator 时间算入芯片性能。

本地 SST 15.0.0 `sst-info merlin.hr_router` 的参数名是实现权威。线上 SST 16 文档只作概念参考，不能复制未在本地 ELI 中出现的字段。

### 19.5 时序与背压规则

1. endpoint 调用 `spaceToSend()` 成功后才发送；否则保留原 packet 和 token。
2. receive callback 只通知，实际每周期最多 dequeue 配置宽度。
3. flit serialization、router/xbar、link buffer 和 credit 由 Merlin 决定。
4. endpoint local queue 满时必须通过 SimpleNetwork 接口停止接收或保留在受控 buffer，不 drop。
5. XY 路由确定性；P4 不引入 adaptive route。
6. SST rank 改变不得改变 packet 内容、logical delivery 或功能 hash。

### 19.6 统计

组合但不重写 Merlin raw stats：send bits/packets、output stalls、xbar stalls、idle time；SnnDL 另报 logical deliveries、packet/flit/header/payload bytes、source queue occupancy、endpoint receive stall、hop count 和 end-to-end latency histogram。

### 19.7 解析测试与压力测试

- 零跳 local、单跳、直线多跳、转弯多跳的 latency 差分。
- packet 恰好 1 flit 与跨 2/4 flit，验证序列化阶梯。
- link_bw、xbar_bw 分别降半，确认对应资源成为瓶颈。
- uniform random、transpose、single-hotspot 和 all-to-one traffic。
- input/output buffer 为 1-2 flit，触发 credit/backpressure。
- 2x2、4x4、8x8；1/2/4 rank 功能一致。
- P3 固定 memory fixture 下的端到端 SNN stress。

### 19.8 交付与退出条件

交付 Merlin mesh builder、endpoint、codec、NoC microbench 和 v5 performance runner。退出要求：

1. 不再构造 v4 north/south/east/west data links。
2. packet/flit/bit 计数按定义闭合，queue drop=0。
3. 无争用延迟和 flit 阶梯与配置解析一致。
4. congestion test 出现 Merlin stall 且最终 drain。
5. P1-P4 end-to-end 与 v4 功能等价。
6. summary 的每个瓶颈结论能追溯到 raw SST stats。

### 19.9 风险与禁止捷径

风险是 Python topology 参数与本地 ELI 不一致、endpoint 无限缓冲、packet size 与计费不一致、control traffic 污染 data。禁止同时运行手写 Link NoC 与 Merlin 后选较快结果；禁止把 `logical_tx` 当 packet；禁止关闭 credit 后仍声称 buffer 有限；禁止用平均 hop 代替实测 latency。

## 20. P5：Multicast 与有成本的 timestep 同步

### 20.1 目标与范围

P5 在已经可信的 P4 unicast 基线上增加两项独立能力：router/branch 级 multicast 和芯片内 timed synchronization。二者必须分别开关和消融，不能把同步流量节省误记为 spike multicast 收益。

### 20.2 Multicast 两级基线

P5 固定两个可比较模式：

1. `source_replication`：源 endpoint 为每个目标 PE 产生 unicast packet。它是功能和性能基线。
2. `router_replication`：源只发送一个带 group ID 的 packet，由合法分支点复制；复制后的所有叶子投递与 source replication 完全等价。

旧 `SpikeKey/TileKey`、INTER/INTRA stage、compact mask 和旧 `MulticastRouter` 不进入 v5。新 multicast format 只表达 group、epoch、tree version 和叶子集合，不表达 GAS/window 状态。

### 20.3 Route/group artifact

`route.bin` 在 P5 扩展为版本化目录：

```text
source_neuron -> route_group_id
route_group_id -> ordered leaf PE/Core set
route_group_id + router_id -> output-port bitmap
tree_version / placement_digest / route_digest
```

branch entry 必须由外部 mapper/compiler 生成并经独立 validator 检查：无环、所有 leaf 恰好一次、无越界 port、tree 与 mesh/placement digest 一致。router 只消费 entry，不在运行时重新求树。

### 20.4 Router/endpoint 设计

优先使用 Merlin 的接口、buffer、flit、credit 和 topology extension point。若 SST 15.0.0 的 `hr_router` 不能在不修改语义的情况下完成 branch replication，则新增窄的 `SnnMulticastTopologyV5`/port-control extension；不得 fork 整个 router 或复制其 buffer/credit 实现。

每个 packet 带唯一 `(timestep, source_pe, source_core, event_seq, group_id)`。leaf endpoint 维护有限 replay/dedup window，仅用于检测协议重复并 fatal；不能静默去重来掩盖树错误。

### 20.5 Timed synchronization

P5 提供两种明确模式：

| 模式 | 用途 | 是否进入性能时间 |
|---|---|---|
| `oracle_out_of_band` | 功能调试、与 v4 对照 | 否，run class 只能 functional |
| `timed_tree` | 性能实验 | 是，所有 control packet/queue/link 都计时 |

`timed_tree` 采用有根 reduction + broadcast：PE/Core local drain -> PE reduce -> tree root 判断全局 tx/rx/token 闭合 -> Seal/Commit broadcast。control 使用独立 VN，但共享 router/xbar/link 的真实资源；是否给予优先级必须显式配置并消融。

### 20.6 Spec 字段

激活 `noc.multicast_mode`、group/table capacity、branch width、tree source、control/data VN、sync topology、fan-in、packet bytes、priority、timeout 和 root placement。performance mode 强制 `sync_mode=timed_tree`；timeout 只终止，不提交。

### 20.7 统计

- multicast：logical leaves、source packets、branch replications、tree edges、flits/bits、table lookup、table miss、branch stall、compression ratio。
- synchronization：reduce/broadcast packet/flit/bytes、per-level latency、root wait reason、barrier wait、control/data contention、timestep sync elapsed。
- 对照字段：source-replication physical traffic、router-replication physical traffic、相同 logical delivery count。

compression ratio 只可按明确分母报告，例如 `source_replication_flits / router_replication_flits`；不能用 logical/physical 混合单位。

### 20.8 测试与规模

1. 1/2/4/16 leaf 的直线、矩形和稀疏树。
2. tree 中每种 turn、同一 router 多 branch 和最深叶子。
3. route entry 损坏、缺 leaf、重复 leaf、环和错误 tree version 的负测试。
4. source replication 与 router replication 的逐步 hash/logical count 等价。
5. control VN 无竞争、与 data hotspot 竞争、不同 priority 的同步延迟。
6. 2x2/4x4/8x8 和 1/2/4 rank，验证 tree 与 partition 无关。
7. multicast table 容量不足时 setup fail，不回退 source replication，除非 spec 明确选择后者。

### 20.9 交付与退出条件

交付无旧 stage 语义的 group format、validator、Merlin extension、timed sync tree 和两组独立消融。退出要求 leaf set 精确等价、零 duplicate/drop、物理流量统计闭合；performance run 不依赖 free Coordinator；同步开销能与数据面开销分别报告。

### 20.10 风险与禁止捷径

最大风险是把 source-side grouping 误称 router multicast、用静默 dedup 掩盖重复、或让 control VN 免费。禁止从运行时 edge list临时建树；禁止 capacity miss 自动转 unicast 后仍标记 router multicast；禁止在性能摘要中减掉 sync 时间；禁止将 control packet 排除于共享 link/xbar contention。

## 21. P6：Artifact 工具链、NIR/workload 接入与规模化

### 21.1 目标与范围

P6 把已经验证的芯片模型连接到稳定的外部模型和数据集，并证明结果可以跨规模、跨 SST rank 重复。SnnDL 仍不负责 mapping optimization；它只消费并严格验证 placement/route/BCSR artifact。

### 21.2 工具链边界

```text
source model / NIR / synthetic graph
        |
        v
semantic importer
        |
        v
external mapper: neuron -> PE/Core
        |
        v
artifact compiler: placement + route + BCSR + stimulus + manifest
        |
        +--> Python v4 oracle
        +--> SST v5 model
```

importer 负责模型语义；mapper 负责 placement 和 route policy；artifact compiler 负责确定二进制布局。三个阶段各自输出 digest，错误不能在下一阶段被修补。

### 21.3 NIR 支持边界

首版只声明经过逐算子和端到端验证的 NIR 子集：Input、LIF/LI、Affine/Linear 或显式 synapse graph、Delay 的受限形式和 Output。unsupported operator、动态 shape、训练状态或连续时间语义必须在 import 阶段失败。

NIR 只作为算法交换层，不携带 PE/Core、bank、cache 或 router 参数。模型变换、量化和 delay 离散化都必须写入 transformation manifest，保留原模型 digest 和变换工具版本。

### 21.4 Workload 层级

P6 使用三级 workload，而不是一上来只跑大型网络：

| 层级 | 内容 | 目的 |
|---|---|---|
| W0 | P1-P5 解析 microbench | 保持周期归因 |
| W1 | synthetic sparse/recurrent/fanout/hotspot | 扫描单一压力维度 |
| W2 | 至少一个 event-audio 与一个 event-vision 网络 | 展示真实活动与拓扑分布 |

候选可以来自 SHD/SSC、N-MNIST 或 DVS Gesture，但只有完成模型来源、预处理、准确率/功能对照和 artifact provenance 的实例才进入正式 cohort。NeuroBench 用于 workload/metric 组织，不自动赋予硬件真实性。

### 21.5 Scale matrix

规模验证不做完整笛卡尔积，而采用覆盖关键资源拐点的分层矩阵：

| Tier | PE mesh | Core/PE | neuron/Core | edges | active spike/step | timestep | rank |
|---|---|---:|---:|---:|---:|---:|---:|
| S0 | 1x1/2x2 | 1-2 | 64 | 10-1K | 1-32 | 2-10 | 1 |
| S1 | 4x4 | 4 | 64-256 | 10K-100K | 0.1%-10% | 10-100 | 1-2 |
| S2 | 8x8 | 4-8 | 256-1024 | 100K-1M+ | workload-derived | 100+ | 1-4+ |

每个点必须报告五个规模轴、artifact bytes、实际活动、资源容量和 rank partition；不能只用网络名称或 total neuron 标签。

### 21.6 MPI/partition 规则

1. 组件 ID、PE/Core owner 和 artifact 地址不依赖 rank。
2. partition 只改变 SST 进程归属，不改变 route/tree/physical address。
3. 跨 rank event 使用可序列化固定版本，禁止发送 raw pointer/host path。
4. 1/2/4 rank 必须产生相同功能 digest、logical count 和 packet count；wall time 不要求相同。
5. rank-local stats 先保存 raw，再按 contract 聚合；histogram 不可按均值再平均。

### 21.7 Spec 与 artifact 扩展

P6 激活 source provenance、model operator inventory、mapping tool/version/policy、partition policy、dataset split、sample range、preprocess digest 和 run cohort ID。任何随机 mapping/stimulus 必须保存 seed 和生成后的 artifact digest；正式运行不得在 SST setup 中重新随机生成。

### 21.8 统计与结果组织

结果按 `workload -> sample/case -> topology -> config -> repeat` 保存。报告至少包括 timestep latency distribution、throughput、synop/s、logical/physical traffic、SRAM/cache/DRAM working set、stall breakdown 和 peak simulator memory。不同 case 不先平均后丢失个体结果。

### 21.9 验证

- importer 单算子与小网络对照。
- artifact compiler deterministic rebuild：相同输入逐字节相同。
- v4/v5 每 timestep digest 和输出 spike train 对照。
- mapping permutation：功能不变，流量/周期可变。
- 1/2/4 rank determinism。
- S0 全覆盖、S1 代表点、S2 至少一个完成点；失败/超时保留状态，不从 cohort 静默删除。
- repeat run 的 deterministic raw count 精确一致；宿主 wall time 只作仿真成本信息。

### 21.10 交付与退出条件

交付 importer、artifact compiler、受支持算子清单、workload manifest、scale launcher 和 per-case aggregator。退出要求至少两个真实 workload 完成功能校验；S0/S1 matrix 完成且 S2 有非平凡点；多 rank 功能/流量确定；任一图可从 source provenance 重建相同 artifact。

### 21.11 风险与禁止捷径

风险是模型转换改变语义、只扩 neuron 不扩活动、失败样本被过滤和多 rank 聚合错误。禁止把 mapping 优化器嵌进 SnnDL runtime；禁止用 synthetic accuracy 代替原任务功能检查；禁止把未完成 cohort 表述为整体趋势；禁止将 sample、case、repeat 混合平均而不保留明细。

## 22. P7：能耗、面积、校准与最终声明

### 22.1 目标与范围

P7 不改变 P1-P6 的功能或周期事件。它把已验证的 action count 映射为可追溯的能耗/面积，并通过微测、工具交叉检查和敏感性分析确定可声明范围。

### 22.2 Action contract

每个硬件块定义互斥、可计数 action：

| 模块 | 典型 action |
|---|---|
| Core | row lookup、synapse lane op、accumulate、LIF update、queue access |
| SRAM | read/write，按 region/width/bank geometry |
| Cache | tag/data lookup、hit、refill、evict/writeback |
| NoC | buffer read/write、xbar traversal、link bit-hop、router replication |
| Sync | control packet/flit 与 sync controller op |
| DMA | descriptor、burst issue、moved byte |
| DRAM | backend command/energy counters |

action count 来自 P1-P6 raw stats，不从 summary 的平均值倒推。每项 action 有 source statistic、coefficient source、unit、voltage/frequency/technology metadata 和 contract digest。

### 22.3 能耗模型

```text
E_dynamic = sum(action_count_i * energy_per_action_i)
E_leakage = leakage_power_i * active_elapsed_time_i
E_total   = E_core + E_sram + E_cache + E_noc + E_sync + E_dma + E_dram
```

片上结构优先用 Accelergy-compatible component/action tables。DRAM 每次运行只能选择一个 primary source：Ramulator2 自带能耗路径或 DRAMPower；另一个可做交叉检查但不能相加。所有结果统一 pJ/nJ/mJ 并保留未舍入值。

### 22.4 面积模型

面积按实例数和实际 geometry 汇总：Core pipeline、SRAM bank、cache、router/link endpoint、DMA/sync controller。只有 coefficient 有明确 technology/process/voltage provenance 时才允许总面积结论；否则只报告结构容量、实例数和 normalized area sensitivity。

Loihi 2/SpiNNaker2 的公开数据只用于合理性范围，不作为 SnnDL 模块逐项面积或能量的直接标定值。

### 22.5 校准层级

| 层级 | 方法 | 通过条件 |
|---|---|---|
| C0 contract | action/count/unit 守恒 | 无漏项、无重复计费 |
| C1 component | Core/SRAM/cache/NoC/DRAM microbench | 只改变目标 action，趋势单调 |
| C2 timing | SST 解析周期与 backend/tool 输出 | 参数作用方向和数量级一致 |
| C3 cross-tool | Accelergy/Ramulator2/DRAMPower 适用部分 | 差异被来源和假设解释 |
| C4 system | workload + scale sensitivity | bottleneck/energy breakdown 可追溯 |

没有真实芯片测量时，最终状态写 `tool-calibrated model`，不能写 silicon-validated。

### 22.6 Spec 与输出

P7 激活 technology node、voltage、temperature、coefficient set ID、tool/version、leakage policy 和 DRAM energy source。输出目录：

```text
energy_area/action_counts.json
energy_area/coefficients.json
energy_area/provenance.json
energy_area/component_energy.json
energy_area/component_area.json
energy_area/sensitivity.json
energy_area/validation.log
```

### 22.7 实验

- 每个 action 的 single-action/linear sweep。
- SRAM size/bank/port、cache size/assoc、flit/link width 的敏感性。
- cold/warm、DMA on/off、unicast/multicast、oracle/timed sync 的能耗消融。
- SimpleMem 不做 DRAM device energy 结论；Ramulator2/DRAMPower 配置必须与 P3 memory geometry 一致。
- 至少报告 energy/timestep、energy/spike、energy/synop、average power、peak storage 和 area breakdown。
- 性能与能耗使用同一 run/artifact，不拼接不同运行的 numerator/denominator。

### 22.8 交付与退出条件

交付 action schema、coefficient set、工具 adapter、component/system report 和 calibration dossier。退出要求全部 action 有 provenance；重复计费审计通过；C0-C3 完成，C4 至少覆盖 P6 正式 workload；能耗、面积和性能共享同一 run ID。

### 22.9 风险与禁止捷径

风险是技术节点混用、DRAM 双计、用未校准旧模型系数、漏算 idle/leakage 和归一化分母漂移。禁止把公开芯片总功耗按 Core 数均分作为系数；禁止把 Ramulator2 command count 与其总 energy 再同时计费；禁止把 normalized proxy 写成 pJ；禁止在 P7 为匹配预期结果修改 P1-P6 周期。

## 23. 跨阶段验证矩阵

| 能力 | P0 | P1 | P2 | P3 | P4 | P5 | P6 | P7 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| strict v5 spec/artifact | R | R | R | R | R | R | R | R |
| v4 functional oracle | R | R | R | R | R | R | R | R |
| Core analytic timing | - | R | R | R | R | R | R | R |
| SRAM capacity/bank timing | - | - | R | R | R | R | R | R |
| cache/L2/DRAM timing | - | - | - | R | R | R | R | R |
| Merlin flit/credit timing | - | - | - | - | R | R | R | R |
| router multicast/timed sync | - | - | - | - | - | R | R | R |
| NIR/workload/multi-rank | - | - | - | - | - | - | R | R |
| energy/area calibration | - | - | - | - | - | - | - | R |

`R` 表示该阶段和以后所有阶段都必须重跑对应保护测试。阶段不能删除早期测试来缩短验收；若参数变化使 analytic expected 改变，必须同时给出公式和 review。

### 23.1 固定实验对照原则

1. 对照实验除目标自变量外使用同一 spec、artifact、seed、SST build 和 rank partition。
2. warm/cold、SimpleMem/Ramulator2、unicast/multicast 属于不同实验臂，标签不可省略。
3. 功能等价先于性能比较；任一 digest/invariant 失败时该运行不进入性能聚合。
4. performance run 至少一次 warm-up 与明确的 measured range；同步 SNN timestep 本身不是宿主 warm-up。
5. deterministic simulator count 应逐字一致；宿主执行时间和峰值 RSS 可波动，单独报告。

## 24. v5 证据目录与发布契约

### 24.1 目录布局

```text
snndl_runs/v5/<cohort>/<run_id>/
  status.json
  user_spec.json
  effective_config.json
  run_manifest.json
  artifact_manifest.json
  command.txt
  stdout.log
  stderr.log
  raw/
    sst_stats.csv
    rank_<n>_sst_stats.csv
    timestep_trace.jsonl
    merlin_stats.csv
    memhierarchy_stats.csv
    ramulator2_stats.txt
  derived/
    essential_summary_v5.json
    validation.json
    bottleneck_breakdown.json
  energy_area/                 P7 optional before phase completion
```

`run_id` 使用 UTC timestamp + effective-config digest 前 12 位，不使用“final”“latest”“good”等不可审计名称。大 artifact 放在内容寻址仓库，run 目录保存 manifest 和 SHA-256；发布 bundle 必须能解析全部引用。

### 24.2 状态机

```text
prepared -> running -> completed
                   \-> failed
                   \-> timed_out
                   \-> aborted
```

只有 `completed` 且 validation 全通过的 run 可进入正式 aggregate。失败目录不得删除或改写为 completed。派生 summary 使用原子临时文件 + rename；run manifest 完成后写最终 file inventory 和 digest。

### 24.3 `essential_summary_v5.json` 最小结构

```text
contract_version / claim_level / run_class
topology / clocks / artifact digests
execution status / timestep range / elapsed_ns
functional invariants and digests
per-timestep latency and activity
core/storage/cache/dma/noc/sync/dram raw-stat references
bottleneck classification with evidence keys
energy/area reference (P7)
warnings / excluded metrics / provenance digest
```

summary 不复制所有 raw stats，也不能成为独立真值。每个派生字段必须能由 `source_stat_keys` 找回分子、分母和 aggregation。

### 24.4 发布检查

- 所有相对链接可解析，所有内容 hash 匹配。
- user spec 与 effective config 同时存在。
- artifact/model/mapping/runtime provenance 完整。
- raw per-rank 与 aggregate 均保留。
- 无绝对开发机路径作为唯一 artifact locator。
- 无 API key、credential、host-private environment dump。
- generated run output 默认不入 Git；只有明确选定的 golden/reference evidence 例外。

## 25. 总体风险登记册

| ID | 风险 | 影响 | 最早检测 | 缓解/关闭条件 | 阶段 |
|---|---|---|---|---|---|
| R1 | v4 oracle 本身漂移 | 功能比较失效 | golden digest | P0 冻结行为与版本 | P0 |
| R2 | spec/ELI 参数双重默认 | 实际架构不可知 | effective config diff | strict resolver，builder 禁止覆写 | P0 |
| R3 | route/weight 双真值 | 错误目标或权重 | artifact digest/shape | 单一 manifest/BCSR image | P0 |
| R4 | GAS/3D 源重新渗入 | 新架构边界失效 | source/DT_NEEDED scan | 独立 v5 manifest 和负面 token 检查 | P0 |
| R5 | callback 无界处理 | 虚假吞吐 | per-tick issue trace | 有限 width/queue | P1 |
| R6 | 乱序 response 改变浮点结果 | 非确定功能 | reorder test | 稳定 retire order | P1 |
| R7 | SRAM 只观测不阻塞 | 虚假周期 | conflict analytic test | request-driven backend | P2 |
| R8 | vector 与 Scratchpad 双存储 | 内容漂移 | mutation/digest test | Scratchpad 唯一 runtime state | P2 |
| R9 | cache coherence 伪流量 | 错误瓶颈 | protocol event audit | noncoherent/read-only baseline | P3 |
| R10 | DMA 与 demand 双计 | 流量/能耗虚高 | request-token ledger | descriptor/request 唯一 ownership | P3 |
| R11 | chip Bus 非目标饱和 | 规模趋势错误 | fabric stall breakdown | P6 前切换可扩展 memory fabric | P3/P6 |
| R12 | endpoint 隐式无限缓冲 | NoC stall 被隐藏 | occupancy/capacity audit | 全队列显式有限 | P4 |
| R13 | logical/packet/flit 单位混淆 | 通信收益错误 | conservation equations | 分层统计 contract | P4 |
| R14 | multicast 漏/重 leaf | 功能错误 | leaf-set validator | versioned tree + exact delivery | P5 |
| R15 | 同步面免费 | timestep 性能乐观 | oracle/timed diff | performance 强制 timed sync | P5 |
| R16 | workload 转换改语义 | 实验无效 | importer/v4 comparison | 受限算子与 transformation manifest | P6 |
| R17 | 只增 neuron 不增工作量 | 虚假 scale | 五轴规模报告 | activity/edge/timestep matrix | P6 |
| R18 | MPI 聚合不正确 | 总量与延迟错误 | 1/2/4 rank compare | raw per-rank + typed aggregation | P6 |
| R19 | 能耗 coefficient 无 provenance | PPA 不可信 | coefficient audit | tool/version/technology metadata | P7 |
| R20 | DRAM energy 双计 | 总能耗错误 | source ledger | 每 run 单一 primary DRAM source | P7 |

高影响风险 R1-R4、R6-R8、R12-R16 和 R19-R20 未关闭时，对应阶段不得签字。风险状态只可为 `open/mitigated/closed`，不能用“暂时看起来正常”。

## 26. 已冻结决策与待决策日志

### 26.1 已冻结，不在 P0 重新讨论

1. 只建模 2D Mesh；GAS 和 3D 永久不属于 v5。
2. 同步 timestep、counted drain、下一步释放 held spike。
3. LIF + BCSR 是首个闭环；mapping 在 SnnDL 外部。
4. Core 私有 L1、PE 共享 L2、芯片共享单 DRAM。
5. state/delta/index/route 是显式 SRAM，weights 是 DRAM-backed 数据。
6. cache baseline noncoherent；SST 原生组件优先。
7. P4 使用 Merlin；P7 只消费既有 action，不改 timing。

### 26.2 待决策项

| Decision ID | 问题 | 最迟阶段 | 选择依据 | 默认状态 |
|---|---|---|---|---|
| D1 | neuron/state/weight 精度与量化 | P1 签字前 | v4 等价 + workload 精度 | binary32 functional baseline |
| D2 | accumulator 数值顺序/宽度 | P1 签字前 | reorder determinism | stable binary32 |
| D3 | SRAM bank mapping | P2 签字前 | conflict microbench | low-order interleave |
| D4 | index 常驻、cache 或 DMA residency | P3 | capacity/traffic sweep | rowptr resident, values DRAM |
| D5 | PE/Chip memory fabric | P3/P6 | Bus bottleneck 与 scale | small Bus, scale fabric TBD |
| D6 | cache inclusion/write policy | P3 | memHierarchy 支持与 microbench | noninclusive, read-only weights |
| D7 | native multicast extension 位置 | P5 | 本地 Merlin 15 API probe | topology/port extension preferred |
| D8 | sync tree fan-in/root placement | P5 | control traffic sweep | deterministic tree |
| D9 | 正式 NIR operator subset | P6 | importer test coverage | minimal verified subset |
| D10 | 正式 workload cohort | P6 | provenance/semantic readiness | 未选择 |
| D11 | 片上 coefficient source | P7 | Accelergy plug-in coverage | 未选择 |
| D12 | DRAM primary energy source | P7 | backend/config completeness | per-run explicit |

任何决定必须新增 ADR 或更新既有 ADR，并提高 contract version（若 wire/spec 变化）。默认状态不是经过校准的硬件结论。

## 27. P0-P7 完成阶梯与允许声明

| 阶段完成 | 系统状态 | 允许声明 | 不允许声明 |
|---|---|---|---|
| P0 | contract-closed | 输入、地址、统计和依赖边界已封闭 | v5 可运行或有周期精度 |
| P1 | compute-timed | Core 流水的解析周期与 stall | 完整 PE/芯片周期 |
| P2 | local-memory-timed | SRAM 容量、bank/port 争用 | cache/DRAM/NoC 瓶颈 |
| P3 | hierarchy-timed | L1/L2/DRAM/DMA 时序和流量 | flit/credit 网络结论 |
| P4 | claim level T | 最小完整 2D 芯片周期、吞吐、瓶颈 | native multicast、真实 workload scale、PPA |
| P5 | communication-complete | multicast 与 timed sync 成本/收益 | workload 普适趋势 |
| P6 | claim level S | 指定 cohort 的 scale/workload 趋势 | silicon 能耗/面积 |
| P7 C0-C3 | claim level E | tool-calibrated energy/area | silicon-validated PPA |
| P7 + silicon data | claim level V | 校准范围内硬件验证 | 超出校准域的泛化 |

文档、图表和论文必须携带 claim level。较高阶段不自动修复较低阶段的失败；例如 P7 有完整能耗表，但 P4 NoC invariant 失败，仍不能发布系统能耗。

## 28. 阶段里程碑与资源优先级

路线图使用依赖和证据里程碑，不给出没有人力/机器预算依据的日历承诺：

| 里程碑 | 覆盖阶段 | 主要价值 | 优先级 |
|---|---|---|---|
| M0 Contract Freeze | P0 | 阻止继续积累歧义 | 立即 |
| M1 Timed Core | P1 | 建立真实计算周期 | 高 |
| M2 Timed PE Memory | P2-P3 | 完成计算-存储闭环 | 高 |
| M3 Timed Chip | P4 | 最小可信架构模型 | 最高研究里程碑 |
| M4 Communication | P5 | 评估保留的通信优化 | M3 后 |
| M5 Workload/Scale | P6 | 形成体系化实验 | M4 后 |
| M6 PPA | P7 | 完整架构评估 | 证据成熟后 |

若资源受限，正确停止点是 P4：它已经是一个可解释的 2D SNN 芯片周期模型。不能跳过 P1-P4 直接做 P7，因为 proxy energy 无法修复错误的 action count 和 timing。

## 29. P0 启动批次

获得本路线图批准后，P0 按以下批次执行：

### Batch A：保护现状

- 只读记录三层版本与 dirty 路径。
- 新目录复跑四个 v4 fixture。
- 生成 normalized golden candidate 和基线审计表。
- 若任一当前基线失败，先定位并把 P0 标记 blocked，不用旧成功目录冒充当前通过。

### Batch B：先定 contract

- 落地 ADR、strict v5 resolver、artifact/address 和 stats/provenance contract。
- 先写 negative tests，再允许有效 minimal v5 spec 通过。
- C++ 只实现纯 contract types，不写周期 Component。

### Batch C：清旧边界

- 归档旧 DMA 簇。
- 重建 source manifest、canonical aggregate 和 boundary checker。
- 串行 build/install，检查 ELI 与 `DT_NEEDED`。

### Batch D：证据闭环

- 定稿四个 v4 golden。
- 实现 run provenance/evidence self-audit。
- 组合 `verify_snndl_v5_p0.sh`，从 clean build 重复一次。
- 只读 review，修完 blocker 后写 sign-off。

每个 Batch 完成后保存 focused diff 和命令结果。Batch B 与 C 都会改 contract/build 边界，不同时修改 `Makefile.am`；Batch D 不改变前面已经签字的 wire format。

## 30. 外部依据与本地实现来源

以下来源用于选择建模维度和复用组件，不替代本地 SST 15.0.0 ELI 与源码检查。链接已于 2026-08-04 访问。

1. SST memHierarchy overview：<https://sst-simulator.org/sst-docs/docs/elements/memHierarchy/intro>
2. SST `StandardMem` interface：<https://sst-simulator.org/sst-docs/docs/core/iface/StandardMem/class>
3. SST Merlin overview：<https://sst-simulator.org/sst-docs/docs/elements/merlin/intro>
4. SST `SimpleNetwork` interface：<https://sst-simulator.org/sst-docs/docs/core/iface/SimpleNetwork/class>
5. 本地 SST 15.0.0 `memHierarchy.Scratchpad`：`src/sst/elements/memHierarchy/scratchpad.{h,cc}`；其 backend 与 Get/Put 路径是 P2/P3 实现依据。
6. 本地 SST 15.0.0 `merlin.hr_router`：`src/sst/elements/merlin/hr_router/`；`sst-info` 暴露的 link/flit/xbar/buffer 参数和 stall stats 是 P4 权威。
7. Intel, *Taking Neuromorphic Computing to the Next Level with Loihi 2*：<https://download.intel.com/newsroom/2021/new-technologies/neuromorphic-computing-loihi-2-brief.pdf>
8. Höppner et al., *The SpiNNaker2 Processing Element Architecture for Hybrid Digital Neuromorphic Computing*, arXiv:2401.04491：<https://arxiv.org/abs/2401.04491>
9. Pedersen et al., *Neuromorphic Intermediate Representation*, Nature Communications 2024，DOI：<https://doi.org/10.1038/s41467-024-52259-9>
10. Yik et al., *NeuroBench*, Nature Communications 2025，DOI：<https://doi.org/10.1038/s41467-025-56739-4>
11. Accelergy：<https://github.com/Accelergy-Project/accelergy>
12. Ramulator2：<https://github.com/CMU-SAFARI/ramulator2>
13. DRAMPower：<https://github.com/tukl-msd/DRAMPower>

引用这些系统只支持以下有限结论：本地 neural memory、DMA、packet NoC、multicast、可编程 neuron dynamics 和分层 benchmark 是合理建模维度。它们不证明 SnnDL 的具体宽度、容量、频率、功耗或面积数值。

## 31. 最终执行边界

本路线图批准后，执行顺序固定为先完成 P0，再按依赖进入 P1-P7，不得跨阶段并行实现。P0 的第一批动作是重新验证 v4 当前基线并冻结 contract；最后动作是人工验收与 sign-off。P0 已签字，后续工作从 P1 开始。

P0.10 sign-off、P1 compute-timed、P2 typed Core SRAM、P3 cache/DRAM/DMA 和
P3.5 Core-memory integration 已完成。当前状态：

```text
roadmap_status = p5_multicast_timed_sync_next
p0_status = accepted
p1_status = accepted
p2a_status = accepted
p2b_status = accepted
p3_status = accepted
p35_status = accepted
p4_status = accepted
v4_status = functional_oracle
v5_runtime_status = merlin_2d_unicast_core_sram_cache_dram_dma_performance_timing
highest_claim_level = T (P1-P4 minimum credible 2D cycle model)
```

P1 sign-off：`docs/reviews/2026-08-05-v5-p1-signoff.md`；P2-A sign-off：
`docs/reviews/2026-08-05-v5-p2-sram-milestone.md`；P2-B sign-off：
`docs/reviews/2026-08-05-v5-p2-core-storage-binding.md`；P3.5 sign-off：
`docs/reviews/2026-08-05-p35-integration-signoff.md`；P4 sign-off：
`docs/reviews/2026-08-05-v5-p4-signoff.md`。P4 已开放 deterministic 2D
unicast/source-replication 的 performance timing 与 claim level T；multicast
和 timed chip-wide synchronization 仍属于 P5。

## 32. P2-A request-driven SRAM milestone

- 新增 `v5/storage/BankedSramV5`：容量边界、有限 request/response queue、低位
  interleave bank mapping、每 bank 多端口、服务后起算的读写 latency、唯一
  byte backing 和 deterministic completion order。
- 新增 `SnnDL.BankedSramV5` SST Component 与 `SnnDL.SramProbeV5`，通过 SST
  link 传递 request/response；queue full 返回 retryable response，容量越界
  fail-closed，不静默丢包。
- 新增 81 项 v5 statistics registry 中的 `storage.sram.*` 原始计数和 JSON
  stats evidence；旧 observe-only `BankedSramModel` 没有进入 v5 runtime。
- 验收命令：
  `SNNDL_V5_P2_ACCEPTANCE_ROOT=snndl_runs/p2_v5_acceptance_20260805_rerun2 bash tools/verify_snndl_v5_p2.sh`
- 真实结果：6 个 SRAM 请求完成、4 次可重试拒绝、1 次 bank conflict、
  16 个 aggregate latency cycles，write/read bytes 各 12；P1 全部场景仍
  通过。

P2-A 尚不允许声明完整 SNN core 的 state/delta/index/route 已使用该存储，
也不允许声明 cache/DRAM/NoC timing；P2-B 已将四类 typed region 接入
`SnnCoreV5`，并保持 P1 functional digest 不变。

## 33. P2-B typed Core storage binding

- 新增 `v5/storage/CoreStorageV5`，为 `CoreState`、`CoreDelta`、`CoreIndex`
  和 `PeRoute` 建立独立 `RegionDescriptor`、owner 和容量边界。每个 region
  只有一个 `BankedSramV5` byte backing，功能访问不再读取 pipeline vector。
- `CorePipeline` 的 state scan、delta append/read/clear、row index touch 和
  PE route touch 均经过 request-driven SRAM；delta count 与 entry bytes
  同样驻留在 `CoreDelta`，没有第二份权重/状态真值。
- P2 参数已开放容量、bank、port、interleave、读写 latency、队列深度和
  delta resident entries；容量不足或 typed address 越界 fail-closed。
- 验收命令：
  `SNNDL_V5_P2_ACCEPTANCE_ROOT=snndl_runs/p2_v5_acceptance_20260805_p2b bash tools/verify_snndl_v5_p2.sh`
- 结果：P0/P1 依赖、34 个 SST timing tests、CoreStorage/流水线 C++ 回归、
  serial build/install、ELI、P2-A SRAM 场景全部通过；最小 Core 证据为
  state reads/writes `192/64`、delta reads/writes `68/132`、index reads `1`、
  route reads `1`。

P2-B 仍不开放 cache/DRAM hierarchy、DMA、Merlin NoC 或 performance class；
下一阶段进入 P3 的每 Core 私有 L1、每 PE 共享 L2 和单芯片 DRAM owner。
