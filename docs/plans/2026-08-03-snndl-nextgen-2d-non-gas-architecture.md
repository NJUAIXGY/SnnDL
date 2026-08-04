# SnnDL 下一代 2D SNN 芯片架构设计与迁移规范（无 GAS）

> 状态：下一代主线已落地；v4 Python 与 C++ SST `MeshPE2D`/`TimestepCoordinator` 入口均可运行，3D research 已隔离为 opt-in 库  
> 日期：2026-08-03  
> 代码范围：`sst_workspace/sst-elements/src/sst/elements/SnnDL/` 及其 spec-first 2D Mesh 运行链路  
> 适用范围：原生 SNN、2D Mesh、BCSR 权重、标准内存、可选通信优化、可选局部存储  
> 明确排除：GAS、3D NoC、3D route、RISC-V/Tensor/Stream/Traffic workload 及历史实验实现

## 1. 决策摘要

下一代 SnnDL 不再把 GAS 视为可开关的执行模式，而是从以下边界中完全移除：

1. 核心 ABI 不出现 `Gas`、`Gather`、`Apply`、`Scatter` 或 window-stage 生命周期。
2. 默认构建目标不编译、不链接 GAS 接口、事件、控制器或内存前端。
3. v4 spec 不接受 `gas`、`exec_mode=gas`、`bcsr_gas` 等字段或取值。
4. canonical runner 不构造 `GlobalGasStepController` 或 `GatherBufferIF`。
5. 当前 GAS 实现作为 legacy 代码归档并保留历史可复现性，但不能进入下一代对象图。

新的执行基线是**确定性的同步 timestep 模型**。每个 timestep 接收带标签的脉冲，完成路由、BCSR 读取和突触增量累加，在全局确认无遗漏输入后统一提交神经元状态；本 timestep 产生的脉冲标记为下一 timestep，并保持到下一 timestep 开始时再进入 NoC。

该选择保留 SNN 所需的突触累加和 timestep 同步，但不保留 GAS 的三阶段窗口、控制命令、窗口预算、内存聚合语义或统计口径。

### 1.1 当前落地入口与边界

本规范已经落地一条可执行的 v4 语义主线，Python 参考模型与 C++ SST 数据面共享同一 spec/descriptor 契约：

- `remote/snndl_system/synchronous_mesh.py` 是独立的同步 2D Mesh 参考执行器，负责 LIF、BCSR、2D Manhattan NoC、内存延迟、held spike、精确 drain 和摘要生成。
- `remote/tools/run_snndl_v4.py` 是 v4 直接入口；`remote/tools/run_snndl_with_time.sh` 在识别 `schema_version=4` 与 `model=snn_mesh_2d` 后只分发到该入口。
- `remote/tools/snndl_spec_cli.py` 和 `remote/sst_dram_si/mesh_template/spec.py` 对 v4 使用严格 resolver；v1-v3 仍明确保留 legacy resolver。
- SnnDL 子模块的 `build/nextgen_sources.am` 生成独立的 `libSnnDLNextCore.la`，只包含 `TimestepTypes`、`TimestepTracker`、`DeltaAccumulator` 和 `NextGenNeuronEngine`，不依赖 SST、GAS、workload 或 research 源码。
- C++ SST v4 入口由 `libSnnDLPlatform2D.la` 提供；`libSnnDL.la` 仅聚合 2D active libraries，不再链接 `libSnnDLResearch.la`。3D ELI 组件只在显式加载 `SnnDLResearch` 时可见。
- `extensions.local_storage.enabled=true` 使用 PE-local BCSR value store；逻辑 memory request/response 计数保持不变，并在摘要中报告 `storage_hits`。

v4 canonical runner 实例化 `SnnDL.MeshPE2D` 与 `SnnDL.TimestepCoordinator`，不实例化 legacy SST `MultiCorePE`、`SnnPESubComponent`、`GlobalGasStepController` 或 `GatherBufferIF`。

## 2. 背景与当前问题

当前主线已经完成若干重要整理：构建库初步分层，BCSR 路由与权重共享数据源契约，内部 ring 实现一跳一拍，NoC 注入拥塞时不再丢包，局部存储由可选服务装配。它们可以作为下一代实现的低层基础。

但是 GAS 已经穿透当前主线的每一层：

- `build/core_sources.am` 把多个 `IGas*` 接口、`GasStepBarrierEvent`、`GasPhaseController` 和 GAS 边收集器列入稳定 Core。
- `ISnnComputeCore` 暴露 `onStageBeginGather`、`onStageBeginApply`、`onStageBeginScatter` 等方法。
- `SnnWorkload` 继承 `IGasStageSink`，并以 GAS 阶段决定何时记录边、读取权重、累加和发放。
- `SnnPESubComponent` 同时实现 GAS 编排、阶段事件接收、内存控制和 workload 转发。
- `GatherBufferIF` 同时承担 StandardMem 前端、读请求聚合、scratchpad、窗口状态机和 timestep 门控。
- `GlobalGasStepController` 混合 timestep barrier、Apply credit、自适应控制和实验 gating。
- Python resolver 即使面对未声明 GAS 的最小 spec，也会默认解析为 `exec_mode=gas` 和 `synapse_weight_mode=bcsr_gas`。
- 摘要、验证器和实验脚本以 `gas.*` 作为正确性与性能口径。

因此，“设置 `gas_enable=0`”只会绕过部分运行时分支，不会形成无 GAS 的架构、API 或构建闭包。

### 2.1 `naive_raw` 不能直接成为新基线

现有 `naive_raw` 解决的只是一个局部问题：输出包带 `step_seq`，目标 PE 将未来 timestep 的包暂存，从而避免部分步内级联。它仍存在以下语义缺口：

1. 非窗口路径每个 SST 时钟周期调用一次 `endCycle()`，把微架构周期误当作神经 timestep。
2. 权重回包到达时立即调用 `onSynapticEvent()`，内存响应顺序可能改变膜电位更新顺序。
3. 非窗口模式的 `hasWork()` 有意忽略 compute core 状态，不能作为严格 timestep 完成条件。
4. `Drain` 策略仍依赖阶段镜像或静默周期，不能证明全网不存在迟到包。
5. 包门控发生在 PE 边界，核心、权重请求、累加器本身没有完整的 timestep 所有权。

下一代实现可以复用 `step_seq` 门控的经验，但必须重新建立端到端 timestep 契约。

## 3. 目标与非目标

### 3.1 目标

- 建立一个可以独立编译、运行和验证的 2D SNN 芯片模型。
- 明确区分神经 timestep、微架构 cycle 和 SST 仿真时间。
- 保证没有步内级联、没有跨 timestep 内存回包污染、没有靠静默超时判断正确性。
- 以 BCSR 作为唯一主线突触格式，并让通信路由与权重内容来自同一个带摘要的图像描述。
- 让 NoC、标准内存、神经计算分别拥有清晰接口。
- 将多播等通信优化和局部存储作为显式扩展接入，关闭扩展时仍可形成完整基线。
- 保持数据面时延、带宽、队列和背压可建模，同时把仿真同步控制与芯片硬件声明分开。
- 保留 legacy GAS 代码和旧实验复现入口，但阻止它们进入新主线。

### 3.2 非目标

- 不支持 3D NoC 或 3D route。
- 不在下一代 SNN ABI 中兼容 Tensor、Stream、Traffic 或 RISC-V workload。
- 不继续维护 GCSS、PULSE 或其他从 GAS 派生的实验路径。
- 不把 `TimestepCoordinator` 宣称为被测芯片中的物理硬件。
- 不以固定周期、静默 N 周期或 wall-clock timeout 代替正确性闭环。
- 不要求第一版实现异步 SNN 或连续时间神经模型。
- 不物理删除旧代码、旧配置、旧结果和历史文档。

## 4. 术语与时间模型

### 4.1 三种时间必须分开

| 名称 | 含义 | 用途 |
|---|---|---|
| `timestep_id` | SNN 离散算法时间，类型为 `uint64_t` | 输入归属、状态提交、输出归属、正确性验证 |
| `cycle` | PE、SRAM、NoC 和内存控制器的微架构周期 | 性能与队列建模 |
| `sim_time` | SST 全局仿真时间 | 组件调度与事件传递 |

一个神经 timestep 可以经历任意数量的微架构周期。内存或 NoC 越慢，完成该 timestep 所需周期越多，但神经状态只能提交一次。

### 4.2 关键术语

- **输入脉冲**：标记为 `k`、在 timestep `k` 被处理的脉冲。
- **输出脉冲**：提交 timestep `k` 时产生，标记为 `k+1`。
- **突触任务**：由一个输入脉冲和一个目标突触生成的稳定任务，归属于唯一 timestep。
- **增量缓冲**：按目标神经元累加本 timestep 的 `delta_v`，不直接修改已提交状态。
- **提交**：对所有本地神经元执行一次动力学推进、增量整合、阈值判断与状态写回。
- **逻辑投递**：一个源脉冲到一个最终目标 PE/core 的语义投递。一个多播物理包可以代表多个逻辑投递。
- **数据面**：Spike、BCSR、内存请求、NoC 包、SRAM 访问和真实队列。
- **同步面**：用于证明 timestep 完整性的控制事件与计数，不计入芯片数据面流量。

## 5. 强制语义不变量

下一代实现必须始终满足以下不变量：

1. 每个输入、突触任务、内存请求、增量和输出都携带唯一 `timestep_id`。
2. timestep `k` 只能读取状态快照 `state[k]`，不能观察尚未提交的 `state[k+1]`。
3. timestep `k` 的突触增量只能写入 `delta[k]`。
4. 每个本地神经元在 timestep `k` 恰好提交一次。
5. timestep `k` 产生的脉冲只能在 `Start(k+1)` 后进入数据 NoC。
6. 内存回包顺序不能改变功能结果；回包只把稳定编号的任务标记为 ready。
7. 任何队列满都必须产生背压，不能删除数据、覆盖数据或隐式降级。
8. 正确性完成条件必须基于精确计数和队列状态，不能基于静默时间猜测。
9. 在 `SealIngress(k)` 后收到 timestep `k` 的新包属于协议错误，必须 fail-fast。
10. 旧 timestep 的包、回包或提交事件必须 fail-fast；不允许静默丢弃。
11. 多播、压缩和局部存储只能改变物理流量或时延，不能改变逻辑投递和神经结果。
12. 一个完成的 run 必须满足所有已开始 timestep 均已提交，除非由显式故障终止。

## 6. 总体架构

```text
Spec v4 / BCSR artifact
          |
          v
  Python Mesh Builder
          |
          +-------------------- TimestepCoordinator
          |                         (simulation semantics)
          v
  +---------------- MeshPE2D ----------------+
  |                                          |
  |  SpikeIngress ---> SynapseEngine         |
  |                       |                  |
  |                       v                  |
  |              BCSR Index/Value Reader     |
  |                       |                  |
  |                       v                  |
  |              DeterministicRetireQueue    |
  |                       |                  |
  |                       v                  |
  |                DeltaAccumulator          |
  |                       |                  |
  |                       v                  |
  |                  NeuronEngine            |
  |                       |                  |
  |                       v                  |
  |                  SpikeEgress             |
  |                                          |
  |  TimestepTracker   Standard Memory       |
  |  Internal Ring     2D NoC Adapter        |
  +------------------------------------------+
          |                         |
          v                         v
   per-PE memory              2D Mesh routers
```

### 6.1 依赖方向

```text
Memory          -> Domain Core
Communication2D -> Domain Core
Platform2D      -> Domain Core + Memory + Communication2D
CommunicationOpt -> Communication2D
LocalStorage     -> Domain Core + Memory
LegacyGas        -> legacy-only dependencies
```

`Domain Core` 不得依赖 SST StandardMem、Merlin、具体 PE、具体内存后端或任何可选扩展。

## 7. Timestep 控制协议

### 7.1 启动

1. 每个 PE 校验 neuron layout、BCSR descriptor、图摘要和本地存储容量。
2. WeightLoader 或初始化路径完成必要的内存写入与 row-index 准备。
3. PE 向 `TimestepCoordinator` 发送 `BootReady`。
4. 所有必需 PE ready 后，协调器广播 `Start(0)` 或 spec 指定的起始 timestep。

启动不得依赖固定 `ready_delay_cycles`。超时只能作为诊断故障，不得强制跳过未完成加载。

### 7.2 正常 timestep

#### A. `Start(k)`

- `TimestepTracker` 打开 timestep `k`。
- `SpikeEgress` 释放此前保存的、标签为 `k` 的输出脉冲。
- 外部 stimulus 注入标签为 `k` 的输入。
- 所有输入生产者开始报告逻辑投递数量。
- `NeuronEngine` 提供只读 `state[k]` 与本 timestep 的 refractory accept mask。

#### B. 数据面执行

- 源侧根据由同一 BCSR artifact 派生的 route directory 选择目标 PE/core。
- NoC 传输带 `timestep_id=k` 的 spike message。
- `SpikeIngress` 验证标签并生成本地 synapse row task。
- `SynapseEngine` 发起 BCSR index/value 读取。
- 内存回包进入稳定编号的 retire entry。
- retire entry 按确定顺序产生 `delta_v`，写入 `DeltaAccumulator[k]`。
- 数据面可以完全流水并行，不需要等待一个全局阶段切换后才发起内存请求。

#### C. 精确封闭输入

每个 PE 在本地所有 timestep `k` 输出均已路由和注入后，发送：

```text
EgressClosed(k, final_logical_tx_count)
```

目标 PE 对已接受到 timestep inbox 的逻辑投递维护累计 `logical_rx_count`。协调器只有在满足以下条件时才能广播 `SealIngress(k)`：

```text
all_producers_closed(k)
and sum(final_logical_tx_count[k]) == sum(logical_rx_count[k])
```

本地投递和多播复制都必须计入逻辑 tx/rx。物理 packet 数单独统计，不能用于正确性封闭。

#### D. 本地 drain

收到 `SealIngress(k)` 后，不会再有合法 timestep `k` 输入。每个 PE 等待以下计数全部归零：

- ingress queue entries；
- route expansion tasks；
- BCSR index reads；
- BCSR value reads；
- memory requests；
- deterministic retire entries；
- accumulator update operations；
- internal ring 与本地投递队列；
- extension-owned、明确注册到 tracker 的 timestep tokens。

归零后发送 `CommitReady(k)`。任何未注册的后台任务不得参与本 timestep 的功能结果。

#### E. 原子提交

协调器收到所有 PE 的 `CommitReady(k)` 后广播 `Commit(k)`。每个 `NeuronEngine` 对本地神经元执行：

```text
old = committed_state[k]

if old.refractory > 0:
    next.refractory = old.refractory - 1
    next.v_mem = refractory_state(old)
    fired = false
else:
    leaked = neuron_model.advance_without_input(old, dt)
    integrated = leaked + delta[k]
    fired, next = neuron_model.threshold_and_reset(integrated)

committed_state[k + 1] = next
```

为了与当前主线的接收门控保持一致，默认语义是在 timestep 开始时 refractory 大于零的神经元拒绝该 timestep 的突触输入。这个判断只读取 `state[k]`，不依赖包到达周期。

发放结果进入 `held_spikes[k+1]`，不得在 `Commit(k)` 内立即进入 NoC。提交完成后 PE 发送 `CommitDone(k)`。

#### F. 推进或结束

- 所有 PE `CommitDone(k)` 后，若达到 `max_timesteps`，协调器结束仿真。
- 否则广播 `Start(k+1)`。

### 7.3 PE 状态机

| 状态 | 允许操作 | 退出条件 |
|---|---|---|
| `Booting` | 加载、校验、初始化 | 本地资源 ready |
| `WaitingStart` | 保存未来 timestep 输出 | 收到 `Start(k)` |
| `Open` | 收包、路由、访存、累加 | 收到 `SealIngress(k)` |
| `Draining` | 完成已有任务，不接受新 `k` 输入 | tracker 全部归零 |
| `CommitReady` | 状态保持不变 | 收到 `Commit(k)` |
| `Committing` | 推进神经状态、产生 `k+1` 输出 | 本地提交完成 |
| `WaitingStart` | 保存 `k+1` 输出 | 协调器推进 |

这些是 timestep 生命周期状态，不是 GAS 阶段，也不控制内存请求聚合策略。

## 8. 核心组件职责

### 8.1 `TimestepCoordinator`

职责：

- 收集 `BootReady`、`EgressClosed`、逻辑 rx 进度、`CommitReady` 和 `CommitDone`。
- 执行精确 tx/rx 封闭与全局提交 barrier。
- 根据 `max_timesteps` 结束仿真。
- 输出同步协议统计和错误诊断。

限制：

- 不携带 Apply credit、memory policy、gating 决策或 workload 自适应控制。
- 不进入数据 NoC 的流量、能耗或带宽统计。
- 第一版是仿真正确性组件，不作为芯片面积或性能贡献进行宣称。

### 8.2 `MeshPE2D`

职责：

- 装配 cores、内部 ring、NIC、memory endpoint 和可选扩展。
- 持有 PE 级 `TimestepTracker` 并聚合 core token。
- 转发同步面事件，但不解释 BCSR、神经动力学或权重内容。
- 汇总 PE 级统计。

`MeshPE2D` 不负责生成测试流量、解析历史 workload、执行负载均衡启发式或承载实验控制器。

### 8.3 `SnnCoreTile`

每个 core 的运行容器，装配：

- `SpikeIngress`；
- `SynapseEngine`；
- `DeltaAccumulator`；
- `NeuronEngine`；
- core-local memory view；
- core-local timestep counters。

它替代当前同时承担通用 workload、GAS 编排和 SNN 数据面的 `SnnPESubComponent`。

### 8.4 `TimestepTracker`

`TimestepTracker` 是纯计数与状态验证模块。建议 token 类型如下：

```cpp
enum class WorkKind : uint8_t {
    Ingress,
    RouteTask,
    IndexRead,
    WeightRead,
    RetireEntry,
    AccumulatorUpdate,
    LocalDelivery,
    Extension
};
```

接口必须成对使用：

```cpp
void acquire(TimestepId step, WorkKind kind, uint64_t count = 1);
void release(TimestepId step, WorkKind kind, uint64_t count = 1);
bool locallyDrained(TimestepId step) const;
DrainSnapshot snapshot(TimestepId step) const;
```

要求：

- 下溢、错 timestep release、未关闭 timestep acquire 均立即报错。
- `DrainSnapshot` 必须能指出具体未归零的 token 类别。
- timeout 只打印 snapshot 或终止仿真，不能把非零 token 强制清零。

### 8.5 `SpikeIngress` 与 `SpikeEgress`

`SpikeIngress`：

- 校验 timestep 标签、目标 core 和 payload 版本。
- 对当前 timestep 入队并获取 ingress token。
- 对下一 timestep 的包仅允许在协议明确支持的 hold queue 中保存。
- 拒绝 stale、过远 future 或 seal 后到达的包。

`SpikeEgress`：

- 保存 `held_spikes[k+1]`。
- 仅在 `Start(k+1)` 后进行 route lookup 和 NoC 注入。
- 在目标列表确定后形成最终逻辑 tx 计数。
- 队列满时向 `NeuronEngine` 或提交路径反压，不得丢包。

### 8.6 `SynapseEngine`

职责：

- 将输入脉冲转换为 BCSR row task。
- 从唯一 descriptor 解析 rowptr、colidx、blockdata 地址。
- 通过 `IWeightStore` 发起 index/value 请求。
- 为每个突触任务分配稳定 retire key。
- 将 ready 结果按确定顺序提交给 `DeltaAccumulator`。
- 暴露精确 pending breakdown。

第一版稳定 retire key 定义为：

```text
(timestep_id, post_local, pre_global, source_event_seq, edge_ordinal)
```

回包到达顺序只改变等待时间，不改变累加顺序。默认采用 per-post deterministic retire，以避免一个慢 post 阻塞所有其他 post，同时保证每个 post 内结果稳定。

### 8.7 `DeltaAccumulator`

突触增量求和属于 SNN 语义，不属于 GAS。新的 `DeltaAccumulator`：

- 只保存一个当前 timestep 的 `delta_v[post]`。
- 使用显式 `float32` 权重和确定顺序累加，避免 MPI/线程调度改变结果。
- 维护 touched bitmap/list，允许统计稀疏活动，但提交语义仍由 neuron model 决定。
- 只在 `Commit(k)` 时提供只读 view。
- `CommitDone(k)` 后才清空 timestep `k` 数据。

旧 `AccumulatorOps` 的“超过高水位后写入 vector spill log”没有真实内存时延，不能直接继承。若容量不足，基线必须背压；启用局部存储扩展时，spill 必须通过真实 `IStateStore`/`IMemoryAccess` 请求建模。

### 8.8 `NeuronEngine`

`NeuronEngine` 只负责：

- 已提交 neuron state；
- timestep 开始时的只读 accept mask；
- `Commit(k)` 时的一次动力学推进；
- threshold、reset、refractory 与输出生成；
- neuron-state SRAM 的可选访问建模。

它不得持有 weight reader、NoC、route table、GAS stage 或 StandardMem 指针。

## 9. 建议接口草案

### 9.1 基础类型

```cpp
using TimestepId = uint64_t;

struct SpikeMessage {
    TimestepId timestep = 0;
    uint32_t source_neuron = 0;
    uint32_t source_node = 0;
    uint16_t source_core = 0;
    uint64_t source_event_seq = 0;
};

struct SynapseContribution {
    TimestepId timestep = 0;
    uint32_t pre_global = 0;
    uint32_t post_local = 0;
    uint32_t multiplicity = 1;
    float weight = 0.0f;
    uint64_t stable_order = 0;
};
```

### 9.2 计算接口

```cpp
class INeuronEngine {
public:
    virtual ~INeuronEngine() = default;
    virtual void beginTimestep(TimestepId step) = 0;
    virtual bool acceptsInput(TimestepId step, uint32_t post_local) const = 0;
    virtual void commitTimestep(TimestepId step, const IDeltaView& deltas) = 0;
    virtual void drainHeldSpikes(TimestepId step, std::vector<SpikeMessage>& out) = 0;
};
```

该接口替代 `ISnnComputeCore` 的 Gather/Apply/Scatter hooks。`onClockTick()` 可以作为微架构资源模型内部接口保留，但不能推进神经 timestep。

### 9.3 突触接口

```cpp
class ISynapseEngine {
public:
    virtual ~ISynapseEngine() = default;
    virtual bool acceptSpike(const SpikeMessage& spike) = 0;
    virtual void tick(uint64_t cycle) = 0;
    virtual bool drained(TimestepId step) const = 0;
    virtual SynapseDrainSnapshot snapshot(TimestepId step) const = 0;
};
```

### 9.4 存储接口

```cpp
class IWeightStore {
public:
    virtual ~IWeightStore() = default;
    virtual RequestId readIndex(TimestepId step, Address addr, size_t bytes,
                                ReadCallback cb) = 0;
    virtual RequestId readValues(TimestepId step, Address addr, size_t bytes,
                                 ReadCallback cb) = 0;
};

class IStateStore {
public:
    virtual ~IStateStore() = default;
    virtual bool readState(uint32_t neuron, NeuronState& out) = 0;
    virtual bool writeState(uint32_t neuron, const NeuronState& state) = 0;
};
```

标准内存、cache、SRAM 和局部存储都通过这些窄接口实现，不让 neuron/synapse 层知道具体后端。

### 9.5 同步事件

```cpp
enum class TimestepControlOp : uint8_t {
    BootReady,
    Start,
    EgressClosed,
    IngressProgress,
    SealIngress,
    CommitReady,
    Commit,
    CommitDone,
    Abort
};
```

`TimestepControlEvent` 只携带 timestep、源 PE、逻辑计数和错误码，不携带 memory policy、credit 或神经数据。

## 10. BCSR 与单一真值

### 10.1 唯一 artifact

下一代 BCSR artifact 必须包含一个机器可校验的 descriptor：

```text
format_version
graph_digest
neuron_layout_digest
total_neurons
mesh_shape
cores_per_pe
neurons_per_core
row_axis = pre_global
column_axis = post_local
block_rows / block_cols
index_bytes / value_bytes
per_core_stride
rowptr / colidx / blockdata offsets
route_directory_digest
```

权重加载、route directory、stimulus reachability 和运行时 BCSR 读取必须引用同一 descriptor 和 `graph_digest`。任何 digest、维度、stride 或 orientation 不一致都在 setup 阶段失败。

### 10.2 路由与权重职责

- 源 PE 使用 route directory 得到目标 PE/core 集合。
- route directory 只包含可达目标，不包含权重值。
- 目标 core 使用本地 BCSR shard 展开 post 和读取权重。
- route directory 必须由同一 BCSR 图像生成，不能由运行时再次独立解析另一个文件。
- `BcsrDataSource` 可继续承担 descriptor 校验和地址计算，但要移除 GAS 命名和窗口入口。

### 10.3 基线格式约束

v4 基线只接受 `format=bcsr` 和 `orientation=pre_to_post`。其他格式必须作为单独扩展定义，不能复用 `synapse_weight_mode` 把执行协议与物理布局混合在一起。

## 11. 内存系统设计

### 11.1 基线

```text
SynapseEngine
  -> IWeightStore
  -> StandardMemAccess
  -> optional cache
  -> per-PE memory controller
  -> SimpleMem or Ramulator2
```

`StandardMemAccess` 继续保持纯地址/字节语义。`StdMemEndpoint` 应缩减为 StandardMem 装配和数据回包分发，不再处理 `CustomReq(GasOpData)`、阶段事件或 GAS stat。

### 11.2 请求合并

`GatherBufferIF` 不进入下一代主线。未来如果需要读请求合并，应新增透明 `ReadCoalescer` 扩展，并满足：

- 不知道 timestep 生命周期；
- 不发送控制命令；
- 不改变单个上游请求的可见数据；
- 可以在任意时刻接收请求；
- 回包和异常保持一一对应；
- 独立统计 payload、physical bytes、overfetch 和 latency。

因此，读合并是 memory microarchitecture optimization，而不是 SNN 执行语义。

### 11.3 容量与背压

- index/value request queue、retire queue、delta buffer 和 output hold queue 都必须有显式容量。
- 队列满时停止上游发起并保持 token，不允许丢弃。
- 所有容量和端口必须进入 effective config。
- 模拟无限容量只能通过显式 `unbounded_for_functional_test=true` 使用，不能作为性能结果。

## 12. 2D NoC 与通信优化

### 12.1 基线 NoC

保留当前 2D Mesh、NIC、router、内部 ring 和背压机制。基线传输只要求普通 `SpikeMessage`：

- router 只解释目标端点和通用网络头；
- spike payload 由 SNN endpoint 解码；
- `timestep_id` 在整个传输过程中保持不变；
- ring 每周期最多一跳；
- source VC 满时保留 pending injection；
- `isIdle()` 必须包含 NIC、ring、pending injection 和本地投递队列。

### 12.2 通信优化扩展

多播、SpikeKey、TileKey、compact mask 和 inter-bundle 可以保留在 `CommunicationOpt`，但必须遵守：

1. 对外仍表现为相同的逻辑 spike deliveries。
2. 每个物理包携带或可推导 `logical_delivery_count`。
3. tx/rx 封闭比较逻辑投递，不比较物理 packet。
4. 解码失败必须报错，不能回退成另一种含糊语义。
5. 扩展开关关闭时，基线库不需要链接其实现。

## 13. 局部存储扩展

局部存储保留为显式扩展，服务对象分为：

- neuron state；
- BCSR index；
- weight values；
- activation ingress；
- delta accumulator；
- small register file / metadata。

接入规则：

- 通过 `IWeightStore`、`IStateStore` 或明确的 queue interface 接入。
- 命中、未命中、spill 和 writeback 都必须消耗真实周期或内存请求。
- 开关局部存储不得改变 spike/state 功能结果。
- 局部存储的后台任务必须注册 `Extension` token，避免提前 commit。
- POD、owner、join 等研究性对象不能进入基线配置；需要时在 LocalStorage 扩展内部单独定义。

## 14. Spec v4 契约

### 14.1 建议最小配置

```json
{
  "schema_version": 4,
  "model": "snn_mesh_2d",
  "platform": {
    "mesh": { "rows": 4, "cols": 4 },
    "pe": { "cores": 4, "neurons_per_core": 4 }
  },
  "execution": {
    "semantics": "synchronous_timestep",
    "start_timestep": 0,
    "max_timesteps": 10,
    "completion": "counted_drain"
  },
  "neuron": {
    "model": "lif",
    "dt_ms": 1.0,
    "tau_mem_ms": 20.0,
    "threshold": 1.0,
    "reset": 0.0,
    "refractory_timesteps": 2
  },
  "synapse": {
    "format": "bcsr",
    "orientation": "pre_to_post",
    "artifact_dir": "weights/bcsr",
    "retire_order": "per_post_deterministic"
  },
  "noc": {
    "type": "merlin_mesh_2d",
    "link_bw": "40GiB/s",
    "num_vns": 2
  },
  "memory": {
    "scope": "per_pe",
    "backend": { "type": "simple", "access_time": "100ns" }
  },
  "extensions": {
    "communication": { "multicast": false },
    "local_storage": { "enabled": false }
  }
}
```

### 14.2 配置规则

- `execution.semantics` 第一版只接受 `synchronous_timestep`。
- `execution.completion` 第一版只接受 `counted_drain`。
- canonical run 必须设置正整数 `max_timesteps`；仿真时间只允许作为 watchdog。
- `synapse.format` 使用 `bcsr`，不再把格式命名为 `bcsr_gas`。
- v4 对 `gas`、`gas_*`、`exec_mode`、`window_read_*`、`apply_acc_*`、`global_gas_*` 直接报 unknown/deprecated error。
- 环境变量不能覆盖结构、神经元数量、BCSR orientation 或执行语义。
- `effective_config.json` 必须忠实记录 descriptor digest、所有容量、时钟、内存和扩展开关。

### 14.3 兼容策略

- v3 spec 只交给 legacy runner。
- v4 resolver 不做静默字段翻译。
- 可以提供独立的离线迁移工具，把可映射的 v3 参数转换为 v4，并输出无法迁移字段清单。
- canonical `tools/snndl_spec_cli.py` 默认验证 v4；legacy 必须显式选择。

## 15. 统计与验证契约

### 15.1 统计命名空间

| 命名空间 | 代表字段 |
|---|---|
| `timestep.*` | started、sealed、committed、cycles、barrier_wait |
| `transport.*` | logical_tx、logical_rx、held_spikes、stale/future errors |
| `noc.*` | physical packets、bytes、hops、latency、backpressure |
| `synapse.*` | spikes accepted、rows expanded、tasks created/retired |
| `memory.*` | requests、responses、payload bytes、physical bytes、latency |
| `accumulator.*` | updates、posts touched、capacity stalls |
| `neuron.*` | commits、neurons evaluated、fired、refractory rejects |
| `storage.*` | SRAM hits/misses、bank conflicts、spill/writeback |

新摘要不得生成 `gas.*` 字段。legacy 摘要保留旧格式，但不能与 v4 聚合混用。

### 15.2 每 timestep 必查不变量

```text
sum(transport.logical_tx[k]) == sum(transport.logical_rx[k])
synapse.tasks_created[k] == synapse.tasks_retired[k]
memory.requests_issued[k] == memory.responses_completed[k]
tracker.outstanding_tokens[k] == 0 before CommitReady(k)
neuron.commit_calls[k] == number_of_participating_cores
held_spikes_created[k+1] == held_spikes_released[k+1]
```

完成 run 还必须满足：

```text
timestep.started_total == timestep.committed_total
no stale packet
no post-seal packet
no token underflow
no queue drop
no BCSR digest mismatch
```

### 15.3 性能口径

- `timestep.cycles` 从 `Start(k)` 到最后一个 `CommitDone(k)`。
- 同步面事件时延单独记录，不归入 NoC bytes 或芯片能耗。
- payload bytes 与 physical bytes 分开。
- 多播结果同时报告 logical deliveries 与 physical packets。
- cache/SRAM 优化必须同时报告功能等价结果和资源命中情况。

## 16. 构建边界与当前落地状态

下一代目标与当前生成目标必须明确区分：

| Library/入口 | 状态 | 内容 | 允许依赖 |
|---|---|---|---|
| `libSnnDLNextCore.la` | 已实现 | `TimestepTypes`、`TimestepTracker`、`DeltaAccumulator`、`NextGenNeuronEngine` | 无具体平台、SST 或可选域依赖 |
| `libSnnDLCore.la` / `libSnnDLComm.la` / `libSnnDLLocal.la` / `libSnnDLRegistry.la` | 已落地 | 当前 2D core、通信、内存/局部存储和 BCSR source contract | 以各自 manifest 为准 |
| `libSnnDLResearch.la` | 已落地 | 显式 3D communication research extensions | Core、Comm、Registry |
| `libSnnDLLegacyGas.la` | 不构建 | 原始对象图仅作为 `archive/legacy_gas/` 文件记录 | legacy only |
| `libSnnDL.la` | 当前兼容入口 | 活动域库的 aggregate anchor；不是 v4 canonical runner | 活动域库 |

强制规则：

- `libSnnDLNextCore.la` 不得包含或链接任何 GAS 源码；其边界由 `tools/check_build_boundaries.py` 检查。
- `libSnnDL.la` 不得链接归档目录或不存在的 legacy target；当前 v4 仍通过 Python 入口作为 canonical runner。
- `core_sources.am` 和新平台 source manifest 中 GAS 关键字为零。
- `CommOpt`、`Local` 关闭时不进入链接闭包。
- 非 SNN workload 和 3D research library 不被 canonical plugin 引用。
- legacy 组件名与新组件名分开，避免同一个 `MultiCorePE` 根据参数暗中切换两套语义。

## 17. 旧组件迁移映射

| 当前组件/接口 | 下一代处理 |
|---|---|
| `GasOps.h` | 无替代，移出主线 |
| `IGasCmdSender` | 无替代，StandardMem 仅保留数据请求 |
| `IGasCreditGate` | 无替代，memory policy 归 memory config |
| `IGasOrchestrator` | 无替代 |
| `IGasStageSink` | 无替代，统计改为领域统计 |
| `IGasStepGate` | 由 timestep participant 协议重新实现，不做一对一兼容 |
| `GasStepBarrierEvent` | 重写为 `TimestepControlEvent` |
| `GlobalGasStepController` | 重写为纯 `TimestepCoordinator` |
| `GatherBufferIF` | 从主线移除；未来可独立实现透明 `ReadCoalescer` |
| `GasPhaseController` | 无替代 |
| `GasCustomCmd` | 无替代 |
| `GasEdgeCollector` | 重写为 `SynapseTaskQueue`/稳定 retire queue |
| `AccumulatorOps` | 重写为 `DeltaAccumulator` |
| `WeightMemorySubsystem` | 拆分为 BCSR readers、`SynapseEngine`、retire queue |
| `SnnWorkload` | 拆分为 timestep pipeline 中的明确组件 |
| `SnnPESubComponent` | 新建较小的 `SnnCoreTile`，旧类归 legacy |
| `MultiCorePE` | 新建 `MeshPE2D`；旧类归 legacy |
| `step_seq` | 改为强类型 `TimestepId` |
| `bcsr_gas` | 改为纯格式名 `bcsr` |

## 18. 文件级迁移计划

采用**平行建立新闭包，再切换 canonical runner**的方式。不要在旧 GAS 对象图上连续重命名，否则容易把旧隐含语义带入新 ABI。

### Phase 0：冻结边界与参考结果

动作：

- 保存当前 v3 minimal spec、effective config、spike/state 摘要和构建边界结果。
- 标记结果为 legacy reference，不把 GAS 输出当作新语义 golden result。
- 新增本设计文档，冻结 v4 决策。

退出条件：当前工作树状态和参考命令可追溯；不修改已有用户改动。

### Phase 1：建立无 GAS Domain Core

新增建议：

```text
api/TimestepTypes.h
api/ITimestepParticipant.h
events/TimestepControlEvent.h
snn/timestep/TimestepTracker.h
snn/timestep/TimestepTracker.cc
snn/compute/INeuronEngine.h
snn/compute/DeltaAccumulator.h
snn/compute/DeltaAccumulator.cc
```

修改：

- `build/core_sources.am`
- `build/extension_sources.am`
- `Makefile.am`
- `tools/check_build_boundaries.py`

退出条件：新 Core target 可独立编译；manifest 中无 GAS 文件；tracker 和 accumulator 单测通过。

### Phase 2：重建计算提交语义

动作：

- 从 `SnnComputeCore` 提取 neuron-state 与 model 逻辑。
- 实现 `beginTimestep`、immutable accept mask、`commitTimestep` 和 held outputs。
- 删除新接口中的 stage hooks、weight reader 和 window fired state。
- 明确 float32 累加与 commit 顺序。

建议测试：

```text
tests/test_timestep_neuron_commit.cc
tests/test_refractory_snapshot_semantics.cc
tests/test_delta_accumulator_determinism.cc
```

退出条件：相同贡献以不同回调顺序到达时，state/spikes 完全一致。

### Phase 3：重建 BCSR SynapseEngine

新增建议：

```text
snn/synapse/engine/SynapseEngine.h
snn/synapse/engine/SynapseEngine.cc
snn/synapse/engine/SynapseTaskQueue.h
snn/synapse/engine/DeterministicRetireQueue.h
snn/synapse/bcsr/BcsrIndexReader.h
snn/synapse/bcsr/BcsrValueReader.h
snn/synapse/bcsr/SynapseImageDescriptor.h
```

复用：

- `snn/synapse/common/BcsrDataSource.h`
- `platform/memory/StandardMemAccess.*`
- 可验证的 BCSR cache/address helpers。

退出条件：1PE 小图能够从模拟内存完成 BCSR 读取、delta 累加和提交，不包含 GatherBuffer 或 GAS custom request。

### Phase 4：建立 `SnnCoreTile` 与 `MeshPE2D`（已完成）

新增建议：

```text
platform/core/SnnCoreTile.h
platform/core/SnnCoreTile.cc
components/MeshPE2D.h
components/MeshPE2D.cc
components/timestep/TimestepCoordinator.h
components/timestep/TimestepCoordinator.cc
snn/transport/SpikeIngress.h
snn/transport/SpikeEgress.h
```

动作：

- 装配 baseline core、memory 和 NoC。
- 将 `TimestepTracker` 与每个异步回调绑定。
- 实现 tx/rx 逻辑计数和 seal/commit 协议。
- 复用已经修正的 ring 与 injection backpressure。

当前状态：同步语义由 Python `synchronous_mesh.py` 和 C++ SST `MeshPE2D`/`SnnCoreTile` 双路径实现并由 v4 runner 对摘要校验；legacy `MultiCorePE` 不在 canonical 对象图中。

### Phase 5：建立 spec-first v4 路径

修改范围：

```text
sst_dram_si/mesh_template/spec.py
sst_dram_si/mesh_template/runtime.py
sst_dram_si/mesh_template/build.py
sst_dram_si/mesh_template/stats.py
snndl_spec/
snndl_system/
tools/snndl_spec_cli.py
tools/run_snndl_with_time.sh
tools/specs/mesh_minimal_v4.json
```

已落地动作：

- 增加严格 v4 schema。
- `resolve_v4_spec()` 构造严格配置；`run_resolved_v4()` 执行同步 timestep 模型。
- v4 canonical path 不调用 `apply_gas_env_overrides()`，也不构造 legacy `gas_cfg`。
- v4 生成 `effective_config.json`、`essential_summary_mesh.json` 和 `validation.log`。

退出条件：已满足 v4 spec-only 构造、运行、总结和 GAS 字段拒绝；2x2、4x4/16-PE 和 Ramulator2 SST fixture 均由专用 gate 覆盖。

### Phase 6：接入保留扩展

顺序：

1. baseline unicast；
2. multicast/compact communication；
3. neuron-state local storage；
4. BCSR index/value local storage（已实现为 PE-local value store）；
5. accumulator capacity model。

每个扩展都先做功能等价，再评估性能，不允许一次同时接入多个扩展。

退出条件：扩展开关前后的 per-timestep logical deliveries、state digest 和 spike digest 一致。

### Phase 7：切换 C++ 默认入口并归档 legacy（已完成）

动作：

- 将 `libSnnDL.la` 从当前兼容注册面切换到 C++ `MeshPE2D` 主线，并使其与 `libSnnDLLegacyGas.la` 彻底断链。
- 旧 GAS 源码迁入 `archive/legacy_gas/` 或独立 legacy target。
- v3 runner/spec 迁入明确的 legacy 入口。
- 更新当前 README、层次文档和构建说明。
- 对 canonical source、build manifests、spec、runner、summary 执行 GAS 零引用检查。

当前状态：`SnnDL.MeshPE2D`/`SnnDL.TimestepCoordinator` 已是 v4 C++ SST 入口；GAS 源码保留在 archive，3D research 使用独立 `SnnDLResearch` ELI/library namespace。

## 19. 验证矩阵

### 19.1 单元测试

1. tracker acquire/release、错 timestep、下溢和 snapshot。
2. delta accumulator 确定顺序与容量反压。
3. neuron commit 恰好一次。
4. refractory snapshot 语义。
5. BCSR descriptor digest、维度、offset 和 orientation 校验。
6. deterministic retire 在不同回包排列下结果一致。
7. held output 只在下一 timestep 释放。
8. stale、future、post-seal packet fail-fast。

### 19.2 小型功能网络

| 网络 | 目的 |
|---|---|
| 1 neuron，无输入 | 验证纯 leak/refractory 推进 |
| 1 pre -> 1 post | 验证一次权重读取和一次 delta |
| 多 pre -> 1 post | 验证稳定累加顺序 |
| 1 pre -> 多 post | 验证 BCSR 展开与 fanout |
| A -> B -> A recurrent | 验证输出只能影响下一 timestep |
| 跨 PE 单边 | 验证 NoC 标签与 tx/rx seal |
| 跨 PE 多播 | 验证物理包压缩但逻辑结果一致 |

### 19.3 扰动测试

- 随机排列内存回包。
- 改变 SST thread/rank 数量。
- 降低 VC、ring、memory 和 hold queue 容量。
- 增大内存延迟和 NoC 拥塞。
- 在每个合法队列边界注入背压。

所有扰动允许改变 cycle、latency 和 queue peak，但不得改变 state/spike digest。

### 19.4 目标验证命令

以下命令是当前 v4 主线和独立 C++ core 的统一验证入口。使用串行构建以避免 Autotools 并行重检查竞争：

```bash
cd remote
REMOTE_ROOT=$(pwd -P)
python3 tools/snndl_spec_cli.py validate tools/specs/mesh_minimal_v4.json
python3 -m unittest snndl_system.test_synchronous_mesh

cd "$REMOTE_ROOT/sst_workspace/sst-elements/src/sst/elements/SnnDL"
make -j1
make check-boundaries
make test-timestep-core
make test-bcsr-source-contract

cd "$REMOTE_ROOT"
bash tools/run_snndl_with_time.sh --spec tools/specs/mesh_minimal_v4.json
```

验证报告必须记录 spec、effective config、state digest、spike digest 和 invariant summary。完整 legacy `sst_dram_si` 测试集中的旧导入失败（Fake SST 接口和历史绝对路径）单独记录，不得伪装成 v4 失败。

## 20. 风险与处理

### 20.1 分布式提前封闭

风险：某 PE 已认为空闲，但远端包仍在 NoC 中。

处理：所有生产者先报告最终 logical tx；协调器只在全局 tx==rx 后 seal。禁止以本地 `isIdle()` 或静默周期替代。

### 20.2 浮点非确定性

风险：不同内存回包顺序改变 float 加法顺序。

处理：回包只标记 retire entry ready；按稳定 key、per-post 顺序累加。必要时输出 contribution digest 诊断。

### 20.3 跨 timestep 回包

风险：旧请求在新 timestep 写入 accumulator。

处理：request metadata 和回调都带 `TimestepId`；旧回包直接协议错误。协调器在 memory outstanding 为零前不会 commit。

### 20.4 输出队列死锁或溢出

风险：提交生成的下一步输出超过 hold queue。

处理：容量不足时 commit 保持未完成并反压；不能部分提交 neuron state。第一版可按最坏每 neuron 一个输出配置容量，并显式报告。

### 20.5 BCSR orientation 与现有数据不一致

风险：旧 artifact 使用不同 row/column 语义。

处理：离线转换为 v4 `pre_to_post`；descriptor 不匹配时禁止运行，不在 runtime 中猜测或转置。

### 20.6 扩展破坏计数闭环

风险：一个多播包代表多个目标，物理 packet 计数与逻辑投递不相等。

处理：codec 和 route directory 显式给出 logical destination count；接收端按最终 endpoint 计数。

### 20.7 同步控制污染性能结果

风险：把协调器事件时延当作芯片 NoC 时延。

处理：同步面使用独立端口和统计；数据面性能与同步等待分别报告。只有未来显式建立物理 timestep controller 时，才把对应开销纳入硬件结果。

## 21. 完成定义

### 21.1 当前 v4 主线完成项

以下条件已经由当前实现和验收命令覆盖，足以称为“GAS 已从 v4 下一代语义主线排除”：

- `build/nextgen_sources.am` 及其 `libSnnDLNextCore.la` 目标不含 GAS 源文件；未来 Core、Memory、Comm2D、Platform2D manifests 仍需沿用同一边界。
- `libSnnDLNextCore.la` 的 manifest、public API 和测试不暴露 GAS 类型或参数。
- canonical v4 spec、resolver、builder、runner、summary 和 validator 无 GAS 路径。
- v4 runner 不加载 `libSnnDL.la`，并在 2x2 多 timestep BCSR SNN 仿真中完成精确 tx/rx、内存请求/回包和状态提交。
- recurrent 小图证明不存在步内级联。
- 反向内存回包和局部存储开关不改变 state/spike digest；多播只改变 physical packet 统计。
- 通信优化和局部存储关闭时，基线仍是完整可运行系统。
- v4 summary 对 logical tx/rx、memory request/response、token/drain、stale/post-seal 和 descriptor digest 提供显式 invariant 字段。

### 21.2 后续增强项（不影响 v4 基线）

- 进一步把 BCSR index metadata 从启动期 descriptor 扩展为可计时的独立 SRAM/cache 请求。
- 将 local-storage 容量、bank conflict 和 eviction 建模接入现有 PE-local value store。
- 为独立 3D research 库维护显式加载与回归脚本，不把它并入 canonical 2D gate。

## 22. 最终架构判断

下一代 SnnDL 的核心不再是“一个通用 PE 加若干 GAS 开关”，而是一个具有严格 timestep 契约的 2D SNN 数据面：

```text
带 timestep 标签的 Spike
-> 2D 通信
-> BCSR SynapseEngine
-> 模拟内存/局部存储
-> 确定性 delta 累加
-> 原子 neuron commit
-> 下一 timestep 的 held Spike
```

通信优化与局部存储围绕这条闭环扩展；GAS、历史 workload 和 3D research 不再决定核心 ABI、构建闭包或默认运行语义。这是后续代码重构、配置整理和实验建模必须共同遵守的架构边界。
