<!--
Phase3 plan (design-only): keep behavior stable, focus on compile-time boundary hardening.
This document is intentionally actionable and regression-driven.
-->

# Phase3：边界硬化收敛（Control / Compute / NoC / Memory / Synapse）

> 目标（主人终局口径）：`MultiCorePE` 变“纯控制壳 + 装配壳”，所有 SNN 事务下沉到子系统；  
> `Memory/NoC` **类型层与语义层都不出现** 权重/突触/BCSR/route；  
> `Compute` 可无缝替换（只关心动力学/发放/学习，可选），不被权重缓存/BCSR/NoC/StandardMem 绑架；  
> `GAS/BCSR/route` 只作为 `services/synapse/**` 的子功能出现。

本 Phase3 只做“结构与接口边界收敛”，不改变 Python 脚本用法与关键统计口径；每个子阶段必须通过 mesh 100us 回归。

---

## 0. 当前状态（Phase2-B 之后）

### 0.1 已达成

- **NoC 类型层通用化已成立**：`services/noc/**` 不依赖 `SpikeEvent`，统一处理 `events/NocPacketEvent*` / `NocPacketBatchEvent*`。
- **Memory 基本去语义化已成立**：主路径为 `api/IMemoryAccess.h` + `services/memory/StandardMemAccess*`（addr → bytes）。
- **Synapse 语义集中已成立**：权重/BCSR/route/GAS 事务集中在 `services/synapse/**`，并通过 `IMemoryAccess` 绑定内存后端。

### 0.2 仍未达标（下一步必须收敛的“硬缺口”）

1) **Compute 接口仍携带权重/缓存语义**（可插拔性被接口绑架）  
   - `compute/ISnnComputeCore.h` 中存在 `requestWeightBCSR/weightCacheTryGet/resolveWeightKey...` 一类接口。
2) **Control 头文件仍强耦合实现细节**（边界不够“硬”）  
   - `control/SnnPESubComponent.h` 直接 include 多个 `services/synapse/**` 实现头以及 `stdMem.h`，导致编译依赖扩散与模块复用困难。
3) **MultiCorePE 仍承担过多 glue**  
   - 当前 MultiCorePE 仍做一部分 NoC packet 编解码/分发 glue（虽已正确，但应进一步“壳化”，将事务性 glue 下沉为可替换子系统/适配器）。

---

## 1. Phase3 的“边界契约”（编译期硬约束）

> Phase3 的关键不是“把代码搬家”，而是把边界变成**编译期约束**：  
> 任何违反边界的 include/类型引用，都应在编译期显性暴露，而不是运行时才发现耦合。

### 1.1 Memory 域契约（绝对纯）

- 允许：`address/size/bytes/callback`，以及“对齐读、切片读、pending、回包分发”等纯传输元信息。
- 禁止（命名与类型层面都禁止）：`weight/synapse/bcsr/route/spike/neuron` 相关字段、结构体、类型。
- 目标接口：`api/IMemoryAccess.h`（作为跨域唯一入口）。

### 1.2 NoC 域契约（payload-agnostic）

- 允许：`send/recv/forward/local-deliver`，payload 统一为 `SST::Event*` 或 `NocPacketEvent*`（NoC 自身不解析 payload）。
- 禁止：`SpikeEvent`（类型层禁用）、fanout/route 选择（语义层禁用）、权重/BCSR（语义层禁用）。
- 目标接口：`api/INocTransport.h`（packet 生命周期语义固定：caller transfer ownership）。

### 1.3 Compute 域契约（可替换）

- 允许：动力学/发放/学习（可选）；输入为 `SpikeEvent`（若需要）或 `SynapticEvent/ΔV`；输出为 `FireEvent`（drain）。
- 禁止：`StandardMem`、`IMemoryAccess` 的具体实现、NoC/LinkControl、BCSR/weight cache 等事务细节。
- 目标：Compute 只通过 `ComputeCoreContext` 注入窄依赖（如 `IWeightReader*` / `writeback_fn`），不通过接口方法暴露“内存语义能力”。

### 1.4 Synapse 域契约（唯一允许出现突触语义）

- 允许：weights/BCSR/meta、fanout/route、GAS 辅助结构（edge collector、accumulator、custom cmd）。
- 允许依赖：`IMemoryAccess`、`INocTransport`（或 `ISpikeTransport`）的抽象接口。
- 禁止：直接触碰 `StandardMem`、直接触碰 NIC/LinkControl 细节。

### 1.5 Control 域契约（纯编排壳）

- 允许：SST 生命周期对接、窗口/GAS 时序、预算控制、统计聚合、诊断开关。
- 禁止：BCSR/meta 解析、权重地址计算、route 构建细节、NoC 发送/转发细节、pending map/cache 容器。
- 关键实现策略：Control 头文件只 include **接口头**；实现 include 下沉到 `.cc`（PImpl 或最小 include）。

---

## 2. Phase3 子阶段拆解（每步都跑 100us）

> 所有子阶段统一验收命令（脚本保持不变）：
>
> ```bash
> cd "sst_workspace/sst-elements/src/sst/elements/SnnDL" && make -j4 && make install
> cd "sst_dram_si" && export MESH_SIM_TIME="100us" && ./tools/run_mesh_with_time.sh
> ```
>
> 对比字段：`essential_summary_mesh.json` 的 `neurons_fired_total / gas_scatter_spikes_emitted_total / step_activation.invocations / nic.packets_sent` 必须量级合理且不出现 0。

### Phase3-A：Compute 接口去权重语义（可插拔的第一刀）

**目标**：让“替换 compute core”不需要实现权重缓存/BCSR/地址映射等事务。

**推荐做法（最安全、最增量）**

1. 保留现有 `compute/ISnnComputeCore` 作为“兼容层”，但新增一个“纯 compute”窄接口（建议命名）：
   - `compute/IComputeKernel.h`（或 `api/IComputeKernel.h`），仅保留：
     - `configure/onInit/onSetup/onFinish`
     - `onStageBegin*/onStageEnd*`
     - `onSpikeDelivered`（若需要；也可后续改为 `onSynapticEvent`）
     - `shouldAcceptSynapticInput`
     - `applySynapticDelta` / `onSynapticEvent`
     - `endCycle/endCycleCandidates`
     - `drainOutputs/getStatistics/read/writeNeuronState`
2. 将以下“内存语义”方法从 compute 的主接口中拆出到可选扩展接口（例如 `compute/IWeightAccessExtensions.h`）：
   - `requestWeight/requestWeightBCSR/weightCacheTryGet/weightCacheStore/resolveWeightKey`
3. Control 侧与 Synapse 侧只依赖 `IComputeKernel`；默认实现通过适配器包装现有 `ISnnComputeCore`（短期兼容）。

**验收**：
- 新增一个最小的“纯 compute core”示例实现（空实现也可），能被装配并跑过 100us（只要不启用其不支持能力即可）。
- `compute/` 目录不再因为 Control/Synapse 的改动而被迫引入更多事务 include。

### Phase3-B：Control 头文件瘦身（边界从软约定变硬约束）

**目标**：`control/SnnPESubComponent.h` 只暴露接口与前置声明；synapse/memory/noc 具体实现 include 全部下沉到 `.cc`。

**推荐做法**

1. 在 `control/SnnPESubComponent.h` 中：
   - 用前置声明替代 `synapse/**` 与 `stdMem.h` 的 include；
   - 将重型成员移入 `struct Impl; std::unique_ptr<Impl> impl_;`（PImpl）。
2. 在 `control/SnnPESubComponent.cc`（或拆分 `SnnPESubComponent_impl.cc`）中：
   - include 所有实现头；
   - `Impl` 内持有：`std::unique_ptr<ISynapseSubsystem>`、`std::unique_ptr<IMemoryAccess>`、`std::unique_ptr<ISpikeTransport>`、`std::unique_ptr<IComputeKernel>` 等。

**验收**：
- Control 头不再 include `stdMem.h`；
- 修改 synapse 实现文件不会导致大量无关目标重编；
- 100us 回归完全一致（统计口径不变）。

### Phase3-C：MultiCorePE 进一步壳化（把 glue 也下沉）

**目标**：MultiCorePE 只做“装配/调度/统计汇聚”，不承载 packet 分发策略与 decode 细节。

**推荐做法**

1. 引入 `services/noc/NocEndpointDispatcher`（或等价命名）：
   - 输入：来自 NIC 的 `SST::Event*`
   - 输出：按 `dst_endpoint` 分发 `NocPacketEvent*`（batch 解包也在此完成，但不解析 payload）
2. 引入 `services/synapse/route/SpikeIngress`（或在现有 `SpikeCommSubsystem` 中新增入口）：
   - 输入：`NocPacketEvent`（kind=Spike）
   - 动作：调用 `SpikeNocCodec::decode`，并将 `SpikeEvent*` 投递到 `SnnPESubComponent`（或 core API）。
3. MultiCorePE 只负责把“NIC 收包回调”接到 dispatcher，把“decoded spike 投递接口”接到 control/core。

**验收**：
- MultiCorePE 中不再出现 `SpikeNocCodec` 的直接调用（编解码下沉到 synapse 域）。
- 100us 回归一致。

---

## 3. 风险与防回归策略（避免再进入“怪圈”）

1) **只允许一次只切一个边界**：Phase3-A/B/C 任何一个子阶段都不混做“搬家 + 行为变更”。  
2) **对 determinism 的硬门槛**：每个子阶段至少连续跑 2 次 100us，关键字段一致；不一致就回滚缩小改动面。  
3) **禁止掩盖问题的兜底**：例如 `use_event_weight_fallback` 必须保持关闭；问题必须靠证据定位、fail-fast 暴露。  
4) **接口能力协商**：compute core 若不支持某能力（例如 window scatter），必须显式声明并在 control 中 fail-fast，而不是 silent fallback。

---

## 4. 交付物（Phase3 完成时应满足）

- `compute/` 有一个“纯 compute”窄接口（或等价结构），且控制层与 synapse 侧主链路仅依赖它。
- `control/SnnPESubComponent.h` 不再 include `stdMem.h` 与 synapse 实现头（编译边界硬化）。
- `MultiCorePE` 变成纯装配/调度壳（编解码与分发细节下沉）。
- mesh 100us 回归通过并稳定；`essential_summary_mesh.json` 关键字段量级合理且不出现 0。

