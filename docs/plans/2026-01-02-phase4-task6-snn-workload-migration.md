# Phase4 Task6.x：SNN 主链路彻底迁入 `workload=snn`（CoreShell 纯壳化）实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 `control/SnnPESubComponent` 收敛为“通用 CoreShell（装配 + 时钟/事件转发 + 统计汇聚）”，把现有 SNN 主链路（compute/weights/route/GAS）完整下沉到 `services/workload/snn/SnnWorkload`，保持 `sst_dram_si/test_mesh_4x4.py` 模板回归口径不变，并持续保证确定性（同 seed 重跑一致）。

**Architecture:** 以“过渡桥接最小化”为原则：每一小步只搬迁一个职责块，并保留一个可回退的 legacy host 适配层；在功能稳定后，再逐步收缩/删除 legacy host 接口，最终让 `SnnWorkload` 形成自洽闭环（只依赖 `api/*` + `services/*` + `events/*`）。

**Tech Stack:** C++17 (SST Elements `SnnDL`), memHierarchy `StandardMem`, Merlin NoC, mesh 模板脚本 `sst_dram_si/tools/run_mesh_with_time.sh`.

---

## 0) 基线与统一验收（必须满足）

### 0.1 当前稳定回归口径（参考）
- 10us：`sst_dram_si/outputs_large/paper2/dram_mesh_4x4/20260102-030930`
- 100us：`sst_dram_si/outputs_large/paper2/dram_mesh_4x4/20260102-031036`

### 0.2 每一步都必须执行的验证
1) 构建安装（改 C++ 必须 install 才生效）：
   - `cd "sst_workspace/sst-elements/src/sst/elements/SnnDL" && make -j4 && make install`
2) mesh 回归（模板方式）：
   - `cd "sst_dram_si" && export "MESH_SIM_TIME=10us" && ./tools/run_mesh_with_time.sh`
   - `cd "sst_dram_si" && export "MESH_SIM_TIME=100us" && ./tools/run_mesh_with_time.sh`
3) 关键字段检查（`essential_summary_mesh.json`）：
   - `spike_activity.neurons_fired_total`：禁止为 0
   - `gas.gather_ns_p95/apply_ns_p95/scatter_ns_p95`：禁止三者同时为 0
   - `step_activation.invocations/spikes_injected_total`：量级合理且稳定
4) 确定性（建议每个里程碑至少做 2 次）：
   - 同配置同 seed 重跑两次关键字段完全一致

---

## 1) 当前进度（起点状态）

- Task6.1 已完成：compute per-tick 驱动下沉到 `SnnWorkload::onClockTick()`，CoreShell tick 统一走 `workload_->onClockTick(total_cycles_)`。
- Task6.2 Step1 已完成：`WeightMemorySubsystem::onClockTick()` 调用点迁入 `SnnWorkload::onClockTick()`（不再由 CoreShell 驱动）。
- Task6.2 Step2 已完成：weights 子系统 ownership 迁入 `workload=snn`（`SnnWorkload` 接管 `IWeightReader/WeightMemorySubsystem`，并在 runtime bind 时绑定 `IMemoryAccess`）。
  - 10us：`sst_dram_si/outputs_large/paper2/dram_mesh_4x4/20260102-035325`
  - 100us：`sst_dram_si/outputs_large/paper2/dram_mesh_4x4/20260102-035420`
- Task6.3 已完成：route/comm（路由构建 + fanout + 发送闭环）迁入 `workload=snn`（CoreShell 仅做 runtime 注入 + 统计汇聚 + gating 转发）。
  - 10us：`sst_dram_si/outputs_large/paper2/dram_mesh_4x4/20260102-044754`
  - 100us：`sst_dram_si/outputs_large/paper2/dram_mesh_4x4/20260102-044853`
- Task6.4 已完成：GAS/window 阶段机迁入 `workload=snn`（CoreShell 仅作为 `IGasStageSink` 做镜像/统计并转发）。
  - 10us：`sst_dram_si/outputs_large/paper2/dram_mesh_4x4/20260102-053548`
  - 100us：`sst_dram_si/outputs_large/paper2/dram_mesh_4x4/20260102-053649`

---

## 2) 大任务拆分（Task6.2 → Task6.4）

> 说明：这里把“一个大任务”拆成可机械验收的小步；每步 10us/100us 回归通过才能进入下一步。

### Task6.2-Step2：把 weights 子系统的“所有权/装配”迁入 `SnnWorkload`

**目标**
- `SnnWorkload` 自己持有并配置 `WeightMemorySubsystem`（或至少持有其核心装配对象：`IWeightReader` + window issue 编排），不再依赖 CoreShell 持有 `weight_reader_adapter_ / weight_mem_subsystem_`。
- legacy host 仅保留“极薄的桥接”（最终会删除）。

**Files（预期）**
- Modify:
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/services/workload/snn/SnnWorkload.{h,cc}`
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPESubComponent.{h,cc}`
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/api/ILegacySnnWorkloadHost.h`
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/services/synapse/weights/WeightMemorySubsystem.{h,cc}`（若需要把配置形态做成 workload 可装配）
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/api/SnnWeightReader.h`（如需新增“纯接口适配器”）

**实施步骤（建议拆成多小步）**
1) 在 `SnnWorkload` 内引入 `WeightMemorySubsystem` 的持有者与配置入口（先不替换旧对象）
2) 把 `ComputeCoreContext.weight_reader` 改为指向 `SnnWorkload` 内部的 `IWeightReader`（而不是 legacy host 的 adapter）
3) 将 CoreShell 中 `weight_reader_adapter_` 退化为“兼容占位”并在日志中 warn（不改变默认行为）
4) 删除 `ILegacySnnWorkloadHost::legacySnnGetWeightReader()` 的依赖路径（最后一步才删接口）

**验收**
- 10us/100us 回归通过；关键字段与基线一致
- `control/` 不再拥有/构造 `weight_reader_adapter_`（或仅保留兼容 stub，且不走主路径）

---

### Task6.3：把 route/comm（路由构建 + fanout + 发送闭环）迁入 `SnnWorkload`

**目标**
- `SnnWorkload` 直接持有并驱动：
  - `SynapseRouteSubsystem`
  - `SpikeCommSubsystem`
  - `ISpikeTransport` 选择（parent / noc）
- CoreShell 只负责把 `INocTransport*` / `SnnPEParentInterface*` 注入 workload runtime，不再在 `setParentInterface()` 内做 route/comm 的装配细节。

**Files（预期）**
- Modify:
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/services/workload/snn/SnnWorkload.{h,cc}`
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPESubComponent.{h,cc}`
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/api/ICoreWorkload.h`（如需扩展 runtime sinks/refs）
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/services/synapse/route/SpikeCommSubsystem.{h,cc}`

**实施步骤**
1) 先把 “构建/绑定 runtime config” 的逻辑从 CoreShell 迁到 `SnnWorkload::onSetup()`（对象仍可暂放在 CoreShell，通过引用传递）
2) 再把对象所有权迁到 `SnnWorkload`（CoreShell 侧指针清空）
3) 将 `drainCoreOutputsAndRoute_()/routeAndSendOutputs_()` 迁到 `SnnWorkload`（CoreShell 不再直接路由/发送）

**验收**
- 10us/100us 回归通过；`gas_scatter_spikes_emitted_total` 与 `neurons_fired_total` 与基线一致

---

### Task6.4：把 GAS/窗口编排（以及 step gate 交互）迁入 `SnnWorkload`（CoreShell 只转发事件）

**目标**
- 让 CoreShell 不再理解 GAS 的阶段机与窗口语义：
  - CoreShell 只转发 “全局 step 开始/结束”、“BeginGather/EndGather/BeginApply/... 等 stage event”
  - `SnnWorkload` 决定在何时发哪些 `GasOp`、何时 issue reads、何时 scatter/route
- 仍然保持“全局同步 step”的确定性语义不变（每个 step 所有 PE 同步开始、同步结束）。

**Files（预期）**
- Modify:
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/services/workload/snn/SnnWorkload.{h,cc}`
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPESubComponent.{h,cc}`
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/api/ICoreWorkload.h`（新增 stage hooks）
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/components/GatherBufferIF.*`（若需要补齐更通用的 gate/notify 接口）

**建议的接口形态（先加 hook，后迁移逻辑）**
- 在 `ICoreWorkload` 增加可选 hook（默认 no-op）：
  - `onGlobalStepStart(uint32_t seq)`
  - `onGlobalStepFinish(uint32_t seq)`
  - `onGasStageEvent(GasOp op, uint32_t seq, ...)`（或拆成 begin/end）
- CoreShell 在收到相应事件时只调用这些 hook，不再直接驱动 weights/route/compute 的 stage begin/end。

**验收**
- 10us/100us 回归通过；并做 2 次确定性重跑
- CoreShell 内不再包含 GAS 阶段机（或只保留极薄“event forwarder”）

---

## 3) 终局收尾（清理 legacy host）

**目标**
- `services/workload/snn/SnnWorkload` 不再需要 `ILegacySnnWorkloadHost`；
- CoreShell 不再暴露 `legacySnn*` 接口；
- `control/` 目录只剩 “装配 + 转发 + 统计汇聚”。

**实施策略**
1) 每删/每替换一个 `legacySnn*` 都必须跑 10us/100us 回归
2) 删除接口前先做 “编译期 fail-fast”（例如：删除调用点、把默认实现 `fatal`），确保不会 silent fallback

---

## 4) 风险点（必须提前规避）

- Tick/阶段重复驱动：必须确保 compute/weights/route/GAS 的 tick 不被双重调用（会导致非确定性或量级漂移）。
- 生命周期顺序：`init/setup` 阶段的构建/绑定顺序必须保持（尤其是 route 构建、weight loader barrier、step gate open/close）。
- 性能：迁移过程中避免在热路径引入 `dynamic_cast/getenv/map<string,...>`；配置应一次性缓存。
