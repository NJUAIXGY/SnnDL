# Task6.4: GAS/窗口编排（含 Step gate 交互）迁入 `workload=snn` Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 把 `control/SnnPESubComponent` 内的 GAS/窗口阶段机（Begin/EndGather、Begin/EndApply、Begin/EndScatter）与 Step gate 交互迁入 `services/workload/snn/SnnWorkload`，使 CoreShell 只做“事件转发 + runtime 注入 + 统计汇聚”，并保持 `sst_dram_si/test_mesh_4x4.py` 模板回归口径稳定、确定性不退化。

**Architecture:** 采用“桥接过渡、一次只搬一块”的策略：
1) 先把“阶段事件的 API”写死到 `api/ICoreWorkload`（默认 no-op），CoreShell 改为仅转发；
2) `workload=snn` 内部引入 `SnnGasWindowController`（建议放在 `services/workload/snn/`，后续可下沉到 `services/synapse/gas/`），承接窗口状态机、调用 `WeightMemorySubsystem`（issue/advance）与 compute core（apply/scatter），并通过已存在的 `ILegacySnnWorkloadHost` 维持统计口径；
3) 最后删除/禁用 CoreShell 内旧阶段机，避免双驱动。

**Tech Stack:** C++17（SST Elements `SnnDL`）、memHierarchy `StandardMem`（经 `IMemoryAccess`）、Merlin NoC、mesh 回归 `sst_dram_si/tools/run_mesh_with_time.sh`。

---

## Hard Constraints（必须遵守）
- 不修改 `sst_dram_si/test_mesh_4x4.py` 的参数名与默认语义（兼容优先）。
- 任何 C++ 变更必须 `make -j4 && make install` 后运行仿真才生效。
- 禁止启用 `use_event_weight_fallback=1`。
- 必须避免“阶段机双驱动”：同一个 window 内的 gather/apply/scatter **只能**有一个来源推进（否则量级漂移或非确定性）。

---

## Acceptance（每个小步都要满足）
1) Build/Install：
   - `cd "sst_workspace/sst-elements/src/sst/elements/SnnDL" && make -j4 && make install`
2) 回归（模板方式）：
   - `cd "sst_dram_si" && export "MESH_SIM_TIME=10us" && ./tools/run_mesh_with_time.sh`
   - `cd "sst_dram_si" && export "MESH_SIM_TIME=100us" && ./tools/run_mesh_with_time.sh`
3) 关键字段（`essential_summary_mesh.json`）：
   - `spike_activity.neurons_fired_total` 禁止为 0
   - `gas.gather_ns_p95/apply_ns_p95/scatter_ns_p95` 禁止三者同时为 0
   - `step_activation.invocations/spikes_injected_total` 量级合理且同 seed 重跑一致
4) 确定性（每个里程碑至少 2 次）：
   - 同配置同 seed 重跑两次关键统计完全一致

---

## Step0: 记录基线（不改代码）
**Command:**
- `cd "sst_dram_si" && export "MESH_SIM_TIME=100us" && ./tools/run_mesh_with_time.sh`

**Expected:**
- 记录 run_dir 作为 Task6.4 baseline（后续每步都对齐比较）

---

## Step1: 为 GAS 阶段事件增加 workload hook（API 先行）
**Files:**
- Modify: `sst_workspace/sst-elements/src/sst/elements/SnnDL/api/ICoreWorkload.h`

**Work:**
- 在 `ICoreWorkload` 增加可选 hooks（默认 no-op）：
  - `onGlobalStepStart(uint32_t seq)`
  - `onGlobalStepFinish(uint32_t seq)`
  - `onGasStageBegin(GasOp op, uint32_t seq, uint64_t now_cycle)`
  - `onGasStageEnd(GasOp op, uint32_t seq, uint64_t now_cycle)`
- 注意：API 层只暴露“阶段事件”，不暴露 `StandardMem`、不暴露 synapse 内部类型。

**Regression:** 10us → 100us

---

## Step2: CoreShell 改为“只转发阶段事件”（不再执行业务逻辑）
**Files:**
- Modify: `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPESubComponent.cc`
- Modify (if needed): `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPEOrchestrators.cc`

**Work:**
- 在现有 `orchestrateBeginGatherWindowSetup / orchestratePrepareApplyWindow / orchestrateBeginApplyWindowEntry / orchestrateBeginScatterSequence / orchestrateEndScatterSequence` 内：
  - 保留 `Impl` 的 stage_events 记录（CSV/DB），但把“核心业务动作”改为调用 `workload_->onGasStageBegin/End(...)`。
- 先做“只增不删”的双路径保护：
  - 增加 `Params` 开关（默认 off）：`workload_snn_owns_gas=0/1`
  - 当为 0：仍走旧路径（用于比对）
  - 当为 1：只走 workload hook（旧路径全部跳过）并对遗漏做 fail-fast

**Regression:** 10us → 100us（两次，确保确定性）

---

## Step3: 在 `workload=snn` 内引入 GAS controller（先桥接、后收敛）
**Files:**
- Create: `sst_workspace/sst-elements/src/sst/elements/SnnDL/services/workload/snn/SnnGasWindowController.h`
- Create: `sst_workspace/sst-elements/src/sst/elements/SnnDL/services/workload/snn/SnnGasWindowController.cc`
- Modify: `sst_workspace/sst-elements/src/sst/elements/SnnDL/services/workload/snn/SnnWorkload.{h,cc}`
- Modify: `sst_workspace/sst-elements/src/sst/elements/SnnDL/Makefile.am`（或对应 build 文件，补齐编译单元）

**Work:**
- `SnnGasWindowController` 承接以下能力（第一版允许使用 legacy_host 桥接）：
  - 接收 `onGasStageBegin/End`，驱动 `WeightMemorySubsystem` 的窗口读推进与 Apply/Scatter 收敛点；
  - 在 Scatter 收敛时调用：`compute_core_->endCycleCandidates / drainOutputs`（取决于现有 window 模式约定）；
  - 调用 `ISnnSpikeCommWorkload::emitNeuronFireBatch` 发送；
  - 通过 `legacy_host_->legacySnnOnNeuronFires(...)` 维持统计口径不漂移。
- 性能要求：
  - 热路径避免 `dynamic_cast/getenv`；所有开关/指针在 `bindRuntime/configureFromParams` 缓存。

**Regression:** 10us → 100us

---

## Step4: Step gate 交互迁入 workload（全局同步语义不变）
**Files:**
- Modify: `sst_workspace/sst-elements/src/sst/elements/SnnDL/services/workload/snn/SnnWorkload.{h,cc}`
- Modify (if needed): `sst_workspace/sst-elements/src/sst/elements/SnnDL/services/stimulus/StepActivationSubsystem.{h,cc}`

**Work:**
- CoreShell 仅转发“全局 step 开始/结束”给 workload；
- `workload=snn` 决定 Step 注入与 GAS window 的边界协调（保持“全 PE 同步开始、同步结束”的语义不变）。

**Regression:** 10us → 100us（至少两次）

---

## Step5: 删除/封存 CoreShell 内旧 GAS 业务实现（避免回退与双驱动）
**Files:**
- Modify: `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPESubComponent.cc`
- Modify: `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPEOrchestrators.cc`

**Work:**
- 将旧路径改为 fail-fast（或 `#if 0` 封存）：
  - 当 `workload_snn_owns_gas=1` 时，任何旧路径被触发都 `fatal`（断言式诊断）。
- 确保 `control/**` 不引入任何 synapse/memory 实现细节回流（只保留 API 头）。

**Regression:** 10us → 100us

---

## Step6: 文档与进度记录（落地结果可追溯）
**Files:**
- Modify: `sst_workspace/sst-elements/src/sst/elements/SnnDL/docs/plans/2026-01-02-phase4-task6-snn-workload-migration.md`
- Modify: `TECH_PROGRESS.md`（按日期追加，记录 run_dir 与关键字段）

**Work:**
- 记录 Task6.4 的 run_dir、关键字段、以及开关 `workload_snn_owns_gas` 的默认值与回退策略。

---

## Status（已完成）

### Build/Install
- ✅ `cd "sst_workspace/sst-elements/src/sst/elements/SnnDL" && make -j4 && make install`

### Regression（mesh 4x4 模板）
- ✅ 10us：`sst_dram_si/outputs_large/paper2/dram_mesh_4x4/20260102-053548`
  - `spike_activity.neurons_fired_total=1`
  - `gas.gather_ns_p95=200 / apply_ns_p95=4 / scatter_ns_p95=40`
- ✅ 100us：`sst_dram_si/outputs_large/paper2/dram_mesh_4x4/20260102-053649`
  - `spike_activity.neurons_fired_total=2220`
  - `gas.gather_ns_p95=44029 / apply_ns_p95=1 / scatter_ns_p95=40`

### 实现摘要
- `StdMemEndpoint` 通过 `IGasStageSink` 将 GAS stage/stat 事件投递到 CoreShell（`SnnPESubComponent`），CoreShell 仅维护最小镜像状态+统计写出，并转发给 `workload=snn`。
- `workload=snn`（`services/workload/snn/SnnWorkload`）接管 window 阶段机业务动作：BeginGather/BeginApply/BeginScatter/EndScatter。
- `WeightMemorySubsystem::overrideAccUpdate()` 支持在 cutover 后重绑 accumulator 回调；window accumulator 迁入 workload（不再依赖 CoreShell 的 `AccumulatorOps` 参与主链路）。
