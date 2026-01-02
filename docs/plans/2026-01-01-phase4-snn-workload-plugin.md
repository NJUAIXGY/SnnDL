# Phase4: SNN Workload 插件化（workload=snn）Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将当前默认 SNN 主链路从 `control/SnnPESubComponent` 迁为 `workload=snn` 插件（`services/workload/snn/*`），使 `SnnPESubComponent` 变为真正的 CoreShell（通用控制壳），并保持 `sst_dram_si/test_mesh_4x4.py` 的 10us/100us 回归口径与性能稳定。

**Architecture:** 采用 **方案 B**：在 `ICoreWorkload` 之上新增 `ISpikeWorkload`（仅 SNN workload 需要 `SpikeEvent`），CoreShell 在构造/初始化阶段做一次 `dynamic_cast` 缓存指针，热路径仅做一次直接调用，避免每 spike RTTI；迁移按“可回归小步”推进：先搭接口/工厂/委托，再把 SNN 事务（GAS + synapse/weights + synapse/route + compute glue）逐块下沉至 `SnnWorkload`。

**Tech Stack:** C++17（SST Elements: `SnnDL`）、SST `StandardMem`（通过 `services/synapse/stdmem/StdMemEndpoint`）、Merlin/NoC（`services/noc/*` + `api/INocTransport`）、Python mesh 模板（`sst_dram_si/tools/run_mesh_with_time.sh`）。

## Progress（更新到 2026-01-01）
- ✅ Task1-3：接口/工厂/SnnWorkload 壳已落地并回归通过（见 `TECH_PROGRESS.md`）。
- ✅ Task4：CoreShell 创建 workload + 生命周期转发已落地并回归通过（见 `TECH_PROGRESS.md`）。
- ✅ Task5-Step1：`deliverSpike/clockTick` 入口切到 workload，采用 `ILegacySnnWorkloadHost` 桥接保持行为不变：
  - 10us：`sst_dram_si/outputs_large/paper2/dram_mesh_4x4/20260101-144844`
  - 100us：`sst_dram_si/outputs_large/paper2/dram_mesh_4x4/20260101-144950`
- ✅ Task5-Step2：`hasWork/getUtilization/getStatistics` 委托（保持 key 兼容；SNN=legacy host 桥接；stream=workload 自管指标）：
  - SNN 10us：`sst_dram_si/outputs_large/paper2/dram_mesh_4x4/20260101-151015`
  - SNN 100us：`sst_dram_si/outputs_large/paper2/dram_mesh_4x4/20260101-151119`
  - stream 10us（env）：`sst_dram_si/outputs_large/paper2/dram_mesh_4x4/20260101-151303`

---

## 0) 硬约束与验收口径（每个 Task 都必须满足）

### 0.1 兼容性硬约束
- 默认配置仍跑 SNN：不要求用户改脚本；`workload_impl` 默认为 `snn`。
- 任何 C++ 变更必须 `make install` 后仿真生效。
- 不引入 `use_event_weight_fallback=1` 作为任何“掩盖问题”的兜底路径（禁用）。

### 0.2 回归命令（统一）
```bash
cd "sst_workspace/sst-elements/src/sst/elements/SnnDL" && make -j4 && make install

cd "sst_dram_si"
MESH_SIM_TIME="10us" ./tools/run_mesh_with_time.sh
MESH_SIM_TIME="100us" ./tools/run_mesh_with_time.sh
```

### 0.3 回归必查字段（来自 `essential_summary_mesh.json`）
- `spike_activity.neurons_fired_total`（禁止为 0）
- `spike_activity.window_spikes_total`
- `spike_activity.gas_scatter_spikes_emitted_total`
- `gas.gather_ns_p95 / gas.apply_ns_p95 / gas.scatter_ns_p95`（禁止全部为 0）

### 0.4 Git 约束（重要）
- 本计划不包含任何 `git commit/push/reset/checkout` 等写操作；若需要提交，由主人明确批准后再做。

---

## 1) 目标分层（Phase4 完成后的“依赖方向”）

### 1.1 CoreShell（`control/SnnPESubComponent`）最终应只保留
- SST 生命周期对接：`init/setup/finish/clockTick`
- 平台装配：`StdMemEndpoint`（产生 `IMemoryAccess` / `IGasStepGate`）、`INocTransport` 注入、统计对象注册（SST Statistic 创建只能在 SST SubComponent 内）
- 分发：把 `deliverSpike / deliverPacket / onClockTick` 交给 workload（`SnnWorkload` 或 `StreamWorkload`）
- 最少量缓存指针：`ISpikeWorkload*`、可选 hooks 缓存（避免热路径 RTTI）

### 1.2 SNN Workload（`services/workload/snn/SnnWorkload`）应独占承载
- 输入路径：`deliverSpike(SpikeEvent*)`（入队/统计/窗口收敛）
- 窗口/GAS 编排（或其桥接到现有 GAS controller）
- Synapse/Weight：`services/synapse/weights/WeightMemorySubsystem` + BCSR/缓存/预算/窗口集合
- Route/Comm：`services/synapse/route/{SynapseRouteSubsystem,SpikeCommSubsystem}` + `ISpikeTransport`（NoC or Parent）
- Compute glue：创建/配置 `compute/ISnnComputeCore`，注入 `IWeightReader`，驱动 `compute_core_->onClockTick/endCycle/drainOutputs`
- 统计与诊断：更新 CoreShell 创建的 SST Statistic（通过 runtime sinks 指针）

### 1.3 Stream Workload（已有 `services/workload/stream/StreamWorkload`）
- 完全不依赖 `SpikeEvent`、不依赖 synapse/gas/weights/route

---

## 2) 接口设计（方案 B：不让 stream 看到 SpikeEvent）

### 2.1 新增 `api/ISpikeWorkload.h`
- `class ISpikeWorkload : public ICoreWorkload { virtual void deliverSpike(SpikeEvent*) = 0; }`
- 说明：只有 `workload=snn` 需要实现；`workload=stream` 不需要看到 `SpikeEvent` 类型。

### 2.2 `ICoreWorkload` 建议扩展（避免 CoreShell 继续持有 compute/队列/统计）
> 这些是“工作负载通用能力”，不引入 SNN 语义，便于 CoreShell 把 `SnnCoreAPI` 必需接口完全委托出去。

- 建议在 `api/ICoreWorkload.h` 增加（默认实现即可，避免强迫每个 workload 都立刻实现）：
  - `virtual void onInitPhase(unsigned /*phase*/) {}`
  - `virtual void onSetup() {}`
  - `virtual void onFinish() {}`
  - `virtual bool hasWork() const { return false; }`
  - `virtual double getUtilization() const { return 0.0; }`
  - `virtual void getStatistics(std::map<std::string, uint64_t>& /*stats*/) const {}`

> 若担心 `std::map` 头文件过重：可后续演进为 callback/flat-map，但 Phase4 先保持与现有 `SnnCoreAPI::getStatistics(std::map<...>&)` 兼容。

---

## 3) 迁移策略（Strangler Fig：先“委托可跑”，再“逐块下沉”）

> 关键：任何一步都必须能 `10us → 100us` 回归；每次移动只动一个“责任块”，防止把 bug 面扩大到不可定位。

### 3.1 Phase4-A：接口 + 工厂 + 最小委托（不改变 SNN 行为）
目标：让 CoreShell 具备“默认创建 `workload=snn`”的框架能力，但仍暂时允许 SNN 逻辑保留在旧位置（只做框架对接）。

### 3.2 Phase4-B：把“调用入口”迁给 workload（行为仍 100% 等价）
目标：`deliverSpike/clockTick/getStatistics/hasWork/getUtilization/setup/finish` 全部由 workload 提供；CoreShell 仅转发。

### 3.3 Phase4-C：按子系统块下沉（真正壳化）
依次将以下块从 CoreShell 迁入 `SnnWorkload`（每块一个 Task）：
1) compute core 创建/配置与驱动
2) synapse/weights（WeightMemorySubsystem + BCSR）
3) synapse/route（route build + gating + SpikeComm）
4) GAS/window 编排与阶段状态（含 `GasPhaseController` / stage seq 等）
5) 统计计数器与输出口径（确保 key 不变）

---

## 4) Task-by-task 详细推进清单（每个 Task 都要 10us/100us 回归）

### Task 1: 新增 `ISpikeWorkload` 接口

**Files:**
- Create: `sst_workspace/sst-elements/src/sst/elements/SnnDL/api/ISpikeWorkload.h`
- Modify: `sst_workspace/sst-elements/src/sst/elements/SnnDL/Makefile.am`

**Step 1: 添加接口头文件（仅声明，不改行为）**
- 定义 `class ISpikeWorkload : public ICoreWorkload`
- 仅声明 `deliverSpike(SpikeEvent*)`

**Step 2: 编译检查**
- Run: `cd "sst_workspace/sst-elements/src/sst/elements/SnnDL" && make -j4`
- Expected: PASS

**Step 3: 回归（10us/100us）**
- Run: 按 0.2
- Expected: 关键字段与 baseline 一致

---

### Task 2: 扩展 `ICoreWorkload`（生命周期 + 指标委托）

**Files:**
- Modify: `sst_workspace/sst-elements/src/sst/elements/SnnDL/api/ICoreWorkload.h`

**Step 1: 添加默认虚函数（不强迫现有 workload 立刻实现）**
- 增加 `onInitPhase/onSetup/onFinish/hasWork/getUtilization/getStatistics`

**Step 2: 编译检查**
- Run: `cd "sst_workspace/sst-elements/src/sst/elements/SnnDL" && make -j4`
- Expected: PASS

**Step 3: 回归（10us/100us）**
- Run: 按 0.2
- Expected: 关键字段与 baseline 一致

---

### Task 3: 新建 `SnnWorkload`（插件壳）

**Files:**
- Create: `sst_workspace/sst-elements/src/sst/elements/SnnDL/services/workload/snn/SnnWorkload.h`
- Create: `sst_workspace/sst-elements/src/sst/elements/SnnDL/services/workload/snn/SnnWorkload.cc`
- Modify: `sst_workspace/sst-elements/src/sst/elements/SnnDL/api/CoreWorkloadFactory.h`
- Modify: `sst_workspace/sst-elements/src/sst/elements/SnnDL/services/workload/CoreWorkloadFactory.cc`
- Modify: `sst_workspace/sst-elements/src/sst/elements/SnnDL/Makefile.am`

**Step 1: 先落地最小实现（仅能编译）**
- `SnnWorkload` 实现 `ISpikeWorkload`
- 所有方法默认 fail-fast 或 no-op（仅用于接线阶段，暂不切换默认路径）

**Step 2: 工厂支持**
- `createWorkloadByName("snn") -> std::make_unique<SnnWorkload>()`

**Step 3: 编译检查**
- Run: `make -j4`
- Expected: PASS

**Step 4: 回归（10us/100us）**
- Run: 按 0.2
- Expected: 关键字段与 baseline 一致（因为默认仍走旧路径，不使用 `SnnWorkload`）

---

### Task 4: CoreShell 接线（创建 workload + 缓存 `ISpikeWorkload*`）

**Files:**
- Modify: `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPESubComponent.h`
- Modify: `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPESubComponent.cc`

**Step 1: 默认创建 snn workload（但先不切换执行入口）**
- `workload_impl` 默认 `snn`
- `workload_ = createWorkloadByName(workload_impl)`
- `spike_workload_ = dynamic_cast<ISpikeWorkload*>(workload_.get())`（仅做一次缓存）

**Step 2: init/setup/finish 时把生命周期事件转发给 workload**
- 在 CoreShell 的 `init(phase)` 末尾调用：`workload_->onInitPhase(phase)`
- 在 CoreShell 的 `setup()` 调用：`workload_->onSetup()`
- 在 CoreShell 的 `finish()` 调用：`workload_->onFinish()`

**Step 3: 编译检查 + 回归**
- Run: `make -j4 && make install` + 10us/100us
- Expected: 关键字段与 baseline 一致

---

### Task 5: 切换执行入口（SNN 仍等价，但入口由 workload 承接）

> 该 Task 完成后，CoreShell 的 `deliverSpike/clockTick/getStatistics/hasWork/getUtilization` 将全部委托给 workload。此时必须让 `SnnWorkload` 真正承载 SNN 逻辑（否则行为会改变）。

**Files:**
- Modify: `services/workload/snn/SnnWorkload.{h,cc}`
- Modify: `control/SnnPESubComponent_{spike,routing,scheme1}.cc`（按块迁移）
- Modify: `control/SnnPESubComponent.cc`（clockTick/统计委托）

**Step 1: 先迁移最小闭环：`clockTick + deliverSpike`**
- 把现有 `SnnPESubComponent::clockTick` 的 SNN 分支与 `deliverSpike` 主链路迁入 `SnnWorkload`
- CoreShell 的 `deliverSpike()` 改为：
  - `spike_workload_->deliverSpike(spike)`（若空则 fatal）
- CoreShell 的 `clockTick()` 改为：
  - `return workload_->onClockTick(...)`（并维护必要的 `total_cycles` 口径：建议将计数也迁到 workload，或由 CoreShell 提供“cycle tick”给 workload）

**Step 2: 迁移 `hasWork/getUtilization/getStatistics`**
- CoreShell 直接委托 `workload_->hasWork/getUtilization/getStatistics`
- 保持 key 映射一致（`spikes_received/neurons_fired/memory_requests/...`）

**Step 3: 回归（10us/100us）**
- Expected: 与 baseline 完全一致

---

### Task 6: 子系统块下沉（按顺序做，每块一个小步回归）

#### Task 6.1: compute core 完整下沉
**目标**：CoreShell 不再持有 `compute_core_`，改由 `SnnWorkload` 负责创建/配置/驱动。

**Files:**
- Modify: `services/workload/snn/SnnWorkload.{h,cc}`
- Modify: `control/SnnPESubComponent.cc`（移除 compute 创建/配置调用）
- Modify: `compute/*`（如需仅 include 调整）

**验收**：10us/100us 回归一致

#### Task 6.2: synapse/weights 下沉
**目标**：CoreShell 不再持有 `WeightMemorySubsystem/WeightAccessor/WeightCacheOps/BcsrWeightManager` 等对象；SnnWorkload 独占并通过 `IMemoryAccess` 绑定。

**关键点**：
- `StdMemEndpoint` 仍由 CoreShell 装配（SST 限制），但把 `IMemoryAccess*` 通过 runtime 注入给 SnnWorkload。

**验收**：10us/100us 回归一致

#### Task 6.3: synapse/route + SpikeComm 下沉
**目标**：gating 决策、route build、spike emit 事务全部归 SnnWorkload。

**关键点**：
- `ICoreControlHooks::applyGatingDecision` 在 CoreShell 侧转发给 workload（建议缓存 `IGatingHooks*`，避免每周期 RTTI）。

**验收**：10us/100us 回归一致

#### Task 6.4: GAS/window 编排与阶段状态下沉
**目标**：CoreShell 不再持有 `gas_enable/window_mode/curr_stage_seq/gas_stage/...`，仅提供 `IGasStepGate`/`IGasCmdSender` 平台能力；阶段逻辑在 SnnWorkload。

**关键点**：
- `IGlobalStepHooks::onGlobalStepStart` 建议拆为两段：
  1) CoreShell：只负责打开 `IGasStepGate::openStep(seq)`（平台语义）
  2) SnnWorkload：接收 step start 通知（SNN 语义：更新 stage seq、驱动窗口）

**验收**：10us/100us 回归一致

---

### Task 7: 边界硬化与清理（Phase4 收尾）

**目标**：`control/SnnPESubComponent.*` 彻底不 include synapse/compute 实现头，仅依赖 `api/*` + 少量平台 glue；SNN 事务文件全部位于 `services/workload/snn/*` 与 `services/synapse/*`。

**Files:**
- Modify: `control/SnnPESubComponent.h/.cc`
- Modify: `docs/PHASE6_UNIVERSAL_WORKLOAD_STREAMING_PLAN.md`（标记 Phase4 完成与验收 run_dir）
- Modify: `TECH_PROGRESS.md`（记录阶段、run_dir、关键指标）

**验收**：
- `grep -R "synapse/" control` 命中应趋近 0（允许 api 级 include）
- `MESH_SIM_TIME=100us` 回归通过且关键字段不变

---

## 5) 风险与缓解

- **风险：迁移过程中引入双重释放/漏删 event**（SpikeEvent/NocPacketEvent ownership）
  - 缓解：每个入口（deliverSpike/deliverPacket）明确“谁 delete”；在 strict/diag 下加断言式诊断。
- **风险：时序漂移导致统计口径变化**
  - 缓解：迁移时保持 `MultiCorePE` 的调用顺序不变；对 `clockTick` 的 cycle 计数口径写成单元测试式的断言（例如 active_cycles/total_cycles 不回退）。
- **风险：性能退化**
  - 缓解：`dynamic_cast` 只在初始化做一次缓存；热路径只做直接指针调用；避免每周期 `std::map` 构造以外的新开销。

---

## 6) 执行前确认点（主人需要明确同意）

1) 是否接受在 Phase4 中对 `api/ICoreWorkload.h` 做一次“通用能力扩展”（生命周期/统计/hasWork/utilization）？
2) SNN workload 的最终命名：`workload_impl=snn`（默认）是否冻结？后续是否允许 `workload_impl=legacy_snn` 作为临时回退（不推荐，除非主人强烈要求）。
