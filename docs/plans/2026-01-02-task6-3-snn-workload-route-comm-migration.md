# Task6.3: SNN route/comm 迁入 `workload=snn` Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将 `control/SnnPESubComponent` 中的 route/comm 装配与发送闭环迁入 `services/workload/snn/SnnWorkload`，使 CoreShell 更纯（装配+事件/阶段转发+统计汇聚），并保持 `sst_dram_si/test_mesh_4x4.py` 模板回归口径稳定。

**Architecture:** `SnnWorkload` owns `SynapseRouteSubsystem + SpikeCommSubsystem + (NocSpikeTransport|ParentSpikeTransport)`；CoreShell 仅通过窄接口 `ISnnSpikeCommWorkload` 触发发送与 gating，并通过 legacy host hook 做计数/统计更新（不把 SST Statistic 句柄下沉到 workload）。

**Tech Stack:** C++17 (SST elements `SnnDL`), memHierarchy `StandardMem`, Merlin NoC, mesh regression via `sst_dram_si/tools/run_mesh_with_time.sh`.

---

## Acceptance（每步都要满足）
- `cd "sst_workspace/sst-elements/src/sst/elements/SnnDL" && make -j4 && make install`
- 10us：`cd "sst_dram_si" && export "MESH_SIM_TIME=10us" && ./tools/run_mesh_with_time.sh`
- 100us：`cd "sst_dram_si" && export "MESH_SIM_TIME=100us" && ./tools/run_mesh_with_time.sh`
- `essential_summary_mesh.json`：
  - `spike_activity.neurons_fired_total` 禁止为 0
  - `gas.gather_ns_p95/apply_ns_p95/scatter_ns_p95` 禁止全为 0

---

## Steps（Step1-7 一次性完成）

### Step1: 补齐 workload runtime 注入（parent + route stats sinks）
- Modify: `sst_workspace/sst-elements/src/sst/elements/SnnDL/api/ICoreWorkload.h`
- Modify: `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPESubComponent.{h,cc}`

### Step2: 新增 `ISnnSpikeCommWorkload` 窄接口
- Create: `sst_workspace/sst-elements/src/sst/elements/SnnDL/api/ISnnSpikeCommWorkload.h`
- Modify: `sst_workspace/sst-elements/src/sst/elements/SnnDL/services/workload/snn/SnnWorkload.{h,cc}`

### Step3: `SnnWorkload` 接管 `SynapseRouteSubsystem/SpikeCommSubsystem` 的所有权与 initRouting
- Modify: `sst_workspace/sst-elements/src/sst/elements/SnnDL/services/workload/snn/SnnWorkload.{h,cc}`
- Modify: `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPESubComponent.cc`（移除原先装配块，避免双装配）

### Step4: gating decision 转发到 workload
- Modify: `sst_workspace/sst-elements/src/sst/elements/SnnDL/api/ILegacySnnWorkloadHost.h`
- Modify: `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPESubComponent_routing.cc`

### Step5: window scatter 的发送委托给 workload（保持统计口径不漂移）
- Modify: `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPEApplyScatter.cc`

### Step6: 非 window 模式的 endCycle+drain+send 闭环迁入 `SnnWorkload`
- Modify: `sst_workspace/sst-elements/src/sst/elements/SnnDL/services/workload/snn/SnnWorkload.cc`
- Modify: `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPESubComponent.cc`（移除 legacy tick 内的 drain/send 调用点）

### Step7: 回归+记录
- Modify: `TECH_PROGRESS.md`
- Modify (optional): `sst_workspace/sst-elements/src/sst/elements/SnnDL/docs/plans/2026-01-02-phase4-task6-snn-workload-migration.md`

---

## Status（已完成）

### Build/Install
- ✅ `cd "sst_workspace/sst-elements/src/sst/elements/SnnDL" && make -j4 && make install`

### Regression（mesh 4x4 模板）
- ✅ 10us：`sst_dram_si/outputs_large/paper2/dram_mesh_4x4/20260102-044754`
  - `spike_activity.neurons_fired_total=1`
  - `gas.gather_ns_p95=200 / apply_ns_p95=4 / scatter_ns_p95=40`
- ✅ 100us：`sst_dram_si/outputs_large/paper2/dram_mesh_4x4/20260102-044853`
  - `spike_activity.neurons_fired_total=2203`
  - `gas.gather_ns_p95=44029 / apply_ns_p95=1 / scatter_ns_p95=40`

### 关键落地项（对应 Step1-7）
- runtime 注入补齐：`ICoreWorkload::Runtime` 增加 `parent_iface` 与 route 统计 sinks；CoreShell 统一在 `bindWorkloadRuntime_()` 绑定。
- 新增窄接口：`api/ISnnSpikeCommWorkload`（CoreShell 不再直接持有/装配 route/comm）。
- `workload=snn` 持有并装配 `SynapseRouteSubsystem + SpikeCommSubsystem`，并按 runtime 选择 transport（NoC 优先，无 NoC 走 parent）。
- gating decision：CoreShell `applyGatingDecision()` 直接转发到 `ISnnSpikeCommWorkload`。
- window scatter：CoreShell 仅做统计汇聚（legacy host），并批量委托 workload 发送（避免逐条构造/调用）。
- 非 window：`endCycle->drain->route/comm` 闭环迁入 `SnnWorkload::onClockTick()`；CoreShell 移除 legacy tick 内的重复推进点（避免漂移/非确定性）。
