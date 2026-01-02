# Phase 5 Boundary Hardening Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 将当前已稳定可回归的子系统化重构（`dev/snndl-modularization-20251227@d6aa7f7`）进一步“硬化边界”，使 Control / Memory / NoC / Synapse / Compute 的依赖方向在**代码层面不可被破坏**，并为后续更大规模的壳化/清理提供稳定基座喵～ (..•˘_˘•..)

**Architecture:** 以“结构改动优先、语义不变”为原则：先通过 PImpl/窄接口把 include 边界写死，再逐步封存 legacy 分支；每一步都必须 `make -j4 && make install` 生效并完成 `MESH_SIM_TIME=100us` 回归验证。

**Tech Stack:** C++17（SST Elements: `SnnDL`）、SST `StandardMem`/`memHierarchy`、Merlin NoC、Python mesh 模板（`sst_dram_si/test_mesh_4x4.py` + `sst_dram_si/tools/run_mesh_with_time.sh`）。

---

## 0) 基线、约束与验收口径（必须遵守）

### 0.1 基线
- 代码基线分支：`dev/snndl-modularization-20251227`
- 基线提交：`d6aa7f7`（`2025-12-27 22:57:31 +0800`）
- 重要历史参考（不建议直接 apply）：`refs/stash@{0}=018f183`（基于 `29b8498`，落后于当前 barrier/GatherBufferIF 修复）

### 0.2 约束（Hard constraints）
- **禁止**启用：`use_event_weight_fallback=1`（脚本已强制报错，任何方案不得依赖它掩盖问题）。
- **任何 C++ 改动必须 install 才生效**：
  - `cd "sst_workspace/sst-elements/src/sst/elements/SnnDL" && make -j4 && make install`
- **每一步都要回归**：先 `10us` 再 `100us`，确保 `neurons_fired_total` 不为 0，且关键字段不出现“gather/apply/scatter 全归零”。
- 不修改 Python 模板脚本的参数名与关键行为语义（兼容优先）。

### 0.3 回归口径（统一命令）
1) 构建安装：
   - `cd "sst_workspace/sst-elements/src/sst/elements/SnnDL" && make -j4 && make install`
2) 运行（模板方式）：
   - `cd "sst_dram_si" && export "MESH_SIM_TIME=10us" && ./tools/run_mesh_with_time.sh`
   - `cd "sst_dram_si" && export "MESH_SIM_TIME=100us" && ./tools/run_mesh_with_time.sh`
3) 验收字段（来自 `essential_summary_mesh.json`）：
   - `spike_activity.neurons_fired_total`（禁止为 0）
   - `spike_activity.window_spikes_total`、`spike_activity.gas_scatter_spikes_emitted_total`
   - `step_activation.invocations`、`step_activation.spikes_injected_total`
   - `gas.gather_ns_p95/apply_ns_p95/scatter_ns_p95`（禁止全部为 0）

---

## 1) 现状复盘（Phase4 结束后的边界形态）

> 这里仅记录“边界是否硬”，不回滚已修复的确定性问题（Global Step barrier + GatherBufferIF 修复）喵～ (๑•̀ㅂ•́) ✧

- Memory 已基本去语义化：`api/IMemoryAccess.h` + `services/memory/StandardMemAccess.*` 为主路径；历史 `StandardMemBackend`（带权重/BCSR 语义字段）已在 Phase5.5 清理删除。
- NoC 已实现“类型层通用化”：`services/noc/*` 以 `NocPacketEvent` 为 payload，`SpikeEvent` 不进入 NoC 事务层。
- Synapse 语义已集中：`services/synapse/{weights,route,gas}/`。
- 仍需硬化的关键点：
  1) `control/SnnPESubComponent.h` include 过重（仍直接 include `stdMem.h`、`SpikeEvent.h`、synapse/weights 实现头等），边界易被破坏；
  2) `control/` 仍存在 `StandardMem` 类型泄露点（只要头里出现嵌套类型 `StandardMem::Request`，就不得不 include `stdMem.h`）；
  3) `compute/ISnnComputeCore.h` 仍携带“权重/缓存请求”语义接口，影响 compute 的可替换性（接口绑架）；
  4) legacy 分支虽已集中，但仍需要明确“只读封存/不再可加载/不再出现在主链路构建”。

---

## 2) Phase 5 总体推进策略（推荐：硬边界优先）

**核心策略（推荐）**：先做“硬边界”再做“清理/删文件”。
- 原因：删文件/大搬家属于高风险操作；先让依赖方向与 include 边界稳定，后续清理才不会反复引入回归。

**推进节奏**：每个小步“结构改动 + 10us→100us 回归 + 记录”，严格避免一次性大改喵～ (..•˘_˘•..)

---

## 3) 任务清单（可执行、可验收）

### Task 5.1：ELI 硬规则落地（SST 可加载对象只能在 `components/**`）

**目标**
- `services/**` 下不出现 `SST_ELI_REGISTER_*`（哪怕文件不参与编译，也禁止留下“可加载”幻象）。

**Files**
- Modify:
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/services/legacy/README.md`

**Steps**
1) 搜索确认违例：
   - `cd "sst_workspace/sst-elements/src/sst/elements/SnnDL" && grep -RIn --include="*.h" --include="*.cc" "SST_ELI_REGISTER" "services" || true`
2) 若存在违例：将带 `SST_ELI_REGISTER_*` 的实现迁移到 `components/**`，并确保 services 下不再保留“可加载”幻象。
3) 再次搜索，确保 `services/**` 下为 0 命中。
4) 回归：10us→100us。

**验收**
- `grep` 无命中；
- 回归关键字段不变。

**已落地实现（2025-12-28）**
- `services/**` 下 `grep -RIn --include="*.h" --include="*.cc" "SST_ELI_REGISTER" "services"` 为 0 命中
- legacy NoC 参考实现已删除（避免重复/可加载幻象），权威位置统一为 `components/noc/*`

---

### Task 5.2：Control PImpl 硬边界（把实现对象与重 include 全部塞进 Impl）

**目标**
- `control/SnnPESubComponent.h` 变成“窄头”：只暴露 public API、仅包含最小 include。
- 所有重依赖（`stdMem.h`、synapse/weights/route 实现头、容器/统计细节）全部移入 `Impl`。

**Files**
- Create（推荐做法，便于多个 `.cc` 共享 Impl）：
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPESubComponent_impl.h`（仅供 `control/*.cc` include，不对外）
- Modify：
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPESubComponent.h`
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPESubComponent.cc`
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/services/synapse/stdmem/SnnPESubComponent_mem.cc`
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPESubComponent_bcsr.cc`
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPESubComponent_routing.cc`
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPESubComponent_spike.cc`
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPESubComponent_scheme1.cc`
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPEApplyScatter.cc`
  - （可选）`sst_workspace/sst-elements/src/sst/elements/SnnDL/control/README.md`

**Steps**
1) 在 `SnnPESubComponent.h` 中：
   - 前置声明 `class SpikeEvent;` 等必要类型；
   - 移除 `#include <sst/core/interfaces/stdMem.h>`、`#include "SpikeEvent.h"`、`#include "synapse/weights/SnnBcsrWeightManager.h"` 等重 include；
   - 仅保留 `std::unique_ptr<Impl> impl_;` 与必要的轻量字段（若任何 inline 需要）。
2) 新增 `SnnPESubComponent_impl.h`：
   - include 所有重依赖；
   - 定义 `struct SnnPESubComponent::Impl`，容纳原本 private 成员。
3) 修改各 `.cc`：
   - include `SnnPESubComponent_impl.h`；
   - 将原 private 成员访问替换为 `impl_->...`。
4) 编译安装 + 10us→100us 回归。

**验收**
- `control/SnnPESubComponent.h` 的 include 列表显著缩小；
- 回归通过。

**已落地实现（2025-12-28）**
- 采用“最小改动先硬化 include 边界”的方式：
  - `control/SnnPESubComponent.h` 移除 synapse 实现头（`synapse/weights/SnnBcsrWeightManager.h`、`synapse/stdmem/StdMemEndpoint.h`）
  - 将 `bcsr_weights_`、`stdmem_ep_` 改为 `std::unique_ptr<>` 持有，并在 `.cc` 中包含实现头与完成构造（避免头文件泄露实现类型）
- 已进一步推进强 PImpl：
  - `control/SnnPESubComponent_impl.h` 成为唯一内部实现承载点：StageEventHub 吸收、`gas_ctrl_` 与统计汇报 `report*` 全部迁入 `Impl`
  - `control/SnnPESubComponent.h` 仅保留 `std::unique_ptr<Impl> impl_`（以及必要的轻量字段/前置声明）
- 相关改动文件：
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPESubComponent.h`
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPESubComponent.cc`
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPESubComponent_bcsr.cc`
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPESubComponent_routing.cc`
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPEOrchestrators.cc`
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/services/synapse/stdmem/SnnPESubComponent_mem.cc`
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPESubComponent_impl.h`
- 回归验收（mesh 模板方式）：
  - 10us：`sst_dram_si/outputs_large/paper2/dram_mesh_4x4/20251228-161122`
  - 100us：`sst_dram_si/outputs_large/paper2/dram_mesh_4x4/20251228-161242`

---

### Task 5.3：Control 完全不出现 StandardMem 类型（Control 不直连 stdMem）

**目标**
- Control 的头文件与实现文件（尽量）不再出现 `SST::Interfaces::StandardMem::Request` 等嵌套类型；
- `StandardMem` 的请求/回包分发全部通过 `services/memory/StandardMemAccess`（`IMemoryAccess`）完成。

**Files**
- Modify：
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPESubComponent_impl.h`
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/services/synapse/stdmem/SnnPESubComponent_mem.cc`
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/services/memory/StandardMemAccess.*`（若需要补齐回调形态/错误语义）

**Steps**
1) 将“内存回包处理函数”变为 `Impl` 私有实现，并确保对外不暴露 `StandardMem::Request*`。
2) `StandardMem` handler 统一改为：
   - Control 侧仅接收 `SST::Event*`/generic callback，然后转交给 `StandardMemAccess::handleMemoryResponse(...)`（或由 `Impl` 内部直接持有并调用）。
3) 搜索确认：
   - `cd "sst_workspace/sst-elements/src/sst/elements/SnnDL" && grep -RIn --include="*.h" --include="*.cc" "StandardMem::" "control" || true`
4) 回归：10us→100us。

**验收**
- `control/**` 无 `StandardMem::` 字样（或只剩极少注释允许，但推荐也清零）；
- 行为不变、回归通过。

**已落地实现（2025-12-28）**
- 采用“实现迁出 control 域”的方式收敛：把含 `StandardMem::` 的实现从 `control/*.cc` 迁到 `services/synapse/stdmem/`，并用 `StdMemEndpoint` 承接 Begin/EndGather 控制面发送。
- grep 验收：`cd "sst_workspace/sst-elements/src/sst/elements/SnnDL" && grep -RIn "StandardMem::" "control" || true` 为 0。
- 回归验收：10us/100us 模板回归通过（run_dir：`sst_dram_si/outputs_large/paper2/dram_mesh_4x4/20251228-020501`、`sst_dram_si/outputs_large/paper2/dram_mesh_4x4/20251228-020614`）。

---

### Task 5.4：Compute 接口收敛（权重/缓存语义迁出 compute 主接口）

**目标**
- `compute/ISnnComputeCore` 成为“动力学/发放判定”主接口；
- 权重/缓存请求通过 `ComputeCoreContext.weight_reader` 注入，compute 不再被迫实现 `requestWeightBCSR/resolveWeightKey/...`。

**Files**
- Create（推荐）：
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/compute/IWeightAwareComputeCore.h`（可选扩展接口，仅当某 compute 需要权重语义时实现）
- Modify：
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/compute/ISnnComputeCore.h`
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/compute/SnnComputeCore.*`（默认实现适配）
  - `sst_workspace/sst-elements/src/sst/elements/SnnDL/control/SnnPESubComponent*.cc`（调用点迁移/降级为 capability）

**Steps**
1) 在 `ISnnComputeCore.h` 中标记/移除权重相关虚函数（或迁移到扩展接口），并用 `ComputeCoreCapabilities` 明确表达“该 core 是否需要 weight cache/weight requests”。
2) Control 侧改为：
   - 默认仅依赖 `IWeightReader`（由 `WeightMemorySubsystem` 提供）；
   - 只有当 compute 实现了 `IWeightAwareComputeCore` 时，才走 legacy/扩展路径（否则 fail-fast）。
3) 10us→100us 回归。

**验收**
- 新 compute core 可以在不实现任何权重相关方法的前提下被加载/替换；
- 回归通过。

**已落地实现（2025-12-28）**
- 新增可选扩展接口：`compute/IWeightAwareComputeCore.h`（权重/缓存语义仅在该接口出现）
- 收敛主接口：`compute/ISnnComputeCore.h` 移除 `requestWeight/requestWeightBCSR/weightCacheTryGet/weightCacheStore/resolveWeightKey`
- 默认实现适配：`compute/SnnComputeCore.*` 的 `DefaultSnnComputeCore` 实现 `IWeightAwareComputeCore`
  - `DefaultSnnComputeCore::getCapabilities()`：`needs_weight_cache=false`（权重读取经 `ComputeCoreContext.weight_reader` 注入）
- 控制层去除对 compute cache 的硬依赖：`control/SnnPESubComponent.cc` 中 `WeightMemorySubsystem::OrchestratorConfig` 的 cache 回调统一落到 control 的 `weightCacheTryGet_/weightCacheStore_`
- 构建文件同步：`Makefile.am/in/Makefile`
- 回归验收（mesh 模板方式）：
  - 10us：`sst_dram_si/outputs_large/paper2/dram_mesh_4x4/20251228-111511`（`neurons_fired_total=1`；`gas.gather/apply/scatter p95` 均非 0）
  - 100us：`sst_dram_si/outputs_large/paper2/dram_mesh_4x4/20251228-111618`（`neurons_fired_total=2220`；`gas.gather/apply/scatter p95` 均非 0）

---

### Task 5.5：收尾清理（高风险：删除/迁移 legacy，需要主人二次确认）

**目标**
- 删除/封存不再使用的旧接口与实现，降低长期维护面；
- 更新文档：冻结 API 口径（`api/README.md`）与职责表（`docs/SUBSYSTEM_MODULARIZATION_ROADMAP.md`）。

**高风险操作提醒**
- 删除文件/大规模移动属于高风险操作：执行前必须由主人明确确认“继续/确认/是”。

**候选动作（执行前需再审计引用与构建文件）**
- `services/legacy/*` 中无引用且不参与构建的实现：删除或迁移到只读 archive。
- Makefile.am/in 中移除 legacy 编译残留（若有）。
- 文档更新与路径对齐。

**验收**
- 全量编译 + 100us 回归通过；
- `docs/README.md` 索引更新；新增/变更的 API 在 `api/README.md` 中可查。

---

## 4) 风险与回退策略（必须写清楚）

- 风险：PImpl 搬迁可能引入遗漏（空指针/生命周期/统计未初始化）
  - 缓解：每搬一个成员就 `make -j4` 快速编译一次；先跑 10us 再跑 100us；必要时打开已有 diag 开关定位。
- 风险：Compute 接口变更可能牵动控制层与默认实现
  - 缓解：先通过“扩展接口 + capabilities”双轨兼容一段时间，避免一次性破坏。
- 回退原则：
  - 只允许在 `SnnDL` 子模块内回退；
  - 回退优先使用“回滚提交/切回稳定 commit”，避免破坏性清理命令（如需执行 `reset --hard`，必须再次征得主人确认）。
