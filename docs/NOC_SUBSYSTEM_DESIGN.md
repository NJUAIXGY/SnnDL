# NoC 子系统设计（Phase4-A1：全覆盖 send/recv/forward + 本地投递）

> 目标（主人选定 A1）：将 **NoC（网络与片上通信）** 抽成独立闭环子系统，覆盖：
> - 本 PE 内：跨 core 投递/转发（ring）
> - 跨 PE：发送/接收/转发（NIC / mesh link）
> - 输入侧：来自 NIC 回调、外部端口、方向链路的事件解包与入队
>
> 并让 `components/MultiCorePE` 朝“纯控制壳”收敛：只装配后端（NIC/ring/links）、转发生命周期与聚合统计，不再承载 NoC 事务逻辑。

---

## 1. 背景：当前差距（MultiCorePE 仍是 NoC 事务中心）

当前输出路径为（Phase4-A1.3 后默认形态）：

```
ComputeCore::drainOutputs
  -> control/SnnPESubComponent
    -> services/SpikeCommSubsystem (fanout 已下沉至 Synapse/Route)
      -> api/ISpikeTransport (NocSpikeTransport)
        -> api/INocTransport
          -> services/NocSubsystem
             -> 本地投递 / ring 路由 / NIC 外发
```

输入路径分散在：
- `MultiCorePE::handleExternalSpikeEvent`（外部端口接收 + hop/TTL 处理 + 可能直接转发）
- `MultiCorePE::handleExternalSpike`（NIC 回调接收，直接入队）
- `MultiCorePE::{north,south,east,west,network} link handlers`（方向链路事件解包）
- `clockTick()` 中对 `external_spike_queue_` 的处理与对 ring 的 tick/receive 循环

这导致：
- NoC 事务逻辑与 MultiCorePE 强绑定，控制壳化困难；
- 输入/转发路径存在重复与口径漂移风险（例如 hop_count/统计更新不一致）；
- 后续想把 MPI/不同 NIC/不同片上互连实现替换为“后端”，会反向牵动 MultiCorePE 大量代码。

---

## 2. 目标与边界（严格）

### 2.1 NoC 子系统负责（必须）
- **send**：接管 `SpikeEvent*` 并决定：
  - 目标在本 PE：直接投递到目标 core（或进入 ring）
  - 目标在其它 PE：外发到 NIC/mesh（并处理自环/TTL 等）
- **recv**：统一接收来源：
  - NIC 回调收到的 `SpikeEvent*`
  - 外部输入端口 / mesh 方向端口收到的 `SST::Event*`（包括 `SpikeEventWrapper`）
- **forward**：当收到的 spike 目标不在本 PE 时，按既有语义进行转发（仍由目标 node 字段决定）
- **本地投递**：由 NoC 做“目标 core 判定”，并调用最小回调 `deliver_to_core(core_id, SpikeEvent*)`
- **片上互连**：对 ring 做 tick/receive，并在收到 ring 消息后投递到 core

### 2.2 NoC 子系统禁止（必须不做）
- 禁止出现 fanout 选择、BCSR/权重解析、路由构建等 **Synapse/Route** 语义；
- 禁止依赖 `control/` 的私有头与容器；
- 禁止改变现有 mesh 模版的统计字段/行为口径（仅搬家 + 适配）。

---

## 3. 设计方案（推荐：适配器起步，逐步搬迁）

为降低风险，Phase4-A1 采用两段式：

### 3.1 Phase4-A1.1（适配器/编排层）——已完成
- `services/NocSubsystem.{h,cc}` 先以编排层形式落地：统一收敛输入侧，并在输出侧做“目的地判定 + 分流策略”。
- MultiCorePE 变薄：其 `sendSpike/handleExternalSpike*/link handlers` 只做转发到 `NocSubsystem`（冻结行为与统计口径）。

优点：改动可控、行为不变风险低、可快速通过 100us 回归。

### 3.2 Phase4-A1.2（后端搬迁）——已完成（NoC 真正闭环）
- `NocSubsystem::Runtime` 由 MultiCorePE 装配并注入后端指针：`SnnInterface* nic`、`OptimizedInternalRing* optimized_ring`、`SST::Link* external_spike_output_link`。
- NoC 内部实现并闭环：
  - ring：`sendMessage/tick/receive/deliver`（优先 optimized ring）
  - 外发/中继：`sendExternal`（计 sent）/ `forwardExternal`（不计 sent）
- 备注：当前主要覆盖 optimized ring；若需要严格兼容 legacy `InternalRing`（`use_optimized_ring=0`），可作为后续补齐项。

### 3.3 Phase4-A1.3（接口化：INocTransport）——已完成
- 新增 `api/INocTransport.h` 冻结 NoC 调用面：`sendFromCore` / `injectLocal` / `sendExternal`。
- `services/NocSubsystem` 实现 `INocTransport`，成为 Control/Step 的唯一 NoC 依赖入口。
- Control/Step 接入：
- `api/NocSpikeTransport.h`：将 `SpikeCommSubsystem` 的 `ISpikeTransport` 落到 `INocTransport::sendFromCore(src_core, ev)`。
  - `services/stimulus/StepActivationSubsystem`：注入/外发优先走 `INocTransport`。

---

## 4. 最小接口草案（面向代码落地）

> Phase4-A1.3 已落地 `api/INocTransport`，用于冻结跨组件 NoC 调用面（Control/Step 只依赖该接口）。

`api/INocTransport`（最小接口）：
- `sendFromCore(int src_core, SpikeEvent*)`：常规发放（NoC 内部分流本地/跨核/跨 PE）。
- `injectLocal(int dst_core, SpikeEvent*)`：本 PE 内本地直达注入（不走 ring；用于 Step/控制面）。
- `sendExternal(SpikeEvent*)`：跨 PE 外发（计入 `external_spikes_sent`）。

`services/NocSubsystem` 关键入口：
- `configure(cfg)`：仅保存配置（如 hop 限制/自环策略/verbose 门控）
- `bindRuntime(rt)`：注入后端与回调（Output/NIC/ring/links/parent hooks）
- `onCoreSend(SpikeEvent* ev)`：来自 compute 输出（ParentSpikeTransport → parent->sendSpike）
- `onNicReceive(SpikeEvent* ev)`：来自 NIC 回调
- `onLinkReceive(SST::Event* ev, const char* dir)`：来自方向链路/外部端口
- `drainIncomingQueue()`：在 `clockTick()` 早期调用，保持与历史顺序一致
- `tickRing(uint64_t now_cycle)`：在 `clockTick()` ring tick 阶段调用

事件所有权：NoC 入口统一 **接管指针生命周期**（与现有 `sendSpike/sendExternalSpike` 语义一致），并在投递/转发失败时负责释放。

---

## 5. 回归与验收（强制）

每一步改动都必须：
1) `cd sst_workspace/sst-elements/src/sst/elements/SnnDL && make -j4 && make install`
2) `cd sst_dram_si && export MESH_SIM_TIME=10us && ./tools/run_mesh_with_time.sh`
3) `cd sst_dram_si && export MESH_SIM_TIME=100us && ./tools/run_mesh_with_time.sh`
4) 同配置重复跑 100us 至少 2 次，关键字段完全一致（确定性）

关键字段（`essential_summary_mesh.json`）：
- `spike_activity.neurons_fired_total`（禁止为 0）
- `spike_activity.gas_scatter_spikes_emitted_total`
- `memory.memory_requests/memory_bytes`
- `nic.packets_sent`
- `window_metrics.*`

---

## 6. 风险与缓解

- **时序/顺序漂移**（queue 处理与 ring tick 顺序变化）：Phase4-A1.1 强制保持原 `clockTick()` 调用顺序（先 drain incoming，再 tick ring）。
- **所有权/重复释放**：NoC 子系统入口统一接管 `SpikeEvent*`；投递到 core 后由 core 接管；向 NIC 外发后由 NIC 接管；其余路径 NoC 负责 delete。
- **legacy InternalRing**：先保持兼容（仍由 MultiCorePE 提供 callback）；稳定后再迁移/封存。
