<!--
SST-SnnDL 设计文档（草稿）
目标：将“脉冲通信/路由/投递”从控制层中抽离为独立子系统，便于后续替换通信实现（SST Link/NIC/MPI/离线回放等）。
-->

# Spike 通信子系统抽象（草稿）

状态：Phase D 已落地（接口以代码为准，本文为设计/口径补充）

相关上位文档：
- `docs/UNIVERSAL_CONTROL_CORE_DESIGN.md`（总体路线与分层目标）

## 1. 背景与问题

当前 `control/SnnPESubComponent` 在输出路径上仍直接依赖：
- `SpikeEvent` 的具体事件类型（SnnDL 事件层）
- `SnnPEParentInterface::sendSpike()`（父组件/框架侧投递）
- 以及（间接）`MultiCorePE` 的“本地/远端”路由策略与 NIC/外部互连实现

这带来两个问题：
1. 控制层无法被复用为“通用控制子核”：即便 Compute Core 可替换，输出路径仍被 SpikeEvent+Parent 强绑定。
2. 后续替换/扩展通信实现（例如：批量包、不同 NIC、MPI、记录回放）需要改动控制层核心代码，破坏解耦目标。

## 2. 目标（Goals）

1. 控制层不再直接调用 `parent_->sendSpike()`；改为调用通信子系统统一入口。
2. 通信实现可替换：控制层只依赖 `ISpikeTransport` 抽象，不依赖 MultiCorePE/NIC/MPI 的具体实现。
3. 保持外部行为一致（兼容性冻结）：默认实现仍走现有 `SnnPEParentInterface` 路径，不改变路由口径与统计口径。
4. 支持未来扩展：可在不改控制层的情况下加入批量发送、传输层统计、或新事件载体（SpikePacket 等）。

## 3. 非目标（Non‑Goals）

- 不改变现有网络拓扑、路由构建策略、门控语义、统计命名。
- 不在本 Phase 引入跨 PE 的新协议/重做 NIC。
- 不一次性替换 `SpikeEvent` 为通用事件（该工作属于更高层的“事件模型抽象”，另行规划）。

## 4. 术语

- **控制层**：`SnnPESubComponent`（control-plane），负责时序/窗口/调度、以及与 Compute Core 的边界交互。
- **Compute Core**：`compute/*`，只负责计算与产出“发放事件”（目前为 `FireEvent`）。
- **通信子系统**：`SpikeCommSubsystem`，负责 fanout 计算、事件构造、并通过传输层投递。
- **传输层**：`ISpikeTransport`，仅提供“把一个 Spike 事件送出去”的最小能力。

## 5. 现状：输出路径调用链

当前（简化）：

```
ComputeCore::drainOutputs -> [FireEvent]
  -> SnnPESubComponent::handleNeuronFire_
      -> route_provider_.computeFanout(...)
          -> new SpikeEvent(...)
              -> parent_->sendSpike(event)
                  -> MultiCorePE::sendSpike
                      -> routeInternalSpike / sendExternalSpike -> NIC/MPI/Link
```

其中 `SnnPESubComponent` 同时承担了：
- fanout 计算（策略/门控）
- 事件构造（SpikeEvent）
- 传输发出（父接口）

## 6. 方案概览：分层与依赖方向

新增两层：

```
Control(Core)  -->  SpikeCommSubsystem  -->  ISpikeTransport  -->  Parent/MultiCorePE/NIC/MPI
        |                 |
        |                 +--(可选) IFanoutProvider（默认用现有 SnnRouteProvider）
        |
        +-- ComputeCore（产出 FireEvent）
```

依赖方向保持“向下依赖抽象”，控制层不再依赖 transport 的具体实现。

## 7. 接口草图

### 7.1 ISpikeTransport（最小传输抽象）

建议新增：`api/ISpikeTransport.h`

```cpp
// -*- c++ -*-
#pragma once

namespace SST { namespace SnnDL {

class SpikeEvent;

class ISpikeTransport {
public:
    virtual ~ISpikeTransport() = default;

    // 语义：接管 SpikeEvent 生命周期（与现有 parent_->sendSpike 一致）
    virtual void send(SpikeEvent* event) = 0;
};

}} // namespace SST::SnnDL
```

默认适配器（保持兼容）：

```cpp
class ParentSpikeTransport final : public ISpikeTransport {
public:
    explicit ParentSpikeTransport(SnnPEParentInterface* parent) : parent_(parent) {}
    void send(SpikeEvent* event) override {
        if (parent_) parent_->sendSpike(event);
        else delete event;
    }
private:
    SnnPEParentInterface* parent_ = nullptr;
};
```

### 7.2 SpikeCommSubsystem（通信子系统）

文件：`services/SpikeCommSubsystem.{h,cc}`（已实现）

职责：
- 接收控制层的“发放事件”（当前为 `FireEvent` 或 `(neuron_idx, source_global)`）
- 调用 fanout provider 计算目的集合
- 构造 `SpikeEvent` 并通过 `ISpikeTransport` 投递

接口要点（已落地的当前代码形态，简化）：

```cpp
struct SpikeCommRoutingConfig { /* weights_template/topk/epsilon/mapping/BCSR/... */ };
struct SpikeCommGatingConfig  { /* event_mode/ttl/scope */ };
struct SpikeCommRuntimeConfig { /* log/transport/node/core/global_base/stats */ };

class SpikeCommSubsystem {
public:
    void configure(const SpikeCommRoutingConfig& routing_cfg,
                   const SpikeCommGatingConfig& gating_cfg);
    void bindRuntime(const SpikeCommRuntimeConfig& rt);
    void initRouting(); // 构建/复用共享路由表，并配置内部 fanout provider

    void emitNeuronFire(uint32_t neuron_idx, uint64_t now_cycle);
    void emitSource(uint32_t source_global, uint32_t source_local, uint64_t now_cycle);

    void applyGatingDecision(uint32_t src_global,
                             const std::vector<uint32_t>& dest_pes,
                             uint64_t current_cycle,
                             uint64_t ttl_cycles);
};
```

实现要点（Phase D）：
- `configure()` 仅保存配置；`initRouting()` 才执行路由表构建/共享缓存注入（避免构造期做重活）。
- 子系统内部组合 `SnnRouteProvider`：
  - 路由表构建与共享缓存（进程级）在 `SpikeCommSubsystem` 内部完成，控制层不再持有 `routes_*`/cache key/层掩码解析等状态；
  - `SnnRouteProvider::computeFanout()` 只依赖“已配置的路由表 + gating_cache”，复用原 fanout 语义。
- `emitNeuronFire()` 内部：`source_global = global_neuron_base + neuron_idx`，并统一走 `emitCommon_()`：
  - 先 fanout，再逐条 `new SpikeEvent(...)`，最后 `transport->send(...)`（transport 接管生命周期）。
- 路由共享缓存 key：保持 v1 语义，且不包含 `(pe,core)` 维度，避免重复构建导致的 RSS/构建耗时爆炸。

## 8. 所有权/生命周期与线程模型

- `SpikeCommSubsystem` 创建 `SpikeEvent*`，并调用 `ISpikeTransport::send()` 后交由传输层接管所有权（保持与当前 `parent_->sendSpike` 一致）。
- 传输层不得回调到 `SpikeCommSubsystem` 持有的临时对象；所有回调应由上层（MultiCorePE/NIC）现有机制处理。
- 现阶段仍假设 SST 单线程事件驱动；若后续引入并发/批量发送，需在 transport 实现内部保证线程安全，控制层不承担锁。

## 9. 迁移步骤（最小风险）

Phase B‑1（传输层抽象，无行为变化）：
1. 新增 `api/ISpikeTransport.h` 与 `ParentSpikeTransport`（适配现有 parent）。
2. `SnnPESubComponent::handleNeuronFire_()` 替换为 `spike_comm_.emitNeuronFire()`。

Phase D（通信子系统闭环化）：
1. 路由构建/共享缓存/门控缓存迁入 `services/SpikeCommSubsystem`；
2. 控制层仅负责配置与生命周期编排，不再持有路由表与缓存。

Phase B‑2（可选增强，不影响语义）：
- 支持批量发送：`emitBatch(const std::vector<...>&)`，减少 `new SpikeEvent` 次数（仅性能优化）。
- 将 fanout provider 抽象为 `IFanoutProvider`，以便未来不依赖 `SnnRouteProvider`。

## 10. 回归验证口径

构建安装：
```bash
cd sst_workspace/sst-elements/src/sst/elements/SnnDL
make -j4 && make install
```

回归实验（以 mesh template 口径为准，详见 `sst_dram_si/docs/mesh_template_guide_20251122-000250.md`）：
```bash
MESH_SIM_TIME=10us sst_dram_si/tools/run_mesh_with_time.sh
MESH_SIM_TIME=100us sst_dram_si/tools/run_mesh_with_time.sh
```

通过标准：
- 仿真可跑完，无 fatal/断言。
- 关键输出/统计口径与基线一致（允许非语义性日志差异）。
