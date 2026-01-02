# api/（稳定接口层）

本目录存放 **SnnDL 内部各层共享的“稳定接口”与最小抽象**，用于降低编译耦合并支撑后续替换不同计算核心（compute core）。

## 职责

- 定义 **SST 组件/子组件之间交互的抽象接口**（不包含具体实现与算法细节）。
- 提供对外/跨层可复用的最小类型集合，避免控制层、计算层、组件层互相直接包含大量实现头文件。

## 主要内容

- `SnnCoreAPI.h`
  - `SnnPESubComponent` 继承的 SubComponent API 基类。
  - 该 API 面向 **MultiCorePE 控制/挂接子核心** 的调用面：`deliverSpike()`、`hasWork()`、`getUtilization()`、`getStatistics()` 等。
- `SnnPEParentInterface.h`
  - 子核心（SnnPESubComponent）与父组件（MultiCorePE）之间的调用接口（发 spike、统计汇聚等）。
- `SnnInterface.h`
  - NIC/PE 等更高层组件使用的接口类型（例如发送/接收事件的抽象入口）。
- `INocTransport.h`
  - NoC 抽象接口：只做 send/recv/forward/本地注入的传输语义，不包含 fanout/权重/路由构建。
  - 典型实现：`services/noc/NocSubsystem`。
- `NocSpikeTransport.h`
  - `ISpikeTransport` 的 NoC 适配器：将“Spike 发送”映射为 `INocTransport::sendFromCore(src_core, packet_event)`（NoC 不解析 Spike 语义）。
- `IMemoryAccess.h`
  - 纯内存访问接口：只做 `addr + size ↔ bytes`，不携带任何 `weight/synapse/bcsr/route` 语义字段。
  - 典型实现：`services/memory/StandardMemAccess`。
- `ISpikeTransport.h`
  - Spike 传输抽象：上层（如 `services/synapse/route/SpikeCommSubsystem`）通过该接口把“要发送的 Spike payload”交给 NoC/本地投递后端。
- `SnnWeightReader.h`
  - `IWeightReader` 抽象：为 compute core 提供统一的权重读取/缓存接口（实现通常在控制层/服务层）。
- `ISynapseRoute.h` / `SynapseRouteBuildConfig.h`
  - Synapse/Route 的对外窄接口：fanout/route 构建与查询（路由语义集中在 synapse 域，不进入 NoC/Memory）。
- `IGlobalStepHooks.h`
  - 全局 Step/GAS 同步钩子：用于 mesh 维度的 step barrier 协调（控制层/组件层调用面收敛于此）。
- `IGasCmdSender.h` / `IGasStageSink.h` / `IGasStepGate.h` / `GasOps.h`
  - GAS/窗口阶段编排的窄接口：用于把“阶段事件发送/接收、gate 控制、统计汇聚”与具体实现解耦。

## 依赖边界（建议）

- 本目录 **不应依赖** `control/`、`components/`、`services/` 的实现细节。
- 允许依赖 `SST core` 的基础头（`sst/core/*`）与 C++ 标准库。
- 上层依赖关系建议：`components/`、`control/`、`compute/`、`services/` 都可以依赖 `api/`，但尽量避免反向依赖。

## 扩展指南

- 新增接口时优先考虑：
  - **最小可用**（KISS）：只暴露跨层必须信息；
  - **可替换**（SOLID-D）：依赖抽象，不依赖具体类；
  - **向后兼容**：避免破坏现有仿真脚本与组件装配方式。

## 稳定性约定（建议）

- `api/` 中的头文件优先视为“跨层稳定契约”，接口变更应遵循：
  - 先新增（保持兼容）→ 回归验证 → 再逐步移除旧接口（若必须，写清迁移路径与回退策略）。
- 若某能力为“可选扩展”（例如 compute 的权重语义），应放到对应域的可选扩展接口中，避免绑架主接口。
