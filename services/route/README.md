# services/route/（Synapse/Route 路由与通信事务子系统）

本目录存放 **Synapse/Route 域** 的实现：路由构建（权重驱动或映射文件驱动）、fanout 查询、gating 决策缓存，以及 spike 事件构造与发送事务的封装。

> 边界原则：Route 域负责 “从 source neuron 到目的集合的选择与发送事务”，但不负责“网络传输细节”（NoC）与“权重字节解析”（Weights）。

---

## 目录结构与组件职责

### `SynapseRouteSubsystem.{h,cc}`
- **定位**：路由构建与共享缓存子系统；实现 `api/ISynapseRoute.h`。
- **核心职责**：
  - 基于 `api/SynapseRouteBuildConfig.h` 构建 `RouteMap`（支持权重驱动/映射文件/层间 mask 等）；
  - 维护进程级共享缓存（避免每个 core 复制 route table）；
  - 承载 gating（`applyGatingDecision`）与 TTL 失效逻辑；
  - 对外提供 `computeFanout(...)`，内部复用 `SnnRouteProvider` 完成扇出列表生成。
- **输出**：
  - `routesShared()`：共享只读路由表；
  - `routesLocalFallback()`：本地 fallback（无共享或构建失败时保守使用）。

### `SnnRouteProvider.{h,cc}`
- **定位**：fanout provider（扇出列表生成器）。
- **核心职责**：
  - 给定 route table + gating cache，计算每个 source 的 fanout 列表；
  - 以 `ISynapseRoute::FanoutEntry` 形式输出（包含目的 PE 等信息）。

### `SpikeCommSubsystem.{h,cc}`
- **定位**：通信事务封装：fanout + SpikeEvent 构造 + 传输调用。
- **核心职责**：
  - 绑定运行时：`ISpikeTransport* transport` 与 `ISynapseRoute* synapse_route`；
  - `emitNeuronFire()`：compute core 报告 neuron_idx 后触发 fanout，并构造 `SpikeEvent` 逐条发送；
  - `applyGatingDecision()`：将 gating 决策转发给 Synapse/Route 子系统（缓存并生效）。
- **说明**：
  - `SpikeCommSubsystem` 通过 `api/ISpikeTransport` 发出事件；常见情况下该 transport 是 `api/NocSpikeTransport`（把 send 映射为 `INocTransport::sendFromCore`）。

---

## 与其他域的交互（典型数据流）

1) `compute/ISnnComputeCore` 产出 fire events；
2) `control/SnnPESubComponent` 调用 `SpikeCommSubsystem::emitNeuronFire()`；
3) `SpikeCommSubsystem` 调用 `ISynapseRoute::computeFanout()` 得到目的集合；
4) `SpikeCommSubsystem` 构造 `events/SpikeEvent` 并调用 `ISpikeTransport::send(...)`；
5) `ISpikeTransport` 的具体实现（通常由 NoC 域承载）完成实际发送与本地投递。

---

## 约束与建议

- **禁止**：Route 域内直接依赖 `StandardMem` 或解析权重 bytes（属于 Weights/Mem 域职责）。
- **建议**：路由构建成本高时优先复用 `SynapseRouteSubsystem` 的进程级缓存（避免每核重复构建）。

