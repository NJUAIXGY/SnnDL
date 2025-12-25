# services/legacy/（历史遗留/参考实现）

本目录存放 **不参与当前主链路（默认不纳入构建）的 legacy/参考实现**。它们用于：
- 对照历史行为；
- 保留曾经的实现思路与调试工具；
- 为后续重构提供回归参考（但不应被新功能继续依赖）。

> 使用原则：如需启用 legacy 文件，应先确认其依赖边界（不得反向依赖 control 私有成员），并在启用前完成 100us 回归验证。

---

## 目录结构与组件职责

### `ReadOrchestrator.{h,cc}`
- **定位**：历史窗口读发起编排器（window-read issue logic）。
- **现状**：已被主路径回收/替代（主路径由 `services/synapse/weights/WeightMemorySubsystem` 闭环承载，并在控制层 orchestrator 中完成窗口读编排）。
- **保留原因**：作为“issueFromEdges/issueFromSets/fallback” 逻辑的参考实现与对照。

### `StandardMemWeightReader.{h,cc}`
- **定位**：历史的 StandardMem 权重读取 shim（实现 `IWeightReader`）。
- **现状**：主路径已由 `services/memory/StandardMemAccess`（纯内存） + `services/synapse/weights/WeightMemorySubsystem`（权重语义）闭环承载。
- **保留原因**：兼容旧接口/旧调试口径的参考。

### `memory/StandardMemBackend.{h,cc}`
- **定位**：历史的 StandardMem pending 跟踪后端（`sendRead/sendWrite + popPending`）。
- **现状**：该实现携带 `MemRequestMeta`（含 BCSR/权重语义字段），已从 `services/memory/` 移出；主链路禁止依赖。
- **保留原因**：用于对照旧 pending/回包分发逻辑与历史调试口径。

### `noc/SnnNetworkAdapter.{h,cc}` / `noc/SimpleNetworkWrapper.{h,cc}`
- **定位**：历史遗留的拓扑适配器/包装器（含 SST ELI 宏）。
- **现状**：ELI 可加载对象的“权威位置”统一收敛在 `components/noc/`；本目录下文件仅保留作参考对照，主链路禁止依赖。
