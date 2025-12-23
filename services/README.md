# services/（可复用服务/子系统）

本目录存放 **控制层与 compute core 之间共享的“服务模块”**：权重访问、缓存策略、GAS 辅助、路由提供器、内部 ring 等。

这些模块的目标是把 `SnnPESubComponent` 从“巨型类”中瘦身出来，并为后续更强的接口解耦奠定基础。

## 主要内容

- 权重/缓存相关：
  - `WeightAccessor.{h,cc}`：权重地址解析、cache key 计算、索引模式处理等。
  - `WeightCacheOps.{h,cc}`：**自带状态** 的 cache 服务（LRU/clock），通过 `configure()` 注入容量/模式/回调，避免窥探控制层私有容器。
  - `SnnBcsrWeightManager.{h,cc}`：BCSR 元数据、rowptr/colidx/block 缓存管理。
  - `StandardMemWeightReader.{h,cc}`：StandardMem 权重读写集中器（**当前未纳入构建**；启用前需先把对控制层私有成员的访问改为接口边界）。
- GAS/窗口相关：
  - `GasPhaseController.{h,cc}`：GAS 阶段状态机控制（BeginGather/BeginApply/...）。
  - `GasEdgeCollector.{h,cc}`：窗口内边集合记录与翻转（Gather→Apply）。
  - `AccumulatorOps.{h,cc}`：dense/map 累加器与验证/统计采集。
  - `ReadOrchestrator.{h,cc}`：历史遗留：窗口读发起编排器（已回收至 `control/SnnPEOrchestrators.cc`，当前不再参与构建；文件保留用于参考）。
  - `GasCustomCmd.h`：GAS 相关 CustomCmd/响应定义。
- 路由/网络辅助：
  - `SpikeCommSubsystem.{h,cc}`：通信子系统（fanout + SpikeEvent 构造 + `api/ISpikeTransport` 发送），并内聚路由构建/共享缓存与门控缓存。
  - `SnnRouteProvider.{h,cc}`：fanout 计算与门控缓存应用（供 `SpikeCommSubsystem` 组合使用；历史上也可由控制层直接调用）。
  - `SimpleNetworkWrapper.{h,cc}`：SimpleNetwork 适配包装。
  - `SnnNetworkAdapter.{h,cc}`：网络侧辅助适配。
- 内部互连：
  - `OptimizedInternalRing.{h,cc}`：MultiCorePE 内部 ring 通信与优化实现。
- 诊断：
  - `SnnProfiler.h`：轻量 profiling（条件编译）。

## 依赖边界（建议）

- `services/` 允许依赖 `api/`、`events/`、标准库与少量 SST 基础类型。
- 长期目标：逐步减少对 `control/SnnPESubComponent` 私有成员的直接访问，改为依赖更小的接口（利于替换控制层实现）。
  - 当前已实现：`WeightAccessor/WeightCacheOps` 不再依赖控制层私有成员；窗口读发起逻辑已回收至 control 层（避免 services→control 的 friend 访问）。

## 扩展指南

- 新增服务模块时优先采用“面向接口”的方式：
  - 参数/回调注入（而非 friend 访问）；
  - 明确输入输出与线程安全边界；
  - 避免把 compute core 的动力学逻辑放到 services（应留在 `compute/`）。
