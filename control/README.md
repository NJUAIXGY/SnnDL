# control/（控制/编排层：通用 Core 壳）

本目录存放 **SnnPESubComponent 及其拆分后的控制面实现**。它的定位是“通用控制壳（control plane）”，面向可替换的 compute core 提供统一的时序、内存与路由编排能力。

## 职责

- SST 生命周期对接：`init/setup/finish`、clock 驱动。
- GAS（Gather/Apply/Scatter）窗口状态机与窗口边界事件上报。
- 内存读写请求编排（StandardMem）、缓存命中统计、请求合并策略等控制面逻辑。
- 路由与 fanout：从 compute core 拉取发放事件，再调用路由提供器发送 SpikeEvent。
- 统计汇总：对上层（MultiCorePE）汇报窗口/内存/发放等统计。

> 重要：本层 **不应包含具体 SNN 动力学**（膜电位、不应期、阈值判定等），这些应全部由 `compute/` 中的 `ISnnComputeCore` 实现。

## 主要内容

- `SnnPESubComponent.{h,cc}`
  - 子核心控制壳本体：持有 `std::unique_ptr<ISnnComputeCore>`，并通过接口喂入输入/时间、拉取输出事件。
  - `compute_core_impl` 参数控制 core 选择（默认 `default`）。
- 拆分实现文件（按功能拆分，便于继续瘦身）：
  - `SnnPESubComponent_spike.cc`：输入侧 spike 递送与本地处理路径。
  - `SnnPESubComponent_mem.cc`：StandardMem 读写/回调/写回等内存控制面。
  - `SnnPESubComponent_bcsr.cc`：BCSR 相关控制面与诊断（不含动力学）。
  - `SnnPESubComponent_routing.cc`：权重驱动/CSV 驱动路由构建与共享缓存。
  - `SnnPESubComponent_scheme1.cc`：scheme1（slice 顺序执行）控制路径。
- `SnnPEApplyScatter.cc`：窗口累加器 Apply/Scatter 的控制面执行与输出收敛点。
- `SnnPEOrchestrators.cc`：控制层小型 orchestrator/统计汇聚辅助结构（含窗口读发起编排：`issueFromEdges_/issueFromSets_/issueFallbackReadsIfNeeded_`）。
- `StageEventHub.{h,cc}`：阶段事件（BeginGather/BeginApply/...）聚合与写出控制。

## 依赖边界（建议）

- 允许依赖：`api/`、`events/`、`services/`、`compute/`。
- 避免：在控制层新增任何“神经元状态存储/扫描发放”等逻辑；新增 compute 范式时应通过 `ISnnComputeCore` 扩展或替换实现。

## 扩展指南（引入新 compute core）

1. 在 `compute/` 实现新的 `ISnnComputeCore`；
2. 在 `compute/createComputeCoreByName()` 注册名称；
3. 在脚本中为 `SnnPESubComponent` 设置 `compute_core_impl=<name>`。
