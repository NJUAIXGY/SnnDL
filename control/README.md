# control/（控制/编排层：通用 Core 壳）

本目录存放 **SnnPESubComponent 及其拆分后的控制面实现**。它的定位是“通用控制壳（control plane）”，面向可替换的 compute core 提供统一的时序、内存与路由编排能力。

## 职责

- SST 生命周期对接：`init/setup/finish`、clock 驱动。
- GAS（Gather/Apply/Scatter）窗口状态机与窗口边界事件上报。
- 内存事务编排（通过 `api/IMemoryAccess` 与 `services/memory/StandardMemAccess`）、缓存命中统计、请求合并策略等控制面逻辑。
- 路由与 fanout：从 compute core 拉取发放事件，交由 `services/synapse/route/SpikeCommSubsystem` 完成 fanout/封包/发送（控制层不持有路由表/缓存）。
- 统计汇总：对上层（MultiCorePE）汇报窗口/内存/发放等统计。

> 重要：本层 **不应包含具体 SNN 动力学**（膜电位、不应期、阈值判定等），这些应全部由 `compute/` 中的 `ISnnComputeCore` 实现。

## 主要内容

- `SnnPESubComponent.{h,cc}`
  - 子核心控制壳本体：持有 `std::unique_ptr<ISnnComputeCore>`，并通过接口喂入输入/时间、拉取输出事件。
  - `compute_core_impl` 参数控制 core 选择（默认 `default`）。
- `SnnPESubComponent_impl.h`（internal）
  - Phase5 边界硬化用的 PImpl 内部状态：阶段事件（Begin*/EndScatter）、统计汇报、`gas_ctrl_` 等重实现对象统一收进 `Impl`。
  - 仅允许被 `.cc` include；禁止在对外头文件传播该依赖。
- 拆分实现文件（按功能拆分，便于继续瘦身）：
  - `SnnPESubComponent_spike.cc`：输入侧 spike 递送与本地处理路径。
  - `services/synapse/stdmem/SnnPESubComponent_mem.cc`：StandardMem glue（与 `StdMemEndpoint`/`StandardMemAccess` 对接）。为满足“control/ 不出现 StandardMem::”约束，该文件刻意迁出 `control/`。
  - `SnnPESubComponent_bcsr.cc`：BCSR 相关控制面与诊断（不含动力学）。
  - `SnnPESubComponent_routing.cc`：历史路由实现封存（已迁入 `services/synapse/route/SpikeCommSubsystem`）；当前仅保留模板解析与门控转发入口。
  - `SnnPESubComponent_scheme1.cc`：scheme1（slice 顺序执行）控制路径。
- `SnnPEApplyScatter.cc`：窗口累加器 Apply/Scatter 的控制面执行与输出收敛点。
- `SnnPEOrchestrators.cc`：控制层小型 orchestrator/辅助结构（含窗口读发起编排：`issueFromEdges_/issueFromSets_/issueFallbackReadsIfNeeded_`）；同时承载 `Impl::report*` 统计汇报实现（Phase5.4）。

## 依赖边界（建议）

- 允许依赖：`api/`、`events/`、`services/`、`compute/`。
- 避免：在控制层新增任何“神经元状态存储/扫描发放”等逻辑；新增 compute 范式时应通过 `ISnnComputeCore` 扩展或替换实现。

## 扩展指南（引入新 compute core）

1. 在 `compute/` 实现新的 `ISnnComputeCore`；
2. 在 `compute/createComputeCoreByName()` 注册名称；
3. 在脚本中为 `SnnPESubComponent` 设置 `compute_core_impl=<name>`。
