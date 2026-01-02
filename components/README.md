# components/（SST 组件集成层）

本目录存放 **SST 可加载的组件/子组件实现**（ELI 注册对象），它们负责与 SST 框架对接：端口、Link、Clock、Stat、init/setup/finish 生命周期等。

## 职责

- 提供完整可运行的 SST 元素（Components/SubComponents），并将下层模块装配成系统。
- 管理网络互连、内存前端、权重加载、门控组件等系统级职责。

## 主要内容

- `MultiCorePE.{h,cc}`
  - 处理单元（PE）顶层组件：挂接多个 `SnnCoreAPI` 子核心（通常是 `control/SnnPESubComponent`）。
  - 负责统计汇聚、阶段事件聚合、内部 ring 互连等。
- `SnnNIC.{h,cc}`
  - NIC 组件：对接 `SimpleNetwork`（如 merlin.linkcontrol），发送/接收 Spike 与门控事件。
- `WeightLoader.{h,cc}`
  - 权重加载组件：通过内存接口将权重预置到 DRAM/缓存可见区域。
- `GatherBufferIF.{h,cc}`
  - GAS window 驱动的 StandardMem 前端（用于 Gather/Apply/Scatter 的窗口化时序）。
- `GatingPE.{h,cc}`
  - 门控组件：生成/传播 `GatingDecisionEvent`，控制 fanout 目的集合。
- `stimulus/SpikeSource.{h,cc}`
  - 可选 spike 源（在部分实验脚本中可能禁用）。
- `MemKCalBench.{h,cc}`
  - micro-benchmark：用于 K 校准或 memory 访问特征测试。
- `SnnPE.{h,cc}`
  - 旧架构兼容组件（deprecated/compat），新功能优先在 MultiCorePE 体系演进。

## 子目录

- `components/stimulus/`：Stimulus 域的注入型组件（例如 `SpikeSource`）。
- `components/mpi/`：MPI 扩展相关组件与类型（可选编译）。
- `components/noc/`：ELI 可加载的 NoC/拓扑适配组件（高级/实验性用途，默认推荐 `SnnNIC`）。
- `components/gas/`：全局 Step/GAS 同步组件（Mesh barrier 控制面）。

## 依赖边界（建议）

- `components/` 可以依赖 `api/`、`events/`、`control/`、`services/`、`compute/`。
- 避免把动力学/算法细节写进组件层：应下沉到 `compute/`。

## 扩展指南

- 新增 SST 组件时：
  - 保持 ELI 注册信息（库名/组件名/参数文档）与现有脚本兼容；
  - 业务逻辑尽量委托给 `control/` 或 `compute/`，组件层只做装配与资源管理。
