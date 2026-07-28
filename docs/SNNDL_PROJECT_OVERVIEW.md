# SnnDL Project Overview

SnnDL 是 SST 元素库中的模块化神经形态与通用 workload 仿真组件。公开兼容面由 ELI 注册名、参数名、端口、统计键和 `api/` 契约组成；目录重构不得改变这些名称。

## 架构分层

```text
SST configuration
      |
components/  ---- ELI registration and assembly
      |
platform/    ---- core, memory, NoC, statistics
      |
workloads/   ---- executable workload state machines
      |
snn/         ---- SNN-only compute, synapse, stimulus
      |
research/    ---- opt-in experimental mechanisms
```

`api/` 和 `events/` 横跨各层提供窄接口与传输载体。依赖应总体向这两个稳定域收敛，避免平台反向依赖 workload 或研究实现。

## 组件面

当前库注册 11 个 Component、7 个 SubComponent 和 2 个 SubComponent API。主要装配入口是 `components/MultiCorePE`，每个 PE 创建 CoreShell、内存与网络资源，再由 `workload_impl` 选择运行逻辑。

`components/multicore/MultiCorePEObservability.cc` 只负责统计注册、采样和 finish 输出；核心装配仍在 `MultiCorePE.cc`。不要在 observability 文件中加入调度决策。

## 平台与 Workload

`platform/core/SnnPESubComponent` 是通用 CoreShell。它绑定 `ICoreWorkload`、转发时钟和 packet，并汇聚统计。其 finish/stat 路径位于 `SnnPESubComponentStats.cc`。

`workloads/CoreWorkloadFactory.cc` 创建 `snn`、`riscv_snn`、`stream`、`traffic`、`traffic_mem` 或 `tensor`。非 SNN workload 不依赖 `snn/compute`。

Tensor 配置、program 执行和统计导出分别位于独立翻译单元；主文件保留内存、tile、collective 和通用 tick 状态机。

## SNN 与研究机制

`snn/compute` 处理神经动力学，`snn/synapse` 处理权重与路由，`snn/stimulus` 处理输入。它们通过 `IMemoryAccess` 和 `INocTransport` 使用平台能力。

`research/` 中的 GAS、local-storage、3D NoC、PE fabric 和 3D route 机制必须由参数显式启用。研究开关默认值和统计键仍需保持可复现。

## 构建与验证

从 SnnDL 目录运行：

```bash
make -j4
make test-compile
make test-riscv-snn-protocols
```

修改 `Makefile.am` 后，从 `sst_workspace/sst-elements/` 运行 Automake 和 `config.status` 重新生成构建文件。提交前确认没有旧目录 include、没有编译产物，并检查 `git diff --check`。
