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
- `SnnWeightReader.h`
  - `IWeightReader` 抽象：为 compute core 提供统一的权重读取/缓存接口（实现通常在控制层/服务层）。

## 依赖边界（建议）

- 本目录 **不应依赖** `control/`、`components/`、`services/` 的实现细节。
- 允许依赖 `SST core` 的基础头（`sst/core/*`）与 C++ 标准库。
- 上层依赖关系建议：`components/`、`control/`、`compute/`、`services/` 都可以依赖 `api/`，但尽量避免反向依赖。

## 扩展指南

- 新增接口时优先考虑：
  - **最小可用**（KISS）：只暴露跨层必须信息；
  - **可替换**（SOLID-D）：依赖抽象，不依赖具体类；
  - **向后兼容**：避免破坏现有仿真脚本与组件装配方式。

