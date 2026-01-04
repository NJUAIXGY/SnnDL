# services/synapse/stdmem/（StandardMem 胶水层：隔离 StandardMem 类型）

本目录存放 **StandardMem 相关的“胶水代码”**，目标是把 `StandardMem::*` 类型与 include 隔离在一个明确的边界内，使得：

- `control/*` 的 `.h/.cc` **完全不出现** `StandardMem::`（包括 include `stdMem.h`）
- 纯内存访问走 `api/IMemoryAccess.h`（实现：`services/memory/StandardMemAccess`）
- GAS 控制面的 stage/stat 事件走 `api/IGasStageSink.h` / `api/IGasCmdSender.h`

---

## 主要内容

### `StdMemEndpoint.{h,cc}`
- **定位**：StandardMem 端点封装（synapse 域胶水），用于装配 StandardMem 子组件并分发回包
- **关键点**：
  - 头文件 **不包含** `stdMem.h`，通过 `handleResponseOpaque(void* req)` 隔离 StandardMem 请求类型
  - 同时对接两类回包：
    1) **数据面**：分发到 `services/memory/StandardMemAccess`（纯 `addr→bytes`）
    2) **控制面**：分发到 `api/IGasStageSink`（BeginGather/BeginApply/... 的 stage/stat 载体）
  - 实现 `api/IGasCmdSender`：用于向 GatherBufferIF 发送 GAS stage custom cmd
  - **fail-fast**：遇到 `stdmem-untracked`（回包无法匹配任何 pending）直接 `fatal`，用于尽早暴露重复回包/错配等严重问题。

### `SnnPESubComponent_mem.cc`
- **定位**：`control/SnnPESubComponent` 的 StandardMem glue 实现文件（刻意放在本目录）
- **原因**：该文件需要 include `stdMem.h` 并与 StandardMem 子组件交互；为满足“control/ 不出现 StandardMem::”边界约束，必须迁出 `control/`。

---

## 依赖与交互

- 本目录可以依赖 StandardMem（`sst/core/interfaces/stdMem.h` 等），但应避免把 StandardMem 类型向外泄露到 `control/` 或 `api/`。
- 对外暴露的交互应收敛到：
  - `api/IMemoryAccess.h`（数据面）
  - `api/IGasStageSink.h` / `api/IGasCmdSender.h`（控制面）
