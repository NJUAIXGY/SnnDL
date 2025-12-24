# services/memory/（纯内存访问子系统）

本目录存放 **Memory 域（纯地址/字节语义）** 的实现，目标是把 StandardMem 的请求发起、pending 跟踪、回包分发收敛为可复用模块，并避免在内存层引入任何 “突触/权重/路由” 语义。

> 边界原则：Memory 只处理 “`addr + size` ↔ `bytes`”，不解释 payload 不做 BCSR 解析。

---

## 目录结构与组件职责

### `StandardMemAccess.{h,cc}`（推荐主入口）
- **定位**：`api/IMemoryAccess.h` 的 StandardMem 实现，负责：
  - 构造 `StandardMem::Read/Write` 请求并 `send()`；
  - 以 `RequestId` 为 key 维护 pending；
  - 在收到 `ReadResp` 时做 **尺寸断言式检查**（`resp_bytes < req_bytes` 直接 `fatal`），再回调上层。
- **关键 API**：
  - `read(addr, bytes, cb)`：cb 形态为 `cb(req_id, addr, data_bytes)`；
  - `write(addr, data, cb)`：cb 形态为 `cb(req_id, addr)`；
  - `handleMemoryResponse(req*)`：用于把 StandardMem 回包分发到对应回调（并释放请求对象）。
- **注意事项**：
  - `ReadResp` 的 `data.size()` 必须覆盖请求的 `bytes`，否则会触发 `[stdmem-access-assert]` 直接 fail-fast；这用于尽早暴露地址映射/对齐/截断问题。

### `StandardMemBackend.{h,cc}`（过渡/工具后端）
- **定位**：更底层的 StandardMem pending 后端：仅提供 `sendRead/sendWrite` 与 `popPending`。
- **说明**：
  - 当前的 `MemRequestMeta` 中仍包含一些与 BCSR/权重诊断耦合的字段（历史包袱/过渡形态）；
  - 长期目标是让该结构退化为“纯内存元信息”（addr/size/对齐/issue_cycle），把权重语义移动到 `services/synapse/weights` 或 `services/synapse/route` 的私有结构里。

---

## 与其他域的交互（典型用法）

- `control/SnnPESubComponent` 在 `setup/init` 时装配：
  - 创建 `StandardMemAccess`（对外提供纯 `IMemoryAccess`）；
  - 仍可保留 `StandardMemBackend` 用于历史路径或更细粒度的 meta 跟踪（逐步收敛中）。
- `services/synapse/weights/WeightMemorySubsystem` 推荐只依赖 `api/IMemoryAccess`：
  - Memory 负责返回 bytes；
  - Weights 负责解释 bytes（例如 float 解码、rowptr/colidx/blockdata 解析）。

---

## 约束与建议

- **禁止**：在 `services/memory` 里出现 `weight/synapse/bcsr/route` 的业务语义。
- **建议**：任何回包解析（float/idx 解码）放到 `services/synapse/weights` 的语义层；Memory 只保证“正确的字节块”。
