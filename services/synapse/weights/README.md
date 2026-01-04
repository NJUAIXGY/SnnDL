# services/synapse/weights/（权重语义与缓存子系统）

本目录存放 **Weights 域** 的实现：权重地址解析、BCSR 元数据与缓存、窗口读编排（budget/outstanding）、以及面向 compute core 的统一 `IWeightReader` 入口。

> 边界原则：Weights 域可以理解 “权重/BCSR/缓存/ΔV 更新” 语义，但不负责 NoC 发送，不构建 fanout 路由表。

---

## 目录结构与组件职责

### `WeightMemorySubsystem.{h,cc}`
- **定位**：权重读取与窗口读编排子系统；实现 `api/SnnWeightReader.h` 的 `IWeightReader`。
- **核心职责**：
  - 为 compute core 提供统一入口：`requestDense()` / `requestBCSR()`；
  - 承载 window-read 的集合/预算/并发（issued/outstanding）与发起编排；
  - 维护窗口触达集合（posts/pres）与边集合（通过 `gas/GasEdgeCollector`）；
  - 提供 BCSR 读路径的安全护栏（例如 abs-max guard、rowptr-ready gate、fallback 策略等）。
- **关键装配面**：
  - `bindMemory(IMemoryAccess*)`：绑定纯内存访问（地址→字节）；
  - `configureOrchestrator(OrchestratorConfig)`：由 `workload=snn` 注入“发起读/写、cache、累加、诊断、统计”等回调与配置。
- **典型调用节奏**：
  - BeginGather：`beginGatherWindow(...)` 迁移/清空窗口集合；
  - BeginApply：`beginApplyWindow(...)` + `flipEdgesForApply(...)`，然后 `issueFromEdges()`；
  - ReadResp：由 `workload=snn` 把 bytes 交给 Weights 语义层解析并回调 `acc_update` 等。

### `SnnBcsrWeightManager.{h,cc}`
- **定位**：BCSR 元数据与缓存管理器（`BcsrWeightManager`）。
- **核心职责**：
  - 维护 BCSR 布局参数（block 行列、idx/val 字节数、rowptr/colidx/blockdata/blockids 基址）；
  - 管理 row-index cache 与 block cache（容量可配）；
  - 提供寻址函数：`blockDataAddr()` / `colIndexAddr()` / `rowBounds()` 等；
  - 支持将 rowptr bytes 安装到 host（`installRowptrFromBytes`）并维护 rowptr-ready 状态。

### `WeightAccessor.{h,cc}`
- **定位**：权重寻址/索引解析工具（从控制层抽出的纯 helper）。
- **核心职责**：
  - 将 `(pre_global, post_local)` 映射为请求索引与 cache key；
  - 保持历史 “allow_remap” 的 modulo 兼容语义（用于 pre 不在本 PE 时的保守映射）。

### `WeightCacheOps.{h,cc}`
- **定位**：权重缓存操作封装（自带状态），用于替代控制层散落的 LRU/clock 容器与淘汰逻辑。
- **核心能力**：
  - `configure()` 注入容量、模式（LRU/clock）与淘汰回调；
  - `tryGet()/store()` 提供统一的 cache hit/miss 行为；
  - 支持 `disable_cache` 强制 miss（诊断用）。

---

## 与其他域的交互

- **Memory 域**：
  - Weights 通过 `api/IMemoryAccess` 发起读写；
  - Memory 回包 bytes 由 Weights 解释为 float/idx/rowptr/blockdata 等语义。
- **GAS 域**：
  - `GasEdgeCollector` 负责 per-window edge 集合的 flip/迭代；
  - Weights 在 Apply 阶段基于 edge 集合批量发起读并更新累加器。
- **Compute 域**：
  - Compute core 只依赖 `IWeightReader`；不直接触碰 StandardMem 与 BCSR 缓存细节。
- **Route 域**：
  - Weights 不构建路由表；路由与 fanout 由 `services/synapse/route` 负责。

---

## 约束与建议

- **禁止**：Weights 域内直接发送 NoC 消息或操作 NIC（应由 `services/noc` 与 `services/synapse/route` 负责）。
- **建议**：对外只暴露 `IWeightReader` 等窄接口；其余复杂编排通过 `configureOrchestrator()` 注入，避免 services→control 反向依赖。
