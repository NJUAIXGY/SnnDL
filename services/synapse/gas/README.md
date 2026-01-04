# services/synapse/gas/（GAS 窗口与累加辅助子系统）

本目录存放 **GAS（Gather / Apply / Scatter）窗口相关的辅助模块**，用于把窗口边集合、累加器实现、以及与 GatherBufferIF 交互的 CustomCmd/统计数据结构从控制层抽离出来。

> 边界原则：GAS 域负责“窗口与累加辅助”，不负责神经动力学（compute），也不负责 NoC/路由表构建。

---

## 目录结构与组件职责

### `GasCustomCmd.h`
- **定位**：GAS 控制面 CustomCmd 定义（承载在 `StandardMem::CustomReq/Resp` 的 payload 中）。
- **主要内容**：
  - `GasOp / GasOpData`：BeginGather/BeginApply/... 等阶段 opcodes；
  - `GasStatData`：GatherBufferIF → PE 的统计载体（unique reads/bytes、row-window merge、burst/payload、inflight peak、buffer peak 等）。

### `GasEdgeCollector.{h,cc}`
- **定位**：per-window 的 edge 采集与翻转迭代器。
- **核心职责**：
  - `recordEdge(post_local, pre_global)` 记录窗口内边（计数）；
  - `flipForApply(...)`：BeginApply 前将 curr → prev（并准备迭代）；
  - `nextPrev(...)`：Apply 阶段逐条迭代 prev 边集合。

### `AccumulatorOps.{h,cc}`
- **定位**：窗口累加器实现（dense/sparse + spill + 可选 shadow verify）。
- **核心职责**：
  - `update(post, dv)`：累加 ΔV；
  - `collectSortedPairs(...)`：以确定性顺序收集更新对（由控制层决定何时 apply/scatter）；
  - `verifyDense(seq)`：可选影子验证（诊断/一致性检查）。
- **特性**：累加器状态自持（避免控制层持有多个容器导致职责混杂）。

### `GasPhaseController.{h,cc}`
- **定位**：阶段事件镜像/轻量编排器（与 `api/IGasOrchestrator` 对接）。
- **作用**：
  - 记录/镜像 BeginGather/BeginApply/... 等阶段事件日志；
  - 将 BeginApply/BeginScatter/EndScatter 这类“序列动作”委托给 orchestrator。

---

## 与其他域的交互

- `components/GatherBufferIF` 使用 `GasCustomCmd` 发送阶段事件/统计到 PE；
- `workload=snn` 使用 `GasEdgeCollector` 与 `AccumulatorOps` 承载窗口边与累加状态（CoreShell 不应持有业务状态机）；
- `services/synapse/weights/WeightMemorySubsystem` 读取 `GasEdgeCollector` 的 prev 边集合，在 Apply 阶段发起窗口读并更新累加器。
