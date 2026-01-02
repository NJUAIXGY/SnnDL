# components/gas/（全局 Step/GAS 同步组件）

本目录存放 **Mesh 级别的 Step/GAS 同步控制组件**，用于把“每个 PE 的 Step 开始/结束”收敛为一个显式的全局 barrier，从而避免各 PE 因窗口节奏抖动导致 Step 触发次数不一致。

> 边界原则：该目录只放 **SST 可加载组件（ELI 注册对象）**；它只做“同步控制面”，不包含任何神经动力学/权重/路由/NoC 事务逻辑。

---

## 主要组件

### `GlobalGasStepController.{h,cc}`
- **类型**：`SST::Component`（ELI：`SnnDL.GlobalGasStepController`）
- **职责**：全局 Step barrier 的控制面状态机
  - 接收来自各 PE 的 barrier 事件：`PeReady` / `PeDone(seq)`（事件类型：`events/GasStepBarrierEvent.h`）
  - 广播 `StartStep(seq)` 到所有已连接的 PE（端口：`pe_link0..pe_linkN`）
  - 语义冻结：
    - 当 `require_all_ready=1`：等待所有 PE 都上报 `PeReady` 后才进入第一个 step
    - 每个 `seq`：等待所有 PE 都上报 `PeDone(seq)` 后才进入 `seq+1`
    - 当 `strict_seq_check=1`：发现乱序/重复/越界的 `seq` 直接 `fatal`（用于尽早暴露同步错误）

---

## 参数与端口（与 ELI 文档一致）

- **参数**：
  - `clock`：组件时钟（默认 `1GHz`）
  - `start_seq`：起始 step 序号（默认 `1`）
  - `require_all_ready`：是否要求所有 PE 先 ready（默认 `1`）
  - `strict_seq_check`：是否对 seq 严格检查（默认 `1`）
  - `verbose`：日志等级（默认 `0`）
- **端口**：
  - `pe_link%(pe)d`：与各 PE 相连的 barrier 链路（event 类型：`SnnDL.GasStepBarrierEvent`）

---

## 依赖与交互

- **依赖**：`events/GasStepBarrierEvent.h`（barrier 事件载体）
- **典型装配**：由 mesh 模板在系统层创建一个 controller，并把每个 PE 的 `pe_linkX` 接到 controller 的 `pe_linkX`。

