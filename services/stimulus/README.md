# services/stimulus/（Stimulus 刺激/注入子系统）

本目录存放 **Stimulus 域** 的实现：提供独立的“注入时基（timebase）+ 选源 + 注入事务”，并通过 NoC/Route 等子系统完成投递与外发。

> 边界原则：Stimulus 负责 *何时注入/注入哪些源*，但不负责 NoC 传输细节，不负责控制层窗口/GAS 的内部实现细节。

---

## 目录结构与组件职责

### `StepActivationSubsystem.{h,cc}`
- **定位**：Step 级随机激活注入子系统（Step Random Activation）。
- **核心能力**：
  - **选源**：按 `fraction` 对全局 neuron 做伯努利采样得到 pre 集合；
  - **生成 fanout**：对每个 pre 生成 `fanout` 个 spike（可选通过 BCSR reachability 约束 post 选择）；
  - **注入/外发**：将 spike 投递到本 PE 的目标 core，或外发到目标 PE；
  - **触发方式**：
    - `period_cycles > 0`：固定周期注入（tick 驱动）；
    - `period_cycles == 0`：legacy BeginGather 触发（`onBeginGather`）。
- **运行时绑定**（`Runtime`）：
  - `INocTransport* noc`：优先使用 NoC 抽象接口完成注入/外发；
  - `deliver_to_core` / `send_external`：与现有 MultiCorePE 后端保持兼容的注入回调；
  - `reset_membranes`：可选每步清膜电位（`reset_mem_each_step`）。
- **统计**（`Stats`）：
  - `invocations / pre_selected / spike_attempts / spikes_injected`；
  - `route_hits / route_misses`（当启用 BCSR reachability 路由采样时）。

---

## 与其他域的交互

- **NoC 域**：通过 `INocTransport` 完成本地投递与外发（Stimulus 不直接操作 NIC/ring）。
- **Route 域**：当前 Step 的 “BCSR reachability” 解析由 Stimulus 内部实现（仅影响 post 选择）；长期可考虑与 `services/route` 共享元信息以避免口径漂移。
- **Control/Components**：
  - `MultiCorePE` 负责把阶段事件（BeginGather/EndScatter）与 tick 转发给 Stimulus；
  - Stimulus 负责把注入事务封装起来，控制层不直接持有注入细节。

