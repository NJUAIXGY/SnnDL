# docs/（设计与重构文档）

本目录存放 **与 SnnDL 设计、重构、阶段性方案相关的文档**，用于记录关键决策、口径与后续演进路线。

## 当前内容

- `UNIVERSAL_CONTROL_CORE_DESIGN.md`
  - 通用控制子核（`control/SnnPESubComponent`）目标形态、分层与迁移路线（Phase A/B/C/D）。
- `SPIKE_COMM_SUBSYSTEM_DESIGN.md`
  - Spike 通信子系统抽象与落地说明（`api/ISpikeTransport` + `services/synapse/route/SpikeCommSubsystem`）。
- `NOC_SUBSYSTEM_DESIGN.md`
  - NoC 子系统化：`services/noc/NocSubsystem` 接管 send/recv/forward + 本地投递，并通过 `api/INocTransport` 冻结跨层调用面。
- `SUBSYSTEM_MODULARIZATION_ROADMAP.md`
  - 子系统化终局路线图（Memory / Synapse+Route / Stimulus / NoC / NeuralCompute）与阶段性验收口径。
- `SNNDL_CLEANUP_PLAN.md`
  - 历史清理与收敛计划（作为执行对照与回归口径参考）。

## 约束与建议

- 文档应描述“为什么/做什么/如何验证/风险与回退”，避免重复粘贴代码实现细节。
- 与可替换 compute core 相关的接口契约，优先维护在仓库根部的 `ISnnComputeCore_SPEC.md`，并在此目录做设计补充与案例记录。
