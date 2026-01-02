# docs/（设计与重构文档）

本目录存放 **与 SnnDL 设计、重构、阶段性方案相关的文档**，用于记录关键决策、口径与后续演进路线。

## 当前内容

- `SNNDL_HIERARCHY_AND_WORKFLOW.md`
  - SnnDL 总览文档：目录层次（Hierarchy）、跨层接口、以及以 `sst_dram_si/test_mesh_4x4.py` 为例的端到端工作流与回归口径。
- `UNIVERSAL_CONTROL_CORE_DESIGN.md`
  - 通用控制子核（`control/SnnPESubComponent`）目标形态、分层与迁移路线（Phase A/B/C/D）。
- `PHASE5_BOUNDARY_HARDENING_PLAN.md`
  - Phase5 边界硬化的可执行任务清单（PImpl/include 硬边界、Control 去 StandardMem、Compute 主接口收敛等），每步要求 10us→100us 回归。
- `PHASE3_MODULAR_BOUNDARIES_PLAN.md`
  - Phase3 边界契约与增量迁移策略（设计对照文档，记录域边界与防回归准则）。
- `SPIKE_COMM_SUBSYSTEM_DESIGN.md`
  - Spike 通信子系统抽象与落地说明（`api/ISpikeTransport` + `services/synapse/route/SpikeCommSubsystem`）。
- `NOC_SUBSYSTEM_DESIGN.md`
  - NoC 子系统化：`services/noc/NocSubsystem` 接管 send/recv/forward + 本地投递，并通过 `api/INocTransport` 冻结跨层调用面。
- `SUBSYSTEM_MODULARIZATION_ROADMAP.md`
  - 子系统化终局路线图（Memory / Synapse+Route / Stimulus / NoC / NeuralCompute）与阶段性验收口径。
- `SNNDL_CLEANUP_PLAN.md`
  - 历史清理与收敛计划（作为执行对照与回归口径参考）。

## 推荐阅读顺序（新人入口）

1. `../README.md`（快速构建/运行 + 目录边界总览）
2. `SNNDL_HIERARCHY_AND_WORKFLOW.md`（从“能跑起来”到“看懂数据流”）
3. 各子域 README：`../api/README.md`、`../components/README.md`、`../control/README.md`、`../compute/README.md`、`../services/README.md`
4. 深入设计与路线图：`SUBSYSTEM_MODULARIZATION_ROADMAP.md` → `PHASE5_BOUNDARY_HARDENING_PLAN.md` → 其余专题设计文档

## 约束与建议

- 文档应描述“为什么/做什么/如何验证/风险与回退”，避免重复粘贴代码实现细节。
- 与可替换 compute core 相关的接口契约，优先维护在仓库根部的 `ISnnComputeCore_SPEC.md`，并在此目录做设计补充与案例记录。
