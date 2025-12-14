# docs/（设计与重构文档）

本目录存放 **与 SnnDL 设计、重构、阶段性方案相关的文档**，用于记录关键决策、口径与后续演进路线。

## 当前内容

- `PHASE3_BCSR_LAYOUT_ACCUMULATOR_DESIGN.md`
  - BCSR 布局、窗口累加器（Accumulator）与相关优化点的设计说明。
- `SNNDL_CLEANUP_PLAN.md`
  - 清理与重构计划的阶段记录（用于对照执行与回归验证）。

## 约束与建议

- 文档应描述“为什么/做什么/如何验证/风险与回退”，避免重复粘贴代码实现细节。
- 与可替换 compute core 相关的接口契约，优先维护在仓库根部的 `ISnnComputeCore_SPEC.md`，并在此目录做设计补充与案例记录。

