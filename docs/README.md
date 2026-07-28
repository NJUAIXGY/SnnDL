# SnnDL Documentation

本目录将当前架构说明与历史设计记录分开。代码和根级 `README.md` 是目录与公共接口的最终依据。

## 当前文档

- `SNNDL_PROJECT_OVERVIEW.md`：组件注册面、四个实现域和依赖方向。
- `SNNDL_HIERARCHY_AND_WORKFLOW.md`：从 spec 到运行、统计和验证的主流程。
- `BCSR_WEIGHT_GENERATION_GUIDE.md`：BCSR 文件布局与生成/校验方法。
- `../ISnnComputeCore_SPEC.md`：compute core 接口契约。

按域继续阅读 `../platform/README.md`、`../workloads/README.md`、`../snn/README.md` 和 `../research/README.md`。

## 历史材料

`plans/`、`SUBSYSTEM_MODULARIZATION_ROADMAP.md` 和 `UNIVERSAL_CONTROL_CORE_DESIGN.md` 记录实施过程，可能包含迁移前路径。它们不定义当前目录结构或默认行为。

## 文档约束

- 当前文档只引用工作树中存在的路径。
- 参数或统计语义变化必须同步更新相邻 README。
- 运行结果、生成统计和 `.deps/.libs` 不作为架构文档。
- 默认内存口径是 cacheline；更大 granule 或 DMA 假设必须在 spec 与结果中显式记录。
