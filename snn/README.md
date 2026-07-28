# snn/（SNN 语义域）

本域包含脉冲神经网络专属逻辑，与通用平台和其他 workload 隔离。

## 子域

- `compute/`：神经动力学、学习核心和 `ISnnComputeCore` 实现。
- `synapse/`：权重、BCSR/GCSS、路由、Spike 通信和 GAS 数据结构。
- `stimulus/`：Step 与外部 spike 注入。
- `profiling/`：SNN 专属 profiling 辅助。

## 依赖规则

SNN 代码通过 `api/IMemoryAccess`、`api/INocTransport` 和 `api/SnnWeightReader` 使用平台能力，不直接持有平台实现。权重语义只放在 `synapse/weights/`，fanout 只放在 `synapse/route/`，动力学只放在 `compute/`。

新增行为时在相邻模块添加确定性回归。权重数据路径变化至少运行 `make test-compile`，协议变化运行 `make test-riscv-snn-protocols`。
