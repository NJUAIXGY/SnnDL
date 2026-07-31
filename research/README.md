# research/（实验机制域）

本域收纳尚未成为稳定平台契约的研究机制，避免实验参数和观测状态污染主平台。

## 子域

- `gas/`：实验性 GAS credit 与预测策略。
- `local_storage/`：PE/Pod 本地对象、shadow gate 和共享元数据。
- `noc3d/`：3D NoC、HBM stack stub 与原生多播实验组件。
- `route3d/`：3D 路由映射与突触路由扩展。

已移出主线的 PE shared-fabric、agenda 和 descriptor 实验保留在
`archive/legacy_optimizations/research/pe_fabric/`，不参与当前构建。

## 稳定性边界

这些实现可以通过已有参数显式启用，但默认值、ELI 名称和统计键仍属于兼容面。实验代码应通过 `api/` 接入稳定域；不要让 `platform/` 或通用 workload 反向依赖单一实验。

将机制提升为主路径前，需要补齐默认关闭验证、启用路径回归和对应 `effective_config` 证据。
