# platform/（通用执行平台）

本域提供与具体 workload 无关的执行基础设施。平台代码只处理生命周期、packet、地址与统计，不解释神经元、突触或张量算子语义。

## 子域

- `core/`：`SnnPESubComponent` CoreShell，负责时钟、packet 投递、运行时绑定和统计汇聚。
- `memory/`：`IMemoryAccess` 实现、DMA 调度和可选 SRAM 时序模型。
- `noc/`：内部环、packet 转发和通用 NoC 子系统。
- `stats/`：按 workload 注册和汇聚统计字段。

## 依赖规则

`platform/` 可以依赖 `api/` 和 `events/`，但不得依赖具体 workload 实现。SNN 或 Tensor 语义应分别留在 `snn/` 和 `workloads/`。跨域调用优先使用 `api/` 的窄接口。

修改 CoreShell 后先运行 `make -j4`，再运行 `make test-compile`；涉及 packet 或内存时还应执行对应协议测试。
