# workloads/（可插拔工作负载）

`CoreWorkloadFactory.cc` 根据 `workload_impl` 创建 workload；环境变量 `SNNDL_WORKLOAD_IMPL` 保留为兼容覆盖。默认实现是 `snn`。

构建时，原生 SNN 工厂和 `SnnWorkload` 位于 `libSnnDLOpt.la`，可选 workload 位于 `libSnnDLWorkloads.la`，3D/Traffic 家族位于独立的 `libSnnDLResearch.la`。因此“源码在本目录”不等于“默认插件会加载它”。

## 实现目录

- `snn/`：Spike、GAS、BCSR 和 Step 主链路。
- `riscv_snn/`：RISC-V 固件、ISS、ABI 和 SNN runtime bridge。
- `stream/`：packet 与内存 read-after-write 校验。
- `traffic/`：NoC、多播和路由验证；由 `libSnnDLResearch.la` 提供。
- `traffic_mem/`：组合通信与内存压力；由 `libSnnDLResearch.la` 提供。
- `tensor/`：GEMM、tile/program 执行、DMA 和 collective。
- `layout/`：跨 workload 的 neuron layout 归一化。
- `common/`：可复用 backend 契约，不放具体 workload 状态机。

Tensor 实现按职责拆为主执行、配置、collective/packet、program 和统计翻译单元。

## 边界

Workload 可以依赖 `api/`、`events/` 和 `platform/` 的稳定接口。只有 SNN workload 可以依赖 `snn/`；`stream`、`traffic` 和 `tensor` 不得引入神经动力学状态。3D 与 Traffic 实现不得回流到 Core、Comm、Local 或默认 Workloads 清单。

布局参数统一解释为：`num_neurons` 是每 core 数量，`neurons_per_pe` 是每 PE 总量，`global_neuron_base` 是当前 core 的全局基址。

## 验证

修改工厂或公共契约后运行：

```bash
make -j4
make test-compile
make test-riscv-snn-protocols
```

Tensor 端到端实验从仓库的 `sst_workloads/tensor_si/` 启动；SNN mesh 从 `sst_dram_si/` 启动。

## 加载 Research workload

`TrafficWorkloadPlugin.cc` 通过静态初始化注册 `traffic` 和 `traffic_mem`。运行这两类 workload 时，需显式把 `libSnnDLResearch.so` 加入进程（例如设置 `LD_PRELOAD`，并把 SnnDL `.libs` 放入 `LD_LIBRARY_PATH`/`SST_ADD_LIB_PATH`）。未加载 Research 库时，选择 `traffic` 或 `traffic_mem` 会按未注册 workload 失败；这是预期的 fail-fast 行为。
