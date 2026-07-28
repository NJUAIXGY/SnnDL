# workloads/（可插拔工作负载）

`CoreWorkloadFactory.cc` 根据 `workload_impl` 创建 workload；环境变量 `SNNDL_WORKLOAD_IMPL` 保留为兼容覆盖。默认实现是 `snn`。

## 实现目录

- `snn/`：Spike、GAS、BCSR 和 Step 主链路。
- `riscv_snn/`：RISC-V 固件、ISS、ABI 和 SNN runtime bridge。
- `stream/`：packet 与内存 read-after-write 校验。
- `traffic/`：NoC、多播和路由验证。
- `traffic_mem/`：组合通信与内存压力。
- `tensor/`：GEMM、tile/program 执行、DMA 和 collective。
- `layout/`：跨 workload 的 neuron layout 归一化。
- `common/`：可复用 backend 契约，不放具体 workload 状态机。

Tensor 实现按职责拆为主执行、配置、collective/packet、program 和统计翻译单元。

## 边界

Workload 可以依赖 `api/`、`events/` 和 `platform/` 的稳定接口。只有 SNN workload 可以依赖 `snn/`；`stream`、`traffic` 和 `tensor` 不得引入神经动力学状态。

布局参数统一解释为：`num_neurons` 是每 core 数量，`neurons_per_pe` 是每 PE 总量，`global_neuron_base` 是当前 core 的全局基址。

## 验证

修改工厂或公共契约后运行：

```bash
make -j4
make test-compile
make test-riscv-snn-protocols
```

Tensor 端到端实验从仓库的 `sst_workloads/tensor_si/` 启动；SNN mesh 从 `sst_dram_si/` 启动。
