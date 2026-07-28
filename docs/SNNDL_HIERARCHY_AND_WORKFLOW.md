# SnnDL Hierarchy and Workflow

## 主流程

1. Python spec 工具解析并校验配置。
2. SST 配置创建 `MultiCorePE`、NIC、内存和可选全局控制组件。
3. `MultiCorePE` 为每个 core 创建 `SnnPESubComponent`。
4. `CoreWorkloadFactory` 根据 `workload_impl` 创建 workload。
5. CoreShell 将拓扑、内存、NoC、日志和时间源作为 runtime 注入 workload。
6. Clock tick 推进 workload；packet 统一通过 `NocPacketEvent` 投递。
7. finish 阶段由 workload、CoreShell 和 PE 逐层汇聚稳定统计键。

## 目录职责

- `components/`：SST 注册、端口、生命周期与资源装配。
- `platform/`：与 workload 无关的 core、memory、NoC、stats。
- `workloads/`：可执行状态机与事务编排。
- `snn/`：仅 SNN 使用的 compute、synapse、stimulus。
- `research/`：显式启用的实验机制。
- `api/`、`events/`：跨域契约与载体。

## 权威配置

外部 spec 和 ELI 参数是输入，运行时注入的 `CorePlatformConfig` 是 core 拓扑权威值。代码不得从局部默认值重新推导 `node_id`、`total_nodes` 或 `total_cores`。Tensor 在 `bindRuntime()` 中采用注入的 `total_cores`，避免配置解析与实际装配漂移。

Neuron layout 使用统一含义：

- `num_neurons`：每 core 的 neuron 数。
- `neurons_per_pe`：每 PE 的 neuron 总数。
- `global_neuron_base`：当前 core 的全局起点。

## 数据边界

Memory 只承诺 `addr + size -> bytes`，NoC 只承诺 packet 传输。权重解析位于 `snn/synapse/weights`，fanout 位于 `snn/synapse/route`，动力学位于 `snn/compute`。

默认系统流量口径是 memHierarchy cacheline 事务。逻辑请求字节、GAS 合并字节和 off-chip 字节必须分别报告。

## 开发循环

```bash
cd remote/sst_workspace/sst-elements/src/sst/elements/SnnDL
make -j4
make test-compile
make test-riscv-snn-protocols
```

涉及实际运行时，从仓库根使用 spec-first 入口：

```bash
python3 tools/snndl_spec_cli.py validate tools/specs/mesh_minimal_v3.json
bash tools/run_snndl_with_time.sh --spec tools/specs/mesh_minimal_v3.json
```

检查 run directory 中的 `effective_config.json`、essential summary 和 `validation.log`。实验性机制必须同时记录启用参数和对应统计证据。

## 修改规则

- 保持 ELI 名称、参数、端口和统计键兼容。
- 新业务状态机放入对应 workload 或 SNN 子域，不放入组件壳。
- 新研究机制先放入 `research/`，通过稳定接口接入。
- 修改 `Makefile.am` 后同步生成 `Makefile.in` 和本地 `Makefile`。
- 不提交 `.deps`、`.libs`、`.lo`、`.o` 或临时运行输出。
