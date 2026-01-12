# services/workload/（Workload 插件域）

本目录存放 **可插拔 workload 插件**：平台核（CoreShell/NoC/Memory）保持通用，“业务语义与执行主链路”由 workload 决定。

> 目标：让同一套平台装配（MultiCorePE + CoreShell + NoC/Mem）可以运行不同范式的工作负载：SNN（Spike/GAS/BCSR/Step）或非 SNN（streaming/packet test 等）。

---

## 如何选择 workload（不改脚本的默认口径）

- 优先：组件参数 `workload_impl=<name>`
- 其次：环境变量 `SNNDL_WORKLOAD_IMPL=<name>`
- 默认：`snn`

mesh 模板示例（切换 stream）：

```bash
cd "sst_dram_si"
export SNNDL_WORKLOAD_IMPL="stream"
export MESH_SIM_TIME="100us"
./tools/run_mesh_with_time.sh
```

---

## 子目录

- `services/workload/snn/`：SNN 主链路 workload（Spike/GAS/BCSR/Step），内部调用 `services/synapse/*` 与 `compute/*`
- `services/workload/stream/`：非 SNN 的 streaming workload（packet-first 通信 + 内存 read-after-write 校验）
- `services/workload/traffic/`：通信/多播验证用的 traffic workload（不建模动力学；用于 SpikeKey/native multicast 等实验）

---

## 依赖边界（必须遵守）

- Workload **允许依赖**：`api/`、`events/`、`services/*`、`compute/*`（按自身语义选择）。
- 平台面（`services/noc`、`services/memory`、`control`）**不得**反向依赖某个具体 workload 的实现细节。
- `workload=stream` **不得**依赖 `SpikeEvent`、`synapse/*`、`stimulus/*`（确保“非 SNN 负载”纯净）。
- `workload=traffic` **不得**依赖 `compute/` 的动力学实现；它是通信/路由验证负载，但允许复用 `services/synapse/route` 的 fanout/multicast 构建与发送事务。
