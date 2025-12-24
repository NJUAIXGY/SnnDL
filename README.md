# SnnDL（SST Element：Spiking Neural Network / Deep Learning）

本目录是 `sst-elements` 中的 **SnnDL 元素库源码**，用于在 SST 中模拟多核 Processing Element（PE）上的脉冲神经网络（SNN）工作负载，并支撑 DRAM/NoC/窗口化（GAS）等系统级实验。

核心设计目标：**组件层只做装配与调度，事务下沉到子系统；compute core 可替换；边界清晰可回归。**

---

## 快速构建与安装（修改生效必须 install）

> 说明：SnnDL 属于 `sst-elements` 的一部分，通常在已 configure 的工作区内增量编译安装。

```bash
cd "sst_workspace/sst-elements/src/sst/elements/SnnDL"
make -j4
make install
```

---

## 快速运行回归（4×4 mesh 模板）

推荐使用项目内 mesh 模板驱动（详细见 `sst_dram_si/docs/mesh_template_guide_20251122-000250.md`）：

```bash
cd "sst_dram_si"
export MESH_SIM_TIME="100us"
./tools/run_mesh_with_time.sh
```

输出目录通常位于：
- `sst_dram_si/outputs_large/paper2/dram_mesh_4x4/YYYYMMDD-HHMMSS/`
  - `essential_summary_mesh.json`：关键指标摘要（用于 100us 回归判定）。

---

## 目录结构（按边界划分）

> 每个子目录都有自己的 README，优先以子目录 README 为准。

```
SnnDL/
├── api/            # 跨层稳定接口（窄抽象）
├── events/         # 事件与数据载体（Spike/Gating 等）
├── components/     # SST 组件装配壳（ELI 注册对象）
├── control/        # 控制/编排壳（GAS/内存/路由调度，不含动力学）
├── compute/        # 可替换 compute core（神经动力学/学习/验证）
├── services/       # 可复用事务子系统（按子域拆分）
│   ├── noc/        # NoC 传输域（send/recv/forward/本地投递）
│   ├── memory/     # 纯内存访问域（地址→字节块）
│   ├── synapse/    # 突触语义域（weights/route/gas 事务闭环）
│   ├── stimulus/   # Stimulus 域（Step 注入/外部刺激）
│   └── legacy/     # 历史遗留/参考实现（默认不进主链路）
├── docs/           # 设计与阶段性方案文档
└── tests/          # include 自检等轻量测试
```

---

## 关键边界（控制 / 计算 / 路由 / 内存 / 权重）

- `components/`：SST 对接与装配壳（端口、Link、Clock、Stat、生命周期）；尽量不写算法事务。
- `control/`：控制/编排壳（窗口/GAS、内存请求编排、统计汇聚）；**不包含神经动力学**。
- `compute/`：神经动力学与学习等计算逻辑（`ISnnComputeCore`）；**不直接触碰 StandardMem/NoC**。
- `services/noc/`：纯传输（send/recv/forward/本地投递）；**不做 fanout/权重语义**。
- `services/memory/`：纯地址/字节访问；**不出现权重/突触/路由语义**。
- `services/synapse/`：权重语义/路由与 fanout/GAS 辅助（weights/route/gas 事务闭环）。
- `services/stimulus/`：注入时基与选源（Step 等刺激）；通过 NoC/Route 完成投递与外发。

---

## 开发提示

- 修改 C++ 后必须执行 `make install` 才会影响实际 `sst` 运行加载的元素库。
- 若需要调整构建清单：优先改 `Makefile.am`，并同步保持 `Makefile.in` 与之匹配（`make` 会通过 `config.status` 重生成 `Makefile`）。
- 回归建议：每个阶段至少跑一次 `MESH_SIM_TIME="100us"`，并对比 `essential_summary_mesh.json` 的关键字段（避免非确定性回归）。
