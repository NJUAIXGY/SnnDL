# SnnDL 清理与收敛方案（Phase 2/3）

本文档记录当前状态、分支策略、分阶段清理计划与回归测试方案。目标是在不改变 Strict GAS 与内存系统行为的前提下，删繁就简、统一路径、默认静默并提升鲁棒性。

## 1. 现状概览
- 语义/架构：Strict GAS 已稳定（Gather→Apply→Scatter）；仅内存权重在 Apply 产生 ΔV，Scatter 判定发放；事件权重/回退默认关闭。
- 拓扑/数据：4×4 mesh，10k/PE（20×500）；BCSR 全局权重（br=1 bc=16 idx=2 val=4）。
- BCSR 布局（10k/PE 基线，64B 对齐）：
  - rowptr=0, colidx=2048, blockdata=226432, blockids=7404928
  - per-core stride = 每 PE core 文件大小的最大值（对齐）
- 一致性：目的节点计算使用 `neurons_per_pe=10000`；mem==file 抽样通过；NIC 统计已由工具聚合。
- 日志：仍有零散诊断输出（WL-*、conv-read、[diag-*]）。需统一门控并默认静默。

## 2. Git 分支
- 根仓库：`/home/xgy/remote`（当前 `feature/gas-refactor-doc`；不影响 SnnDL 子仓库工作）
- sst-elements：`HEAD`（detached），暂不改动
- SnnDL 子仓库：`sst_workspace/sst-elements/src/sst/elements/SnnDL`
  - `main`：已包含“clearify phase1”
  - 当前开发分支：`snndl_phase2_cleanup`（基于 `main`）
  - 远端：`origin = https://github.com/NJUAIXGY/SnnDL.git`（暂不推送）

## 3. 目标与约束
- 目标：统一唯一运行路径；默认静默，诊断可控；保持行为与统计口径不变。
- 约束（严格）：
  - 不改变内存系统行为：不动 GatherBufferIF 的合并/突发/并发/延迟策略，不引入缓存/重排。
  - Strict GAS 语义不变：仅 Apply 通过内存权重产生 ΔV；Scatter 发放；事件权重回退默认关闭。

## 4. 分阶段清理计划

### Phase 2（默认静默 / 诊断门控统一 / 冻结唯一路径）
- 诊断门控统一：
  - 环境变量：`SNNDL_DEBUG=0/1`
  - 参数：`window_read_debug=0/1`
  - 将 WL-*、conv-read、[diag-gbi]、[diag-stage]、[step-*] 等全部归口到上述门控；默认关闭。
- 冻结唯一路径：
  - GatherBufferIF：仅保留自动窗口；`manual_window_drive` 路径封存且默认禁用。
  - SnnPESubComponent：仅在 Gather 阶段记录边（recordEdge_）；Apply/Scatter 记录变体默认禁用。
  - 回调路径：仅保留“ID 直通”；脚本侧保持 `defer_issue_until_apply=0`；去除历史 deferral-ID 重映射分支。
  - MultiCorePE：Step BCSR 路由加载失败仅计数 + 限流告警；uniform 回退默认禁用。
- 自检（默认不打印）：
  - 计数/断言（仅 `SNNDL_DEBUG=1` 生效）：pending 插入/回调/清理配平；窗口翻转时 pending=0；edge_collector curr/prev 翻转后清空；BCSR 响应尺寸与权重守护；ΔV 每窗 touched_posts 与 updates 一致性。

### Phase 3（代码结构收敛与小型优化，不改访存）
- BCSR 布局集中：引入 `BcsrLayout{rows, cols, br, bc, idx_bytes, val_bytes, offsets, stride}`，集中校验与寻址，替代分散计算。
- 容器/分配优化（不改请求数/行为）：
  - edge 收集器 curr/prev 预留容量；pending map 预留；块解码复用 scratch buffer；减少小对象分配。
- 稀疏累加器重置优化：
  - 引入 touched_list_/bitmap，仅 reset 被触达索引（先以 shadow 比对验证 ΔV 等价，再切换为默认实现）。
- 路由/节点计算抽象：
  - `neurons_per_pe` 读取与目标节点计算抽为 helper，避免漂移与重复。

### Phase 4（文档与工具）
- `sst_dram_si/docs/SNNDL_DEBUG_LOGGING_GUIDE.md`：列出全部开关、默认值与影响面
- `sst_dram_si/TECH_PROGRESS.md`：记录变更、运行方法、基线指标
- 统计说明：明确 spike_activity（Scatter 发放）与 SnnNIC（网络收发）是不同维度

## 5. 回归测试方案（默认静默）

### 构建
```bash
cd sst_workspace/sst-elements/src/sst/elements/SnnDL
make -j && make install
```

### 10us 快测
```bash
cd sst_dram_si
MESH_ALT_STOP=1 MESH_SIM_TIME=10us tools/run_mesh_with_time.sh
# 若无 summary，补算：
python3 tools/compute_essential_summary_mesh.py --run-dir outputs_large/paper2/dram_mesh_4x4/<timestamp>
```
- 验收（对比基线，允差 ±1%）：
  - memory.memory_requests 与 memory_bytes 稳定
  - gas.windows ~ 570–580
  - spike_activity 非零（total_spikes_processed、neurons_fired_total > 0）
  - nic.spikes_sent/recv 非零
  - mesh_run.log 默认无 WL-*/conv-read/[diag-*]

### 100us 稳态
```bash
cd sst_dram_si
MESH_ALT_STOP=1 MESH_SIM_TIME=100us tools/run_mesh_with_time.sh
```
- 生成并检查 essential_summary_mesh.json，量级与基线一致

### 诊断开关（仅定位时使用）
- `SNNDL_DEBUG=1` 或 `window_read_debug=1` 临时开启日志
- 定位结束后关闭并复测 10us/100us

## 6. 风险与回滚
- 所有改动在 `snndl_phase2_cleanup` 分支推进；指标偏差>±1% 或触发断言则停止推进，切回 `main` 定点修复。
- 不进行破坏性 git 操作；是否推送远端需另行批准。

## 7. 近期执行顺序
1) Phase 2：统一门控、冻结唯一路径、加自检计数（默认不打印），10us→100us 回归
2) Phase 3：BcsrLayout 集中、容器/分配优化、累加器 touched reset（shadow 验证），10us→100us 回归
3) Phase 4：完善文档与回归脚本，保持默认静默与稳定统计口径

