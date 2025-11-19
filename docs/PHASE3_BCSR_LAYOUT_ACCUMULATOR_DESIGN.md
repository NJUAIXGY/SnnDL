# Phase 3 设计：BCSR 布局抽象 + 累加器 touched reset（含影子验证）

> 目标：在不改变 Strict GAS 语义与统计口径的前提下，收敛 BCSR 寻址代码并降低 Apply/Scatter 窗口重置开销；通过影子验证保障 ΔV 等价，默认保持现有行为，风险可控逐步切换。

## 1. 约束与范围

- 不可改变：
  - Strict GAS：Gather 仅记录边；Apply 读取权重累加 ΔV；Scatter 应用 ΔV 发放；事件权重回退默认关闭。
  - 统计口径与字段名：不改现有统计名（供 `compute_essential_summary_mesh.py` 使用）。
  - Mesh 模板行为：`sst_dram_si/test_mesh_4x4.py` 与 `SnnDL_Basic/scripts/test_classification_4x4.py` 无修改即可通过 10us/100us 回归（允差±1%）。
- 可改变：
  - BCSR 布局寻址从分散参数拼装收敛到集中结构体；
  - 累加器内部容器与 reset 策略（对外 ΔV 语义不变）。

## 2. 现状问题

- BCSR 参数（br/bc/idx/val/offset/stride）在 `SnnPESubComponent.cc` 多处散落，校验/日志重复，易漂移。
- 累加器两套：稀疏 map（`acc_delta_`）开销大；致密向量（`apply_dense_acc_enable_`）已有但默认关闭。
- 缺少系统性的 ΔV 等价校验（新旧路径切换风险缺乏防护）。

## 3. 方案概述

- 引入 BcsrLayout 抽象（集中校验与寻址），替代分散 offset 拼接；仅“搬家”，不改偏移/对齐/缓存策略。
- 扩展累加器致密路径（沿用 `apply_dense_acc_enable_`）为目标默认实现；切换前先做影子验证。
- 影子验证（shadow verify）：窗口内并行维护“对照累加器”，BeginScatter 前比较 per-post ΔV；默认关闭，仅调试启用，发现不一致仅日志告警。

## 4. 结构与接口

### 4.1 BcsrLayout（集中定义，建议放置 `SnnPESubComponent.h`）

- 字段：
  - `rows, cols`；`block_rows, block_cols`；`idx_bytes, val_bytes`；
  - `rowptr_offset, colidx_offset, blockdata_offset, blockids_offset`；
  - `per_core_stride`（可选，不影响现有寻址即可为 0）。
- 方法：
  - `validate(base, out, debug, core_id, node_id) -> bool`：检查单调与 64B 对齐，打印一次性诊断；返回校验结果（不 fatal）。
  - `addr*()` 可选（直接 `base+offset` 即可）。
- 生成与使用：
  - 在 `SnnPESubComponent::setup()` 一处生成并 `validate()`；
  - 将 `bcsr_rowptr_addr_` 等由 `base+offset` 统一赋值，避免散落计算与日志。

### 4.2 累加器（Apply/Scatter 窗口）

- 稀疏路径：`std::unordered_map<uint32_t,float> acc_delta_`（保留、默认仍启用）。
- 致密路径（已有 `apply_dense_acc_enable_`）：
  - `std::vector<float> acc_dense_` 长度=`num_neurons_`；
  - `std::vector<uint8_t> acc_touched_bitmap_` + `std::vector<uint32_t> acc_touched_list_`；
  - `accUpdate_(post,dv)`：累加 `acc_dense_[post]`；首次触达入 `touched_list_`，bitmap 置 1；统计 `posts_touched++`；
  - BeginScatter：仅遍历 `touched_list_` 应用 ΔV 并判阈值；
  - `accReset_()`：仅清除 `touched_list_` 所在元素与 bitmap；map 路径仍是 `clear()`。

### 4.3 影子验证（shadow verify）

- 容器：`std::unordered_map<uint32_t,float> acc_shadow_map_`；开关：`acc_shadow_verify_enable_`（默认 0，只有 `SNNDL_DEBUG=1` 时才允许置 1）。
- 当致密路径启用且 shadow 开启：每次 `accUpdate_` 同步写 shadow_map；BeginScatter 比较：
  - 对 `touched_list_` 内每个 post，比 `acc_dense_[post]` 与 `acc_shadow_map_[post]`，|Δ|<=1e-6 视为等价；
  - 对 `shadow_map` 有而 `touched_list_` 无的 post，也输出一次性告警；
  - 仅日志 `[acc-shadow]`，不 fatal。

## 5. 参数与兼容性

- 外部参数：不新增。复用 `apply_dense_acc_enable` 与 `window_read_debug`；
- 内部开关（可选）：`acc_shadow_verify_enable`，默认 0；当 `SNNDL_DEBUG=1` 且 `window_read_debug=1` 时方可由参数打开；
- 默认行为不变：`apply_dense_acc_enable=0`（map）；影子验证默认关闭。

## 6. 实施步骤

1) BcsrLayout 引入与替换（零行为变更）：
   - 在 `SnnPESubComponent.h` 定义 `struct BcsrLayout`；
   - 在 `SnnPESubComponent::setup()` 读取参数生成 layout，`validate()` 一次；
   - 用 `base+offset` 统一赋值 `bcsr_rowptr_addr_` 等；保留旧字段一版周期后再清理。

2) 影子验证加入（不改默认路径）：
   - 新增 `acc_shadow_map_` 与开关；致密路径启用时镜像更新；
   - BeginScatter 比较，`SNNDL_DEBUG=1` 时如有差异打印 `[acc-shadow]` 一次性告警。

3) 回归验证：
   - 10us：`MESH_SIM_TIME=10us`，开启致密+影子（`apply_dense_acc_enable=1`，`acc_shadow_verify_enable=1`，`SNNDL_DEBUG=1`，`window_read_debug=1`），确认无告警；
   - 100us：同上；
   - 关闭影子后，默认 `apply_dense_acc_enable=0` 再跑一轮，确认指标未偏移。

4) 切换默认（可选，待审批）：
   - 将 `apply_dense_acc_enable` 默认值改为 1（保持可通过参数回退）。

## 7. 验证与验收

- 编译：`make -j && make install` 通过；
- 10us 快测：gas.windows≈570–580；memory.requests/bytes 与基线±1%；nic spikes sent/recv 非零；日志默认静默；影子验证无告警；
- 100us 稳态：与基线量级一致；
- SnnDL_Basic：`scripts/test_classification_4x4.py` 正常运行，不强依赖新路径。

## 8. 性能预期

- BcsrLayout：减少重复代码与漂移点（性能≈0 影响）。
- 致密累加器：reset 从 O(N) → O(touched)，降低 rehash 与清零成本；
- 额外内存：O(N) floats/核；例如 500 行/核 × 20 核/PE × 4B ≈ 40KB/PE。

## 9. 风险与回滚

- 风险：
  - BCSR offset 校验条件错误 → 仅日志告警，不影响路径；
  - ΔV 不一致 → 影子验证仅告警，不切换默认。
- 回滚：
  - BcsrLayout 只是“集中来源”，保留原地址变量即可快速回退；
  - 累加器：`apply_dense_acc_enable=0` 立刻回退 map；影子验证默认关闭。

## 10. 代码触点（规划）

- `SnnPESubComponent.h/.cc`：
  - 新增 `struct BcsrLayout` 与 `acc_shadow_map_`；
  - `setup()` 处构造 layout 并 validate；
  - `accUpdate_/accReset_/BeginScatter` 加入影子验证（仅调试）。
- 不改：`GatherBufferIF`、`SnnNIC`、`WeightLoader`、mesh 脚本。

## 11. 排期建议

- T+1：落地 BcsrLayout（零行为变更）+ 10us 验证；
- T+2：加入影子验证（默认关闭）+ 10us/100us 验证；
- T+3：评审后决定是否将致密路径设为默认，后续清理旧字段与冗余日志。

