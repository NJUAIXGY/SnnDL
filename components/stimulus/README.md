# components/stimulus/（Stimulus 注入型组件）

本目录存放 **Stimulus 域的 SST 组件实现**：以“数据源/刺激源”的形式在仿真中注入 Spike 事件。

> 边界原则：该目录只放 *SST 可加载组件*（ELI 注册对象），具体注入策略/事务编排应优先放到 `services/stimulus/`，组件层只做装配与时序驱动。

---

## 主要内容

### `SpikeSource.{h,cc}`
- **定位**：脉冲数据源组件（`SST::Component`）。
- **功能**：
  - 从数据集文件加载 spike 序列，并按时间戳在仿真中注入；
  - 支持多种格式（以实现为准）：TEXT / NMNIST_AER / SHD_HDF5；
  - 支持 `time_scale`、`start_time_us`、`neuron_offset` 等常见注入参数；
  - 可选“严格 slice 分段释放”（`segmented_release` + `slices_per_superstep` + `slice_window_us/slice_gap_us`）。
- **输出端口**：
  - `spike_output`：输出 `events/SpikeEvent`（类型：`SnnDL.SpikeEvent`）。
- **典型用途**：
  - 用于外部数据集驱动的 SNN 输入侧刺激；
  - 用于最小链路连通性/端到端注入验证（不依赖 Step 注入逻辑）。

---

## 与 services/stimulus 的关系

- `components/stimulus/`：**组件层**（SST 生命周期 + 时钟驱动 + 端口输出）。
- `services/stimulus/`：**事务层**（例如 Step 注入的 timebase/选源/注入编排）。

