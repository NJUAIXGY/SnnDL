# components/gather/（GatherBufferIF 参数解析模块）

本目录用于承载 **GatherBufferIF 的构造期参数解析**，以降低 `components/GatherBufferIF.cc` 中 `params.find(...)` 的噪音。

## 设计原则

- **不新增运行期组件/装配点**：仅新增解析函数与配置结构体；SST ELI 注册对象仍只有 `GatherBufferIF` 本体。
- **不改变参数面**：所有参数名/默认值保持与 `GatherBufferIF` 现有实现一致（仅搬运解析位置）。

## 文件

- `GatherBufferIFConfig.{h,cc}`
  - `parseGatherBufferIFConfig(const SST::Params&) -> GatherBufferIFConfig`
  - 负责读取 `verbose/merge_policy/sram_bytes/window_auto/...` 等参数并返回结构化配置。

---

## 建模口径提示：merge_policy 会改变“读粒度语义”

`merge_policy/gap_merge/burst_bytes_max/k_adapt` 会影响 GatherBufferIF 最终向下游发起读请求的“合并粒度”，从而改变 memHierarchy 的 cacheline 事务量：

- **默认语义（通用 memHierarchy/DRAM）**：cacheline 粒度（例如 64B）。
- **row-streaming/DMA**：属于显式架构假设，必须单列结果，不与 cacheline 模式混算。

为避免 `local_run_config.json` 写 `auto` 但实际被装配层覆写导致误读，mesh 模板会把最终生效参数写入运行目录的 `effective_config.json`（并汇总到 `essential_summary_mesh.json` 的 `model.effective`）。
