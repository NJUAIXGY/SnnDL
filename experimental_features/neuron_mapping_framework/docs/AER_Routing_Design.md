# neuron_mapping_framework AER 路由设计（对标 SpiNNaker）

本文面向 neuron_mapping_framework，给出一套遵循 SpiNNaker 地址事件表示（AER, Address-Event Representation）理念的映射与路由设计方案，用于指导后续实现与与 SST‑SnnDL 集成。方案坚持“离线编译、运行时硬件样式转发”的思想：所有复杂工作在编译期完成，运行时仅基于预置路由表进行按键匹配的多播/单播转发。

## 1. 目标与原则
- 目标：在不修改现有 SnnDL 核心接口前提下，提供可落地的 AER 风格“应用图 → 机器图 → 分布式路由表”流水线。
- 兼容：保持 neuron_mapping_framework 现有数据结构与接口命名（AddressEvent, RoutingTable, RoutingTableGenerator, RoutingAwareGraphPartitioningStrategy 等）。
- 最小化变更：优先复用现有头文件/类；新增内容以实现和导出为主。
- 可配置：表项规模、掩码策略、多播树构造、压缩阈值等均以配置项控制。

## 2. 总体流程
```
高层模型(应用图) → 分区与映射(路由感知) → 全局ID分配(AER键) → 多播组检测
                               ↓                     ↓
                         机器图(PE/Core)      多播树/路由生成(每PE)
                               ↓                     ↓
                         路由表压缩/验证 → 分布式路由表导出(JSON/CSV/二进制)
```

对应现有模块：
- 分割/映射：RoutingAwareGraphPartitioningStrategy + MappingSolution
- AER 事件键：AddressEvent（32 位 GlobalNeuronId）
- 路由生成：RoutingTableGenerator（最短路、负载均衡、多播优化等）
- 路由表：RoutingTable（key, mask, routes 三元组；最长前缀匹配风格）

## 3. 数据与键位设计（AER）
- 32 位全局神经元 ID：沿用 AddressEvent 中的约定
  - local_id: 16b | pe_id: 12b | core_id: 4b
  - 全局路由键 key = global_id（运行时查表即对该 key 做匹配）
- 掩码匹配：RoutingEntry 使用 `(packet_key & mask) == (key & mask)`。
- 关键约束：不修改 AddressEvent 接口的前提下，默认采用“每神经元唯一键”的模式实现单播/多播；可通过“键分配策略 + 掩码压缩”获取一定聚合度。
- 进阶（可选）：保留“组键”能力作为将来扩展（MULTICAST 事件 + 组键），不影响默认路径（文末 TODO）。

## 4. 路由生成算法（对标 SpiNNaker）

4.1 分区与映射（Routing-aware）
- 目标：满足单 PE 存储上限约束（示例：128KB 级 SRAM）与计算/带宽预算，在此约束下最小化跨 PE 通信与路由拥塞。
- 方法：
  - 基线：多层图分割（MULTILEVEL）+ 局部改良（如 Kernighan–Lin/Fiduccia–Mattheyses）
  - 路由感知：在代价函数中加入路由项（跳数、链路负载、热点度、估算拥塞），复用 RoutingAwareGraphPartitioningStrategy::RoutingAwareConfig。
  - 指标：通信边切割率、平均跳数、链路最大利用率、预计表项数。

4.2 全局 ID 分配
- `RoutingTableGenerator::assignGlobalNeuronIds(...)`
- 约束保持：位宽布局不变；在 local_id 分配上优先“局部连续”，利于后续掩码聚合（例如同 PE、同核的一组连续 local_id 可被较短前缀掩码覆盖）。

4.3 多播组检测与构建
- `detectMulticastGroups(network, mapping, min_group_size)`：
  - 候选标准：
    - 同源群体对多个目标群体复用度高（fan-out 大且重复目标集合相似）
    - 边权（发放率×权重）大、跨 PE 目标多
  - 生成 `MulticastGroup{ group_id, members, target_pes }`
- 多播树：
  - `buildMulticastTree(group, topology, root_pe)`：在目标 PE 集合上构建近似 Steiner 树（最小生成树 + 局部替换/延迟约束），根一般选源侧或网络中心性较好的 PE。

4.4 路径计算
- 单播：最短路（BFS/Dijkstra，按拓扑链接权重；二维 Mesh 即曼哈顿路径）
- 负载均衡（可选）：K 最短路备选 + 链路热度惩罚，按权重轮转

4.5 表项生成（每路由器/PE）
- 单播：沿路径上每个路由器插入一条 `(key = global_id, mask = 0xFFFFFFFF, routes = {下一跳/或 LOCAL})`
- 多播：对树上分叉节点插入多出向方向；叶子节点含 LOCAL。
- 优先级：本地路由 > 组播/压缩项 > 回退项；可借助 `priority` 字段排序。

4.6 表压缩与合并
- `compressRoutingTables(..., method)`：
  - 前缀聚合（PREFIX_AGGREGATION）：同 PE、相邻 local_id 且路由完全相同的条目聚合为短掩码（例如 8/16 个连续键 → 单条前缀项）
  - 路由合并（ROUTE_MERGING）：多条键不同但 routes 相同且掩码兼容时合并
  - 多播树压缩（MULTICAST_TREES）：将重复子树上多条相同出向的单播项“上提”为一次分叉
- 目标：在准确性不变前提下降低表项数；保持 `validateTable()` 与端到端连通性检查通过。

## 5. 运行时执行模型
- 事件生成：PE 上神经元发放 → AddressEvent(global_id, ts, weight)
- 路由查找：硬件风格 CAM 匹配（RoutingTable::lookupRoute 或 SpikeRouter::routePacket 语义）
- 多播复制：在路由器侧按 routes 复制转发；LOCAL 方向交付至本地核心/核内队列
- 无状态、静态：运行时不做路径搜索，仅表项匹配与复制

## 6. 关键配置项（建议默认）
- 路由策略：SHORTEST_PATH；可切换 LOAD_BALANCED/MULTICAST_OPTIMIZED
- 压缩：enable_compression=true，method=PREFIX_AGGREGATION，目标压缩比≥50%
- 多播：enable_multicast=true，min_group_size=3，multicast_tree_depth≤4
- 表项预算：per‑PE max_entries（如 1024）与内存预算（如 32–64KB）；超限时触发更强压缩或回退（提示统计）

## 7. 导出格式（SST‑SnnDL 友好）
建议在 experimental_features/neuron_mapping_framework/export/ 生成以下文件：
- mapping.json：神经元 → {pe, core, global_id}
- routing_tables.json：pe → [ {key, mask, routes, priority} ]
- multicast_groups.json：组定义与统计（可选）
- routing_stats.csv：生成统计（表项总数、平均跳数、压缩比、热点链路等）

示例（routing_tables.json 片段）：
```json
{
  "pe_12": {
    "entries": [
      {"key": 305419896, "mask": 4294967295, "routes": ["EAST"], "priority": 10},
      {"key": 305419904, "mask": 4294967280, "routes": ["NORTH", "LOCAL"], "priority": 8}
    ]
  }
}
```
说明：routes 取值集合与 `RouteDirection` 一致（LOCAL/NORTH/SOUTH/EAST/WEST/UP/DOWN）。

## 8. 与 SST‑SnnDL 的集成方式
- 单独构建：保持 mapping framework 独立；仅通过导出产物喂给 SnnDL 配置生成器
- 对接位置：沿用现有 `snnDL_mapping_integration_design.md` 的“预处理式映射优化”路径
  - 由转换器将 mapping.json/routing_tables.json 转写为 SnnDL Python 脚本中的 PE 参数与网络连线
  - 可在 SnnDL 仿真脚本中增加一个可选参数 `--use_mapping_artifacts` 指向导出目录（不改核心接口）
- 统计联动：仿真结束后可回抛统计至 `sst_visualizer/` 工具链

## 9. 正确性与验证
- 结构校验：
  - `RoutingTable::validateTable()` 每表无冲突/无悬空
  - `DistributedRoutingTable::validateGlobalConsistency()` 端到端可达
- 功能回归：
  - 单播/多播混合网络的端到端路径还原，与逻辑连通关系一致
- 预算约束：
  - 表项数不超过预算；如超过，自动提高压缩等级并生成告警
- 统计输出：
  - 跳数分布、链路利用率、热点度 Top‑K、压缩比与多播效率

## 10. 渐进式落地计划（TODO）
1) 导出管线与格式
   - [ ] 在 `RoutingTableGenerator` 增补 `exportArtifacts(dir, formats)`（或复用现有 export API）
   - [ ] 产出 mapping.json / routing_tables.json / routing_stats.csv
2) 路由生成与压缩
   - [ ] 打通 `generateShortestPathRouting`/`generateMulticastOptimizedRouting`
   - [ ] 实装 `performPrefixAggregation`/`performRouteMerging`/`performMulticastTreeCompression`
3) 路由感知分割
   - [ ] `RoutingAwareGraphPartitioningStrategy` 中开启路由项；暴露权重参数
4) 校验与可视化
   - [ ] 分布式一致性校验 + Graphviz/JSON 导出（供 `sst_visualizer` 使用）
5) SnnDL 集成试运行
   - [ ] 基于 `snnDL_tests/basic_tests` 的最小网络贯通；观察表项与延迟
6) 进阶（可选，不破坏兼容）
   - [ ] 组键模式（MULTICAST 事件携带 group_id，分配“组键”以获得跨 PE 的前缀聚合），需要运行时在特定路径生成组播包；默认关闭
   - [ ] K 最短路 + 链路热度惩罚的在线重平衡（仍在编译期预计算，运行时不改表）

## 11. 风险与缓解
- 表项爆炸：在高扇出网络中条目数上升 → 默认启用前缀聚合与多播树压缩；设置硬阈值并降级
- 键位耦合：32 位布局对“跨 PE 前缀聚合”不友好 → 通过“组键模式”作为增量增强，默认不启用
- 与 SnnDL 对接细节：严格走“预处理+导入”路径，避免改 SST 组件；若需要新参数，新增为可选项并保持默认关闭

## 12. 开发与测试清单
- 构建与运行：
  - `cd experimental_features/neuron_mapping_framework && make all && make run-demo && make run-test`
  - 导出：`make run-benchmark` 同时输出路由与统计至 export/
- 编译自检：`make test-compile`（保持新增 C++ 改动可编译）
- 评估指标：
  - 256→16 PEs、512→32 PEs：目标通信成本下降 ≥96%（对比均匀映射）
  - 路由表压缩比 ≥50%，平均跳数降低，热点链路削峰

—— 以上方案完成后即可进入代码实现阶段；实现时严格遵循“先编译验证、再最小化集成”的准则。
