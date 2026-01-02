# Phase 6：通用 Workload 插件化（A 路线）+ Stream（通信+内存）验证工作负载推进计划

> 目标：在 **不破坏现有 SNN 回归口径** 的前提下，把 `control/SnnPESubComponent` 演进为“通用 core 壳（CoreShell）”，使未来 compute/workload 可以完全不是 SNN；并落地首个非 SNN workload：**streaming read/write + RawBytes 通信**（带 read-after-write 校验）。

---

## 0) 约束与成功标准（必须满足）

### 0.1 硬约束（Hard Constraints）
- **兼容性优先**：默认配置仍按 SNN 路径运行；`sst_dram_si/test_mesh_4x4.py` 作为主模板不应被破坏。
- **安装生效**：任何 C++ 变更必须 `make -j4 && make install` 后运行仿真才会生效。
- **确定性**：同配置同 seed 重跑（至少 2 次）关键统计必须一致（SNN 路径 + stream 路径都要做到）。
- **验证模式选择：B**：stream workload 必须支持 **read-after-write 校验**，且默认验收要求 `verify_fail_total==0`（可允许抽样降低开销，但默认要能证明正确性）。

### 0.2 成功标准（Acceptance）
#### A) SNN 路径（不变性）
- 10us / 100us 回归：`essential_summary_mesh.json` 的关键字段与 baseline **完全一致**（同 seed）。

#### B) Stream 路径（新能力）
- 10us / 100us：`stream_mem_verify_fail_total == 0`
- `stream_mem_bytes_written_total > 0` 且 `stream_mem_bytes_read_total > 0`
- `stream_pkt_sent_total > 0` 且 `stream_pkt_recv_total > 0`（至少在 2 节点场景成立）
- 同 seed 重跑两次，所有 stream 统计计数 **完全一致**（确定性）

---

## 1) 终局架构（Platform vs Workload）

### 1.1 Platform（平台核，稳定）
- NoC：`services/noc/*` 只处理 `events/NocPacketEvent`（payload-agnostic）
- Memory：`api/IMemoryAccess` + `services/memory/StandardMemAccess` 只处理 `addr↔bytes`
- 组件装配：`components/MultiCorePE` 只做端口/clock/装配与队列调度，不承载 workload 语义

### 1.2 Workload（语义插件，可替换）
- `workload=snn`：现有 SNN 语义（GAS + synapse + spike）作为一个 workload 插件保留
- `workload=stream`：新的非 SNN streaming 负载（通信 + 内存），完全不依赖 Spike/Neuron/GAS 语义

---

## 2) 关键缺口（为什么必须先打通“非 Spike 包”）

当前投递链路的核心问题是：packet → core 的最后一步仍默认“当 Spike 解码”。
要实现非 SNN workload，必须先做到：
- NoC 仍只投递 `NocPacketEvent`
- CoreShell 能接收 **`kind != Spike`** 的 packet，并将其交给 workload（而不是硬解码为 Spike）

因此 Phase6 的第一个可执行里程碑是：**非 Spike packet 的 demux（分流）与投递**。

---

## 3) 接口与协议（为扩展性写死边界）

### 3.1 CoreShell 的最小新增调用面（优先不破坏现有接口）
- 在 `api/SnnCoreAPI` 增加一个新虚函数（默认空实现）：
  - `bool deliverPacket(NocPacketEvent* pkt)`：用于 `NocPacketKind::RawBytes/Control` 等非 spike 包
    - 返回 `true`：core 接管 `pkt` 生命周期（负责 `delete`）
    - 返回 `false`：core 未处理，上层负责 `delete`
- 保持原有 `deliverSpike(SpikeEvent*)` 语义不变（SNN 继续走快路径）

**理由**：这是最小侵入的“扩展点”，不会强迫现有 SNN 路径改名/改类型，同时为 future workload 打开入口。

### 3.2 Packet demux 的归属（避免污染 synapse 域）
- `services/synapse/route/SpikePacketBridge` 仅负责 spike 编解码（保持边界清晰）
- `components/MultiCorePE`（或其装配的 endpoint dispatcher）负责依据 `NocPacketEvent.kind` 分流：
  - `kind==Spike` → `SpikePacketBridge::deliverPacketToEndpoint()`
  - `kind!=Spike` → `core->deliverPacket(pkt)`（或丢弃 + fail-fast）

### 3.3 Stream workload 的 payload 协议（可版本化）
使用 `NocPacketKind::RawBytes`，payload 的前缀固定为：
- `magic`（8B）：例如 `"SNNDLSTR"`
- `version`（u16）
- `msg_type`（u16）：DATA / ACK / CTRL
- `seq`（u32）
- `src_node/src_core/dst_node/dst_core`（可选）
- `payload_len`（u32）
- `crc32`（u32）
- `payload_bytes[...]`

**原则**：收包端若 magic/version 不匹配，直接计数并丢弃（或 strict 模式 fatal），避免 silent wrong。

---

## 4) Stream workload：行为定义（通信 + 内存，B 校验）

### 4.1 内存 streaming（read-after-write）
每个 core 维护一个确定性的地址生成器（由 seed + core_id 派生），循环遍历：
- 地址区间：`[stream_base_addr, stream_base_addr + stream_region_bytes)`
- 访问步长：`stream_stride_bytes`（默认 64）
- 请求大小：`stream_req_bytes`（默认 64，避免大读触发 cache payload 问题）

**B 验证模式（默认）**：
1) 先发起 write：写入 pattern（由 `seed + addr + seq` 生成）
2) write 完成后对同地址发起 read
3) readResp 到达时逐字节校验 pattern（允许抽样：例如每 16B 取 1B，但默认建议全量）

并发控制：
- `stream_max_outstanding` 限制在途请求数量（避免无界 pending）

### 4.2 通信 streaming（RawBytes）
每 `comm_period_cycles` 发送一条 DATA 包到指定目的（先固定，再扩展模式）：
- `dst_node`：可配置为固定节点/邻居/环形 next
- `dst_endpoint`：可配置为固定 core（或对端 core0）

接收端：
- 校验 magic/version/crc
- 统计计数：recv_total / bad_crc_total / unknown_total
- 可选 ACK：回送 ACK(seq) 以验证双向路径

---

## 5) 分阶段推进路线（每步可回归、可回退）

> 每个阶段都必须：`make -j4 && make install` + 10us → 100us 回归（SNN 不变 + stream 新能力）。

### Phase 0：准备与记录（不改行为）
1) 记录当前 SNN baseline run_dir（100us，固定 seed）
2) 创建 Phase6 的 OpenSpec change（建议 change-id：`add-universal-workload-stream`），写清 Why/What/Impact 与验收口径
3) 新建开发分支（只在 SnnDL 子模块内）

### Phase 1：非 Spike packet 投递链路打通（最小侵入）
1) 扩展 `api/SnnCoreAPI`：新增 `deliverPacket(NocPacketEvent*)`（默认空实现）
2) `MultiCorePE` 的 endpoint 投递处按 `NocPacketEvent.kind` 分流：
   - spike：继续走 `SpikePacketBridge`
   - 非 spike：调用 `core->deliverPacket(pkt)`；若返回 false 则由 `MultiCorePE` 回收 `pkt`
3) `SnnPESubComponent` 临时实现 `deliverPacket`：仅统计 + 丢弃（不影响 SNN）

**验收**：
- SNN 10us/100us 回归全等（同 seed）
- 注入一个 `RawBytes` 包（可用现有 test 注入点）能被 core 计数到

### Phase 2：在 `SnnPESubComponent` 内实现最小 stream workload（先不抽插件）
1) 新增参数：
   - `workload_impl = snn|stream`（默认 snn）
   - stream/comm 参数（见第 4 节）
   - **保持脚本不改动**：当 Params 未显式提供 `workload_impl` 时，允许通过环境变量切换：
     - `SNNDL_WORKLOAD_IMPL=stream`
2) 当 `workload_impl=stream`：
   - 不启用/不依赖 synapse/route/weights/GAS（或保持构造但不进入主逻辑）
   - `onClockTick` 中推进 stream 状态机：发起 write→read→verify
   - 使用 `IMemoryAccess`（来自 `StdMemEndpoint::memoryAccess()`）发起读写
   - 使用 `INocTransport` 发包（构造 `NocPacketEvent(kind=RawBytes)`）
3) 为 stream 增加统计字段（独立命名空间：`stream_*`）
   - 重要：在 `SNNDL_WORKLOAD_IMPL=stream` 下必须禁用会污染负载的 SNN-only 事务：
     - `components/WeightLoader` 自动 no-op（避免覆盖/污染 stream 校验区）
     - `components/MultiCorePE` 自动关闭 StepActivation 注入（避免 Step/GAS 驱动产生 spikes/mem 噪声）
   - 补充（收敛一致性）：stream 选择来源优先 `Params.workload_impl`，其次环境变量；避免“仅设置 Params 但子组件仍按 SNN 行为运行”。
     - `services/synapse/stdmem/StdMemEndpoint` 在 stream 下强制配置 `StandardMemAccess` 为 non-cacheable（纯内存语义）
     - `components/MultiCorePE` 在 stream 下禁用 StepActivation（支持 Params 或 env）

**验收**：
- stream：10us/100us `verify_fail_total==0` 且收发/读写计数非零
- 同 seed 重跑两次 stream 结果全等
- SNN 仍可回归（不要求与 stream 同脚本同跑，但必须“默认 snn”不变）

### Phase 3：抽取 Workload 插件接口（为“compute 非 SNN”铺路）
1) 新增 `api/ICoreWorkload` + `createWorkloadByName()`
2) 把 Phase2 的 stream 逻辑迁到 `workloads/stream/*`（或 `services/workload/stream/*`）
3) `SnnPESubComponent` 变为 CoreShell：
   - 负责装配：noc/memory/stats/timers
   - 负责分发：deliverSpike/deliverPacket → workload

**验收**：
- stream 与 SNN 都按 workload_impl 路径稳定回归

**当前落地状态（Phase3 子集已完成）**：
- 已把 Phase2 的 stream 逻辑下沉为独立类：`services/workload/stream/StreamWorkload.{h,cc}`
- 已新增 workload 插件接口与工厂：`api/ICoreWorkload.h` + `api/CoreWorkloadFactory.h`（实现：`services/workload/CoreWorkloadFactory.cc`）
- `services/workload/stream/StreamWorkload` 已实现 `ICoreWorkload`
- `control/SnnPESubComponent` 在 `workload_impl=stream` 时通过工厂创建并委托 workload（clockTick/deliverPacket），默认 SNN 行为不变
- env 读取入口统一并做一次性缓存（避免热路径反复 `getenv`）：`api/WorkloadConfig.h`
- 尚未把 SNN 主链路迁为 `workload=snn`：当前仍保持“默认 SNN 快路径在 CoreShell 内”的结构，确保回归口径与性能稳定

### Phase 4：把 SNN 迁成 `workload=snn`（完成通用壳）
1) 将现有 GAS/synapse/compute glue 下沉到 `workloads/snn/*`
2) CoreShell 中移除 SNN 专用成员，留下纯“控制/装配壳”

**验收**：
- SNN 指标与 baseline 完全一致
- stream 指标稳定一致

---

## 6) 测试与回归命令（统一口径）

### 编译安装
```bash
cd "sst_workspace/sst-elements/src/sst/elements/SnnDL"
make -j4
make install
```

### SNN 回归（模板方式）
```bash
cd "sst_dram_si"
export "MESH_SIM_TIME=10us"
./tools/run_mesh_with_time.sh

export "MESH_SIM_TIME=100us"
./tools/run_mesh_with_time.sh
```

### Stream 回归（建议新增模板开关）
- 保持脚本不变：通过环境变量切换 workload：
```bash
cd "sst_dram_si"
export "SNNDL_WORKLOAD_IMPL=stream"
export "MESH_SIM_TIME=10us"
./tools/run_mesh_with_time.sh

export "MESH_SIM_TIME=100us"
./tools/run_mesh_with_time.sh
```

---

## 7) 风险与回退

- 风险：把 packet 分流逻辑写错导致 spike 丢失或双重释放
  - 缓解：所有 `NocPacketEvent*` 的 ownership 必须明确（谁 delete），并在 debug 模式下对非 spike 包走 strict/fail-fast
- 风险：stream 读写影响 SNN 权重地址空间
  - 缓解：Phase2 采用两层保护：
    - `WeightLoader` 在 stream 下 no-op（避免权重写入覆盖 stream 校验区）
    - stream 的 Read 请求通过 `F_NONCACHEABLE` + GatherBufferIF passthrough 走“纯内存”路径（不经过 GAS coalesce/SRAM cache）
- 回退：每个 phase 以最小 patch 落地，必要时可按 commit/阶段逐步回退；严禁依赖“掩盖问题”的 fallback（例如 `use_event_weight_fallback`）。
