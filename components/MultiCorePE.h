// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// MultiCorePE.h: 真正的多核脉冲神经网络处理单元头文件
//

#ifndef _MULTICOREPE_H
#define _MULTICOREPE_H

#include <sst/core/component.h>
#include <sst/core/subcomponent.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <sst/core/params.h>
#include <sst/core/event.h>
#include <sst/core/clock.h>
#include <sst/core/interfaces/simpleNetwork.h>
#include <sst/core/interfaces/stdMem.h>
#include <sst/core/shared/sharedArray.h>

#include <vector>
#include <string>
#include <array>
#include <cstdint>
#include <map>
#include <memory>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <random>
#include <limits>

#include "events/SpikeEvent.h"
#include "events/NocPacketEvent.h"
#include "SnnInterface.h"
#include "IPeAggregation.h"
#include "CoreShellAPI.h"
#include "../api/IDmaSchedulerProvider.h"
#include "../api/ILocalStorageProvider.h"
#include "../api/IPePodSharedMetadataProvider.h"
#include "../api/IPeWeightObjectPlaneProvider.h"
#include "../api/GlobalNeuronLayout.h"
#include "platform/noc/OptimizedInternalRing.h"
#include "snn/stimulus/ExternalSpikeInputSubsystem.h"
#include "snn/stimulus/StepActivationSubsystem.h"
#include "platform/noc/NocSubsystem.h"
#include "snn/synapse/route/SpikePacketBridge.h"
#include "snn/synapse/weights/WeightMemorySubsystem.h"
#include "platform/stats/IWorkloadStatsModule.h"
#include "../api/IGasStageSink.h"

namespace SST {
namespace SnnDL {

// 前置声明
class MultiCoreController;
class SnnNetworkAdapter;
class NocPacketEvent;
class LocalStorageHierarchyController;
class PodMetadataObjectPlane;
class PodOwnerServiceTable;
class PeLocalServiceObjectTable;
class PeWeightObjectPlane;

// RingMessage和RingMessageType现在定义在OptimizedInternalRing.h中

/**
 * @brief 处理单元状态
 */
struct ProcessingUnitState {
    int unit_id;
    int neuron_id_start;
    int neuron_count;
    bool is_active;
    uint64_t spikes_processed;
    uint64_t neurons_fired;
    double utilization;
    
    ProcessingUnitState() : unit_id(-1), neuron_id_start(0), neuron_count(0), 
                           is_active(false), spikes_processed(0), neurons_fired(0),
                           utilization(0.0) {}
};

/**
 * @brief 真正的多核处理单元组件
 * 
 * 集成多个ProcessingUnit、共享L2缓存、内部互连网络
 */
class MultiCorePE : public SST::Component,
                    public IPeAggregation,
                    public IWorkloadStatRegistrar,
                    public IDmaSchedulerProvider,
                    public ILocalStorageProvider,
                    public IPePodSharedMetadataProvider,
                    public IPeWeightObjectPlaneProvider {
public:
    // ELI注册信息
    SST_ELI_REGISTER_COMPONENT(
        MultiCorePE,                    // 类名
        "SnnDL",                       // 库名  
        "MultiCorePE",                 // 组件名
        SST_ELI_ELEMENT_VERSION(1,0,0), // 版本
        "真正的多核脉冲神经网络处理单元",  // 描述
        COMPONENT_CATEGORY_PROCESSOR    // 类别
    )

    // 参数文档
    SST_ELI_DOCUMENT_PARAMS(
        {"clock",            "多核PE内部时钟频率", "1GHz"},
        {"num_cores",        "处理单元数量", "4"},
        {"neurons_per_core", "每个处理单元的神经元数量", "64"},
        {"l2_cache_size",    "共享L2缓存大小", "256KB"},
        {"l2_associativity", "L2缓存关联度", "8"},
        {"l2_cache_line_size", "L2缓存行大小", "64B"},
        {"internal_ring_latency", "内部环形网络延迟", "1ns"},
        {"verbose",          "日志详细级别", "0"},
        {"node_id",          "网络节点ID", "0"},
        {"base_addr",        "全局内存基地址", "0"},
        {"per_core_stride",  "每个核心权重区域的地址跨度（字节）", "0"},
        {"sim_stop_ns",      "组件主控结束仿真（纳秒）。>0时注册为primary并在达到该时间点时OKToEndSim", "0"},
        {"weights_file",     "权重文件路径", ""},
        {"enable_numa",      "启用NUMA优化", "1"},
        {"workload_impl",    "workload实现选择(snn|stream|traffic|tensor). 为空则回退 env:SNNDL_WORKLOAD_IMPL", ""},
        {"workload_stats_modules", "workload统计模块(逗号分隔). 为空则按workload_impl自动选择", ""},
        {"exec_mode",        "执行模式提示(gas|naive_raw). 仅用于实验可观测性，不改变行为", "gas"},
        {"dma_enable", "启用 PE 级共享 DMA 读调度（仅 SNN workload）", "0"},
        {"dma_bytes_per_cycle", "PE 级 DMA 每周期总发射字节预算（0=不限）", "0"},
        {"dma_read_engines", "PE 级 DMA 每周期 read burst 发射数（0=不限）", "0"},
        {"dma_max_inflight", "PE 级 DMA 允许的最大在途 burst 数（0=不限）", "0"},
        {"dma_queue_depth", "PE 级 DMA 主队列深度（0=不限）", "0"},
        {"dma_overflow_policy", "PE 级 DMA 队列溢出策略：block/fail_fast", "block"},
        {"dma_burst_bytes", "PE 级 DMA 单次 burst 上限字节数（0=自动）", "0"},
        {"dma_setup_cycles", "PE 级 DMA 请求 setup 延迟（周期）", "0"},
        {"dma_channels", "PE 级 DMA 通道数", "1"},
        {"dma_channel_bytes_per_cycle", "单通道 DMA 每周期发射字节预算（0=不限）", "0"},
        {"dma_channel_interleave_bytes", "DMA 通道地址交织粒度（字节）", "256"},
        {"dma_stage_budget_scale_gather_p0", "Gather 阶段 P0 预算缩放（permille）", "1000"},
        {"dma_stage_budget_scale_gather_p1", "Gather 阶段 P1 预算缩放（permille）", "1000"},
        {"dma_stage_budget_scale_gather_p2", "Gather 阶段 P2 预算缩放（permille）", "1000"},
        {"dma_stage_budget_scale_gather_p3", "Gather 阶段 P3 预算缩放（permille）", "200"},
        {"dma_stage_budget_scale_apply_p0", "Apply 阶段 P0 预算缩放（permille）", "1000"},
        {"dma_stage_budget_scale_apply_p1", "Apply 阶段 P1 预算缩放（permille）", "1000"},
        {"dma_stage_budget_scale_apply_p2", "Apply 阶段 P2 预算缩放（permille）", "0"},
        {"dma_stage_budget_scale_apply_p3", "Apply 阶段 P3 预算缩放（permille）", "200"},
        {"dma_stage_budget_scale_scatter_p0", "Scatter 阶段 P0 预算缩放（permille）", "1000"},
        {"dma_stage_budget_scale_scatter_p1", "Scatter 阶段 P1 预算缩放（permille）", "1000"},
        {"dma_stage_budget_scale_scatter_p2", "Scatter 阶段 P2 预算缩放（permille）", "250"},
        {"dma_stage_budget_scale_scatter_p3", "Scatter 阶段 P3 预算缩放（permille）", "200"},
        {"dma_stage_budget_scale_idle_p0", "Idle 阶段 P0 预算缩放（permille）", "1000"},
        {"dma_stage_budget_scale_idle_p1", "Idle 阶段 P1 预算缩放（permille）", "1000"},
        {"dma_stage_budget_scale_idle_p2", "Idle 阶段 P2 预算缩放（permille）", "250"},
        {"dma_stage_budget_scale_idle_p3", "Idle 阶段 P3 预算缩放（permille）", "200"},
        {"local_storage_enable", "启用 PE 级片上本地存储层级控制器（Phase A 骨架，仅 SNN workload）", "0"},
        {"pe_internal_cpe_enable", "启用 PE 内部 C/P/E 三层 scope 实验架构总开关（默认关闭）", "0"},
        {"pe_internal_pod_enable", "启用 P-scope pod 对象注册与隔离（默认关闭）", "0"},
        {"pe_internal_pod_count", "显式 pod 数；0=由 pod_size 或 num_cores 推导", "0"},
        {"pe_internal_pod_size", "显式每个 pod 覆盖 core 数；0=不按 size 推导", "0"},
        {"pe_internal_pod_metadata_enable", "启用 pod metadata object plane 对象注册", "0"},
        {"pe_internal_pod_metadata_capacity_bytes", "pod metadata store 容量", "0"},
        {"pe_internal_pod_metadata_banks", "pod metadata store bank 数", "1"},
        {"pe_internal_pod_owner_enable", "启用 pod owner table 对象注册", "0"},
        {"pe_internal_pod_owner_entries", "pod owner table entry 数", "0"},
        {"pe_internal_pod_owner_entry_bytes", "pod owner table 每 entry 字节数", "16"},
        {"pe_internal_pod_join_enable", "启用 pod join table 对象注册", "0"},
        {"pe_internal_pod_join_entries", "pod join table entry 数", "0"},
        {"pe_internal_pod_join_entry_bytes", "pod join table 每 entry 字节数", "16"},
        {"pe_internal_pod_ready_enable", "启用 pod ready table 对象注册", "0"},
        {"pe_internal_pod_ready_entries", "pod ready queue 深度", "0"},
        {"ls_state_enable", "显式 state store 对象开关（未设置时兼容 state_sram_enable）", "0"},
        {"ls_state_capacity_bytes", "显式 state store 容量（未设置时兼容 state_sram_capacity_bytes）", "0"},
        {"ls_state_banks", "显式 state store bank 数（未设置时兼容 state_sram_banks）", "16"},
        {"ls_state_read_ports", "显式 state store 读端口数", "1"},
        {"ls_state_write_ports", "显式 state store 写端口数", "1"},
        {"ls_state_update_ports", "显式 state store 更新端口数", "1"},
        {"ls_weight_idx_enable", "显式 weight idx store 对象开关（未设置时兼容 weight_idx_sram_enable）", "0"},
        {"ls_weight_idx_capacity_bytes", "显式 weight idx store 容量（未设置时兼容 weight_idx_sram_capacity_bytes）", "0"},
        {"ls_weight_value_enable", "显式 weight value store 对象开关（未设置时兼容 weight_l0_sram_enable）", "0"},
        {"ls_weight_value_capacity_bytes", "显式 weight value store 容量（未设置时兼容 weight_l0_sram_capacity_bytes）", "0"},
        {"ls_activation_ingress_enable", "显式 activation ingress queue 对象开关", "0"},
        {"ls_activation_ingress_entries", "显式 activation ingress queue 深度", "0"},
        {"ls_activation_core_enable", "显式 activation per-core queue 对象开关", "0"},
        {"ls_activation_core_entries", "显式 activation per-core queue 深度", "0"},
        {"ls_acc_enable", "显式 accumulator store 对象开关", "0"},
        {"ls_acc_capacity_bytes", "显式 accumulator store 容量", "0"},
        {"ls_rf_enable", "显式 register file 对象开关", "0"},
        {"ls_rf_entries", "显式 register file entry 数", "0"},
        {"ls_rf_entry_bytes", "显式 register file entry 字节数", "4"},
        {"v_thresh",         "触发脉冲的膜电位阈值", "1.0"},
        {"v_reset",          "脉冲发放后膜电位重置值", "0.0"},
        {"v_rest",           "静息膜电位", "0.0"},
        {"tau_mem",          "膜电位泄漏时间常数(ms)", "20.0"},
        {"t_ref",            "不应期时长(时钟周期)", "2"},
        {"enable_test_traffic", "是否启用网络测试流量", "0"},
        {"test_traffic_packet_kind", "测试流量包类型(spike|spikekey)", "spike"},
        {"test_target_node", "测试流量的目标节点ID", "0"},
        {"test_period",      "测试流量发送周期(周期数)", "100"},
        {"test_spikes_per_burst", "每次周期性发送的测试脉冲数量", "4"},
        {"test_weight",      "测试脉冲权重", "0.2"},
        {"use_optimized_ring", "使用优化的环形网络实现(1)或原始实现(0)", "1"},
        // 时间窗口化统计（Batch‑B，可选，默认关闭）
        {"window_stats_enable", "启用时间窗口化统计(1=启用,0=关闭)", "0"},
        {"window_us", "统计窗口长度（微秒）", "20"},
        {"window_csv", "窗口化统计输出CSV路径（为空则不输出）", ""},
        {"diag_fire_log", "启用发放统计诊断日志(1=开启)", "0"},
        {"manual_gas_gather_cycles", "已弃用：旧 manual drive 兼容参数（当前无效；保留仅为兼容）", "200"},
        {"step_activation_enable", "启用步级随机激活", "0"},
        {"step_activation_fraction", "步级随机激活伯努利概率", "0.0"},
        {"step_activation_fanout", "步级随机激活fanout", "0"},
        {"step_activation_seed", "步级随机激活随机种子", "0xdecafbad"},
        {"step_activation_period_cycles", "固定周期触发step随机激活(>0启用; 0=沿用BeginGather触发)", "0"},
        {"step_activation_event_weight", "步级注入事件权重（非严格模式备用）", "0.0"},
        {"step_activation_pre_pattern", "step随机激活 pre 选择模式：bernoulli/clustered", "bernoulli"},
        {"step_activation_pre_cluster_len", "clustered模式：连续 pre 段长度（neuron）；0=自动(64)", "0"},
        {"step_reset_mem_each_step", "步末复位膜电位", "0"},
        {"step_activation_use_bcsr_routes", "随机激活是否使用BCSR路由表", "0"},
        {"step_seed_only_mode", "实验模式：每个step仅由随机激活驱动（自动复位膜状态并禁止运行时发放跨step反馈）", "0"},
        {"sentinel_enable", "启用 sentinel 调试输出(0/1)", "0"},
        {"progress_log_interval_ns", "每隔N纳秒打印一次进度(0=关闭)", "0"},
        {"progress_log_node", "仅对指定node_id打印进度(-1=所有)", "-1"},
        {"step_diag_cap", "步级诊断采样上限（>0 启用）", "0"},
        {"step_diag_enable", "启用步级诊断(0/1)", "0"},
        {"step_activation_bcsr_template", "BCSR权重文件模板({core})", ""},
        {"step_activation_bcsr_rows_per_core", "BCSR每核行数", "0"},
        {"step_activation_bcsr_br", "BCSR块行大小", "16"},
        {"step_activation_bcsr_bc", "BCSR块列大小", "16"},
        {"step_activation_bcsr_idx_bytes", "BCSR colidx字节数", "2"},
        {"step_activation_bcsr_val_bytes", "BCSR权重字节数", "4"},
        {"step_activation_bcsr_rowptr_offset", "BCSR rowptr段偏移", "0"},
        {"step_activation_bcsr_colidx_offset", "BCSR colidx段偏移", "0"},
        {"step_activation_bcsr_blockdata_offset", "BCSR blockdata段偏移", "0"},
        {"step_activation_bcsr_blockids_offset", "BCSR blockids段偏移", "0"},
        {"step_activation_bcsr_stride_bytes", "BCSR per-core stride字节数", "0"},
        {"step_activation_bcsr_weight_epsilon", "判定非零连边的权重阈值", "0.0"},
        {"step_activation_log_enable", "启用步级激活BCSR路由构建日志(0/1)", "0"}
        ,
        {"global_step_sync_enable", "启用全局 Step/GAS barrier 同步(0/1)", "0"}
        ,
        {"global_step_done_policy", "全局 step 完成判定策略：endscatter/drain/quiescent/fixed_cycles（默认endscatter）", "endscatter"}
        ,
        {"global_step_quiescent_min_cycles", "quiescent 模式：每 step 至少等待的周期数（避免同拍开始即完成）", "1"}
        ,
        {"global_step_drain_min_cycles", "drain 模式：判定无在途事务后仍需保持静默的周期数（覆盖网络最坏延迟；语义等价对比推荐）", "200"}
        ,
        {"loader_done_key", "SharedArray key toggled by WeightLoader upon completion（用于 Step-limited 下延迟 PE_READY）", ""}
        ,
        {"loader_done_timeout_cycles", "等待 loader_done_key 超时后降级发送 PE_READY 的周期数（0=禁用）", "0"}
        ,
        {"global_step_ready_delay_cycles", "global_step_sync_enable=1 时：在 loader_done 后额外等待 N 周期再发送 PE_READY（避免 naive_* 在 rowptr 未就绪时积压）", "0"}
    )

    // 子组件槽位文档
    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS(
        {"core0", "CoreShell 子核心0", "SST::SnnDL::CoreShellAPI"},
        {"core1", "CoreShell 子核心1", "SST::SnnDL::CoreShellAPI"},
        {"core2", "CoreShell 子核心2", "SST::SnnDL::CoreShellAPI"},
        {"core3", "CoreShell 子核心3", "SST::SnnDL::CoreShellAPI"},
        {"core4", "CoreShell 子核心4", "SST::SnnDL::CoreShellAPI"},
        {"core5", "CoreShell 子核心5", "SST::SnnDL::CoreShellAPI"},
        {"core6", "CoreShell 子核心6", "SST::SnnDL::CoreShellAPI"},
        {"core7", "CoreShell 子核心7", "SST::SnnDL::CoreShellAPI"},
        {"l2_cache", "共享L2缓存", "SST::MemHierarchy::Cache"},
        {"external_nic", "外部网络接口", "SST::SnnDL::SnnInterface"}
    )

    // 端口文档 
    SST_ELI_DOCUMENT_PORTS(
        {"external_spike_input",  "外部脉冲输入端口", {"SnnDL.SpikeEvent"}},
        {"external_spike_output", "外部脉冲输出端口", {"SnnDL.SpikeEvent"}},
        {"gas_step_ctrl", "全局 Step/GAS 同步控制器端口（可选）", {"SnnDL.GasStepBarrierEvent", "SnnDL.GatingDecisionEvent"}},
        {"loader_done", "WeightLoader完成事件（可选，桥接跨rank loader_done_key）", {"SnnDL.LoaderDoneEvent"}},
        {"network", "网络连接端口（用于direct_link模式）", {"SnnDL.SpikeEvent", "SimpleNetwork"}},
        {"north", "北向网络连接端口（网格拓扑）", {"SnnDL.SpikeEvent"}},
        {"south", "南向网络连接端口（网格拓扑）", {"SnnDL.SpikeEvent"}},
        {"east", "东向网络连接端口（网格拓扑）", {"SnnDL.SpikeEvent"}},
        {"west", "西向网络连接端口（网格拓扑）", {"SnnDL.SpikeEvent"}},
        {"mem_link", "内存层次结构连接端口（保留，通用用途）", {"memHierarchy.MemEventBase"}},
        {"core0_mem", "核心0的内存连接端口", {"memHierarchy.MemEventBase"}},
        {"core1_mem", "核心1的内存连接端口", {"memHierarchy.MemEventBase"}},
        {"core2_mem", "核心2的内存连接端口", {"memHierarchy.MemEventBase"}},
        {"core3_mem", "核心3的内存连接端口", {"memHierarchy.MemEventBase"}},
        {"core4_mem", "核心4的内存连接端口", {"memHierarchy.MemEventBase"}},
        {"core5_mem", "核心5的内存连接端口", {"memHierarchy.MemEventBase"}},
        {"core6_mem", "核心6的内存连接端口", {"memHierarchy.MemEventBase"}},
        {"core7_mem", "核心7的内存连接端口", {"memHierarchy.MemEventBase"}}
    )

    // 统计信息文档
    SST_ELI_DOCUMENT_STATISTICS(
        {"total_spikes_processed", "处理的脉冲总数", "spikes", 1},
        {"inter_core_messages", "核间消息数量", "messages", 1}, 
        {"l2_cache_hits", "L2缓存命中数", "hits", 1},
        {"l2_cache_misses", "L2缓存缺失数", "misses", 1},
        {"memory_requests", "内存请求数", "requests", 1},
        {"avg_core_utilization", "平均核心利用率", "percentage", 1},
	        {"total_neurons_fired", "总神经元发放数", "neurons", 1},
	        {"unique_neurons_fired_total", "至少发放一次的不同神经元数（总）", "neurons", 1},
	        {"external_spikes_sent", "发送的外部脉冲数", "spikes", 1},
	        {"external_spikes_received", "接收的外部脉冲数", "spikes", 1},
            {"riscv_snn_workload_selected", "riscv_snn runtime bridge: constructor/config 路径选择了 riscv_snn workload 的 core 数（PE聚合）", "count", 1},
            {"riscv_snn_firmware_elf_present", "riscv_snn runtime bridge: 配置了非空 firmware ELF 的 core 数（PE聚合）", "count", 1},
            {"riscv_snn_firmware_loaded", "riscv_snn runtime bridge: 已成功装载 firmware 的 core 数（PE聚合）", "count", 1},
            {"riscv_snn_backend_runtime_bridge", "riscv_snn runtime bridge: backend 配置为 runtime_bridge 的 core 数（PE聚合）", "count", 1},
            {"riscv_snn_firmware_started_count", "riscv_snn runtime bridge: firmware 真正开始执行的次数（PE聚合）", "count", 1},
            {"riscv_snn_submitted_commands", "riscv_snn runtime bridge: 软件提交 command 次数（PE聚合）", "count", 1},
            {"riscv_snn_accepted_commands", "riscv_snn runtime bridge: backend 接受 command 次数（PE聚合）", "count", 1},
            {"riscv_snn_completion_visible_count", "riscv_snn runtime bridge: completion 对软件可见次数（PE聚合）", "count", 1},
            {"riscv_snn_completion_consumed_count", "riscv_snn runtime bridge: completion 被软件消费次数（PE聚合）", "count", 1},
            {"riscv_snn_fused_step_completion_count", "riscv_snn runtime bridge: FUSED_STEP 完成次数（PE聚合）", "count", 1},
            {"riscv_snn_fault_count", "riscv_snn runtime bridge: fault 次数（PE聚合）", "count", 1},
            {"riscv_snn_last_completion_status", "riscv_snn runtime bridge: 最近 completion status（PE聚合）", "status", 1},
            {"riscv_snn_last_fault_csr", "riscv_snn runtime bridge: 最近 fault CSR（PE聚合）", "csr", 1},
            {"riscv_snn_backend_runtime_bridge_provider_bound", "riscv_snn runtime bridge: provider ready 的 core 数（PE聚合）", "count", 1},
            {"snn_tx_spike_packets_total", "SNN workload: 发送的Spike包数（PE聚合）", "packets", 1},
            {"snn_tx_spikekey_packets_total", "SNN workload: 发送的SpikeKey包数（PE聚合）", "packets", 1},
            {"snn_tx_spiketilekey_packets_total", "SNN workload: 发送的SpikeTileKey包数（PE聚合）", "packets", 1},
            {"snn_tx_spikekey_v4_packets_total", "SNN workload: 发送的SpikeKey-v4包数（PE聚合）", "packets", 1},
            {"snn_tx_spiketilekey_v4_packets_total", "SNN workload: 发送的SpikeTileKey-v4包数（PE聚合）", "packets", 1},
            {"snn_tx_bundle_v1_packets_total", "SNN workload: 发送的bundle-v1包数（PE聚合）", "packets", 1},
            {"snn_tx_bundle_v2_packets_total", "SNN workload: 发送的bundle-v2包数（PE聚合）", "packets", 1},
            {"snn_tx_bundle_v3_packets_total", "SNN workload: 发送的bundle-v3包数（PE聚合）", "packets", 1},
            {"snn_rx_spike_packets_total", "SNN workload: 接收的Spike包数（PE聚合）", "packets", 1},
            {"snn_rx_spikekey_total", "SNN workload: 接收的SpikeKey包数（PE聚合）", "packets", 1},
            {"snn_rx_spiketilekey_total", "SNN workload: 接收的SpikeTileKey包数（PE聚合）", "packets", 1},
            {"snn_rx_spikekey_v4_total", "SNN workload: 接收的SpikeKey-v4包数（PE聚合）", "packets", 1},
            {"snn_rx_spiketilekey_v4_total", "SNN workload: 接收的SpikeTileKey-v4包数（PE聚合）", "packets", 1},
            {"snn_rx_fastpath_packets_total", "SNN workload: 接收侧fastpath处理包数（PE聚合）", "packets", 1},
            {"snn_rx_fallback_packets_total", "SNN workload: 接收侧fallback处理包数（PE聚合）", "packets", 1},
            {"snn_rx_decode_fail_total", "SNN workload: SpikeKey/SpikeTileKey解码失败数（PE聚合）", "packets", 1},
            {"snn_rx_fastpath_posts_total", "SNN workload: fastpath展开post节点总数（PE聚合）", "posts", 1},
            {"snn_rx_fastpath_accept_total", "SNN workload: fastpath通过门控并接收的post数（PE聚合）", "posts", 1},
            {"snn_rx_fastpath_reject_total", "SNN workload: fastpath门控拒绝的post数（PE聚合）", "posts", 1},
            {"snn_rx_fastpath_edges_recorded_total", "SNN workload: fastpath写入window edge记录数（PE聚合）", "edges", 1},
            {"snn_edge_record_attempt_total", "SNN workload: edge-record 尝试次数（PE聚合）", "edges", 1},
            {"snn_edge_record_commit_total", "SNN workload: edge-record 成功写入次数（PE聚合）", "edges", 1},
            {"snn_edge_record_skip_gate_total", "SNN workload: edge-record 因 gate/backend/WMS 缺失跳过次数（PE聚合）", "edges", 1},
            {"snn_edge_record_skip_stage_total", "SNN workload: edge-record 因 stage 不允许跳过次数（PE聚合）", "edges", 1},
            {"snn_edge_record_skip_capacity_total", "SNN workload: edge-record 因 collector 容量不足跳过次数（PE聚合）", "edges", 1},
            {"snn_edge_record_skip_reject_total", "SNN workload: edge-record 因 compute gating reject 跳过次数（PE聚合）", "edges", 1},
            {"snn_edge_record_fastpath_handler_entry_total", "SNN workload: fastpath edge-record handler 入口次数（PE聚合）", "calls", 1},
            {"snn_edge_record_fastpath_wms_missing_total", "SNN workload: fastpath handler 中 WMS 缺失次数（PE聚合）", "calls", 1},
            {"snn_edge_record_fastpath_backend_not_ready_total", "SNN workload: fastpath handler 中 backend 未就绪次数（PE聚合）", "calls", 1},
            {"snn_edge_record_fastpath_stage_block_total", "SNN workload: fastpath handler 中 stage block 次数（PE聚合）", "calls", 1},
            {"snn_edge_record_process_local_handler_entry_total", "SNN workload: processLocalSpike edge-record handler 入口次数（PE聚合）", "calls", 1},
            {"snn_edge_record_process_local_wms_missing_total", "SNN workload: processLocalSpike handler 中 WMS 缺失次数（PE聚合）", "calls", 1},
            {"snn_edge_record_process_local_backend_not_ready_total", "SNN workload: processLocalSpike handler 中 backend 未就绪次数（PE聚合）", "calls", 1},
            {"snn_edge_record_process_local_stage_block_total", "SNN workload: processLocalSpike handler 中 stage block 次数（PE聚合）", "calls", 1},
            {"snn_edge_record_deliver_window_handler_entry_total", "SNN workload: deliverSpike window edge-record handler 入口次数（PE聚合）", "calls", 1},
            {"snn_edge_record_deliver_window_wms_missing_total", "SNN workload: deliverSpike window handler 中 WMS 缺失次数（PE聚合）", "calls", 1},
            {"snn_edge_record_deliver_window_backend_not_ready_total", "SNN workload: deliverSpike window handler 中 backend 未就绪次数（PE聚合）", "calls", 1},
            {"snn_edge_record_deliver_window_stage_block_total", "SNN workload: deliverSpike window handler 中 stage block 次数（PE聚合）", "calls", 1},
            {"route3d_native_activation_total", "route3d native runtime activation总次数（PE聚合）", "count", 1},
            {"route3d_native_gating_activation_total", "route3d native runtime由gating命中的激活次数（PE聚合）", "count", 1},
            {"route3d_native_direct_activation_total", "route3d native runtime由direct route命中的激活次数（PE聚合）", "count", 1},
            {"route3d_native_unique_sources_total", "route3d native runtime触发过的不同source数量（PE聚合）", "count", 1},
            {"route_native_source_fanout_active", "route authority: native source fanout是否处于激活状态（PE聚合 one-hot）", "count", 1},
            {"route_native_target_synthesis_active", "route authority: native target synthesis是否处于激活状态（PE聚合 one-hot）", "count", 1},
            {"route_bootstrap_dependency_active", "route authority: native route是否依赖bootstrap源（PE聚合 one-hot）", "count", 1},
            {"route_real_synapse_inputs_available", "route authority: real synapse inputs是否可用（PE聚合 one-hot）", "count", 1},
            {"route_native_synapse_source_candidate", "route authority: source-side是否具备native synapse候选资格（PE聚合 one-hot）", "count", 1},
            {"route_source_semantics_authority_legacy_provider", "route authority: source semantics=legacy_provider（PE聚合 one-hot）", "count", 1},
            {"route_source_semantics_authority_legacy_built_routes_3d", "route authority: source semantics=legacy_built_routes_3d（PE聚合 one-hot）", "count", 1},
            {"route_source_semantics_authority_native_3d_route_table", "route authority: source semantics=native_3d_route_table（PE聚合 one-hot）", "count", 1},
            {"route_source_primary_kind_legacy_only", "route authority: source primary kind=legacy_only（PE聚合 one-hot）", "count", 1},
            {"route_source_primary_kind_edges_csv_bootstrap", "route authority: source primary kind=edges_csv_bootstrap（PE聚合 one-hot）", "count", 1},
            {"route_source_primary_kind_legacy_route_tables_bootstrap", "route authority: source primary kind=legacy_route_tables_bootstrap（PE聚合 one-hot）", "count", 1},
            {"route_source_primary_kind_legacy_route_tables_with_real_synapse_inputs", "route authority: source primary kind=legacy_route_tables_with_real_synapse_inputs（PE聚合 one-hot）", "count", 1},
            {"route_source_primary_kind_native_3d_route_table_with_real_synapse_inputs", "route authority: source primary kind=native_3d_route_table_with_real_synapse_inputs（PE聚合 one-hot）", "count", 1},
            {"route_source_primary_kind_real_synapse_inputs_only", "route authority: source primary kind=real_synapse_inputs_only（PE聚合 one-hot）", "count", 1},
            {"route_native_bootstrap_source_edges_csv", "route authority: native bootstrap source=edges_csv（PE聚合 one-hot）", "count", 1},
            {"route_native_bootstrap_source_legacy_route_tables", "route authority: native bootstrap source=legacy_route_tables（PE聚合 one-hot）", "count", 1},
            {"route_topology_mesh_2d", "route authority: route topology=mesh_2d（PE聚合 one-hot）", "count", 1},
            {"route_topology_mesh_3d", "route authority: route topology=mesh_3d（PE聚合 one-hot）", "count", 1},
            {"route_target_semantics_authority_legacy_multicast_fallback", "route authority: target semantics=legacy_multicast_fallback（PE聚合 one-hot）", "count", 1},
            {"route_target_semantics_authority_compat_3d_target_synthesis", "route authority: target semantics=compat_3d_target_synthesis（PE聚合 one-hot）", "count", 1},
            {"route_target_semantics_authority_native_3d_target_synthesis", "route authority: target semantics=native_3d_target_synthesis（PE聚合 one-hot）", "count", 1},
	        // Stream workload（PE聚合，供 essential_summary_mesh 汇总）
	        {"stream_mem_writes_issued_total", "Stream workload: total writes issued（PE聚合）", "requests", 1},
	        {"stream_mem_reads_issued_total", "Stream workload: total reads issued（PE聚合）", "requests", 1},
	        {"stream_mem_bytes_written_total", "Stream workload: bytes written（issued, PE聚合）", "bytes", 1},
	        {"stream_mem_bytes_read_total", "Stream workload: bytes read（issued, PE聚合）", "bytes", 1},
	        {"metadata_lookup_writes_issued_total", "Semantic memory: metadata lookup writes issued（PE聚合）", "requests", 1},
	        {"metadata_lookup_reads_issued_total", "Semantic memory: metadata lookup reads issued（PE聚合）", "requests", 1},
	        {"metadata_lookup_bytes_written_total", "Semantic memory: metadata lookup bytes written（PE聚合）", "bytes", 1},
	        {"metadata_lookup_bytes_read_total", "Semantic memory: metadata lookup bytes read（PE聚合）", "bytes", 1},
	        {"synapse_gather_writes_issued_total", "Semantic memory: synapse gather writes issued（PE聚合）", "requests", 1},
	        {"synapse_gather_reads_issued_total", "Semantic memory: synapse gather reads issued（PE聚合）", "requests", 1},
	        {"synapse_gather_bytes_written_total", "Semantic memory: synapse gather bytes written（PE聚合）", "bytes", 1},
	        {"synapse_gather_bytes_read_total", "Semantic memory: synapse gather bytes read（PE聚合）", "bytes", 1},
	        {"stream_region_writes_issued_total", "Semantic memory: stream region writes issued（PE聚合）", "requests", 1},
	        {"stream_region_reads_issued_total", "Semantic memory: stream region reads issued（PE聚合）", "requests", 1},
	        {"stream_region_bytes_written_total", "Semantic memory: stream region bytes written（PE聚合）", "bytes", 1},
	        {"stream_region_bytes_read_total", "Semantic memory: stream region bytes read（PE聚合）", "bytes", 1},
	        {"writeback_region_writes_issued_total", "Semantic memory: writeback region writes issued（PE聚合）", "requests", 1},
	        {"writeback_region_reads_issued_total", "Semantic memory: writeback region reads issued（PE聚合）", "requests", 1},
	        {"writeback_region_bytes_written_total", "Semantic memory: writeback region bytes written（PE聚合）", "bytes", 1},
	        {"writeback_region_bytes_read_total", "Semantic memory: writeback region bytes read（PE聚合）", "bytes", 1},
	        {"stream_mem_verify_pass_total", "Stream workload: read-after-write 校验通过次数（PE聚合）", "count", 1},
	        {"stream_mem_verify_fail_total", "Stream workload: read-after-write 校验失败次数（PE聚合）", "count", 1},
	        {"stream_pkt_sent_total", "Stream workload: raw-bytes packets sent（PE聚合）", "packets", 1},
	        {"stream_pkt_recv_total", "Stream workload: raw-bytes packets received（PE聚合）", "packets", 1},
	        {"stream_pkt_bad_crc_total", "Stream workload: bad CRC packets（PE聚合）", "packets", 1},
	        {"stream_pkt_bad_magic_total", "Stream workload: bad magic packets（PE聚合）", "packets", 1},
	        {"traffic_tx_batches", "Traffic workload: emitted spike batches（PE聚合）", "batches", 1},
	        {"traffic_tx_pre_total", "Traffic workload: emitted pre neurons（PE聚合）", "neurons", 1},
	        {"traffic_tx_spike_pkts", "Traffic workload: spike packets sent（PE聚合）", "packets", 1},
	        {"traffic_tx_spikekey_pkts", "Traffic workload: spikekey packets sent（PE聚合）", "packets", 1},
	        {"traffic_tx_spiketilekey_pkts", "Traffic workload: spiketilekey packets sent（PE聚合）", "packets", 1},
	        {"traffic_tx_spikekey_v4_pkts", "Traffic workload: v4 spikekey packets sent（PE聚合）", "packets", 1},
	        {"traffic_tx_spiketilekey_v4_pkts", "Traffic workload: v4 spiketilekey packets sent（PE聚合）", "packets", 1},
	        {"traffic_tx_bundle_v1_pkts", "Traffic workload: bundle-v1 packets sent（PE聚合）", "packets", 1},
	        {"traffic_tx_bundle_v2_pkts", "Traffic workload: bundle-v2 packets sent（PE聚合）", "packets", 1},
	        {"traffic_tx_bundle_v3_pkts", "Traffic workload: bundle-v3 packets sent（PE聚合）", "packets", 1},
	        {"traffic_rx_spikekey_v4_total", "Traffic workload: received v4 spikekey packets（PE聚合）", "packets", 1},
	        {"traffic_rx_spiketilekey_v4_total", "Traffic workload: received v4 spiketilekey packets（PE聚合）", "packets", 1},
	        {"traffic_semantic_metadata_lookup_demands_total", "Traffic-driven semantic runtime: metadata lookup demands（PE聚合）", "requests", 1},
	        {"traffic_semantic_synapse_gather_demands_total", "Traffic-driven semantic runtime: synapse gather demands（PE聚合）", "requests", 1},
	        {"traffic_semantic_stream_region_demands_total", "Traffic-driven semantic runtime: stream region demands（PE聚合）", "requests", 1},
	        {"traffic_semantic_writeback_region_demands_total", "Traffic-driven semantic runtime: writeback region demands（PE聚合）", "requests", 1},
	        {"traffic_semantic_tier_local_home_gather_demands_total", "Traffic-driven semantic runtime: tier-local-home gather demands（PE聚合）", "requests", 1},
	        {"traffic_semantic_same_xy_cross_tier_gather_demands_total", "Traffic-driven semantic runtime: same-xy cross-tier gather demands（PE聚合）", "requests", 1},
	        {"traffic_semantic_remote_home_gather_demands_total", "Traffic-driven semantic runtime: remote-home gather demands（PE聚合）", "requests", 1},
	        {"traffic_semantic_tier_local_home_stream_region_demands_total", "Traffic-driven semantic runtime: tier-local-home stream demands（PE聚合）", "requests", 1},
	        {"traffic_semantic_same_xy_cross_tier_stream_region_demands_total", "Traffic-driven semantic runtime: same-xy cross-tier stream demands（PE聚合）", "requests", 1},
	        {"traffic_semantic_remote_home_stream_region_demands_total", "Traffic-driven semantic runtime: remote-home stream demands（PE聚合）", "requests", 1},
	        // Tensor workload（PE聚合，供 essential_summary_mesh 汇总）
        {"tensor_mem_reads_issued_total", "Tensor workload: total reads issued（PE聚合）", "requests", 1},
        {"tensor_mem_writes_issued_total", "Tensor workload: total writes issued（PE聚合）", "requests", 1},
        {"tensor_mem_bytes_read_total", "Tensor workload: bytes read（issued, PE聚合）", "bytes", 1},
        {"tensor_mem_bytes_write_total", "Tensor workload: bytes written（issued, PE聚合）", "bytes", 1},
        {"tensor_mem_read_latency_cycles_total", "Tensor workload: cumulative mem read latency（PE聚合）", "cycles", 1},
        {"tensor_mem_read_latency_cycles_max", "Tensor workload: max mem read latency（PE聚合）", "cycles", 1},
        {"tensor_mem_read_latency_samples_total", "Tensor workload: mem read latency samples（PE聚合）", "count", 1},
        {"tensor_mem_write_latency_cycles_total", "Tensor workload: cumulative mem write latency（PE聚合）", "cycles", 1},
        {"tensor_mem_write_latency_cycles_max", "Tensor workload: max mem write latency（PE聚合）", "cycles", 1},
        {"tensor_mem_write_latency_samples_total", "Tensor workload: mem write latency samples（PE聚合）", "count", 1},
        {"tensor_mem_row_hit_total", "Tensor workload: DRAM row-buffer hits（PE聚合）", "count", 1},
        {"tensor_mem_row_miss_total", "Tensor workload: DRAM row-buffer misses（PE聚合）", "count", 1},
        {"tensor_mem_row_conflict_total", "Tensor workload: DRAM row-buffer conflicts（PE聚合）", "count", 1},
        {"tensor_mem_bank_queue_full_total", "Tensor workload: DRAM bank queue full events（PE聚合）", "count", 1},
        {"tensor_mem_bank_queue_wait_cycles_total", "Tensor workload: DRAM bank queue wait cycles（PE聚合）", "cycles", 1},
        {"tensor_mem_sched_fifo_pick_total", "Tensor workload: DRAM scheduler FIFO picks（PE聚合）", "count", 1},
        {"tensor_mem_sched_frfcfs_pick_total", "Tensor workload: DRAM scheduler FRFCFS picks（PE聚合）", "count", 1},
        {"tensor_mem_cmd_act_total", "Tensor workload: DRAM ACT command proxy count（PE聚合）", "count", 1},
        {"tensor_mem_cmd_pre_total", "Tensor workload: DRAM PRE command proxy count（PE聚合）", "count", 1},
        {"tensor_mem_cmd_rdwr_total", "Tensor workload: DRAM RD/WR command proxy count（PE聚合）", "count", 1},
        {"tensor_mem_row_service_cycles_total", "Tensor workload: DRAM row service cycles（PE聚合）", "cycles", 1},
        {"tensor_mem_refresh_block_cycles_total", "Tensor workload: DRAM refresh blocked cycles（PE聚合）", "cycles", 1},
        {"tensor_mem_proxy_delay_cycles_total", "Tensor workload: proxy timing delay cycles（PE聚合）", "cycles", 1},
        {"tensor_mem_proxy_delay_cycles_max", "Tensor workload: proxy timing max delay cycles（PE聚合）", "cycles", 1},
        {"tensor_mem_bank_active_cycles_total", "Tensor workload: DRAM bank active cycles（PE聚合）", "cycles", 1},
        {"tensor_mem_cmd_queue_slots_total", "Tensor workload: memory command queue slots（PE聚合）", "slots", 1},
        {"tensor_mem_cmd_queue_depth_max", "Tensor workload: memory command queue depth max（PE聚合）", "slots", 1},
        {"tensor_mem_cmd_bus_wait_cycles_total", "Tensor workload: memory command bus wait cycles（PE聚合）", "cycles", 1},
        {"tensor_mem_cmd_bus_bg_switch_total", "Tensor workload: memory command bus bank-group switch count（PE聚合）", "count", 1},
        {"tensor_mem_cmd_issue_total", "Tensor workload: memory command issue count（PE聚合）", "count", 1},
	        {"tensor_compute_cycles_total", "Tensor workload: compute cycles（PE聚合）", "cycles", 1},
	        {"tensor_compute_math_cycles_total", "Tensor workload: compute math cycles（PE聚合）", "cycles", 1},
	        {"tensor_compute_pipeline_cycles_total", "Tensor workload: compute pipeline cycles（PE聚合）", "cycles", 1},
	        {"tensor_mxu_wavefront_cycles_total", "Tensor workload: MXU wavefront cycles（PE聚合）", "cycles", 1},
	        {"tensor_mxu_io_busy_cycles_total", "Tensor workload: MXU IO busy cycles（PE聚合）", "cycles", 1},
	        {"tensor_compute_precision_profile_id", "Tensor workload: compute precision profile id（PE聚合）", "id", 1},
	        {"tensor_mac_ops_total", "Tensor workload: MAC ops（PE聚合）", "ops", 1},
        {"tensor_dma_stall_cycles_total", "Tensor workload: DMA stall cycles（PE聚合）", "cycles", 1},
        {"tensor_iter_cycles_total", "Tensor workload: iteration cycles（PE聚合）", "cycles", 1},
        {"tensor_stall_dma_budget_cycles_total", "Tensor workload: stall (DMA budget) cycles（PE聚合）", "cycles", 1},
        {"tensor_stall_dma_hbm_channel_budget_cycles_total", "Tensor workload: stall (DMA HBM channel budget) cycles（PE聚合）", "cycles", 1},
        {"tensor_stall_mem_outstanding_cycles_total", "Tensor workload: stall (mem outstanding) cycles（PE聚合）", "cycles", 1},
        {"tensor_stall_wait_read_cycles_total", "Tensor workload: stall (wait read) cycles（PE聚合）", "cycles", 1},
        {"tensor_stall_wait_write_cycles_total", "Tensor workload: stall (wait write) cycles（PE聚合）", "cycles", 1},
        {"tensor_stall_collective_cycles_total", "Tensor workload: stall (collective barrier) cycles（PE聚合）", "cycles", 1},
        {"tensor_stall_onchip_capacity_cycles_total", "Tensor workload: stall (on-chip capacity) cycles（PE聚合）", "cycles", 1},
        {"tensor_stall_onchip_port_cycles_total", "Tensor workload: stall (on-chip ports) cycles（PE聚合）", "cycles", 1},
        {"tensor_stall_onchip_bank_conflict_cycles_total", "Tensor workload: stall (on-chip bank conflict) cycles（PE聚合）", "cycles", 1},
        {"tensor_stall_spill_budget_cycles_total", "Tensor workload: stall (spill budget) cycles（PE聚合）", "cycles", 1},
	        {"tensor_dma_cycles_total", "Tensor workload: DMA cycles（PE聚合）", "cycles", 1},
	        {"tensor_dram_bytes_total", "Tensor workload: DRAM bytes（PE聚合）", "bytes", 1},
	        {"tensor_onchip_bytes_total", "Tensor workload: on-chip bytes（PE聚合）", "bytes", 1},
	        {"tensor_cfg_ub_bytes", "Tensor workload: cfg ub bytes (debug/contract)（PE聚合）", "bytes", 1},
	        {"tensor_cfg_weight_bytes", "Tensor workload: cfg weight bytes (debug/contract)（PE聚合）", "bytes", 1},
	        {"tensor_onchip_weight_occupancy_bytes_max", "Tensor workload: weight pool occupancy max（PE聚合）", "bytes", 1},
	        {"tensor_onchip_weight_bank_occupancy_bytes_max", "Tensor workload: weight pool per-bank occupancy max（PE聚合）", "bytes", 1},
	        {"tensor_onchip_a_resident_tiles_max", "Tensor workload: A resident tiles max（PE聚合）", "tiles", 1},
	        {"tensor_onchip_b_resident_tiles_max", "Tensor workload: B resident tiles max（PE聚合）", "tiles", 1},
	        {"tensor_tile_count_total", "Tensor workload: tile count（PE聚合）", "tiles", 1},
	        {"tensor_spill_bytes_total", "Tensor workload: spill bytes（PE聚合）", "bytes", 1},
        {"tensor_spill_pkts_total", "Tensor workload: spill packets（PE聚合）", "packets", 1},
        {"tensor_collective_bytes_sent_total", "Tensor workload: collective bytes sent（PE聚合）", "bytes", 1},
        {"tensor_collective_bytes_recv_total", "Tensor workload: collective bytes received（PE聚合）", "bytes", 1},
        {"tensor_collective_pkts_sent_total", "Tensor workload: collective packets sent（PE聚合）", "packets", 1},
        {"tensor_collective_pkts_recv_total", "Tensor workload: collective packets received（PE聚合）", "packets", 1},
        {"tensor_collective_cycles_total", "Tensor workload: collective cycles（PE聚合）", "cycles", 1},
        {"tensor_collective_pending_cycles_total", "Tensor workload: collective pending cycles（PE聚合）", "cycles", 1},
        {"tensor_collective_issue_cycles_total", "Tensor workload: collective issue cycles（PE聚合）", "cycles", 1},
        {"tensor_collective_chunk_groups_total", "Tensor workload: collective chunk groups（PE聚合）", "groups", 1},
        {"tensor_collective_ring_steps_total", "Tensor workload: collective ring steps（PE聚合）", "steps", 1},
        {"tensor_collective_2d_row_rs_steps_total", "Tensor workload: collective 2D row reduce-scatter steps（PE聚合）", "steps", 1},
        {"tensor_collective_2d_col_rs_steps_total", "Tensor workload: collective 2D col reduce-scatter steps（PE聚合）", "steps", 1},
        {"tensor_collective_2d_col_ag_steps_total", "Tensor workload: collective 2D col all-gather steps（PE聚合）", "steps", 1},
        {"tensor_collective_2d_row_ag_steps_total", "Tensor workload: collective 2D row all-gather steps（PE聚合）", "steps", 1},
        {"tensor_collective_2d_row_rs_bytes_sent_total", "Tensor workload: collective 2D row reduce-scatter bytes sent（PE聚合）", "bytes", 1},
        {"tensor_collective_2d_col_rs_bytes_sent_total", "Tensor workload: collective 2D col reduce-scatter bytes sent（PE聚合）", "bytes", 1},
        {"tensor_collective_2d_col_ag_bytes_sent_total", "Tensor workload: collective 2D col all-gather bytes sent（PE聚合）", "bytes", 1},
        {"tensor_collective_2d_row_ag_bytes_sent_total", "Tensor workload: collective 2D row all-gather bytes sent（PE聚合）", "bytes", 1},
        {"tensor_collective_reduce_wait_cycles_total", "Tensor workload: collective reduce-wait cycles（PE聚合）", "cycles", 1},
        {"tensor_collective_2d_reduce_wait_cycles_total", "Tensor workload: collective 2D reduce-wait cycles（PE聚合）", "cycles", 1},
        {"tensor_collective_epoch_done_total", "Tensor workload: blocking collective epochs completed（PE聚合）", "count", 1},
        {"tensor_collective_epoch_latency_cycles_total", "Tensor workload: cumulative blocking collective epoch latency（PE聚合）", "cycles", 1},
        {"tensor_collective_epoch_latency_cycles_max", "Tensor workload: max blocking collective epoch latency（PE聚合）", "cycles", 1},
        {"tensor_collective_algo_id", "Tensor workload: collective algorithm id (0=legacy_bytes,1=ring_chunked,2=torus_2d_rs_ag)", "id", 1},
        {"tensor_collective_credit_stall_cycles_total", "Tensor workload: collective credit stall cycles（PE聚合）", "cycles", 1},
        {"tensor_collective_backpressure_stall_cycles_total", "Tensor workload: collective backpressure stall cycles（PE聚合）", "cycles", 1},
        {"tensor_collective_inflight_chunks_max", "Tensor workload: max collective inflight chunks（PE聚合）", "chunks", 1},
        {"tensor_collective_credit_return_pkts_sent_total", "Tensor workload: credit return packets sent（PE聚合）", "packets", 1},
        {"tensor_collective_credit_return_pkts_recv_total", "Tensor workload: credit return packets received（PE聚合）", "packets", 1},
        {"tensor_collective_credit_return_orphan_total", "Tensor workload: unmatched credit return packets（PE聚合）", "packets", 1},
        {"tensor_collective_credit_return_dup_total", "Tensor workload: duplicate credit return packets（PE聚合）", "packets", 1},
        {"tensor_collective_credit_return_latency_cycles_total", "Tensor workload: cumulative credit return latency（PE聚合）", "cycles", 1},
        {"tensor_collective_credit_return_latency_cycles_max", "Tensor workload: max credit return latency（PE聚合）", "cycles", 1},
        {"tensor_bank_queue_occupancy_max", "Tensor workload: max on-chip bank queue occupancy（PE聚合）", "entries", 1},
        {"tensor_stall_noc_budget_cycles_total", "Tensor workload: stall (NoC budget) cycles（PE聚合）", "cycles", 1},
        {"tensor_overlap_compute_collective_cycles_total", "Tensor workload: overlap (compute+collective) cycles（PE聚合）", "cycles", 1},
        {"tensor_overlap_compute_mem_cycles_total", "Tensor workload: overlap (compute+mem issue) cycles（PE聚合）", "cycles", 1},
        {"tensor_pkt_sent_total", "Tensor workload: RawBytes packets sent（PE聚合）", "packets", 1},
        {"tensor_pkt_recv_total", "Tensor workload: RawBytes packets received（PE聚合）", "packets", 1},
        {"tensor_pkt_bytes_sent_total", "Tensor workload: RawBytes bytes sent（PE聚合）", "bytes", 1},
        {"tensor_pkt_bytes_recv_total", "Tensor workload: RawBytes bytes received（PE聚合）", "bytes", 1},
        {"tensor_vector_cycles_total", "Tensor workload: vector cycles（PE聚合）", "cycles", 1},
        {"tensor_program_any_busy_cycles_total", "Tensor workload: program any-busy cycles（PE聚合）", "cycles", 1},
        {"tensor_program_dma_busy_cycles_total", "Tensor workload: program DMA busy cycles（PE聚合）", "cycles", 1},
        {"tensor_program_mxu_busy_cycles_total", "Tensor workload: program MXU busy cycles（PE聚合）", "cycles", 1},
        {"tensor_program_vec_busy_cycles_total", "Tensor workload: program vector busy cycles（PE聚合）", "cycles", 1},
        {"tensor_program_coll_busy_cycles_total", "Tensor workload: program collective busy cycles（PE聚合）", "cycles", 1},
        {"tensor_program_ops_total", "Tensor workload: program ops executed（PE聚合）", "ops", 1},
        {"tensor_program_iters_total", "Tensor workload: program iterations（PE聚合）", "iters", 1},
        {"tensor_program_fence_count_total", "Tensor workload: program fences executed（PE聚合）", "count", 1},
        {"tensor_program_fence_wait_cycles_total", "Tensor workload: program fence wait cycles（PE聚合）", "cycles", 1},
        {"tensor_program_ub_stall_cycles_total", "Tensor workload: program UB stall cycles（PE聚合）", "cycles", 1},
        {"tensor_program_mem_stall_cycles_total", "Tensor workload: program mem stall cycles（PE聚合）", "cycles", 1},
        {"tensor_program_ub_occupancy_bytes_max", "Tensor workload: program UB occupancy max（PE聚合）", "bytes", 1},
	        {"mem_read_latency_cycles", "端到端内存读延迟（cycles）", "cycles", 1},
	        {"mem_read_latency_cycles_weights", "权重访问读延迟（cycles）", "cycles", 1},
	        {"mem_read_latency_cycles_state", "非权重访问读延迟（cycles）", "cycles", 1},
        {"mem_req_size_bytes", "发起时请求大小（bytes）", "bytes", 1},
        {"gas_unique_reads_total", "GAS 下游唯一合并读事务数（总）", "reads", 1},
        {"gas_unique_bytes_total", "GAS 下游唯一合并读覆盖字节（总）", "bytes", 1},
        {"gas_row_window_triggers_total", "GAS 行窗口触发次数（总）", "count", 1},
        {"gas_row_window_bytes_total", "GAS 行窗口触发覆盖字节（总）", "bytes", 1},
        {"gas_total_bursts", "GAS granule/突发 数（总）", "bursts", 1},
        {"gas_total_payload_bytes", "GAS granule 有效载荷字节总和", "bytes", 1},
        {"gas_frontend_staged_reads_total", "GAS 前端诊断：进入 GatherBufferIF build 的 staged reads 总数", "reads", 1},
        {"gas_frontend_staged_line_touches_total", "GAS 前端诊断：staged reads 触达的 line touch 总数", "touches", 1},
        {"gas_frontend_granules_built_total", "GAS 前端诊断：build 阶段新建 granule 总数", "granules", 1},
        {"gas_unique_line_count_total", "GAS 诊断：近似 unique cacheline 数（总；DRAM-aware）", "lines", 1},
        {"gas_covered_line_count_total", "GAS 诊断：近似 covered cacheline 数（总；DRAM-aware）", "lines", 1},
        {"gas_overfetch_bytes_total", "GAS 诊断：overfetch bytes（总；DRAM-aware）", "bytes", 1},
        {"gas_apply_bank_credit_effective_total", "GAS Apply 并发额度：窗口有效 bank credit（总）", "count", 1},
        {"gas_cmd_cost_veto_total", "GAS cmd-cost 护栏：veto 次数（总）", "count", 1},
        {"gas_cmd_cost_veto_fine_gap_total", "GAS cmd-cost 护栏：fine-gap veto 次数（总）", "count", 1},
        {"gas_cmd_cost_veto_row_window_total", "GAS cmd-cost 护栏：row-window veto 次数（总）", "count", 1},
        {"gas_apply_acc_updates_total", "Apply阶段的delta累加次数（总）", "count", 1},
        {"gas_acc_posts_touched_total", "Apply阶段触达post个数（总）", "posts", 1},
        {"gas_scatter_spikes_emitted_total", "Scatter阶段发放spike个数（总）", "spikes", 1},
        {"stall_on_step_gate_cycles_total", "step_gate 等待下一次 openStep 的累计周期（PE聚合）", "cycles", 1},
        {"gas_acc_high_watermark_bytes_total", "累加器峰值占用（总）", "bytes", 1},
        {"gas_acc_spill_records_total", "溢写记录条数（总）", "records", 1},
        {"gas_acc_spilled_bytes_total", "溢写有效字节（总）", "bytes", 1},
            {"gas_retire_global_hol_cycles_total", "Global retire路径head未ready且存在ready-edge时的阻塞周期（PE聚合）", "cycles", 1},
            {"gas_retire_ready_but_blocked_edges_total", "Global retire路径中ready但被head阻塞的edge累计量（PE聚合）", "edge_cycles", 1},
            {"gas_retire_per_post_progress_total", "Per-post retire模式下实际退役次数（PE聚合）", "count", 1},
            {"gas_retire_samepost_blocked_edges_total", "Global retire路径中同post ready-edge被head阻塞的累计量（PE聚合）", "edge_cycles", 1},
            {"gas_retire_crosspost_blocked_edges_total", "Global retire路径中跨post ready-edge被head阻塞的累计量（PE聚合）", "edge_cycles", 1},
            {"gas_retire_policy_loss_cycles_total", "Global retire路径中可由更弱contract释放的阻塞周期（PE聚合）", "cycles", 1},
            {"gas_retire_policy_loss_edges_total", "Global retire路径中可由更弱contract释放的blocked edge累计量（PE聚合）", "edge_cycles", 1},
            {"gas_retire_shadow_per_post_recoverable_cycles_total", "Shadow per-post retire可恢复HOL周期（PE聚合）", "cycles", 1},
        {"gas_retire_shadow_per_post_recoverable_edges_total", "Shadow per-post retire可恢复edge累计量（PE聚合）", "edge_cycles", 1},
        {"gas_retire_shadow_per_post_ready_posts_peak", "Shadow per-post retire ready-post峰值（PE聚合）", "posts", 1},
        {"gas_retire_shadow_per_post_committable_edges_peak", "Shadow per-post retire可提交edge峰值（PE聚合）", "edges", 1},
        {"weight_read_dense_reqs_total", "Issued weight-read requests classified as dense（PE聚合）", "requests", 1},
        {"weight_read_dense_bytes_total", "Issued weight-read bytes classified as dense（PE聚合）", "bytes", 1},
        {"weight_read_rowptr_reqs_total", "Issued weight-read requests classified as BCSR rowptr（PE聚合）", "requests", 1},
        {"weight_read_rowptr_bytes_total", "Issued weight-read bytes classified as BCSR rowptr（PE聚合）", "bytes", 1},
        {"weight_read_colidx_reqs_total", "Issued weight-read requests classified as BCSR colidx（PE聚合）", "requests", 1},
        {"weight_read_colidx_bytes_total", "Issued weight-read bytes classified as BCSR colidx（PE聚合）", "bytes", 1},
        {"weight_read_blockdata_reqs_total", "Issued weight-read requests classified as BCSR blockdata（PE聚合）", "requests", 1},
        {"weight_read_blockdata_bytes_total", "Issued weight-read bytes classified as BCSR blockdata（PE聚合）", "bytes", 1},
        {"weight_idx_sram_reads_total", "Observe-only weight idx SRAM reads（PE聚合）", "reads", 1},
        {"weight_idx_sram_writes_total", "Observe-only weight idx SRAM writes（PE聚合）", "writes", 1},
        {"weight_idx_sram_bytes_read_total", "Observe-only weight idx SRAM read bytes（PE聚合）", "bytes", 1},
        {"weight_idx_sram_bytes_write_total", "Observe-only weight idx SRAM write bytes（PE聚合）", "bytes", 1},
        {"weight_idx_sram_bank_conflict_ticks_total", "Observe-only weight idx SRAM conflict ticks（PE聚合）", "ticks", 1},
        {"weight_idx_sram_predicted_extra_cycles_total", "Observe-only weight idx SRAM predicted extra cycles（PE聚合）", "cycles", 1},
        {"weight_idx_sram_resident_bytes_peak", "Observe-only weight idx SRAM resident bytes peak（PE聚合）", "bytes", 1},
        {"weight_idx_sram_bank_peak_accesses_per_tick", "Weight idx SRAM peak accesses on any bank（PE聚合）", "accesses", 1},
        {"weight_idx_sram_energy_read_pj_total", "Weight idx SRAM read energy（PE聚合）", "pJ", 1},
        {"weight_idx_sram_energy_write_pj_total", "Weight idx SRAM write energy（PE聚合）", "pJ", 1},
        {"weight_l0_sram_reads_total", "Observe-only weight L0 SRAM reads（PE聚合）", "reads", 1},
        {"weight_l0_sram_writes_total", "Observe-only weight L0 SRAM writes（PE聚合）", "writes", 1},
        {"weight_l0_sram_bytes_read_total", "Observe-only weight L0 SRAM read bytes（PE聚合）", "bytes", 1},
        {"weight_l0_sram_bytes_write_total", "Observe-only weight L0 SRAM write bytes（PE聚合）", "bytes", 1},
        {"weight_l0_sram_bank_conflict_ticks_total", "Observe-only weight L0 SRAM conflict ticks（PE聚合）", "ticks", 1},
        {"weight_l0_sram_predicted_extra_cycles_total", "Observe-only weight L0 SRAM predicted extra cycles（PE聚合）", "cycles", 1},
        {"weight_l0_sram_resident_bytes_peak", "Observe-only weight L0 SRAM resident bytes peak（PE聚合）", "bytes", 1},
        {"weight_l0_sram_bank_peak_accesses_per_tick", "Weight L0 SRAM peak accesses on any bank（PE聚合）", "accesses", 1},
        {"weight_l0_sram_energy_read_pj_total", "Weight L0 SRAM read energy（PE聚合）", "pJ", 1},
        {"weight_l0_sram_energy_write_pj_total", "Weight L0 SRAM write energy（PE聚合）", "pJ", 1},
        {"weight_sram_enforced_stall_cycles_total", "Weight SRAM enforced stall cycles（PE聚合）", "cycles", 1},
        {"weight_l0_lookup_total", "Weight L0 lookups（PE聚合）", "lookups", 1},
        {"weight_l0_hit_total", "Weight L0 hits（PE聚合）", "hits", 1},
        {"weight_l0_fill_total", "Weight L0 fills（PE聚合）", "fills", 1},
        {"weight_l0_evict_total", "Weight L0 evictions（PE聚合）", "evictions", 1},
        {"core_state_sram_reads_total", "Observe-only state SRAM reads（PE聚合）", "reads", 1},
        {"core_state_sram_writes_total", "Observe-only state SRAM writes（PE聚合）", "writes", 1},
        {"core_state_sram_bytes_read_total", "Observe-only state SRAM read bytes（PE聚合）", "bytes", 1},
        {"core_state_sram_bytes_write_total", "Observe-only state SRAM write bytes（PE聚合）", "bytes", 1},
        {"core_state_sram_bank_conflict_ticks_total", "Observe-only state SRAM conflict ticks（PE聚合）", "ticks", 1},
        {"core_state_sram_predicted_extra_cycles_total", "Observe-only state SRAM predicted extra cycles（PE聚合）", "cycles", 1},
        {"core_state_sram_resident_bytes_peak", "Observe-only state SRAM resident bytes peak（PE聚合）", "bytes", 1},
        {"core_state_sram_bank_peak_accesses_per_tick", "Neuron-state SRAM peak accesses on any bank（PE聚合）", "accesses", 1},
        {"core_state_sram_energy_read_pj_total", "Neuron-state SRAM read energy（PE聚合）", "pJ", 1},
        {"core_state_sram_energy_write_pj_total", "Neuron-state SRAM write energy（PE聚合）", "pJ", 1},
        {"core_state_sram_stall_cycles_total", "Neuron-state SRAM enforced stall cycles（PE聚合）", "cycles", 1},
        {"mem_outstanding_at_issue", "发起时并发请求数", "count", 1},
        {"gas_activity_f", "GAS 窗口内活跃度 f（活跃轴数/列宽）", "ratio", 1},
        {"sim_cycles_total", "总仿真周期（组件clock tick累计）", "cycles", 1},
        {"compute_active_cycles_total", "近似计算活跃周期（仅统计发生 synaptic/apply 推进的周期）", "cycles", 1},
        {"global_steps_done_total", "全局 Step/GAS 同步：本PE已完成的step数（PeDone次数）", "count", 1},
        {"step_activation_invocations", "step随机激活触发次数", "count", 1},
        {"step_activation_pre_selected", "step随机激活选中的pre节点数", "count", 1},
        {"step_activation_spike_attempts", "step随机激活尝试产生的spike数量", "count", 1},
        {"step_activation_spikes_injected", "step随机激活成功注入本PE的spike数", "spikes", 1},
        {"step_activation_route_hits", "step随机激活经由BCSR路由成功命中的次数", "count", 1},
        {"step_activation_route_misses", "step随机激活启用BCSR但路由为空的次数", "count", 1},
        {"step_activation_local_drops", "step随机激活因目标不在本PE而丢弃的spike数", "spikes", 1},
        // Experiment profile observability (sst_dram_si: experiment_profile=universal_core_eval)
        {"compat_unexpected_stream_activity_total", "非 stream workload 下仍出现 stream 统计活动（计数）", "count", 1},
        {"compat_naive_gas_stage_events_total", "naive_raw 下观测到 GAS 阶段事件(BeginApply/BeginScatter/EndApply)（计数）", "count", 1},
        {"compat_finish_incomplete_total", "global_step_sync_enable=1 但本PE在 active step 未完成时结束（计数）", "count", 1},
        {"loader_done_timeout_fallback_total", "loader_done_key 等待超时后触发降级发送 PE_READY 的次数", "count", 1},
        {"dma_issue_reqs_total", "PE DMA 发射的读 burst 总数", "requests", 1},
        {"dma_issue_bytes_total", "PE DMA 发射的总字节数", "bytes", 1},
        {"dma_queue_depth_max_p0", "PE DMA P0 最大队列深度", "requests", 1},
        {"dma_queue_depth_max_p1", "PE DMA P1 最大队列深度", "requests", 1},
        {"dma_queue_depth_max_p2", "PE DMA P2 最大队列深度", "requests", 1},
        {"dma_queue_depth_max_p3", "PE DMA P3 最大队列深度", "requests", 1},
        {"dma_inflight_max", "PE DMA 最大在途 burst 数", "requests", 1},
        {"dma_stall_cycles_budget", "PE DMA 因预算受限的停顿计数", "cycles", 1},
        {"dma_stall_cycles_engine", "PE DMA 因引擎受限的停顿计数", "cycles", 1},
        {"dma_stall_cycles_inflight", "PE DMA 因 inflight 上限受限的停顿计数", "cycles", 1},
        {"dma_stall_cycles_stage_gate", "PE DMA 因阶段门控受限的停顿计数", "cycles", 1},
        {"dma_stall_cycles_queue_full", "PE DMA 因队列满受限的停顿计数", "cycles", 1},
        {"ls_objects_registered_total", "Local storage: 已注册对象总数", "objects", 1},
        {"ls_objects_enabled_total", "Local storage: 有效对象总数", "objects", 1},
        {"ls_capacity_bytes_total", "Local storage: 总容量字节数", "bytes", 1},
        {"ls_queue_slots_total", "Local storage: 总队列槽位数", "slots", 1},
    )

    /**
     * @brief 构造函数
     */
    MultiCorePE(SST::ComponentId_t id, SST::Params& params);

    /**
     * @brief 析构函数  
     */
    ~MultiCorePE();

    // SST组件生命周期方法
    void init(unsigned int phase) override;
    void complete(unsigned int phase) override;
    void setup() override;
    void finish() override;

    // ===== 公共接口方法 =====
    
    /**
     * @brief 处理来自其他MultiCorePE的外部脉冲
     */
    void handleExternalSpike(SpikeEvent* spike);
    
    /**
     * @brief 发送脉冲到其他MultiCorePE  
     */
    void sendExternalSpike(SpikeEvent* spike);
    
    /**
     * @brief 获取处理单元状态信息
     */
    const ProcessingUnitState& getProcessingUnitState(int unit_id) const;
    
    /**
     * @brief 获取多核PE统计信息
     */
    void getStatistics(std::map<std::string, uint64_t>& stats) const;

    // Batch-A 聚合接口（供核心调用）
    void accumulateMemReadLatency(uint64_t latency_cycles, bool is_weight);
    void accumulateIssueStats(uint64_t req_size_bytes, uint64_t inflight);
    void accumulateGasStats(uint64_t unique_bytes, uint64_t unique_reads);
    void accumulateGasStatsExt(uint64_t unique_bytes, uint64_t unique_reads,
                               uint64_t rowwin_triggers, uint64_t rowwin_bytes,
                               uint64_t bursts, uint64_t payload_bytes,
                               uint64_t window_inflight_peak = 0,
                               uint64_t window_buffer_max_bytes = 0,
                               uint64_t frontend_staged_reads = 0,
                               uint64_t frontend_staged_line_touches = 0,
                               uint64_t frontend_granules_built = 0,
                               uint64_t unique_line_count = 0,
                               uint64_t covered_line_count = 0,
                               uint64_t overfetch_bytes = 0,
                               uint64_t apply_bank_credit_effective = 0,
                               uint64_t cmd_cost_veto = 0,
                               uint64_t cmd_cost_veto_fine_gap = 0,
                               uint64_t cmd_cost_veto_row_window = 0,
                               uint64_t stall_on_step_gate_cycles = 0);
    void accumulateActivityF(double f);
    void accumulateApplyScatterStats(uint64_t acc_updates, uint64_t posts_touched,
                                     uint64_t spikes_emitted, uint64_t hwm_bytes,
                                     uint64_t spill_records, uint64_t spilled_bytes);
    void recordStepActivationSummary(uint32_t seq,
                                     uint64_t pre_selected,
                                     uint64_t spike_attempts,
                                     uint64_t spikes_injected,
                                     uint64_t route_hits,
                                     uint64_t route_misses,
                                     uint64_t local_drops);
    void recordStepRxPacket(uint32_t seq, NocPacketKind kind, bool is_local, bool before_begin_gather);
    void recordStepRxGate(uint32_t seq,
                          uint64_t accept_total,
                          uint64_t reject_refractory_total,
                          uint64_t direct_accept_total,
                          uint64_t direct_reject_refractory_total,
                          uint64_t fastpath_accept_total,
                          uint64_t fastpath_reject_refractory_total);
    void recordStepGasStat(uint32_t seq, const GasStatEvent& st);
    void recordCoreStepGasStat(int core_id, uint32_t seq, const GasStatEvent& st);
    void recordStepApplyScatter(uint32_t seq,
                                uint64_t acc_updates,
                                uint64_t posts_touched,
                                uint64_t spikes_emitted,
                                uint64_t hwm_bytes,
                                uint64_t spill_records,
                                uint64_t spilled_bytes);
    void recordCoreStepApplyScatter(int core_id,
                                    uint32_t seq,
                                    uint64_t acc_updates,
                                    uint64_t posts_touched,
                                    uint64_t spikes_emitted,
                                    uint64_t hwm_bytes,
                                    uint64_t spill_records,
                                    uint64_t spilled_bytes);
    void accumulateRiscvSnnRuntimeStats(uint64_t workload_selected,
                                        uint64_t firmware_elf_present,
                                        uint64_t firmware_loaded,
                                        uint64_t backend_runtime_bridge,
                                        uint64_t firmware_started_count,
                                        uint64_t submitted_commands,
                                        uint64_t accepted_commands,
                                        uint64_t completion_visible_count,
                                        uint64_t completion_consumed_count,
                                        uint64_t fused_step_completion_count,
                                        uint64_t fault_count,
                                        uint64_t last_completion_status,
                                        uint64_t last_fault_csr,
                                        uint64_t backend_runtime_bridge_provider_bound);
    void accumulateSynapseReadStats(uint64_t dense_reqs_total,
                                    uint64_t dense_bytes_total,
                                    uint64_t rowptr_reqs_total,
                                    uint64_t rowptr_bytes_total,
                                    uint64_t colidx_reqs_total,
                                    uint64_t colidx_bytes_total,
                                    uint64_t blockdata_reqs_total,
                                    uint64_t blockdata_bytes_total,
                                    uint64_t weight_idx_sram_reads_total,
                                    uint64_t weight_idx_sram_writes_total,
                                    uint64_t weight_idx_sram_bytes_read_total,
                                    uint64_t weight_idx_sram_bytes_write_total,
                                    uint64_t weight_idx_sram_bank_conflict_ticks_total,
                                    uint64_t weight_idx_sram_predicted_extra_cycles_total,
                                    uint64_t weight_idx_sram_resident_bytes_peak,
                                    uint64_t weight_idx_sram_bank_peak_accesses_per_tick,
                                    uint64_t weight_idx_sram_energy_read_pj_total,
                                    uint64_t weight_idx_sram_energy_write_pj_total,
                                    uint64_t weight_l0_sram_reads_total,
                                    uint64_t weight_l0_sram_writes_total,
                                    uint64_t weight_l0_sram_bytes_read_total,
                                    uint64_t weight_l0_sram_bytes_write_total,
                                    uint64_t weight_l0_sram_bank_conflict_ticks_total,
                                    uint64_t weight_l0_sram_predicted_extra_cycles_total,
                                    uint64_t weight_l0_sram_resident_bytes_peak,
                                    uint64_t weight_l0_sram_bank_peak_accesses_per_tick,
                                    uint64_t weight_l0_sram_energy_read_pj_total,
                                    uint64_t weight_l0_sram_energy_write_pj_total,
                                    uint64_t weight_sram_enforced_stall_cycles_total,
                                    uint64_t weight_l0_lookup_total,
                                    uint64_t weight_l0_hit_total,
                                    uint64_t weight_l0_fill_total,
                                    uint64_t weight_l0_evict_total,
                                    uint64_t core_state_sram_reads_total,
                                    uint64_t core_state_sram_writes_total,
                                    uint64_t core_state_sram_bytes_read_total,
                                    uint64_t core_state_sram_bytes_write_total,
                                    uint64_t core_state_sram_bank_conflict_ticks_total,
                                    uint64_t core_state_sram_predicted_extra_cycles_total,
                                    uint64_t core_state_sram_resident_bytes_peak,
                                    uint64_t core_state_sram_bank_peak_accesses_per_tick,
                                    uint64_t core_state_sram_energy_read_pj_total,
                                    uint64_t core_state_sram_energy_write_pj_total,
                                    uint64_t core_state_sram_stall_cycles_total,
                                    uint64_t gas_retire_global_hol_cycles_total,
                                    uint64_t gas_retire_ready_but_blocked_edges_total,
                                    uint64_t gas_retire_per_post_progress_total,
                                    uint64_t gas_retire_samepost_blocked_edges_total,
                                    uint64_t gas_retire_crosspost_blocked_edges_total,
                                    uint64_t gas_retire_policy_loss_cycles_total,
                                            uint64_t gas_retire_policy_loss_edges_total,
                                            uint64_t gas_retire_shadow_per_post_recoverable_cycles_total,
                                            uint64_t gas_retire_shadow_per_post_recoverable_edges_total,
                                            uint64_t gas_retire_shadow_per_post_ready_posts_peak,
                                            uint64_t gas_retire_shadow_per_post_committable_edges_peak);
    // PE级阶段事件收敛：核心通知PE，PE只写一次（每窗一行）
    // 接收核心上报的阶段事件；当 event=EndScatter 且 spikes>0 时，同时记录本窗发放
    void notifyStageEvent(uint32_t seq, const std::string& event, uint64_t ts_ns,
                          uint64_t spikes_emitted = 0, int core_id = -1);
    void accumulateUniqueNeuronFired(uint64_t cnt) {
        if (cnt && stat_unique_neurons_fired_total_) stat_unique_neurons_fired_total_->addData(cnt);
    }
    void accumulateWindowSpikes(uint32_t seq, uint64_t count) {
        window_spikes_pe_[seq] += count;
    }
    Statistic<uint64_t>* getComputeActiveCyclesTotalStatistic() const {
        return stat_compute_active_cycles_total_;
    }

    SST::Statistics::Statistic<uint64_t>* registerU64(const std::string& stat_name) override {
        return registerStatistic<uint64_t>(stat_name);
    }
    PeDmaScheduler* dmaScheduler() override { return dma_scheduler_.get(); }
    LocalStorageHierarchyController* localStorageHierarchy() override { return local_storage_controller_.get(); }
    PodMetadataObjectPlane* pePodMetadataObjectPlane() override { return pod_metadata_object_plane_.get(); }
    PodOwnerServiceTable* pePodOwnerServiceTable() override { return pod_owner_service_table_.get(); }
    PeLocalServiceObjectTable* peLocalServiceObjectTable() override {
        return pe_local_service_object_table_.get();
    }
    PeWeightObjectPlane* peWeightObjectPlane() override { return pe_weight_object_plane_.get(); }
    bool peInternalCpeEnabledConfig() const { return pe_internal_cpe_enable_cfg_; }
    bool peInternalPodEnabledConfig() const { return pe_internal_pod_enable_cfg_; }
    bool peInternalPodMetadataEnabledConfig() const {
        return pe_internal_pod_metadata_enable_cfg_;
    }
    bool peInternalPodOwnerEnabledConfig() const {
        return pe_internal_pod_owner_enable_cfg_;
    }
    uint32_t peInternalPodCountConfig() const { return pe_internal_pod_count_cfg_; }
    uint32_t peInternalPodSizeConfig() const { return pe_internal_pod_size_cfg_; }

    // 友元类声明
    friend class MultiCoreController;

private:
    // 诊断打印节流（成员化，替代函数静态）
    bool first_tick_logged_ = false;
    // P2: 参数化门控（优先参数，其次回退环境变量）
    bool sentinel_enabled_ = false;
    long step_diag_cap_cfg_ = -1;
    int  step_diag_enable_cfg_ = 0;

    // ===== 配置参数 =====
    
    int num_cores_;
    int neurons_per_core_;
    int total_neurons_;
    uint32_t neurons_per_pe_cfg_ = 0; // 脚本可传入，否则按 num_cores*neurons_per_core 推导（Step/NIC 计算用）
    int node_id_;
    int total_nodes_ = 1;
    uint64_t sim_stop_ns_ = 0;      // 组件主控停止时间（ns）；>0 启用
    bool primary_registered_ = false;
    uint64_t global_neuron_base_;
    uint64_t base_addr_ = 0;
    uint64_t per_core_stride_ = 0;
    SST::Params core_params_template_;
    std::string workload_impl_ = "snn";
    int verbose_;
    std::string clock_freq_ = "1GHz";
    std::string weights_file_;
    bool enable_numa_;
    bool enable_test_traffic_;
    bool dma_enable_ = false;

    // 全局 neuron_id 布局（单一真源，供 Step/Route/NoC 等口径复用）
    GlobalNeuronLayout global_layout_{};
    
    // 神经元参数
    float v_thresh_;
    float v_reset_;
    float v_rest_;
    float tau_mem_;
    int t_ref_;
    
    // 测试流量参数
    std::string test_traffic_packet_kind_ = "spike";
    int test_target_node_;
    int test_period_;
    int test_spikes_per_burst_;
    float test_weight_;
    int test_max_spikes_;  // 最大测试脉冲数限制
    
    // 环形网络实现选择
    bool use_optimized_ring_;
    // 输出控制
    bool print_node_summary_ = true; // 控制finish时是否打印节点汇总
    bool primary_keepalive_ = false; // 仅在单PE脚本需要保持仿真推进时启用
    bool ok_to_end_sent_ = false; // 防止重复触发 primaryComponentOKToEndSim 导致退出事件重复调度
    bool manual_core_drive_enable_ = false; // 诊断回退：手动驱动子核clock/EndGather
    uint64_t manual_gas_gather_cycles_ = 200; // 已弃用：旧 manual drive 兼容参数（当前无效；保留仅为兼容）
    
    // 权重验证参数
    bool verify_weights_;
    uint32_t weight_verify_samples_;
    float expected_weight_value_;
    bool verify_log_each_sample_;
    
    // 权重回退参数
    bool use_event_weight_fallback_;
    bool enable_memory_weights_;
    bool write_weights_on_init_;
    bool step_seed_only_mode_ = false;
    std::unique_ptr<PeDmaScheduler> dma_scheduler_;
    bool pe_internal_pod_requested_ = false;
    bool pe_internal_pod_metadata_requested_ = false;
    bool pe_internal_pod_owner_requested_ = false;
    bool pe_internal_cpe_enable_cfg_ = false;
    bool pe_internal_pod_enable_cfg_ = false;
    bool pe_internal_pod_metadata_enable_cfg_ = false;
    bool pe_internal_pod_owner_enable_cfg_ = false;
    uint32_t pe_internal_pod_count_cfg_ = 1;
    uint32_t pe_internal_pod_size_cfg_ = 1;

    // ===== 组件对象 =====
    
    // 时钟和输出
    SST::Output* output_;
    
    // 统计变量
    Statistic<uint64_t>* stat_spikes_processed_;
    Statistic<uint64_t>* stat_inter_core_messages_;
    Statistic<uint64_t>* stat_l2_hits_;
    Statistic<uint64_t>* stat_l2_misses_;
    Statistic<uint64_t>* stat_memory_requests_;
    Statistic<double>* stat_avg_utilization_;
    Statistic<uint64_t>* stat_neurons_fired_;
    Statistic<uint64_t>* stat_unique_neurons_fired_total_ = nullptr;
    Statistic<uint64_t>* stat_external_spikes_sent_;
    Statistic<uint64_t>* stat_external_spikes_received_;
    Statistic<uint64_t>* stat_riscv_snn_workload_selected_ = nullptr;
    Statistic<uint64_t>* stat_riscv_snn_firmware_elf_present_ = nullptr;
    Statistic<uint64_t>* stat_riscv_snn_firmware_loaded_ = nullptr;
    Statistic<uint64_t>* stat_riscv_snn_backend_runtime_bridge_ = nullptr;
    Statistic<uint64_t>* stat_riscv_snn_firmware_started_count_ = nullptr;
    Statistic<uint64_t>* stat_riscv_snn_submitted_commands_ = nullptr;
    Statistic<uint64_t>* stat_riscv_snn_accepted_commands_ = nullptr;
    Statistic<uint64_t>* stat_riscv_snn_completion_visible_count_ = nullptr;
    Statistic<uint64_t>* stat_riscv_snn_completion_consumed_count_ = nullptr;
    Statistic<uint64_t>* stat_riscv_snn_fused_step_completion_count_ = nullptr;
    Statistic<uint64_t>* stat_riscv_snn_fault_count_ = nullptr;
    Statistic<uint64_t>* stat_riscv_snn_last_completion_status_ = nullptr;
    Statistic<uint64_t>* stat_riscv_snn_last_fault_csr_ = nullptr;
    Statistic<uint64_t>* stat_riscv_snn_backend_runtime_bridge_provider_bound_ = nullptr;
    Statistic<uint64_t>* stat_dma_issue_reqs_total_ = nullptr;
    Statistic<uint64_t>* stat_dma_issue_bytes_total_ = nullptr;
    Statistic<uint64_t>* stat_dma_queue_depth_max_p0_ = nullptr;
    Statistic<uint64_t>* stat_dma_queue_depth_max_p1_ = nullptr;
    Statistic<uint64_t>* stat_dma_queue_depth_max_p2_ = nullptr;
    Statistic<uint64_t>* stat_dma_queue_depth_max_p3_ = nullptr;
    Statistic<uint64_t>* stat_dma_inflight_max_ = nullptr;
    Statistic<uint64_t>* stat_dma_stall_cycles_budget_ = nullptr;
    Statistic<uint64_t>* stat_dma_stall_cycles_engine_ = nullptr;
    Statistic<uint64_t>* stat_dma_stall_cycles_inflight_ = nullptr;
    Statistic<uint64_t>* stat_dma_stall_cycles_stage_gate_ = nullptr;
    Statistic<uint64_t>* stat_dma_stall_cycles_queue_full_ = nullptr;
    Statistic<uint64_t>* stat_ls_objects_registered_total_ = nullptr;
    Statistic<uint64_t>* stat_ls_objects_enabled_total_ = nullptr;
    Statistic<uint64_t>* stat_ls_capacity_bytes_total_ = nullptr;
    Statistic<uint64_t>* stat_ls_queue_slots_total_ = nullptr;
    // Batch-A 组件级直方图统计
    Statistic<uint64_t>* stat_mem_read_latency_cycles_ = nullptr;
    Statistic<uint64_t>* stat_mem_read_latency_cycles_weights_ = nullptr;
    Statistic<uint64_t>* stat_mem_read_latency_cycles_state_ = nullptr;
    Statistic<uint64_t>* stat_mem_req_size_bytes_ = nullptr;
    Statistic<uint64_t>* stat_gas_unique_reads_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_unique_bytes_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_rowwin_triggers_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_rowwin_bytes_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_total_bursts_ = nullptr;
    Statistic<uint64_t>* stat_gas_total_payload_bytes_ = nullptr;
    Statistic<uint64_t>* stat_gas_frontend_staged_reads_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_frontend_staged_line_touches_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_frontend_granules_built_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_unique_line_count_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_covered_line_count_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_overfetch_bytes_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_apply_bank_credit_effective_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_cmd_cost_veto_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_cmd_cost_veto_fine_gap_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_cmd_cost_veto_row_window_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_apply_acc_updates_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_acc_posts_touched_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_scatter_spikes_emitted_total_ = nullptr;
    Statistic<uint64_t>* stat_stall_on_step_gate_cycles_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_acc_hwm_bytes_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_acc_spill_records_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_acc_spilled_bytes_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_retire_global_hol_cycles_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_retire_ready_but_blocked_edges_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_retire_per_post_progress_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_retire_samepost_blocked_edges_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_retire_crosspost_blocked_edges_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_retire_policy_loss_cycles_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_retire_policy_loss_edges_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_retire_shadow_per_post_recoverable_cycles_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_retire_shadow_per_post_recoverable_edges_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_retire_shadow_per_post_ready_posts_peak_ = nullptr;
    Statistic<uint64_t>* stat_gas_retire_shadow_per_post_committable_edges_peak_ = nullptr;
    Statistic<uint64_t>* stat_weight_read_dense_reqs_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_read_dense_bytes_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_read_rowptr_reqs_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_read_rowptr_bytes_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_read_colidx_reqs_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_read_colidx_bytes_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_read_blockdata_reqs_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_read_blockdata_bytes_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_idx_sram_reads_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_idx_sram_writes_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_idx_sram_bytes_read_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_idx_sram_bytes_write_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_idx_sram_bank_conflict_ticks_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_idx_sram_predicted_extra_cycles_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_idx_sram_resident_bytes_peak_ = nullptr;
    Statistic<uint64_t>* stat_weight_idx_sram_bank_peak_accesses_per_tick_ = nullptr;
    Statistic<uint64_t>* stat_weight_idx_sram_energy_read_pj_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_idx_sram_energy_write_pj_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_idx_lookup_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_idx_lookup_idx2_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_l0_sram_reads_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_l0_sram_writes_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_l0_sram_bytes_read_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_l0_sram_bytes_write_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_l0_sram_bank_conflict_ticks_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_l0_sram_predicted_extra_cycles_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_l0_sram_resident_bytes_peak_ = nullptr;
    Statistic<uint64_t>* stat_weight_l0_sram_bank_peak_accesses_per_tick_ = nullptr;
    Statistic<uint64_t>* stat_weight_l0_sram_energy_read_pj_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_l0_sram_energy_write_pj_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_sram_enforced_stall_cycles_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_l0_lookup_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_l0_hit_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_l0_fill_total_ = nullptr;
    Statistic<uint64_t>* stat_weight_l0_evict_total_ = nullptr;
    Statistic<uint64_t>* stat_core_state_sram_reads_total_ = nullptr;
    Statistic<uint64_t>* stat_core_state_sram_writes_total_ = nullptr;
    Statistic<uint64_t>* stat_core_state_sram_bytes_read_total_ = nullptr;
    Statistic<uint64_t>* stat_core_state_sram_bytes_write_total_ = nullptr;
    Statistic<uint64_t>* stat_core_state_sram_bank_conflict_ticks_total_ = nullptr;
    Statistic<uint64_t>* stat_core_state_sram_predicted_extra_cycles_total_ = nullptr;
    Statistic<uint64_t>* stat_core_state_sram_resident_bytes_peak_ = nullptr;
    Statistic<uint64_t>* stat_core_state_sram_bank_peak_accesses_per_tick_ = nullptr;
    Statistic<uint64_t>* stat_core_state_sram_energy_read_pj_total_ = nullptr;
    Statistic<uint64_t>* stat_core_state_sram_energy_write_pj_total_ = nullptr;
    Statistic<uint64_t>* stat_core_state_sram_stall_cycles_total_ = nullptr;
    Statistic<double>*  stat_gas_activity_f_ = nullptr;
    Statistic<uint64_t>* stat_mem_outstanding_at_issue_ = nullptr;
    Statistic<uint64_t>* stat_sim_cycles_total_ = nullptr;
    Statistic<uint64_t>* stat_compute_active_cycles_total_ = nullptr;
    Statistic<uint64_t>* stat_global_steps_done_total_ = nullptr;
    Statistic<uint64_t>* stat_step_activation_invocations_ = nullptr;
    Statistic<uint64_t>* stat_step_activation_pre_selected_ = nullptr;
    Statistic<uint64_t>* stat_step_activation_spike_attempts_ = nullptr;
    Statistic<uint64_t>* stat_step_activation_spikes_injected_ = nullptr;
	    Statistic<uint64_t>* stat_step_activation_route_hits_ = nullptr;
	    Statistic<uint64_t>* stat_step_activation_route_misses_ = nullptr;
	    Statistic<uint64_t>* stat_step_activation_local_drops_ = nullptr;
	    Statistic<uint64_t>* stat_compat_unexpected_stream_activity_total_ = nullptr;
	    Statistic<uint64_t>* stat_compat_naive_gas_stage_events_total_ = nullptr;
	    Statistic<uint64_t>* stat_compat_finish_incomplete_total_ = nullptr;
    Statistic<uint64_t>* stat_loader_done_timeout_fallback_total_ = nullptr;
    std::vector<std::unique_ptr<IWorkloadStatsModule>> workload_stats_modules_;

    // 本地统计：仅在环形跨核投递成功时累加
    uint64_t inter_core_messages_count_ = 0;

    // PE-level per-window spikes aggregation
    std::unordered_map<uint32_t, uint64_t> window_spikes_pe_;
    std::string stage_events_csv_path_;
    std::string stats_csv_path_;

    // 进度日志：用于定位仿真是否在推进/卡住（默认关闭）
    uint64_t progress_log_interval_ns_ = 0;
    int progress_log_node_ = -1;
    // PE级阶段事件聚合（每seq聚合一次）
    struct StageMarks { uint64_t bg=0, ga=0, bs=0, ea=0, es=0; };
    struct CoreStepKey {
        int core = -1;
        uint32_t seq = 0;

        bool operator==(const CoreStepKey& other) const {
            return core == other.core && seq == other.seq;
        }
    };
    struct CoreStepKeyHash {
        std::size_t operator()(const CoreStepKey& key) const {
            return (static_cast<std::size_t>(static_cast<uint32_t>(key.core)) << 32) ^
                   static_cast<std::size_t>(key.seq);
        }
    };
    struct StepEndScatterResetState {
        std::vector<uint8_t> cores_seen;
        uint32_t done_count = 0;
        bool reset_fired = false;
    };
    struct StepPerfAgg {
        uint64_t seed_pre_selected = 0;
        uint64_t seed_spike_attempts = 0;
        uint64_t seed_spikes_injected = 0;
        uint64_t seed_route_hits = 0;
        uint64_t seed_route_misses = 0;
        uint64_t seed_local_drops = 0;
        uint64_t rx_packets_total = 0;
        uint64_t rx_spike_packets_total = 0;
        uint64_t rx_spikekey_packets_total = 0;
        uint64_t rx_spiketilekey_packets_total = 0;
        uint64_t rx_local_packets_total = 0;
        uint64_t rx_remote_packets_total = 0;
        uint64_t rx_packets_before_bg_total = 0;
        uint64_t rx_local_packets_before_bg_total = 0;
        uint64_t rx_remote_packets_before_bg_total = 0;
        uint64_t rx_packets_during_gather_total = 0;
        uint64_t rx_packets_during_apply_total = 0;
        uint64_t rx_packets_during_scatter_total = 0;
        uint64_t rx_gate_pending_peak = 0;
        uint64_t rx_gate_accept_total = 0;
        uint64_t rx_gate_reject_refractory_total = 0;
        uint64_t rx_gate_direct_accept_total = 0;
        uint64_t rx_gate_direct_reject_refractory_total = 0;
        uint64_t rx_gate_fastpath_accept_total = 0;
        uint64_t rx_gate_fastpath_reject_refractory_total = 0;
        uint64_t acc_updates = 0;
        uint64_t posts_touched = 0;
        uint64_t spill_records = 0;
        uint64_t spilled_bytes = 0;
        uint64_t hwm_bytes_max = 0;
        uint64_t scatter_spikes_emitted = 0;
        uint64_t unique_reads = 0;
        uint64_t unique_bytes = 0;
        uint64_t rowwin_triggers = 0;
        uint64_t rowwin_bytes = 0;
        uint64_t bursts = 0;
        uint64_t payload_bytes = 0;
        uint64_t window_inflight_peak = 0;
        uint64_t window_buffer_max_bytes = 0;
        uint64_t gap_absorbed_bytes = 0;
        uint64_t frontend_staged_reads = 0;
        uint64_t frontend_staged_line_touches = 0;
        uint64_t frontend_granules_built = 0;
        uint64_t unique_line_count = 0;
        uint64_t covered_line_count = 0;
        uint64_t overfetch_bytes = 0;
        uint64_t apply_bank_credit_effective = 0;
        uint64_t cmd_cost_veto = 0;
        uint64_t cmd_cost_veto_fine_gap = 0;
        uint64_t cmd_cost_veto_row_window = 0;
        uint64_t apply_issue_attempt_total = 0;
        uint64_t apply_issue_success_total = 0;
        uint64_t apply_issue_block_no_ready_total = 0;
        uint64_t apply_issue_block_inflight_cap_total = 0;
        uint64_t apply_issue_block_bank_credit_total = 0;
        uint64_t apply_issue_block_downstream_busy_total = 0;
        uint64_t apply_issue_block_retire_guard_total = 0;
        uint64_t apply_ready_queue_peak = 0;
        uint64_t apply_ready_queue_nonempty_cycles_total = 0;
        uint64_t apply_first_issue_delay_ns = 0;
        uint64_t apply_first_down_resp_delay_ns = 0;
        uint64_t apply_first_granule_done_delay_ns = 0;
        uint64_t apply_first_up_resp_delay_ns = 0;
        uint64_t apply_down_resp_total = 0;
        uint64_t apply_completed_granules_total = 0;
        uint64_t apply_emitted_subreads_total = 0;
        uint64_t apply_backlog_granules_residual = 0;
        uint64_t apply_backlog_pending_up_reads_residual = 0;
        uint64_t apply_backlog_inflight_residual = 0;
        uint64_t apply_backlog_granules_peak_after_due = 0;
        uint64_t apply_backlog_pending_up_reads_peak_after_due = 0;
        uint64_t apply_backlog_inflight_peak_after_due = 0;
        uint64_t retire_global_hol_cycles_total = 0;
        uint64_t retire_ready_but_blocked_edges_total = 0;
        uint64_t retire_per_post_progress_total = 0;
        uint64_t retire_wait_cycles_total = 0;
        uint64_t retire_wait_cycles_due_to_hol_total = 0;
        uint64_t retire_wait_cycles_due_to_barrier_total = 0;
        uint64_t retire_wait_cycles_due_to_not_ready_total = 0;
        uint64_t retire_samepost_blocked_edges_total = 0;
        uint64_t retire_crosspost_blocked_edges_total = 0;
        uint64_t retire_policy_loss_cycles_total = 0;
        uint64_t retire_policy_loss_edges_total = 0;
        uint64_t retire_shadow_per_post_recoverable_cycles_total = 0;
        uint64_t retire_shadow_per_post_recoverable_edges_total = 0;
        uint64_t retire_shadow_per_post_ready_posts_peak = 0;
        uint64_t retire_shadow_per_post_committable_edges_peak = 0;
        uint64_t retire_head_hol_cycles_dense_total = 0;
        uint64_t retire_head_hol_cycles_cache_total = 0;
        uint64_t retire_head_hol_cycles_miss_total = 0;
        uint64_t retire_head_hol_cycles_bcsr_total = 0;
        uint64_t retire_head_hol_cycles_bcsr_file_total = 0;
        uint64_t retire_head_blocked_edges_dense_total = 0;
        uint64_t retire_head_blocked_edges_cache_total = 0;
        uint64_t retire_head_blocked_edges_miss_total = 0;
        uint64_t retire_head_blocked_edges_bcsr_total = 0;
        uint64_t retire_head_blocked_edges_bcsr_file_total = 0;
        uint64_t retire_begin_apply_windows_total = 0;
        uint64_t retire_begin_apply_prev_edges_total = 0;
        uint64_t retire_begin_apply_outstanding_carryin_total = 0;
        uint64_t retire_begin_apply_outstanding_carryin_windows_total = 0;
        uint64_t retire_begin_apply_loader_not_ready_windows_total = 0;
        uint64_t retire_edge_retire_registered_total = 0;
        uint64_t retire_edge_retire_retired_total = 0;
        uint64_t retire_end_scatter_pending_direct_reads_residual_total = 0;
        uint64_t retire_end_scatter_outstanding_residual_total = 0;
        uint64_t retire_end_scatter_residual_work_windows_total = 0;
        uint64_t retire_ready_queue_peak = 0;
        uint64_t retire_unblock_events_total = 0;
        uint64_t step_barrier_wait_ns = 0;
    };
    std::unordered_map<uint32_t, StageMarks> stage_marks_;
    std::unordered_map<CoreStepKey, StageMarks, CoreStepKeyHash> core_stage_marks_;
    std::unordered_map<uint32_t, StepEndScatterResetState> step_endscatter_reset_state_;
    std::unordered_map<uint32_t, StepPerfAgg> step_perf_;
    std::unordered_map<CoreStepKey, StepPerfAgg, CoreStepKeyHash> core_step_perf_;

    // 测试注入一次跨核脉冲
    bool test_injected_ = false;
    
    // 子组件
    std::vector<CoreShellAPI*> cores_;
    SST::Interfaces::StandardMem* l2_cache_;
    SST::SnnDL::SnnInterface* external_nic_;
    
    // 内部架构组件
    OptimizedInternalRing* optimized_ring_;
    MultiCoreController* controller_;
    
    // 处理单元状态跟踪
    std::vector<ProcessingUnitState> unit_states_;
    
    // 外部端口
    SST::Link* external_spike_input_link_;
    SST::Link* external_spike_output_link_;
    SST::Link* mem_link_;
    
    // 网络方向端口（用于端口代理机制）
    SST::Link* north_link_;
    SST::Link* south_link_;
    SST::Link* east_link_;
    SST::Link* west_link_;
    SST::Link* network_link_;
    
    // 时钟计数器和测试流量
    uint64_t current_cycle_;
    uint64_t test_cycle_counter_;
    int test_spikes_sent_;  // 已发送的测试脉冲计数
    bool pure_snn_datapath_workload_eligible_ = false;
    bool local_storage_enable_ = false;

    // ===== NoC 端到端延迟画像（native multicast lab）=====
    uint32_t noc_lat_hist_max_ = 131072; // cycles；>hist_max 归入 overflow bin（hist_max+1）
    uint64_t noc_lat_spike_cnt_ = 0;
    uint64_t noc_lat_spike_sum_ = 0;
    uint64_t noc_lat_spike_max_ = 0;
    std::vector<uint64_t> noc_lat_spike_hist_{};
    uint64_t noc_lat_spikekey_cnt_ = 0;
    uint64_t noc_lat_spikekey_sum_ = 0;
    uint64_t noc_lat_spikekey_max_ = 0;
    std::vector<uint64_t> noc_lat_spikekey_hist_{};
    std::unique_ptr<LocalStorageHierarchyController> local_storage_controller_;
    std::unique_ptr<PodMetadataObjectPlane> pod_metadata_object_plane_;
    std::unique_ptr<PodOwnerServiceTable> pod_owner_service_table_;
    std::unique_ptr<PeLocalServiceObjectTable> pe_local_service_object_table_;
    std::unique_ptr<PeWeightObjectPlane> pe_weight_object_plane_;

    // ===== 时间窗口化统计（Batch‑B） =====
    bool window_stats_enable_ = false;
    uint64_t window_us_ = 20;      // 默认窗口长度
    uint64_t window_ns_ = 20000;   // 对应纳秒（1us=1000ns）
    std::string window_csv_;
    std::string window_metrics_csv_;
    bool diag_fire_log_ = false;
    // Exec-mode / workload hint (experiment observability only; does not change behavior)
    std::string exec_mode_;
    struct WindowAgg {
        uint64_t start_ns = 0;
        uint64_t end_ns = 0;
        uint64_t read_count = 0;          // 响应计数
        uint64_t read_latency_sum = 0;    // cycles 累计
        uint64_t issue_count = 0;         // 发起计数
        uint64_t req_size_sum = 0;        // bytes 累计
        uint64_t outstanding_sum = 0;     // 并发累计
        double   activity_f_sum = 0.0;    // 活跃度f累计（窗口内多次上报求均值）
        uint64_t activity_f_count = 0;    // 活跃度f样本数
        uint64_t sb_peak_bytes = 0;       // SRAM窗口峰值
        uint64_t inflight_peak = 0;       // 在途峰值
    };
    std::vector<WindowAgg> windows_;
    inline size_t getWindowIndex_(uint64_t now_ns) const {
        if (window_ns_ == 0) return 0; return static_cast<size_t>(now_ns / window_ns_);
    }
    inline WindowAgg& getOrCreateWindow_(uint64_t now_ns) {
        size_t idx = getWindowIndex_(now_ns);
        if (idx >= windows_.size()) windows_.resize(idx + 1);
        WindowAgg& w = windows_[idx];
        if (w.end_ns == 0) {
            uint64_t start = idx * window_ns_;
            w.start_ns = start;
            w.end_ns = start + window_ns_;
        }
        return w;
    }
    StepPerfAgg& getOrCreateStepPerf_(uint32_t seq);
    StepPerfAgg& getOrCreateCoreStepPerf_(int core_id, uint32_t seq);
    StageMarks& getOrCreateCoreStageMarks_(int core_id, uint32_t seq);

    // 外部端口 SpikeEvent 直注入（语义：仅本地投递；Stimulus 域）
    ExternalSpikeInputSubsystem external_spike_input_subsys_{};

    // Step-level random activation injection (Phase3-B): 下沉为独立子系统
    StepActivationSubsystem step_activation_subsys_{};

    // Global Step/GAS barrier sync (Phase-step-sync)
    enum class GlobalStepDonePolicy : uint8_t {
        EndScatter = 0,
        Drain = 1,      // drain-based：显式检查 core/NoC/NIC 后，再要求静默 N 周期
        Quiescent = 2,  // legacy：仅基于 hasWork/incomingQueue 的静默判定（保留兼容）
        FixedCycles = 3 // 每 step 运行固定 cycles 后强制完成（吞吐口径/调试用；不用于语义等价对比）
    };
    bool global_step_sync_enable_ = false;
    bool global_step_sync_ready_ = false;
    bool global_step_ready_sent_ = false;
    SST::Link* gas_step_ctrl_link_ = nullptr;
    SST::Link* loader_done_link_ = nullptr;
    bool global_step_start_pending_ = false;
    uint32_t global_step_pending_seq_ = 0;
    // Optional per-step apply bank credit target carried by START_STEP(seq) (0 means "no override").
    uint32_t global_step_pending_apply_bank_credit_target_ = 0;
    uint32_t global_step_active_seq_ = 0;
    uint32_t global_step_active_apply_bank_credit_target_ = 0;
    uint32_t global_step_last_seen_seq_ = 0;
    uint64_t global_step_start_dup_total_ = 0;
    uint64_t global_step_start_stale_total_ = 0;
    uint64_t global_step_start_jump_total_ = 0;
    bool global_step_done_sent_ = false;
    GlobalStepDonePolicy global_step_done_policy_ = GlobalStepDonePolicy::EndScatter;
    uint64_t global_step_begin_cycle_ = 0;
    uint64_t global_step_quiescent_min_cycles_ = 1;
    uint64_t global_step_drain_min_cycles_ = 200;
    uint64_t global_step_fixed_cycles_ = 0;
    uint64_t global_step_last_activity_cycle_ = 0;
    uint32_t global_step_drain_diag_count_ = 0;

    // step_seq 门控（naive_raw baseline）：用于禁用“步内级联”，将 step_seq>active 的 Spike packet 暂存到目标 step 再投递。
    struct DeferredNocPacket {
        int endpoint_id = -1;
        NocPacketEvent* pkt = nullptr;
    };
    std::unordered_map<uint32_t, std::vector<DeferredNocPacket>> deferred_packets_by_seq_{};
    void flushDeferredPacketsForSeq_(uint32_t seq);
    void clearAllDeferredPackets_();
    // Step-limited 下的就绪门控：等待 WeightLoader 发布 done 再发送 PE_READY（避免 naive_* 在 rowptr 未就绪时爆炸式积压）
    std::string loader_done_key_;
    bool wait_for_loader_done_ = false;
    bool loader_ready_latched_ = false;
    uint64_t loader_ready_cycle_ = 0;
    uint64_t loader_done_timeout_cycles_ = 0;
    bool loader_wait_started_ = false;
    uint64_t loader_wait_start_cycle_ = 0;
    bool loader_ready_forced_ = false;
    uint64_t global_step_ready_delay_cycles_ = 0;
    std::unique_ptr<SST::Shared::SharedArray<int>> loader_done_shared_;
    std::vector<uint8_t> global_step_done_cores_{};
    // 全局 Step 诊断：记录当前 active_seq 的每核最后一次阶段事件，便于在 finish() 处输出“卡在哪”
    // code: 0=None 1=BeginGather 2=BeginApply 3=EndApply 4=BeginScatter 5=EndScatter
    std::vector<uint8_t> global_step_last_stage_code_{};
    std::vector<uint32_t> global_step_last_stage_seq_{};
    std::vector<uint64_t> global_step_last_stage_ts_ns_{};
    std::vector<uint64_t> global_step_last_stage_spikes_{};
    // Spike packet bridge (Phase3-C): Spike 编解码与投递 glue 下沉到 synapse 域
    SpikePacketBridge spike_packet_bridge_{};
    // NoC 子系统（Phase4-A1.1）：收敛 send/recv/forward + 本地投递
    NocSubsystem noc_subsys_{};
    void writeWindowCsv_();
    void mergeWindowMetricsFromCsv_();
    
    // ===== 核心方法 =====

    void maybeInjectTestTraffic_(SST::Cycle_t current_cycle);
    void driveCoresManually_(SST::Cycle_t current_cycle);
    void refreshProcessingUnitStates_();
    
    /**
     * @brief 时钟滴答处理器
     */
    bool clockTick(SST::Cycle_t current_cycle);
    
    /**
     * @brief 处理外部脉冲事件（从Link接收）
     */
    void handleExternalSpikeEvent(SST::Event* ev);

    /**
     * @brief 处理全局 Step/GAS 控制器事件（StartStep）
     */
    void handleGasStepCtrlEvent(SST::Event* ev);

    /**
     * @brief 处理 WeightLoader 完成事件（跨rank桥接 loader_done_key）
     */
    void handleLoaderDoneEvent(SST::Event* ev);

    /**
     * @brief 在时钟边界打开所有 core 的新窗口（由 StartStep 驱动）
     */
    void beginGlobalStep_(uint32_t seq);
    void noteStepBarrierWaitNs_(uint32_t seq, uint64_t now_ns);
    
    /**
     * @brief 处理内部脉冲路由
     */
    /**
     * @brief 加载和分布权重
     */
    void loadAndDistributeWeights();
    
    /**
     * @brief 初始化SnnPE SubComponent核心
     */
    void initializeProcessingUnits();
    
    /**
     * @brief 初始化网络接口适配器
     */
    void initializeNetworkInterface();
    
    /**
     * @brief 初始化方向链路代理机制
     */
    void initializeDirectionLinks();
    
    /**
     * @brief 向指定核心递送脉冲
     */
    /**
     * @brief 将NoC packet按kind分流并投递到指定endpoint/core
     */
    void deliverPacketToEndpoint_(int endpoint_id, NocPacketEvent* pkt);
    void deliverPacketDirectToCore_(int endpoint_id, NocPacketEvent* pkt);
    void resetAllCoreMembranes();
    
    // === 网络端口事件处理器 ===
    
    /**
     * @brief 处理北向链路事件
     */
    void handleNorthLinkEvent(SST::Event* event);
    
    /**
     * @brief 处理南向链路事件
     */
    void handleSouthLinkEvent(SST::Event* event);
    
    /**
     * @brief 处理东向链路事件
     */
    void handleEastLinkEvent(SST::Event* event);
    
    /**
     * @brief 处理西向链路事件
     */
    void handleWestLinkEvent(SST::Event* event);
    
    /**
     * @brief 处理通用网络链路事件
     */
    void handleNetworkLinkEvent(SST::Event* event);
    

    
    /**
     * @brief 初始化内部互连
     */
    void initializeInternalRing();
    
    /**
     * @brief 初始化统计收集
     */
    void initializeStatistics();
    
    /**
     * @brief 更新统计信息
     */
    void updateStatistics();
    
    /**
     * @brief 生成测试流量
     */
    void generateTestTraffic();
    
    /**
     * @brief 负载均衡检查
     */
    void checkLoadBalance();
    
    /**
     * @brief 确定脉冲的目标处理单元
     */
    int determineTargetUnit(int neuron_id) const;
    
    /**
     * @brief 确定神经元ID是否属于本MultiCorePE
     */
    bool isLocalNeuron(int neuron_id) const;
};



/**
 * @brief 多核控制器
 */
class MultiCoreController {
public:
    MultiCoreController(MultiCorePE* parent, SST::Output* output);
    ~MultiCoreController();
    
    // 调度和负载均衡
    void scheduleWork();
    void balanceLoad();
    void tick();
    
    // 性能监控
    void updatePerformanceCounters();
    double getCoreUtilization(int core_id) const;
    double getOverallUtilization() const;
    
    // 统计信息
    uint64_t getTotalWorkDistributed() const { return total_work_distributed_; }
    int getLoadImbalanceCount() const { return load_imbalance_count_; }

private:
    MultiCorePE* parent_pe_;
    SST::Output* output_;
    
    // 负载均衡状态
    std::vector<double> core_utilization_history_;
    std::vector<uint64_t> core_work_count_;
    
    // 统计信息
    uint64_t total_work_distributed_;
    int load_imbalance_count_;
    double load_balance_threshold_;
    
    // 内部方法
    void redistributeWork();
    int findLeastLoadedCore() const;
    int findMostLoadedCore() const;
};

} // namespace SnnDL
} // namespace SST

#endif // _MULTICOREPE_H
