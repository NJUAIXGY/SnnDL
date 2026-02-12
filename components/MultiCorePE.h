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
#include <random>
#include <limits>

#include "events/SpikeEvent.h"
#include "SnnInterface.h"
#include "SnnPEParentInterface.h"
#include "IPeAggregation.h"
#include "CoreShellAPI.h"
#include "../api/GlobalNeuronLayout.h"
#include "noc/OptimizedInternalRing.h"
#include "stimulus/ExternalSpikeInputSubsystem.h"
#include "stimulus/StepActivationSubsystem.h"
#include "noc/NocSubsystem.h"
#include "synapse/route/SpikePacketBridge.h"
#include "workload_stats/IWorkloadStatsModule.h"

namespace SST {
namespace SnnDL {

// 前置声明
class MultiCoreController;
class SnnNetworkAdapter;
class NocPacketEvent;

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
class MultiCorePE : public SST::Component, public SnnPEParentInterface, public IPeAggregation, public IWorkloadStatRegistrar {
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
        {"sim_stop_ns",      "组件主控结束仿真（纳秒）。>0时注册为primary并在达到该时间点时OKToEndSim", "0"},
        {"weights_file",     "权重文件路径", ""},
        {"enable_numa",      "启用NUMA优化", "1"},
        {"workload_impl",    "workload实现选择(snn|stream|traffic|tensor). 为空则回退 env:SNNDL_WORKLOAD_IMPL", ""},
        {"workload_stats_modules", "workload统计模块(逗号分隔). 为空则按workload_impl自动选择", ""},
        {"exec_mode",        "执行模式提示(gas|naive_raw). 仅用于实验可观测性，不改变行为", "gas"},
        {"v_thresh",         "触发脉冲的膜电位阈值", "1.0"},
        {"v_reset",          "脉冲发放后膜电位重置值", "0.0"},
        {"v_rest",           "静息膜电位", "0.0"},
        {"tau_mem",          "膜电位泄漏时间常数(ms)", "20.0"},
        {"t_ref",            "不应期时长(时钟周期)", "2"},
        {"enable_test_traffic", "是否启用网络测试流量", "0"},
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
        {"memory_interface", "内存接口", "SST::Interfaces::StandardMem"},
        {"external_nic", "外部网络接口", "SST::SnnDL::SnnInterface"}
    )

    // 端口文档 
    SST_ELI_DOCUMENT_PORTS(
        {"external_spike_input",  "外部脉冲输入端口", {"SnnDL.SpikeEvent"}},
        {"external_spike_output", "外部脉冲输出端口", {"SnnDL.SpikeEvent"}},
        {"gas_step_ctrl", "全局 Step/GAS 同步控制器端口（可选）", {"SnnDL.GasStepBarrierEvent"}},
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
	        // Stream workload（PE聚合，供 essential_summary_mesh 汇总）
	        {"stream_mem_writes_issued_total", "Stream workload: total writes issued（PE聚合）", "requests", 1},
	        {"stream_mem_reads_issued_total", "Stream workload: total reads issued（PE聚合）", "requests", 1},
	        {"stream_mem_bytes_written_total", "Stream workload: bytes written（issued, PE聚合）", "bytes", 1},
	        {"stream_mem_bytes_read_total", "Stream workload: bytes read（issued, PE聚合）", "bytes", 1},
	        {"stream_mem_verify_pass_total", "Stream workload: read-after-write 校验通过次数（PE聚合）", "count", 1},
	        {"stream_mem_verify_fail_total", "Stream workload: read-after-write 校验失败次数（PE聚合）", "count", 1},
	        {"stream_pkt_sent_total", "Stream workload: raw-bytes packets sent（PE聚合）", "packets", 1},
	        {"stream_pkt_recv_total", "Stream workload: raw-bytes packets received（PE聚合）", "packets", 1},
	        {"stream_pkt_bad_crc_total", "Stream workload: bad CRC packets（PE聚合）", "packets", 1},
	        {"stream_pkt_bad_magic_total", "Stream workload: bad magic packets（PE聚合）", "packets", 1},
	        // Tensor workload（PE聚合，供 essential_summary_mesh 汇总）
	        {"tensor_mem_reads_issued_total", "Tensor workload: total reads issued（PE聚合）", "requests", 1},
	        {"tensor_mem_writes_issued_total", "Tensor workload: total writes issued（PE聚合）", "requests", 1},
	        {"tensor_mem_bytes_read_total", "Tensor workload: bytes read（issued, PE聚合）", "bytes", 1},
	        {"tensor_mem_bytes_write_total", "Tensor workload: bytes written（issued, PE聚合）", "bytes", 1},
        {"tensor_compute_cycles_total", "Tensor workload: compute cycles（PE聚合）", "cycles", 1},
        {"tensor_mac_ops_total", "Tensor workload: MAC ops（PE聚合）", "ops", 1},
        {"tensor_dma_stall_cycles_total", "Tensor workload: DMA stall cycles（PE聚合）", "cycles", 1},
        {"tensor_iter_cycles_total", "Tensor workload: iteration cycles（PE聚合）", "cycles", 1},
        {"tensor_stall_dma_budget_cycles_total", "Tensor workload: stall (DMA budget) cycles（PE聚合）", "cycles", 1},
        {"tensor_stall_mem_outstanding_cycles_total", "Tensor workload: stall (mem outstanding) cycles（PE聚合）", "cycles", 1},
        {"tensor_stall_wait_read_cycles_total", "Tensor workload: stall (wait read) cycles（PE聚合）", "cycles", 1},
        {"tensor_stall_wait_write_cycles_total", "Tensor workload: stall (wait write) cycles（PE聚合）", "cycles", 1},
        {"tensor_stall_collective_cycles_total", "Tensor workload: stall (collective barrier) cycles（PE聚合）", "cycles", 1},
        {"tensor_dma_cycles_total", "Tensor workload: DMA cycles（PE聚合）", "cycles", 1},
        {"tensor_dram_bytes_total", "Tensor workload: DRAM bytes（PE聚合）", "bytes", 1},
        {"tensor_onchip_bytes_total", "Tensor workload: on-chip bytes（PE聚合）", "bytes", 1},
        {"tensor_tile_count_total", "Tensor workload: tile count（PE聚合）", "tiles", 1},
        {"tensor_collective_bytes_sent_total", "Tensor workload: collective bytes sent（PE聚合）", "bytes", 1},
        {"tensor_collective_bytes_recv_total", "Tensor workload: collective bytes received（PE聚合）", "bytes", 1},
        {"tensor_collective_pkts_sent_total", "Tensor workload: collective packets sent（PE聚合）", "packets", 1},
        {"tensor_collective_pkts_recv_total", "Tensor workload: collective packets received（PE聚合）", "packets", 1},
        {"tensor_collective_cycles_total", "Tensor workload: collective cycles（PE聚合）", "cycles", 1},
        {"tensor_pkt_sent_total", "Tensor workload: RawBytes packets sent（PE聚合）", "packets", 1},
        {"tensor_pkt_recv_total", "Tensor workload: RawBytes packets received（PE聚合）", "packets", 1},
        {"tensor_pkt_bytes_sent_total", "Tensor workload: RawBytes bytes sent（PE聚合）", "bytes", 1},
        {"tensor_pkt_bytes_recv_total", "Tensor workload: RawBytes bytes received（PE聚合）", "bytes", 1},
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
        {"gas_apply_acc_updates_total", "Apply阶段的delta累加次数（总）", "count", 1},
        {"gas_acc_posts_touched_total", "Apply阶段触达post个数（总）", "posts", 1},
        {"gas_scatter_spikes_emitted_total", "Scatter阶段发放spike个数（总）", "spikes", 1},
        {"gas_acc_high_watermark_bytes_total", "累加器峰值占用（总）", "bytes", 1},
        {"gas_acc_spill_records_total", "溢写记录条数（总）", "records", 1},
        {"gas_acc_spilled_bytes_total", "溢写有效字节（总）", "bytes", 1},
        {"mem_outstanding_at_issue", "发起时并发请求数", "count", 1},
        {"gas_activity_f", "GAS 窗口内活跃度 f（活跃轴数/列宽）", "ratio", 1},
        {"sim_cycles_total", "总仿真周期（组件clock tick累计）", "cycles", 1},
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
        {"loader_done_timeout_fallback_total", "loader_done_key 等待超时后触发降级发送 PE_READY 的次数", "count", 1}
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
                               uint64_t window_buffer_max_bytes = 0);
    void accumulateActivityF(double f);
    void accumulateApplyScatterStats(uint64_t acc_updates, uint64_t posts_touched,
                                     uint64_t spikes_emitted, uint64_t hwm_bytes,
                                     uint64_t spill_records, uint64_t spilled_bytes);
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

    // ===== SnnPEParentInterface 实现 =====
    
    /**
     * @brief 向父级组件发送脉冲（从SnnPE SubComponent调用）
     */
    void sendSpike(SpikeEvent* event) override;
    
    /**
     * @brief 向父级组件请求内存访问（从SnnPE SubComponent调用）
     */
    void requestMemoryAccess(uint64_t address, size_t size, 
                                    std::function<void(const void*)> callback) override;
    
    /**
     * @brief 获取当前仿真周期
     */
    uint64_t getCurrentCycle() const override { return current_cycle_; }
    
    /**
     * @brief 获取本PE的节点ID
     */
    int getNodeId() const override { return node_id_; }
    
    /**
     * @brief 获取本PE管理的神经元总数
     */
    int getTotalNeurons() const override { return total_neurons_; }

    SST::Statistics::Statistic<uint64_t>* registerU64(const std::string& stat_name) override {
        return registerStatistic<uint64_t>(stat_name);
    }

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
    int verbose_;
    std::string clock_freq_ = "1GHz";
    std::string weights_file_;
    bool enable_numa_;
    bool enable_test_traffic_;

    // 全局 neuron_id 布局（单一真源，供 Step/Route/NoC 等口径复用）
    GlobalNeuronLayout global_layout_{};
    
    // 神经元参数
    float v_thresh_;
    float v_reset_;
    float v_rest_;
    float tau_mem_;
    int t_ref_;
    
    // 测试流量参数
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
    Statistic<uint64_t>* stat_gas_apply_acc_updates_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_acc_posts_touched_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_scatter_spikes_emitted_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_acc_hwm_bytes_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_acc_spill_records_total_ = nullptr;
    Statistic<uint64_t>* stat_gas_acc_spilled_bytes_total_ = nullptr;
    Statistic<double>*  stat_gas_activity_f_ = nullptr;
    Statistic<uint64_t>* stat_mem_outstanding_at_issue_ = nullptr;
    Statistic<uint64_t>* stat_sim_cycles_total_ = nullptr;
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
    std::unordered_map<uint32_t, StageMarks> stage_marks_;

    // 测试注入一次跨核脉冲
    bool test_injected_ = false;
    
    // 子组件
    std::vector<CoreShellAPI*> cores_;
    SST::Interfaces::StandardMem* l2_cache_;
    SST::Interfaces::StandardMem* memory_interface_;
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
    uint32_t global_step_active_seq_ = 0;
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
