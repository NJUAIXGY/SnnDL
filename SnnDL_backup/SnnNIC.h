// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// SnnNIC.h: SNN网络接口控制器头文件
// Recommended NIC for SnnDL. Integrates with merlin.linkcontrol (SimpleNetwork)
// and is the default path for MultiCorePE + SnnPESubComponent simulations.
//

#ifndef _SNNNIC_H
#define _SNNNIC_H

#include <sst/core/interfaces/simpleNetwork.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <sst/core/statapi/statbase.h>
#include <sst/core/clock.h>
#include <queue>
#include <vector>
#include "SnnInterface.h"
#include "SpikeEvent.h"
#include "GatingDecisionEvent.h"

namespace SST {
namespace SnnDL {

/**
 * @brief SNN网络接口控制器
 * 
 * 该组件实现了SnnInterface接口，作为SnnPE与merlin网络之间的适配器。
 * 它将SpikeEvent转换为网络数据包，并处理网络通信的复杂性。
 */
class SnnNIC : public SnnInterface {
public:
    // SST组件注册宏
    SST_ELI_REGISTER_SUBCOMPONENT(
        SnnNIC,                                    // 类名
        "SnnDL",                                  // 元素库名称
        "SnnNIC",                                 // 组件名称
        SST_ELI_ELEMENT_VERSION(1, 0, 0),         // 版本号
        "SNN网络接口控制器",                        // 描述
        SST::SnnDL::SnnInterface                  // 父接口
    )

    // 参数文档
    SST_ELI_DOCUMENT_PARAMS(
        {"node_id", "网络节点ID", "0"},
        {"link_bw", "网络链路带宽", "40GiB/s"},
        {"input_buf_size", "输入缓冲区大小", "1KiB"},
        {"output_buf_size", "输出缓冲区大小", "1KiB"},
        {"port_name", "网络端口名称", "network"},
        {"use_direct_link", "是否使用直接Link模式", "false"},
        {"verbose", "日志详细级别", "0"},
        // 兼容参数（已忽略/禁用）：SnnNIC 现强制单VN(VN=0)、禁用跨Rank聚合
        {"virtual_channels", "[已忽略] 虚拟通道数量(num_vns)，强制为1", "1"},
        {"network_num_vns", "[已忽略] LinkControl虚拟通道数，强制为1", "0"},
        {"vn_spike_data", "[已忽略] 单条脉冲使用的虚拟通道，强制为0", "0"},
        {"vn_batch_data", "[已忽略] 批量脉冲使用的虚拟通道，强制为0", "1"},
        {"flush_on_credit", "[可选] 是否在credit回调中刷新批处理(1=开,0=关)", "1"},
        {"probe_vn_on_setup", "[已忽略] setup阶段VN探测(0/1)", "0"},
        {"enable_batching", "是否启用批处理发送(0/1)", "0"},
        {"batch_size_local", "邻居节点批量大小阈值", "8"},
        {"batch_size_remote", "远程节点批量大小阈值", "32"},
        {"batch_flush_window", "批处理刷新窗口(ns)", "1000"},
        {"total_nodes", "网络总节点数(用于邻居判定)", "16"},
        {"enable_inter_rank_batching", "[已禁用] 启用跨Rank代理聚合(0/1)", "0"},
        {"inter_rank_batch_window", "[已禁用] 跨Rank批量窗口(ns)", "0"},
        {"nodes_per_rank", "[已禁用] 每个rank的节点数(用于简化映射)", "0"}
    )

    // 端口文档
    SST_ELI_DOCUMENT_PORTS(
        {"network", "连接到merlin.linkcontrol或路由器的端口", {"SimpleNetwork"}}
    )

    // SubComponent槽位文档
    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS(
        {"linkcontrol", "merlin LinkControl子组件", "SST::Interfaces::SimpleNetwork"}
    )

    // 统计信息文档
    SST_ELI_DOCUMENT_STATISTICS(
        {"spikes_sent", "发送的脉冲数量", "spikes", 1},
        {"spikes_received", "接收的脉冲数量", "spikes", 1},
        {"packets_sent", "发送的网络数据包数量", "packets", 1},
        {"packets_received", "接收的网络数据包数量", "packets", 1},
        {"batches_sent", "发送的批量包数量", "batches", 1},
        {"inter_rank_batches_sent", "跨Rank代理批量发送次数", "batches", 1}
    )

    /**
     * @brief 构造函数
     * @param id 组件ID
     * @param params 配置参数
     */
    SnnNIC(SST::ComponentId_t id, SST::Params& params);

    /**
     * @brief 析构函数
     */
    ~SnnNIC();

    // === SnnInterface 接口实现 ===
    void setSpikeHandler(SpikeHandler handler) override;
    void sendSpike(SpikeEvent* spike_event) override;
    void setNodeId(uint32_t node_id) override;
    uint32_t getNodeId() const override;
    std::string getNetworkStatus() const override;

    // === SimpleNetwork 回调方法 ===
    bool handleIncoming(int vn);
    bool spaceAvailable(int vn);
    
    // === 直接Link 回调方法 ===
    void handleDirectSpikeEvent(SST::Event* event);

    // === SST组件生命周期方法 ===
    void init(unsigned int phase) override;
    void setup() override;
    void finish() override;

private:
    // === 内部方法 ===
    
    /**
     * @brief 将SpikeEvent打包成网络请求
     * @param spike_event 脉冲事件
     * @param dest_node 目标节点ID
     * @return 网络请求对象
     */
    SST::Interfaces::SimpleNetwork::Request* createNetworkRequest(
        SpikeEvent* spike_event, uint32_t dest_node);
    
    /**
     * @brief 从网络请求中解包SpikeEvent
     * @param req 网络请求
     * @return 解包的脉冲事件（如果成功）
     */
    SpikeEvent* extractSpikeEvent(SST::Interfaces::SimpleNetwork::Request* req);

    // === 成员变量 ===
    
    // SST基础设施
    SST::Output* output;                        ///< 日志输出对象
    SST::Interfaces::SimpleNetwork* network;   ///< 网络接口
    SST::Link* direct_link;                     ///< 直接链接接口
    
    // 网络配置
    uint32_t node_id;                          ///< 本节点ID
    std::string link_bw;                       ///< 链路带宽
    std::string input_buf_size;                ///< 输入缓冲区大小
    std::string output_buf_size;               ///< 输出缓冲区大小
    bool use_direct_link;                      ///< 是否使用直接链接模式
    uint32_t virtual_channels_ = 1;            ///< [强制单VN] 虚拟通道数量固定为1
    uint32_t network_num_vns_ = 1;             ///< [强制单VN] LinkControl虚拟通道数固定为1
    uint32_t effective_num_vns_ = 1;           ///< [强制单VN]
    uint32_t vn_spike_data_ = 0;               ///< [忽略] 单条脉冲VN固定为0
    uint32_t vn_batch_data_ = 0;               ///< [忽略] 批量脉冲VN固定为0
    uint32_t vn_control_ = 0;                  ///< [忽略] 控制VN固定为0
    bool auto_vn_fallback_ = false;            ///< [忽略] 禁用VN降级逻辑
    bool enable_batching_ = false;             ///< 是否启用批处理
    bool flush_on_credit_ = true;              ///< 在spaceAvailable回调中是否触发刷新
    bool probe_vn_on_setup_ = false;           ///< 在setup阶段主动探测VN以预热
    uint32_t batch_size_local_ = 8;            ///< 邻居批大小
    uint32_t batch_size_remote_ = 32;          ///< 远程批大小
    uint64_t batch_flush_window_ns_ = 1000;    ///< 刷新窗口
    uint32_t total_nodes_ = 16;                ///< 节点总数(网格尺寸推断)
    bool vn_guard_warned_ = false;             ///< VN越界回退仅提示一次
    
    // 回调处理器
    SpikeHandler spike_handler;                ///< 脉冲接收处理器
    std::function<void(SST::Event*)> control_handler_; ///< 控制事件处理器（可选）
    
    // 统计计数器
    uint64_t spikes_sent_count;
    uint64_t spikes_received_count;
    uint64_t packets_sent_count;
    uint64_t packets_received_count;
    
    // 统计对象
    Statistic<uint64_t>* stat_spikes_sent;
    Statistic<uint64_t>* stat_spikes_received;
    Statistic<uint64_t>* stat_packets_sent;
    Statistic<uint64_t>* stat_packets_received;
    Statistic<uint64_t>* stat_batches_sent = nullptr;   ///< 批量发送次数
    Statistic<uint64_t>* stat_ir_batches_sent = nullptr;///< 跨Rank代理批量发送次数
    
    // 待发送队列（可选，用于流量控制）
    std::queue<SpikeEvent*> pending_spikes;

    // 简单批处理：按目标节点分桶
    std::map<uint32_t, std::vector<SpikeEvent*>> batch_buckets_;  ///< dest_node -> spikes
    std::map<uint32_t, uint64_t> batch_earliest_ts_;              ///< dest_node -> earliest ts

    // 批处理辅助
    void tryBatchSpike(SpikeEvent* spike);
    void flushBatchToNode(uint32_t dest_node);
    void flushAllBatches();
    bool isNeighborNode(uint32_t dest_node) const;
    uint32_t manhattanDistance(uint32_t src_node, uint32_t dst_node) const;

    // 跨Rank代理聚合（可选，默认关闭）
    bool enable_inter_rank_batching_ = false;   ///< [禁用] 跨Rank代理聚合
    uint64_t inter_rank_batch_window_ns_ = 0;   ///< [禁用]
    uint32_t nodes_per_rank_ = 0;               ///< [禁用]
    std::map<uint32_t, std::vector<SpikeEvent*>> ir_buckets_;     ///< dest_rank -> spikes
    std::map<uint32_t, uint64_t> ir_earliest_ts_;                 ///< dest_rank -> earliest ts
    void tryInterRankBatch(SpikeEvent* spike);
    void flushInterRankTo(uint32_t dest_rank);
    void flushInterRankAll();
    inline uint32_t computeRankForNode(uint32_t node) const {
        return (nodes_per_rank_ > 0) ? (node / nodes_per_rank_) : 0;
    }
    inline uint32_t computeGatewayNodeForRank(uint32_t rank) const {
        return rank * nodes_per_rank_; // 选用该rank首个节点作为网关
    }

    // 定时刷新（时间窗）
    bool flushClockTick(SST::Cycle_t currentCycle);

    // 网络初始化与就绪状态
    bool network_ready_ = false;               ///< 在setup完成后标记为就绪，避免过早发送
    bool link_ready_ = false;                  ///< 首次收到发送回调后标记为就绪，确保端口握手完成
    bool init_done_ = false;                   ///< 在init阶段>=2后标记为就绪，确保VN映射完成
    std::vector<bool> vn_ready_;              ///< 按VN记录是否已收到发送回调

    // 调试辅助：仅首次打印门控快照，避免刷屏
    bool gating_logged_send_ = false;
    bool gating_logged_flush_ = false;
    void logGatingSnapshot(const char* ctx) const;

public:
    // 可选：设置控制事件处理器
    void setControlHandler(std::function<void(SST::Event*)> handler) { control_handler_ = std::move(handler); }
    // 可选：发送控制事件到目标节点
    bool sendControl(SST::Event* ev, uint32_t dest_node);
};

} // namespace SnnDL
} // namespace SST

#endif /* _SNNNIC_H */
