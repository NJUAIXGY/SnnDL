// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// SnnNIC.h: SNN网络接口控制器头文件
//

#ifndef _SNNNIC_H
#define _SNNNIC_H

#include <sst/core/interfaces/simpleNetwork.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <sst/core/statapi/statbase.h>
#include <sst/core/clock.h>
#include <queue>
#include "SnnInterface.h"
#include "SpikeEvent.h"

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
        {"use_direct_link", "是否使用直接Link模式", "true"},
        {"verbose", "日志详细级别", "0"},
        // 新增：虚拟通道与批处理参数（默认保持兼容）
        {"virtual_channels", "虚拟通道数量(num_vns)", "2"},
        {"vn_spike_data", "单条脉冲使用的虚拟通道", "0"},
        {"vn_batch_data", "批量脉冲使用的虚拟通道", "1"},
        {"enable_batching", "是否启用批处理发送(0/1)", "0"},
        {"batch_size_local", "邻居节点批量大小阈值", "8"},
        {"batch_size_remote", "远程节点批量大小阈值", "32"},
        {"batch_flush_window", "批处理刷新窗口(ns)", "1000"},
        {"total_nodes", "网络总节点数(用于邻居判定)", "16"}
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
        {"batches_sent", "发送的批量包数量", "batches", 1}
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
    uint32_t virtual_channels_ = 2;            ///< 虚拟通道数量
    uint32_t vn_spike_data_ = 0;               ///< 单条脉冲VN
    uint32_t vn_batch_data_ = 1;               ///< 批量脉冲VN
    bool enable_batching_ = false;             ///< 是否启用批处理
    uint32_t batch_size_local_ = 8;            ///< 邻居批大小
    uint32_t batch_size_remote_ = 32;          ///< 远程批大小
    uint64_t batch_flush_window_ns_ = 1000;    ///< 刷新窗口
    uint32_t total_nodes_ = 16;                ///< 节点总数(网格尺寸推断)
    
    // 回调处理器
    SpikeHandler spike_handler;                ///< 脉冲接收处理器
    
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

    // 定时刷新（时间窗）
    bool flushClockTick(SST::Cycle_t currentCycle);
};

} // namespace SnnDL
} // namespace SST

#endif /* _SNNNIC_H */
