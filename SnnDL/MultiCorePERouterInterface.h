// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// MultiCorePERouterInterface.h: MultiCorePE专用的hr_router接口头文件
//

#ifndef _MULTICORPEROUTERINTERFACE_H
#define _MULTICORPEROUTERINTERFACE_H

#include <sst/core/subcomponent.h>
#include <sst/core/interfaces/simpleNetwork.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <sst/core/statapi/statbase.h>
#include <sst/core/statapi/stataccumulator.h>
#include <sst/core/event.h>
#include <queue>
#include <memory>
#include <functional>

#include "SnnInterface.h"
#include "SpikeEvent.h"
#include "SpikeEventWrapper.h"

namespace SST {
namespace SnnDL {

/**
 * @brief MultiCorePE专用的hr_router网络接口
 * 
 * 专门为MultiCorePE设计的网络接口，解决端口暴露和协议适配问题。
 * 不依赖父组件的injectDirectionLink机制，自主管理网络端口。
 */
class MultiCorePERouterInterface : public SnnInterface {
public:
    // SST组件注册宏
    SST_ELI_REGISTER_SUBCOMPONENT(
        MultiCorePERouterInterface,                    // 类名
        "SnnDL",                                       // 元素库名称  
        "MultiCorePERouterInterface",                  // 组件名称
        SST_ELI_ELEMENT_VERSION(1, 0, 0),             // 版本号
        "MultiCorePE专用hr_router网络接口",            // 描述
        SST::SnnDL::SnnInterface                      // 父接口
    )

    // 参数文档
    SST_ELI_DOCUMENT_PARAMS(
        {"node_id", "网络节点ID", "0"},
        {"link_bw", "链路带宽", "40GiB/s"},
        {"input_buf_size", "输入缓冲区大小", "2KiB"},
        {"output_buf_size", "输出缓冲区大小", "2KiB"},
        {"port_name", "网络端口名称", "network"},
        {"verbose", "日志详细级别", "0"},
        
        // LinkControl配置参数
        {"job_id", "作业ID（LinkControl需要）", "0"},
        {"job_size", "作业大小（LinkControl需要）", "1"}, 
        {"logical_nid", "逻辑节点ID（LinkControl需要）", "0"}
    )

    // 端口文档
    SST_ELI_DOCUMENT_PORTS(
        {"network", "连接到merlin.hr_router的端口", {"SimpleNetwork"}}
    )

    // 统计项文档
    SST_ELI_DOCUMENT_STATISTICS(
        {"spikes_sent", "发送的脉冲数量", "spikes", 1},
        {"spikes_received", "接收的脉冲数量", "spikes", 1},
        {"bytes_sent", "发送的字节数", "bytes", 1},
        {"bytes_received", "接收的字节数", "bytes", 1},
        {"packets_sent", "发送的数据包数", "packets", 1},
        {"packets_received", "接收的数据包数", "packets", 1},
        {"send_buffer_occupancy", "发送缓冲区占用率", "percent", 2},
        {"recv_buffer_occupancy", "接收缓冲区占用率", "percent", 2}
    )

    /**
     * @brief 构造函数
     * @param id 组件ID
     * @param params 参数集合
     */
    MultiCorePERouterInterface(SST::ComponentId_t id, SST::Params& params);

    /**
     * @brief 析构函数
     */
    virtual ~MultiCorePERouterInterface();

    // === SnnInterface接口实现 ===
    
    /**
     * @brief 设置脉冲接收处理器
     * @param handler 脉冲处理回调函数
     */
    void setSpikeHandler(SpikeHandler handler) override;

    /**
     * @brief 发送脉冲事件
     * @param spike_event 要发送的脉冲事件
     */
    void sendSpike(SpikeEvent* spike_event) override;

    /**
     * @brief 设置网络节点ID
     * @param node_id 节点ID
     */
    void setNodeId(uint32_t node_id) override;

    /**
     * @brief 获取网络节点ID
     * @return 当前节点ID
     */
    uint32_t getNodeId() const override;

    /**
     * @brief 获取网络状态信息
     * @return 状态字符串
     */
    std::string getNetworkStatus() const override;

    // === SST组件生命周期方法 ===
    
    /**
     * @brief 组件初始化
     * @param phase 初始化阶段
     */
    void init(unsigned int phase) override;

    /**
     * @brief 组件配置
     */
    void setup() override;

    /**
     * @brief 组件完成
     */
    void finish() override;

private:
    // === 核心配置 ===
    uint32_t node_id_;              ///< 网络节点ID
    uint32_t verbose_;              ///< 日志详细级别
    std::string port_name_;         ///< 网络端口名称
    
    // === 缓冲区配置 ===
    std::string link_bw_;           ///< 链路带宽
    std::string input_buf_size_;    ///< 输入缓冲区大小
    std::string output_buf_size_;   ///< 输出缓冲区大小
    
    // === SimpleNetwork集成 ===
    SST::Interfaces::SimpleNetwork* router_;   ///< LinkControl SubComponent
    
    // === 事件处理 ===
    SpikeHandler spike_handler_;    ///< 脉冲处理回调
    std::queue<SpikeEvent*> send_queue_;       ///< 发送队列
    
    // === 输出和统计 ===
    SST::Output* output_;           ///< 日志输出
    
    // 统计项
    Statistic<uint64_t>* stat_spikes_sent_;
    Statistic<uint64_t>* stat_spikes_received_;
    Statistic<uint64_t>* stat_bytes_sent_;
    Statistic<uint64_t>* stat_bytes_received_;
    Statistic<uint64_t>* stat_packets_sent_;
    Statistic<uint64_t>* stat_packets_received_;
    Statistic<double>* stat_send_buffer_occupancy_;
    Statistic<double>* stat_recv_buffer_occupancy_;
    
    // === 内部方法 ===
    
    /**
     * @brief 初始化SimpleNetwork接口
     */
    void initializeSimpleNetwork();
    
    /**
     * @brief 处理网络接收事件
     * @param vn 虚拟网络ID
     * @return 是否处理成功
     */
    bool handleNetworkEvent(int vn);
    
    /**
     * @brief 处理发送队列
     */
    void processSendQueue();
    
    /**
     * @brief 将SpikeEvent转换为SimpleNetwork::Request
     * @param spike_event 脉冲事件
     * @return 网络请求
     */
    SST::Interfaces::SimpleNetwork::Request* convertSpikeToRequest(SpikeEvent* spike_event);
    
    /**
     * @brief 将SimpleNetwork::Request转换为SpikeEvent
     * @param request 网络请求
     * @return 脉冲事件
     */
    SpikeEvent* convertRequestToSpike(SST::Interfaces::SimpleNetwork::Request* request);
    
    /**
     * @brief 更新缓冲区占用率统计
     */
    void updateBufferStats();
    
    /**
     * @brief 输出调试信息
     * @param level 日志级别
     * @param format 格式字符串
     */
    void debugPrint(uint32_t level, const char* format, ...);
};

} // namespace SnnDL
} // namespace SST

#endif /* _MULTICORPEROUTERINTERFACE_H */