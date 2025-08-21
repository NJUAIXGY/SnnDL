// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// SnnInterface.h: SNN网络接口基类
//

#ifndef _SNNINTERFACE_H
#define _SNNINTERFACE_H

#include <sst/core/subcomponent.h>
#include <sst/core/params.h>
#include <functional>
#include <string>
#include "SpikeEvent.h"

namespace SST {
namespace SnnDL {

/**
 * @brief SNN网络接口基类
 * 
 * 这是一个SubComponent基类，定义了SnnPE与网络通信的标准接口。
 * 类似于miranda.BaseCPU使用memory和generator插槽的方式，
 * SnnPE可以使用这个接口插槽来实现网络通信的抽象。
 */
class SnnInterface : public SST::SubComponent {
public:
    /**
     * @brief 脉冲处理回调函数类型
     * @param spike_event 接收到的脉冲事件指针
     */
    typedef std::function<void(SpikeEvent*)> SpikeHandler;

    /**
     * @brief 构造函数
     * @param id 组件ID
     * @param params 参数集合
     */
    SnnInterface(SST::ComponentId_t id, SST::Params& params)
        : SST::SubComponent(id)
    {
        // 基类构造器
    }

    /**
     * @brief 虚析构函数
     */
    virtual ~SnnInterface() {}

    // === ELI注册宏 ===
    SST_ELI_REGISTER_SUBCOMPONENT_API(SST::SnnDL::SnnInterface)

    // === 纯虚接口方法 ===

    /**
     * @brief 设置脉冲接收处理器
     * @param handler 脉冲处理回调函数
     */
    virtual void setSpikeHandler(SpikeHandler handler) = 0;

    /**
     * @brief 发送脉冲事件
     * @param spike_event 要发送的脉冲事件
     */
    virtual void sendSpike(SpikeEvent* spike_event) = 0;

    /**
     * @brief 设置网络节点ID
     * @param node_id 节点ID
     */
    virtual void setNodeId(uint32_t node_id) = 0;

    /**
     * @brief 获取网络节点ID
     * @return 当前节点ID
     */
    virtual uint32_t getNodeId() const = 0;

    /**
     * @brief 获取网络状态信息
     * @return 状态字符串
     */
    virtual std::string getNetworkStatus() const = 0;
};

} // namespace SnnDL
} // namespace SST

#endif /* _SNNINTERFACE_H */
