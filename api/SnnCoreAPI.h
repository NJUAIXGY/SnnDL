// -*- c++ -*-
//
// SnnCoreAPI.h: SnnPE计算核心的SubComponent接口(API)
//

#ifndef _SNNCORE_API_H
#define _SNNCORE_API_H

#include <sst/core/subcomponent.h>
#include <sst/core/params.h>
#include <sst/core/link.h>
#include <map>

namespace SST {
namespace SnnDL {

class SpikeEvent;
class NocPacketEvent;
class SnnPEParentInterface;

class SnnCoreAPI : public SST::SubComponent {
public:
    // 注册为可加载的SubComponent API，构造签名为 (ComponentId_t, Params&)
    SST_ELI_REGISTER_SUBCOMPONENT_API(SST::SnnDL::SnnCoreAPI)

    virtual ~SnnCoreAPI() = default;

    // 生命周期（可选覆写）
    using SubComponent::init;
    using SubComponent::setup;
    using SubComponent::finish;

    // 与父组件通信
    virtual void setParentInterface(SnnPEParentInterface* parent) = 0;

    // 业务接口
    virtual void deliverSpike(SpikeEvent* spike) = 0;
    // 通用 NoC packet 输入（非 Spike 语义）：默认不处理，由上层回收 packet 内存。
    // 返回 true 表示本 core 接管 packet 生命周期；返回 false 表示未处理、由调用方负责 delete。
    virtual bool deliverPacket(NocPacketEvent* /*packet*/) { return false; }
    virtual void getStatistics(std::map<std::string, uint64_t>& stats) const = 0;
    virtual bool hasWork() const = 0;
    virtual double getUtilization() const = 0;
    // 可选：设置内存连接，默认空实现，具体实现可覆盖
    virtual void setMemoryLink(SST::Link* /*link*/) {}
    // 步级 reset：默认不做操作，具体实现可覆盖
    virtual void resetMembraneState(float /*v_rest*/) {}
    // 手动驱动：强制当前Gather窗口结束（上层在manual_window_drive模式下触发）
    virtual void forceEndGather() {}

protected:
    // 提供构造函数以便派生类在初始化列表中正确调用
    SnnCoreAPI(SST::ComponentId_t id, SST::Params& /*params*/) : SST::SubComponent(id) {}
};

} // namespace SnnDL
} // namespace SST

#endif // _SNNCORE_API_H
