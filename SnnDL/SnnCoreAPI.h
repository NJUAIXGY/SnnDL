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

#include "SpikeEvent.h"
#include "SnnPEParentInterface.h"

namespace SST {
namespace SnnDL {

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
    virtual void getStatistics(std::map<std::string, uint64_t>& stats) const = 0;
    virtual bool hasWork() const = 0;
    virtual double getUtilization() const = 0;
    // 可选：设置内存连接，默认空实现，具体实现可覆盖
    virtual void setMemoryLink(SST::Link* /*link*/) {}

protected:
    // 提供构造函数以便派生类在初始化列表中正确调用
    SnnCoreAPI(SST::ComponentId_t id, SST::Params& /*params*/) : SST::SubComponent(id) {}
};

} // namespace SnnDL
} // namespace SST

#endif // _SNNCORE_API_H

