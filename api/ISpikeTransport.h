// -*- c++ -*-
//
// ISpikeTransport: 抽象脉冲传输层，屏蔽父组件/NIC/MPI等具体实现。
//

#pragma once

#include "SpikeEvent.h"
#include "SnnPEParentInterface.h"

namespace SST { namespace SnnDL {

class ISpikeTransport {
public:
    virtual ~ISpikeTransport() = default;
    // 语义：接管 SpikeEvent 生命周期（与 parent_->sendSpike 一致）
    virtual void send(SpikeEvent* event) = 0;
};

// 默认适配器：复用现有 SnnPEParentInterface 路径，保持兼容。
class ParentSpikeTransport final : public ISpikeTransport {
public:
    explicit ParentSpikeTransport(SnnPEParentInterface* parent) : parent_(parent) {}
    void setParent(SnnPEParentInterface* parent) { parent_ = parent; }
    void send(SpikeEvent* event) override {
        if (parent_) parent_->sendSpike(event);
        else delete event;
    }
private:
    SnnPEParentInterface* parent_ = nullptr;
};

}} // namespace SST::SnnDL
