// -*- c++ -*-
//
// NocSpikeTransport: ISpikeTransport 的 NoC 适配器
// - 将 SpikeCommSubsystem 的 transport->send(ev) 映射为 INocTransport::sendFromCore(src_core, ev)
//

#pragma once

#include "INocTransport.h"
#include "ISpikeTransport.h"

namespace SST { namespace SnnDL {

class NocSpikeTransport final : public ISpikeTransport {
public:
    NocSpikeTransport() = default;
    NocSpikeTransport(INocTransport* noc, int src_core) : noc_(noc), src_core_(src_core) {}

    void setNocTransport(INocTransport* noc) { noc_ = noc; }
    void setSourceCore(int src_core) { src_core_ = src_core; }

    void send(SpikeEvent* event) override {
        if (noc_) noc_->sendFromCore(src_core_, event);
        else delete event;
    }

private:
    INocTransport* noc_ = nullptr;  // 非拥有
    int src_core_ = 0;
};

}} // namespace SST::SnnDL

