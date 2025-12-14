// -*- c++ -*-
//
// StageEventHub: GAS阶段事件调度与统计汇报助手

#pragma once

#include <cstdint>

namespace SST { namespace SnnDL {

class SnnPESubComponent;

struct StageEventHub {
    SnnPESubComponent* core = nullptr;
    uint64_t t_begin_gather = 0;
    uint64_t t_begin_apply = 0;
    uint64_t t_begin_scatter = 0;
    bool have_begin_gather = false;
    bool have_begin_apply = false;
    bool have_begin_scatter = false;

    void init(SnnPESubComponent* owner) { core = owner; }
    void markBeginGather(uint32_t seq);
    void markBeginApply(uint32_t seq);
    void markBeginScatter(uint32_t seq);
    void markEndScatter(uint32_t seq, uint64_t spikes_emitted);
};

}} // namespace SST::SnnDL
