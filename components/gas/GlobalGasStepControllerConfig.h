// -*- c++ -*-
//
// GlobalGasStepControllerConfig:
// - 将 GlobalGasStepController 构造期 params.find(...) 收敛到一处，降低“胶水”分散与漂移风险。
// - 仅做参数解析/默认值，行为由使用侧决定。
//

#pragma once

#include <cstdint>

#include <sst/core/params.h>

namespace SST { namespace SnnDL {

struct GlobalGasStepControllerConfig {
    int verbose = 0;
    uint32_t start_seq = 1;
    uint32_t max_steps = 0;
    bool require_all_ready = true;
    bool strict_seq_check = false;
};

inline GlobalGasStepControllerConfig parseGlobalGasStepControllerConfig(const SST::Params& params) {
    GlobalGasStepControllerConfig c{};
    c.verbose = params.find<int>("verbose", 0);
    c.start_seq = params.find<uint32_t>("start_seq", 1);
    c.max_steps = params.find<uint32_t>("max_steps", 0);
    c.require_all_ready = params.find<int>("require_all_ready", 1) != 0;
    c.strict_seq_check = params.find<int>("strict_seq_check", 0) != 0;
    return c;
}

}} // namespace SST::SnnDL
