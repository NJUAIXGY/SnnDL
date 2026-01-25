// -*- c++ -*-
//
// MulticastNICConfig:
// - 将 MulticastNIC 构造期 params.find(...) 收敛到一处，降低“胶水”分散与漂移风险。
// - 仅做参数解析/默认值，不改变既有实现结构。
//

#pragma once

#include <cstdint>
#include <string>

#include <sst/core/params.h>

namespace SST { namespace SnnDL {

struct MulticastNICConfig {
    uint32_t node_id = 0;
    std::string port_name = "network";
    int verbose = 0;
};

inline MulticastNICConfig parseMulticastNICConfig(const SST::Params& params) {
    MulticastNICConfig c{};
    c.node_id = params.find<uint32_t>("node_id", 0);
    c.port_name = params.find<std::string>("port_name", "network");
    c.verbose = params.find<int>("verbose", 0);
    return c;
}

}} // namespace SST::SnnDL

