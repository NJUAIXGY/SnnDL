// -*- c++ -*-
//
// MulticastRouterConfig:
// - 将 MulticastRouter 构造期 params.find(...) 收敛到一处，降低“胶水”分散与漂移风险。
// - 仅做参数解析/默认值，不改变现有行为。
//

#pragma once

#include <cstdint>
#include <string>

#include <sst/core/params.h>

namespace SST { namespace SnnDL {

struct MulticastRouterConfig {
    int verbose = 0;
    uint32_t node_id = 0;
    uint64_t router_latency_cycles = 0;
    bool serialize_output_enable = false;
    uint64_t serialize_service_cycles = 1;
    std::string multicast_inter_policy = "xy";
    std::string multicast_intra_policy = "manhattan_x_first";
    std::string mesh_shape = "4x4";
};

inline MulticastRouterConfig parseMulticastRouterConfig(const SST::Params& params) {
    MulticastRouterConfig c{};
    c.verbose = params.find<int>("verbose", 0);
    c.node_id = params.find<uint32_t>("node_id", 0);
    c.router_latency_cycles = params.find<uint64_t>("router_latency_cycles", 0);
    c.serialize_output_enable = params.find<int>("serialize_output_enable", 0) != 0;
    c.serialize_service_cycles = params.find<uint64_t>("serialize_service_cycles", 1);
    if (c.serialize_service_cycles == 0) c.serialize_service_cycles = 1;
    c.multicast_inter_policy = params.find<std::string>("multicast_inter_policy", "xy");
    c.multicast_intra_policy = params.find<std::string>("multicast_intra_policy", "manhattan_x_first");
    c.mesh_shape = params.find<std::string>("mesh_shape", "4x4");
    return c;
}

}} // namespace SST::SnnDL

