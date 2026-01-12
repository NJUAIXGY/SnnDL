// -*- c++ -*-
//
// MulticastRouter:
// - SnnDL 原生 NoC 路由组件（blocked multicast + block 间单播到 ingress）
// - 仅处理 NocPacketEvent（payload bytes 不透明；仅在 kind=SpikeKey 时解析路由头）
//

#pragma once

#include <array>
#include <cstdint>
#include <string>

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <sst/core/timeConverter.h>

namespace SST { namespace SnnDL {

class NocPacketEvent;

class MulticastRouter final : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(
        MulticastRouter,
        "SnnDL",
        "MulticastRouter",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "SnnDL native multicast router (blocked multicast)",
        COMPONENT_CATEGORY_NETWORK
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"node_id", "Router node id (must match PE node_id)", "0"},
        {"mesh_shape", "Mesh shape as 'WxH' (e.g. '4x4')", "4x4"},
        {"router_latency_cycles", "Additional per-hop latency in cycles (1 cycle ~= 1ns under 1GHz)", "0"},
        {"serialize_output_enable", "Enable per-output-port serialization (simple contention model)", "0"},
        {"serialize_service_cycles", "Per-packet service time per output port (cycles)", "1"},
        {"multicast_inter_policy", "SpikeKey INTER-stage unicast routing policy (xy|yx|hash_xy)", "xy"},
        {"multicast_intra_policy", "SpikeKey INTRA-stage blocked multicast tree policy (manhattan_x_first|manhattan_y_first)", "manhattan_x_first"},
        {"verbose", "日志详细级别", "0"}
    )

    SST_ELI_DOCUMENT_PORTS(
        {"local", "本地端口（连接到 MultiCorePE.network）", {"SnnDL.NocPacketEvent"}},
        {"north", "北向端口（连接到相邻 router）", {"SnnDL.NocPacketEvent"}},
        {"south", "南向端口（连接到相邻 router）", {"SnnDL.NocPacketEvent"}},
        {"east", "东向端口（连接到相邻 router）", {"SnnDL.NocPacketEvent"}},
        {"west", "西向端口（连接到相邻 router）", {"SnnDL.NocPacketEvent"}}
    )

    MulticastRouter(SST::ComponentId_t id, SST::Params& params);
    ~MulticastRouter();

    void init(unsigned int phase) override;
    void setup() override;
    void finish() override;

private:
    enum class InterBlockRoutePolicy : uint8_t { XY = 0, YX = 1, HashXY = 2 };
    enum class IntraBlockTreePolicy : uint8_t { ManhattanXFirst = 0, ManhattanYFirst = 1 };

    void handleLocalLinkEvent(SST::Event* event);
    void handleNorthLinkEvent(SST::Event* event);
    void handleSouthLinkEvent(SST::Event* event);
    void handleEastLinkEvent(SST::Event* event);
    void handleWestLinkEvent(SST::Event* event);
    void handlePacket_(NocPacketEvent* pkt);

    void routeUnicastXY_(NocPacketEvent* pkt);
    void routeUnicastYX_(NocPacketEvent* pkt);
    void routeSpikeKey_(NocPacketEvent* pkt);

    void sendToLocal_(NocPacketEvent* pkt);
    void sendToNorth_(NocPacketEvent* pkt);
    void sendToSouth_(NocPacketEvent* pkt);
    void sendToEast_(NocPacketEvent* pkt);
    void sendToWest_(NocPacketEvent* pkt);
    void sendOnPort_(SST::Link* link, uint32_t port_idx, NocPacketEvent* pkt);

    static bool parseMeshShape_(const std::string& shape, uint32_t& out_w, uint32_t& out_h);
    static bool parseInterPolicy_(const std::string& s, InterBlockRoutePolicy& out);
    static bool parseIntraPolicy_(const std::string& s, IntraBlockTreePolicy& out);

    SST::Output out_;
    uint32_t node_id_ = 0;
    uint32_t mesh_w_ = 0;
    uint32_t mesh_h_ = 0;
    SST::TimeConverter timebase_;
    uint64_t router_latency_cycles_ = 0;
    bool serialize_output_enable_ = false;
    uint64_t serialize_service_cycles_ = 1;
    InterBlockRoutePolicy multicast_inter_policy_ = InterBlockRoutePolicy::XY;
    IntraBlockTreePolicy multicast_intra_policy_ = IntraBlockTreePolicy::ManhattanXFirst;
    std::array<uint64_t, 5> port_next_free_cycle_{{0, 0, 0, 0, 0}}; // local,n,s,e,w

    SST::Link* local_link_ = nullptr;
    SST::Link* north_link_ = nullptr;
    SST::Link* south_link_ = nullptr;
    SST::Link* east_link_ = nullptr;
    SST::Link* west_link_ = nullptr;

    uint64_t pkts_in_total_ = 0;
    uint64_t pkts_out_total_ = 0;
    uint64_t pkts_to_local_ = 0;
    uint64_t pkts_forward_xy_ = 0;

    uint64_t spikekey_in_total_ = 0;
    uint64_t spikekey_stage_inter_ = 0;
    uint64_t spikekey_stage_intra_ = 0;
    uint64_t spikekey_clones_total_ = 0;
};

}} // namespace SST::SnnDL
