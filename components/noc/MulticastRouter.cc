// -*- c++ -*-
//
// MulticastRouter implementation
//

#include "MulticastRouter.h"

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <string>
#include <inttypes.h>

#include <sst/core/event.h>
#include <sst/core/params.h>

#include "NocPacketEvent.h"
#include "synapse/route/SpikeNocCodec.h"

namespace SST { namespace SnnDL {

namespace {

inline bool anyMask_(const SpikeNocCodec::WireSpikeKeyV2& ws, uint32_t cells) {
    const uint32_t n = std::min<uint32_t>(cells, kMaxMulticastBlockCells);
    for (uint32_t i = 0; i < n; ++i) {
        if (ws.core_mask[i] != 0) return true;
    }
    return false;
}

inline uint32_t idxOfLocal_(uint32_t local_x, uint32_t local_y, uint32_t block_w) {
    return local_y * block_w + local_x;
}

} // namespace

MulticastRouter::MulticastRouter(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id),
      out_("SnnDL.MulticastRouter", params.find<int>("verbose", 0), 0, SST::Output::STDOUT),
      timebase_(getTimeConverter("1ns")) {

    node_id_ = params.find<uint32_t>("node_id", 0);
    router_latency_cycles_ = params.find<uint64_t>("router_latency_cycles", 0);
    serialize_output_enable_ = params.find<int>("serialize_output_enable", 0) != 0;
    serialize_service_cycles_ = params.find<uint64_t>("serialize_service_cycles", 1);
    if (serialize_service_cycles_ == 0) serialize_service_cycles_ = 1;

    {
        const std::string inter = params.find<std::string>("multicast_inter_policy", "xy");
        if (!parseInterPolicy_(inter, multicast_inter_policy_)) {
            out_.fatal(CALL_INFO, -1, "Invalid multicast_inter_policy='%s' (supported: xy,yx)\n", inter.c_str());
        }
        const std::string intra = params.find<std::string>("multicast_intra_policy", "manhattan_x_first");
        if (!parseIntraPolicy_(intra, multicast_intra_policy_)) {
            out_.fatal(
                CALL_INFO, -1,
                "Invalid multicast_intra_policy='%s' (supported: manhattan_x_first,manhattan_y_first)\n",
                intra.c_str());
        }
    }

    const std::string mesh_shape = params.find<std::string>("mesh_shape", "4x4");
    if (!parseMeshShape_(mesh_shape, mesh_w_, mesh_h_) || mesh_w_ == 0 || mesh_h_ == 0) {
        out_.fatal(CALL_INFO, -1, "Invalid mesh_shape='%s'\n", mesh_shape.c_str());
    }

    local_link_ = configureLink("local", new SST::Event::Handler2<MulticastRouter, &MulticastRouter::handleLocalLinkEvent>(this));
    north_link_ = configureLink("north", new SST::Event::Handler2<MulticastRouter, &MulticastRouter::handleNorthLinkEvent>(this));
    south_link_ = configureLink("south", new SST::Event::Handler2<MulticastRouter, &MulticastRouter::handleSouthLinkEvent>(this));
    east_link_ = configureLink("east", new SST::Event::Handler2<MulticastRouter, &MulticastRouter::handleEastLinkEvent>(this));
    west_link_ = configureLink("west", new SST::Event::Handler2<MulticastRouter, &MulticastRouter::handleWestLinkEvent>(this));
}

MulticastRouter::~MulticastRouter() = default;

void MulticastRouter::init(unsigned int /*phase*/) {}
void MulticastRouter::setup() {}
void MulticastRouter::finish() {
    out_.output(
        "[mcast-router] node=%u in=%" PRIu64 " out=%" PRIu64 " local=%" PRIu64 " fwd_xy=%" PRIu64
        " spikekey_in=%" PRIu64 " stage_inter=%" PRIu64 " stage_intra=%" PRIu64 " clones=%" PRIu64
        "\n",
        node_id_,
        pkts_in_total_,
        pkts_out_total_,
        pkts_to_local_,
        pkts_forward_xy_,
        spikekey_in_total_,
        spikekey_stage_inter_,
        spikekey_stage_intra_,
        spikekey_clones_total_);
}

void MulticastRouter::handleLocalLinkEvent(SST::Event* event) {
    auto* pkt = dynamic_cast<NocPacketEvent*>(event);
    if (!pkt) { delete event; return; }
    handlePacket_(pkt);
}

void MulticastRouter::handleNorthLinkEvent(SST::Event* event) {
    auto* pkt = dynamic_cast<NocPacketEvent*>(event);
    if (!pkt) { delete event; return; }
    handlePacket_(pkt);
}

void MulticastRouter::handleSouthLinkEvent(SST::Event* event) {
    auto* pkt = dynamic_cast<NocPacketEvent*>(event);
    if (!pkt) { delete event; return; }
    handlePacket_(pkt);
}

void MulticastRouter::handleEastLinkEvent(SST::Event* event) {
    auto* pkt = dynamic_cast<NocPacketEvent*>(event);
    if (!pkt) { delete event; return; }
    handlePacket_(pkt);
}

void MulticastRouter::handleWestLinkEvent(SST::Event* event) {
    auto* pkt = dynamic_cast<NocPacketEvent*>(event);
    if (!pkt) { delete event; return; }
    handlePacket_(pkt);
}

void MulticastRouter::handlePacket_(NocPacketEvent* pkt) {
    if (!pkt) return;
    pkt->hop_count += 1;
    pkts_in_total_ += 1;
    if (pkt->packetKind() == NocPacketKind::SpikeKey) spikekey_in_total_ += 1;

    if (pkt->packetKind() == NocPacketKind::SpikeKey) {
        routeSpikeKey_(pkt);
        return;
    }

    routeUnicastXY_(pkt);
}

void MulticastRouter::routeUnicastXY_(NocPacketEvent* pkt) {
    if (!pkt) return;
    if (pkt->dst_node == node_id_) {
        sendToLocal_(pkt);
        return;
    }

    if (mesh_w_ == 0) { delete pkt; return; }
    const uint32_t x = node_id_ % mesh_w_;
    const uint32_t y = node_id_ / mesh_w_;

    const uint32_t dest_x = pkt->dst_node % mesh_w_;
    const uint32_t dest_y = pkt->dst_node / mesh_w_;

    if (x < dest_x) { sendToEast_(pkt); return; }
    if (x > dest_x) { sendToWest_(pkt); return; }
    if (y < dest_y) { sendToSouth_(pkt); return; }
    if (y > dest_y) { sendToNorth_(pkt); return; }

    sendToLocal_(pkt);
}

void MulticastRouter::routeUnicastYX_(NocPacketEvent* pkt) {
    if (!pkt) return;
    if (pkt->dst_node == node_id_) {
        sendToLocal_(pkt);
        return;
    }

    if (mesh_w_ == 0) { delete pkt; return; }
    const uint32_t x = node_id_ % mesh_w_;
    const uint32_t y = node_id_ / mesh_w_;

    const uint32_t dest_x = pkt->dst_node % mesh_w_;
    const uint32_t dest_y = pkt->dst_node / mesh_w_;

    if (y < dest_y) { sendToSouth_(pkt); return; }
    if (y > dest_y) { sendToNorth_(pkt); return; }
    if (x < dest_x) { sendToEast_(pkt); return; }
    if (x > dest_x) { sendToWest_(pkt); return; }

    sendToLocal_(pkt);
}

void MulticastRouter::routeSpikeKey_(NocPacketEvent* pkt) {
    if (!pkt) return;

    SpikeNocCodec::WireSpikeKeyV2 ws{};
    if (!SpikeNocCodec::decodeSpikeKeyAny(pkt->payload, ws)) {
        delete pkt;
        return;
    }
    if (ws.version != 1 && ws.version != 2) { delete pkt; return; }

    constexpr uint16_t kStageInter = 0;
    constexpr uint16_t kStageIntra = 1;

    const uint32_t block_w = static_cast<uint32_t>((ws.block_w_h >> 8) & 0xffu);
    const uint32_t block_h = static_cast<uint32_t>((ws.block_w_h) & 0xffu);
    if (block_w == 0 || block_h == 0) { delete pkt; return; }
    const uint32_t block_cells = block_w * block_h;
    if (block_cells == 0 || block_cells > kMaxMulticastBlockCells) { delete pkt; return; }
    if (!anyMask_(ws, block_cells)) { delete pkt; return; }

    // INTER: unicast XY towards ingress (dst_node should already be ingress_node).
    if (ws.stage == kStageInter) {
        spikekey_stage_inter_ += 1;
        if (node_id_ == ws.ingress_node) {
            ws.stage = kStageIntra;
            SpikeNocCodec::encodeSpikeKey(ws, pkt->payload);
            // fallthrough to INTRA routing on ingress
        } else {
            pkt->dst_node = ws.ingress_node;
            if (multicast_inter_policy_ == InterBlockRoutePolicy::HashXY) {
                const uint64_t h = ws.group_id ^ static_cast<uint64_t>(ws.pre_global) ^ static_cast<uint64_t>(ws.block_id);
                if (h & 1ull) routeUnicastYX_(pkt);
                else routeUnicastXY_(pkt);
            } else if (multicast_inter_policy_ == InterBlockRoutePolicy::YX) {
                routeUnicastYX_(pkt);
            } else {
                routeUnicastXY_(pkt);
            }
            return;
        }
    }

    if (ws.stage != kStageIntra) {
        delete pkt;
        return;
    }
    spikekey_stage_intra_ += 1;

    // INTRA: blocked multicast tree (arbitrary block_w x block_h, rooted at ingress_node).
    if (mesh_w_ == 0) { delete pkt; return; }
    const uint32_t self_x = node_id_ % mesh_w_;
    const uint32_t self_y = node_id_ / mesh_w_;

    // block origin derived from block_id (core_mask indices are always in block-top-left order)
    const uint32_t blocks_w = mesh_w_ / block_w;
    const uint32_t blocks_h = mesh_h_ / block_h;
    if (blocks_w == 0 || blocks_h == 0) { delete pkt; return; }
    if (ws.block_id >= blocks_w * blocks_h) { delete pkt; return; }
    const uint32_t block_x0 = (ws.block_id % blocks_w) * block_w;
    const uint32_t block_y0 = (ws.block_id / blocks_w) * block_h;

    if (self_x < block_x0 || self_y < block_y0) { delete pkt; return; }
    const uint32_t local_x = self_x - block_x0;
    const uint32_t local_y = self_y - block_y0;
    if (local_x >= block_w || local_y >= block_h) { delete pkt; return; }

    const uint32_t idx_self = idxOfLocal_(local_x, local_y, block_w); // block-relative
    if (idx_self >= block_cells || idx_self >= kMaxMulticastBlockCells) { delete pkt; return; }

    const uint32_t ingress_x = ws.ingress_node % mesh_w_;
    const uint32_t ingress_y = ws.ingress_node / mesh_w_;
    if (ingress_x < block_x0 || ingress_y < block_y0) { delete pkt; return; }
    if (ingress_x >= block_x0 + block_w || ingress_y >= block_y0 + block_h) { delete pkt; return; }
    const uint32_t ingress_lx = ingress_x - block_x0;
    const uint32_t ingress_ly = ingress_y - block_y0;
    if (ingress_lx >= block_w || ingress_ly >= block_h) { delete pkt; return; }

    auto deliverLocalMask = [&](uint32_t core_mask) {
        if (!core_mask) return;
        for (uint32_t bit = 0; bit < 32; ++bit) {
            if ((core_mask & (1u << bit)) == 0) continue;
            auto* c = static_cast<NocPacketEvent*>(pkt->clone());
            spikekey_clones_total_ += 1;
            c->dst_node = node_id_;
            c->dst_endpoint = static_cast<uint16_t>(bit);
            sendToLocal_(c);
        }
    };

    const bool x_first = (multicast_intra_policy_ == IntraBlockTreePolicy::ManhattanXFirst);
    auto parentOf = [&](uint32_t cx, uint32_t cy, uint32_t& out_px, uint32_t& out_py) {
        out_px = cx;
        out_py = cy;
        if (cx == ingress_lx && cy == ingress_ly) return; // root has no parent
        if (x_first) {
            if (cx != ingress_lx) {
                out_px = (ingress_lx > cx) ? (cx + 1u) : (cx - 1u);
                return;
            }
            if (cy != ingress_ly) {
                out_py = (ingress_ly > cy) ? (cy + 1u) : (cy - 1u);
                return;
            }
        } else {
            if (cy != ingress_ly) {
                out_py = (ingress_ly > cy) ? (cy + 1u) : (cy - 1u);
                return;
            }
            if (cx != ingress_lx) {
                out_px = (ingress_lx > cx) ? (cx + 1u) : (cx - 1u);
                return;
            }
        }
    };

    auto isChild = [&](uint32_t child_x, uint32_t child_y) -> bool {
        if (child_x >= block_w || child_y >= block_h) return false;
        uint32_t px = 0, py = 0;
        parentOf(child_x, child_y, px, py);
        return (px == local_x) && (py == local_y);
    };

    auto subtreeHasAny = [&](uint32_t child_x, uint32_t child_y) -> bool {
        // quick reject for invalid child
        if (!isChild(child_x, child_y)) return false;
        for (uint32_t idx = 0; idx < block_cells && idx < kMaxMulticastBlockCells; ++idx) {
            if (ws.core_mask[idx] == 0) continue;
            const uint32_t cx0 = idx % block_w;
            const uint32_t cy0 = idx / block_w;
            if (cx0 == local_x && cy0 == local_y) continue;

            uint32_t cx = cx0;
            uint32_t cy = cy0;
            while (!(cx == ingress_lx && cy == ingress_ly) &&
                   !(cx == local_x && cy == local_y) &&
                   !(cx == child_x && cy == child_y)) {
                uint32_t px = 0, py = 0;
                parentOf(cx, cy, px, py);
                if (px == cx && py == cy) break;
                cx = px;
                cy = py;
            }
            if (cx == child_x && cy == child_y) return true;
        }
        return false;
    };

    // Forward to tree children if their subtree contains at least one recipient.
    if (local_x + 1u < block_w && subtreeHasAny(local_x + 1u, local_y)) {
        auto* c = static_cast<NocPacketEvent*>(pkt->clone());
        spikekey_clones_total_ += 1;
        sendToEast_(c);
    }
    if (local_x > 0u && subtreeHasAny(local_x - 1u, local_y)) {
        auto* c = static_cast<NocPacketEvent*>(pkt->clone());
        spikekey_clones_total_ += 1;
        sendToWest_(c);
    }
    if (local_y + 1u < block_h && subtreeHasAny(local_x, local_y + 1u)) {
        auto* c = static_cast<NocPacketEvent*>(pkt->clone());
        spikekey_clones_total_ += 1;
        sendToSouth_(c);
    }
    if (local_y > 0u && subtreeHasAny(local_x, local_y - 1u)) {
        auto* c = static_cast<NocPacketEvent*>(pkt->clone());
        spikekey_clones_total_ += 1;
        sendToNorth_(c);
    }

    deliverLocalMask(ws.core_mask[idx_self]);
    delete pkt;
}

void MulticastRouter::sendToLocal_(NocPacketEvent* pkt) {
    if (!pkt) return;
    if (!local_link_) { delete pkt; return; }
    pkts_out_total_ += 1;
    pkts_to_local_ += 1;
    sendOnPort_(local_link_, 0, pkt);
}

void MulticastRouter::sendToNorth_(NocPacketEvent* pkt) {
    if (!pkt) return;
    if (!north_link_) { delete pkt; return; }
    pkts_out_total_ += 1;
    pkts_forward_xy_ += 1;
    sendOnPort_(north_link_, 1, pkt);
}

void MulticastRouter::sendToSouth_(NocPacketEvent* pkt) {
    if (!pkt) return;
    if (!south_link_) { delete pkt; return; }
    pkts_out_total_ += 1;
    pkts_forward_xy_ += 1;
    sendOnPort_(south_link_, 2, pkt);
}

void MulticastRouter::sendToEast_(NocPacketEvent* pkt) {
    if (!pkt) return;
    if (!east_link_) { delete pkt; return; }
    pkts_out_total_ += 1;
    pkts_forward_xy_ += 1;
    sendOnPort_(east_link_, 3, pkt);
}

void MulticastRouter::sendToWest_(NocPacketEvent* pkt) {
    if (!pkt) return;
    if (!west_link_) { delete pkt; return; }
    pkts_out_total_ += 1;
    pkts_forward_xy_ += 1;
    sendOnPort_(west_link_, 4, pkt);
}

void MulticastRouter::sendOnPort_(SST::Link* link, uint32_t port_idx, NocPacketEvent* pkt) {
    if (!pkt) return;
    if (!link) { delete pkt; return; }
    if (!serialize_output_enable_) {
        link->send(router_latency_cycles_, timebase_, pkt);
        return;
    }
    if (port_idx >= port_next_free_cycle_.size()) {
        link->send(router_latency_cycles_, timebase_, pkt);
        return;
    }

    const uint64_t now = static_cast<uint64_t>(getCurrentSimTime(timebase_));
    const uint64_t next_free = port_next_free_cycle_[port_idx];
    const uint64_t start = (next_free > now) ? next_free : now;
    const uint64_t wait = start - now;
    const uint64_t delay = wait + router_latency_cycles_;
    link->send(delay, timebase_, pkt);
    port_next_free_cycle_[port_idx] = start + serialize_service_cycles_;
}

bool MulticastRouter::parseMeshShape_(const std::string& shape, uint32_t& out_w, uint32_t& out_h) {
    out_w = 0;
    out_h = 0;
    if (shape.empty()) return false;

    std::string s = shape;
    for (auto& ch : s) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));

    const auto pos = s.find('x');
    if (pos == std::string::npos) return false;
    const std::string a = s.substr(0, pos);
    const std::string b = s.substr(pos + 1);
    if (a.empty() || b.empty()) return false;

    char* endp = nullptr;
    const long w = std::strtol(a.c_str(), &endp, 10);
    if (!endp || *endp != '\0' || w <= 0) return false;
    endp = nullptr;
    const long h = std::strtol(b.c_str(), &endp, 10);
    if (!endp || *endp != '\0' || h <= 0) return false;

    out_w = static_cast<uint32_t>(w);
    out_h = static_cast<uint32_t>(h);
    return true;
}

bool MulticastRouter::parseInterPolicy_(const std::string& s, InterBlockRoutePolicy& out) {
    std::string v = s;
    for (auto& ch : v) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (v == "xy") { out = InterBlockRoutePolicy::XY; return true; }
    if (v == "yx") { out = InterBlockRoutePolicy::YX; return true; }
    if (v == "hash_xy" || v == "hashxy") { out = InterBlockRoutePolicy::HashXY; return true; }
    return false;
}

bool MulticastRouter::parseIntraPolicy_(const std::string& s, IntraBlockTreePolicy& out) {
    std::string v = s;
    for (auto& ch : v) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    if (v == "manhattan_x_first" || v == "x_first" || v == "xfirst") { out = IntraBlockTreePolicy::ManhattanXFirst; return true; }
    if (v == "manhattan_y_first" || v == "y_first" || v == "yfirst") { out = IntraBlockTreePolicy::ManhattanYFirst; return true; }
    return false;
}

}} // namespace SST::SnnDL
