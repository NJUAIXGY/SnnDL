// -*- c++ -*-
//
// SpikeNocCodec:
// - 将 SpikeEvent 编解码为 NoC 通用包（NocPacketEvent）的 payload bytes
// - 目的：让 services/noc 与 NIC 层完全不依赖 SpikeEvent（严格通用化）
//

#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "api/MulticastLimits.h"
#include "GlobalNeuronLayout.h"
#include "NocPacketEvent.h"
#include "events/SpikeEvent.h"

namespace SST { namespace SnnDL {

class SpikeNocCodec final {
public:
    struct WireSpike final {
        uint32_t src_neuron = 0;
        uint32_t dst_neuron = 0;
        double weight = 0.0;
        uint64_t timestamp = 0;
    };

    static_assert(std::is_trivially_copyable<WireSpike>::value, "WireSpike must be trivially copyable");

    struct WireSpikeKeyV1 final {
        uint16_t version = 1;
        uint16_t route_mode = 0;
        uint16_t stage = 0;
        uint16_t block_w_h = 0;

        uint32_t mesh_w = 0;
        uint32_t mesh_h = 0;
        uint32_t block_id = 0;
        uint32_t ingress_node = 0;

        uint32_t pre_global = 0;
        uint64_t group_id = 0;

        uint32_t core_mask[4] = {0, 0, 0, 0};
    };

    static_assert(std::is_trivially_copyable<WireSpikeKeyV1>::value, "WireSpikeKeyV1 must be trivially copyable");

    // V2: 支持可配置 block_w/block_h（由 block_w_h=(w<<8)|h 表示），并扩展 core_mask 为固定上限数组。
    // 兼容性：前缀字段保持与 V1 一致，便于 router/workload 在不引入可变长 payload 的情况下扩展。
    struct WireSpikeKeyV2 final {
        uint16_t version = 2;
        uint16_t route_mode = 0;
        uint16_t stage = 0;
        uint16_t block_w_h = 0;

        uint32_t mesh_w = 0;
        uint32_t mesh_h = 0;
        uint32_t block_id = 0;
        uint32_t ingress_node = 0;

        uint32_t pre_global = 0;
        uint64_t group_id = 0;

        std::array<uint32_t, kMaxMulticastBlockCells> core_mask{};
    };

    static_assert(std::is_trivially_copyable<WireSpikeKeyV2>::value, "WireSpikeKeyV2 must be trivially copyable");

    static void encodeSpikeKey(const WireSpikeKeyV1& ws, std::vector<uint8_t>& out_payload) {
        out_payload.resize(sizeof(WireSpikeKeyV1));
        std::memcpy(out_payload.data(), &ws, sizeof(WireSpikeKeyV1));
    }

    static void encodeSpikeKey(const WireSpikeKeyV2& ws, std::vector<uint8_t>& out_payload) {
        out_payload.resize(sizeof(WireSpikeKeyV2));
        std::memcpy(out_payload.data(), &ws, sizeof(WireSpikeKeyV2));
    }

    static bool decodeSpikeKey(const std::vector<uint8_t>& payload, WireSpikeKeyV1& out_ws) {
        if (payload.size() != sizeof(WireSpikeKeyV1)) return false;
        std::memcpy(&out_ws, payload.data(), sizeof(WireSpikeKeyV1));
        return true;
    }

    // Decode V1 or V2 into a V2 struct (V1 fields are expanded; unused mask slots are zeroed).
    static bool decodeSpikeKeyAny(const std::vector<uint8_t>& payload, WireSpikeKeyV2& out_ws) {
        if (payload.size() == sizeof(WireSpikeKeyV2)) {
            std::memcpy(&out_ws, payload.data(), sizeof(WireSpikeKeyV2));
            return true;
        }
        if (payload.size() == sizeof(WireSpikeKeyV1)) {
            WireSpikeKeyV1 v1{};
            std::memcpy(&v1, payload.data(), sizeof(WireSpikeKeyV1));
            out_ws = WireSpikeKeyV2{};
            out_ws.version = v1.version;
            out_ws.route_mode = v1.route_mode;
            out_ws.stage = v1.stage;
            out_ws.block_w_h = v1.block_w_h;
            out_ws.mesh_w = v1.mesh_w;
            out_ws.mesh_h = v1.mesh_h;
            out_ws.block_id = v1.block_id;
            out_ws.ingress_node = v1.ingress_node;
            out_ws.pre_global = v1.pre_global;
            out_ws.group_id = v1.group_id;
            for (size_t i = 0; i < 4; ++i) out_ws.core_mask[i] = v1.core_mask[i];
            return true;
        }
        return false;
    }

    static NocPacketEvent* encode(const SpikeEvent& spike, const GlobalNeuronLayout& layout) {
        const uint64_t src_global = static_cast<uint64_t>(spike.getSourceNeuron());
        const uint64_t dst_global = static_cast<uint64_t>(spike.getDestinationNeuron());

        auto* pkt = new NocPacketEvent();
        pkt->src_node = layout.nodeOf(src_global);
        pkt->dst_node = layout.nodeOf(dst_global);
        pkt->src_endpoint = static_cast<uint16_t>(layout.coreOf(src_global));
        pkt->dst_endpoint = static_cast<uint16_t>(layout.coreOf(dst_global));
        pkt->kind = static_cast<uint16_t>(NocPacketKind::Spike);
        pkt->hop_count = static_cast<uint16_t>(spike.getHopCount());
        pkt->timestamp = spike.getTimestamp();

        WireSpike ws;
        ws.src_neuron = spike.getSourceNeuron();
        ws.dst_neuron = spike.getDestinationNeuron();
        ws.weight = spike.getWeight();
        ws.timestamp = spike.getTimestamp();

        pkt->payload.resize(sizeof(WireSpike));
        std::memcpy(pkt->payload.data(), &ws, sizeof(WireSpike));
        return pkt;
    }

    static SpikeEvent* decode(const NocPacketEvent& pkt) {
        if (pkt.payload.size() != sizeof(WireSpike)) return nullptr;
        WireSpike ws;
        std::memcpy(&ws, pkt.payload.data(), sizeof(WireSpike));

        auto* spike = new SpikeEvent(ws.src_neuron, ws.dst_neuron, pkt.dst_node, ws.weight, ws.timestamp);
        spike->hop_count = pkt.hop_count;
        return spike;
    }
};

}} // namespace SST::SnnDL
