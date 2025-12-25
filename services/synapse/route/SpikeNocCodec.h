// -*- c++ -*-
//
// SpikeNocCodec:
// - 将 SpikeEvent 编解码为 NoC 通用包（NocPacketEvent）的 payload bytes
// - 目的：让 services/noc 与 NIC 层完全不依赖 SpikeEvent（严格通用化）
//

#pragma once

#include <cstdint>
#include <cstring>
#include <type_traits>

#include "GlobalNeuronLayout.h"
#include "NocPacketEvent.h"
#include "SpikeEvent.h"

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

