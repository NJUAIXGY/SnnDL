// -*- c++ -*-
//
// SpikeTileBatchEmitter:
// - Experimental sender-side aggregation for SpikeTileKey.
// - Aggregates fired pre neurons by (target block_id, ingress_node, block_col)
//   and emits one packet with block_col + pre_mask.
//

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <unordered_map>
#include <vector>

#include "INocTransport.h"
#include "NocPacketEvent.h"
#include "SpikeInterBundleCodec.h"
#include "SynapseRouteSubsystem.h"
#include "SpikeTileNocCodec.h"

namespace SST { namespace SnnDL {

struct SpikeTileBatchEmitConfig final {
    uint32_t source_node = 0;
    uint16_t source_core = 0;
    uint32_t block_w = 0;
    uint32_t block_h = 0;
    uint32_t block_cols = 0;
    uint32_t max_pre_bits = 64;
    uint64_t* emit_seq = nullptr; // required for group-id uniqueness
    bool experimental_compact_mask_enable = false;
    bool experimental_inter_bundle_enable = false;
    uint32_t experimental_inter_bundle_max_entries = 64;
    bool experimental_inter_bundle_v2_enable = false;
};

namespace detail {

struct TileAggKey final {
    uint32_t block_id = 0;
    uint32_t ingress_node = 0;
    uint32_t block_col = 0;

    bool operator==(const TileAggKey& rhs) const {
        return block_id == rhs.block_id &&
               ingress_node == rhs.ingress_node &&
               block_col == rhs.block_col;
    }
};

struct TileAggKeyHash final {
    std::size_t operator()(const TileAggKey& key) const {
        const uint64_t a = (static_cast<uint64_t>(key.block_id) << 32) | static_cast<uint64_t>(key.ingress_node);
        const uint64_t b = static_cast<uint64_t>(key.block_col);
        return static_cast<std::size_t>((a * 11400714819323198485ull) ^ (b * 14029467366897019727ull));
    }
};

struct TileAggValue final {
    std::array<uint32_t, kMaxMulticastBlockCells> core_mask{};
    uint64_t pre_mask = 0;
};

} // namespace detail

inline bool emitSpikeTileBatchExperimental(const std::vector<uint32_t>& neuron_indices,
                                           uint64_t global_neuron_base,
                                           uint64_t now_cycle,
                                           const SynapseRouteSubsystem& route,
                                           INocTransport& noc,
                                           const SpikeTileBatchEmitConfig& cfg,
                                           uint64_t* emitted_packets = nullptr) {
    if (neuron_indices.empty()) return false;
    if (cfg.block_cols == 0 || cfg.block_cols > 64) return false;
    if (cfg.max_pre_bits == 0 || cfg.max_pre_bits > 64) return false;
    if (cfg.block_cols > cfg.max_pre_bits) return false;
    if (cfg.experimental_inter_bundle_max_entries == 0) return false;
    if (cfg.block_w == 0 || cfg.block_h == 0) return false;
    if (!cfg.emit_seq) return false;
    if (!route.multicastEnabled()) return false;
    const uint16_t block_w_h = static_cast<uint16_t>(((cfg.block_w & 0xffu) << 8) | (cfg.block_h & 0xffu));
    if (cfg.experimental_inter_bundle_enable &&
        cfg.experimental_inter_bundle_v2_enable &&
        SpikeInterBundleCodec::blockCellsFromBlockWH(block_w_h) == 0) {
        return false;
    }

    std::unordered_map<detail::TileAggKey, detail::TileAggValue, detail::TileAggKeyHash> agg;
    agg.reserve(neuron_indices.size());

    for (uint32_t neuron_idx : neuron_indices) {
        const uint32_t pre_global = static_cast<uint32_t>(global_neuron_base + static_cast<uint64_t>(neuron_idx));
        const uint32_t bit = pre_global % cfg.block_cols;
        if (bit >= cfg.max_pre_bits) continue;
        const uint32_t block_col = pre_global / cfg.block_cols;
        const uint64_t bit_mask = (1ull << bit);

        std::vector<SynapseRouteSubsystem::BlockTarget> targets;
        bool applied_gating = false;
        if (!route.computeMulticastTargets(pre_global, neuron_idx, now_cycle, targets, applied_gating) || targets.empty()) continue;
        (void)applied_gating;

        for (const auto& target : targets) {
            detail::TileAggKey key{};
            key.block_id = target.block_id;
            key.ingress_node = target.ingress_node;
            key.block_col = block_col;

            auto it = agg.find(key);
            if (it == agg.end()) {
                detail::TileAggValue init{};
                init.pre_mask = bit_mask;
                init.core_mask = target.core_mask;
                agg.emplace(key, std::move(init));
            } else {
                it->second.pre_mask |= bit_mask;
                for (uint32_t i = 0; i < kMaxMulticastBlockCells; ++i) {
                    it->second.core_mask[i] |= target.core_mask[i];
                }
            }
        }
    }

    if (agg.empty()) return false;

    uint64_t local_emitted_packets = 0;
    std::vector<std::vector<uint8_t>> bundle_entries_v1;
    std::vector<SpikeInterBundleCodec::BundleEntryV2> bundle_entries_v2;
    if (cfg.experimental_inter_bundle_enable) {
        if (cfg.experimental_inter_bundle_v2_enable) {
            bundle_entries_v2.reserve(agg.size());
        } else {
            bundle_entries_v1.reserve(agg.size());
        }
    }
    for (const auto& kv : agg) {
        const auto& key = kv.first;
        const auto& val = kv.second;
        if (val.pre_mask == 0) continue;
        const uint64_t seq = ++(*cfg.emit_seq);
        const uint64_t group_id = (seq << 32) | static_cast<uint64_t>(key.block_col);
        const uint32_t pre_global = static_cast<uint32_t>(
            static_cast<uint64_t>(key.block_col) * static_cast<uint64_t>(cfg.block_cols));

        if (cfg.experimental_inter_bundle_enable && cfg.experimental_inter_bundle_v2_enable) {
            SpikeInterBundleCodec::BundleEntryV2 entry{};
            entry.meta.block_id = key.block_id;
            entry.meta.ingress_node = key.ingress_node;
            entry.meta.pre_global = pre_global;
            entry.meta.group_id = group_id;
            entry.meta.tile_version =
                cfg.experimental_compact_mask_enable
                    ? static_cast<uint16_t>(SpikeTileNocCodec::kTileVersionV2)
                    : static_cast<uint16_t>(SpikeTileNocCodec::kTileVersionV1);
            entry.meta.tile_block_col = key.block_col;
            entry.meta.tile_pre_mask = val.pre_mask;
            entry.core_mask = val.core_mask;
            bundle_entries_v2.emplace_back(std::move(entry));
            continue;
        }

        SpikeTileNocCodec::WireSpikeTileKeyV1 ws{};
        ws.route.version = 2;
        ws.route.route_mode = SpikeTileNocCodec::kRouteModeSpikeTile;
        ws.route.stage = 0; // INTER
        ws.route.block_w_h = block_w_h;
        ws.route.block_id = key.block_id;
        ws.route.ingress_node = key.ingress_node;
        ws.route.pre_global = pre_global;
        ws.route.group_id = group_id;
        ws.route.core_mask = val.core_mask;
        ws.tile.tile_version = SpikeTileNocCodec::kTileVersionV1;
        ws.tile.block_col = key.block_col;
        ws.tile.pre_mask = val.pre_mask;

        std::vector<uint8_t> encoded_payload;
        bool encoded = false;
        if (cfg.experimental_compact_mask_enable) {
            encoded = SpikeTileNocCodec::encodeCompactV2(
                ws.route,
                ws.tile.block_col,
                ws.tile.pre_mask,
                encoded_payload);
        }
        if (!encoded) {
            SpikeTileNocCodec::encode(ws, encoded_payload);
        }

        if (cfg.experimental_inter_bundle_enable) {
            bundle_entries_v1.emplace_back(std::move(encoded_payload));
            continue;
        }

        auto* pkt = new NocPacketEvent();
        pkt->src_node = cfg.source_node;
        pkt->dst_node = key.ingress_node;
        pkt->src_endpoint = cfg.source_core;
        pkt->dst_endpoint = 0;
        pkt->kind = static_cast<uint16_t>(NocPacketKind::SpikeTileKey);
        pkt->timestamp = now_cycle;
        pkt->payload = std::move(encoded_payload);
        noc.sendFromCore(static_cast<int>(cfg.source_core), pkt);
        local_emitted_packets += 1;
    }

    if (cfg.experimental_inter_bundle_enable &&
        !cfg.experimental_inter_bundle_v2_enable &&
        !bundle_entries_v1.empty()) {
        const size_t chunk = static_cast<size_t>(cfg.experimental_inter_bundle_max_entries);
        for (size_t begin = 0; begin < bundle_entries_v1.size(); begin += chunk) {
            const size_t end = std::min(bundle_entries_v1.size(), begin + chunk);
            std::vector<std::vector<uint8_t>> part(bundle_entries_v1.begin() + begin, bundle_entries_v1.begin() + end);
            auto* pkt = new NocPacketEvent();
            pkt->src_node = cfg.source_node;
            pkt->dst_node = cfg.source_node;
            pkt->src_endpoint = cfg.source_core;
            pkt->dst_endpoint = 0;
            pkt->kind = static_cast<uint16_t>(NocPacketKind::SpikeTileKey);
            pkt->timestamp = now_cycle;
            const uint64_t bseq = ++(*cfg.emit_seq);
            const uint64_t bundle_id = (bseq << 32) | static_cast<uint64_t>(cfg.source_node);
            if (!SpikeInterBundleCodec::encode(
                    static_cast<uint16_t>(NocPacketKind::SpikeTileKey),
                    bundle_id,
                    part,
                    pkt->payload)) {
                delete pkt;
                continue;
            }
            noc.sendFromCore(static_cast<int>(cfg.source_core), pkt);
            local_emitted_packets += 1;
        }
    }
    if (cfg.experimental_inter_bundle_enable &&
        cfg.experimental_inter_bundle_v2_enable &&
        !bundle_entries_v2.empty()) {
        const size_t chunk = static_cast<size_t>(cfg.experimental_inter_bundle_max_entries);
        for (size_t begin = 0; begin < bundle_entries_v2.size(); begin += chunk) {
            const size_t end = std::min(bundle_entries_v2.size(), begin + chunk);
            std::vector<SpikeInterBundleCodec::BundleEntryV2> part(
                bundle_entries_v2.begin() + begin,
                bundle_entries_v2.begin() + end);
            auto* pkt = new NocPacketEvent();
            pkt->src_node = cfg.source_node;
            pkt->dst_node = cfg.source_node;
            pkt->src_endpoint = cfg.source_core;
            pkt->dst_endpoint = 0;
            pkt->kind = static_cast<uint16_t>(NocPacketKind::SpikeTileKey);
            pkt->timestamp = now_cycle;
            const uint64_t bseq = ++(*cfg.emit_seq);
            const uint64_t bundle_id = (bseq << 32) | static_cast<uint64_t>(cfg.source_node);
            if (!SpikeInterBundleCodec::encodeV2(
                    static_cast<uint16_t>(NocPacketKind::SpikeTileKey),
                    /*route_mode=*/SpikeTileNocCodec::kRouteModeSpikeTile,
                    /*stage=*/0,
                    block_w_h,
                    bundle_id,
                    part,
                    pkt->payload)) {
                delete pkt;
                continue;
            }
            noc.sendFromCore(static_cast<int>(cfg.source_core), pkt);
            local_emitted_packets += 1;
        }
    }

    if (emitted_packets) {
        *emitted_packets += local_emitted_packets;
    }

    return true;
}

}} // namespace SST::SnnDL
