// -*- c++ -*-
// GasCustomCmd.h: Custom control-plane commands for GAS phases

#pragma once

#include <sst/core/interfaces/stdMem.h>
#include <string>

#include "GasOps.h"

namespace SST { namespace SnnDL {

// CustomData payload carried inside StandardMem::CustomReq/Resp
struct GasOpData : public SST::Interfaces::StandardMem::CustomData {
    GasOp op = GasOp::BeginGather;
    uint32_t superstep = 0;
    uint32_t slice = 0;
    uint32_t total_slices = 1;
    bool flag = false; // generic flag (e.g., flush)

    GasOpData() = default;
    GasOpData(GasOp _op, uint32_t ss=0, uint32_t sl=0, uint32_t tot=1, bool fl=false)
        : op(_op), superstep(ss), slice(sl), total_slices(tot), flag(fl) {}

    // CustomData API
    SST::Interfaces::StandardMem::CustomData* makeResponse() override {
        // One-way by default
        return new GasOpData(GasOp::EndScatter, superstep, slice, total_slices, flag);
    }
    bool needsResponse() override { return false; }
    SST::Interfaces::StandardMem::Addr getRoutingAddress() override { return 0; }
    uint64_t getSize() override { return 0; }
    std::string getString() override {
        return std::string("GasOp:") + std::to_string((int)op) +
               ",ss=" + std::to_string(superstep) +
               ",sl=" + std::to_string(slice) +
               ",tot=" + std::to_string(total_slices) +
               ",fl=" + (flag?"1":"0");
    }
    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        SST_SER(op);
        SST_SER(superstep);
        SST_SER(slice);
        SST_SER(total_slices);
        SST_SER(flag);
    }
    ImplementSerializable(SST::SnnDL::GasOpData);
};

// Upstream statistic update payload from GatherBufferIF to PE
struct GasStatData : public SST::Interfaces::StandardMem::CustomData {
    // Downstream unique reads/bytes (granule returns)
    uint64_t unique_reads = 0;
    uint64_t unique_bytes = 0;
    // Row-window coarse-merge counters per segment (issued at Apply build time)
    uint64_t rowwin_triggers = 0;
    uint64_t rowwin_bytes = 0;
    // Segment/burst counters regardless of row-window (issued at Apply build time)
    uint64_t bursts = 0;            // number of granules (segments) built
    uint64_t payload_bytes = 0;     // sum of sub-read sizes within the segment (useful bytes)
    uint64_t window_inflight_peak = 0;     // per-window inflight peak (two-buffer sum)
    uint64_t window_buffer_max_bytes = 0;  // per-window SRAM occupancy peak
    uint64_t gap_absorbed_bytes = 0;       // fine merge: absorbed gap bytes (window-level or segment-level additive)
    // DRAM-aware Apply diagnostics (optional; emitted only when enabled)
    uint64_t unique_line_count = 0;        // approx unique cachelines touched by staged reads (per window)
    uint64_t covered_line_count = 0;       // approx covered cachelines by issued segments (per window)
    uint64_t overfetch_bytes = 0;          // issued_bytes - payload_bytes (per window)

    GasStatData() = default;
    GasStatData(uint64_t r, uint64_t b,
                uint64_t rwt=0, uint64_t rwb=0,
                uint64_t bursts_=0, uint64_t payload_=0,
                uint64_t inflight_peak_=0, uint64_t buffer_peak_=0,
                uint64_t gap_abs_=0,
                uint64_t unique_lines_=0,
                uint64_t covered_lines_=0,
                uint64_t overfetch_bytes_=0)
        : unique_reads(r), unique_bytes(b),
          rowwin_triggers(rwt), rowwin_bytes(rwb),
          bursts(bursts_), payload_bytes(payload_),
          window_inflight_peak(inflight_peak_),
          window_buffer_max_bytes(buffer_peak_),
          gap_absorbed_bytes(gap_abs_),
          unique_line_count(unique_lines_),
          covered_line_count(covered_lines_),
          overfetch_bytes(overfetch_bytes_) {}

    // CustomData API
    SST::Interfaces::StandardMem::CustomData* makeResponse() override { return new GasStatData(0, 0); }
    bool needsResponse() override { return false; }
    SST::Interfaces::StandardMem::Addr getRoutingAddress() override { return 0; }
    uint64_t getSize() override { return sizeof(GasStatData); }
    std::string getString() override {
        return std::string("GasStatData:reads=") + std::to_string(unique_reads) +
               ",bytes=" + std::to_string(unique_bytes) +
               ",rowwin_trig=" + std::to_string(rowwin_triggers) +
               ",rowwin_bytes=" + std::to_string(rowwin_bytes) +
               ",bursts=" + std::to_string(bursts) +
               ",payload=" + std::to_string(payload_bytes) +
               ",inflight_peak=" + std::to_string(window_inflight_peak) +
               ",buffer_peak=" + std::to_string(window_buffer_max_bytes) +
               ",gap_abs=" + std::to_string(gap_absorbed_bytes) +
               ",uniq_lines=" + std::to_string(unique_line_count) +
               ",cov_lines=" + std::to_string(covered_line_count) +
               ",overfetch=" + std::to_string(overfetch_bytes);
    }
    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        SST_SER(unique_reads);
        SST_SER(unique_bytes);
        SST_SER(rowwin_triggers);
        SST_SER(rowwin_bytes);
        SST_SER(bursts);
        SST_SER(payload_bytes);
        SST_SER(window_inflight_peak);
        SST_SER(window_buffer_max_bytes);
        SST_SER(gap_absorbed_bytes);
        SST_SER(unique_line_count);
        SST_SER(covered_line_count);
        SST_SER(overfetch_bytes);
    }
    ImplementSerializable(SST::SnnDL::GasStatData);
};

}} // namespace
