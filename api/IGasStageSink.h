// -*- c++ -*-
//
// IGasStageSink: GAS 控制面阶段事件/统计汇聚入口（StdMem glue → Control）
// - 目的：让 control/ 不解析 StandardMem::CustomResp，也不依赖 GasOpData/StandardMem 类型
//

#pragma once

#include <cstdint>

#include "GasOps.h"

namespace SST { namespace SnnDL {

struct GasStageEvent {
    GasOp op = GasOp::BeginGather;
    uint32_t superstep = 0;
    uint32_t slice = 0;
    uint32_t total_slices = 1;
    bool flag = false;
};

struct GasStatEvent {
    uint64_t unique_reads = 0;
    uint64_t unique_bytes = 0;
    uint64_t rowwin_triggers = 0;
    uint64_t rowwin_bytes = 0;
    uint64_t bursts = 0;
    uint64_t payload_bytes = 0;
    uint64_t window_inflight_peak = 0;
    uint64_t window_buffer_max_bytes = 0;
    uint64_t gap_absorbed_bytes = 0;
    // DRAM-aware Apply diagnostics (optional; only meaningful when enabled)
    uint64_t unique_line_count = 0;
    uint64_t covered_line_count = 0;
    uint64_t overfetch_bytes = 0;
};

class IGasStageSink {
public:
    virtual ~IGasStageSink() = default;

    virtual void onGasStageEvent(const GasStageEvent& ev) = 0;
    virtual void onGasStatEvent(const GasStatEvent& st) = 0;
};

}} // namespace SST::SnnDL
