// -*- c++ -*-
//
// IPeAggregation: PE 级汇聚接口（统计/阶段事件）。
//
// 目的：
// - 让 control 层（SnnPESubComponent 等）仅依赖窄接口上报统计与阶段事件；
// - 消除对 components/MultiCorePE.h 的硬依赖，降低耦合与编译依赖面。
//

#pragma once

#include <cstdint>
#include <string>

namespace SST { namespace SnnDL {

class IPeAggregation {
public:
    virtual ~IPeAggregation() = default;

    // === 阶段事件（用于 PE 级落盘/聚合） ===
    virtual void notifyStageEvent(uint32_t seq,
                                  const std::string& event,
                                  uint64_t ts_ns,
                                  uint64_t spikes_emitted,
                                  int core_id) = 0;

    // === Batch-A: memory access aggregation ===
    virtual void accumulateMemReadLatency(uint64_t latency_cycles, bool is_weight) = 0;
    virtual void accumulateIssueStats(uint64_t req_size_bytes, uint64_t inflight) = 0;

    // === GAS window stats aggregation ===
    virtual void accumulateGasStatsExt(uint64_t unique_bytes, uint64_t unique_reads,
                                       uint64_t rowwin_triggers, uint64_t rowwin_bytes,
                                       uint64_t bursts, uint64_t payload_bytes,
                                       uint64_t window_inflight_peak,
                                       uint64_t window_buffer_max_bytes,
                                       uint64_t unique_line_count,
                                       uint64_t covered_line_count,
                                       uint64_t overfetch_bytes) = 0;

    virtual void accumulateActivityF(double f) = 0;
    virtual void accumulateApplyScatterStats(uint64_t acc_updates, uint64_t posts_touched,
                                             uint64_t spikes_emitted, uint64_t hwm_bytes,
                                             uint64_t spill_records, uint64_t spilled_bytes) = 0;

    virtual void accumulateUniqueNeuronFired(uint64_t cnt) = 0;
    virtual void accumulateWindowSpikes(uint32_t seq, uint64_t count) = 0;
};

}} // namespace SST::SnnDL
