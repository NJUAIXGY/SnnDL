// -*- c++ -*-
//
// TensorWorkload: 张量/矩阵加速器（Systolic）周期级 microbench workload
// - 完全非 SNN：不依赖 SpikeEvent / synapse/* / stimulus/*
// - 通过 IMemoryAccess(addr↔bytes) 建模 DMA/DRAM 访问，通过 INocTransport 可选建模跨节点搬运
//

#pragma once

#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "api/ICoreWorkload.h"

namespace SST { namespace SnnDL {

class IMemoryAccess;
class INocTransport;
class NocPacketEvent;

class TensorWorkload final : public ICoreWorkload {
public:
    struct Config {
        // Microbench shape (GEMM by default): C[MxN] = A[MxK] * B[KxN]
        uint32_t m = 256;
        uint32_t n = 256;
        uint32_t k = 256;
        uint32_t element_bytes = 2; // default: fp16

        // Systolic array shape (peak MACs/cycle ~= array_m * array_n)
        uint32_t array_m = 32;
        uint32_t array_n = 32;
        float compute_efficiency = 1.0f; // [0,1], applied to peak macs/cycle

        // Runtime scheduling
        bool overlap_enable = true; // overlap DMA-read with compute
        uint64_t start_cycle = 1;
        uint32_t iterations = 0; // 0 = run forever (until sim ends)

        // Memory behavior (addr↔bytes)
        bool mem_enable = true;
        uint64_t mem_region_bytes = 1ull << 20; // wrap-around region per core
        uint32_t mem_req_bytes = 64;
        uint32_t mem_max_outstanding = 32;

        // Dataflow + tiling
        std::string dataflow = "os"; // os/ws/is
        uint32_t tile_m = 0; // 0 = use full M
        uint32_t tile_n = 0; // 0 = use full N
        uint32_t tile_k = 0; // 0 = use full K
        // Execution semantics
        // - bulk: legacy iteration-level model (fast; weaker dependency/backpressure semantics)
        // - tile: tile-level model with stalls/backpressure (Level-2 NPU semantics)
        std::string exec_mode = "bulk"; // bulk/tile
        // Tile loop schedule (tile exec only):
        // - auto: choose a reuse-friendly schedule based on dataflow (is->mkn, ws->nkm, os->mnk)
        // - mnk/mkn/nkm: explicit loop order
        std::string tile_schedule = "auto";
        // Writeback policy (tile exec only)
        std::string writeback_policy = "at_end_of_k";

        // On-chip buffer model (bytes)
        uint64_t ub_bytes = 0;
        uint64_t acc_bytes = 0;

        // Optional analytical DMA bandwidth (bytes/cycle). 0 = disabled
        uint64_t dma_bandwidth_bytes_per_cycle = 0;
        bool double_buffer = false;

        std::string collective_type = "none";
        uint64_t collective_bytes = 0;
        uint64_t collective_period_cycles = 0;
        std::string collective_pattern = "ring";
        uint32_t collective_packet_bytes = 256;
        // If true, treat collective as a barrier between iterations (Level-2).
        bool collective_blocking = false;
        // Collective scope (future extension; currently per_core is the only fully-defined behavior)
        std::string collective_scope = "per_core"; // per_core/per_pe/per_system

        // Optional NoC payload traffic (RawBytes)
        bool comm_enable = false;
        uint64_t comm_period_cycles = 0;
        uint32_t comm_payload_bytes = 0;

        bool strict = true; // if true, missing mem/noc fails fast
        uint64_t seed_base = 0;
    };

    TensorWorkload() = default;
    explicit TensorWorkload(const Config& cfg) : cfg_(cfg) {}

    void configureFromParams(const SST::Params& params) override;
    void bindRuntime(const Runtime& rt) override;

    bool onClockTick(uint64_t now_cycle) override;
    bool deliverPacket(NocPacketEvent* packet) override;
    void onGlobalStepStart(uint32_t seq) override;

    bool hasWork() const override;
    double getUtilization() const override;
    void getStatistics(std::map<std::string, uint64_t>& stats) const override;

private:
    enum class ReqKind : uint8_t { Read = 0, Write = 1 };

    enum class MemTag : uint8_t {
        Generic = 0,
        ReadA = 1,
        ReadB = 2,
        WriteC = 3,
    };

    struct InflightReq {
        ReqKind kind = ReqKind::Read;
        MemTag tag = MemTag::Generic;
        uint64_t epoch = 0;
        uint32_t bytes = 0;
    };

    struct TileSegState {
        bool valid = false;
        uint64_t epoch = 0;
        uint64_t seg_index = 0;
        uint32_t mi = 0;
        uint32_t ni = 0;
        uint32_t ki = 0;

        uint64_t need_a_bytes = 0;
        uint64_t need_b_bytes = 0;
        uint64_t issued_a_bytes = 0;
        uint64_t issued_b_bytes = 0;
        uint64_t done_a_bytes = 0;
        uint64_t done_b_bytes = 0;

        uint64_t rem_compute_cycles = 0;
    };

    struct WritebackState {
        uint64_t epoch = 0;
        uint32_t mi = 0;
        uint32_t ni = 0;
        uint64_t total_bytes = 0;
        uint64_t issued_bytes = 0;
        uint64_t done_bytes = 0;
    };

    static uint64_t ceilDivU64_(uint64_t a, uint64_t b);
    static uint64_t clampNonZero_(uint64_t v, uint64_t fallback);
    static uint64_t splitmix64_next_(uint64_t& x);
    static void fillBytesDeterministic_(uint64_t seed, uint64_t addr, uint32_t seq, std::vector<uint8_t>& out);

    inline uint64_t peakMacsPerCycle_() const {
        return static_cast<uint64_t>(cfg_.array_m ? cfg_.array_m : 1u) *
               static_cast<uint64_t>(cfg_.array_n ? cfg_.array_n : 1u);
    }

    inline uint64_t effectivePeakMacsPerCycle_() const {
        const uint64_t peak = peakMacsPerCycle_();
        const float eff = (cfg_.compute_efficiency < 0.0f) ? 0.0f : (cfg_.compute_efficiency > 1.0f ? 1.0f : cfg_.compute_efficiency);
        const uint64_t scaled = static_cast<uint64_t>(static_cast<double>(peak) * static_cast<double>(eff));
        return scaled ? scaled : 1ull;
    }

    void startIteration_();
    void resetTileIteration_();
    bool generateNextTileSeg_(TileSegState& out);
    bool advanceTileIndices_(uint32_t& mi, uint32_t& ni, uint32_t& ki) const;
    uint64_t tileSegComputeCycles_(uint64_t seg_index) const;
    uint64_t tileNeedReadABytes_(uint32_t mi, uint32_t ni, uint32_t ki) const;
    uint64_t tileNeedReadBBytes_(uint32_t mi, uint32_t ni, uint32_t ki) const;
    void scheduleWriteback_(uint32_t mi, uint32_t ni);
    void retireCompletedWritebacks_();

    uint32_t issueMemReadTagged_(uint32_t max_bytes, MemTag tag, uint64_t epoch);
    uint32_t issueMemWriteTagged_(uint32_t max_bytes, MemTag tag, uint64_t epoch);
    void onMemComplete_(ReqKind kind, MemTag tag, uint64_t epoch, uint32_t bytes);
    uint32_t issueMemRead_();
    uint32_t issueMemWrite_();
    bool tickCompute_();
    bool emitCollectiveTraffic_(uint64_t now_cycle);
    void serviceCollectiveBarrier_(uint64_t now_cycle);
    void onCollectiveControlPacket_(const std::vector<uint8_t>& payload, uint64_t now_cycle);
    void maybeNotifyCollectiveDone_(uint64_t now_cycle);
    void maybeEmitCollectiveRelease_(uint64_t now_cycle);
    void onCollectiveRelease_(uint32_t seq, uint32_t next_seq);
    uint64_t collectiveRecvBytesForSeq_(uint32_t seq) const;
    bool onClockTickTile_(uint64_t now_cycle);

    inline bool commReady_() const { return cfg_.comm_enable && rt_.noc && cfg_.comm_period_cycles > 0 && cfg_.comm_payload_bytes > 0; }
    inline bool collectiveReady_() const {
        return (cfg_.collective_type != "none" && rt_.noc && cfg_.collective_bytes > 0 && cfg_.collective_period_cycles > 0);
    }
    inline bool tileModelEnabled_() const {
        return (cfg_.tile_m > 0 || cfg_.tile_n > 0 || cfg_.tile_k > 0 || cfg_.ub_bytes > 0 || cfg_.acc_bytes > 0 || cfg_.dma_bandwidth_bytes_per_cycle > 0);
    }

    Config cfg_{};
    Runtime rt_{};

    bool configured_ = false;
    bool started_ = false;
    bool step_gated_ = false; // enabled when onGlobalStepStart() is observed
    bool step_open_ = false;  // when step_gated_: allow running one iteration in current step
    uint32_t step_seq_ = 0;

    // Derived per-iteration workload
    uint64_t bytes_read_per_iter_ = 0;
    uint64_t bytes_write_per_iter_ = 0;
    uint64_t mac_ops_per_iter_ = 0;
    uint64_t compute_cycles_per_iter_ = 0;
    uint64_t dram_bytes_per_iter_ = 0;
    uint64_t onchip_bytes_per_iter_ = 0;
    uint64_t dma_cycles_per_iter_ = 0;
    uint64_t tile_count_per_iter_ = 0;

    // Iteration state
    uint32_t iter_done_ = 0;
    uint32_t iter_seq_ = 1;
    bool iter_active_ = false;
    uint64_t rem_read_bytes_ = 0;
    uint64_t rem_write_bytes_ = 0;
    uint64_t rem_compute_cycles_ = 0;
    uint64_t rem_macs_ = 0;
    bool compute_started_ = false;

    // Tile-mode derived params (exec_mode=tile)
    std::string tile_schedule_eff_ = "mnk";
    bool tile_keep_a_ = false;
    bool tile_keep_b_ = false;
    uint32_t tile_tm_ = 1;
    uint32_t tile_tn_ = 1;
    uint32_t tile_tk_ = 1;
    uint32_t tile_mt_ = 1;
    uint32_t tile_nt_ = 1;
    uint32_t tile_kt_ = 1;
    uint64_t tile_a_bytes_ = 0;
    uint64_t tile_b_bytes_ = 0;
    uint64_t tile_c_bytes_ = 0;
    uint64_t tile_seg_cycles_base_ = 0;
    uint64_t tile_seg_cycles_remainder_ = 0;

    // Tile-mode iteration state
    bool tile_mode_active_ = false;
    uint64_t tile_seg_gen_index_ = 0;
    uint32_t tile_gen_mi_ = 0;
    uint32_t tile_gen_ni_ = 0;
    uint32_t tile_gen_ki_ = 0;
    bool tile_gen_done_ = false;
    uint64_t tile_seg_done_ = 0;
    TileSegState tile_cur_{};
    TileSegState tile_next_{};

    // Tile-mode writeback queue
    std::deque<uint64_t> tile_writeback_queue_{};
    std::unordered_map<uint64_t, WritebackState> tile_writebacks_{};

    uint64_t read_off_ = 0;
    uint64_t write_off_ = 0;

    std::unordered_map<uint64_t, InflightReq> inflight_;

    // Activity & counters (per-core; MultiCorePE pulls via getStatistics())
    uint64_t total_cycles_ = 0;
    uint64_t active_cycles_ = 0;
    uint64_t memory_requests_ = 0;

    uint64_t tensor_mem_reads_issued_total_ = 0;
    uint64_t tensor_mem_writes_issued_total_ = 0;
    uint64_t tensor_mem_bytes_read_total_ = 0;
    uint64_t tensor_mem_bytes_write_total_ = 0;
    uint64_t tensor_compute_cycles_total_ = 0;
    uint64_t tensor_mac_ops_total_ = 0;
    uint64_t tensor_dma_stall_cycles_total_ = 0;
    uint64_t tensor_iter_cycles_total_ = 0;
    uint64_t tensor_stall_dma_budget_cycles_total_ = 0;
    uint64_t tensor_stall_mem_outstanding_cycles_total_ = 0;
    uint64_t tensor_stall_wait_read_cycles_total_ = 0;
    uint64_t tensor_stall_wait_write_cycles_total_ = 0;
    uint64_t tensor_stall_collective_cycles_total_ = 0;
    uint64_t tensor_dma_cycles_total_ = 0;
    uint64_t tensor_dram_bytes_total_ = 0;
    uint64_t tensor_onchip_bytes_total_ = 0;
    uint64_t tensor_tile_count_total_ = 0;
    uint64_t tensor_collective_bytes_sent_total_ = 0;
    uint64_t tensor_collective_bytes_recv_total_ = 0;
    uint64_t tensor_collective_pkts_sent_total_ = 0;
    uint64_t tensor_collective_pkts_recv_total_ = 0;
    uint64_t tensor_collective_cycles_total_ = 0;

    uint64_t comm_last_cycle_ = 0;
    uint64_t collective_last_cycle_ = 0;
    uint64_t collective_seq_ = 0;
    bool collective_epoch_active_ = false;
    uint32_t collective_epoch_seq_ = 0;
    uint64_t collective_epoch_expected_recv_bytes_ = 0;
    uint64_t collective_epoch_recv_bytes_ = 0;
    bool collective_done_notified_ = false;
    bool collective_release_pending_ = false; // leader/root: pending broadcast of release for current seq
    uint32_t collective_release_pending_seq_ = 0;
    uint32_t total_cores_cfg_ = 1; // per-PE cores (from control param total_cores)
    uint32_t collective_barrier_done_count_ = 0;
    std::vector<uint8_t> collective_barrier_done_bitmap_{};
    std::unordered_map<uint32_t, uint64_t> collective_recv_bytes_by_seq_{};
    uint64_t tensor_pkt_sent_total_ = 0;
    uint64_t tensor_pkt_recv_total_ = 0;
    uint64_t tensor_pkt_bytes_sent_total_ = 0;
    uint64_t tensor_pkt_bytes_recv_total_ = 0;
};

}} // namespace SST::SnnDL
