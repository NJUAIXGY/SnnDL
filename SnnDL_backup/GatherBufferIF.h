// -*- c++ -*-
// GatherBufferIF.h: GAS-aware StandardMem front-end with scratchpad buffering

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <vector>
#include <cstdint>

#include <sst/core/sst_config.h>
#include <sst/core/output.h>
#include <sst/core/interfaces/stdMem.h>

#include "GasCustomCmd.h"

namespace SST { namespace SnnDL {

class GatherBufferIF : public SST::Interfaces::StandardMem {
public:
    SST_ELI_REGISTER_SUBCOMPONENT(
        GatherBufferIF,
        "SnnDL", "GatherBufferIF", SST_ELI_ELEMENT_VERSION(1,0,0),
        "GAS scratchpad + coalescing StandardMem front-end", SST::Interfaces::StandardMem)

    SST_ELI_DOCUMENT_PARAMS(
        {"verbose", "verbosity", "0"},
        {"merge_policy", "none|cacheline|row|auto", "auto"},
        {"sort_policy", "addr|row|bank_row", "row"},
        {"row_bytes_guess", "row bytes guess for ordering", "8192"},
        {"sram_bytes", "scratchpad capacity bytes", "262144"},
        {"sram_access_ns", "SRAM access latency (ns) for response emission", "0"},
        {"max_inflight_reads", "Gather phase max inflight to downstream", "128"},
        {"tail_wait_timeout_ns", "EndGather tail wait timeout (0=wait-all)", "0"},
        {"allow_apply_miss_read", "allow fetching in APPLY on miss", "0"},
        {"flush_after_scatter", "flush SRAM after scatter", "1"},
        {"strict_mode", "gate reads strictly to Gather phase (1=strict)", "1"},
        {"window_auto", "enable time-driven GAS windows", "0"},
        {"window_cycles_gather", "cycles of Gather window when window_auto=1", "0"},
        {"window_cycles_apply", "cycles to hold before emitting Apply responses", "0"},
        {"window_cycles_scatter", "cycles of Scatter window", "0"},
        {"clock", "subcomponent clock when window_auto=1", "1GHz"},
        {"port", "shared port name for downstream standardInterface when loaded anonymously"}
    )

    SST_ELI_DOCUMENT_PORTS(
        {"lowlink", "Port to memory hierarchy (shared with downstream standardInterface)", {}}
    )

    SST_ELI_DOCUMENT_STATISTICS(
        {"gas_unique_reads", "Number of unique coalesced reads issued downstream", "reads", 1},
        {"gas_unique_bytes", "Bytes covered by unique coalesced reads", "bytes", 1},
        {"gas_upstream_reads", "Number of upstream read requests received during Gather windows", "reads", 1},
        {"gas_evictions", "Number of SRAM evictions (granules)", "evictions", 1},
        {"gas_inflight_peak", "Peak inflight downstream reads during Gather", "reads", 1},
        {"gas_tail_wait_ns", "Tail wait duration between EndGather and Apply (ns)", "ns", 1},
        {"gas_stage_cycles_gather", "Cycles spent in Gather stage (window mode)", "cycles", 1},
        {"gas_stage_cycles_apply", "Cycles spent in Apply stage (window mode)", "cycles", 1},
        {"gas_stage_cycles_scatter", "Cycles spent in Scatter stage (window mode)", "cycles", 1},
        {"gas_req_coalesce_size_bytes", "Granule size distribution (bytes)", "bytes", 1},
        {"gas_buffer_occupancy_bytes", "SRAM buffer occupancy sampled on fills (bytes)", "bytes", 1}
    )

    // API ctor
    GatherBufferIF(ComponentId_t id, Params& params, TimeConverter* time, HandlerBase* handler);
    // Serialization-only default
    GatherBufferIF() : SST::Interfaces::StandardMem() {}
    ~GatherBufferIF() override;

    // StandardMem virtuals
    void sendUntimedData(Request* req) override;
    Request* recvUntimedData() override;
    void send(Request* req) override;
    Request* poll() override; // unused: push-based
    Addr getLineSize() override;
    void setMemoryMappedAddressRegion(Addr start, Addr size) override;

    // Phases
    void init(unsigned int phase) override;
    void setup() override;
    void finish() override;

private:
    // Internal types
    enum class Stage { Idle=0, Gather=1, Apply=2, Scatter=3 };
    enum class Merge { None=0, Cacheline=1, Row=2, Auto=3 };
    enum class Sort { Addr=0, Row=1, BankRow=2 };

    struct SubReq { Request::id_t up_id; uint64_t offset; uint32_t size; SST::Interfaces::StandardMem::Read* up_read; };
    struct Granule { uint64_t base; uint32_t size; std::vector<SubReq> subs; bool issued=false; bool ready=false; uint64_t down_id=0; };

    // Helpers
    Merge parseMerge(const std::string& s) const;
    Sort parseSort(const std::string& s) const;
    uint64_t alignDown(uint64_t addr, uint64_t bytes) const { return (addr/bytes)*bytes; }
    uint64_t granuleSize() const;
    uint64_t rowIndex(uint64_t addr) const { return row_bytes_guess_? (addr / row_bytes_guess_) : 0; }
    void issueGranule_(uint64_t key, Granule& g);
    void onDownstreamResp_(Request* r);
    void maybeEnterApply_();
    void emitApplyResponses_();
    void doFlush_();
    bool clockTick(Cycle_t currentCycle);

    // Backend
    SST::Interfaces::StandardMem* backend_ = nullptr; // downstream standardInterface
    TimeConverter* time_ = nullptr;
    HandlerBase* upstream_handler_ = nullptr; // to deliver responses to parent
    Output out_;

    // Config
    Merge merge_ = Merge::Auto;
    Sort sort_ = Sort::Row;
    uint32_t row_bytes_guess_ = 8192;
    uint64_t sram_bytes_ = 256*1024;
    uint32_t sram_access_ns_ = 0;
    uint32_t max_inflight_reads_ = 128;
    uint64_t tail_wait_timeout_ns_ = 0; // 0 = wait all
    bool allow_apply_miss_read_ = false;
    bool flush_after_scatter_ = true;
    bool strict_mode_ = true; // 非Gather阶段是否严格拦截读（默认开）
    bool window_auto_ = false;
    uint64_t win_cyc_gather_ = 0, win_cyc_apply_ = 0, win_cyc_scatter_ = 0;
    std::string clock_freq_ = "1GHz";

    // State
    Stage stage_ = Stage::Idle;
    uint64_t current_gather_id_ = 0;
    uint64_t bytes_in_sram_ = 0;
    std::unordered_map<uint64_t, Granule> granules_; // key->granule
    std::unordered_map<Request::id_t, SST::Interfaces::StandardMem::Read*> pending_up_reads_; // upstream read map
    std::unordered_map<Request::id_t, uint64_t> inflight_down_; // down_id -> granule key
    std::deque<uint64_t> lru_list_; // most-recent at back
    std::unordered_map<uint64_t, std::vector<uint8_t>> sram_blocks_; // key->data
    std::unordered_set<uint64_t> required_set_; // required granules for current gather
    bool end_gather_seen_ = false;
    uint64_t stage_counter_ = 0; // cycles elapsed in current stage
    bool apply_pending_emit_ = false;
    std::vector<SST::Interfaces::StandardMem::Read*> queued_non_gather_reads_; // 非Gather阶段缓存的上游读

    // Utilities for LRU
    void touchLRU_(uint64_t key);
    void ensureCapacity_(uint64_t need);
    
    // Stats
    Statistic<uint64_t>* stat_unique_reads_ = nullptr;
    Statistic<uint64_t>* stat_unique_bytes_ = nullptr;
    Statistic<uint64_t>* stat_up_reads_ = nullptr;
    Statistic<uint64_t>* stat_evictions_ = nullptr;
    Statistic<uint64_t>* stat_inflight_peak_ = nullptr;
    Statistic<uint64_t>* stat_tail_wait_ns_ = nullptr;
    Statistic<uint64_t>* stat_stage_cycles_g_ = nullptr;
    Statistic<uint64_t>* stat_stage_cycles_a_ = nullptr;
    Statistic<uint64_t>* stat_stage_cycles_s_ = nullptr;
    Statistic<uint64_t>* stat_coalesce_granule_size_ = nullptr; // bytes per granule
    Statistic<uint64_t>* stat_buffer_occupancy_bytes_ = nullptr; // sampled on store

    // Time tracking (ns) for non-window mode
    uint64_t last_stage_change_ns_ = 0;
    uint64_t tail_wait_start_ns_ = 0;
};

}} // namespace
