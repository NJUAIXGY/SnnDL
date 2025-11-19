// GatherBufferIF.cc: GAS-aware StandardMem front-end with scratchpad buffering

#include <sst/core/sst_config.h>
#include "GatherBufferIF.h"

#include <algorithm>
#include <cstring>
#include <cmath>
#include <fstream>
#include <cstdlib>

using namespace SST;
using namespace SST::Interfaces;
using namespace SST::SnnDL;

namespace {
inline bool snndlGlobalDebugEnabled() {
    static int cached = -1;
    if (cached == -1) {
        const char* env = std::getenv("SNNDL_DEBUG");
        if (!env || std::atoi(env) == 0) {
            env = std::getenv("SNNDL_DIAG_ENABLE");
        }
        cached = (env && std::atoi(env) != 0) ? 1 : 0;
    }
    return cached == 1;
}
}

GatherBufferIF::GatherBufferIF(ComponentId_t id, Params& params, TimeConverter* time, HandlerBase* handler)
    : StandardMem(id, params, time, handler), time_(time), upstream_handler_(handler)
{
    setDefaultTimeBase(*time);
    int verbose = params.find<int>("verbose", 0);
    out_.init("GatherBufferIF[@p:@l]: ", verbose, 0, Output::STDOUT);

    merge_ = parseMerge(params.find<std::string>("merge_policy", "auto"));
    sort_  = parseSort(params.find<std::string>("sort_policy", "row"));
    row_bytes_guess_ = params.find<uint32_t>("row_bytes_guess", 8192);
    sram_bytes_ = params.find<uint64_t>("sram_bytes", 256*1024);
    gap_merge_enable_ = params.find<int>("gap_merge_enable", 1) != 0;
    gap_k_bytes_ = params.find<uint64_t>("gap_merge_k_bytes", 0);
    burst_bytes_max_ = params.find<uint64_t>("burst_bytes_max", 64*1024);
    bank_bits_ = params.find<uint32_t>("bank_bits", 0);
    bank_shift_ = params.find<uint32_t>("bank_shift", 0);
    bank_auto_enable_ = params.find<int>("bank_auto_enable", 1) != 0;
    bank_auto_min_banks_ = params.find<uint32_t>("bank_auto_min_banks", 4);
    bank_auto_max_banks_ = params.find<uint32_t>("bank_auto_max_banks", 32);
    row_window_enable_ = params.find<int>("row_window_enable", 0) != 0;
    row_window_bytes_  = params.find<uint64_t>("row_window_bytes", 0);
    row_window_timeout_ns_ = params.find<uint64_t>("row_window_timeout_ns", 0);
    sram_access_ns_ = params.find<uint32_t>("sram_access_ns", 0);
    max_inflight_reads_ = params.find<uint32_t>("max_inflight_reads", 128);
    tail_wait_timeout_ns_ = params.find<uint64_t>("tail_wait_timeout_ns", 0);
    allow_apply_miss_read_ = params.find<int>("allow_apply_miss_read", 0) != 0;
    flush_after_scatter_ = params.find<int>("flush_after_scatter", 1) != 0;
    strict_mode_ = params.find<int>("strict_mode", 1) != 0;
    double_buffer_enable_ = params.find<int>("double_buffer_enable", 1) != 0;
    window_auto_ = params.find<int>("window_auto", 0) != 0;
    // Deprecate manual_window_drive: keep param for compatibility but force-disable
    (void)params.find<int>("manual_window_drive", 0);
    manual_window_drive_ = false;
    win_cyc_gather_ = params.find<uint64_t>("window_cycles_gather", 0);
    win_cyc_apply_  = params.find<uint64_t>("window_cycles_apply", 0);
    win_cyc_scatter_= params.find<uint64_t>("window_cycles_scatter", 0);
    apply_auto_end_enable_ = params.find<int>("apply_auto_end_enable", 1) != 0;
    scatter_immediate_complete_ = params.find<int>("scatter_immediate_complete", 0) != 0;
    clock_freq_ = params.find<std::string>("clock", "1GHz");
    emit_stage_events_ = params.find<int>("emit_stage_events", 0) != 0;
    emit_lenient_ = params.find<int>("emit_stage_events_lenient", 0) != 0;
    stage_cycles_csv_ = params.find<std::string>("stage_cycles_csv", "");
    stage_cycles_export_enable_ = !stage_cycles_csv_.empty();
    probe_csv_path_ = params.find<std::string>("probe_gas_csv", "");
    defer_issue_until_apply_ = params.find<int>("defer_issue_until_apply", 0) != 0;
    gather_auto_end_bytes_ = params.find<uint64_t>("gather_auto_end_bytes", 0);
    gather_auto_end_reads_ = params.find<uint64_t>("gather_auto_end_reads", 0);
    // Gating: if k or Lmax invalid, disable gap merge
    if (!gap_merge_enable_ || gap_k_bytes_ == 0 || burst_bytes_max_ == 0) {
        gap_merge_enable_ = false;
    }

    // Load downstream memHierarchy.standardInterface anonymously, sharing our 'lowlink' port
    Params bp; bp.insert("port", std::string("lowlink"));
    backend_ = loadAnonymousSubComponent<SST::Interfaces::StandardMem>(
        "memHierarchy.standardInterface", "lowlink", 0,
        ComponentInfo::SHARE_PORTS | ComponentInfo::INSERT_STATS,
        bp, time_, new StandardMem::Handler2<GatherBufferIF, &GatherBufferIF::onDownstreamResp_>(this));
    if (!backend_) {
        out_.fatal(CALL_INFO, -1, "Failed to load downstream memHierarchy.standardInterface\n");
    }
    // Stats registration
    stat_unique_reads_ = registerStatistic<uint64_t>("gas_unique_reads");
    stat_unique_bytes_ = registerStatistic<uint64_t>("gas_unique_bytes");
    stat_up_reads_     = registerStatistic<uint64_t>("gas_upstream_reads");
    stat_evictions_    = registerStatistic<uint64_t>("gas_evictions");
    stat_inflight_peak_= registerStatistic<uint64_t>("gas_inflight_peak");
    stat_tail_wait_ns_ = registerStatistic<uint64_t>("gas_tail_wait_ns");
    stat_stage_cycles_g_ = registerStatistic<uint64_t>("gas_stage_cycles_gather");
    stat_stage_cycles_a_ = registerStatistic<uint64_t>("gas_stage_cycles_apply");
    stat_stage_cycles_s_ = registerStatistic<uint64_t>("gas_stage_cycles_scatter");
    stat_coalesce_granule_size_ = registerStatistic<uint64_t>("gas_req_coalesce_size_bytes");
    stat_buffer_occupancy_bytes_ = registerStatistic<uint64_t>("gas_buffer_occupancy_bytes");
    stat_gap_absorbed_bytes_ = registerStatistic<uint64_t>("gas_gap_absorbed_bytes");
    stat_row_window_triggers_ = registerStatistic<uint64_t>("gas_row_window_triggers");
    stat_row_window_bytes_    = registerStatistic<uint64_t>("gas_row_window_bytes");
    // 门控诊断：下发到下游的唯一granule读次数
    stat_reads_issued_        = registerStatistic<uint64_t>("gas_reads_issued");
    // Adaptive k stats (optional)
    stat_k_dyn_bytes_         = registerStatistic<uint64_t>("gas_k_dyn_bytes");
    stat_oeff_ns_avg_         = registerStatistic<uint64_t>("gas_oeff_ns_avg");
    stat_bw_eff_bytes_per_us_ = registerStatistic<uint64_t>("gas_bw_eff_bytes_per_us");
    // Control stats
    stat_ctrl_probes_  = registerStatistic<uint64_t>("gas_ctrl_probes");
    stat_ctrl_adopts_  = registerStatistic<uint64_t>("gas_ctrl_adopts");
    stat_ctrl_reverts_ = registerStatistic<uint64_t>("gas_ctrl_reverts");
    inflight_down_.reserve(128);
    queued_non_gather_reads_.reserve(32);
    last_stage_change_ns_ = getCurrentSimTimeNano();
    // Adaptive k params
    k_adapt_enable_   = params.find<int>("k_adapt_enable", 0) != 0;
    k_adapt_window_N_ = params.find<uint32_t>("k_adapt_window_N", 8);
    k_alpha_bw_       = params.find<double>("k_alpha_bw", 0.2);
    k_alpha_k_        = params.find<double>("k_alpha_k", 0.2);
    k_min_bytes_      = params.find<uint64_t>("k_min_bytes", 512);
    k_max_bytes_      = params.find<uint64_t>("k_max_bytes", 64*1024);
    k_delta_bytes_    = params.find<uint64_t>("k_delta_bytes", 512);
    // Adaptive control params
    ctrl_enable_ = params.find<int>("ctrl_enable", 0) != 0;
    ctrl_probe_every_N_ = params.find<uint32_t>("ctrl_probe_every_N", 4);
    ctrl_cooldown_N_ = params.find<uint32_t>("ctrl_cooldown_N", 4);
    ctrl_eps_reqs_ = params.find<double>("ctrl_eps_reqs", 0.02);
    ctrl_eps_burst_ = params.find<double>("ctrl_eps_burst", 0.05);
    ctrl_lat_tol_ns_ = params.find<uint64_t>("ctrl_lat_tol_ns", 1);
    ctrl_k_list_ = parseCsvU64_(params.find<std::string>("ctrl_k_list", "512,1024,2048,4096,8192"));
    ctrl_rowwin_list_ = parseCsvU64_(params.find<std::string>("ctrl_rowwin_list", "0,16384,32768,65536"));
    ctrl_timeout_list_ = parseCsvU64_(params.find<std::string>("ctrl_timeout_list", "0,300,600"));
    // Initial control cfg from current params
    ctrl_curr_cfg_.k = gap_k_bytes_;
    ctrl_curr_cfg_.rowwin_bytes = row_window_bytes_;
    ctrl_curr_cfg_.timeout_ns = row_window_timeout_ns_;

    // P1-2: granule export CSV (optional)
    export_granules_csv_ = params.find<std::string>("export_granules_csv", "");
    node_id_param_ = params.find<uint32_t>("node_id", 0);
    export_window_metrics_csv_ = params.find<std::string>("export_window_metrics_csv", "");

    resetGatherAutoCounters_();
}

GatherBufferIF::~GatherBufferIF() {
    if (!export_window_metrics_csv_.empty() && !window_metrics_written_) {
        exportWindowMetricsRow_(current_gather_id_, win_payload_bytes_, win_bursts_, win_inflight_peak_, win_buffer_max_bytes_);
    }
}

void GatherBufferIF::init(unsigned int phase) {
    backend_->init(phase);
    // 记录所有 phase 的初始化情况（仅在显式启用时打印）
    const char* sent_env = std::getenv("SNNDL_SENTINEL_ENABLE");
        if (sent_env && std::atoi(sent_env) != 0) {
            printf("[[sentinel-gbi-init]] phase=%u window_auto=%d manual=%d clock=%s win_cyc_g=%" PRIu64 " win_cyc_a=%" PRIu64 " win_cyc_s=%" PRIu64 " auto_bytes=%" PRIu64 " auto_reads=%" PRIu64 " defer=%d\n",
               phase, window_auto_ ? 1 : 0, manual_window_drive_ ? 1 : 0, clock_freq_.c_str(),
               win_cyc_gather_, win_cyc_apply_, win_cyc_scatter_,
               gather_auto_end_bytes_, gather_auto_end_reads_, defer_issue_until_apply_ ? 1 : 0);
        }
    if (phase == 0 && window_auto_) {
        if (out_.getVerboseLevel() >= 2) {
            out_.verbose(CALL_INFO, 0, 0,
                "[diag-gbi-init] window_auto=%d manual=%d(clock-forced) clock=%s win_cyc_g=%" PRIu64 " win_cyc_a=%" PRIu64 " win_cyc_s=%" PRIu64 " auto_bytes=%" PRIu64 " auto_reads=%" PRIu64 " defer=%d\n",
                window_auto_ ? 1 : 0, 0, clock_freq_.c_str(), win_cyc_gather_, win_cyc_apply_, win_cyc_scatter_,
                gather_auto_end_bytes_, gather_auto_end_reads_, defer_issue_until_apply_ ? 1 : 0);
        }
        // Always register clock for time-driven windows (manual drive deprecated)
        registerClock(clock_freq_, new Clock::Handler2<GatherBufferIF, &GatherBufferIF::clockTick>(this));
        stage_ = Stage::Gather; stage_counter_ = 0; current_gather_id_++;
        resetWindowMetrics_();
        resetGatherAutoCounters_();
        if (emit_stage_events_ && upstream_handler_) {
            auto* ev = new GasOpData(GasOp::BeginGather, (uint32_t)current_gather_id_, 0, 1, false);
            auto* cr = new StandardMem::CustomResp((StandardMem::Request::id_t)0, ev, 0, 0, 0);
            (*upstream_handler_)(cr);
        }
        sb_[gather_buf_index_].end_gather_seen = false;
    }
}

void GatherBufferIF::setup() { backend_->setup(); }

void GatherBufferIF::finish() {
    if (!export_window_metrics_csv_.empty() && !window_metrics_written_) {
        exportWindowMetricsRow_(current_gather_id_, win_payload_bytes_, win_bursts_, win_inflight_peak_, win_buffer_max_bytes_);
    }
    backend_->finish();
}

void GatherBufferIF::sendUntimedData(Request* req) { backend_->sendUntimedData(req); }

StandardMem::Request* GatherBufferIF::recvUntimedData() { return backend_->recvUntimedData(); }

StandardMem::Addr GatherBufferIF::getLineSize() { return backend_ ? backend_->getLineSize() : 64; }

void GatherBufferIF::setMemoryMappedAddressRegion(Addr start, Addr size) {
    if (backend_) backend_->setMemoryMappedAddressRegion(start, size);
}

void GatherBufferIF::manualWindowTick() {
    // Deprecated no-op for legacy compatibility.
    return;
}

void GatherBufferIF::send(Request* req) {
    // Control-plane: CustomReq
    if (auto* cr = dynamic_cast<StandardMem::CustomReq*>(req)) {
        auto* data = dynamic_cast<GasOpData*>(&cr->getData());
        if (data) {
            // In auto mode, ignore external GasOp control requests (manual drive deprecated)
            if (window_auto_) {
                if (!warned_auto_custom_req_ && diagEnabled_()) {
                    warned_auto_custom_req_ = true;
                    out_.verbose(CALL_INFO, 0, 0,
                        "[diag-gbi] CustomReq ignored: window_auto=1 manual_drive=0 (logged once)");
                }
                delete req; return;
            }
            if (out_.getVerboseLevel() >= 1) {
                out_.verbose(CALL_INFO, 1, 0,
                    "[diag-gbi] CustomReq op=%d (manual=%d) stage=%d gid=%" PRIu64 "\n",
                    (int)data->op, manual_window_drive_ ? 1 : 0, (int)stage_, current_gather_id_);
            }
            switch (data->op) {
                case GasOp::BeginGather:
                    stage_ = Stage::Gather; sb_[gather_buf_index_].end_gather_seen = false; current_gather_id_++;
                    resetGatherAutoCounters_();
                    sb_[gather_buf_index_].granules.clear(); sb_[gather_buf_index_].required_set.clear();
                    // 将非Gather阶段缓存的读导入当前Gather窗口
                    if (!queued_non_gather_reads_.empty()) {
                        for (auto* r : queued_non_gather_reads_) {
                            sb_[gather_buf_index_].pending_up_reads[r->getID()] = r;
                            uint64_t gsz = granuleSize();
                            bool use_row = (merge_==Merge::Row) || (merge_==Merge::Auto && row_bytes_guess_ > gsz);
                            uint64_t base = (merge_==Merge::Cacheline || (merge_==Merge::Auto && !use_row)) ? alignDown(r->pAddr, gsz)
                                               : (use_row ? alignDown(r->pAddr, row_bytes_guess_) : r->pAddr);
                            uint32_t sz = (merge_==Merge::None) ? r->size : (use_row ? row_bytes_guess_ : gsz);
                            uint64_t key = (base << 32) ^ (uint64_t)sz;
                            auto& g = sb_[gather_buf_index_].granules[key];
                            if (g.subs.empty()) { g.base = base; g.size = sz; g.window_id = current_gather_id_; sb_[gather_buf_index_].required_set.insert(key); if (stat_coalesce_granule_size_) stat_coalesce_granule_size_->addData((uint64_t)sz); }
                            uint64_t off = (r->pAddr >= base) ? (r->pAddr - base) : 0;
                            g.subs.push_back({r->getID(), off, (uint32_t)r->size, r});
                            if (!g.issued) { issueGranuleBuf_(gather_buf_index_, key, g); }
                            if (stat_up_reads_) stat_up_reads_->addData(1);
                        }
                        queued_non_gather_reads_.clear();
                    }
                    break;
                case GasOp::EndGather:
                    sb_[gather_buf_index_].end_gather_seen = true; tail_wait_start_ns_ = getCurrentSimTimeNano();
                    if (out_.getVerboseLevel() >= 1) {
                        out_.verbose(CALL_INFO, 1, 0,
                            "[diag-gbi] EndGather recv: granules=%zu required_set=%zu stage=%d\n",
                            sb_[gather_buf_index_].granules.size(), sb_[gather_buf_index_].required_set.size(), (int)stage_);
                    }
                    maybeEnterApply_();
                    break;
                case GasOp::BeginApply:
                    stage_ = Stage::Apply; emitApplyResponsesBuf_(apply_buf_index_); break;
                case GasOp::EndApply:
                    stage_ = Stage::Scatter; if (flush_after_scatter_) doFlushBuf_(apply_buf_index_); stage_ = Stage::Idle; break;
                case GasOp::BeginScatter:
                    stage_ = Stage::Scatter; break;
                case GasOp::EndScatter:
                    if (flush_after_scatter_) doFlushBuf_(apply_buf_index_); stage_ = Stage::Idle; break;
                case GasOp::FlushSRAM:
                    doFlushBuf_(apply_buf_index_); break;
                case GasOp::SetSlice:
                    // no-op for v1
                    break;
            }
            delete req; return;
        }
        // Unknown CustomReq: pass-through
    }

    // Data-plane
    if (auto* rd = dynamic_cast<StandardMem::Read*>(req)) {
        // Decide whether we can treat this as gather for the next SB
        bool can_gather_now = (stage_ == Stage::Gather) || (double_buffer_enable_ && stage_ == Stage::Apply);
        if (can_gather_now) {
            // 在Gather阶段，进入当前gather缓冲；在Apply阶段，上游按边读取属于当前apply窗口，应进入apply缓冲
            int tgt = (stage_ == Stage::Gather) ? gather_buf_index_ : apply_buf_index_;
            sb_[tgt].pending_up_reads[rd->getID()] = rd;
            gather_bytes_accum_ += (uint64_t)rd->size;
            gather_reads_accum_++;
            if (stage_ == Stage::Gather) {
                tryAutoEndGather_();
            }
            bool take_staging_path = gap_merge_enable_ && defer_issue_until_apply_;
            out_.verbose(CALL_INFO, 2, 0,
                "[diag-gbi] send() Read: stage=%d tgt_buf=%d gap_merge=%d defer=%d -> path=%s\n",
                (int)stage_, tgt, (int)gap_merge_enable_, (int)defer_issue_until_apply_,
                take_staging_path ? "STAGING" : "IMMEDIATE");
            if (defer_issue_until_apply_ && take_staging_path && !warned_defer_issue_path_ && diagEnabled_()) {
                warned_defer_issue_path_ = true;
                out_.verbose(CALL_INFO, 0, 0,
                    "[diag-gbi] defer_issue_until_apply=1: upstream Read IDs will be remapped via staged granules (logged once)");
            }
            if (take_staging_path) {
                // Stage later for gap/Lmax (and optional row-window) merging
                sb_[tgt].staging_reads.push_back(rd);
                sb_[tgt].staged_arrival_ns[rd->getID()] = getCurrentSimTimeNano();
            } else {
                uint64_t gsz = granuleSize();
                bool use_row = (merge_==Merge::Row) || (merge_==Merge::Auto && row_bytes_guess_ > gsz);
                uint64_t base = (merge_==Merge::Cacheline || (merge_==Merge::Auto && !use_row)) ? alignDown(rd->pAddr, gsz)
                               : (use_row ? alignDown(rd->pAddr, row_bytes_guess_) : rd->pAddr);
                uint32_t sz = (merge_==Merge::None) ? rd->size : (use_row ? row_bytes_guess_ : gsz);
                uint64_t key = (base << 32) ^ (uint64_t)sz; // compact key
                auto& g = sb_[tgt].granules[key];
                if (g.subs.empty()) { g.base = base; g.size = sz; g.window_id = current_gather_id_; sb_[tgt].required_set.insert(key); if (stat_coalesce_granule_size_) stat_coalesce_granule_size_->addData((uint64_t)sz); }
                uint64_t off = (rd->pAddr >= base) ? (rd->pAddr - base) : 0;
                g.subs.push_back({rd->getID(), off, (uint32_t)rd->size, rd});
                if (!defer_issue_until_apply_) {
                    if (!g.issued) { issueGranuleBuf_(tgt, key, g); }
                }
            }
            if (stat_up_reads_) stat_up_reads_->addData(1);
            return; // keep rd queued; deletion deferred after we emit ReadResp
        } else {
            if (strict_mode_) { queued_non_gather_reads_.push_back(rd); return; }
            backend_->send(req); return;
        }
    }

    // Writes/flush: pass-through; optionally invalidate overlapping SRAM
    if (auto* wr = dynamic_cast<StandardMem::Write*>(req)) {
        // Invalidate overlapping blocks for both SBs
        for (int b=0;b<2;++b) {
            if (!sb_[b].sram_blocks.empty()) {
                std::vector<uint64_t> toErase;
                for (auto& kv : sb_[b].sram_blocks) {
                    uint64_t base = kv.first >> 32; uint32_t sz = (uint32_t)(kv.first & 0xffffffffu);
                    if (!(wr->pAddr + wr->size <= base || wr->pAddr >= base + sz)) toErase.push_back(kv.first);
                }
                for (auto k : toErase) { sb_[b].sram_blocks.erase(k); }
            }
        }
        backend_->send(req); return;
    }

    // Other requests: pass-through
    backend_->send(req);
}

StandardMem::Request* GatherBufferIF::poll() {
    // We use push-based handler; upstream does not poll. Return nullptr.
    return nullptr;
}

// === helpers ===
GatherBufferIF::Merge GatherBufferIF::parseMerge(const std::string& s) const {
    if (s == "none") return Merge::None;
    if (s == "cacheline") return Merge::Cacheline;
    if (s == "row") return Merge::Row;
    return Merge::Auto;
}

GatherBufferIF::Sort GatherBufferIF::parseSort(const std::string& s) const {
    if (s == "addr") return Sort::Addr;
    if (s == "bank_row") return Sort::BankRow;
    return Sort::Row;
}

uint64_t GatherBufferIF::granuleSize() const {
    // Use downstream line size if available for cacheline policy
    auto ls = backend_ ? backend_->getLineSize() : 64;
    if (ls == 0) ls = 64;
    return (uint64_t)ls;
}

void GatherBufferIF::issueGranuleBuf_(int buf, uint64_t key, Granule& g) {
    // respect max_inflight (best-effort: simple check) across both buffers
    if ((inflight_counts_[0] + inflight_counts_[1]) >= max_inflight_reads_) return; // defer; will retry on next resp
    auto* rd = new StandardMem::Read(g.base, g.size);
    g.down_id = rd->getID();
    g.issued = true;
    g.issue_ns = getCurrentSimTimeNano();
    out_.verbose(CALL_INFO, 2, 0,
        "[diag-gbi] issue granule buf=%d key=0x%lx base=0x%lx size=%u id=%" PRIu64 "\n",
        buf, (unsigned long)key, (unsigned long)g.base, g.size, (uint64_t)g.down_id);
    inflight_down_[g.down_id] = std::make_pair(buf, key);
    inflight_counts_[buf]++;
    if (stat_reads_issued_) stat_reads_issued_->addData(1);
    backend_->send(rd);
}

void GatherBufferIF::onDownstreamResp_(Request* r) {
    // Only expecting ReadResp for now
    if (auto* rr = dynamic_cast<StandardMem::ReadResp*>(r)) {
        out_.verbose(CALL_INFO, 1, 0,
            "[diag-gbi] got ReadResp id=%" PRIu64 " bytes=%zu\n",
            (uint64_t)rr->getID(), rr->data.size());
        auto it = inflight_down_.find(rr->getID());
        if (it != inflight_down_.end()) {
            int buf_index = it->second.first; uint64_t key = it->second.second; inflight_down_.erase(it);
            if (buf_index >=0 && buf_index < 2) {
                auto& sb = sb_[buf_index];
                auto git = sb.granules.find(key);
                if (git != sb.granules.end()) {
                // store into SRAM
                ensureCapacity_(buf_index, git->second.size);
                auto& buf = sb.sram_blocks[key];
                buf.resize(git->second.size);
                if (!rr->data.empty()) {
                    std::memcpy(buf.data(), rr->data.data(), std::min<size_t>(buf.size(), rr->data.size()));
                }
                touchLRU_(buf_index, key);
                git->second.ready = true;
                if (out_.getVerboseLevel() >= 1) {
                    out_.verbose(CALL_INFO, 1, 0,
                        "[diag-gbi] mark ready buf=%d key=0x%lx subs=%zu\n",
                        buf_index, (unsigned long)key, (size_t)git->second.subs.size());
                }
                if (inflight_counts_[buf_index] > 0) inflight_counts_[buf_index]--;
                if (stat_unique_reads_) stat_unique_reads_->addData(1);
                if (stat_unique_bytes_) stat_unique_bytes_->addData(git->second.size);
                // Diagnostic: dump per-sub first-float values into CSV
                // Use SRAM block (blk) as the source of truth so we still produce samples even if rr->data is empty.
                if (!probe_csv_path_.empty()) {
                    FILE* fp = fopen(probe_csv_path_.c_str(), probe_csv_header_written_? "a" : "w");
                    if (fp) {
                        if (!probe_csv_header_written_) { fprintf(fp, "abs_addr,size,f0\n"); probe_csv_header_written_ = true; }
                        for (auto &sub : git->second.subs) {
                            float f0 = 0.0f;
                            if ((size_t)sub.offset + sizeof(float) <= buf.size()) {
                                std::memcpy(&f0, buf.data()+sub.offset, sizeof(float));
                            }
                            unsigned long abs = (unsigned long)(git->second.base + (uint64_t)sub.offset);
                            fprintf(fp, "0x%lx,%u,%.6f\n", abs, (unsigned)sub.size, f0);
                        }
                        fclose(fp);
                    }
                }
                // --- Adaptive k: segment timing and payload ---
                if (k_adapt_enable_ || ctrl_enable_) {
                    uint64_t end_ns = getCurrentSimTimeNano();
                    uint64_t t_seg_ns = (end_ns >= git->second.issue_ns) ? (end_ns - git->second.issue_ns) : 0;
                    // sum payload bytes of subs
                    uint64_t payload = 0;
                    for (auto &s : git->second.subs) payload += (uint64_t)s.size;
                    // accumulate window metrics
                    win_payload_bytes_ += payload;
                    win_bursts_ += 1;
                    if (t_seg_ns > 0) { win_seg_latency_sum_ns_ += t_seg_ns; win_seg_count_ += 1; }
                    // update B_eff (EMA) using segment throughput
                    if (k_adapt_enable_ && t_seg_ns > 0 && payload > 0) {
                        double inst_bw = (double)payload / (double)t_seg_ns; // bytes/ns
                        if (bw_eff_bytes_per_ns_ <= 0.0) bw_eff_bytes_per_ns_ = inst_bw;
                        else bw_eff_bytes_per_ns_ = (1.0 - k_alpha_bw_) * bw_eff_bytes_per_ns_ + k_alpha_bw_ * inst_bw;
                        // derive O_eff sample (ns): max(0, T - payload/B)
                        double ideal_ns = (bw_eff_bytes_per_ns_ > 0.0) ? ((double)payload / bw_eff_bytes_per_ns_) : (double)t_seg_ns;
                        double oeff = std::max(0.0, (double)t_seg_ns - ideal_ns);
                        oeff_samples_ns_.push_back((uint64_t)std::llround(oeff));
                    }
                }
                // Also propagate a lightweight stat update upstream so PE层可累计到 CSV
                if (upstream_handler_) {
                    auto* s = new GasStatData(1, (uint64_t)git->second.size);
                    auto* cr = new StandardMem::CustomResp((StandardMem::Request::id_t)0, s, 0, 0, 0);
                    (*upstream_handler_)(cr);
                }
                if (stat_buffer_occupancy_bytes_) {
                    uint64_t sum_bytes = sb_[0].bytes_in_sram + sb_[1].bytes_in_sram;
                    stat_buffer_occupancy_bytes_->addData(sum_bytes);
                    if (sum_bytes > win_buffer_max_bytes_) win_buffer_max_bytes_ = sum_bytes;
                }
                // 即时回传：若处于Apply阶段且本响应属于当前apply缓冲，尽快向上游发回，避免窗口切换时机错过
                if (stage_ == Stage::Apply && buf_index == apply_buf_index_) {
                    emitApplyResponsesBuf_(apply_buf_index_);
                }
            }
            }
        }
        delete r;
        // 进入或推进阶段判定
        if (stage_ == Stage::Gather) {
            // If all granules returned and EndGather seen, move to APPLY
            maybeEnterApply_();
            // 非延后模式：仍在Gather，继续发射未发granule
            if (!defer_issue_until_apply_ && stage_ == Stage::Gather) {
                for (auto& kv : sb_[gather_buf_index_].granules) {
                    if (!kv.second.issued) { issueGranuleBuf_(gather_buf_index_, kv.first, kv.second); }
                }
            }
        } else if (stage_ == Stage::Apply && apply_pending_emit_) {
            // 检查是否全部ready，若是且无需等待Apply窗口，则发放
            bool allReady = true;
            for (auto key : sb_[apply_buf_index_].required_set) {
                auto it2 = sb_[apply_buf_index_].granules.find(key);
                if (it2 == sb_[apply_buf_index_].granules.end() || !it2->second.ready) { allReady = false; break; }
            }
            if (!allReady && out_.getVerboseLevel() >= 2) {
                size_t ready_cnt = 0;
                for (auto key : sb_[apply_buf_index_].required_set) {
                    auto it2 = sb_[apply_buf_index_].granules.find(key);
                    if (it2 != sb_[apply_buf_index_].granules.end() && it2->second.ready) ++ready_cnt;
                }
                out_.verbose(CALL_INFO, 2, 0,
                    "[diag-gbi] Apply waiting: ready=%zu / total=%zu\n",
                    ready_cnt, sb_[apply_buf_index_].required_set.size());
            }
            if (allReady) {
                if (!window_auto_ || win_cyc_apply_ == 0) {
                    emitApplyResponsesBuf_(apply_buf_index_);
                    // EndApply barrier: ensure no inflight downstream reads
                    if (emit_stage_events_ && upstream_handler_) {
                        if (inflight_counts_[apply_buf_index_] == 0) {
                            auto* ea = new GasOpData(GasOp::EndApply, (uint32_t)current_gather_id_, 0, 1, false);
                            auto* cr = new StandardMem::CustomResp((StandardMem::Request::id_t)0, ea, 0, 0, 0);
                            (*upstream_handler_)(cr);
                        } else {
                            apply_pending_emit_ = true; // wait for drain
                            return;
                        }
                    }
                    stage_ = Stage::Scatter; stage_counter_ = 0;
                    if (emit_stage_events_ && upstream_handler_) {
                        auto* bs = new GasOpData(GasOp::BeginScatter, (uint32_t)current_gather_id_, 0, 1, false);
                        auto* cr2 = new StandardMem::CustomResp((StandardMem::Request::id_t)0, bs, 0, 0, 0);
                        (*upstream_handler_)(cr2);
                    }
                    if (flush_after_scatter_) doFlushBuf_(apply_buf_index_);
                    if (emit_stage_events_ && upstream_handler_) {
                        auto* es = new GasOpData(GasOp::EndScatter, (uint32_t)current_gather_id_, 0, 1, false);
                        auto* cr3 = new StandardMem::CustomResp((StandardMem::Request::id_t)0, es, 0, 0, 0);
                        (*upstream_handler_)(cr3);
                    }
                    stage_ = window_auto_ ? Stage::Gather : Stage::Idle;
                    if (window_auto_) { sb_[gather_buf_index_].end_gather_seen = false; current_gather_id_++; }
                    if (window_auto_ && emit_stage_events_ && upstream_handler_) {
                        auto* bg = new GasOpData(GasOp::BeginGather, (uint32_t)current_gather_id_, 0, 1, false);
                        auto* cr4 = new StandardMem::CustomResp((StandardMem::Request::id_t)0, bg, 0, 0, 0);
                        (*upstream_handler_)(cr4);
                    }
                }
            }
        }
        return;
    }

    // Pass-through for other responses
    if (upstream_handler_) (*upstream_handler_)(r);
}

void GatherBufferIF::maybeEnterApply_() {
    if (stage_ != Stage::Gather || !sb_[gather_buf_index_].end_gather_seen) {
        if (out_.getVerboseLevel() >= 2) {
            out_.verbose(CALL_INFO, 2, 0,
                "[diag-gbi] maybeEnterApply_ gate: stage=%d end_gather_seen=%d required_set=%zu granules=%zu\n",
                (int)stage_, sb_[gather_buf_index_].end_gather_seen?1:0,
                sb_[gather_buf_index_].required_set.size(), sb_[gather_buf_index_].granules.size());
        }
        return;
    }
    // 进入 APPLY；对于延后发射模式，在此时统一构建/排序并下发；否则要求全部ready再进入
    if (out_.getVerboseLevel() >= 1) {
        out_.verbose(CALL_INFO, 1, 0,
            "[diag-gbi] maybeEnterApply_: pre-switch stage=%d gid=%" PRIu64 " granules=%zu required_set=%zu inflight_g=%" PRIu64 " inflight_a=%" PRIu64 "\n",
            (int)stage_, current_gather_id_,
            sb_[gather_buf_index_].granules.size(), sb_[gather_buf_index_].required_set.size(),
            (uint64_t)inflight_counts_[gather_buf_index_], (uint64_t)inflight_counts_[apply_buf_index_]);
    }
    flushStageCycles_(Stage::Gather, false);
    stage_ = Stage::Apply; stage_counter_ = 0; apply_pending_emit_ = true;
    if (emit_stage_events_ && upstream_handler_) {
        auto* ev = new GasOpData(GasOp::BeginApply, (uint32_t)current_gather_id_, 0, 1, false);
        auto* cr = new StandardMem::CustomResp((StandardMem::Request::id_t)0, ev, 0, 0, 0);
        (*upstream_handler_)(cr);
    }
    // [DEBUG] 在切换前记录当前 Gather 缓冲区状态
    out_.verbose(CALL_INFO, 1, 0,
        "[diag-gbi] BEFORE switch: gather_buf=%d granules=%zu required_set=%zu pending_up_reads=%zu\n",
        gather_buf_index_, sb_[gather_buf_index_].granules.size(),
        sb_[gather_buf_index_].required_set.size(), sb_[gather_buf_index_].pending_up_reads.size());
    // Double buffer: 当前Gather缓冲区切换为Apply缓冲区，Gather切到另一页
    apply_buf_index_ = gather_buf_index_;
    gather_buf_index_ ^= 1;
    resetGatherAutoCounters_();
    // [DEBUG] 切换后记录 Apply 缓冲区状态
    out_.verbose(CALL_INFO, 1, 0,
        "[diag-gbi] AFTER switch: apply_buf=%d gather_buf=%d; apply granules=%zu required_set=%zu\n",
        apply_buf_index_, gather_buf_index_, sb_[apply_buf_index_].granules.size(),
        sb_[apply_buf_index_].required_set.size());
    if (defer_issue_until_apply_) {
        // 若启用 gap/Lmax 合并，则先按 bank/row 聚合构建 granule
        if (gap_merge_enable_) buildGranulesWithGapMergeBuf_(apply_buf_index_);
        // 排序并发射所有未发的granule（支持 bank_row）
        auto& gmap = sb_[apply_buf_index_].granules;
        std::vector<std::pair<uint64_t,uint64_t>> sorted; sorted.reserve(gmap.size());
        for (auto &kv : gmap) {
            // 优化2：使用缓存的排序键，避免重复计算
            if (!kv.second.sort_key_valid) {
                if (sort_ == Sort::Addr) {
                    kv.second.cached_sort_key = kv.second.base;
                } else if (sort_ == Sort::BankRow) {
                    kv.second.cached_sort_key = bankRowIndex(kv.second.base);
                } else {
                    kv.second.cached_sort_key = rowIndex(kv.second.base);
                }
                kv.second.sort_key_valid = true;
            }
            sorted.emplace_back(kv.second.cached_sort_key, kv.first);
        }
        std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
        // Compute pseudo row-adjacency for this window (before issuing)
        if (sorted.size() > 1) {
            for (size_t i = 1; i < sorted.size(); ++i) {
                auto itp = gmap.find(sorted[i-1].second);
                auto itc = gmap.find(sorted[i].second);
                if (itp != gmap.end() && itc != gmap.end()) {
                    ++win_row_adj_total_;
                    if (sort_ == Sort::BankRow) {
                        if (bankRowIndex(itp->second.base) == bankRowIndex(itc->second.base)) ++win_row_adj_same_;
                    } else {
                        if (rowIndex(itp->second.base) == rowIndex(itc->second.base)) ++win_row_adj_same_;
                    }
                }
            }
        }
        for (auto &p : sorted) {
            auto it = gmap.find(p.second);
            if (it != gmap.end() && !it->second.issued) issueGranuleBuf_(apply_buf_index_, p.second, it->second);
        }
    } else {
        // 非延后：只有当全部ready时才会立刻发放
        bool allReady = true;
        for (auto key : sb_[apply_buf_index_].required_set) {
            auto it = sb_[apply_buf_index_].granules.find(key);
            if (it == sb_[apply_buf_index_].granules.end() || !it->second.ready) { allReady = false; break; }
        }
        if (!allReady && out_.getVerboseLevel() >= 2) {
            size_t ready_cnt = 0;
            for (auto key : sb_[apply_buf_index_].required_set) {
                auto it2 = sb_[apply_buf_index_].granules.find(key);
                if (it2 != sb_[apply_buf_index_].granules.end() && it2->second.ready) ++ready_cnt;
            }
            out_.verbose(CALL_INFO, 2, 0,
                "[diag] BeginApply: waiting for ReadResp (%zu/%zu granules ready)\n",
                ready_cnt, sb_[apply_buf_index_].required_set.size());
        }
        // 标记需要在窗口结束时统一回传（FinishApplyWindow_将无条件emit）
        apply_pending_emit_ = true;
        if (!window_auto_ && win_cyc_apply_ == 0 && allReady) {
            emitApplyResponsesBuf_(apply_buf_index_);
            stage_ = Stage::Scatter; stage_counter_ = 0;
            if (flush_after_scatter_) doFlushBuf_(apply_buf_index_);
            stage_ = Stage::Idle;
            return;
        }
    }
    if (tail_wait_start_ns_ && stat_tail_wait_ns_) {
        uint64_t now = getCurrentSimTimeNano();
        if (now >= tail_wait_start_ns_) stat_tail_wait_ns_->addData(now - tail_wait_start_ns_);
        tail_wait_start_ns_ = 0;
    }
    if (out_.getVerboseLevel() >= 1) {
        out_.verbose(CALL_INFO, 1, 0,
            "[diag-gbi] maybeEnterApply_: post-switch apply_buf=%d granules=%zu required_set=%zu\n",
            apply_buf_index_, sb_[apply_buf_index_].granules.size(), sb_[apply_buf_index_].required_set.size());
    }
}

// Build granules from staged reads with gap/Lmax and bank/row grouping for a specific SB
void GatherBufferIF::buildGranulesWithGapMergeBuf_(int buf) {
    auto& S = sb_[buf];
    if (S.staging_reads.empty()) return;
    // Heuristic bank bits/shift detection if requested
    if (!bank_auto_done_ && bank_auto_enable_ && bank_bits_ == 0) {
        // Simple heuristic: reuse staged addrs of this buffer
        // (keep existing API by inlining logic here)
        std::vector<uint64_t> addrs;
        addrs.reserve(S.staging_reads.size());
        for (auto* rd : S.staging_reads) addrs.push_back(rd->pAddr);
        if (addrs.size() >= 16) {
            uint32_t best_bits = 0, best_shift = 0; uint64_t best_score = (uint64_t)-1;
            uint32_t minb = std::max<uint32_t>(1, bank_auto_min_banks_);
            uint32_t maxb = std::max<uint32_t>(minb, bank_auto_max_banks_);
            uint32_t target = (minb + maxb) / 2;
            for (uint32_t bits = 2; bits <= 6; ++bits) {
                for (uint32_t shift = 12; shift <= 24; ++shift) {
                    std::unordered_set<uint32_t> banks; banks.reserve(addrs.size());
                    for (auto a : addrs) {
                        uint32_t b = (uint32_t)((a >> shift) & ((1ull << bits) - 1ull));
                        banks.insert(b); if (banks.size() > maxb) break;
                    }
                    uint32_t uniq = (uint32_t)banks.size();
                    if (uniq < minb || uniq > maxb) continue;
                    uint64_t score = (uniq > target) ? (uniq - target) : (target - uniq);
                    score = score * 8 + (shift > 16 ? (shift - 16) : (16 - shift));
                    if (score < best_score) { best_score = score; best_bits = bits; best_shift = shift; }
                }
            }
            if (best_bits != 0) { bank_bits_ = best_bits; bank_shift_ = best_shift; }
            bank_auto_done_ = true;
        }
    }
    // Group by (bank,rowIndex)
    struct ReadItem { uint64_t addr; uint32_t size; uint64_t arr_ns; StandardMem::Read* rd; };
    std::unordered_map<uint64_t, std::vector<ReadItem>> groups;

    // 优化3：预分配groups容器（假设平均每组4个读请求）
    size_t estimated_groups = S.staging_reads.size() / 4;
    if (estimated_groups < 8) estimated_groups = 8;  // 最少预留8组
    groups.reserve(estimated_groups);

    for (auto* rd : S.staging_reads) {
        uint64_t addr = rd->pAddr; uint32_t sz = rd->size;
        uint64_t key = (bank_bits_? (bankIndex(addr)<<32):0) | (uint32_t)rowIndex(addr);
        uint64_t arr = 0;
        auto itst = S.staged_arrival_ns.find(rd->getID());
        if (itst != S.staged_arrival_ns.end()) arr = itst->second;

        // 优化3：预分配每组的vector空间（假设平均每组8个元素）
        auto& vec = groups[key];
        if (vec.empty()) vec.reserve(8);
        vec.push_back({addr, sz, arr, rd});
    }
    uint64_t gap_abs_sum = 0;
    // For each group, sort by addr and build segments
    for (auto &kv : groups) {
        auto &vec = kv.second;
        std::sort(vec.begin(), vec.end(), [](const ReadItem&a, const ReadItem&b){ return a.addr < b.addr; });
        uint64_t cur_base = 0, cur_end = 0; bool has=false;
        uint64_t seg_sum_bytes = 0;      // sum of sub-read sizes in current segment
        bool seg_used_row_window = false; // whether coarse row-window absorption was used
        uint64_t seg_start_ns = 0;        // arrival time of first sub-read in segment
        // temp list of sub-reads per current segment
        std::vector<ReadItem> segSubs;
        // 优化3：预分配segSubs容器（假设每段平均包含vec的一半元素）
        segSubs.reserve(vec.size() / 2 + 2);
        auto flush_segment = [&](bool mark_rowwin=false) {
            if (!has) return;
            uint64_t base = cur_base; uint32_t sz = (uint32_t)(cur_end - cur_base);
            uint64_t gkey = (base << 32) ^ (uint64_t)sz;
            auto &g = S.granules[gkey];
            if (g.subs.empty()) {
                g.base = base; g.size = sz; g.window_id = current_gather_id_;
                S.required_set.insert(gkey);
                if (stat_coalesce_granule_size_) stat_coalesce_granule_size_->addData((uint64_t)sz);
                // 优化3：预分配subs空间（已知即将添加segSubs.size()个元素）
                g.subs.reserve(segSubs.size());
            }
            for (auto &it : segSubs) {
                uint64_t off = (it.addr >= base) ? (it.addr - base) : 0;
                g.subs.push_back({it.rd->getID(), off, it.size, it.rd});
            }
            // P1-2: export granule row if enabled
            if (!export_granules_csv_.empty()) {
                exportGranuleRow_(seg_start_ns, (uint32_t)sz);
            }
            // Row-window stats: count only if we actually used row-window absorption or explicitly marked (e.g., timeout)
            if (row_window_enable_ && stat_row_window_triggers_ && stat_row_window_bytes_) {
                if (seg_used_row_window || mark_rowwin) {
                    stat_row_window_triggers_->addData(1);
                    stat_row_window_bytes_->addData((uint64_t)sz); // bytes actually issued on DRAM (including holes)
                }
            }
            // Upstream aggregation: report per-segment metrics (bursts/payload, and row-window if used)
            if (upstream_handler_) {
                uint64_t rwt = (seg_used_row_window || mark_rowwin) ? 1 : 0;
                uint64_t rwb = (seg_used_row_window || mark_rowwin) ? (uint64_t)sz : 0;
                uint64_t bursts = 1;
                uint64_t payload = seg_sum_bytes; // sum of sub-read sizes in this segment
                auto* s = new GasStatData(0, 0, rwt, rwb, bursts, payload);
                auto* cr = new StandardMem::CustomResp((StandardMem::Request::id_t)0, s, 0, 0, 0);
                (*upstream_handler_)(cr);
            }
            segSubs.clear(); has=false;
            seg_sum_bytes = 0; seg_used_row_window = false; seg_start_ns = 0;
        };
        for (auto &it : vec) {
            uint64_t a = it.addr, b = it.addr + it.size;
            if (!has) {
                cur_base = a; cur_end = b; seg_sum_bytes = it.size; segSubs.clear(); segSubs.push_back(it); has=true; seg_start_ns = it.arr_ns; continue;
            }
            if (a <= cur_end) {
                // overlap/adjacent
                if (b > cur_end) cur_end = b;
                segSubs.push_back(it);
                seg_sum_bytes += it.size;
            } else {
                uint64_t gap = a - cur_end;
                uint64_t new_len = (b - cur_base);
                bool absorbed = false;
                if (gap_merge_enable_ && gap_k_bytes_>0 && gap <= gap_k_bytes_ && new_len <= burst_bytes_max_) {
                    // absorb gap
                    cur_end = b; gap_abs_sum += gap; segSubs.push_back(it); absorbed = true; seg_sum_bytes += it.size;
                }
                // Row-window timeout trigger: if enabled and window has waited too long, flush before adding
                if (!absorbed && row_window_enable_ && row_window_timeout_ns_>0 && seg_start_ns>0 && it.arr_ns>0) {
                    if (it.arr_ns >= seg_start_ns && (it.arr_ns - seg_start_ns) >= row_window_timeout_ns_) {
                        // flush current segment as a row-window burst due to timeout
                        flush_segment(/*mark_rowwin=*/true);
                        // start new segment with current item
                        cur_base = a; cur_end = b; has=true; segSubs.push_back(it); seg_sum_bytes = it.size; seg_start_ns = it.arr_ns; continue;
                    }
                }
                // Row-window coarse merge (allow larger gap if within row_window_bytes threshold of sum-bytes, and not exceeding Lmax)
                if (!absorbed && row_window_enable_ && row_window_bytes_>0) {
                    uint64_t tentative_sum = seg_sum_bytes + it.size;
                    if (tentative_sum <= row_window_bytes_ && new_len <= burst_bytes_max_) {
                        cur_end = b; // absorb regardless of gap size
                        segSubs.push_back(it); absorbed = true; seg_sum_bytes = tentative_sum; seg_used_row_window = true;
                    }
                }
                if (!absorbed) {
                    // neither fine nor coarse merge possible: flush and start new
                    flush_segment();
                    cur_base = a; cur_end = b; has=true; segSubs.push_back(it); seg_sum_bytes = it.size; seg_start_ns = it.arr_ns; seg_used_row_window = false;
                }
            }
        }
        // Flush the last segment; mark as row-window only if we used row-window absorption at least once
        flush_segment(/*mark_rowwin=*/seg_used_row_window);
    }
    if (stat_gap_absorbed_bytes_ && gap_abs_sum) stat_gap_absorbed_bytes_->addData(gap_abs_sum);
    S.staging_reads.clear();
    S.staged_arrival_ns.clear();
}

void GatherBufferIF::emitApplyResponsesBuf_(int buf) {
    // For each upstream Read queued in this gather, emit a ReadResp from SRAM
    auto& S = sb_[buf];
    out_.verbose(CALL_INFO, 1, 0,
        "[diag-gbi] emitApplyResponses buf=%d granules=%zu pending_up_reads=%zu\n",
        buf, (size_t)S.granules.size(), S.pending_up_reads.size());

    // 收集已处理完成的 granule keys（ready 且已发回所有 sub-reads）
    std::vector<uint64_t> completed_keys;
    completed_keys.reserve(S.granules.size());

    for (auto& kvg : S.granules) {
        uint64_t key = kvg.first; auto& g = kvg.second;

        // 跳过未 ready 的 granule（保留到下次处理）
        if (!g.ready) {
            out_.verbose(CALL_INFO, 1, 0,
                "[diag-gbi] granule key=0x%lx NOT ready; window=%" PRIu64 " buf=%d (will retry next Apply)\n",
                (unsigned long)key, (uint64_t)g.window_id, buf);
            continue;
        }

        auto bit = S.sram_blocks.find(key);
        if (bit == S.sram_blocks.end()) {
            out_.verbose(CALL_INFO, 1, 0,
                "[diag-gbi] granule key=0x%lx missing SRAM entry (buf=%d)\n",
                (unsigned long)key, buf);
            continue;
        }

        const auto& blk = bit->second;
        if (g.subs.empty()) {
            out_.verbose(CALL_INFO, 1, 0,
                "[diag-gbi] granule key=0x%lx READY but subs=0 (buf=%d) pending_up_reads=%zu\n",
                (unsigned long)key, buf, S.pending_up_reads.size());
        }
        out_.verbose(CALL_INFO, 1, 0,
            "[diag-gbi] granule key=0x%lx subs=%zu ready=%d buf=%d\n",
            (unsigned long)key, (size_t)g.subs.size(), (int)g.ready, buf);

        // 发回所有 sub-reads
        for (auto& s : g.subs) {
            auto* resp = new StandardMem::ReadResp(s.up_read, std::vector<uint8_t>(s.size));
            // Fill payload
            if (s.offset + s.size <= blk.size()) {
                std::memcpy(resp->data.data(), blk.data() + s.offset, s.size);
            }
            out_.verbose(CALL_INFO, 1, 0,
                "[diag-gbi] respond up_id=%" PRIu64 " key=0x%lx off=%u size=%u pending=%zu data_empty=%d buf=%d\n",
                (uint64_t)s.up_id, (unsigned long)key, (unsigned)s.offset,
                (unsigned)s.size, (size_t)S.pending_up_reads.size(), resp->data.empty()?1:0, buf);
            // Diagnostic probe (no-ops when verbose==0): print first-float value of this sub-read
            if (out_.getVerboseLevel() >= 1 && !resp->data.empty()) {
                float f0 = 0.0f;
                if (resp->data.size() >= sizeof(float)) {
                    std::memcpy(&f0, resp->data.data(), sizeof(float));
                }
                uint64_t abs_addr = g.base + (uint64_t)s.offset;
                out_.verbose(CALL_INFO, 1, 0,
                    "[VERIFY][probe-gas] granule_base=0x%lx sub_off=%u abs=0x%lx size=%u f0=%.6f\n",
                    (unsigned long)g.base, (unsigned)s.offset,
                    (unsigned long)abs_addr, (unsigned)s.size, f0);
            }
            if (!probe_csv_path_.empty()) {
                // write one line per sub-read with addr,size,first-float (read from SRAM block)
                FILE* fp = fopen(probe_csv_path_.c_str(), probe_csv_header_written_? "a" : "w");
                if (fp) {
                    if (!probe_csv_header_written_) { fprintf(fp, "abs_addr,size,f0\n"); probe_csv_header_written_ = true; }
                    float f0 = 0.0f;
                    if ((size_t)s.offset + sizeof(float) <= blk.size()) std::memcpy(&f0, blk.data() + s.offset, sizeof(float));
                    unsigned long abs = (unsigned long)(g.base + (uint64_t)s.offset);
                    fprintf(fp, "0x%lx,%u,%.6f\n", abs, (unsigned)s.size, f0);
                    fclose(fp);
                }
            }
            if (upstream_handler_) (*upstream_handler_)(resp);
            // 注意：上游StandardMem::Read的所有权由上游组件管理。
            // 这里不再delete上游请求指针，仅从追踪表移除，避免重复释放导致的崩溃。
            auto itup = S.pending_up_reads.find(s.up_id);
            if (itup != S.pending_up_reads.end()) {
                S.pending_up_reads.erase(itup);
                out_.verbose(CALL_INFO, 1, 0,
                    "[diag-gbi] pending erase up_id=%" PRIu64 " remaining=%zu\n",
                    (uint64_t)s.up_id, (size_t)S.pending_up_reads.size());
            } else {
                out_.verbose(CALL_INFO, 1, 0,
                    "[diag-gbi] pending MISS up_id=%" PRIu64 "\n",
                    (uint64_t)s.up_id);
            }
        }

        // 标记此 granule 为已完成（可以清空）
        completed_keys.push_back(key);
    }

    // 只清空已完成的 granules（ready 且已发回的）
    for (auto key : completed_keys) {
        S.granules.erase(key);
        S.required_set.erase(key);
    }

    // 诊断：报告有多少 granules 被保留到下次处理
    size_t remaining_granules = S.granules.size();
    size_t remaining_pending = S.pending_up_reads.size();
    if (remaining_granules > 0 || remaining_pending > 0) {
        out_.verbose(CALL_INFO, 1, 0,
            "[diag-gbi] Apply buf=%d RETAINED: granules=%zu (not ready), pending_up_reads=%zu\n",
            buf, remaining_granules, remaining_pending);
    }

    // 清理其他窗口状态（但保留未完成的 granules 和 pending_up_reads）
    S.end_gather_seen = false;
    S.staging_reads.clear();
    apply_pending_emit_ = false;
}

void GatherBufferIF::doFlushBuf_(int buf) {
    auto& S = sb_[buf];

    // 清理LRU结构（RAII容器直接清空）
    S.lru_list.clear();
    S.lru_map.clear();

    // 清理SRAM数据
    S.sram_blocks.clear();
    S.bytes_in_sram = 0;
}

void GatherBufferIF::touchLRU_(int buf, uint64_t key) {
    auto& S = sb_[buf];

    auto it = S.lru_map.find(key);
    if (it != S.lru_map.end()) {
        // 移动到队尾（最近使用）
        S.lru_list.splice(S.lru_list.end(), S.lru_list, it->second);
    } else {
        // 新key：插入尾部并记录迭代器
        S.lru_list.push_back(key);
        auto it_list = std::prev(S.lru_list.end());
        S.lru_map.emplace(key, it_list);
    }
}

void GatherBufferIF::ensureCapacity_(int buf, uint64_t need) {
    auto& S = sb_[buf];

    // 淘汰最久未使用的节点（从front开始）
    while (S.bytes_in_sram + need > sram_bytes_ && !S.lru_list.empty()) {
        uint64_t k = S.lru_list.front();
        S.lru_list.pop_front();
        S.lru_map.erase(k);
        // 从SRAM中移除对应数据（若存在）
        auto it = S.sram_blocks.find(k);
        if (it != S.sram_blocks.end()) {
            S.bytes_in_sram -= it->second.size();
            S.sram_blocks.erase(it);
            if (stat_evictions_) {
                stat_evictions_->addData(1);
            }
        }
    }

    S.bytes_in_sram += need;
}

bool GatherBufferIF::clockTick(Cycle_t) {
    if (!window_auto_) return false;
    if (!clock_tick_logged_) {
        if (out_.getVerboseLevel() >= 2) {
            out_.verbose(CALL_INFO, 0, 0,
                "[diag-gbi-clock] first tick stage=%d counter=%" PRIu64 " win_cyc_g=%" PRIu64 "\n",
                (int)stage_, stage_counter_, win_cyc_gather_);
        }
        clock_tick_logged_ = true;
    }
    stage_counter_++;
    if (stage_ == Stage::Gather) stage_cycles_accum_[0]++;
    else if (stage_ == Stage::Apply) stage_cycles_accum_[1]++;
    else if (stage_ == Stage::Scatter) stage_cycles_accum_[2]++;
    if (stage_ == Stage::Gather) {
        const bool empty_window = sb_[gather_buf_index_].granules.empty() && sb_[gather_buf_index_].required_set.empty();
        if (empty_window) {
            // 尝试主动结束 Gather 并进入 Apply（即便没有边），保持窗口推进
            if (win_cyc_gather_ == 0 || stage_counter_ >= win_cyc_gather_) {
                sb_[gather_buf_index_].end_gather_seen = true;
                maybeEnterApply_();
            }
        } else {
            if (win_cyc_gather_ && stage_counter_ >= win_cyc_gather_) {
                sb_[gather_buf_index_].end_gather_seen = true;
                maybeEnterApply_();
                // in immediate path stage_ may now be Idle or Scatter; optional wait handled there
            } else {
                tryAutoEndGather_();
            }
        }
    } else if (stage_ == Stage::Apply) {
        // 并行：在Apply阶段也推进下一窗口的Gather构建与下发
        if (double_buffer_enable_ && defer_issue_until_apply_ && !sb_[gather_buf_index_].staging_reads.empty()) {
            buildGranulesWithGapMergeBuf_(gather_buf_index_);
            // 直接按当前排序策略下发新窗口的granule，实现与Apply并行
            auto& gmap = sb_[gather_buf_index_].granules;
            std::vector<std::pair<uint64_t,uint64_t>> sorted; sorted.reserve(gmap.size());
            for (auto &kv : gmap) {
                if (kv.second.issued) continue;
                // 优化2：使用缓存的排序键
                if (!kv.second.sort_key_valid) {
                    if (sort_ == Sort::Addr) {
                        kv.second.cached_sort_key = kv.second.base;
                    } else if (sort_ == Sort::BankRow) {
                        kv.second.cached_sort_key = bankRowIndex(kv.second.base);
                    } else {
                        kv.second.cached_sort_key = rowIndex(kv.second.base);
                    }
                    kv.second.sort_key_valid = true;
                }
                sorted.emplace_back(kv.second.cached_sort_key, kv.first);
            }
            std::sort(sorted.begin(), sorted.end(), [](const auto& a, const auto& b){ return a.first < b.first; });
            for (auto &p : sorted) {
                auto it = gmap.find(p.second);
                if (it != gmap.end() && !it->second.issued) issueGranuleBuf_(gather_buf_index_, p.second, it->second);
            }
        }
        if (apply_auto_end_enable_ && win_cyc_apply_ > 0) {
            if (tryAutoEndApply_()) return false;
        }
        if (win_cyc_apply_ && stage_counter_ >= win_cyc_apply_) {
            if (!finishApplyWindow_()) return false;
        }
    } else if (stage_ == Stage::Scatter) {
        bool scatter_due = false;
        if (scatter_immediate_complete_ && stage_counter_ >= 1) {
            scatter_due = true;
        } else if (win_cyc_scatter_ && stage_counter_ >= win_cyc_scatter_) {
            scatter_due = true;
        }
        if (scatter_due) {
            flushStageCycles_(Stage::Scatter, true);
            // start new gather window
            if (emit_stage_events_ && upstream_handler_) {
                auto* es = new GasOpData(GasOp::EndScatter, (uint32_t)current_gather_id_, 0, 1, false);
                auto* cr3 = new StandardMem::CustomResp((StandardMem::Request::id_t)0, es, 0, 0, 0);
                (*upstream_handler_)(cr3);
            }
            stage_ = Stage::Gather; stage_counter_ = 0; sb_[gather_buf_index_].end_gather_seen = false; current_gather_id_++;
            if (emit_stage_events_ && upstream_handler_) {
                auto* bg = new GasOpData(GasOp::BeginGather, (uint32_t)current_gather_id_, 0, 1, false);
                auto* cr4 = new StandardMem::CustomResp((StandardMem::Request::id_t)0, bg, 0, 0, 0);
                (*upstream_handler_)(cr4);
            }
            // 不清理 gather 缓冲，允许在 Apply 阶段已开始的下一窗口继续累积
            // 导入非Gather缓存
            if (!queued_non_gather_reads_.empty()) {
                for (auto* r : queued_non_gather_reads_) {
                    sb_[gather_buf_index_].pending_up_reads[r->getID()] = r;
                    uint64_t gsz = granuleSize();
                    uint64_t base = (merge_==Merge::Cacheline || merge_==Merge::Auto) ? alignDown(r->pAddr, gsz)
                                   : (merge_==Merge::Row ? alignDown(r->pAddr, row_bytes_guess_) : r->pAddr);
                    uint32_t sz = (merge_==Merge::None) ? r->size : (merge_==Merge::Row ? row_bytes_guess_ : gsz);
                    uint64_t key = (base << 32) ^ (uint64_t)sz;
                    auto& g = sb_[gather_buf_index_].granules[key];
                    if (g.subs.empty()) { g.base = base; g.size = sz; g.window_id = current_gather_id_; sb_[gather_buf_index_].required_set.insert(key); if (stat_coalesce_granule_size_) stat_coalesce_granule_size_->addData((uint64_t)sz); }
                    uint64_t off = (r->pAddr >= base) ? (r->pAddr - base) : 0;
                    g.subs.push_back({r->getID(), off, (uint32_t)r->size, r});
                    if (!g.issued) { issueGranuleBuf_(gather_buf_index_, key, g); }
                    if (stat_up_reads_) stat_up_reads_->addData(1);
                }
                queued_non_gather_reads_.clear();
            }
        }
    }
    // track inflight peak (sum of both buffers)
    uint64_t inflight_total = (uint64_t)(inflight_counts_[0] + inflight_counts_[1]);
    if (stat_inflight_peak_) stat_inflight_peak_->addData(inflight_total);
    if (inflight_total > win_inflight_peak_) win_inflight_peak_ = inflight_total;
    return false; // continue ticking
}

// === Adaptive-control helpers (definitions) ===
void GatherBufferIF::resetWindowMetrics_() {
    win_payload_bytes_ = 0;
    win_bursts_ = 0;
    win_seg_latency_sum_ns_ = 0;
    win_seg_count_ = 0;
    win_inflight_peak_ = inflight_counts_[0] + inflight_counts_[1];
    win_row_adj_same_ = 0;
    win_row_adj_total_ = 0;
    win_buffer_max_bytes_ = 0;
}

void GatherBufferIF::resetGatherAutoCounters_() {
    gather_bytes_accum_ = 0;
    gather_reads_accum_ = 0;
    gather_auto_triggered_ = false;
}

void GatherBufferIF::tryAutoEndGather_() {
    if (!window_auto_) return;
    if (!gather_auto_end_bytes_ && !gather_auto_end_reads_) return;
    // 避免在无任何需求时空转切窗
    if (sb_[gather_buf_index_].granules.empty() && sb_[gather_buf_index_].required_set.empty()) return;
    const bool bytes_hit = (gather_auto_end_bytes_ && gather_bytes_accum_ >= gather_auto_end_bytes_);
    const bool reads_hit = (gather_auto_end_reads_ && gather_reads_accum_ >= gather_auto_end_reads_);
    if (!bytes_hit && !reads_hit) return;
    gather_auto_triggered_ = true;
    if (out_.getVerboseLevel() >= 1) {
        out_.verbose(CALL_INFO, 1, 0,
            "[diag-gbi] AutoEndGather hit: stage=%d counter=%" PRIu64 " bytes=%" PRIu64 "/%" PRIu64 " reads=%" PRIu64 "/%" PRIu64 " granules=%zu required_set=%zu end_gather_seen=%d\n",
            (int)stage_, stage_counter_,
            gather_bytes_accum_, gather_auto_end_bytes_,
            gather_reads_accum_, gather_auto_end_reads_,
            sb_[gather_buf_index_].granules.size(), sb_[gather_buf_index_].required_set.size(),
            sb_[gather_buf_index_].end_gather_seen ? 1 : 0);
    }
    if (!sb_[gather_buf_index_].end_gather_seen) {
        sb_[gather_buf_index_].end_gather_seen = true;
    }
    if (stage_ == Stage::Gather) {
        maybeEnterApply_();
    }
}

bool GatherBufferIF::tryAutoEndApply_() {
    if (!window_auto_) return false;
    if (!apply_auto_end_enable_) return false;
    if (stage_ != Stage::Apply) return false;
    if (win_cyc_apply_ == 0) return false;
    bool allReady = true;
    for (auto key : sb_[apply_buf_index_].required_set) {
        auto it = sb_[apply_buf_index_].granules.find(key);
        if (it == sb_[apply_buf_index_].granules.end() || !it->second.ready) {
            allReady = false;
            break;
        }
    }
    if (!allReady) return false;
    return finishApplyWindow_();
}

bool GatherBufferIF::finishApplyWindow_() {
    flushStageCycles_(Stage::Apply, false);
    // 始终在窗口结束时向上游发回已ready的响应，避免非defer路径丢失回传
    emitApplyResponsesBuf_(apply_buf_index_);
    if (emit_stage_events_ && upstream_handler_) {
        if (inflight_counts_[apply_buf_index_] == 0 || emit_lenient_) {
            auto* ea = new GasOpData(GasOp::EndApply, (uint32_t)current_gather_id_, 0, 1, false);
            auto* cr = new StandardMem::CustomResp((StandardMem::Request::id_t)0, ea, 0, 0, 0);
            (*upstream_handler_)(cr);
        } else {
            apply_pending_emit_ = true;
            return false; // wait for downstream drain
        }
    }
    stage_ = Stage::Scatter;
    stage_counter_ = 0;
    if (emit_stage_events_ && upstream_handler_) {
        auto* bs = new GasOpData(GasOp::BeginScatter, (uint32_t)current_gather_id_, 0, 1, false);
        auto* cr2 = new StandardMem::CustomResp((StandardMem::Request::id_t)0, bs, 0, 0, 0);
        (*upstream_handler_)(cr2);
    }
    if (k_adapt_enable_) {
        if (!oeff_samples_ns_.empty() && bw_eff_bytes_per_ns_ > 0.0) {
            uint64_t sum = 0; for (auto v : oeff_samples_ns_) sum += v;
            uint64_t oeff_avg = sum / (uint64_t)oeff_samples_ns_.size();
            double k_cand = (double)oeff_avg * bw_eff_bytes_per_ns_;
            uint64_t k_cand_bytes = (uint64_t)std::llround(k_cand);
            if (k_cand_bytes < k_min_bytes_) k_cand_bytes = k_min_bytes_;
            if (k_cand_bytes > k_max_bytes_) k_cand_bytes = k_max_bytes_;
            if (k_dyn_bytes_ == 0) k_dyn_bytes_ = k_cand_bytes;
            else {
                k_dyn_bytes_ = (uint64_t)std::llround((1.0 - k_alpha_k_) * (double)k_dyn_bytes_ + k_alpha_k_ * (double)k_cand_bytes);
            }
            k_window_updates_++;
            if (k_window_updates_ >= k_adapt_window_N_) {
                k_window_updates_ = 0;
                uint64_t delta = (k_dyn_bytes_ > gap_k_bytes_) ? (k_dyn_bytes_ - gap_k_bytes_) : (gap_k_bytes_ - k_dyn_bytes_);
                if ((gap_k_bytes_ == 0) || (delta >= k_delta_bytes_)) {
                    gap_k_bytes_ = k_dyn_bytes_;
                }
            }
            if (stat_k_dyn_bytes_) stat_k_dyn_bytes_->addData(k_dyn_bytes_);
            if (stat_oeff_ns_avg_) stat_oeff_ns_avg_->addData(oeff_avg);
            if (stat_bw_eff_bytes_per_us_) stat_bw_eff_bytes_per_us_->addData((uint64_t)std::llround(bw_eff_bytes_per_ns_ * 1000.0));
        }
        oeff_samples_ns_.clear();
    }
    if (flush_after_scatter_) doFlushBuf_(apply_buf_index_);
    controlStep_();
    return true;
}

void GatherBufferIF::flushStageCycles_(Stage stage, bool window_boundary) {
    if (!window_auto_) return;
    int idx = stageIndex_(stage);
    if (idx < 0) return;
    uint64_t val = stage_cycles_accum_[idx];
    stage_cycles_last_[idx] = val;
    stage_cycles_accum_[idx] = 0;
    Statistic<uint64_t>* stat = nullptr;
    if (idx == 0) stat = stat_stage_cycles_g_;
    else if (idx == 1) stat = stat_stage_cycles_a_;
    else if (idx == 2) stat = stat_stage_cycles_s_;
    if (stat) stat->addData(val);
    if (window_boundary && stage == Stage::Scatter) {
        if (stage_cycles_export_enable_ && !stage_cycles_csv_.empty()) {
            if (!stage_cycles_stream_.is_open()) {
                stage_cycles_stream_.open(stage_cycles_csv_, std::ios::out | std::ios::app);
                if (stage_cycles_stream_.good() && stage_cycles_stream_.tellp() > 0) {
                    stage_cycles_header_written_ = true;
                }
                if (stage_cycles_stream_.good() && !stage_cycles_header_written_) {
                    stage_cycles_stream_ << "window_id,gather_cycles,apply_cycles,scatter_cycles\n";
                    stage_cycles_header_written_ = true;
                }
            }
            if (stage_cycles_stream_.good()) {
                stage_cycles_stream_ << current_gather_id_ << ','
                                     << stage_cycles_last_[0] << ','
                                     << stage_cycles_last_[1] << ','
                                     << stage_cycles_last_[2] << '\n';
            }
        }
        stage_cycles_last_[0] = stage_cycles_last_[1] = stage_cycles_last_[2] = 0;
    }
}

int GatherBufferIF::stageIndex_(Stage stage) const {
    switch (stage) {
        case Stage::Gather: return 0;
        case Stage::Apply: return 1;
        case Stage::Scatter: return 2;
        default: return -1;
    }
}

std::vector<uint64_t> GatherBufferIF::parseCsvU64_(const std::string& s) {
    std::vector<uint64_t> v; std::string num;
    for (char c : s) {
        if (c==',' || c==' ' || c=='\t') { if (!num.empty()) { v.push_back((uint64_t)std::stoull(num)); num.clear(); } }
        else num.push_back(c);
    }
    if (!num.empty()) v.push_back((uint64_t)std::stoull(num));
    if (v.empty()) v.push_back(0);
    return v;
}

bool GatherBufferIF::diagEnabled_(int level) const {
    if (out_.getVerboseLevel() >= level) return true;
    return snndlGlobalDebugEnabled();
}

void GatherBufferIF::exportGranuleRow_(uint64_t start_ns, uint32_t bytes) {
    if (export_granules_csv_.empty()) return;
    if (!granule_stream_.is_open()) {
        granule_stream_.open(export_granules_csv_, std::ios::out | std::ios::app);
        if (!granule_stream_.good()) return;
        if (granule_stream_.tellp() > 0) export_header_written_ = true;
        if (!export_header_written_) {
            granule_stream_ << "step,req_tile,owner_tile,burst_bytes,start_ns\n";
            export_header_written_ = true;
        }
    }
    granule_stream_ << current_gather_id_ << ','
                    << node_id_param_ << ','
                    << node_id_param_ << ','
                    << bytes << ','
                    << start_ns << '\n';
}

void GatherBufferIF::exportWindowMetricsRow_(uint64_t window_id, uint64_t payload_bytes, uint64_t bursts,
                                 uint64_t inflight_peak, uint64_t buffer_max) {
    if (export_window_metrics_csv_.empty()) return;
    if (!window_stream_.is_open()) {
        window_stream_.open(export_window_metrics_csv_, std::ios::out | std::ios::app);
        if (!window_stream_.good()) return;
        if (window_stream_.tellp() > 0) window_metrics_header_written_ = true;
        if (!window_metrics_header_written_) {
            window_stream_ << "window_id,payload_bytes,bursts,inflight_peak,buffer_max_bytes\n";
            window_metrics_header_written_ = true;
        }
    }
    window_stream_ << window_id << ',' << payload_bytes << ',' << bursts << ',' << inflight_peak << ',' << buffer_max << '\n';
    window_metrics_written_ = true;
    // reset buffer max for next window
    win_buffer_max_bytes_ = 0;
}

void GatherBufferIF::applyCtrlConfig_(uint64_t k, uint64_t rowwin_bytes, uint64_t timeout_ns) {
    // Apply k
    if (k == 0 || burst_bytes_max_ == 0) { gap_merge_enable_ = false; gap_k_bytes_ = 0; }
    else { gap_merge_enable_ = true; gap_k_bytes_ = k; }
    // Apply row-window params
    row_window_bytes_ = rowwin_bytes;
    row_window_enable_ = (row_window_bytes_ > 0);
    row_window_timeout_ns_ = timeout_ns;
}

void GatherBufferIF::controlStep_() {
    uint64_t win_payload = win_payload_bytes_;
    uint64_t win_bursts = win_bursts_;
    uint64_t win_inflight = win_inflight_peak_;
    uint64_t win_buffer = win_buffer_max_bytes_;
    if (!export_window_metrics_csv_.empty()) {
        exportWindowMetricsRow_(current_gather_id_, win_payload, win_bursts, win_inflight, win_buffer);
    }
    if (upstream_handler_ && (win_inflight > 0 || win_buffer > 0)) {
        auto* s = new GasStatData(0, 0, 0, 0, 0, 0, win_inflight, win_buffer);
        auto* cr = new StandardMem::CustomResp((StandardMem::Request::id_t)0, s, 0, 0, 0);
        (*upstream_handler_)(cr);
    }
    if (!ctrl_enable_) { resetWindowMetrics_(); return; }
    // Compute window metrics
    double payload_mib = (win_payload_bytes_>0) ? ((double)win_payload_bytes_ / (1024.0*1024.0)) : 0.0;
    double reqs_per_mib = (payload_mib>0.0 && win_bursts_>0) ? ((double)win_bursts_/payload_mib) : 0.0;
    double avg_seg_lat = (win_seg_count_>0) ? ((double)win_seg_latency_sum_ns_/ (double)win_seg_count_) : 0.0;
    // Initialize baseline after first full window
    ctrl_windows_seen_++;
    if (ctrl_windows_seen_ == 1) {
        ctrl_last_score_ = (reqs_per_mib>0)? reqs_per_mib : 1e12;
        ctrl_last_lat_ = avg_seg_lat;
        ctrl_curr_cfg_.k = gap_k_bytes_;
        ctrl_curr_cfg_.rowwin_bytes = row_window_bytes_;
        ctrl_curr_cfg_.timeout_ns = row_window_timeout_ns_;
        resetWindowMetrics_();
        return;
    }
    auto frac_improve = [](double base, double now){ return (base>0) ? ((base - now)/base) : 0.0; };
    bool lat_ok = (avg_seg_lat <= 0.0) || (ctrl_last_lat_<=0.0) || (avg_seg_lat <= ctrl_last_lat_ + (double)ctrl_lat_tol_ns_);
    if (ctrl_probe_active_) {
        // Decide adoption
        double imp_reqs = frac_improve(ctrl_last_score_, reqs_per_mib);
        double old_burst = (win_bursts_>0) ? ((double)win_payload_bytes_/(double)win_bursts_) : 0.0;
        // Allow adoption if reqs_per_mib improves or avg_burst increases, while latency is acceptable
        bool adopt = ((imp_reqs >= ctrl_eps_reqs_) || (old_burst > 0)) && lat_ok;
        if (adopt) {
            ctrl_curr_cfg_ = ctrl_probe_cfg_;
            ctrl_last_score_ = (reqs_per_mib>0)? reqs_per_mib : ctrl_last_score_;
            ctrl_last_lat_ = (avg_seg_lat>0)? avg_seg_lat : ctrl_last_lat_;
            if (stat_ctrl_adopts_) stat_ctrl_adopts_->addData(1);
            ctrl_cooldown_left_ = ctrl_cooldown_N_;
        } else {
            // revert
            applyCtrlConfig_(ctrl_curr_cfg_.k, ctrl_curr_cfg_.rowwin_bytes, ctrl_curr_cfg_.timeout_ns);
            if (stat_ctrl_reverts_) stat_ctrl_reverts_->addData(1);
            ctrl_cooldown_left_ = ctrl_cooldown_N_;
        }
        ctrl_probe_active_ = false;
    } else if (ctrl_cooldown_left_>0) {
        ctrl_cooldown_left_--;
    } else if (ctrl_probe_every_N_>0 && (ctrl_windows_seen_ % ctrl_probe_every_N_ == 0)) {
        // Choose next neighbor along current dimension
        auto idxOf=[&](const std::vector<uint64_t>&v, uint64_t x){ for(size_t i=0;i<v.size();++i) if (v[i]==x) return (int)i; return -1; };
        if (ctrl_dim_cursor_==0 && !ctrl_k_list_.empty()){
            int idx = idxOf(ctrl_k_list_, ctrl_curr_cfg_.k); if (idx<0) idx=0; int nxt = (idx+1) % (int)ctrl_k_list_.size();
            ctrl_probe_cfg_ = { ctrl_k_list_[nxt], ctrl_curr_cfg_.rowwin_bytes, ctrl_curr_cfg_.timeout_ns };
        } else if (ctrl_dim_cursor_==1 && !ctrl_rowwin_list_.empty()){
            int idx = idxOf(ctrl_rowwin_list_, ctrl_curr_cfg_.rowwin_bytes); if (idx<0) idx=0; int nxt = (idx+1) % (int)ctrl_rowwin_list_.size();
            ctrl_probe_cfg_ = { ctrl_curr_cfg_.k, ctrl_rowwin_list_[nxt], ctrl_curr_cfg_.timeout_ns };
        } else if (ctrl_dim_cursor_==2 && !ctrl_timeout_list_.empty()){
            int idx = idxOf(ctrl_timeout_list_, ctrl_curr_cfg_.timeout_ns); if (idx<0) idx=0; int nxt = (idx+1) % (int)ctrl_timeout_list_.size();
            ctrl_probe_cfg_ = { ctrl_curr_cfg_.k, ctrl_curr_cfg_.rowwin_bytes, ctrl_timeout_list_[nxt] };
        } else {
            resetWindowMetrics_();
            return;
        }
        applyCtrlConfig_(ctrl_probe_cfg_.k, ctrl_probe_cfg_.rowwin_bytes, ctrl_probe_cfg_.timeout_ns);
        ctrl_probe_active_ = true;
        ctrl_dim_cursor_ = (ctrl_dim_cursor_+1) % 3;
        if (stat_ctrl_probes_) stat_ctrl_probes_->addData(1);
    }
    // Reset metrics
    resetWindowMetrics_();
}
