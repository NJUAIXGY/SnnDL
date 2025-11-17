// GatherBufferIF.cc: GAS-aware StandardMem front-end with scratchpad buffering

#include <sst/core/sst_config.h>
#include "GatherBufferIF.h"

#include <algorithm>
#include <cstring>

using namespace SST;
using namespace SST::Interfaces;
using namespace SST::SnnDL;

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
    sram_access_ns_ = params.find<uint32_t>("sram_access_ns", 0);
    max_inflight_reads_ = params.find<uint32_t>("max_inflight_reads", 128);
    tail_wait_timeout_ns_ = params.find<uint64_t>("tail_wait_timeout_ns", 0);
    allow_apply_miss_read_ = params.find<int>("allow_apply_miss_read", 0) != 0;
    flush_after_scatter_ = params.find<int>("flush_after_scatter", 1) != 0;
    strict_mode_ = params.find<int>("strict_mode", 1) != 0;
    window_auto_ = params.find<int>("window_auto", 0) != 0;
    win_cyc_gather_ = params.find<uint64_t>("window_cycles_gather", 0);
    win_cyc_apply_  = params.find<uint64_t>("window_cycles_apply", 0);
    win_cyc_scatter_= params.find<uint64_t>("window_cycles_scatter", 0);
    clock_freq_ = params.find<std::string>("clock", "1GHz");

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
    last_stage_change_ns_ = getCurrentSimTimeNano();
}

GatherBufferIF::~GatherBufferIF() {}

void GatherBufferIF::init(unsigned int phase) {
    backend_->init(phase);
    if (phase == 0 && window_auto_) {
        // Register clock for time-driven windows
        registerClock(clock_freq_, new Clock::Handler2<GatherBufferIF, &GatherBufferIF::clockTick>(this));
        stage_ = Stage::Gather; stage_counter_ = 0; current_gather_id_++;
        end_gather_seen_ = false;
    }
}

void GatherBufferIF::setup() { backend_->setup(); }

void GatherBufferIF::finish() { backend_->finish(); }

void GatherBufferIF::sendUntimedData(Request* req) { backend_->sendUntimedData(req); }

StandardMem::Request* GatherBufferIF::recvUntimedData() { return backend_->recvUntimedData(); }

StandardMem::Addr GatherBufferIF::getLineSize() { return backend_ ? backend_->getLineSize() : 64; }

void GatherBufferIF::setMemoryMappedAddressRegion(Addr start, Addr size) {
    if (backend_) backend_->setMemoryMappedAddressRegion(start, size);
}

void GatherBufferIF::send(Request* req) {
    // Control-plane: CustomReq
    if (auto* cr = dynamic_cast<StandardMem::CustomReq*>(req)) {
        auto* data = dynamic_cast<GasOpData*>(&cr->getData());
        if (data) {
            if (window_auto_) { delete req; return; } // ignore external control in window mode
            switch (data->op) {
                case GasOp::BeginGather:
                    stage_ = Stage::Gather; end_gather_seen_ = false; current_gather_id_++;
                    granules_.clear(); required_set_.clear();
                    // 将非Gather阶段缓存的读导入当前Gather窗口
                    if (!queued_non_gather_reads_.empty()) {
                        for (auto* r : queued_non_gather_reads_) {
                            pending_up_reads_[r->getID()] = r;
                    uint64_t gsz = granuleSize();
                    bool use_row = (merge_==Merge::Row) || (merge_==Merge::Auto && row_bytes_guess_ > gsz);
                    uint64_t base = (merge_==Merge::Cacheline || (merge_==Merge::Auto && !use_row)) ? alignDown(r->pAddr, gsz)
                                           : (use_row ? alignDown(r->pAddr, row_bytes_guess_) : r->pAddr);
                    uint32_t sz = (merge_==Merge::None) ? r->size : (use_row ? row_bytes_guess_ : gsz);
                            uint64_t key = (base << 32) ^ (uint64_t)sz;
                            auto& g = granules_[key];
                            if (g.subs.empty()) { g.base = base; g.size = sz; required_set_.insert(key); if (stat_coalesce_granule_size_) stat_coalesce_granule_size_->addData((uint64_t)sz); }
                            uint64_t off = (r->pAddr >= base) ? (r->pAddr - base) : 0;
                            g.subs.push_back({r->getID(), off, (uint32_t)r->size, r});
                            if (!g.issued) { issueGranule_(key, g); }
                            if (stat_up_reads_) stat_up_reads_->addData(1);
                        }
                        queued_non_gather_reads_.clear();
                    }
                    break;
                case GasOp::EndGather:
                    end_gather_seen_ = true; tail_wait_start_ns_ = getCurrentSimTimeNano();
                    maybeEnterApply_();
                    break;
                case GasOp::BeginApply:
                    stage_ = Stage::Apply; emitApplyResponses_(); break;
                case GasOp::EndApply:
                    stage_ = Stage::Scatter; if (flush_after_scatter_) doFlush_(); stage_ = Stage::Idle; break;
                case GasOp::BeginScatter:
                    stage_ = Stage::Scatter; break;
                case GasOp::EndScatter:
                    if (flush_after_scatter_) doFlush_(); stage_ = Stage::Idle; break;
                case GasOp::FlushSRAM:
                    doFlush_(); break;
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
        if (stage_ == Stage::Gather) {
            pending_up_reads_[rd->getID()] = rd;
            uint64_t gsz = granuleSize();
            bool use_row = (merge_==Merge::Row) || (merge_==Merge::Auto && row_bytes_guess_ > gsz);
            uint64_t base = (merge_==Merge::Cacheline || (merge_==Merge::Auto && !use_row)) ? alignDown(rd->pAddr, gsz)
                           : (use_row ? alignDown(rd->pAddr, row_bytes_guess_) : rd->pAddr);
            uint32_t sz = (merge_==Merge::None) ? rd->size : (use_row ? row_bytes_guess_ : gsz);
            uint64_t key = (base << 32) ^ (uint64_t)sz; // compact key
            auto& g = granules_[key];
            if (g.subs.empty()) { g.base = base; g.size = sz; required_set_.insert(key); }
            if (stat_coalesce_granule_size_ && g.subs.empty()) stat_coalesce_granule_size_->addData((uint64_t)sz);
            // record sub-req offset
            uint64_t off = (rd->pAddr >= base) ? (rd->pAddr - base) : 0;
            g.subs.push_back({rd->getID(), off, (uint32_t)rd->size, rd});
            if (!g.issued) { issueGranule_(key, g); }
            if (stat_up_reads_) stat_up_reads_->addData(1);
            return; // keep rd queued; deletion deferred after we emit ReadResp
        } else if (stage_ == Stage::Apply || stage_ == Stage::Scatter || stage_ == Stage::Idle) {
            if (strict_mode_) {
                // 缓存到下一次Gather窗口再处理
                queued_non_gather_reads_.push_back(rd);
                return;
            } else {
                backend_->send(req); return;
            }
        }
    }

    // Writes/flush: pass-through; optionally invalidate overlapping SRAM
    if (auto* wr = dynamic_cast<StandardMem::Write*>(req)) {
        // Invalidate overlapping blocks for simplicity
        if (!sram_blocks_.empty()) {
            std::vector<uint64_t> toErase;
            for (auto& kv : sram_blocks_) {
                uint64_t base = kv.first >> 32; uint32_t sz = (uint32_t)(kv.first & 0xffffffffu);
                if (!(wr->pAddr + wr->size <= base || wr->pAddr >= base + sz)) toErase.push_back(kv.first);
            }
            for (auto k : toErase) { sram_blocks_.erase(k); }
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

void GatherBufferIF::issueGranule_(uint64_t key, Granule& g) {
    // respect max_inflight (best-effort: simple check)
    if (inflight_down_.size() >= max_inflight_reads_) return; // defer; will retry on next resp
    auto* rd = new StandardMem::Read(g.base, g.size);
    g.down_id = rd->getID();
    g.issued = true;
    inflight_down_[g.down_id] = key;
    backend_->send(rd);
}

void GatherBufferIF::onDownstreamResp_(Request* r) {
    // Only expecting ReadResp for now
    if (auto* rr = dynamic_cast<StandardMem::ReadResp*>(r)) {
        auto it = inflight_down_.find(rr->getID());
        if (it != inflight_down_.end()) {
            uint64_t key = it->second; inflight_down_.erase(it);
            auto git = granules_.find(key);
            if (git != granules_.end()) {
                // store into SRAM
                ensureCapacity_(git->second.size);
                auto& buf = sram_blocks_[key];
                buf.resize(git->second.size);
                if (!rr->data.empty()) {
                    std::memcpy(buf.data(), rr->data.data(), std::min<size_t>(buf.size(), rr->data.size()));
                }
                touchLRU_(key);
                git->second.ready = true;
                if (stat_unique_reads_) stat_unique_reads_->addData(1);
                if (stat_unique_bytes_) stat_unique_bytes_->addData(git->second.size);
                if (stat_buffer_occupancy_bytes_) stat_buffer_occupancy_bytes_->addData(bytes_in_sram_);
            }
        }
        delete r;
        // If all granules returned and EndGather seen, move to APPLY
        maybeEnterApply_();
        // Re-issue deferred granules if any and window still in Gather
        if (stage_ == Stage::Gather) {
            for (auto& kv : granules_) {
                if (!kv.second.issued) { issueGranule_(kv.first, kv.second); }
            }
        }
        return;
    }

    // Pass-through for other responses
    if (upstream_handler_) (*upstream_handler_)(r);
}

void GatherBufferIF::maybeEnterApply_() {
    if (stage_ != Stage::Gather || !end_gather_seen_) return;
    // Check if all required granules are ready
    for (auto key : required_set_) {
        auto it = granules_.find(key);
        if (it == granules_.end() || !it->second.ready) return; // still waiting
    }
    // Enter APPLY; optionally hold for window cycles; record tail-wait
    stage_ = Stage::Apply; stage_counter_ = 0; apply_pending_emit_ = true;
    if (tail_wait_start_ns_ && stat_tail_wait_ns_) {
        uint64_t now = getCurrentSimTimeNano();
        if (now >= tail_wait_start_ns_) stat_tail_wait_ns_->addData(now - tail_wait_start_ns_);
        tail_wait_start_ns_ = 0;
    }
    if (!window_auto_ || win_cyc_apply_ == 0) {
        emitApplyResponses_();
        stage_ = Stage::Scatter; stage_counter_ = 0;
        if (flush_after_scatter_) doFlush_();
        stage_ = window_auto_ ? Stage::Gather : Stage::Idle; // in window mode, loop to next Gather
        if (window_auto_) { end_gather_seen_ = false; granules_.clear(); required_set_.clear(); current_gather_id_++; }
    }
}

void GatherBufferIF::emitApplyResponses_() {
    // For each upstream Read queued in this gather, emit a ReadResp from SRAM
    for (auto& kvg : granules_) {
        uint64_t key = kvg.first; auto& g = kvg.second;
        if (!g.ready) continue; // should not happen
        auto bit = sram_blocks_.find(key);
        if (bit == sram_blocks_.end()) continue;
        const auto& blk = bit->second;
        for (auto& s : g.subs) {
            auto* resp = new StandardMem::ReadResp(s.up_read, std::vector<uint8_t>(s.size));
            // Fill payload
            if (s.offset + s.size <= blk.size()) {
                std::memcpy(resp->data.data(), blk.data() + s.offset, s.size);
            }
            if (upstream_handler_) (*upstream_handler_)(resp);
            // delete original upstream read now that it has been answered
            auto itup = pending_up_reads_.find(s.up_id);
            if (itup != pending_up_reads_.end()) { delete itup->second; pending_up_reads_.erase(itup); }
        }
    }
    // Clear per-gather structures except cached blocks
    granules_.clear();
    required_set_.clear();
    end_gather_seen_ = false;
}

void GatherBufferIF::doFlush_() {
    sram_blocks_.clear(); lru_list_.clear(); bytes_in_sram_ = 0;
}

void GatherBufferIF::touchLRU_(uint64_t key) {
    auto it = std::find(lru_list_.begin(), lru_list_.end(), key);
    if (it != lru_list_.end()) lru_list_.erase(it);
    lru_list_.push_back(key);
}

void GatherBufferIF::ensureCapacity_(uint64_t need) {
    while (bytes_in_sram_ + need > sram_bytes_ && !lru_list_.empty()) {
        uint64_t k = lru_list_.front(); lru_list_.pop_front();
        auto it = sram_blocks_.find(k);
        if (it != sram_blocks_.end()) { bytes_in_sram_ -= it->second.size(); sram_blocks_.erase(it); if (stat_evictions_) stat_evictions_->addData(1); }
    }
    bytes_in_sram_ += need;
}

bool GatherBufferIF::clockTick(Cycle_t) {
    if (!window_auto_) return false;
    stage_counter_++;
    // accumulate per-stage cycles
    if (stage_ == Stage::Gather && stat_stage_cycles_g_) stat_stage_cycles_g_->addData(1);
    else if (stage_ == Stage::Apply && stat_stage_cycles_a_) stat_stage_cycles_a_->addData(1);
    else if (stage_ == Stage::Scatter && stat_stage_cycles_s_) stat_stage_cycles_s_->addData(1);
    if (stage_ == Stage::Gather) {
        if (win_cyc_gather_ && stage_counter_ >= win_cyc_gather_) {
            end_gather_seen_ = true;
            maybeEnterApply_();
            // in immediate path stage_ may now be Idle or Scatter; optional wait handled there
        }
    } else if (stage_ == Stage::Apply) {
        if (win_cyc_apply_ && stage_counter_ >= win_cyc_apply_) {
            if (apply_pending_emit_) emitApplyResponses_();
            stage_ = Stage::Scatter; stage_counter_ = 0;
            if (flush_after_scatter_) doFlush_();
        }
    } else if (stage_ == Stage::Scatter) {
        if (win_cyc_scatter_ && stage_counter_ >= win_cyc_scatter_) {
            // start new gather window
            stage_ = Stage::Gather; stage_counter_ = 0; end_gather_seen_ = false; current_gather_id_++;
            granules_.clear(); required_set_.clear();
            // 导入非Gather缓存
            if (!queued_non_gather_reads_.empty()) {
                for (auto* r : queued_non_gather_reads_) {
                    pending_up_reads_[r->getID()] = r;
                    uint64_t gsz = granuleSize();
                    uint64_t base = (merge_==Merge::Cacheline || merge_==Merge::Auto) ? alignDown(r->pAddr, gsz)
                                   : (merge_==Merge::Row ? alignDown(r->pAddr, row_bytes_guess_) : r->pAddr);
                    uint32_t sz = (merge_==Merge::None) ? r->size : (merge_==Merge::Row ? row_bytes_guess_ : gsz);
                    uint64_t key = (base << 32) ^ (uint64_t)sz;
                    auto& g = granules_[key];
                    if (g.subs.empty()) { g.base = base; g.size = sz; required_set_.insert(key); if (stat_coalesce_granule_size_) stat_coalesce_granule_size_->addData((uint64_t)sz); }
                    uint64_t off = (r->pAddr >= base) ? (r->pAddr - base) : 0;
                    g.subs.push_back({r->getID(), off, (uint32_t)r->size, r});
                    if (!g.issued) { issueGranule_(key, g); }
                    if (stat_up_reads_) stat_up_reads_->addData(1);
                }
                queued_non_gather_reads_.clear();
            }
        }
    }
    // track inflight peak
    if (stat_inflight_peak_) stat_inflight_peak_->addData((uint64_t)inflight_down_.size());
    return false; // continue ticking
}
