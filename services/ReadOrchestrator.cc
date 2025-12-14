// -*- c++ -*-
//
// ReadOrchestrator: window-read issue logic extracted from SnnPESubComponent.

#include "ReadOrchestrator.h"
#include "SnnPESubComponent.h"

#include <algorithm>
#include <cstdint>
#include <fstream>

using namespace SST::SnnDL;

bool DiagSink::enabled() const {
    return core && core->window_read_debug_ && core->output_;
}

SST::Output* DiagSink::out() const {
    return core ? core->output_ : nullptr;
}

void ReadOrchestrator::issueFromEdges() {
    if (!core || !(core->apply_acc_enable_ && core->gas_window_mode_)) return;
    // 仅在真正要发起权重读取前检查 loader 是否就绪
    if (!core->ensureLoaderReady_()) {
        diag_.log(1, "[diag-window-read] BeginApply: core=%u window=%u defer edge issue (loader not ready)\n",
            core->core_id_, core->curr_stage_seq_);
        return;
    }
    if (core->use_bcsr_ && !core->bcsr_weights_.isRowptrReady()) {
        diag_.log(1, "[diag-window-read] BeginApply: core=%u window=%u defer edge issue (BCSR rowptr not ready)\n",
            core->core_id_, core->curr_stage_seq_);
        return;
    }
    while (canIssueMoreReads_()) {
        uint64_t key = 0;
        uint32_t count = 0;
        if (!core->edge_collector_.nextPrev(key, count)) {
            diag_.log(0, "[diag-edge-loop] core=%d window=%u no-prev size=%zu iter=%zu",
                core->core_id_, core->curr_stage_seq_,
                core->edge_collector_.prevSize(), core->edge_collector_.prevIter());
            break;
        }
        uint32_t post_local = static_cast<uint32_t>(key >> 32);
        uint32_t pre_global = static_cast<uint32_t>(key & 0xffffffffu);
        if (post_local >= core->num_neurons_) continue;

        if (core->use_bcsr_) {
            if (core->bcsr_force_file_read_) {
                // 诊断路径：直接从权重文件读取块，避免内存可见性/一致性导致的错误
                uint32_t br = (core->bcsr_br_>0? core->bcsr_br_:1);
                uint32_t bc = (core->bcsr_bc_>0? core->bcsr_bc_:16);
                uint32_t block_row = post_local / br;
                uint32_t intra_row = post_local % br;
                uint32_t blk_col = (bc? (pre_global / bc) : 0);
                uint32_t intra_col = (bc? (pre_global % bc) : 0);
                float resolved = 0.0f;
                do {
                    const auto& rowptr = core->bcsr_weights_.rowptrHost();
                    if (block_row + 1 > rowptr.size()) break;
                    uint32_t start = rowptr[block_row];
                    uint32_t end   = (block_row + 1 < rowptr.size() ? rowptr[block_row+1] : start);
                    if (end <= start) break;
                    std::string bin_path = core->resolveWeightTemplate(core->node_id_, core->core_id_);
                    if (bin_path.empty()) break;
                    uint32_t rows=0, colsN=0, brM=0, bcM=0, idxB=0, valB=0, totalB=0;
                    uint64_t rp_off=0, ci_off=0, bd_off=0, id_off=0;
                    std::string meta_path = bin_path + ".meta.json";
                    if (!core->parseBcsrMeta(meta_path, rows, colsN, brM, bcM, idxB, valB, rp_off, ci_off, bd_off, id_off, totalB)) break;
                    std::ifstream fin(bin_path, std::ios::binary);
                    if (!fin.good()) break;
                    int idx_in_row = -1;
                    for (uint32_t j=0; j < (end - start); ++j) {
                        fin.seekg(static_cast<std::streamoff>(ci_off + (size_t)(start + j) * idxB), std::ios::beg);
                        uint32_t colv = 0; if (idxB == 2) { uint16_t v=0; fin.read(reinterpret_cast<char*>(&v), 2); colv = v; } else { fin.read(reinterpret_cast<char*>(&colv), 4); }
                        if (!fin.good()) break;
                        if (colv == blk_col) { idx_in_row = (int)j; break; }
                    }
                    if (idx_in_row < 0) break;
                    size_t blk_bytes = (size_t)(brM?brM:br) * (size_t)(bcM?bcM:bc) * (size_t)valB;
                    fin.seekg(static_cast<std::streamoff>(bd_off + (uint64_t)(start + (uint32_t)idx_in_row) * blk_bytes), std::ios::beg);
                    std::vector<float> blk((brM?brM:br) * (bcM?bcM:bc), 0.0f);
                    if (valB == 4) fin.read(reinterpret_cast<char*>(blk.data()), blk_bytes);
                    if (!fin.good()) break;
                    uint32_t off = intra_row * (bcM?bcM:bc) + intra_col;
                    if (off < blk.size()) resolved = blk[off];
                } while(0);
                if (core->readresp_zero_fallback_ && resolved == 0.0f) resolved = core->init_default_weight_;
                core->accUpdate_(post_local, resolved * static_cast<float>(count));
                core->diagEdgeWeight_("bcsr-file", post_local, pre_global, resolved, count);
                // 继续发起下一条边
                issueFromEdges();
            } else {
                core->outstanding_requests_++;
                core->stats_reporter_.updatePendingPeak(core->outstanding_requests_);
                core->window_reads_issued_this_apply_++;
                uint32_t pre_capture = pre_global;
                auto cb = [this, post_local, count, pre_capture](float w) {
                    float resolved = w;
                    if (core->readresp_zero_fallback_ && resolved == 0.0f) resolved = core->init_default_weight_;
                    core->accUpdate_(post_local, resolved * static_cast<float>(count));
                    core->diagEdgeWeight_("bcsr", post_local, pre_capture, resolved, count);
                    if (core->outstanding_requests_ > 0) core->outstanding_requests_--;
                    issueFromEdges();
                };
                bool issued = false;
                if (core->compute_core_) issued = core->compute_core_->requestWeightBCSR(pre_global, post_local, cb);
                if (!issued) core->requestWeightBCSR(pre_global, post_local, cb);
            }
            continue;
        }

        uint32_t req_pre = 0;
        uint32_t req_post = 0;
        uint64_t cache_key = 0;
        if (!core->weight_accessor_.resolve(pre_global, post_local, req_pre, req_post, cache_key)) {
            diag_.log(0, "[diag-edge-resolve] core=%d window=%u bad-edge pre=%u post=%u",
                core->core_id_, core->curr_stage_seq_, pre_global, post_local);
            continue;
        }

        float cached = 0.0f;
        bool cache_hit = false;
        if (core->compute_core_) cache_hit = core->compute_core_->weightCacheTryGet(cache_key, cached);
        else cache_hit = core->weightCacheTryGet_(cache_key, cached);
        if (cache_hit) {
            if (core->readresp_zero_fallback_ && cached == 0.0f) cached = core->init_default_weight_;
            core->accUpdate_(post_local, cached * static_cast<float>(count));
            core->diagEdgeWeight_("cache", post_local, pre_global, cached, count);
            core->stats_reporter_.reportCacheAccess(true);
            continue;
        }

        core->stats_reporter_.reportCacheAccess(false);
        core->outstanding_requests_++;
        core->stats_reporter_.updatePendingPeak(core->outstanding_requests_);
        core->window_reads_issued_this_apply_++;
        uint32_t pre_capture = pre_global;
        auto miss_cb = [this, post_local, count, cache_key, pre_capture](float w) {
                float resolved = w;
                if (core->readresp_zero_fallback_ && resolved == 0.0f) resolved = core->init_default_weight_;
                if (core->compute_core_) core->compute_core_->weightCacheStore(cache_key, resolved);
                else core->weightCacheStore_(cache_key, resolved);
                core->accUpdate_(post_local, resolved * static_cast<float>(count));
                core->diagEdgeWeight_("miss", post_local, pre_capture, resolved, count);
                if (core->outstanding_requests_ > 0) core->outstanding_requests_--;
                issueFromEdges();
        };
        bool issued = false;
        if (core->compute_core_) issued = core->compute_core_->requestWeight(req_pre, req_post, miss_cb);
        if (!issued) core->requestWeight(req_pre, req_post, miss_cb);
    }
}

void ReadOrchestrator::issueFromSets(
    const std::vector<uint32_t>* posts_to_use,
    const std::unordered_set<uint32_t>* pres_to_use) {
    if (!core || !posts_to_use || !pres_to_use) return;
    if (core->use_bcsr_) {
        issueFromSetsBcsr(posts_to_use, pres_to_use);
        return;
    }
    uint32_t issued = 0;
    for (const auto& pre_g : *pres_to_use) {
        for (uint32_t post_l : *posts_to_use) {
            if (!canIssueMoreReads_()) break;
            uint32_t req_pre = 0;
            uint32_t req_post = 0;
            uint64_t cache_key = 0;
            if (!core->weight_accessor_.resolve(pre_g, post_l, req_pre, req_post, cache_key)) continue;
            core->stats_reporter_.reportCacheAccess(false);
            core->outstanding_requests_++;
            core->stats_reporter_.updatePendingPeak(core->outstanding_requests_);
            core->window_reads_issued_this_apply_++;
            issued++;
            auto cache_cb = [this, cache_key](float w){
                if (core->compute_core_) core->compute_core_->weightCacheStore(cache_key, w);
                else core->weightCacheStore_(cache_key, w);
                if (core->outstanding_requests_ > 0) core->outstanding_requests_--;
            };
            bool issued_req = false;
            if (core->compute_core_) issued_req = core->compute_core_->requestWeight(req_pre, req_post, cache_cb);
            if (!issued_req) core->requestWeight(req_pre, req_post, cache_cb);
        }
        if (!canIssueMoreReads_()) break;
    }
    logIssuedStats_(issued);
}

void ReadOrchestrator::issueFallbackReadsIfNeeded(bool strict_gas_active) {
    if (!core) return;
    if (!core->window_read_enable_ || !core->enable_weight_fetch_ || !core->memory_ || !core->memory_ready_) return;
    bool need_sets = false;
    if (strict_gas_active) {
        need_sets = core->edge_collector_.prevEmpty();
    } else {
        need_sets = true;
    }
    if (!need_sets) return;

    bool have_posts_prev = !core->posts_list_prev_window_.empty();
    bool have_pres_prev  = !core->active_pre_prev_window_.empty();
    bool have_posts_curr = !core->posts_list_window_.empty();
    bool have_pres_curr  = !core->active_pre_window_.empty();

    bool use_fallback = (!have_posts_prev || !have_pres_prev) && (have_posts_curr && have_pres_curr);
    logWindowReadSummary_(static_cast<uint32_t>(core->posts_list_prev_window_.size()),
                          static_cast<uint32_t>(core->active_pre_prev_window_.size()),
                          static_cast<uint32_t>(core->posts_list_window_.size()),
                          static_cast<uint32_t>(core->active_pre_window_.size()),
                          use_fallback);

    if ((!have_posts_prev && !have_posts_curr) || (!have_pres_prev && !have_pres_curr)) {
        diag_.log(1, "[diag-window-read] BeginApply: core=%u window=%u skip read (both windows empty)\n",
                  core->core_id_, core->curr_stage_seq_);
        return;
    }

    const std::vector<uint32_t>* posts = use_fallback ? &core->posts_list_window_ : &core->posts_list_prev_window_;
    const std::unordered_set<uint32_t>* pres = use_fallback ? &core->active_pre_window_ : &core->active_pre_prev_window_;
    if (use_fallback) {
        logFallbackSwitch_();
    }
    core->window_reads_issued_this_apply_ = 0;
    issueFromSets(posts, pres);
}

void ReadOrchestrator::issueFromSetsBcsr(
    const std::vector<uint32_t>* posts_to_use,
    const std::unordered_set<uint32_t>* pres_to_use) {
    if (!core || !posts_to_use || !pres_to_use) return;
    if (!core->bcsr_weights_.isRowptrReady()) {
        diag_.log(1, "[diag-window-read] BeginApply: core=%u window=%u skip set-priming (BCSR rowptr not ready)\n",
            core->core_id_, core->curr_stage_seq_);
        return;
    }
    uint32_t primed = 0;
    for (const auto& pre_g : *pres_to_use) {
        for (uint32_t post_l : *posts_to_use) {
            if (!canIssueMoreReads_()) break;
            core->stats_reporter_.reportCacheAccess(false);
            core->outstanding_requests_++;
            core->stats_reporter_.updatePendingPeak(core->outstanding_requests_);
            core->window_reads_issued_this_apply_++;
            primed++;
            core->requestWeightBCSR(pre_g, post_l, [this](float) {
                if (core->outstanding_requests_ > 0) core->outstanding_requests_--;
            });
        }
        if (!canIssueMoreReads_()) break;
    }
    diag_.log(1,
        "[diag-window-read] BCSR priming: core=%u window=%u issued=%u outstanding=%u\n",
        core->core_id_, core->curr_stage_seq_, primed,
        core->outstanding_requests_);
}

void ReadOrchestrator::logWindowReadSummary_(
    uint32_t posts_prev, uint32_t pres_prev,
    uint32_t posts_curr, uint32_t pres_curr,
    bool fallback) const {
    if (!core) return;
    diag_.log(1,
        "[diag-window-read] BeginApply: core=%u window=%u prev(posts=%u pre=%u) curr(posts=%u pre=%u) budget=%u max_out=%u fallback=%d\n",
        core->core_id_,
        core->curr_stage_seq_,
        posts_prev, pres_prev, posts_curr, pres_curr,
        core->window_read_budget_,
        core->max_outstanding_requests_,
        fallback ? 1 : 0);
}

void ReadOrchestrator::logFallbackSwitch_() const {
    if (!core) return;
    diag_.log(1,
        "[diag-window-read] BeginApply: core=%u window=%u FALLBACK to current window (prev empty, curr has data)\n",
        core->core_id_, core->curr_stage_seq_);
}

void ReadOrchestrator::logIssuedStats_(uint32_t issued) const {
    if (!core) return;
    diag_.log(1,
        "[diag-window-read] BeginApply: core=%u window=%u issued=%u (this window) outstanding_reqs=%u pending=%zu\n",
        core->core_id_, core->curr_stage_seq_, issued,
        core->outstanding_requests_,
        static_cast<size_t>(core->pending_memory_requests_.size()));
}

void ReadOrchestrator::logEdgeFetchStart(
    size_t prev_edges, uint32_t issued, uint32_t outstanding, uint32_t budget) const {
    if (!core) return;
    diag_.log(1,
        "[diag-edge-fetch] core=%d stage=%d prev_edges=%zu issued=%u outstanding=%u budget=%u\n",
        core->core_id_, static_cast<int>(core->gas_stage_), prev_edges,
        issued, outstanding, budget);
}

bool ReadOrchestrator::canIssueMoreReads_() const {
    if (!core) return false;
    if (core->window_read_budget_ && core->window_reads_issued_this_apply_ >= core->window_read_budget_) {
        diag_.log(0, "[diag-edge-loop] core=%d budget hit issued=%u\n", core->core_id_, core->window_reads_issued_this_apply_);
        return false;
    }
    if (core->outstanding_requests_ >= core->max_outstanding_requests_) {
        diag_.log(0, "[diag-edge-loop] core=%d outstanding=%u limit=%u\n",
            core->core_id_, core->outstanding_requests_, core->max_outstanding_requests_);
        return false;
    }
    return true;
}
