#include "SnnPESubComponent.h"
#include "SnnPESubComponent_impl.h"
#include "IPeAggregation.h"
#include "synapse/weights/WeightAccessor.h"
#include "synapse/weights/WeightMemorySubsystem.h"
#include "synapse/weights/SnnBcsrWeightManager.h"

#include <algorithm>
#include <cstdint>
#include <fstream>

using namespace SST::SnnDL;

// ==== Statistics reporter (Phase5.4: moved into Impl) ====
void SnnPESubComponent::Impl::reportMemoryIssue(size_t bytes, bool count_weight_read) const {
    if (!core) return;
    const uint64_t inflight = static_cast<uint64_t>(core->pendingMemSize_()) + 1ULL;
    if (auto* pe = core->parent_pe_cached_) {
        pe->accumulateIssueStats(static_cast<uint64_t>(bytes), inflight);
    }
    if (core->stat_mem_req_size_bytes_) {
        core->stat_mem_req_size_bytes_->addData(static_cast<uint64_t>(bytes));
    }
    if (core->stat_mem_outstanding_at_issue_) {
        core->stat_mem_outstanding_at_issue_->addData(inflight);
    }
    if (core->stat_memory_requests_) core->stat_memory_requests_->addData(1);
    core->count_memory_requests_++;
    if (count_weight_read && core->stat_weight_read_requests_) {
        core->stat_weight_read_requests_->addData(1);
    }
}

void SnnPESubComponent::Impl::reportApplyScatter(uint64_t acc_updates, uint64_t posts_touched,
                                uint64_t spikes_emitted, uint64_t hwm_bytes,
                                uint64_t spill_records, uint64_t spilled_bytes) const {
    if (!core) return;
    if (auto* pe = core->parent_pe_cached_) {
        pe->accumulateApplyScatterStats(acc_updates, posts_touched, spikes_emitted,
                                        hwm_bytes, spill_records, spilled_bytes);
    }
}

void SnnPESubComponent::Impl::reportWindowSpikes(uint32_t seq, uint64_t spikes_emitted) const {
    if (!core || spikes_emitted == 0) return;
    if (auto* pe = core->parent_pe_cached_) {
        pe->accumulateWindowSpikes(seq, spikes_emitted);
    }
}

void SnnPESubComponent::Impl::reportCacheAccess(bool hit) const {
    if (!core) return;
    if (hit) {
        if (core->stat_weight_cache_hits_) core->stat_weight_cache_hits_->addData(1);
        core->count_cache_hits_++;
    } else {
        if (core->stat_weight_cache_misses_) core->stat_weight_cache_misses_->addData(1);
        core->count_cache_misses_++;
    }
}

void SnnPESubComponent::Impl::updatePendingPeak(uint32_t outstanding) const {
    if (!core) return;
    if (outstanding > core->pending_reqs_peak_) {
        core->pending_reqs_peak_ = outstanding;
        if (core->stat_pending_reqs_peak_) core->stat_pending_reqs_peak_->addData(outstanding);
    }
}

// ==== Window-read issue helpers (control-plane) ====
bool SnnPESubComponent::canIssueMoreReads_() const {
    if (windowStateCanIssue_()) return true;
    if (window_read_debug_ && output_) {
        output_->verbose(CALL_INFO, 0, 0,
            "[diag-edge-loop] core=%d budget/ostd stop issued=%u outstanding=%u budget=%u limit=%u\n",
            core_id_, windowStateIssued_(), windowStateOutstanding_(), window_read_budget_, max_outstanding_requests_);
    }
    return false;
}

void SnnPESubComponent::issueFromEdges_() {
    if (!(apply_acc_enable_ && gas_window_mode_)) return;
    if (weight_mem_subsystem_) {
        weight_mem_subsystem_->issueFromEdges();
        return;
    }
#if 0
    // 仅在真正要发起权重读取前检查 loader 是否就绪
    if (!ensureLoaderReady_()) {
        if (window_read_debug_ && output_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-window-read] BeginApply: core=%u window=%u defer edge issue (loader not ready)\n",
                core_id_, curr_stage_seq_);
        }
        return;
    }
    if (use_bcsr_ && !bcsr_weights_->isRowptrReady()) {
        if (window_read_debug_ && output_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-window-read] BeginApply: core=%u window=%u defer edge issue (BCSR rowptr not ready)\n",
                core_id_, curr_stage_seq_);
        }
        return;
    }

    while (canIssueMoreReads_()) {
        uint64_t key = 0;
        uint32_t count = 0;
        if (!edge_collector_.nextPrev(key, count)) {
            if (window_read_debug_ && output_) {
                output_->verbose(CALL_INFO, 0, 0,
                    "[diag-edge-loop] core=%d window=%u no-prev size=%zu iter=%zu",
                    core_id_, curr_stage_seq_,
                    edge_collector_.prevSize(), edge_collector_.prevIter());
            }
            break;
        }
        uint32_t post_local = static_cast<uint32_t>(key >> 32);
        uint32_t pre_global = static_cast<uint32_t>(key & 0xffffffffu);
        if (post_local >= num_neurons_) continue;

        if (use_bcsr_) {
            if (bcsr_force_file_read_) {
                // 诊断路径：直接从权重文件读取块，避免内存可见性/一致性导致的错误
                uint32_t br = (bcsr_br_>0? bcsr_br_:1);
                uint32_t bc = (bcsr_bc_>0? bcsr_bc_:16);
                uint32_t block_row = post_local / br;
                uint32_t intra_row = post_local % br;
                uint32_t blk_col = (bc? (pre_global / bc) : 0);
                uint32_t intra_col = (bc? (pre_global % bc) : 0);
                float resolved = 0.0f;
                do {
                    const auto& rowptr = bcsr_weights_->rowptrHost();
                    if (block_row + 1 > rowptr.size()) break;
                    uint32_t start = rowptr[block_row];
                    uint32_t end   = (block_row + 1 < rowptr.size() ? rowptr[block_row+1] : start);
                    if (end <= start) break;
                    std::string bin_path = resolveWeightTemplate(node_id_, core_id_);
                    if (bin_path.empty()) break;
                    uint32_t rows=0, colsN=0, brM=0, bcM=0, idxB=0, valB=0, totalB=0;
                    uint64_t rp_off=0, ci_off=0, bd_off=0, id_off=0;
                    std::string meta_path = bin_path + ".meta.json";
                    if (!parseBcsrMeta(meta_path, rows, colsN, brM, bcM, idxB, valB, rp_off, ci_off, bd_off, id_off, totalB)) break;
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
            if (readresp_zero_fallback_ && resolved == 0.0f) resolved = init_default_weight_;
            accUpdate_(post_local, resolved * static_cast<float>(count));
            diagEdgeWeight_("bcsr-file", post_local, pre_global, resolved, count);
        } else {
            windowStateNoteIssue_();
            uint32_t pre_capture = pre_global;
            auto cb = [this, post_local, count, pre_capture](float w) {
                float resolved = w;
                if (readresp_zero_fallback_ && resolved == 0.0f) resolved = init_default_weight_;
                accUpdate_(post_local, resolved * static_cast<float>(count));
                diagEdgeWeight_("bcsr", post_local, pre_capture, resolved, count);
                windowStateNoteComplete_();
                issueFromEdges_();
            };
            bool issued = false;
            if (compute_core_) issued = compute_core_->requestWeightBCSR(pre_global, post_local, cb);
            if (!issued) requestWeightBCSR(pre_global, post_local, cb);
            }
            continue;
        }

        uint32_t req_pre = 0;
        uint32_t req_post = 0;
        uint64_t cache_key = 0;
        if (!weight_accessor_ || !weight_accessor_->resolve(pre_global, post_local, req_pre, req_post, cache_key)) {
            if (window_read_debug_ && output_) {
                output_->verbose(CALL_INFO, 0, 0,
                    "[diag-edge-resolve] core=%d window=%u bad-edge pre=%u post=%u",
                    core_id_, curr_stage_seq_, pre_global, post_local);
            }
            continue;
        }

        float cached = 0.0f;
        bool cache_hit = false;
        if (compute_core_) cache_hit = compute_core_->weightCacheTryGet(cache_key, cached);
        else cache_hit = weightCacheTryGet_(cache_key, cached);
        if (cache_hit) {
            if (readresp_zero_fallback_ && cached == 0.0f) cached = init_default_weight_;
            accUpdate_(post_local, cached * static_cast<float>(count));
            diagEdgeWeight_("cache", post_local, pre_global, cached, count);
            stats_reporter_.reportCacheAccess(true);
            continue;
        }

        stats_reporter_.reportCacheAccess(false);
        windowStateNoteIssue_();
        uint32_t pre_capture = pre_global;
        auto miss_cb = [this, post_local, count, cache_key, pre_capture](float w) {
            float resolved = w;
            if (readresp_zero_fallback_ && resolved == 0.0f) resolved = init_default_weight_;
            if (compute_core_) compute_core_->weightCacheStore(cache_key, resolved);
            else weightCacheStore_(cache_key, resolved);
            accUpdate_(post_local, resolved * static_cast<float>(count));
            diagEdgeWeight_("miss", post_local, pre_capture, resolved, count);
            windowStateNoteComplete_();
            issueFromEdges_();
        };
        bool issued = false;
        if (compute_core_) issued = compute_core_->requestWeight(req_pre, req_post, miss_cb);
        if (!issued) requestWeight(req_pre, req_post, miss_cb);
    }
#endif
}

void SnnPESubComponent::issueFromSets_(
    const std::vector<uint32_t>* posts_to_use,
    const std::unordered_set<uint32_t>* pres_to_use) {
    if (!posts_to_use || !pres_to_use) return;
    if (weight_mem_subsystem_) {
        weight_mem_subsystem_->issueFromSets(posts_to_use, pres_to_use);
    }
}

void SnnPESubComponent::issueFallbackReadsIfNeeded_(bool strict_gas_active) {
    if (!window_read_enable_ || !enable_weight_fetch_ || !ensureMemoryReady_()) return;
    if (!weight_mem_subsystem_) return;
    bool need_sets = false;
    if (strict_gas_active) {
        need_sets = weight_mem_subsystem_->edgesPrevEmpty();
    } else {
        need_sets = true;
    }
    if (!need_sets) return;

    bool have_posts_prev = weight_mem_subsystem_->postsPrevSize() > 0;
    bool have_pres_prev  = weight_mem_subsystem_->presPrevSize() > 0;
    bool have_posts_curr = weight_mem_subsystem_->postsCurrSize() > 0;
    bool have_pres_curr  = weight_mem_subsystem_->presCurrSize() > 0;

    bool use_fallback = (!have_posts_prev || !have_pres_prev) && (have_posts_curr && have_pres_curr);
    if (window_read_debug_ && output_) {
        output_->verbose(CALL_INFO, 0, 0,
            "[diag-window-read] BeginApply: core=%u window=%u prev(posts=%u pre=%u) curr(posts=%u pre=%u) budget=%u max_out=%u fallback=%d\n",
            core_id_,
            curr_stage_seq_,
            static_cast<uint32_t>(weight_mem_subsystem_->postsPrevSize()),
            static_cast<uint32_t>(weight_mem_subsystem_->presPrevSize()),
            static_cast<uint32_t>(weight_mem_subsystem_->postsCurrSize()),
            static_cast<uint32_t>(weight_mem_subsystem_->presCurrSize()),
            window_read_budget_,
            max_outstanding_requests_,
            use_fallback ? 1 : 0);
    }

    if ((!have_posts_prev && !have_posts_curr) || (!have_pres_prev && !have_pres_curr)) {
        if (window_read_debug_ && output_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-window-read] BeginApply: core=%u window=%u skip read (both windows empty)\n",
                core_id_, curr_stage_seq_);
        }
        return;
    }

    const std::vector<uint32_t>* posts = use_fallback ? &weight_mem_subsystem_->postsCurr()
                                                      : &weight_mem_subsystem_->postsPrev();
    const std::unordered_set<uint32_t>* pres = use_fallback ? &weight_mem_subsystem_->presCurr()
                                                            : &weight_mem_subsystem_->presPrev();
    if (use_fallback && window_read_debug_ && output_) {
        output_->verbose(CALL_INFO, 0, 0,
            "[diag-window-read] BeginApply: core=%u window=%u FALLBACK to current window (prev empty, curr has data)\n",
            core_id_, curr_stage_seq_);
    }
    windowStateBegin_();
    issueFromSets_(posts, pres);
}

void SnnPESubComponent::issueFromSetsBcsr_(
    const std::vector<uint32_t>* posts_to_use,
    const std::unordered_set<uint32_t>* pres_to_use) {
    if (!posts_to_use || !pres_to_use) return;
    if (weight_mem_subsystem_) {
        weight_mem_subsystem_->issueFromSetsBcsr(posts_to_use, pres_to_use);
    }
}
