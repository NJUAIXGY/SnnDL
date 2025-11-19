// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// SnnPESubComponent.cc: SnnPE SubComponent版本实现文件
//

#include <sst/core/sst_config.h>
#include "SnnPESubComponent.h"
#include <fstream>
#include "GasCustomCmd.h"
#include "MultiCorePE.h"
#include "GatherBufferIF.h"

#include <sstream>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <cstdio>   // TEMP: debug file sink for acc-shadow verification
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <list>
#include <limits>
#include <iomanip>
#include <cctype>
#include <iterator>

using namespace SST;
using namespace SST::SnnDL;

// 诊断门控改为参数化：由 enable_extended_diagnostics_ 成员控制

bool SnnPESubComponent::BcsrLayout::validate(uint64_t base, Output* out, bool debug, uint32_t core_id, uint32_t node_id) const {
    const bool monotonic = (colidx_offset >= rowptr_offset) && (blockdata_offset >= colidx_offset);
    const bool aligned64 = ((rowptr_offset | colidx_offset | blockdata_offset | blockids_offset) & 0x3FULL) == 0;
    const uint64_t stride = per_core_stride;
    const uint64_t max_off = maxOffset();
    const bool stride_ok = (stride == 0) ? true : (max_off < stride);
    if (debug && out) {
        out->verbose(CALL_INFO, 0, 0,
            "[diag-bcsr-base] node=%u core=%u base=0x%lx rp=0x%lx ci=0x%lx bd=0x%lx ids=0x%lx stride=%" PRIu64 " stride_ok=%d align64=%d mono=%d\n",
            node_id, core_id,
            (unsigned long)base,
            (unsigned long)(base + rowptr_offset),
            (unsigned long)(base + colidx_offset),
            (unsigned long)(base + blockdata_offset),
            (unsigned long)(blockids_offset ? base + blockids_offset : 0),
            stride, stride_ok ? 1 : 0,
            aligned64 ? 1 : 0,
            monotonic ? 1 : 0);
        if (!aligned64 || !monotonic || !stride_ok) {
            out->verbose(CALL_INFO, 0, 0,
                "[diag-bcsr-base] offsets suspect (aligned64=%d monotonic=%d stride_ok=%d max_off=0x%llx stride=0x%llx)\n",
                aligned64 ? 1 : 0, monotonic ? 1 : 0, stride_ok ? 1 : 0,
                (unsigned long long)max_off, (unsigned long long)stride);
        }
    }
    return monotonic && aligned64;
}


// Lightweight logging helpers (file-local). Use consistent style across SnnDL.
#ifndef SNNDL_LOGPTR
#define SNNDL_LOGPTR(ptr, lvl, ...) do { if (ptr) (ptr)->verbose(CALL_INFO, (lvl), 0, __VA_ARGS__); } while(0)
#endif
#ifndef SNNDL_LOG
#define SNNDL_LOG(lvl, ...) SNNDL_LOGPTR(output_, (lvl), __VA_ARGS__)
#endif

#ifdef SNNDL_ENABLE_DEBUG_LOG
#define SNNDL_DEBUG_ENABLED 1
#define SNNDL_DEBUG_LOG(lvl, ...) SNNDL_LOG(lvl, __VA_ARGS__)
#define SNNDL_DEBUG_BLOCK(stmt) do { stmt; } while(0)
#else
#define SNNDL_DEBUG_ENABLED 0
#define SNNDL_DEBUG_LOG(lvl, ...) do {} while(0)
#define SNNDL_DEBUG_BLOCK(stmt) do {} while(0)
#endif

constexpr uint32_t BCSR_SENTINEL_ID = 0xFFFFFFFFu;

// === 静态共享路由缓存定义 ===
std::mutex SnnPESubComponent::s_route_cache_mtx_;
std::unordered_map<std::string, std::weak_ptr<const SnnPESubComponent::RouteMap>> SnnPESubComponent::s_route_cache_;
std::mutex SnnPESubComponent::s_stage_csv_mutex_;
std::unordered_set<std::string> SnnPESubComponent::s_stage_csv_files_;
static std::unordered_set<std::string> s_stage_written_once_; // key: path:seq:event

void SnnPESubComponent::appendStageEventRow_(const char* event_name, uint64_t now_ns, uint64_t spikes_emitted) {
    // 仅由 core0 代表本 PE 写入阶段事件，且每 (seq,event) 仅写一次
    if (event_name == nullptr) return;
    // 仅 EndScatter 允许所有核心上报（用于聚合发放数）；其余事件由 core0 代表写一次
    if (core_id_ != 0) {
        if (std::string(event_name) != "EndScatter") return;
    }
    std::lock_guard<std::mutex> lock(s_stage_csv_mutex_);
    // 改为通知父 PE 统一写入阶段事件（避免多核重复与多次落盘）；同时传递本窗发放数量
    if (parent_) {
        if (auto* pe = parent_pe_cached_) {
            pe->notifyStageEvent(static_cast<uint32_t>(curr_stage_seq_), std::string(event_name), now_ns, spikes_emitted);
        }
    }
}

void SnnPESubComponent::handleStageEventWithoutApply_(const GasOpData* op) {
    if (!op) return;
    uint64_t now = getCurrentSimTimeNano();
    switch (op->op) {
        case GasOp::BeginGather:
            curr_stage_seq_ = op->superstep;
            appendStageEventRow_("BeginGather", now, 0);
            break;
        case GasOp::BeginApply:
            appendStageEventRow_("BeginApply", now, 0);
            break;
        case GasOp::EndApply:
            appendStageEventRow_("EndApply", now, 0);
            break;
        case GasOp::BeginScatter:
            appendStageEventRow_("BeginScatter", now, 0);
            break;
        case GasOp::EndScatter:
            appendStageEventRow_("EndScatter", now, 0);
            break;
        default:
            break;
    }
}

void SnnPESubComponent::prepareEdgeWindowForApply_() {
    if (!(apply_acc_enable_ && gas_window_mode_)) return;
    edge_collector_.flipForApply(window_read_debug_, output_, core_id_, curr_stage_seq_);
    window_reads_issued_this_apply_ = 0;
    issueEdgeWeightFetches_();
}

#ifdef SNNDL_ENABLE_DEBUG_LOG
void SnnPESubComponent::diagEdgeWeight_(const char* tag, uint32_t post_local,
                                        uint32_t pre_global, float weight,
                                        uint32_t count) {
    if (!window_read_debug_ || !output_) return;
    output_->verbose(CALL_INFO, 0, 0,
        "[diag-weight] %s core=%d window=%u post_local=%u pre_global=%u weight=%.6f count=%u\n",
        tag ? tag : "edge", core_id_, curr_stage_seq_, post_local, pre_global,
        (double)weight, count);
}
#else
void SnnPESubComponent::diagEdgeWeight_(const char*, uint32_t, uint32_t, float, uint32_t) {}
#endif

namespace {
bool extractUnsigned(const std::string& text, const char* key, uint64_t& value) {
    auto pos = text.find(key);
    if (pos == std::string::npos) return false;
    pos = text.find(':', pos);
    if (pos == std::string::npos) return false;
    ++pos;
    while (pos < text.size() && (std::isspace(static_cast<unsigned char>(text[pos])) || text[pos] == '"')) ++pos;
    size_t end = pos;
    while (end < text.size() && (std::isdigit(static_cast<unsigned char>(text[end])) || text[end] == 'x' || text[end] == 'X')) ++end;
    if (end <= pos) return false;
    value = std::strtoull(text.substr(pos, end - pos).c_str(), nullptr, 0);
    return true;
}
}

void SnnPESubComponent::logRoutingSummary_(const char* phase, const char* reason) {
    if (!output_) return;
    if (core_id_ != 0) return; // 每个PE仅核心0打印一次，避免刷屏
    if (!window_read_debug_ && output_->getVerboseLevel() < 1) return;
    const size_t shared_entries = routes_shared_ ? routes_shared_->size() : 0;
    const size_t local_entries = shared_entries ? 0 : routes_by_source_.size();
    const char* mode = routing_weight_driven_ ? "weight_driven" : "fixed";
    output_->verbose(CALL_INFO, 0, 0,
        "[diag-route] core=%d phase=%s mode=%s use_bcsr=%d shared=%zu local=%zu template=%s reason=%s\n",
        core_id_, phase ? phase : "setup", mode, use_bcsr_ ? 1 : 0,
        shared_entries, local_entries,
        weights_template_.empty() ? "" : weights_template_.c_str(),
        reason ? reason : "-");
}

void SnnPESubComponent::logBcsrWindowStats_(const char* tag) {
    if (!window_read_debug_ || !use_bcsr_ || !output_) return;
    if (bcsr_req_edges_ == 0 && bcsr_req_wait_rowptr_ == 0 &&
        bcsr_req_block_hit_ == 0 && bcsr_req_block_miss_ == 0) {
        return;
    }
    output_->verbose(CALL_INFO, 0, 0,
        "[diag-bcsr-window] core=%d window=%u tag=%s edges=%" PRIu64
        " rowptr_wait=%" PRIu64 " hits=%" PRIu64 " miss=%" PRIu64 "\n",
        core_id_, curr_stage_seq_, tag ? tag : "-",
        bcsr_req_edges_, bcsr_req_wait_rowptr_,
        bcsr_req_block_hit_, bcsr_req_block_miss_);
}

void SnnPESubComponent::resetBcsrWindowCounters_() {
    bcsr_req_edges_ = 0;
    bcsr_req_wait_rowptr_ = 0;
    bcsr_req_block_hit_ = 0;
    bcsr_req_block_miss_ = 0;
}

void SnnPESubComponent::StageEventHub::markBeginGather(uint32_t) {
    if (!core) return;
    if (core->use_bcsr_) {
        core->logBcsrWindowStats_("prev");
        core->resetBcsrWindowCounters_();
    }
    uint64_t now = core->getCurrentSimTimeNano();
    core->appendStageEventRow_("BeginGather", now, 0);
    t_begin_gather = now;
    have_begin_gather = true;
    have_begin_apply = false;
    have_begin_scatter = false;
    core->window_spikes_all_ = 0;
}

void SnnPESubComponent::StageEventHub::markBeginApply(uint32_t) {
    if (!core) return;
    uint64_t now = core->getCurrentSimTimeNano();
    core->appendStageEventRow_("BeginApply", now, 0);
    t_begin_apply = now;
    have_begin_apply = true;
}

void SnnPESubComponent::StageEventHub::markBeginScatter(uint32_t) {
    if (!core) return;
    uint64_t now = core->getCurrentSimTimeNano();
    core->appendStageEventRow_("BeginScatter", now, 0);
    t_begin_scatter = now;
    have_begin_scatter = true;
    if (core->apply_acc_enable_ && core->gas_window_mode_) {
        if (core->fired_this_window_.size() != core->num_neurons_) {
            core->fired_this_window_.assign(core->num_neurons_, 0);
        } else {
            std::fill(core->fired_this_window_.begin(), core->fired_this_window_.end(), 0);
        }
    }
}

void SnnPESubComponent::StageEventHub::markEndScatter(uint32_t seq, uint64_t spikes_emitted) {
    if (!core) return;
    uint64_t now = core->getCurrentSimTimeNano();
    core->appendStageEventRow_("EndScatter", now, spikes_emitted);
    core->stats_reporter_.reportWindowSpikes(static_cast<uint32_t>(seq), spikes_emitted);
    core->spikes_emitted_window_ = 0;
    core->window_spikes_all_ = 0;
    if (core->apply_acc_enable_ && core->gas_window_mode_) {
        if (!core->fired_this_window_.empty()) {
            std::fill(core->fired_this_window_.begin(), core->fired_this_window_.end(), 0);
        }
    }
    if (core->stat_gas_superstep_total_cycles_) {
        if (have_begin_gather) {
            uint64_t total = (now >= t_begin_gather) ? (now - t_begin_gather) : 0ULL;
            core->stat_gas_superstep_total_cycles_->addData(total);
        }
        if (have_begin_gather && have_begin_apply && core->stat_gas_superstep_gather_cycles_) {
            uint64_t g = (t_begin_apply >= t_begin_gather) ? (t_begin_apply - t_begin_gather) : 0ULL;
            core->stat_gas_superstep_gather_cycles_->addData(g);
        }
        if (have_begin_apply && core->stat_gas_superstep_apply_cycles_) {
            uint64_t a = (t_begin_scatter >= t_begin_apply) ? (t_begin_scatter - t_begin_apply) : 0ULL;
            core->stat_gas_superstep_apply_cycles_->addData(a);
        }
        if (have_begin_scatter && core->stat_gas_superstep_scatter_cycles_) {
            uint64_t s = (now >= t_begin_scatter) ? (now - t_begin_scatter) : 0ULL;
            core->stat_gas_superstep_scatter_cycles_->addData(s);
        }
    }
    have_begin_gather = have_begin_apply = have_begin_scatter = false;
}

void SnnPESubComponent::reserveWindowContainers_() {
    size_t post_cap = std::max<size_t>(64, num_neurons_ / 8);
    posts_list_window_.reserve(post_cap);
    posts_list_prev_window_.reserve(post_cap);
    active_pre_window_.reserve(256);
    active_pre_prev_window_.reserve(256);
}

bool SnnPESubComponent::ensureLoaderReady_() {
    if (!wait_for_loader_done_) return true;
    if (loader_ready_latched_) return true;
    if (!loader_done_shared_initialized_) return true;
    if (loader_done_shared_.size() == 0) return false;
    int ready = loader_done_shared_.mutex_read(0);
    if (ready != 0) {
        loader_ready_latched_ = true;
        if (window_read_debug_ && !loader_ready_logged_ && output_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-loader] core=%d weights_ready at cycle=%" PRIu64 "\n",
                core_id_, total_cycles_);
        }
        loader_ready_logged_ = true;
        return true;
    }
    return false;
}

void SnnPESubComponent::StatsReporter::reportMemoryIssue(size_t bytes, bool count_weight_read) const {
    if (!core) return;
    const uint64_t inflight = static_cast<uint64_t>(core->pending_memory_requests_.size()) + 1ULL;
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

void SnnPESubComponent::StatsReporter::reportApplyScatter(uint64_t acc_updates, uint64_t posts_touched,
                                uint64_t spikes_emitted, uint64_t hwm_bytes,
                                uint64_t spill_records, uint64_t spilled_bytes) const {
    if (!core) return;
    if (auto* pe = core->parent_pe_cached_) {
        pe->accumulateApplyScatterStats(acc_updates, posts_touched, spikes_emitted,
                                        hwm_bytes, spill_records, spilled_bytes);
    }
}

void SnnPESubComponent::StatsReporter::reportWindowSpikes(uint32_t seq, uint64_t spikes_emitted) const {
    if (!core || spikes_emitted == 0) return;
    if (auto* pe = core->parent_pe_cached_) {
        pe->accumulateWindowSpikes(seq, spikes_emitted);
    }
}

void SnnPESubComponent::StatsReporter::reportCacheAccess(bool hit) const {
    if (!core) return;
    if (hit) {
        if (core->stat_weight_cache_hits_) core->stat_weight_cache_hits_->addData(1);
        core->count_cache_hits_++;
    } else {
        if (core->stat_weight_cache_misses_) core->stat_weight_cache_misses_->addData(1);
        core->count_cache_misses_++;
    }
}

void SnnPESubComponent::StatsReporter::updatePendingPeak(uint32_t outstanding) const {
    if (!core) return;
    if (outstanding > core->pending_reqs_peak_) {
        core->pending_reqs_peak_ = outstanding;
        if (core->stat_pending_reqs_peak_) core->stat_pending_reqs_peak_->addData(outstanding);
    }
}

void SnnPESubComponent::issueEdgeWeightFetches_() {
    read_orchestrator_.logEdgeFetchStart(
        edge_collector_.prevSize(), window_reads_issued_this_apply_,
        outstanding_requests_, window_read_budget_);
    read_orchestrator_.issueFromEdges();
}

void SnnPESubComponent::ReadOrchestrator::issueFromEdges() {
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
                uint32_t br = core->bcsr_weights_.effectiveBlockRows();
                br = (br > 0 ? br : 1);
                uint32_t bc = core->bcsr_weights_.effectiveBlockCols();
                uint32_t block_row = post_local / br;
                uint32_t intra_row = post_local % br;
                uint32_t blk_col = (bc? (pre_global / bc) : 0);
                uint32_t intra_col = (bc? (pre_global % bc) : 0);
                // 使用已加载的 rowptr_host_ 计算全局块索引
                float resolved = 0.0f;
                do {
                    const auto& rowptr = core->bcsr_weights_.rowptrHost();
                    if (block_row + 1 > rowptr.size()) break;
                    uint32_t start = rowptr[block_row];
                    uint32_t end   = (block_row + 1 < rowptr.size() ? rowptr[block_row+1] : start);
                    if (end <= start) break;
                    // 解析 meta + 读取 colidx 与对应块
                    std::string bin_path = core->resolveWeightTemplate(core->node_id_, core->core_id_);
                    if (bin_path.empty()) break;
                    uint32_t rows=0, colsN=0, brM=0, bcM=0, idxB=0, valB=0, totalB=0;
                    uint64_t rp_off=0, ci_off=0, bd_off=0, id_off=0;
                    std::string meta_path = bin_path + ".meta.json";
                    if (!core->parseBcsrMeta(meta_path, rows, colsN, brM, bcM, idxB, valB, rp_off, ci_off, bd_off, id_off, totalB)) break;
                    std::ifstream fin(bin_path, std::ios::binary);
                    if (!fin.good()) break;
                    // 在该行查找目标块列
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
                core->requestWeightBCSR(pre_global, post_local,
                    [this, post_local, count, pre_capture](float w) {
                        float resolved = w;
                        if (core->readresp_zero_fallback_ && resolved == 0.0f) resolved = core->init_default_weight_;
                        core->accUpdate_(post_local, resolved * static_cast<float>(count));
                        core->diagEdgeWeight_("bcsr", post_local, pre_capture, resolved, count);
                        if (core->outstanding_requests_ > 0) core->outstanding_requests_--;
                        issueFromEdges();
                    });
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
        if (core->weightCacheTryGet_(cache_key, cached)) {
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
        core->requestWeight(req_pre, req_post,
            [this, post_local, count, cache_key, pre_capture](float w) {
                float resolved = w;
                if (core->readresp_zero_fallback_ && resolved == 0.0f) resolved = core->init_default_weight_;
                core->weightCacheStore_(cache_key, resolved);
                core->accUpdate_(post_local, resolved * static_cast<float>(count));
                core->diagEdgeWeight_("miss", post_local, pre_capture, resolved, count);
                if (core->outstanding_requests_ > 0) core->outstanding_requests_--;
                issueFromEdges();
            });
    }
}

void SnnPESubComponent::ReadOrchestrator::issueFromSets(
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
            core->requestWeight(req_pre, req_post, [this, cache_key](float w){
                core->weightCacheStore_(cache_key, w);
                if (core->outstanding_requests_ > 0) core->outstanding_requests_--;
            });
        }
        if (!canIssueMoreReads_()) break;
    }
    logIssuedStats_(issued);
}

void SnnPESubComponent::ReadOrchestrator::issueFallbackReadsIfNeeded(bool strict_gas_active) {
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
    bool have_pres_prev = !core->active_pre_prev_window_.empty();
    bool have_posts_curr = !core->posts_list_window_.empty();
    bool have_pres_curr = !core->active_pre_window_.empty();

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

void SnnPESubComponent::ReadOrchestrator::issueFromSetsBcsr(
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

void SnnPESubComponent::ReadOrchestrator::logWindowReadSummary_(
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

void SnnPESubComponent::ReadOrchestrator::logFallbackSwitch_() const {
    if (!core) return;
    diag_.log(1,
        "[diag-window-read] BeginApply: core=%u window=%u FALLBACK to current window (prev empty, curr has data)\n",
        core->core_id_, core->curr_stage_seq_);
}

void SnnPESubComponent::ReadOrchestrator::logIssuedStats_(uint32_t issued) const {
    if (!core) return;
    diag_.log(1,
        "[diag-window-read] BeginApply: core=%u window=%u issued=%u (this window) outstanding_reqs=%u pending=%zu\n",
        core->core_id_, core->curr_stage_seq_, issued,
        core->outstanding_requests_,
        static_cast<size_t>(core->pending_memory_requests_.size()));
}

void SnnPESubComponent::ReadOrchestrator::logEdgeFetchStart(
    size_t prev_edges, uint32_t issued, uint32_t outstanding, uint32_t budget) const {
    if (!core) return;
    diag_.log(1,
        "[diag-edge-fetch] core=%d stage=%d prev_edges=%zu issued=%u outstanding=%u budget=%u\n",
        core->core_id_, static_cast<int>(core->gas_stage_), prev_edges,
        issued, outstanding, budget);
}

bool SnnPESubComponent::ReadOrchestrator::canIssueMoreReads_() const {
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

SnnPESubComponent::SnnPESubComponent(ComponentId_t id, Params& params)
    : SnnCoreAPI(id, params),
      parent_(nullptr),
      output_(nullptr),
      memory_(nullptr),
      gather_buffer_if_(nullptr),
      memory_link_(nullptr) {
    // 构造期最早哨兵（默认静默，除非显式启用 SNNDL_SENTINEL_ENABLE）。
    // 避免直接使用 stdout，统一走 SST Output。
    static const bool kSentinelOn = [](){
        const char* env = std::getenv("SNNDL_SENTINEL_ENABLE");
        return env && std::atoi(env) != 0;
    }();
    // 提前构建一个最低等级的输出对象，避免后续早期初始化路径使用 output_ 时发生空指针
    // 真实 verbose 等级稍后在解析完参数后再生效（此处仅用于早期诊断与防护）
    if (!output_) {
        try {
            output_ = new Output("SnnPESubComponent[@p:@l]: ", /*verbose*/0, 0, Output::STDOUT);
        } catch (...) {
            output_ = nullptr; // 最小化风险，保持后续分支都做空指针判定
        }
    }
    // 注意：其余子模块初始化挪到参数解析之后，避免早期未初始化成员被使用
    
    // 读取配置参数
    core_id_ = params.find<int>("core_id", 0);
    if (kSentinelOn && output_) {
        SNNDL_LOG(0, "[[sentinel-core-ctor]] core_ctor enter\n");
        SNNDL_LOG(0, "[[sentinel-core-ctor]] after params: core_id=%d\n", core_id_);
    }
    total_cores_ = params.find<int>("total_cores", 8);
    global_neuron_base_ = params.find<uint64_t>("global_neuron_base", 0);
    num_neurons_ = params.find<uint32_t>("num_neurons", 64);
    v_thresh_ = params.find<float>("v_thresh", 1.0f);
    v_reset_ = params.find<float>("v_reset", 0.0f);
    v_rest_ = params.find<float>("v_rest", 0.0f);
    tau_mem_ = params.find<float>("tau_mem", 20.0f);
    t_ref_ = params.find<uint32_t>("t_ref", 2);
    base_addr_ = params.find<uint64_t>("base_addr", 0);
    node_id_ = params.find<uint32_t>("node_id", 0);
    verbose_ = params.find<int>("verbose", 0);
    enable_extended_diagnostics_ = params.find<int>("enable_extended_diagnostics", 0) != 0;
    if (kSentinelOn && output_) {
        SNNDL_LOG(0, "[[sentinel-core-ctor]] after params2: node_id=%u num_neurons=%u base_addr=%" PRIu64 "\n",
                node_id_, num_neurons_, (uint64_t)base_addr_);
    }
    enable_weight_fetch_ = params.find<int>("enable_weight_fetch", 0) != 0;
    // 现在再初始化依赖core指针的轻量结构
    stage_event_hub_.init(this);
    stats_reporter_.init(this);
    read_orchestrator_.init(this);
#if SNNDL_DEBUG_ENABLED
    if (window_read_debug_) {
        SNNDL_DEBUG_LOG(1, "[diag-init] core=%d enable_weight_fetch=%d\n", core_id_, enable_weight_fetch_ ? 1 : 0);
    }
#endif
    write_weights_on_init_ = params.find<int>("write_weights_on_init", 1) != 0;
    memory_warmup_cycles_ = params.find<uint64_t>("memory_warmup_cycles", 1000);
    init_default_weight_ = params.find<float>("init_default_weight", 0.5f);
    readresp_zero_fallback_ = params.find<int>("readresp_zero_fallback", 0) != 0;
    max_outstanding_requests_ = params.find<uint32_t>("max_outstanding_requests", 16);
    max_cache_entries_ = params.find<uint32_t>("max_cache_entries", 65536);
    use_event_weight_fallback_ = params.find<int>("use_event_weight_fallback", 0) != 0;
    event_weight_fallback_warned_ = false;
    merge_read_cacheline_ = params.find<int>("merge_read_cacheline", 1) != 0;
    merge_read_row_ = params.find<int>("merge_read_row", 0) != 0;
    weight_cache_.reserve(max_cache_entries_ ? max_cache_entries_ : 1);
    gas_enable_ = params.find<int>("gas_enable", 0) != 0; // 默认关闭
    gas_window_mode_ = params.find<int>("gas_window_mode", 0) != 0; // 当为true时采用GatherBufferIF的window驱动
    // Deprecate manual window driving: read param for compatibility but force-disable
    bool manual_drive_param = params.find<int>("gas_manual_window_drive", 0) != 0;
    gas_manual_window_drive_ = false;
    if (manual_drive_param && output_) {
        output_->verbose(CALL_INFO, 0, 0,
            "[diag-gas-config] core=%d gas_manual_window_drive 已弃用，仍将使用自动窗口驱动\n",
            core_id_);
    }
    manual_gas_gather_cycles_cfg_ = params.find<uint64_t>("gas_window_cycles_gather", 200);
    if (manual_gas_gather_cycles_cfg_ == 0) manual_gas_gather_cycles_cfg_ = 1;
    // 方案1（slice顺序执行）
    scheme1_enable_ = params.find<int>("scheme1_enable", 0) != 0;
    scheme1_slices_ = params.find<uint32_t>("scheme1_slices", 8);
    scheme1_gather_cycles_cfg_ = params.find<uint64_t>("scheme1_gather_cycles", 100);
    scheme1_slice_gap_cycles_ = params.find<uint64_t>("scheme1_slice_gap_cycles", 0);
    scheme1_scatter_cycles_ = params.find<uint64_t>("scheme1_scatter_cycles", 1);
    scheme1_partition_mod_ = params.find<int>("scheme1_partition_mod", 0) != 0;
    merge_read_auto_ = params.find<int>("merge_read_auto", 0) != 0; // default off
    line_size_bytes_ = params.find<uint32_t>("line_size_bytes", 64);
    loader_done_key_ = params.find<std::string>("loader_done_key", "");
    wait_for_loader_done_ = !loader_done_key_.empty();
    if (wait_for_loader_done_) {
        loader_done_shared_.initialize(loader_done_key_, 1, 0);
        loader_done_shared_initialized_ = true;
        if (window_read_debug_ && output_) {
            output_->verbose(CALL_INFO, 1, 0,
                "[diag-loader] core=%d init loader_done_key=%s\n",
                core_id_, loader_done_key_.c_str());
        }
    }
    disable_weight_cache_ = params.find<int>("disable_weight_cache", 0) != 0;
    window_read_enable_ = params.find<int>("window_read_enable", 0) != 0;
    window_read_debug_ = params.find<int>("window_read_debug", 0) != 0;
    window_read_budget_ = params.find<uint32_t>("window_read_budget", 1024);
    read_force_single_ = params.find<int>("read_force_single", 0) != 0;
    // 边集合容量上限（极端保护）
    edge_collector_max_capacity_ = static_cast<size_t>(params.find<uint64_t>("edge_collector_max_capacity", 1000000));
    if (window_read_enable_) {
        reserveWindowContainers_();
    if (!record_edge_idle_enable_ && !record_edge_scatter_enable_ && window_read_debug_ && output_ && enable_extended_diagnostics_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-record-edge] core=%d 仅在Gather阶段记录边 (Apply/Idle/Scatter=0)", core_id_);
        }
    } else if (window_read_debug_ && output_ && enable_extended_diagnostics_) {
        output_->verbose(CALL_INFO, 0, 0,
            "[diag-record-edge] core=%d window_read_enable=0 => 忽略 window_read_debug", core_id_);
    }
    // 全网读取扩展参数
    weights_cols_ = params.find<uint32_t>("weights_cols", 0);
    std::string index_mode_str = params.find<std::string>("index_mode", "pre_row_post_col");
    use_soa_state_ = params.find<int>("use_soa_neuron_state", 0) != 0;
    use_aosoa_state_ = params.find<int>("use_aosoa_neuron_state", 0) != 0;
    aosoa_block_rows_ = params.find<uint32_t>("aosoa_block_rows", 0);
    if (use_aosoa_state_) use_soa_state_ = true;
    verify_routing_weights_ = params.find<int>("verify_routing_weights", 0) != 0;
    use_post_row_pre_col_ = (index_mode_str == "post_row_pre_col");
    if (index_mode_str == "bcsr_post_row") {
        use_bcsr_ = true;
        use_post_row_pre_col_ = true;
    } else if (index_mode_str == "csr_post_row") {
        // CSR 模式已弃用：统一禁用
        use_post_row_pre_col_ = true;
        SNNDL_DEBUG_LOG(1, "[CSR] 索引模式已禁用，改用密集/BCSR读取\n");
    }
    if (weights_cols_ == 0) weights_cols_ = num_neurons_; // 默认沿用旧行宽
    // 预计算dense权重区域上界（按行*列*4B）
    {
        uint64_t bytes = static_cast<uint64_t>(num_neurons_) * static_cast<uint64_t>(weights_cols_) * static_cast<uint64_t>(sizeof(float));
        weight_region_end_ = base_addr_ + bytes;
    }
    enable_detailed_map_log_ = params.find<int>("enable_detailed_map_log", 0) != 0;
    log_weight_details_ = params.find<int>("log_weight_details", 0) != 0;
    loader_barrier_cycles_ = params.find<uint64_t>("loader_barrier_cycles", 0);
    // BCSR 参数
    bcsr_layout_.rows = num_neurons_;
    bcsr_layout_.cols = params.find<uint32_t>("weights_cols", num_neurons_);
    bcsr_layout_.block_rows = params.find<uint32_t>("bcsr_block_rows", 16);
    bcsr_layout_.block_cols = params.find<uint32_t>("bcsr_block_cols", 16);
    bcsr_layout_.idx_bytes = params.find<uint32_t>("bcsr_idx_bytes", 2);
    bcsr_layout_.val_bytes = params.find<uint32_t>("bcsr_val_bytes", 4);
    bcsr_layout_.rowptr_offset = params.find<uint64_t>("bcsr_rowptr_offset", 0);
    bcsr_layout_.colidx_offset = params.find<uint64_t>("bcsr_colidx_offset", 0);
    bcsr_layout_.blockdata_offset = params.find<uint64_t>("bcsr_blockdata_offset", 0);
    bcsr_layout_.blockids_offset = params.find<uint64_t>("bcsr_blockids_offset", 0);
    bcsr_layout_.per_core_stride = params.find<uint64_t>("per_core_stride", 0);
    bcsr_layout_.validate(base_addr_, output_, (window_read_debug_ || enable_extended_diagnostics_), core_id_, node_id_);
    uint64_t bcsr_rowptr_addr = base_addr_ + bcsr_layout_.rowptr_offset;
    uint64_t bcsr_colidx_addr = base_addr_ + bcsr_layout_.colidx_offset;
    uint64_t bcsr_blockdata_addr = base_addr_ + bcsr_layout_.blockdata_offset;
    uint64_t bcsr_blockids_addr = bcsr_layout_.blockids_offset ? base_addr_ + bcsr_layout_.blockids_offset : 0;
    bcsr_weights_.configure(bcsr_rowptr_addr,
                            bcsr_colidx_addr,
                            bcsr_blockdata_addr,
                            bcsr_blockids_addr,
                            bcsr_layout_.block_rows,
                            bcsr_layout_.block_cols,
                            bcsr_layout_.idx_bytes,
                            bcsr_layout_.val_bytes);
    const uint32_t row_index_cache_cap = params.find<uint32_t>("bcsr_row_index_cache_cap", 64);
    const uint32_t block_cache_cap = params.find<uint32_t>("bcsr_block_cache_cap", 256);
    bcsr_weights_.setRowIndexCacheCapacity(row_index_cache_cap);
    bcsr_weights_.setBlockCacheCapacity(block_cache_cap);
    if (aosoa_block_rows_ == 0) aosoa_block_rows_ = bcsr_weights_.effectiveBlockRows();
    // GAS Apply/Scatter Phase‑1
    apply_acc_enable_ = params.find<int>("apply_acc_enable", 0) != 0;
    acc_hwm_bytes_ = params.find<uint64_t>("acc_high_watermark_bytes", 16*1024*1024);
    acc_spill_enable_ = params.find<int>("acc_spill_enable", 1) != 0;
    stage_events_csv_ = params.find<std::string>("stage_events_csv", "");
    if (aosoa_block_rows_ == 0) aosoa_block_rows_ = 16;
    // CSR 参数已移除
    bcsr_prefetch_all_ = params.find<int>("bcsr_prefetch_all", 0) != 0;
    // 权重验证参数
    verify_weights_ = params.find<int>("verify_weights", 0) != 0;
    bcsr_force_file_read_ = params.find<int>("bcsr_force_file_read", 0) != 0;
    bcsr_rowptr_file_fallback_enable_ = params.find<int>("bcsr_rowptr_file_fallback_enable", 0) != 0;
    weight_verify_samples_ = params.find<uint32_t>("weight_verify_samples", 16);
    expected_weight_value_ = params.find<float>("expected_weight_value", 0.0f);
    verify_epsilon_ = params.find<float>("verify_epsilon", 1e-4f);
    verify_log_each_sample_ = params.find<int>("verify_log_each_sample", 0) != 0;
    verify_against_file_ = params.find<int>("verify_against_file", 0) != 0;
    verify_cluster_enable_ = params.find<int>("verify_cluster_enable", 0) != 0;
    verify_file_template_ = params.find<std::string>("verify_file_template", "");
    quiet_finish_logs_ = params.find<int>("quiet_finish_logs", 0) != 0;
    const int record_apply_default = (gas_window_mode_ && apply_acc_enable_) ? 1 : 0;
    record_edge_apply_enable_ = params.find<int>("record_edge_apply_enable", record_apply_default) != 0;
    record_edge_idle_enable_ = params.find<int>("record_edge_idle_enable", 0) != 0;
    if (record_edge_idle_enable_ && output_ && enable_extended_diagnostics_) {
        output_->verbose(CALL_INFO, 0, 0,
            "[diag-gas-config] core=%d 已启用 record_edge_idle（诊断配置，回归默认关闭）\n",
            core_id_);
    }
    record_edge_scatter_enable_ = params.find<int>("record_edge_scatter_enable", 0) != 0;

    // ---- Profiling init (optional; minimal overhead when disabled) ----
#ifdef SNNDL_ENABLE_PROFILING
    profiler_enabled_ = params.find<int>("enable_profiler", 0) != 0;
    profiler_csv_prefix_ = params.find<std::string>("profiler_csv_prefix", "");
    if (profiler_enabled_) {
        try {
            profiler_ = new SST::SnnDL::Profiler(std::string("SnnPESubComponent_Core") + std::to_string(core_id_));
        } catch (...) {
            profiler_ = nullptr;
            profiler_enabled_ = false;
        }
    }
#endif
    // 代码级内存优化开关（可选）
    use_clock_weight_cache_ = params.find<int>("use_clock_weight_cache", 0) != 0;
    // 默认启用致密累加器（与头文件参数表一致）
    apply_dense_acc_enable_ = params.find<int>("apply_dense_acc_enable", 1) != 0;
    acc_shadow_verify_enable_ = apply_dense_acc_enable_ && params.find<int>("acc_shadow_verify_enable", 0) != 0;
    if (acc_shadow_verify_enable_ && !enable_extended_diagnostics_) {
        acc_shadow_verify_enable_ = false;
    }
    if (acc_shadow_verify_enable_) {
        // TEMP(debug): write a one-shot breadcrumb so we can confirm shadow verification is actually enabled at runtime.
        // This will be removed after verification of runs (10us/100us).
        FILE* fp = std::fopen("/tmp/acc_shadow.log", "a");
        if (fp) {
            std::fprintf(fp, "[acc-shadow-enabled] node=%u core=%u seq=%u time_ns=%" PRIu64 "\n",
                        node_id_, core_id_, curr_stage_seq_, (uint64_t)getCurrentSimTimeNano());
            std::fclose(fp);
        }
        // 强制落盘到 stdout，便于调试捕获（调试用，跑完撤销）
        fprintf(stdout, "[acc-shadow-enable] core=%d acc_dense=1 shadow=1\n", core_id_);
        fflush(stdout);
    }
    if (use_clock_weight_cache_ && max_cache_entries_ > 0) {
        wcache_cap_ = max_cache_entries_;
        wcache_keys_.resize(wcache_cap_);
        wcache_vals_.resize(wcache_cap_);
        wcache_access_.assign(wcache_cap_, 0);
        wcache_index_.reserve(wcache_cap_);
        wcache_hand_ = 0; wcache_size_ = 0;
    }
    if (apply_dense_acc_enable_) {
        acc_dense_.assign(num_neurons_, 0.0f);
        acc_touched_bitmap_.assign(num_neurons_, 0);
        acc_touched_list_.reserve(num_neurons_ / 10 + 8);
    }
    // 路由模式参数
    std::string routing_mode = params.find<std::string>("routing_mode", "fixed");
    routing_weight_driven_ = (routing_mode == "weight_driven");
    weights_template_ = params.find<std::string>("weights_template", "");
    total_nodes_cfg_ = params.find<uint32_t>("total_nodes", 16);
    // 默认按 num_cores * num_neurons 推导；但允许Python侧显式传入 neurons_per_pe
    uint32_t np_from_params = params.find<uint32_t>("neurons_per_pe", 0);
    uint32_t computed_neurons_per_pe = static_cast<uint32_t>(total_cores_) * static_cast<uint32_t>(num_neurons_);
    if (np_from_params > 0) {
        neurons_per_pe_cfg_ = np_from_params;
        if (np_from_params != computed_neurons_per_pe && output_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-gas-config] core=%d neurons_per_pe=%u (脚本指定) ≠ cores*rows=%u\n",
                core_id_, np_from_params, computed_neurons_per_pe);
        }
    } else {
        neurons_per_pe_cfg_ = computed_neurons_per_pe;
    }
    routing_epsilon_ = params.find<float>("routing_epsilon", 1e-8f);
    routing_topk_ = params.find<uint32_t>("routing_topk", 0);
    routing_topk_per_pe_ = params.find<uint32_t>("routing_topk_per_pe", 0);
    route_exclude_self_pe_ = params.find<int>("route_exclude_self_pe", 0) != 0;
    route_layers_mask_ = params.find<std::string>("route_layers_mask", "");
    route_filter_warn_ = params.find<int>("route_filter_warn", 1) != 0;
    // 映射框架集成
    mapping_mode_ = params.find<std::string>("mapping_mode", "off");
    mapping_edges_file_ = params.find<std::string>("mapping_edges_file", "");
    mapping_csv_has_header_ = params.find<int>("mapping_csv_has_header", 1) != 0;
    mapping_csv_separator_ = params.find<std::string>("mapping_csv_separator", ",");
    mapping_assume_block_ids_ = params.find<int>("mapping_assume_block_ids", 1) != 0;
    // 解析层间许可掩码
    allowed_layer_edges_.clear();
    allow_all_layers_ = true;
    if (!route_layers_mask_.empty()) {
        allow_all_layers_ = false;
        auto mask = route_layers_mask_;
        // 统一大小写，分隔符支持逗号或分号
        for (auto &ch : mask) ch = (char)std::toupper((unsigned char)ch);
        std::vector<std::string> toks;
        size_t start = 0;
        for (size_t i = 0; i <= mask.size(); ++i) {
            if (i == mask.size() || mask[i] == ',' || mask[i] == ';') {
                if (i > start) toks.emplace_back(mask.substr(start, i - start));
                start = i + 1;
            }
        }
        auto layerId = [](const std::string& s)->int{
            if (s == "I") return 0;      // Input 0-3
            if (s == "H1") return 1;     // Hidden1 4-7
            if (s == "H2") return 2;     // Hidden2 8-11
            if (s == "O") return 3;      // Output 12-15
            return -1;
        };
        for (auto &t : toks) {
            size_t p = t.find('>');
            if (p == std::string::npos) continue;
            std::string a = t.substr(0, p);
            std::string b = t.substr(p+1);
            int la = layerId(a); int lb = layerId(b);
            if (la >= 0 && lb >= 0) {
                uint32_t key = ((uint32_t)la << 8) | (uint32_t)lb;
                allowed_layer_edges_.insert(key);
            }
        }
    }
    
    // 获取权重文件路径
    weights_file_path_ = params.find<std::string>("weights_file", "");

    // === Supervised-learning (Phase 1) params ===
    learning_enabled_   = params.find<int>("learning_enabled", 0) != 0;
    learn_window_cycles_ = params.find<uint64_t>("learn_window_cycles", 1000);
    record_membrane_    = params.find<int>("record_membrane", 0) != 0;
    record_spike_times_ = params.find<int>("record_spike_times", 1) != 0;
    surrogate_type_     = params.find<std::string>("surrogate_type", "superspike");
    surrogate_beta_     = params.find<float>("surrogate_beta", 5.0f);
    error_file_template_ = params.find<std::string>("error_file", "");
    grad_accum_limit_    = (size_t) params.find<uint32_t>("grad_accum_limit", 0);
    apply_writeback_     = params.find<int>("apply_writeback", 0) != 0;
    apply_every_n_windows_ = params.find<uint32_t>("apply_every_n_windows", 1);
    learning_rate_       = params.find<float>("learning_rate", 0.001f);
    weight_decay_        = params.find<float>("weight_decay", 0.0f);

    // 参数日志改至 setup 以避免构造早期潜在问题
    
    // 初始化输出对象（若前面已创建则不重复）
    if (!output_) {
        output_ = new Output("SnnPESubComponent[@p:@l]: ", verbose_, 0, Output::STDOUT);
    }
    if (window_read_debug_) {
        output_->verbose(CALL_INFO, 0, 0,
            "[diag-bcsr-base] node=%u core=%d base=0x%llx rowptr=0x%llx colidx=0x%llx blockdata=0x%llx blockids=0x%llx\n",
            node_id_, core_id_,
            (unsigned long long)base_addr_,
            (unsigned long long)bcsr_weights_.rowptrAddr(),
            (unsigned long long)bcsr_weights_.colidxAddr(),
            (unsigned long long)bcsr_weights_.blockdataAddr(),
            (unsigned long long)bcsr_weights_.blockidsAddr());
    }
    if (use_bcsr_) {
        if (weights_template_.empty() && window_read_debug_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-bcsr] node=%u core=%d warning: weights_template empty while BCSR enabled\n",
                node_id_, core_id_);
        } else if (bcsr_rowptr_file_fallback_enable_ && !weights_template_.empty() &&
                   !bcsr_weights_.isRowptrReady() && loadBcsrRowptrFromFile_()) {
            if (window_read_debug_) {
                output_->verbose(CALL_INFO, 0, 0,
                    "[diag-bcsr] core=%u preload rowptr entries=%zu first=%u second=%u\n",
                    core_id_, bcsr_weights_.rowptrHost().size(),
                    bcsr_weights_.rowptrHost().empty()?0u:bcsr_weights_.rowptrHost()[0],
                    bcsr_weights_.rowptrHost().size()>1?bcsr_weights_.rowptrHost()[1]:0u);
            }
        }
    }

    // output_->verbose(CALL_INFO, 1, 0, "🔧 初始化SnnPE SubComponent (核心%d, %u个神经元)\n", 
    //                 core_id_, num_neurons_);
    
    // 输出权重验证参数以便调试
    // 降低默认日志级别，避免常规运行下产生噪声
    output_->verbose(CALL_INFO, 3, 0, "🔍 权重验证配置: verify_weights=%d, samples=%u, expected=%.3f, log_each=%d\n",
                     verify_weights_ ? 1 : 0, weight_verify_samples_, expected_weight_value_, verify_log_each_sample_ ? 1 : 0);
    
    // 初始化神经元状态（复用SnnPE逻辑）
    if (use_soa_state_) {
        soa_v_mem_.assign(num_neurons_, v_rest_);
        soa_refrac_.assign(num_neurons_, 0);
        soa_last_spike_.assign(num_neurons_, 0);
        neuron_states_.clear();
    } else {
        neuron_states_.resize(num_neurons_);
        for (uint32_t i = 0; i < num_neurons_; i++) {
            neuron_states_[i] = NeuronState(v_rest_);
        }
        soa_v_mem_.clear();
        soa_refrac_.clear();
        soa_last_spike_.clear();
    }
    // 去重发放统计位图（默认全0）
    fired_ever_.assign(num_neurons_, 0);
    
    // 初始化内存访问
    memory_link_ = nullptr;
    memory_ = nullptr;
    next_request_id_ = 1;

    // Initialize learning window
    window_start_cycle_ = 0;
    current_window_index_ = 0;
    if (learning_enabled_) {
        error_buffer_.assign(num_neurons_, 0.0f);
    }


    
    // 初始化统计变量
    total_cycles_ = 0;
    active_cycles_ = 0;
    boot_read_sent_ = false;
    boot_write_sent_ = false;
    delayed_read_counter_ = 0;
    delayed_read_triggered_ = false;
    weights_initialized_ = false;
    memory_ready_ = false;
    stat_spikes_received_ = nullptr;
    stat_spikes_generated_ = nullptr;
    stat_neurons_fired_ = nullptr;
    stat_memory_requests_ = nullptr;
    stat_weight_cache_hits_ = nullptr;
    stat_weight_cache_misses_ = nullptr;
    stat_merged_reads_rows_ = nullptr;
    stat_merged_reads_cls_ = nullptr;
    stat_weights_verify_count_ = nullptr;
    stat_weights_mismatch_count_ = nullptr;
    stat_weights_verify_sum_ = nullptr;
    
    // 初始化内部计数器
    count_spikes_received_ = 0;
    count_spikes_generated_ = 0;
    count_neurons_fired_ = 0;
    count_memory_requests_ = 0;
    
    // 配置时钟
    std::string clock_freq = "1GHz";
    registerClock(clock_freq, new Clock::Handler2<SnnPESubComponent,&SnnPESubComponent::clockTick>(this));
    
    // 立即注册统计，避免在调用 getStatistics 前指针为空
    initializeStatistics();

    // === 读取门控事件参数 ===
    std::string gating_mode = params.find<std::string>("gating_mode", "off");
    gating_event_mode_ = (gating_mode == "event");
    gating_ttl_cycles_cfg_ = params.find<uint64_t>("gating_ttl_cycles", 1000);
    std::string gating_scope = params.find<std::string>("gating_scope", "inputs");
    gating_scope_inputs_only_ = (gating_scope != "all");

    // output_->verbose(CALL_INFO, 2, 0, "✅ SnnPE SubComponent核心%d初始化完成\n", core_id_);
}

// 生成共享路由缓存键：尽量覆盖会影响路由构建结果的所有参数
std::string SnnPESubComponent::buildRouteCacheKey() const {
    try {
        std::ostringstream ss;
        // 路由模式与输入来源
        ss << "mode=" << (routing_weight_driven_ ? "wd" : "fixed");
        ss << ";mapping_mode=" << mapping_mode_;
        ss << ";weights_tpl=" << weights_template_;
        ss << ";edges_file=" << mapping_edges_file_;
        ss << ";csv_hdr=" << (mapping_csv_has_header_ ? 1 : 0);
        ss << ";csv_sep=" << mapping_csv_separator_;
        ss << ";assume_blk=" << (mapping_assume_block_ids_ ? 1 : 0);
        // 过滤/裁剪参数
        ss << ";eps=" << routing_epsilon_;
        ss << ";topk=" << routing_topk_;
        ss << ";topk_pe=" << routing_topk_per_pe_;
        ss << ";ex_self=" << (route_exclude_self_pe_ ? 1 : 0);
        ss << ";layers=" << route_layers_mask_;
        // 维度/规模参数
        ss << ";total_nodes=" << total_nodes_cfg_;
        ss << ";rows(perPE)=" << num_neurons_;
        ss << ";cols(global)=" << weights_cols_;
        return ss.str();
    } catch (...) {
        return std::string();
    }
}

SnnPESubComponent::~SnnPESubComponent() {
    // output_->verbose(CALL_INFO, 1, 0, "🗑️ 销毁SnnPE SubComponent核心%d\n", core_id_);
    parent_pe_cached_ = nullptr;

    // 清理脉冲队列
    while (!incoming_spikes_.empty()) {
        delete incoming_spikes_.front();
        incoming_spikes_.pop();
    }
#ifdef SNNDL_ENABLE_PROFILING
    if (profiler_) { delete profiler_; profiler_ = nullptr; }
#endif
    // 避免析构次序竞态：不手动delete日志对象
    output_ = nullptr;
}

// === Activity f (per-window active axons ratio) ===
void SnnPESubComponent::activityFlush_() {
    if (!activity_stats_enable_) return;
    if (!parent_) return;
    if (weights_cols_ == 0) return;
    double f = (double)activity_pre_set_.size() / (double)weights_cols_;
    if (auto* pe = parent_pe_cached_) {
        pe->accumulateActivityF(f);
    }
    activityReset_();
}

void SnnPESubComponent::setParentInterface(SnnPEParentInterface* parent) {
    parent_ = parent;
    parent_pe_cached_ = dynamic_cast<MultiCorePE*>(parent);
    // output_->verbose(CALL_INFO, 2, 0, "🔗 核心%d设置父级接口\n", core_id_);
}

void SnnPESubComponent::init(unsigned int phase) {
    // 提前构建输出对象，避免在init早期使用output_时空指针
    if (!output_) {
        output_ = new Output("SnnPESubComponent[@p:@l]: ", verbose_, 0, Output::STDOUT);
    }
    // output_->verbose(CALL_INFO, 1, 0, "🔄 核心%d init phase %u\n", core_id_, phase);
    
    if (phase == 0) {
        // 初始化统计收集
        initializeStatistics();
        
        // 配置内存端口（可选，但不覆盖已设置的链接）
        if (!memory_link_) {
            memory_link_ = configureLink("mem_link");
            if (memory_link_) output_->verbose(CALL_INFO, 2, 0, "🔗 核心%d配置mem_link\n", core_id_);
        }
        
        // 加载StandardMem接口（Python可通过槽位提供）
        memory_ = loadUserSubComponent<SST::Interfaces::StandardMem>(
            "memory", ComponentInfo::SHARE_NONE,
            registerTimeBase("1ns"),
            new SST::Interfaces::StandardMem::Handler2<SnnPESubComponent, &SnnPESubComponent::handleMemoryResponse>(this));
        if (memory_) {
            // output_->verbose(CALL_INFO, 1, 0, "✅ 核心%d加载StandardMem成功\n", core_id_);
            // 若已成功加载 StandardMem 子组件，则认为内存就绪（即使未显式提供 memory_link_）
            memory_ready_ = true;
            if (gas_manual_window_drive_) {
                gather_buffer_if_ = dynamic_cast<GatherBufferIF*>(memory_);
                if (!gather_buffer_if_) {
                    output_->verbose(CALL_INFO, 0, 0,
                        "⚠️ 核心%d启用gas_manual_window_drive但memory不是GatherBufferIF，降级为自动窗口\n",
                        core_id_);
                    gas_manual_window_drive_ = false;
                } else {
                    output_->verbose(CALL_INFO, 0, 0,
                        "[diag-gas] 核心%d启用manual窗口驱动 (GatherBufferIF) gather_cycles=%" PRIu64 "\n",
                        core_id_, manual_gas_gather_cycles_cfg_);
                }
            }
        } else {
            // output_->verbose(CALL_INFO, 1, 0, "⚠️ 核心%d未加载StandardMem，将使用默认权重\n", core_id_);
        }

        // 路由表构建（可选，支持共享缓存）
        if (routing_weight_driven_) {
            bool ok = false;
            // 生成共享缓存key，尽量覆盖影响路由的所有参数
            std::string cache_key = buildRouteCacheKey();
            if (!cache_key.empty()) {
                std::shared_ptr<const RouteMap> hit;
                {
                    std::lock_guard<std::mutex> g(s_route_cache_mtx_);
                    auto it = s_route_cache_.find(cache_key);
                    if (it != s_route_cache_.end()) hit = it->second.lock();
                }
                if (hit) {
                    routes_shared_ = hit;
                    ok = true;
                    // 统计共享表条目总数
                    uint64_t total_entries = 0; for (auto &kv : *routes_shared_) total_entries += (uint64_t)kv.second.size();
                    if (stat_routes_entries_) stat_routes_entries_->addData(total_entries);
                    // 降低默认日志级别
                    SNNDL_LOG(3, "🔁 命中共享路由缓存: 核心%d, 源条目=%zu, 总目的=%" PRIu64 "\n",
                        core_id_, routes_shared_->size(), total_entries);
                }
            }

            if (!ok) {
                // 未命中缓存，则构建一次
                if (mapping_mode_ == "edges_csv" && !mapping_edges_file_.empty()) {
                    ok = buildRoutesFromEdgesCSV();
                } else {
                    ok = buildWeightDrivenRoutes();
                }

                if (!ok) {
                    output_->verbose(CALL_INFO, 1, 0, "⚠️ 核心%d权重驱动路由构建失败，将回退fixed路由\n", core_id_);
                    routing_weight_driven_ = false;
                } else {
                    // 将本地表封装为共享指针，并写入缓存
                    auto built = std::make_shared<RouteMap>(routes_by_source_);
                    routes_shared_ = built;
                    // 统计
                    uint64_t total_entries = 0; for (auto &kv : *routes_shared_) total_entries += (uint64_t)kv.second.size();
                    if (stat_routes_entries_) stat_routes_entries_->addData(total_entries);
                    // 统计本地/远端目的比例（仅一次性日志，调试用）
                    uint64_t local_edges = 0, remote_edges = 0;
                    const uint32_t denom = (neurons_per_pe_cfg_ > 0) ? neurons_per_pe_cfg_ : num_neurons_;
                    for (const auto &kv : *routes_shared_) {
                        for (auto post_global : kv.second) {
                            uint32_t pe_of_post = (denom ? (post_global / denom) : 0);
                            if (pe_of_post == node_id_) ++local_edges; else ++remote_edges;
                        }
                    }
                    if (!route_summary_logged_) {
                        route_summary_logged_ = true;
                        double local_ratio = (total_entries > 0) ? (double)local_edges / (double)total_entries : 0.0;
                        double remote_ratio = (total_entries > 0) ? (double)remote_edges / (double)total_entries : 0.0;
                        output_->verbose(CALL_INFO, 0, 0,
                            "[route-summary] node=%u core=%d entries=%zu total=%" PRIu64 " local=%" PRIu64 " (%.2f) remote=%" PRIu64 " (%.2f)\n",
                            node_id_, core_id_, routes_shared_->size(), total_entries,
                            local_edges, local_ratio, remote_edges, remote_ratio);
                    }
                    // 发布到进程级缓存
                    std::string cache_key = buildRouteCacheKey();
                    if (!cache_key.empty()) {
                        std::lock_guard<std::mutex> g(s_route_cache_mtx_);
                        s_route_cache_[cache_key] = built;
                    }
                    // 释放本地副本以减内存占用（后续使用共享表）
                    routes_by_source_.clear();
                }
            }
        }

        if (window_read_debug_ || output_->getVerboseLevel() >= 1) {
            logRoutingSummary_("setup", routing_weight_driven_ ? "active" : "fallback_fixed");
        }

        // 加载用于验证的权重文件（可选）
        if (verify_against_file_ && !verify_file_template_.empty()) {
            std::string path = verify_file_template_;
            size_t pos = path.find("{pe:02d}");
            if (pos != std::string::npos) {
                char buf[16]; std::snprintf(buf, sizeof(buf), "%02u", node_id_);
                path.replace(pos, 8, buf);
            } else {
                pos = path.find("{pe}");
                if (pos != std::string::npos) path.replace(pos, 4, std::to_string(node_id_));
            }
            std::ifstream fin(path, std::ios::binary);
            if (fin.good()) {
                fin.seekg(0, std::ios::end);
                std::streamsize bytes = fin.tellg();
                fin.seekg(0, std::ios::beg);
                if (bytes > 0 && (bytes % sizeof(float) == 0)) {
                    size_t count = static_cast<size_t>(bytes / sizeof(float));
                    verify_file_buf_.resize(count);
                    fin.read(reinterpret_cast<char*>(verify_file_buf_.data()), bytes);
                    verify_file_loaded_ = true;
                    output_->verbose(CALL_INFO, 1, 0, "✅ 验证文件加载完成: %s (floats=%zu)\n", path.c_str(), verify_file_buf_.size());
                } else {
                    output_->verbose(CALL_INFO, 1, 0, "⚠️ 验证文件尺寸异常: %s\n", path.c_str());
                }
            } else {
                output_->verbose(CALL_INFO, 1, 0, "⚠️ 无法打开验证文件: %s\n", path.c_str());
            }
        }
    }

    // 将 init 相位转发给 StandardMem，以建立地址映射与握手
    if (memory_) {
        memory_->init(phase);
    }

    // Default weight initialization disabled, relying on WeightLoader
    if (phase == 4) {
        // 所有init阶段结束，允许后续时钟中发起访问
        memory_ready_ = true;
        // 重置验证状态
        verify_started_ = false;
        verify_requested_ = 0;
        verify_completed_ = 0;
        verify_sum_ = 0.0;
        verify_mismatch_count_ = 0;
    }
}

void SnnPESubComponent::setup() {
    // output_->verbose(CALL_INFO, 1, 0, "🔧 核心%d setup 进入\n", core_id_);
    // output_->verbose(CALL_INFO, 1, 0,
    //     "🧩 参数: init_default_weight=%.3f, fallback=%d, merge_row=%d, merge_cl=%d, line=%uB, base_addr=%" PRIu64 ", N=%u\n",
    //     init_default_weight_, use_event_weight_fallback_, merge_read_row_, merge_read_cacheline_, line_size_bytes_, base_addr_, num_neurons_);
    
    // 验证组件状态
    if (!parent_) {
        output_->fatal(CALL_INFO, -1, "❌ 错误: 核心%d没有父级接口\n", core_id_);
    }
    // 注意：此处不直接发起内存访问，避免在setup阶段 MemLink 尚未建立时触发 memHierarchy fatal
    if (!memory_) {
        // output_->verbose(CALL_INFO, 1, 0, "⚠️ 核心%d未配置StandardMem，检查是否有直接权重文件\n", core_id_);
        
        // 权重将由WeightLoader组件通过内存接口加载
        if (!weights_file_path_.empty()) {
            // output_->verbose(CALL_INFO, 1, 0, "🔧 核心%d权重文件路径: %s (将由WeightLoader加载)\n", core_id_, weights_file_path_.c_str());
        } else {
            output_->verbose(CALL_INFO, 1, 0, "⚠️ 核心%d未配置权重文件，将使用默认权重\n", core_id_);
        }
    }

    // 打印映射模式与GAS端到端配置（一次性调试信息）
    {
        const char* idx_name = use_bcsr_ ? "bcsr_post_row" : (use_post_row_pre_col_ ? "post_row_pre_col" : "pre_row_post_col");
        int diag_lvl = window_read_debug_ ? 0 : 1;
        output_->verbose(CALL_INFO, diag_lvl, 0,
            "[GAS-Debug] core=%d index_mode=%s use_post_row_pre_col=%d apply_acc_enable=%d gas_enable=%d gas_window_mode=%d\n",
            core_id_, idx_name, use_post_row_pre_col_ ? 1 : 0, apply_acc_enable_ ? 1 : 0, gas_enable_ ? 1 : 0, gas_window_mode_ ? 1 : 0);
        output_->verbose(CALL_INFO, diag_lvl, 0,
            "[Init] core=%d global_base=%" PRIu64 " num_neurons=%u weights_cols=%u\n",
            core_id_, (uint64_t)global_neuron_base_, num_neurons_, weights_cols_);
    }
    // 配置一致性：启用窗口端到端语义时要求 window 模式的 GAS
    if (apply_acc_enable_ && (!gas_enable_ || !gas_window_mode_)) {
        output_->fatal(CALL_INFO, -1, "❌ 配置错误：apply_acc_enable=1 需要 GAS 启用且 gas_window_mode=1 (window_auto)。\n");
    }
    // output_->verbose(CALL_INFO, 1, 0, "✅ SnnPE SubComponent核心%d setup完成\n", core_id_);
}

void SnnPESubComponent::finish() {
    // 统计聚合（保持原路径）
    if (stat_pending_reqs_peak_) stat_pending_reqs_peak_->addData(pending_reqs_peak_);
    double avg_lat = (count_mem_responses_ > 0) ? ((double)accum_mem_latency_cycles_ / (double)count_mem_responses_) : 0.0;
    double utilization = (total_cycles_ > 0) ? (double)active_cycles_ / (double)total_cycles_ : 0.0;
    if (!quiet_finish_logs_) {
        // 输出统计信息（使用内部计数器获得正确值）
        output_->verbose(CALL_INFO, 1, 0, "📊 核心%d统计: 接收脉冲=%" PRIu64 ", 生成脉冲=%" PRIu64 ", 神经元发放=%" PRIu64 "\n",
                        core_id_, count_spikes_received_, count_spikes_generated_, count_neurons_fired_);
        if (verify_weights_) {
            output_->verbose(CALL_INFO, 1, 0, "🔍 权重验证: 完成=%u, 不匹配=%" PRIu64 "\n",
                             verify_completed_, verify_mismatch_count_);
        }
        output_->verbose(CALL_INFO, 0, 0,
            "📈 核心%d性能摘要: total_cycles=%" PRIu64 ", active_cycles=%" PRIu64 ", utilization=%.4f, memory_req=%" PRIu64 ", cache_hit=%" PRIu64 ", cache_miss=%" PRIu64 ", pending_peak=%u, avg_mem_lat=%.2f\n",
            core_id_, total_cycles_, active_cycles_, utilization,
            count_memory_requests_, count_cache_hits_, count_cache_misses_, pending_reqs_peak_, avg_lat);
    }

#ifdef SNNDL_ENABLE_PROFILING
    if (profiler_enabled_ && profiler_) {
        // 控制台摘要
        profiler_->generate_report(std::cout, 3.0);
        // CSV 导出：prefix 优先；否则回退到工程级 analysis
        std::string csv = profiler_csv_prefix_.empty() ? std::string("analysis/profile_core") : profiler_csv_prefix_;
        csv += std::string("_c") + std::to_string(core_id_) + std::string(".csv");
        profiler_->export_csv(csv, 3.0);
    }
#endif
}

bool SnnPESubComponent::clockTick(Cycle_t current_cycle) {
    total_cycles_++;
    bool has_activity = false;
    if (!clock_tick_logged_ && window_read_debug_) {
        output_->verbose(CALL_INFO, 0, 0,
            "[diag-gas] 核心%d clockTick start stage=%d gas_enable=%d window_mode=%d manual_drive=%d\n",
            core_id_, (int)gas_stage_, gas_enable_ ? 1 : 0, gas_window_mode_ ? 1 : 0,
            gas_manual_window_drive_ ? 1 : 0);
        clock_tick_logged_ = true;
    }
    // 不在时钟顶层阻断loader，允许deliver/recordEdge提前进行；
    // 在发起权重读取时（ReadOrchestrator::issueFromEdges）再检查loader是否就绪。
    // 方案1：优先处理 slice 顺序执行路径；若启用则该函数接管整个周期流程
    if (scheme1_enable_) {
        if (scheme1Tick_()) return false; // 已完成本周期
    }
    // GAS: mark gather window start for this cycle (barrier-based)
    if (gas_enable_ && !gas_window_mode_ && ensureMemoryReady_()) {
        auto *begin_g = new SST::Interfaces::StandardMem::CustomReq(
            new SST::SnnDL::GasOpData(SST::SnnDL::GasOp::BeginGather, /*ss*/0, /*slice*/0, /*tot*/1));
        memory_->send(begin_g);
    }
    if (gas_enable_ && gas_window_mode_ && gas_manual_window_drive_ && gather_buffer_if_) {
        gather_buffer_if_->manualWindowTick();
        manual_gas_counter_++;
        if (!manual_tick_sampled_ && manual_gas_counter_ <= manual_gas_gather_cycles_cfg_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-gas] 核心%d manual_tick stage=%d counter=%" PRIu64 " threshold=%" PRIu64 "\n",
                core_id_, (int)gas_stage_, manual_gas_counter_, manual_gas_gather_cycles_cfg_);
            if (manual_gas_counter_ >= manual_gas_gather_cycles_cfg_) manual_tick_sampled_ = true;
        }
        if (manual_gas_counter_ >= manual_gas_gather_cycles_cfg_) {
            auto *end_g = new SST::Interfaces::StandardMem::CustomReq(
                new SST::SnnDL::GasOpData(SST::SnnDL::GasOp::EndGather, /*ss*/0, /*slice*/0, /*tot*/1));
            memory_->send(end_g);
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-gas] 核心%d手动发出 EndGather (stage=%d cnt=%" PRIu64 ")\n",
                core_id_, (int)gas_stage_, manual_gas_counter_);
            manual_gas_counter_ = 0;
            manual_tick_sampled_ = false;
        }
    }
    // Learning window boundary check (Phase 2)
    if (learning_enabled_ && learn_window_cycles_ > 0) {
        uint64_t win_idx = (uint64_t)(total_cycles_ / learn_window_cycles_);
        if (total_cycles_ == 1 || win_idx != current_window_index_) {
            onWindowBoundary_(win_idx);
            current_window_index_ = win_idx;
        }
    }
    
    // 调试权重验证状态 (仅在前几个周期输出)
    /*
    if (verify_weights_ && total_cycles_ < 10) {
        output_->verbose(CALL_INFO, 2, 0, "🔍 核心%d状态检查: verify_weights=%d, memory_link=%s, memory_ready=%d, cycles=%lu, warmup=%lu\n",
                        core_id_, verify_weights_ ? 1 : 0, memory_link_ ? "yes" : "no", memory_ready_ ? 1 : 0, 
                        total_cycles_, memory_warmup_cycles_);
    }
    */
    
    // BCSR rowptr 预取（延迟到运行期且满足暖机与barrier）
    if (use_bcsr_ && memory_ && memory_ready_ && !bcsr_weights_.isRowptrReady() &&
        !bcsr_weights_.isRowptrReadPending() &&
        total_cycles_ >= memory_warmup_cycles_ &&
        (loader_barrier_cycles_ == 0 || total_cycles_ >= loader_barrier_cycles_)) {
        uint32_t rows = num_neurons_;
        uint32_t br = bcsr_weights_.effectiveBlockRows();
        uint32_t nBlockRows = (rows + br - 1) / br;
        size_t bytes = static_cast<size_t>(nBlockRows + 1) * sizeof(uint32_t);
        auto* read = new SST::Interfaces::StandardMem::Read(bcsr_weights_.rowptrAddr(), bytes);
        auto reqId = read->getID();
        PendingMemoryRequest pmr;
        pmr.request_id = reqId;
        pmr.address = bcsr_weights_.rowptrAddr();
        pmr.size = bytes;
        pmr.bcsr_kind = 1;
        pmr.issue_cycle = total_cycles_;
        stats_reporter_.reportMemoryIssue(bytes, false);
        pending_memory_requests_[reqId] = pmr;
        if (window_read_debug_ && output_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-bcsr] core=%u issue rowptr read addr=0x%llx bytes=%zu\n",
                core_id_, (unsigned long long)bcsr_weights_.rowptrAddr(), bytes);
        }
        memory_->send(read);
        bcsr_weights_.setRowptrReadPending(true);
    }


    // 处理输入脉冲队列
    while (!incoming_spikes_.empty()) {
        SpikeEvent* spike = incoming_spikes_.front();
        incoming_spikes_.pop();
        
        processLocalSpike(spike);
        has_activity = true;
        
        delete spike;
    }

    // Apply窗口内的机会式发起：若BeginApply时未能捕获到上一窗集合，
    // 但在本窗内由于deliverSpike处理使得 prev/curr 集合非空，则本窗内立即发起读取。
    if (gas_stage_ == GasStage::Apply && window_read_enable_ && enable_weight_fetch_ && memory_ && memory_ready_) {
        if (window_reads_issued_this_apply_ < window_read_budget_) {
            const bool have_prev = (!active_pre_prev_window_.empty() && !posts_list_prev_window_.empty());
            const auto* pres_src  = have_prev ? &active_pre_prev_window_ : &active_pre_window_;
            const auto* posts_src = have_prev ? &posts_list_prev_window_ : &posts_list_window_;
            if (pres_src && posts_src && !pres_src->empty() && !posts_src->empty()) {
                read_orchestrator_.issueFromSets(posts_src, pres_src);
            }
        }
    }
    
    // 启动后按需读取权重（受暖机周期与开关控制）
    if (enable_weight_fetch_ && memory_ && memory_ready_ && total_cycles_ >= memory_warmup_cycles_ &&
        (loader_barrier_cycles_ == 0 || total_cycles_ >= loader_barrier_cycles_)) {
        // 示例：周期性读取一个权重并累加到某个神经元上（验证通路）
        // 实际模型应在突触更新处调用 requestWeight
        if (!delayed_read_triggered_) {
            uint32_t pre = 0;
            uint32_t post = 0;
            requestWeight(pre, post, [this, pre, post](float w){
                if (!neuron_states_.empty()) {
                    neuron_states_[post % num_neurons_].v_mem += 0.0f; // 仅拉通读路径，不直接修改
                }
            });
            delayed_read_triggered_ = true;
        }
    }
    // 权重正确性验证：在暖机完成后进行固定次数采样读取，对比 expected_weight_value_
    if (verify_weights_ && memory_ && memory_ready_ && total_cycles_ >= memory_warmup_cycles_ &&
        (loader_barrier_cycles_ == 0 || total_cycles_ >= loader_barrier_cycles_)) {
        if (!verify_started_) {
            verify_started_ = true;
            // 降低默认日志级别
            output_->verbose(CALL_INFO, 3, 0, "🎯 核心%d权重验证启动: 周期=%lu, 暖机阈值=%lu\n", 
                            core_id_, total_cycles_, memory_warmup_cycles_);
        }
        // 每个周期发起至多一个样本，避免拥塞
        if (verify_completed_ < weight_verify_samples_ && verify_requested_ - verify_completed_ < max_outstanding_requests_) {
            uint32_t sample_idx = verify_requested_;
            // 采样若干 (row, col)
            uint32_t row;
            uint32_t col;
            if (verify_cluster_enable_) {
                // 将前weight_verify_samples_个样本聚类到同一cacheline：固定行0，列在一个cacheline范围内循环
                uint32_t fpl = std::max<uint32_t>(1, line_size_bytes_ / (uint32_t)sizeof(float));
                row = 0;
                col = (sample_idx % fpl);
                if (use_post_row_pre_col_) {
                    // 新模式 col 表示 pre_global；为了命中同一CL，选取一小段连续 pre_global
                    // 这里使用全局列 0..fpl-1，足够验证命中
                } else {
                    // 旧模式 col=post_local，同样聚到同一CL
                }
            } else {
                row = (sample_idx * 13) % num_neurons_;                // 本地目标行
                col = use_post_row_pre_col_ ? ((sample_idx * 7) % std::max<uint32_t>(1, weights_cols_))
                                             : ((sample_idx * 7) % num_neurons_);
            }
            // 新模式传参：(pre_global=col, post_local=row)；旧模式：(pre_local=row, post_local=col)
            uint32_t arg0 = use_post_row_pre_col_ ? col : row;
            uint32_t arg1 = use_post_row_pre_col_ ? row : col;
            requestWeight(arg0, arg1, [this, row, col](float w){
                verify_completed_++;
                verify_sum_ += static_cast<double>(w);
                bool mismatch = false;
                if (verify_against_file_ && verify_file_loaded_) {
                    // 使用文件中的期望值（row-major: row*weights_cols_ + col）
                    uint64_t idx = static_cast<uint64_t>(row) * static_cast<uint64_t>(weights_cols_) + static_cast<uint64_t>(col);
                    float expected = 0.0f;
                    if (idx < verify_file_buf_.size()) expected = verify_file_buf_[idx];
                    mismatch = (std::fabs(w - expected) > verify_epsilon_);
                    if (verify_log_each_sample_) {
                        output_->verbose(CALL_INFO, 1, 0,
                            "🔎 权重样本(FILE): row=%u col=%u value=%.6f expected=%.6f diff=%.6f %s\n",
                            row, col, w, expected, std::fabs(w-expected), (mismatch?"MISMATCH":"OK"));
                    }
                } else {
                    // 回退到常数期望（兼容旧行为）
                    mismatch = (std::fabs(w - expected_weight_value_) > verify_epsilon_);
                    if (verify_log_each_sample_) {
                        output_->verbose(CALL_INFO, 1, 0,
                            "🔎 权重样本(CONST): row=%u col=%u value=%.6f expected=%.6f diff=%.6f %s\n",
                            row, col, w, expected_weight_value_, std::fabs(w-expected_weight_value_), (mismatch?"MISMATCH":"OK"));
                    }
                }
                if (mismatch) verify_mismatch_count_++;
                // 详细调试权重读取值（禁用默认回调日志；仅当逐样本日志开启时输出）
                if (verify_log_each_sample_) {
                    output_->verbose(CALL_INFO, 2, 0,
                        "🔎 权重验证回调: core=%d row=%u col=%u value=%.6f sum=%.6f count=%u\n",
                        core_id_, row, col, w, verify_sum_, verify_completed_);
                }
                if (stat_weights_verify_count_) stat_weights_verify_count_->addData(1);
                if (verify_mismatch_count_ && stat_weights_mismatch_count_) stat_weights_mismatch_count_->addData(1);
                if (stat_weights_verify_sum_) stat_weights_verify_sum_->addData(verify_sum_);
            });
            verify_requested_++;
        }
    }

    // BCSR探针：尽力在同一窗口发起一次按边读，扫描一个块内的列，促成 1.0 样本出现（仅诊断；不影响GAS语义）
    if (verify_weights_ && use_bcsr_ && memory_ && memory_ready_ && bcsr_weights_.isRowptrReady() &&
        total_cycles_ >= memory_warmup_cycles_ && (loader_barrier_cycles_ == 0 || total_cycles_ >= loader_barrier_cycles_)) {
        if (!verify_bcsr_done_ && !verify_bcsr_inflight_) {
            uint32_t br = bcsr_weights_.effectiveBlockRows();
            uint32_t bc = bcsr_weights_.effectiveBlockCols();
            uint32_t nBlockRows = (num_neurons_ + br - 1) / br;
            if (!verify_bcsr_started_) {
                const auto& rp = bcsr_weights_.rowptrHost();
                for (uint32_t r = 0; r < nBlockRows; ++r) {
                    if (r + 1 >= rp.size()) break;
                    uint32_t start = rp[r];
                    uint32_t end   = rp[r+1];
                    if (end > start) { verify_bcsr_post_local_ = r * br; verify_bcsr_block_col_ = 0; verify_bcsr_intra_col_ = 0; verify_bcsr_started_ = true; if (output_) output_->verbose(CALL_INFO, 1, 0, "[VERIFY][init-bcsr] core=%d post_local=%u rowptr=(%u,%u)\n", core_id_, verify_bcsr_post_local_, start, end); break; }
                }
                if (!verify_bcsr_started_) verify_bcsr_done_ = true;
            }
            if (verify_bcsr_started_ && !verify_bcsr_done_) {
                uint32_t r = verify_bcsr_post_local_ / br;
                // 若该行的colidx已缓存，解析出第一个block_col；否则先触发一次colidx读取
                if (!verify_bcsr_block_resolved_) {
                    std::vector<uint32_t> cols;
                    if (bcsrRowIndexGet_(r, cols) && !cols.empty()) {
                        verify_bcsr_block_col_ = cols[0];
                        verify_bcsr_block_resolved_ = true;
                        if (output_) output_->verbose(CALL_INFO, 1, 0, "[VERIFY][bcsr-colidx] core=%d row=%u first_block_col=%u\n", core_id_, r, verify_bcsr_block_col_);
                    } else {
                        // 文件直读一次，解析该行第一个块并且定位块内首个非零位置（诊断用途）
                        std::string bin_path = resolveWeightTemplate(node_id_, core_id_);
                        if (!bin_path.empty()) {
                            uint32_t rows=0, colsN=0, brM=0, bcM=0, idxB=0, valB=0, totalB=0;
                            uint64_t rp_off=0, ci_off=0, bd_off=0, id_off=0;
                            std::string meta_path = bin_path + ".meta.json";
                            if (parseBcsrMeta(meta_path, rows, colsN, brM, bcM, idxB, valB, rp_off, ci_off, bd_off, id_off, totalB)) {
                                std::ifstream fin(bin_path, std::ios::binary);
                                if (fin.good()) {
                                    const auto& rp = bcsr_weights_.rowptrHost();
                                    uint32_t start = (r+1 < rp.size() ? rp[r] : 0);
                                    uint32_t end   = (r+1 < rp.size() ? rp[r+1] : start);
                                    uint32_t brEff = (brM? brM : 1), bcEff = (bcM? bcM : 16);
                                    size_t blk_bytes = (size_t)brEff * (size_t)bcEff * (size_t)valB;
                                    uint32_t nblocks = (end > start ? (end - start) : 0);
                                    for (uint32_t j = 0; j < nblocks && !verify_bcsr_block_resolved_; ++j) {
                                        // 读第 j 个块列值
                                        fin.seekg(static_cast<std::streamoff>(ci_off + (size_t)(start + j) * idxB), std::ios::beg);
                                        uint32_t blk_col = 0; if (idxB == 2) { uint16_t v=0; fin.read(reinterpret_cast<char*>(&v), 2); blk_col = v; } else { fin.read(reinterpret_cast<char*>(&blk_col), 4); }
                                        if (!fin.good()) break;
                                        // 读该块数据
                                        fin.seekg(static_cast<std::streamoff>(bd_off + (uint64_t)(start + j) * blk_bytes), std::ios::beg);
                                        std::vector<float> blk(brEff*bcEff, 0.0f);
                                        if (valB == 4) fin.read(reinterpret_cast<char*>(blk.data()), blk_bytes);
                                        if (!fin.good()) break;
                                        for (uint32_t cc = 0; cc < bcEff; ++cc) {
                                            if (std::fabs(blk[cc]) > verify_epsilon_) { verify_bcsr_block_col_ = blk_col; verify_bcsr_intra_col_ = cc; verify_bcsr_block_resolved_ = true; break; }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                uint32_t intra = verify_bcsr_intra_col_;
                uint32_t pre_global = (verify_bcsr_block_resolved_ ? (verify_bcsr_block_col_ * bc + intra) : intra);
                uint32_t post_local = verify_bcsr_post_local_;
                verify_bcsr_inflight_ = true;
                uint32_t block_row = verify_bcsr_post_local_ / br;
                uint32_t bc_eff = bcsr_weights_.effectiveBlockCols();
                uint32_t blk_col = (bc_eff ? (pre_global / bc_eff) : 0);
                size_t block_bytes = bcsr_weights_.blockBytes();
                uint32_t start = 0;
                uint64_t block_addr = 0;
                const auto& rp = bcsr_weights_.rowptrHost();
                if (block_row + 1 < rp.size()) {
                    start = rp[block_row];
                    block_addr = bcsr_weights_.blockDataAddr(start + verify_bcsr_block_col_);
                }
                if (output_) {
                    output_->verbose(CALL_INFO, 0, 0,
                        "[VERIFY][mem-addr] core=%d post=%u pre=%u blk_row=%u blk_col=%u block_addr=0x%llx rowptr_start=%u block_bytes=%zu\n",
                        core_id_, post_local, pre_global, block_row, blk_col,
                        (unsigned long long)block_addr, start, block_bytes);
                }
                requestWeightBCSR(pre_global, post_local, [this, post_local, pre_global](float w){
                    if (output_) {
                        output_->verbose(CALL_INFO, 1, 0, "[VERIFY][probe-bcsr] post=%u pre=%u value=%.6f\n", post_local, pre_global, w);
                    }
                    verify_completed_++;
                    verify_sum_ += static_cast<double>(w);
                    // 仅诊断：读取文件中的同一位置，打印文件权重（不改变语义）
                    do {
                        std::string bin_path = resolveWeightTemplate(node_id_, core_id_);
                        if (bin_path.empty()) break;
                        uint32_t rows=0, colsN=0, brM=0, bcM=0, idxB=0, valB=0, totalB=0;
                        uint64_t rp_off=0, ci_off=0, bd_off=0, id_off=0;
                        std::string meta_path = bin_path + ".meta.json";
                        if (!parseBcsrMeta(meta_path, rows, colsN, brM, bcM, idxB, valB, rp_off, ci_off, bd_off, id_off, totalB)) break;
                        std::ifstream fin(bin_path, std::ios::binary);
                        if (!fin.good()) break;
                        uint32_t brEff = (brM? brM : 1), bcEff = (bcM? bcM : 16);
                        uint32_t r = (brEff? (post_local / brEff) : 0);
                        // 直接从文件读取 rowptr
                        fin.seekg(static_cast<std::streamoff>(rp_off + (size_t)r * sizeof(uint32_t)), std::ios::beg);
                        uint32_t start=0, end=0; fin.read(reinterpret_cast<char*>(&start), 4); fin.read(reinterpret_cast<char*>(&end), 4);
                        if (!fin.good() || end <= start) break;
                        uint32_t total_blocks = end - start;
                        if (total_blocks > 1'000'000) break; // 防止异常数据
                        uint32_t blk_col_target = (bcEff? (pre_global / bcEff) : 0);
                        int idx_in_row = -1;
                        for (uint32_t j=0; j < total_blocks; ++j) {
                            fin.seekg(static_cast<std::streamoff>(ci_off + (size_t)(start + j) * idxB), std::ios::beg);
                            uint32_t blk_col = 0; if (idxB == 2) { uint16_t v=0; fin.read(reinterpret_cast<char*>(&v), 2); blk_col = v; } else { fin.read(reinterpret_cast<char*>(&blk_col), 4); }
                            if (!fin.good()) break;
                            if (blk_col == blk_col_target) { idx_in_row = (int)j; break; }
                        }
                        if (idx_in_row < 0) break;
                        // 读该块数据并取 intra 列值
                        size_t blk_bytes = (size_t)brEff * (size_t)bcEff * (size_t)valB;
                        fin.seekg(static_cast<std::streamoff>(bd_off + (uint64_t)(start + (uint32_t)idx_in_row) * blk_bytes), std::ios::beg);
                        std::vector<float> blk(brEff*bcEff, 0.0f);
                        if (valB == 4) fin.read(reinterpret_cast<char*>(blk.data()), blk_bytes);
                        if (!fin.good()) break;
                        uint32_t intra = (bcEff? (pre_global % bcEff) : 0);
                        float fv = (intra < blk.size()? blk[intra] : 0.0f);
                        if (output_) output_->verbose(CALL_INFO, 1, 0, "[VERIFY][file-probe] post=%u pre=%u file_value=%.6f mem_value=%.6f\n", post_local, pre_global, fv, w);
                    } while(0);
                    if (std::fabs(w) > verify_epsilon_) {
                        verify_bcsr_done_ = true;
                    } else {
                        uint32_t bc = bcsr_weights_.effectiveBlockCols();
                        if (verify_bcsr_intra_col_ + 1 < bc) {
                            // 立即尝试下一列
                            verify_bcsr_intra_col_++;
                            uint32_t next_pre = (verify_bcsr_block_resolved_ ? (verify_bcsr_block_col_ * bc + verify_bcsr_intra_col_) : verify_bcsr_intra_col_);
                            requestWeightBCSR(next_pre, post_local, [this, post_local, next_pre](float w2){
                                if (output_) output_->verbose(CALL_INFO, 1, 0, "[VERIFY][probe-bcsr] post=%u pre=%u value=%.6f\n", post_local, next_pre, w2);
                                verify_completed_++;
                                verify_sum_ += static_cast<double>(w2);
                                if (std::fabs(w2) > verify_epsilon_) {
                                    verify_bcsr_done_ = true;
                                } else {
                                    // 若仍未命中，则留给下一tick继续
                                }
                                verify_bcsr_inflight_ = false;
                            });
                            return; // 由内层回调负责清理 inflight
                        } else {
                            verify_bcsr_done_ = true;
                        }
                    }
                    verify_bcsr_inflight_ = false;
                });
            }
        }
    }

    // 更新神经元状态（复用SnnPE逻辑）
    updateNeuronStates();
    
    // 在 GAS 窗口累加语义下（apply_acc_enable_ && gas_window_mode_），发放应当发生在 Scatter 阶段，
    // 避免在常规时钟路径重复触发。仅当未启用窗口语义时，才在每周期检查发放。
    if (!(apply_acc_enable_ && gas_window_mode_)) {
        for (uint32_t i = 0; i < num_neurons_; i++) {
            checkAndFireSpike(i);
        }
    }
    
    if (has_activity) {
        active_cycles_++;
    }
    // GAS: end of gather window for this cycle; GatherBufferIF will reply upon '读齐'
    if (gas_enable_ && !gas_window_mode_ && ensureMemoryReady_()) {
        auto *end_g = new SST::Interfaces::StandardMem::CustomReq(
            new SST::SnnDL::GasOpData(SST::SnnDL::GasOp::EndGather, /*ss*/0, /*slice*/0, /*tot*/1));
        memory_->send(end_g);
    }

    return false;  // 继续时钟
}

void SnnPESubComponent::forceEndGather() {
    if (!(gas_enable_ && gas_window_mode_ && gas_manual_window_drive_ && memory_ && memory_ready_)) return;
    auto *end_g = new SST::Interfaces::StandardMem::CustomReq(
        new SST::SnnDL::GasOpData(SST::SnnDL::GasOp::EndGather, /*ss*/0, /*slice*/0, /*tot*/1));
    memory_->send(end_g);
    if (window_read_debug_ && output_) {
        output_->verbose(CALL_INFO, 0, 0, "[diag-gas] 核心%d 手动触发 EndGather\n", core_id_);
    }
}
void SnnPESubComponent::deliverSpike(SpikeEvent* spike) {
    if (!spike) return;

    spike->clearLocalCache();
    output_->verbose(CALL_INFO, 4, 0, "📨 核心%d接收脉冲: 源全局ID=%u, 目标全局ID=%u, 目标神经元=%u, 权重%.3f\n",
                    core_id_, spike->getSourceNeuron(), spike->getDestinationNeuron(), spike->getDestinationNeuron(), spike->getWeight());

    // 将脉冲加入队列，在时钟周期中处理
    incoming_spikes_.push(spike);

    if (window_read_enable_) {
        if (posts_seen_window_.size() != num_neurons_) posts_seen_window_.assign(num_neurons_, 0);
        uint32_t dest = spike->getDestinationNeuron();
        uint32_t post_local = dest;
        bool post_local_valid = true;
        if (dest >= num_neurons_) {
            if (dest >= global_neuron_base_ && dest < global_neuron_base_ + num_neurons_) {
                post_local = static_cast<uint32_t>(dest - global_neuron_base_);
            } else {
                post_local = UINT32_MAX;
                post_local_valid = false;
            }
        }
        if (post_local_valid) {
            spike->setCachedPostLocal(post_local);
        }
        if (!use_post_row_pre_col_) {
            spike->setCachedPreLocal(mapPreGlobalToLocal_(spike->getSourceNeuron()));
        }
        static const uint32_t kLogLimit = 8;
        if (debug_window_log_count_ < kLogLimit) {
            output_->verbose(CALL_INFO, 1, 0,
                "[DeliverDebug] core=%d dest=%u base=%" PRIu64 " num=%u -> post_local=%u queue=%zu\n",
                core_id_, dest, (uint64_t)global_neuron_base_, num_neurons_, post_local, incoming_spikes_.size());
            debug_window_log_count_++;
        }
        if (post_local < num_neurons_ && !posts_seen_window_[post_local]) {
            posts_seen_window_[post_local] = 1;
            posts_list_window_.push_back(post_local);
        }
        active_pre_window_.insert(spike->getSourceNeuron());
        recordActivePre_(spike->getSourceNeuron());
    }
    
    // 更新两种统计：SST统计对象和内部计数器
    stat_spikes_received_->addData(1);
    count_spikes_received_++;
    
    // Debug output disabled to prevent excessive logging
    // printf("DEBUG: SnnPESubComponent核心%d接收脉冲，内部计数器更新: count_spikes_received_=%lu\n", 
    //        core_id_, count_spikes_received_);
}

void SnnPESubComponent::resetMembraneState(float v_rest_value) {
    if (use_soa_state_) {
        std::fill(soa_v_mem_.begin(), soa_v_mem_.end(), v_rest_value);
        std::fill(soa_refrac_.begin(), soa_refrac_.end(), 0u);
        std::fill(soa_last_spike_.begin(), soa_last_spike_.end(), (Cycle_t)0);
    } else {
        for (auto& st : neuron_states_) {
            st.v_mem = v_rest_value;
            st.refractory_timer = 0;
            st.last_spike_time = 0;
        }
    }
    if (!fired_this_window_.empty()) {
        std::fill(fired_this_window_.begin(), fired_this_window_.end(), 0);
    }
    accReset_();
}

void SnnPESubComponent::setMemoryLink(SST::Link* link) {
    memory_link_ = link;
    
    // ★ 关键修正：直接使用提供的Link进行内存操作 ★
    if (memory_link_) {
        // output_->verbose(CALL_INFO, 2, 0, "🔗 核心%d设置内存连接成功\n", core_id_);
        memory_ready_ = true;  // 标记内存已准备就绪
    } else {
        output_->verbose(CALL_INFO, 2, 0, "🔗 核心%d设置内存连接失败 (link=nullptr)\n", core_id_);
        memory_ready_ = false;
    }
}

bool SnnPESubComponent::hasWork() const {
    if (!incoming_spikes_.empty()) return true;
    if (use_soa_state_) {
        return std::any_of(soa_v_mem_.begin(), soa_v_mem_.end(), [](float v){ return v > 0.1f; });
    }
    return std::any_of(neuron_states_.begin(), neuron_states_.end(),
                      [](const NeuronState& state) { return state.v_mem > 0.1f; });
}

double SnnPESubComponent::getUtilization() const {
    if (total_cycles_ == 0) return 0.0;
    return static_cast<double>(active_cycles_) / static_cast<double>(total_cycles_);
}

void SnnPESubComponent::getStatistics(std::map<std::string, uint64_t>& stats) const {
    // 使用内部计数器而不是getCollectionCount()来获取正确的累计值
    stats["spikes_received"] = count_spikes_received_;
    stats["spikes_generated"] = count_spikes_generated_;
    stats["neurons_fired"] = count_neurons_fired_;
    stats["memory_requests"] = count_memory_requests_;
    stats["total_cycles"] = total_cycles_;
    stats["active_cycles"] = active_cycles_;
    stats["cycles_update_neuron"] = count_cycles_update_neuron_;
    stats["synaptic_accesses"] = count_synaptic_accesses_;
}

// ===== 核心计算方法（复用SnnPE实现）=====

void SnnPESubComponent::updateNeuronStates() {
#ifdef SNNDL_ENABLE_PROFILING
    if (profiler_enabled_) { SNNDL_PROFILE_FUNCTION(profiler_); }
#endif
    if (use_aosoa_state_) {
        updateNeuronStatesAoSoA_();
    } else if (use_soa_state_) {
        updateNeuronStatesSoA_();
    } else {
        updateNeuronStatesAoS_();
    }
}

void SnnPESubComponent::updateNeuronStatesAoS_() {
    if (tau_mem_ <= 0.0f) return;
    const float decay = std::exp(-1.0f / tau_mem_);
    uint64_t local_updates = 0;
    for (uint32_t i = 0; i < num_neurons_; i++) {
        auto& neuron = neuron_states_[i];
        local_updates++;
        if (neuron.refractory_timer > 0) {
            neuron.refractory_timer--;
            continue;
        }
        if (neuron.v_mem > v_rest_) {
            neuron.v_mem = v_rest_ + (neuron.v_mem - v_rest_) * decay;
        }
    }
    count_cycles_update_neuron_ += local_updates;
    if (stat_cycles_update_neuron_) stat_cycles_update_neuron_->addData(local_updates);
}

void SnnPESubComponent::updateNeuronStatesSoA_() {
    if (tau_mem_ <= 0.0f) return;
    const float decay = std::exp(-1.0f / tau_mem_);
    uint64_t local_updates = 0;
    for (uint32_t i = 0; i < num_neurons_; ++i) {
        local_updates++;
        if (soa_refrac_[i] > 0) {
            soa_refrac_[i]--;
            continue;
        }
        float v = soa_v_mem_[i];
        if (v > v_rest_) {
            soa_v_mem_[i] = v_rest_ + (v - v_rest_) * decay;
        }
    }
    count_cycles_update_neuron_ += local_updates;
    if (stat_cycles_update_neuron_) stat_cycles_update_neuron_->addData(local_updates);
}

void SnnPESubComponent::updateNeuronStatesAoSoA_() {
    if (tau_mem_ <= 0.0f) return;
    const float decay = std::exp(-1.0f / tau_mem_);
    const uint32_t lane = (aosoa_block_rows_ > 0) ? aosoa_block_rows_ : 16;
    uint64_t local_updates = 0;
    for (uint32_t block = 0; block < num_neurons_; block += lane) {
        uint32_t limit = std::min(num_neurons_, block + lane);
        for (uint32_t idx = block; idx < limit; ++idx) {
            local_updates++;
            if (soa_refrac_[idx] > 0) {
                soa_refrac_[idx]--;
                continue;
            }
            float v = soa_v_mem_[idx];
            if (v > v_rest_) {
                soa_v_mem_[idx] = v_rest_ + (v - v_rest_) * decay;
            }
        }
    }
    count_cycles_update_neuron_ += local_updates;
    if (stat_cycles_update_neuron_) stat_cycles_update_neuron_->addData(local_updates);
}

void SnnPESubComponent::checkAndFireSpike(uint32_t neuron_idx) {
    if (use_aosoa_state_) {
        checkAndFireSpikeAoSoA_(neuron_idx);
    } else if (use_soa_state_) {
        checkAndFireSpikeSoA_(neuron_idx);
    } else {
        checkAndFireSpikeAoS_(neuron_idx);
    }
}

void SnnPESubComponent::checkAndFireSpikeAoS_(uint32_t neuron_idx) {
    if (neuron_idx >= num_neurons_) return;
    // 严格窗口语义：启用窗口累加时，仅允许在 Scatter 阶段触发发放
    if (apply_acc_enable_ && gas_window_mode_ && gas_stage_ != GasStage::Scatter) return;
    // 每窗口一次发放门控：同一窗口内每个神经元最多发一次
    if (apply_acc_enable_ && gas_window_mode_) {
        if (fired_this_window_.size() != num_neurons_) fired_this_window_.assign(num_neurons_, 0);
        if (fired_this_window_[neuron_idx]) return;
    }
    auto& neuron = neuron_states_[neuron_idx];
    float v_before = neuron.v_mem;
    bool will_fire = neuron_model_ ? neuron_model_->shouldFire(neuron_idx, neuron)
                                   : (neuron.v_mem >= v_thresh_ && neuron.refractory_timer == 0);
    if (!will_fire) return;
    if (neuron_model_) {
        neuron_model_->onFired(neuron_idx, neuron);
    } else {
        neuron.v_mem = v_reset_;
        neuron.refractory_timer = t_ref_;
    }
    neuron.last_spike_time = total_cycles_;
    if (apply_acc_enable_ && gas_window_mode_) {
        fired_this_window_[neuron_idx] = 1;
    }
    handleNeuronFire_(neuron_idx, v_before, neuron.v_mem);
}

void SnnPESubComponent::checkAndFireSpikeSoA_(uint32_t neuron_idx) {
    if (neuron_idx >= num_neurons_) return;
    // 严格窗口语义：启用窗口累加时，仅允许在 Scatter 阶段触发发放
    if (apply_acc_enable_ && gas_window_mode_ && gas_stage_ != GasStage::Scatter) return;
    // 每窗口一次发放门控：同一窗口内每个神经元最多发一次
    if (apply_acc_enable_ && gas_window_mode_) {
        if (fired_this_window_.size() != num_neurons_) fired_this_window_.assign(num_neurons_, 0);
        if (fired_this_window_[neuron_idx]) return;
    }
    NeuronState tmp(v_rest_);
    tmp.v_mem = soa_v_mem_[neuron_idx];
    tmp.refractory_timer = soa_refrac_[neuron_idx];
    tmp.last_spike_time = soa_last_spike_[neuron_idx];
    float v_before = tmp.v_mem;
    bool will_fire = neuron_model_ ? neuron_model_->shouldFire(neuron_idx, tmp)
                                   : (tmp.v_mem >= v_thresh_ && tmp.refractory_timer == 0);
    if (!will_fire) {
        if (neuron_model_) {
            soa_v_mem_[neuron_idx] = tmp.v_mem;
            soa_refrac_[neuron_idx] = tmp.refractory_timer;
            soa_last_spike_[neuron_idx] = tmp.last_spike_time;
        }
        return;
    }
    if (neuron_model_) {
        neuron_model_->onFired(neuron_idx, tmp);
        soa_v_mem_[neuron_idx] = tmp.v_mem;
        soa_refrac_[neuron_idx] = tmp.refractory_timer;
        soa_last_spike_[neuron_idx] = tmp.last_spike_time;
    } else {
        soa_v_mem_[neuron_idx] = v_reset_;
        soa_refrac_[neuron_idx] = t_ref_;
        soa_last_spike_[neuron_idx] = total_cycles_;
    }
    if (apply_acc_enable_ && gas_window_mode_) {
        fired_this_window_[neuron_idx] = 1;
    }
    handleNeuronFire_(neuron_idx, v_before, soa_v_mem_[neuron_idx]);
}

void SnnPESubComponent::checkAndFireSpikeAoSoA_(uint32_t neuron_idx) {
    // AoSoA 逻辑沿用 SoA 单元素处理，仅在迭代粒度上与 SoA 区分
    // 严格窗口语义 gating 由 checkAndFireSpikeSoA_ 内部完成
    checkAndFireSpikeSoA_(neuron_idx);
}

void SnnPESubComponent::handleNeuronFire_(uint32_t neuron_idx, float v_before, float v_after) {
    // Phase 1: record spike timeline (lightweight, default on when learning enabled)
    if (learning_enabled_ && record_spike_times_) {
        uint32_t g_id = static_cast<uint32_t>(global_neuron_base_ + neuron_idx);
        float v_fire = record_membrane_ ? v_before : 0.0f;
        spike_history_.emplace_back(g_id, static_cast<uint64_t>(total_cycles_), v_fire);
    }

    stat_neurons_fired_->addData(1);
    stat_spikes_generated_->addData(1);
    count_neurons_fired_++;
    count_spikes_generated_++;
    if (apply_acc_enable_ && gas_window_mode_) {
        window_spikes_all_++;
    }
    // 去重发放统计：首次发放上报到父PE聚合
    if (neuron_idx < fired_ever_.size() && fired_ever_[neuron_idx] == 0) {
        fired_ever_[neuron_idx] = 1;
        if (parent_) {
            if (auto* pe = parent_pe_cached_) {
                // 累加一次唯一发放
                pe->accumulateUniqueNeuronFired(1);
            }
        }
    }

    output_->verbose(CALL_INFO, 3, 0, "🔥 核心%d神经元%d发放脉冲! v_before=%.3f -> v_after=%.3f\n",
                    core_id_, neuron_idx, v_before, v_after);

    uint32_t source_global = static_cast<uint32_t>(global_neuron_base_ + neuron_idx);
    if (routing_weight_driven_) {
        // 在线门控事件覆盖（仅输入层或全局）
        bool applied_gating = false;
        if (gating_event_mode_) {
            bool scope_ok = !gating_scope_inputs_only_ ? true : (node_id_ <= 3);
            if (scope_ok) {
                    auto itg = gating_cache_.find(source_global);
                    if (itg != gating_cache_.end() && total_cycles_ <= itg->second.expire_cycle) {
                        const auto& dpes = itg->second.dest_pes;
                        if (!dpes.empty()) {
                            if (stat_fanout_per_spike_) stat_fanout_per_spike_->addData((uint64_t)dpes.size());
                            for (uint32_t dpe : dpes) {
                                uint32_t dest_global = dpe * num_neurons_ + neuron_idx; // 同row映射
                                uint32_t dest_node = dpe;
                                float output_weight = 1.0f;
                                SpikeEvent* output_spike = new SpikeEvent(
                                    source_global,
                                    dest_global,
                                    dest_node,
                                    output_weight,
                                    total_cycles_);
                                if (parent_) parent_->sendSpike(output_spike); else delete output_spike;
                            }
                            applied_gating = true;
                            output_->verbose(CALL_INFO, 2, 0, "🎯 门控命中: 源g=%u, 目的PE数=%zu\n", source_global, dpes.size());
                        }
                    }
                }
            }
            if (applied_gating) return; // 门控覆盖完成
            // 优先使用共享路由表，回退到本地表
            const RouteMap* route_tbl = routes_shared_ ? routes_shared_.get() : &routes_by_source_;
            auto itrt = route_tbl->find(source_global);
            if (itrt != route_tbl->end()) {
                const auto& dests = itrt->second;
                if (stat_fanout_per_spike_) stat_fanout_per_spike_->addData((uint64_t)dests.size());
                for (uint32_t dest_global : dests) {
                    // 注意：这里要用每个PE的神经元数来计算目的节点，而不是本core行数
                    uint32_t denom = (neurons_per_pe_cfg_ > 0) ? neurons_per_pe_cfg_ : num_neurons_;
                    uint32_t dest_node = dest_global / denom;
                    float output_weight = 1.0f;
                    SpikeEvent* output_spike = new SpikeEvent(
                        source_global,
                        dest_global,
                        dest_node,
                        output_weight,
                        total_cycles_);
                    if (parent_) parent_->sendSpike(output_spike); else delete output_spike;
                }
            if (log_weight_details_) {
                output_->verbose(CALL_INFO, 2, 0, "🌐 权重驱动扇出: 源g=%u, 目的数=%zu\n", source_global, dests.size());
            }
            } else {
                // 无路由目标，静默
            }
    } else {
        // 原固定层间映射
        uint32_t target_neuron = 0;
        uint32_t target_node = node_id_;
        float output_weight = 1.0f;
        if (node_id_ >= 0 && node_id_ <= 3) {
            uint32_t target_hidden_base = (node_id_ < 2) ? 4 : 8;
            uint32_t target_hidden_node = target_hidden_base + (node_id_ % 2) * 2 + (neuron_idx % 2);
            target_node = target_hidden_node;
            target_neuron = target_hidden_node * 16 + neuron_idx;
            output_->verbose(CALL_INFO, 2, 0, "🔥 输入层节点%d神经元%d -> 隐藏层节点%d神经元%d\n",
                             node_id_, neuron_idx, target_node, target_neuron);
        } else if (node_id_ >= 4 && node_id_ <= 11) {
            uint32_t target_output_node = 12 + ((node_id_ - 4) / 2);
            target_node = target_output_node;
            target_neuron = target_output_node * 16 + (neuron_idx % 16);
            output_->verbose(CALL_INFO, 2, 0, "🔥 隐藏层节点%d神经元%d -> 输出层节点%d神经元%d\n",
                             node_id_, neuron_idx, target_node, target_neuron);
        } else {
            output_->verbose(CALL_INFO, 2, 0, "🔥 输出层节点%d神经元%d发放，不发送外部脉冲\n",
                             node_id_, neuron_idx);
            return;
        }
        SpikeEvent* output_spike = new SpikeEvent(
            source_global,
            target_neuron,
            target_node,
            output_weight,
            total_cycles_);
        if (parent_) parent_->sendSpike(output_spike); else delete output_spike;
    }
}

// === Phase 2: Learning window handling and error loading ===
void SnnPESubComponent::onWindowBoundary_(uint64_t window_idx) {
    // For now, just (re)load error buffer from file each window if template provided.
    // Future: could switch to per-epoch/batch semantics.
    (void)window_idx;
    loadErrorsForWindow_(window_idx);
    // Optional: prune gradient map if exceeds cap
    if (grad_accum_limit_ > 0 && local_grad_.size() > grad_accum_limit_) {
        // Minimal policy: clear all to keep memory bounded
        local_grad_.clear();
    }
    // Apply updates periodically if enabled
    if (apply_writeback_ && apply_every_n_windows_ > 0 && (window_idx % apply_every_n_windows_ == 0)) {
        if (memory_ && memory_ready_) {
            applyLocalWeightUpdates_();
        } else {
            output_->verbose(CALL_INFO, 1, 0, "⚠️ 学习: 写回启用但内存接口不可用，跳过本窗写回\n");
        }
    }
}

void SnnPESubComponent::loadErrorsForWindow_(uint64_t window_idx) {
    if (!learning_enabled_) return;
    if (error_file_template_.empty()) {
        // No file specified -> keep zeros (do nothing)
        if (error_buffer_.size() != num_neurons_) error_buffer_.assign(num_neurons_, 0.0f);
        return;
    }
    std::string path = replacePlaceholders_(error_file_template_);
    // Also support {win} placeholder
    {
        size_t p = path.find("{win}");
        if (p != std::string::npos) {
            std::string w = std::to_string(window_idx);
            path.replace(p, 5, w);
        }
    }
    std::ifstream fin(path);
    if (!fin.good()) {
        output_->verbose(CALL_INFO, 1, 0, "⚠️ 学习: 无法打开误差文件 %s, 本窗使用0误差\n", path.c_str());
        if (error_buffer_.size() != num_neurons_) error_buffer_.assign(num_neurons_, 0.0f);
        else std::fill(error_buffer_.begin(), error_buffer_.end(), 0.0f);
        return;
    }
    // Parse: supports lines of "id val" or single float per line (indexed by line)
    std::vector<float> tmp(num_neurons_, 0.0f);
    std::string line;
    uint32_t line_idx = 0;
    while (std::getline(fin, line)) {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream iss(line);
        uint32_t id = UINT32_MAX; float val = 0.0f;
        if ((iss >> id >> val)) {
            if (id < num_neurons_) tmp[id] = val;
        } else {
            iss.clear(); iss.str(line);
            if (iss >> val) {
                if (line_idx < num_neurons_) tmp[line_idx] = val;
                line_idx++;
            }
        }
    }
    error_buffer_.swap(tmp);
}

void SnnPESubComponent::applyLocalWeightUpdates_() {
    if (!apply_writeback_) return;
    if (local_grad_.empty()) return;
    const uint32_t width = use_post_row_pre_col_ ? weights_cols_ : num_neurons_;
    const size_t bytes_per_float = sizeof(float);
    uint64_t total_writes = 0;

    // Iterate over accumulated gradients and issue per-weight writes.
    // Note: this minimal implementation only updates weights present in cache
    // to compute absolute new values; others are skipped to avoid corrupting memory.
    size_t skipped_uncached = 0;
    for (const auto& kv : local_grad_) {
        uint64_t key = kv.first;
        float grad = kv.second;
        uint32_t row = (uint32_t)(key / width);
        uint32_t col = (uint32_t)(key % width);
        (void)row; (void)col;

        float old_w = 0.0f;
        if (!weightCacheTryGet_(key, old_w)) {
            skipped_uncached++;
            continue; // skip if not in cache
        }
        float new_w = old_w - learning_rate_ * grad;
        if (weight_decay_ != 0.0f) {
            new_w -= weight_decay_ * old_w; // simple L2 decay
        }

        uint64_t offset = key; // linear index within matrix
        uint64_t addr = base_addr_ + offset * bytes_per_float;
        std::vector<uint8_t> data(bytes_per_float);
        std::memcpy(data.data(), &new_w, bytes_per_float);
        auto* w = new SST::Interfaces::StandardMem::Write(addr, data.size(), data, false);
        stats_reporter_.reportMemoryIssue(data.size(), false);
        memory_->send(w);
        total_writes++;

        // Update cache with new value
        weightCacheStore_(key, new_w);
    }
    // Clear gradients after applying
    local_grad_.clear();
    output_->verbose(CALL_INFO, 1, 0, "📝 学习: 写回完成 writes=%" PRIu64 ", 跳过(未缓存)=%zu\n", total_writes, skipped_uncached);
}

void SnnPESubComponent::processLocalSpike(SpikeEvent* spike_event) {
#ifdef SNNDL_ENABLE_PROFILING
    if (profiler_enabled_) { SNNDL_PROFILE_FUNCTION(profiler_); }
#endif
    // 复用SnnPE的本地脉冲处理逻辑
    if (!spike_event) return;
    
    uint32_t dest = spike_event->getDestinationNeuron();
    switch (gas_stage_) {
        case GasStage::Gather: diag_spikes_stage_gather_++; break;
        case GasStage::Apply:  diag_spikes_stage_apply_++;  break;
        case GasStage::Scatter:diag_spikes_stage_scatter_++;break;
        default: diag_spikes_stage_idle_++; break;
    }
    uint32_t target_neuron = dest;
    bool used_cached_post = false;
    if (spike_event->hasCachedPostLocal()) {
        target_neuron = spike_event->getCachedPostLocal();
        used_cached_post = true;
    }
    // 全局ID → 本地ID 映射
    if (!used_cached_post && dest >= num_neurons_) {
        if (dest >= global_neuron_base_ && dest < global_neuron_base_ + num_neurons_) {
            target_neuron = static_cast<uint32_t>(dest - global_neuron_base_);
        } else {
            output_->verbose(CALL_INFO, 2, 0, "⚠️ 核心%d收到无法映射的目标神经元%d的脉冲\n", core_id_, dest);
            return;
        }
    }

#ifdef SNNDL_ENABLE_DEBUG_LOG
    if (window_read_debug_) {
        output_->verbose(CALL_INFO, 0, 0,
            "[diag-spike] core=%d stage=%d pre_g=%u post_g=%u dest_l=%u queue=%zu\n",
            core_id_, (int)gas_stage_, spike_event->getSourceNeuron(), spike_event->getDestinationNeuron(),
            target_neuron, incoming_spikes_.size());
    }
#endif
    
    uint32_t refr = getRefrac_(target_neuron);
    if (refr > 0) {
        output_->verbose(CALL_INFO, 4, 0, "⚠️ 核心%d神经元%d在不应期，忽略脉冲\n", 
                        core_id_, target_neuron);
        return;
    }
    float neuron_v = getMem_(target_neuron);
    Cycle_t last_spike_time = getLastSpike_(target_neuron);
    
    // 使用权重缓存/按需读取
    float weight = 0.0f;
    bool have_mem_weight = false;
    const bool logDetail = log_weight_details_ || enable_detailed_map_log_;
    if (enable_weight_fetch_ && memory_ && memory_ready_) {
        uint32_t pre_global = spike_event->getSourceNeuron();
        uint32_t post_global = spike_event->getDestinationNeuron();
        uint32_t post_local = target_neuron;
        if (spike_event->hasCachedPostLocal()) {
            post_local = spike_event->getCachedPostLocal();
        } else if (post_global >= global_neuron_base_ && post_global < global_neuron_base_ + num_neurons_) {
            post_local = static_cast<uint32_t>(post_global - global_neuron_base_);
        }
        recordEdge_(post_local, pre_global);
        uint32_t req_pre_param = 0;
        uint32_t req_post_param = 0;
        uint64_t cache_key = 0;
        const bool allow_remap = !use_post_row_pre_col_;
        if (weight_accessor_.resolve(pre_global, post_local, req_pre_param, req_post_param, cache_key, allow_remap)) {
            if (use_post_row_pre_col_) {
                recordActivePre_(pre_global);
            }
            float cached = 0.0f;
            if (weightCacheTryGet_(cache_key, cached)) {
                weight = cached;
                if (readresp_zero_fallback_ && weight == 0.0f) weight = init_default_weight_;
                have_mem_weight = true;
                stats_reporter_.reportCacheAccess(true);
                if (logDetail && !first_cache_hit_logged_) {
                    if (use_post_row_pre_col_) {
                        SNNDL_LOG(2, "🟢 首次命中(全网): row(post_l)=%u, col(pre_g)=%u, key=%" PRIu64 ", weight=%.3f\n",
                            post_local, pre_global, cache_key, weight);
                    } else {
                        uint32_t pre_local_dbg = spike_event->hasCachedPreLocal()
                            ? spike_event->getCachedPreLocal()
                            : mapPreGlobalToLocal_(pre_global);
                        SNNDL_LOG(2, "🟢 首次命中: pre_l=%u, post_l=%u, key=%" PRIu64 ", weight=%.3f\n",
                            pre_local_dbg, post_local, cache_key, weight);
                    }
                    first_cache_hit_logged_ = true;
                }
            } else if (!window_read_enable_ && outstanding_requests_ < max_outstanding_requests_) {
                stats_reporter_.reportCacheAccess(false);
                outstanding_requests_++;
                stats_reporter_.updatePendingPeak(outstanding_requests_);
                requestWeight(req_pre_param, req_post_param, [this, cache_key](float w){
                    weightCacheStore_(cache_key, w);
                    if (outstanding_requests_ > 0) outstanding_requests_--;
                });
                if (logDetail && !first_cache_miss_logged_) {
                    if (use_post_row_pre_col_) {
                        SNNDL_LOG(2, "🟡 首次未命中并发起读(全网): row(post_l)=%u, col(pre_g)=%u, key=%" PRIu64 "\n",
                            post_local, pre_global, cache_key);
                    } else {
                        uint32_t pre_local_dbg = spike_event->hasCachedPreLocal()
                            ? spike_event->getCachedPreLocal()
                            : mapPreGlobalToLocal_(pre_global);
                        SNNDL_LOG(2, "🟡 首次未命中并发起读: pre_l=%u, post_l=%u, key=%" PRIu64 "\n",
                            pre_local_dbg, post_local, cache_key);
                    }
                    first_cache_miss_logged_ = true;
                }
            } else if (window_read_enable_) {
                stats_reporter_.reportCacheAccess(false);
            }
        } else {
#if SNNDL_DEBUG_ENABLED
            if (window_read_debug_) {
                SNNDL_DEBUG_LOG(0, "[diag-edge-resolve] core=%d skipped pre=%u post=%u (resolve失败)\n",
                    core_id_, pre_global, post_local);
            }
#endif
        }
    } else {
#ifdef SNNDL_ENABLE_DEBUG_LOG
        if (window_read_debug_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-edge-gate] core=%d skip recordEdge: ewf=%d mem=%s ready=%d stage=%d\n",
                core_id_, enable_weight_fetch_ ? 1 : 0, memory_ ? "set" : "null", memory_ready_ ? 1 : 0, (int)gas_stage_);
        }
#endif
    }
    // 记录本窗触达的post（仅用于窗口读发起；不改变语义）
    // window_read_enable_ 情况下，posts_seen_window_ 已在 deliverSpike 时打点，
    // 这里无需重复标记，但为稳妥保留一次兜底，判断是否已有记录
    if (window_read_enable_) {
        if (posts_seen_window_.size() != num_neurons_) posts_seen_window_.assign(num_neurons_, 0);
        if (target_neuron < num_neurons_ && !posts_seen_window_[target_neuron]) {
            posts_seen_window_[target_neuron] = 1;
            posts_list_window_.push_back(target_neuron);
            active_pre_window_.insert(spike_event->getSourceNeuron());
            recordActivePre_(spike_event->getSourceNeuron());
        }
    }
    if (!have_mem_weight) {
        // 回退策略：可选择使用事件权重，或直接使用默认初始权重（与内存一致）
        if (use_event_weight_fallback_) {
            weight = spike_event->getWeight();
            if (!event_weight_fallback_warned_) {
                output_->verbose(CALL_INFO, 1, 0, "⚠️ 核心%d启用事件权重回退，事件权重=%.3f\n", core_id_, weight);
                event_weight_fallback_warned_ = true;
            }
        } else {
            weight = 0.0f;
        }
    }
    if (apply_acc_enable_) {
        // End-to-end semantics: accumulate now if we already have the weight (cache hit)
        if (have_mem_weight && use_post_row_pre_col_) {
            uint32_t post_local = target_neuron;
            // Debug probe (Option 2/3): cache-hit accumulation path
            if (output_) {
                output_->verbose(CALL_INFO, 2, 0,
                    "[GAS][Delta][cache-hit] core=%u post=%u dv=%.6f (key-cached)\n",
                    core_id_, post_local, weight);
            }
            // 诊断：累计ΔV
            diag_dv_sum_window_ += (double)weight;
            if (weight != 0.0f) diag_dv_updates_nonzero_++;
            accUpdate_(post_local, weight);
        }
        recordSynapticAccess_();
        return;
    }
    if (neuron_model_) {
        NeuronState tmp(v_rest_);
        tmp.v_mem = neuron_v;
        tmp.refractory_timer = refr;
        tmp.last_spike_time = last_spike_time;
        neuron_model_->onSynapticEvent(target_neuron, weight, tmp);
        neuron_v = tmp.v_mem;
        setMem_(target_neuron, tmp.v_mem);
        setRefrac_(target_neuron, tmp.refractory_timer);
        setLastSpike_(target_neuron, tmp.last_spike_time);
    } else {
        neuron_v += weight;
        setMem_(target_neuron, neuron_v);
    }
    recordSynapticAccess_();

    // Phase 2: online gradient accumulation using windowed error
    if (learning_enabled_) {
        const float err = (target_neuron < error_buffer_.size()) ? error_buffer_[target_neuron] : 0.0f;
        if (err != 0.0f) {
        const float sgrad = computeSurrogateGrad_(neuron_v);
            const float contrib = err * sgrad; // pre spike counts as 1 event
            uint64_t key = 0;
            uint32_t req_pre_tmp = 0;
            uint32_t req_post_tmp = 0;
            if (weight_accessor_.resolve(spike_event->getSourceNeuron(), target_neuron,
                                         req_pre_tmp, req_post_tmp, key)) {
                local_grad_[key] += contrib;
            }
            if (grad_accum_limit_ > 0 && local_grad_.size() > grad_accum_limit_) {
                local_grad_.clear();
            }
        }
    }

    // 一次性详细日志：打印全局/本地映射与地址
    if (logDetail && (enable_detailed_map_log_ || !detailed_log_emitted_)) {
        uint32_t pre_global = spike_event->getSourceNeuron();
        uint32_t post_global = spike_event->getDestinationNeuron();
        uint32_t pre_local_dbg = spike_event->hasCachedPreLocal()
            ? spike_event->getCachedPreLocal()
            : mapPreGlobalToLocal_(pre_global);
        uint32_t post_local_dbg = target_neuron;
        uint64_t offset_dbg = 0;
        if (use_post_row_pre_col_) {
            offset_dbg = static_cast<uint64_t>(post_local_dbg) * static_cast<uint64_t>(weights_cols_) + static_cast<uint64_t>(pre_global);
        } else {
            offset_dbg = static_cast<uint64_t>(pre_local_dbg) * static_cast<uint64_t>(num_neurons_) + post_local_dbg;
        }
        uint64_t addr_dbg = base_addr_ + offset_dbg * sizeof(float);
        output_->verbose(CALL_INFO, 1, 0,
            "🧪 详细权重调试: 事件权重=%.3f, 内存权重=%s, 最终权重=%.3f, 回退=%s\n",
            spike_event->getWeight(), have_mem_weight ? "有" : "无", weight, use_event_weight_fallback_ ? "启用" : "禁用");
        output_->verbose(CALL_INFO, 1, 0,
            "🧪 一次性详细映射: pre_g=%u->pre_l=%u, post_g=%u->post_l=%u, base=%" PRIu64 ", off=%" PRIu64 ", addr=%" PRIu64 ", weight=%.3f\n",
            pre_global, pre_local_dbg, post_global, post_local_dbg, base_addr_, offset_dbg, addr_dbg, weight);
        detailed_log_emitted_ = true;
    }
    if (logDetail) {
        output_->verbose(CALL_INFO, 5, 0, "⚡ 核心%d神经元%d: v_mem=%.3f (添加权重%.3f)\n",
                        core_id_, target_neuron, neuron_v, weight);
    }
    
    // 检查是否达到阈值并发放脉冲（仅当未启用窗口累加时）
    if (!apply_acc_enable_) checkAndFireSpike(target_neuron);
}

void SnnPESubComponent::requestWeight(uint32_t pre_neuron, uint32_t post_neuron, 
                                    std::function<void(float)> callback) {
    if (use_bcsr_ && use_post_row_pre_col_) {
        requestWeightBCSR(pre_neuron, post_neuron, callback);
        return;
    }
    uint32_t req_pre = 0;
    uint32_t req_post = 0;
    uint64_t cache_key = 0;
    if (!weight_accessor_.resolve(pre_neuron, post_neuron, req_pre, req_post, cache_key)) {
        if (callback) callback(init_default_weight_);
        return;
    }

    if (!ensureMemoryReady_()) {
        if (callback) callback(0.5f);
        return;
    }

    uint64_t req_addr = 0; size_t req_size = sizeof(float);
    bool is_row = false; uint32_t col_start = req_post; uint32_t count_floats = 1;
    prepareDenseRead_(req_pre, req_post, use_post_row_pre_col_ ? weights_cols_ : num_neurons_,
                      req_addr, req_size, is_row, col_start, count_floats);
    if (window_read_debug_) {
        output_->verbose(CALL_INFO, 2, 0, "[diag-read] core=%d requestWeight row=%u col=%u is_row=%d col_start=%u count=%u addr=0x%llx size=%zu\n",
                         core_id_, req_pre, req_post, (int)is_row, col_start, count_floats,
                         (unsigned long long)req_addr, req_size);
    }
    issueReadCommon_(req_addr, req_size, is_row, req_pre, col_start, count_floats, callback, cache_key);
}

float SnnPESubComponent::readBcsrWeightFromFile_(uint32_t post_local, uint32_t pre_global) const {
    // 基于 meta.json 与二进制文件，解析对应块并取出 (intra_row, intra_col) 的值
    uint32_t br = bcsr_weights_.effectiveBlockRows();
    if (br == 0) br = 1;
    uint32_t bc = bcsr_weights_.effectiveBlockCols();
    uint32_t block_row = (br? (post_local / br) : 0);
    uint32_t intra_row = (br? (post_local % br) : 0);
    uint32_t blk_col = (bc? (pre_global / bc) : 0);
    uint32_t intra_col = (bc? (pre_global % bc) : 0);
    std::string bin_path = resolveWeightTemplate(node_id_, core_id_);
    if (bin_path.empty()) return 0.0f;
    uint32_t rows=0, colsN=0, brM=0, bcM=0, idxB=0, valB=0, totalB=0;
    uint64_t rp_off=0, ci_off=0, bd_off=0, id_off=0;
    std::string meta_path = bin_path + ".meta.json";
    if (!parseBcsrMeta(meta_path, rows, colsN, brM, bcM, idxB, valB, rp_off, ci_off, bd_off, id_off, totalB)) return 0.0f;
    // 读取 rowptr
    std::ifstream fin(bin_path, std::ios::binary);
    if (!fin.good()) return 0.0f;
    uint32_t start = 0, end = 0;
    fin.seekg(static_cast<std::streamoff>(rp_off + (size_t)block_row * sizeof(uint32_t)), std::ios::beg);
    fin.read(reinterpret_cast<char*>(&start), 4);
    fin.read(reinterpret_cast<char*>(&end), 4);
    if (!fin.good() || end <= start) return 0.0f;
    // 在该行查找目标块列
    int idx_in_row = -1;
    for (uint32_t j=0; j < (end - start); ++j) {
        fin.seekg(static_cast<std::streamoff>(ci_off + (size_t)(start + j) * idxB), std::ios::beg);
        uint32_t colv = 0;
        if (idxB == 2) { uint16_t v=0; fin.read(reinterpret_cast<char*>(&v), 2); colv = v; }
        else { fin.read(reinterpret_cast<char*>(&colv), 4); }
        if (!fin.good()) return 0.0f;
        if (colv == blk_col) { idx_in_row = (int)j; break; }
    }
    if (idx_in_row < 0) return 0.0f;
    uint32_t brEff = (brM? brM : br);
    uint32_t bcEff = (bcM? bcM : bc);
    size_t blk_bytes = (size_t)brEff * (size_t)bcEff * (size_t)valB;
    fin.seekg(static_cast<std::streamoff>(bd_off + (uint64_t)(start + (uint32_t)idx_in_row) * blk_bytes), std::ios::beg);
    std::vector<float> blk((size_t)brEff * (size_t)bcEff, 0.0f);
    if (valB == 4) fin.read(reinterpret_cast<char*>(blk.data()), blk_bytes);
    if (!fin.good()) return 0.0f;
    uint32_t off = intra_row * bcEff + intra_col;
    if (off >= blk.size()) return 0.0f;
    return blk[off];
}

void SnnPESubComponent::handleMemoryResponse(SST::Interfaces::StandardMem::Request* req) {
#ifdef SNNDL_ENABLE_PROFILING
    if (profiler_enabled_) { SNNDL_PROFILE_FUNCTION(profiler_); }
#endif
    // 轻量哨兵：进入/退出标记（限量输出，便于定位崩溃前最后一步）
    static uint32_t s_sentinel_enter = 0;
    if (window_read_debug_ && s_sentinel_enter < 64) {
        const char* t = "req";
        if (dynamic_cast<SST::Interfaces::StandardMem::ReadResp*>(req)) t = "ReadResp";
        else if (dynamic_cast<SST::Interfaces::StandardMem::CustomResp*>(req)) t = "CustomResp";
        output_->verbose(CALL_INFO, 0, 0, "[sentinel] core=%d enter handleMemoryResponse type=%s id=%" PRIu64 "\n",
                         core_id_, t, req ? req->getID() : 0ULL);
        ++s_sentinel_enter;
    }
    // Capture GAS upstream stat updates or stage events
    if (auto* cust = dynamic_cast<SST::Interfaces::StandardMem::CustomResp*>(req)) {
        auto& data = cust->getData();
        if (auto* op = dynamic_cast<GasOpData*>(&data)) {
            // Stage event handling (Phase‑1): requires apply_acc_enable_
            if (apply_acc_enable_) {
                switch (op->op) {
                    case GasOp::BeginGather:
                        gas_stage_ = GasStage::Gather; curr_stage_seq_ = op->superstep; accReset_();
                        if (window_read_debug_) {
                            output_->verbose(CALL_INFO, 0, 0,
                                "[diag-edges] BeginGather seq=%u edges_curr=%zu edges_prev=%zu\n",
                                curr_stage_seq_, edge_collector_.currSize(), edge_collector_.prevSize());
                        }
                        if (window_read_debug_) {
                        SNNDL_DEBUG_LOG(1, "[diag-stage] BeginGather window=%" PRIu64 " stage=%d posts_prev=%zu active_prev=%zu posts_curr=%zu active_curr=%zu qsize=%zu\n",
                            (uint64_t)op->superstep, static_cast<int>(gas_stage_),
                            posts_list_prev_window_.size(), active_pre_prev_window_.size(),
                            posts_list_window_.size(), active_pre_window_.size(), incoming_spikes_.size());
                        }
                        // reset per-window counters
                        {
                        stage_event_hub_.markBeginGather(curr_stage_seq_);
                        // 新窗开始：复位边集合容量告警
                        edge_collector_capacity_warned_ = false;
                            if (window_read_enable_) {
                                // 位图迁移
                                if (!posts_seen_window_.empty()) {
                                    if (posts_seen_prev_window_.size() != posts_seen_window_.size()) {
                                        posts_seen_prev_window_.assign(posts_seen_window_.size(), 0);
                                    }
                                    posts_seen_prev_window_ = posts_seen_window_;
                                    std::fill(posts_seen_window_.begin(), posts_seen_window_.end(), 0);
                                } else if (posts_seen_prev_window_.size() != num_neurons_) {
                                    posts_seen_prev_window_.assign(num_neurons_, 0);
                                }
                                // 列表迁移
                                posts_list_prev_window_.swap(posts_list_window_);
                                posts_list_window_.clear();
                                // pre 集合迁移
                                active_pre_prev_window_ = std::move(active_pre_window_);
                                active_pre_window_.clear();
                            }
                        }
                        acc_updates_count_ = 0; acc_posts_touched_count_ = 0;
                        diag_edges_record_hits_ = 0; diag_edges_stage_skips_ = 0; diag_edges_cond_skips_ = 0;
                        diag_spikes_stage_gather_ = diag_spikes_stage_apply_ = diag_spikes_stage_scatter_ = 0;
                        diag_spikes_stage_idle_ = 0;
                        acc_spill_records_count_ = 0; acc_spilled_bytes_sum_ = 0; acc_hwm_bytes_max_ = 0;
                        break;
                    case GasOp::BeginApply:
                        gas_stage_ = GasStage::Apply; curr_stage_seq_ = op->superstep; // start accepting accum
                        prepareEdgeWindowForApply_();
#if SNNDL_DEBUG_ENABLED
                        if (window_read_debug_) {
                            SNNDL_DEBUG_LOG(1, "[diag-stage] BeginApply window=%" PRIu64 " stage=%d posts_prev=%zu active_prev=%zu posts_curr=%zu active_curr=%zu\n",
                                (uint64_t)op->superstep, static_cast<int>(gas_stage_),
                                posts_list_prev_window_.size(), active_pre_prev_window_.size(),
                                posts_list_window_.size(), active_pre_window_.size());
                        }
#endif
                stage_event_hub_.markBeginApply(curr_stage_seq_);
                read_orchestrator_.issueFallbackReadsIfNeeded(apply_acc_enable_ && gas_window_mode_);
#if SNNDL_DEBUG_ENABLED
                if (window_read_debug_ && output_) {
                    output_->verbose(CALL_INFO, 1, 0,
                        "[diag-edges-summary] core=%u window=%u recorded=%" PRIu64 " stage_skip=%" PRIu64 " cond_skip=%" PRIu64 " edges_prev=%zu\n",
                        core_id_, curr_stage_seq_, diag_edges_record_hits_, diag_edges_stage_skips_, diag_edges_cond_skips_,
                        edge_collector_.prevSize());
                    output_->verbose(CALL_INFO, 1, 0,
                        "[diag-spike-stage] core=%u window=%u gather=%" PRIu64 " apply=%" PRIu64 " scatter=%" PRIu64 " idle=%" PRIu64 "\n",
                        core_id_, curr_stage_seq_, diag_spikes_stage_gather_, diag_spikes_stage_apply_,
                        diag_spikes_stage_scatter_, diag_spikes_stage_idle_);
                }
#endif
                diag_spikes_stage_gather_ = 0;
                break;
                    case GasOp::EndApply:
                        gas_stage_ = GasStage::Apply; // remain until BeginScatter
                        stats_reporter_.reportApplyScatter(
                            acc_updates_count_, acc_posts_touched_count_, 0,
                            acc_hwm_bytes_max_, acc_spill_records_count_, acc_spilled_bytes_sum_);
                        appendStageEventRow_("EndApply", getCurrentSimTimeNano(), 0);
                        break;
                    case GasOp::BeginScatter:
                        gas_stage_ = GasStage::Scatter; // apply accumulated deltas deterministically
#if SNNDL_DEBUG_ENABLED
                        if (window_read_debug_ && output_) {
                            SNNDL_DEBUG_LOG(1, "[diag-stage] BeginScatter window=%" PRIu64 " stage=%d\n",
                                (uint64_t)op->superstep, static_cast<int>(gas_stage_));
                            output_->verbose(CALL_INFO, 1, 0,
                                "[diag-spike-stage] core=%u window=%u (apply) gather=%" PRIu64 " apply=%" PRIu64 " scatter=%" PRIu64 " idle=%" PRIu64 "\n",
                                core_id_, curr_stage_seq_, diag_spikes_stage_gather_, diag_spikes_stage_apply_,
                                diag_spikes_stage_scatter_, diag_spikes_stage_idle_);
                        }
#endif
                        diag_spikes_stage_apply_ = 0;
                        stage_event_hub_.markBeginScatter(curr_stage_seq_);
                        // 记录窗口发放基线：用于 EndScatter 兜底对齐口径（O(1) 开销）
                        spikes_generated_base_ = count_spikes_generated_;
                        // Scheme-C (debug aid): print per-window delta summary just before applying
                        // Helps verify that Apply phase indeed accumulated into this window.
                        // We intentionally use a low verbosity so it shows up when users enable logs.
#if SNNDL_DEBUG_ENABLED
                        if (window_read_debug_ && output_) {
                            // Use updates/posts as主要口径，避免因 acc_delta_ 已在溢写合并前为空而误判
                            output_->verbose(CALL_INFO, 1, 0,
                                "[GAS][Delta] core=%u seq=%u updates=%" PRIu64 ", posts=%" PRIu64 ", hwm_bytes=%" PRIu64 ", spill_records=%" PRIu64 ", spilled_bytes=%" PRIu64 "\n",
                                core_id_, curr_stage_seq_, acc_updates_count_, acc_posts_touched_count_,
                                acc_hwm_bytes_max_, acc_spill_records_count_, acc_spilled_bytes_sum_);
                            output_->verbose(CALL_INFO, 1, 0,
                                "[GAS][Delta][sum] core=%u seq=%u dv_sum=%.6f nonzero_updates=%" PRIu64 "\n",
                                core_id_, curr_stage_seq_, diag_dv_sum_window_, diag_dv_updates_nonzero_);
                        }
#endif
                        if (!acc_spill_log_.empty()) {
                            // Merge spill into map: sort by post and reduce
                            std::sort(acc_spill_log_.begin(), acc_spill_log_.end(), [](auto&a, auto&b){return a.first<b.first;});
                            uint32_t curp=UINT32_MAX; float sum=0.0f;
                            for (auto &pr : acc_spill_log_) {
                                if (pr.first != curp) {
                                    if (curp!=UINT32_MAX) accUpdate_(curp, sum);
                                    curp = pr.first; sum = pr.second;
                                } else { sum += pr.second; }
                            }
                            if (curp!=UINT32_MAX) accUpdate_(curp, sum);
                            acc_spill_log_.clear();
                        }
                        {
                            uint64_t spikes_emitted=0;
                            if (apply_dense_acc_enable_) {
                                // 仅遍历触及列表，顺序化按索引排序（可选）
                                if (!acc_touched_list_.empty()) {
                                    std::sort(acc_touched_list_.begin(), acc_touched_list_.end());
                                }
                                for (auto post : acc_touched_list_) {
                                    if (post >= acc_dense_.size()) continue;
                                    float dv = acc_dense_[post];
                                    if (dv == 0.0f) continue;
                                    float v_before = getMem_(post);
                                    float v = v_before + dv;
                                    setMem_(post, v);
                                    bool willFire = (v >= v_thresh_ && getRefrac_(post) == 0);
#if SNNDL_DEBUG_ENABLED
                                    if (window_read_debug_ && output_) {
                                        output_->verbose(CALL_INFO, 0, 0,
                                            "[diag-scatter] core=%u window=%u post=%u v_before=%.6f dv=%.6f v_after=%.6f thr=%.6f refrac=%u willFire=%d\n",
                                            core_id_, curr_stage_seq_, post,
                                            (double)v_before, (double)dv, (double)v, (double)v_thresh_,
                                            (unsigned)getRefrac_(post), willFire ? 1 : 0);
                                    }
#endif
                                    checkAndFireSpike(post);
                                    if (willFire) {
                                        spikes_emitted++;
                                        if (stat_gas_scatter_spikes_emitted_total_) stat_gas_scatter_spikes_emitted_total_->addData(1);
                                    }
                                }
                                accReset_();
                            } else if (!acc_delta_.empty()) {
                                std::vector<uint32_t> posts; posts.reserve(acc_delta_.size());
                                for (auto &kv : acc_delta_) posts.push_back(kv.first);
                                std::sort(posts.begin(), posts.end());
                                for (auto post : posts) {
                                    float dv = acc_delta_[post];
                                    float v_before = getMem_(post);
                                    float v = v_before + dv;
                                    setMem_(post, v);
                                    bool willFire = (v >= v_thresh_ && getRefrac_(post) == 0);
#if SNNDL_DEBUG_ENABLED
                                    if (window_read_debug_ && output_) {
                                        output_->verbose(CALL_INFO, 0, 0,
                                            "[diag-scatter] core=%u window=%u post=%u v_before=%.6f dv=%.6f v_after=%.6f thr=%.6f refrac=%u willFire=%d\n",
                                            core_id_, curr_stage_seq_, post,
                                            (double)v_before, (double)dv, (double)v, (double)v_thresh_,
                                            (unsigned)getRefrac_(post), willFire ? 1 : 0);
                                    }
#endif
                                    checkAndFireSpike(post);
                                    if (willFire) {
                                        spikes_emitted++;
                                        if (stat_gas_scatter_spikes_emitted_total_) stat_gas_scatter_spikes_emitted_total_->addData(1);
                                    }
                                }
                                accReset_();
                            }
                            // cache this window's spikes for EndScatter CSV emit
                            spikes_emitted_window_ = spikes_emitted;
                            if (apply_dense_acc_enable_) {
                                verifyDenseAccumulator_(curr_stage_seq_);
                            }
                            if (spikes_emitted>0 && parent_) {
                                if (auto* pe = parent_pe_cached_) {
                                    pe->accumulateApplyScatterStats(0, 0, spikes_emitted, 0, 0, 0);
                                }
                            }
                        }
                        break;
                    case GasOp::EndScatter:
                        gas_stage_ = GasStage::Idle;
                        {
                            uint64_t to_emit = window_spikes_all_ ? window_spikes_all_ : spikes_emitted_window_;
                            if (to_emit == 0) {
                                uint64_t delta = 0;
                                if (count_spikes_generated_ >= spikes_generated_base_) {
                                    delta = count_spikes_generated_ - spikes_generated_base_;
                                }
                                if (delta > 0) to_emit = delta;
                            }
                            // 诊断：当两种口径均非零但不一致时，提示一次（不改变行为）
#if SNNDL_DEBUG_ENABLED
                            if (window_read_debug_ && window_spikes_all_ > 0 && spikes_emitted_window_ > 0 &&
                                window_spikes_all_ != spikes_emitted_window_) {
                                output_->verbose(CALL_INFO, 0, 0,
                                    "[diag-window] ⚠️ inconsistent spikes: all=%" PRIu64 " vs emitted=%" PRIu64 " (seq=%u)\n",
                                    window_spikes_all_, spikes_emitted_window_, curr_stage_seq_);
                            }
#endif
                            if (window_read_debug_ && output_) {
                                output_->verbose(CALL_INFO, 1, 0,
                                    "[diag-spike-stage] core=%u window=%u (scatter) gather=%" PRIu64 " apply=%" PRIu64 " scatter=%" PRIu64 " idle=%" PRIu64 "\n",
                                    core_id_, curr_stage_seq_, diag_spikes_stage_gather_, diag_spikes_stage_apply_,
                                    diag_spikes_stage_scatter_, diag_spikes_stage_idle_);
                            }
                            diag_spikes_stage_scatter_ = 0;
                            stage_event_hub_.markEndScatter(curr_stage_seq_, to_emit);
                        }
                        break;
                    default: break;
                }
            }
            // Unconditional: activity f window tracking (independent of apply_acc_enable_)
            switch (op->op) {
                case GasOp::BeginGather:
                    activity_window_seq_ = op->superstep;
                    activityReset_();
                    break;
                case GasOp::BeginApply:
                    // 上一Gather窗口结束，计算其活跃度
                    activityFlush_();
                    break;
                case GasOp::EndApply:
                case GasOp::BeginScatter:
                    // flush activity for the finished window
                    activityFlush_();
                    break;
                default: break;
            }
            delete req; return;
        }
        if (auto* stat = dynamic_cast<GasStatData*>(&data)) {
            // Accumulate at PE level to ensure CSV visibility
            if (window_read_debug_ && output_) output_->verbose(CALL_INFO, 2, 0,
                "[GAS] stat recv: reads=%" PRIu64 ", bytes=%" PRIu64 ", rwt=%" PRIu64 ", rwb=%" PRIu64 ", bursts=%" PRIu64 ", payload=%" PRIu64 "\n",
                (uint64_t)stat->unique_reads, (uint64_t)stat->unique_bytes,
                (uint64_t)stat->rowwin_triggers, (uint64_t)stat->rowwin_bytes,
                (uint64_t)stat->bursts, (uint64_t)stat->payload_bytes);
            if (parent_) {
                if (auto* pe = parent_pe_cached_) {
                    pe->accumulateGasStatsExt(stat->unique_bytes, stat->unique_reads,
                                              stat->rowwin_triggers, stat->rowwin_bytes,
                                              stat->bursts, stat->payload_bytes,
                                              stat->window_inflight_peak, stat->window_buffer_max_bytes);
                }
            }
            if ((stat->window_inflight_peak || stat->window_buffer_max_bytes) && window_read_debug_ && output_) {
                output_->verbose(CALL_INFO, 2, 0,
                    "[GAS] window metrics: inflight_peak=%" PRIu64 ", buffer_peak=%" PRIu64 "\n",
                    (uint64_t)stat->window_inflight_peak, (uint64_t)stat->window_buffer_max_bytes);
            }
            // Local (per-core) copies for unique_* only (optional)
            if (stat_gas_unique_reads_total_ && stat->unique_reads) stat_gas_unique_reads_total_->addData(stat->unique_reads);
            if (stat_gas_unique_bytes_total_ && stat->unique_bytes) stat_gas_unique_bytes_total_->addData(stat->unique_bytes);
            delete req; return;
        }
        // Unknown CustomResp: drop through to delete
    }
    if (!req) return;
    
    output_->verbose(CALL_INFO, 4, 0, "📨 核心%d收到内存响应: ID=%" PRIu64 "\n", 
                    core_id_, req->getID());
    auto it = pending_memory_requests_.find(req->getID());
    if (it != pending_memory_requests_.end()) {
        // 计算往返延迟
        uint64_t ic = it->second.issue_cycle;
        if (total_cycles_ >= ic) {
            uint64_t lat = static_cast<uint64_t>(total_cycles_ - ic);
            accum_mem_latency_cycles_ += lat;
            count_mem_responses_++;
            if (parent_) {
                if (auto* pe = parent_pe_cached_) {
                    pe->accumulateMemReadLatency(lat, it->second.is_weight);
                }
            }
        }
    }
    static uint32_t s_sentinel_exit = 0;
    if (window_read_debug_ && s_sentinel_exit < 64) {
        output_->verbose(CALL_INFO, 0, 0, "[sentinel] core=%d exit handleMemoryResponse\n", core_id_);
        ++s_sentinel_exit;
    }
    
    // 查找对应的挂起请求
    if (it != pending_memory_requests_.end()) {
        PendingMemoryRequest pending_req = it->second; // 拷贝一份，便于先erase
    pending_memory_requests_.erase(it);
        // 方案1：预取计数回退
        if (pending_req.scheme1_prefetch && scheme1_enable_) {
            if (scheme1_pending_prefetch_ > 0) scheme1_pending_prefetch_--;
        }

        auto finalize_rowptr_ready = [&]() {
            if (window_read_debug_ && output_) {
                const auto& rp = bcsr_weights_.rowptrHost();
                output_->verbose(CALL_INFO, 0, 0,
                    "[diag-bcsr] core=%u rowptr ready entries=%zu first=%u second=%u\n",
                    core_id_, rp.size(),
                    rp.empty() ? 0u : rp[0],
                    rp.size() > 1 ? rp[1] : 0u);
            }
            bcsrPrefetchAll_();
            if (apply_acc_enable_ && gas_window_mode_ && gas_stage_ == GasStage::Apply) {
                issueEdgeWeightFetches_();
            }
        };

        auto* readResp = dynamic_cast<SST::Interfaces::StandardMem::ReadResp*>(req);
        if (readResp && window_read_debug_) {
            const std::vector<uint8_t>& bytes = readResp->data;
            const char* kind_label = "-";
            switch (pending_req.bcsr_kind) {
                case 1: kind_label = "rowptr"; break;
                case 2: kind_label = "colidx"; break;
                case 3: kind_label = "block"; break;
                default: break;
            }
            output_->verbose(CALL_INFO, 1, 0,
                "[diag-resp] core=%d kind=%s id=%" PRIu64 " addr=0x%llx bytes=%zu\n",
                core_id_, kind_label, req->getID(), (unsigned long long)pending_req.address,
                bytes.size());
        }
        if (pending_req.bcsr_kind == 1) {
            bcsr_weights_.setRowptrReadPending(false);
        }
        if (readResp && !readResp->data.empty()) {
            const std::vector<uint8_t>& bytes = readResp->data;
            size_t float_count = bytes.size() / sizeof(float);
            const float* fptr = reinterpret_cast<const float*>(bytes.data());
            output_->verbose(CALL_INFO, 3, 0, "📥 内存响应: addr=0x%lx, bytes=%zu, floats=%zu\n",
                              pending_req.address, bytes.size(), float_count);

            if (pending_req.bcsr_kind == 1) {
                if (!installRowptrFromBytes_(bytes.data(), bytes.size(), "dram", true)) {
                    ensureRowptrReadyOrFatal_("rowptr load failed (dram)");
                }
                finalize_rowptr_ready();
            } else if (pending_req.bcsr_kind == 2) {
                const size_t expect = pending_req.size;
                if (bytes.size() != expect) {
                    output_->verbose(CALL_INFO, 0, 0,
                        "[diag-bcsr][warn] colidx read size mismatch: got=%zu expect=%zu addr=0x%llx\n",
                        bytes.size(), expect, (unsigned long long)pending_req.address);
                    if (pending_req.has_single_cb && pending_req.single_cb) pending_req.single_cb(0.0f);
                    return;
                }
                uint32_t idx_bytes = bcsr_weights_.effectiveIdxBytes();
                size_t n = (idx_bytes ? bytes.size() / idx_bytes : 0);
                std::vector<uint32_t> cols(n);
                if (idx_bytes == 2) {
                    for (size_t i=0;i<n;i++) cols[i] = ((const uint16_t*)bytes.data())[i];
                } else {
                    for (size_t i=0;i<n;i++) cols[i] = ((const uint32_t*)bytes.data())[i];
                }
                bcsrRowIndexPut_(pending_req.bcsr_block_row, cols);
                bcsr_count_row_index_fills_++;
                bcsr_count_colidx_reads_++;
                bcsr_bytes_idx_ += bytes.size();
                if (pending_req.bcsr_prefetch_all) {
                    bcsrPrefetchRowBlocks_(pending_req.bcsr_block_row, cols, pending_req.bcsr_row_start);
                } else {
                    // 查找目标块
                    uint32_t idx_in_row = 0; bool found=false;
                    for (size_t i=0;i<cols.size();++i){ if (cols[i]==pending_req.bcsr_target_block_col){ idx_in_row=(uint32_t)i; found=true; break; } }
                    if (!found) {
                        if (pending_req.has_single_cb && pending_req.single_cb) pending_req.single_cb(0.0f);
                    } else {
                        uint32_t start = pending_req.bcsr_row_start;
                        // 块数据按 rowptr 顺序线性排列；blockids 段为 Reachability 用途，不能作为寻址 ID。
                        uint32_t global_block_index = start + idx_in_row;
                        std::vector<float> blk;
                        const uint32_t bc_eff = bcsr_weights_.effectiveBlockCols();
                        if (bcsrBlockGet_(pending_req.bcsr_block_row, pending_req.bcsr_target_block_col, blk)) {
                            uint32_t off = pending_req.bcsr_intra_row * bc_eff + pending_req.bcsr_intra_col;
                            float w = (off<blk.size()? blk[off] : 0.0f);
                            if (pending_req.has_single_cb && pending_req.single_cb) pending_req.single_cb(w);
                        } else {
                            size_t block_bytes = bcsr_weights_.blockBytes();
                            uint64_t addr = bcsr_weights_.blockDataAddr(global_block_index);
                            auto* rd = new SST::Interfaces::StandardMem::Read(addr, block_bytes);
                            auto id = rd->getID();
                            PendingMemoryRequest pm;
                            pm.request_id = id; pm.address = addr; pm.size = block_bytes; pm.issue_cycle = total_cycles_;
                            pm.is_weight = true; // BCSR 块数据属于权重
                            pm.bcsr_kind = 3; pm.bcsr_block_row = pending_req.bcsr_block_row; pm.bcsr_target_block_col = pending_req.bcsr_target_block_col;
                            pm.bcsr_intra_row = pending_req.bcsr_intra_row; pm.bcsr_intra_col = pending_req.bcsr_intra_col;
                            pm.bcsr_row_start = start; pm.bcsr_idx_in_row = idx_in_row; pm.bcsr_global_block_index = global_block_index;
                            pm.has_single_cb = pending_req.has_single_cb; pm.single_cb = pending_req.single_cb;
                            stats_reporter_.reportMemoryIssue(block_bytes, true);
                            pending_memory_requests_[id] = pm;
                            memory_->send(rd);
                            bcsr_count_block_misses_++;
                        }
                    }
                }
            } else if (pending_req.bcsr_kind == 3) {
                size_t n = static_cast<size_t>(bcsr_weights_.effectiveBlockRows()) *
                           static_cast<size_t>(bcsr_weights_.effectiveBlockCols());
                std::vector<float> blk(n); // 默认0填充；若响应为空/不足，避免未定义行为
                const size_t expect_bytes = n * static_cast<size_t>(bcsr_weights_.effectiveValBytes());
                bool used_mem_block = false;
                if (bcsr_weights_.effectiveValBytes() == 4 && bytes.size() >= expect_bytes) {
                    std::memcpy(blk.data(), bytes.data(), expect_bytes);
                    used_mem_block = true;
                } else {
                    if (window_read_debug_ && output_) {
                        output_->verbose(CALL_INFO, 0, 0,
                            "[diag-bcsr][warn] block read size mismatch: got=%zu expect=%zu addr=0x%llx row=%u col=%u (fallback=file)\n",
                            bytes.size(), expect_bytes, (unsigned long long)pending_req.address,
                            pending_req.bcsr_block_row, pending_req.bcsr_target_block_col);
                    }
                    // 尝试从权重文件直接加载该块（仅诊断/兜底，不改变既有口径；成功后仍按相同路径accUpdate_）
                    // 需要：rowptr已从文件加载；根据 global_block_index 精确定位到块数据
                    const uint32_t global_block_index = pending_req.bcsr_row_start + pending_req.bcsr_idx_in_row;
                    std::string path = weights_template_;
                    // 替换 {pe:02d} 和 {core:02d}
                    size_t p = path.find("{pe:02d}");
                    if (p != std::string::npos) {
                        char buf[16]; std::snprintf(buf, sizeof(buf), "%02u", node_id_);
                        path.replace(p, 8, buf);
                    } else {
                        p = path.find("{pe}");
                        if (p != std::string::npos) path.replace(p, 4, std::to_string(node_id_));
                    }
                    p = path.find("{core:02d}");
                    if (p != std::string::npos) {
                        char buf2[16]; std::snprintf(buf2, sizeof(buf2), "%02u", core_id_);
                        path.replace(p, 10, buf2);
                    } else {
                        p = path.find("{core}");
                        if (p != std::string::npos) path.replace(p, 6, std::to_string(core_id_));
                    }
                    size_t block_bytes = bcsr_weights_.blockBytes();
                    uint64_t blockdata_off = (bcsr_weights_.blockdataAddr() > base_addr_) ?
                        (bcsr_weights_.blockdataAddr() - base_addr_) : 0;
                    uint64_t file_off = blockdata_off + (uint64_t)global_block_index * (uint64_t)block_bytes;
                    std::ifstream fin(path, std::ios::binary);
                    if (fin.good()) {
                        fin.seekg(0, std::ios::end);
                        std::streamsize fsz = fin.tellg();
                        if (fsz >= 0 && (uint64_t)fsz >= (file_off + block_bytes)) {
                            fin.seekg((std::streamoff)file_off, std::ios::beg);
                            std::vector<uint8_t> tmp(block_bytes);
                            fin.read(reinterpret_cast<char*>(tmp.data()), (std::streamsize)block_bytes);
                            if (fin.gcount() == (std::streamsize)block_bytes && bcsr_weights_.effectiveValBytes() == 4) {
                                std::memcpy(blk.data(), tmp.data(), block_bytes);
                                used_mem_block = true; // 实际来源为文件，但后续处理一致
                                if (window_read_debug_ && output_) {
                                    output_->verbose(CALL_INFO, 1, 0,
                                        "[diag-bcsr][file] loaded block row=%u idx_in_row=%u gb=%u off=0x%llx bytes=%zu\n",
                                        pending_req.bcsr_block_row, pending_req.bcsr_idx_in_row, global_block_index,
                                        (unsigned long long)file_off, (size_t)block_bytes);
                }
        } else if (pending_req.bcsr_kind == 1) {
            ensureRowptrReadyOrFatal_("rowptr read returned empty payload");
            finalize_rowptr_ready();
        }
                        }
                    }
                }
                bcsrBlockPut_(pending_req.bcsr_block_row, pending_req.bcsr_target_block_col, blk);
                bcsr_count_block_reads_++;
                bcsr_bytes_val_ += bytes.size();
                const uint32_t bc_hit = bcsr_weights_.effectiveBlockCols();
                uint32_t off = pending_req.bcsr_intra_row * bc_hit + pending_req.bcsr_intra_col;
                float w = (off < blk.size() ? blk[off] : 0.0f);
                if (bcsr_weight_guard_enable_) {
                    if (!std::isfinite(w) || std::fabs(w) > bcsr_weight_abs_max_) {
                        if (window_read_debug_ && output_) {
                            output_->verbose(CALL_INFO, 0, 0,
                                "[diag-bcsr][guard] core=%u row=%u col=%u raw=%.6e -> clamped(0)\n",
                                core_id_, pending_req.bcsr_block_row,
                                pending_req.bcsr_target_block_col, (double)w);
                        }
                        w = 0.0f;
                        bcsr_bad_weight_count_++;
                    }
                }
                if (readresp_zero_fallback_ && w == 0.0f) {
                    w = init_default_weight_;
                }
                bcsr_req_block_hit_++;
                if (window_read_debug_ && output_) {
                    uint32_t post_local = pending_req.bcsr_block_row * bcsr_weights_.effectiveBlockRows() + pending_req.bcsr_intra_row;
                    uint32_t post_global = global_neuron_base_ + post_local;
                    uint32_t pre_global_effective = pending_req.bcsr_target_block_col * bc_hit + pending_req.bcsr_intra_col;
                    output_->verbose(CALL_INFO, 1, 0,
                        "[diag-bcsr-weight] core=%u post_local=%u post_global=%u pre_global=%u weight=%.6f source=%s\n",
                        core_id_, post_local, post_global, pre_global_effective, w, used_mem_block ? "OK" : "FallbackFile");
                }
                // 窗口累加：严格GAS下在Apply读到目标权重后，直接将该pair的ΔV累加到当前窗口的acc，Scatter阶段统一应用
                if (apply_acc_enable_ && gas_window_mode_) {
                    uint32_t post_local = pending_req.bcsr_block_row * bcsr_weights_.effectiveBlockRows() + pending_req.bcsr_intra_row;
                    accUpdate_(post_local, w);
                }
                if (pending_req.has_single_cb && pending_req.single_cb) pending_req.single_cb(w);
            } else {
                // Dense：行/缓存线回填；在窗口Apply启用时，仅对本次请求的目标(pre_global=cb_post)进行累加，避免整行累加误差
                uint32_t width = use_post_row_pre_col_ ? weights_cols_ : num_neurons_;
                // 健壮性：若为行合并且返回字节与期望不符，仅打印告警（不改语义，不自动重发）
                if (pending_req.is_row) {
                    size_t expect_bytes = static_cast<size_t>(width) * sizeof(float);
                    if (pending_req.size != expect_bytes) {
                        output_->verbose(CALL_INFO, 1, 0,
                            "[GAS][Warn] row-merge size mismatch: got=%zu expect=%zu (row=%u)\n",
                            (size_t)pending_req.size, expect_bytes, pending_req.pre);
                    }
                }
                // 仅日志：无论是否命中窗口累加，若启用权重验证且开启了日志，则打印当前切片的首元素（不改变语义）
                if (verify_weights_ && output_ && verbose_ >= 1) {
                    float wprobe = (float_count > 0 ? fptr[0] : 0.0f);
                    uint32_t probe_row = pending_req.pre;
                    uint32_t probe_col = pending_req.post_start; // 切片首列（read_force_single=1 时即目标列）
                    output_->verbose(CALL_INFO, 1, 0,
                        "[VERIFY][probe-any] row=%u col=%u value=%.6f size=%zu floats=%zu is_row=%d\n",
                        probe_row, probe_col, wprobe, (size_t)pending_req.size, (size_t)float_count, pending_req.is_row ? 1 : 0);
                }
                if (apply_acc_enable_ && use_post_row_pre_col_) {
                    uint32_t target_col = pending_req.cb_post; // 目标 pre_global 列
                    if (target_col >= pending_req.post_start && target_col < pending_req.post_start + float_count && target_col < width) {
                        size_t idx = static_cast<size_t>(target_col - pending_req.post_start);
                        float w = fptr[idx];
                        if (bcsr_weight_guard_enable_) {
                            if (!std::isfinite(w) || std::fabs(w) > bcsr_weight_abs_max_) {
                                if (window_read_debug_ && output_) {
                                    output_->verbose(CALL_INFO, 0, 0,
                                        "[diag-bcsr][guard] core=%u row(post)=%u col(pre)=%u raw=%.6e -> clamped(0)\n",
                                        core_id_, pending_req.pre, target_col, (double)w);
                                }
                                w = 0.0f;
                                bcsr_bad_weight_count_++;
                            }
                        }
                        if (readresp_zero_fallback_ && w == 0.0f) w = init_default_weight_;
                        uint64_t key = static_cast<uint64_t>(pending_req.pre) * static_cast<uint64_t>(width) + target_col;
                        weightCacheStore_(key, w);
                        uint32_t post_local = pending_req.pre; // row is post
                        if (output_) {
                            size_t idx = static_cast<size_t>(target_col - pending_req.post_start);
                            output_->verbose(CALL_INFO, 2, 0,
                                "[GAS][Delta][readresp] core=%u seq=%u post(row)=%u pre(col)=%u dv=%.6f (single col) addr=0x%lx start=%u idx=%zu size=%zu\n",
                                core_id_, curr_stage_seq_, post_local, target_col, w,
                                (unsigned long)pending_req.address, pending_req.post_start,
                                idx, (size_t)pending_req.size);
                            // 仅日志钩子：在诊断场景下输出样本值（不改变语义）
                            if (verify_weights_) {
                                output_->verbose(CALL_INFO, 1, 0,
                                    "[VERIFY][probe] row=%u col=%u value=%.6f\n",
                                    post_local, target_col, w);
                            }
                        }
                        accUpdate_(post_local, w);
                    }
                } else {
                    // 非窗口模式或旧映射：按元素填入缓存，并在旧模式采用回调
                    for (size_t i = 0; i < float_count; ++i) {
                        uint32_t col_idx = pending_req.post_start + static_cast<uint32_t>(i);
                        if (col_idx >= width) break;
                        uint64_t key = static_cast<uint64_t>(pending_req.pre) * static_cast<uint64_t>(width) + col_idx;
                        float w = fptr[i];
                        if (readresp_zero_fallback_ && w == 0.0f) w = init_default_weight_;
                        weightCacheStore_(key, w);
                        if (pending_req.has_single_cb && pending_req.single_cb && !use_post_row_pre_col_) {
                            pending_req.single_cb(w);
                        }
                    }
                }
            }
        } else {
            if (pending_req.has_single_cb && pending_req.single_cb) pending_req.single_cb(0.0f);
        }
        if (outstanding_requests_ > 0) outstanding_requests_--;
        // Apply 窗口内的补发：在收到 ReadResp 后，若预算/并发未满则继续发起下一批
        if (!(apply_acc_enable_ && gas_window_mode_) && window_read_enable_ && gas_stage_ == GasStage::Apply && enable_weight_fetch_ && memory_ && memory_ready_) {
            // 允许使用当前窗作为补发来源（当 prev 为空但 curr 非空时）
            bool have_prev_refill = (!active_pre_prev_window_.empty() && !posts_list_prev_window_.empty());
            bool have_curr_refill = (!active_pre_window_.empty()      && !posts_list_window_.empty());
            const auto& pres_src_refill  = (have_prev_refill ? active_pre_prev_window_ : active_pre_window_);
            const auto& posts_src_refill = (have_prev_refill ? posts_list_prev_window_ : posts_list_window_);
            if (window_reads_issued_this_apply_ < window_read_budget_ && !pres_src_refill.empty() && !posts_src_refill.empty()) {
                const uint32_t width_refill = use_post_row_pre_col_ ? weights_cols_ : num_neurons_;
                for (const auto& pre_g : pres_src_refill) {
                    for (uint32_t post_l : posts_src_refill) {
                        if (window_reads_issued_this_apply_ >= window_read_budget_) break;
                        if (outstanding_requests_ >= max_outstanding_requests_) break;
                        uint32_t arg0 = use_post_row_pre_col_ ? pre_g : post_l;
                        uint32_t arg1 = use_post_row_pre_col_ ? post_l : pre_g;
                        const uint64_t key = (uint64_t)post_l * (uint64_t)width_refill + (use_post_row_pre_col_ ? (uint64_t)pre_g : (uint64_t)post_l);
                        stats_reporter_.reportCacheAccess(false);
                        outstanding_requests_++;
                        stats_reporter_.updatePendingPeak(outstanding_requests_);
                        requestWeight(arg0, arg1, [this, key](float w){
                            weightCacheStore_(key, w);
                            if (outstanding_requests_ > 0) outstanding_requests_--;
                        });
                        window_reads_issued_this_apply_++;
                        if (outstanding_requests_ >= max_outstanding_requests_) break;
                    }
                    if (window_reads_issued_this_apply_ >= window_read_budget_ || outstanding_requests_ >= max_outstanding_requests_) break;
                }
            }
        }
        if (apply_acc_enable_ && gas_window_mode_) {
            issueEdgeWeightFetches_();
        }
    }
    
    delete req;
}

// ===== 方案1：slice 顺序执行实现 =====
void SnnPESubComponent::scheme1Reset_() {
    // 进入一个新的 superstep（基线方案1：每个superstep仅处理一个slice/块）
    scheme1_stage_ = Scheme1Stage::Gather;
    scheme1_stage_counter_ = 0;
    if (scheme1_first_superstep_) {
        scheme1_current_slice_ = 0;
        scheme1_first_superstep_ = false;
    }
    scheme1_prefetch_issued_ = false;
    scheme1_pending_prefetch_ = 0;
    if (!scheme1_queues_inited_) {
        scheme1_slice_queues_.assign(std::max<uint32_t>(1, scheme1_slices_), {});
        scheme1_queues_inited_ = true;
    }
}

bool SnnPESubComponent::scheme1Tick_() {
    // 初始进入 Gather
    if (scheme1_stage_ == Scheme1Stage::Idle) {
        scheme1Reset_();
    }

    // Gather：收集本 superstep 的所有脉冲（仅缓存，不处理）
    if (scheme1_stage_ == Scheme1Stage::Gather) {
        while (!incoming_spikes_.empty()) {
            auto* sp = incoming_spikes_.front(); incoming_spikes_.pop();
            uint32_t pre_g = sp->getSourceNeuron();
            uint32_t slice = scheme1SliceFromPreGlobal_(pre_g);
            if (slice >= scheme1_slice_queues_.size()) slice = (uint32_t)scheme1_slice_queues_.size() - 1;
            scheme1_slice_queues_[slice].push_back(sp);
        }
        scheme1_stage_counter_++;
        if (scheme1_stage_counter_ >= scheme1_gather_cycles_cfg_) {
            scheme1_stage_ = Scheme1Stage::Apply;
            scheme1_stage_counter_ = 0;
            scheme1_current_slice_ = 0;
            scheme1_prefetch_issued_ = false;
            scheme1_pending_prefetch_ = 0;
        }
        // Gather 阶段暂不进行膜电位演化/发放，返回即可
        return true;
    }

    // Apply：仅处理当前slice（基线方案1：每个superstep只处理一个slice）
    if (scheme1_stage_ == Scheme1Stage::Apply) {
        // 发起预取（按行扫描该 slice 的列区间：基线方案，不做按需优化）
        if (!scheme1_prefetch_issued_) {
            scheme1PrefetchSlice_(scheme1_current_slice_);
            scheme1_prefetch_issued_ = true;
        }
        // 等待预取返回
        if (scheme1_pending_prefetch_ > 0) {
            return true; // 等下一批响应
        }
        // 处理该 slice 的所有事件（其他slice队列保留到后续superstep）
        auto& q = scheme1_slice_queues_[scheme1_current_slice_];
        while (!q.empty()) {
            auto* sp = q.front(); q.pop_front();
            processLocalSpike(sp);
            delete sp;
        }
        // 当前slice处理完毕，清空其队列
        q.clear();
        // slice 间隔
        scheme1_stage_counter_++;
        if (scheme1_stage_counter_ < scheme1_slice_gap_cycles_) return true;
        // 本superstep结束，进入Scatter
        scheme1_stage_counter_ = 0;
        scheme1_prefetch_issued_ = false;
        scheme1_stage_ = Scheme1Stage::Scatter;
        return true;
    }

    // Scatter：统一触发膜电位发放（可配置持续若干周期）
    if (scheme1_stage_ == Scheme1Stage::Scatter) {
        // 更新膜电位并尝试触发发放
        updateNeuronStates();
        for (uint32_t i = 0; i < num_neurons_; i++) {
            checkAndFireSpike(i);
        }
        scheme1_stage_counter_++;
        if (scheme1_stage_counter_ >= scheme1_scatter_cycles_) {
            // 轮换到下一个slice（块），进入下一superstep的Gather阶段
            scheme1_current_slice_ = (scheme1_current_slice_ + 1) % std::max<uint32_t>(1, scheme1_slices_);
            scheme1_stage_ = Scheme1Stage::Idle; // 下一tick会调用 scheme1Reset_() 进入下个superstep
        }
        return true;
    }

    return true; // 默认不落入旧路径
}

void SnnPESubComponent::scheme1PrefetchSlice_(uint32_t slice_idx) {
    if (!ensureMemoryReady_()) return;
    if (weights_cols_ == 0) return;
    // 计算该 slice 的列区间 [beg, end)
    uint32_t width = weights_cols_;
    uint32_t seg = std::max<uint32_t>(1, (width + scheme1_slices_ - 1) / scheme1_slices_);
    uint32_t beg = std::min<uint32_t>(slice_idx * seg, width);
    uint32_t end = std::min<uint32_t>(beg + seg, width);
    if (beg >= end) return;
    const uint32_t fpl = std::max<uint32_t>(1, line_size_bytes_ / (uint32_t)sizeof(float));
    s1_is_issuing_prefetch_ = true;
    for (uint32_t row = 0; row < num_neurons_; ++row) {
        // 按 cacheline 对齐扫描该区间
        for (uint32_t c = (beg / fpl) * fpl; c < end; c += fpl) {
            uint64_t req_addr = base_addr_ + (static_cast<uint64_t>(row) * width + c) * sizeof(float);
            size_t req_size = std::min<uint32_t>(fpl, end - c) * (uint32_t)sizeof(float);
            if (stat_s1_bytes_read_) stat_s1_bytes_read_->addData(static_cast<uint64_t>(req_size));
            // 通过公共发起路径以便回填到权重缓存
            // 设置 is_row=false、col_start=c、count_floats=...，不需要回调
            auto* rd = new SST::Interfaces::StandardMem::Read(req_addr, req_size);
            auto id = rd->getID();
            PendingMemoryRequest pm;
            pm.request_id = id; pm.address = req_addr; pm.size = req_size; pm.is_row = false;
            pm.pre = row; pm.post_start = c; pm.count_floats = (uint32_t)(req_size / sizeof(float));
            pm.has_single_cb = false; pm.cb_post = 0; pm.issue_cycle = total_cycles_;
            pm.is_weight = (req_addr >= base_addr_ && req_addr < weight_region_end_);
            pm.scheme1_prefetch = true;
            stats_reporter_.reportMemoryIssue(req_size, true);
            pending_memory_requests_[id] = pm;
            memory_->send(rd);
            scheme1_pending_prefetch_++;
        }
    }
    s1_is_issuing_prefetch_ = false;
}

void SnnPESubComponent::verifyDenseAccumulator_(uint32_t seq) {
    if (!acc_shadow_verify_enable_) return;
    constexpr double kEps = 1e-5;
    // TEMP(debug): append a per-window breadcrumb indicating verify path executed and basics of touched size.
    {
        FILE* fp = std::fopen("/tmp/acc_shadow.log", "a");
        if (fp) {
            std::fprintf(fp, "[acc-shadow-verify] node=%u core=%u seq=%u touched=%zu time_ns=%" PRIu64 "\n",
                        node_id_, core_id_, seq, acc_touched_list_.size(), (uint64_t)getCurrentSimTimeNano());
            std::fclose(fp);
        }
    }
    auto log_once = [&](const char* reason, uint32_t post, double dense, double reference) {
        if (acc_shadow_mismatch_logged_ || !output_) return;
        output_->verbose(CALL_INFO, 0, 0,
            "[acc-shadow] core=%u seq=%u %s post=%u dense=%.6f ref=%.6f diff=%.6g\n",
            core_id_, seq, reason ? reason : "-", post,
            dense, reference, dense - reference);
        acc_shadow_mismatch_logged_ = true;
    };

    for (auto post : acc_touched_list_) {
        double dense = (post < acc_dense_.size()) ? (double)acc_dense_[post] : 0.0;
        double reference = 0.0;
        auto it = acc_shadow_map_.find(post);
        if (it != acc_shadow_map_.end()) {
            reference = (double)it->second;
            acc_shadow_map_.erase(it);
        }
        if (std::fabs(dense - reference) > kEps) {
            log_once("mismatch", post, dense, reference);
        }
    }
    if (!acc_shadow_map_.empty()) {
        auto kv = *acc_shadow_map_.begin();
        log_once("unused", kv.first, 0.0, kv.second);
        acc_shadow_map_.clear();
    }
    acc_shadow_map_.clear();
    acc_shadow_mismatch_logged_ = false;
}

// === Helpers implementations ===
bool SnnPESubComponent::prepareDenseRead_(uint32_t row, uint32_t col, uint32_t width,
                           uint64_t& req_addr, size_t& req_size,
                           bool& is_row, uint32_t& col_start, uint32_t& count_floats) const {
    const uint32_t bpf = sizeof(float);
    // 默认先按单元素读取初始化（便于切换诊断开关）
    req_addr = base_addr_ + (static_cast<uint64_t>(row) * width + col) * bpf;
    req_size = sizeof(float);
    is_row = false;
    col_start = col;
    count_floats = 1;
    if (read_force_single_) {
        return true;
    }
    if (merge_read_auto_) {
        uint32_t fpl = std::max<uint32_t>(1, line_size_bytes_ / bpf);
        size_t bytes_row = static_cast<size_t>(width) * bpf;
        size_t bytes_cl  = static_cast<size_t>(fpl) * bpf;
        bool choose_row = merge_read_row_ && (bytes_row <= bytes_cl);
        if (choose_row && merge_read_row_) {
            is_row = true;
            col_start = 0;
            count_floats = width;
            req_addr = base_addr_ + static_cast<uint64_t>(row) * width * bpf;
            req_size = static_cast<size_t>(count_floats) * bpf;
        } else if (merge_read_cacheline_) {
            col_start = (col / fpl) * fpl;
            count_floats = std::min<uint32_t>(fpl, width - col_start);
            req_addr = base_addr_ + (static_cast<uint64_t>(row) * width + col_start) * bpf;
            req_size = static_cast<size_t>(count_floats) * bpf;
        }
    } else if (merge_read_row_) {
        is_row = true;
        col_start = 0;
        count_floats = width;
        req_addr = base_addr_ + static_cast<uint64_t>(row) * width * bpf;
        req_size = static_cast<size_t>(count_floats) * bpf;
    } else if (merge_read_cacheline_) {
        uint32_t fpl = std::max<uint32_t>(1, line_size_bytes_ / bpf);
        col_start = (col / fpl) * fpl;
        count_floats = std::min<uint32_t>(fpl, width - col_start);
        req_addr = base_addr_ + (static_cast<uint64_t>(row) * width + col_start) * bpf;
        req_size = static_cast<size_t>(count_floats) * bpf;
    }
    return true;
}

void SnnPESubComponent::issueReadCommon_(uint64_t req_addr, size_t req_size,
                          bool is_row, uint32_t row, uint32_t col_start, uint32_t count_floats,
                          std::function<void(float)> single_cb, uint32_t single_col) {
    auto* read = new SST::Interfaces::StandardMem::Read(req_addr, req_size);
    uint64_t reqId = read->getID();
    PendingMemoryRequest pmr;
    pmr.request_id = reqId;
    pmr.address = req_addr;
    pmr.size = req_size;
    pmr.is_row = is_row;
    pmr.pre = row;
    pmr.post_start = col_start;
    pmr.count_floats = count_floats;
    pmr.has_single_cb = (single_cb != nullptr);
    pmr.cb_post = single_col;
    pmr.single_cb = single_cb;
    pmr.issue_cycle = total_cycles_;
    // Dense权重读取：视为权重区（地址位于 base_addr_ 段内）
    pmr.is_weight = (req_addr >= base_addr_ && req_addr < weight_region_end_);
    stats_reporter_.reportMemoryIssue(req_size, true);
    pending_memory_requests_[reqId] = pmr;
    if (window_read_debug_) {
        output_->verbose(CALL_INFO, 1, 0,
            "[diag-issue] core=%d send Read id=%" PRIu64 " addr=0x%llx size=%zu is_row=%d col_start=%u count=%u outstanding=%zu\n",
            core_id_, reqId, (unsigned long long)req_addr, req_size, (int)is_row,
            col_start, count_floats, (size_t)pending_memory_requests_.size());
    }
    memory_->send(read);
}

void SnnPESubComponent::initializeStatistics() {
    // output_->verbose(CALL_INFO, 2, 0, "📊 核心%d初始化统计收集\n", core_id_);
    
    stat_spikes_received_ = registerStatistic<uint64_t>("spikes_received");
    stat_spikes_generated_ = registerStatistic<uint64_t>("spikes_generated");
    stat_neurons_fired_ = registerStatistic<uint64_t>("neurons_fired");
    stat_memory_requests_ = registerStatistic<uint64_t>("memory_requests");
    stat_weight_cache_hits_ = registerStatistic<uint64_t>("weight_cache_hits");
    stat_weight_cache_misses_ = registerStatistic<uint64_t>("weight_cache_misses");
    stat_merged_reads_rows_ = registerStatistic<uint64_t>("merged_reads_rows");
    stat_merged_reads_cls_ = registerStatistic<uint64_t>("merged_reads_cls");
    stat_weights_verify_count_ = registerStatistic<uint64_t>("weights_verify_count");
    stat_weights_mismatch_count_ = registerStatistic<uint64_t>("weights_mismatch_count");
    stat_weights_verify_sum_ = registerStatistic<double>("weights_verify_sum");
    // 扩展统计
    stat_routes_entries_ = registerStatistic<uint64_t>("routes_entries");
    stat_fanout_per_spike_ = registerStatistic<uint64_t>("fanout_per_spike");
    stat_cache_evictions_ = registerStatistic<uint64_t>("cache_evictions");
    stat_pending_reqs_peak_ = registerStatistic<uint64_t>("pending_reqs_peak");
    stat_cycles_update_neuron_ = registerStatistic<uint64_t>("cycles_update_neuron");
    stat_synaptic_accesses_ = registerStatistic<uint64_t>("synaptic_accesses");
    stat_s1_bytes_read_ = registerStatistic<uint64_t>("scheme1_bytes_read");
    // 门控诊断：权重读请求发起次数
    stat_weight_read_requests_ = registerStatistic<uint64_t>("weight_read_requests");
    // GAS totals accumulated from GatherBufferIF via CustomResp
    stat_gas_unique_reads_total_ = registerStatistic<uint64_t>("gas_unique_reads_total");
    stat_gas_unique_bytes_total_ = registerStatistic<uint64_t>("gas_unique_bytes_total");
    // 边集合溢出计数（仅在容量保护触发时递增）
    stat_gas_edge_overflow_ = registerStatistic<uint64_t>("gas_edge_overflow");
    // Apply/Scatter端到端统计（Phase‑1）
    stat_gas_apply_acc_updates_total_ = registerStatistic<uint64_t>("gas_apply_acc_updates_total");
    stat_gas_acc_posts_touched_total_ = registerStatistic<uint64_t>("gas_acc_posts_touched_total");
    stat_gas_scatter_spikes_emitted_total_ = registerStatistic<uint64_t>("gas_scatter_spikes_emitted_total");
    stat_gas_acc_hwm_bytes_total_ = registerStatistic<uint64_t>("gas_acc_high_watermark_bytes_total");
    stat_gas_acc_spill_records_total_ = registerStatistic<uint64_t>("gas_acc_spill_records_total");
    stat_gas_acc_spilled_bytes_total_ = registerStatistic<uint64_t>("gas_acc_spilled_bytes_total");
    // GAS superstep durations（cycles@1GHz == ns）
    // 留空注册，默认不向 SST 统计输出；统一使用 stage_events_csv + 离线脚本聚合
    // Batch-A additions（若需要SST统计输出，可在后续版本开放）
    // stat_mem_read_latency_cycles_ = registerStatistic<uint64_t>("mem_read_latency_cycles");
    // stat_mem_read_latency_cycles_weights_ = registerStatistic<uint64_t>("mem_read_latency_cycles_weights");
    // stat_mem_read_latency_cycles_state_ = registerStatistic<uint64_t>("mem_read_latency_cycles_state");
    // 记录单次内存请求大小与发起时的未完成请求数（Mesh 汇总使用）
    stat_mem_req_size_bytes_ = registerStatistic<uint64_t>("mem_req_size_bytes");
    stat_mem_outstanding_at_issue_ = registerStatistic<uint64_t>("mem_outstanding_at_issue");
    
    // output_->verbose(CALL_INFO, 2, 0, "✅ 核心%d统计收集初始化完成\n", core_id_);
}
bool SnnPESubComponent::buildWeightDrivenRoutes() {
    // 需要 weights_template_ 包含 {pe} 占位符；需要 weights_cols_ 和 num_neurons_ 定义行/列
    if (weights_template_.empty()) {
        output_->verbose(CALL_INFO, 1, 0, "⚠️ 路由构建失败：weights_template 未提供\n");
        return false;
    }
    // 当前实现按 row-major dense float 读取；若模板指向 BCSR/稀疏文件，提前告警并放弃构建，避免误读导致垃圾路由。
    if (weights_template_.find(".bcsr") != std::string::npos || weights_template_.find(".BCSR") != std::string::npos) {
        return buildWeightDrivenRoutesFromBcsr();
    }
    routes_by_source_.clear();
    const uint32_t rows = num_neurons_;          // 每PE行数（本地目标神经元数）
    const uint32_t cols = weights_cols_;         // 全网列数（全局源神经元数）
    const uint32_t total_nodes = total_nodes_cfg_;
    const size_t expected = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    // 临时结构：pre_global -> list of (abs(weight), dest_global)
    std::unordered_map<uint32_t, std::vector<std::pair<float,uint32_t>>> tmp;
    tmp.reserve(cols);
    uint64_t dropped_self_pe = 0;
    uint64_t dropped_layer_mask = 0;
    // 遍历所有PE的权重文件，建立 pre_global -> 目的候选列表
    for (uint32_t pe = 0; pe < total_nodes; ++pe) {
        // 生成路径
        std::string path = weights_template_;
        // 支持 {pe:02d} 和 {pe}
        size_t pos = path.find("{pe:02d}");
        if (pos != std::string::npos) {
            char buf[16]; std::snprintf(buf, sizeof(buf), "%02u", pe);
            path.replace(pos, 8, buf);
        } else {
            pos = path.find("{pe}");
            if (pos != std::string::npos) path.replace(pos, 4, std::to_string(pe));
        }
        std::ifstream fin(path, std::ios::binary);
        if (!fin.good()) {
            output_->verbose(CALL_INFO, 1, 0, "⚠️ 路由构建：无法读取权重文件 %s\n", path.c_str());
            continue;
        }
        fin.seekg(0, std::ios::end);
        std::streamsize bytes = fin.tellg();
        fin.seekg(0, std::ios::beg);
        if (bytes <= 0 || (bytes % sizeof(float) != 0)) {
            output_->verbose(CALL_INFO, 1, 0, "⚠️ 路由构建：文件尺寸异常 %s\n", path.c_str());
            continue;
        }
        size_t count = static_cast<size_t>(bytes / sizeof(float));
        if (count < expected) {
            output_->verbose(CALL_INFO, 1, 0, "⚠️ 路由构建：文件过短 %s (%zu<%zu)\n", path.c_str(), count, expected);
            continue;
        }
        std::vector<float> buf(count);
        fin.read(reinterpret_cast<char*>(buf.data()), bytes);
        // 行优先：row-major，index = row*cols + col
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t col = 0; col < cols; ++col) {
                size_t idx = static_cast<size_t>(row) * static_cast<size_t>(cols) + col;
                float w = buf[idx];
                if (std::fabs(w) > routing_epsilon_) {
                    uint32_t pre_global = col;
                    uint32_t dest_global = pe * rows + row; // 每PE的全局基按 rows 间隔
                    // 过滤：同PE
                    if (route_exclude_self_pe_) {
                        uint32_t src_pe = pre_global / rows;
                        if (src_pe == pe) { dropped_self_pe++; continue; }
                    }
                    // 过滤：层间掩码
                    if (!allow_all_layers_) {
                        uint32_t src_pe = pre_global / rows;
                        uint32_t la = getLayerIdFromPE(src_pe);
                        uint32_t lb = getLayerIdFromPE(pe);
                        uint32_t key = (la<<8) | lb;
                        if (allowed_layer_edges_.find(key) == allowed_layer_edges_.end()) { dropped_layer_mask++; continue; }
                    }
                    tmp[pre_global].emplace_back(w, dest_global);
                }
            }
        }
    }
    // 通用构建：对每个源应用 TopK/去重 并写入 routes_by_source_
    buildRoutesFromCandidates(tmp, rows, /*group_by_pe=*/true);
    // 统计与提醒
    uint64_t total_entries = 0;
    for (auto &kv : routes_by_source_) total_entries += (uint64_t)kv.second.size();
    if (stat_routes_entries_) stat_routes_entries_->addData(total_entries);
        if (route_exclude_self_pe_ || !allow_all_layers_) {
        if (route_filter_warn_) {
            SNNDL_LOG(1,
                "⚠️ 路由过滤已启用: exclude_self_pe=%d, layers_mask='%s' (丢弃: self_pe=%" PRIu64 ", layer_mask=%" PRIu64 ")\n",
                route_exclude_self_pe_ ? 1 : 0, route_layers_mask_.c_str(), dropped_self_pe, dropped_layer_mask);
        } else {
            SNNDL_LOG(2,
                "路由过滤: exclude_self_pe=%d, layers_mask='%s' (丢弃: self_pe=%" PRIu64 ", layer_mask=%" PRIu64 ")\n",
                route_exclude_self_pe_ ? 1 : 0, route_layers_mask_.c_str(), dropped_self_pe, dropped_layer_mask);
        }
    }
    return !routes_by_source_.empty();
}

uint32_t SnnPESubComponent::getLayerIdFromPE(uint32_t pe) const {
    // 固定4x4网格层划分：I:0-3, H1:4-7, H2:8-11, O:12-15
    if (pe <= 3) return 0;
    if (pe <= 7) return 1;
    if (pe <= 11) return 2;
    return 3;
}

bool SnnPESubComponent::buildRoutesFromEdgesCSV() {
    routes_by_source_.clear();
    const uint32_t rows = num_neurons_;
    std::ifstream fin(mapping_edges_file_);
    if (!fin.good()) {
        output_->verbose(CALL_INFO, 1, 0, "⚠️ 无法打开映射边文件: %s\n", mapping_edges_file_.c_str());
        return false;
    }
    std::string line;
    if (mapping_csv_has_header_) std::getline(fin, line);
    auto split = [this](const std::string& s)->std::vector<std::string>{
        std::vector<std::string> out; std::string cur; char sep = mapping_csv_separator_.empty() ? ',' : mapping_csv_separator_[0];
        std::istringstream ss(s);
        while (std::getline(ss, cur, sep)) out.push_back(cur);
        if (out.empty()) { std::istringstream ss2(s); while (ss2 >> cur) out.push_back(cur); }
        return out;
    };
    std::unordered_map<uint32_t, std::vector<std::pair<float,uint32_t>>> tmp;
    uint64_t dropped_self = 0, dropped_layer = 0;
    while (std::getline(fin, line)) {
        if (line.empty()) continue;
        auto toks = split(line);
        if (toks.size() < 2) continue;
        uint32_t src = (uint32_t) std::stoul(toks[0]);
        uint32_t dst = (uint32_t) std::stoul(toks[1]);
        float w = 1.0f;
        if (toks.size() >= 3) { try { w = std::stof(toks[2]); } catch(...) { w = 1.0f; } }
        if (std::fabs(w) <= routing_epsilon_) continue;
        if (mapping_assume_block_ids_) {
            uint32_t src_pe = src / rows;
            uint32_t dst_pe = dst / rows;
            if (route_exclude_self_pe_ && src_pe == dst_pe) { dropped_self++; continue; }
            if (!allow_all_layers_) {
                uint32_t la = getLayerIdFromPE(src_pe);
                uint32_t lb = getLayerIdFromPE(dst_pe);
                uint32_t key = (la<<8) | lb;
                if (allowed_layer_edges_.find(key) == allowed_layer_edges_.end()) { dropped_layer++; continue; }
            }
        }
        tmp[src].emplace_back(std::fabs(w), dst);
    }
    // 通用构建：若 global_id 不保证 block 映射，则不按PE分组
    buildRoutesFromCandidates(tmp, rows, /*group_by_pe=*/mapping_assume_block_ids_);
    uint64_t total_entries = 0; for (auto &kv : routes_by_source_) total_entries += (uint64_t)kv.second.size();
    if (stat_routes_entries_) stat_routes_entries_->addData(total_entries);
    if ((route_exclude_self_pe_ || !allow_all_layers_) && route_filter_warn_) {
        SNNDL_LOG(1,
            "⚠️ 路由过滤(映射CSV)启用: exclude_self_pe=%d, layers_mask='%s' (丢弃: self=%" PRIu64 ", layer=%" PRIu64 ")\n",
            route_exclude_self_pe_?1:0, route_layers_mask_.c_str(), dropped_self, dropped_layer);
    }
    return !routes_by_source_.empty();
}

// Shared route-building helper to remove duplicated TopK/merge logic
void SnnPESubComponent::buildRoutesFromCandidates(
    const std::unordered_map<uint32_t, std::vector<std::pair<float,uint32_t>>>& tmp,
    uint32_t rows,
    bool group_by_pe)
{
    for (auto &kv : tmp) {
        uint32_t pre = kv.first;
        const auto &lst_in = kv.second;
        if (lst_in.empty()) continue;

        // Work on a copy when mutations are needed
        std::vector<std::pair<float,uint32_t>> lst = lst_in;
        std::vector<uint32_t> final_routes;

        if (routing_topk_per_pe_ > 0) {
            // Group by PE (or single group when group_by_pe=false)
            std::unordered_map<uint32_t, std::vector<std::pair<float,uint32_t>>> by_group;
            by_group.reserve(16);
            for (auto &p : lst) {
                uint32_t gid = group_by_pe ? (p.second / rows) : 0u;
                by_group[gid].push_back(p);
            }
            std::vector<uint32_t> out;
            for (auto &g : by_group) {
                auto &vec = g.second;
                if (vec.size() > routing_topk_per_pe_) {
                    std::partial_sort(vec.begin(), vec.begin()+routing_topk_per_pe_, vec.end(),
                        [](const auto& a, const auto& b){
                            float aw = std::fabs(a.first);
                            float bw = std::fabs(b.first);
                            if (aw == bw) return a.second < b.second;
                            return aw > bw; 
                        });
                    vec.resize(routing_topk_per_pe_);
                }
                for (auto &p : vec) out.push_back(p.second);
            }
            if (routing_topk_ > 0 && out.size() > routing_topk_) {
                // Re-score using original weights to apply global top-k
                std::vector<std::pair<float,uint32_t>> tmp2; tmp2.reserve(out.size());
                std::unordered_set<uint32_t> keep(out.begin(), out.end());
                for (auto &p : lst) if (keep.count(p.second)) tmp2.push_back(p);
                std::partial_sort(tmp2.begin(), tmp2.begin()+routing_topk_, tmp2.end(),
                    [](const auto& a, const auto& b){
                        float aw = std::fabs(a.first);
                        float bw = std::fabs(b.first);
                        if (aw == bw) return a.second < b.second;
                        return aw > bw;
                    });
                tmp2.resize(routing_topk_);
                std::vector<uint32_t> final_out; final_out.reserve(tmp2.size());
                for (auto &p : tmp2) final_out.push_back(p.second);
                final_routes = std::move(final_out);
            } else {
                final_routes = std::move(out);
            }
        } else if (routing_topk_ > 0) {
            if (lst.size() > routing_topk_) {
                std::partial_sort(lst.begin(), lst.begin()+routing_topk_, lst.end(),
                    [](const auto& a, const auto& b){
                        float aw = std::fabs(a.first);
                        float bw = std::fabs(b.first);
                        if (aw == bw) return a.second < b.second;
                        return aw > bw;
                    });
                lst.resize(routing_topk_);
            }
            std::vector<uint32_t> out; out.reserve(lst.size());
            for (auto &p : lst) out.push_back(p.second);
            final_routes = std::move(out);
        } else {
            // No top-k: just deduplicate by destination
            std::sort(lst.begin(), lst.end(), [](const auto& a, const auto& b){ return a.second < b.second; });
            lst.erase(std::unique(lst.begin(), lst.end(), [](const auto& a, const auto& b){ return a.second==b.second; }), lst.end());
            std::vector<uint32_t> out; out.reserve(lst.size());
            for (auto &p : lst) out.push_back(p.second);
            final_routes = std::move(out);
        }

        if (verify_routing_weights_) {
            uint64_t mismatch = 0;
            float min_abs = std::numeric_limits<float>::infinity();
            for (uint32_t dest : final_routes) {
                float w = 0.0f;
                bool found = false;
                for (const auto& cand : lst_in) {
                    if (cand.second == dest) { w = cand.first; found = true; break; }
                }
                if (!found) {
                    output_->verbose(CALL_INFO, 0, 0,
                                     "⚠️ 路由验证: pre=%u dest=%u 未在候选列表中找到原始权重\n",
                                     pre, dest);
                    continue;
                }
                float absw = std::fabs(w);
                if (absw < min_abs) min_abs = absw;
                if (absw <= routing_epsilon_) {
                    mismatch++;
                    output_->verbose(CALL_INFO, 0, 0,
                                     "⚠️ 路由验证: pre=%u dest=%u weight=%.6f <= epsilon=%.6f\n",
                                     pre, dest, w, routing_epsilon_);
                }
            }
        if (!final_routes.empty()) {
            float report_min = std::isfinite(min_abs) ? min_abs : 0.0f;
            output_->verbose(CALL_INFO, 0, 0,
                "🔍 路由验证: pre=%u fanout=%zu min_abs_weight=%.6f\n",
                pre, final_routes.size(), report_min);
        }
        if (mismatch > 0) {
            output_->verbose(CALL_INFO, 0, 0,
                "⚠️ 路由验证: pre=%u 存在%" PRIu64 "个未达阈值的扇出\n",
                pre, mismatch);
        }
        }

        routes_by_source_[pre] = std::move(final_routes);
    }
    // stats updated by caller (to keep original behavior/placement)
}
// === LRU cache helpers ===
bool SnnPESubComponent::cacheGet_(uint64_t key, float& out) {
    if (disable_weight_cache_) return false; // 诊断：强制miss以触发requestWeight
    if (use_clock_weight_cache_) {
        return clockGet_(key, out);
    }
    auto it = weight_cache_.find(key);
    if (it == weight_cache_.end()) return false;
    cache_lru_list_.erase(it->second.it);
    cache_lru_list_.push_front(key);
    it->second.it = cache_lru_list_.begin();
    out = it->second.value;
    return true;
}

bool SnnPESubComponent::buildWeightDrivenRoutesFromBcsr() {
    routes_by_source_.clear();
    const uint32_t cores_per_pe = (total_cores_ > 0) ? static_cast<uint32_t>(total_cores_) : 1u;
    uint32_t rows_hint = (cores_per_pe > 0) ? static_cast<uint32_t>((num_neurons_ + cores_per_pe - 1) / cores_per_pe) : num_neurons_;
    if (rows_hint == 0) rows_hint = num_neurons_;
    bool ok = true;
    for (uint32_t pe = 0; pe < total_nodes_cfg_; ++pe) {
        for (uint32_t core = 0; core < cores_per_pe; ++core) {
            std::string path = resolveWeightTemplate(pe, static_cast<int>(core));
            if (path.empty()) { ok = false; break; }
            if (!appendRoutesFromBcsrFile(path, pe, static_cast<int>(core), rows_hint)) {
                ok = false;
                break;
            }
        }
        if (!ok) break;
    }
    if (!ok) {
        routes_by_source_.clear();
        return false;
    }
    return !routes_by_source_.empty();
}

bool SnnPESubComponent::appendRoutesFromBcsrFile(const std::string& path, uint32_t pe_index, int core_index, uint32_t rows_hint) {
    uint32_t rows = rows_hint;
    uint32_t cols = weights_cols_;
    uint32_t br = bcsr_weights_.effectiveBlockRows();
    uint32_t bc = bcsr_weights_.effectiveBlockCols();
    uint32_t idx_bytes = bcsr_weights_.effectiveIdxBytes();
    uint32_t val_bytes = bcsr_weights_.effectiveValBytes();
    uint64_t rowptr_off = (bcsr_weights_.rowptrAddr() > base_addr_) ? (bcsr_weights_.rowptrAddr() - base_addr_) : 0;
    uint64_t colidx_off = (bcsr_weights_.colidxAddr() > base_addr_) ? (bcsr_weights_.colidxAddr() - base_addr_) : 0;
    uint64_t blockdata_off = (bcsr_weights_.blockdataAddr() > base_addr_) ? (bcsr_weights_.blockdataAddr() - base_addr_) : 0;
    uint64_t blockids_off = (bcsr_weights_.blockidsAddr() > base_addr_) ? (bcsr_weights_.blockidsAddr() - base_addr_) : 0;
    uint32_t total_blocks = 0;
    uint32_t meta_cols = cols;
    const uint64_t pe_base_global = static_cast<uint64_t>(pe_index) * static_cast<uint64_t>(num_neurons_);
    const std::string meta_path = path + ".meta.json";
    if (parseBcsrMeta(meta_path, rows, meta_cols, br, bc, idx_bytes, val_bytes,
                      rowptr_off, colidx_off, blockdata_off, blockids_off, total_blocks)) {
        if (meta_cols > 0) cols = meta_cols;
    } else {
        total_blocks = 0; // will be derived from rowptr later
    }
    if (rows == 0 || cols == 0) {
        output_->verbose(CALL_INFO, 0, 0, "⚠️ BCSR路由: 元数据缺失 rows/cols %s\n", path.c_str());
        return false;
    }
    if (val_bytes != 4) {
        output_->verbose(CALL_INFO, 0, 0, "⚠️ BCSR路由: val_bytes=%u 不受支持 %s\n", val_bytes, path.c_str());
        return false;
    }
    const uint32_t n_block_rows = (rows + br - 1) / br;
    const uint64_t core_offset_global = (core_index > 0) ? static_cast<uint64_t>(core_index) * static_cast<uint64_t>(rows) : 0ULL;
    std::ifstream fin(path, std::ios::binary);
    if (!fin.good()) {
        output_->verbose(CALL_INFO, 0, 0, "⚠️ BCSR路由: 无法读取 %s\n", path.c_str());
        return false;
    }
    fin.seekg(static_cast<std::streamoff>(rowptr_off), std::ios::beg);
    std::vector<uint32_t> rowptr(n_block_rows + 1, 0);
    fin.read(reinterpret_cast<char*>(rowptr.data()), rowptr.size() * sizeof(uint32_t));
    if (!fin.good()) {
        output_->verbose(CALL_INFO, 0, 0, "⚠️ BCSR路由: 读取rowptr失败 %s\n", path.c_str());
        return false;
    }
    uint32_t derived_blocks = rowptr.back();
    if (total_blocks == 0) total_blocks = derived_blocks;
    std::vector<uint32_t> block_cols(total_blocks, 0u);
    fin.seekg(static_cast<std::streamoff>(colidx_off), std::ios::beg);
    if (idx_bytes == 2) {
        std::vector<uint16_t> tmp(total_blocks, 0);
        fin.read(reinterpret_cast<char*>(tmp.data()), tmp.size() * sizeof(uint16_t));
        if (!fin.good()) {
            output_->verbose(CALL_INFO, 0, 0, "⚠️ BCSR路由: 读取colidx失败 %s\n", path.c_str());
            return false;
        }
        for (uint32_t i = 0; i < total_blocks; ++i) block_cols[i] = tmp[i];
    } else {
        fin.read(reinterpret_cast<char*>(block_cols.data()), block_cols.size() * sizeof(uint32_t));
        if (!fin.good()) {
            output_->verbose(CALL_INFO, 0, 0, "⚠️ BCSR路由: 读取colidx失败 %s\n", path.c_str());
            return false;
        }
    }
    std::ifstream fdata(path, std::ios::binary);
    std::ifstream fids(path, std::ios::binary);
    fdata.seekg(static_cast<std::streamoff>(blockdata_off), std::ios::beg);
    if (blockids_off > 0) {
        fids.seekg(static_cast<std::streamoff>(blockids_off), std::ios::beg);
    }
    const size_t floats_per_block = static_cast<size_t>(br) * static_cast<size_t>(bc);
    std::vector<float> blockdata(floats_per_block, 0.0f);
    std::vector<uint32_t> blockids(floats_per_block, BCSR_SENTINEL_ID);
    const uint64_t total_global_neurons = static_cast<uint64_t>(total_nodes_cfg_) * static_cast<uint64_t>(neurons_per_pe_cfg_ > 0 ? neurons_per_pe_cfg_ : num_neurons_);
    uint64_t block_counter = 0;
    for (uint32_t block_row = 0; block_row < n_block_rows; ++block_row) {
        uint32_t begin = rowptr[block_row];
        uint32_t end = rowptr[block_row + 1];
        for (uint32_t idx = begin; idx < end; ++idx, ++block_counter) {
            if (block_counter >= block_cols.size()) break;
            uint32_t block_col = block_cols[idx];
            fdata.read(reinterpret_cast<char*>(blockdata.data()), blockdata.size() * sizeof(float));
            if (blockids_off > 0) {
                fids.read(reinterpret_cast<char*>(blockids.data()), blockids.size() * sizeof(uint32_t));
            }
            if (!fdata.good() || (blockids_off > 0 && !fids.good())) {
                output_->verbose(CALL_INFO, 0, 0, "⚠️ BCSR路由: 读取block数据失败 %s\n", path.c_str());
                return false;
            }
            for (uint32_t rr = 0; rr < br; ++rr) {
                uint32_t post_local = block_row * br + rr;
                if (post_local >= rows) continue;
                for (uint32_t cc = 0; cc < bc; ++cc) {
                    size_t off = static_cast<size_t>(rr) * bc + cc;
                    float weight = blockdata[off];
                    if (std::fabs(weight) <= routing_epsilon_) continue;
                    uint32_t post_global = (blockids_off > 0) ? blockids[off]
                        : static_cast<uint32_t>(pe_base_global + core_offset_global + post_local);
                    if (post_global == BCSR_SENTINEL_ID) continue;
                    if (post_global >= total_global_neurons) continue;
                    uint32_t pre_global = block_col * bc + cc;
                    if (pre_global >= cols) continue;
                    routes_by_source_[pre_global].push_back(post_global);
                }
            }
        }
    }
    return true;
}

bool SnnPESubComponent::parseBcsrMeta(const std::string& meta_path, uint32_t& rows_out, uint32_t& cols_out,
                                      uint32_t& br_out, uint32_t& bc_out,
                                      uint32_t& idx_bytes_out, uint32_t& val_bytes_out,
                                      uint64_t& rowptr_off_out, uint64_t& colidx_off_out,
                                      uint64_t& blockdata_off_out, uint64_t& blockids_off_out,
                                      uint32_t& total_blocks_out) const {
    std::ifstream meta(meta_path);
    if (!meta.good()) return false;
    std::string text((std::istreambuf_iterator<char>(meta)), std::istreambuf_iterator<char>());
    uint64_t value = 0;
    bool ok = false;
    if (extractUnsigned(text, "\"rows\"", value)) { rows_out = static_cast<uint32_t>(value); ok = true; }
    if (extractUnsigned(text, "\"cols\"", value)) { cols_out = static_cast<uint32_t>(value); ok = true; }
    if (extractUnsigned(text, "\"br\"", value)) { br_out = static_cast<uint32_t>(value); ok = true; }
    if (extractUnsigned(text, "\"bc\"", value)) { bc_out = static_cast<uint32_t>(value); ok = true; }
    if (extractUnsigned(text, "\"idx_bytes\"", value)) { idx_bytes_out = static_cast<uint32_t>(value); ok = true; }
    if (extractUnsigned(text, "\"val_bytes\"", value)) { val_bytes_out = static_cast<uint32_t>(value); ok = true; }
    if (extractUnsigned(text, "\"rowptr_offset\"", value)) { rowptr_off_out = value; ok = true; }
    if (extractUnsigned(text, "\"colidx_offset\"", value)) { colidx_off_out = value; ok = true; }
    if (extractUnsigned(text, "\"blockdata_offset\"", value)) { blockdata_off_out = value; ok = true; }
    if (extractUnsigned(text, "\"blockids_offset\"", value)) { blockids_off_out = value; ok = true; }
    if (extractUnsigned(text, "\"total_blocks\"", value)) { total_blocks_out = static_cast<uint32_t>(value); ok = true; }
    return ok;
}

std::string SnnPESubComponent::resolveWeightTemplate(uint32_t pe, int core) const {
    if (weights_template_.empty()) return "";
    std::string path = weights_template_;
    auto replaceIndexed = [&](const std::string& marker, uint32_t value, int width) {
        size_t pos = 0;
        while ((pos = path.find(marker, pos)) != std::string::npos) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%0*u", width, value);
            path.replace(pos, marker.size(), buf);
            pos += width;
        }
    };
    auto replaceSimple = [&](const std::string& marker, uint32_t value) {
        size_t pos = 0;
        std::string text = std::to_string(value);
        while ((pos = path.find(marker, pos)) != std::string::npos) {
            path.replace(pos, marker.size(), text);
            pos += text.size();
        }
    };
    replaceIndexed("{pe:02d}", pe, 2);
    replaceSimple("{pe}", pe);
    replaceIndexed("{core:02d}", static_cast<uint32_t>(core), 2);
    replaceSimple("{core}", static_cast<uint32_t>(core));
    return path;
}

void SnnPESubComponent::cachePut_(uint64_t key, float value) {
    if (use_clock_weight_cache_) {
        clockPut_(key, value);
        return;
    }
    auto it = weight_cache_.find(key);
    if (it != weight_cache_.end()) {
        it->second.value = value;
        cache_lru_list_.erase(it->second.it);
        cache_lru_list_.push_front(key);
        it->second.it = cache_lru_list_.begin();
    } else {
        cache_lru_list_.push_front(key);
        CacheEntry entry{value, cache_lru_list_.begin()};
        weight_cache_.emplace(key, std::move(entry));
        if (max_cache_entries_ > 0 && weight_cache_.size() > max_cache_entries_) {
            uint64_t victim = cache_lru_list_.back();
            cache_lru_list_.pop_back();
            weight_cache_.erase(victim);
            if (stat_cache_evictions_) stat_cache_evictions_->addData(1);
            count_cache_evictions_++;
        }
    }
}

bool SnnPESubComponent::clockGet_(uint64_t key, float& out) {
    auto it = wcache_index_.find(key);
    if (it == wcache_index_.end()) return false;
    uint32_t idx = it->second;
    if (idx >= wcache_size_) return false;
    wcache_access_[idx] = 1;
    out = wcache_vals_[idx];
    return true;
}

void SnnPESubComponent::clockPut_(uint64_t key, float value) {
    auto it = wcache_index_.find(key);
    if (it != wcache_index_.end()) {
        uint32_t idx = it->second;
        if (idx < wcache_size_) {
            wcache_vals_[idx] = value;
            wcache_access_[idx] = 1;
            return;
        }
        wcache_index_.erase(it);
    }
    if (wcache_cap_ == 0) return;
    if (wcache_size_ < wcache_cap_) {
        uint32_t idx = wcache_size_++;
        wcache_keys_[idx] = key;
        wcache_vals_[idx] = value;
        wcache_access_[idx] = 1;
        wcache_index_[key] = idx;
    } else {
        while (wcache_access_[wcache_hand_]) {
            wcache_access_[wcache_hand_] = 0;
            wcache_hand_ = (wcache_hand_ + 1) % wcache_cap_;
        }
        uint32_t idx = wcache_hand_;
        uint64_t victim = wcache_keys_[idx];
        wcache_index_.erase(victim);
        wcache_keys_[idx] = key;
        wcache_vals_[idx] = value;
        wcache_access_[idx] = 1;
        wcache_index_[key] = idx;
        wcache_hand_ = (wcache_hand_ + 1) % wcache_cap_;
    }
}
void SnnPESubComponent::applyGatingDecision(uint32_t src_global, const std::vector<uint32_t>& dest_pes,
                             uint64_t current_cycle, uint64_t ttl_cycles)
{
    if (!gating_event_mode_) return;
    GatingEntry e; e.dest_pes = dest_pes; e.expire_cycle = current_cycle + (ttl_cycles ? ttl_cycles : gating_ttl_cycles_cfg_);
    gating_cache_[src_global] = std::move(e);
    output_->verbose(CALL_INFO, 3, 0, "📥 应用门控: src_g=%u, k=%zu, expire=%" PRIu64 "\n", src_global, dest_pes.size(), gating_cache_[src_global].expire_cycle);
}
// === BCSR 辅助实现 ===
bool SnnPESubComponent::bcsrRowIndexGet_(uint32_t block_row, std::vector<uint32_t>& out) {
    if (!use_bcsr_) return false;
    if (!bcsr_weights_.rowIndexGet(block_row, out)) return false;
    bcsr_count_row_index_hits_++;
    return true;
}

void SnnPESubComponent::bcsrRowIndexPut_(uint32_t block_row, std::vector<uint32_t>& cols) {
    if (!use_bcsr_) return;
    bcsr_weights_.rowIndexPut(block_row, std::move(cols));
}

bool SnnPESubComponent::bcsrBlockGet_(uint32_t block_row, uint32_t block_col, std::vector<float>& out) {
    if (!use_bcsr_) return false;
    if (!bcsr_weights_.blockGet(block_row, block_col, out)) return false;
    bcsr_count_block_hits_++;
    return true;
}

void SnnPESubComponent::bcsrBlockPut_(uint32_t block_row, uint32_t block_col, std::vector<float>& data) {
    if (!use_bcsr_) return;
    if (!data.empty()) {
        bcsrPopulateWeightCache_(block_row, block_col, data);
    }
    bcsr_weights_.blockPut(block_row, block_col, std::move(data));
}

// CSR 读路径已移除

void SnnPESubComponent::requestWeightBCSR(uint32_t pre_global, uint32_t post_local, std::function<void(float)> cb) {
    recordActivePre_(pre_global);
    bcsr_req_edges_++;
    // 诊断分支：强制从文件读取（仅在bcsr_force_file_read_开启时，用于验证权重正确性/可见性）
    if (bcsr_force_file_read_) {
        float w = readBcsrWeightFromFile_(post_local, pre_global);
        if (cb) cb(w);
        return;
    }
    // 前置检查
    if (!bcsr_weights_.isRowptrReady()) {
        bcsr_req_wait_rowptr_++;
        if (window_read_debug_ && output_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-bcsr] core=%u rowptr not ready pre=%u post=%u\n",
                core_id_, pre_global, post_local);
        }
        if (cb) cb(0.0f);
        return;
    }
    uint32_t rows = num_neurons_;
    uint32_t br = bcsr_weights_.effectiveBlockRows();
    uint32_t bc = bcsr_weights_.effectiveBlockCols();
    uint32_t block_row = (br ? (post_local / br) : 0);
    uint32_t intra_row = (br ? (post_local % br) : 0);
    uint32_t block_col = (bc ? (pre_global / bc) : 0);
    uint32_t intra_col = (bc ? (pre_global % bc) : 0);
    uint32_t start = 0, end = 0;
    if (!bcsr_weights_.rowBounds(block_row, start, end)) {
        if (window_read_debug_ && output_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-bcsr] core=%u block_row=%u rowptr_size=%zu out-of-range\n",
                core_id_, block_row, bcsr_weights_.rowptrHost().size());
        }
        if (cb) cb(0.0f);
        return;
    }
    if (end <= start) {
        if (window_read_debug_ && output_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-bcsr] core=%u row empty post_local=%u block_row=%u start=%u end=%u\n",
                core_id_, post_local, block_row, start, end);
        }
        if (cb) cb(0.0f);
        return;
    }
    // 获取行索引段
    std::vector<uint32_t> cols;
    if (!bcsrRowIndexGet_(block_row, cols)) {
        // 发起读取 colidx 段
        size_t bytes = bcsr_weights_.colIndexBytes(end - start);
        uint64_t addr = bcsr_weights_.colIndexAddr(start);
        if (window_read_debug_ && output_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-bcsr] core=%u issue colidx row=%u start=%u end=%u bytes=%zu addr=0x%llx\n",
                core_id_, block_row, start, end, bytes, (unsigned long long)addr);
        }
        auto* rd = new SST::Interfaces::StandardMem::Read(addr, bytes);
        auto id = rd->getID();
        PendingMemoryRequest pm; pm.request_id = id; pm.address = addr; pm.size = bytes; pm.issue_cycle = total_cycles_;
        pm.bcsr_kind = 2; pm.bcsr_block_row = block_row; pm.bcsr_target_block_col = block_col; pm.bcsr_intra_row = intra_row; pm.bcsr_intra_col = intra_col;
        pm.bcsr_row_start = start; pm.has_single_cb = (cb!=nullptr); pm.single_cb = cb;
        stats_reporter_.reportMemoryIssue(bytes, false);
        pending_memory_requests_[id] = pm;
        memory_->send(rd);
        return;
    }
    // 查找目标块列
    uint32_t idx_in_row=0; bool found=false;
    for (size_t i=0;i<cols.size();++i){ if (cols[i]==block_col){ idx_in_row=(uint32_t)i; found=true; break; } }
        if (!found) {
            bcsr_req_block_miss_++;
            if (window_read_debug_ && output_) {
                std::string sample_cols;
                if (!cols.empty()) {
                    const size_t limit = std::min<size_t>(cols.size(), 8);
                    sample_cols.reserve(limit * 6);
                    for (size_t si = 0; si < limit; ++si) {
                        sample_cols += std::to_string(cols[si]);
                        if (si + 1 < limit) sample_cols += ",";
                    }
                }
                output_->verbose(CALL_INFO, 0, 0,
                    "[diag-bcsr] core=%u block_row=%u pre=%u block_col=%u miss colidx range=[%u,%u) sample=[%s]%s\n",
                    core_id_, block_row, pre_global, block_col, start, end,
                    sample_cols.c_str(), cols.size() > 8 ? "..." : "");
            }
            if (cb) cb(0.0f);
            return;
        }
    // 尝试块缓存
    std::vector<float> blk;
    if (bcsrBlockGet_(block_row, block_col, blk)) {
        bcsr_req_block_hit_++;
        uint32_t off = intra_row * bc + intra_col;
        float w = (off < blk.size() ? blk[off] : 0.0f);
        if (window_read_debug_ && output_) {
            uint32_t post_local = block_row * br + intra_row;
            uint32_t post_global = global_neuron_base_ + post_local;
            uint32_t pre_global_effective = block_col * bc + intra_col;
            output_->verbose(CALL_INFO, 1, 0,
                "[diag-bcsr-weight] core=%u post_local=%u post_global=%u pre_global=%u weight=%.6f source=BcsrCache\n",
                core_id_, post_local, post_global, pre_global_effective, w);
        }
        if (cb) cb(w);
        return;
    }
    // 读取块数据
    size_t block_bytes = bcsr_weights_.blockBytes();
    uint32_t global_block_index = start + idx_in_row;
    uint64_t addr = bcsr_weights_.blockDataAddr(global_block_index);
    auto* rd = new SST::Interfaces::StandardMem::Read(addr, block_bytes);
    auto id = rd->getID();
    PendingMemoryRequest pm; pm.request_id = id; pm.address = addr; pm.size = block_bytes; pm.issue_cycle = total_cycles_;
    pm.bcsr_kind = 3; pm.bcsr_block_row = block_row; pm.bcsr_target_block_col = block_col; pm.bcsr_intra_row = intra_row; pm.bcsr_intra_col = intra_col;
    pm.bcsr_row_start = start; pm.bcsr_idx_in_row = idx_in_row; pm.bcsr_global_block_index = global_block_index;
    pm.has_single_cb = (cb!=nullptr); pm.single_cb = cb;
    stats_reporter_.reportMemoryIssue(block_bytes, true);
    pending_memory_requests_[id] = pm;
    memory_->send(rd);
    bcsr_count_block_misses_++;
}

void SnnPESubComponent::bcsrPrefetchAll_() {
    if (!use_bcsr_) return;
    if (!bcsr_prefetch_all_) return;
    if (bcsr_prefetch_issued_) return;
    if (!memory_) return;
    if (!bcsr_weights_.isRowptrReady()) return;

    uint32_t rows = num_neurons_;
    uint32_t br = bcsr_weights_.effectiveBlockRows();
    uint32_t nBlockRows = (rows + br - 1) / br;
    for (uint32_t block_row = 0; block_row < nBlockRows; ++block_row) {
        const auto& rp = bcsr_weights_.rowptrHost();
        if (block_row + 1 >= rp.size()) break;
        uint32_t start = rp[block_row];
        uint32_t end   = rp[block_row+1];
        if (end <= start) continue;
        std::vector<uint32_t> cached_cols;
        if (bcsrRowIndexGet_(block_row, cached_cols)) {
            bcsrPrefetchRowBlocks_(block_row, cached_cols, start);
            continue;
        }
        size_t bytes = bcsr_weights_.colIndexBytes(end - start);
        uint64_t addr = bcsr_weights_.colIndexAddr(start);
        auto* rd = new SST::Interfaces::StandardMem::Read(addr, bytes);
        auto id = rd->getID();
        PendingMemoryRequest pm;
        pm.request_id = id;
        pm.address = addr;
        pm.size = bytes;
        pm.issue_cycle = total_cycles_;
        pm.is_weight = true; // colidx 属于权重索引段
        pm.bcsr_kind = 2;
        pm.bcsr_block_row = block_row;
        pm.bcsr_row_start = start;
        pm.bcsr_prefetch_all = true;
        pm.bcsr_target_block_col = UINT32_MAX;
        stats_reporter_.reportMemoryIssue(bytes, false);
        pending_memory_requests_[id] = pm;
        memory_->send(rd);
    }
    bcsr_prefetch_issued_ = true;
}

void SnnPESubComponent::bcsrPrefetchRowBlocks_(uint32_t block_row, const std::vector<uint32_t>& cols, uint32_t row_start) {
    if (!use_bcsr_) return;
    if (!memory_) return;
    uint32_t br = bcsr_weights_.effectiveBlockRows();
    uint32_t bc = bcsr_weights_.effectiveBlockCols();
    size_t block_bytes = bcsr_weights_.blockBytes();
    for (size_t i = 0; i < cols.size(); ++i) {
        uint32_t block_col = cols[i];
        if (bcsr_weights_.hasBlock(block_row, block_col)) continue;
        uint32_t global_block_index = row_start + static_cast<uint32_t>(i);
        uint64_t addr = bcsr_weights_.blockDataAddr(global_block_index);
        auto* rd = new SST::Interfaces::StandardMem::Read(addr, block_bytes);
        auto id = rd->getID();
        PendingMemoryRequest pm;
        pm.request_id = id;
        pm.address = addr;
        pm.size = block_bytes;
        pm.issue_cycle = total_cycles_;
        pm.is_weight = true; // 块数据属于权重
        pm.bcsr_kind = 3;
        pm.bcsr_block_row = block_row;
        pm.bcsr_target_block_col = block_col;
        pm.bcsr_row_start = row_start;
        pm.bcsr_idx_in_row = static_cast<uint32_t>(i);
        pm.bcsr_global_block_index = global_block_index;
        pm.bcsr_prefetch_all = true;
        stats_reporter_.reportMemoryIssue(block_bytes, true);
        pending_memory_requests_[id] = pm;
        memory_->send(rd);
        bcsr_count_block_misses_++;
    }
}

void SnnPESubComponent::bcsrPopulateWeightCache_(uint32_t block_row, uint32_t block_col, const std::vector<float>& blk) {
    if (!use_bcsr_) return;
    if (blk.empty()) return;
    uint32_t br = bcsr_weights_.effectiveBlockRows();
    uint32_t bc = bcsr_weights_.effectiveBlockCols();
    uint32_t row_base = block_row * br;
    uint32_t col_base = block_col * bc;
    for (uint32_t rr = 0; rr < br; ++rr) {
        uint32_t post_local = row_base + rr;
        if (post_local >= num_neurons_) break;
        for (uint32_t cc = 0; cc < bc; ++cc) {
            uint32_t pre_global = col_base + cc;
            if (pre_global >= weights_cols_) break;
            size_t idx = static_cast<size_t>(rr) * static_cast<size_t>(bc) + cc;
            if (idx >= blk.size()) break;
            float w = blk[idx];
            uint64_t key = (uint64_t)post_local * (uint64_t)weights_cols_ + pre_global;
            weightCacheStore_(key, w);
        }
    }
}

size_t SnnPESubComponent::expectedRowptrEntries_() const {
    return bcsr_weights_.expectedRowptrEntries(num_neurons_);
}

size_t SnnPESubComponent::expectedRowptrBytes_() const {
    return bcsr_weights_.expectedRowptrBytes(num_neurons_);
}

bool SnnPESubComponent::installRowptrFromBytes_(const uint8_t* data, size_t bytes, const char* source, bool count_stats) {
    if (!bcsr_weights_.installRowptrFromBytes(data, bytes, num_neurons_)) {
        SNNDL_DEBUG_LOG(0,
            "[diag-bcsr] core=%u rowptr install failed source=%s bytes=%zu expect=%zu\n",
            core_id_, source ? source : "-", bytes, expectedRowptrBytes_());
        return false;
    }
    if (count_stats) {
        bcsr_count_row_reads_++;
        bcsr_bytes_idx_ += bytes;
    }
    if (window_read_debug_ && output_) {
        const auto& rp = bcsr_weights_.rowptrHost();
        output_->verbose(CALL_INFO, 0, 0,
            "[diag-bcsr] core=%u rowptr ready entries=%zu first=%u second=%u\n",
            core_id_, rp.size(),
            rp.empty()?0u:rp[0],
            rp.size()>1?rp[1]:0u);
    }
    return true;
}

bool SnnPESubComponent::loadBcsrRowptrFromFile_() {
    if (weights_template_.empty()) return false;
    std::string bin_path = resolveWeightTemplate(node_id_, core_id_);
    if (bin_path.empty()) return false;
    std::string meta_path = bin_path + ".meta.json";
    uint32_t rows = 0, cols = 0, br = 0, bc = 0, idx_bytes = 0, val_bytes = 0, total_blocks = 0;
    uint64_t rowptr_off = 0, colidx_off = 0, blockdata_off = 0, blockids_off = 0;
    if (!parseBcsrMeta(meta_path, rows, cols, br, bc, idx_bytes, val_bytes,
                       rowptr_off, colidx_off, blockdata_off, blockids_off, total_blocks)) {
        SNNDL_DEBUG_LOG(0, "[diag-bcsr] core=%u fallback meta parse failed %s\n", core_id_, meta_path.c_str());
        return false;
    }
    std::ifstream fin(bin_path, std::ios::binary);
    if (!fin.good()) {
        SNNDL_DEBUG_LOG(0, "[diag-bcsr] core=%u fallback open failed %s\n", core_id_, bin_path.c_str());
        return false;
    }
    size_t bytes = expectedRowptrBytes_();
    std::vector<uint8_t> buffer(bytes);
    fin.seekg(static_cast<std::streamoff>(rowptr_off), std::ios::beg);
    fin.read(reinterpret_cast<char*>(buffer.data()), bytes);
    if (!fin.good()) {
        SNNDL_DEBUG_LOG(0,
            "[diag-bcsr] core=%u fallback read failed %s rowptr_off=%" PRIu64 " bytes=%zu\n",
            core_id_, bin_path.c_str(), rowptr_off, bytes);
        return false;
    }
    if (!installRowptrFromBytes_(buffer.data(), buffer.size(), "file", false)) {
        return false;
    }
    return true;
}

void SnnPESubComponent::ensureRowptrReadyOrFatal_(const char* reason) {
    const char* msg = reason ? reason : "unknown";
    if (!bcsr_rowptr_file_fallback_enable_) {
        output_->fatal(CALL_INFO, -1,
            "BCSR rowptr load failed for core=%u node=%u (%s). Enable parameter 'bcsr_rowptr_file_fallback_enable=1' or fix rowptr data.\n",
            core_id_, node_id_, msg);
    }
    if (!loadBcsrRowptrFromFile_()) {
        output_->fatal(CALL_INFO, -1,
            "BCSR rowptr load failed (%s) and fallback weight file could not be loaded (core=%u node=%u).\n",
            msg, core_id_, node_id_);
    }
    if (window_read_debug_ && output_) {
        output_->verbose(CALL_INFO, 0, 0,
            "[diag-bcsr] core=%u rowptr fallback applied (%s)\n",
            core_id_, msg);
    }
}
