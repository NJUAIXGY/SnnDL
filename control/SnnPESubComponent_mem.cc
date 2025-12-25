// -*- c++ -*-
//
// SnnPESubComponent_mem.cc: StandardMem request/response tracking and dense read/write
// helpers extracted from SnnPESubComponent.cc. Behavior preserved; only file split.
//

#include <sst/core/sst_config.h>
#include "SnnPESubComponent.h"
#include "synapse/gas/GasCustomCmd.h"
#include "synapse/gas/GasPhaseController.h"
#include "IPeAggregation.h"
#include "memory/StandardMemAccess.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <inttypes.h>

using namespace SST;
using namespace SST::SnnDL;

// Lightweight logging helpers (file-local). Keep consistent with other split units.
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

// === Learning writeback (called by compute core) ===
bool SnnPESubComponent::applyLocalWeightUpdates_(const std::unordered_map<uint64_t, float>& grads,
                                                float learning_rate,
                                                float weight_decay) {
    if (grads.empty()) return true;
    if (!memory_ || !memory_ready_ || !stdmem_access_) return false;
    const uint32_t width = use_post_row_pre_col_ ? weights_cols_ : num_neurons_;
    const size_t bytes_per_float = sizeof(float);
    uint64_t total_writes = 0;
    size_t skipped_uncached = 0;
    for (const auto& kv : grads) {
        uint64_t key = kv.first;
        float grad = kv.second;
        float old_w = 0.0f;
        if (!weightCacheTryGet_(key, old_w)) {
            skipped_uncached++;
            continue;
        }
        float new_w = old_w - learning_rate * grad;
        if (weight_decay != 0.0f) {
            new_w -= weight_decay * old_w;
        }
        uint64_t addr = base_addr_ + key * bytes_per_float;
        std::vector<uint8_t> data(bytes_per_float);
        std::memcpy(data.data(), &new_w, bytes_per_float);
        stats_reporter_.reportMemoryIssue(data.size(), false);
        stdmem_access_->write(addr, data, nullptr);
        total_writes++;
        weightCacheStore_(key, new_w);
    }
    if (output_) {
        output_->verbose(CALL_INFO, 1, 0,
            "📝 学习: 写回完成 writes=%" PRIu64 ", 跳过(未缓存)=%zu\n",
            total_writes, skipped_uncached);
    }
    return true;
}

void SnnPESubComponent::requestWeight(uint32_t pre_neuron, uint32_t post_neuron,
                                     std::function<void(float)> callback) {
    if (use_bcsr_ && use_post_row_pre_col_) {
        requestWeightBCSR(pre_neuron, post_neuron, std::move(callback));
        return;
    }
    if (!ensureMemoryReady_() || !stdmem_access_) {
        if (callback) callback(init_default_weight_);
        return;
    }

    const uint32_t width = use_post_row_pre_col_ ? weights_cols_ : num_neurons_;
    const uint32_t row = use_post_row_pre_col_ ? post_neuron : pre_neuron;
    const uint32_t col = use_post_row_pre_col_ ? pre_neuron : post_neuron;
    if (width == 0 || row >= num_neurons_ || col >= width) {
        if (callback) callback(0.0f);
        return;
    }

    const uint64_t req_addr =
        base_addr_ + (static_cast<uint64_t>(row) * static_cast<uint64_t>(width) + static_cast<uint64_t>(col)) * sizeof(float);
    const uint64_t issue_cycle = static_cast<uint64_t>(total_cycles_);

    if (window_read_debug_ && output_) {
        output_->verbose(CALL_INFO, 2, 0,
            "[diag-read] core=%d requestWeight dense row=%u col=%u addr=0x%llx size=%zu\n",
            core_id_, row, col, (unsigned long long)req_addr, sizeof(float));
    }

    stats_reporter_.reportMemoryIssue(sizeof(float), /*count_weight_read*/true);
    stdmem_access_->read(req_addr, sizeof(float),
        [this, cb = std::move(callback), issue_cycle](IMemoryAccess::RequestId, uint64_t, std::vector<uint8_t>&& data) mutable {
            float w = 0.0f;
            if (data.size() >= sizeof(float)) {
                std::memcpy(&w, data.data(), sizeof(float));
            }
            if (readresp_zero_fallback_ && w == 0.0f) w = init_default_weight_;
            if (cb) cb(w);

            const uint64_t now = static_cast<uint64_t>(total_cycles_);
            if (now >= issue_cycle) {
                const uint64_t lat = now - issue_cycle;
                accum_mem_latency_cycles_ += lat;
                count_mem_responses_++;
                if (auto* pe = parent_pe_cached_) {
                    pe->accumulateMemReadLatency(lat, /*is_weight*/true);
                }
            }
        });
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
                         core_id_, t, static_cast<uint64_t>(req ? req->getID() : 0));
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
                            const size_t edges_curr = weight_mem_subsystem_ ? weight_mem_subsystem_->edgesCurrSize() : 0;
                            const size_t edges_prev = weight_mem_subsystem_ ? weight_mem_subsystem_->edgesPrevSize() : 0;
                            output_->verbose(CALL_INFO, 0, 0,
                                "[diag-edges] BeginGather seq=%u edges_curr=%zu edges_prev=%zu\n",
                                curr_stage_seq_, edges_curr, edges_prev);
                        }
                        if (window_read_debug_) {
                        const size_t posts_prev = weight_mem_subsystem_ ? weight_mem_subsystem_->postsPrevSize() : 0;
                        const size_t pres_prev  = weight_mem_subsystem_ ? weight_mem_subsystem_->presPrevSize() : 0;
                        const size_t posts_curr = weight_mem_subsystem_ ? weight_mem_subsystem_->postsCurrSize() : 0;
                        const size_t pres_curr  = weight_mem_subsystem_ ? weight_mem_subsystem_->presCurrSize() : 0;
                        SNNDL_DEBUG_LOG(1, "[diag-stage] BeginGather window=%" PRIu64 " stage=%d posts_prev=%zu active_prev=%zu posts_curr=%zu active_curr=%zu qsize=%zu\n",
                            (uint64_t)op->superstep, static_cast<int>(gas_stage_),
                            posts_prev, pres_prev,
                            posts_curr, pres_curr, incoming_spikes_.size());
                        }
                        // reset per-window counters
                        {
                        orchestrateBeginGatherWindowSetup();
                        // 新窗开始：复位容量告警 + 迁移 window-read 集合
                        record_edge_capacity_warned_ = false;
                        if (weight_mem_subsystem_) {
                            weight_mem_subsystem_->beginGatherWindow(window_read_enable_, num_neurons_);
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
                        if (gas_ctrl_) { gas_ctrl_->beginApplyFullSequence(apply_acc_enable_ && gas_window_mode_); }
                        else { orchestratePrepareApplyWindow(); orchestrateApplyWindowEntry(); orchestrateBeginApplyIssueFallback(apply_acc_enable_ && gas_window_mode_); }
#if SNNDL_DEBUG_ENABLED
                        if (window_read_debug_) {
                            const size_t posts_prev = weight_mem_subsystem_ ? weight_mem_subsystem_->postsPrevSize() : 0;
                            const size_t pres_prev  = weight_mem_subsystem_ ? weight_mem_subsystem_->presPrevSize() : 0;
                            const size_t posts_curr = weight_mem_subsystem_ ? weight_mem_subsystem_->postsCurrSize() : 0;
                            const size_t pres_curr  = weight_mem_subsystem_ ? weight_mem_subsystem_->presCurrSize() : 0;
                            SNNDL_DEBUG_LOG(1, "[diag-stage] BeginApply window=%" PRIu64 " stage=%d posts_prev=%zu active_prev=%zu posts_curr=%zu active_curr=%zu\n",
                                (uint64_t)op->superstep, static_cast<int>(gas_stage_),
                                posts_prev, pres_prev,
                                posts_curr, pres_curr);
                        }
#endif
                stage_event_hub_.markBeginApply(curr_stage_seq_);
                issueFallbackReadsIfNeeded_(apply_acc_enable_ && gas_window_mode_);
#if SNNDL_DEBUG_ENABLED
                if (window_read_debug_ && output_) {
                    output_->verbose(CALL_INFO, 1, 0,
                        "[diag-edges-summary] core=%u window=%u recorded=%" PRIu64 " stage_skip=%" PRIu64 " cond_skip=%" PRIu64 " edges_prev=%zu\n",
                        core_id_, curr_stage_seq_, diag_edges_record_hits_, diag_edges_stage_skips_, diag_edges_cond_skips_,
                        weight_mem_subsystem_ ? weight_mem_subsystem_->edgesPrevSize() : 0);
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
                        if (gas_ctrl_) { gas_ctrl_->beginScatterSequence(); }
                        else { orchestrateBeginScatterSequence(); }
                        // Scheme-C (debug aid): print per-window delta summary for diagnostics.
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
                        // BeginScatter 的累加应用/发放已由 orchestrateBeginScatterSequence()
                        // 与 applyAccumulatedWindowAndScatter_ 完成，避免重复执行。
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
                            if (gas_ctrl_) { gas_ctrl_->endScatterSequence(); }
                            else { orchestrateEndScatterSequence(); }
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

    // Phase E/Phase1: data-plane StandardMem responses should first be dispatched by StandardMemAccess.
    // Control layer keeps only CustomResp (GAS control-plane) processing above.
    if (weight_mem_subsystem_) {
        weight_mem_subsystem_->setNowCycle(static_cast<uint64_t>(total_cycles_));
    }
    if (stdmem_access_ && stdmem_access_->handleMemoryResponse(req)) {
        return;
    }

    // Phase1-A: fail-fast only for *data-plane* read responses. Other response types may be internal
    // (e.g., GatherBufferIF flush/write maintenance) and can be safely ignored here if untracked.
    if (dynamic_cast<SST::Interfaces::StandardMem::ReadResp*>(req)) {
        if (output_) {
            output_->fatal(CALL_INFO, -1,
                "[stdmem-untracked] node=%u core=%u type=ReadResp id=%" PRIu64 "\n",
                static_cast<uint32_t>(node_id_),
                static_cast<uint32_t>(core_id_),
                static_cast<uint64_t>(req->getID()));
        }
        delete req;
        return;
    }
    // Untracked non-ReadResp: drop.
    delete req;
    return;

#if 0
    output_->verbose(CALL_INFO, 4, 0, "📨 核心%d收到内存响应: ID=%" PRIu64 "\n",
                    core_id_, req->getID());
    MemRequestMeta pending_req;
    const bool found = mem_backend_ && mem_backend_->popPending(req->getID(), pending_req);
    if (found) {
        // 计算往返延迟
        uint64_t ic = pending_req.issue_cycle;
        if (total_cycles_ >= ic) {
            uint64_t lat = static_cast<uint64_t>(total_cycles_ - ic);
            accum_mem_latency_cycles_ += lat;
            count_mem_responses_++;
            if (parent_) {
                if (auto* pe = parent_pe_cached_) {
                    pe->accumulateMemReadLatency(lat, pending_req.is_weight);
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
    if (!found) { delete req; return; }
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
                    delete req;
                    return;
                }
                size_t n = bytes.size() / bcsr_idx_bytes_;
                std::vector<uint32_t> cols(n);
                if (bcsr_idx_bytes_ == 2) {
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
                        if (bcsrBlockGet_(pending_req.bcsr_block_row, pending_req.bcsr_target_block_col, blk)) {
                            uint32_t off = pending_req.bcsr_intra_row * bcsr_bc_ + pending_req.bcsr_intra_col;
                            float w = (off<blk.size()? blk[off] : 0.0f);
                            if (pending_req.has_single_cb && pending_req.single_cb) pending_req.single_cb(w);
                        } else {
                            size_t block_bytes = (size_t)bcsr_br_ * (size_t)bcsr_bc_ * bcsr_val_bytes_;
                            uint64_t addr = bcsr_blockdata_addr_ + (uint64_t)global_block_index * block_bytes;
                            PendingMemoryRequest pm{};
                            pm.address = addr; pm.size = block_bytes; pm.issue_cycle = total_cycles_;
                            pm.is_weight = true; // BCSR 块数据属于权重
                            pm.bcsr_kind = 3; pm.bcsr_block_row = pending_req.bcsr_block_row; pm.bcsr_target_block_col = pending_req.bcsr_target_block_col;
                            pm.bcsr_intra_row = pending_req.bcsr_intra_row; pm.bcsr_intra_col = pending_req.bcsr_intra_col;
                            pm.bcsr_row_start = start; pm.bcsr_idx_in_row = idx_in_row; pm.bcsr_global_block_index = global_block_index;
                            pm.has_single_cb = pending_req.has_single_cb; pm.single_cb = pending_req.single_cb;
                            stats_reporter_.reportMemoryIssue(block_bytes, true);
                            if (mem_backend_) mem_backend_->sendRead(addr, block_bytes, pm);
                            bcsr_count_block_misses_++;
                        }
                    }
                }
            } else if (pending_req.bcsr_kind == 3) {
                size_t n = (size_t)bcsr_br_ * (size_t)bcsr_bc_;
                std::vector<float> blk(n); // 默认0填充；若响应为空/不足，避免未定义行为
                const size_t expect_bytes = n * (size_t)bcsr_val_bytes_;
                bool used_mem_block = false;
                if (bcsr_val_bytes_ == 4 && bytes.size() >= expect_bytes) {
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
                    size_t block_bytes = (size_t)bcsr_br_ * (size_t)bcsr_bc_ * bcsr_val_bytes_;
                    // 以文件起始为基准的blockdata偏移 = (bcsr_blockdata_addr_ - base_addr_) + global_block_index * block_bytes
                    uint64_t file_off = (uint64_t)(bcsr_blockdata_addr_ - base_addr_) + (uint64_t)global_block_index * (uint64_t)block_bytes;
                    std::ifstream fin(path, std::ios::binary);
                    if (fin.good()) {
                        fin.seekg(0, std::ios::end);
                        std::streamsize fsz = fin.tellg();
                        if (fsz >= 0 && (uint64_t)fsz >= (file_off + block_bytes)) {
                            fin.seekg((std::streamoff)file_off, std::ios::beg);
                            std::vector<uint8_t> tmp(block_bytes);
                            fin.read(reinterpret_cast<char*>(tmp.data()), (std::streamsize)block_bytes);
                            if (fin.gcount() == (std::streamsize)block_bytes && bcsr_val_bytes_ == 4) {
                                std::memcpy(blk.data(), tmp.data(), block_bytes);
                                used_mem_block = true; // 实际来源为文件，但后续处理一致
                                if (window_read_debug_ && output_) {
                                    output_->verbose(CALL_INFO, 1, 0,
                                        "[diag-bcsr][file] loaded block row=%u idx_in_row=%u gb=%u off=0x%llx bytes=%zu\n",
                                        pending_req.bcsr_block_row, pending_req.bcsr_idx_in_row, global_block_index,
                                        (unsigned long long)file_off, (size_t)block_bytes);
                                }
                            }
                        }
                    }
                }
                bcsrBlockPut_(pending_req.bcsr_block_row, pending_req.bcsr_target_block_col, blk);
                bcsr_count_block_reads_++;
                bcsr_bytes_val_ += bytes.size();
                uint32_t off = pending_req.bcsr_intra_row * bcsr_bc_ + pending_req.bcsr_intra_col;
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
                    uint32_t post_local = pending_req.bcsr_block_row * bcsr_br_ + pending_req.bcsr_intra_row;
                    uint32_t post_global = global_neuron_base_ + post_local;
                    uint32_t pre_global_effective = pending_req.bcsr_target_block_col * bcsr_bc_ + pending_req.bcsr_intra_col;
                    output_->verbose(CALL_INFO, 1, 0,
                        "[diag-bcsr-weight] core=%u post_local=%u post_global=%u pre_global=%u weight=%.6f source=%s\n",
                        core_id_, post_local, post_global, pre_global_effective, w, used_mem_block ? "OK" : "FallbackFile");
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
        } else if (pending_req.bcsr_kind == 1) {
            ensureRowptrReadyOrFatal_("rowptr read returned empty payload");
            finalize_rowptr_ready();
        } else {
            if (pending_req.has_single_cb && pending_req.single_cb) pending_req.single_cb(0.0f);
        }
        const bool counted_request = (pending_req.bcsr_kind == 0) && pending_req.has_single_cb;
        if (counted_request) {
            windowStateNoteComplete_();
        }
        // Apply 窗口内的补发：在收到 ReadResp 后，若预算/并发未满则继续发起下一批
        if (!(apply_acc_enable_ && gas_window_mode_) && window_read_enable_ && gas_stage_ == GasStage::Apply && enable_weight_fetch_ && memory_ && memory_ready_) {
            // 允许使用当前窗作为补发来源（当 prev 为空但 curr 非空时）
            if (!weight_mem_subsystem_) {
                // 子系统缺失时跳过补发
                delete req;
                return;
            }
            bool have_prev_refill = (weight_mem_subsystem_->presPrevSize() > 0) && (weight_mem_subsystem_->postsPrevSize() > 0);
            bool have_curr_refill = (weight_mem_subsystem_->presCurrSize() > 0) && (weight_mem_subsystem_->postsCurrSize() > 0);
            (void)have_curr_refill;
            const auto& pres_src_refill  = have_prev_refill ? weight_mem_subsystem_->presPrev()
                                                            : weight_mem_subsystem_->presCurr();
            const auto& posts_src_refill = have_prev_refill ? weight_mem_subsystem_->postsPrev()
                                                            : weight_mem_subsystem_->postsCurr();
            if (windowStateCanIssue_() && !pres_src_refill.empty() && !posts_src_refill.empty()) {
                const uint32_t width_refill = use_post_row_pre_col_ ? weights_cols_ : num_neurons_;
                for (const auto& pre_g : pres_src_refill) {
                    for (uint32_t post_l : posts_src_refill) {
                        if (!windowStateCanIssue_()) break;
                        uint32_t arg0 = use_post_row_pre_col_ ? pre_g : post_l;
                        uint32_t arg1 = use_post_row_pre_col_ ? post_l : pre_g;
                        const uint64_t key = (uint64_t)post_l * (uint64_t)width_refill + (use_post_row_pre_col_ ? (uint64_t)pre_g : (uint64_t)post_l);
                        stats_reporter_.reportCacheAccess(false);
                        windowStateNoteIssue_();
                        requestWeight(arg0, arg1, [this, key](float w){
                            weightCacheStore_(key, w);
                            windowStateNoteComplete_();
                        });
                        if (!windowStateCanIssue_()) break;
                    }
                    if (!windowStateCanIssue_()) break;
                }
            }
        }
        if (apply_acc_enable_ && gas_window_mode_ && counted_request) {
            issueEdgeWeightFetches_();
    }

    delete req;
#endif
}

void SnnPESubComponent::scheme1PrefetchSlice_(uint32_t slice_idx) {
    if (!ensureMemoryReady_()) return;
    if (weights_cols_ == 0) return;
    if (!weight_mem_subsystem_) return;
    // 计算该 slice 的列区间 [beg, end)
    uint32_t width = weights_cols_;
    uint32_t seg = std::max<uint32_t>(1, (width + scheme1_slices_ - 1) / scheme1_slices_);
    uint32_t beg = std::min<uint32_t>(slice_idx * seg, width);
    uint32_t end = std::min<uint32_t>(beg + seg, width);
    if (beg >= end) return;
    const uint32_t fpl = std::max<uint32_t>(1, line_size_bytes_ / (uint32_t)sizeof(float));
    s1_is_issuing_prefetch_ = true;
    weight_mem_subsystem_->setNowCycle(static_cast<uint64_t>(total_cycles_));
    for (uint32_t row = 0; row < num_neurons_; ++row) {
        // 按 cacheline 对齐扫描该区间
        for (uint32_t c = (beg / fpl) * fpl; c < end; c += fpl) {
            uint64_t req_addr = base_addr_ + (static_cast<uint64_t>(row) * width + c) * sizeof(float);
            size_t req_size = std::min<uint32_t>(fpl, end - c) * (uint32_t)sizeof(float);
            if (stat_s1_bytes_read_) stat_s1_bytes_read_->addData(static_cast<uint64_t>(req_size));
            const uint32_t count_floats = static_cast<uint32_t>(req_size / sizeof(float));
            const bool issued = weight_mem_subsystem_->issueDensePrefetchRaw(
                req_addr, req_size, row, c, count_floats, /*scheme1_prefetch*/true);
            if (issued) scheme1_pending_prefetch_++;
        }
    }
    s1_is_issuing_prefetch_ = false;
}
