// -*- c++ -*-
//
// StandardMemWeightReader.cc: implementation.
//

#include <sst/core/sst_config.h>
#include "StandardMemWeightReader.h"
#include "SnnPESubComponent.h"
#include "MultiCorePE.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <inttypes.h>

using namespace SST;
using namespace SST::SnnDL;

void StandardMemWeightReader::requestDense(uint32_t pre_neuron, uint32_t post_neuron,
                                          std::function<void(float)> callback) {
    if (!core_) return;
    if (core_->use_bcsr_ && core_->use_post_row_pre_col_) {
        requestBCSR(pre_neuron, post_neuron, std::move(callback));
        return;
    }
    uint32_t req_pre = 0;
    uint32_t req_post = 0;
    uint64_t cache_key = 0;
    if (!core_->weight_accessor_.resolve(pre_neuron, post_neuron, req_pre, req_post, cache_key)) {
        if (callback) callback(core_->init_default_weight_);
        return;
    }

    if (!core_->ensureMemoryReady_()) {
        if (callback) callback(0.5f);
        return;
    }

    uint64_t req_addr = 0; size_t req_size = sizeof(float);
    bool is_row = false; uint32_t col_start = req_post; uint32_t count_floats = 1;
    prepareDenseRead_(req_pre, req_post,
                      core_->use_post_row_pre_col_ ? core_->weights_cols_ : core_->num_neurons_,
                      req_addr, req_size, is_row, col_start, count_floats);
    if (core_->window_read_debug_) {
        core_->output_->verbose(CALL_INFO, 2, 0, "[diag-read] core=%d requestWeight row=%u col=%u is_row=%d col_start=%u count=%u addr=0x%llx size=%zu\n",
                         core_->core_id_, req_pre, req_post, (int)is_row, col_start, count_floats,
                         (unsigned long long)req_addr, req_size);
    }
    issueReadCommon_(req_addr, req_size, is_row, req_pre, col_start, count_floats, callback, cache_key);
}

void StandardMemWeightReader::requestBCSR(uint32_t pre_global, uint32_t post_local,
                                         std::function<void(float)> cb) {
    if (!core_) return;
    core_->requestWeightBCSR(pre_global, post_local, std::move(cb));
}

bool StandardMemWeightReader::tryCache(uint64_t key, float& out) {
    if (!core_) return false;
    return core_->weightCacheTryGet_(key, out);
}

void StandardMemWeightReader::putCache(uint64_t key, float value) {
    if (!core_) return;
    core_->weightCacheStore_(key, value);
}

bool StandardMemWeightReader::applyLocalWeightUpdates(const std::unordered_map<uint64_t, float>& grads,
                                                      float learning_rate,
                                                      float weight_decay) {
    if (!core_) return false;
    if (grads.empty()) return true;
    if (!core_->memory_ || !core_->memory_ready_) return false;
    const size_t bytes_per_float = sizeof(float);
    uint64_t total_writes = 0;
    size_t skipped_uncached = 0;
    for (const auto& kv : grads) {
        uint64_t key = kv.first;
        float grad = kv.second;
        float old_w = 0.0f;
        if (!core_->weightCacheTryGet_(key, old_w)) {
            skipped_uncached++;
            continue;
        }
        float new_w = old_w - learning_rate * grad;
        if (weight_decay != 0.0f) {
            new_w -= weight_decay * old_w;
        }
        uint64_t addr = core_->base_addr_ + key * bytes_per_float;
        std::vector<uint8_t> data(bytes_per_float);
        std::memcpy(data.data(), &new_w, bytes_per_float);
        auto* w = new SST::Interfaces::StandardMem::Write(addr, data.size(), data, false);
        core_->stats_reporter_.reportMemoryIssue(data.size(), false);
        core_->memory_->send(w);
        total_writes++;
        core_->weightCacheStore_(key, new_w);
    }
    if (core_->output_) {
        core_->output_->verbose(CALL_INFO, 1, 0,
            "📝 学习: 写回完成 writes=%" PRIu64 ", 跳过(未缓存)=%zu\n",
            total_writes, skipped_uncached);
    }
    return true;
}

void StandardMemWeightReader::scheme1PrefetchSlice(uint32_t slice_idx) {
    if (!core_) return;
    if (!core_->ensureMemoryReady_()) return;
    if (core_->weights_cols_ == 0) return;
    uint32_t width = core_->weights_cols_;
    uint32_t seg = std::max<uint32_t>(1, (width + core_->scheme1_slices_ - 1) / core_->scheme1_slices_);
    uint32_t beg = std::min<uint32_t>(slice_idx * seg, width);
    uint32_t end = std::min<uint32_t>(beg + seg, width);
    if (beg >= end) return;
    const uint32_t fpl = std::max<uint32_t>(1, core_->line_size_bytes_ / (uint32_t)sizeof(float));
    core_->s1_is_issuing_prefetch_ = true;
    for (uint32_t row = 0; row < core_->num_neurons_; ++row) {
        for (uint32_t c = (beg / fpl) * fpl; c < end; c += fpl) {
            uint64_t req_addr = core_->base_addr_ + (static_cast<uint64_t>(row) * width + c) * sizeof(float);
            size_t req_size = std::min<uint32_t>(fpl, end - c) * (uint32_t)sizeof(float);
            if (core_->stat_s1_bytes_read_) core_->stat_s1_bytes_read_->addData(static_cast<uint64_t>(req_size));
            auto* rd = new SST::Interfaces::StandardMem::Read(req_addr, req_size);
            auto id = rd->getID();
            SnnPESubComponent::PendingMemoryRequest pm;
            pm.request_id = id; pm.address = req_addr; pm.size = req_size; pm.is_row = false;
            pm.pre = row; pm.post_start = c; pm.count_floats = (uint32_t)(req_size / sizeof(float));
            pm.has_single_cb = false; pm.cb_post = 0; pm.issue_cycle = core_->total_cycles_;
            pm.is_weight = (req_addr >= core_->base_addr_ && req_addr < core_->weight_region_end_);
            pm.scheme1_prefetch = true;
            core_->stats_reporter_.reportMemoryIssue(req_size, true);
            core_->pending_memory_requests_[id] = pm;
            core_->memory_->send(rd);
            core_->scheme1_pending_prefetch_++;
        }
    }
    core_->s1_is_issuing_prefetch_ = false;
}

bool StandardMemWeightReader::prepareDenseRead_(uint32_t row, uint32_t col, uint32_t width,
                                               uint64_t& req_addr, size_t& req_size,
                                               bool& is_row, uint32_t& col_start, uint32_t& count_floats) const {
    if (!core_) return false;
    const uint32_t bpf = sizeof(float);
    req_addr = core_->base_addr_ + (static_cast<uint64_t>(row) * width + col) * bpf;
    req_size = sizeof(float);
    is_row = false;
    col_start = col;
    count_floats = 1;
    if (core_->read_force_single_) {
        return true;
    }
    if (core_->merge_read_auto_) {
        uint32_t fpl = std::max<uint32_t>(1, core_->line_size_bytes_ / bpf);
        size_t bytes_row = static_cast<size_t>(width) * bpf;
        size_t bytes_cl  = static_cast<size_t>(fpl) * bpf;
        bool choose_row = core_->merge_read_row_ && (bytes_row <= bytes_cl);
        if (choose_row && core_->merge_read_row_) {
            is_row = true;
            col_start = 0;
            count_floats = width;
            req_addr = core_->base_addr_ + static_cast<uint64_t>(row) * width * bpf;
            req_size = static_cast<size_t>(count_floats) * bpf;
        } else if (core_->merge_read_cacheline_) {
            col_start = (col / fpl) * fpl;
            count_floats = std::min<uint32_t>(fpl, width - col_start);
            req_addr = core_->base_addr_ + (static_cast<uint64_t>(row) * width + col_start) * bpf;
            req_size = static_cast<size_t>(count_floats) * bpf;
        }
    } else if (core_->merge_read_row_) {
        is_row = true;
        col_start = 0;
        count_floats = width;
        req_addr = core_->base_addr_ + static_cast<uint64_t>(row) * width * bpf;
        req_size = static_cast<size_t>(count_floats) * bpf;
    } else if (core_->merge_read_cacheline_) {
        uint32_t fpl = std::max<uint32_t>(1, core_->line_size_bytes_ / bpf);
        col_start = (col / fpl) * fpl;
        count_floats = std::min<uint32_t>(fpl, width - col_start);
        req_addr = core_->base_addr_ + (static_cast<uint64_t>(row) * width + col_start) * bpf;
        req_size = static_cast<size_t>(count_floats) * bpf;
    }
    return true;
}

void StandardMemWeightReader::issueReadCommon_(uint64_t req_addr, size_t req_size,
                                              bool is_row, uint32_t row, uint32_t col_start, uint32_t count_floats,
                                              std::function<void(float)> single_cb, uint32_t single_col) {
    if (!core_) return;
    auto* read = new SST::Interfaces::StandardMem::Read(req_addr, req_size);
    uint64_t reqId = read->getID();
    SnnPESubComponent::PendingMemoryRequest pmr;
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
    pmr.issue_cycle = core_->total_cycles_;
    pmr.is_weight = (req_addr >= core_->base_addr_ && req_addr < core_->weight_region_end_);
    core_->stats_reporter_.reportMemoryIssue(req_size, true);
    core_->pending_memory_requests_[reqId] = pmr;
    if (core_->window_read_debug_) {
        core_->output_->verbose(CALL_INFO, 1, 0,
            "[diag-issue] core=%d send Read id=%" PRIu64 " addr=0x%llx size=%zu is_row=%d col_start=%u count=%u outstanding=%zu\n",
            core_->core_id_, reqId, (unsigned long long)req_addr, req_size, (int)is_row,
            col_start, count_floats, (size_t)core_->pending_memory_requests_.size());
    }
    core_->memory_->send(read);
}

bool StandardMemWeightReader::handleMemoryResponse(SST::Interfaces::StandardMem::Request* req) {
    if (!core_ || !req) return false;

    auto it = core_->pending_memory_requests_.find(req->getID());
    if (it != core_->pending_memory_requests_.end()) {
        uint64_t ic = it->second.issue_cycle;
        if (core_->total_cycles_ >= ic) {
            uint64_t lat = static_cast<uint64_t>(core_->total_cycles_ - ic);
            core_->accum_mem_latency_cycles_ += lat;
            core_->count_mem_responses_++;
            if (core_->parent_) {
                if (auto* pe = core_->parent_pe_cached_) {
                    pe->accumulateMemReadLatency(lat, it->second.is_weight);
                }
            }
        }
    }

    auto* readResp = dynamic_cast<SST::Interfaces::StandardMem::ReadResp*>(req);
    if (!readResp) {
        delete req;
        return true;
    }
    if (it == core_->pending_memory_requests_.end()) {
        delete req;
        return true;
    }

    SnnPESubComponent::PendingMemoryRequest pending_req = it->second;
    core_->pending_memory_requests_.erase(it);
    if (pending_req.scheme1_prefetch && core_->scheme1_enable_) {
        if (core_->scheme1_pending_prefetch_ > 0) core_->scheme1_pending_prefetch_--;
    }

    auto finalize_rowptr_ready = [&]() {
        if (core_->window_read_debug_ && core_->output_) {
            const auto& rp = core_->bcsr_weights_.rowptrHost();
            core_->output_->verbose(CALL_INFO, 0, 0,
                "[diag-bcsr] core=%u rowptr ready entries=%zu first=%u second=%u\n",
                core_->core_id_, rp.size(),
                rp.empty() ? 0u : rp[0],
                rp.size() > 1 ? rp[1] : 0u);
        }
        core_->bcsrPrefetchAll_();
        if (core_->apply_acc_enable_ && core_->gas_window_mode_ && core_->gas_stage_ == SnnPESubComponent::GasStage::Apply) {
            core_->issueEdgeWeightFetches_();
        }
    };

    if (core_->window_read_debug_) {
        const std::vector<uint8_t>& bytes = readResp->data;
        const char* kind_label = "-";
        switch (pending_req.bcsr_kind) {
            case 1: kind_label = "rowptr"; break;
            case 2: kind_label = "colidx"; break;
            case 3: kind_label = "block"; break;
            default: break;
        }
        core_->output_->verbose(CALL_INFO, 1, 0,
            "[diag-resp] core=%d kind=%s id=%" PRIu64 " addr=0x%llx bytes=%zu\n",
            core_->core_id_, kind_label, req->getID(), (unsigned long long)pending_req.address,
            bytes.size());
    }
    if (pending_req.bcsr_kind == 1) {
        core_->bcsr_weights_.setRowptrReadPending(false);
    }

    if (!readResp->data.empty()) {
        const std::vector<uint8_t>& bytes = readResp->data;
        size_t float_count = bytes.size() / sizeof(float);
        const float* fptr = reinterpret_cast<const float*>(bytes.data());

        if (pending_req.bcsr_kind == 1) {
            if (!core_->installRowptrFromBytes_(bytes.data(), bytes.size(), "dram", true)) {
                core_->ensureRowptrReadyOrFatal_("rowptr load failed (dram)");
            }
            finalize_rowptr_ready();
        } else if (pending_req.bcsr_kind == 2) {
            const size_t expect = pending_req.size;
            if (bytes.size() != expect) {
                core_->output_->verbose(CALL_INFO, 0, 0,
                    "[diag-bcsr][warn] colidx read size mismatch: got=%zu expect=%zu addr=0x%llx\n",
                    bytes.size(), expect, (unsigned long long)pending_req.address);
                if (pending_req.has_single_cb && pending_req.single_cb) pending_req.single_cb(0.0f);
                delete req;
                return true;
            }
            size_t n = bytes.size() / core_->bcsr_idx_bytes_;
            std::vector<uint32_t> cols(n);
            if (core_->bcsr_idx_bytes_ == 2) {
                for (size_t i=0;i<n;i++) cols[i] = ((const uint16_t*)bytes.data())[i];
            } else {
                for (size_t i=0;i<n;i++) cols[i] = ((const uint32_t*)bytes.data())[i];
            }
            core_->bcsrRowIndexPut_(pending_req.bcsr_block_row, cols);
            core_->bcsr_count_row_index_fills_++;
            core_->bcsr_count_colidx_reads_++;
            core_->bcsr_bytes_idx_ += bytes.size();
            if (pending_req.bcsr_prefetch_all) {
                core_->bcsrPrefetchRowBlocks_(pending_req.bcsr_block_row, cols, pending_req.bcsr_row_start);
            } else {
                uint32_t idx_in_row = 0; bool found=false;
                for (size_t i=0;i<cols.size();++i){ if (cols[i]==pending_req.bcsr_target_block_col){ idx_in_row=(uint32_t)i; found=true; break; } }
                if (!found) {
                    if (pending_req.has_single_cb && pending_req.single_cb) pending_req.single_cb(0.0f);
                } else {
                    uint32_t start = pending_req.bcsr_row_start;
                    uint32_t global_block_index = start + idx_in_row;
                    std::vector<float> blk;
                    if (core_->bcsrBlockGet_(pending_req.bcsr_block_row, pending_req.bcsr_target_block_col, blk)) {
                        uint32_t off = pending_req.bcsr_intra_row * core_->bcsr_bc_ + pending_req.bcsr_intra_col;
                        float w = (off<blk.size()? blk[off] : 0.0f);
                        if (pending_req.has_single_cb && pending_req.single_cb) pending_req.single_cb(w);
                    } else {
                        size_t block_bytes = (size_t)core_->bcsr_br_ * (size_t)core_->bcsr_bc_ * core_->bcsr_val_bytes_;
                        uint64_t addr = core_->bcsr_blockdata_addr_ + (uint64_t)global_block_index * block_bytes;
                        auto* rd = new SST::Interfaces::StandardMem::Read(addr, block_bytes);
                        auto id = rd->getID();
                        SnnPESubComponent::PendingMemoryRequest pm;
                        pm.request_id = id; pm.address = addr; pm.size = block_bytes; pm.issue_cycle = core_->total_cycles_;
                        pm.is_weight = true;
                        pm.bcsr_kind = 3; pm.bcsr_block_row = pending_req.bcsr_block_row; pm.bcsr_target_block_col = pending_req.bcsr_target_block_col;
                        pm.bcsr_intra_row = pending_req.bcsr_intra_row; pm.bcsr_intra_col = pending_req.bcsr_intra_col;
                        pm.bcsr_row_start = start; pm.bcsr_idx_in_row = idx_in_row; pm.bcsr_global_block_index = global_block_index;
                        pm.has_single_cb = pending_req.has_single_cb; pm.single_cb = pending_req.single_cb;
                        core_->stats_reporter_.reportMemoryIssue(block_bytes, true);
                        core_->pending_memory_requests_[id] = pm;
                        core_->memory_->send(rd);
                        core_->bcsr_count_block_misses_++;
                    }
                }
            }
        } else if (pending_req.bcsr_kind == 3) {
            size_t n = (size_t)core_->bcsr_br_ * (size_t)core_->bcsr_bc_;
            std::vector<float> blk(n);
            const size_t expect_bytes = n * (size_t)core_->bcsr_val_bytes_;
            bool used_mem_block = false;
            if (core_->bcsr_val_bytes_ == 4 && bytes.size() >= expect_bytes) {
                std::memcpy(blk.data(), bytes.data(), expect_bytes);
                used_mem_block = true;
            } else {
                if (core_->window_read_debug_ && core_->output_) {
                    core_->output_->verbose(CALL_INFO, 0, 0,
                        "[diag-bcsr][warn] block read size mismatch: got=%zu expect=%zu addr=0x%llx row=%u col=%u (fallback=file)\n",
                        bytes.size(), expect_bytes, (unsigned long long)pending_req.address,
                        pending_req.bcsr_block_row, pending_req.bcsr_target_block_col);
                }
                const uint32_t global_block_index = pending_req.bcsr_row_start + pending_req.bcsr_idx_in_row;
                std::string path = core_->weights_template_;
                size_t p = path.find("{pe:02d}");
                if (p != std::string::npos) {
                    char buf[16]; std::snprintf(buf, sizeof(buf), "%02u", core_->node_id_);
                    path.replace(p, 8, buf);
                } else {
                    p = path.find("{pe}");
                    if (p != std::string::npos) path.replace(p, 4, std::to_string(core_->node_id_));
                }
                p = path.find("{core:02d}");
                if (p != std::string::npos) {
                    char buf2[16]; std::snprintf(buf2, sizeof(buf2), "%02u", core_->core_id_);
                    path.replace(p, 10, buf2);
                } else {
                    p = path.find("{core}");
                    if (p != std::string::npos) path.replace(p, 6, std::to_string(core_->core_id_));
                }
                size_t block_bytes = (size_t)core_->bcsr_br_ * (size_t)core_->bcsr_bc_ * core_->bcsr_val_bytes_;
                uint64_t file_off = (uint64_t)(core_->bcsr_blockdata_addr_ - core_->base_addr_) +
                                    (uint64_t)global_block_index * (uint64_t)block_bytes;
                std::ifstream fin(path, std::ios::binary);
                if (fin.good()) {
                    fin.seekg(0, std::ios::end);
                    std::streamsize fsz = fin.tellg();
                    if (fsz >= 0 && (uint64_t)fsz >= (file_off + block_bytes)) {
                        fin.seekg((std::streamoff)file_off, std::ios::beg);
                        std::vector<uint8_t> tmp(block_bytes);
                        fin.read(reinterpret_cast<char*>(tmp.data()), (std::streamsize)block_bytes);
                        if (fin.gcount() == (std::streamsize)block_bytes && core_->bcsr_val_bytes_ == 4) {
                            std::memcpy(blk.data(), tmp.data(), block_bytes);
                            used_mem_block = true;
                            if (core_->window_read_debug_ && core_->output_) {
                                core_->output_->verbose(CALL_INFO, 1, 0,
                                    "[diag-bcsr][file] loaded block row=%u idx_in_row=%u gb=%u off=0x%llx bytes=%zu\n",
                                    pending_req.bcsr_block_row, pending_req.bcsr_idx_in_row, global_block_index,
                                    (unsigned long long)file_off, (size_t)block_bytes);
                            }
                        }
                    }
                }
            }
            core_->bcsrBlockPut_(pending_req.bcsr_block_row, pending_req.bcsr_target_block_col, blk);
            core_->bcsr_count_block_reads_++;
            core_->bcsr_bytes_val_ += bytes.size();
            uint32_t off = pending_req.bcsr_intra_row * core_->bcsr_bc_ + pending_req.bcsr_intra_col;
            float w = (off < blk.size() ? blk[off] : 0.0f);
            if (core_->bcsr_weight_guard_enable_) {
                if (!std::isfinite(w) || std::fabs(w) > core_->bcsr_weight_abs_max_) {
                    if (core_->window_read_debug_ && core_->output_) {
                        core_->output_->verbose(CALL_INFO, 0, 0,
                            "[diag-bcsr][guard] core=%u row=%u col=%u raw=%.6e -> clamped(0)\n",
                            core_->core_id_, pending_req.bcsr_block_row,
                            pending_req.bcsr_target_block_col, (double)w);
                    }
                    w = 0.0f;
                    core_->bcsr_bad_weight_count_++;
                }
            }
            if (core_->readresp_zero_fallback_ && w == 0.0f) {
                w = core_->init_default_weight_;
            }
            core_->bcsr_req_block_hit_++;
            if (core_->window_read_debug_ && core_->output_) {
                uint32_t post_local = pending_req.bcsr_block_row * core_->bcsr_br_ + pending_req.bcsr_intra_row;
                uint32_t post_global = core_->global_neuron_base_ + post_local;
                uint32_t pre_global_effective = pending_req.bcsr_target_block_col * core_->bcsr_bc_ + pending_req.bcsr_intra_col;
                core_->output_->verbose(CALL_INFO, 1, 0,
                    "[diag-bcsr-weight] core=%u post_local=%u post_global=%u pre_global=%u weight=%.6f source=%s\n",
                    core_->core_id_, post_local, post_global, pre_global_effective, w, used_mem_block ? "OK" : "FallbackFile");
            }
            if (core_->apply_acc_enable_ && core_->gas_window_mode_) {
                uint32_t post_local = pending_req.bcsr_block_row * core_->bcsr_br_ + pending_req.bcsr_intra_row;
                core_->accUpdate_(post_local, w);
            }
            if (pending_req.has_single_cb && pending_req.single_cb) pending_req.single_cb(w);
        } else {
            uint32_t width = core_->use_post_row_pre_col_ ? core_->weights_cols_ : core_->num_neurons_;
            if (pending_req.is_row) {
                size_t expect_bytes = static_cast<size_t>(width) * sizeof(float);
                if (pending_req.size != expect_bytes) {
                    core_->output_->verbose(CALL_INFO, 1, 0,
                        "[GAS][Warn] row-merge size mismatch: got=%zu expect=%zu (row=%u)\n",
                        (size_t)pending_req.size, expect_bytes, pending_req.pre);
                }
            }
            if (core_->verify_weights_ && core_->output_ && core_->verbose_ >= 1) {
                float wprobe = (float_count > 0 ? fptr[0] : 0.0f);
                uint32_t probe_row = pending_req.pre;
                uint32_t probe_col = pending_req.post_start;
                core_->output_->verbose(CALL_INFO, 1, 0,
                    "[VERIFY][probe-any] row=%u col=%u value=%.6f size=%zu floats=%zu is_row=%d\n",
                    probe_row, probe_col, wprobe, (size_t)pending_req.size, (size_t)float_count, pending_req.is_row ? 1 : 0);
            }
            if (core_->apply_acc_enable_ && core_->use_post_row_pre_col_) {
                uint32_t target_col = pending_req.cb_post;
                if (target_col >= pending_req.post_start && target_col < pending_req.post_start + float_count && target_col < width) {
                    size_t idx = static_cast<size_t>(target_col - pending_req.post_start);
                    float w = fptr[idx];
                    if (core_->bcsr_weight_guard_enable_) {
                        if (!std::isfinite(w) || std::fabs(w) > core_->bcsr_weight_abs_max_) {
                            if (core_->window_read_debug_ && core_->output_) {
                                core_->output_->verbose(CALL_INFO, 0, 0,
                                    "[diag-bcsr][guard] core=%u row(post)=%u col(pre)=%u raw=%.6e -> clamped(0)\n",
                                    core_->core_id_, pending_req.pre, target_col, (double)w);
                            }
                            w = 0.0f;
                            core_->bcsr_bad_weight_count_++;
                        }
                    }
                    if (core_->readresp_zero_fallback_ && w == 0.0f) w = core_->init_default_weight_;
                    uint64_t key = static_cast<uint64_t>(pending_req.pre) * static_cast<uint64_t>(width) + target_col;
                    core_->weightCacheStore_(key, w);
                    uint32_t post_local = pending_req.pre;
                    if (core_->output_) {
                        core_->output_->verbose(CALL_INFO, 2, 0,
                            "[GAS][Delta][readresp] core=%u seq=%u post(row)=%u pre(col)=%u dv=%.6f (single col) addr=0x%lx start=%u idx=%zu size=%zu\n",
                            core_->core_id_, core_->curr_stage_seq_, post_local, target_col, w,
                            (unsigned long)pending_req.address, pending_req.post_start,
                            idx, (size_t)pending_req.size);
                        if (core_->verify_weights_) {
                            core_->output_->verbose(CALL_INFO, 1, 0,
                                "[VERIFY][probe] row=%u col=%u value=%.6f\n",
                                post_local, target_col, w);
                        }
                    }
                    core_->accUpdate_(post_local, w);
                }
            } else {
                for (size_t i = 0; i < float_count; ++i) {
                    uint32_t col_idx = pending_req.post_start + static_cast<uint32_t>(i);
                    if (col_idx >= width) break;
                    uint64_t key = static_cast<uint64_t>(pending_req.pre) * static_cast<uint64_t>(width) + col_idx;
                    float w = fptr[i];
                    if (core_->readresp_zero_fallback_ && w == 0.0f) w = core_->init_default_weight_;
                    core_->weightCacheStore_(key, w);
                    if (pending_req.has_single_cb && pending_req.single_cb && !core_->use_post_row_pre_col_) {
                        pending_req.single_cb(w);
                    }
                }
            }
        }
    } else if (pending_req.bcsr_kind == 1) {
        core_->ensureRowptrReadyOrFatal_("rowptr read returned empty payload");
        finalize_rowptr_ready();
    } else {
        if (pending_req.has_single_cb && pending_req.single_cb) pending_req.single_cb(0.0f);
    }

    if (core_->outstanding_requests_ > 0) core_->outstanding_requests_--;
    if (!(core_->apply_acc_enable_ && core_->gas_window_mode_) &&
        core_->window_read_enable_ && core_->gas_stage_ == SnnPESubComponent::GasStage::Apply &&
        core_->enable_weight_fetch_ && core_->memory_ && core_->memory_ready_) {
        bool have_prev_refill = (!core_->active_pre_prev_window_.empty() && !core_->posts_list_prev_window_.empty());
        bool have_curr_refill = (!core_->active_pre_window_.empty()      && !core_->posts_list_window_.empty());
        const auto& pres_src_refill  = (have_prev_refill ? core_->active_pre_prev_window_ : core_->active_pre_window_);
        const auto& posts_src_refill = (have_prev_refill ? core_->posts_list_prev_window_ : core_->posts_list_window_);
        if (core_->window_reads_issued_this_apply_ < core_->window_read_budget_ &&
            !pres_src_refill.empty() && !posts_src_refill.empty()) {
            const uint32_t width_refill = core_->use_post_row_pre_col_ ? core_->weights_cols_ : core_->num_neurons_;
            for (const auto& pre_g : pres_src_refill) {
                for (uint32_t post_l : posts_src_refill) {
                    if (core_->window_reads_issued_this_apply_ >= core_->window_read_budget_) break;
                    if (core_->outstanding_requests_ >= core_->max_outstanding_requests_) break;
                    uint32_t arg0 = core_->use_post_row_pre_col_ ? pre_g : post_l;
                    uint32_t arg1 = core_->use_post_row_pre_col_ ? post_l : pre_g;
                    const uint64_t key = (uint64_t)post_l * (uint64_t)width_refill +
                        (core_->use_post_row_pre_col_ ? (uint64_t)pre_g : (uint64_t)post_l);
                    core_->stats_reporter_.reportCacheAccess(false);
                    core_->outstanding_requests_++;
                    core_->stats_reporter_.updatePendingPeak(core_->outstanding_requests_);
                    requestDense(arg0, arg1, [this, key](float w){
                        if (core_) core_->weightCacheStore_(key, w);
                        if (core_ && core_->outstanding_requests_ > 0) core_->outstanding_requests_--;
                    });
                    core_->window_reads_issued_this_apply_++;
                    if (core_->outstanding_requests_ >= core_->max_outstanding_requests_) break;
                }
                if (core_->window_reads_issued_this_apply_ >= core_->window_read_budget_ ||
                    core_->outstanding_requests_ >= core_->max_outstanding_requests_) break;
            }
        }
    }
    if (core_->apply_acc_enable_ && core_->gas_window_mode_) {
        core_->issueEdgeWeightFetches_();
    }

    delete req;
    return true;
}

