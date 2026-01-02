// -*- c++ -*-
//
// SnnPESubComponent_mem.cc: StandardMem glue + dense read/write helpers for SnnPESubComponent.
//

#include <sst/core/sst_config.h>
#include <sst/core/componentInfo.h>
#include <sst/core/interfaces/stdMem.h>
#include "SnnPESubComponent.h"
#include "SnnPESubComponent_impl.h"
#include "IPeAggregation.h"
#include "IManualWindowDrive.h"
#include "synapse/stdmem/StdMemEndpoint.h"
#include "synapse/weights/SnnBcsrWeightManager.h"
#include "synapse/weights/WeightMemorySubsystem.h"
#include "memory/StandardMemAccess.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <inttypes.h>

using namespace SST;
using namespace SST::SnnDL;

template <>
void SnnPESubComponent::handleMemoryResponse<SST::Interfaces::StandardMem::Request>(
    SST::Interfaces::StandardMem::Request* req);

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

void SnnPESubComponent::initStdMemPhase0_() {
    // 加载 StandardMem 接口（Python 可通过槽位提供）。
    // 注意：该逻辑放在 synapse/stdmem 域，避免 control/*.cc 出现 StandardMem::。
    auto* stdmem = loadUserSubComponent<SST::Interfaces::StandardMem>(
        "memory", ComponentInfo::SHARE_NONE,
        registerTimeBase("1ns"),
        new SST::Interfaces::StandardMem::Handler2<SnnPESubComponent,
                                                   &SnnPESubComponent::handleMemoryResponse<SST::Interfaces::StandardMem::Request>>(this));

    // bind endpoint for control-plane sends (Begin/EndGather etc.)
    // Phase6：stream workload 需要“纯内存语义”，强制设置 non-cacheable（同时保留 env fallback 兼容路径）。
    {
        StdMemEndpoint::Config cfg{};
        cfg.force_noncacheable = isStreamWorkload_();
        stdmem_ep_->configure(cfg);
    }
    StdMemEndpoint::Runtime rt{};
    rt.log = output_;
    rt.node_id = static_cast<uint32_t>(node_id_);
    rt.core_id = static_cast<uint32_t>(core_id_);
    // Phase4-Task6.4: GAS stage/stat events are dispatched via StdMemEndpoint to CoreShell (IGasStageSink).
    rt.gas_stage_sink = this;
    rt.now_cycle = [this]() { return static_cast<uint64_t>(total_cycles_); };
    rt.before_data_plane_dispatch = [this](uint64_t now_cycle) {
        if (weight_mem_subsystem_) weight_mem_subsystem_->setNowCycle(now_cycle);
    };
    stdmem_ep_->bindRuntime(rt);
    stdmem_ep_->bindStdMem(stdmem);

    if (stdmem_ep_ && stdmem_ep_->available()) {
        // 若已成功加载 StandardMem 子组件，则认为内存就绪（即使未显式提供 memory_link_）
        memory_ready_ = true;
        if (gas_manual_window_drive_) {
            auto* drive = stdmem_ep_->manualWindowDrive();
            if (!drive) {
                if (output_) {
                    output_->verbose(CALL_INFO, 0, 0,
                        "⚠️ 核心%d启用gas_manual_window_drive但memory不支持IManualWindowDrive，降级为自动窗口\n",
                        core_id_);
                }
                gas_manual_window_drive_ = false;
            } else {
                if (output_) {
                    output_->verbose(CALL_INFO, 0, 0,
                        "[diag-gas] 核心%d启用manual窗口驱动 (IManualWindowDrive) gather_cycles=%" PRIu64 "\n",
                        core_id_, manual_gas_gather_cycles_cfg_);
                }
            }
        }
    } else {
        memory_ready_ = false;
    }
}

// === Learning writeback (called by compute core) ===
bool SnnPESubComponent::applyLocalWeightUpdates_(const std::unordered_map<uint64_t, float>& grads,
                                                float learning_rate,
                                                float weight_decay) {
    if (grads.empty()) return true;
    if (!ensureMemoryReady_()) return false;
    auto* mem = stdmem_ep_ ? stdmem_ep_->memoryAccess() : nullptr;
    if (!mem) return false;
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
	        if (impl_) impl_->reportMemoryIssue(data.size(), false);
	        mem->write(addr, data, nullptr);
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
    auto* mem = stdmem_ep_ ? stdmem_ep_->memoryAccess() : nullptr;
    if (!ensureMemoryReady_() || !mem) {
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

	    if (impl_) impl_->reportMemoryIssue(sizeof(float), /*count_weight_read*/true);
	    mem->read(req_addr, sizeof(float),
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

template <>
void SnnPESubComponent::handleMemoryResponse<SST::Interfaces::StandardMem::Request>(SST::Interfaces::StandardMem::Request* req) {
#ifdef SNNDL_ENABLE_PROFILING
    if (profiler_enabled_) { SNNDL_PROFILE_FUNCTION(profiler_); }
#endif
    // Phase4-Task6.4: StdMemEndpoint 统一分发：
    // - GAS 控制面（CustomResp）→ IGasStageSink（CoreShell），再转发到 workload=snn
    // - 数据面（ReadResp/WriteResp）→ StandardMemAccess（纯内存语义）
    if (stdmem_ep_ && stdmem_ep_->available()) {
        stdmem_ep_->handleResponseOpaque(req);
        return;
    }
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
            const auto& rp = bcsr_weights_->rowptrHost();
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
        bcsr_weights_->setRowptrReadPending(false);
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
	                            if (impl_) impl_->reportMemoryIssue(block_bytes, true);
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
        if (!(apply_acc_enable_ && gas_window_mode_) && window_read_enable_ && gas_stage_ == GasStage::Apply &&
            enable_weight_fetch_ && ensureMemoryReady_()) {
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
	                        if (impl_) impl_->reportCacheAccess(false);
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
