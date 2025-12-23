// -*- c++ -*-
//
// SnnPESubComponent_bcsr.cc: BCSR-related helpers extracted from the monolithic
// SnnPESubComponent.cc. Behavior preserved; only file-level split.
//

#include <sst/core/sst_config.h>
#include "SnnPESubComponent.h"

#include <fstream>
#include <sstream>
#include <cmath>
#include <cctype>
#include <iterator>
#include <algorithm>
#include <cstdlib>
#include "StandardMemBackend.h"

using namespace SST;
using namespace SST::SnnDL;
using PendingMemoryRequest = MemRequestMeta;

// Lightweight logging helpers (file-local). Keep consistent with SnnPESubComponent.cc.
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
} // namespace

bool SnnPESubComponent::BcsrLayout::validate(uint64_t base, Output* out, bool debug, uint32_t core_id, uint32_t node_id) const {
    const bool monotonic = (colidx_offset >= rowptr_offset) && (blockdata_offset >= colidx_offset);
    const bool aligned64 = ((rowptr_offset | colidx_offset | blockdata_offset | blockids_offset) & 0x3FULL) == 0;
    const uint64_t stride = per_core_stride;
    const uint64_t max_off = maxOffset();
    const bool stride_ok = (stride == 0) ? true : (max_off < stride);
    if (debug && out) {
        out->verbose(CALL_INFO, 0, 0,
            "[diag-bcsr-base] node=%u core=%u base=0x%lx rp=0x%lx ci=0x%lx bd=0x%lx ids=0x%lx stride=%" PRIu64 " stride_ok=%d align64=%d mono=%d br=%u bc=%u idx=%u val=%u\n",
            node_id, core_id,
            (unsigned long)base,
            (unsigned long)(base + rowptr_offset),
            (unsigned long)(base + colidx_offset),
            (unsigned long)(base + blockdata_offset),
            (unsigned long)(blockids_offset ? base + blockids_offset : 0),
            stride, stride_ok ? 1 : 0,
            aligned64 ? 1 : 0,
            monotonic ? 1 : 0,
            block_rows ? block_rows : 0,
            block_cols ? block_cols : 0,
            idx_bytes ? idx_bytes : 0,
            val_bytes ? val_bytes : 0);
        if (!aligned64 || !monotonic || !stride_ok) {
            out->verbose(CALL_INFO, 0, 0,
                "[diag-bcsr-base] offsets suspect (aligned64=%d monotonic=%d stride_ok=%d max_off=0x%llx stride=0x%llx)\n",
                aligned64 ? 1 : 0, monotonic ? 1 : 0, stride_ok ? 1 : 0,
                (unsigned long long)max_off, (unsigned long long)stride);
        }
    }
    return monotonic && aligned64;
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

float SnnPESubComponent::readBcsrWeightFromFile_(uint32_t post_local, uint32_t pre_global) const {
    uint32_t br = (bcsr_br_>0? bcsr_br_:1);
    uint32_t bc = (bcsr_bc_>0? bcsr_bc_:16);
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
    std::ifstream fin(bin_path, std::ios::binary);
    if (!fin.good()) return 0.0f;
    uint32_t start = 0, end = 0;
    fin.seekg(static_cast<std::streamoff>(rp_off + (size_t)block_row * sizeof(uint32_t)), std::ios::beg);
    fin.read(reinterpret_cast<char*>(&start), 4);
    fin.read(reinterpret_cast<char*>(&end), 4);
    if (!fin.good() || end <= start) return 0.0f;
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

bool SnnPESubComponent::bcsrRowIndexGet_(uint32_t block_row, std::vector<uint32_t>& out) {
    auto it = bcsr_row_index_cache_.find(block_row);
    if (it == bcsr_row_index_cache_.end()) return false;
    out = it->second;
    bcsr_count_row_index_hits_++;
    return true;
}

void SnnPESubComponent::bcsrRowIndexPut_(uint32_t block_row, std::vector<uint32_t>& data) {
    if (bcsr_row_index_cache_.size() >= bcsr_row_index_cache_cap_) {
        auto it = bcsr_row_index_cache_.begin();
        bcsr_row_index_cache_.erase(it);
    }
    bcsr_row_index_cache_[block_row] = data;
}

bool SnnPESubComponent::bcsrBlockGet_(uint32_t block_row, uint32_t block_col, std::vector<float>& out) {
    uint64_t key = ((uint64_t)block_row << 32) | block_col;
    auto it = bcsr_block_cache_.find(key);
    if (it == bcsr_block_cache_.end()) return false;
    out = it->second.data;
    bcsr_count_block_hits_++;
    return true;
}

void SnnPESubComponent::bcsrBlockPut_(uint32_t block_row, uint32_t block_col, std::vector<float>& data) {
    uint64_t key = ((uint64_t)block_row << 32) | block_col;
    auto it = bcsr_block_cache_.find(key);
    if (it != bcsr_block_cache_.end()) {
        it->second.data = data;
    } else {
        if (bcsr_block_cache_.size() >= bcsr_block_cache_cap_) {
            auto it2 = bcsr_block_cache_.begin();
            bcsr_block_cache_.erase(it2);
        }
        BcsrBlockEntry e; e.data = data; bcsr_block_cache_[key] = std::move(e);
    }
    bcsrPopulateWeightCache_(block_row, block_col, bcsr_block_cache_[key].data);
}

void SnnPESubComponent::requestWeightBCSR(uint32_t pre_global, uint32_t post_local, std::function<void(float)> cb) {
    recordActivePre_(pre_global);
    bcsr_req_edges_++;
    if (bcsr_force_file_read_) {
        float w = readBcsrWeightFromFile_(post_local, pre_global);
        if (cb) cb(w);
        return;
    }
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
    uint32_t br = (bcsr_br_>0? bcsr_br_:16);
    uint32_t bc = (bcsr_bc_>0? bcsr_bc_:16);
    uint32_t block_row = post_local / br;
    uint32_t intra_row = post_local % br;
    uint32_t block_col = pre_global / bc;
    uint32_t intra_col = pre_global % bc;
    const auto& rowptr_host = bcsr_weights_.rowptrHost();
    if (block_row+1 >= rowptr_host.size()) {
        if (window_read_debug_ && output_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-bcsr] core=%u block_row=%u rowptr_size=%zu out-of-range\n",
                core_id_, block_row, rowptr_host.size());
        }
        if (cb) cb(0.0f);
        return;
    }
    uint32_t start = rowptr_host[block_row];
    uint32_t end   = rowptr_host[block_row+1];
    if (end <= start) {
        if (window_read_debug_ && output_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-bcsr] core=%u row empty post_local=%u block_row=%u start=%u end=%u\n",
                core_id_, post_local, block_row, start, end);
        }
        if (cb) cb(0.0f);
        return;
    }
    std::vector<uint32_t> cols;
    if (!bcsrRowIndexGet_(block_row, cols)) {
        size_t bytes = (size_t)(end - start) * bcsr_idx_bytes_;
        uint64_t addr = bcsr_colidx_addr_ + (uint64_t)start * bcsr_idx_bytes_;
        if (window_read_debug_ && output_) {
            output_->verbose(CALL_INFO, 0, 0,
                "[diag-bcsr] core=%u issue colidx row=%u start=%u end=%u bytes=%zu addr=0x%llx\n",
                core_id_, block_row, start, end, bytes, (unsigned long long)addr);
        }
        PendingMemoryRequest pm{};
        pm.address = addr; pm.size = bytes; pm.issue_cycle = total_cycles_;
        pm.bcsr_kind = 2; pm.bcsr_block_row = block_row; pm.bcsr_target_block_col = block_col; pm.bcsr_intra_row = intra_row; pm.bcsr_intra_col = intra_col;
        pm.bcsr_row_start = start; pm.has_single_cb = (cb!=nullptr); pm.single_cb = cb;
        stats_reporter_.reportMemoryIssue(bytes, false);
        if (mem_backend_) mem_backend_->sendRead(addr, bytes, pm);
        return;
    }
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
    std::vector<float> blk;
    if (bcsrBlockGet_(block_row, block_col, blk)) {
        bcsr_req_block_hit_++;
        uint32_t off = intra_row * bc + intra_col;
        float w = (off < blk.size() ? blk[off] : 0.0f);
        if (window_read_debug_ && output_) {
            uint32_t post_local_dbg = block_row * br + intra_row;
            uint32_t post_global = global_neuron_base_ + post_local_dbg;
            uint32_t pre_global_effective = block_col * bc + intra_col;
            output_->verbose(CALL_INFO, 1, 0,
                "[diag-bcsr-weight] core=%u post_local=%u post_global=%u pre_global=%u weight=%.6f source=BcsrCache\n",
                core_id_, post_local_dbg, post_global, pre_global_effective, w);
        }
        if (cb) cb(w);
        return;
    }
    size_t block_bytes = (size_t)br * (size_t)bc * bcsr_val_bytes_;
    uint32_t global_block_index = start + idx_in_row;
    uint64_t addr = bcsr_blockdata_addr_ + (uint64_t)global_block_index * block_bytes;
    PendingMemoryRequest pm{}; pm.address = addr; pm.size = block_bytes; pm.issue_cycle = total_cycles_;
    pm.bcsr_kind = 3; pm.bcsr_block_row = block_row; pm.bcsr_target_block_col = block_col; pm.bcsr_intra_row = intra_row; pm.bcsr_intra_col = intra_col;
    pm.bcsr_row_start = start; pm.bcsr_idx_in_row = idx_in_row; pm.bcsr_global_block_index = global_block_index;
    pm.has_single_cb = (cb!=nullptr); pm.single_cb = cb;
    stats_reporter_.reportMemoryIssue(block_bytes, true);
    if (mem_backend_) mem_backend_->sendRead(addr, block_bytes, pm);
    bcsr_count_block_misses_++;
}

void SnnPESubComponent::bcsrPrefetchAll_() {
    if (!use_bcsr_) return;
    if (!bcsr_prefetch_all_) return;
    if (bcsr_prefetch_issued_) return;
    if (!memory_) return;
    if (!bcsr_weights_.isRowptrReady()) return;

    uint32_t rows = num_neurons_;
    uint32_t br = (bcsr_br_>0? bcsr_br_:16);
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
        size_t bytes = (size_t)(end - start) * bcsr_idx_bytes_;
        uint64_t addr = bcsr_colidx_addr_ + (uint64_t)start * bcsr_idx_bytes_;
        if (!mem_backend_ && memory_) {
            mem_backend_ = std::make_unique<StandardMemBackend>(memory_);
        }
        if (!mem_backend_) {
            if (output_) {
                output_->verbose(CALL_INFO, 0, 0,
                    "[diag-bcsr][warn] core=%u colidx prefetch skipped: mem_backend_未就绪\n",
                    core_id_);
            }
            continue;
        }
        PendingMemoryRequest pm{};
        pm.address = addr;
        pm.size = bytes;
        pm.issue_cycle = total_cycles_;
        pm.is_weight = true;
        pm.bcsr_kind = 2;
        pm.bcsr_block_row = block_row;
        pm.bcsr_row_start = start;
        pm.bcsr_prefetch_all = true;
        pm.bcsr_target_block_col = UINT32_MAX;
        stats_reporter_.reportMemoryIssue(bytes, false);
        mem_backend_->sendRead(addr, bytes, pm);
    }
    bcsr_prefetch_issued_ = true;
}

void SnnPESubComponent::bcsrPrefetchRowBlocks_(uint32_t block_row, const std::vector<uint32_t>& cols, uint32_t row_start) {
    if (!use_bcsr_) return;
    if (!memory_) return;
    uint32_t br = (bcsr_br_>0? bcsr_br_:16);
    uint32_t bc = (bcsr_bc_>0? bcsr_bc_:16);
    size_t block_bytes = (size_t)br * (size_t)bc * bcsr_val_bytes_;
    for (size_t i = 0; i < cols.size(); ++i) {
        uint32_t block_col = cols[i];
        uint64_t key = ((uint64_t)block_row << 32) | block_col;
        if (bcsr_block_cache_.find(key) != bcsr_block_cache_.end()) continue;
        uint32_t global_block_index = row_start + static_cast<uint32_t>(i);
        uint64_t addr = bcsr_blockdata_addr_ + (uint64_t)global_block_index * block_bytes;
        if (!mem_backend_ && memory_) {
            mem_backend_ = std::make_unique<StandardMemBackend>(memory_);
        }
        if (!mem_backend_) {
            if (output_) {
                output_->verbose(CALL_INFO, 0, 0,
                    "[diag-bcsr][warn] core=%u block prefetch skipped: mem_backend_未就绪 row=%u col=%u\n",
                    core_id_, block_row, block_col);
            }
            continue;
        }
        PendingMemoryRequest pm{};
        pm.address = addr;
        pm.size = block_bytes;
        pm.issue_cycle = total_cycles_;
        pm.is_weight = true;
        pm.bcsr_kind = 3;
        pm.bcsr_block_row = block_row;
        pm.bcsr_target_block_col = block_col;
        pm.bcsr_row_start = row_start;
        pm.bcsr_idx_in_row = static_cast<uint32_t>(i);
        pm.bcsr_global_block_index = global_block_index;
        pm.bcsr_prefetch_all = true;
        stats_reporter_.reportMemoryIssue(block_bytes, true);
        mem_backend_->sendRead(addr, block_bytes, pm);
        bcsr_count_block_misses_++;
    }
}

void SnnPESubComponent::bcsrPopulateWeightCache_(uint32_t block_row, uint32_t block_col, const std::vector<float>& blk) {
    if (!use_bcsr_) return;
    if (blk.empty()) return;
    uint32_t br = (bcsr_br_>0? bcsr_br_:16);
    uint32_t bc = (bcsr_bc_>0? bcsr_bc_:16);
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
