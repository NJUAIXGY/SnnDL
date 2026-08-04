// -*- c++ -*-
//
// SnnPESubComponent_bcsr.cc: BCSR-related helpers extracted from the monolithic
// SnnPESubComponent.cc. Behavior preserved; only file-level split.
//

#include <sst/core/sst_config.h>
#include "SnnPESubComponent.h"
#include "snn/synapse/weights/WeightMemorySubsystem.h"
#include "snn/synapse/weights/SnnBcsrWeightManager.h"
#include "SnnDLLogging.h"
#include "snn/synapse/common/BcsrMeta.h"
#include "snn/synapse/common/BcsrDataSource.h"

#include <fstream>
#include <sstream>
#include <cmath>
#include <cctype>
#include <iterator>
#include <algorithm>
#include <cstdlib>
#include <cstring>

using namespace SST;
using namespace SST::SnnDL;

// NOTE(Universal-core experiments):
// Globally disable all BCSR optimizations in legacy control helpers as well:
// - rowIndex cache
// - block cache
// - populate dense weight cache from BCSR blocks
// BCSR remains a storage format/addressing scheme only.
static constexpr bool kEnableLegacyBcsrOptimizations = false;

bool SnnPESubComponent::BcsrLayout::validate(uint64_t base, Output* out, bool debug, uint32_t core_id, uint32_t node_id) const {
    std::string mode = layout_mode;
    std::transform(mode.begin(), mode.end(), mode.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (mode.empty()) mode = "flat";
    const bool monotonic = (colidx_offset >= rowptr_offset) && (blockdata_offset >= colidx_offset);
    const bool aligned64 = ((rowptr_offset | colidx_offset | blockdata_offset | blockids_offset) & 0x3FULL) == 0;
    const size_t block_row_bytes =
        static_cast<size_t>(block_cols ? block_cols : 0) * static_cast<size_t>(val_bytes ? val_bytes : 0);
    const bool rowpack_stride_ok =
        (mode != "rowpack_v1") ||
        ((colidx_row_stride_bytes >= (idx_bytes ? idx_bytes : 2)) &&
         (blockdata_row_stride_bytes >= block_row_bytes));
    const uint64_t stride = per_core_stride;
    const uint64_t max_off = maxOffset();
    const bool stride_ok = (stride == 0) ? true : (max_off < stride);
    if (debug && out) {
        out->verbose(CALL_INFO, 2, 0,
            "[diag-bcsr-base] node=%u core=%u base=0x%lx rp=0x%lx ci=0x%lx bd=0x%lx ids=0x%lx stride=%" PRIu64
            " stride_ok=%d align64=%d mono=%d mode=%s ci_row_stride=%u bd_row_stride=%u ids_row_stride=%u rowpack_ok=%d br=%u bc=%u idx=%u val=%u\n",
            node_id, core_id,
            (unsigned long)base,
            (unsigned long)(base + rowptr_offset),
            (unsigned long)(base + colidx_offset),
            (unsigned long)(base + blockdata_offset),
            (unsigned long)(blockids_offset ? base + blockids_offset : 0),
            stride, stride_ok ? 1 : 0,
            aligned64 ? 1 : 0,
            monotonic ? 1 : 0,
            mode.c_str(),
            colidx_row_stride_bytes,
            blockdata_row_stride_bytes,
            blockids_row_stride_bytes,
            rowpack_stride_ok ? 1 : 0,
            block_rows ? block_rows : 0,
            block_cols ? block_cols : 0,
            idx_bytes ? idx_bytes : 0,
            val_bytes ? val_bytes : 0);
        if (!aligned64 || !monotonic || !stride_ok || !rowpack_stride_ok) {
            out->verbose(CALL_INFO, 2, 0,
                "[diag-bcsr-base] offsets/layout suspect (aligned64=%d monotonic=%d stride_ok=%d rowpack_ok=%d max_off=0x%llx stride=0x%llx)\n",
                aligned64 ? 1 : 0, monotonic ? 1 : 0, stride_ok ? 1 : 0, rowpack_stride_ok ? 1 : 0,
                (unsigned long long)max_off, (unsigned long long)stride);
        }
    }
    return monotonic && aligned64 && rowpack_stride_ok;
}

bool SnnPESubComponent::parseBcsrMeta(const std::string& meta_path, uint32_t& rows_out, uint32_t& cols_out,
                                      uint32_t& br_out, uint32_t& bc_out,
                                      uint32_t& idx_bytes_out, uint32_t& val_bytes_out,
                                      uint64_t& rowptr_off_out, uint64_t& colidx_off_out,
                                      uint64_t& blockdata_off_out, uint64_t& blockids_off_out,
                                      uint32_t& total_blocks_out) const {
    BcsrMeta meta{};
    if (!parseBcsrMetaJsonFile(meta_path, meta)) return false;

    if (meta.rows == 0 || meta.cols == 0) return false;
    if (meta.rowptr_offset == 0 && meta.colidx_offset == 0 && meta.blockdata_offset == 0) return false;

    rows_out = meta.rows;
    cols_out = meta.cols;
    br_out = meta.br;
    bc_out = meta.bc;
    idx_bytes_out = meta.idx_bytes;
    val_bytes_out = meta.val_bytes;
    rowptr_off_out = meta.rowptr_offset;
    colidx_off_out = meta.colidx_offset;
    blockdata_off_out = meta.blockdata_offset;
    blockids_off_out = meta.blockids_offset;
    total_blocks_out = meta.total_blocks;
    return true;
}

float SnnPESubComponent::readBcsrWeightFromFile_(uint32_t post_local, uint32_t pre_global) const {
    std::string bin_path = resolveWeightTemplate(node_id_, core_id_);
    if (bin_path.empty()) return 0.0f;
    BcsrFileSource source;
    if (!loadBcsrFileSource(bin_path, num_neurons_, source, nullptr)) return 0.0f;
    float weight = 0.0f;
    return readBcsrWeight(source, post_local, pre_global, weight) ? weight : 0.0f;
}

bool SnnPESubComponent::bcsrRowIndexGet_(uint32_t block_row, std::vector<uint32_t>& out) {
    if (!kEnableLegacyBcsrOptimizations) return false;
    auto it = bcsr_row_index_cache_.find(block_row);
    if (it == bcsr_row_index_cache_.end()) return false;
    out = it->second;
    bcsr_count_row_index_hits_++;
    return true;
}

void SnnPESubComponent::bcsrRowIndexPut_(uint32_t block_row, std::vector<uint32_t>& data) {
    if (!kEnableLegacyBcsrOptimizations) return;
    if (bcsr_row_index_cache_cap_ == 0) return; // avoid UB: erase(begin()) on empty map
    if (bcsr_row_index_cache_.size() >= bcsr_row_index_cache_cap_) {
        auto it = bcsr_row_index_cache_.begin();
        bcsr_row_index_cache_.erase(it);
    }
    bcsr_row_index_cache_[block_row] = data;
}

bool SnnPESubComponent::bcsrBlockGet_(uint32_t block_row, uint32_t block_col, std::vector<float>& out) {
    if (!kEnableLegacyBcsrOptimizations) return false;
    uint64_t key = ((uint64_t)block_row << 32) | block_col;
    auto it = bcsr_block_cache_.find(key);
    if (it == bcsr_block_cache_.end()) return false;
    out = it->second.data;
    bcsr_count_block_hits_++;
    return true;
}

void SnnPESubComponent::bcsrBlockPut_(uint32_t block_row, uint32_t block_col, std::vector<float>& data) {
    if (!kEnableLegacyBcsrOptimizations) return;
    if (bcsr_block_cache_cap_ == 0) return; // avoid UB: erase(begin()) on empty map
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
    // Phase-1A: BCSR 读语义由 WeightMemorySubsystem 统一承载；控制层仅保留轻量封装。
    if (!weight_mem_subsystem_) {
        if (cb) cb(0.0f);
        return;
    }
    weight_mem_subsystem_->requestBCSR(pre_global, post_local, std::move(cb));
}

void SnnPESubComponent::bcsrPopulateWeightCache_(uint32_t block_row, uint32_t block_col, const std::vector<float>& blk) {
    if (!kEnableLegacyBcsrOptimizations) return;
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
    return bcsr_weights_->expectedRowptrEntries(num_neurons_);
}

size_t SnnPESubComponent::expectedRowptrBytes_() const {
    return bcsr_weights_->expectedRowptrBytes(num_neurons_);
}

bool SnnPESubComponent::installRowptrFromBytes_(const uint8_t* data, size_t bytes, const char* source, bool count_stats) {
    if (!bcsr_weights_->installRowptrFromBytes(data, bytes, num_neurons_)) {
        SNNDL_DEBUG_LOG(2,
            "[diag-bcsr] core=%u rowptr install failed source=%s bytes=%zu expect=%zu\n",
            core_id_, source ? source : "-", bytes, expectedRowptrBytes_());
        return false;
    }
    if (count_stats) {
        bcsr_count_row_reads_++;
        bcsr_bytes_idx_ += bytes;
    }
    if (window_read_debug_ && output_ && output_->getVerboseLevel() >= 2) {
        const auto& rp = bcsr_weights_->rowptrHost();
        output_->verbose(CALL_INFO, 2, 0,
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
    BcsrFileSource source;
    std::string source_error;
    if (!loadBcsrFileSource(bin_path, num_neurons_, source, &source_error)) {
        SNNDL_DEBUG_LOG(2, "[diag-bcsr] core=%u fallback source rejected %s: %s\n",
                        core_id_, bin_path.c_str(), source_error.c_str());
        return false;
    }
    size_t bytes = expectedRowptrBytes_();
    std::vector<uint8_t> buffer(bytes);
    std::vector<uint32_t> rowptr;
    if (!readBcsrRowptr(source, num_neurons_, rowptr) ||
        rowptr.size() * sizeof(uint32_t) != bytes) {
        SNNDL_DEBUG_LOG(2,
            "[diag-bcsr] core=%u fallback read failed %s bytes=%zu\n",
            core_id_, bin_path.c_str(), bytes);
        return false;
    }
    std::memcpy(buffer.data(), rowptr.data(), bytes);
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
    if (window_read_debug_ && output_ && output_->getVerboseLevel() >= 2) {
        output_->verbose(CALL_INFO, 2, 0,
            "[diag-bcsr] core=%u rowptr fallback applied (%s)\n",
            core_id_, msg);
    }
}
