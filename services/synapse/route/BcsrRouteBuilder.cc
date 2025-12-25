// -*- c++ -*-
//
// BcsrRouteBuilder implementation
//

#include "synapse/route/BcsrRouteBuilder.h"

#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <vector>

#include <sst/core/output.h>

namespace SST { namespace SnnDL {

namespace {

constexpr uint32_t kBcsrSentinelId = 0xFFFFFFFFu;

bool extractUnsigned_(const std::string& text, const char* key, uint64_t& value) {
    auto pos = text.find(key);
    if (pos == std::string::npos) return false;
    pos = text.find(':', pos);
    if (pos == std::string::npos) return false;
    ++pos;
    while (pos < text.size() && (std::isspace(static_cast<unsigned char>(text[pos])) || text[pos] == '"')) ++pos;
    size_t end = pos;
    while (end < text.size() &&
           (std::isdigit(static_cast<unsigned char>(text[end])) || text[end] == 'x' || text[end] == 'X')) {
        ++end;
    }
    if (end <= pos) return false;
    value = std::strtoull(text.substr(pos, end - pos).c_str(), nullptr, 0);
    return true;
}

} // namespace

bool parseBcsrMetaJson(const std::string& meta_path,
                       uint32_t& rows_out, uint32_t& cols_out,
                       uint32_t& br_out, uint32_t& bc_out,
                       uint32_t& idx_bytes_out, uint32_t& val_bytes_out,
                       uint64_t& rowptr_off_out, uint64_t& colidx_off_out,
                       uint64_t& blockdata_off_out, uint64_t& blockids_off_out,
                       uint32_t& total_blocks_out) {
    std::ifstream meta(meta_path);
    if (!meta.good()) return false;
    std::string text((std::istreambuf_iterator<char>(meta)), std::istreambuf_iterator<char>());
    uint64_t value = 0;
    bool ok = false;
    if (extractUnsigned_(text, "\"rows\"", value)) { rows_out = static_cast<uint32_t>(value); ok = true; }
    if (extractUnsigned_(text, "\"cols\"", value)) { cols_out = static_cast<uint32_t>(value); ok = true; }
    if (extractUnsigned_(text, "\"br\"", value)) { br_out = static_cast<uint32_t>(value); ok = true; }
    if (extractUnsigned_(text, "\"bc\"", value)) { bc_out = static_cast<uint32_t>(value); ok = true; }
    if (extractUnsigned_(text, "\"idx_bytes\"", value)) { idx_bytes_out = static_cast<uint32_t>(value); ok = true; }
    if (extractUnsigned_(text, "\"val_bytes\"", value)) { val_bytes_out = static_cast<uint32_t>(value); ok = true; }
    if (extractUnsigned_(text, "\"rowptr_offset\"", value)) { rowptr_off_out = value; ok = true; }
    if (extractUnsigned_(text, "\"colidx_offset\"", value)) { colidx_off_out = value; ok = true; }
    if (extractUnsigned_(text, "\"blockdata_offset\"", value)) { blockdata_off_out = value; ok = true; }
    if (extractUnsigned_(text, "\"blockids_offset\"", value)) { blockids_off_out = value; ok = true; }
    if (extractUnsigned_(text, "\"total_blocks\"", value)) { total_blocks_out = static_cast<uint32_t>(value); ok = true; }
    return ok;
}

std::string resolveBcsrTemplate(const std::string& tmpl, uint32_t pe, int core) {
    if (tmpl.empty()) return "";
    std::string path = tmpl;
    auto replaceIndexed = [&](const std::string& marker, uint32_t value, int width) {
        size_t pos = 0;
        while ((pos = path.find(marker, pos)) != std::string::npos) {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%0*u", width, value);
            path.replace(pos, marker.size(), buf);
            pos += static_cast<size_t>(width);
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

bool appendRoutesFromBcsrFile(const SynapseRouteBuildConfig& cfg,
                              SST::Output* out,
                              const std::string& path,
                              uint32_t pe_index,
                              int core_index,
                              uint32_t rows_hint,
                              ISynapseRoute::RouteMap& routes_out,
                              const BcsrAppendOptions& opt) {
    uint32_t rows = rows_hint;
    uint32_t cols = cfg.cols;
    uint32_t br = cfg.bcsr_br ? cfg.bcsr_br : 16;
    uint32_t bc = cfg.bcsr_bc ? cfg.bcsr_bc : 16;
    uint32_t idx_bytes = cfg.bcsr_idx_bytes ? cfg.bcsr_idx_bytes : 2;
    uint32_t val_bytes = cfg.bcsr_val_bytes ? cfg.bcsr_val_bytes : 4;
    uint64_t rowptr_off = (cfg.bcsr_rowptr_addr > cfg.base_addr) ? (cfg.bcsr_rowptr_addr - cfg.base_addr) : 0;
    uint64_t colidx_off = (cfg.bcsr_colidx_addr > cfg.base_addr) ? (cfg.bcsr_colidx_addr - cfg.base_addr) : 0;
    uint64_t blockdata_off = (cfg.bcsr_blockdata_addr > cfg.base_addr) ? (cfg.bcsr_blockdata_addr - cfg.base_addr) : 0;
    uint64_t blockids_off = (cfg.bcsr_blockids_addr > cfg.base_addr) ? (cfg.bcsr_blockids_addr - cfg.base_addr) : 0;
    uint32_t total_blocks = 0;
    uint32_t meta_cols = cols;
    const std::string meta_path = path + ".meta.json";
    if (parseBcsrMetaJson(meta_path, rows, meta_cols, br, bc, idx_bytes, val_bytes,
                          rowptr_off, colidx_off, blockdata_off, blockids_off, total_blocks)) {
        if (meta_cols > 0) cols = meta_cols;
    } else {
        total_blocks = 0;
    }
    if (rows == 0 || cols == 0) {
        if (out) out->verbose(CALL_INFO, 0, 0, "⚠️ BCSR路由: 元数据缺失 rows/cols %s\n", path.c_str());
        return false;
    }
    if (val_bytes != 4) {
        if (out) out->verbose(CALL_INFO, 0, 0, "⚠️ BCSR路由: val_bytes=%u 不受支持 %s\n", val_bytes, path.c_str());
        return false;
    }
    const uint32_t n_block_rows = (rows + br - 1) / br;
    const uint64_t neurons_per_pe =
        (cfg.neurons_per_pe > 0)
            ? static_cast<uint64_t>(cfg.neurons_per_pe)
            : (cfg.cores_per_pe > 0 ? static_cast<uint64_t>(cfg.cores_per_pe) * static_cast<uint64_t>(rows)
                                    : static_cast<uint64_t>(rows));
    const uint64_t core_offset_global = (core_index > 0) ? static_cast<uint64_t>(core_index) * static_cast<uint64_t>(rows) : 0ULL;
    std::ifstream fin(path, std::ios::binary);
    if (!fin.good()) {
        if (out) out->verbose(CALL_INFO, 0, 0, "⚠️ BCSR路由: 无法读取 %s\n", path.c_str());
        return false;
    }
    fin.seekg(static_cast<std::streamoff>(rowptr_off), std::ios::beg);
    std::vector<uint32_t> rowptr(n_block_rows + 1, 0);
    fin.read(reinterpret_cast<char*>(rowptr.data()), rowptr.size() * sizeof(uint32_t));
    if (!fin.good()) {
        if (out) out->verbose(CALL_INFO, 0, 0, "⚠️ BCSR路由: 读取rowptr失败 %s\n", path.c_str());
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
            if (out) out->verbose(CALL_INFO, 0, 0, "⚠️ BCSR路由: 读取colidx失败 %s\n", path.c_str());
            return false;
        }
        for (uint32_t i = 0; i < total_blocks; ++i) block_cols[i] = tmp[i];
    } else {
        fin.read(reinterpret_cast<char*>(block_cols.data()), block_cols.size() * sizeof(uint32_t));
        if (!fin.good()) {
            if (out) out->verbose(CALL_INFO, 0, 0, "⚠️ BCSR路由: 读取colidx失败 %s\n", path.c_str());
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
    std::vector<uint32_t> blockids(floats_per_block, kBcsrSentinelId);
    const uint64_t total_global_neurons =
        static_cast<uint64_t>(cfg.total_nodes) * (neurons_per_pe > 0 ? neurons_per_pe : 1ULL);
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
                if (out) out->verbose(CALL_INFO, 0, 0, "⚠️ BCSR路由: 读取block数据失败 %s\n", path.c_str());
                return false;
            }
            for (uint32_t rr = 0; rr < br; ++rr) {
                uint32_t post_local = block_row * br + rr;
                if (post_local >= rows) continue;
                for (uint32_t cc = 0; cc < bc; ++cc) {
                    size_t off = static_cast<size_t>(rr) * bc + cc;
                    float weight = blockdata[off];
                    if (std::fabs(weight) <= cfg.routing_epsilon) continue;
                    // 重要语义：
                    // - blockids 仅作为有效位/哨兵（0xFFFFFFFF 表示无边），不能作为 post_global。
                    // - post_global 必须由 (pe, core, post_local) 计算，保证与 WeightMemorySubsystem 的寻址/落盘口径一致。
                    if (blockids_off > 0 && blockids[off] == kBcsrSentinelId) continue;
                    const uint64_t post_global_64 =
                        static_cast<uint64_t>(pe_index) * neurons_per_pe + core_offset_global + static_cast<uint64_t>(post_local);
                    if (post_global_64 >= total_global_neurons) continue;
                    const uint32_t post_global = static_cast<uint32_t>(post_global_64);
                    uint32_t pre_global = block_col * bc + cc;
                    if (pre_global >= cols) continue;
                    if (pre_global < opt.pre_begin || pre_global >= opt.pre_end) continue;
                    routes_out[pre_global].push_back(post_global);
                }
            }
        }
    }
    return true;
}

}} // namespace SST::SnnDL
