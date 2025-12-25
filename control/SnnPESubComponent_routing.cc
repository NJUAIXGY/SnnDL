// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// SnnPESubComponent_routing.cc: 路由构建与共享缓存逻辑拆分
//

#include <sst/core/sst_config.h>

#include "SnnPESubComponent.h"
#include "synapse/route/SpikeCommSubsystem.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <inttypes.h>
#include <iterator>
#include <limits>
#include <sstream>

using namespace SST;
using namespace SST::SnnDL;

#if 0
// Phase D:
// 旧的“控制层持有路由表 + 共享缓存 + 路由构建”实现已迁入 services/SpikeCommSubsystem，
// 控制层不再持有 routes/pending/cache 等路由细节。此处保留历史实现以便对照，但不参与编译。

// Logging helpers (local to this TU)
#ifndef SNNDL_LOGPTR
#define SNNDL_LOGPTR(ptr, lvl, ...) do { if (ptr) (ptr)->verbose(CALL_INFO, (lvl), 0, __VA_ARGS__); } while(0)
#endif
#ifndef SNNDL_LOG
#define SNNDL_LOG(lvl, ...) SNNDL_LOGPTR(output_, (lvl), __VA_ARGS__)
#endif

// === 路由共享缓存静态定义 ===
std::mutex SnnPESubComponent::s_route_cache_mtx_;
std::unordered_map<std::string, std::weak_ptr<const SnnPESubComponent::RouteMap>> SnnPESubComponent::s_route_cache_;

void SnnPESubComponent::logRoutingSummary_(const char* phase, const char* reason) {
    if (!(route_summary_enable_ && output_)) return;
    const size_t shared_entries = routes_shared_ ? routes_shared_->size() : 0;
    const size_t local_entries = shared_entries ? 0 : routes_by_source_.size();
    output_->verbose(CALL_INFO, 0, 0,
        "[route-summary][%s] node=%u core=%d routes_shared=%zu routes_local=%zu %s\n",
        phase ? phase : "-", node_id_, core_id_, shared_entries, local_entries,
        reason ? reason : "");
}

std::string SnnPESubComponent::buildRouteCacheKey() const {
    std::ostringstream oss;
    // 路由表是“全网全核心共享”的：只要权重模板与路由参数一致，不应因 (pe,core) 不同而重复构建，
    // 否则会导致巨量重复内存与构建耗时（每核心一份 routes_by_source_）。
    oss << "routes-v1"
        << "-rows" << num_neurons_
        << "-cols" << weights_cols_
        << "-total_nodes" << total_nodes_cfg_
        << "-cores_per_pe" << total_cores_
        << "-neurons_per_pe" << neurons_per_pe_cfg_
        << "-idxmode" << (use_post_row_pre_col_ ? "post_row_pre_col" : "pre_row_post_col")
        << "-template" << weights_template_
        << "-bcsr_br" << bcsr_br_
        << "-bcsr_bc" << bcsr_bc_
        << "-bcsr_ib" << bcsr_idx_bytes_
        << "-bcsr_vb" << bcsr_val_bytes_
        << "-topk" << routing_topk_
        << "-topkpe" << routing_topk_per_pe_
        << "-exself" << (route_exclude_self_pe_ ? 1 : 0)
        << "-layers" << route_layers_mask_
        << "-eps" << routing_epsilon_
        << "-mapmode" << mapping_mode_
        << "-mapfile" << mapping_edges_file_
        << "-maphdr" << (mapping_csv_has_header_ ? 1 : 0)
        << "-mapsplit" << mapping_csv_separator_
        << "-mapblk" << (mapping_assume_block_ids_ ? 1 : 0);
    return oss.str();
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
    uint32_t br = bcsr_br_ ? bcsr_br_ : 16;
    uint32_t bc = bcsr_bc_ ? bcsr_bc_ : 16;
    uint32_t idx_bytes = bcsr_idx_bytes_ ? bcsr_idx_bytes_ : 2;
    uint32_t val_bytes = bcsr_val_bytes_ ? bcsr_val_bytes_ : 4;
    uint64_t rowptr_off = (bcsr_weights_.rowptrAddr() > base_addr_) ? (bcsr_weights_.rowptrAddr() - base_addr_) : 0;
    uint64_t colidx_off = (bcsr_colidx_addr_ > base_addr_) ? (bcsr_colidx_addr_ - base_addr_) : 0;
    uint64_t blockdata_off = (bcsr_blockdata_addr_ > base_addr_) ? (bcsr_blockdata_addr_ - base_addr_) : 0;
    uint64_t blockids_off = (bcsr_blockids_addr_ > base_addr_) ? (bcsr_blockids_addr_ - base_addr_) : 0;
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

#endif // 0

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

void SnnPESubComponent::applyGatingDecision(uint32_t src_global, const std::vector<uint32_t>& dest_pes,
                             uint64_t current_cycle, uint64_t ttl_cycles)
{
    if (spike_comm_) {
        spike_comm_->applyGatingDecision(src_global, dest_pes, current_cycle, ttl_cycles);
    }
}
