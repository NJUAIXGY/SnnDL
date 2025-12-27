// -*- c++ -*-
//
// SpikeCommSubsystem: fanout + 事件构造 + 传输调用
//

#include "SpikeCommSubsystem.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <inttypes.h>
#include <iterator>
#include <limits>
#include <sstream>
#include <unordered_set>

#include <sst/core/statapi/stataccumulator.h>

#include "SpikeEvent.h"

namespace SST { namespace SnnDL {

#if 0
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

bool parseBcsrMeta_(const std::string& meta_path,
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

uint32_t layerIdFromPe_(uint32_t pe) {
    // 固定4x4网格层划分：I:0-3, H1:4-7, H2:8-11, O:12-15
    if (pe <= 3) return 0;
    if (pe <= 7) return 1;
    if (pe <= 11) return 2;
    return 3;
}

void parseLayerMask_(const std::string& mask_in,
                     std::unordered_set<uint32_t>& allowed_edges,
                     bool& allow_all_layers) {
    allowed_edges.clear();
    allow_all_layers = true;
    if (mask_in.empty()) return;

    allow_all_layers = false;
    std::string mask = mask_in;
    for (auto& ch : mask) ch = (char)std::toupper((unsigned char)ch);

    std::vector<std::string> toks;
    size_t start = 0;
    for (size_t i = 0; i <= mask.size(); ++i) {
        if (i == mask.size() || mask[i] == ',' || mask[i] == ';') {
            if (i > start) toks.emplace_back(mask.substr(start, i - start));
            start = i + 1;
        }
    }

    auto layerId = [](const std::string& s)->int{
        if (s == "I") return 0;
        if (s == "H1") return 1;
        if (s == "H2") return 2;
        if (s == "O") return 3;
        return -1;
    };

    for (auto& t : toks) {
        size_t p = t.find('>');
        if (p == std::string::npos) continue;
        std::string a = t.substr(0, p);
        std::string b = t.substr(p + 1);
        int la = layerId(a);
        int lb = layerId(b);
        if (la >= 0 && lb >= 0) {
            uint32_t key = ((uint32_t)la << 8) | (uint32_t)lb;
            allowed_edges.insert(key);
        }
    }
}

std::string buildRouteCacheKey_(const SpikeCommRoutingConfig& cfg) {
    std::ostringstream oss;
    oss << "routes-v1"
        << "-rows" << cfg.rows
        << "-cols" << cfg.cols
        << "-total_nodes" << cfg.total_nodes
        << "-cores_per_pe" << cfg.cores_per_pe
        << "-neurons_per_pe" << cfg.neurons_per_pe
        << "-idxmode" << (cfg.use_post_row_pre_col ? "post_row_pre_col" : "pre_row_post_col")
        << "-template" << cfg.weights_template
        << "-bcsr_br" << cfg.bcsr_br
        << "-bcsr_bc" << cfg.bcsr_bc
        << "-bcsr_ib" << cfg.bcsr_idx_bytes
        << "-bcsr_vb" << cfg.bcsr_val_bytes
        << "-topk" << cfg.routing_topk
        << "-topkpe" << cfg.routing_topk_per_pe
        << "-exself" << (cfg.route_exclude_self_pe ? 1 : 0)
        << "-layers" << cfg.route_layers_mask
        << "-eps" << cfg.routing_epsilon
        << "-mapmode" << cfg.mapping_mode
        << "-mapfile" << cfg.mapping_edges_file
        << "-maphdr" << (cfg.mapping_csv_has_header ? 1 : 0)
        << "-mapsplit" << cfg.mapping_csv_separator
        << "-mapblk" << (cfg.mapping_assume_block_ids ? 1 : 0);
    return oss.str();
}

void buildRoutesFromCandidates_(const SpikeCommRoutingConfig& cfg,
                               Output* out,
                               bool verify_routing_weights,
                               const std::unordered_map<uint32_t, std::vector<std::pair<float,uint32_t>>>& tmp,
                               uint32_t rows,
                               bool group_by_pe,
                               SpikeCommSubsystem::RouteMap& routes_out) {
    for (auto& kv : tmp) {
        uint32_t pre = kv.first;
        const auto& lst_in = kv.second;
        if (lst_in.empty()) continue;

        std::vector<std::pair<float,uint32_t>> lst = lst_in;
        std::vector<uint32_t> final_routes;

        if (cfg.routing_topk_per_pe > 0) {
            std::unordered_map<uint32_t, std::vector<std::pair<float,uint32_t>>> by_group;
            by_group.reserve(16);
            for (auto& p : lst) {
                uint32_t gid = group_by_pe ? (p.second / rows) : 0u;
                by_group[gid].push_back(p);
            }
            std::vector<uint32_t> out_routes;
            for (auto& g : by_group) {
                auto& vec = g.second;
                if (vec.size() > cfg.routing_topk_per_pe) {
                    std::partial_sort(vec.begin(), vec.begin() + cfg.routing_topk_per_pe, vec.end(),
                        [](const auto& a, const auto& b) {
                            float aw = std::fabs(a.first);
                            float bw = std::fabs(b.first);
                            if (aw == bw) return a.second < b.second;
                            return aw > bw;
                        });
                    vec.resize(cfg.routing_topk_per_pe);
                }
                for (auto& p : vec) out_routes.push_back(p.second);
            }
            if (cfg.routing_topk > 0 && out_routes.size() > cfg.routing_topk) {
                std::vector<std::pair<float,uint32_t>> tmp2;
                tmp2.reserve(out_routes.size());
                std::unordered_set<uint32_t> keep(out_routes.begin(), out_routes.end());
                for (auto& p : lst) if (keep.count(p.second)) tmp2.push_back(p);
                std::partial_sort(tmp2.begin(), tmp2.begin() + cfg.routing_topk, tmp2.end(),
                    [](const auto& a, const auto& b) {
                        float aw = std::fabs(a.first);
                        float bw = std::fabs(b.first);
                        if (aw == bw) return a.second < b.second;
                        return aw > bw;
                    });
                tmp2.resize(cfg.routing_topk);
                std::vector<uint32_t> final_out;
                final_out.reserve(tmp2.size());
                for (auto& p : tmp2) final_out.push_back(p.second);
                final_routes = std::move(final_out);
            } else {
                final_routes = std::move(out_routes);
            }
        } else if (cfg.routing_topk > 0) {
            if (lst.size() > cfg.routing_topk) {
                std::partial_sort(lst.begin(), lst.begin() + cfg.routing_topk, lst.end(),
                    [](const auto& a, const auto& b) {
                        float aw = std::fabs(a.first);
                        float bw = std::fabs(b.first);
                        if (aw == bw) return a.second < b.second;
                        return aw > bw;
                    });
                lst.resize(cfg.routing_topk);
            }
            std::vector<uint32_t> out_routes;
            out_routes.reserve(lst.size());
            for (auto& p : lst) out_routes.push_back(p.second);
            final_routes = std::move(out_routes);
        } else {
            std::sort(lst.begin(), lst.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
            lst.erase(std::unique(lst.begin(), lst.end(),
                                  [](const auto& a, const auto& b) { return a.second == b.second; }),
                      lst.end());
            std::vector<uint32_t> out_routes;
            out_routes.reserve(lst.size());
            for (auto& p : lst) out_routes.push_back(p.second);
            final_routes = std::move(out_routes);
        }

        if (verify_routing_weights && out) {
            uint64_t mismatch = 0;
            float min_abs = std::numeric_limits<float>::infinity();
            for (uint32_t dest : final_routes) {
                float w = 0.0f;
                bool found = false;
                for (const auto& cand : lst_in) {
                    if (cand.second == dest) { w = cand.first; found = true; break; }
                }
                if (!found) {
                    out->verbose(CALL_INFO, 0, 0,
                                 "⚠️ 路由验证: pre=%u dest=%u 未在候选列表中找到原始权重\n",
                                 pre, dest);
                    continue;
                }
                float absw = std::fabs(w);
                if (absw < min_abs) min_abs = absw;
                if (absw <= cfg.routing_epsilon) {
                    mismatch++;
                    out->verbose(CALL_INFO, 0, 0,
                                 "⚠️ 路由验证: pre=%u dest=%u weight=%.6f <= epsilon=%.6f\n",
                                 pre, dest, w, cfg.routing_epsilon);
                }
            }
            if (!final_routes.empty()) {
                float report_min = std::isfinite(min_abs) ? min_abs : 0.0f;
                out->verbose(CALL_INFO, 0, 0,
                             "🔍 路由验证: pre=%u fanout=%zu min_abs_weight=%.6f\n",
                             pre, final_routes.size(), report_min);
            }
            if (mismatch > 0) {
                out->verbose(CALL_INFO, 0, 0,
                             "⚠️ 路由验证: pre=%u 存在%" PRIu64 "个未达阈值的扇出\n",
                             pre, mismatch);
            }
        }

        // 确定性保证：
        // routes 的“发送顺序”会影响同一时间戳下事件入队次序（尤其在 MPI 多 rank），
        // 进而放大为窗口边界/发放统计的非确定性。这里统一对目的列表做排序与去重，
        // 保证跨运行的路由发射顺序稳定且无重复目的。
        std::sort(final_routes.begin(), final_routes.end());
        final_routes.erase(std::unique(final_routes.begin(), final_routes.end()), final_routes.end());

        routes_out[pre] = std::move(final_routes);
    }
}

bool buildRoutesFromEdgesCSV_(const SpikeCommRoutingConfig& cfg,
                             Output* out,
                             const std::unordered_set<uint32_t>& allowed_layer_edges,
                             bool allow_all_layers,
                             SpikeCommSubsystem::RouteMap& routes_out) {
    routes_out.clear();
    const uint32_t rows = cfg.rows;
    std::ifstream fin(cfg.mapping_edges_file);
    if (!fin.good()) {
        if (out) out->verbose(CALL_INFO, 1, 0, "⚠️ 无法打开映射边文件: %s\n", cfg.mapping_edges_file.c_str());
        return false;
    }
    std::string line;
    if (cfg.mapping_csv_has_header) std::getline(fin, line);
    auto split = [&cfg](const std::string& s)->std::vector<std::string>{
        std::vector<std::string> outv;
        std::string cur;
        char sep = cfg.mapping_csv_separator.empty() ? ',' : cfg.mapping_csv_separator[0];
        std::istringstream ss(s);
        while (std::getline(ss, cur, sep)) outv.push_back(cur);
        if (outv.empty()) { std::istringstream ss2(s); while (ss2 >> cur) outv.push_back(cur); }
        return outv;
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
        if (std::fabs(w) <= cfg.routing_epsilon) continue;
        if (cfg.mapping_assume_block_ids) {
            uint32_t src_pe = src / rows;
            uint32_t dst_pe = dst / rows;
            if (cfg.route_exclude_self_pe && src_pe == dst_pe) { dropped_self++; continue; }
            if (!allow_all_layers) {
                uint32_t la = layerIdFromPe_(src_pe);
                uint32_t lb = layerIdFromPe_(dst_pe);
                uint32_t key = (la<<8) | lb;
                if (allowed_layer_edges.find(key) == allowed_layer_edges.end()) { dropped_layer++; continue; }
            }
        }
        tmp[src].emplace_back(std::fabs(w), dst);
    }
    buildRoutesFromCandidates_(cfg, out, cfg.verify_routing_weights, tmp, rows,
                              /*group_by_pe=*/cfg.mapping_assume_block_ids, routes_out);
    if ((cfg.route_exclude_self_pe || !allow_all_layers) && cfg.route_filter_warn && out) {
        out->verbose(CALL_INFO, 1, 0,
            "⚠️ 路由过滤(映射CSV)启用: exclude_self_pe=%d, layers_mask='%s' (丢弃: self=%" PRIu64 ", layer=%" PRIu64 ")\n",
            cfg.route_exclude_self_pe ? 1 : 0, cfg.route_layers_mask.c_str(), dropped_self, dropped_layer);
    }
    return !routes_out.empty();
}

bool appendRoutesFromBcsrFile_(const SpikeCommRoutingConfig& cfg,
                               Output* out,
                               const std::string& path,
                               uint32_t pe_index,
                               int core_index,
                               uint32_t rows_hint,
                               SpikeCommSubsystem::RouteMap& routes_out) {
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
    if (parseBcsrMeta_(meta_path, rows, meta_cols, br, bc, idx_bytes, val_bytes,
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
                    // 重要语义：blockids 仅作为有效位/哨兵，不能作为 post_global。
                    if (blockids_off > 0 && blockids[off] == kBcsrSentinelId) continue;
                    const uint64_t post_global_64 =
                        static_cast<uint64_t>(pe_index) * neurons_per_pe + core_offset_global + static_cast<uint64_t>(post_local);
                    if (post_global_64 >= total_global_neurons) continue;
                    const uint32_t post_global = static_cast<uint32_t>(post_global_64);
                    uint32_t pre_global = block_col * bc + cc;
                    if (pre_global >= cols) continue;
                    routes_out[pre_global].push_back(post_global);
                }
            }
        }
    }
    return true;
}

std::string resolveBcsrTemplate_(const std::string& tmpl, uint32_t pe, int core) {
    if (tmpl.empty()) return "";
    std::string path = tmpl;
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

bool buildWeightDrivenRoutesFromBcsr_(const SpikeCommRoutingConfig& cfg,
                                     Output* out,
                                     SpikeCommSubsystem::RouteMap& routes_out) {
    routes_out.clear();
    const uint32_t cores_per_pe = (cfg.cores_per_pe > 0) ? cfg.cores_per_pe : 1u;
    uint32_t rows_hint = (cores_per_pe > 0) ? static_cast<uint32_t>((cfg.rows + cores_per_pe - 1) / cores_per_pe) : cfg.rows;
    if (rows_hint == 0) rows_hint = cfg.rows;
    bool ok = true;
    for (uint32_t pe = 0; pe < cfg.total_nodes; ++pe) {
        for (uint32_t core = 0; core < cores_per_pe; ++core) {
            std::string path = resolveBcsrTemplate_(cfg.weights_template, pe, static_cast<int>(core));
            if (path.empty()) { ok = false; break; }
            if (!appendRoutesFromBcsrFile_(cfg, out, path, pe, static_cast<int>(core), rows_hint, routes_out)) {
                ok = false;
                break;
            }
        }
        if (!ok) break;
    }
    if (!ok) {
        routes_out.clear();
        return false;
    }
    return !routes_out.empty();
}

bool buildWeightDrivenRoutesDense_(const SpikeCommRoutingConfig& cfg,
                                  Output* out,
                                  const std::unordered_set<uint32_t>& allowed_layer_edges,
                                  bool allow_all_layers,
                                  SpikeCommSubsystem::RouteMap& routes_out) {
    routes_out.clear();
    if (cfg.weights_template.empty()) {
        if (out) out->verbose(CALL_INFO, 1, 0, "⚠️ 路由构建失败：weights_template 未提供\n");
        return false;
    }
    if (cfg.weights_template.find(".bcsr") != std::string::npos ||
        cfg.weights_template.find(".BCSR") != std::string::npos) {
        return buildWeightDrivenRoutesFromBcsr_(cfg, out, routes_out);
    }

    const uint32_t rows = cfg.rows;
    const uint32_t cols = cfg.cols;
    const uint32_t total_nodes = cfg.total_nodes;
    const size_t expected = static_cast<size_t>(rows) * static_cast<size_t>(cols);
    std::unordered_map<uint32_t, std::vector<std::pair<float,uint32_t>>> tmp;
    tmp.reserve(cols);
    uint64_t dropped_self_pe = 0;
    uint64_t dropped_layer_mask = 0;

    for (uint32_t pe = 0; pe < total_nodes; ++pe) {
        std::string path = cfg.weights_template;
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
            if (out) out->verbose(CALL_INFO, 1, 0, "⚠️ 路由构建：无法读取权重文件 %s\n", path.c_str());
            continue;
        }
        fin.seekg(0, std::ios::end);
        std::streamsize bytes = fin.tellg();
        fin.seekg(0, std::ios::beg);
        if (bytes <= 0 || (bytes % sizeof(float) != 0)) {
            if (out) out->verbose(CALL_INFO, 1, 0, "⚠️ 路由构建：文件尺寸异常 %s\n", path.c_str());
            continue;
        }
        size_t count = static_cast<size_t>(bytes / sizeof(float));
        if (count < expected) {
            if (out) out->verbose(CALL_INFO, 1, 0, "⚠️ 路由构建：文件过短 %s (%zu<%zu)\n", path.c_str(), count, expected);
            continue;
        }
        std::vector<float> buf(count);
        fin.read(reinterpret_cast<char*>(buf.data()), bytes);
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t col = 0; col < cols; ++col) {
                size_t idx = static_cast<size_t>(row) * static_cast<size_t>(cols) + col;
                float w = buf[idx];
                if (std::fabs(w) > cfg.routing_epsilon) {
                    uint32_t pre_global = col;
                    uint32_t dest_global = pe * rows + row;
                    if (cfg.route_exclude_self_pe) {
                        uint32_t src_pe = pre_global / rows;
                        if (src_pe == pe) { dropped_self_pe++; continue; }
                    }
                    if (!allow_all_layers) {
                        uint32_t src_pe = pre_global / rows;
                        uint32_t la = layerIdFromPe_(src_pe);
                        uint32_t lb = layerIdFromPe_(pe);
                        uint32_t key = (la<<8) | lb;
                        if (allowed_layer_edges.find(key) == allowed_layer_edges.end()) { dropped_layer_mask++; continue; }
                    }
                    tmp[pre_global].emplace_back(w, dest_global);
                }
            }
        }
    }

    buildRoutesFromCandidates_(cfg, out, cfg.verify_routing_weights, tmp, rows,
                              /*group_by_pe=*/true, routes_out);
    if (cfg.route_exclude_self_pe || !allow_all_layers) {
        if (cfg.route_filter_warn && out) {
            out->verbose(CALL_INFO, 1, 0,
                "⚠️ 路由过滤已启用: exclude_self_pe=%d, layers_mask='%s' (丢弃: self_pe=%" PRIu64 ", layer_mask=%" PRIu64 ")\n",
                cfg.route_exclude_self_pe ? 1 : 0, cfg.route_layers_mask.c_str(), dropped_self_pe, dropped_layer_mask);
        } else if (out) {
            out->verbose(CALL_INFO, 2, 0,
                "路由过滤: exclude_self_pe=%d, layers_mask='%s' (丢弃: self_pe=%" PRIu64 ", layer_mask=%" PRIu64 ")\n",
                cfg.route_exclude_self_pe ? 1 : 0, cfg.route_layers_mask.c_str(), dropped_self_pe, dropped_layer_mask);
        }
    }
    return !routes_out.empty();
}

} // namespace
#endif

void SpikeCommSubsystem::configure() {
    route_provider_ready_ = false;
}

void SpikeCommSubsystem::bindRuntime(const SpikeCommRuntimeConfig& rt) {
    if (rt.log) log_ = rt.log;
    if (rt.transport) transport_ = rt.transport;
    if (rt.synapse_route) synapse_route_ = rt.synapse_route;
    global_neuron_base_ = rt.global_neuron_base;
    route_provider_ready_ = false;
}

void SpikeCommSubsystem::initRouting() {
    route_provider_ready_ = false;
    if (!synapse_route_) return;
    // Phase3：fanout provider 已下沉至 Synapse/Route；SpikeComm 仅确保其初始化。
    (void)synapse_route_->initRoutes();
    route_provider_ready_ = true;
}

void SpikeCommSubsystem::emitNeuronFire(uint32_t neuron_idx, uint64_t now_cycle) {
    uint32_t source_global = static_cast<uint32_t>(global_neuron_base_ + neuron_idx);
    emitCommon_(source_global, neuron_idx, now_cycle);
}

uint64_t SpikeCommSubsystem::emitNeuronFireBatch(const std::vector<uint32_t>& neuron_indices, uint64_t now_cycle) {
    if (neuron_indices.empty()) return 0;
    uint64_t emitted = 0;
    for (uint32_t neuron_idx : neuron_indices) {
        emitNeuronFire(neuron_idx, now_cycle);
        emitted += 1;
    }
    return emitted;
}

void SpikeCommSubsystem::emitSource(uint32_t source_global, uint32_t source_local, uint64_t now_cycle) {
    emitCommon_(source_global, source_local, now_cycle);
}

void SpikeCommSubsystem::emitCommon_(uint32_t source_global, uint32_t source_local, uint64_t now_cycle) {
    if (!transport_ || !route_provider_ready_ || !synapse_route_) return;
    std::vector<ISynapseRoute::FanoutEntry> fanouts;
    bool applied_gating = false;
    synapse_route_->computeFanout(source_global, source_local, now_cycle, fanouts, applied_gating);
    if (fanouts.empty()) return;

    for (const auto& fe : fanouts) {
        auto* ev = new SpikeEvent(
            source_global,
            fe.dest_global,
            fe.dest_node,
            /*weight=*/1.0,
            now_cycle);
        // 传输层接管生命周期
        transport_->send(ev);
    }
}

void SpikeCommSubsystem::applyGatingDecision(uint32_t src_global,
                                             const std::vector<uint32_t>& dest_pes,
                                             uint64_t current_cycle,
                                             uint64_t ttl_cycles) {
    if (!synapse_route_) return;
    synapse_route_->applyGatingDecision(src_global, dest_pes, current_cycle, ttl_cycles);
}

}} // namespace SST::SnnDL
