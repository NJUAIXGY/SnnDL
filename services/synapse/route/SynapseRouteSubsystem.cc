// -*- c++ -*-
//
// SynapseRouteSubsystem: 权重驱动路由构建 + 进程级共享缓存
//

#include "SynapseRouteSubsystem.h"

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

#include <sst/core/output.h>
#include <sst/core/statapi/stataccumulator.h>

#include "synapse/route/BcsrRouteBuilder.h"

namespace SST { namespace SnnDL {

// Process-wide shared route cache (avoid per-core route table duplication).
std::mutex SynapseRouteSubsystem::s_route_cache_mtx_;
std::unordered_map<std::string, std::weak_ptr<const SynapseRouteSubsystem::RouteMap>> SynapseRouteSubsystem::s_route_cache_;

namespace {

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

std::string buildRouteCacheKey_(const SynapseRouteBuildConfig& cfg) {
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

void buildRoutesFromCandidates_(const SynapseRouteBuildConfig& cfg,
                               Output* out,
                               bool verify_routing_weights,
                               const std::unordered_map<uint32_t, std::vector<std::pair<float,uint32_t>>>& tmp,
                               uint32_t rows,
                               bool group_by_pe,
                               SynapseRouteSubsystem::RouteMap& routes_out) {
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

bool buildRoutesFromEdgesCSV_(const SynapseRouteBuildConfig& cfg,
                             Output* out,
                             const std::unordered_set<uint32_t>& allowed_layer_edges,
                             bool allow_all_layers,
                             SynapseRouteSubsystem::RouteMap& routes_out) {
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

bool buildWeightDrivenRoutesFromBcsr_(const SynapseRouteBuildConfig& cfg,
                                     Output* out,
                                     SynapseRouteSubsystem::RouteMap& routes_out) {
    routes_out.clear();
    const uint32_t cores_per_pe = (cfg.cores_per_pe > 0) ? cfg.cores_per_pe : 1u;
    uint32_t rows_hint = (cores_per_pe > 0) ? static_cast<uint32_t>((cfg.rows + cores_per_pe - 1) / cores_per_pe) : cfg.rows;
    if (rows_hint == 0) rows_hint = cfg.rows;
    bool ok = true;
    for (uint32_t pe = 0; pe < cfg.total_nodes; ++pe) {
        for (uint32_t core = 0; core < cores_per_pe; ++core) {
            std::string path = resolveBcsrTemplate(cfg.weights_template, pe, static_cast<int>(core));
            if (path.empty()) { ok = false; break; }
            if (!appendRoutesFromBcsrFile(cfg, out, path, pe, static_cast<int>(core),
                                          rows_hint, routes_out, BcsrAppendOptions{})) {
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

bool buildWeightDrivenRoutesDense_(const SynapseRouteBuildConfig& cfg,
                                  Output* out,
                                  const std::unordered_set<uint32_t>& allowed_layer_edges,
                                  bool allow_all_layers,
                                  SynapseRouteSubsystem::RouteMap& routes_out) {
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
                    tmp[pre_global].emplace_back(std::fabs(w), dest_global);
                }
            }
        }
    }
    buildRoutesFromCandidates_(cfg, out, cfg.verify_routing_weights, tmp, rows, /*group_by_pe=*/true, routes_out);
    if ((cfg.route_exclude_self_pe || !allow_all_layers) && cfg.route_filter_warn && out) {
        out->verbose(CALL_INFO, 1, 0,
            "⚠️ 路由过滤启用: exclude_self_pe=%d, layers_mask='%s' (丢弃: self_pe=%" PRIu64 ", layer_mask=%" PRIu64 ")\n",
            cfg.route_exclude_self_pe ? 1 : 0, cfg.route_layers_mask.c_str(), dropped_self_pe, dropped_layer_mask);
    }
    return !routes_out.empty();
}

} // namespace

void SynapseRouteSubsystem::configure(const SynapseRouteBuildConfig& cfg) {
    cfg_ = cfg;
    routing_weight_driven_active_ = false;
    routes_shared_.reset();
    routes_local_fallback_.clear();
    route_summary_logged_ = false;
    fanout_provider_ready_ = false;
    gating_cache_.clear();
}

void SynapseRouteSubsystem::configureGating(bool gating_event_mode,
                                            uint64_t gating_ttl_cycles,
                                            bool gating_scope_inputs_only) {
    gating_event_mode_ = gating_event_mode;
    gating_ttl_cycles_ = gating_ttl_cycles;
    gating_scope_inputs_only_ = gating_scope_inputs_only;
}

void SynapseRouteSubsystem::bindRuntime(Output* log,
                                        uint32_t node_id,
                                        uint32_t core_id,
                                        uint32_t num_neurons,
                                        uint32_t neurons_per_pe_cfg,
                                        SST::Statistics::Statistic<uint64_t>* stat_routes_entries) {
    if (log) log_ = log;
    node_id_ = node_id;
    core_id_ = core_id;
    num_neurons_ = num_neurons;
    neurons_per_pe_cfg_ = neurons_per_pe_cfg;
    if (stat_routes_entries) stat_routes_entries_ = stat_routes_entries;
}

void SynapseRouteSubsystem::bindFanoutStat(SST::Statistics::Statistic<uint64_t>* stat_fanout_per_spike) {
    if (stat_fanout_per_spike) stat_fanout_per_spike_ = stat_fanout_per_spike;
}

bool SynapseRouteSubsystem::initRoutes() {
    routes_shared_.reset();
    routes_local_fallback_.clear();
    fanout_provider_ready_ = false;

    routing_weight_driven_active_ = cfg_.routing_weight_driven;
    if (!routing_weight_driven_active_) {
        // fixed 路由模式：无需构建 routes，但仍需配置 fanout provider。
        configureFanoutProvider_();
        return false;
    }

    const std::string cache_key = buildRouteCacheKey_(cfg_);
    bool ok = false;
    bool hit_cache = false;

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
            hit_cache = true;
            uint64_t total_entries = 0;
            for (auto& kv : *routes_shared_) total_entries += (uint64_t)kv.second.size();
            if (stat_routes_entries_) stat_routes_entries_->addData(total_entries);
            if (log_) {
                log_->verbose(CALL_INFO, 3, 0,
                    "🔁 命中共享路由缓存: 核心%u, 源条目=%zu, 总目的=%" PRIu64 "\n",
                    core_id_, routes_shared_->size(), total_entries);
            }
        }
    }

    if (!ok) {
        std::unordered_set<uint32_t> allowed_layer_edges;
        bool allow_all_layers = true;
        parseLayerMask_(cfg_.route_layers_mask, allowed_layer_edges, allow_all_layers);

        RouteMap built_routes;
        if (cfg_.mapping_mode == "edges_csv" && !cfg_.mapping_edges_file.empty()) {
            ok = buildRoutesFromEdgesCSV_(cfg_, log_, allowed_layer_edges, allow_all_layers, built_routes);
        } else {
            ok = buildWeightDrivenRoutesDense_(cfg_, log_, allowed_layer_edges, allow_all_layers, built_routes);
        }

        if (!ok) {
            if (log_) {
                log_->verbose(CALL_INFO, 1, 0,
                    "⚠️ 核心%u权重驱动路由构建失败，将回退fixed路由\n",
                    core_id_);
            }
            routing_weight_driven_active_ = false;
        } else {
            auto shared = std::make_shared<RouteMap>(std::move(built_routes));
            routes_shared_ = shared;
            uint64_t total_entries = 0;
            for (auto& kv : *routes_shared_) total_entries += (uint64_t)kv.second.size();
            if (stat_routes_entries_) stat_routes_entries_->addData(total_entries);

            // One-shot local/remote ratio summary (per core)
            if (cfg_.route_summary_enable && log_ && !route_summary_logged_) {
                route_summary_logged_ = true;
                uint64_t local_edges = 0, remote_edges = 0;
                const uint32_t denom = (neurons_per_pe_cfg_ > 0) ? neurons_per_pe_cfg_ : num_neurons_;
                for (const auto& kv : *routes_shared_) {
                    for (auto post_global : kv.second) {
                        uint32_t pe_of_post = (denom ? (post_global / denom) : 0);
                        if (pe_of_post == node_id_) ++local_edges; else ++remote_edges;
                    }
                }
                double local_ratio = (total_entries > 0) ? (double)local_edges / (double)total_entries : 0.0;
                double remote_ratio = (total_entries > 0) ? (double)remote_edges / (double)total_entries : 0.0;
                log_->verbose(CALL_INFO, 0, 0,
                    "[route-summary] node=%u core=%u entries=%zu total=%" PRIu64
                    " local=%" PRIu64 " (%.2f) remote=%" PRIu64 " (%.2f)\n",
                    node_id_, core_id_, routes_shared_->size(), total_entries,
                    local_edges, local_ratio, remote_edges, remote_ratio);
            }

            if (!cache_key.empty()) {
                std::lock_guard<std::mutex> g(s_route_cache_mtx_);
                s_route_cache_[cache_key] = shared;
            }
        }
    }

    if (cfg_.route_summary_enable && log_) {
        logRoutingSummary_("init", routing_weight_driven_active_ ? (hit_cache ? "active(cache)" : "active") : "fallback_fixed");
    }

    configureFanoutProvider_();
    return routing_weight_driven_active_;
}

void SynapseRouteSubsystem::computeFanout(uint32_t source_global, uint32_t neuron_idx,
                                          uint64_t now_cycles,
                                          std::vector<FanoutEntry>& out_entries,
                                          bool& applied_gating) const {
    out_entries.clear();
    applied_gating = false;
    if (!fanout_provider_ready_) return;
    fanout_provider_.computeFanout(source_global, neuron_idx, now_cycles, out_entries, applied_gating);
}

void SynapseRouteSubsystem::applyGatingDecision(uint32_t src_global,
                                                const std::vector<uint32_t>& dest_pes,
                                                uint64_t current_cycle,
                                                uint64_t ttl_cycles) {
    if (!gating_event_mode_) return;
    GatingEntry e;
    e.dest_pes = dest_pes;
    e.expire_cycle = current_cycle + (ttl_cycles ? ttl_cycles : gating_ttl_cycles_);
    gating_cache_[src_global] = std::move(e);
    if (log_) {
        log_->verbose(CALL_INFO, 3, 0,
            "📥 应用门控: src_g=%u, k=%zu, expire=%" PRIu64 "\n",
            src_global, dest_pes.size(), gating_cache_[src_global].expire_cycle);
    }
}

void SynapseRouteSubsystem::configureFanoutProvider_() {
    SnnRouteProvider::Config cfg{};
    cfg.routing_weight_driven = routing_weight_driven_active_;
    cfg.log_weight_details = cfg_.log_weight_details;
    cfg.num_neurons = num_neurons_;
    cfg.neurons_per_pe_cfg = neurons_per_pe_cfg_;
    cfg.node_id = node_id_;
    cfg.gating_event_mode = gating_event_mode_;
    cfg.gating_scope_inputs_only = gating_scope_inputs_only_;
    cfg.gating_cache = &gating_cache_;
    cfg.out = log_;
    cfg.stat_fanout = stat_fanout_per_spike_;

    fanout_provider_.configure(cfg, routes_shared_, &routes_local_fallback_);
    fanout_provider_ready_ = true;
}

void SynapseRouteSubsystem::logRoutingSummary_(const char* phase, const char* reason) const {
    if (!(cfg_.route_summary_enable && log_)) return;
    const size_t shared_entries = routes_shared_ ? routes_shared_->size() : 0;
    const size_t local_entries = shared_entries ? 0 : routes_local_fallback_.size();
    log_->verbose(CALL_INFO, 0, 0,
        "[route-summary][%s] node=%u core=%u routes_shared=%zu routes_local=%zu %s\n",
        phase ? phase : "-", node_id_, core_id_, shared_entries, local_entries,
        reason ? reason : "");
}

}} // namespace SST::SnnDL
