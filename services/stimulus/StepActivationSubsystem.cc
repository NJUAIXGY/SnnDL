// -*- c++ -*-
//
// StepActivationSubsystem implementation
//

#include "StepActivationSubsystem.h"

#include <algorithm>
#include <cmath>
#include <cinttypes>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>

#include <sst/core/output.h>
#include <sst/core/statapi/stataccumulator.h>

#include "../../api/GlobalNeuronLayout.h"
#include "INocTransport.h"
#include "SpikeEvent.h"

#include "synapse/route/StepBcsrReachability.h"

namespace SST { namespace SnnDL {

#ifndef STEP_LOG
#define STEP_LOG(lvl, ...) do { if (rt_.log) rt_.log->verbose(CALL_INFO, (lvl), 0, __VA_ARGS__); } while(0)
#endif

void StepActivationSubsystem::configure(const Config& cfg) {
    cfg_ = cfg;
    injection_ready_ = false;
    pending_step_inject_ = false;
    pending_step_seq_ = 0;
    pending_step_ts_ns_ = 0;
    next_cycle_ = 0;
    fixed_seq_ = 1;
    last_injection_seq_ = std::numeric_limits<uint32_t>::max();
    last_reset_seq_ = std::numeric_limits<uint32_t>::max();
    seq_warn_count_ = 0;
    route_diag_done_ = false;
    route_ack_logged_ = false;
    route_warned_ = false;
    step_routes_.clear();
    pre_with_routes_.clear();
    if (cfg_.bcsr_align == 0) cfg_.bcsr_align = 64;
}

void StepActivationSubsystem::bindRuntime(const Runtime& rt) {
    rt_ = rt;

    if (!rt_.layout || !rt_.layout->valid()) {
        if (rt_.log) rt_.log->fatal(CALL_INFO, -1, "StepActivationSubsystem fatal: runtime.layout 为空或无效，无法进行全局ID映射（fail-fast）\n");
        return;
    }

    const uint64_t derived_neurons_per_pe =
        static_cast<uint64_t>(rt_.num_cores) * static_cast<uint64_t>(rt_.neurons_per_core);
    const uint64_t layout_neurons_per_pe = rt_.layout->neuronsPerPE();
    if (layout_neurons_per_pe != derived_neurons_per_pe) {
        if (rt_.log) rt_.log->fatal(
            CALL_INFO, -1,
            "StepActivationSubsystem fatal: neurons_per_pe 口径不一致（fail-fast）：layout=%" PRIu64 " derived(num_cores*neurons_per_core)=%" PRIu64 "\n",
            layout_neurons_per_pe, derived_neurons_per_pe);
        return;
    }
    if (rt_.neurons_per_pe_cfg != 0 && static_cast<uint64_t>(rt_.neurons_per_pe_cfg) != layout_neurons_per_pe) {
        if (rt_.log) rt_.log->fatal(
            CALL_INFO, -1,
            "StepActivationSubsystem fatal: neurons_per_pe_cfg(%u) 与 layout.neuronsPerPE(%" PRIu64 ") 不一致（fail-fast）\n",
            rt_.neurons_per_pe_cfg, layout_neurons_per_pe);
        return;
    }
    const uint64_t expected_base = rt_.layout->globalBaseOfNode(static_cast<uint32_t>(rt_.node_id));
    if (rt_.global_neuron_base != expected_base) {
        if (rt_.log) rt_.log->fatal(
            CALL_INFO, -1,
            "StepActivationSubsystem fatal: global_neuron_base(0x%" PRIx64 ") 与 layout.base(node=%d)=0x%" PRIx64 " 不一致（fail-fast）\n",
            rt_.global_neuron_base, rt_.node_id, expected_base);
        return;
    }

    if (cfg_.trigger_core < -1 || cfg_.trigger_core >= rt_.num_cores) {
        STEP_LOG(1, "⚠️ step_activation_trigger_core=%d 超出范围[-1,%d)，回退到0\n",
                 cfg_.trigger_core, rt_.num_cores);
        cfg_.trigger_core = 0;
    }
}

void StepActivationSubsystem::bindStats(const Stats& st) {
    st_ = st;
}

void StepActivationSubsystem::initBcsrReachabilityIfEnabled() {
    if (!(cfg_.enable && cfg_.use_bcsr_routes)) return;
    if (!loadBcsrReachability_()) {
        STEP_LOG(1, "⚠️ step_activation_use_bcsr_routes=1 但 BCSR 索引加载失败，回退到均匀采样\n");
        cfg_.use_bcsr_routes = false;
    }
}

void StepActivationSubsystem::tick(uint64_t current_cycle) {
    if (!cfg_.enable) return;

    // legacy: BeginGather 触发（当 period==0 时保留原始行为）
    if (cfg_.period_cycles == 0 && pending_step_inject_ && injection_ready_) {
        injectStepActivations_(pending_step_seq_, current_cycle);
        last_injection_seq_ = pending_step_seq_;
        pending_step_inject_ = false;
    }

    // 固定周期触发（去耦 BeginGather 时基漂移）
    if (cfg_.period_cycles > 0 && injection_ready_) {
        if (next_cycle_ == 0) {
            next_cycle_ = current_cycle;
            fixed_seq_ = 1;
        }
        if (current_cycle >= next_cycle_) {
            injectStepActivations_(fixed_seq_, current_cycle);
            ++fixed_seq_;
            next_cycle_ += cfg_.period_cycles;
        }
    }
}

void StepActivationSubsystem::onBeginGather(uint32_t seq, uint64_t ts_ns, int core_id) {
    if (!cfg_.enable) return;
    if (cfg_.period_cycles != 0) return;

    const bool core_ok = (cfg_.trigger_core < 0) ||
                         (core_id < 0) ||
                         (core_id == cfg_.trigger_core);
    if (!core_ok) return;

    const bool first_inject = (last_injection_seq_ == std::numeric_limits<uint32_t>::max());
    if (first_inject || seq > last_injection_seq_) {
        if (injection_ready_) {
            injectStepActivations_(seq, ts_ns);
            last_injection_seq_ = seq;
        } else {
            pending_step_inject_ = true;
            pending_step_seq_ = seq;
            pending_step_ts_ns_ = ts_ns;
        }
    } else {
        if (rt_.log && seq_warn_count_ < 16) {
            STEP_LOG(1, "[step-warn] non-monotonic BeginGather ignored: node=%d core=%d seq=%u last=%u\n",
                     rt_.node_id, core_id, seq, last_injection_seq_);
            ++seq_warn_count_;
        }
    }
}

void StepActivationSubsystem::onEndScatter(uint32_t seq) {
    if (!cfg_.reset_mem_each_step) return;
    if (last_reset_seq_ == seq) return;
    if (rt_.reset_membranes) rt_.reset_membranes();
    last_reset_seq_ = seq;
}

int StepActivationSubsystem::determineTargetUnit_(uint32_t neuron_id) const {
    if (!rt_.layout || !rt_.layout->valid()) return -1;
    if (!rt_.layout->inGlobalRange(static_cast<uint64_t>(neuron_id))) return -1;
    if (!rt_.layout->isLocalToNode(static_cast<uint64_t>(neuron_id), static_cast<uint32_t>(rt_.node_id))) return -1;
    const uint32_t core = rt_.layout->coreOf(static_cast<uint64_t>(neuron_id));
    return (core < static_cast<uint32_t>(rt_.num_cores)) ? static_cast<int>(core) : -1;
}

void StepActivationSubsystem::injectStepActivations_(uint32_t seq, uint64_t sim_time_ns) {
    if (!rt_.layout || !rt_.layout->valid()) return;
    const uint64_t neurons_per_pe = rt_.layout->neuronsPerPE();
    const int total_neurons = static_cast<int>(neurons_per_pe);
    if (!cfg_.enable || cfg_.fanout == 0 || total_neurons <= 0) return;

    double fraction = cfg_.fraction;
    if (fraction <= 0.0) return;
    if (fraction > 1.0) fraction = 1.0;

    if (st_.invocations) st_.invocations->addData(1);

    const uint64_t local_total = static_cast<uint64_t>(total_neurons);
    const uint64_t max_global = rt_.layout->maxGlobalNeurons();
    const uint64_t diag_cap = (rt_.step_diag_cap_cfg > 0) ? static_cast<uint64_t>(rt_.step_diag_cap_cfg) : 0ULL;
    std::mt19937_64 rng(cfg_.seed ^ (static_cast<uint64_t>(seq) + (static_cast<uint64_t>(rt_.node_id) << 32)));
    std::uniform_int_distribution<uint64_t> post_dist(0, local_total - 1);
    std::bernoulli_distribution pick(fraction);
    const bool activate_all = (fraction >= 0.999999);
    uint64_t spikes_injected = 0;
    uint64_t sources_selected = 0;
    uint64_t spike_attempts = 0;
    uint64_t route_hits = 0;
    uint64_t route_misses = 0;
    uint64_t local_drops = 0;
    bool diag_cap_hit = false;

    const bool step_diag_enabled = (rt_.step_diag_enable_cfg != 0);

    if (step_diag_enabled && !route_diag_done_ && rt_.node_id == 0 && seq <= 1 && cfg_.use_bcsr_routes) {
        uint64_t with_routes = 0;
        uint64_t max_routes = 0;
        uint64_t local_edges = 0;
        uint64_t remote_edges = 0;
        for (size_t i = 0; i < step_routes_.size(); ++i) {
            const auto& v = step_routes_[i];
            if (!v.empty()) {
                ++with_routes;
                if (v.size() > max_routes) max_routes = static_cast<uint64_t>(v.size());
                for (auto post : v) {
                    uint32_t pe_of_post = rt_.layout->nodeOf(static_cast<uint64_t>(post));
                    if (pe_of_post == static_cast<uint32_t>(rt_.node_id)) ++local_edges; else ++remote_edges;
                }
            }
        }
        if (rt_.log) {
            double denom_edges = (local_edges + remote_edges) ? static_cast<double>(local_edges + remote_edges) : 1.0;
            double local_ratio = static_cast<double>(local_edges) / denom_edges;
            double remote_ratio = static_cast<double>(remote_edges) / denom_edges;
            STEP_LOG(0,
                "[step-activation-summary] node=%d routes_nonempty=%" PRIu64 " total_pre=%zu max_routes=%" PRIu64
                " max_global=%" PRIu64 " local=%" PRIu64 " (%.2f) remote=%" PRIu64 " (%.2f)\n",
                rt_.node_id, with_routes, step_routes_.size(), max_routes, max_global,
                local_edges, local_ratio, remote_edges, remote_ratio);
        }
        route_diag_done_ = true;
    }

    const bool single_pe_bcsr = (rt_.total_nodes == 1) && cfg_.use_bcsr_routes && !pre_with_routes_.empty();

    if (single_pe_bcsr) {
        for (uint32_t pre_global : pre_with_routes_) {
            if (max_global > 0 && pre_global >= max_global) continue;
            if (!activate_all && !pick(rng)) continue;
            ++sources_selected;
            const bool use_routes = cfg_.use_bcsr_routes && pre_global < step_routes_.size();
            const auto* routes = use_routes ? &step_routes_[pre_global] : nullptr;
            if (!routes || routes->empty()) continue;

            static uint64_t route_sampled = 0;
            if (step_diag_enabled && use_routes && rt_.node_id == 0 && seq <= 1 && route_sampled < 16) {
                STEP_LOG(1, "[[step-diag-pre]] node=%d seq=%u pre_global=%u routes=%zu\n",
                         rt_.node_id, seq, pre_global, routes->size());
                ++route_sampled;
            }
            for (uint32_t fan = 0; fan < cfg_.fanout; ++fan) {
                ++spike_attempts;
                std::uniform_int_distribution<size_t> route_pick(0, routes->size() - 1);
                uint32_t post_global = (*routes)[route_pick(rng)];
                if (max_global > 0 && static_cast<uint64_t>(post_global) >= max_global) {
                    ++local_drops;
                    continue;
                }
                route_hits++;

                auto* spike = new SpikeEvent(pre_global, post_global, static_cast<uint32_t>(rt_.node_id), 1.0f, sim_time_ns);
                int dst_core = determineTargetUnit_(post_global);
                if (dst_core >= 0) {
                    if (rt_.deliver_to_core) rt_.deliver_to_core(dst_core, spike);
                    else delete spike;
                    spikes_injected++;
                } else {
                    delete spike;
                    ++local_drops;
                }
                if (diag_cap && spikes_injected >= diag_cap) { diag_cap_hit = true; break; }
            }
            if (diag_cap_hit) break;
        }
    } else {
        for (int core = 0; core < rt_.num_cores; ++core) {
            uint64_t base = rt_.global_neuron_base +
                static_cast<uint64_t>(core) * static_cast<uint64_t>(rt_.neurons_per_core);
            for (int n = 0; n < rt_.neurons_per_core; ++n) {
                if (!activate_all && !pick(rng)) continue;
                ++sources_selected;
                uint64_t pre_global_64 = base + static_cast<uint64_t>(n);
                if (max_global > 0 && pre_global_64 >= max_global) continue;
                uint32_t pre_global = static_cast<uint32_t>(pre_global_64);
                const bool use_routes = cfg_.use_bcsr_routes && pre_global < step_routes_.size();
                const auto* routes = use_routes ? &step_routes_[pre_global] : nullptr;

                static uint64_t route_sampled = 0;
                if (step_diag_enabled && use_routes && routes && !routes->empty() &&
                    rt_.node_id == 0 && seq <= 1 && route_sampled < 16) {
                    STEP_LOG(1, "[[step-diag-pre]] node=%d seq=%u pre_global=%u routes=%zu\n",
                             rt_.node_id, seq, pre_global, routes->size());
                    ++route_sampled;
                }

                for (uint32_t fan = 0; fan < cfg_.fanout; ++fan) {
                    ++spike_attempts;
                    uint64_t post_global_64 = static_cast<uint64_t>(rt_.global_neuron_base) + post_dist(rng);
                    if (max_global > 0 && post_global_64 >= max_global) continue;
                    uint32_t post_global = static_cast<uint32_t>(post_global_64);
                    if (use_routes) {
                        if (routes && !routes->empty()) {
                            std::uniform_int_distribution<size_t> route_pick(0, routes->size() - 1);
                            post_global = (*routes)[route_pick(rng)];
                            route_hits++;
                        } else {
                            route_misses++;
                        }
                    }

                    auto* spike = new SpikeEvent(pre_global, post_global, static_cast<uint32_t>(rt_.node_id), 1.0f, sim_time_ns);
                    int dst_core = determineTargetUnit_(post_global);
                    if (dst_core >= 0) {
                        if (rt_.deliver_to_core) rt_.deliver_to_core(dst_core, spike);
                        else delete spike;
                        spikes_injected++;
                    } else {
                        uint32_t dest_node = rt_.layout->nodeOf(static_cast<uint64_t>(post_global));
                        spike->setDestinationNode(dest_node);
                        if (rt_.send_external) rt_.send_external(spike);
                        else delete spike;
                        spikes_injected++;
                    }
                    if (diag_cap && spikes_injected >= diag_cap) { diag_cap_hit = true; break; }
                }
                if (diag_cap_hit) break;
            }
            if (diag_cap_hit) break;
        }
    }

    if (st_.pre_selected && sources_selected) st_.pre_selected->addData(sources_selected);
    if (st_.spike_attempts && spike_attempts) st_.spike_attempts->addData(spike_attempts);
    if (st_.spikes_injected && spikes_injected) st_.spikes_injected->addData(spikes_injected);
    if (st_.route_hits && route_hits) st_.route_hits->addData(route_hits);
    if (st_.route_misses && route_misses) st_.route_misses->addData(route_misses);
    if (st_.local_drops && local_drops) st_.local_drops->addData(local_drops);

    if (rt_.log && rt_.log->getVerboseLevel() >= 1) {
        STEP_LOG(1,
            "[step-activation] seq=%u summary sources=%llu attempts=%llu spikes_ok=%llu route_hits=%llu route_miss=%llu local_drop=%llu fraction=%.4g fanout=%u use_routes=%d\n",
            seq,
            static_cast<unsigned long long>(sources_selected),
            static_cast<unsigned long long>(spike_attempts),
            static_cast<unsigned long long>(spikes_injected),
            static_cast<unsigned long long>(route_hits),
            static_cast<unsigned long long>(route_misses),
            static_cast<unsigned long long>(local_drops),
            fraction,
            cfg_.fanout,
            cfg_.use_bcsr_routes ? 1 : 0);
    }

    if (step_diag_enabled && rt_.node_id == 0 && seq <= 1) {
        STEP_LOG(1, "[[step-diag-stats]] node=%d seq=%u sources=%llu attempts=%llu spikes=%llu hits=%llu miss=%llu cap=%" PRIu64 " cap_hit=%d\n",
                 rt_.node_id, seq,
                 (unsigned long long)sources_selected,
                 (unsigned long long)spike_attempts,
                 (unsigned long long)spikes_injected,
                 (unsigned long long)route_hits,
                 (unsigned long long)route_misses,
                 diag_cap,
                 diag_cap_hit ? 1 : 0);
    }
}

#if 0
// Phase5‑5.3：Stimulus 不再自持 BCSR 解析/offset 推导与文件扫描逻辑。
// 旧实现保留作对照参考（不参与编译），reachability 构建统一委托 synapse/route/StepBcsrReachability。
std::string StepActivationSubsystem::formatBcsrPath_(int pe, int core) const {
    if (cfg_.bcsr_template.empty()) return std::string();
    std::string path = cfg_.bcsr_template;
    auto posp = path.find("{pe");
    if (posp != std::string::npos) {
        auto endp = path.find('}', posp);
        if (endp == std::string::npos) return std::string();
        int widthp = 0;
        auto colonp = path.find(':', posp);
        if (colonp != std::string::npos && colonp < endp) {
            auto spec_endp = path.find_first_of("diu", colonp);
            if (spec_endp != std::string::npos && spec_endp < endp) {
                std::string width_str = path.substr(colonp + 1, spec_endp - colonp - 1);
                widthp = std::atoi(width_str.c_str());
            }
        }
        std::ostringstream ossp;
        if (widthp > 0) ossp << std::setfill('0') << std::setw(widthp);
        ossp << pe;
        path.replace(posp, endp - posp + 1, ossp.str());
    }
    auto pos = path.find("{core");
    if (pos == std::string::npos) return path;
    auto end = path.find('}', pos);
    if (end == std::string::npos) return std::string();
    int width = 0;
    auto colon = path.find(':', pos);
    if (colon != std::string::npos && colon < end) {
        auto spec_end = path.find_first_of("diu", colon);
        if (spec_end != std::string::npos && spec_end < end) {
            std::string width_str = path.substr(colon + 1, spec_end - colon - 1);
            width = std::atoi(width_str.c_str());
        }
    }
    std::ostringstream oss;
    if (width > 0) oss << std::setfill('0') << std::setw(width);
    oss << core;
    path.replace(pos, end - pos + 1, oss.str());
    return path;
}

uint64_t StepActivationSubsystem::alignUp_(uint64_t value, uint64_t align) const {
    if (align == 0) return value;
    uint64_t rem = value % align;
    return rem ? (value + align - rem) : value;
}

bool StepActivationSubsystem::computeBcsrOffsets_(uint32_t n_block_rows, uint32_t total_blocks,
                                                  uint64_t block_bytes,
                                                  uint64_t& rowptr_offset, uint64_t& colidx_offset,
                                                  uint64_t& blockdata_offset, uint64_t& blockids_offset) const {
    const uint64_t align = cfg_.bcsr_align ? cfg_.bcsr_align : 64;
    rowptr_offset = 0;
    colidx_offset = alignUp_(rowptr_offset + (uint64_t)(n_block_rows + 1) * sizeof(uint32_t), align);
    blockdata_offset = alignUp_(colidx_offset + (uint64_t)total_blocks * cfg_.bcsr_idx_bytes, align);
    blockids_offset  = alignUp_(blockdata_offset + (uint64_t)total_blocks * block_bytes, align);
    return true;
}

bool StepActivationSubsystem::checkBcsrOffsets_(uint64_t file_size, uint32_t n_block_rows,
                                                uint32_t total_blocks, uint64_t block_bytes,
                                                uint64_t& rowptr_offset, uint64_t& colidx_offset,
                                                uint64_t& blockdata_offset, uint64_t& blockids_offset) const {
    auto valid = [&](uint64_t off) { return off < file_size; };
    if (!valid(rowptr_offset) || !valid(colidx_offset) ||
        !valid(blockdata_offset) || !valid(blockids_offset)) {
        computeBcsrOffsets_(n_block_rows, total_blocks, block_bytes,
                            rowptr_offset, colidx_offset, blockdata_offset, blockids_offset);
    }
    if (rowptr_offset >= file_size) return false;
    if (colidx_offset >= file_size) return false;
    if (blockdata_offset >= file_size) return false;
    if (blockids_offset >= file_size) return false;
    const uint64_t need_rowptr = (uint64_t)(n_block_rows + 1) * sizeof(uint32_t);
    const uint64_t need_colidx = (uint64_t)total_blocks * cfg_.bcsr_idx_bytes;
    const uint64_t need_block  = (uint64_t)total_blocks * block_bytes;
    if (rowptr_offset + need_rowptr > file_size) return false;
    if (colidx_offset + need_colidx > file_size) return false;
    if (blockdata_offset + need_block > file_size) return false;
    if (blockids_offset + need_block > file_size) return false;
    return true;
}

bool StepActivationSubsystem::buildRoutesFromBcsrFile_(const std::string& path, uint32_t pe_id, uint32_t core_index) {
    const uint32_t rows_per_core = cfg_.bcsr_rows_per_core;
    const uint32_t br = cfg_.bcsr_br ? cfg_.bcsr_br : 16;
    const uint32_t bc = cfg_.bcsr_bc ? cfg_.bcsr_bc : 16;
    const uint32_t n_block_rows = (rows_per_core + br - 1) / br;
    const size_t floats_per_block = static_cast<size_t>(br) * static_cast<size_t>(bc);
    const uint32_t neurons_per_pe = static_cast<uint32_t>(rt_.neurons_per_core) * static_cast<uint32_t>(rt_.num_cores);
    const uint32_t local_pre_begin = (neurons_per_pe > 0) ? static_cast<uint32_t>(rt_.node_id) * neurons_per_pe : 0u;
    const uint32_t local_pre_end = local_pre_begin + neurons_per_pe;
    const uint32_t max_global = (rt_.total_nodes > 0 && neurons_per_pe > 0)
        ? static_cast<uint32_t>(rt_.total_nodes) * neurons_per_pe
        : 0u;

    std::ifstream fin(path, std::ios::binary);
    if (!fin.good()) {
        STEP_LOG(0, "⚠️ 无法读取BCSR文件: %s\n", path.c_str());
        return false;
    }

    fin.seekg(0, std::ios::end);
    const std::streamoff file_size = fin.tellg();
    fin.clear();
    fin.seekg(0, std::ios::beg);
    uint64_t rowptr_off = cfg_.bcsr_rowptr_offset;
    uint64_t colidx_off = cfg_.bcsr_colidx_offset;
    uint64_t blockdata_off = cfg_.bcsr_blockdata_offset;
    uint64_t blockids_off = cfg_.bcsr_blockids_offset;
    const uint64_t bytes_per_block_data = floats_per_block * sizeof(float);
    const uint64_t bytes_per_block_ids  = floats_per_block * sizeof(uint32_t);
    const uint64_t avail_rowptr_bytes = (rowptr_off < (uint64_t)file_size) ? ((uint64_t)file_size - rowptr_off) : 0ULL;
    const uint64_t avail_colidx_bytes = (colidx_off < blockdata_off && blockdata_off <= (uint64_t)file_size)
        ? (blockdata_off - colidx_off) : 0ULL;
    const uint64_t avail_blockdata_bytes = (blockdata_off < (uint64_t)file_size) ? ((uint64_t)file_size - blockdata_off) : 0ULL;
    const uint64_t avail_blockids_bytes  = (blockids_off  < (uint64_t)file_size) ? ((uint64_t)file_size - blockids_off)  : 0ULL;

    fin.seekg(cfg_.bcsr_rowptr_offset, std::ios::beg);
    const uint64_t want_rowptr_bytes = (uint64_t)(n_block_rows + 1) * sizeof(uint32_t);
    const uint64_t take_rowptr_bytes = std::min<uint64_t>(want_rowptr_bytes, avail_rowptr_bytes);
    const uint32_t rowptr_elems = static_cast<uint32_t>(take_rowptr_bytes / sizeof(uint32_t));
    if (rowptr_elems < 2) {
        STEP_LOG(0, "⚠️ rowptr区域不足: have=%" PRIu64 " need=%" PRIu64 " file=%lld\n",
                 (uint64_t)take_rowptr_bytes, (uint64_t)want_rowptr_bytes, (long long)file_size);
        return false;
    }
    std::vector<uint32_t> rowptr(rowptr_elems, 0);
    fin.read(reinterpret_cast<char*>(rowptr.data()), rowptr_elems * sizeof(uint32_t));
    if (!fin.good()) {
        STEP_LOG(0, "⚠️ 读取rowptr失败: %s\n", path.c_str());
        return false;
    }
    const uint32_t total_blocks_rowptr = rowptr.back();
    const uint64_t max_blocks_colidx = (cfg_.bcsr_idx_bytes > 0)
        ? (avail_colidx_bytes / (uint64_t)cfg_.bcsr_idx_bytes)
        : 0ULL;
    const uint64_t max_blocks_data = (bytes_per_block_data > 0) ? (avail_blockdata_bytes / bytes_per_block_data) : 0ULL;
    const uint64_t max_blocks_ids  = (bytes_per_block_ids  > 0) ? (avail_blockids_bytes  / bytes_per_block_ids ) : 0ULL;
    const uint64_t max_blocks_by_file = std::min(std::min(max_blocks_data, max_blocks_ids), max_blocks_colidx);
    const uint32_t total_blocks = static_cast<uint32_t>(std::min<uint64_t>(total_blocks_rowptr, max_blocks_by_file));
    if (total_blocks == 0) {
        STEP_LOG(0, "⚠️ total_blocks=0 (rowptr=%u, by_file=%" PRIu64 ") path=%s\n",
                 total_blocks_rowptr, max_blocks_by_file, path.c_str());
        return false;
    }
    if (!checkBcsrOffsets_((uint64_t)file_size, n_block_rows, total_blocks,
                           bytes_per_block_data,
                           rowptr_off, colidx_off, blockdata_off, blockids_off)) {
        if (!route_warned_) {
            STEP_LOG(0,
                "⚠️ BCSR offsets/size mismatch: node=%d path=%s fsize=%lld blocks(rowptr)=%u br=%u bc=%u idxB=%u valB=%u (recomputed: rowptr=%llu colidx=%llu blockdata=%llu blockids=%llu)\n",
                rt_.node_id, path.c_str(), (long long)file_size, total_blocks_rowptr, br, bc,
                cfg_.bcsr_idx_bytes, cfg_.bcsr_val_bytes,
                (unsigned long long)rowptr_off, (unsigned long long)colidx_off,
                (unsigned long long)blockdata_off, (unsigned long long)blockids_off);
        }
        return false;
    }

    std::vector<uint32_t> block_cols(total_blocks, 0);
    fin.seekg(cfg_.bcsr_colidx_offset, std::ios::beg);
    if (cfg_.bcsr_idx_bytes == 2) {
        std::vector<uint16_t> tmp(total_blocks, 0);
        fin.read(reinterpret_cast<char*>(tmp.data()), tmp.size() * sizeof(uint16_t));
        if (!fin.good()) {
            STEP_LOG(0, "⚠️ 读取colidx(2B)失败: %s blocks=%u\n", path.c_str(), total_blocks);
            return false;
        }
        for (uint32_t i = 0; i < total_blocks; ++i) block_cols[i] = tmp[i];
    } else {
        fin.read(reinterpret_cast<char*>(block_cols.data()), block_cols.size() * sizeof(uint32_t));
        if (!fin.good()) {
            STEP_LOG(0, "⚠️ 读取colidx(4B)失败: %s blocks=%u\n", path.c_str(), total_blocks);
            return false;
        }
    }

    std::ifstream fdata(path, std::ios::binary);
    std::ifstream fids(path, std::ios::binary);
    fdata.seekg(cfg_.bcsr_blockdata_offset, std::ios::beg);
    fids.seekg(cfg_.bcsr_blockids_offset, std::ios::beg);
    std::vector<float> blockdata(floats_per_block, 0.0f);
    std::vector<uint32_t> blockids(floats_per_block, 0u);

    uint32_t block_index = 0;
    uint64_t skipped_blocks = 0;
    auto flush_skips = [&](uint64_t n) -> bool {
        if (n == 0) return true;
        const uint64_t data_skip = n * bytes_per_block_data;
        const uint64_t ids_skip  = n * bytes_per_block_ids;
        fdata.seekg(static_cast<std::streamoff>(data_skip), std::ios::cur);
        fids.seekg(static_cast<std::streamoff>(ids_skip), std::ios::cur);
        if (!fdata.good() || !fids.good()) {
            STEP_LOG(0, "⚠️ BCSR seek skip failed: %s skip_blocks=%" PRIu64 "\n", path.c_str(), (uint64_t)n);
            return false;
        }
        return true;
    };

    for (uint32_t block_row = 0; block_row < n_block_rows; ++block_row) {
        if ((size_t)block_row + 1 >= rowptr.size()) break;
        uint32_t begin = rowptr[block_row];
        uint32_t end = rowptr[block_row + 1];
        if (begin >= total_blocks) break;
        if (end > total_blocks) end = total_blocks;
        for (uint32_t idx = begin; idx < end; ++idx, ++block_index) {
            if (block_index >= total_blocks) break;
            uint32_t block_col = block_cols[idx];
            const uint64_t pre_block_begin = static_cast<uint64_t>(block_col) * static_cast<uint64_t>(bc);
            const uint64_t pre_block_end = pre_block_begin + static_cast<uint64_t>(bc);
            const bool overlap_local = (neurons_per_pe > 0) &&
                !(pre_block_end <= (uint64_t)local_pre_begin || pre_block_begin >= (uint64_t)local_pre_end);
            if (!overlap_local) { ++skipped_blocks; continue; }
            if (!flush_skips(skipped_blocks)) return false;
            skipped_blocks = 0;

            fdata.read(reinterpret_cast<char*>(blockdata.data()), blockdata.size() * sizeof(float));
            fids.read(reinterpret_cast<char*>(blockids.data()), blockids.size() * sizeof(uint32_t));
            if (!fdata.good() || !fids.good()) {
                STEP_LOG(0, "⚠️ 读取block数据失败: %s (block_index=%u/%u, fsize=%lld)\n",
                         path.c_str(), block_index, total_blocks, (long long)file_size);
                return false;
            }

            if (block_col >= std::numeric_limits<uint32_t>::max() / bc) continue;
            const uint32_t pre_base = static_cast<uint32_t>(pre_block_begin);
            for (uint32_t rr = 0; rr < br; ++rr) {
                uint32_t post_local = block_row * br + rr;
                if (post_local >= rows_per_core) continue;
                const uint64_t post_global_64 =
                    static_cast<uint64_t>(pe_id) * static_cast<uint64_t>(neurons_per_pe) +
                    static_cast<uint64_t>(core_index) * static_cast<uint64_t>(rows_per_core) +
                    static_cast<uint64_t>(post_local);
                if (max_global > 0u && post_global_64 >= static_cast<uint64_t>(max_global)) continue;
                const uint32_t post_global = static_cast<uint32_t>(post_global_64);

                for (uint32_t cc = 0; cc < bc; ++cc) {
                    size_t off = static_cast<size_t>(rr) * bc + cc;
                    if (blockids[off] == 0xFFFFFFFFu) continue;
                    float weight = blockdata[off];
                    if (std::fabs(weight) <= cfg_.bcsr_weight_epsilon) continue;
                    uint32_t pre_global = pre_base + cc;
                    if (pre_global < local_pre_begin || pre_global >= local_pre_end) continue;
                    if (pre_global >= step_routes_.size()) continue;
                    step_routes_[pre_global].push_back(post_global);
                }
            }
        }
    }

    if (rt_.log && cfg_.log_enable && rt_.log->getVerboseLevel() >= 1) {
        uint64_t edges = 0;
        for (auto& v : step_routes_) edges += (uint64_t)v.size();
        STEP_LOG(1,
            "[step-activation] BCSR reachability loaded: pe=%u core=%u rows=%u br=%u bc=%u total_blocks(rowptr)=%u used=%u edges=%llu\n",
            pe_id, core_index, rows_per_core, br, bc, total_blocks_rowptr, total_blocks,
            static_cast<unsigned long long>(edges));
    }
    return true;
}
#endif // 0

bool StepActivationSubsystem::loadBcsrReachability_() {
    if (rt_.log && cfg_.log_enable) {
        STEP_LOG(0,
            "[step-activation] node=%d loadBcsrReachability enable=%d use_bcsr=%d template=%s build_local_only=%d rows_per_core=%u br=%u bc=%u\n",
            rt_.node_id, (int)cfg_.enable, (int)cfg_.use_bcsr_routes,
            cfg_.bcsr_template.c_str(), (int)cfg_.build_local_only,
            cfg_.bcsr_rows_per_core, cfg_.bcsr_br, cfg_.bcsr_bc);
    }
    if (cfg_.bcsr_template.empty()) {
        if (rt_.log && cfg_.log_enable) {
            STEP_LOG(0, "⚠️ 未提供 step_activation_bcsr_template，无法加载BCSR索引\n");
        }
        return false;
    }

    const uint64_t total_neurons = static_cast<uint64_t>(rt_.num_cores) * static_cast<uint64_t>(rt_.neurons_per_core);
    const uint64_t total_pre = rt_.global_neuron_base + total_neurons;
    step_routes_.assign(static_cast<size_t>(total_pre), {});

    StepBcsrReachabilityConfig rcfg{};
    rcfg.bcsr_template = cfg_.bcsr_template;
    rcfg.build_local_only = cfg_.build_local_only;
    rcfg.log_enable = cfg_.log_enable;
    rcfg.bcsr_rows_per_core = (cfg_.bcsr_rows_per_core > 0)
        ? cfg_.bcsr_rows_per_core
        : static_cast<uint32_t>(rt_.neurons_per_core);
    rcfg.bcsr_br = cfg_.bcsr_br;
    rcfg.bcsr_bc = cfg_.bcsr_bc;
    rcfg.bcsr_idx_bytes = cfg_.bcsr_idx_bytes;
    rcfg.bcsr_val_bytes = cfg_.bcsr_val_bytes;
    rcfg.bcsr_rowptr_offset = cfg_.bcsr_rowptr_offset;
    rcfg.bcsr_colidx_offset = cfg_.bcsr_colidx_offset;
    rcfg.bcsr_blockdata_offset = cfg_.bcsr_blockdata_offset;
    rcfg.bcsr_blockids_offset = cfg_.bcsr_blockids_offset;
    rcfg.bcsr_weight_epsilon = cfg_.bcsr_weight_epsilon;

    StepBcsrReachabilityRuntime rrt{};
    rrt.log = rt_.log;
    rrt.node_id = static_cast<uint32_t>(rt_.node_id);
    rrt.total_nodes = static_cast<uint32_t>(rt_.total_nodes > 0 ? rt_.total_nodes : 1);
    rrt.num_cores = static_cast<uint32_t>(rt_.num_cores);
    rrt.neurons_per_core = static_cast<uint32_t>(rt_.neurons_per_core);
    rrt.neurons_per_pe_cfg = rt_.neurons_per_pe_cfg;
    rrt.global_neuron_base = rt_.global_neuron_base;

    ISynapseRoute::RouteMap routes_map;
    std::vector<uint32_t> pre_with_routes;
    bool success = buildStepBcsrReachabilityRoutes(rcfg, rrt, routes_map, pre_with_routes);
    if (success) {
        for (auto& kv : routes_map) {
            const uint32_t pre = kv.first;
            if (pre < step_routes_.size()) step_routes_[pre] = std::move(kv.second);
        }
        if (rt_.total_nodes == 1) pre_with_routes_ = std::move(pre_with_routes);
        else pre_with_routes_.clear();
    }

    if (success) {
        size_t with_routes = 0;
        for (const auto& vec : step_routes_) if (!vec.empty()) ++with_routes;
        if (cfg_.log_enable && !route_ack_logged_) {
            STEP_LOG(1, "[step-activation] BCSR reachability loaded: pre_with_routes=%zu total_pre=%zu\n",
                     with_routes, step_routes_.size());
            computeRouteRatios_();
            route_ack_logged_ = true;
        }
        if (rt_.log && cfg_.log_enable) {
            STEP_LOG(0, "[step-activation] node=%d loadBcsrReachability success route_vectors=%zu\n",
                     rt_.node_id, step_routes_.size());
        }
        if (rt_.total_nodes == 1 && rt_.log && cfg_.log_enable) {
            STEP_LOG(1, "[step-activation] node=%d single-PE pre_with_routes_list size=%zu\n",
                     rt_.node_id, pre_with_routes_.size());
        }
    } else {
        pre_with_routes_.clear();
        if (cfg_.log_enable && !route_warned_) {
            STEP_LOG(0, "⚠️ step_activation BCSR route build failed, routes cleared; using fallback sampling\n");
            route_warned_ = true;
        }
        step_routes_.clear();
        if (rt_.log && cfg_.log_enable) {
            STEP_LOG(0, "[step-activation] node=%d loadBcsrReachability FAILED\n", rt_.node_id);
        }
    }
    return success;
}

void StepActivationSubsystem::computeRouteRatios_() const {
    uint64_t local_edges = 0, remote_edges = 0, total_edges = 0;
    const uint32_t neurons_per_pe = static_cast<uint32_t>(rt_.neurons_per_core) * static_cast<uint32_t>(rt_.num_cores);
    if (neurons_per_pe > 0) {
        for (size_t pre = 0; pre < step_routes_.size(); ++pre) {
            const auto& vec = step_routes_[pre];
            total_edges += static_cast<uint64_t>(vec.size());
            for (auto post_global : vec) {
                uint32_t pe_of_post = static_cast<uint32_t>(post_global / neurons_per_pe);
                if (pe_of_post == static_cast<uint32_t>(rt_.node_id)) ++local_edges;
                else ++remote_edges;
            }
        }
    }
    double local_ratio = (total_edges ? (double)local_edges / (double)total_edges : 0.0);
    double remote_ratio = (total_edges ? (double)remote_edges / (double)total_edges : 0.0);
    if (rt_.log && cfg_.log_enable) {
        STEP_LOG(1,
            "[step-activation] route_ratio: node=%d local_edges=%" PRIu64 " remote_edges=%" PRIu64
            " total=%" PRIu64 " local_ratio=%.4f remote_ratio=%.4f\n",
            rt_.node_id, local_edges, remote_edges, total_edges, local_ratio, remote_ratio);
    }
}

}} // namespace SST::SnnDL
