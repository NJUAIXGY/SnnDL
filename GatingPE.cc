// -*- c++ -*-
#include "GatingPE.h"
#include <sstream>
#include <random>
#include "SnnNIC.h"
#include "SnnInterface.h"
#include "GatingDecisionEvent.h"

using namespace SST;
using namespace SST::SnnDL;

GatingPE::GatingPE(ComponentId_t id, Params& params)
    : Component(id)
{
    total_nodes_  = params.find<uint32_t>("total_nodes", 16);
    rows_per_pe_  = params.find<uint32_t>("rows_per_pe", 16);
    top_k_        = params.find<uint32_t>("top_k", 4);
    weight_value_ = params.find<float>("weight_value", 1.0f);
    transitions_  = params.find<std::string>("transitions", "0-3:4-7,4-7:8-11,8-11:12-15");
    edges_path_   = params.find<std::string>("edges_output_file", "");
    csv_header_   = params.find<int>("csv_header", 1) != 0;
    selection_    = params.find<std::string>("selection", "round_robin");
    seed_         = params.find<uint32_t>("seed", 42);

    int verbose = params.find<int>("verbose", 0);
    out_ = new Output("GatingPE[@p:@l]: ", verbose, 0, Output::STDOUT);

    // stats
    stat_tokens_ = registerStatistic<uint64_t>("gating_tokens");
    stat_edges_  = registerStatistic<uint64_t>("edges_generated");
    stat_evt_sent_ = registerStatistic<uint64_t>("gating_events_sent");

    // No clock required
}

GatingPE::~GatingPE() {
    if (out_) { delete out_; out_ = nullptr; }
}

void GatingPE::init(unsigned int phase) {
    if (phase == 0) {
        if (edges_path_.empty()) {
            out_->verbose(CALL_INFO, 1, 0, "⚠️ 未设置edges_output_file，跳过CSV生成\n");
        } else {
            writeCSV();
        }
        // 加载NIC（用于控制事件发射）
        Params p;
        p.insert("node_id", std::to_string(nic_node_id_));
        p.insert("link_bw", nic_link_bw_);
        p.insert("input_buf_size", nic_in_buf_);
        p.insert("output_buf_size", nic_out_buf_);
        p.insert("virtual_channels", std::to_string(nic_vns_));
        p.insert("vn_control", std::to_string(nic_vn_control_));
        p.insert("total_nodes", std::to_string(total_nodes_));
        nic_ = loadAnonymousSubComponent<SnnInterface>("SnnDL.SnnNIC", "network_interface", 0,
            ComponentInfo::SHARE_PORTS | ComponentInfo::INSERT_STATS, p);
        if (!nic_) {
            out_->verbose(CALL_INFO, 1, 0, "⚠️ 未能加载SnnNIC，控制事件将不可用\n");
        }
        // 注册定时器
        if (emit_period_ns_ > 0 && nic_) {
            std::string per = std::to_string(emit_period_ns_) + "ns";
            registerClock(per, new SST::Clock::Handler2<GatingPE, &GatingPE::onEmitTick>(this));
        }
    }
}

void GatingPE::setup() { /* no-op */ }
void GatingPE::finish() { /* no-op */ }

bool GatingPE::parseTransitions(const std::string& s, std::vector<Transition>& out) {
    // format: a-b:c-d, e-f:g-h, ... (inclusive bounds)
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        size_t colon = tok.find(':');
        if (colon == std::string::npos) return false;
        std::string left = tok.substr(0, colon);
        std::string right = tok.substr(colon+1);
        auto parseRange = [](const std::string& r)->Range{
            size_t dash = r.find('-');
            Range R{};
            if (dash == std::string::npos) { R.a = R.b = (uint32_t) std::stoul(r); }
            else { R.a = (uint32_t) std::stoul(r.substr(0,dash)); R.b = (uint32_t) std::stoul(r.substr(dash+1)); }
            return R;
        };
        Transition tr; tr.src = parseRange(left); tr.dst = parseRange(right);
        out.push_back(tr);
    }
    return !out.empty();
}

void GatingPE::writeCSV() {
    std::vector<Transition> trs; trs.reserve(8);
    if (!parseTransitions(transitions_, trs)) {
        out_->verbose(CALL_INFO, 1, 0, "⚠️ transitions参数解析失败: %s\n", transitions_.c_str());
        return;
    }
    std::ofstream fout(edges_path_);
    if (!fout.good()) {
        out_->verbose(CALL_INFO, 1, 0, "⚠️ 无法打开CSV输出路径: %s\n", edges_path_.c_str());
        return;
    }
    if (csv_header_) fout << "src_global,dst_global,weight\n";

    std::mt19937 rng(seed_);

    uint64_t edges = 0, tokens = 0;
    for (const auto& tr : trs) {
        if (tr.src.b < tr.src.a || tr.dst.b < tr.dst.a) continue;
        std::vector<uint32_t> dst_pes;
        for (uint32_t p = tr.dst.a; p <= tr.dst.b; ++p) dst_pes.push_back(p);
        const uint32_t cnt = (uint32_t) dst_pes.size();
        if (cnt == 0) continue;
        for (uint32_t src_pe = tr.src.a; src_pe <= tr.src.b; ++src_pe) {
            uint32_t off = (selection_ == "round_robin") ? (src_pe - tr.src.a) % cnt
                                                          : 0u; // 其他策略后续补充
            uint32_t choose = std::min(top_k_, cnt);
            for (uint32_t row = 0; row < rows_per_pe_; ++row) {
                uint32_t src_global = src_pe * rows_per_pe_ + row;
                for (uint32_t i = 0; i < choose; ++i) {
                    uint32_t dst_pe = dst_pes[(off + i) % cnt];
                    uint32_t dst_global = dst_pe * rows_per_pe_ + row;
                    fout << src_global << "," << dst_global << "," << weight_value_ << "\n";
                    edges++;
                }
                tokens++;
            }
        }
    }
    if (stat_edges_) stat_edges_->addData(edges);
    if (stat_tokens_) stat_tokens_->addData(tokens);
    out_->verbose(CALL_INFO, 1, 0, "✅ 生成门控CSV: %s (edges=%" PRIu64 ", tokens=%" PRIu64 ")\n",
                  edges_path_.c_str(), edges, tokens);
}

bool GatingPE::parseTargets(const std::string& s, std::vector<uint32_t>& out) {
    // format: a-b,c,d-e
    std::istringstream ss(s);
    std::string tok;
    while (std::getline(ss, tok, ',')) {
        if (tok.empty()) continue;
        size_t dash = tok.find('-');
        if (dash == std::string::npos) {
            out.push_back((uint32_t)std::stoul(tok));
        } else {
            uint32_t a = (uint32_t)std::stoul(tok.substr(0,dash));
            uint32_t b = (uint32_t)std::stoul(tok.substr(dash+1));
            for (uint32_t x=a; x<=b; ++x) out.push_back(x);
        }
    }
    return !out.empty();
}

bool GatingPE::onEmitTick(SST::Cycle_t) {
    if (!nic_) return false;
    // 解析一次targets
    static std::vector<uint32_t> targets; if (targets.empty()) parseTargets(gate_targets_, targets);
    // 解析一次transitions
    static std::vector<Transition> trs; if (trs.empty()) parseTransitions(transitions_, trs);
    if (targets.empty() || trs.empty()) return false;

    // 简单轮转：每个目标源PE发一个决策（src_row按emit_sent_轮转）
    uint32_t row = (rows_per_pe_ == 0) ? 0u : (uint32_t)(emit_sent_ % rows_per_pe_);
    for (uint32_t src_pe : targets) {
        // 找到对应的dst范围
        std::vector<uint32_t> dst_pes;
        for (auto &tr : trs) if (src_pe >= tr.src.a && src_pe <= tr.src.b) {
            for (uint32_t p=tr.dst.a; p<=tr.dst.b; ++p) dst_pes.push_back(p);
        }
        if (dst_pes.empty()) continue;
        // 选Top-k（round-robin起点）
        uint32_t cnt = (uint32_t)dst_pes.size();
        uint32_t choose = std::min(top_k_, cnt);
        uint32_t off = (src_pe + (uint32_t)emit_sent_) % cnt;
        std::vector<uint32_t> pick; pick.reserve(choose);
        for (uint32_t i=0;i<choose;i++) pick.push_back(dst_pes[(off+i)%cnt]);
        // 组包并发送
        auto* ev = new GatingDecisionEvent();
        ev->token_id = (uint32_t)emit_sent_;
        ev->src_pe = src_pe;
        ev->src_row = row;
        ev->top_k = choose;
        ev->ttl_cycles = 1000; // 简易TTL，可加入参数化
        ev->dest_pes = pick;
        if (auto* nic_impl = dynamic_cast<SnnNIC*>(nic_)) {
            if (nic_impl->sendControl(ev, src_pe)) {
                if (stat_evt_sent_) stat_evt_sent_->addData(1);
            } else {
                delete ev; // 发送失败，避免泄漏
            }
        } else {
            delete ev;
        }
    }
    emit_sent_++;
    if (emit_count_ > 0 && emit_sent_ >= emit_count_) return true; // 停止
    return false; // 继续
}
