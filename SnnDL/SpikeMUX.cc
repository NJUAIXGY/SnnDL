// -*- c++ -*-
// SpikeMUX.cc: 跨 PE 脉冲多路复用/转发组件（最小实现）

#include <sst/core/sst_config.h>
#include "SpikeMUX.h"

using namespace SST;
using namespace SST::SnnDL;

SpikeMUX::SpikeMUX(ComponentId_t id, Params& params) : Component(id) {
    verbose_ = params.find<int>("verbose", 0);
    output_ = new Output("SpikeMUX[@p:@l]: ", verbose_, 0, Output::STDOUT);

    num_ports_ = params.find<int>("num_ports", 2);
    per_pe_neurons_ = params.find<uint64_t>("per_pe_neurons", 0);
    max_queue_depth_ = params.find<uint32_t>("max_queue_depth", 0);
    fwd_latency_cycles_ = params.find<uint32_t>("fwd_latency_cycles", 0);
    use_dest_node_first_ = params.find<int>("use_dest_node_first", 1) != 0;

    stat_forwarded_ = registerStatistic<uint64_t>("mux_forwarded");
    stat_dropped_ = registerStatistic<uint64_t>("mux_dropped");
    stat_queue_peak_ = registerStatistic<uint64_t>("mux_queue_peak");

    in_links_.resize(num_ports_, nullptr);
    out_links_.resize(num_ports_, nullptr);
    out_queues_.resize(num_ports_);

    for (int i = 0; i < num_ports_; i++) {
        std::string in_name = "core_in" + std::to_string(i);
        std::string out_name = "core_out" + std::to_string(i);
        in_links_[i] = configureLink(in_name, new Event::Handler2<SpikeMUX, &SpikeMUX::handleInbound>(this));
        out_links_[i] = configureLink(out_name);
        output_->verbose(CALL_INFO, 2, 0, "🔗 端口i=%d: in=%s, out=%s\n", i, in_name.c_str(), out_name.c_str());
    }

    output_->verbose(CALL_INFO, 1, 0, "🚦 SpikeMUX 初始化: ports=%d, per_pe_neurons=%" PRIu64 ", max_q=%u, fwd_lat=%u\n",
                     num_ports_, per_pe_neurons_, max_queue_depth_, fwd_latency_cycles_);
}

SpikeMUX::~SpikeMUX() {
    delete output_;
}

void SpikeMUX::init(unsigned int phase) {
    if (phase == 0) {
        output_->verbose(CALL_INFO, 2, 0, "🔄 SpikeMUX init phase %u\n", phase);
    }
}

void SpikeMUX::setup() {
    output_->verbose(CALL_INFO, 2, 0, "✅ SpikeMUX setup 完成\n");
}

void SpikeMUX::finish() {
    output_->verbose(CALL_INFO, 1, 0, "🏁 SpikeMUX 完成仿真\n");
}

int SpikeMUX::selectPortForDestination(uint32_t dest_global, uint32_t dest_node) const {
    if (per_pe_neurons_ == 0 || num_ports_ <= 0) return -1;

    // 基于dest_global优先计算端口索引（更鲁棒）
    uint64_t calc_index64 = static_cast<uint64_t>(dest_global) / per_pe_neurons_;
    int calc_index = (calc_index64 < static_cast<uint64_t>(num_ports_)) ? static_cast<int>(calc_index64) : -1;

    if (use_dest_node_first_) {
        // 若配置要求优先用dest_node，则仅在其与计算结果一致时采用
        if (dest_node < static_cast<uint32_t>(num_ports_)) {
            int node_index = static_cast<int>(dest_node);
            if (calc_index >= 0 && node_index == calc_index) {
                return node_index;
            }
        }
        // 不一致时回退到calc_index，避免被错误的dest_node误导
        return calc_index;
    }

    return calc_index;
}

void SpikeMUX::handleInbound(Event* ev) {
    auto* sev = dynamic_cast<SpikeEvent*>(ev);
    if (!sev) { delete ev; return; }

    uint32_t dest = sev->getDestinationNeuron();
    uint32_t dest_node = sev->getDestinationNode();
    int out_port = selectPortForDestination(dest, dest_node);
    if (out_port < 0 || out_port >= num_ports_ || out_links_[out_port] == nullptr) {
        output_->verbose(CALL_INFO, 2, 0, "⚠️ 无法路由: dest=%u -> out_port=%d\n", dest, out_port);
        if (stat_dropped_) stat_dropped_->addData(1);
        delete sev;
        return;
    }

    // 背压/队列与延迟：根据 fwd_latency_cycles_ 决定立即发送或入队
    if (fwd_latency_cycles_ == 0) {
        // 立即尝试发送，如需模拟端口容量亦可入队
        output_->verbose(CALL_INFO, 3, 0, "➡️ 转发: dest=%u node=%u -> core_out%d\n", dest, dest_node, out_port);
        out_links_[out_port]->send(sev);
        if (stat_forwarded_) stat_forwarded_->addData(1);
    } else {
        auto& q = out_queues_[out_port];
        if (max_queue_depth_ > 0 && q.size() >= max_queue_depth_) {
            output_->verbose(CALL_INFO, 2, 0, "⚠️ 队列溢出: port=%d size=%zu, 丢弃dest=%u\n", out_port, q.size(), dest);
            if (stat_dropped_) stat_dropped_->addData(1);
            delete sev; return;
        }
        Pending p; p.ev = sev; p.ready_cycle = current_cycle_ + fwd_latency_cycles_; p.out_port = out_port;
        q.push_back(p);
        if (q.size() > queue_peak_) {
            queue_peak_ = static_cast<uint32_t>(q.size());
            if (stat_queue_peak_) stat_queue_peak_->addData(queue_peak_);
        }
    }
}

bool SpikeMUX::onClockTick(Cycle_t cycle) {
    current_cycle_ = cycle;
    // 逐端口检查可出队的消息
    for (int p = 0; p < num_ports_; ++p) {
        auto& q = out_queues_[p];
        while (!q.empty() && q.front().ready_cycle <= current_cycle_) {
            Pending item = q.front(); q.pop_front();
            if (out_links_[p]) {
                out_links_[p]->send(item.ev);
                if (stat_forwarded_) stat_forwarded_->addData(1);
                output_->verbose(CALL_INFO, 4, 0, "⏩ 出队转发: port=%d dest=%u\n", p, item.ev->getDestinationNeuron());
            } else {
                delete item.ev;
            }
        }
    }
    return false;
}

