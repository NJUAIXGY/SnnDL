// -*- c++ -*-
// SpikeMUX.h: 跨 PE 脉冲多路复用/转发组件（最小实现）

#ifndef _SPIKEMUX_H
#define _SPIKEMUX_H

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <sst/core/params.h>
#include <vector>
#include <string>
#include <cstdint>

#include "SpikeEvent.h"

namespace SST {
namespace SnnDL {

class SpikeMUX : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(
        SpikeMUX,
        "SnnDL",
        "SpikeMUX",
        SST_ELI_ELEMENT_VERSION(1,0,0),
        "跨PE脉冲转发器（按目标神经元范围选择输出端口）",
        COMPONENT_CATEGORY_NETWORK
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"num_ports", "连接的MultiCorePE数量（成对的in/out端口）", "2"},
        {"per_pe_neurons", "每个PE负责的全局神经元数量（用于范围映射）", "0"},
        {"verbose", "日志详细级别", "0"},
        {"max_queue_depth", "每个输出端口的最大排队深度(0=无限)", "0"},
        {"fwd_latency_cycles", "转发延迟(周期)", "0"},
        {"use_dest_node_first", "路由优先使用事件目的节点(1=是,0=否)", "1"}
    )

    // 动态端口命名：core_in0..core_inN-1, core_out0..core_outN-1
    SST_ELI_DOCUMENT_PORTS(
        {"core_in%(num_ports)d", "来自第i个PE的外发脉冲输入", {"SnnDL.SpikeEvent"}},
        {"core_out%(num_ports)d", "发往第i个PE的外部脉冲输出", {"SnnDL.SpikeEvent"}}
    )

    SST_ELI_DOCUMENT_STATISTICS(
        {"mux_forwarded", "成功转发的脉冲数量", "spikes", 1},
        {"mux_dropped", "因无法判定目标或队列溢出而丢弃的脉冲数量", "spikes", 1},
        {"mux_queue_peak", "所有端口中的队列峰值(条)", "entries", 1}
    )

    SpikeMUX(SST::ComponentId_t id, SST::Params& params);
    ~SpikeMUX() override;

    void init(unsigned int phase) override;
    void setup() override;
    void finish() override;

private:
    void handleInbound(SST::Event* ev);
    bool onClockTick(Cycle_t cycle);
    int selectPortForDestination(uint32_t dest_global, uint32_t dest_node) const;

    struct Pending {
        SpikeEvent* ev;
        uint64_t ready_cycle;
        int out_port;
        Pending() : ev(nullptr), ready_cycle(0), out_port(-1) {}
    };

    int num_ports_;
    uint64_t per_pe_neurons_;
    int verbose_;
    uint32_t max_queue_depth_;
    uint32_t fwd_latency_cycles_;
    bool use_dest_node_first_;

    std::vector<SST::Link*> in_links_;
    std::vector<SST::Link*> out_links_;
    std::vector<std::deque<Pending>> out_queues_;

    SST::Output* output_;

    Statistic<uint64_t>* stat_forwarded_;
    Statistic<uint64_t>* stat_dropped_;
    Statistic<uint64_t>* stat_queue_peak_;

    uint64_t current_cycle_ {0};
    uint32_t queue_peak_ {0};
};

} // namespace SnnDL
} // namespace SST

#endif // _SPIKEMUX_H

