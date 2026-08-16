#ifndef SST_SNN_DL_V5_MULTICAST_ROUTER_V5_H
#define SST_SNN_DL_V5_MULTICAST_ROUTER_V5_H

#include "NocEventsV5.h"
#include "MulticastBranchTableV5.h"
#include "MulticastCreditV5.h"

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>

#include <array>
#include <cstdint>
#include <deque>
#include <string>

namespace SST { namespace SnnDL { namespace v5 {

class MulticastRouterV5 final : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(MulticastRouterV5, "SnnDL", "MulticastRouterV5",
        SST_ELI_ELEMENT_VERSION(1,0,0),
        "SST-native finite-buffer deterministic 2D multicast Router", COMPONENT_CATEGORY_NETWORK)
    SST_ELI_DOCUMENT_PARAMS(
        {"pe_id", "Router/PE identifier", "0"},
        {"mesh_rows", "2D mesh rows", "1"},
        {"mesh_cols", "2D mesh columns", "1"},
        {"branch_table", "route_id:output_mask:local_core_mask records", ""},
        {"input_queue_entries", "Finite packet entries per input", "8"},
        {"flit_size_bytes", "Physical flit bytes", "32"},
        {"route_latency_cycles", "Branch table lookup latency", "1"},
        {"output_latency_cycles", "Output pipeline latency", "1"},
        {"output_json", "Router evidence path", ""},
        {"clock", "Router clock", "1GHz"},
        {"verbose", "Verbose logging level", "0"})
    SST_ELI_DOCUMENT_PORTS(
        {"local", "Bidirectional PE endpoint link", {"SnnDL.NocPacketV5Event", "SnnDL.NocCreditV5Event"}},
        {"east", "Bidirectional east Router link", {"SnnDL.NocPacketV5Event", "SnnDL.NocCreditV5Event"}},
        {"west", "Bidirectional west Router link", {"SnnDL.NocPacketV5Event", "SnnDL.NocCreditV5Event"}},
        {"south", "Bidirectional south Router link", {"SnnDL.NocPacketV5Event", "SnnDL.NocCreditV5Event"}},
        {"north", "Bidirectional north Router link", {"SnnDL.NocPacketV5Event", "SnnDL.NocCreditV5Event"}})
    SST_ELI_DOCUMENT_STATISTICS(
        {"mcast.source_packets", "Packets accepted at local injection", "packets", 1},
        {"mcast.router_clones", "Additional branch copies created", "events", 1},
        {"mcast.branch_transmissions", "All branch transmissions", "packets", 1},
        {"mcast.link_traversals", "Non-local branch transmissions", "packets", 1},
        {"mcast.flit_traversals", "Non-local flit traversals", "flits", 1},
        {"mcast.route_lookups", "Branch table lookups", "requests", 1},
        {"mcast.credit_stall_cycles", "Atomic branch credit stalls", "cycles", 1},
        {"mcast.output_stall_cycles", "Atomic branch output stalls", "cycles", 1})

    MulticastRouterV5(SST::ComponentId_t, SST::Params&);
    ~MulticastRouterV5() override;
    void finish() override;

private:
    enum Port : std::uint8_t { Local=0, East=1, West=2, South=3, North=4, PortCount=5 };
    struct QueuedPacket { NocPacketV5Event* packet=nullptr; std::uint64_t ready_cycle=0; };

    void handle_(SST::Event*, int port);
    bool tick_(SST::Cycle_t);
    void writeEvidence_() const;
    static std::uint8_t branchBit_(Port);

    SST::Output out_;
    std::array<SST::Link*, PortCount> links_{};
    std::array<std::deque<QueuedPacket>, PortCount> inputs_;
    std::array<MulticastCreditV5, PortCount> credits_{};
    std::array<std::uint64_t, PortCount> output_busy_until_{};
    MulticastBranchTableV5 branches_;
    std::uint32_t pe_id_=0, rows_=1, cols_=1, input_entries_=8, flit_bytes_=32;
    std::uint64_t route_latency_=1, output_latency_=1, cycle_=0;
    std::size_t round_robin_=0;
    std::string output_json_;
    std::uint64_t source_packets_=0, clones_=0, branch_transmissions_=0;
    std::uint64_t link_traversals_=0, flit_traversals_=0, route_lookups_=0;
    std::uint64_t credit_stalls_=0, output_stalls_=0, input_peak_=0, drops_=0;
};

}}}
#endif
