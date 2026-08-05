#ifndef SST_SNN_DL_V5_PE_ENDPOINT_V5_H
#define SST_SNN_DL_V5_PE_ENDPOINT_V5_H

#include "NocEventsV5.h"
#include "v5/events/CoreEvents.h"
#include <sst/core/component.h>
#include <sst/core/interfaces/simpleNetwork.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <cstdint>
#include <deque>
#include <string>

namespace SST { namespace SnnDL { namespace v5 {

class PeEndpointV5 final : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(PeEndpointV5, "SnnDL", "PeEndpointV5",
        SST_ELI_ELEMENT_VERSION(1,0,0), "P4 bounded SNN endpoint for SST Merlin", COMPONENT_CATEGORY_NETWORK)
    SST_ELI_DOCUMENT_PARAMS(
        {"pe_id", "Logical PE/network endpoint identifier", "0"},
        {"mesh_x", "2D mesh X dimension", "1"},
        {"tx_queue_entries", "Finite source packet queue", "16"},
        {"rx_queue_entries", "Finite receive/dispatch queue", "16"},
        {"flit_size_bytes", "Accounting flit size", "32"},
        {"destination_pe", "Destination PE for Core egress", "0"},
        {"destination_core", "Destination Core for Core egress", "0"},
        {"payload_bytes", "Core-spike payload bytes", "0"},
        {"core_attached", "Enable Core/provider proxy links", "0"},
        {"core_held_spike_entries", "Attached Core held-spike capacity; TX must cover one release burst", "32"},
        {"output_json", "Endpoint evidence path", ""},
        {"clock", "Endpoint clock", "1GHz"},
        {"verbose", "Verbose level", "0"})
    SST_ELI_DOCUMENT_PORTS(
        {"probe_in", "Probe packet injection", {"SnnDL.NocPacketV5Event"}},
        {"probe_out", "Probe ACK and delivered packet", {"SnnDL.NocPacketV5Event", "SnnDL.NocInjectionAckV5Event"}},
        {"provider_control", "Provider control input", {"SnnDL.CoreControlEvent"}},
        {"core_control", "Core control output", {"SnnDL.CoreControlEvent"}},
        {"provider_spike", "Provider local spike input", {"SnnDL.CoreSpikeEvent"}},
        {"provider_ack", "Provider local spike ACK", {"SnnDL.CoreSpikeAckEvent"}},
        {"core_spike", "Core spike ingress", {"SnnDL.CoreSpikeEvent"}},
        {"core_ack", "Core spike ACK", {"SnnDL.CoreSpikeAckEvent"}},
        {"core_egress", "Core held-spike egress", {"SnnDL.CoreSpikeEvent"}},
        {"provider_monitor", "Core output monitor", {"SnnDL.CoreSpikeEvent"}})
    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS(
        {"networkIF", "Merlin linkcontrol interface", "SST::Interfaces::SimpleNetwork"})
    SST_ELI_DOCUMENT_STATISTICS(
        {"noc.packets", "Physical packets transmitted", "packets", 1},
        {"noc.flits", "Physical flits transmitted", "flits", 1},
        {"noc.logical_deliveries", "Logical packets delivered", "deliveries", 1},
        {"noc.tx_stall_cycles", "TX credit stalls", "cycles", 1},
        {"noc.rx_stall_cycles", "RX local-capacity stalls", "cycles", 1},
        {"noc.core_retry_cycles", "Core retry cycles", "cycles", 1})

    PeEndpointV5(SST::ComponentId_t, SST::Params&);
    ~PeEndpointV5() override;
    void init(unsigned int phase) override;
    void setup() override;
    void finish() override;
private:
    void handleProbe_(SST::Event*);
    void handleProviderControl_(SST::Event*);
    void handleProviderSpike_(SST::Event*);
    void handleCoreAck_(SST::Event*);
    void handleCoreEgress_(SST::Event*);
    bool tick_(SST::Cycle_t);
    void enqueue_(NocPacketV5Event*, SST::Link* ack_link);
    void receive_();
    void transmit_();
    void dispatch_();
    void writeEvidence_() const;

    SST::Output out_;
    SST::Interfaces::SimpleNetwork* network_ = nullptr;
    SST::Link *probe_in_=nullptr, *probe_out_=nullptr, *provider_control_=nullptr,
        *core_control_=nullptr, *provider_spike_=nullptr, *provider_ack_=nullptr,
        *core_spike_=nullptr, *core_ack_=nullptr, *core_egress_=nullptr, *provider_monitor_=nullptr;
    std::deque<NocPacketV5Event*> tx_, rx_;
    std::uint32_t pe_id_=0, mesh_x_=1, tx_capacity_=16, rx_capacity_=16, flit_bytes_=32;
    std::uint32_t destination_pe_=0, destination_core_=0, payload_bytes_=0, core_held_capacity_=32;
    bool core_attached_=false, core_started_=false, core_inflight_=false;
    CoreControlEvent* pending_seal_=nullptr;
    std::string output_json_;
    std::uint64_t cycles_=0, tx_packets_=0, rx_packets_=0, logical_deliveries_=0;
    std::uint64_t tx_bits_=0, tx_flits_=0, tx_stalls_=0, rx_stalls_=0, core_retries_=0;
    std::uint64_t latency_sum_ns_=0, latency_max_ns_=0, hop_sum_=0, hop_max_=0, drops_=0;
};

}}}
#endif
