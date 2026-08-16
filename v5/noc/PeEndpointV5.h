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
#include <map>
#include <string>
#include <vector>

namespace SST { namespace SnnDL { namespace v5 {

class PeEndpointV5 final : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(PeEndpointV5, "SnnDL", "PeEndpointV5",
        SST_ELI_ELEMENT_VERSION(2,0,0), "P5 bounded multi-Core endpoint for SST Merlin", COMPONENT_CATEGORY_NETWORK)
    SST_ELI_DOCUMENT_PARAMS(
        {"pe_id", "Logical PE/network endpoint identifier", "0"},
        {"mesh_x", "2D mesh X dimension", "1"},
        {"cores_per_pe", "Core dispatch ports behind this endpoint", "1"},
        {"tx_queue_entries", "Finite source data packet queue", "16"},
        {"rx_queue_entries", "Finite receive queue per Core", "16"},
        {"control_queue_entries", "Finite control packet queue", "32"},
        {"flit_size_bytes", "Accounting flit size", "32"},
        {"destination_pe", "Default destination PE for Core egress", "0"},
        {"destination_core", "Default destination Core for Core egress", "0"},
        {"core_destination_pes", "Optional comma-separated destination PE per local Core", ""},
        {"core_destination_cores", "Optional comma-separated destination Core per local Core", ""},
        {"core_route_table", "Artifact-v2 Core/local source routes and target masks", ""},
        {"route_contract_v2", "Treat absent local source routes as zero fanout", "0"},
        {"external_stimulus_to_network", "Inject provider stimuli as source-owned external spikes", "0"},
        {"multicast_mode", "unicast, source_replication, or native_tree", "source_replication"},
        {"native_link_credits", "Initial native Router input credits", "8"},
        {"payload_bytes", "Core-spike payload bytes", "0"},
        {"core_attached", "Enable Core/provider proxy links", "0"},
        {"timed_control", "Enable VN1 epoch control protocol", "0"},
        {"coordinator_pe", "PE hosting EpochCoordinatorV5", "0"},
        {"core_held_spike_entries", "Attached Core held-spike capacity", "32"},
        {"output_json", "Endpoint evidence path", ""},
        {"clock", "Endpoint clock", "1GHz"},
        {"verbose", "Verbose level", "0"})
    SST_ELI_DOCUMENT_PORTS(
        {"probe_in", "Probe packet injection", {"SnnDL.NocPacketV5Event"}},
        {"probe_out", "Probe ACK and delivered packet", {"SnnDL.NocPacketV5Event", "SnnDL.NocInjectionAckV5Event"}},
        {"epoch_command", "Coordinator command input", {"SnnDL.NocControlV5Event"}},
        {"epoch_status", "Coordinator status output", {"SnnDL.NocControlV5Event"}},
        {"native_network", "Native multicast Router data link", {"SnnDL.NocPacketV5Event", "SnnDL.NocCreditV5Event"}},
        {"provider_control", "Legacy single-Core provider control", {"SnnDL.CoreControlEvent"}},
        {"core_control", "Legacy single-Core control output", {"SnnDL.CoreControlEvent"}},
        {"provider_spike", "Legacy single-Core provider spike", {"SnnDL.CoreSpikeEvent"}},
        {"provider_ack", "Legacy single-Core provider ACK", {"SnnDL.CoreSpikeAckEvent"}},
        {"core_spike", "Legacy single-Core spike ingress", {"SnnDL.CoreSpikeEvent"}},
        {"core_ack", "Legacy single-Core spike ACK", {"SnnDL.CoreSpikeAckEvent"}},
        {"core_egress", "Legacy single-Core held-spike egress", {"SnnDL.CoreSpikeEvent"}},
        {"provider_monitor", "Legacy single-Core output monitor", {"SnnDL.CoreSpikeEvent"}},
        {"core_status", "Legacy single-Core status input", {"SnnDL.CoreStatusEvent"}},
        {"provider_status", "Legacy single-Core status monitor", {"SnnDL.CoreStatusEvent"}},
        {"provider_control%(core)d", "Provider control for each Core", {"SnnDL.CoreControlEvent"}},
        {"core_control%(core)d", "Control output for each Core", {"SnnDL.CoreControlEvent"}},
        {"provider_spike%(core)d", "Provider spike for each Core", {"SnnDL.CoreSpikeEvent"}},
        {"provider_ack%(core)d", "Provider ACK for each Core", {"SnnDL.CoreSpikeAckEvent"}},
        {"core_spike%(core)d", "Spike ingress for each Core", {"SnnDL.CoreSpikeEvent"}},
        {"core_ack%(core)d", "Spike ACK for each Core", {"SnnDL.CoreSpikeAckEvent"}},
        {"core_egress%(core)d", "Held-spike egress for each Core", {"SnnDL.CoreSpikeEvent"}},
        {"provider_monitor%(core)d", "Output monitor for each Core", {"SnnDL.CoreSpikeEvent"}},
        {"core_status%(core)d", "Status input for each Core", {"SnnDL.CoreStatusEvent"}},
        {"provider_status%(core)d", "Status monitor for each Core", {"SnnDL.CoreStatusEvent"}})
    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS(
        {"networkIF", "Merlin linkcontrol interface", "SST::Interfaces::SimpleNetwork"})
    SST_ELI_DOCUMENT_STATISTICS(
        {"noc.packets", "Physical data packets transmitted", "packets", 1},
        {"noc.flits", "Physical data flits transmitted", "flits", 1},
        {"noc.logical_deliveries", "Logical data packets delivered", "deliveries", 1},
        {"noc.control_packets", "Physical control packets transmitted", "packets", 1},
        {"noc.control_flits", "Physical control flits transmitted", "flits", 1},
        {"noc.tx_stall_cycles", "TX credit stalls", "cycles", 1},
        {"noc.rx_stall_cycles", "RX local-capacity stalls", "cycles", 1},
        {"noc.core_retry_cycles", "Core retry cycles", "cycles", 1})

    PeEndpointV5(SST::ComponentId_t, SST::Params&);
    ~PeEndpointV5() override;
    void init(unsigned int phase) override;
    void setup() override;
    void finish() override;

private:
    enum class AckOrigin : std::uint8_t { Provider, Network };
    struct RouteTarget {
        std::uint32_t pe = 0;
        std::uint64_t core_mask = 0;
    };
    struct SourceRoute {
        std::uint64_t source_global = 0;
        std::uint64_t route_id = 0;
        std::vector<RouteTarget> targets;
    };
    struct CorePort {
        SST::Link* provider_control = nullptr;
        SST::Link* core_control = nullptr;
        SST::Link* provider_spike = nullptr;
        SST::Link* provider_ack = nullptr;
        SST::Link* core_spike = nullptr;
        SST::Link* core_ack = nullptr;
        SST::Link* core_egress = nullptr;
        SST::Link* provider_monitor = nullptr;
        SST::Link* core_status = nullptr;
        SST::Link* provider_status = nullptr;
        std::deque<NocPacketV5Event*> rx;
        std::deque<AckOrigin> ack_origins;
        CoreControlEvent* pending_seal = nullptr;
        std::uint32_t destination_pe = 0;
        std::uint32_t destination_core = 0;
        std::map<std::uint32_t, SourceRoute> source_routes;
        bool started = false;
        bool network_inflight = false;
    };

    void handleProbe_(SST::Event*);
    void handleEpochCommand_(SST::Event*);
    void handleProviderControl_(SST::Event*, int core);
    void handleProviderSpike_(SST::Event*, int core);
    void handleCoreAck_(SST::Event*, int core);
    void handleCoreEgress_(SST::Event*, int core);
    void routeSourceSpike_(SST::Event*, int core, bool monitor);
    void handleCoreStatus_(SST::Event*, int core);
    void handleNative_(SST::Event*);
    bool tick_(SST::Cycle_t);
    void enqueueData_(NocPacketV5Event*, SST::Link* ack_link);
    void enqueueControl_(NocControlV5Event*);
    void receiveData_();
    void receiveControl_();
    void transmitData_();
    void transmitControl_();
    void distributeData_();
    void dispatchCores_();
    void deliverControl_(NocControlV5Event*);
    void sendStatus_(CoreControlOp operation, std::uint64_t epoch, std::uint32_t core, std::uint64_t count = 0);
    void writeEvidence_() const;
    bool drainedForSeal_(std::size_t core) const;
    static std::vector<std::uint32_t> parseDestinations_(const std::string&, std::uint32_t count, std::uint32_t fallback);
    void parseRoutes_(const std::string&);

    SST::Output out_;
    SST::Interfaces::SimpleNetwork* network_ = nullptr;
    SST::Link *probe_in_=nullptr, *probe_out_=nullptr, *epoch_command_=nullptr, *epoch_status_=nullptr, *native_link_=nullptr;
    std::vector<CorePort> cores_;
    std::deque<NocPacketV5Event*> data_tx_, ingress_rx_;
    std::deque<NocControlV5Event*> control_tx_;
    std::uint32_t pe_id_=0, mesh_x_=1, cores_per_pe_=1;
    std::uint32_t tx_capacity_=16, rx_capacity_=16, control_capacity_=32, flit_bytes_=32;
    std::uint32_t payload_bytes_=0, core_held_capacity_=32, coordinator_pe_=0;
    bool core_attached_=false, timed_control_=false, legacy_ports_=false, native_tree_=false, route_contract_v2_=false;
    bool external_stimulus_to_network_=false;
    std::string output_json_;
    std::uint64_t cycles_=0, tx_packets_=0, rx_packets_=0, logical_deliveries_=0;
    std::uint64_t control_tx_packets_=0, control_rx_packets_=0, control_deliveries_=0;
    std::uint64_t tx_bits_=0, tx_flits_=0, control_bits_=0, control_flits_=0;
    std::uint64_t tx_stalls_=0, rx_stalls_=0, core_retries_=0;
    std::uint64_t latency_sum_ns_=0, latency_max_ns_=0, hop_sum_=0, hop_max_=0, drops_=0;
    std::vector<std::vector<std::uint64_t>> core_epoch_enqueued_;
    std::uint64_t next_packet_id_=1, logical_spikes_=0, source_packets_=0, zero_fanout_=0;
    std::uint64_t external_zero_fanout_=0, core_zero_fanout_=0;
    std::uint32_t native_credits_=0, native_credit_limit_=0;
};

}}}
#endif
