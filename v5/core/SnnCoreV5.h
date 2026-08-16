#ifndef SST_SNN_DL_V5_SNN_CORE_V5_H
#define SST_SNN_DL_V5_SNN_CORE_V5_H

#include "CorePipeline.h"
#include "v5/events/CoreEvents.h"
#include "v5/api/StatisticNames.h"

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>

#include <cstdint>
#include <map>
#include <string>

namespace SST {
namespace SnnDL {
namespace v5 {

class SnnCoreV5 final : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(
        SnnCoreV5,
        "SnnDL",
        "SnnCoreV5",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "P1 independent SNN core with bounded cycle pipeline",
        COMPONENT_CATEGORY_PROCESSOR
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"core_id", "Logical core identifier", "0"},
        {"pe_id", "Logical PE owner for the shared route region", "0"},
        {"neurons", "Neurons owned by this core", "64"},
        {"core_state_sram_capacity_bytes", "CoreState SRAM capacity; zero derives from neurons", "0"},
        {"core_delta_sram_capacity_bytes", "CoreDelta SRAM capacity; zero derives from queue geometry", "0"},
        {"core_index_sram_capacity_bytes", "CoreIndex SRAM capacity", "4096"},
        {"pe_route_sram_capacity_bytes", "PE route SRAM capacity", "4096"},
        {"core_state_sram_banks", "CoreState SRAM banks", "1"},
        {"core_delta_sram_banks", "CoreDelta SRAM banks", "1"},
        {"core_index_sram_banks", "CoreIndex SRAM banks", "1"},
        {"pe_route_sram_banks", "PE route SRAM banks", "1"},
        {"storage_ports_per_bank", "Ports per typed SRAM bank", "1"},
        {"storage_interleave_bytes", "Typed SRAM low-order bank interleave", "4"},
        {"storage_read_latency_cycles", "Typed SRAM read latency", "1"},
        {"storage_write_latency_cycles", "Typed SRAM write latency", "1"},
        {"storage_request_queue_entries", "Typed SRAM request queue entries", "16"},
        {"storage_response_queue_entries", "Typed SRAM response queue entries", "16"},
        {"storage_max_delta_entries_per_neuron", "Resident CoreDelta entries per neuron", "32"},
        {"core_index_bytes", "Typed CoreIndex region bytes", "4096"},
        {"pe_route_bytes", "Typed PeRoute region bytes", "4096"},
        {"ingress_queue_entries", "Bounded spike ingress entries", "16"},
        {"row_queue_entries", "Bounded row request entries", "16"},
        {"synapse_queue_entries", "Bounded ready synapse entries", "32"},
        {"retire_queue_entries", "Bounded deterministic retire entries", "32"},
        {"accumulator_queue_entries", "Bounded accumulator entries", "32"},
        {"held_spike_queue_entries", "Bounded held output entries", "32"},
        {"ingress_width", "Ingress width per cycle", "1"},
        {"ingress_latency_cycles", "Ingress latency", "1"},
        {"row_lookup_width", "Row lookup width per cycle", "1"},
        {"row_lookup_latency_cycles", "Row lookup latency", "1"},
        {"synapse_width", "Synapse width per cycle", "1"},
        {"synapse_latency_cycles", "Synapse latency", "1"},
        {"retire_width", "Retire width per cycle", "1"},
        {"retire_latency_cycles", "Retire latency", "1"},
        {"accumulator_width", "Accumulator width per cycle", "1"},
        {"accumulator_latency_cycles", "Accumulator latency", "1"},
        {"neuron_lanes", "Neuron lanes per cycle", "1"},
        {"neuron_latency_cycles", "Neuron batch latency", "1"},
        {"dt_ms", "LIF integration interval", "1.0"},
        {"tau_mem_ms", "LIF membrane time constant", "20.0"},
        {"threshold", "LIF firing threshold", "1.0"},
        {"reset", "LIF reset potential", "0.0"},
        {"refractory_timesteps", "LIF refractory duration", "0"},
        {"clock", "Core clock", "1GHz"},
        {"verbose", "Verbose logging level", "0"}
    )

    SST_ELI_DOCUMENT_STATISTICS(
        {"core.ingress.accepted", "Accepted spike ingress entries", "events", 1},
        {"core.ingress.full_cycles", "Ingress queue full cycles", "cycles", 1},
        {"core.ingress.occupancy", "Ingress queue occupancy", "events", 1},
        {"core.ingress.stall_cycles", "Ingress downstream stall cycles", "cycles", 1},
        {"core.row_lookup.accepted", "Spikes accepted by row lookup", "operations", 1},
        {"core.row_lookup.issued", "Row requests issued", "operations", 1},
        {"core.row_lookup.completed", "Rows completed", "operations", 1},
        {"core.row_lookup.occupancy", "Row lookup occupancy", "events", 1},
        {"core.row_lookup.busy_cycles", "Row lookup busy cycles", "cycles", 1},
        {"core.row_lookup.stall_cycles", "Row lookup stall cycles", "cycles", 1},
        {"core.row_lookup.full_cycles", "Row lookup full cycles", "cycles", 1},
        {"core.synapse.accepted", "Synapse responses accepted", "operations", 1},
        {"core.synapse.issued", "Synapse operations issued", "operations", 1},
        {"core.synapse.completed", "Synapse operations completed", "operations", 1},
        {"core.synapse.occupancy", "Synapse queue occupancy", "events", 1},
        {"core.synapse.busy_cycles", "Synapse busy cycles", "cycles", 1},
        {"core.synapse.stall_cycles", "Synapse stall cycles", "cycles", 1},
        {"core.synapse.full_cycles", "Synapse full cycles", "cycles", 1},
        {"core.retire.accepted", "Retire entries accepted", "operations", 1},
        {"core.retire.issued", "Retire entries issued", "operations", 1},
        {"core.retire.completed", "Retire entries completed", "operations", 1},
        {"core.retire.occupancy", "Retire occupancy", "events", 1},
        {"core.retire.busy_cycles", "Retire busy cycles", "cycles", 1},
        {"core.retire.stall_cycles", "Retire stall cycles", "cycles", 1},
        {"core.retire.full_cycles", "Retire full cycles", "cycles", 1},
        {"core.accumulator.accepted", "Accumulator entries accepted", "operations", 1},
        {"core.accumulator.issued", "Accumulator updates issued", "operations", 1},
        {"core.accumulator.completed", "Accumulator updates completed", "operations", 1},
        {"core.accumulator.occupancy", "Accumulator occupancy", "events", 1},
        {"core.accumulator.busy_cycles", "Accumulator busy cycles", "cycles", 1},
        {"core.accumulator.stall_cycles", "Accumulator stall cycles", "cycles", 1},
        {"core.accumulator.full_cycles", "Accumulator full cycles", "cycles", 1},
        {"core.neuron.evaluated", "Neuron evaluations", "operations", 1},
        {"core.neuron.fired", "Neuron firing events", "events", 1},
        {"core.neuron.occupancy", "Neuron pipeline occupancy", "events", 1},
        {"core.neuron.busy_cycles", "Neuron busy cycles", "cycles", 1},
        {"core.neuron.stall_cycles", "Neuron stall cycles", "cycles", 1},
        {"core.held_spike.accepted", "Held spikes accepted", "events", 1},
        {"core.held_spike.released", "Held spikes released", "events", 1},
        {"core.held_spike.occupancy", "Held spike occupancy", "events", 1},
        {"core.held_spike.full_cycles", "Held spike full cycles", "cycles", 1},
        {"core.held_spike.stall_cycles", "Held spike stall cycles", "cycles", 1},
        {"storage.state.reads", "Typed CoreState reads", "requests", 1},
        {"storage.delta.reads", "Typed CoreDelta reads", "requests", 1},
        {"timestep.core_elapsed_ns", "Core elapsed time", "ns", 1}
    )

    SST_ELI_DOCUMENT_PORTS(
        {"control", "Start/seal/commit control", {"SnnDL.CoreControlEvent"}},
        {"spike_in", "Targeted spike ingress", {"SnnDL.CoreSpikeEvent"}},
        {"spike_ack", "Spike ingress accepted/retry acknowledgement", {"SnnDL.CoreSpikeAckEvent"}},
        {"spike_out", "Held spike egress", {"SnnDL.CoreSpikeEvent"}},
        {"row_provider", "Ideal row and weight provider", {"SnnDL.CoreRowRequestEvent", "SnnDL.CoreSynapseResponseEvent", "SnnDL.CoreRowDoneEvent"}},
        {"status", "Commit and counter status", {"SnnDL.CoreStatusEvent"}}
    )

    SnnCoreV5(SST::ComponentId_t id, SST::Params& params);
    ~SnnCoreV5() override;

    void init(unsigned int phase) override;
    void setup() override;
    void finish() override;

private:
    void handleControl_(SST::Event* event);
    void handleSpike_(SST::Event* event);
    void handleProvider_(SST::Event* event);
    bool clockTick_(SST::Cycle_t cycle);
    void sendStatus_(CoreControlOp operation);
    void sendRequests_();
    void sendReleasedSpikes_();
    void sendSpikeAck_(const CoreSpikeEvent& spike, bool accepted, bool retryable);
    void sendProviderAck_(std::uint64_t timestep, std::uint64_t source_neuron,
                          std::uint64_t source_event_seq, std::uint64_t edge_ordinal,
                          bool row_done, bool accepted, bool retryable);
    void publishStatistics_();

    SST::Output out_;
    SST::Link* control_link_ = nullptr;
    SST::Link* spike_in_link_ = nullptr;
    SST::Link* spike_ack_link_ = nullptr;
    SST::Link* spike_out_link_ = nullptr;
    SST::Link* row_provider_link_ = nullptr;
    SST::Link* status_link_ = nullptr;
    std::uint32_t core_id_ = 0;
    bool commit_ready_sent_ = false;
    bool stats_published_ = false;
    std::uint64_t timestep_start_ns_ = 0;
    std::uint64_t last_elapsed_ns_ = 0;
    CorePipeline pipeline_;
    std::map<std::string, Statistic<std::uint64_t>*> statistics_;
};

} // namespace v5
} // namespace SnnDL
} // namespace SST

#endif
