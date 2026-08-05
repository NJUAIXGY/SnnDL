#include <sst/core/sst_config.h>

#include "SnnCoreV5.h"

#include <algorithm>
#include <cinttypes>
#include <stdexcept>

namespace SST {
namespace SnnDL {
namespace v5 {

// Referenced by the canonical aggregate so the loader retains this ELI
// library and its component registration constructors.
extern "C" void snndl_v5_core_anchor() {}

namespace {
using namespace ::SnnDL::v5;

std::uint32_t positive(const SST::Params& params, const char* name, std::uint32_t fallback) {
    return std::max<std::uint32_t>(1, params.find<std::uint32_t>(name, fallback));
}
}

SnnCoreV5::SnnCoreV5(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id), out_("SnnDL.SnnCoreV5", 0, 0, SST::Output::STDOUT), core_id_(params.find<std::uint32_t>("core_id", 0)),
      pipeline_([&params]() {
          CorePipelineConfig config;
          config.neurons = positive(params, "neurons", 64);
          config.ingress_entries = positive(params, "ingress_queue_entries", 16);
          config.row_entries = positive(params, "row_queue_entries", 16);
          config.synapse_entries = positive(params, "synapse_queue_entries", 32);
          config.retire_entries = positive(params, "retire_queue_entries", 32);
          config.accumulator_entries = positive(params, "accumulator_queue_entries", 32);
          config.held_spike_entries = positive(params, "held_spike_queue_entries", 32);
          config.ingress.width = positive(params, "ingress_width", 1);
          config.ingress.latency_cycles = params.find<std::uint32_t>("ingress_latency_cycles", 1);
          config.row_lookup.width = positive(params, "row_lookup_width", 1);
          config.row_lookup.latency_cycles = params.find<std::uint32_t>("row_lookup_latency_cycles", 1);
          config.synapse.width = positive(params, "synapse_width", 1);
          config.synapse.latency_cycles = params.find<std::uint32_t>("synapse_latency_cycles", 1);
          config.retire.width = positive(params, "retire_width", 1);
          config.retire.latency_cycles = params.find<std::uint32_t>("retire_latency_cycles", 1);
          config.accumulator.width = positive(params, "accumulator_width", 1);
          config.accumulator.latency_cycles = params.find<std::uint32_t>("accumulator_latency_cycles", 1);
          config.neuron.width = positive(params, "neuron_lanes", 1);
          config.neuron.latency_cycles = params.find<std::uint32_t>("neuron_latency_cycles", 1);
          config.lif.dt_ms = params.find<float>("dt_ms", 1.0f);
          config.lif.tau_mem_ms = params.find<float>("tau_mem_ms", 20.0f);
          config.lif.threshold = params.find<float>("threshold", 1.0f);
          config.lif.reset = params.find<float>("reset", 0.0f);
          config.lif.refractory_timesteps = params.find<std::uint32_t>("refractory_timesteps", 0);
          config.storage.core_id = params.find<std::uint32_t>("core_id", 0);
          config.storage.pe_id = params.find<std::uint32_t>("pe_id", 0);
          config.storage.index_bytes = positive(params, "core_index_bytes", 4096);
          config.storage.route_bytes = positive(params, "pe_route_bytes", 4096);
          config.storage.max_delta_entries_per_neuron =
              params.find<std::size_t>("storage_max_delta_entries_per_neuron", 32);
          const auto ports = positive(params, "storage_ports_per_bank", 1);
          const auto interleave = std::max<std::uint64_t>(
              1, params.find<std::uint64_t>("storage_interleave_bytes", 4));
          const auto read_latency = positive(params, "storage_read_latency_cycles", 1);
          const auto write_latency = positive(params, "storage_write_latency_cycles", 1);
          const auto request_queue = static_cast<std::size_t>(
              positive(params, "storage_request_queue_entries", 16));
          const auto response_queue = static_cast<std::size_t>(
              positive(params, "storage_response_queue_entries", 16));
          auto configureStorage = [&](BankedSramV5Config& storage, const char* capacity,
                                      const char* banks) {
              storage.capacity_bytes = params.find<std::uint64_t>(capacity, 0);
              storage.banks = positive(params, banks, 1);
              storage.ports_per_bank = ports;
              storage.interleave_bytes = interleave;
              storage.read_latency_cycles = read_latency;
              storage.write_latency_cycles = write_latency;
              storage.request_queue_entries = request_queue;
              storage.response_queue_entries = response_queue;
          };
          configureStorage(config.storage.state_sram, "core_state_sram_capacity_bytes", "core_state_sram_banks");
          configureStorage(config.storage.delta_sram, "core_delta_sram_capacity_bytes", "core_delta_sram_banks");
          configureStorage(config.storage.index_sram, "core_index_sram_capacity_bytes", "core_index_sram_banks");
          configureStorage(config.storage.route_sram, "pe_route_sram_capacity_bytes", "pe_route_sram_banks");
          return config;
      }()) {
    out_.setVerboseLevel(params.find<int>("verbose", 0));
    control_link_ = configureLink("control", new SST::Event::Handler2<SnnCoreV5, &SnnCoreV5::handleControl_>(this));
    spike_in_link_ = configureLink("spike_in", new SST::Event::Handler2<SnnCoreV5, &SnnCoreV5::handleSpike_>(this));
    spike_ack_link_ = configureLink("spike_ack");
    spike_out_link_ = configureLink("spike_out");
    row_provider_link_ = configureLink("row_provider", new SST::Event::Handler2<SnnCoreV5, &SnnCoreV5::handleProvider_>(this));
    status_link_ = configureLink("status");
    if (!control_link_ || !spike_in_link_ || !spike_ack_link_ || !spike_out_link_ || !row_provider_link_ || !status_link_) {
        out_.fatal(CALL_INFO, -1, "SnnCoreV5 requires control, spike_in, spike_ack, spike_out, row_provider and status links\n");
    }

    const char* names[] = {
        kStatisticNameCoreIngressAccepted, kStatisticNameCoreIngressFullCycles,
        kStatisticNameCoreIngressOccupancy, kStatisticNameCoreIngressStallCycles,
        kStatisticNameCoreRowLookupAccepted, kStatisticNameCoreRowLookupIssued,
        kStatisticNameCoreRowLookupCompleted, kStatisticNameCoreRowLookupOccupancy,
        kStatisticNameCoreRowLookupBusyCycles, kStatisticNameCoreRowLookupStallCycles,
        kStatisticNameCoreRowLookupFullCycles, kStatisticNameCoreSynapseAccepted,
        kStatisticNameCoreSynapseIssued, kStatisticNameCoreSynapseCompleted,
        kStatisticNameCoreSynapseOccupancy, kStatisticNameCoreSynapseBusyCycles,
        kStatisticNameCoreSynapseStallCycles, kStatisticNameCoreSynapseFullCycles,
        kStatisticNameCoreRetireAccepted, kStatisticNameCoreRetireIssued,
        kStatisticNameCoreRetireCompleted, kStatisticNameCoreRetireOccupancy,
        kStatisticNameCoreRetireBusyCycles, kStatisticNameCoreRetireStallCycles,
        kStatisticNameCoreRetireFullCycles, kStatisticNameCoreAccumulatorAccepted,
        kStatisticNameCoreAccumulatorIssued, kStatisticNameCoreAccumulatorCompleted,
        kStatisticNameCoreAccumulatorOccupancy, kStatisticNameCoreAccumulatorBusyCycles,
        kStatisticNameCoreAccumulatorStallCycles, kStatisticNameCoreAccumulatorFullCycles,
        kStatisticNameCoreNeuronEvaluated, kStatisticNameCoreNeuronFired,
        kStatisticNameCoreNeuronOccupancy, kStatisticNameCoreNeuronBusyCycles,
        kStatisticNameCoreNeuronStallCycles, kStatisticNameCoreHeldSpikeAccepted,
        kStatisticNameCoreHeldSpikeReleased, kStatisticNameCoreHeldSpikeOccupancy,
        kStatisticNameCoreHeldSpikeFullCycles, kStatisticNameCoreHeldSpikeStallCycles,
        kStatisticNameStorageStateReads, kStatisticNameStorageDeltaReads,
        kStatisticNameTimestepCoreElapsedNs
    };
    for (const auto* name : names) statistics_.emplace(name, registerStatistic<std::uint64_t>(name));
    registerClock(params.find<std::string>("clock", "1GHz"), new SST::Clock::Handler2<SnnCoreV5, &SnnCoreV5::clockTick_>(this));
}

SnnCoreV5::~SnnCoreV5() = default;
void SnnCoreV5::init(unsigned int) {}
void SnnCoreV5::setup() {}

void SnnCoreV5::handleControl_(SST::Event* event) {
    auto* control = dynamic_cast<CoreControlEvent*>(event);
    if (!control) {
        delete event;
        out_.fatal(CALL_INFO, -1, "SnnCoreV5 received an unexpected control event\n");
    }
    try {
        switch (control->operation) {
        case CoreControlOp::Start:
            pipeline_.start(control->timestep);
            timestep_start_ns_ = static_cast<std::uint64_t>(getCurrentSimTimeNano());
            last_elapsed_ns_ = 0;
            commit_ready_sent_ = false;
            sendReleasedSpikes_();
            break;
        case CoreControlOp::SealIngress:
            pipeline_.sealIngress();
            break;
        case CoreControlOp::Commit:
            if (!pipeline_.readyToCommit()) throw std::logic_error("Commit before core drain");
            pipeline_.commit();
            sendStatus_(CoreControlOp::CommitDone);
            break;
        case CoreControlOp::Abort:
            out_.fatal(CALL_INFO, -1, "SnnCoreV5 aborted by controller\n");
            break;
        default:
            break;
        }
    } catch (const std::exception& error) {
        delete control;
        out_.fatal(CALL_INFO, -1, "SnnCoreV5 control error: %s\n", error.what());
    }
    delete control;
}

void SnnCoreV5::handleSpike_(SST::Event* event) {
    auto* spike = dynamic_cast<CoreSpikeEvent*>(event);
    if (!spike) {
        delete event;
        out_.fatal(CALL_INFO, -1, "SnnCoreV5 received an unexpected spike event\n");
    }
    const bool accepted = pipeline_.submitSpike(SpikeInput{spike->timestep, spike->source_neuron, spike->source_event_seq});
    sendSpikeAck_(*spike, accepted, !accepted);
    delete spike;
}

void SnnCoreV5::handleProvider_(SST::Event* event) {
    bool accepted = false;
    std::uint64_t timestep = 0;
    std::uint32_t source_neuron = 0;
    std::uint64_t source_event_seq = 0;
    std::uint64_t edge_ordinal = 0;
    bool row_done = false;
    bool provider_event = false;
    if (auto* response = dynamic_cast<CoreSynapseResponseEvent*>(event)) {
        provider_event = true;
        timestep = response->timestep;
        source_neuron = response->source_neuron;
        source_event_seq = response->source_event_seq;
        edge_ordinal = response->edge_ordinal;
        accepted = pipeline_.acceptSynapseResponse(SynapseResponse{
            response->timestep, response->source_neuron, response->source_event_seq,
            response->post_neuron, response->edge_ordinal, response->weight,
            response->row_complete, response->row_edge_count});
    } else if (auto* done = dynamic_cast<CoreRowDoneEvent*>(event)) {
        provider_event = true;
        timestep = done->timestep;
        source_neuron = done->source_neuron;
        source_event_seq = done->source_event_seq;
        row_done = true;
        accepted = pipeline_.acceptRowDone(RowDone{
            done->timestep, done->source_neuron, done->source_event_seq, done->edge_count});
    }
    delete event;
    if (provider_event) {
        sendProviderAck_(timestep, source_neuron, source_event_seq, edge_ordinal, row_done,
                         accepted, !accepted);
    } else {
        out_.fatal(CALL_INFO, -1, "SnnCoreV5 received an unexpected row-provider event\n");
    }
}

void SnnCoreV5::sendSpikeAck_(const CoreSpikeEvent& spike, bool accepted, bool retryable) {
    auto* ack = new CoreSpikeAckEvent();
    ack->timestep = spike.timestep;
    ack->source_neuron = spike.source_neuron;
    ack->source_event_seq = spike.source_event_seq;
    ack->accepted = accepted;
    ack->retryable = retryable;
    spike_ack_link_->send(ack);
}

void SnnCoreV5::sendProviderAck_(std::uint64_t timestep, std::uint32_t source_neuron,
                                 std::uint64_t source_event_seq, std::uint64_t edge_ordinal,
                                 bool row_done, bool accepted, bool retryable) {
    auto* ack = new CoreProviderAckEvent();
    ack->timestep = timestep;
    ack->source_neuron = source_neuron;
    ack->source_event_seq = source_event_seq;
    ack->edge_ordinal = edge_ordinal;
    ack->row_done = row_done;
    ack->accepted = accepted;
    ack->retryable = retryable;
    row_provider_link_->send(ack);
}

void SnnCoreV5::sendRequests_() {
    for (const auto& request : pipeline_.takeRowRequests()) {
        auto* event = new CoreRowRequestEvent();
        event->timestep = request.timestep;
        event->source_neuron = request.source_neuron;
        event->source_event_seq = request.source_event_seq;
        event->row_id = request.row_id;
        row_provider_link_->send(event);
    }
}

void SnnCoreV5::sendReleasedSpikes_() {
    for (const auto& spike : pipeline_.takeReleasedSpikes()) {
        auto* event = new CoreSpikeEvent();
        event->timestep = spike.timestep;
        event->source_neuron = spike.post_neuron;
        event->target_neuron = spike.post_neuron;
        event->source_event_seq = spike.source_event_seq;
        spike_out_link_->send(event);
    }
}

void SnnCoreV5::sendStatus_(CoreControlOp operation) {
    auto* status = new CoreStatusEvent();
    const auto& stats = pipeline_.stats();
    status->operation = operation;
    status->timestep = pipeline_.timestep();
    status->core_cycles = stats.cycles;
    status->ingress_accepted = stats.ingress_accepted;
    status->row_requests = stats.row_requests;
    status->synapse_issued = stats.synapse_issued;
    status->retire_retired = stats.retire_retired;
    status->accumulator_updates = stats.accumulator_updates;
    status->neurons_evaluated = stats.neurons_evaluated;
    status->neurons_fired = stats.neurons_fired;
    status->held_released = stats.held_released;
    status->ingress_full_cycles = stats.ingress_full_cycles;
    status->row_stall_cycles = stats.row_stall_cycles;
    status->synapse_stall_cycles = stats.synapse_stall_cycles;
    status->retire_stall_cycles = stats.retire_stall_cycles;
    status->accumulator_stall_cycles = stats.accumulator_stall_cycles;
    status->held_full_cycles = stats.held_full_cycles;
    status->functional_hash = pipeline_.functionalHash();
    const auto& state_storage = pipeline_.storage().stats(::SnnDL::v5::AddressSpaceId::CoreState);
    const auto& delta_storage = pipeline_.storage().stats(::SnnDL::v5::AddressSpaceId::CoreDelta);
    const auto& index_storage = pipeline_.storage().stats(::SnnDL::v5::AddressSpaceId::CoreIndex);
    const auto& route_storage = pipeline_.storage().stats(::SnnDL::v5::AddressSpaceId::PeRoute);
    status->storage_state_reads = state_storage.reads;
    status->storage_state_writes = state_storage.writes;
    status->storage_delta_reads = delta_storage.reads;
    status->storage_delta_writes = delta_storage.writes;
    status->storage_index_reads = index_storage.reads;
    status->storage_route_reads = route_storage.reads;
    status->core_elapsed_ns = static_cast<std::uint64_t>(getCurrentSimTimeNano()) - timestep_start_ns_;
    last_elapsed_ns_ = status->core_elapsed_ns;
    const auto copyStage = [](const CorePipelineStats::Stage& from, CoreStageCounters& to) {
        to.accepted = from.accepted;
        to.issued = from.issued;
        to.completed = from.completed;
        to.occupancy = from.occupancy;
        to.busy_cycles = from.busy_cycles;
        to.full_cycles = from.full_cycles;
        to.stall_cycles = from.stall_cycles;
    };
    copyStage(stats.ingress, status->ingress);
    copyStage(stats.row_lookup, status->row_lookup);
    copyStage(stats.synapse, status->synapse);
    copyStage(stats.retire, status->retire);
    copyStage(stats.accumulator, status->accumulator);
    copyStage(stats.neuron, status->neuron);
    copyStage(stats.held_spike, status->held_spike);
    status_link_->send(status);
}

bool SnnCoreV5::clockTick_(SST::Cycle_t) {
    pipeline_.tick();
    sendRequests_();
    if (pipeline_.readyToCommit() && !commit_ready_sent_) {
        commit_ready_sent_ = true;
        sendStatus_(CoreControlOp::CommitReady);
    }
    return false;
}

void SnnCoreV5::publishStatistics_() {
    if (stats_published_) return;
    stats_published_ = true;
    const auto& s = pipeline_.stats();
    const auto add = [this](const char* name, std::uint64_t value) {
        auto it = statistics_.find(name);
        if (it != statistics_.end() && it->second) it->second->addData(value);
    };
    const auto publishStage = [&add](const char* prefix, const CorePipelineStats::Stage& stage) {
        std::string name(prefix);
        add((name + ".accepted").c_str(), stage.accepted);
        add((name + ".issued").c_str(), stage.issued);
        add((name + ".completed").c_str(), stage.completed);
        add((name + ".occupancy").c_str(), stage.occupancy);
        add((name + ".busy_cycles").c_str(), stage.busy_cycles);
        add((name + ".full_cycles").c_str(), stage.full_cycles);
        add((name + ".stall_cycles").c_str(), stage.stall_cycles);
    };
    publishStage("core.ingress", s.ingress);
    publishStage("core.row_lookup", s.row_lookup);
    publishStage("core.synapse", s.synapse);
    publishStage("core.retire", s.retire);
    publishStage("core.accumulator", s.accumulator);
    add("core.neuron.evaluated", s.neurons_evaluated);
    add("core.neuron.fired", s.neurons_fired);
    add("core.neuron.occupancy", s.neuron.occupancy);
    add("core.neuron.busy_cycles", s.neuron.busy_cycles);
    add("core.neuron.stall_cycles", s.neuron.stall_cycles);
    add("core.held_spike.accepted", s.held_spike.accepted);
    add("core.held_spike.released", s.held_released);
    add("core.held_spike.occupancy", s.held_spike.occupancy);
    add("core.held_spike.full_cycles", s.held_spike.full_cycles);
    add("core.held_spike.stall_cycles", s.held_spike.stall_cycles);
    add(kStatisticNameStorageStateReads,
        pipeline_.storage().stats(::SnnDL::v5::AddressSpaceId::CoreState).reads);
    add(kStatisticNameStorageDeltaReads,
        pipeline_.storage().stats(::SnnDL::v5::AddressSpaceId::CoreDelta).reads);
    add("timestep.core_elapsed_ns", last_elapsed_ns_);
}

void SnnCoreV5::finish() {
    publishStatistics_();
    out_.verbose(CALL_INFO, 1, 0,
                 "[snndl-v5-core] core=%u cycles=%" PRIu64 " rows=%" PRIu64
                 " synapses=%" PRIu64 " neurons=%" PRIu64 " hash=%" PRIu64 "\n",
                 core_id_, pipeline_.stats().cycles, pipeline_.stats().row_requests,
                 pipeline_.stats().synapse_issued, pipeline_.stats().neurons_evaluated,
                 pipeline_.functionalHash());
}

} // namespace v5
} // namespace SnnDL
} // namespace SST
