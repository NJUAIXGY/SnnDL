#ifndef SST_SNN_DL_V5_CORE_EVENTS_H
#define SST_SNN_DL_V5_CORE_EVENTS_H

#include <cstdint>

#include <sst/core/event.h>
#include <sst/core/serialization/serialize.h>

namespace SST {
namespace SnnDL {
namespace v5 {

enum class CoreControlOp : std::uint8_t {
    Start = 0,
    SealIngress,
    Commit,
    Abort,
    CommitReady,
    CommitDone,
};

class CoreControlEvent final : public SST::Event {
public:
    CoreControlOp operation = CoreControlOp::Abort;
    std::uint64_t timestep = 0;

    CoreControlEvent() = default;
    CoreControlEvent(CoreControlOp op, std::uint64_t step) : operation(op), timestep(step) {}

    CoreControlEvent* clone() override {
        auto* copy = new CoreControlEvent();
        copy->operation = operation;
        copy->timestep = timestep;
        return copy;
    }

    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(operation);
        SST_SER(timestep);
    }

private:
    ImplementSerializable(SST::SnnDL::v5::CoreControlEvent)
};

class CoreSpikeEvent final : public SST::Event {
public:
    std::uint64_t timestep = 0;
    std::uint32_t source_neuron = 0;
    std::uint32_t target_neuron = 0;
    std::uint64_t source_event_seq = 0;

    CoreSpikeEvent() = default;

    CoreSpikeEvent* clone() override {
        auto* copy = new CoreSpikeEvent();
        copy->timestep = timestep;
        copy->source_neuron = source_neuron;
        copy->target_neuron = target_neuron;
        copy->source_event_seq = source_event_seq;
        return copy;
    }

    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(timestep);
        SST_SER(source_neuron);
        SST_SER(target_neuron);
        SST_SER(source_event_seq);
    }

private:
    ImplementSerializable(SST::SnnDL::v5::CoreSpikeEvent)
};

class CoreSpikeAckEvent final : public SST::Event {
public:
    std::uint64_t timestep = 0;
    std::uint32_t source_neuron = 0;
    std::uint64_t source_event_seq = 0;
    bool accepted = false;
    bool retryable = true;

    CoreSpikeAckEvent() = default;
    CoreSpikeAckEvent* clone() override {
        auto* copy = new CoreSpikeAckEvent();
        copy->timestep = timestep;
        copy->source_neuron = source_neuron;
        copy->source_event_seq = source_event_seq;
        copy->accepted = accepted;
        copy->retryable = retryable;
        return copy;
    }
    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(timestep);
        SST_SER(source_neuron);
        SST_SER(source_event_seq);
        SST_SER(accepted);
        SST_SER(retryable);
    }

private:
    ImplementSerializable(SST::SnnDL::v5::CoreSpikeAckEvent)
};

class CoreRowRequestEvent final : public SST::Event {
public:
    std::uint64_t timestep = 0;
    std::uint32_t source_neuron = 0;
    std::uint64_t source_event_seq = 0;
    std::uint64_t row_id = 0;

    CoreRowRequestEvent() = default;

    CoreRowRequestEvent* clone() override {
        auto* copy = new CoreRowRequestEvent();
        copy->timestep = timestep;
        copy->source_neuron = source_neuron;
        copy->source_event_seq = source_event_seq;
        copy->row_id = row_id;
        return copy;
    }

    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(timestep);
        SST_SER(source_neuron);
        SST_SER(source_event_seq);
        SST_SER(row_id);
    }

private:
    ImplementSerializable(SST::SnnDL::v5::CoreRowRequestEvent)
};

class CoreSynapseResponseEvent final : public SST::Event {
public:
    std::uint64_t timestep = 0;
    std::uint32_t source_neuron = 0;
    std::uint64_t source_event_seq = 0;
    std::uint32_t post_neuron = 0;
    std::uint64_t edge_ordinal = 0;
    float weight = 0.0f;
    bool row_complete = false;
    std::uint32_t row_edge_count = 0;

    CoreSynapseResponseEvent() = default;

    CoreSynapseResponseEvent* clone() override {
        auto* copy = new CoreSynapseResponseEvent();
        copy->timestep = timestep;
        copy->source_neuron = source_neuron;
        copy->source_event_seq = source_event_seq;
        copy->post_neuron = post_neuron;
        copy->edge_ordinal = edge_ordinal;
        copy->weight = weight;
        copy->row_complete = row_complete;
        copy->row_edge_count = row_edge_count;
        return copy;
    }

    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(timestep);
        SST_SER(source_neuron);
        SST_SER(source_event_seq);
        SST_SER(post_neuron);
        SST_SER(edge_ordinal);
        SST_SER(weight);
        SST_SER(row_complete);
        SST_SER(row_edge_count);
    }

private:
    ImplementSerializable(SST::SnnDL::v5::CoreSynapseResponseEvent)
};

class CoreRowDoneEvent final : public SST::Event {
public:
    std::uint64_t timestep = 0;
    std::uint32_t source_neuron = 0;
    std::uint64_t source_event_seq = 0;
    std::uint32_t edge_count = 0;

    CoreRowDoneEvent() = default;

    CoreRowDoneEvent* clone() override {
        auto* copy = new CoreRowDoneEvent();
        copy->timestep = timestep;
        copy->source_neuron = source_neuron;
        copy->source_event_seq = source_event_seq;
        copy->edge_count = edge_count;
        return copy;
    }

    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(timestep);
        SST_SER(source_neuron);
        SST_SER(source_event_seq);
        SST_SER(edge_count);
    }

private:
    ImplementSerializable(SST::SnnDL::v5::CoreRowDoneEvent)
};

class CoreProviderAckEvent final : public SST::Event {
public:
    std::uint64_t timestep = 0;
    std::uint32_t source_neuron = 0;
    std::uint64_t source_event_seq = 0;
    std::uint64_t edge_ordinal = 0;
    bool row_done = false;
    bool accepted = false;
    bool retryable = true;

    CoreProviderAckEvent() = default;
    CoreProviderAckEvent* clone() override {
        auto* copy = new CoreProviderAckEvent();
        copy->timestep = timestep;
        copy->source_neuron = source_neuron;
        copy->source_event_seq = source_event_seq;
        copy->edge_ordinal = edge_ordinal;
        copy->row_done = row_done;
        copy->accepted = accepted;
        copy->retryable = retryable;
        return copy;
    }
    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(timestep);
        SST_SER(source_neuron);
        SST_SER(source_event_seq);
        SST_SER(edge_ordinal);
        SST_SER(row_done);
        SST_SER(accepted);
        SST_SER(retryable);
    }

private:
    ImplementSerializable(SST::SnnDL::v5::CoreProviderAckEvent)
};

struct CoreStageCounters {
    std::uint64_t accepted = 0;
    std::uint64_t issued = 0;
    std::uint64_t completed = 0;
    std::uint64_t occupancy = 0;
    std::uint64_t busy_cycles = 0;
    std::uint64_t full_cycles = 0;
    std::uint64_t stall_cycles = 0;

    void serialize_order(SST::Core::Serialization::serializer& ser) {
        SST_SER(accepted);
        SST_SER(issued);
        SST_SER(completed);
        SST_SER(occupancy);
        SST_SER(busy_cycles);
        SST_SER(full_cycles);
        SST_SER(stall_cycles);
    }
};

class CoreStatusEvent final : public SST::Event {
public:
    CoreControlOp operation = CoreControlOp::Abort;
    std::uint64_t timestep = 0;
    std::uint64_t core_cycles = 0;
    std::uint64_t ingress_accepted = 0;
    std::uint64_t row_requests = 0;
    std::uint64_t synapse_issued = 0;
    std::uint64_t retire_retired = 0;
    std::uint64_t accumulator_updates = 0;
    std::uint64_t neurons_evaluated = 0;
    std::uint64_t neurons_fired = 0;
    std::uint64_t held_released = 0;
    std::uint64_t ingress_full_cycles = 0;
    std::uint64_t row_stall_cycles = 0;
    std::uint64_t synapse_stall_cycles = 0;
    std::uint64_t retire_stall_cycles = 0;
    std::uint64_t accumulator_stall_cycles = 0;
    std::uint64_t held_full_cycles = 0;
    std::uint64_t storage_state_reads = 0;
    std::uint64_t storage_state_writes = 0;
    std::uint64_t storage_delta_reads = 0;
    std::uint64_t storage_delta_writes = 0;
    std::uint64_t storage_index_reads = 0;
    std::uint64_t storage_route_reads = 0;
    std::uint64_t functional_hash = 0;
    std::uint64_t core_elapsed_ns = 0;
    CoreStageCounters ingress;
    CoreStageCounters row_lookup;
    CoreStageCounters synapse;
    CoreStageCounters retire;
    CoreStageCounters accumulator;
    CoreStageCounters neuron;
    CoreStageCounters held_spike;

    CoreStatusEvent* clone() override {
        auto* copy = new CoreStatusEvent();
        copy->operation = operation;
        copy->timestep = timestep;
        copy->core_cycles = core_cycles;
        copy->ingress_accepted = ingress_accepted;
        copy->row_requests = row_requests;
        copy->synapse_issued = synapse_issued;
        copy->retire_retired = retire_retired;
        copy->accumulator_updates = accumulator_updates;
        copy->neurons_evaluated = neurons_evaluated;
        copy->neurons_fired = neurons_fired;
        copy->held_released = held_released;
        copy->ingress_full_cycles = ingress_full_cycles;
        copy->row_stall_cycles = row_stall_cycles;
        copy->synapse_stall_cycles = synapse_stall_cycles;
        copy->retire_stall_cycles = retire_stall_cycles;
        copy->accumulator_stall_cycles = accumulator_stall_cycles;
        copy->held_full_cycles = held_full_cycles;
        copy->storage_state_reads = storage_state_reads;
        copy->storage_state_writes = storage_state_writes;
        copy->storage_delta_reads = storage_delta_reads;
        copy->storage_delta_writes = storage_delta_writes;
        copy->storage_index_reads = storage_index_reads;
        copy->storage_route_reads = storage_route_reads;
        copy->functional_hash = functional_hash;
        copy->core_elapsed_ns = core_elapsed_ns;
        copy->ingress = ingress;
        copy->row_lookup = row_lookup;
        copy->synapse = synapse;
        copy->retire = retire;
        copy->accumulator = accumulator;
        copy->neuron = neuron;
        copy->held_spike = held_spike;
        return copy;
    }

    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(operation);
        SST_SER(timestep);
        SST_SER(core_cycles);
        SST_SER(ingress_accepted);
        SST_SER(row_requests);
        SST_SER(synapse_issued);
        SST_SER(retire_retired);
        SST_SER(accumulator_updates);
        SST_SER(neurons_evaluated);
        SST_SER(neurons_fired);
        SST_SER(held_released);
        SST_SER(ingress_full_cycles);
        SST_SER(row_stall_cycles);
        SST_SER(synapse_stall_cycles);
        SST_SER(retire_stall_cycles);
        SST_SER(accumulator_stall_cycles);
        SST_SER(held_full_cycles);
        SST_SER(storage_state_reads);
        SST_SER(storage_state_writes);
        SST_SER(storage_delta_reads);
        SST_SER(storage_delta_writes);
        SST_SER(storage_index_reads);
        SST_SER(storage_route_reads);
        SST_SER(functional_hash);
        SST_SER(core_elapsed_ns);
        SST_SER(ingress);
        SST_SER(row_lookup);
        SST_SER(synapse);
        SST_SER(retire);
        SST_SER(accumulator);
        SST_SER(neuron);
        SST_SER(held_spike);
    }

private:
    ImplementSerializable(SST::SnnDL::v5::CoreStatusEvent)
};

} // namespace v5
} // namespace SnnDL
} // namespace SST

#endif
