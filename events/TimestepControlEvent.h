#ifndef SST_SNN_DL_TIMESTEP_CONTROL_EVENT_H
#define SST_SNN_DL_TIMESTEP_CONTROL_EVENT_H

#include "api/TimestepTypes.h"

#include <cstdint>

#include <sst/core/event.h>
#include <sst/core/serialization/serialize.h>

namespace SST {
namespace SnnDL {

enum class TimestepControlOp : std::uint8_t {
    BootReady = 0,
    Start,
    EgressClosed,
    IngressProgress,
    SealIngress,
    CommitReady,
    Commit,
    CommitDone,
    Abort
};

class TimestepControlEvent : public SST::Event {
public:
    TimestepControlOp operation = TimestepControlOp::Abort;
    TimestepId timestep = 0;
    std::uint32_t source_pe = 0;
    std::uint64_t logical_count = 0;
    std::uint64_t physical_count = 0;
    std::uint64_t memory_count = 0;
    std::uint64_t memory_response_count = 0;
    std::uint64_t storage_hits = 0;
    std::uint64_t synapse_count = 0;
    std::uint64_t retired_count = 0;
    std::uint64_t fired_count = 0;
    std::uint64_t neuron_count = 0;
    std::uint64_t cycle_count = 0;
    std::uint64_t state_hash = 0;
    std::uint64_t spike_hash = 0;
    std::uint64_t queue_drops = 0;
    std::uint64_t backpressure_events = 0;
    std::uint64_t stale_events = 0;
    std::uint64_t future_events = 0;
    std::uint64_t post_seal_events = 0;
    std::uint64_t tracked_tokens = 0;
    std::uint64_t queue_depth = 0;
    std::uint64_t blocked_routes = 0;
    std::int32_t error_code = 0;

    TimestepControlEvent() = default;

    TimestepControlEvent* clone() override {
        auto* copy = new TimestepControlEvent();
        copy->operation = operation;
        copy->timestep = timestep;
        copy->source_pe = source_pe;
        copy->logical_count = logical_count;
        copy->physical_count = physical_count;
        copy->memory_count = memory_count;
        copy->memory_response_count = memory_response_count;
        copy->storage_hits = storage_hits;
        copy->synapse_count = synapse_count;
        copy->retired_count = retired_count;
        copy->fired_count = fired_count;
        copy->neuron_count = neuron_count;
        copy->cycle_count = cycle_count;
        copy->state_hash = state_hash;
        copy->spike_hash = spike_hash;
        copy->queue_drops = queue_drops;
        copy->backpressure_events = backpressure_events;
        copy->stale_events = stale_events;
        copy->future_events = future_events;
        copy->post_seal_events = post_seal_events;
        copy->tracked_tokens = tracked_tokens;
        copy->queue_depth = queue_depth;
        copy->blocked_routes = blocked_routes;
        copy->error_code = error_code;
        return copy;
    }

    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(operation);
        SST_SER(timestep);
        SST_SER(source_pe);
        SST_SER(logical_count);
        SST_SER(physical_count);
        SST_SER(memory_count);
        SST_SER(memory_response_count);
        SST_SER(storage_hits);
        SST_SER(synapse_count);
        SST_SER(retired_count);
        SST_SER(fired_count);
        SST_SER(neuron_count);
        SST_SER(cycle_count);
        SST_SER(state_hash);
        SST_SER(spike_hash);
        SST_SER(queue_drops);
        SST_SER(backpressure_events);
        SST_SER(stale_events);
        SST_SER(future_events);
        SST_SER(post_seal_events);
        SST_SER(tracked_tokens);
        SST_SER(queue_depth);
        SST_SER(blocked_routes);
        SST_SER(error_code);
    }

private:
    ImplementSerializable(SST::SnnDL::TimestepControlEvent)
};

} // namespace SnnDL
} // namespace SST

#endif
