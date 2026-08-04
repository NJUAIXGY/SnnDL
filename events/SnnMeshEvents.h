#ifndef SST_SNN_DL_MESH_EVENTS_H
#define SST_SNN_DL_MESH_EVENTS_H

#include <cstdint>

#include <sst/core/event.h>
#include <sst/core/serialization/serialize.h>

namespace SST { namespace SnnDL {

class MeshSpikeEvent final : public SST::Event {
public:
    static constexpr std::uint16_t kAllCores = 0xffffu;

    std::uint64_t timestep = 0;
    std::uint32_t source_neuron = 0;
    std::uint32_t source_pe = 0;
    std::uint16_t source_core = 0;
    std::uint16_t destination_core = kAllCores;
    std::uint32_t destination_pe = 0;
    std::uint64_t source_event_seq = 0;
    std::uint64_t logical_deliveries = 1;
    std::uint16_t hop_count = 0;

    MeshSpikeEvent() = default;

    MeshSpikeEvent* clone() override {
        auto* copy = new MeshSpikeEvent();
        copy->timestep = timestep;
        copy->source_neuron = source_neuron;
        copy->source_pe = source_pe;
        copy->source_core = source_core;
        copy->destination_core = destination_core;
        copy->destination_pe = destination_pe;
        copy->source_event_seq = source_event_seq;
        copy->logical_deliveries = logical_deliveries;
        copy->hop_count = hop_count;
        return copy;
    }

    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(timestep);
        SST_SER(source_neuron);
        SST_SER(source_pe);
        SST_SER(source_core);
        SST_SER(destination_core);
        SST_SER(destination_pe);
        SST_SER(source_event_seq);
        SST_SER(logical_deliveries);
        SST_SER(hop_count);
    }

private:
    ImplementSerializable(SST::SnnDL::MeshSpikeEvent)
};

class MeshMemoryRequestEvent final : public SST::Event {
public:
    std::uint64_t request_id = 0;
    std::uint64_t timestep = 0;
    std::uint32_t source_pe = 0;
    std::uint64_t address = 0;
    std::uint32_t bytes = 0;
    float value = 0.0f;

    MeshMemoryRequestEvent() = default;

    MeshMemoryRequestEvent* clone() override {
        auto* copy = new MeshMemoryRequestEvent();
        copy->request_id = request_id;
        copy->timestep = timestep;
        copy->source_pe = source_pe;
        copy->address = address;
        copy->bytes = bytes;
        copy->value = value;
        return copy;
    }

    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(request_id);
        SST_SER(timestep);
        SST_SER(source_pe);
        SST_SER(address);
        SST_SER(bytes);
        SST_SER(value);
    }

private:
    ImplementSerializable(SST::SnnDL::MeshMemoryRequestEvent)
};

class MeshMemoryResponseEvent final : public SST::Event {
public:
    std::uint64_t request_id = 0;
    std::uint64_t timestep = 0;
    std::uint32_t source_pe = 0;
    std::uint64_t address = 0;
    float value = 0.0f;

    MeshMemoryResponseEvent() = default;

    MeshMemoryResponseEvent* clone() override {
        auto* copy = new MeshMemoryResponseEvent();
        copy->request_id = request_id;
        copy->timestep = timestep;
        copy->source_pe = source_pe;
        copy->address = address;
        copy->value = value;
        return copy;
    }

    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(request_id);
        SST_SER(timestep);
        SST_SER(source_pe);
        SST_SER(address);
        SST_SER(value);
    }

private:
    ImplementSerializable(SST::SnnDL::MeshMemoryResponseEvent)
};

}} // namespace SST::SnnDL

#endif
