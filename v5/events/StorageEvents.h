#ifndef SST_SNN_DL_V5_STORAGE_EVENTS_H
#define SST_SNN_DL_V5_STORAGE_EVENTS_H

#include <cstdint>
#include <vector>

#include <sst/core/event.h>
#include <sst/core/serialization/serialize.h>

namespace SST {
namespace SnnDL {
namespace v5 {

class SramRequestEvent final : public SST::Event {
public:
    std::uint64_t request_id = 0;
    std::uint64_t address = 0;
    std::vector<std::uint8_t> data;
    bool write = false;

    SramRequestEvent() = default;
    SramRequestEvent* clone() override {
        auto* copy = new SramRequestEvent();
        copy->request_id = request_id;
        copy->address = address;
        copy->data = data;
        copy->write = write;
        return copy;
    }
    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(request_id);
        SST_SER(address);
        SST_SER(data);
        SST_SER(write);
    }

private:
    ImplementSerializable(SST::SnnDL::v5::SramRequestEvent)
};

class SramResponseEvent final : public SST::Event {
public:
    std::uint64_t request_id = 0;
    std::uint64_t address = 0;
    std::uint64_t service_cycle = 0;
    std::uint64_t completion_cycle = 0;
    std::uint32_t bank = 0;
    std::vector<std::uint8_t> data;
    bool accepted = false;
    bool completed = false;
    bool retryable = false;
    std::uint8_t reject_reason = 0;

    SramResponseEvent() = default;
    SramResponseEvent* clone() override {
        auto* copy = new SramResponseEvent();
        copy->request_id = request_id;
        copy->address = address;
        copy->service_cycle = service_cycle;
        copy->completion_cycle = completion_cycle;
        copy->bank = bank;
        copy->data = data;
        copy->accepted = accepted;
        copy->completed = completed;
        copy->retryable = retryable;
        copy->reject_reason = reject_reason;
        return copy;
    }
    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(request_id);
        SST_SER(address);
        SST_SER(service_cycle);
        SST_SER(completion_cycle);
        SST_SER(bank);
        SST_SER(data);
        SST_SER(accepted);
        SST_SER(completed);
        SST_SER(retryable);
        SST_SER(reject_reason);
    }

private:
    ImplementSerializable(SST::SnnDL::v5::SramResponseEvent)
};

// A DMA descriptor is an address-only contract.  The engine never interprets
// neuron, route, or GAS state; owners and address spaces are retained solely
// for provenance and fail-closed validation at the engine boundary.
class DmaDescriptorEvent final : public SST::Event {
public:
    std::uint64_t descriptor_id = 0;
    std::uint64_t timestep_id = 0;
    std::uint8_t source_space = 0;
    std::uint8_t destination_space = 0;
    std::uint32_t source_owner = 0;
    std::uint32_t destination_owner = 0;
    std::uint64_t source_address = 0;
    std::uint64_t destination_address = 0;
    std::uint64_t bytes = 0;
    std::uint64_t burst_bytes = 0;
    std::uint64_t completion_token = 0;

    DmaDescriptorEvent() = default;
    DmaDescriptorEvent* clone() override {
        auto* copy = new DmaDescriptorEvent();
        copy->descriptor_id = descriptor_id;
        copy->timestep_id = timestep_id;
        copy->source_space = source_space;
        copy->destination_space = destination_space;
        copy->source_owner = source_owner;
        copy->destination_owner = destination_owner;
        copy->source_address = source_address;
        copy->destination_address = destination_address;
        copy->bytes = bytes;
        copy->burst_bytes = burst_bytes;
        copy->completion_token = completion_token;
        return copy;
    }
    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(descriptor_id);
        SST_SER(timestep_id);
        SST_SER(source_space);
        SST_SER(destination_space);
        SST_SER(source_owner);
        SST_SER(destination_owner);
        SST_SER(source_address);
        SST_SER(destination_address);
        SST_SER(bytes);
        SST_SER(burst_bytes);
        SST_SER(completion_token);
    }

private:
    ImplementSerializable(SST::SnnDL::v5::DmaDescriptorEvent)
};

class DmaCompletionEvent final : public SST::Event {
public:
    std::uint64_t descriptor_id = 0;
    std::uint64_t timestep_id = 0;
    std::uint64_t completion_token = 0;
    std::uint64_t bytes = 0;
    std::uint64_t bursts = 0;
    std::uint64_t elapsed_cycles = 0;
    bool accepted = false;
    bool completed = false;
    bool retryable = false;
    std::uint8_t error = 0;

    DmaCompletionEvent() = default;
    DmaCompletionEvent* clone() override {
        auto* copy = new DmaCompletionEvent();
        copy->descriptor_id = descriptor_id;
        copy->timestep_id = timestep_id;
        copy->completion_token = completion_token;
        copy->bytes = bytes;
        copy->bursts = bursts;
        copy->elapsed_cycles = elapsed_cycles;
        copy->accepted = accepted;
        copy->completed = completed;
        copy->retryable = retryable;
        copy->error = error;
        return copy;
    }
    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(descriptor_id);
        SST_SER(timestep_id);
        SST_SER(completion_token);
        SST_SER(bytes);
        SST_SER(bursts);
        SST_SER(elapsed_cycles);
        SST_SER(accepted);
        SST_SER(completed);
        SST_SER(retryable);
        SST_SER(error);
    }

private:
    ImplementSerializable(SST::SnnDL::v5::DmaCompletionEvent)
};

} // namespace v5
} // namespace SnnDL
} // namespace SST

#endif
