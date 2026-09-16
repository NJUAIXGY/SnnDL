#ifndef SST_SNN_DL_V5_TRACE_EVENTS_H
#define SST_SNN_DL_V5_TRACE_EVENTS_H

#include <cstdint>
#include <string>
#include <vector>

#include <sst/core/event.h>
#include <sst/core/serialization/serialize.h>

namespace SST { namespace SnnDL { namespace v5 {

enum class TraceControlOp : std::uint8_t {
    Start = 0,
    Abort = 1,
};

enum class TraceStatusOp : std::uint8_t {
    Ready = 0,
    Drained = 1,
    Failed = 2,
};

enum class TraceNoCMulticastMode : std::uint8_t {
    Unicast = 0,
    SourceReplication = 1,
    NativeTree = 2,
};

class TraceNoCInjectionV5Event final : public SST::Event {
public:
    static constexpr std::uint16_t kFormatVersion = 1;
    std::uint16_t format_version = kFormatVersion;
    std::string event_id;
    std::uint64_t event_token = 0;
    std::uint64_t timestep = 0;
    std::uint32_t source_pe = 0;
    std::uint32_t source_core = 0;
    std::uint64_t source_neuron = 0;
    std::uint64_t source_event_seq = 0;
    std::uint64_t route_id = 0;
    std::uint32_t payload_bytes = 0;
    std::uint32_t virtual_network = 0;
    TraceNoCMulticastMode multicast_mode = TraceNoCMulticastMode::SourceReplication;
    std::vector<std::uint32_t> destination_pes;
    std::vector<std::uint64_t> destination_core_masks;

    TraceNoCInjectionV5Event* clone() override { return new TraceNoCInjectionV5Event(*this); }
    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(format_version); SST_SER(event_id); SST_SER(event_token); SST_SER(timestep);
        SST_SER(source_pe); SST_SER(source_core); SST_SER(source_neuron); SST_SER(source_event_seq);
        SST_SER(route_id); SST_SER(payload_bytes); SST_SER(virtual_network); SST_SER(multicast_mode);
        SST_SER(destination_pes); SST_SER(destination_core_masks);
    }
private:
    ImplementSerializable(SST::SnnDL::v5::TraceNoCInjectionV5Event)
};

class TraceNoCInjectionAckV5Event final : public SST::Event {
public:
    std::uint16_t format_version = TraceNoCInjectionV5Event::kFormatVersion;
    std::string event_id;
    std::uint64_t event_token = 0;
    bool accepted = false;
    bool retryable = true;
    std::uint32_t physical_packets = 0;

    TraceNoCInjectionAckV5Event* clone() override { return new TraceNoCInjectionAckV5Event(*this); }
    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser); SST_SER(format_version); SST_SER(event_id); SST_SER(event_token);
        SST_SER(accepted); SST_SER(retryable); SST_SER(physical_packets);
    }
private:
    ImplementSerializable(SST::SnnDL::v5::TraceNoCInjectionAckV5Event)
};

class TraceNoCDeliveryV5Event final : public SST::Event {
public:
    std::uint16_t format_version = TraceNoCInjectionV5Event::kFormatVersion;
    std::string event_id;
    std::uint64_t event_token = 0;
    std::uint64_t timestep = 0;
    std::uint32_t source_pe = 0;
    std::uint32_t source_core = 0;
    std::uint64_t source_neuron = 0;
    std::uint64_t source_event_seq = 0;
    std::uint32_t destination_pe = 0;
    std::uint32_t destination_core = 0;
    std::uint32_t payload_bytes = 0;

    TraceNoCDeliveryV5Event* clone() override { return new TraceNoCDeliveryV5Event(*this); }
    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser); SST_SER(format_version); SST_SER(event_id); SST_SER(event_token);
        SST_SER(timestep); SST_SER(source_pe); SST_SER(source_core); SST_SER(source_neuron);
        SST_SER(source_event_seq); SST_SER(destination_pe); SST_SER(destination_core); SST_SER(payload_bytes);
    }
private:
    ImplementSerializable(SST::SnnDL::v5::TraceNoCDeliveryV5Event)
};

class TraceNoCDeliveryAckV5Event final : public SST::Event {
public:
    std::uint16_t format_version = TraceNoCInjectionV5Event::kFormatVersion;
    std::string event_id;
    std::uint64_t event_token = 0;
    std::uint32_t destination_pe = 0;
    std::uint32_t destination_core = 0;
    bool accepted = false;
    bool retryable = true;

    TraceNoCDeliveryAckV5Event* clone() override { return new TraceNoCDeliveryAckV5Event(*this); }
    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser); SST_SER(format_version); SST_SER(event_id); SST_SER(event_token);
        SST_SER(destination_pe); SST_SER(destination_core); SST_SER(accepted); SST_SER(retryable);
    }
private:
    ImplementSerializable(SST::SnnDL::v5::TraceNoCDeliveryAckV5Event)
};

class TraceControlEvent final : public SST::Event {
public:
    TraceControlOp operation = TraceControlOp::Abort;
    std::uint32_t source_id = 0;
    std::uint64_t timestep = 0;

    TraceControlEvent() = default;
    TraceControlEvent(TraceControlOp op, std::uint32_t id, std::uint64_t step)
        : operation(op), source_id(id), timestep(step) {}

    TraceControlEvent* clone() override {
        return new TraceControlEvent(*this);
    }

    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(operation);
        SST_SER(source_id);
        SST_SER(timestep);
    }

private:
    ImplementSerializable(SST::SnnDL::v5::TraceControlEvent)
};

class TraceStatusEvent final : public SST::Event {
public:
    TraceStatusOp operation = TraceStatusOp::Failed;
    std::uint32_t source_id = 0;
    std::uint64_t expected_records = 0;
    std::uint64_t eligible = 0;
    std::uint64_t offered = 0;
    std::uint64_t injected = 0;
    std::uint64_t completed = 0;
    std::uint64_t request_bytes = 0;
    std::uint64_t outstanding = 0;
    std::uint64_t cycle = 0;
    std::string error;

    TraceStatusEvent() = default;
    TraceStatusEvent* clone() override {
        return new TraceStatusEvent(*this);
    }

    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(operation);
        SST_SER(source_id);
        SST_SER(expected_records);
        SST_SER(eligible);
        SST_SER(offered);
        SST_SER(injected);
        SST_SER(completed);
        SST_SER(request_bytes);
        SST_SER(outstanding);
        SST_SER(cycle);
        SST_SER(error);
    }

private:
    ImplementSerializable(SST::SnnDL::v5::TraceStatusEvent)
};

}}}

#endif
