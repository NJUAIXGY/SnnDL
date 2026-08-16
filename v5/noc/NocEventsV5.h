#ifndef SST_SNN_DL_V5_NOC_EVENTS_V5_H
#define SST_SNN_DL_V5_NOC_EVENTS_V5_H

#include <cstdint>
#include <sst/core/event.h>
#include <sst/core/serialization/serialize.h>

#include "v5/events/CoreEvents.h"

namespace SST { namespace SnnDL { namespace v5 {

inline constexpr std::uint16_t kNocPacketV5FormatVersion = 1;
inline constexpr std::uint32_t kNocPacketV5HeaderBytes = 48;
inline constexpr std::uint32_t kNocControlV5HeaderBytes = 48;
inline constexpr std::uint32_t kNocDataVn = 0;
inline constexpr std::uint32_t kNocControlVn = 1;

class NocPacketV5Event final : public SST::Event {
public:
    std::uint16_t format_version = kNocPacketV5FormatVersion;
    std::uint64_t packet_id = 0;
    std::uint64_t timestep = 0;
    std::uint32_t source_pe = 0;
    std::uint32_t source_core = 0;
    std::uint64_t source_neuron = 0;
    std::uint64_t route_id = 0;
    std::uint32_t destination_pe = 0;
    std::uint32_t destination_core = 0;
    std::uint64_t destination_core_mask = 0;
    std::uint32_t target_neuron = 0;
    std::uint64_t source_event_seq = 0;
    std::uint64_t injection_time_ns = 0;
    std::uint32_t payload_bytes = 0;

    std::uint64_t wireBytes() const { return kNocPacketV5HeaderBytes + payload_bytes; }
    NocPacketV5Event* clone() override { return new NocPacketV5Event(*this); }
    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(format_version); SST_SER(packet_id); SST_SER(timestep);
        SST_SER(source_pe); SST_SER(source_core); SST_SER(source_neuron); SST_SER(route_id);
        SST_SER(destination_pe); SST_SER(destination_core); SST_SER(destination_core_mask); SST_SER(target_neuron);
        SST_SER(source_event_seq); SST_SER(injection_time_ns); SST_SER(payload_bytes);
    }
private:
    ImplementSerializable(SST::SnnDL::v5::NocPacketV5Event)
};

class NocInjectionAckV5Event final : public SST::Event {
public:
    std::uint64_t packet_id = 0;
    bool accepted = false;
    bool retryable = true;
    NocInjectionAckV5Event* clone() override { return new NocInjectionAckV5Event(*this); }
    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser); SST_SER(packet_id); SST_SER(accepted); SST_SER(retryable);
    }
private:
    ImplementSerializable(SST::SnnDL::v5::NocInjectionAckV5Event)
};

class NocCreditV5Event final : public SST::Event {
public:
    std::uint32_t credits = 1;
    NocCreditV5Event* clone() override { return new NocCreditV5Event(*this); }
    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser); SST_SER(credits);
    }
private:
    ImplementSerializable(SST::SnnDL::v5::NocCreditV5Event)
};

enum class NocControlV5Kind : std::uint8_t {
    Command = 0,
    Status = 1,
};

class NocControlV5Event final : public SST::Event {
public:
    std::uint16_t format_version = kNocPacketV5FormatVersion;
    NocControlV5Kind kind = NocControlV5Kind::Command;
    CoreControlOp operation = CoreControlOp::Abort;
    std::uint64_t epoch = 0;
    std::uint32_t source_pe = 0;
    std::uint32_t source_core = 0;
    std::uint32_t destination_pe = 0;
    std::uint32_t destination_core = 0;
    std::uint64_t logical_count = 0;
    std::uint64_t injection_time_ns = 0;

    std::uint64_t wireBytes() const { return kNocControlV5HeaderBytes; }
    NocControlV5Event* clone() override { return new NocControlV5Event(*this); }
    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(format_version); SST_SER(kind); SST_SER(operation); SST_SER(epoch);
        SST_SER(source_pe); SST_SER(source_core);
        SST_SER(destination_pe); SST_SER(destination_core); SST_SER(logical_count);
        SST_SER(injection_time_ns);
    }
private:
    ImplementSerializable(SST::SnnDL::v5::NocControlV5Event)
};

}}}
#endif
