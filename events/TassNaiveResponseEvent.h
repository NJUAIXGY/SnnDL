// -*- c++ -*-
#ifndef SNNDL_TASS_NAIVE_RESPONSE_EVENT_H
#define SNNDL_TASS_NAIVE_RESPONSE_EVENT_H

#include <utility>

#include <sst/core/event.h>
#include <sst/core/serialization/serialize.h>

#include "IPeAggregation.h"

namespace SST { namespace SnnDL {

class TassNaiveResponseEvent final : public SST::Event {
public:
    uint32_t source_node = 0;
    std::vector<TassNaiveResponseEntry> entries{};

    TassNaiveResponseEvent() = default;
    TassNaiveResponseEvent(uint32_t src_node, std::vector<TassNaiveResponseEntry> e)
        : SST::Event(), source_node(src_node), entries(std::move(e)) {}

    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        SST_SER(source_node);
        SST_SER(entries);
    }

private:
    ImplementSerializable(SST::SnnDL::TassNaiveResponseEvent)
};

}} // namespace SST::SnnDL

#endif // SNNDL_TASS_NAIVE_RESPONSE_EVENT_H
