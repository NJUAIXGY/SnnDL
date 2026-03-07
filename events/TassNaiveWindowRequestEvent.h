// -*- c++ -*-
#ifndef SNNDL_TASS_NAIVE_WINDOW_REQUEST_EVENT_H
#define SNNDL_TASS_NAIVE_WINDOW_REQUEST_EVENT_H

#include <sst/core/event.h>
#include <sst/core/serialization/serialize.h>

#include "IPeAggregation.h"

namespace SST { namespace SnnDL {

class TassNaiveWindowRequestEvent final : public SST::Event {
public:
    TassNaiveWindowRequest request{};

    TassNaiveWindowRequestEvent() = default;
    explicit TassNaiveWindowRequestEvent(const TassNaiveWindowRequest& r) : SST::Event(), request(r) {}

    void serialize_order(SST::Core::Serialization::serializer& ser) override {
        Event::serialize_order(ser);
        request.serialize_order(ser);
    }

private:
    ImplementSerializable(SST::SnnDL::TassNaiveWindowRequestEvent)
};

}} // namespace SST::SnnDL

#endif // SNNDL_TASS_NAIVE_WINDOW_REQUEST_EVENT_H
