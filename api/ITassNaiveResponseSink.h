// -*- c++ -*-
#pragma once

#include <vector>

#include "IPeAggregation.h"

namespace SST { namespace SnnDL {

class ITassNaiveResponseSink {
public:
    virtual ~ITassNaiveResponseSink() = default;
    virtual void onTassNaiveResponses(const std::vector<TassNaiveResponseEntry>& entries) = 0;
};

}} // namespace SST::SnnDL
