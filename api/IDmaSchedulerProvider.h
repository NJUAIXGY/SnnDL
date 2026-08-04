// -*- c++ -*-
//
// Narrow provider interface for exposing a PE-shared DMA scheduler
// without coupling callers to a concrete SST component.

#pragma once

namespace SST { namespace SnnDL {

class PeDmaScheduler;

class IDmaSchedulerProvider {
public:
    virtual ~IDmaSchedulerProvider() = default;
    virtual PeDmaScheduler* dmaScheduler() = 0;
};

}} // namespace SST::SnnDL
