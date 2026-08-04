// -*- c++ -*-
//
// Composite capability injection point for CoreShell implementations.
// The PE assembler owns the adapter; cores consume only the capabilities
// they need instead of dynamic-casting the parent component for every
// optional domain.

#pragma once

#include "IDmaSchedulerProvider.h"
#include "ILocalStorageProvider.h"
#include "IPeDiagnosticsSink.h"
#include "IPePodSharedMetadataProvider.h"
#include "IPeRuntimeConfig.h"
#include "IPeWeightObjectPlaneProvider.h"

namespace SST { namespace SnnDL {

class IPeCapabilityProvider {
public:
    virtual ~IPeCapabilityProvider() = default;

    virtual IPeDiagnosticsSink* peDiagnosticsSink() = 0;
    virtual IPeRuntimeConfig* peRuntimeConfig() = 0;
    virtual IDmaSchedulerProvider* dmaSchedulerProvider() = 0;
    virtual ILocalStorageProvider* localStorageProvider() = 0;
    virtual IPePodSharedMetadataProvider* podMetadataProvider() = 0;
    virtual IPeWeightObjectPlaneProvider* weightObjectPlaneProvider() = 0;
};

}} // namespace SST::SnnDL
