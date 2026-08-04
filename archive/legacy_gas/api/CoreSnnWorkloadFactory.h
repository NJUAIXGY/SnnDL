// -*- c++ -*-
//
// CoreSnnWorkloadFactory: construct the native SNN workload.
//
// The core factory deliberately knows about only SnnWorkload. Optional
// workload implementations are exposed through OptionalWorkloadRegistry.
//

#pragma once

#include <memory>

namespace SST { namespace SnnDL {

class ICoreWorkload;

// Construct the native SNN workload. The returned object is always the core
// SNN implementation; optional workload names are handled by the registry.
std::unique_ptr<ICoreWorkload> createCoreSnnWorkload();

}} // namespace SST::SnnDL
