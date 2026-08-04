// -*- c++ -*-
//
// CoreWorkloadFactory: compatibility entry point for workload plugins.
//
// Native SNN construction lives in CoreSnnWorkloadFactory. Non-SNN workload
// implementations are registered by OptionalWorkloadRegistry. Keep this
// function for callers that still use the historical single-factory API.
//

#pragma once

#include <memory>
#include <string>

namespace SST { namespace SnnDL {

class ICoreWorkload;

// Returns nullptr if name is unknown.
std::unique_ptr<ICoreWorkload> createWorkloadByName(const std::string& name);

}} // namespace SST::SnnDL
