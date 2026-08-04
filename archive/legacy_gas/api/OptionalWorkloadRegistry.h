// -*- c++ -*-
//
// OptionalWorkloadRegistry: extension-side workload registration.
//

#pragma once

#include <memory>
#include <string>

namespace SST { namespace SnnDL {

class ICoreWorkload;
class ISnnAccelRuntimeServices;

using OptionalWorkloadFactory = std::unique_ptr<ICoreWorkload>(*)();
using OptionalRuntimeServiceFactory =
    std::unique_ptr<ISnnAccelRuntimeServices>(*)();

// Register an optional workload implementation. Registration is idempotent;
// a later registration replaces the factory for the same normalized name.
void registerOptionalWorkload(const std::string& name,
                              OptionalWorkloadFactory factory);

// Returns nullptr when no optional workload is registered for name.
std::unique_ptr<ICoreWorkload> createOptionalWorkloadByName(const std::string& name);

// Register an optional runtime-service provider. Providers are normally
// registered by an explicitly loaded extension library.
void registerOptionalRuntimeService(const std::string& name,
                                    OptionalRuntimeServiceFactory factory);

// Returns nullptr when no runtime-service provider is registered for name.
std::unique_ptr<ISnnAccelRuntimeServices>
createOptionalRuntimeServiceByName(const std::string& name);

}} // namespace SST::SnnDL
