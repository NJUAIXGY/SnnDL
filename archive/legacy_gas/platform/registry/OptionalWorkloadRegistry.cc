// -*- c++ -*-
//
// Process-local registry ABI shared by the core PE and explicitly loaded
// workload/research extensions.  Concrete workload classes stay outside this
// library so the native SNN/2D assembly has no reverse dependency on them.

#include "api/OptionalWorkloadRegistry.h"
#include "api/ICoreWorkload.h"
#include "api/ISnnAccelRuntimeServices.h"

#include "api/SnnDLStringUtil.h"

#include <algorithm>
#include <vector>

namespace SST { namespace SnnDL {

namespace {

struct WorkloadEntry {
    std::string name;
    OptionalWorkloadFactory factory = nullptr;
};

struct RuntimeEntry {
    std::string name;
    OptionalRuntimeServiceFactory factory = nullptr;
};

std::vector<WorkloadEntry>& workloadRegistry_() {
    static std::vector<WorkloadEntry> entries;
    return entries;
}

std::vector<RuntimeEntry>& runtimeRegistry_() {
    static std::vector<RuntimeEntry> entries;
    return entries;
}

std::string normalize_(const std::string& name) {
    return toLowerCopy(name);
}

} // namespace

void registerOptionalWorkload(const std::string& name,
                              OptionalWorkloadFactory factory) {
    const std::string normalized = normalize_(name);
    if (normalized.empty() || !factory) return;

    auto& entries = workloadRegistry_();
    const auto it = std::find_if(entries.begin(), entries.end(),
                                 [&normalized](const WorkloadEntry& entry) {
                                     return entry.name == normalized;
                                 });
    if (it != entries.end()) {
        it->factory = factory;
        return;
    }
    entries.push_back({normalized, factory});
}

std::unique_ptr<ICoreWorkload>
createOptionalWorkloadByName(const std::string& name) {
    const std::string normalized = normalize_(name);
    const auto& entries = workloadRegistry_();
    const auto it = std::find_if(entries.begin(), entries.end(),
                                 [&normalized](const WorkloadEntry& entry) {
                                     return entry.name == normalized;
                                 });
    return it == entries.end() || !it->factory ? nullptr : it->factory();
}

void registerOptionalRuntimeService(const std::string& name,
                                    OptionalRuntimeServiceFactory factory) {
    const std::string normalized = normalize_(name);
    if (normalized.empty() || !factory) return;

    auto& entries = runtimeRegistry_();
    const auto it = std::find_if(entries.begin(), entries.end(),
                                 [&normalized](const RuntimeEntry& entry) {
                                     return entry.name == normalized;
                                 });
    if (it != entries.end()) {
        it->factory = factory;
        return;
    }
    entries.push_back({normalized, factory});
}

std::unique_ptr<ISnnAccelRuntimeServices>
createOptionalRuntimeServiceByName(const std::string& name) {
    const std::string normalized = normalize_(name);
    const auto& entries = runtimeRegistry_();
    const auto it = std::find_if(entries.begin(), entries.end(),
                                 [&normalized](const RuntimeEntry& entry) {
                                     return entry.name == normalized;
                                 });
    return it == entries.end() || !it->factory ? nullptr : it->factory();
}

}} // namespace SST::SnnDL
