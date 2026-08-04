// Explicit research plugin registration for the traffic/3D workload family.
// The default SnnDL compatibility library intentionally does not register it.

#include "api/OptionalWorkloadRegistry.h"
#include "api/ICoreWorkload.h"
#include "workloads/traffic/TrafficWorkload.h"
#include "workloads/traffic_mem/TrafficMemWorkload.h"

namespace SST { namespace SnnDL {

namespace {
std::unique_ptr<ICoreWorkload> makeTraffic_() {
    return std::make_unique<TrafficWorkload>();
}

std::unique_ptr<ICoreWorkload> makeTrafficMem_() {
    return makeTrafficMemWorkload();
}

struct TrafficRegistration final {
    TrafficRegistration() {
        registerOptionalWorkload("traffic", &makeTraffic_);
        registerOptionalWorkload("traffic_mem", &makeTrafficMem_);
    }
};

const TrafficRegistration kTrafficRegistration{};
} // namespace

}} // namespace SST::SnnDL
