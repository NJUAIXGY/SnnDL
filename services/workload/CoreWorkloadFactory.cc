// -*- c++ -*-

#include <sst/core/sst_config.h>

#include "CoreWorkloadFactory.h"

#include "ICoreWorkload.h"
#include "workload/snn/SnnWorkload.h"
#include "workload/stream/StreamWorkload.h"
#include "workload/traffic/TrafficWorkload.h"

namespace SST { namespace SnnDL {

std::unique_ptr<ICoreWorkload> createWorkloadByName(const std::string& name) {
    if (name == "snn") return std::make_unique<SnnWorkload>();
    if (name == "stream") return std::make_unique<StreamWorkload>();
    if (name == "traffic") return std::make_unique<TrafficWorkload>();
    return nullptr;
}

}} // namespace SST::SnnDL
