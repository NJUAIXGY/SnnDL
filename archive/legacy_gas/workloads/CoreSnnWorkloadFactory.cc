// -*- c++ -*-

#include <sst/core/sst_config.h>

#include "CoreSnnWorkloadFactory.h"

#include "ICoreWorkload.h"
#include "workloads/snn/SnnWorkload.h"

namespace SST { namespace SnnDL {

std::unique_ptr<ICoreWorkload> createCoreSnnWorkload() {
    return std::make_unique<SnnWorkload>();
}

}} // namespace SST::SnnDL
