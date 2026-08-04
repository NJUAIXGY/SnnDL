// -*- c++ -*-

#include <sst/core/sst_config.h>

#include "CoreWorkloadFactory.h"

#include "ICoreWorkload.h"
#include "CoreSnnWorkloadFactory.h"
#include "OptionalWorkloadRegistry.h"
#include "SnnDLStringUtil.h"

namespace SST { namespace SnnDL {

std::unique_ptr<ICoreWorkload> createWorkloadByName(const std::string& name) {
    if (toLowerCopy(name) == "snn") {
        return createCoreSnnWorkload();
    }
    return createOptionalWorkloadByName(name);
}

}} // namespace SST::SnnDL
