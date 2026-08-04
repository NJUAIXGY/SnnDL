// Optional PE configuration capabilities exposed to core subcomponents.
#pragma once

#include <cstdint>

namespace SST { namespace SnnDL {

class IPeRuntimeConfig {
public:
    virtual ~IPeRuntimeConfig() = default;

    virtual bool peInternalCpeEnabledConfig() const { return false; }
    virtual bool peInternalPodEnabledConfig() const { return false; }
    virtual bool peInternalPodMetadataEnabledConfig() const { return false; }
    virtual bool peInternalPodOwnerEnabledConfig() const { return false; }
    virtual uint32_t peInternalPodCountConfig() const { return 1; }
    virtual uint32_t peInternalPodSizeConfig() const { return 1; }
};

}} // namespace SST::SnnDL
