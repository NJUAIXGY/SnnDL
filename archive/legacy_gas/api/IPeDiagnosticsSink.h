// Narrow PE-level diagnostics sink used by core subcomponents.
#pragma once

#include <cstdint>

#include "IGasStageSink.h"

namespace SST { namespace Statistics {
template <class T> class Statistic;
}}

namespace SST { namespace SnnDL {

class IPeDiagnosticsSink {
public:
    virtual ~IPeDiagnosticsSink() = default;

    virtual SST::Statistics::Statistic<uint64_t>*
    getComputeActiveCyclesTotalStatistic() const { return nullptr; }

    virtual void recordStepGasStat(uint32_t, const GasStatEvent&) {}
    virtual void recordCoreStepGasStat(int, uint32_t, const GasStatEvent&) {}

    virtual void recordStepApplyScatter(uint32_t,
                                        uint64_t, uint64_t, uint64_t,
                                        uint64_t, uint64_t, uint64_t) {}
    virtual void recordCoreStepApplyScatter(int, uint32_t,
                                            uint64_t, uint64_t, uint64_t,
                                            uint64_t, uint64_t, uint64_t) {}

    virtual void accumulateRiscvSnnRuntimeStats(uint64_t, uint64_t, uint64_t,
                                                uint64_t, uint64_t, uint64_t,
                                                uint64_t, uint64_t, uint64_t,
                                                uint64_t, uint64_t, uint64_t,
                                                uint64_t, uint64_t) {}
};

}} // namespace SST::SnnDL
