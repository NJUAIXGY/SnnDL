#ifndef SST_SNN_DL_DELTA_ACCUMULATOR_H
#define SST_SNN_DL_DELTA_ACCUMULATOR_H

#include "api/TimestepTypes.h"

#include <cstdint>
#include <vector>

namespace SST {
namespace SnnDL {

class DeltaAccumulator {
public:
    explicit DeltaAccumulator(std::uint32_t posts);

    void begin(TimestepId timestep);
    void add(const SynapseContribution& contribution);
    std::vector<float> view(TimestepId timestep) const;
    void commitDone(TimestepId timestep);
    std::uint64_t updates() const { return updates_; }

private:
    struct Entry {
        std::uint64_t stable_order = 0;
        float value = 0.0f;
    };

    TimestepId active_timestep_ = 0;
    bool active_ = false;
    std::vector<std::vector<Entry>> entries_;
    std::uint64_t updates_ = 0;
};

} // namespace SnnDL
} // namespace SST

#endif
