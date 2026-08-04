#ifndef SST_SNN_DL_TIMESTEP_TRACKER_H
#define SST_SNN_DL_TIMESTEP_TRACKER_H

#include "api/TimestepTypes.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <map>

namespace SST {
namespace SnnDL {

enum class WorkKind : std::uint8_t {
    Ingress = 0,
    RouteTask,
    IndexRead,
    WeightRead,
    RetireEntry,
    AccumulatorUpdate,
    LocalDelivery,
    HoldQueue,
    Extension,
    Count
};

struct DrainSnapshot {
    TimestepId timestep = 0;
    bool opened = false;
    bool sealed = false;
    std::array<std::uint64_t, static_cast<std::size_t>(WorkKind::Count)> outstanding{};

    bool empty() const;
};

class TimestepTracker {
public:
    void open(TimestepId timestep);
    void sealIngress(TimestepId timestep);
    void acquire(TimestepId timestep, WorkKind kind, std::uint64_t count = 1);
    void release(TimestepId timestep, WorkKind kind, std::uint64_t count = 1);
    bool locallyDrained(TimestepId timestep) const;
    DrainSnapshot snapshot(TimestepId timestep) const;

private:
    struct StepState {
        bool opened = false;
        bool sealed = false;
        std::array<std::uint64_t, static_cast<std::size_t>(WorkKind::Count)> outstanding{};
    };

    static std::size_t index_(WorkKind kind);
    StepState& require_(TimestepId timestep);
    const StepState& require_(TimestepId timestep) const;

    std::map<TimestepId, StepState> steps_;
};

} // namespace SnnDL
} // namespace SST

#endif
