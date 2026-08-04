#include "TimestepTracker.h"

#include <map>
#include <stdexcept>

namespace SST {
namespace SnnDL {

bool DrainSnapshot::empty() const {
    for (std::uint64_t value : outstanding) {
        if (value != 0) return false;
    }
    return true;
}

std::size_t TimestepTracker::index_(WorkKind kind) {
    const auto value = static_cast<std::size_t>(kind);
    if (value >= static_cast<std::size_t>(WorkKind::Count)) {
        throw std::logic_error("invalid timestep work kind");
    }
    return value;
}

TimestepTracker::StepState& TimestepTracker::require_(TimestepId timestep) {
    auto it = steps_.find(timestep);
    if (it == steps_.end() || !it->second.opened) {
        throw std::logic_error("timestep is not open");
    }
    return it->second;
}

const TimestepTracker::StepState& TimestepTracker::require_(TimestepId timestep) const {
    auto it = steps_.find(timestep);
    if (it == steps_.end() || !it->second.opened) {
        throw std::logic_error("timestep is not open");
    }
    return it->second;
}

void TimestepTracker::open(TimestepId timestep) {
    auto& state = steps_[timestep];
    if (state.opened) throw std::logic_error("timestep was already opened");
    state = StepState{};
    state.opened = true;
}

void TimestepTracker::sealIngress(TimestepId timestep) {
    auto& state = require_(timestep);
    if (state.sealed) throw std::logic_error("timestep ingress is already sealed");
    state.sealed = true;
}

void TimestepTracker::acquire(TimestepId timestep, WorkKind kind, std::uint64_t count) {
    auto& state = require_(timestep);
    if (state.sealed) throw std::logic_error("cannot acquire work after ingress seal");
    state.outstanding[index_(kind)] += count;
}

void TimestepTracker::release(TimestepId timestep, WorkKind kind, std::uint64_t count) {
    auto& state = require_(timestep);
    auto& outstanding = state.outstanding[index_(kind)];
    if (count > outstanding) throw std::logic_error("timestep work token underflow");
    outstanding -= count;
}

bool TimestepTracker::locallyDrained(TimestepId timestep) const {
    const auto& state = require_(timestep);
    if (!state.sealed) return false;
    for (std::uint64_t value : state.outstanding) {
        if (value != 0) return false;
    }
    return true;
}

DrainSnapshot TimestepTracker::snapshot(TimestepId timestep) const {
    const auto& state = require_(timestep);
    DrainSnapshot result;
    result.timestep = timestep;
    result.opened = state.opened;
    result.sealed = state.sealed;
    result.outstanding = state.outstanding;
    return result;
}

} // namespace SnnDL
} // namespace SST
