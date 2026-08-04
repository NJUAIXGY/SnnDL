#include "DeltaAccumulator.h"

#include <algorithm>
#include <stdexcept>

namespace SST {
namespace SnnDL {

DeltaAccumulator::DeltaAccumulator(std::uint32_t posts) : entries_(posts) {}

void DeltaAccumulator::begin(TimestepId timestep) {
    if (active_) throw std::logic_error("delta accumulator already active");
    active_timestep_ = timestep;
    active_ = true;
    for (auto& entries : entries_) entries.clear();
    updates_ = 0;
}

void DeltaAccumulator::add(const SynapseContribution& contribution) {
    if (!active_ || contribution.timestep != active_timestep_) {
        throw std::logic_error("delta contribution has the wrong timestep");
    }
    if (contribution.post_local >= entries_.size()) {
        throw std::out_of_range("delta post index is out of range");
    }
    entries_[contribution.post_local].push_back(
        Entry{contribution.stable_order, contribution.weight});
    ++updates_;
}

std::vector<float> DeltaAccumulator::view(TimestepId timestep) const {
    if (!active_ || timestep != active_timestep_) {
        throw std::logic_error("delta view has the wrong timestep");
    }
    std::vector<float> result(entries_.size(), 0.0f);
    for (std::size_t post = 0; post < entries_.size(); ++post) {
        auto ordered = entries_[post];
        std::stable_sort(ordered.begin(), ordered.end(), [](const Entry& lhs, const Entry& rhs) {
            return lhs.stable_order < rhs.stable_order;
        });
        for (const auto& entry : ordered) result[post] += entry.value;
    }
    return result;
}

void DeltaAccumulator::commitDone(TimestepId timestep) {
    if (!active_ || timestep != active_timestep_) {
        throw std::logic_error("delta commit has the wrong timestep");
    }
    active_ = false;
    for (auto& entries : entries_) entries.clear();
}

} // namespace SnnDL
} // namespace SST
