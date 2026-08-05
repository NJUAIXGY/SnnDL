#include "DeterministicRetireQueue.h"

#include <stdexcept>

namespace SST {
namespace SnnDL {
namespace v5 {

DeterministicRetireQueue::DeterministicRetireQueue(std::size_t capacity) : capacity_(capacity) {
    if (capacity_ == 0) throw std::invalid_argument("retire queue capacity must be positive");
}

bool DeterministicRetireQueue::push(const RetireEntry& entry) {
    if (full() || entries_.find(entry.key) != entries_.end()) return false;
    entries_.emplace(entry.key, entry);
    return true;
}

const RetireEntry& DeterministicRetireQueue::front() const {
    if (entries_.empty()) throw std::out_of_range("retire queue is empty");
    return entries_.begin()->second;
}

RetireEntry DeterministicRetireQueue::pop() {
    if (entries_.empty()) throw std::out_of_range("retire queue is empty");
    auto it = entries_.begin();
    RetireEntry result = it->second;
    entries_.erase(it);
    return result;
}

} // namespace v5
} // namespace SnnDL
} // namespace SST
