#ifndef SST_SNN_DL_V5_DETERMINISTIC_RETIRE_QUEUE_H
#define SST_SNN_DL_V5_DETERMINISTIC_RETIRE_QUEUE_H

#include <cstdint>
#include <map>

namespace SST {
namespace SnnDL {
namespace v5 {

struct RetireKey {
    std::uint32_t post_neuron = 0;
    std::uint64_t source_event_seq = 0;
    std::uint64_t edge_ordinal = 0;

    bool operator<(const RetireKey& other) const {
        if (post_neuron != other.post_neuron) return post_neuron < other.post_neuron;
        if (source_event_seq != other.source_event_seq) return source_event_seq < other.source_event_seq;
        return edge_ordinal < other.edge_ordinal;
    }
    bool operator==(const RetireKey& other) const {
        return post_neuron == other.post_neuron && source_event_seq == other.source_event_seq &&
               edge_ordinal == other.edge_ordinal;
    }
};

struct RetireEntry {
    RetireKey key;
    std::uint64_t timestep = 0;
    float weight = 0.0f;
};

class DeterministicRetireQueue {
public:
    explicit DeterministicRetireQueue(std::size_t capacity);

    bool push(const RetireEntry& entry);
    bool empty() const { return entries_.empty(); }
    bool full() const { return entries_.size() >= capacity_; }
    std::size_t size() const { return entries_.size(); }
    std::size_t capacity() const { return capacity_; }
    const RetireEntry& front() const;
    RetireEntry pop();
    void clear() { entries_.clear(); }

private:
    std::size_t capacity_;
    std::map<RetireKey, RetireEntry> entries_;
};

} // namespace v5
} // namespace SnnDL
} // namespace SST

#endif
