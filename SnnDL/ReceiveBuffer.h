#ifndef SNNDL_RECEIVE_BUFFER_H
#define SNNDL_RECEIVE_BUFFER_H

#include <queue>
#include <mutex>
#include <cstddef>
#include "SpikeEvent.h"

namespace SST { namespace SnnDL {

class ReceiveBuffer {
public:
    explicit ReceiveBuffer(size_t max_size = 1024) : max_buffer_size_(max_size) {}

    bool enqueueSpike(SpikeEvent* spike) {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        if (pending_spikes_.size() >= max_buffer_size_) return false;
        pending_spikes_.push(spike);
        return true;
    }

    SpikeEvent* dequeueSpike() {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        if (pending_spikes_.empty()) return nullptr;
        SpikeEvent* s = pending_spikes_.front();
        pending_spikes_.pop();
        return s;
    }

    bool isFull() const {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        return pending_spikes_.size() >= max_buffer_size_;
    }

    size_t getSize() const {
        std::lock_guard<std::mutex> lock(buffer_mutex_);
        return pending_spikes_.size();
    }

private:
    std::queue<SpikeEvent*> pending_spikes_;
    size_t max_buffer_size_;
    mutable std::mutex buffer_mutex_;
};

}} // namespace

#endif // SNNDL_RECEIVE_BUFFER_H
