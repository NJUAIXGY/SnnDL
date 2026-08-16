#ifndef SST_SNN_DL_V5_MULTICAST_CREDIT_V5_H
#define SST_SNN_DL_V5_MULTICAST_CREDIT_V5_H

#include <cstdint>

namespace SST { namespace SnnDL { namespace v5 {

class MulticastCreditV5 {
public:
    void reset(std::uint32_t capacity) {
        capacity_ = capacity;
        available_ = capacity;
    }

    bool consume() {
        if (available_ == 0) return false;
        --available_;
        return true;
    }

    bool restore(std::uint32_t credits = 1) {
        if (credits > capacity_ - available_) return false;
        available_ += credits;
        return true;
    }

    std::uint32_t available() const { return available_; }
    std::uint32_t capacity() const { return capacity_; }

private:
    std::uint32_t capacity_ = 0;
    std::uint32_t available_ = 0;
};

}}}
#endif
