#include "SnnBcsrWeightManager.h"

#include <cstring>

using namespace SST::SnnDL;

void BcsrWeightManager::configure(uint64_t rowptr_addr,
                                  uint32_t block_rows,
                                  uint32_t block_cols,
                                  uint32_t idx_bytes,
                                  uint32_t val_bytes) {
    rowptr_addr_ = rowptr_addr;
    block_rows_ = block_rows;
    block_cols_ = block_cols;
    idx_bytes_ = idx_bytes;
    val_bytes_ = val_bytes;
}

size_t BcsrWeightManager::expectedRowptrEntries(uint32_t num_neurons) const {
    uint32_t br = (block_rows_ > 0 ? block_rows_ : 16);
    uint32_t nBlockRows = (num_neurons + br - 1) / br;
    return static_cast<size_t>(nBlockRows + 1);
}

size_t BcsrWeightManager::expectedRowptrBytes(uint32_t num_neurons) const {
    return expectedRowptrEntries(num_neurons) * sizeof(uint32_t);
}

bool BcsrWeightManager::installRowptrFromBytes(const uint8_t* data, size_t bytes, uint32_t num_neurons) {
    if (!data) return false;
    const size_t expect = expectedRowptrBytes(num_neurons);
    if (bytes != expect) return false;
    bcsr_rowptr_host_.resize(expectedRowptrEntries(num_neurons));
    std::memcpy(bcsr_rowptr_host_.data(), data, bytes);
    // Require non-decreasing sequence and at least one non-zero delta
    bool non_decreasing = true;
    bool has_progress = false;
    for (size_t idx = 0; idx + 1 < bcsr_rowptr_host_.size(); ++idx) {
        uint32_t cur = bcsr_rowptr_host_[idx];
        uint32_t nxt = bcsr_rowptr_host_[idx + 1];
        if (nxt < cur) { non_decreasing = false; break; }
        if (nxt > cur) has_progress = true;
    }
    if (!non_decreasing || !has_progress) {
        bcsr_rowptr_host_.clear();
        return false;
    }
    rowptr_ready_ = true;
    rowptr_read_pending_ = false;
    return true;
}
