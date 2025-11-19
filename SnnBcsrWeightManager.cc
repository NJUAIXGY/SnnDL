#include "SnnBcsrWeightManager.h"

#include <cstring>
#include <utility>

using namespace SST::SnnDL;

void BcsrWeightManager::configure(uint64_t rowptr_addr,
                                  uint64_t colidx_addr,
                                  uint64_t blockdata_addr,
                                  uint64_t blockids_addr,
                                  uint32_t block_rows,
                                  uint32_t block_cols,
                                  uint32_t idx_bytes,
                                  uint32_t val_bytes) {
    rowptr_addr_ = rowptr_addr;
    colidx_addr_ = colidx_addr;
    blockdata_addr_ = blockdata_addr;
    blockids_addr_ = blockids_addr;
    block_rows_ = block_rows;
    block_cols_ = block_cols;
    idx_bytes_ = idx_bytes;
    val_bytes_ = val_bytes;
    rowptr_ready_ = false;
    rowptr_read_pending_ = false;
    bcsr_rowptr_host_.clear();
    resetCaches();
}

void BcsrWeightManager::setRowIndexCacheCapacity(uint32_t cap) {
    row_index_cache_cap_ = cap;
    if (row_index_cache_cap_ == 0) {
        row_index_cache_.clear();
        return;
    }
    while (row_index_cache_.size() > row_index_cache_cap_) {
        row_index_cache_.erase(row_index_cache_.begin());
    }
}

void BcsrWeightManager::setBlockCacheCapacity(uint32_t cap) {
    block_cache_cap_ = cap;
    if (block_cache_cap_ == 0) {
        block_cache_.clear();
        return;
    }
    while (block_cache_.size() > block_cache_cap_) {
        block_cache_.erase(block_cache_.begin());
    }
}

void BcsrWeightManager::resetCaches() {
    row_index_cache_.clear();
    block_cache_.clear();
}

size_t BcsrWeightManager::expectedRowptrEntries(uint32_t num_neurons) const {
    uint32_t br = (block_rows_ > 0 ? block_rows_ : 16);
    uint32_t nBlockRows = (num_neurons + br - 1) / br;
    return static_cast<size_t>(nBlockRows + 1);
}

size_t BcsrWeightManager::expectedRowptrBytes(uint32_t num_neurons) const {
    return expectedRowptrEntries(num_neurons) * sizeof(uint32_t);
}

uint32_t BcsrWeightManager::effectiveBlockRows() const {
    return block_rows_ ? block_rows_ : 16;
}

uint32_t BcsrWeightManager::effectiveBlockCols() const {
    return block_cols_ ? block_cols_ : 16;
}

uint32_t BcsrWeightManager::effectiveIdxBytes() const {
    return idx_bytes_ ? idx_bytes_ : 2;
}

uint32_t BcsrWeightManager::effectiveValBytes() const {
    return val_bytes_ ? val_bytes_ : 4;
}

size_t BcsrWeightManager::blockBytes() const {
    return static_cast<size_t>(effectiveBlockRows()) *
           static_cast<size_t>(effectiveBlockCols()) *
           static_cast<size_t>(effectiveValBytes());
}

uint64_t BcsrWeightManager::blockDataAddr(uint32_t global_block_index) const {
    return blockdata_addr_ + static_cast<uint64_t>(global_block_index) * blockBytes();
}

uint64_t BcsrWeightManager::colIndexAddr(uint32_t start_index) const {
    return colidx_addr_ + static_cast<uint64_t>(start_index) * effectiveIdxBytes();
}

size_t BcsrWeightManager::colIndexBytes(uint32_t block_count) const {
    return static_cast<size_t>(block_count) * effectiveIdxBytes();
}

bool BcsrWeightManager::rowBounds(uint32_t block_row, uint32_t& start, uint32_t& end) const {
    if (block_row + 1 >= bcsr_rowptr_host_.size()) return false;
    start = bcsr_rowptr_host_[block_row];
    end = bcsr_rowptr_host_[block_row + 1];
    return true;
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

bool BcsrWeightManager::rowIndexGet(uint32_t block_row, std::vector<uint32_t>& out) const {
    auto it = row_index_cache_.find(block_row);
    if (it == row_index_cache_.end()) return false;
    out = it->second;
    return true;
}

void BcsrWeightManager::rowIndexPut(uint32_t block_row, std::vector<uint32_t>&& cols) {
    if (row_index_cache_cap_ == 0) return;
    evictRowIndexIfNeeded(block_row);
    row_index_cache_[block_row] = std::move(cols);
}

bool BcsrWeightManager::blockGet(uint32_t block_row, uint32_t block_col, std::vector<float>& out) const {
    const uint64_t key = makeBlockKey(block_row, block_col);
    auto it = block_cache_.find(key);
    if (it == block_cache_.end()) return false;
    out = it->second;
    return true;
}

void BcsrWeightManager::blockPut(uint32_t block_row, uint32_t block_col, std::vector<float>&& data) {
    if (block_cache_cap_ == 0) return;
    const uint64_t key = makeBlockKey(block_row, block_col);
    evictBlockIfNeeded(key);
    block_cache_[key] = std::move(data);
}

bool BcsrWeightManager::hasBlock(uint32_t block_row, uint32_t block_col) const {
    const uint64_t key = makeBlockKey(block_row, block_col);
    return block_cache_.find(key) != block_cache_.end();
}

uint64_t BcsrWeightManager::makeBlockKey(uint32_t block_row, uint32_t block_col) {
    return (static_cast<uint64_t>(block_row) << 32) | block_col;
}

void BcsrWeightManager::evictRowIndexIfNeeded(uint32_t incoming_key) {
    if (row_index_cache_cap_ == 0) return;
    if (row_index_cache_.size() < row_index_cache_cap_) return;
    if (row_index_cache_.find(incoming_key) != row_index_cache_.end()) return;
    row_index_cache_.erase(row_index_cache_.begin());
}

void BcsrWeightManager::evictBlockIfNeeded(uint64_t incoming_key) {
    if (block_cache_cap_ == 0) return;
    if (block_cache_.size() < block_cache_cap_) return;
    if (block_cache_.find(incoming_key) != block_cache_.end()) return;
    block_cache_.erase(block_cache_.begin());
}
