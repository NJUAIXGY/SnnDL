#ifndef SST_ELEMENTS_SNNDL_SNNBCSRWEIGHTMANAGER_H
#define SST_ELEMENTS_SNNDL_SNNBCSRWEIGHTMANAGER_H

#include <cstdint>
#include <vector>
#include <cstddef>

namespace SST {
namespace SnnDL {

class BcsrWeightManager {
public:
    BcsrWeightManager() = default;

    void configure(uint64_t rowptr_addr,
                   uint32_t block_rows,
                   uint32_t block_cols,
                   uint32_t idx_bytes,
                   uint32_t val_bytes);

    uint64_t rowptrAddr() const { return rowptr_addr_; }
    void setRowptrAddr(uint64_t addr) { rowptr_addr_ = addr; }
    uint32_t blockRows() const { return block_rows_; }
    uint32_t blockCols() const { return block_cols_; }
    uint32_t idxBytes() const { return idx_bytes_; }
    uint32_t valBytes() const { return val_bytes_; }

    bool isRowptrReady() const { return rowptr_ready_; }
    void setRowptrReady(bool ready) { rowptr_ready_ = ready; }
    bool isRowptrReadPending() const { return rowptr_read_pending_; }
    void setRowptrReadPending(bool pending) { rowptr_read_pending_ = pending; }

    std::vector<uint32_t>& rowptrHost() { return bcsr_rowptr_host_; }
    const std::vector<uint32_t>& rowptrHost() const { return bcsr_rowptr_host_; }

    size_t expectedRowptrEntries(uint32_t num_neurons) const;
    size_t expectedRowptrBytes(uint32_t num_neurons) const;
    bool installRowptrFromBytes(const uint8_t* data, size_t bytes, uint32_t num_neurons);

private:
    uint64_t rowptr_addr_ = 0;
    uint32_t block_rows_ = 16;
    uint32_t block_cols_ = 16;
    uint32_t idx_bytes_ = 2;
    uint32_t val_bytes_ = 4;

    std::vector<uint32_t> bcsr_rowptr_host_;
    bool rowptr_ready_ = false;
    bool rowptr_read_pending_ = false;
};

} // namespace SnnDL
} // namespace SST

#endif // SST_ELEMENTS_SNNDL_SNNBCSRWEIGHTMANAGER_H
