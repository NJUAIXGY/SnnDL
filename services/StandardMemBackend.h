// -*- c++ -*-
//
// StandardMemBackend: 将 StandardMem 读/写请求的 pending 跟踪与回调分发收敛为独立后端，
// 以便控制层不再持有 pending map/request_id。

#pragma once

#include <cstdint>
#include <cstddef>
#include <functional>
#include <unordered_map>
#include <vector>

#include <sst/core/interfaces/stdMem.h>

namespace SST { namespace SnnDL {

struct MemRequestMeta {
    uint64_t address = 0;
    size_t size = 0;
    bool is_row = false;
    uint32_t pre = 0;
    uint32_t post_start = 0;
    uint32_t count_floats = 0;
    bool is_weight = true;
    uint64_t issue_cycle = 0;
    // BCSR 扩展
    int bcsr_kind = 0;              // 0=dense, 1=rowptr, 2=colidx, 3=blockdata
    uint32_t bcsr_block_row = 0;
    uint32_t bcsr_target_block_col = 0;
    uint32_t bcsr_intra_row = 0;
    uint32_t bcsr_intra_col = 0;
    uint32_t bcsr_row_start = 0;
    uint32_t bcsr_idx_in_row = 0;
    uint32_t bcsr_global_block_index = 0;
    bool bcsr_prefetch_all = false;
    bool scheme1_prefetch = false;
    bool has_single_cb = false;
    uint32_t cb_post = 0;
    std::function<void(float)> single_cb;
};

class StandardMemBackend {
public:
    explicit StandardMemBackend(SST::Interfaces::StandardMem* mem) : mem_(mem) {}

    SST::Interfaces::StandardMem::Request::id_t sendRead(uint64_t addr, size_t bytes,
                                                         const MemRequestMeta& meta);
    SST::Interfaces::StandardMem::Request::id_t sendWrite(uint64_t addr, const std::vector<uint8_t>& data,
                                                          const MemRequestMeta& meta);

    bool popPending(SST::Interfaces::StandardMem::Request::id_t id, MemRequestMeta& out);
    size_t pendingSize() const { return pending_.size(); }

private:
    SST::Interfaces::StandardMem* mem_ = nullptr;
    std::unordered_map<SST::Interfaces::StandardMem::Request::id_t, MemRequestMeta> pending_;
};

}} // namespace SST::SnnDL
