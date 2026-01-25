// -*- c++ -*-

#include <sst/core/sst_config.h>

#include "StandardMemAccess.h"

#include <algorithm>
#include <cstring>

#include <sst/core/output.h>

#include <inttypes.h>

namespace SST { namespace SnnDL {

uint64_t StandardMemAccess::lineBytes_() const {
    if (!mem_) return 0;
    const auto lb = mem_->getLineSize();
    if (lb != 0) return static_cast<uint64_t>(lb);
    // Some StandardMem front-ends (e.g. memHierarchy.standardInterface) do not report line size.
    // For safety (and to avoid memHierarchy Incoherent cache overflow on >line reads), use a conservative default.
    // NOTE: reading smaller-than-actual cacheline is safe (may increase traffic but preserves correctness).
    static bool warned = false;
    if (!warned && out_) {
        warned = true;
        out_->verbose(CALL_INFO, 0, 0,
                      "WARN: stdmem getLineSize()=0, fallback to 64B (node=%d core=%d)\n",
                      node_id_, core_id_);
    }
    return 64;
}

uint64_t StandardMemAccess::allocFragmentGroupId_() {
    // Use a high-bit prefix to avoid colliding with StandardMem IDs (+1 offset, typically small).
    const uint64_t id = (1ull << 63) | next_group_id_;
    next_group_id_ += 1;
    // Avoid ever returning 0.
    if (id == 0) return allocFragmentGroupId_();
    return id;
}

void StandardMemAccess::onFragmentResp_(uint64_t group_id, uint64_t line_index, std::vector<uint8_t> data) {
    auto it = frag_groups_.find(group_id);
    if (it == frag_groups_.end()) return;
    FragmentGroup& g = it->second;
    if (g.line_bytes == 0) return;
    const size_t line_bytes = static_cast<size_t>(g.line_bytes);
    const size_t off = static_cast<size_t>(line_index) * line_bytes;
    const size_t copy_n = std::min(line_bytes, data.size());
    if (off + copy_n <= g.buf.size()) {
        std::memcpy(g.buf.data() + off, data.data(), copy_n);
    }

    if (g.remaining > 0) g.remaining -= 1;
    if (g.remaining != 0) return;

    std::vector<uint8_t> sliced;
    if (g.slice_len != 0 && g.slice_off + g.slice_len <= g.buf.size()) {
        sliced.assign(g.buf.begin() + static_cast<ptrdiff_t>(g.slice_off),
                      g.buf.begin() + static_cast<ptrdiff_t>(g.slice_off + g.slice_len));
    }

    ReadCallback cb = std::move(g.cb);
    const uint64_t orig_addr = g.orig_addr;
    frag_groups_.erase(it);
    if (cb) cb(group_id, orig_addr, std::move(sliced));
}

IMemoryAccess::RequestId
StandardMemAccess::read(uint64_t addr, size_t bytes, ReadCallback cb) {
    if (!mem_ || bytes == 0) {
        if (cb) cb(0, addr, {});
        return 0;
    }

    const uint64_t line_bytes = lineBytes_();
    const bool need_fragment = (line_bytes != 0 && bytes > static_cast<size_t>(line_bytes));
    if (need_fragment) {
        const uint64_t base = (line_bytes != 0) ? (addr / line_bytes) * line_bytes : addr;
        const uint64_t end = addr + static_cast<uint64_t>(bytes);
        const uint64_t aligned_end =
            (line_bytes != 0) ? ((end + line_bytes - 1) / line_bytes) * line_bytes : end;
        const size_t total_bytes = static_cast<size_t>(aligned_end - base);
        const size_t nlines = (line_bytes != 0) ? (total_bytes / static_cast<size_t>(line_bytes)) : 0;

        const uint64_t group_id = allocFragmentGroupId_();
        FragmentGroup g{};
        g.orig_addr = addr;
        g.slice_off = static_cast<size_t>(addr - base);
        g.slice_len = bytes;
        g.line_bytes = line_bytes;
        g.remaining = nlines;
        g.buf.resize(total_bytes);
        g.cb = std::move(cb);
        frag_groups_[group_id] = std::move(g);

        for (size_t i = 0; i < nlines; ++i) {
            const uint64_t sub_addr = base + static_cast<uint64_t>(i) * line_bytes;
            auto* req = new SST::Interfaces::StandardMem::Read(sub_addr, line_bytes);
            if (force_noncacheable_) {
                req->setNoncacheable();
            }
            const auto sub_id = static_cast<uint64_t>(req->getID()) + 1;
            PendingEntry pe{};
            pe.address = sub_addr;
            pe.bytes = static_cast<size_t>(line_bytes);
            pe.is_write = false;
            pe.read_cb = [this, group_id, i](RequestId /*id*/, uint64_t /*addr*/, std::vector<uint8_t> data) {
                onFragmentResp_(group_id, static_cast<uint64_t>(i), std::move(data));
            };
            pending_[sub_id] = std::move(pe);
            mem_->send(req);
        }

        return group_id;
    }

    auto* req = new SST::Interfaces::StandardMem::Read(addr, bytes);
    if (force_noncacheable_) {
        req->setNoncacheable();
    }
    // StandardMem::Request::id_t 可能为 0；IMemoryAccess 约定 0 为“失败/无效”。
    // 因此对外暴露的 RequestId 做 +1 偏移，保证成功请求永不返回 0。
    const auto id = static_cast<uint64_t>(req->getID()) + 1;
    PendingEntry pe{};
    pe.address = addr;
    pe.bytes = bytes;
    pe.is_write = false;
    pe.read_cb = std::move(cb);
    pending_[id] = std::move(pe);
    mem_->send(req);
    return id;
}

IMemoryAccess::RequestId
StandardMemAccess::write(uint64_t addr, const std::vector<uint8_t>& data, WriteCallback cb) {
    if (!mem_ || data.empty()) {
        if (cb) cb(0, addr);
        return 0;
    }
    auto* req = new SST::Interfaces::StandardMem::Write(addr, static_cast<uint64_t>(data.size()),
                                                        data, /*posted*/false);
    if (force_noncacheable_) {
        req->setNoncacheable();
    }
    const auto id = static_cast<uint64_t>(req->getID()) + 1;
    PendingEntry pe{};
    pe.address = addr;
    pe.bytes = data.size();
    pe.is_write = true;
    pe.write_cb = std::move(cb);
    pending_[id] = std::move(pe);
    mem_->send(req);
    return id;
}

bool StandardMemAccess::handleMemoryResponse(SST::Interfaces::StandardMem::Request* req) {
    if (!req) return true;
    const uint64_t id = static_cast<uint64_t>(req->getID()) + 1;
    auto it = pending_.find(id);
    if (it == pending_.end()) {
        // Phase10+: Some StandardMem front-ends may alter the response ID while preserving (addr,size).
        // Keep memory layer semantics pure: only match by exact (addr,size) for reads and only when unambiguous.
        if (auto* rr = dynamic_cast<SST::Interfaces::StandardMem::ReadResp*>(req)) {
            uint64_t match_id = 0;
            for (const auto& kv : pending_) {
                const PendingEntry& pe = kv.second;
                if (pe.is_write) continue;
                if (pe.address != rr->pAddr) continue;
                if (pe.bytes != static_cast<size_t>(rr->size)) continue;
                if (match_id != 0) {
                    if (out_) {
                        out_->fatal(CALL_INFO, -1,
                                    "[stdmem-access-assert] ambiguous addr-match node=%d core=%d resp_id=%" PRIu64
                                    " addr=0x%llx bytes=%zu pending=%zu\n",
                                    node_id_, core_id_, (uint64_t)id,
                                    (unsigned long long)rr->pAddr,
                                    (size_t)rr->size,
                                    pending_.size());
                    }
                    abort();
                }
                match_id = kv.first;
            }
            if (match_id != 0) {
                if (out_) {
                    out_->verbose(CALL_INFO, 1, 0,
                                  "[stdmem-access] id-mismatch node=%d core=%d resp_id=%" PRIu64
                                  " matched_id=%" PRIu64 " addr=0x%llx bytes=%zu\n",
                                  node_id_, core_id_, (uint64_t)id, (uint64_t)match_id,
                                  (unsigned long long)rr->pAddr, (size_t)rr->size);
                }
                it = pending_.find(match_id);
            } else {
                return false;
            }
        } else {
            return false;
        }
    }
    const uint64_t effective_id = it->first;
    PendingEntry pe = std::move(it->second);
    pending_.erase(it);

    if (auto* rr = dynamic_cast<SST::Interfaces::StandardMem::ReadResp*>(req)) {
        std::vector<uint8_t> data = std::move(rr->data);
        if (data.size() < pe.bytes) {
            if (out_) {
                out_->fatal(CALL_INFO, -1,
                            "[stdmem-access-assert] node=%d core=%d id=%" PRIu64 " addr=0x%llx req_bytes=%zu resp_bytes=%zu\n",
                            node_id_, core_id_, (uint64_t)id,
                            (unsigned long long)pe.address,
                            pe.bytes, data.size());
            } else {
                // No Output available; still fail-fast.
                abort();
            }
        }
        if (data.size() > pe.bytes) data.resize(pe.bytes);
        if (pe.read_cb) pe.read_cb(effective_id, pe.address, std::move(data));
        delete req;
        return true;
    }

    // Treat any non-ReadResp as handled for this id.
    if (pe.is_write) {
        if (pe.write_cb) pe.write_cb(effective_id, pe.address);
    } else {
        // Unexpected response type for a read: surface as failure (Phase1-A: empty data).
        if (pe.read_cb) pe.read_cb(effective_id, pe.address, {});
    }
    delete req;
    return true;
}

}} // namespace SST::SnnDL
