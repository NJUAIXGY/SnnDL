// -*- c++ -*-

#include <sst/core/sst_config.h>

#include "StandardMemAccess.h"

#include <sst/core/output.h>

#include <inttypes.h>

#include "WorkloadConfig.h"

namespace SST { namespace SnnDL {

IMemoryAccess::RequestId
StandardMemAccess::read(uint64_t addr, size_t bytes, ReadCallback cb) {
    if (!mem_ || bytes == 0) {
        if (cb) cb(0, addr, {});
        return 0;
    }
    auto* req = new SST::Interfaces::StandardMem::Read(addr, bytes);
    if (force_noncacheable_ || isStreamWorkloadEnv()) {
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
    if (force_noncacheable_ || isStreamWorkloadEnv()) {
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
        return false;
    }
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
        if (pe.read_cb) pe.read_cb(id, pe.address, std::move(data));
        delete req;
        return true;
    }

    // Treat any non-ReadResp as handled for this id.
    if (pe.is_write) {
        if (pe.write_cb) pe.write_cb(id, pe.address);
    } else {
        // Unexpected response type for a read: surface as failure (Phase1-A: empty data).
        if (pe.read_cb) pe.read_cb(id, pe.address, {});
    }
    delete req;
    return true;
}

}} // namespace SST::SnnDL
