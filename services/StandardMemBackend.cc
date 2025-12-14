// -*- c++ -*-
//
// StandardMemBackend: 收敛 StandardMem pending/回调分发为独立后端。

#include "StandardMemBackend.h"

namespace SST { namespace SnnDL {

SST::Interfaces::StandardMem::Request::id_t
StandardMemBackend::sendRead(uint64_t addr, size_t bytes, const MemRequestMeta& meta) {
    if (!mem_) return 0;
    auto* req = new SST::Interfaces::StandardMem::Read(addr, bytes);
    auto id = req->getID();
    pending_[id] = meta;
    mem_->send(req);
    return id;
}

SST::Interfaces::StandardMem::Request::id_t
StandardMemBackend::sendWrite(uint64_t addr, const std::vector<uint8_t>& data, const MemRequestMeta& meta) {
    if (!mem_) return 0;
    auto* req = new SST::Interfaces::StandardMem::Write(addr, static_cast<uint64_t>(data.size()), data, /*posted*/false);
    auto id = req->getID();
    pending_[id] = meta;
    mem_->send(req);
    return id;
}

bool StandardMemBackend::popPending(SST::Interfaces::StandardMem::Request::id_t id, MemRequestMeta& out) {
    auto it = pending_.find(id);
    if (it == pending_.end()) return false;
    out = it->second;
    pending_.erase(it);
    return true;
}

}} // namespace SST::SnnDL
