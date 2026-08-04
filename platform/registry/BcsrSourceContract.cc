// -*- c++ -*-

#include "api/BcsrSourceContract.h"

#include <mutex>
#include <sstream>
#include <unordered_map>

extern "C" void snndl_registry_anchor() {}

namespace SST { namespace SnnDL {

namespace {

struct Binding {
    std::string path;
    BcsrSourceIdentity identity{};
    std::string owner;
};

std::mutex& contractMutex() {
    static std::mutex mutex;
    return mutex;
}

std::unordered_map<std::string, Binding>& bindings() {
    static std::unordered_map<std::string, Binding> table;
    return table;
}

bool identityMatches(const BcsrSourceIdentity& lhs, const BcsrSourceIdentity& rhs) {
    if (lhs.file_size != rhs.file_size) return false;
    if (lhs.content_fingerprint != 0 || rhs.content_fingerprint != 0) {
        return lhs.content_fingerprint != 0 &&
               rhs.content_fingerprint != 0 &&
               lhs.content_fingerprint == rhs.content_fingerprint;
    }
    return lhs.descriptor_fingerprint != 0 &&
           rhs.descriptor_fingerprint != 0 &&
           lhs.descriptor_fingerprint == rhs.descriptor_fingerprint;
}

} // namespace

bool bindBcsrSourceContract(const std::string& slot,
                            const std::string& path,
                            const BcsrSourceIdentity& identity,
                            const char* owner,
                            std::string* error_out) {
    if (error_out) error_out->clear();
    if (slot.empty() || path.empty() || identity.file_size == 0) {
        if (error_out) *error_out = "invalid BCSR source contract binding";
        return false;
    }

    std::lock_guard<std::mutex> guard(contractMutex());
    auto& table = bindings();
    auto it = table.find(slot);
    if (it == table.end()) {
        Binding binding;
        binding.path = path;
        binding.identity = identity;
        binding.owner = owner ? owner : "unknown";
        table.emplace(slot, std::move(binding));
        return true;
    }

    if (identityMatches(it->second.identity, identity)) return true;

    if (error_out) {
        std::ostringstream msg;
        msg << "BCSR source mismatch for slot " << slot
            << ": existing owner=" << it->second.owner
            << " path=" << it->second.path
            << " size=" << it->second.identity.file_size
            << " descriptor=0x" << std::hex << it->second.identity.descriptor_fingerprint
            << " content=0x" << it->second.identity.content_fingerprint
            << "; observed owner=" << (owner ? owner : "unknown")
            << " path=" << path
            << " size=" << std::dec << identity.file_size
            << " descriptor=0x" << std::hex << identity.descriptor_fingerprint
            << " content=0x" << identity.content_fingerprint;
        *error_out = msg.str();
    }
    return false;
}

}} // namespace SST::SnnDL
