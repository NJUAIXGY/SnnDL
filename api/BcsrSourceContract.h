// -*- c++ -*-
//
// Process-local contract for the BCSR source used by routing and loading.
// The implementation lives in libSnnDLRegistry so independently loaded SnnDL
// libraries observe one registry rather than one copy per shared object.
//
#pragma once

#include <cstdint>
#include <string>

namespace SST { namespace SnnDL {

struct BcsrSourceIdentity {
    uint64_t descriptor_fingerprint = 0;
    uint64_t content_fingerprint = 0;
    uint64_t file_size = 0;
};

// Bind one logical PE/core slot to the BCSR source it observes.  Rebinding the
// same slot is idempotent only when identity matches; a path/content mismatch
// is returned as false and described in error_out.
bool bindBcsrSourceContract(const std::string& slot,
                            const std::string& path,
                            const BcsrSourceIdentity& identity,
                            const char* owner,
                            std::string* error_out = nullptr);

}} // namespace SST::SnnDL
