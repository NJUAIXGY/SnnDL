// -*- c++ -*-
#pragma once

#include <cstdint>
#include <unordered_map>

namespace SST { namespace SnnDL {

class ITassLfP0Hooks {
public:
    struct LocalWindowReport {
        uint32_t seq = 0;
        uint32_t line_size_bytes = 64;
        uint64_t payload_bytes_total = 0;
        uint64_t current_vlf_line_groups_total = 0;
        std::unordered_map<uint32_t, uint64_t> pre_payload_bytes;
    };

    virtual ~ITassLfP0Hooks() = default;
    virtual bool takeTassLfP0LocalWindowReport(uint32_t seq, LocalWindowReport& out) = 0;
};

}} // namespace SST::SnnDL
