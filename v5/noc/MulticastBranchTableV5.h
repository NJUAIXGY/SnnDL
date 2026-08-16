#ifndef SST_SNN_DL_V5_MULTICAST_BRANCH_TABLE_V5_H
#define SST_SNN_DL_V5_MULTICAST_BRANCH_TABLE_V5_H

#include <cstdint>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace SST { namespace SnnDL { namespace v5 {

struct MulticastBranchActionV5 {
    std::uint8_t output_mask = 0;
    std::uint64_t local_core_mask = 0;
};

using MulticastBranchTableV5 = std::map<std::uint64_t, MulticastBranchActionV5>;

inline std::uint64_t parseMulticastUnsignedV5(const std::string& value) {
    if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos) {
        throw std::invalid_argument("multicast branch field must be an unsigned decimal integer");
    }
    try {
        return std::stoull(value);
    } catch (const std::exception&) {
        throw std::invalid_argument("multicast branch field is outside uint64 range");
    }
}

inline MulticastBranchTableV5 parseMulticastBranchTableV5(const std::string& encoded) {
    constexpr std::uint8_t kValidOutputs = 0x1f;
    constexpr std::uint8_t kLocalOutput = 1u << 4;
    MulticastBranchTableV5 table;
    std::stringstream records(encoded);
    std::string record;
    while (std::getline(records, record, ';')) {
        if (record.empty()) continue;
        std::stringstream fields(record);
        std::string route;
        std::string outputs;
        std::string local;
        std::string extra;
        if (!std::getline(fields, route, ':') || !std::getline(fields, outputs, ':') ||
            !std::getline(fields, local, ':') || std::getline(fields, extra, ':')) {
            throw std::invalid_argument("invalid branch table record");
        }
        const auto route_id = parseMulticastUnsignedV5(route);
        const auto output_value = parseMulticastUnsignedV5(outputs);
        const MulticastBranchActionV5 action{
            static_cast<std::uint8_t>(output_value), parseMulticastUnsignedV5(local)};
        const bool has_local_output = (action.output_mask & kLocalOutput) != 0;
        if (route_id == 0 || output_value > kValidOutputs ||
            has_local_output != (action.local_core_mask != 0) ||
            !table.emplace(route_id, action).second) {
            throw std::invalid_argument("invalid or duplicate multicast branch action");
        }
    }
    return table;
}

}}}
#endif
