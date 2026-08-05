#include <cassert>
#include <cstring>
#include <set>
#include <string>

#include "v5/api/StatisticNames.h"

int main() {
    assert(SnnDL::v5::kStatisticNameCount >= 64u);
    assert(std::strlen(SnnDL::v5::kStatisticsContractSha256) == 64u);
    std::set<std::string> names;
    for (const char* name : SnnDL::v5::kStatisticNames) {
        assert(name != nullptr && name[0] != '\0');
        names.emplace(name);
    }
    assert(names.size() == SnnDL::v5::kStatisticNameCount);
    assert(names.count("core.neuron.fired") == 1u);
    assert(names.count("timestep.committed") == 1u);
    assert(names.count("storage.sram.bank_conflicts") == 1u);
    assert(names.count("storage.sram.requests.retryable_rejects") == 1u);
    return 0;
}
