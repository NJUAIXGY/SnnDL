#include <iostream>
#include <string>

#include "api/BcsrSourceContract.h"

using SST::SnnDL::BcsrSourceIdentity;
using SST::SnnDL::bindBcsrSourceContract;

namespace {

bool expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        return false;
    }
    return true;
}

} // namespace

int main() {
    const BcsrSourceIdentity source{0x11u, 0x22u, 128u};
    std::string error;

    if (!expect(bindBcsrSourceContract("test/contract/0", "/tmp/a.bcsr",
                                      source, "test", &error),
                "first binding is accepted")) return 1;
    if (!expect(error.empty(), "successful binding clears the error")) return 1;
    if (!expect(bindBcsrSourceContract("test/contract/0", "/tmp/alias.bcsr",
                                      source, "test-repeat", &error),
                "matching identity is idempotent")) return 1;
    if (!expect(!bindBcsrSourceContract(
                    "test/contract/0", "/tmp/changed.bcsr",
                    BcsrSourceIdentity{0x11u, 0x33u, 128u}, "test-mismatch",
                    &error),
                "content mismatch is rejected")) return 1;
    if (!expect(!error.empty(), "mismatch reports an error")) return 1;
    if (!expect(!bindBcsrSourceContract("", "/tmp/invalid.bcsr", source,
                                       "test-invalid", &error),
                "empty slot is rejected")) return 1;
    if (!expect(bindBcsrSourceContract("test/contract/descriptor-only",
                                      "/tmp/descriptor.bcsr",
                                      BcsrSourceIdentity{0x44u, 0u, 256u},
                                      "test", &error),
                "descriptor-only identity is accepted")) return 1;
    if (!expect(!bindBcsrSourceContract(
                    "test/contract/descriptor-only", "/tmp/changed.bcsr",
                    BcsrSourceIdentity{0x45u, 0u, 256u}, "test-mismatch",
                    &error),
                "descriptor mismatch is rejected")) return 1;

    std::cout << "BCSR source contract: PASS\n";
    return 0;
}
