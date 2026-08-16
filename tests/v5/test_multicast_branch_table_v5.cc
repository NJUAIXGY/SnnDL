#include "v5/noc/MulticastBranchTableV5.h"
#include "v5/noc/MulticastCreditV5.h"

#include <cassert>
#include <iostream>
#include <stdexcept>

using namespace SST::SnnDL::v5;

namespace {

void expectInvalid(const char* encoded) {
    bool rejected = false;
    try {
        (void)parseMulticastBranchTableV5(encoded);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
}

} // namespace

int main() {
    const auto table = parseMulticastBranchTableV5("1:17:3;2:0:0;3:4:0");
    assert(table.size() == 3);
    assert(table.at(1).output_mask == 17);
    assert(table.at(1).local_core_mask == 3);
    assert(table.at(2).output_mask == 0);
    assert(table.at(2).local_core_mask == 0);

    expectInvalid("0:1:0");
    expectInvalid("1:16:0");
    expectInvalid("1:0:1");
    expectInvalid("1:32:0");
    expectInvalid("1:256:0");
    expectInvalid("-1:1:0");
    expectInvalid("1x:1:0");
    expectInvalid("18446744073709551616:1:0");
    expectInvalid("1:1:0;1:2:0");
    expectInvalid("1:1");
    expectInvalid("1:1:0:4");

    MulticastCreditV5 credits;
    credits.reset(2);
    assert(credits.capacity() == 2 && credits.available() == 2);
    assert(credits.consume());
    assert(credits.consume());
    assert(!credits.consume());
    assert(credits.restore());
    assert(credits.available() == 1);
    assert(credits.consume());
    assert(!credits.restore(3));
    assert(credits.restore(2));
    assert(credits.available() == 2);

    std::cout << "v5 multicast branch table: PASS\n";
    return 0;
}
