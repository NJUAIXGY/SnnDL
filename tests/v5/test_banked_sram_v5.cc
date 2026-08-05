#include <cassert>
#include <cstdint>
#include <vector>

#include "v5/storage/BankedSramV5.h"

using SST::SnnDL::v5::BankedSramV5;
using SST::SnnDL::v5::BankedSramV5Config;
using SST::SnnDL::v5::BankedSramV5Reject;
using SST::SnnDL::v5::BankedSramV5Request;

static BankedSramV5Request writeRequest(std::uint64_t id, std::uint64_t address,
                                        std::initializer_list<std::uint8_t> bytes) {
    BankedSramV5Request request;
    request.request_id = id;
    request.address = address;
    request.write = true;
    request.data.assign(bytes.begin(), bytes.end());
    return request;
}

static BankedSramV5Request readRequest(std::uint64_t id, std::uint64_t address, std::size_t bytes) {
    BankedSramV5Request request;
    request.request_id = id;
    request.address = address;
    request.write = false;
    request.data.assign(bytes, 0);
    return request;
}

static void test_backing_round_trip_and_service_latency() {
    BankedSramV5Config config;
    config.capacity_bytes = 32;
    config.banks = 2;
    config.ports_per_bank = 1;
    config.interleave_bytes = 4;
    config.read_latency_cycles = 2;
    config.write_latency_cycles = 1;
    BankedSramV5 model(config);

    assert(model.accept(writeRequest(1, 0, {1, 2, 3, 4}), 0));
    model.tick(0);
    assert(model.takeResponses().empty());
    model.tick(1);
    auto write_responses = model.takeResponses();
    assert(write_responses.size() == 1);
    assert(write_responses[0].completion_cycle == 1);

    assert(model.accept(readRequest(2, 0, 4), 1));
    model.tick(1);
    model.tick(2);
    assert(model.takeResponses().empty());
    model.tick(3);
    auto read_responses = model.takeResponses();
    assert(read_responses.size() == 1);
    assert(read_responses[0].data == std::vector<std::uint8_t>({1, 2, 3, 4}));
    assert(model.stats().reads == 1);
    assert(model.stats().writes == 1);
    assert(model.stats().latency_cycles == 3);
}

static void test_bank_conflict_is_real_queueing() {
    BankedSramV5Config config;
    config.capacity_bytes = 64;
    config.banks = 2;
    config.ports_per_bank = 1;
    config.interleave_bytes = 4;
    config.read_latency_cycles = 1;
    config.request_queue_entries = 4;
    config.response_queue_entries = 4;
    BankedSramV5 model(config);

    assert(model.accept(readRequest(1, 0, 4), 0));
    assert(model.accept(readRequest(2, 8, 4), 0));
    model.tick(0);
    model.tick(1);
    assert(model.stats().bank_conflicts == 1);
    assert(model.stats().port_stall_cycles == 1);
    assert(model.stats().requests_issued == 2);
    assert(model.stats().requests_completed == 1);
    model.tick(2);
    assert(model.stats().requests_completed == 2);
}

static void test_queue_and_capacity_fail_closed() {
    BankedSramV5Config config;
    config.capacity_bytes = 8;
    config.banks = 1;
    config.request_queue_entries = 1;
    config.response_queue_entries = 1;
    BankedSramV5 model(config);

    assert(model.accept(readRequest(1, 0, 4), 0));
    SST::SnnDL::v5::BankedSramV5Response rejection;
    assert(!model.accept(readRequest(2, 4, 4), 0, &rejection));
    assert(rejection.reject == BankedSramV5Reject::QueueFull);
    assert(rejection.retryable);
    model.tick(0);
    assert(!model.accept(readRequest(3, 7, 2), 0, &rejection));
    assert(rejection.reject == BankedSramV5Reject::Capacity);
    assert(!rejection.retryable);
    assert(model.stats().retryable_rejects == 1);
    assert(model.stats().capacity_rejects == 1);

    model.tick(1);
    assert(!model.accept(readRequest(1, 0, 4), 1, &rejection));
    assert(rejection.reject == BankedSramV5Reject::Invalid);
}

int main() {
    test_backing_round_trip_and_service_latency();
    test_bank_conflict_is_real_queueing();
    test_queue_and_capacity_fail_closed();
    return 0;
}
