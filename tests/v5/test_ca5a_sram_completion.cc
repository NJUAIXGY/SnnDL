#include "v5/core/CorePipeline.h"
#include "v5/storage/CoreStorageV5.h"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

using namespace SST::SnnDL::v5;
using AddressSpaceId = ::SnnDL::v5::AddressSpaceId;

namespace {

CoreStorageV5Config asyncStorage(std::uint32_t neurons = 2) {
    CoreStorageV5Config config;
    config.core_id = 1;
    config.pe_id = 1;
    config.neurons = neurons;
    config.max_delta_entries_per_neuron = 4;
    config.index_bytes = 64;
    config.route_bytes = 64;
    config.execution_profile = CoreStorageExecutionProfile::AsyncCompletion;
    config.state_sram.banks = 1;
    config.delta_sram.banks = 1;
    config.index_sram.banks = 1;
    config.route_sram.banks = 1;
    config.state_sram.ports_per_bank = 1;
    config.delta_sram.ports_per_bank = 1;
    config.index_sram.ports_per_bank = 1;
    config.route_sram.ports_per_bank = 1;
    config.state_sram.request_queue_entries = 8;
    config.delta_sram.request_queue_entries = 8;
    config.index_sram.request_queue_entries = 8;
    config.route_sram.request_queue_entries = 8;
    config.state_sram.response_queue_entries = 8;
    config.delta_sram.response_queue_entries = 8;
    config.index_sram.response_queue_entries = 8;
    config.route_sram.response_queue_entries = 8;
    return config;
}

void setLatency(CoreStorageV5Config& config, std::uint32_t latency) {
    config.state_sram.read_latency_cycles = latency;
    config.state_sram.write_latency_cycles = latency;
    config.delta_sram.read_latency_cycles = latency;
    config.delta_sram.write_latency_cycles = latency;
    config.index_sram.read_latency_cycles = latency;
    config.index_sram.write_latency_cycles = latency;
    config.route_sram.read_latency_cycles = latency;
    config.route_sram.write_latency_cycles = latency;
}

void setLatency(CorePipelineConfig& config, std::uint32_t latency) {
    setLatency(config.storage, latency);
}

bool fieldBool(const std::string& line, const char* key) {
    const auto marker = std::string("\"") + key + "\":";
    const auto pos = line.find(marker);
    if (pos == std::string::npos) return false;
    return line.compare(pos + marker.size(), 4, "true") == 0;
}

bool fieldNumber(const std::string& line, const char* key, std::uint64_t& value) {
    const auto marker = std::string("\"") + key + "\":";
    const auto pos = line.find(marker);
    if (pos == std::string::npos) return false;
    value = std::strtoull(line.c_str() + pos + marker.size(), nullptr, 10);
    return true;
}

std::vector<std::string> readLines(const std::string& path) {
    std::ifstream input(path);
    std::vector<std::string> lines;
    std::string line;
    while (std::getline(input, line)) {
        if (!line.empty()) lines.push_back(line);
    }
    return lines;
}

void drainToken(CoreStorageV5& storage, StorageToken token, std::uint64_t& cycle) {
    for (int guard = 0; guard < 10000; ++guard) {
        storage.advance(cycle);
        auto completions = storage.takeCompletions();
        for (const auto& completion : completions) {
            assert(completion.ok);
            assert(cycle >= completion.completion_cycle);
            assert(storage.consume(completion.token));
        }
        storage.admit(cycle);
        if (storage.drained()) return;
        if (cycle == UINT64_MAX) break;
        ++cycle;
    }
    assert(storage.drained());
    (void)token;
}

void testLatencyFormula() {
    std::vector<std::uint64_t> observed;
    for (const std::uint32_t latency : {1u, 2u, 5u}) {
        auto config = asyncStorage(1);
        setLatency(config, latency);
        CoreStorageV5 storage(config);
        storage.beginTimestep(3);
        storage.beginEpoch();
        storage.advance(0);
        const auto submitted = storage.submitRead(
            AddressSpaceId::CoreState, storage.stateByteOffset(0), CoreStorageV5::kStateBytes, 0);
        assert(submitted.accepted);
        std::uint64_t cycle = 1;
        bool consumed = false;
        for (int guard = 0; guard < 1000 && !consumed; ++guard) {
            storage.advance(cycle);
            auto completions = storage.takeCompletions();
            if (completions.empty()) {
                assert(storage.pending() > 0);
            } else {
                assert(completions.size() == 1);
                assert(completions[0].token.id == submitted.token.id);
                assert(completions[0].completion_cycle == completions[0].service_cycle + latency);
                assert(cycle >= completions[0].completion_cycle);
                assert(cycle == static_cast<std::uint64_t>(latency) + 1);
                assert(storage.consume(completions[0].token));
                consumed = true;
            }
            if (!consumed) ++cycle;
        }
        assert(consumed);
        assert(storage.drained());
        observed.push_back(cycle);
    }
    assert(observed[0] < observed[1] && observed[1] < observed[2]);
    assert(observed[0] == 2 && observed[1] == 3 && observed[2] == 6);
}

void testRegionsAdvanceOnce() {
    CoreStorageV5 storage(asyncStorage());
    storage.advance(4);
    assert(storage.advanceCount(AddressSpaceId::CoreState) == 1);
    assert(storage.advanceCount(AddressSpaceId::CoreDelta) == 1);
    assert(storage.advanceCount(AddressSpaceId::CoreIndex) == 1);
    assert(storage.advanceCount(AddressSpaceId::PeRoute) == 1);
    bool rejected = false;
    try {
        storage.advance(4);
    } catch (const std::logic_error&) {
        rejected = true;
    }
    assert(rejected);
    storage.advance(5);
    assert(storage.advanceCount(AddressSpaceId::CoreState) == 2);
    assert(storage.advanceCount(AddressSpaceId::CoreDelta) == 2);
}

void testRetryDoesNotDropOrRepeat() {
    auto config = asyncStorage();
    config.state_sram.request_queue_entries = 1;
    config.state_sram.response_queue_entries = 1;
    CoreStorageV5 storage(config);
    storage.beginEpoch();
    const LifNeuronState first{1.25f, 3};
    const LifNeuronState second{2.5f, 9};
    const auto first_submit = storage.submitWrite(
        AddressSpaceId::CoreState, storage.stateByteOffset(0), CoreStorageV5::encodeLifRecord(first), 0);
    const auto second_submit = storage.submitWrite(
        AddressSpaceId::CoreState, storage.stateByteOffset(1), CoreStorageV5::encodeLifRecord(second), 0);
    assert(first_submit.accepted);
    assert(second_submit.token);
    assert(!second_submit.accepted);
    assert(second_submit.retryable);
    std::uint32_t second_attempt = 0;
    std::uint64_t second_logical = 0;
    int second_completions = 0;
    std::uint64_t cycle = 0;
    for (int guard = 0; guard < 10000 && !storage.drained(); ++guard) {
        storage.advance(cycle);
        for (const auto& completion : storage.takeCompletions()) {
            assert(completion.ok);
            assert(cycle >= completion.completion_cycle);
            if (completion.token.id == second_submit.token.id) {
                ++second_completions;
                second_attempt = completion.attempt_id;
                second_logical = completion.logical_operation_id;
            }
            assert(storage.consume(completion.token));
        }
        storage.admit(cycle);
        ++cycle;
    }
    assert(storage.drained());
    assert(second_completions == 1);
    assert(second_attempt >= 2);
    assert(second_logical != 0);
    assert(storage.stats(AddressSpaceId::CoreState).writes == 2);
    LifNeuronState restored;
    assert(storage.readCompletedState(0, restored));
    assert(restored.membrane == first.membrane && restored.refractory == first.refractory);
    assert(storage.readCompletedState(1, restored));
    assert(restored.membrane == second.membrane && restored.refractory == second.refractory);
}

void testCompletionOrderBindsTokens() {
    CoreStorageV5 storage(asyncStorage());
    storage.beginEpoch();
    const LifNeuronState first{0.5f, 1};
    const LifNeuronState second{-1.0f, 4};
    const auto first_bytes = CoreStorageV5::encodeLifRecord(first);
    const auto second_bytes = CoreStorageV5::encodeLifRecord(second);
    assert(storage.submitWrite(AddressSpaceId::CoreState, storage.stateByteOffset(0), first_bytes, 0).accepted);
    assert(storage.submitWrite(AddressSpaceId::CoreState, storage.stateByteOffset(1), second_bytes, 0).accepted);
    std::uint64_t cycle = 0;
    drainToken(storage, {}, cycle);
    ++cycle;
    const auto first_read = storage.submitRead(
        AddressSpaceId::CoreState, storage.stateByteOffset(0), CoreStorageV5::kStateBytes, cycle);
    const auto second_read = storage.submitRead(
        AddressSpaceId::CoreState, storage.stateByteOffset(1), CoreStorageV5::kStateBytes, cycle);
    assert(first_read.token && second_read.token);
    std::map<std::uint64_t, std::vector<std::uint8_t>> expected;
    expected[first_read.token.id] = first_bytes;
    expected[second_read.token.id] = second_bytes;
    std::vector<StorageCompletion> completions;
    for (int guard = 0; guard < 10000 && completions.size() < 2; ++guard) {
        storage.advance(cycle);
        auto ready = storage.takeCompletions();
        completions.insert(completions.end(), ready.begin(), ready.end());
        storage.admit(cycle);
        ++cycle;
    }
    assert(completions.size() == 2);
    std::reverse(completions.begin(), completions.end());
    for (const auto& completion : completions) {
        assert(completion.ok);
        assert(completion.data == expected.at(completion.token.id));
        assert(storage.consume(completion.token));
    }
    assert(storage.drained());
}

void testEpochRefusesOutstanding() {
    CoreStorageV5 storage(asyncStorage());
    storage.beginEpoch();
    assert(storage.epoch() == 1);
    const auto submitted = storage.submitWrite(
        AddressSpaceId::CoreState, storage.stateByteOffset(0),
        CoreStorageV5::encodeLifRecord(LifNeuronState{1.0f, 1}), 0);
    assert(submitted.token);
    assert(!storage.drained());
    bool rejected = false;
    try {
        storage.beginEpoch();
    } catch (const std::logic_error&) {
        rejected = true;
    }
    assert(rejected);
    assert(storage.epoch() == 1);
    std::uint64_t cycle = 0;
    drainToken(storage, submitted.token, cycle);
    storage.beginEpoch();
    assert(storage.epoch() == 2);
    assert(storage.drained());
}

void testPortConflictIsObservable() {
    auto config = asyncStorage();
    config.state_sram.banks = 1;
    config.state_sram.ports_per_bank = 1;
    config.state_sram.read_latency_cycles = 2;
    config.state_sram.write_latency_cycles = 2;
    const std::string path = "/tmp/snndl_ca5a_port_conflict.jsonl";
    std::remove(path.c_str());
    config.trace_json = path;
    CoreStorageV5 storage(config);
    storage.beginTimestep(1);
    storage.beginEpoch();
    assert(storage.submitWrite(AddressSpaceId::CoreState, storage.stateByteOffset(0),
                               CoreStorageV5::encodeLifRecord(LifNeuronState{1.0f, 1}), 0).accepted);
    assert(storage.submitWrite(AddressSpaceId::CoreState, storage.stateByteOffset(1),
                               CoreStorageV5::encodeLifRecord(LifNeuronState{2.0f, 2}), 0).accepted);
    storage.advance(0);
    assert(storage.stats(AddressSpaceId::CoreState).bank_conflicts > 0);
    std::uint64_t cycle = 1;
    bool saw_port_busy = false;
    std::uint64_t previous_completion = 0;
    bool have_previous = false;
    for (int guard = 0; guard < 10000 && !storage.drained(); ++guard) {
        storage.advance(cycle);
        for (const auto& completion : storage.takeCompletions()) {
            if (have_previous) assert(completion.completion_cycle >= previous_completion);
            previous_completion = completion.completion_cycle;
            have_previous = true;
            if (completion.stall_reason == "port_busy") saw_port_busy = true;
            assert(storage.consume(completion.token));
        }
        storage.admit(cycle);
        ++cycle;
    }
    assert(storage.drained());
    assert(saw_port_busy);
    bool trace_has_stall = false;
    for (const auto& line : readLines(path)) {
        if (line.find("\"stall_reason\":\"port_busy\"") != std::string::npos) trace_has_stall = true;
        assert(line.find("\"timebase\":\"core_pipeline_cycle\"") != std::string::npos);
    }
    assert(trace_has_stall);
    std::remove(path.c_str());
}

void testBlockingApiRejectedOnAsync() {
    CoreStorageV5 storage(asyncStorage());
    LifNeuronState state;
    bool rejected = false;
    try {
        (void)storage.readState(0, state);
    } catch (const std::logic_error&) {
        rejected = true;
    }
    assert(rejected);
    assert(storage.advanceCount(AddressSpaceId::CoreState) == 0);
}

struct PipelineResult {
    std::uint64_t cycles = 0;
    std::uint64_t hash = 0;
    float membrane = 0.0f;
};

PipelineResult runSeal(std::uint32_t neurons, std::uint32_t latency, const std::string& trace) {
    CorePipelineConfig config;
    config.neurons = neurons;
    config.neuron.width = 1;
    config.lif.threshold = 1.0f;
    config.execution_profile = CoreStorageExecutionProfile::AsyncCompletion;
    setLatency(config, latency);
    config.storage.trace_json = trace;
    config.storage.state_sram.banks = 1;
    config.storage.delta_sram.banks = 1;
    config.storage.state_sram.ports_per_bank = 1;
    config.storage.delta_sram.ports_per_bank = 1;
    PipelineResult result;
    {
        CorePipeline pipeline(config);
        pipeline.start(0);
        assert(pipeline.storage().advanceCount(AddressSpaceId::CoreState) == 0);
        pipeline.tick();
        assert(pipeline.storage().advanceCount(AddressSpaceId::CoreState) == 1);
        assert(pipeline.storage().advanceCount(AddressSpaceId::CoreDelta) == 1);
        assert(pipeline.storage().advanceCount(AddressSpaceId::CoreIndex) == 1);
        assert(pipeline.storage().advanceCount(AddressSpaceId::PeRoute) == 1);
        assert(!pipeline.readyToCommit());
        bool early_commit = false;
        try {
            pipeline.commit();
        } catch (const std::logic_error&) {
            early_commit = true;
        }
        assert(early_commit);
        pipeline.sealIngress();
        for (int i = 0; i < 100000 && !pipeline.readyToCommit(); ++i) pipeline.tick();
        assert(pipeline.readyToCommit());
        assert(pipeline.storage().drained());
        assert(pipeline.storage().pending() == 0);
        const auto advances = pipeline.storage().advanceCount(AddressSpaceId::CoreState);
        assert(advances == pipeline.stats().cycles);
        assert(advances == pipeline.storage().advanceCount(AddressSpaceId::CoreDelta));
        const auto before = advances;
        result.membrane = pipeline.state()[0].membrane;
        result.hash = pipeline.functionalHash();
        assert(pipeline.storage().advanceCount(AddressSpaceId::CoreState) == before);
        result.cycles = pipeline.stats().cycles;
        pipeline.commit();
        pipeline.start(1);
        pipeline.sealIngress();
        for (int i = 0; i < 100000 && !pipeline.readyToCommit(); ++i) pipeline.tick();
        assert(pipeline.readyToCommit());
        assert(pipeline.storage().drained());
        pipeline.commit();
    }
    return result;
}

void testPipelineLatencyAndChain() {
    const std::string path = "/tmp/snndl_ca5a_pipeline_latency.jsonl";
    std::remove(path.c_str());
    const auto latency1 = runSeal(1, 1, path);
    auto lines = readLines(path);
    assert(!lines.empty());
    bool saw_read = false;
    bool checked_write = false;
    std::uint64_t read_consume = 0;
    for (const auto& line : lines) {
        assert(line.find("\"timebase\":\"core_pipeline_cycle\"") != std::string::npos);
        assert(line.find("\"timestep\":") != std::string::npos);
        std::uint64_t consume = 0;
        std::uint64_t completion = 0;
        std::uint64_t service = 0;
        std::uint64_t issue = 0;
        std::uint64_t region = 0;
        if (!fieldBool(line, "completed")) continue;
        assert(fieldNumber(line, "consume_cycle", consume));
        assert(fieldNumber(line, "completion_cycle", completion));
        assert(fieldNumber(line, "service_cycle", service));
        assert(fieldNumber(line, "issue_cycle", issue));
        assert(fieldNumber(line, "region_id", region));
        assert(consume >= completion);
        assert(completion == service + 1);
        const bool state_region = region == static_cast<std::uint64_t>(AddressSpaceId::CoreState);
        const bool is_write = line.find("\"write\":true") != std::string::npos;
        if (!checked_write && state_region && !is_write && !saw_read) {
            saw_read = true;
            read_consume = consume;
        } else if (!checked_write && state_region && is_write && saw_read) {
            assert(issue >= read_consume);
            checked_write = true;
        }
    }
    assert(saw_read && checked_write);
    std::remove(path.c_str());

    const auto latency2 = runSeal(1, 2, "");
    const auto latency5 = runSeal(1, 5, "");
    assert(latency1.hash == latency2.hash && latency2.hash == latency5.hash);
    assert(latency1.membrane == latency2.membrane);
    assert(latency1.cycles < latency2.cycles && latency2.cycles < latency5.cycles);
}

PipelineResult runStimulus(CoreStorageExecutionProfile profile, std::uint32_t latency) {
    CorePipelineConfig config;
    config.neurons = 4;
    config.ingress_entries = 4;
    config.row_entries = 4;
    config.synapse_entries = 8;
    config.retire_entries = 8;
    config.accumulator_entries = 8;
    config.held_spike_entries = 8;
    config.lif.threshold = 1.0f;
    config.execution_profile = profile;
    setLatency(config, latency);
    CorePipeline pipeline(config);
    pipeline.start(0);
    assert(pipeline.submitSpike(SpikeInput{0, 0, 7}));
    std::vector<RowRequest> requests;
    for (int i = 0; i < 256 && requests.empty(); ++i) {
        pipeline.tick();
        requests = pipeline.takeRowRequests();
    }
    assert(requests.size() == 1);
    assert(pipeline.acceptSynapseResponse(SynapseResponse{0, 0, 7, 1, 1, 0.5f, false, 0}));
    assert(pipeline.acceptSynapseResponse(SynapseResponse{0, 0, 7, 1, 0, 0.6f, false, 0}));
    assert(pipeline.acceptRowDone(RowDone{0, 0, 7, 2}));
    pipeline.sealIngress();
    for (int i = 0; i < 100000 && !pipeline.readyToCommit(); ++i) pipeline.tick();
    assert(pipeline.readyToCommit());
    assert(pipeline.storage().drained());
    assert(pipeline.stats().neurons_fired == 1);
    PipelineResult result;
    result.membrane = pipeline.state()[1].membrane;
    result.hash = pipeline.functionalHash();
    result.cycles = pipeline.stats().cycles;
    pipeline.commit();
    return result;
}

void testLegacyMatchesAsyncFunction() {
    const auto async_fast = runStimulus(CoreStorageExecutionProfile::AsyncCompletion, 1);
    const auto legacy = runStimulus(CoreStorageExecutionProfile::LegacyBlocking, 4);
    const auto async_slow = runStimulus(CoreStorageExecutionProfile::AsyncCompletion, 4);
    assert(async_fast.hash == legacy.hash && legacy.hash == async_slow.hash);
    assert(async_fast.membrane == 0.0f && legacy.membrane == 0.0f);
    assert(async_slow.cycles > async_fast.cycles);
    assert(async_slow.cycles > legacy.cycles);
}

} // namespace

int main() {
    testLatencyFormula();
    testRegionsAdvanceOnce();
    testRetryDoesNotDropOrRepeat();
    testCompletionOrderBindsTokens();
    testEpochRefusesOutstanding();
    testPortConflictIsObservable();
    testBlockingApiRejectedOnAsync();
    testPipelineLatencyAndChain();
    testLegacyMatchesAsyncFunction();
    std::cout << "v5 CA-5A SRAM completion: PASS\n";
    return 0;
}
