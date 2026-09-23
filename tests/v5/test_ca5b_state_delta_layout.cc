#include "v5/core/CorePipeline.h"
#include "v5/storage/CoreStorageV5.h"

#include <cassert>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace SST::SnnDL::v5;
using AddressSpaceId = ::SnnDL::v5::AddressSpaceId;

namespace {

CoreStorageV5Config storageConfig(std::uint32_t neurons, std::uint32_t slots, const std::string& path) {
    CoreStorageV5Config config;
    config.neurons = neurons;
    config.max_delta_entries_per_neuron = slots;
    config.execution_profile = CoreStorageExecutionProfile::AsyncCompletion;
    config.descriptor_json = path;
    config.index_bytes = 4096;
    config.route_bytes = 4096;
    return config;
}

void testIdentityFileMatchesImplicitLayout() {
    auto described = storageConfig(3, 3, "/tmp/snndl_ca5b_identity.json");
    auto implicit = described;
    implicit.descriptor_json.clear();
    CoreStorageV5 from_compiler(described);
    CoreStorageV5 from_builtin(implicit);
    assert(from_compiler.layout().identityOrder());
    assert(!from_compiler.layout().digest().empty());
    for (std::uint32_t neuron = 0; neuron < 3; ++neuron) {
        assert(from_compiler.stateByteOffset(neuron) == from_builtin.stateByteOffset(neuron));
        assert(from_compiler.deltaCountByteOffset(neuron) == from_builtin.deltaCountByteOffset(neuron));
        for (std::uint32_t slot = 0; slot < 3; ++slot) {
            assert(from_compiler.deltaEntryByteOffset(neuron, slot) == from_builtin.deltaEntryByteOffset(neuron, slot));
        }
    }
    assert(from_compiler.stateByteOffset(1) == 8);
    assert(from_compiler.deltaCountByteOffset(2) == 8);
    assert(from_compiler.deltaEntryByteOffset(1, 2) == from_builtin.deltaEntryByteOffset(1, 2));
}

void testDescriptorRequiresAsync() {
    auto config = storageConfig(3, 3, "/tmp/snndl_ca5b_identity.json");
    config.execution_profile = CoreStorageExecutionProfile::LegacyBlocking;
    bool rejected = false;
    try {
        CoreStorageV5 storage(config);
        (void)storage;
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    assert(rejected);
}

void testInvalidDescriptorsFailClosed() {
    for (const char* path : {"/tmp/snndl_ca5b_alias.json", "/tmp/snndl_ca5b_bad_digest.json",
                             "/tmp/snndl_ca5b_misaligned.json", "/tmp/snndl_ca5b_global_base.json"}) {
        bool rejected = false;
        try {
            CoreStorageV5 storage(storageConfig(3, 3, path));
            (void)storage;
        } catch (const std::invalid_argument&) {
            rejected = true;
        }
        assert(rejected);
    }
}

struct Run {
    std::uint64_t hash = 0;
    float membrane = 0.0f;
    std::vector<std::uint64_t> state_addresses;
    bool completion_ordered = true;
};

Run runCase(const std::string& descriptor, const std::string& trace, std::uint32_t neurons, std::uint32_t slots,
            bool spike) {
    CorePipelineConfig config;
    config.neurons = neurons;
    config.execution_profile = CoreStorageExecutionProfile::AsyncCompletion;
    config.lif.threshold = 2.0f;
    config.storage.max_delta_entries_per_neuron = slots;
    config.storage.descriptor_json = descriptor;
    config.storage.trace_json = trace;
    config.storage.state_sram.banks = 2;
    config.storage.state_sram.interleave_bytes = 4;
    config.storage.state_sram.ports_per_bank = 1;
    Run result;
    {
        CorePipeline pipeline(config);
        pipeline.start(0);
        if (spike) {
            assert(pipeline.submitSpike(SpikeInput{0, 0, 7}));
            std::vector<RowRequest> requests;
            for (int i = 0; i < 256 && requests.empty(); ++i) {
                pipeline.tick();
                requests = pipeline.takeRowRequests();
            }
            assert(requests.size() == 1);
            assert(pipeline.acceptSynapseResponse(SynapseResponse{0, 0, 7, 1, 0, 0.6f, false, 0}));
            assert(pipeline.acceptSynapseResponse(SynapseResponse{0, 0, 7, 1, 1, 0.5f, false, 0}));
            assert(pipeline.acceptRowDone(RowDone{0, 0, 7, 2}));
        }
        pipeline.sealIngress();
        for (int i = 0; i < 100000 && !pipeline.readyToCommit(); ++i) pipeline.tick();
        assert(pipeline.readyToCommit());
        assert(pipeline.storage().drained());
        assert(pipeline.storage().pending() == 0);
        result.membrane = pipeline.state()[1].membrane;
        result.hash = pipeline.functionalHash();
        pipeline.commit();
    }
    std::ifstream input(trace);
    std::string line;
    while (std::getline(input, line)) {
        if (line.find("\"completed\":true") == std::string::npos) continue;
        const auto consume_at = line.find("\"consume_cycle\":");
        const auto completion_at = line.find("\"completion_cycle\":");
        assert(consume_at != std::string::npos && completion_at != std::string::npos);
        const auto consume = std::strtoull(line.c_str() + consume_at + std::string("\"consume_cycle\":").size(), nullptr, 10);
        const auto completion = std::strtoull(line.c_str() + completion_at + std::string("\"completion_cycle\":").size(), nullptr, 10);
        if (consume < completion) result.completion_ordered = false;
        if (line.find("\"region_id\":0") != std::string::npos) {
            const auto address_at = line.find("\"address\":");
            result.state_addresses.push_back(std::strtoull(line.c_str() + address_at + std::string("\"address\":").size(), nullptr, 10));
        }
    }
    return result;
}

void testAsyncFunctionalParityAndBankObservation() {
    const auto identity = runCase("/tmp/snndl_ca5b_identity.json", "/tmp/snndl_ca5b_identity_trace.jsonl", 3, 3, true);
    const auto fields = runCase("/tmp/snndl_ca5b_field_major.json", "/tmp/snndl_ca5b_field_trace.jsonl", 3, 3, true);
    const auto builtin = runCase("", "/tmp/snndl_ca5b_builtin_trace.jsonl", 3, 3, true);
    assert(identity.completion_ordered && fields.completion_ordered && builtin.completion_ordered);
    assert(identity.hash == fields.hash && fields.hash == builtin.hash);
    assert(identity.membrane == fields.membrane);
    assert(identity.membrane != 0.0f);
    auto contains = [](const std::vector<std::uint64_t>& values, std::uint64_t address) {
        for (const auto value : values) if (value == address) return true;
        return false;
    };
    assert(contains(identity.state_addresses, 8));
    assert(!contains(identity.state_addresses, 4));
    assert(contains(fields.state_addresses, 4));
    const auto slots = runCase("/tmp/snndl_ca5b_slot_major.json", "/tmp/snndl_ca5b_slot_trace.jsonl", 2, 2, true);
    const auto slot_identity = runCase("", "/tmp/snndl_ca5b_slot_identity_trace.jsonl", 2, 2, true);
    assert(slots.hash == slot_identity.hash);
    assert(slots.membrane == slot_identity.membrane);
    assert(slots.completion_ordered);
}

} // namespace

int main() {
    testIdentityFileMatchesImplicitLayout();
    testDescriptorRequiresAsync();
    testInvalidDescriptorsFailClosed();
    testAsyncFunctionalParityAndBankObservation();
    std::cout << "v5 CA-5B state/delta descriptor: PASS\n";
    return 0;
}
