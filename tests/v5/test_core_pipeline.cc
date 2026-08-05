#include "v5/core/CorePipeline.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace SST::SnnDL::v5;

namespace {

CorePipelineConfig baseConfig(std::uint32_t neurons = 4) {
    CorePipelineConfig config;
    config.neurons = neurons;
    config.ingress_entries = 4;
    config.row_entries = 4;
    config.synapse_entries = 8;
    config.retire_entries = 8;
    config.accumulator_entries = 8;
    config.held_spike_entries = 8;
    config.ingress.width = 1;
    config.row_lookup.width = 1;
    config.synapse.width = 1;
    config.retire.width = 1;
    config.accumulator.width = 1;
    config.neuron.width = 1;
    config.lif.threshold = 1.0f;
    config.lif.refractory_timesteps = 0;
    return config;
}

std::uint64_t runOne(bool reverse) {
    auto config = baseConfig();
    CorePipeline pipeline(config);
    pipeline.start(0);
    assert(pipeline.submitSpike(SpikeInput{0, 0, 7}));
    std::vector<RowRequest> requests;
    for (int i = 0; i < 16 && requests.empty(); ++i) {
        pipeline.tick();
        requests = pipeline.takeRowRequests();
    }
    assert(requests.size() == 1);
    const std::vector<SynapseResponse> responses = {
        SynapseResponse{0, 0, 7, 1, 0, 0.6f, false, 0},
        SynapseResponse{0, 0, 7, 1, 1, 0.5f, false, 0},
    };
    if (reverse) {
        assert(pipeline.acceptSynapseResponse(responses[1]));
        assert(pipeline.acceptSynapseResponse(responses[0]));
    } else {
        assert(pipeline.acceptSynapseResponse(responses[0]));
        assert(pipeline.acceptSynapseResponse(responses[1]));
    }
    assert(pipeline.acceptRowDone(RowDone{0, 0, 7, 2}));
    pipeline.sealIngress();
    for (int i = 0; i < 128 && !pipeline.readyToCommit(); ++i) pipeline.tick();
    assert(pipeline.readyToCommit());
    assert(pipeline.state()[1].membrane == 0.0f);
    assert(pipeline.stats().neurons_fired == 1);
    const auto hash = pipeline.functionalHash();
    pipeline.commit();
    return hash;
}

std::uint64_t scanCycles(std::uint32_t neurons, std::uint32_t lanes) {
    auto config = baseConfig(neurons);
    config.neuron.width = lanes;
    config.held_spike_entries = neurons + 1;
    CorePipeline pipeline(config);
    pipeline.start(0);
    pipeline.sealIngress();
    for (int i = 0; i < 4096 && !pipeline.readyToCommit(); ++i) pipeline.tick();
    assert(pipeline.readyToCommit());
    return pipeline.stats().cycles;
}

void testQueueBackpressure() {
    auto config = baseConfig();
    config.ingress_entries = 1;
    CorePipeline pipeline(config);
    pipeline.start(0);
    assert(pipeline.submitSpike(SpikeInput{0, 0, 1}));
    assert(!pipeline.submitSpike(SpikeInput{0, 1, 2}));
    pipeline.tick();
    pipeline.tick();
    assert(pipeline.submitSpike(SpikeInput{0, 1, 2}));
    assert(pipeline.stats().ingress_full_cycles > 0);
}

void testStageBackpressureAndCounters() {
    auto config = baseConfig(8);
    config.ingress_entries = 3;
    config.row_entries = 1;
    config.synapse_entries = 1;
    config.retire_entries = 1;
    config.accumulator_entries = 1;
    config.accumulator.latency_cycles = 4;
    config.ingress.width = 1;
    config.row_lookup.width = 1;
    config.synapse.width = 1;
    config.retire.width = 1;
    config.accumulator.width = 1;
    config.neuron.width = 8;
    CorePipeline pipeline(config);
    pipeline.start(0);
    assert(pipeline.submitSpike(SpikeInput{0, 0, 1}));
    assert(pipeline.submitSpike(SpikeInput{0, 0, 2}));
    assert(pipeline.submitSpike(SpikeInput{0, 0, 3}));

    std::vector<RowRequest> requests;
    for (int i = 0; i < 32 && requests.size() < 3; ++i) {
        pipeline.tick();
        auto batch = pipeline.takeRowRequests();
        requests.insert(requests.end(), batch.begin(), batch.end());
    }
    assert(requests.size() == 3);
    for (const auto& request : requests) {
        assert(pipeline.acceptSynapseResponse(SynapseResponse{
            request.timestep, request.source_neuron, request.source_event_seq,
            1, request.row_id, 1.0f, false, 0}));
        assert(pipeline.acceptRowDone(RowDone{
            request.timestep, request.source_neuron, request.source_event_seq, 1}));
        for (int i = 0; i < 8; ++i) pipeline.tick();
    }
    pipeline.sealIngress();
    for (int i = 0; i < 512 && !pipeline.readyToCommit(); ++i) pipeline.tick();
    assert(pipeline.readyToCommit());
    assert(pipeline.stats().ingress.full_cycles > 0 || pipeline.stats().ingress.stall_cycles > 0);
    assert(pipeline.stats().row_lookup.full_cycles > 0 || pipeline.stats().row_lookup.stall_cycles > 0);
    assert(pipeline.stats().synapse.issued == pipeline.stats().synapse.accepted);
    assert(pipeline.stats().retire.issued == pipeline.stats().retire.accepted);
    assert(pipeline.stats().accumulator.accepted == pipeline.stats().accumulator.completed);
    assert(pipeline.stats().neuron.completed == 8);
}

void testDeltaCapacityIsIndependentOfRetireQueue() {
    auto config = baseConfig(4);
    config.retire_entries = 1;
    // Leave the storage capacity at its default. Two edges targeting one
    // neuron must still drain through a one-entry transient retire queue.
    CorePipeline pipeline(config);
    pipeline.start(0);
    assert(pipeline.submitSpike(SpikeInput{0, 0, 11}));

    std::vector<RowRequest> requests;
    for (int i = 0; i < 32 && requests.empty(); ++i) {
        pipeline.tick();
        requests = pipeline.takeRowRequests();
    }
    assert(requests.size() == 1);
    assert(pipeline.acceptSynapseResponse(
        SynapseResponse{0, 0, 11, 1, 0, 0.6f, false, 0}));
    assert(pipeline.acceptSynapseResponse(
        SynapseResponse{0, 0, 11, 1, 1, 0.5f, false, 0}));
    assert(pipeline.acceptRowDone(RowDone{0, 0, 11, 2}));
    pipeline.sealIngress();
    for (int i = 0; i < 512 && !pipeline.readyToCommit(); ++i) pipeline.tick();
    assert(pipeline.readyToCommit());
    assert(pipeline.stats().retire_retired == 2);
}

void testUnrelatedStageIsolation() {
    auto baseline = baseConfig(64);
    baseline.neuron.width = 4;
    CorePipeline first(baseline);
    first.start(0);
    first.sealIngress();
    for (int i = 0; i < 512 && !first.readyToCommit(); ++i) first.tick();
    assert(first.readyToCommit());

    auto changed = baseline;
    changed.row_lookup.latency_cycles = 17;
    changed.synapse.latency_cycles = 23;
    changed.retire.latency_cycles = 11;
    CorePipeline second(changed);
    second.start(0);
    second.sealIngress();
    for (int i = 0; i < 512 && !second.readyToCommit(); ++i) second.tick();
    assert(second.readyToCommit());
    assert(first.stats().cycles == second.stats().cycles);
}

void testRetireOrder() {
    DeterministicRetireQueue queue(4);
    assert(queue.push(RetireEntry{RetireKey{2, 4, 0}, 0, 1.0f}));
    assert(queue.push(RetireEntry{RetireKey{1, 9, 0}, 0, 1.0f}));
    assert(queue.push(RetireEntry{RetireKey{1, 3, 1}, 0, 1.0f}));
    assert(queue.pop().key == (RetireKey{1, 3, 1}));
    assert(queue.pop().key == (RetireKey{1, 9, 0}));
    assert(queue.pop().key == (RetireKey{2, 4, 0}));
}

} // namespace

int main() {
    testRetireOrder();
    testQueueBackpressure();
    testStageBackpressureAndCounters();
    testDeltaCapacityIsIndependentOfRetireQueue();
    testUnrelatedStageIsolation();
    const auto forward_hash = runOne(false);
    const auto reverse_hash = runOne(true);
    assert(forward_hash == reverse_hash);

    const auto width1 = scanCycles(64, 1);
    const auto width2 = scanCycles(64, 2);
    const auto width4 = scanCycles(64, 4);
    assert(width1 > width2 && width2 > width4);
    assert(scanCycles(256, 4) > width4);
    assert(scanCycles(1024, 4) > scanCycles(256, 4));

    // Independent cores have independent cycle state and no static memory slot.
    for (std::uint32_t cores : {1u, 2u, 4u, 8u}) {
        std::vector<CorePipeline> bank;
        for (std::uint32_t core = 0; core < cores; ++core) bank.emplace_back(baseConfig(8));
        for (auto& pipeline : bank) {
            pipeline.start(0);
            pipeline.sealIngress();
            for (int i = 0; i < 256 && !pipeline.readyToCommit(); ++i) pipeline.tick();
            assert(pipeline.readyToCommit());
            pipeline.commit();
        }
    }

    std::cout << "v5 core pipeline: PASS\n";
    std::cout << "scan_cycles width1=" << width1 << " width2=" << width2 << " width4=" << width4 << "\n";
    return 0;
}
