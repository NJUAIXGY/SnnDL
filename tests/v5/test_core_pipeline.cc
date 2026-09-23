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
    for (int i = 0; i < 64 && requests.empty(); ++i) {
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
    for (int i = 0; i < 8192 && !pipeline.readyToCommit(); ++i) pipeline.tick();
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
    for (int i = 0; i < 250000 && !pipeline.readyToCommit(); ++i) pipeline.tick();
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

void testScheduleAdmissionReservations() {
    auto config = baseConfig(4);
    config.schedule_admission_enabled = true;
    config.schedule_ingress_entries = 1;
    config.schedule_synapse_entries = 1;
    CorePipeline pipeline(config);
    pipeline.start(0);
    assert(pipeline.submitSpike(SpikeInput{0, 0, 31}));
    // The physical ingress queue has room for four entries, but the compiled
    // SchedulePlan reservation exposes only one admission slot.
    assert(!pipeline.submitSpike(SpikeInput{0, 1, 32}));
    std::vector<RowRequest> requests;
    for (int i = 0; i < 64 && requests.empty(); ++i) {
        pipeline.tick();
        requests = pipeline.takeRowRequests();
    }
    assert(requests.size() == 1);
    assert(pipeline.acceptSynapseResponse(
        SynapseResponse{0, 0, 31, 1, 0, 1.0f, false, 0}));
    assert(!pipeline.acceptSynapseResponse(
        SynapseResponse{0, 0, 31, 2, 1, 1.0f, false, 0}));
    bool released = false;
    for (int i = 0; i < 8 && !released; ++i) {
        pipeline.tick();
        released = pipeline.acceptSynapseResponse(
            SynapseResponse{0, 0, 31, 2, 1, 1.0f, true, 2});
    }
    assert(released);
}

void testScheduleEarliestCycleAndStageWidth() {
    auto config = baseConfig(4);
    config.ingress.width = 4;
    config.schedule_admission_enabled = true;
    config.schedule_ingress_entries = 4;
    config.schedule_synapse_entries = 4;
    const char* ids[] = {"preload", "ingress", "row-lookup", "synapse-issue",
                         "neuron-update", "retire", "seal", "commit", "drain"};
    for (std::size_t index = 0; index < 9; ++index) {
        ScheduleStageDescriptor stage;
        stage.id = ids[index];
        stage.operation = ids[index];
        stage.resource = ids[index];
        stage.request_class = ids[index];
        stage.retry_policy = "runtime";
        stage.issue_group_id = ids[index];
        stage.earliest_cycle = index == 1 ? 3 : 0;
        stage.max_inflight = index == 1 ? 1 : 4;
        if (index > 0) stage.dependencies.push_back(ids[index - 1]);
        config.schedule_stages.push_back(std::move(stage));
    }
    CorePipeline pipeline(config);
    pipeline.start(0);
    assert(pipeline.submitSpike(SpikeInput{0, 0, 41}));
    for (int cycle = 0; cycle < 3; ++cycle) {
        pipeline.tick();
        assert(pipeline.takeRowRequests().empty());
    }
    std::vector<RowRequest> requests;
    for (int cycle = 0; cycle < 16 && requests.empty(); ++cycle) {
        pipeline.tick();
        requests = pipeline.takeRowRequests();
    }
    assert(requests.size() == 1);
    assert(pipeline.acceptSynapseResponse(
        SynapseResponse{0, 0, 41, 1, 0, 1.0f, true, 1}));
    pipeline.sealIngress();
    for (int cycle = 0; cycle < 8192 && !pipeline.readyToCommit(); ++cycle) pipeline.tick();
    assert(pipeline.readyToCommit());
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
    for (int i = 0; i < 128 && requests.size() < 3; ++i) {
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
    for (int i = 0; i < 20000 && !pipeline.readyToCommit(); ++i) pipeline.tick();
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
    for (int i = 0; i < 128 && requests.empty(); ++i) {
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
    for (int i = 0; i < 20000 && !pipeline.readyToCommit(); ++i) pipeline.tick();
    assert(pipeline.readyToCommit());
    assert(pipeline.stats().retire_retired == 2);
}

void testUnrelatedStageIsolation() {
    auto baseline = baseConfig(64);
    baseline.neuron.width = 4;
    CorePipeline first(baseline);
    first.start(0);
    first.sealIngress();
    for (int i = 0; i < 20000 && !first.readyToCommit(); ++i) first.tick();
    assert(first.readyToCommit());

    auto changed = baseline;
    changed.row_lookup.latency_cycles = 17;
    changed.synapse.latency_cycles = 23;
    changed.retire.latency_cycles = 11;
    CorePipeline second(changed);
    second.start(0);
    second.sealIngress();
    for (int i = 0; i < 20000 && !second.readyToCommit(); ++i) second.tick();
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

void testCubaLifOperatorPath() {
    auto config = baseConfig(2);
    config.neuron_operator = NeuronOperatorKind::CubaLif;
    config.cuba_lif.dt_seconds = 1.0f;
    config.cuba_lif.tau_syn_seconds = 1.0f;
    config.cuba_lif.tau_mem_seconds = 1.0f;
    config.cuba_lif.resistance = 1.0f;
    config.cuba_lif.threshold = 0.1f;
    config.cuba_lif.input_weight = 1.0f;
    config.cuba_lif.reset_mode = CubaLifNeuronOp::ResetMode::Subtract;
    CorePipeline pipeline(config);
    pipeline.start(0);
    assert(pipeline.submitSpike(SpikeInput{0, 0, 19}));
    std::vector<RowRequest> requests;
    for (int i = 0; i < 128 && requests.empty(); ++i) {
        pipeline.tick();
        requests = pipeline.takeRowRequests();
    }
    assert(requests.size() == 1);
    assert(pipeline.acceptSynapseResponse(
        SynapseResponse{0, 0, 19, 1, 0, 1.0f, true, 1}));
    pipeline.sealIngress();
    for (int i = 0; i < 8192 && !pipeline.readyToCommit(); ++i) pipeline.tick();
    assert(pipeline.readyToCommit());
    assert(pipeline.stats().neurons_fired == 1);
    assert(pipeline.cubaState()[1].synaptic_current == 1.0f);
    assert(std::fabs(pipeline.cubaState()[1].membrane - 0.5321205258369446f) < 1.0e-7f);
    bool rejected_lif_snapshot = false;
    try {
        (void)pipeline.state();
    } catch (const std::logic_error&) {
        rejected_lif_snapshot = true;
    }
    assert(rejected_lif_snapshot);
}

void testMixedLocalOperatorBinding() {
    auto config = baseConfig(4);
    config.neuron_bindings.resize(4);
    config.neuron_bindings[0].kind = NeuronOperatorKind::Padding;
    config.neuron_bindings[1].kind = NeuronOperatorKind::Lif;
    config.neuron_bindings[1].lif = config.lif;
    config.neuron_bindings[2].kind = NeuronOperatorKind::CubaLif;
    config.neuron_bindings[2].cuba_lif.dt_seconds = 1.0f;
    config.neuron_bindings[2].cuba_lif.tau_syn_seconds = 1.0f;
    config.neuron_bindings[2].cuba_lif.tau_mem_seconds = 1.0f;
    config.neuron_bindings[2].cuba_lif.resistance = 1.0f;
    config.neuron_bindings[2].cuba_lif.threshold = 0.1f;
    config.neuron_bindings[3].kind = NeuronOperatorKind::Padding;
    CorePipeline pipeline(config);
    pipeline.start(0);
    assert(pipeline.submitSpike(SpikeInput{0, 0, 23}));
    std::vector<RowRequest> requests;
    for (int i = 0; i < 128 && requests.empty(); ++i) {
        pipeline.tick();
        requests = pipeline.takeRowRequests();
    }
    assert(requests.size() == 1);
    assert(pipeline.acceptSynapseResponse(
        SynapseResponse{0, 0, 23, 2, 0, 1.0f, false, 0}));
    assert(pipeline.acceptRowDone(RowDone{0, 0, 23, 1}));
    pipeline.sealIngress();
    for (int i = 0; i < 8192 && !pipeline.readyToCommit(); ++i) pipeline.tick();
    assert(pipeline.readyToCommit());
    assert(pipeline.stats().neurons_evaluated == 4);
    assert(pipeline.stats().neurons_fired == 1);
    assert(pipeline.functionalHash() != 0);
    bool rejected_lif_snapshot = false;
    try {
        (void)pipeline.state();
    } catch (const std::logic_error&) {
        rejected_lif_snapshot = true;
    }
    assert(rejected_lif_snapshot);
    bool rejected_cuba_snapshot = false;
    try {
        (void)pipeline.cubaState();
    } catch (const std::logic_error&) {
        rejected_cuba_snapshot = true;
    }
    assert(rejected_cuba_snapshot);
}

} // namespace

int main() {
    testRetireOrder();
    testCubaLifOperatorPath();
    testMixedLocalOperatorBinding();
    testQueueBackpressure();
    testScheduleAdmissionReservations();
    testScheduleEarliestCycleAndStageWidth();
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
            for (int i = 0; i < 8192 && !pipeline.readyToCommit(); ++i) pipeline.tick();
            assert(pipeline.readyToCommit());
            pipeline.commit();
        }
    }

    std::cout << "v5 core pipeline: PASS\n";
    std::cout << "scan_cycles width1=" << width1 << " width2=" << width2 << " width4=" << width4 << "\n";
    return 0;
}
