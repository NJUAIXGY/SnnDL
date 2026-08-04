#include <sst/core/sst_config.h>

#include "TimestepCoordinator.h"

#include <algorithm>
#include <cinttypes>
#include <fstream>
#include <iomanip>
#include <sstream>

namespace SST { namespace SnnDL {

TimestepCoordinator::TimestepCoordinator(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id), out_("SnnDL.TimestepCoordinator", 0, 0, SST::Output::STDOUT) {
    total_pes_ = std::max<std::uint32_t>(1, params.find<std::uint32_t>("total_pes", 1));
    start_timestep_ = params.find<std::uint64_t>("start_timestep", 0);
    max_timesteps_ = std::max<std::uint64_t>(1, params.find<std::uint64_t>("max_timesteps", 1));
    output_json_ = params.find<std::string>("output_json", "");
    mesh_rows_ = std::max<std::uint32_t>(1, params.find<std::uint32_t>("mesh_rows", 1));
    mesh_cols_ = std::max<std::uint32_t>(1, params.find<std::uint32_t>("mesh_cols", 1));
    cores_per_pe_ = std::max<std::uint32_t>(1, params.find<std::uint32_t>("cores_per_pe", 1));
    neurons_per_core_ = std::max<std::uint32_t>(1, params.find<std::uint32_t>("neurons_per_core", 1));
    memory_scope_ = params.find<std::string>("memory_scope", "chip_shared");
    memory_backend_ = params.find<std::string>("memory_backend", "simple");
    local_storage_ = params.find<int>("local_storage", 0) != 0;
    noc_type_ = params.find<std::string>("noc_type", "sst_xy_mesh_2d");
    noc_link_bw_ = params.find<std::string>("noc_link_bw", "");
    noc_num_vns_ = std::max<std::uint32_t>(1, params.find<std::uint32_t>("noc_num_vns", 1));
    noc_router_latency_cycles_ = std::max<std::uint32_t>(
        1, params.find<std::uint32_t>("noc_router_latency_cycles", 1));
    noc_queue_capacity_ = std::max<std::uint64_t>(
        1, params.find<std::uint64_t>("noc_queue_capacity", 64));
    weight_image_bytes_ = params.find<std::uint64_t>("weight_image_bytes", 0);
    graph_digest_ = params.find<std::string>("graph_digest", "");
    route_digest_ = params.find<std::string>("route_digest", "");
    descriptor_digest_ = params.find<std::string>("descriptor_digest", "");
    out_.setVerboseLevel(params.find<int>("verbose", 0));

    steps_.resize(max_timesteps_);
    for (auto& step : steps_) step.pe.resize(total_pes_);
    pe_links_.reserve(total_pes_);
    for (std::uint32_t pe = 0; pe < total_pes_; ++pe) {
        auto* link = configureLink(
            "pe" + std::to_string(pe),
            new SST::Event::Handler2<TimestepCoordinator, &TimestepCoordinator::handleControl_>(this));
        if (!link) out_.fatal(CALL_INFO, -1, "TimestepCoordinator requires pe%u port\n", pe);
        pe_links_.push_back(link);
    }
    registerAsPrimaryComponent();
    primaryComponentDoNotEndSim();
}

TimestepCoordinator::~TimestepCoordinator() = default;
void TimestepCoordinator::init(unsigned int) {}
void TimestepCoordinator::setup() {}

void TimestepCoordinator::finish() {
    if (!finished_) out_.verbose(CALL_INFO, 1, 0, "[snndl-coordinator] simulation ended before completion\n");
}

void TimestepCoordinator::handleControl_(SST::Event* event) {
    auto* control = dynamic_cast<TimestepControlEvent*>(event);
    if (!control) {
        delete event;
        out_.fatal(CALL_INFO, -1, "TimestepCoordinator received an unexpected event\n");
    }
    if (control->operation == TimestepControlOp::BootReady) {
        ++boot_ready_;
        if (!start_sent_ && boot_ready_ == total_pes_) {
            start_sent_ = true;
            current_timestep_ = start_timestep_;
            sendToAll_(TimestepControlOp::Start, current_timestep_);
        }
        delete control;
        return;
    }

    const auto step = control->timestep;
    if (step < start_timestep_ || step >= start_timestep_ + max_timesteps_) {
        delete control;
        out_.fatal(CALL_INFO, -1, "TimestepCoordinator received out-of-range timestep\n");
    }
    auto& aggregate = steps_.at(step - start_timestep_);
    auto& pe = aggregate.pe.at(control->source_pe);
    switch (control->operation) {
    case TimestepControlOp::EgressClosed:
        if (pe.egress_closed) {
            delete control;
            out_.fatal(CALL_INFO, -1, "duplicate EgressClosed from pe%u\n", control->source_pe);
        }
        pe.egress_closed = true;
        pe.tx = control->logical_count;
        pe.physical = control->physical_count;
        pe.memory_requests = control->memory_count;
        pe.synapse_created = control->synapse_count;
        maybeSeal_(step);
        break;
    case TimestepControlOp::IngressProgress:
        aggregate.logical_rx += control->logical_count;
        maybeSeal_(step);
        break;
    case TimestepControlOp::CommitReady:
        if (pe.commit_ready) {
            delete control;
            out_.fatal(CALL_INFO, -1, "duplicate CommitReady from pe%u\n", control->source_pe);
        }
        pe.commit_ready = true;
        maybeCommit_(step);
        break;
    case TimestepControlOp::CommitDone:
        if (pe.commit_done) {
            delete control;
            out_.fatal(CALL_INFO, -1, "duplicate CommitDone from pe%u\n", control->source_pe);
        }
        pe.commit_done = true;
        pe.tx = control->logical_count;
        pe.physical = control->physical_count;
        pe.memory_requests = control->memory_count;
        pe.memory_responses = control->memory_response_count;
        pe.storage_hits = control->storage_hits;
        pe.synapse_created = control->synapse_count;
        pe.synapse_retired = control->retired_count;
        pe.fired = control->fired_count;
        pe.neurons = control->neuron_count;
        pe.cycles = control->cycle_count;
        pe.state_hash = control->state_hash;
        pe.spike_hash = control->spike_hash;
        pe.queue_drops = control->queue_drops;
        pe.backpressure_events = control->backpressure_events;
        pe.stale_events = control->stale_events;
        pe.future_events = control->future_events;
        pe.post_seal_events = control->post_seal_events;
        pe.tracked_tokens = control->tracked_tokens;
        pe.queue_depth = control->queue_depth;
        pe.blocked_routes = control->blocked_routes;
        completeStep_(step);
        break;
    case TimestepControlOp::Abort:
        delete control;
        out_.fatal(CALL_INFO, -1, "TimestepCoordinator aborted\n");
        break;
    default:
        break;
    }
    delete control;
}

void TimestepCoordinator::sendToAll_(TimestepControlOp operation, std::uint64_t timestep) {
    for (auto* link : pe_links_) {
        auto* control = new TimestepControlEvent();
        control->operation = operation;
        control->timestep = timestep;
        link->send(control);
    }
}

void TimestepCoordinator::maybeSeal_(std::uint64_t timestep) {
    auto& aggregate = steps_.at(timestep - start_timestep_);
    if (aggregate.sealed) return;
    if (!std::all_of(aggregate.pe.begin(), aggregate.pe.end(),
                     [](const PeStep& pe) { return pe.egress_closed; })) {
        return;
    }
    std::uint64_t tx = 0;
    for (const auto& pe : aggregate.pe) tx += pe.tx;
    if (tx != aggregate.logical_rx) return;
    aggregate.sealed = true;
    sendToAll_(TimestepControlOp::SealIngress, timestep);
}

void TimestepCoordinator::maybeCommit_(std::uint64_t timestep) {
    auto& aggregate = steps_.at(timestep - start_timestep_);
    if (!aggregate.sealed || aggregate.commit_sent) return;
    if (!std::all_of(aggregate.pe.begin(), aggregate.pe.end(),
                     [](const PeStep& pe) { return pe.commit_ready; })) {
        return;
    }
    aggregate.commit_sent = true;
    sendToAll_(TimestepControlOp::Commit, timestep);
}

void TimestepCoordinator::completeStep_(std::uint64_t timestep) {
    auto& aggregate = steps_.at(timestep - start_timestep_);
    if (!std::all_of(aggregate.pe.begin(), aggregate.pe.end(),
                     [](const PeStep& pe) { return pe.commit_done; })) {
        return;
    }
    if (timestep + 1 < start_timestep_ + max_timesteps_) {
        current_timestep_ = timestep + 1;
        sendToAll_(TimestepControlOp::Start, current_timestep_);
        return;
    }
    writeSummary_();
    finished_ = true;
    primaryComponentOKToEndSim();
}

std::uint64_t TimestepCoordinator::hashMix_(std::uint64_t hash, std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    return hash;
}

void TimestepCoordinator::writeSummary_() const {
    if (output_json_.empty()) return;
    std::ofstream out(output_json_, std::ios::out | std::ios::trunc);
    if (!out.good()) return;
    std::uint64_t total_tx = 0, total_rx = 0, total_physical = 0;
    std::uint64_t total_mem_req = 0, total_mem_resp = 0, total_storage_hits = 0;
    std::uint64_t total_created = 0, total_retired = 0, total_fired = 0, total_cycles = 0;
    std::uint64_t total_queue_drops = 0, total_backpressure = 0;
    std::uint64_t total_stale = 0, total_future = 0, total_post_seal = 0;
    bool invariants = true;
    out << "{\n  \"schema_version\": 4,\n"
        << "  \"model\": \"snn_mesh_2d\",\n"
        << "  \"backend\": \"sst\",\n"
        << "  \"topology\": {\n"
        << "    \"mesh_rows\": " << mesh_rows_ << ",\n"
        << "    \"mesh_cols\": " << mesh_cols_ << ",\n"
        << "    \"total_pes\": " << total_pes_ << ",\n"
        << "    \"cores_per_pe\": " << cores_per_pe_ << ",\n"
        << "    \"neurons_per_core\": " << neurons_per_core_ << ",\n"
        << "    \"private_l1_per_core\": true,\n"
        << "    \"shared_l2_per_pe\": true,\n"
        << "    \"chip_shared_dram\": true\n"
        << "  },\n"
        << "  \"memory\": {\"scope\": \"" << memory_scope_
        << "\", \"backend\": \"" << memory_backend_
        << "\", \"local_storage\": " << (local_storage_ ? "true" : "false")
        << ", \"weight_image_bytes\": " << weight_image_bytes_ << "},\n"
        << "  \"noc\": {\"type\": \"" << noc_type_
        << "\", \"link_bw\": \"" << noc_link_bw_
        << "\", \"num_vns\": " << noc_num_vns_
        << ", \"router_latency_cycles\": " << noc_router_latency_cycles_
        << ", \"queue_capacity\": " << noc_queue_capacity_ << "},\n"
        << "  \"descriptor\": {\n"
        << "    \"graph_digest\": \"" << graph_digest_ << "\",\n"
        << "    \"route_digest\": \"" << route_digest_ << "\",\n"
        << "    \"descriptor_digest\": \"" << descriptor_digest_ << "\"\n"
        << "  },\n"
        << "  \"diagnostics\": {\n"
        << "    \"queue_drops\": ";
    for (const auto& step : steps_) {
        for (const auto& pe : step.pe) {
            total_queue_drops += pe.queue_drops;
            total_backpressure += pe.backpressure_events;
            total_stale += pe.stale_events;
            total_future += pe.future_events;
            total_post_seal += pe.post_seal_events;
        }
    }
    out << total_queue_drops << ",\n"
        << "    \"backpressure_events\": " << total_backpressure << ",\n"
        << "    \"stale_events\": " << total_stale << ",\n"
        << "    \"future_events\": " << total_future << ",\n"
        << "    \"post_seal_events\": " << total_post_seal << "\n"
        << "  },\n"
        << "  \"execution\": {\n"
        << "    \"started\": " << max_timesteps_ << ",\n"
        << "    \"committed\": " << max_timesteps_ << ",\n"
        << "    \"all_invariants_pass\": ";
    for (const auto& step : steps_) {
        std::uint64_t tx = 0, physical = 0, mem_req = 0, mem_resp = 0, storage_hits = 0;
        std::uint64_t created = 0, retired = 0;
        for (const auto& pe : step.pe) {
            tx += pe.tx;
            physical += pe.physical;
            mem_req += pe.memory_requests;
            mem_resp += pe.memory_responses;
            storage_hits += pe.storage_hits;
            created += pe.synapse_created;
            retired += pe.synapse_retired;
        }
        invariants = invariants && (tx == step.logical_rx) && (mem_req == mem_resp) &&
                     (created == retired);
    }
    invariants = invariants && total_queue_drops == 0 && total_stale == 0 &&
                 total_future == 0 && total_post_seal == 0;
    out << (invariants ? "true" : "false") << "\n  },\n  \"step_breakdown\": {\n";
    for (std::size_t index = 0; index < steps_.size(); ++index) {
        const auto& step = steps_[index];
        std::uint64_t tx = 0, physical = 0, mem_req = 0, mem_resp = 0, storage_hits = 0;
        std::uint64_t created = 0, retired = 0, fired = 0, neurons = 0, cycles = 0;
        std::uint64_t tracked_tokens = 0, queue_depth = 0, blocked_routes = 0;
        std::uint64_t state_hash = 0, spike_hash = 0;
        for (std::uint32_t pe_id = 0; pe_id < total_pes_; ++pe_id) {
            const auto& pe = step.pe[pe_id];
            tx += pe.tx; physical += pe.physical;
            mem_req += pe.memory_requests; mem_resp += pe.memory_responses;
            storage_hits += pe.storage_hits;
            created += pe.synapse_created; retired += pe.synapse_retired;
            fired += pe.fired; neurons += pe.neurons; cycles = std::max(cycles, pe.cycles);
            tracked_tokens += pe.tracked_tokens;
            queue_depth += pe.queue_depth;
            blocked_routes += pe.blocked_routes;
            state_hash = hashMix_(state_hash, hashMix_(pe_id, pe.state_hash));
            spike_hash = hashMix_(spike_hash, hashMix_(pe_id, pe.spike_hash));
        }
        total_tx += tx; total_rx += step.logical_rx; total_physical += physical;
        total_mem_req += mem_req; total_mem_resp += mem_resp; total_storage_hits += storage_hits;
        total_created += created; total_retired += retired; total_fired += fired; total_cycles += cycles;
        if (index != 0) out << ",\n";
        out << "    \"timestep_" << (start_timestep_ + index) << "\": {\"logical_tx\": " << tx
            << ", \"logical_rx\": " << step.logical_rx
            << ", \"physical_packets\": " << physical
            << ", \"memory_requests\": " << mem_req
            << ", \"memory_responses\": " << mem_resp
            << ", \"storage_hits\": " << storage_hits
            << ", \"synapse_tasks_created\": " << created
            << ", \"synapse_tasks_retired\": " << retired
            << ", \"fired\": " << fired
            << ", \"neurons_committed\": " << neurons
            << ", \"cycles\": " << cycles
            << ", \"tracked_tokens\": " << tracked_tokens
            << ", \"queue_depth\": " << queue_depth
            << ", \"blocked_routes\": " << blocked_routes
            << ", \"state_hash\": \"" << std::hex << state_hash << std::dec
            << "\", \"spike_hash\": \"" << std::hex << spike_hash << std::dec << "\"}";
    }
    out << "\n  },\n  \"per_timestep\": [\n";
    for (std::size_t index = 0; index < steps_.size(); ++index) {
        const auto& step = steps_[index];
        std::uint64_t tx = 0, physical = 0, mem_req = 0, mem_resp = 0, storage_hits = 0;
        std::uint64_t created = 0, retired = 0, fired = 0, neurons = 0, cycles = 0;
        std::uint64_t tracked_tokens = 0, queue_depth = 0, blocked_routes = 0;
        std::uint64_t step_backpressure = 0;
        for (const auto& pe : step.pe) {
            tx += pe.tx; physical += pe.physical; mem_req += pe.memory_requests;
            mem_resp += pe.memory_responses; created += pe.synapse_created;
            storage_hits += pe.storage_hits;
            retired += pe.synapse_retired; fired += pe.fired; neurons += pe.neurons;
            cycles = std::max(cycles, pe.cycles);
            tracked_tokens += pe.tracked_tokens;
            queue_depth += pe.queue_depth;
            blocked_routes += pe.blocked_routes;
            step_backpressure += pe.backpressure_events;
        }
        if (index != 0) out << ",\n";
        out << "    {\"timestep\": " << (start_timestep_ + index)
            << ", \"logical_tx\": " << tx << ", \"logical_rx\": " << step.logical_rx
            << ", \"physical_packets\": " << physical << ", \"memory_requests\": " << mem_req
            << ", \"memory_responses\": " << mem_resp
            << ", \"storage_hits\": " << storage_hits
            << ", \"synapse_tasks_created\": " << created
            << ", \"synapse_tasks_retired\": " << retired << ", \"fired\": " << fired
            << ", \"neurons_committed\": " << neurons << ", \"cycles\": " << cycles
            << ", \"tracked_tokens\": " << tracked_tokens
            << ", \"queue_depth\": " << queue_depth
            << ", \"blocked_routes\": " << blocked_routes
            << ", \"backpressure_events\": " << step_backpressure << "}";
    }
    out << "\n  ],\n  \"totals\": {\"logical_tx\": " << total_tx
        << ", \"logical_rx\": " << total_rx << ", \"physical_packets\": " << total_physical
        << ", \"memory_requests\": " << total_mem_req << ", \"memory_responses\": " << total_mem_resp
        << ", \"storage_hits\": " << total_storage_hits
        << ", \"synapse_tasks_created\": " << total_created
        << ", \"synapse_tasks_retired\": " << total_retired
        << ", \"fired\": " << total_fired << ", \"cycles\": " << total_cycles << "}\n}\n";
}

}} // namespace SST::SnnDL
