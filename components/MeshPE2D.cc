#include <sst/core/sst_config.h>

#include "MeshPE2D.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace SST { namespace SnnDL {

// Keep the aggregate SnnDL plugin's platform dependency reachable.  The
// symbol belongs to the active 2D platform, not the archived event-memory
// component.
extern "C" void snndl_platform2d_anchor() {}

namespace {

std::uint64_t parseUnsigned_(const std::string& value, const char* name) {
    try {
        std::size_t consumed = 0;
        const auto parsed = std::stoull(value, &consumed, 10);
        if (consumed != value.size()) throw std::invalid_argument("trailing characters");
        return parsed;
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string("invalid ") + name + " value: " + value);
    }
}

} // namespace

MeshPE2D::MeshPE2D(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id), out_("SnnDL.MeshPE2D", 0, 0, SST::Output::STDOUT) {
    pe_id_ = params.find<std::uint32_t>("pe_id", 0);
    weight_region_.owner_id = pe_id_;
    rows_ = std::max<std::uint32_t>(1, params.find<std::uint32_t>("rows", 1));
    cols_ = std::max<std::uint32_t>(1, params.find<std::uint32_t>("cols", 1));
    cores_per_pe_ = std::max<std::uint32_t>(1, params.find<std::uint32_t>("cores_per_pe", 1));
    neurons_per_core_ = std::max<std::uint32_t>(1, params.find<std::uint32_t>("neurons_per_core", 1));
    start_timestep_ = params.find<std::uint64_t>("start_timestep", 0);
    max_timesteps_ = std::max<std::uint32_t>(1, params.find<std::uint32_t>("max_timesteps", 1));
    neurons_per_pe_ = cores_per_pe_ * neurons_per_core_;
    memory_bytes_ = std::max<std::uint32_t>(1, params.find<std::uint32_t>("memory_bytes", 4));
    memory_latency_cycles_ = std::max<std::uint32_t>(
        1, params.find<std::uint32_t>("memory_latency_cycles", 10));
    local_storage_ = params.find<int>("local_storage", 0) != 0;
    multicast_ = params.find<int>("multicast", 0) != 0;
    use_standard_memory_ = params.find<int>("use_standard_memory", 1) != 0;
    descriptor_file_ = params.find<std::string>("descriptor_file", "");
    weight_image_file_ = params.find<std::string>("weight_image_file", "");
    descriptor_digest_ = params.find<std::string>("descriptor_digest", "");
    noc_link_bw_ = params.find<std::string>("noc_link_bw", "");
    noc_num_vns_ = std::max<std::uint32_t>(1, params.find<std::uint32_t>("noc_num_vns", 1));
    noc_router_latency_cycles_ = std::max<std::uint32_t>(
        1, params.find<std::uint32_t>("noc_router_latency_cycles", 1));
    noc_queue_capacity_ = std::max<std::size_t>(
        1, params.find<std::size_t>("noc_queue_capacity", 64));
    noc_queues_.resize(4u * noc_num_vns_);

    if (!use_standard_memory_) {
        out_.fatal(CALL_INFO, -1,
                   "MeshPE2D v4 requires memHierarchy StandardMem; event-memory fallback is archived\n");
    }

    out_.setVerboseLevel(params.find<int>("verbose", 0));

    control_link_ = configureLink(
        "control",
        new SST::Event::Handler2<MeshPE2D, &MeshPE2D::handleControl_>(this));
    memory_link_ = configureLink(
        "memory",
        new SST::Event::Handler2<MeshPE2D, &MeshPE2D::handleMemory_>(this));
    north_link_ = configureLink(
        "north", new SST::Event::Handler2<MeshPE2D, &MeshPE2D::handleSpike_>(this));
    south_link_ = configureLink(
        "south", new SST::Event::Handler2<MeshPE2D, &MeshPE2D::handleSpike_>(this));
    east_link_ = configureLink(
        "east", new SST::Event::Handler2<MeshPE2D, &MeshPE2D::handleSpike_>(this));
    west_link_ = configureLink(
        "west", new SST::Event::Handler2<MeshPE2D, &MeshPE2D::handleSpike_>(this));
    if (!control_link_ || !memory_link_) {
        if (!control_link_) {
            out_.fatal(CALL_INFO, -1, "MeshPE2D requires a control link\n");
        }
        if (!use_standard_memory_ && !memory_link_) {
            out_.fatal(CALL_INFO, -1, "MeshPE2D requires a memory link in event-memory mode\n");
        }
    }

    cores_.reserve(cores_per_pe_);
    const auto dt_ms = params.find<float>("dt_ms", 1.0f);
    const auto tau_ms = params.find<float>("tau_mem_ms", 20.0f);
    const auto threshold = params.find<float>("threshold", 1.0f);
    const auto reset = params.find<float>("reset", 0.0f);
    const auto refractory = params.find<std::uint32_t>("refractory_timesteps", 0);
    for (std::uint32_t core = 0; core < cores_per_pe_; ++core) {
        cores_.emplace_back(pe_id_, static_cast<std::uint16_t>(core), neurons_per_core_,
                            dt_ms, tau_ms, threshold, reset, refractory);
    }

    if (use_standard_memory_ && !local_storage_) {
        core_memories_.reserve(cores_per_pe_);
        for (std::uint32_t core = 0; core < cores_per_pe_; ++core) {
            const auto slot = "core_memory" + std::to_string(core);
            auto* memory = loadUserSubComponent<SST::Interfaces::StandardMem>(
                slot, ComponentInfo::SHARE_NONE, registerTimeBase("1ns"),
                new SST::Interfaces::StandardMem::Handler2<
                    MeshPE2D, &MeshPE2D::handleStandardMemory_>(this));
            if (!memory) {
                out_.fatal(CALL_INFO, -1,
                           "MeshPE2D requires StandardMem subcomponent '%s'\n", slot.c_str());
            }
            core_memories_.push_back(memory);
        }
    }

    try {
        if (!descriptor_file_.empty()) {
            parseDescriptor_(descriptor_file_);
        } else {
            // Kept only for old hand-written SDLs.  The schema-v4 runner
            // always supplies one descriptor for routing and weight ordinals.
            parseEdges_(params.find<std::string>("edges", ""));
        }
        if (local_storage_) loadLocalWeightStore_();
        parseStimuli_(params.find<std::string>("stimuli", ""));
    } catch (const std::exception& error) {
        out_.fatal(CALL_INFO, -1, "MeshPE2D graph parse failed: %s\n", error.what());
    }

    const auto clock = params.find<std::string>("clock", "1GHz");
    registerClock(clock, new SST::Clock::Handler2<MeshPE2D, &MeshPE2D::clockTick_>(this));
}

MeshPE2D::~MeshPE2D() = default;

void MeshPE2D::init(unsigned int phase) {
    for (auto* memory : core_memories_) memory->init(phase);
}

void MeshPE2D::setup() {
    for (auto* memory : core_memories_) memory->setup();
}

void MeshPE2D::finish() {
    for (auto* memory : core_memories_) memory->finish();
    out_.verbose(CALL_INFO, 1, 0,
                 "[snndl-pe] pe=%u active=%d tx=%" PRIu64 " rx=%" PRIu64
                 " mem=%" PRIu64 "/%" PRIu64 " tasks=%" PRIu64 "/%" PRIu64 "\n",
                 pe_id_, active_ ? 1 : 0, counters_.logical_tx, counters_.logical_rx,
                 counters_.memory_requests, counters_.memory_responses,
                 counters_.synapse_tasks_created, counters_.synapse_tasks_retired);
}

bool MeshPE2D::clockTick_(SST::Cycle_t cycle) {
    if (!boot_sent_) {
        boot_sent_ = true;
        sendControl_(TimestepControlOp::BootReady);
    }
    serviceNoc_(cycle);
    if (active_) processIngress_();
    if (active_ && local_storage_) serviceLocalStorage_(cycle);
    maybeCommitReady_();
    return false;
}

void MeshPE2D::handleControl_(SST::Event* event) {
    auto* control = dynamic_cast<TimestepControlEvent*>(event);
    if (!control) {
        delete event;
        out_.fatal(CALL_INFO, -1, "MeshPE2D received an unexpected control event\n");
    }
    const auto operation = control->operation;
    const auto step = control->timestep;
    if (operation == TimestepControlOp::Start) {
        if (active_) {
            delete control;
            out_.fatal(CALL_INFO, -1, "MeshPE2D received Start while another timestep is active\n");
        }
        startTimestep_(step, static_cast<SST::Cycle_t>(getCurrentSimTimeNano()));
    } else if (operation == TimestepControlOp::SealIngress) {
        if (!active_ || step != active_timestep_) {
            delete control;
            out_.fatal(CALL_INFO, -1, "MeshPE2D received SealIngress for the wrong timestep\n");
        }
        sealed_ = true;
        try {
            tracker_.sealIngress(step);
        } catch (const std::logic_error& error) {
            delete control;
            out_.fatal(CALL_INFO, -1, "MeshPE2D tracker seal failed: %s\n", error.what());
        }
        maybeCommitReady_();
    } else if (operation == TimestepControlOp::Commit) {
        if (!active_ || step != active_timestep_ || !commit_ready_sent_) {
            delete control;
            out_.fatal(CALL_INFO, -1, "MeshPE2D received Commit before local drain\n");
        }
        commitTimestep_(step, static_cast<SST::Cycle_t>(getCurrentSimTimeNano()));
    } else if (operation == TimestepControlOp::Abort) {
        delete control;
        out_.fatal(CALL_INFO, -1, "MeshPE2D aborted by timestep coordinator\n");
    }
    delete control;
}

void MeshPE2D::handleSpike_(SST::Event* event) {
    auto* spike = dynamic_cast<MeshSpikeEvent*>(event);
    if (!spike) {
        delete event;
        out_.fatal(CALL_INFO, -1, "MeshPE2D received an unexpected NoC event\n");
    }
    if (sealed_ && spike->timestep == active_timestep_) {
        ++counters_.post_seal_events;
    }
    if (!active_ || spike->timestep != active_timestep_) {
        if (spike->timestep < active_timestep_) ++counters_.stale_events;
        if (spike->timestep > active_timestep_) ++counters_.future_events;
        const auto step = spike->timestep;
        delete spike;
        out_.fatal(CALL_INFO, -1,
                   "MeshPE2D received a stale/future spike (active=%" PRIu64 " packet=%" PRIu64 ")\n",
                   active_timestep_, step);
    }
    if (spike->destination_pe == pe_id_) {
        try {
            tracker_.acquire(active_timestep_, WorkKind::Ingress);
        } catch (const std::logic_error& error) {
            delete spike;
            out_.fatal(CALL_INFO, -1, "MeshPE2D ingress token failed: %s\n", error.what());
        }
        ingress_queue_.push_back(*spike);
    } else {
        routeSpike_(*spike);
    }
    delete spike;
}

void MeshPE2D::serviceLocalStorage_(SST::Cycle_t cycle) {
    while (!local_ready_tasks_.empty() && local_ready_tasks_.front().ready_cycle <= cycle) {
        const auto request_id = local_ready_tasks_.front().request_id;
        local_ready_tasks_.pop_front();
        const auto it = pending_tasks_.find(request_id);
        if (it == pending_tasks_.end()) {
            out_.fatal(CALL_INFO, -1,
                       "MeshPE2D local-storage completion has unknown request id=%" PRIu64 "\n",
                       request_id);
        }
        const auto value_it = local_weight_store_.find(it->second.address);
        if (value_it == local_weight_store_.end()) {
            out_.fatal(CALL_INFO, -1,
                       "MeshPE2D local-storage has no BCSR value at address=%" PRIu64 "\n",
                       it->second.address);
        }
        if (std::memcmp(&value_it->second, &it->second.weight, sizeof(float)) != 0) {
            out_.fatal(CALL_INFO, -1,
                       "MeshPE2D local-storage value mismatch at address=%" PRIu64 "\n",
                       it->second.address);
        }
        retireTask_(request_id, value_it->second);
    }
}

void MeshPE2D::retireTask_(std::uint64_t request_id, float value) {
    const auto it = pending_tasks_.find(request_id);
    if (it == pending_tasks_.end()) {
        out_.fatal(CALL_INFO, -1,
                   "MeshPE2D received an unknown weight response id=%" PRIu64 "\n",
                   request_id);
    }
    const Task task = it->second;
    if (!active_ || task.timestep != active_timestep_) {
        if (task.timestep < active_timestep_) ++counters_.stale_events;
        if (task.timestep > active_timestep_) ++counters_.future_events;
        out_.fatal(CALL_INFO, -1,
                   "MeshPE2D received a stale/future weight response"
                   " (active=%" PRIu64 " task=%" PRIu64 ")\n",
                   active_timestep_, task.timestep);
    }
    pending_tasks_.erase(it);
    try {
        tracker_.release(task.timestep, WorkKind::RouteTask);
        tracker_.release(task.timestep, WorkKind::WeightRead);
    } catch (const std::logic_error& error) {
        out_.fatal(CALL_INFO, -1, "MeshPE2D weight token release failed: %s\n", error.what());
    }
    ++counters_.memory_responses;
    ++counters_.synapse_tasks_retired;
    if (local_storage_ && task.accepted) ++counters_.storage_hits;
    if (task.accepted) {
        const auto core = static_cast<std::uint32_t>(task.post_local / neurons_per_core_);
        const auto local = task.post_local % neurons_per_core_;
        cores_.at(core).addDelta(task.timestep, local, value, task.stable_order);
    }
    try {
        tracker_.release(task.timestep, WorkKind::RetireEntry);
        tracker_.release(task.timestep, WorkKind::AccumulatorUpdate);
    } catch (const std::logic_error& error) {
        out_.fatal(CALL_INFO, -1, "MeshPE2D retire token release failed: %s\n", error.what());
    }
    maybeCommitReady_();
}

void MeshPE2D::handleMemory_(SST::Event* event) {
    auto* response = dynamic_cast<MeshMemoryResponseEvent*>(event);
    if (!response) {
        delete event;
        out_.fatal(CALL_INFO, -1, "MeshPE2D received an unexpected memory event\n");
    }
    if (!active_ || response->timestep != active_timestep_) {
        if (response->timestep < active_timestep_) ++counters_.stale_events;
        if (response->timestep > active_timestep_) ++counters_.future_events;
        delete response;
        out_.fatal(CALL_INFO, -1, "MeshPE2D received a stale/future memory response\n");
    }
    retireTask_(response->request_id, response->value);
    delete response;
}

void MeshPE2D::handleStandardMemory_(SST::Interfaces::StandardMem::Request* request) {
    if (!request) return;
    auto* response = dynamic_cast<SST::Interfaces::StandardMem::ReadResp*>(request);
    if (!response) {
        delete request;
        out_.fatal(CALL_INFO, -1, "MeshPE2D received a non-read StandardMem response\n");
    }
    const auto request_id = static_cast<std::uint64_t>(request->getID());
    const auto it = pending_tasks_.find(request_id);
    if (it == pending_tasks_.end()) {
        delete request;
        out_.fatal(CALL_INFO, -1,
                   "MeshPE2D received an unknown StandardMem response id=%" PRIu64 "\n",
                   request_id);
    }
    const Task task = it->second;
    if (response->pAddr != task.address) {
        delete request;
        out_.fatal(CALL_INFO, -1,
                   "MeshPE2D received a StandardMem response at unexpected address"
                   " id=%" PRIu64 " address=%" PRIu64 " expected=%" PRIu64 "\n",
                   request_id, static_cast<std::uint64_t>(response->pAddr), task.address);
    }
    if (response->data.size() < sizeof(float)) {
        delete request;
        out_.fatal(CALL_INFO, -1,
                   "MeshPE2D received a short StandardMem response id=%" PRIu64
                   " bytes=%zu\n", request_id, response->data.size());
    }
    float weight = 0.0f;
    std::memcpy(&weight, response->data.data(), sizeof(weight));
    if (std::memcmp(&weight, &task.weight, sizeof(weight)) != 0) {
        delete request;
        out_.fatal(CALL_INFO, -1,
                   "MeshPE2D received a weight/image mismatch id=%" PRIu64
                   " address=%" PRIu64 " response=%g expected=%g\n",
                   request_id, task.address, static_cast<double>(weight),
                   static_cast<double>(task.weight));
    }
    retireTask_(request_id, weight);
    delete request;
}

void MeshPE2D::startTimestep_(std::uint64_t timestep, SST::Cycle_t cycle) {
    if (timestep < start_timestep_ || timestep >= start_timestep_ + max_timesteps_) {
        out_.fatal(CALL_INFO, -1, "MeshPE2D received timestep beyond max_timesteps\n");
    }
    active_ = true;
    sealed_ = false;
    egress_closed_ = false;
    commit_ready_sent_ = false;
    active_timestep_ = timestep;
    active_start_cycle_ = cycle;
    counters_ = StepCounters{};
    counters_.start_cycle = cycle;
    try {
        tracker_.open(timestep);
    } catch (const std::logic_error& error) {
        out_.fatal(CALL_INFO, -1, "MeshPE2D tracker open failed: %s\n", error.what());
    }
    for (auto& core : cores_) core.begin(timestep);

    std::vector<SpikeMessage> inputs;
    const auto held = held_spikes_.find(timestep);
    if (held != held_spikes_.end()) {
        inputs.insert(inputs.end(), held->second.begin(), held->second.end());
        try {
            tracker_.acquire(timestep, WorkKind::HoldQueue, held->second.size());
            tracker_.release(timestep, WorkKind::HoldQueue, held->second.size());
        } catch (const std::logic_error& error) {
            out_.fatal(CALL_INFO, -1, "MeshPE2D hold-queue token failed: %s\n", error.what());
        }
        held_spikes_.erase(held);
    }
    const auto external = stimuli_.find(timestep);
    if (external != stimuli_.end()) {
        for (const auto neuron : external->second) {
            inputs.push_back(SpikeMessage{timestep, neuron % neurons_per_pe_, pe_id_,
                                          coreForNeuron_(neuron), external_event_seq_++});
        }
    }
    std::stable_sort(inputs.begin(), inputs.end(), [](const SpikeMessage& lhs, const SpikeMessage& rhs) {
        if (lhs.source_neuron != rhs.source_neuron) return lhs.source_neuron < rhs.source_neuron;
        return lhs.source_event_seq < rhs.source_event_seq;
    });
    for (const auto& input : inputs) emitSourceSpike_(input);
    closeEgress_();
}

void MeshPE2D::closeEgress_() {
    if (egress_closed_) return;
    egress_closed_ = true;
    sendControl_(TimestepControlOp::EgressClosed);
}

void MeshPE2D::processIngress_() {
    while (!ingress_queue_.empty()) {
        MeshSpikeEvent spike = ingress_queue_.front();
        ingress_queue_.pop_front();
        counters_.logical_rx += spike.logical_deliveries;
        processSpikeAtDestination_(spike);
        try {
            tracker_.release(active_timestep_, WorkKind::Ingress);
        } catch (const std::logic_error& error) {
            out_.fatal(CALL_INFO, -1, "MeshPE2D ingress token release failed: %s\n", error.what());
        }

        TimestepControlEvent progress;
        progress.operation = TimestepControlOp::IngressProgress;
        progress.timestep = active_timestep_;
        progress.source_pe = pe_id_;
        progress.logical_count = spike.logical_deliveries;
        control_link_->send(new TimestepControlEvent(progress));
    }
}

void MeshPE2D::serviceNoc_(SST::Cycle_t cycle) {
    // First admit blocked packets whenever a finite virtual-network queue has
    // room. Packets remain owned by the upstream PE until they can enter a
    // router input queue; no packet is silently dropped.
    for (auto it = blocked_routes_.begin(); it != blocked_routes_.end();) {
        auto& queue = noc_queues_.at(it->direction * noc_num_vns_ + it->virtual_network);
        if (queue.size() >= noc_queue_capacity_) {
            ++it;
            continue;
        }
        queue.push_back(std::move(*it));
        it = blocked_routes_.erase(it);
    }

    // Each physical direction transfers at most one event per cycle. VNs are
    // selected round-robin, while router latency delays every hop explicitly.
    for (std::uint32_t direction = 0; direction < 4; ++direction) {
        const auto start = noc_rr_.at(direction) % noc_num_vns_;
        for (std::uint32_t attempt = 0; attempt < noc_num_vns_; ++attempt) {
            const auto vn = (start + attempt) % noc_num_vns_;
            auto& queue = noc_queues_.at(direction * noc_num_vns_ + vn);
            if (queue.empty() || queue.front().ready_cycle > cycle) continue;
            auto queued = std::move(queue.front());
            queue.pop_front();
            queued.link->send(new MeshSpikeEvent(queued.spike));
            ++counters_.physical_packets;
            noc_rr_.at(direction) = (vn + 1) % noc_num_vns_;
            break;
        }
    }
}

void MeshPE2D::processSpikeAtDestination_(const MeshSpikeEvent& spike) {
    const auto edges = incoming_edges_.find(spike.source_neuron);
    if (edges == incoming_edges_.end()) return;
    for (const auto& edge : edges->second) {
        const auto destination_core = coreForNeuron_(edge.post);
        if (spike.destination_core != MeshSpikeEvent::kAllCores &&
            destination_core != spike.destination_core) {
            continue;
        }
        issueTask_(spike.timestep, edge, spike.source_event_seq);
    }
}

void MeshPE2D::emitSourceSpike_(const SpikeMessage& input) {
    const auto global_source = input.source_node * neurons_per_pe_ + input.source_neuron;
    const auto edges = outgoing_edges_.find(global_source);
    if (edges == outgoing_edges_.end()) return;

    std::map<std::pair<std::uint32_t, std::uint16_t>, std::uint64_t> endpoints;
    for (const auto& edge : edges->second) {
        endpoints[{peForNeuron_(edge.post), coreForNeuron_(edge.post)}] += 1;
    }
    if (multicast_) {
        std::map<std::uint32_t, std::uint64_t> by_pe;
        for (const auto& endpoint : endpoints) by_pe[endpoint.first.first] += 1;
        for (const auto& entry : by_pe) {
            MeshSpikeEvent packet;
            packet.timestep = active_timestep_;
            packet.source_neuron = global_source;
            packet.source_pe = input.source_node;
            packet.source_core = input.source_core;
            packet.destination_pe = entry.first;
            packet.destination_core = MeshSpikeEvent::kAllCores;
            packet.source_event_seq = input.source_event_seq;
            packet.logical_deliveries = entry.second;
            counters_.logical_tx += entry.second;
            routeSpike_(packet);
        }
    } else {
        for (const auto& endpoint : endpoints) {
            MeshSpikeEvent packet;
            packet.timestep = active_timestep_;
            packet.source_neuron = global_source;
            packet.source_pe = input.source_node;
            packet.source_core = input.source_core;
            packet.destination_pe = endpoint.first.first;
            packet.destination_core = endpoint.first.second;
            packet.source_event_seq = input.source_event_seq;
            packet.logical_deliveries = 1;
            ++counters_.logical_tx;
            routeSpike_(packet);
        }
    }
}

void MeshPE2D::issueTask_(std::uint64_t timestep, const Edge& edge,
                          std::uint64_t source_event_seq) {
    const auto post_local = localNeuron_(edge.post);
    const auto core = static_cast<std::uint32_t>(post_local / neurons_per_core_);
    const auto local = post_local % neurons_per_core_;
    const bool accepted = cores_.at(core).accepts(timestep, local);
    const auto stable_order = hashMix_(hashMix_(source_event_seq, edge.pre), edge.post);
    ++counters_.synapse_tasks_created;
    ++counters_.memory_requests;

    try {
        // A task owns all of its lifetime tokens before ingress can be sealed.
        // Responses release the route/read/retire/accumulator tokens together,
        // so no token is acquired after the barrier has closed.
        tracker_.acquire(timestep, WorkKind::RouteTask);
        tracker_.acquire(timestep, WorkKind::WeightRead);
        tracker_.acquire(timestep, WorkKind::RetireEntry);
        tracker_.acquire(timestep, WorkKind::AccumulatorUpdate);
    } catch (const std::logic_error& error) {
        out_.fatal(CALL_INFO, -1, "MeshPE2D task token acquire failed: %s\n", error.what());
    }

    const auto address = weightAddress_(edge);
    const Task task{timestep, post_local, address, edge.weight, accepted, stable_order};
    if (local_storage_) {
        const auto request_id = next_event_request_id_++;
        if (!pending_tasks_.emplace(request_id, task).second) {
            out_.fatal(CALL_INFO, -1,
                       "MeshPE2D detected a duplicate local-storage request id=%" PRIu64 "\n",
                       request_id);
        }
        // The local store is a PE-owned copy of the canonical BCSR value image.
        // Keep a ready queue so the extension changes timing and accounting,
        // but never changes logical request/response semantics.
        local_ready_tasks_.push_back(
            LocalReadyTask{request_id,
                           static_cast<SST::Cycle_t>(getCurrentSimTimeNano()) +
                               memory_latency_cycles_});
    } else if (use_standard_memory_) {
        auto* request = new SST::Interfaces::StandardMem::Read(address, memory_bytes_);
        const auto standard_id = static_cast<std::uint64_t>(request->getID());
        if (!pending_tasks_.emplace(standard_id, task).second) {
            delete request;
            out_.fatal(CALL_INFO, -1,
                       "MeshPE2D detected a duplicate StandardMem request id=%" PRIu64 "\n",
                       standard_id);
        }
        core_memories_.at(core)->send(request);
    } else {
        const auto request_id = next_event_request_id_++;
        if (!pending_tasks_.emplace(request_id, task).second) {
            out_.fatal(CALL_INFO, -1,
                       "MeshPE2D detected a duplicate event-memory request id=%" PRIu64 "\n",
                       request_id);
        }
        auto* request = new MeshMemoryRequestEvent();
        request->request_id = request_id;
        request->timestep = timestep;
        request->source_pe = pe_id_;
        request->address = address;
        request->bytes = memory_bytes_;
        request->value = edge.weight;
        memory_link_->send(request);
    }
}

void MeshPE2D::routeSpike_(const MeshSpikeEvent& original) {
    MeshSpikeEvent spike = original;
    if (spike.destination_pe == pe_id_) {
        try {
            tracker_.acquire(active_timestep_, WorkKind::Ingress);
        } catch (const std::logic_error& error) {
            out_.fatal(CALL_INFO, -1, "MeshPE2D local-delivery token acquire failed: %s\n", error.what());
        }
        ingress_queue_.push_back(spike);
        return;
    }
    auto* link = nextHop_(spike.destination_pe, spike);
    if (!link) {
        out_.fatal(CALL_INFO, -1, "MeshPE2D has no route from pe%u to pe%u\n",
                   pe_id_, spike.destination_pe);
    }
    const auto direction = directionForLink_(link);
    const auto virtual_network = static_cast<std::uint32_t>(
        spike.source_event_seq % static_cast<std::uint64_t>(noc_num_vns_));
    QueuedSpike queued{link, spike,
                       static_cast<SST::Cycle_t>(getCurrentSimTimeNano() +
                                                 noc_router_latency_cycles_),
                       direction, virtual_network};
    auto& queue = noc_queues_.at(direction * noc_num_vns_ + virtual_network);
    if (queue.size() >= noc_queue_capacity_) {
        // Preserve the packet and model upstream backpressure.  A packet is
        // never dropped merely because a finite virtual-network queue fills.
        ++counters_.backpressure_events;
        blocked_routes_.push_back(std::move(queued));
    } else {
        queue.push_back(std::move(queued));
    }
}

SST::Link* MeshPE2D::nextHop_(std::uint32_t destination_pe, MeshSpikeEvent& spike) {
    if (destination_pe >= rows_ * cols_) {
        out_.fatal(CALL_INFO, -1, "MeshPE2D destination pe%u is outside %ux%u mesh\n",
                   destination_pe, rows_, cols_);
    }
    const auto source_row = pe_id_ / cols_;
    const auto source_col = pe_id_ % cols_;
    const auto destination_row = destination_pe / cols_;
    const auto destination_col = destination_pe % cols_;
    SST::Link* link = nullptr;
    // Deterministic dimension-order routing: columns first, then rows.
    if (source_col < destination_col) {
        link = east_link_;
    } else if (source_col > destination_col) {
        link = west_link_;
    } else if (source_row < destination_row) {
        link = south_link_;
    } else if (source_row > destination_row) {
        link = north_link_;
    }
    ++spike.hop_count;
    return link;
}

std::uint32_t MeshPE2D::directionForLink_(SST::Link* link) const {
    if (link == north_link_) return 0;
    if (link == south_link_) return 1;
    if (link == east_link_) return 2;
    if (link == west_link_) return 3;
    out_.fatal(CALL_INFO, -1, "MeshPE2D received an unknown NoC output link\n");
    return 0;
}

bool MeshPE2D::nocQueuesEmpty_() const {
    if (!blocked_routes_.empty()) return false;
    for (const auto& queue : noc_queues_) {
        if (!queue.empty()) return false;
    }
    return true;
}

void MeshPE2D::sendControl_(TimestepControlOp operation) {
    auto* control = new TimestepControlEvent();
    control->operation = operation;
    control->timestep = active_timestep_;
    control->source_pe = pe_id_;
    if (operation == TimestepControlOp::EgressClosed) {
        control->logical_count = counters_.logical_tx;
        control->physical_count = counters_.physical_packets;
        control->memory_count = counters_.memory_requests;
        control->memory_response_count = counters_.memory_responses;
        control->synapse_count = counters_.synapse_tasks_created;
    } else if (operation == TimestepControlOp::CommitDone) {
        control->logical_count = counters_.logical_tx;
        control->physical_count = counters_.physical_packets;
        control->memory_count = counters_.memory_requests;
        control->memory_response_count = counters_.memory_responses;
        control->storage_hits = counters_.storage_hits;
        control->synapse_count = counters_.synapse_tasks_created;
        control->retired_count = counters_.synapse_tasks_retired;
        control->fired_count = counters_.fired;
        control->neuron_count = counters_.neurons_committed;
        control->cycle_count = counters_.commit_cycle - counters_.start_cycle;
        control->state_hash = counters_.state_hash;
        control->spike_hash = counters_.spike_hash;
        control->queue_drops = counters_.queue_drops;
        control->backpressure_events = counters_.backpressure_events;
        control->stale_events = counters_.stale_events;
        control->future_events = counters_.future_events;
        control->post_seal_events = counters_.post_seal_events;
        control->tracked_tokens = counters_.tracked_tokens;
        control->queue_depth = counters_.queue_depth;
        control->blocked_routes = counters_.blocked_routes;
    }
    control_link_->send(control);
}

void MeshPE2D::maybeCommitReady_() {
    if (!active_ || !sealed_ || !egress_closed_ || commit_ready_sent_ ||
        !ingress_queue_.empty() || !pending_tasks_.empty() || !local_ready_tasks_.empty() ||
        !nocQueuesEmpty_()) {
        return;
    }
    if (counters_.synapse_tasks_created != counters_.synapse_tasks_retired ||
        counters_.memory_requests != counters_.memory_responses) {
        return;
    }
    try {
        if (!tracker_.locallyDrained(active_timestep_)) return;
    } catch (const std::logic_error& error) {
        out_.fatal(CALL_INFO, -1, "MeshPE2D tracker drain check failed: %s\n", error.what());
    }
    commit_ready_sent_ = true;
    sendControl_(TimestepControlOp::CommitReady);
}

void MeshPE2D::commitTimestep_(std::uint64_t timestep, SST::Cycle_t cycle) {
    std::vector<SpikeMessage> all_fired;
    for (auto& core : cores_) {
        auto fired = core.commit(timestep);
        counters_.neurons_committed += core.neurons();
        for (auto& spike : fired) {
            spike.source_neuron = core.coreId() * neurons_per_core_ + spike.source_neuron;
            spike.source_node = pe_id_;
            spike.source_core = core.coreId();
            spike.source_event_seq = external_event_seq_++;
            all_fired.push_back(spike);
        }
    }
    counters_.fired = all_fired.size();
    counters_.commit_cycle = cycle;
    counters_.state_hash = stateHash_();
    counters_.spike_hash = spikeHash_(all_fired);
    try {
        const auto snapshot = tracker_.snapshot(timestep);
        for (const auto value : snapshot.outstanding) counters_.tracked_tokens += value;
    } catch (const std::logic_error& error) {
        out_.fatal(CALL_INFO, -1, "MeshPE2D tracker snapshot failed: %s\n", error.what());
    }
    for (const auto& queue : noc_queues_) counters_.queue_depth += queue.size();
    counters_.blocked_routes = blocked_routes_.size();
    held_spikes_[timestep + 1] = all_fired;
    sendControl_(TimestepControlOp::CommitDone);
    active_ = false;
    sealed_ = false;
    egress_closed_ = false;
    commit_ready_sent_ = false;
}

std::uint32_t MeshPE2D::peForNeuron_(std::uint32_t neuron) const {
    return neuron / neurons_per_pe_;
}

std::uint16_t MeshPE2D::coreForNeuron_(std::uint32_t neuron) const {
    return static_cast<std::uint16_t>((neuron % neurons_per_pe_) / neurons_per_core_);
}

std::uint32_t MeshPE2D::localNeuron_(std::uint32_t neuron) const {
    return neuron % neurons_per_pe_;
}

std::vector<std::string> MeshPE2D::split_(const std::string& value, char separator) {
    std::vector<std::string> result;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, separator)) {
        if (!token.empty()) result.push_back(token);
    }
    return result;
}

void MeshPE2D::parseEdges_(const std::string& encoded) {
    const auto total_neurons = rows_ * cols_ * neurons_per_pe_;
    std::uint64_t edge_ordinal = 0;
    std::uint64_t max_ordinal = 0;
    bool have_edge = false;
    for (const auto& item : split_(encoded, ';')) {
        const auto fields = split_(item, ':');
        if (fields.size() != 3 && fields.size() != 4) {
            throw std::invalid_argument("edge must be pre:post:weight[:ordinal]");
        }
        const auto pre = static_cast<std::uint32_t>(parseUnsigned_(fields[0], "edge pre"));
        const auto post = static_cast<std::uint32_t>(parseUnsigned_(fields[1], "edge post"));
        if (pre >= total_neurons || post >= total_neurons) {
            throw std::out_of_range("edge neuron is outside the configured layout");
        }
        const auto weight = std::stof(fields[2]);
        const auto ordinal = fields.size() == 4
            ? parseUnsigned_(fields[3], "edge ordinal")
            : edge_ordinal;
        if (ordinal == UINT64_MAX) {
            throw std::out_of_range("edge ordinal exceeds addressable weight region");
        }
        const Edge edge{pre, post, weight, ordinal};
        if (peForNeuron_(pre) == pe_id_) outgoing_edges_[pre].push_back(edge);
        if (peForNeuron_(post) == pe_id_) incoming_edges_[pre].push_back(edge);
        max_ordinal = std::max(max_ordinal, ordinal);
        have_edge = true;
        ++edge_ordinal;
    }
    setWeightRegionSize_(have_edge ? max_ordinal + 1 : 0);
}

void MeshPE2D::parseDescriptor_(const std::string& path) {
    std::ifstream stream(path);
    if (!stream.good()) {
        throw std::invalid_argument("cannot open BCSR descriptor: " + path);
    }

    std::string line;
    if (!std::getline(stream, line) || line != "SNNDL_BCSR_V4") {
        throw std::invalid_argument("invalid BCSR descriptor magic");
    }
    std::string descriptor_digest;
    std::uint64_t edge_count = 0;
    bool have_count = false;
    std::vector<Edge> edges;
    while (std::getline(stream, line)) {
        if (line.empty()) continue;
        if (line.rfind("descriptor_digest=", 0) == 0) {
            descriptor_digest = line.substr(std::string("descriptor_digest=").size());
            continue;
        }
        if (line.rfind("edge_count=", 0) == 0) {
            edge_count = parseUnsigned_(
                line.substr(std::string("edge_count=").size()), "descriptor edge count");
            have_count = true;
            continue;
        }
        if (line.rfind("edge ", 0) != 0) {
            throw std::invalid_argument("invalid BCSR descriptor record");
        }
        std::istringstream fields(line.substr(5));
        std::uint64_t pre = 0, post = 0, ordinal = 0;
        float weight = 0.0f;
        std::string extra;
        if (!(fields >> pre >> post >> weight >> ordinal) || (fields >> extra)) {
            throw std::invalid_argument("invalid BCSR descriptor edge record");
        }
        if (!std::isfinite(weight)) throw std::invalid_argument("non-finite BCSR weight");
        edges.push_back(Edge{static_cast<std::uint32_t>(pre),
                             static_cast<std::uint32_t>(post), weight, ordinal});
    }
    if (!have_count || edge_count != edges.size()) {
        throw std::invalid_argument("BCSR descriptor edge_count does not match records");
    }
    if (!descriptor_digest_.empty() && descriptor_digest != descriptor_digest_) {
        throw std::invalid_argument("BCSR descriptor digest mismatch");
    }

    const auto total_neurons = rows_ * cols_ * neurons_per_pe_;
    std::set<std::uint64_t> ordinals;
    for (const auto& edge : edges) {
        if (edge.pre >= total_neurons || edge.post >= total_neurons) {
            throw std::out_of_range("BCSR descriptor neuron is outside the configured layout");
        }
        if (!ordinals.insert(edge.ordinal).second) {
            throw std::invalid_argument("BCSR descriptor contains duplicate ordinals");
        }
        if (peForNeuron_(edge.pre) == pe_id_) outgoing_edges_[edge.pre].push_back(edge);
        if (peForNeuron_(edge.post) == pe_id_) incoming_edges_[edge.pre].push_back(edge);
    }
    for (std::uint64_t ordinal = 0; ordinal < edge_count; ++ordinal) {
        if (!ordinals.count(ordinal)) {
            throw std::invalid_argument("BCSR descriptor ordinals must be contiguous");
        }
    }
    setWeightRegionSize_(edge_count);
}

void MeshPE2D::setWeightRegionSize_(std::uint64_t element_count) {
    if (memory_bytes_ == 0 || element_count > UINT64_MAX / memory_bytes_) {
        throw std::overflow_error("weight region size overflows byte address space");
    }
    weight_region_.size_bytes = element_count * memory_bytes_;
}

std::uint64_t MeshPE2D::weightAddress_(const Edge& edge) const {
    ::SnnDL::v5::TypedAddress address{};
    std::uint64_t physical = 0;
    if (!::SnnDL::v5::typedAddressForElement(
            weight_region_, edge.ordinal, memory_bytes_, address) ||
        !::SnnDL::v5::resolveRegionAddress(address, weight_region_, physical)) {
        throw std::out_of_range("edge ordinal is outside the PE weight region");
    }
    return physical;
}

void MeshPE2D::loadLocalWeightStore_() {
    std::ifstream image;
    if (!weight_image_file_.empty()) {
        if (memory_bytes_ != sizeof(float)) {
            throw std::invalid_argument(
                "local-storage v4 weight image requires memory_bytes=4");
        }
        image.open(weight_image_file_, std::ios::in | std::ios::binary);
        if (!image.good()) {
            throw std::invalid_argument(
                "cannot open local-storage BCSR value image: " + weight_image_file_);
        }
    }

    for (const auto& entry : incoming_edges_) {
        for (const auto& edge : entry.second) {
            const auto address = weightAddress_(edge);
            float value = edge.weight;
            if (image.is_open()) {
                image.clear();
                image.seekg(static_cast<std::streamoff>(address), std::ios::beg);
                image.read(reinterpret_cast<char*>(&value), sizeof(value));
                if (image.gcount() != static_cast<std::streamsize>(sizeof(value))) {
                    throw std::invalid_argument(
                        "local-storage BCSR value image is shorter than descriptor");
                }
                if (std::memcmp(&value, &edge.weight, sizeof(value)) != 0) {
                    throw std::invalid_argument(
                        "local-storage BCSR value image does not match descriptor");
                }
            }
            if (!local_weight_store_.emplace(address, value).second) {
                throw std::invalid_argument(
                    "local-storage BCSR value image contains a duplicate address");
            }
        }
    }
}

void MeshPE2D::parseStimuli_(const std::string& encoded) {
    const auto total_neurons = rows_ * cols_ * neurons_per_pe_;
    for (const auto& item : split_(encoded, ';')) {
        const auto fields = split_(item, ':');
        if (fields.size() != 2) throw std::invalid_argument("stimulus must be timestep:neuron");
        const auto timestep = parseUnsigned_(fields[0], "stimulus timestep");
        const auto neuron = static_cast<std::uint32_t>(parseUnsigned_(fields[1], "stimulus neuron"));
        if (timestep < start_timestep_ ||
            timestep >= start_timestep_ + max_timesteps_ || neuron >= total_neurons) {
            throw std::out_of_range("stimulus is outside the configured run");
        }
        if (peForNeuron_(neuron) == pe_id_) stimuli_[timestep].push_back(neuron);
    }
}

std::uint64_t MeshPE2D::hashMix_(std::uint64_t hash, std::uint64_t value) {
    hash ^= value + 0x9e3779b97f4a7c15ULL + (hash << 6) + (hash >> 2);
    return hash;
}

std::uint64_t MeshPE2D::stateHash_() const {
    std::uint64_t hash = 1469598103934665603ULL;
    for (const auto& core : cores_) {
        for (const auto& state : core.state()) {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &state.v_mem, sizeof(bits));
            hash = hashMix_(hash, bits);
            hash = hashMix_(hash, state.refractory);
        }
    }
    return hash;
}

std::uint64_t MeshPE2D::spikeHash_(const std::vector<SpikeMessage>& spikes) const {
    std::uint64_t hash = 1099511628211ULL;
    for (const auto& spike : spikes) {
        hash = hashMix_(hash, spike.timestep);
        // The core engine stores a PE-local neuron index.  The oracle and the
        // external event contract hash the globally-addressed neuron instead.
        const auto global_source = spike.source_node * neurons_per_pe_ + spike.source_neuron;
        hash = hashMix_(hash, global_source);
        hash = hashMix_(hash, spike.source_core);
        hash = hashMix_(hash, spike.source_event_seq);
    }
    return hash;
}

}} // namespace SST::SnnDL
