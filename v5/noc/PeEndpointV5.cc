#include <sst/core/sst_config.h>

#include "PeEndpointV5.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace SST { namespace SnnDL { namespace v5 {

namespace {
std::string portName(const char* base, std::uint32_t core, bool legacy) {
    return legacy ? std::string(base) : std::string(base) + std::to_string(core);
}
}

std::vector<std::uint32_t> PeEndpointV5::parseDestinations_(
    const std::string& encoded, std::uint32_t count, std::uint32_t fallback) {
    std::vector<std::uint32_t> values(count, fallback);
    if (encoded.empty()) return values;
    std::stringstream stream(encoded);
    std::string item;
    std::uint32_t index = 0;
    while (std::getline(stream, item, ',')) {
        if (index >= count || item.empty()) throw std::invalid_argument("invalid Core destination vector");
        values[index++] = static_cast<std::uint32_t>(std::stoul(item));
    }
    if (index != count) throw std::invalid_argument("Core destination vector length mismatch");
    return values;
}

void PeEndpointV5::parseRoutes_(const std::string& encoded) {
    if (encoded.empty()) return;
    std::stringstream records(encoded);
    std::string record;
    while (std::getline(records, record, ';')) {
        if (record.empty()) continue;
        std::vector<std::string> fields;
        std::stringstream field_stream(record);
        std::string field;
        while (std::getline(field_stream, field, '|')) fields.push_back(field);
        if (fields.size() != 5) throw std::invalid_argument("source route must have five fields");
        const auto core = static_cast<std::uint32_t>(std::stoul(fields[0]));
        const auto local = static_cast<std::uint32_t>(std::stoul(fields[1]));
        if (core >= cores_.size()) throw std::invalid_argument("source route Core is outside this PE");
        SourceRoute route;
        route.source_global = std::stoull(fields[2]);
        route.route_id = std::stoull(fields[3]);
        if (route.route_id == 0) throw std::invalid_argument("source route id must be nonzero");
        std::stringstream targets(fields[4]);
        std::string target;
        while (std::getline(targets, target, '&')) {
            const auto separator = target.find(':');
            if (separator == std::string::npos) throw std::invalid_argument("route target must be PE:CoreMask");
            RouteTarget parsed;
            parsed.pe = static_cast<std::uint32_t>(std::stoul(target.substr(0, separator)));
            parsed.core_mask = std::stoull(target.substr(separator + 1));
            if (parsed.core_mask == 0 || parsed.core_mask >> cores_per_pe_)
                throw std::invalid_argument("route target contains an invalid Core mask");
            route.targets.push_back(parsed);
        }
        if (route.targets.empty() || !cores_[core].source_routes.emplace(local, std::move(route)).second)
            throw std::invalid_argument("source route is empty or duplicates a local neuron");
    }
}

PeEndpointV5::PeEndpointV5(SST::ComponentId_t id, SST::Params& p)
    : Component(id), out_("SnnDL.PeEndpointV5", 0, 0, Output::STDOUT),
      pe_id_(p.find<std::uint32_t>("pe_id", 0)),
      mesh_x_(std::max(1u, p.find<std::uint32_t>("mesh_x", 1))),
      cores_per_pe_(std::max(1u, p.find<std::uint32_t>("cores_per_pe", 1))),
      tx_capacity_(std::max(1u, p.find<std::uint32_t>("tx_queue_entries", 16))),
      rx_capacity_(std::max(1u, p.find<std::uint32_t>("rx_queue_entries", 16))),
      control_capacity_(std::max(1u, p.find<std::uint32_t>("control_queue_entries", 32))),
      flit_bytes_(std::max(1u, p.find<std::uint32_t>("flit_size_bytes", 32))),
      payload_bytes_(p.find<std::uint32_t>("payload_bytes", 0)),
      core_held_capacity_(std::max(1u, p.find<std::uint32_t>("core_held_spike_entries", 32))),
      coordinator_pe_(p.find<std::uint32_t>("coordinator_pe", 0)),
      core_attached_(p.find<int>("core_attached", 0) != 0),
      timed_control_(p.find<int>("timed_control", 0) != 0),
      route_contract_v2_(p.find<int>("route_contract_v2", 0) != 0),
      external_stimulus_to_network_(p.find<int>("external_stimulus_to_network", 0) != 0),
      output_json_(p.find<std::string>("output_json", "")) {
    out_.setVerboseLevel(p.find<int>("verbose", 0));
    if (timed_control_ && flit_bytes_ != 32) out_.fatal(CALL_INFO, -1, "P5 wire contract requires flit_size_bytes=32\n");
    if (timed_control_ && !core_attached_) out_.fatal(CALL_INFO, -1, "timed control requires attached Cores\n");
    if (core_attached_ && tx_capacity_ < cores_per_pe_ * core_held_capacity_) {
        out_.fatal(CALL_INFO, -1, "endpoint TX capacity must cover all Core held-spike release bursts\n");
    }

    network_ = loadUserSubComponent<SST::Interfaces::SimpleNetwork>(
        "networkIF", ComponentInfo::SHARE_NONE, timed_control_ ? 2 : 1);
    if (!network_) out_.fatal(CALL_INFO, -1, "PeEndpointV5 requires networkIF=merlin.linkcontrol\n");
    probe_in_ = configureLink("probe_in", new Event::Handler2<PeEndpointV5, &PeEndpointV5::handleProbe_>(this));
    probe_out_ = configureLink("probe_out");
    epoch_command_ = configureLink("epoch_command", new Event::Handler2<PeEndpointV5, &PeEndpointV5::handleEpochCommand_>(this));
    epoch_status_ = configureLink("epoch_status");

    legacy_ports_ = cores_per_pe_ == 1 && isPortConnected("core_control");
    const auto default_pe = p.find<std::uint32_t>("destination_pe", 0);
    const auto default_core = p.find<std::uint32_t>("destination_core", 0);
    std::vector<std::uint32_t> destination_pes;
    std::vector<std::uint32_t> destination_cores;
    try {
        destination_pes = parseDestinations_(p.find<std::string>("core_destination_pes", ""), cores_per_pe_, default_pe);
        destination_cores = parseDestinations_(p.find<std::string>("core_destination_cores", ""), cores_per_pe_, default_core);
    } catch (const std::exception& error) {
        out_.fatal(CALL_INFO, -1, "PeEndpointV5 destination configuration error: %s\n", error.what());
    }

    cores_.resize(cores_per_pe_);
    core_epoch_enqueued_.resize(cores_per_pe_, std::vector<std::uint64_t>(1, 0));
    for (std::uint32_t core = 0; core < cores_per_pe_; ++core) {
        auto& port = cores_[core];
        port.destination_pe = destination_pes[core];
        port.destination_core = destination_cores[core];
        port.provider_control = configureLink(
            portName("provider_control", core, legacy_ports_),
            new Event::Handler2<PeEndpointV5, &PeEndpointV5::handleProviderControl_, int>(this, core));
        port.core_control = configureLink(portName("core_control", core, legacy_ports_));
        port.provider_spike = configureLink(
            portName("provider_spike", core, legacy_ports_),
            new Event::Handler2<PeEndpointV5, &PeEndpointV5::handleProviderSpike_, int>(this, core));
        port.provider_ack = configureLink(portName("provider_ack", core, legacy_ports_));
        port.core_spike = configureLink(portName("core_spike", core, legacy_ports_));
        port.core_ack = configureLink(
            portName("core_ack", core, legacy_ports_),
            new Event::Handler2<PeEndpointV5, &PeEndpointV5::handleCoreAck_, int>(this, core));
        port.core_egress = configureLink(
            portName("core_egress", core, legacy_ports_),
            new Event::Handler2<PeEndpointV5, &PeEndpointV5::handleCoreEgress_, int>(this, core));
        port.provider_monitor = configureLink(portName("provider_monitor", core, legacy_ports_));
        port.core_status = configureLink(
            portName("core_status", core, legacy_ports_),
            new Event::Handler2<PeEndpointV5, &PeEndpointV5::handleCoreStatus_, int>(this, core));
        port.provider_status = configureLink(portName("provider_status", core, legacy_ports_));
        if (core_attached_ && (!port.provider_control || !port.core_control || !port.provider_spike ||
                              !port.provider_ack || !port.core_spike || !port.core_ack || !port.core_egress)) {
            out_.fatal(CALL_INFO, -1, "core-attached endpoint is missing proxy links for Core %u\n", core);
        }
        if (timed_control_ && (!port.core_status || !port.provider_status)) {
            out_.fatal(CALL_INFO, -1, "timed endpoint is missing status links for Core %u\n", core);
        }
    }
    try {
        const auto mode = p.find<std::string>("multicast_mode", "source_replication");
        if (mode != "unicast" && mode != "source_replication" && mode != "native_tree")
            throw std::invalid_argument("unknown multicast_mode");
        native_tree_ = mode == "native_tree";
        parseRoutes_(p.find<std::string>("core_route_table", ""));
    } catch (const std::exception& error) {
        out_.fatal(CALL_INFO, -1, "PeEndpointV5 route configuration error: %s\n", error.what());
    }
    if (native_tree_) {
        native_credit_limit_ = std::max(1u, p.find<std::uint32_t>("native_link_credits", 8));
        native_credits_ = native_credit_limit_;
        native_link_ = configureLink("native_network",
            new Event::Handler2<PeEndpointV5, &PeEndpointV5::handleNative_>(this));
        if (!native_link_) out_.fatal(CALL_INFO, -1, "native_tree requires native_network link\n");
    }
    if (timed_control_ && pe_id_ == coordinator_pe_ && (!epoch_command_ || !epoch_status_)) {
        out_.fatal(CALL_INFO, -1, "coordinator endpoint requires epoch_command and epoch_status links\n");
    }

    registerClock(p.find<std::string>("clock", "1GHz"), new Clock::Handler2<PeEndpointV5, &PeEndpointV5::tick_>(this));
    for (const char* name : {"noc.packets", "noc.flits", "noc.logical_deliveries",
                             "noc.control_packets", "noc.control_flits", "noc.tx_stall_cycles",
                             "noc.rx_stall_cycles", "noc.core_retry_cycles"}) {
        registerStatistic<std::uint64_t>(name);
    }
}

PeEndpointV5::~PeEndpointV5() {
    for (auto* packet : data_tx_) delete packet;
    for (auto* packet : ingress_rx_) delete packet;
    for (auto* packet : control_tx_) delete packet;
    for (auto& core : cores_) {
        for (auto* packet : core.rx) delete packet;
        delete core.pending_seal;
    }
    delete network_;
}

void PeEndpointV5::init(unsigned int phase) { network_->init(phase); }
void PeEndpointV5::setup() {
    network_->setup();
    if (network_->getEndpointID() != pe_id_) out_.fatal(CALL_INFO, -1, "endpoint/linkcontrol nid mismatch\n");
}

void PeEndpointV5::enqueueData_(NocPacketV5Event* packet, SST::Link* ack_link) {
    auto* ack = new NocInjectionAckV5Event();
    ack->packet_id = packet->packet_id;
    if (packet->format_version != kNocPacketV5FormatVersion) {
        ack->accepted = false; ack->retryable = false; delete packet;
    } else if (data_tx_.size() >= tx_capacity_) {
        ack->accepted = false; ack->retryable = true; delete packet;
    } else {
        packet->source_pe = pe_id_;
        packet->injection_time_ns = getCurrentSimTimeNano();
        data_tx_.push_back(packet);
        ack->accepted = true;
    }
    if (ack_link) ack_link->send(ack); else delete ack;
}

void PeEndpointV5::enqueueControl_(NocControlV5Event* packet) {
    if (!timed_control_ || packet->format_version != kNocPacketV5FormatVersion) {
        delete packet;
        out_.fatal(CALL_INFO, -1, "invalid P5 control packet\n");
    }
    if (control_tx_.size() >= control_capacity_) {
        delete packet;
        out_.fatal(CALL_INFO, -1, "P5 control queue overflow\n");
    }
    packet->source_pe = pe_id_;
    packet->injection_time_ns = getCurrentSimTimeNano();
    control_tx_.push_back(packet);
}

void PeEndpointV5::handleProbe_(SST::Event* event) {
    auto* packet = dynamic_cast<NocPacketV5Event*>(event);
    if (!packet) { delete event; out_.fatal(CALL_INFO, -1, "bad probe event\n"); }
    enqueueData_(packet, probe_out_);
}

void PeEndpointV5::handleNative_(SST::Event* event) {
    if (auto* credit = dynamic_cast<NocCreditV5Event*>(event)) {
        if (credit->credits > native_credit_limit_ - std::min(native_credit_limit_, native_credits_)) {
            delete credit; out_.fatal(CALL_INFO, -1, "native endpoint credit overflow\n");
        }
        native_credits_ += credit->credits;
        delete credit;
        return;
    }
    auto* packet = dynamic_cast<NocPacketV5Event*>(event);
    if (!packet || packet->format_version != kNocPacketV5FormatVersion ||
        packet->destination_pe != pe_id_ || packet->destination_core_mask == 0) {
        delete event; out_.fatal(CALL_INFO, -1, "invalid native multicast ejection\n");
    }
    if (ingress_rx_.size() >= native_credit_limit_) {
        delete packet; out_.fatal(CALL_INFO, -1, "native endpoint receive buffer overflow\n");
    }
    const auto now = std::uint64_t(getCurrentSimTimeNano());
    latency_sum_ns_ += now >= packet->injection_time_ns ? now - packet->injection_time_ns : 0;
    ++rx_packets_;
    ingress_rx_.push_back(packet);
}

void PeEndpointV5::handleEpochCommand_(SST::Event* event) {
    auto* packet = dynamic_cast<NocControlV5Event*>(event);
    if (!packet || packet->kind != NocControlV5Kind::Command) {
        delete event; out_.fatal(CALL_INFO, -1, "bad epoch command\n");
    }
    enqueueControl_(packet);
}

void PeEndpointV5::sendStatus_(CoreControlOp operation, std::uint64_t epoch, std::uint32_t core, std::uint64_t count) {
    auto* packet = new NocControlV5Event();
    packet->kind = NocControlV5Kind::Status;
    packet->operation = operation;
    packet->epoch = epoch;
    packet->source_core = core;
    packet->destination_pe = coordinator_pe_;
    packet->destination_core = 0;
    packet->logical_count = count;
    enqueueControl_(packet);
}

void PeEndpointV5::handleProviderControl_(SST::Event* event, int core_index) {
    auto* control = dynamic_cast<CoreControlEvent*>(event);
    if (!control || core_index < 0 || static_cast<std::size_t>(core_index) >= cores_.size()) {
        delete event; out_.fatal(CALL_INFO, -1, "bad provider control\n");
    }
    auto& core = cores_[core_index];
    if (timed_control_ && (control->operation == CoreControlOp::PreloadReady ||
                           control->operation == CoreControlOp::IngressReady)) {
        const auto& epoch_enqueued = core_epoch_enqueued_[core_index];
        const auto count = control->operation == CoreControlOp::IngressReady &&
                                   control->timestep < epoch_enqueued.size()
                               ? epoch_enqueued[control->timestep]
                               : 0;
        sendStatus_(control->operation, control->timestep, core_index, count);
        delete control;
        return;
    }
    if (control->operation == CoreControlOp::Start) core.started = true;
    if (control->operation == CoreControlOp::SealIngress && !drainedForSeal_(core_index)) {
        delete core.pending_seal;
        core.pending_seal = control;
    } else {
        core.core_control->send(control);
    }
}

void PeEndpointV5::handleProviderSpike_(SST::Event* event, int core_index) {
    auto* spike = dynamic_cast<CoreSpikeEvent*>(event);
    if (!spike || core_index < 0 || static_cast<std::size_t>(core_index) >= cores_.size()) {
        delete event; out_.fatal(CALL_INFO, -1, "bad provider spike\n");
    }
    auto& core = cores_[core_index];
    if (external_stimulus_to_network_) {
        const auto timestep = spike->timestep;
        const auto source_neuron = spike->source_neuron;
        const auto source_event_seq = spike->source_event_seq;
        auto route = core.source_routes.end();
        for (auto candidate = core.source_routes.begin(); candidate != core.source_routes.end(); ++candidate) {
            if (candidate->second.source_global == source_neuron) {
                route = candidate;
                break;
            }
        }
        if (route == core.source_routes.end()) {
            ++logical_spikes_;
            ++zero_fanout_;
            ++external_zero_fanout_;
            delete spike;
        } else {
            // Core egress uses the local neuron id as the route-table key;
            // public StimulusIR carries the global source neuron id.
            spike->source_neuron = route->first;
            routeSourceSpike_(spike, core_index, false);
        }
        auto* ack = new CoreSpikeAckEvent();
        ack->timestep = timestep;
        ack->source_neuron = source_neuron;
        ack->source_event_seq = source_event_seq;
        ack->accepted = true;
        ack->retryable = false;
        core.provider_ack->send(ack);
        return;
    }
    core.ack_origins.push_back(AckOrigin::Provider);
    core.core_spike->send(spike);
}

void PeEndpointV5::handleCoreAck_(SST::Event* event, int core_index) {
    auto* ack = dynamic_cast<CoreSpikeAckEvent*>(event);
    if (!ack || core_index < 0 || static_cast<std::size_t>(core_index) >= cores_.size()) {
        delete event; out_.fatal(CALL_INFO, -1, "bad Core ACK\n");
    }
    auto& core = cores_[core_index];
    if (core.ack_origins.empty()) {
        delete ack; out_.fatal(CALL_INFO, -1, "Core ACK has no dispatch origin\n");
    }
    const auto origin = core.ack_origins.front();
    core.ack_origins.pop_front();
    if (origin == AckOrigin::Provider) {
        core.provider_ack->send(ack);
        return;
    }
    if (!core.network_inflight || core.rx.empty()) {
        delete ack; out_.fatal(CALL_INFO, -1, "network Core ACK has no packet\n");
    }
    if (ack->accepted) {
        const auto timestep = core.rx.front()->timestep;
        delete core.rx.front();
        core.rx.pop_front();
        ++logical_deliveries_;
        if (timed_control_) sendStatus_(CoreControlOp::IngressProgress, timestep, core_index, 1);
    } else if (ack->retryable) {
        ++core_retries_;
    } else {
        delete ack; out_.fatal(CALL_INFO, -1, "Core permanently rejected network packet\n");
    }
    core.network_inflight = false;
    delete ack;
}

void PeEndpointV5::handleCoreEgress_(SST::Event* event, int core_index) {
    routeSourceSpike_(event, core_index, true);
}

void PeEndpointV5::routeSourceSpike_(SST::Event* event, int core_index, bool monitor) {
    auto* spike = dynamic_cast<CoreSpikeEvent*>(event);
    if (!spike || core_index < 0 || static_cast<std::size_t>(core_index) >= cores_.size()) {
        delete event; out_.fatal(CALL_INFO, -1, "bad Core egress\n");
    }
    auto& core = cores_[core_index];
    if (monitor && core.provider_monitor) core.provider_monitor->send(spike->clone());
    ++logical_spikes_;
    const auto route = core.source_routes.find(static_cast<std::uint32_t>(spike->source_neuron));
    if (route_contract_v2_ && route == core.source_routes.end()) {
        ++zero_fanout_;
        ++core_zero_fanout_;
        delete spike;
        return;
    }

    std::vector<RouteTarget> targets;
    std::uint64_t source_global = spike->source_neuron;
    std::uint64_t route_id = 0;
    if (route != core.source_routes.end()) {
        targets = route->second.targets;
        source_global = route->second.source_global;
        route_id = route->second.route_id;
    } else {
        targets.push_back(RouteTarget{core.destination_pe, std::uint64_t(1) << core.destination_core});
    }
    std::uint64_t deliveries = 0;
    for (const auto& target : targets) deliveries += static_cast<std::uint64_t>(__builtin_popcountll(target.core_mask));
    const auto physical_packets = native_tree_ ? std::uint64_t(1) : deliveries;
    if (data_tx_.size() + physical_packets > tx_capacity_) {
        delete spike; out_.fatal(CALL_INFO, -1, "Core multicast egress exceeds finite endpoint TX capacity\n");
    }
    auto& epoch_enqueued = core_epoch_enqueued_[core_index];
    if (spike->timestep >= epoch_enqueued.size()) epoch_enqueued.resize(spike->timestep + 1, 0);
    epoch_enqueued[spike->timestep] += deliveries;
    if (native_tree_) {
        auto* packet = new NocPacketV5Event();
        packet->packet_id = (std::uint64_t(pe_id_) << 56) ^ next_packet_id_++;
        packet->timestep = spike->timestep;
        packet->source_pe = pe_id_;
        packet->source_core = core_index;
        packet->source_neuron = source_global;
        packet->route_id = route_id;
        packet->source_event_seq = spike->source_event_seq;
        packet->payload_bytes = payload_bytes_;
        packet->injection_time_ns = getCurrentSimTimeNano();
        data_tx_.push_back(packet);
        ++source_packets_;
    } else for (const auto& target : targets) {
        for (std::uint32_t destination_core = 0; destination_core < cores_per_pe_; ++destination_core) {
            if ((target.core_mask & (std::uint64_t(1) << destination_core)) == 0) continue;
            auto* packet = new NocPacketV5Event();
            packet->packet_id = (std::uint64_t(pe_id_) << 56) ^ next_packet_id_++;
            packet->timestep = spike->timestep;
            packet->source_pe = pe_id_;
            packet->source_core = core_index;
            packet->source_neuron = source_global;
            packet->route_id = route_id;
            packet->source_event_seq = spike->source_event_seq;
            packet->destination_pe = target.pe;
            packet->destination_core = destination_core;
            packet->target_neuron = 0;
            packet->payload_bytes = payload_bytes_;
            packet->injection_time_ns = getCurrentSimTimeNano();
            data_tx_.push_back(packet);
            ++source_packets_;
        }
    }
    delete spike;
}

void PeEndpointV5::handleCoreStatus_(SST::Event* event, int core_index) {
    auto* status = dynamic_cast<CoreStatusEvent*>(event);
    if (!status || core_index < 0 || static_cast<std::size_t>(core_index) >= cores_.size()) {
        delete event; out_.fatal(CALL_INFO, -1, "bad Core status\n");
    }
    auto& core = cores_[core_index];
    if (core.provider_status) core.provider_status->send(status->clone());
    if (timed_control_ && (status->operation == CoreControlOp::CommitReady ||
                           status->operation == CoreControlOp::CommitDone)) {
        sendStatus_(status->operation, status->timestep, core_index);
    }
    delete status;
}

void PeEndpointV5::transmitData_() {
    if (data_tx_.empty()) return;
    auto* packet = data_tx_.front();
    if (native_tree_) {
        if (native_credits_ == 0) { ++tx_stalls_; return; }
        native_link_->send(packet);
        --native_credits_;
        data_tx_.pop_front();
        ++tx_packets_;
        const auto bits = packet->wireBytes() * 8;
        tx_bits_ += bits;
        tx_flits_ += (packet->wireBytes() + flit_bytes_ - 1) / flit_bytes_;
        return;
    }
    if (packet->destination_pe == pe_id_) {
        data_tx_.pop_front();
        ingress_rx_.push_back(packet);
        return;
    }
    const auto bits = packet->wireBytes() * 8;
    if (!network_->spaceToSend(kNocDataVn, bits)) { ++tx_stalls_; return; }
    auto* request = new SST::Interfaces::SimpleNetwork::Request(
        packet->destination_pe, pe_id_, bits, true, true, packet);
    request->vn = kNocDataVn;
    request->allow_adaptive = false;
    network_->send(request, kNocDataVn);
    data_tx_.pop_front();
    ++tx_packets_;
    tx_bits_ += bits;
    tx_flits_ += (packet->wireBytes() + flit_bytes_ - 1) / flit_bytes_;
}

void PeEndpointV5::transmitControl_() {
    if (control_tx_.empty()) return;
    auto* packet = control_tx_.front();
    if (packet->destination_pe == pe_id_) {
        control_tx_.pop_front();
        deliverControl_(packet);
        return;
    }
    const auto bits = packet->wireBytes() * 8;
    if (!network_->spaceToSend(kNocControlVn, bits)) { ++tx_stalls_; return; }
    auto* request = new SST::Interfaces::SimpleNetwork::Request(
        packet->destination_pe, pe_id_, bits, true, true, packet);
    request->vn = kNocControlVn;
    request->allow_adaptive = false;
    network_->send(request, kNocControlVn);
    control_tx_.pop_front();
    ++control_tx_packets_;
    control_bits_ += bits;
    control_flits_ += (packet->wireBytes() + flit_bytes_ - 1) / flit_bytes_;
}

void PeEndpointV5::receiveData_() {
    if (native_tree_) return;
    if (ingress_rx_.size() >= static_cast<std::size_t>(rx_capacity_) * cores_per_pe_) {
        if (network_->requestToReceive(kNocDataVn)) ++rx_stalls_;
        return;
    }
    if (!network_->requestToReceive(kNocDataVn)) return;
    auto* request = network_->recv(kNocDataVn);
    if (!request) return;
    auto* packet = dynamic_cast<NocPacketV5Event*>(request->takePayload());
    if (!packet || packet->format_version != kNocPacketV5FormatVersion || packet->destination_pe != pe_id_) {
        delete packet; delete request; out_.fatal(CALL_INFO, -1, "invalid received v5 data packet\n");
    }
    const auto now = std::uint64_t(getCurrentSimTimeNano());
    const auto latency = now >= packet->injection_time_ns ? now - packet->injection_time_ns : 0;
    const auto sx = packet->source_pe % mesh_x_, sy = packet->source_pe / mesh_x_;
    const auto dx = pe_id_ % mesh_x_, dy = pe_id_ / mesh_x_;
    const auto hops = std::uint64_t(sx > dx ? sx - dx : dx - sx) + std::uint64_t(sy > dy ? sy - dy : dy - sy);
    latency_sum_ns_ += latency; latency_max_ns_ = std::max(latency_max_ns_, latency);
    hop_sum_ += hops; hop_max_ = std::max(hop_max_, hops);
    ++rx_packets_;
    ingress_rx_.push_back(packet);
    delete request;
}

void PeEndpointV5::receiveControl_() {
    if (!timed_control_ || !network_->requestToReceive(kNocControlVn)) return;
    auto* request = network_->recv(kNocControlVn);
    if (!request) return;
    auto* packet = dynamic_cast<NocControlV5Event*>(request->takePayload());
    if (!packet || packet->format_version != kNocPacketV5FormatVersion || packet->destination_pe != pe_id_) {
        delete packet; delete request; out_.fatal(CALL_INFO, -1, "invalid received v5 control packet\n");
    }
    ++control_rx_packets_;
    delete request;
    deliverControl_(packet);
}

void PeEndpointV5::distributeData_() {
    if (ingress_rx_.empty()) return;
    auto* packet = ingress_rx_.front();
    if (!core_attached_) {
        if (probe_out_) probe_out_->send(packet->clone());
        delete packet;
        ingress_rx_.pop_front();
        ++logical_deliveries_;
        return;
    }
    if (packet->destination_core_mask != 0) {
        if (packet->destination_core_mask >> cores_per_pe_)
            out_.fatal(CALL_INFO, -1, "native packet contains an invalid Core mask\n");
        for (std::uint32_t core=0; core<cores_per_pe_; ++core)
            if ((packet->destination_core_mask & (std::uint64_t(1) << core)) && cores_[core].rx.size() >= rx_capacity_) {
                ++rx_stalls_; return;
            }
        ingress_rx_.pop_front();
        for (std::uint32_t core=0; core<cores_per_pe_; ++core) {
            if ((packet->destination_core_mask & (std::uint64_t(1) << core)) == 0) continue;
            auto* copy = packet->clone();
            copy->destination_core = core;
            copy->destination_core_mask = 0;
            cores_[core].rx.push_back(copy);
        }
        delete packet;
        if (native_link_) native_link_->send(new NocCreditV5Event());
        return;
    }
    if (packet->destination_core >= cores_per_pe_) {
        out_.fatal(CALL_INFO, -1, "received packet has invalid destination_core=%u\n", packet->destination_core);
    }
    auto& core = cores_[packet->destination_core];
    if (core.rx.size() >= rx_capacity_) { ++rx_stalls_; return; }
    ingress_rx_.pop_front();
    core.rx.push_back(packet);
}

void PeEndpointV5::dispatchCores_() {
    for (auto& core : cores_) {
        if (!core.started || core.network_inflight || core.rx.empty()) continue;
        auto* packet = core.rx.front();
        auto* spike = new CoreSpikeEvent();
        spike->timestep = packet->timestep;
        spike->source_neuron = packet->source_neuron;
        spike->target_neuron = packet->target_neuron;
        spike->source_event_seq = packet->source_event_seq;
        core.ack_origins.push_back(AckOrigin::Network);
        core.core_spike->send(spike);
        core.network_inflight = true;
    }
}

void PeEndpointV5::deliverControl_(NocControlV5Event* packet) {
    if (packet->kind == NocControlV5Kind::Status) {
        if (pe_id_ != coordinator_pe_ || !epoch_status_) {
            delete packet; out_.fatal(CALL_INFO, -1, "control status reached a non-coordinator endpoint\n");
        }
        ++control_deliveries_;
        epoch_status_->send(packet);
        return;
    }
    if (packet->destination_core >= cores_per_pe_) {
        delete packet; out_.fatal(CALL_INFO, -1, "control command has invalid destination Core\n");
    }
    auto& core = cores_[packet->destination_core];
    auto* control = new CoreControlEvent(packet->operation, packet->epoch);
    if (packet->operation == CoreControlOp::Start) {
        core.started = true;
        if (core.provider_control) core.provider_control->send(control->clone());
    }
    if (packet->operation == CoreControlOp::SealIngress && !drainedForSeal_(packet->destination_core)) {
        delete core.pending_seal;
        core.pending_seal = control;
    } else {
        core.core_control->send(control);
    }
    ++control_deliveries_;
    delete packet;
}

bool PeEndpointV5::drainedForSeal_(std::size_t core) const {
    return ingress_rx_.empty() && cores_[core].rx.empty() && !cores_[core].network_inflight &&
           cores_[core].ack_origins.empty();
}

bool PeEndpointV5::tick_(SST::Cycle_t) {
    ++cycles_;
    receiveControl_();
    receiveData_();
    distributeData_();
    dispatchCores_();
    transmitControl_();
    transmitData_();
    for (std::size_t core = 0; core < cores_.size(); ++core) {
        if (cores_[core].pending_seal && drainedForSeal_(core)) {
            cores_[core].core_control->send(cores_[core].pending_seal);
            cores_[core].pending_seal = nullptr;
        }
    }
    return false;
}

void PeEndpointV5::writeEvidence_() const {
    if (output_json_.empty()) return;
    std::uint64_t core_rx_remaining = 0, ack_remaining = 0;
    for (const auto& core : cores_) {
        core_rx_remaining += core.rx.size();
        ack_remaining += core.ack_origins.size();
    }
    std::ofstream out(output_json_);
    out << "{\n  \"pe_id\": " << pe_id_
        << ",\n  \"cores_per_pe\": " << cores_per_pe_
        << ",\n  \"cycles\": " << cycles_
        << ",\n  \"tx_packets\": " << tx_packets_
        << ",\n  \"rx_packets\": " << rx_packets_
        << ",\n  \"logical_deliveries\": " << logical_deliveries_
        << ",\n  \"logical_spikes\": " << logical_spikes_
        << ",\n  \"source_packets\": " << source_packets_
        << ",\n  \"zero_fanout\": " << zero_fanout_
        << ",\n  \"external_zero_fanout\": " << external_zero_fanout_
        << ",\n  \"core_zero_fanout\": " << core_zero_fanout_
        << ",\n  \"multicast_mode\": \"" << (native_tree_ ? "native_tree" : "source_replication") << "\""
        << ",\n  \"control_tx_packets\": " << control_tx_packets_
        << ",\n  \"control_rx_packets\": " << control_rx_packets_
        << ",\n  \"control_deliveries\": " << control_deliveries_
        << ",\n  \"tx_bits\": " << tx_bits_
        << ",\n  \"tx_flits\": " << tx_flits_
        << ",\n  \"control_bits\": " << control_bits_
        << ",\n  \"control_flits\": " << control_flits_
        << ",\n  \"tx_stall_cycles\": " << tx_stalls_
        << ",\n  \"rx_stall_cycles\": " << rx_stalls_
        << ",\n  \"core_retry_cycles\": " << core_retries_
        << ",\n  \"latency_sum_ns\": " << latency_sum_ns_
        << ",\n  \"latency_max_ns\": " << latency_max_ns_
        << ",\n  \"hop_sum\": " << hop_sum_
        << ",\n  \"hop_max\": " << hop_max_
        << ",\n  \"tx_queue_remaining\": " << data_tx_.size()
        << ",\n  \"control_queue_remaining\": " << control_tx_.size()
        << ",\n  \"rx_queue_remaining\": " << (ingress_rx_.size() + core_rx_remaining)
        << ",\n  \"ack_queue_remaining\": " << ack_remaining
        << ",\n  \"drops\": " << drops_ << "\n}\n";
}

void PeEndpointV5::finish() {
    writeEvidence_();
    registerStatistic<std::uint64_t>("noc.packets")->addData(tx_packets_);
    registerStatistic<std::uint64_t>("noc.flits")->addData(tx_flits_);
    registerStatistic<std::uint64_t>("noc.logical_deliveries")->addData(logical_deliveries_);
    registerStatistic<std::uint64_t>("noc.control_packets")->addData(control_tx_packets_);
    registerStatistic<std::uint64_t>("noc.control_flits")->addData(control_flits_);
    registerStatistic<std::uint64_t>("noc.tx_stall_cycles")->addData(tx_stalls_);
    registerStatistic<std::uint64_t>("noc.rx_stall_cycles")->addData(rx_stalls_);
    registerStatistic<std::uint64_t>("noc.core_retry_cycles")->addData(core_retries_);
    network_->finish();
}

}}}
