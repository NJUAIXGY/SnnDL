#include <sst/core/sst_config.h>

#include "IdealSynapseSource.h"
#include "v5/events/StorageEvents.h"

#include <algorithm>
#include <cinttypes>
#include <cstring>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace SST {
namespace SnnDL {
namespace v5 {

namespace {
std::uint64_t parseUnsigned(const std::string& value, const char* name) {
    try {
        std::size_t consumed = 0;
        const auto parsed = std::stoull(value, &consumed, 10);
        if (consumed != value.size()) throw std::invalid_argument("trailing characters");
        return parsed;
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string("invalid ") + name + ": " + value);
    }
}
}

IdealSynapseSource::IdealSynapseSource(SST::ComponentId_t id, SST::Params& params)
    : SST::Component(id), out_("SnnDL.IdealSynapseSource", 0, 0, SST::Output::STDOUT),
      neurons_(std::max<std::uint32_t>(1, params.find<std::uint32_t>("neurons", 64))),
      timesteps_(std::max<std::uint64_t>(1, params.find<std::uint64_t>("timesteps", 1))),
      reverse_responses_(params.find<int>("reverse_responses", 0) != 0),
      memory_backed_weights_(params.find<int>("memory_backed_weights", 0) != 0),
      weight_image_base_(params.find<std::uint64_t>("weight_image_base", 0)),
      weight_read_base_(params.find<std::uint64_t>("weight_read_base", 0)),
      output_json_(params.find<std::string>("output_json", "")) {
    out_.setVerboseLevel(params.find<int>("verbose", 0));
    try {
        parseEdges_(params.find<std::string>("edges", "0:0:1.0:0"));
        parseStimuli_(params.find<std::string>("stimuli", "0:0:1"));
        buildWeightImage_();
    } catch (const std::exception& error) {
        out_.fatal(CALL_INFO, -1, "IdealSynapseSource configuration error: %s\n", error.what());
    }
    control_link_ = configureLink("control", new SST::Event::Handler2<IdealSynapseSource, &IdealSynapseSource::handleControl_>(this));
    spike_out_link_ = configureLink("spike_out");
    spike_ack_link_ = configureLink("spike_ack", new SST::Event::Handler2<IdealSynapseSource, &IdealSynapseSource::handleSpikeAck_>(this));
    spike_in_link_ = configureLink("spike_in", new SST::Event::Handler2<IdealSynapseSource, &IdealSynapseSource::handleSpike_>(this));
    row_provider_link_ = configureLink("row_provider", new SST::Event::Handler2<IdealSynapseSource, &IdealSynapseSource::handleProvider_>(this));
    status_link_ = configureLink("status", new SST::Event::Handler2<IdealSynapseSource, &IdealSynapseSource::handleStatus_>(this));
    preload_link_ = configureLink("preload", new SST::Event::Handler2<IdealSynapseSource, &IdealSynapseSource::handlePreload_>(this));
    if (!control_link_ || !spike_out_link_ || !spike_ack_link_ || !spike_in_link_ || !row_provider_link_ || !status_link_) {
        out_.fatal(CALL_INFO, -1, "IdealSynapseSource requires all six links\n");
    }
    if (memory_backed_weights_) {
        if (!preload_link_) out_.fatal(CALL_INFO, -1, "memory-backed IdealSynapseSource requires preload link\n");
        memory_ = loadUserSubComponent<SST::Interfaces::StandardMem>(
            "memory", ComponentInfo::SHARE_NONE, registerTimeBase("1ns"),
            new SST::Interfaces::StandardMem::Handler2<
                IdealSynapseSource, &IdealSynapseSource::handleMemory_>(this));
        if (!memory_) out_.fatal(CALL_INFO, -1, "memory-backed IdealSynapseSource requires StandardMem slot 'memory'\n");
    } else {
        preload_ready_ = true;
    }
    registerClock(params.find<std::string>("clock", "1GHz"), new SST::Clock::Handler2<IdealSynapseSource, &IdealSynapseSource::clockTick_>(this));
    registerAsPrimaryComponent();
    primaryComponentDoNotEndSim();
}

IdealSynapseSource::~IdealSynapseSource() = default;
void IdealSynapseSource::init(unsigned int phase) {
    if (!memory_) return;
    memory_->init(phase);
    if (phase == 0 && !image_initialized_) {
        memory_->sendUntimedData(new SST::Interfaces::StandardMem::Write(
            weight_image_base_, weight_image_.size(), weight_image_, true));
        image_initialized_ = true;
    }
    while (auto* response = memory_->recvUntimedData()) delete response;
}
void IdealSynapseSource::setup() { if (memory_) memory_->setup(); }

void IdealSynapseSource::buildWeightImage_() {
    static constexpr std::size_t record_bytes = sizeof(std::uint32_t) + sizeof(float) + sizeof(std::uint64_t);
    for (const auto& item : edges_by_pre_) {
        row_locations_[item.first] = RowLocation{weight_image_.size(), item.second.size()};
        for (const auto& edge : item.second) {
            const auto begin = weight_image_.size();
            weight_image_.resize(begin + record_bytes);
            std::memcpy(weight_image_.data() + begin, &edge.post, sizeof(edge.post));
            std::memcpy(weight_image_.data() + begin + sizeof(edge.post), &edge.weight, sizeof(edge.weight));
            std::memcpy(weight_image_.data() + begin + sizeof(edge.post) + sizeof(edge.weight),
                        &edge.ordinal, sizeof(edge.ordinal));
            image_weight_sum_ += edge.weight;
        }
    }
    if (weight_image_.empty()) weight_image_.resize(record_bytes, 0);
}

std::vector<std::string> IdealSynapseSource::split_(const std::string& value, char separator) {
    std::vector<std::string> result;
    std::string item;
    std::stringstream stream(value);
    while (std::getline(stream, item, separator)) result.push_back(item);
    return result;
}

void IdealSynapseSource::parseEdges_(const std::string& encoded) {
    for (const auto& item : split_(encoded, ';')) {
        if (item.empty()) continue;
        const auto fields = split_(item, ':');
        if (fields.size() != 4) throw std::invalid_argument("edge must be pre:post:weight:ordinal");
        Edge edge;
        edge.pre = static_cast<std::uint32_t>(parseUnsigned(fields[0], "edge pre"));
        edge.post = static_cast<std::uint32_t>(parseUnsigned(fields[1], "edge post"));
        if (edge.post >= neurons_) throw std::invalid_argument("edge post is outside provider neuron count");
        edge.weight = std::stof(fields[2]);
        edge.ordinal = parseUnsigned(fields[3], "edge ordinal");
        edges_.push_back(edge);
        edges_by_pre_[edge.pre].push_back(edge);
    }
    for (auto& item : edges_by_pre_) {
        std::sort(item.second.begin(), item.second.end(), [](const Edge& lhs, const Edge& rhs) {
            return lhs.ordinal < rhs.ordinal;
        });
    }
}

void IdealSynapseSource::parseStimuli_(const std::string& encoded) {
    for (const auto& item : split_(encoded, ';')) {
        if (item.empty()) continue;
        const auto fields = split_(item, ':');
        if (fields.size() < 2 || fields.size() > 3) throw std::invalid_argument("stimulus must be timestep:source[:sequence]");
        Stimulus stimulus;
        stimulus.timestep = parseUnsigned(fields[0], "stimulus timestep");
        stimulus.source = static_cast<std::uint32_t>(parseUnsigned(fields[1], "stimulus source"));
        stimulus.sequence = fields.size() == 3 ? parseUnsigned(fields[2], "stimulus sequence") : stimuli_.size() + 1;
        if (stimulus.timestep >= timesteps_ || stimulus.source >= neurons_) throw std::invalid_argument("stimulus is outside configured range");
        stimuli_.push_back(stimulus);
    }
    std::stable_sort(stimuli_.begin(), stimuli_.end(), [](const Stimulus& lhs, const Stimulus& rhs) {
        if (lhs.timestep != rhs.timestep) return lhs.timestep < rhs.timestep;
        return lhs.sequence < rhs.sequence;
    });
}

void IdealSynapseSource::sendControl_(CoreControlOp operation, std::uint64_t timestep) {
    auto* event = new CoreControlEvent(operation, timestep);
    control_link_->send(event);
}

void IdealSynapseSource::sendStimuli_() {
    if (stimulus_in_flight_) return;
    while (stimulus_cursor_ < stimuli_.size() && stimuli_[stimulus_cursor_].timestep < current_timestep_) {
        ++stimulus_cursor_;
    }
    if (stimulus_cursor_ == stimuli_.size() || stimuli_[stimulus_cursor_].timestep != current_timestep_) {
        stimuli_sent_ = true;
        return;
    }
    stimulus_sent_ = stimuli_[stimulus_cursor_];
    auto* event = new CoreSpikeEvent();
    event->timestep = stimulus_sent_.timestep;
    event->source_neuron = stimulus_sent_.source;
    event->source_event_seq = stimulus_sent_.sequence;
    spike_out_link_->send(event);
    stimulus_in_flight_ = true;
}

void IdealSynapseSource::respondToRow_(const CoreRowRequestEvent& request) {
    if (memory_backed_weights_) {
        ProviderTransaction transaction;
        transaction.request = request;
        const auto found = row_locations_.find(request.source_neuron);
        if (found != row_locations_.end()) {
            transaction.memory_offset = found->second.byte_offset;
            transaction.memory_edges = found->second.edge_count;
        }
        provider_transactions_.push_back(std::move(transaction));
        ++rows_served_;
        issueMemoryRead_();
        return;
    }
    std::vector<Edge> row;
    const auto found = edges_by_pre_.find(request.source_neuron);
    if (found != edges_by_pre_.end()) row = found->second;
    if (reverse_responses_) std::reverse(row.begin(), row.end());
    provider_transactions_.push_back(ProviderTransaction{request, std::move(row)});
    ++rows_served_;
    sendNextProviderItem_();
}

void IdealSynapseSource::issueMemoryRead_() {
    if (!memory_ || provider_transactions_.empty() || pending_memory_) return;
    auto& transaction = provider_transactions_.front();
    if (transaction.memory_reads_completed >= transaction.memory_edges) {
        sendNextProviderItem_();
        return;
    }
    static constexpr std::uint64_t record_bytes = sizeof(std::uint32_t) + sizeof(float) + sizeof(std::uint64_t);
    const auto address = weight_read_base_ + transaction.memory_offset +
                         transaction.memory_reads_completed * record_bytes;
    auto* request = new SST::Interfaces::StandardMem::Read(address, record_bytes);
    pending_memory_request_ = static_cast<std::uint64_t>(request->getID());
    pending_memory_ = true;
    transaction.memory_read_in_flight = true;
    memory_->send(request);
    ++memory_reads_;
    memory_read_bytes_ += record_bytes;
}

void IdealSynapseSource::handleMemory_(SST::Interfaces::StandardMem::Request* request) {
    auto* response = dynamic_cast<SST::Interfaces::StandardMem::ReadResp*>(request);
    if (!response || provider_transactions_.empty() ||
        static_cast<std::uint64_t>(request->getID()) != pending_memory_request_) {
        delete request;
        out_.fatal(CALL_INFO, -1, "IdealSynapseSource received an unexpected StandardMem response\n");
    }
    static constexpr std::size_t record_bytes = sizeof(std::uint32_t) + sizeof(float) + sizeof(std::uint64_t);
    if (response->data.size() < record_bytes) {
        delete response;
        out_.fatal(CALL_INFO, -1, "IdealSynapseSource received a truncated weight record\n");
    }
    Edge edge;
    std::memcpy(&edge.post, response->data.data(), sizeof(edge.post));
    std::memcpy(&edge.weight, response->data.data() + sizeof(edge.post), sizeof(edge.weight));
    std::memcpy(&edge.ordinal, response->data.data() + sizeof(edge.post) + sizeof(edge.weight), sizeof(edge.ordinal));
    if (edge.post >= neurons_) {
        delete response;
        out_.fatal(CALL_INFO, -1, "IdealSynapseSource decoded an invalid post neuron\n");
    }
    auto& transaction = provider_transactions_.front();
    transaction.row.push_back(edge);
    ++transaction.memory_reads_completed;
    transaction.memory_read_in_flight = false;
    decoded_weight_sum_ += edge.weight;
    if (transaction.memory_reads_completed == transaction.memory_edges && reverse_responses_) {
        std::reverse(transaction.row.begin(), transaction.row.end());
    }
    pending_memory_ = false;
    delete response;
    issueMemoryRead_();
}

void IdealSynapseSource::handlePreload_(SST::Event* event) {
    auto* completion = dynamic_cast<DmaCompletionEvent*>(event);
    if (!completion || !completion->accepted || !completion->completed || completion->error != 0) {
        delete event;
        out_.fatal(CALL_INFO, -1, "IdealSynapseSource received a failed preload completion\n");
    }
    if (preload_ready_) {
        delete completion;
        out_.fatal(CALL_INFO, -1, "IdealSynapseSource received duplicate preload completion\n");
    }
    preload_ready_ = true;
    preload_ready_cycle_ = tick_count_;
    delete completion;
}

void IdealSynapseSource::sendNextProviderItem_() {
    if (provider_transactions_.empty()) return;
    auto& transaction = provider_transactions_.front();
    if (memory_backed_weights_ && transaction.memory_reads_completed < transaction.memory_edges) {
        issueMemoryRead_();
        return;
    }
    if (transaction.in_flight) return;
    if (transaction.next_edge < transaction.row.size()) {
        const auto& edge = transaction.row[transaction.next_edge];
        auto* response = new CoreSynapseResponseEvent();
        response->timestep = transaction.request.timestep;
        response->source_neuron = transaction.request.source_neuron;
        response->source_event_seq = transaction.request.source_event_seq;
        response->post_neuron = edge.post;
        response->edge_ordinal = edge.ordinal;
        response->weight = edge.weight;
        row_provider_link_->send(response);
        transaction.in_flight = true;
        ++response_attempts_;
        return;
    }
    if (!transaction.done_sent) {
        auto* done = new CoreRowDoneEvent();
        done->timestep = transaction.request.timestep;
        done->source_neuron = transaction.request.source_neuron;
        done->source_event_seq = transaction.request.source_event_seq;
        done->edge_count = transaction.row.size();
        row_provider_link_->send(done);
        transaction.done_sent = true;
        transaction.in_flight = true;
    }
}

void IdealSynapseSource::handleControl_(SST::Event* event) {
    delete event;
}

void IdealSynapseSource::handleSpike_(SST::Event* event) {
    auto* spike = dynamic_cast<CoreSpikeEvent*>(event);
    if (spike) {
        ++output_spikes_;
        functional_hash_ ^= spike->source_neuron + 0x9e3779b97f4a7c15ULL + (functional_hash_ << 6) + (functional_hash_ >> 2);
    }
    delete event;
}

void IdealSynapseSource::handleSpikeAck_(SST::Event* event) {
    auto* ack = dynamic_cast<CoreSpikeAckEvent*>(event);
    if (!ack) {
        delete event;
        out_.fatal(CALL_INFO, -1, "IdealSynapseSource received an unexpected spike ack\n");
    }
    if (!stimulus_in_flight_ || ack->timestep != stimulus_sent_.timestep ||
        ack->source_neuron != stimulus_sent_.source || ack->source_event_seq != stimulus_sent_.sequence) {
        delete ack;
        out_.fatal(CALL_INFO, -1, "IdealSynapseSource received a mismatched spike ack\n");
    }
    if (!ack->accepted && !ack->retryable) {
        delete ack;
        out_.fatal(CALL_INFO, -1, "IdealSynapseSource spike ingress was rejected permanently\n");
    }
    if (ack->accepted) {
        ++spike_acks_accepted_;
        ++stimulus_cursor_;
    } else {
        ++spike_retries_;
    }
    stimulus_in_flight_ = false;
    delete ack;
}

void IdealSynapseSource::handleProvider_(SST::Event* event) {
    if (auto* request = dynamic_cast<CoreRowRequestEvent*>(event)) {
        respondToRow_(*request);
        delete request;
        return;
    }
    auto* ack = dynamic_cast<CoreProviderAckEvent*>(event);
    if (!ack) {
        delete event;
        out_.fatal(CALL_INFO, -1, "IdealSynapseSource received an unexpected provider event\n");
    }
    if (provider_transactions_.empty()) {
        delete ack;
        out_.fatal(CALL_INFO, -1, "IdealSynapseSource received a provider ack without a transaction\n");
    }
    auto& transaction = provider_transactions_.front();
    if (ack->timestep != transaction.request.timestep || ack->source_neuron != transaction.request.source_neuron ||
        ack->source_event_seq != transaction.request.source_event_seq ||
        (ack->row_done && transaction.next_edge != transaction.row.size())) {
        delete ack;
        out_.fatal(CALL_INFO, -1, "IdealSynapseSource received a mismatched provider ack\n");
    }
    if (!ack->accepted && !ack->retryable) {
        delete ack;
        out_.fatal(CALL_INFO, -1, "IdealSynapseSource provider response was rejected permanently\n");
    }
    if (!ack->accepted) ++provider_retries_;
    transaction.in_flight = false;
    if (ack->accepted && ack->row_done) {
        provider_transactions_.pop_front();
    } else if (ack->accepted) {
        ++responses_served_;
        ++transaction.next_edge;
    }
    delete ack;
    sendNextProviderItem_();
}

void IdealSynapseSource::handleStatus_(SST::Event* event) {
    auto* status = dynamic_cast<CoreStatusEvent*>(event);
    if (!status) {
        delete event;
        out_.fatal(CALL_INFO, -1, "IdealSynapseSource received an unexpected status event\n");
    }
    last_status_ = *status;
    if (status->operation == CoreControlOp::CommitReady) {
        sendControl_(CoreControlOp::Commit, current_timestep_);
    } else if (status->operation == CoreControlOp::CommitDone) {
        if (current_timestep_ + 1 < timesteps_) {
            ++current_timestep_;
            started_ = false;
            stimuli_sent_ = false;
            seal_sent_ = false;
        } else if (!finished_) {
            finished_ = true;
            primaryComponentOKToEndSim();
        }
    }
    delete status;
}

bool IdealSynapseSource::clockTick_(SST::Cycle_t) {
    ++tick_count_;
    if (finished_) return false;
    if (!preload_ready_) {
        ++preload_wait_cycles_;
        return false;
    }
    if (!started_) {
        sendControl_(CoreControlOp::Start, current_timestep_);
        started_ = true;
        if (start_cycle_ == 0) start_cycle_ = tick_count_;
        return false;
    }
    if (!stimuli_sent_) {
        sendStimuli_();
    } else if (!seal_sent_) {
        sendControl_(CoreControlOp::SealIngress, current_timestep_);
        seal_sent_ = true;
    }
    return false;
}

void IdealSynapseSource::writeEvidence_() const {
    if (output_json_.empty()) return;
    std::ofstream out(output_json_, std::ios::out | std::ios::trunc);
    if (!out.good()) return;
    const auto writeStage = [&out](const char* name, const CoreStageCounters& stage, bool trailing_comma) {
        out << "    \"" << name << "\": {\"accepted\": " << stage.accepted
            << ", \"issued\": " << stage.issued
            << ", \"completed\": " << stage.completed
            << ", \"occupancy\": " << stage.occupancy
            << ", \"busy_cycles\": " << stage.busy_cycles
            << ", \"full_cycles\": " << stage.full_cycles
            << ", \"stall_cycles\": " << stage.stall_cycles << "}";
        if (trailing_comma) out << ",";
        out << "\n";
    };
    out << "{\n"
        << "  \"run_class\": \"development\",\n"
        << "  \"timing_evidence\": true,\n"
        << "  \"weight_mode\": \"" << (memory_backed_weights_ ? "memory_backed" : "ideal") << "\",\n"
        << "  \"weight_image_bytes\": " << weight_image_.size() << ",\n"
        << "  \"weight_image_sum\": " << image_weight_sum_ << ",\n"
        << "  \"decoded_weight_sum\": " << decoded_weight_sum_ << ",\n"
        << "  \"memory_reads\": " << memory_reads_ << ",\n"
        << "  \"memory_read_bytes\": " << memory_read_bytes_ << ",\n"
        << "  \"preload_ready\": " << (preload_ready_ ? "true" : "false") << ",\n"
        << "  \"preload_ready_cycle\": " << preload_ready_cycle_ << ",\n"
        << "  \"start_cycle\": " << start_cycle_ << ",\n"
        << "  \"preload_wait_cycles\": " << preload_wait_cycles_ << ",\n"
        << "  \"timesteps\": " << timesteps_ << ",\n"
        << "  \"rows_served\": " << rows_served_ << ",\n"
        << "  \"responses_served\": " << responses_served_ << ",\n"
        << "  \"response_attempts\": " << response_attempts_ << ",\n"
        << "  \"output_spikes\": " << output_spikes_ << ",\n"
        << "  \"spike_acks_accepted\": " << spike_acks_accepted_ << ",\n"
        << "  \"spike_retries\": " << spike_retries_ << ",\n"
        << "  \"provider_retries\": " << provider_retries_ << ",\n"
        << "  \"functional_hash\": " << functional_hash_ << ",\n"
        << "  \"core_cycles\": " << last_status_.core_cycles << ",\n"
        << "  \"core_elapsed_ns\": " << last_status_.core_elapsed_ns << ",\n"
        << "  \"ingress_accepted\": " << last_status_.ingress_accepted << ",\n"
        << "  \"synapse_issued\": " << last_status_.synapse_issued << ",\n"
        << "  \"retire_retired\": " << last_status_.retire_retired << ",\n"
        << "  \"neurons_evaluated\": " << last_status_.neurons_evaluated << ",\n"
        << "  \"storage\": {\"core_state_reads\": " << last_status_.storage_state_reads
        << ", \"core_state_writes\": " << last_status_.storage_state_writes
        << ", \"core_delta_reads\": " << last_status_.storage_delta_reads
        << ", \"core_delta_writes\": " << last_status_.storage_delta_writes
        << ", \"core_index_reads\": " << last_status_.storage_index_reads
        << ", \"pe_route_reads\": " << last_status_.storage_route_reads << "},\n"
        << "  \"stall\": {\"ingress_full_cycles\": " << last_status_.ingress_full_cycles
        << ", \"row\": " << last_status_.row_stall_cycles
        << ", \"synapse\": " << last_status_.synapse_stall_cycles
        << ", \"retire\": " << last_status_.retire_stall_cycles
        << ", \"accumulator\": " << last_status_.accumulator_stall_cycles
        << ", \"held\": " << last_status_.held_full_cycles << "},\n"
        << "  \"stages\": {\n";
    writeStage("ingress", last_status_.ingress, true);
    writeStage("row_lookup", last_status_.row_lookup, true);
    writeStage("synapse", last_status_.synapse, true);
    writeStage("retire", last_status_.retire, true);
    writeStage("accumulator", last_status_.accumulator, true);
    writeStage("neuron", last_status_.neuron, true);
    writeStage("held_spike", last_status_.held_spike, false);
    out << "  }\n"
        << "}\n";
}

void IdealSynapseSource::finish() {
    if (memory_) memory_->finish();
    if (pending_memory_ || !provider_transactions_.empty()) {
        out_.fatal(CALL_INFO, -1, "IdealSynapseSource finished with outstanding memory/provider state\n");
    }
    writeEvidence_();
    out_.verbose(CALL_INFO, 1, 0, "[snndl-v5-ideal] timesteps=%" PRIu64 " rows=%" PRIu64 " responses=%" PRIu64 "\n", timesteps_, rows_served_, responses_served_);
}

} // namespace v5
} // namespace SnnDL
} // namespace SST
