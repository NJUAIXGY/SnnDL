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
      core_id_(params.find<std::uint32_t>("core_id", 0)),
      timesteps_(std::max<std::uint64_t>(1, params.find<std::uint64_t>("timesteps", 1))),
      start_timestep_(params.find<std::uint64_t>("start_timestep", 0)),
      reverse_responses_(params.find<int>("reverse_responses", 0) != 0),
      memory_backed_weights_(params.find<int>("memory_backed_weights", 0) != 0),
      external_control_(params.find<int>("external_control", 0) != 0),
      weight_image_base_(params.find<std::uint64_t>("weight_image_base", 0)),
      weight_read_base_(params.find<std::uint64_t>("weight_read_base", 0)),
      weight_cache_line_bytes_(std::max<std::size_t>(16, params.find<std::size_t>(
          "weight_cache_line_bytes", 64))),
      weight_image_write_bytes_(std::max<std::size_t>(16, params.find<std::size_t>(
          "weight_image_write_bytes", 64))),
      weight_read_granularity_bytes_(std::max<std::size_t>(16, params.find<std::size_t>(
          "weight_read_granularity_bytes", 16))),
      dram_bank_count_(std::max<std::size_t>(1, params.find<std::size_t>("dram_bank_count", 1))),
      dram_bank_interleave_bytes_(std::max<std::size_t>(1, params.find<std::size_t>("dram_bank_interleave_bytes", 64))),
      dram_bank_policy_(params.find<std::string>("dram_bank_policy", "low_bits")),
      output_json_(params.find<std::string>("output_json", "")),
      request_trace_json_(params.find<std::string>("request_trace_json", "")) {
    if (!request_trace_json_.empty()) {
        request_trace_stream_.open(request_trace_json_, std::ios::out | std::ios::trunc);
    }
    readout_start_ = params.find<std::uint64_t>("readout_start", 0);
    readout_count_ = params.find<std::uint32_t>("readout_count", 0);
    if (readout_count_ > 0) readout_spike_counts_.assign(readout_count_, 0);
    if (weight_read_granularity_bytes_ % 16 != 0) {
        out_.fatal(CALL_INFO, -1, "weight_read_granularity_bytes must be a multiple of 16\n");
    }
    if (weight_cache_line_bytes_ < 16 || weight_cache_line_bytes_ % 16 != 0) {
        out_.fatal(CALL_INFO, -1, "weight_cache_line_bytes must be a positive multiple of 16\n");
    }
    if (weight_image_write_bytes_ < 16 || weight_image_write_bytes_ % 16 != 0) {
        out_.fatal(CALL_INFO, -1, "weight_image_write_bytes must be a positive multiple of 16\n");
    }
    if (dram_bank_policy_ != "low_bits") {
        out_.fatal(CALL_INFO, -1, "unsupported dram_bank_policy; only low_bits is legal\n");
    }
    current_timestep_ = start_timestep_;
    out_.setVerboseLevel(params.find<int>("verbose", 0));
    try {
        const auto artifact_file = params.find<std::string>("artifact_weight_file", "");
        artifact_mode_ = !artifact_file.empty();
        artifact_digest_ = params.find<std::string>("artifact_digest", "");
        if (artifact_mode_) {
            parseArtifactRows_(params.find<std::string>("artifact_rows", ""));
            parseStimuli_(params.find<std::string>("artifact_stimuli", ""));
            loadArtifactWeights_(
                artifact_file,
                params.find<std::uint64_t>("artifact_weight_file_offset", 0),
                params.find<std::uint64_t>("artifact_weight_bytes", 0));
        } else {
            if (params.find<int>("artifact_required", 0) != 0) {
                throw std::invalid_argument("canonical provider requires artifact_weight_file");
            }
            parseEdges_(params.find<std::string>("edges", "0:0:1.0:0"));
            parseStimuli_(params.find<std::string>("stimuli", "0:0:1"));
            buildWeightImage_();
        }
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
        out_.fatal(CALL_INFO, -1, "IdealSynapseSource requires control, spike, row-provider, and status links\n");
    }
    if (memory_backed_weights_) {
        memory_ = loadUserSubComponent<SST::Interfaces::StandardMem>(
            "memory", ComponentInfo::SHARE_NONE, registerTimeBase("1ns"),
            new SST::Interfaces::StandardMem::Handler2<
                IdealSynapseSource, &IdealSynapseSource::handleMemory_>(this));
        if (!memory_) out_.fatal(CALL_INFO, -1, "memory-backed IdealSynapseSource requires StandardMem slot 'memory'\n");
        // The canonical v5 path reads weights directly from ChipDram through
        // this StandardMem client.  The artifact image is populated during
        // initialization; no DMA preload or local weight scratchpad is involved.
        preload_ready_ = true;
    } else {
        preload_ready_ = true;
    }
    registerClock(params.find<std::string>("clock", "1GHz"), new SST::Clock::Handler2<IdealSynapseSource, &IdealSynapseSource::clockTick_>(this));
    // In the canonical external-control topology the epoch coordinator is
    // the sole simulation-lifetime authority.  Keeping every provider in the
    // SST Exit refcount lets a provider that has locally drained release the
    // simulation before a slower Core has reported CommitReady.  Standalone
    // provider tests still use the original primary-component contract.
    if (!external_control_) {
        registerAsPrimaryComponent();
        primaryComponentDoNotEndSim();
    }
}

IdealSynapseSource::~IdealSynapseSource() = default;
void IdealSynapseSource::init(unsigned int phase) {
    if (!memory_) return;
    memory_->init(phase);
    if (phase == 0 && !image_initialized_) {
        for (std::size_t offset = 0; offset < weight_image_.size(); offset += weight_image_write_bytes_) {
            const auto bytes = std::min(weight_image_write_bytes_, weight_image_.size() - offset);
            std::vector<std::uint8_t> payload(
                weight_image_.begin() + static_cast<std::ptrdiff_t>(offset),
                weight_image_.begin() + static_cast<std::ptrdiff_t>(offset + bytes));
            memory_->sendUntimedData(new SST::Interfaces::StandardMem::Write(
                weight_image_base_ + offset, bytes, std::move(payload), true));
        }
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

void IdealSynapseSource::parseArtifactRows_(const std::string& encoded) {
    for (const auto& item : split_(encoded, ';')) {
        if (item.empty()) continue;
        const auto fields = split_(item, ':');
        if (fields.size() != 3) throw std::invalid_argument("artifact row must be pre:byte_offset:edge_count");
        const auto pre = parseUnsigned(fields[0], "artifact row pre");
        if (!row_locations_.emplace(pre, RowLocation{
                parseUnsigned(fields[1], "artifact row byte offset"),
                static_cast<std::size_t>(parseUnsigned(fields[2], "artifact row edge count"))}).second) {
            throw std::invalid_argument("duplicate artifact row");
        }
    }
}

void IdealSynapseSource::loadArtifactWeights_(const std::string& path, std::uint64_t offset, std::uint64_t bytes) {
    static constexpr std::uint64_t record_bytes = sizeof(std::uint32_t) + sizeof(float) + sizeof(std::uint64_t);
    if (bytes == 0 || bytes % record_bytes != 0) {
        throw std::invalid_argument("artifact weight region must contain complete 16-byte records");
    }
    std::ifstream input(path, std::ios::binary);
    if (!input.good()) throw std::invalid_argument("cannot open artifact weight file: " + path);
    input.seekg(0, std::ios::end);
    const auto file_bytes = static_cast<std::uint64_t>(input.tellg());
    if (offset > file_bytes || bytes > file_bytes - offset) {
        throw std::invalid_argument("artifact weight region exceeds file size");
    }
    input.seekg(static_cast<std::streamoff>(offset), std::ios::beg);
    weight_image_.resize(static_cast<std::size_t>(bytes));
    input.read(reinterpret_cast<char*>(weight_image_.data()), static_cast<std::streamsize>(bytes));
    if (static_cast<std::uint64_t>(input.gcount()) != bytes) {
        throw std::invalid_argument("artifact weight region was truncated while reading");
    }
    for (std::size_t position = 0; position < weight_image_.size(); position += record_bytes) {
        float weight = 0.0f;
        std::memcpy(&weight, weight_image_.data() + position + sizeof(std::uint32_t), sizeof(weight));
        image_weight_sum_ += weight;
    }
    for (const auto& row : row_locations_) {
        if (row.second.byte_offset > bytes || row.second.edge_count >
                (bytes - row.second.byte_offset) / record_bytes) {
            throw std::invalid_argument("artifact row exceeds weight region");
        }
    }
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
        edge.pre = parseUnsigned(fields[0], "edge pre");
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
        stimulus.source = parseUnsigned(fields[1], "stimulus source");
        stimulus.sequence = fields.size() == 3 ? parseUnsigned(fields[2], "stimulus sequence") : stimuli_.size() + 1;
        if (stimulus.timestep < start_timestep_ || stimulus.timestep >= start_timestep_ + timesteps_)
            throw std::invalid_argument("stimulus is outside configured range");
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
    const auto line_remaining = weight_cache_line_bytes_ - (address % weight_cache_line_bytes_);
    const auto records = std::min<std::size_t>(
        std::min(weight_read_granularity_bytes_ / record_bytes,
                 static_cast<std::size_t>(line_remaining / record_bytes)),
        transaction.memory_edges - transaction.memory_reads_completed);
    if (records == 0) {
        out_.fatal(CALL_INFO, -1, "IdealSynapseSource cannot issue a cache-line-bounded weight read\n");
    }
    const auto request_bytes = records * record_bytes;
    const auto bank = (address / dram_bank_interleave_bytes_) % dram_bank_count_;
    auto* request = new SST::Interfaces::StandardMem::Read(address, request_bytes);
    const auto request_id = static_cast<std::uint64_t>(request->getID());
    pending_memory_request_ = request_id;
    pending_memory_records_ = records;
    pending_memory_ = true;
    transaction.memory_read_in_flight = true;
    memory_->send(request);
    std::ostringstream identity;
    identity << "t" << transaction.request.timestep << ".core" << core_id_
             << ".row" << transaction.request.row_id << ".chunk"
             << transaction.memory_reads_completed;
    pending_memory_trace_ = MemoryRequestTrace{
        identity.str(), request_id, transaction.request.timestep,
        transaction.request.source_neuron, transaction.request.source_event_seq,
        transaction.request.row_id,
        transaction.memory_reads_completed, address,
        static_cast<std::uint64_t>(request_bytes), static_cast<std::uint64_t>(records),
        tick_count_, 0, bank, false,
    };
    pending_memory_trace_valid_ = true;
    memory_reads_ += records;
    ++memory_requests_;
    memory_read_bytes_ += request_bytes;
}

void IdealSynapseSource::handleMemory_(SST::Interfaces::StandardMem::Request* request) {
    auto* response = dynamic_cast<SST::Interfaces::StandardMem::ReadResp*>(request);
    if (!response || provider_transactions_.empty() ||
        static_cast<std::uint64_t>(request->getID()) != pending_memory_request_) {
        delete request;
        out_.fatal(CALL_INFO, -1, "IdealSynapseSource received an unexpected StandardMem response\n");
    }
    static constexpr std::size_t record_bytes = sizeof(std::uint32_t) + sizeof(float) + sizeof(std::uint64_t);
    auto& transaction = provider_transactions_.front();
    const auto records = pending_memory_records_;
    const auto response_bytes = records * record_bytes;
    if (response->data.size() < response_bytes) {
        delete response;
        out_.fatal(CALL_INFO, -1, "IdealSynapseSource received a truncated weight record\n");
    }
    for (std::size_t index = 0; index < records; ++index) {
        const auto* record = response->data.data() + index * record_bytes;
        Edge edge;
        std::memcpy(&edge.post, record, sizeof(edge.post));
        std::memcpy(&edge.weight, record + sizeof(edge.post), sizeof(edge.weight));
        std::memcpy(&edge.ordinal, record + sizeof(edge.post) + sizeof(edge.weight), sizeof(edge.ordinal));
        if (edge.post >= neurons_) {
            delete response;
            out_.fatal(CALL_INFO, -1,
                       "IdealSynapseSource decoded invalid post=%" PRIu32
                       " core=%" PRIu32 " row_offset=%" PRIu64 " record=%zu\n",
                       edge.post, core_id_, transaction.memory_offset,
                       transaction.memory_reads_completed + index);
        }
        transaction.row.push_back(edge);
        decoded_weight_sum_ += edge.weight;
    }
    transaction.memory_reads_completed += records;
    if (!pending_memory_trace_valid_ ||
        pending_memory_trace_.request_id != pending_memory_request_) {
        delete response;
        out_.fatal(CALL_INFO, -1, "IdealSynapseSource memory trace state is inconsistent\n");
    }
    pending_memory_trace_.completion_cycle = tick_count_;
    pending_memory_trace_.completed = true;
    appendRequestTrace_();
    pending_memory_trace_valid_ = false;
    transaction.memory_read_in_flight = false;
    if (transaction.memory_reads_completed == transaction.memory_edges && reverse_responses_) {
        std::reverse(transaction.row.begin(), transaction.row.end());
    }
    pending_memory_ = false;
    delete response;
    issueMemoryRead_();
    maybeFinish_();
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
    auto* control = dynamic_cast<CoreControlEvent*>(event);
    if (!control) {
        delete event;
        out_.fatal(CALL_INFO, -1, "IdealSynapseSource received an unexpected control event\n");
    }
    if (external_control_) {
        if (control->operation == CoreControlOp::Start) {
            if (!preload_ready_) {
                delete control;
                out_.fatal(CALL_INFO, -1, "external Start arrived before preload readiness\n");
            }
            current_timestep_ = control->timestep;
            started_ = true;
            stimuli_sent_ = false;
            seal_sent_ = false;
            ingress_ready_reported_ = false;
            if (start_cycle_ == 0) start_cycle_ = tick_count_;
        } else if (control->operation == CoreControlOp::Abort) {
            delete control;
            out_.fatal(CALL_INFO, -1, "ArtifactSynapseSourceV5 aborted by coordinator\n");
        }
    }
    delete control;
}

void IdealSynapseSource::handleSpike_(SST::Event* event) {
    auto* spike = dynamic_cast<CoreSpikeEvent*>(event);
    if (spike) {
        ++output_spikes_;
        output_spikes_by_timestep_[spike->timestep].push_back(
            static_cast<std::uint32_t>(spike->source_neuron));
        if (readout_count_ > 0 && spike->source_neuron >= readout_start_ &&
            spike->source_neuron < readout_start_ + readout_count_) {
            const auto index = static_cast<std::size_t>(spike->source_neuron - readout_start_);
            ++readout_spike_counts_[index];
            readout_spikes_by_timestep_[spike->timestep].push_back(static_cast<std::uint32_t>(index));
        }
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
    maybeFinish_();
}

void IdealSynapseSource::maybeFinish_() {
    if (finished_ || !final_commit_done_pending_ || pending_memory_ ||
        !provider_transactions_.empty()) {
        return;
    }
    finished_ = true;
    primaryComponentOKToEndSim();
}

void IdealSynapseSource::handleStatus_(SST::Event* event) {
    auto* status = dynamic_cast<CoreStatusEvent*>(event);
    if (!status) {
        delete event;
        out_.fatal(CALL_INFO, -1, "IdealSynapseSource received an unexpected status event\n");
    }
    last_status_ = *status;
    if (status->operation == CoreControlOp::CommitReady && !external_control_) {
        sendControl_(CoreControlOp::Commit, current_timestep_);
    } else if (status->operation == CoreControlOp::CommitDone) {
        ++completed_timesteps_;
        functional_hash_by_timestep_[status->timestep] = status->functional_hash;
        total_core_cycles_ += status->core_cycles;
        total_ingress_accepted_ += status->ingress_accepted;
        total_synapse_issued_ += status->synapse_issued;
        total_retire_retired_ += status->retire_retired;
        total_neurons_evaluated_ += status->neurons_evaluated;
        if (current_timestep_ + 1 < start_timestep_ + timesteps_) {
            ++current_timestep_;
            started_ = false;
            stimuli_sent_ = false;
            seal_sent_ = false;
        } else if (!finished_) {
            final_commit_done_pending_ = true;
            maybeFinish_();
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
    if (external_control_) {
        if (!preload_reported_) {
            sendControl_(CoreControlOp::PreloadReady, current_timestep_);
            preload_reported_ = true;
            return false;
        }
        if (!started_) return false;
        if (!stimuli_sent_) {
            sendStimuli_();
        } else if (!ingress_ready_reported_) {
            sendControl_(CoreControlOp::IngressReady, current_timestep_);
            ingress_ready_reported_ = true;
        }
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
        << "  \"weight_mode\": \"" << (artifact_mode_ ? "canonical_artifact" : (memory_backed_weights_ ? "memory_backed" : "ideal")) << "\",\n"
        << "  \"artifact_digest\": \"" << artifact_digest_ << "\",\n"
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
        << "  \"memory_requests\": " << memory_requests_ << ",\n"
        << "  \"memory_requests_issued\": " << memory_requests_ << ",\n"
        << "  \"memory_requests_accepted\": " << memory_requests_ << ",\n"
        << "  \"memory_requests_retried\": 0,\n"
        << "  \"weight_record_bytes\": 16,\n"
        << "  \"weight_cache_line_bytes\": " << weight_cache_line_bytes_ << ",\n"
        << "  \"weight_image_write_bytes\": " << weight_image_write_bytes_ << ",\n"
        << "  \"memory_read_granularity_bytes\": " << weight_read_granularity_bytes_ << ",\n"
        << "  \"output_spikes\": " << output_spikes_ << ",\n"
        << "  \"readout_start\": " << readout_start_ << ",\n"
        << "  \"readout_count\": " << readout_count_ << ",\n"
        << "  \"readout_spike_counts\": [";
    for (std::size_t index = 0; index < readout_spike_counts_.size(); ++index) {
        if (index != 0) out << ", ";
        out << readout_spike_counts_[index];
    }
    out << "],\n"
        << "  \"output_spikes_by_timestep\": [";
    bool first_output_timestep = true;
    for (const auto& item : output_spikes_by_timestep_) {
        auto neurons = item.second;
        std::sort(neurons.begin(), neurons.end());
        if (!first_output_timestep) out << ",";
        out << "{\"timestep\": " << item.first << ", \"core\": " << core_id_
            << ", \"local_neurons\": [";
        for (std::size_t index = 0; index < neurons.size(); ++index) {
            if (index != 0) out << ", ";
            out << neurons[index];
        }
        out << "]}";
        first_output_timestep = false;
    }
    out << "],\n"
        << "  \"spike_acks_accepted\": " << spike_acks_accepted_ << ",\n"
        << "  \"spike_retries\": " << spike_retries_ << ",\n"
        << "  \"provider_retries\": " << provider_retries_ << ",\n"
        << "  \"functional_hash\": " << functional_hash_ << ",\n"
        << "  \"functional_hash_by_timestep\": [";
    bool first_hash_timestep = true;
    for (const auto& item : functional_hash_by_timestep_) {
        if (!first_hash_timestep) out << ",";
        out << "{\"timestep\": " << item.first << ", \"core\": " << core_id_
            << ", \"functional_hash\": " << item.second << "}";
        first_hash_timestep = false;
    }
    out << "],\n"
        << "  \"completed_timesteps\": " << completed_timesteps_ << ",\n"
        << "  \"final_timestep\": " << last_status_.timestep << ",\n"
        << "  \"core_functional_hash\": " << last_status_.functional_hash << ",\n"
        << "  \"total_core_cycles\": " << total_core_cycles_ << ",\n"
        << "  \"total_ingress_accepted\": " << total_ingress_accepted_ << ",\n"
        << "  \"total_synapse_issued\": " << total_synapse_issued_ << ",\n"
        << "  \"total_retire_retired\": " << total_retire_retired_ << ",\n"
        << "  \"total_neurons_evaluated\": " << total_neurons_evaluated_ << ",\n"
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

void IdealSynapseSource::writeRequestTrace_() const {
    if (request_trace_stream_.is_open()) request_trace_stream_.flush();
}

void IdealSynapseSource::appendRequestTrace_() const {
    if (!request_trace_stream_.good() || !pending_memory_trace_valid_) return;
    const auto& trace = pending_memory_trace_;
    request_trace_stream_ << "{\"id\": \"" << trace.identity << "\", \"request_id\": " << trace.request_id
            << ", \"timestep\": " << trace.timestep
            << ", \"source_neuron\": " << trace.source_neuron
            << ", \"source_event_seq\": " << trace.source_event_seq
            << ", \"source_core\": " << core_id_
            << ", \"row_id\": " << trace.row_id
            << ", \"sequence\": " << trace.sequence
            << ", \"region_id\": \"Weights\""
            << ", \"request_class\": \"weight-read\""
            << ", \"issue_cycle_class\": \"synapse_issue\""
            << ", \"byte_address\": " << trace.byte_address
            << ", \"bytes\": " << trace.bytes
            << ", \"records\": " << trace.records
            << ", \"issue_cycle\": " << trace.issue_cycle
            << ", \"completion_cycle\": " << trace.completion_cycle
            << ", \"bank\": " << trace.bank
            << ", \"accepted\": true, \"retried\": 0"
            << ", \"completed\": " << (trace.completed ? "true" : "false")
            // The provider owns the logical request identity, but it does not
            // observe cache/controller internals. Keep those dimensions
            // explicit so a missing instrument is never interpreted as zero.
            << ", \"observations\": {\"l1\": \"unavailable\", \"l2\": \"unavailable\","
            << " \"sram_port\": \"unavailable\", \"noc\": \"unavailable\","
            << " \"mc\": \"unavailable\", \"ramulator2\": {\"channel\": \"unavailable\"," 
            << " \"bank\": \"unavailable\", \"row\": \"unavailable\"}}"
            << "}\n";
}

void IdealSynapseSource::finish() {
    if (memory_) memory_->finish();
    if (pending_memory_ || !provider_transactions_.empty()) {
        out_.fatal(CALL_INFO, -1, "IdealSynapseSource finished with outstanding memory/provider state\n");
    }
    writeEvidence_();
    writeRequestTrace_();
    out_.verbose(CALL_INFO, 1, 0, "[snndl-v5-ideal] timesteps=%" PRIu64 " rows=%" PRIu64 " responses=%" PRIu64 "\n", timesteps_, rows_served_, responses_served_);
}

} // namespace v5
} // namespace SnnDL
} // namespace SST
