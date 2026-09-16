#ifndef SST_SNN_DL_V5_IDEAL_SYNAPSE_SOURCE_H
#define SST_SNN_DL_V5_IDEAL_SYNAPSE_SOURCE_H

#include "v5/events/CoreEvents.h"

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/interfaces/stdMem.h>
#include <sst/core/output.h>

#include <cstdint>
#include <deque>
#include <fstream>
#include <map>
#include <string>
#include <sstream>
#include <vector>

namespace SST {
namespace SnnDL {
namespace v5 {

class IdealSynapseSource : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(
        IdealSynapseSource,
        "SnnDL",
        "IdealSynapseSource",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "P1 deterministic ideal row/weight provider and driver",
        COMPONENT_CATEGORY_PROCESSOR
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"timesteps", "Number of synchronous timesteps", "1"},
        {"start_timestep", "First synchronous timestep", "0"},
        {"core_id", "Global Core ordinal used in functional evidence", "0"},
        {"neurons", "Neuron count used for input validation", "64"},
        {"edges", "Semicolon-separated pre:post:weight:ordinal rows", "0:0:1.0:0"},
        {"stimuli", "Semicolon-separated timestep:source[:sequence] spikes", "0:0:1"},
        {"reverse_responses", "Return row responses in reverse order", "0"},
        {"memory_backed_weights", "Read row records through StandardMem from the ChipDram hierarchy", "0"},
        {"external_control", "Receive Start/Seal/Commit from EpochCoordinatorV5", "0"},
        {"artifact_required", "Reject legacy edge-string mode", "0"},
        {"artifact_weight_file", "Validated canonical weight artifact file", ""},
        {"artifact_weight_file_offset", "Byte offset of the validated weight region", "0"},
        {"artifact_weight_bytes", "Byte length of the validated weight region", "0"},
        {"artifact_rows", "Canonical pre:byte_offset:edge_count row index", ""},
        {"artifact_stimuli", "Canonical timestep:source:sequence stimuli", ""},
        {"artifact_digest", "Canonical manifest graph digest", ""},
        {"weight_image_base", "ChipDram base address for the untimed weight image", "0"},
        {"weight_read_base", "ChipDram base address for timed row reads", "0"},
        {"weight_cache_line_bytes", "Cache-line boundary used to split timed row reads", "64"},
        {"weight_image_write_bytes", "Maximum bytes per untimed weight-image write", "64"},
        {"weight_read_granularity_bytes", "Physical StandardMem row-read granularity (multiple of the 16-byte weight record)", "16"},
        {"dram_bank_count", "Compiler logical chip-DRAM bank count for request evidence", "1"},
        {"dram_bank_interleave_bytes", "Compiler logical chip-DRAM bank interleave", "64"},
        {"dram_bank_policy", "Compiler logical chip-DRAM bank policy", "low_bits"},
        {"output_json", "Optional JSON evidence path", ""},
        {"request_trace_json", "Optional JSONL path for canonical StandardMem request identity trace", ""},
        {"readout_start", "Global first neuron in the classification readout population", "0"},
        {"readout_count", "Number of neurons in the classification readout population; zero disables readout", "0"},
        {"clock", "Driver clock", "1GHz"},
        {"verbose", "Verbose logging level", "0"}
    )

    SST_ELI_DOCUMENT_PORTS(
        {"control", "Bidirectional core control link", {"SnnDL.CoreControlEvent"}},
        {"spike_out", "Spike input link to the core", {"SnnDL.CoreSpikeEvent"}},
        {"spike_ack", "Spike ingress acknowledgement", {"SnnDL.CoreSpikeAckEvent"}},
        {"spike_in", "Held spike output from the core", {"SnnDL.CoreSpikeEvent"}},
        {"row_provider", "Bidirectional ideal row provider link", {"SnnDL.CoreRowRequestEvent", "SnnDL.CoreSynapseResponseEvent", "SnnDL.CoreRowDoneEvent", "SnnDL.CoreProviderAckEvent"}},
        {"status", "Core status input", {"SnnDL.CoreStatusEvent"}},
        {"preload", "Optional DMA preload completion input", {"SnnDL.DmaCompletionEvent"}}
    )

    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS(
        {"memory", "Optional StandardMem interface for memory-backed weights", "SST::Interfaces::StandardMem"}
    )

    IdealSynapseSource(SST::ComponentId_t id, SST::Params& params);
    ~IdealSynapseSource() override;

    void init(unsigned int phase) override;
    void setup() override;
    void finish() override;

private:
    struct Edge {
        std::uint64_t pre = 0;
        std::uint32_t post = 0;
        float weight = 0.0f;
        std::uint64_t ordinal = 0;
    };
    struct Stimulus {
        std::uint64_t timestep = 0;
        std::uint64_t source = 0;
        std::uint64_t sequence = 0;
    };

    void handleControl_(SST::Event* event);
    void handleSpike_(SST::Event* event);
    void handleSpikeAck_(SST::Event* event);
    void handleProvider_(SST::Event* event);
    void handleStatus_(SST::Event* event);
    void handlePreload_(SST::Event* event);
    void handleMemory_(SST::Interfaces::StandardMem::Request* request);
    bool clockTick_(SST::Cycle_t cycle);
    void sendControl_(CoreControlOp operation, std::uint64_t timestep);
    void sendStimuli_();
    void respondToRow_(const CoreRowRequestEvent& request);
    void issueMemoryRead_();
    void buildWeightImage_();
    void sendNextProviderItem_();
    void maybeFinish_();
    void writeEvidence_() const;
    static std::vector<std::string> split_(const std::string& value, char separator);
    void parseEdges_(const std::string& encoded);
    void parseStimuli_(const std::string& encoded);
    void parseArtifactRows_(const std::string& encoded);
    void loadArtifactWeights_(const std::string& path, std::uint64_t offset, std::uint64_t bytes);
    void writeRequestTrace_() const;
    void appendRequestTrace_() const;

    SST::Output out_;
    SST::Link* control_link_ = nullptr;
    SST::Link* spike_out_link_ = nullptr;
    SST::Link* spike_ack_link_ = nullptr;
    SST::Link* spike_in_link_ = nullptr;
    SST::Link* row_provider_link_ = nullptr;
    SST::Link* status_link_ = nullptr;
    SST::Link* preload_link_ = nullptr;
    SST::Interfaces::StandardMem* memory_ = nullptr;
    std::uint32_t neurons_ = 64;
    std::uint32_t core_id_ = 0;
    std::uint64_t timesteps_ = 1;
    std::uint64_t start_timestep_ = 0;
    bool reverse_responses_ = false;
    bool memory_backed_weights_ = false;
    bool external_control_ = false;
    bool artifact_mode_ = false;
    std::uint64_t weight_image_base_ = 0;
    std::uint64_t weight_read_base_ = 0;
    std::size_t weight_cache_line_bytes_ = 64;
    std::size_t weight_image_write_bytes_ = 64;
    std::size_t weight_read_granularity_bytes_ = 16;
    std::size_t dram_bank_count_ = 1;
    std::size_t dram_bank_interleave_bytes_ = 64;
    std::string dram_bank_policy_ = "low_bits";
    std::string output_json_;
    std::string request_trace_json_;
    std::vector<Edge> edges_;
    std::vector<Stimulus> stimuli_;
    std::map<std::uint64_t, std::vector<Edge>> edges_by_pre_;
    struct RowLocation {
        std::uint64_t byte_offset = 0;
        std::size_t edge_count = 0;
    };
    std::map<std::uint64_t, RowLocation> row_locations_;
    std::vector<std::uint8_t> weight_image_;
    struct ProviderTransaction {
        CoreRowRequestEvent request;
        std::vector<Edge> row;
        std::size_t next_edge = 0;
        bool done_sent = false;
        bool in_flight = false;
        std::uint64_t memory_offset = 0;
        std::size_t memory_edges = 0;
        std::size_t memory_reads_completed = 0;
        bool memory_read_in_flight = false;
    };
    std::deque<ProviderTransaction> provider_transactions_;
    std::size_t stimulus_cursor_ = 0;
    bool stimulus_in_flight_ = false;
    Stimulus stimulus_sent_;
    std::uint64_t current_timestep_ = 0;
    std::uint64_t tick_count_ = 0;
    bool started_ = false;
    bool stimuli_sent_ = false;
    bool seal_sent_ = false;
    bool finished_ = false;
    bool final_commit_done_pending_ = false;
    bool preload_ready_ = false;
    bool image_initialized_ = false;
    bool preload_reported_ = false;
    bool ingress_ready_reported_ = false;
    std::string artifact_digest_;
    std::uint64_t pending_memory_request_ = 0;
    std::size_t pending_memory_records_ = 0;
    bool pending_memory_ = false;
    std::uint64_t preload_ready_cycle_ = 0;
    std::uint64_t start_cycle_ = 0;
    std::uint64_t preload_wait_cycles_ = 0;
    std::uint64_t memory_reads_ = 0;
    std::uint64_t memory_requests_ = 0;
    std::uint64_t memory_read_bytes_ = 0;
    std::uint64_t next_memory_trace_sequence_ = 0;
    struct MemoryRequestTrace {
        std::string identity;
        std::uint64_t request_id = 0;
        std::uint64_t timestep = 0;
        std::uint64_t source_neuron = 0;
        std::uint64_t source_event_seq = 0;
        std::uint64_t row_id = 0;
        std::uint64_t sequence = 0;
        std::uint64_t byte_address = 0;
        std::uint64_t bytes = 0;
        std::uint64_t records = 0;
        std::uint64_t issue_cycle = 0;
        std::uint64_t completion_cycle = 0;
        std::uint64_t bank = 0;
        bool completed = false;
    };
    // At most one StandardMem request is outstanding per provider. Keep only
    // that mutable record; completed identities are streamed to JSONL so a
    // long run does not retain millions of trace objects in RAM.
    MemoryRequestTrace pending_memory_trace_;
    bool pending_memory_trace_valid_ = false;
    mutable std::ofstream request_trace_stream_;
    double image_weight_sum_ = 0.0;
    double decoded_weight_sum_ = 0.0;
    std::uint64_t output_spikes_ = 0;
    std::map<std::uint64_t, std::vector<std::uint32_t>> output_spikes_by_timestep_;
    std::uint64_t readout_start_ = 0;
    std::uint32_t readout_count_ = 0;
    std::vector<std::uint64_t> readout_spike_counts_;
    std::map<std::uint64_t, std::vector<std::uint32_t>> readout_spikes_by_timestep_;
    std::uint64_t rows_served_ = 0;
    std::uint64_t responses_served_ = 0;
    std::uint64_t response_attempts_ = 0;
    std::uint64_t spike_acks_accepted_ = 0;
    std::uint64_t spike_retries_ = 0;
    std::uint64_t provider_retries_ = 0;
    std::uint64_t functional_hash_ = 0;
    std::map<std::uint64_t, std::uint64_t> functional_hash_by_timestep_;
    std::uint64_t completed_timesteps_ = 0;
    std::uint64_t total_core_cycles_ = 0;
    std::uint64_t total_ingress_accepted_ = 0;
    std::uint64_t total_synapse_issued_ = 0;
    std::uint64_t total_retire_retired_ = 0;
    std::uint64_t total_neurons_evaluated_ = 0;
    CoreStatusEvent last_status_;
};

class ArtifactSynapseSourceV5 final : public IdealSynapseSource {
public:
    SST_ELI_REGISTER_COMPONENT(
        ArtifactSynapseSourceV5,
        "SnnDL",
        "ArtifactSynapseSourceV5",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "P6 canonical manifest-backed row, weight, and stimulus provider",
        COMPONENT_CATEGORY_PROCESSOR
    )
    ArtifactSynapseSourceV5(SST::ComponentId_t id, SST::Params& params)
        : IdealSynapseSource(id, params) {}
};

} // namespace v5
} // namespace SnnDL
} // namespace SST

#endif
