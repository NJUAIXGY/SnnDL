#ifndef SST_SNN_DL_V5_IDEAL_SYNAPSE_SOURCE_H
#define SST_SNN_DL_V5_IDEAL_SYNAPSE_SOURCE_H

#include "v5/events/CoreEvents.h"

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/interfaces/stdMem.h>
#include <sst/core/output.h>

#include <cstdint>
#include <deque>
#include <map>
#include <string>
#include <vector>

namespace SST {
namespace SnnDL {
namespace v5 {

class IdealSynapseSource final : public SST::Component {
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
        {"neurons", "Neuron count used for input validation", "64"},
        {"edges", "Semicolon-separated pre:post:weight:ordinal rows", "0:0:1.0:0"},
        {"stimuli", "Semicolon-separated timestep:source[:sequence] spikes", "0:0:1"},
        {"reverse_responses", "Return row responses in reverse order", "0"},
        {"memory_backed_weights", "Read row records through StandardMem after DMA preload", "0"},
        {"weight_image_base", "ChipDram base address for the untimed weight image", "0"},
        {"weight_read_base", "Local weight scratchpad base address", "0"},
        {"output_json", "Optional JSON evidence path", ""},
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
        std::uint32_t pre = 0;
        std::uint32_t post = 0;
        float weight = 0.0f;
        std::uint64_t ordinal = 0;
    };
    struct Stimulus {
        std::uint64_t timestep = 0;
        std::uint32_t source = 0;
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
    void writeEvidence_() const;
    static std::vector<std::string> split_(const std::string& value, char separator);
    void parseEdges_(const std::string& encoded);
    void parseStimuli_(const std::string& encoded);

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
    std::uint64_t timesteps_ = 1;
    bool reverse_responses_ = false;
    bool memory_backed_weights_ = false;
    std::uint64_t weight_image_base_ = 0;
    std::uint64_t weight_read_base_ = 0;
    std::string output_json_;
    std::vector<Edge> edges_;
    std::vector<Stimulus> stimuli_;
    std::map<std::uint32_t, std::vector<Edge>> edges_by_pre_;
    struct RowLocation {
        std::uint64_t byte_offset = 0;
        std::size_t edge_count = 0;
    };
    std::map<std::uint32_t, RowLocation> row_locations_;
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
    bool preload_ready_ = false;
    bool image_initialized_ = false;
    std::uint64_t pending_memory_request_ = 0;
    bool pending_memory_ = false;
    std::uint64_t preload_ready_cycle_ = 0;
    std::uint64_t start_cycle_ = 0;
    std::uint64_t preload_wait_cycles_ = 0;
    std::uint64_t memory_reads_ = 0;
    std::uint64_t memory_read_bytes_ = 0;
    double image_weight_sum_ = 0.0;
    double decoded_weight_sum_ = 0.0;
    std::uint64_t output_spikes_ = 0;
    std::uint64_t rows_served_ = 0;
    std::uint64_t responses_served_ = 0;
    std::uint64_t response_attempts_ = 0;
    std::uint64_t spike_acks_accepted_ = 0;
    std::uint64_t spike_retries_ = 0;
    std::uint64_t provider_retries_ = 0;
    std::uint64_t functional_hash_ = 0;
    CoreStatusEvent last_status_;
};

} // namespace v5
} // namespace SnnDL
} // namespace SST

#endif
