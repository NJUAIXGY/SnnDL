#ifndef SST_SNN_DL_MESH_PE_2D_H
#define SST_SNN_DL_MESH_PE_2D_H

#include "events/SnnMeshEvents.h"
#include "events/TimestepControlEvent.h"
#include "platform/core/SnnCoreTile.h"
#include "snn/timestep/TimestepTracker.h"
#include "v5/api/AddressSpace.h"

#include <sst/core/clock.h>
#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <sst/core/interfaces/stdMem.h>

#include <cstdint>
#include <deque>
#include <array>
#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

namespace SST { namespace SnnDL {

class MeshPE2D final : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(
        MeshPE2D,
        "SnnDL",
        "MeshPE2D",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "GAS-free 2D mesh processing element for schema v4",
        COMPONENT_CATEGORY_PROCESSOR
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"pe_id", "PE id in row-major mesh order", "0"},
        {"rows", "Mesh rows", "1"},
        {"cols", "Mesh columns", "1"},
        {"cores_per_pe", "Number of SNN cores in this PE", "1"},
        {"neurons_per_core", "Neurons per SNN core", "1"},
        {"start_timestep", "First allowed timestep", "0"},
        {"max_timesteps", "Maximum number of synchronous timesteps", "1"},
        {"dt_ms", "Neuron integration interval", "1.0"},
        {"tau_mem_ms", "LIF membrane time constant", "20.0"},
        {"threshold", "LIF threshold", "1.0"},
        {"reset", "LIF reset value", "0.0"},
        {"refractory_timesteps", "Refractory duration", "0"},
        {"memory_bytes", "Bytes per BCSR value read", "4"},
        {"local_storage", "Use the local-storage latency path", "0"},
        {"memory_latency_cycles", "Reference memory latency for reporting", "10"},
        {"multicast", "Group endpoint packets per destination PE", "0"},
        {"use_standard_memory", "Use memHierarchy StandardMem core interfaces", "1"},
        {"descriptor_file", "Canonical v4 BCSR route/value descriptor", ""},
        {"weight_image_file", "Canonical v4 BCSR value image for local storage", ""},
        {"descriptor_digest", "Expected descriptor digest", ""},
        {"noc_link_bw", "Configured NoC link bandwidth", ""},
        {"noc_num_vns", "Number of virtual networks", "1"},
        {"noc_router_latency_cycles", "Router latency per hop", "1"},
        {"noc_queue_capacity", "Finite packets per virtual-network queue", "64"},
        {"edges", "BCSR graph edges pre:post:weight;...", ""},
        {"stimuli", "External spikes step:global_neuron;...", ""},
        {"clock", "PE clock", "1GHz"},
        {"verbose", "Verbose logging level", "0"}
    )

    SST_ELI_DOCUMENT_PORTS(
        {"control", "Bidirectional timestep control channel", {"SnnDL.TimestepControlEvent"}},
        {"memory", "Bidirectional shared-memory channel", {"SST::Event"}},
        {"north", "North 2D mesh link", {"SnnDL.MeshSpikeEvent"}},
        {"south", "South 2D mesh link", {"SnnDL.MeshSpikeEvent"}},
        {"east", "East 2D mesh link", {"SnnDL.MeshSpikeEvent"}},
        {"west", "West 2D mesh link", {"SnnDL.MeshSpikeEvent"}}
    )

    SST_ELI_DOCUMENT_SUBCOMPONENT_SLOTS(
        {"core_memory0", "StandardMem interface for core 0", "SST::Interfaces::StandardMem"},
        {"core_memory1", "StandardMem interface for core 1", "SST::Interfaces::StandardMem"},
        {"core_memory2", "StandardMem interface for core 2", "SST::Interfaces::StandardMem"},
        {"core_memory3", "StandardMem interface for core 3", "SST::Interfaces::StandardMem"},
        {"core_memory4", "StandardMem interface for core 4", "SST::Interfaces::StandardMem"},
        {"core_memory5", "StandardMem interface for core 5", "SST::Interfaces::StandardMem"},
        {"core_memory6", "StandardMem interface for core 6", "SST::Interfaces::StandardMem"},
        {"core_memory7", "StandardMem interface for core 7", "SST::Interfaces::StandardMem"}
    )

    MeshPE2D(SST::ComponentId_t id, SST::Params& params);
    ~MeshPE2D() override;

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

    struct Task {
        std::uint64_t timestep = 0;
        std::uint32_t post_local = 0;
        std::uint64_t address = 0;
        float weight = 0.0f;
        bool accepted = false;
        std::uint64_t stable_order = 0;
    };

    struct StepCounters {
        std::uint64_t logical_tx = 0;
        std::uint64_t logical_rx = 0;
        std::uint64_t physical_packets = 0;
        std::uint64_t memory_requests = 0;
        std::uint64_t memory_responses = 0;
        std::uint64_t storage_hits = 0;
        std::uint64_t synapse_tasks_created = 0;
        std::uint64_t synapse_tasks_retired = 0;
        std::uint64_t fired = 0;
        std::uint64_t neurons_committed = 0;
        std::uint64_t start_cycle = 0;
        std::uint64_t commit_cycle = 0;
        std::uint64_t state_hash = 0;
        std::uint64_t spike_hash = 0;
        std::uint64_t queue_drops = 0;
        std::uint64_t backpressure_events = 0;
        std::uint64_t stale_events = 0;
        std::uint64_t future_events = 0;
        std::uint64_t post_seal_events = 0;
        std::uint64_t tracked_tokens = 0;
        std::uint64_t queue_depth = 0;
        std::uint64_t blocked_routes = 0;
    };

    struct QueuedSpike {
        SST::Link* link = nullptr;
        MeshSpikeEvent spike;
        SST::Cycle_t ready_cycle = 0;
        std::uint32_t direction = 0;
        std::uint32_t virtual_network = 0;
    };

    struct LocalReadyTask {
        std::uint64_t request_id = 0;
        SST::Cycle_t ready_cycle = 0;
    };

    void handleControl_(SST::Event* event);
    void handleSpike_(SST::Event* event);
    void handleMemory_(SST::Event* event);
    void handleStandardMemory_(SST::Interfaces::StandardMem::Request* request);
    bool clockTick_(SST::Cycle_t cycle);

    void startTimestep_(std::uint64_t timestep, SST::Cycle_t cycle);
    void closeEgress_();
    void processIngress_();
    void serviceNoc_(SST::Cycle_t cycle);
    void serviceLocalStorage_(SST::Cycle_t cycle);
    void processSpikeAtDestination_(const MeshSpikeEvent& spike);
    void issueTask_(std::uint64_t timestep, const Edge& edge,
                    std::uint64_t source_event_seq);
    void emitSourceSpike_(const SpikeMessage& spike);
    void routeSpike_(const MeshSpikeEvent& spike);
    void sendControl_(TimestepControlOp operation);
    void maybeCommitReady_();
    void commitTimestep_(std::uint64_t timestep, SST::Cycle_t cycle);
    void retireTask_(std::uint64_t request_id, float value);

    SST::Link* nextHop_(std::uint32_t destination_pe, MeshSpikeEvent& spike);
    std::uint32_t peForNeuron_(std::uint32_t neuron) const;
    std::uint16_t coreForNeuron_(std::uint32_t neuron) const;
    std::uint32_t localNeuron_(std::uint32_t neuron) const;
    static std::vector<std::string> split_(const std::string& value, char separator);
    void parseEdges_(const std::string& encoded);
    void parseDescriptor_(const std::string& path);
    void setWeightRegionSize_(std::uint64_t element_count);
    std::uint64_t weightAddress_(const Edge& edge) const;
    void loadLocalWeightStore_();
    void parseStimuli_(const std::string& encoded);
    std::uint32_t directionForLink_(SST::Link* link) const;
    bool nocQueuesEmpty_() const;
    static std::uint64_t hashMix_(std::uint64_t hash, std::uint64_t value);
    std::uint64_t stateHash_() const;
    std::uint64_t spikeHash_(const std::vector<SpikeMessage>& spikes) const;

    SST::Output out_;
    SST::Link* control_link_ = nullptr;
    SST::Link* memory_link_ = nullptr;
    SST::Link* north_link_ = nullptr;
    SST::Link* south_link_ = nullptr;
    SST::Link* east_link_ = nullptr;
    SST::Link* west_link_ = nullptr;
    std::vector<SST::Interfaces::StandardMem*> core_memories_;

    std::uint32_t pe_id_ = 0;
    std::uint32_t rows_ = 1;
    std::uint32_t cols_ = 1;
    std::uint32_t cores_per_pe_ = 1;
    std::uint32_t neurons_per_core_ = 1;
    std::uint32_t neurons_per_pe_ = 1;
    std::uint64_t start_timestep_ = 0;
    std::uint32_t max_timesteps_ = 1;
    std::uint32_t memory_bytes_ = 4;
    std::uint32_t memory_latency_cycles_ = 10;
    std::uint32_t noc_num_vns_ = 1;
    std::uint32_t noc_router_latency_cycles_ = 1;
    std::size_t noc_queue_capacity_ = 64;
    std::string noc_link_bw_;
    std::string descriptor_file_;
    std::string weight_image_file_;
    std::string descriptor_digest_;
    ::SnnDL::v5::RegionDescriptor weight_region_{
        ::SnnDL::v5::AddressSpaceId::PeWeightSpm, 0, 0, 0, false};
    bool local_storage_ = false;
    bool multicast_ = false;
    bool use_standard_memory_ = true;
    bool boot_sent_ = false;
    bool active_ = false;
    bool egress_closed_ = false;
    bool sealed_ = false;
    bool commit_ready_sent_ = false;
    std::uint64_t active_timestep_ = 0;
    SST::Cycle_t active_start_cycle_ = 0;
    std::uint64_t next_event_request_id_ = 1;
    std::uint64_t external_event_seq_ = 0;

    std::vector<SnnCoreTile> cores_;
    std::unordered_map<std::uint32_t, std::vector<Edge>> outgoing_edges_;
    std::unordered_map<std::uint32_t, std::vector<Edge>> incoming_edges_;
    std::unordered_map<std::uint64_t, float> local_weight_store_;
    std::map<std::uint64_t, std::vector<SpikeMessage>> held_spikes_;
    std::map<std::uint64_t, std::vector<std::uint32_t>> stimuli_;
    std::deque<MeshSpikeEvent> ingress_queue_;
    std::unordered_map<std::uint64_t, Task> pending_tasks_;
    std::deque<LocalReadyTask> local_ready_tasks_;
    std::vector<std::deque<QueuedSpike>> noc_queues_;
    std::deque<QueuedSpike> blocked_routes_;
    std::array<std::size_t, 4> noc_rr_{{0, 0, 0, 0}};
    TimestepTracker tracker_;
    StepCounters counters_;
};

}} // namespace SST::SnnDL

#endif
