#ifndef SST_SNN_DL_TIMESTEP_COORDINATOR_H
#define SST_SNN_DL_TIMESTEP_COORDINATOR_H

#include "events/TimestepControlEvent.h"

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>

#include <cstdint>
#include <string>
#include <vector>

namespace SST { namespace SnnDL {

class TimestepCoordinator final : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(
        TimestepCoordinator,
        "SnnDL",
        "TimestepCoordinator",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "Counted-drain timestep barrier for the v4 2D SNN model",
        COMPONENT_CATEGORY_PROCESSOR
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"total_pes", "Number of participating PEs", "1"},
        {"start_timestep", "First timestep", "0"},
        {"max_timesteps", "Number of timesteps to execute", "1"},
        {"output_json", "Path for the SST-produced summary JSON", ""},
        {"mesh_rows", "Configured mesh rows", "1"},
        {"mesh_cols", "Configured mesh columns", "1"},
        {"cores_per_pe", "Configured cores per PE", "1"},
        {"neurons_per_core", "Configured neurons per core", "1"},
        {"memory_scope", "Resolved memory scope", "chip_shared"},
        {"memory_backend", "Resolved memory backend", "simple"},
        {"local_storage", "Use the PE-local BCSR value store", "0"},
        {"noc_type", "Resolved NoC type", "sst_xy_mesh_2d"},
        {"noc_link_bw", "Resolved NoC link bandwidth", ""},
        {"noc_num_vns", "Resolved NoC virtual networks", "1"},
        {"noc_router_latency_cycles", "Resolved router latency", "1"},
        {"noc_queue_capacity", "Resolved NoC queue capacity", "64"},
        {"weight_image_bytes", "Bytes in the BCSR value image", "0"},
        {"graph_digest", "BCSR graph digest", ""},
        {"route_digest", "BCSR route digest", ""},
        {"descriptor_digest", "BCSR descriptor digest", ""},
        {"verbose", "Verbose logging level", "0"}
    )

    SST_ELI_DOCUMENT_PORTS(
        {"pe%(pe)d", "Bidirectional control port for each PE", {"SnnDL.TimestepControlEvent"}}
    )

    TimestepCoordinator(SST::ComponentId_t id, SST::Params& params);
    ~TimestepCoordinator() override;

    void init(unsigned int phase) override;
    void setup() override;
    void finish() override;

private:
    struct PeStep {
        bool egress_closed = false;
        bool commit_ready = false;
        bool commit_done = false;
        std::uint64_t tx = 0;
        std::uint64_t physical = 0;
        std::uint64_t memory_requests = 0;
        std::uint64_t memory_responses = 0;
        std::uint64_t storage_hits = 0;
        std::uint64_t synapse_created = 0;
        std::uint64_t synapse_retired = 0;
        std::uint64_t fired = 0;
        std::uint64_t neurons = 0;
        std::uint64_t cycles = 0;
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

    struct Step {
        std::uint64_t logical_rx = 0;
        bool sealed = false;
        bool commit_sent = false;
        std::vector<PeStep> pe;
    };

    void handleControl_(SST::Event* event);
    void sendToAll_(TimestepControlOp operation, std::uint64_t timestep);
    void maybeSeal_(std::uint64_t timestep);
    void maybeCommit_(std::uint64_t timestep);
    void completeStep_(std::uint64_t timestep);
    void writeSummary_() const;
    static std::uint64_t hashMix_(std::uint64_t hash, std::uint64_t value);

    SST::Output out_;
    std::vector<SST::Link*> pe_links_;
    std::vector<Step> steps_;
    std::uint32_t total_pes_ = 1;
    std::uint64_t start_timestep_ = 0;
    std::uint64_t max_timesteps_ = 1;
    std::string output_json_;
    std::uint32_t mesh_rows_ = 1;
    std::uint32_t mesh_cols_ = 1;
    std::uint32_t cores_per_pe_ = 1;
    std::uint32_t neurons_per_core_ = 1;
    std::string memory_scope_ = "chip_shared";
    std::string memory_backend_ = "simple";
    bool local_storage_ = false;
    std::string noc_type_ = "sst_xy_mesh_2d";
    std::string noc_link_bw_;
    std::uint32_t noc_num_vns_ = 1;
    std::uint32_t noc_router_latency_cycles_ = 1;
    std::uint64_t noc_queue_capacity_ = 64;
    std::uint64_t weight_image_bytes_ = 0;
    std::string graph_digest_;
    std::string route_digest_;
    std::string descriptor_digest_;
    std::uint64_t boot_ready_ = 0;
    std::uint64_t current_timestep_ = 0;
    bool start_sent_ = false;
    bool finished_ = false;
};

}} // namespace SST::SnnDL

#endif
