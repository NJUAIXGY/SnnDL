#ifndef SST_SNN_DL_V5_EPOCH_COORDINATOR_V5_H
#define SST_SNN_DL_V5_EPOCH_COORDINATOR_V5_H

#include "v5/noc/NocEventsV5.h"

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>

#include <cstdint>
#include <string>
#include <vector>

namespace SST { namespace SnnDL { namespace v5 {

class EpochCoordinatorV5 final : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(EpochCoordinatorV5, "SnnDL", "EpochCoordinatorV5",
        SST_ELI_ELEMENT_VERSION(1,0,0), "P5 timed control-VN epoch barrier", COMPONENT_CATEGORY_PROCESSOR)
    SST_ELI_DOCUMENT_PARAMS(
        {"mesh_pes", "Number of participating PEs", "1"},
        {"cores_per_pe", "Participating Cores per PE", "1"},
        {"start_timestep", "First epoch", "0"},
        {"timesteps", "Number of epochs", "1"},
        {"timeout_cycles", "Maximum cycles without barrier progress", "1000000"},
        {"output_json", "Coordinator evidence path", ""},
        {"clock", "Coordinator clock", "1GHz"},
        {"verbose", "Verbose level", "0"})
    SST_ELI_DOCUMENT_PORTS(
        {"command", "Commands sent through the coordinator PE endpoint", {"SnnDL.NocControlV5Event"}},
        {"status", "Status packets returned through control VN", {"SnnDL.NocControlV5Event"}})
    SST_ELI_DOCUMENT_STATISTICS(
        {"sync.commands", "Control commands emitted", "events", 1},
        {"sync.reports", "Control reports received", "events", 1},
        {"sync.epochs_completed", "Epochs committed", "events", 1},
        {"sync.barrier_wait_ns", "Nanoseconds waiting at epoch barriers", "ns", 1},
        {"sync.timeouts", "Detected control timeouts", "events", 1})

    EpochCoordinatorV5(SST::ComponentId_t, SST::Params&);
    ~EpochCoordinatorV5() override = default;
    void finish() override;

private:
    enum class Phase : std::uint8_t { Preload, Ingress, CommitReady, CommitDone, Finished };
    void handleStatus_(SST::Event*);
    bool tick_(SST::Cycle_t);
    void resetReports_();
    void sendAll_(CoreControlOp operation, std::uint64_t epoch);
    void advance_(CoreControlOp operation);
    void writeEvidence_() const;
    bool allReports_() const;
    std::size_t participant_(const NocControlV5Event&) const;

    SST::Output out_;
    SST::Link* command_ = nullptr;
    SST::Link* status_ = nullptr;
    std::uint32_t pes_ = 1, cores_per_pe_ = 1;
    std::uint64_t start_timestep_ = 0, timesteps_ = 1, timeout_cycles_ = 1000000;
    std::uint64_t epoch_ = 0, cycle_ = 0, last_progress_cycle_ = 0, phase_start_ns_ = 0;
    std::uint64_t commands_ = 0, reports_ = 0, epochs_completed_ = 0;
    std::uint64_t barrier_wait_ns_ = 0, timeouts_ = 0, expected_data_ = 0, delivered_data_ = 0;
    Phase phase_ = Phase::Preload;
    std::vector<bool> seen_;
    std::string output_json_;
};

}}}
#endif
