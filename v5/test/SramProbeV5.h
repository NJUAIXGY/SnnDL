#ifndef SST_SNN_DL_V5_SRAM_PROBE_V5_H
#define SST_SNN_DL_V5_SRAM_PROBE_V5_H

#include "v5/events/StorageEvents.h"

#include <sst/core/component.h>
#include <sst/core/link.h>
#include <sst/core/output.h>

#include <cstdint>
#include <deque>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace SST {
namespace SnnDL {
namespace v5 {

class SramProbeV5 final : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(
        SramProbeV5,
        "SnnDL",
        "SramProbeV5",
        SST_ELI_ELEMENT_VERSION(1, 0, 0),
        "P2 deterministic SRAM timing probe",
        COMPONENT_CATEGORY_PROCESSOR
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"output_json", "Evidence output path", ""},
        {"clock", "Probe clock", "1GHz"},
        {"verbose", "Verbose logging level", "0"}
    )

    SST_ELI_DOCUMENT_PORTS(
        {"request", "SRAM request output", {"SnnDL.SramRequestEvent"}},
        {"response", "SRAM response input", {"SnnDL.SramResponseEvent"}}
    )

    SramProbeV5(SST::ComponentId_t id, SST::Params& params);
    ~SramProbeV5() override;

    void init(unsigned int phase) override;
    void setup() override;
    void finish() override;

private:
    struct Request {
        std::uint64_t id = 0;
        std::uint64_t address = 0;
        std::vector<std::uint8_t> data;
        bool write = false;
    };

    void handleResponse_(SST::Event* event);
    bool clockTick_(SST::Cycle_t cycle);
    void sendPending_();
    void writeEvidence_() const;

    SST::Output out_;
    SST::Link* request_link_ = nullptr;
    SST::Link* response_link_ = nullptr;
    std::string output_json_;
    std::deque<Request> pending_;
    std::map<std::uint64_t, Request> in_flight_;
    std::uint64_t completed_ = 0;
    std::uint64_t retries_ = 0;
    std::uint64_t max_service_cycle_ = 0;
    std::uint64_t max_completion_cycle_ = 0;
    std::uint64_t digest_ = 0;
    std::set<std::uint64_t> completed_ids_;
    bool finished_ = false;
};

} // namespace v5
} // namespace SnnDL
} // namespace SST

#endif
