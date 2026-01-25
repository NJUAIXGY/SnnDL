// -*- c++ -*-
//
// GlobalGasStepController:
// - Mesh 级 Step/GAS 同步控制器（控制面）
// - 语义：所有 PE 同时开始 START_STEP(seq)，并等待所有 PE 报告 PE_DONE(seq) 后进入下一步
//

#ifndef SNNDL_GLOBAL_GAS_STEP_CONTROLLER_H
#define SNNDL_GLOBAL_GAS_STEP_CONTROLLER_H

#include <cstdint>
#include <string>
#include <vector>

#include <sst/core/component.h>
#include <sst/core/event.h>
#include <sst/core/link.h>
#include <sst/core/output.h>
#include <sst/core/params.h>

namespace SST { namespace SnnDL {

class GlobalGasStepController final : public SST::Component {
public:
    SST_ELI_REGISTER_COMPONENT(
        GlobalGasStepController,
        "SnnDL",
        "GlobalGasStepController",
        SST_ELI_ELEMENT_VERSION(1,0,0),
        "Global step-level barrier controller for GAS windows",
        COMPONENT_CATEGORY_UNCATEGORIZED
    )

    SST_ELI_DOCUMENT_PARAMS(
        {"clock", "controller clock", "1GHz"},
        {"verbose", "verbosity", "0"},
        {"start_seq", "initial step sequence id", "1"},
        {"max_steps", "stop after completing this many steps (0=unbounded)", "0"},
        {"require_all_ready", "wait for PeReady from all connected PEs (0/1)", "1"},
        {"strict_seq_check", "fatal on unexpected seq transitions (0/1)", "0"}
    )

    SST_ELI_DOCUMENT_PORTS(
        {"pe_link%(pe)d", "Links to PEs. Connect pe_link0, pe_link1, ...", {"SnnDL.GasStepBarrierEvent"}}
    )

    GlobalGasStepController(SST::ComponentId_t id, SST::Params& params);
    ~GlobalGasStepController() override;

    void init(unsigned int phase) override;
    void setup() override;
    void finish() override;

private:
    bool clockTick_(SST::Cycle_t);
    void handleBarrierEvent_(SST::Event* ev, int pe_index);
    void broadcastStart_(uint32_t seq);
    bool allReady_() const;
    bool allDone_() const;

    SST::Output* out_ = nullptr;
    std::vector<SST::Link*> pe_links_;
    std::vector<uint8_t> pe_ready_;
    std::vector<uint8_t> pe_done_;

    std::string clock_freq_ = "1GHz";
    bool clock_registered_ = false;

    uint32_t start_seq_ = 1;
    uint32_t current_seq_ = 0;
    bool started_ = false;
    uint32_t max_steps_ = 0;         // 0=unbounded
    uint32_t steps_completed_ = 0;   // count of completed steps (allDone barriers observed)
    bool primary_keepalive_ = false; // only enabled when max_steps_ > 0

    int verbose_ = 0;
    bool require_all_ready_ = true;
    bool strict_seq_check_ = false;
    uint32_t warn_count_ = 0;
};

}} // namespace SST::SnnDL

#endif // SNNDL_GLOBAL_GAS_STEP_CONTROLLER_H
