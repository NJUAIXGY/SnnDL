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
        {"strict_seq_check", "fatal on unexpected seq transitions (0/1)", "0"},
        // Experimental: progress watchdog (disabled by default; debugging only).
        {"experimental_progress_enable", "enable periodic step-sync progress log (0/1)", "0"},
        {"experimental_progress_period_cycles", "progress log period in controller cycles (0=disable)", "0"},
        {"experimental_progress_max_reports", "cap number of progress reports (0=unbounded)", "0"},
        {"experimental_progress_dump_first_n", "dump first N not-done PEs in progress log", "8"},
        // Experimental: step-level global apply bank credit control.
        {"credit_ctrl_enable", "enable global apply bank credit control (0/1)", "0"},
        {"credit_ctrl_credit_min", "global credit: default apply bank credit for non-critical PEs", "1"},
        {"credit_ctrl_credit_max", "global credit: apply bank credit for selected critical PEs", "2"},
        {"credit_ctrl_top_k", "global credit: number of slowest PEs to assign credit_max each step (0=disable)", "0"},
        // Experimental (p0b): budgeted (raise + drop) redistribution around a finite base credit.
        {"credit_ctrl_mode", "global credit: 0=legacy(top-k slowest->max, else min); 1=p0b(budgeted around base_credit); 2=p0c-pred(predict next-step activation load)", "0"},
        {"credit_ctrl_base_credit", "global credit(p0b): baseline apply bank credit per PE", "0"},
        {"credit_ctrl_apply_ratio_min_permille", "global credit(p0b): only raise if apply/total >= threshold (permille)", "0"},
        {"credit_ctrl_rank_by_apply", "global credit(p0b): rank metric 1=apply_ns, 0=total_ns", "1"},
        // Experimental (p0c-pred): microbench-oriented deterministic next-step load prediction.
        {"credit_ctrl_pred_seed", "global credit(p0c): predictor seed (mirror step_activation_seed)", "0"},
        {"credit_ctrl_pred_fraction", "global credit(p0c): predictor fraction (mirror step_activation_fraction)", "0"},
        {"credit_ctrl_pred_neurons_per_pe", "global credit(p0c): predictor neurons per PE", "0"},
        {"credit_ctrl_pred_fanout", "global credit(p0c): predictor fanout (mirror step_activation_fanout)", "0"}
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
    void computeNextCredits_(uint32_t completed_seq);

    SST::Output* out_ = nullptr;
    std::vector<SST::Link*> pe_links_;
    std::vector<uint8_t> pe_ready_;
    std::vector<uint8_t> pe_done_;
    // Per-step telemetry (valid for current_seq_ when PE_DONE arrives).
    std::vector<uint64_t> pe_step_total_ns_;
    std::vector<uint64_t> pe_step_apply_ns_;
    // Per-PE credit target to be sent on next START_STEP(seq).
    std::vector<uint32_t> pe_next_apply_bank_credit_;

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

    // Experimental: progress watchdog (disabled by default).
    bool experimental_progress_enable_ = false;
    uint64_t experimental_progress_period_cycles_ = 0;
    uint32_t experimental_progress_max_reports_ = 0;
    uint32_t experimental_progress_dump_first_n_ = 8;
    uint32_t experimental_progress_reports_ = 0;

    // Experimental: global criticality-aware credit control (disabled by default).
    bool credit_ctrl_enable_ = false;
    uint32_t credit_ctrl_credit_min_ = 1;
    uint32_t credit_ctrl_credit_max_ = 2;
    uint32_t credit_ctrl_top_k_ = 0;
    // credit_ctrl_mode:
    // - 0: legacy (top-k slowest -> credit_max, else credit_min)
    // - 1: p0b (budgeted raise+drop around credit_ctrl_base_credit_)
    // - 2: p0c-pred (predict next-step activation load; rank by predicted sources_selected)
    uint32_t credit_ctrl_mode_ = 0;
    uint32_t credit_ctrl_base_credit_ = 0;
    uint32_t credit_ctrl_apply_ratio_min_permille_ = 0;
    bool credit_ctrl_rank_by_apply_ = true;
    // p0c predictor params (only used when credit_ctrl_mode_==2).
    uint64_t credit_ctrl_pred_seed_ = 0;
    double credit_ctrl_pred_fraction_ = 0.0;
    uint32_t credit_ctrl_pred_neurons_per_pe_ = 0;
    uint32_t credit_ctrl_pred_fanout_ = 0;
    // best-effort mapping from pe_index -> runtime node_id (src_node in barrier events)
    std::vector<uint32_t> pe_node_ids_;
};

}} // namespace SST::SnnDL

#endif // SNNDL_GLOBAL_GAS_STEP_CONTROLLER_H
