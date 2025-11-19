#include "GasPhaseController.h"
#include "SnnPESubComponent.h"
#include <fstream>
#include <sstream>

using namespace SST::SnnDL;

void GasPhaseController::beginApplyOrchestrate() {
    if (!core_) return;
    // Delegate to core minimal entry point (no behavior change in this step)
    core_->orchestrateApplyWindowEntry();
}

uint64_t GasPhaseController::getNow_() const {
    if (!core_) return 0ULL;
    return core_->getCurrentSimTimeNano();
}

void GasPhaseController::beginGatherWindowSetup() {
    if (!core_) return;
    core_->orchestrateBeginGatherWindowSetup();
}

void GasPhaseController::beginApplyIssueReads(bool strict_active) {
    if (!core_) return;
    core_->orchestrateBeginApplyIssueFallback(strict_active);
}

void GasPhaseController::continueIssueReads() {
    if (!core_) return;
    core_->orchestrateContinueIssueReads();
}

void GasPhaseController::issueFromEdges() {
    if (!core_) return;
    core_->orchestrateIssueFromEdgesDirect();
}

void GasPhaseController::beginApplyFullSequence(bool strict_active) {
    if (!core_) return;
    core_->orchestrateApplyWindowEntry();
    core_->orchestrateMarkBeginApply();
    core_->orchestrateBeginApplyIssueFallback(strict_active);
}

void GasPhaseController::onScatterFinalize(uint32_t seq,
                           uint64_t spikes_emitted,
                           uint64_t window_spikes_all,
                           uint64_t acc_updates,
                           uint64_t acc_posts_touched,
                           uint64_t acc_hwm_bytes_max,
                           uint64_t acc_spill_records,
                           uint64_t acc_spilled_bytes)
{
    scatter_seq_ = seq;
    scatter_spikes_emitted_ = spikes_emitted;
    scatter_window_spikes_all_ = window_spikes_all;
    scatter_acc_updates_ = acc_updates;
    scatter_acc_posts_touched_ = acc_posts_touched;
    scatter_acc_hwm_bytes_max_ = acc_hwm_bytes_max;
    scatter_acc_spill_records_ = acc_spill_records;
    scatter_acc_spilled_bytes_ = acc_spilled_bytes;
    if (out_ && (window_read_debug_ || out_->getVerboseLevel() >= 2)) {
        out_->verbose(CALL_INFO, 2, 0,
            "[gas-ctrl] scatter-finalize seq=%u spikes_emitted=%" PRIu64 " window_spikes_all=%" PRIu64
            " acc_updates=%" PRIu64 " posts=%" PRIu64 " hwm_max=%" PRIu64 " spill_rec=%" PRIu64 " spill_bytes=%" PRIu64 "\n",
            seq, spikes_emitted, window_spikes_all, acc_updates, acc_posts_touched,
            acc_hwm_bytes_max, acc_spill_records, acc_spilled_bytes);
    }
    // Optional CSV mirror: when stage_events_csv_ not empty, write a parallel scatter summary CSV
    if (!stage_events_csv_.empty()) {
        if (scatter_csv_path_.empty()) {
            // Derive a path by appending suffix to avoid clobbering original CSV (if any)
            scatter_csv_path_ = stage_events_csv_ + ".ctrl_scatter.csv";
        }
        std::ofstream ofs(scatter_csv_path_, std::ios::out | std::ios::app);
        if (ofs.good()) {
            if (!scatter_csv_header_written_) {
                ofs << "seq,spikes_emitted,window_spikes_all,acc_updates,acc_posts,acc_hwm_max,acc_spill_records,acc_spilled_bytes\n";
                scatter_csv_header_written_ = true;
            }
            ofs << seq << ','
                << spikes_emitted << ','
                << window_spikes_all << ','
                << acc_updates << ','
                << acc_posts_touched << ','
                << acc_hwm_bytes_max << ','
                << acc_spill_records << ','
                << acc_spilled_bytes << "\n";
        }
    }
}
