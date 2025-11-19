// Minimal skeleton for GAS phase orchestration controller.
// Step2-1: interface only, no logic change. It mirrors stage events so that
// later steps can migrate window orchestration safely.
#ifndef SST_ELEMENTS_SNNDL_GAS_PHASE_CONTROLLER_H
#define SST_ELEMENTS_SNNDL_GAS_PHASE_CONTROLLER_H

#include <cstdint>
#include <string>
#include <utility>
#include <sst/core/output.h>
#include <unordered_set>

namespace SST { namespace SnnDL {

class SnnPESubComponent; // fwd

class GasPhaseController {
public:
    GasPhaseController() = default;

    void init(SnnPESubComponent* core, Output* out) {
        core_ = core; out_ = out;
    }
    void setDebug(bool window_read_debug, bool extended_diag) {
        window_read_debug_ = window_read_debug; extended_diag_ = extended_diag;
    }
    void setStageEventsCsv(const std::string& path) { stage_events_csv_ = path; }

    // Stage events (no-op for now; keep side effects zero to preserve behavior)
    void onBeginGather(uint32_t seq) {
        seq_ = seq; have_begin_gather_ = true; t_begin_gather_ = getNow_();
        log_("BeginGather", seq, 0);
    }
    void onBeginApply(uint32_t seq) {
        log_("BeginApply", seq, 0);
        seq_ = seq; have_begin_apply_ = true; t_begin_apply_ = getNow_();
        window_reads_issued_ = 0;
    }
    void onBeginScatter(uint32_t seq) {
        log_("BeginScatter", seq, 0);
        seq_ = seq; have_begin_scatter_ = true; t_begin_scatter_ = getNow_();
    }
    void onEndScatter(uint32_t seq, uint64_t spikes_emitted) {
        log_("EndScatter", seq, spikes_emitted);
        spikes_emitted_window_ = spikes_emitted;
    }

    // Apply window orchestration entry (delegates back to core; implemented in .cc)
    void beginApplyOrchestrate();
    void beginGatherWindowSetup();
    void beginApplyIssueReads(bool strict_active);
    void continueIssueReads();
    void issueFromEdges();
    void beginApplyFullSequence(bool strict_active);
    void logEdgeFetchStart(uint64_t prev_edges, uint32_t issued, uint32_t outstanding, uint32_t budget) {
        // Mirror-only: record last snapshot; avoid duplicate logs by default.
        last_prev_edges_ = prev_edges;
        last_issued_ = issued;
        last_outstanding_ = outstanding;
        last_budget_ = budget;
        if (out_ && (window_read_debug_ && out_->getVerboseLevel() >= 3)) {
            out_->verbose(CALL_INFO, 3, 0,
                "[gas-ctrl] edge-fetch-start seq=%u prev_edges=%" PRIu64 " issued=%u outstanding=%u budget=%u\n",
                seq_, prev_edges, issued, outstanding, budget);
        }
    }

    // Mirror accumulation updates (no behavior change)
    void onWeightResolved(uint32_t post_local, float value, uint32_t count) {
        (void)count; // currently unused; reserved for detailed accounting
        mirror_acc_updates_++;
        // Track touched posts (best-effort, no capacity guarantees)
        try { posts_touched_.insert(post_local); } catch (...) {}
        if (window_read_debug_ && out_ && out_->getVerboseLevel() >= 3) {
            out_->verbose(CALL_INFO, 3, 0,
                "[gas-ctrl] acc mirror post=%u dv=%.6f seq=%u\n", post_local, (double)value, seq_);
        }
    }

    // Mirror window finalize hook
    void onWindowFinalize() {
        // Reset per-window mirrors (no behavior change)
        mirror_acc_updates_ = 0;
        try { posts_touched_.clear(); } catch (...) {}
        spikes_emitted_window_ = 0;
        window_reads_issued_ = 0;
    }

    // Stage state queries (mirror-only)
    uint32_t currentSeq() const { return seq_; }
    bool hasBeginGather() const { return have_begin_gather_; }
    bool hasBeginApply() const { return have_begin_apply_; }
    bool hasBeginScatter() const { return have_begin_scatter_; }
    uint64_t tBeginGather() const { return t_begin_gather_; }
    uint64_t tBeginApply() const { return t_begin_apply_; }
    uint64_t tBeginScatter() const { return t_begin_scatter_; }
    // windowReadsIssued()/windowReadBudget() declared earlier

    // Scatter finalize (mirror totals; no behavior change)
    void onScatterFinalize(uint32_t seq,
                           uint64_t spikes_emitted,
                           uint64_t window_spikes_all,
                           uint64_t acc_updates,
                           uint64_t acc_posts_touched,
                           uint64_t acc_hwm_bytes_max,
                           uint64_t acc_spill_records,
                           uint64_t acc_spilled_bytes);

    // Window-read accounting (mirror only; no behavior changes)
    void setWindowReadBudget(uint32_t budget) { window_read_budget_ = budget; }
    void onReadIssued() { if (window_read_budget_) ++window_reads_issued_; }
    void resetWindowReads() { window_reads_issued_ = 0; }
    uint32_t windowReadsIssued() const { return window_reads_issued_; }
    uint32_t windowReadBudget() const { return window_read_budget_; }
    bool canIssueMoreReads(uint32_t local_issued, uint32_t local_budget,
                           uint32_t outstanding, uint32_t max_outstanding) const {
        // Keep exact original behavior: both budget and outstanding must allow issuing
        const uint32_t eff_budget = window_read_budget_ ? window_read_budget_ : local_budget;
        const uint32_t eff_issued = window_reads_issued_ ? window_reads_issued_ : local_issued;
        if (eff_budget && eff_issued >= eff_budget) return false;
        if (max_outstanding && outstanding >= max_outstanding) return false;
        return true;
    }

private:
    void log_(const char* ev, uint32_t seq, uint64_t spikes) const {
        // Only lightweight diagnostic, gated strictly; no behavior change.
        if (!out_ || (!window_read_debug_ && out_->getVerboseLevel() < 1)) return;
        out_->verbose(CALL_INFO, 3, 0,
            "[gas-ctrl] core-bridge ev=%s seq=%u spikes=%" PRIu64 "\n", ev?ev:"-", seq, spikes);
    }
    uint64_t getNow_() const;

    SnnPESubComponent* core_ = nullptr;
    Output* out_ = nullptr;
    bool window_read_debug_ = false;
    bool extended_diag_ = false;
    std::string stage_events_csv_;
    // Stage mirror
    uint32_t seq_ = 0;
    bool have_begin_gather_ = false;
    bool have_begin_apply_ = false;
    bool have_begin_scatter_ = false;
    uint64_t t_begin_gather_ = 0;
    uint64_t t_begin_apply_ = 0;
    uint64_t t_begin_scatter_ = 0;
    uint64_t spikes_emitted_window_ = 0;
    // Read issuance accounting (mirror)
    uint32_t window_reads_issued_ = 0;
    uint32_t window_read_budget_ = 0;
    // Last edge-fetch start mirror (optional)
    uint64_t last_prev_edges_ = 0;
    uint32_t last_issued_ = 0;
    uint32_t last_outstanding_ = 0;
    uint32_t last_budget_ = 0;
    // Acc mirror
    uint64_t mirror_acc_updates_ = 0;
    std::unordered_set<uint32_t> posts_touched_;
    // Scatter totals mirror
    uint32_t scatter_seq_ = 0;
    uint64_t scatter_spikes_emitted_ = 0;
    uint64_t scatter_window_spikes_all_ = 0;
    uint64_t scatter_acc_updates_ = 0;
    uint64_t scatter_acc_posts_touched_ = 0;
    uint64_t scatter_acc_hwm_bytes_max_ = 0;
    uint64_t scatter_acc_spill_records_ = 0;
    uint64_t scatter_acc_spilled_bytes_ = 0;
    // Optional CSV mirror (disabled when empty). Derived from stage_events_csv_ by caller.
    std::string scatter_csv_path_;
    bool scatter_csv_header_written_ = false;
};

}} // namespace

#endif
