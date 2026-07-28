// -*- c++ -*-
//
// Copyright 2025 SST Contributors
//
// SnnPESubComponent.cc: SnnPE SubComponent版本实现文件
//

#include <sst/core/sst_config.h>
#include "SnnPESubComponent.h"
#include "SnnPESubComponent_impl.h"
#include "SnnPESubComponentConfig.h"
#include <fstream>
#include "snn/synapse/gas/GasCustomCmd.h"
#include "snn/synapse/gas/GasPhaseController.h"
#include "snn/synapse/gas/AccumulatorOps.h"
#include "IPeAggregation.h"
#include "IGasCreditGate.h"
#include "IGasStepGate.h"
#include "api/IPePodSharedMetadataProvider.h"
#include "api/IPeSharedCoreFabricProvider.h"
#include "api/IPeWeightObjectPlaneProvider.h"
#include "ISnnComputeCore.h"
#include "snn/synapse/stdmem/StdMemEndpoint.h"
#include "snn/synapse/weights/SnnBcsrWeightManager.h"
#include "snn/synapse/weights/WeightMemorySubsystem.h"
#include "snn/synapse/weights/WeightCacheOps.h"
#include "snn/synapse/weights/WeightAccessor.h"
#include "snn/synapse/weights/DenseWeightLayout.h"
#include "research/local_storage/PeLocalServiceObjectTable.h"
#include "research/local_storage/PeWeightObjectPlane.h"
#include "ISpikeTransport.h"
#include "NocSpikeTransport.h"
#include "NocPacketEvent.h"
#include "IMemoryAccess.h"
#include "CoreWorkloadFactory.h"
#include "ISpikeWorkload.h"
#include "ISnnSpikeCommWorkload.h"
#include "IWeightReaderAdopter.h"
#include "snn/synapse/route/SpikeCommSubsystem.h"
#include "snn/synapse/route/SynapseRouteSubsystem.h"
#include "WorkloadConfig.h"
#include "SnnDLLogging.h"
#include "components/MultiCorePE.h"
#include "workloads/riscv_snn/RiscvSnnShadowRuntimeServices.h"

#include <sstream>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <cstdlib>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <limits>
#include <vector>

using namespace SST;
using namespace SST::SnnDL;

namespace {
static std::string normalizePulseMetadataMaskToken_(std::string token) {
    for (char& ch : token) {
        ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
    }
    token.erase(
        std::remove_if(
            token.begin(),
            token.end(),
            [](unsigned char ch) {
                return ch == '_' || ch == '-' || std::isspace(ch) != 0;
            }),
        token.end());
    return token;
}

static uint32_t parsePulseMetadataObjectMask_(const std::string& raw_mask) {
    if (raw_mask.empty()) return 0u;

    uint32_t mask = 0u;
    size_t start = 0u;
    while (start <= raw_mask.size()) {
        const size_t end = raw_mask.find_first_of(",|+; ", start);
        const std::string token = normalizePulseMetadataMaskToken_(
            raw_mask.substr(start, (end == std::string::npos) ? std::string::npos : (end - start)));
        if (!token.empty()) {
            if (token == "all") {
                mask |= PeLocalServiceObjectTable::kMetadataKindMaskAll;
            } else if (token == "preband") {
                mask |= PeLocalServiceObjectTable::kMetadataKindMaskPreband;
            } else if (token == "prebase" || token == "base" || token == "premphfbase") {
                mask |= PeLocalServiceObjectTable::kMetadataKindMaskPreMphfBase;
            } else if (token == "band" || token == "premphfband") {
                mask |= PeLocalServiceObjectTable::kMetadataKindMaskPreMphfBand;
            } else if (token == "idx2" || token == "idx2row") {
                mask |= PeLocalServiceObjectTable::kMetadataKindMaskIdx2Row;
            } else if (token == "rowidx" || token == "rowindex") {
                mask |= PeLocalServiceObjectTable::kMetadataKindMaskRowIndex;
            } else if (token == "rowdescriptor") {
                mask |= PeLocalServiceObjectTable::kMetadataKindMaskRowDescriptor;
            }
        }
        if (end == std::string::npos) break;
        start = end + 1u;
    }
    return mask;
}
} // namespace

// 诊断门控改为参数化：由 enable_extended_diagnostics_ 成员控制

void SnnPESubComponent::reportStreamMemIssueThunk_(void* ctx, size_t bytes) {
    auto* core = static_cast<SnnPESubComponent*>(ctx);
    if (!core) return;
    if (core->impl_) {
        core->impl_->reportMemoryIssue(bytes, /*count_weight_read=*/false);
    }
}

void SnnPESubComponent::reportSnnMemIssueThunk_(void* ctx, size_t bytes) {
    auto* core = static_cast<SnnPESubComponent*>(ctx);
    if (!core) return;
    if (core->impl_) {
        core->impl_->reportMemoryIssue(bytes, /*count_weight_read=*/true);
    }
}

void SnnPESubComponent::reportApplyScatterThunk_(void* ctx,
                                                 uint64_t acc_updates,
                                                 uint64_t posts_touched,
                                                 uint64_t spikes_emitted,
                                                 uint64_t hwm_bytes,
                                                 uint64_t spill_records,
                                                 uint64_t spilled_bytes) {
    auto* core = static_cast<SnnPESubComponent*>(ctx);
    if (!core) return;
    if (core->impl_) {
        core->impl_->reportApplyScatter(acc_updates, posts_touched, spikes_emitted,
                                        hwm_bytes, spill_records, spilled_bytes);
    }
    if (auto* pe = core->snn_parent_observer_) {
        pe->recordStepApplyScatter(static_cast<uint32_t>(core->curr_stage_seq_),
                                   acc_updates,
                                   posts_touched,
                                   spikes_emitted,
                                   hwm_bytes,
                                   spill_records,
                                   spilled_bytes);
        pe->recordCoreStepApplyScatter(core->core_id_,
                                       static_cast<uint32_t>(core->curr_stage_seq_),
                                       acc_updates,
                                       posts_touched,
                                       spikes_emitted,
                                       hwm_bytes,
                                       spill_records,
                                       spilled_bytes);
    }
}

void SnnPESubComponent::requestGasEndGatherThunk_(void* ctx, uint32_t superstep) {
    auto* core = static_cast<SnnPESubComponent*>(ctx);
    if (!core) return;
    if (superstep == 0) return;
    if (!core->stdmem_ep_ || !core->stdmem_ep_->available()) return;
    core->stdmem_ep_->sendGasCmd(GasOp::EndGather, superstep, /*slice*/0, /*tot*/1, /*flag*/false);
}

void SnnPESubComponent::requestGasEndScatterThunk_(void* ctx, uint32_t superstep) {
    auto* core = static_cast<SnnPESubComponent*>(ctx);
    if (!core) return;
    if (superstep == 0) return;
    if (!core->stdmem_ep_ || !core->stdmem_ep_->available()) return;
    core->stdmem_ep_->sendGasCmd(GasOp::EndScatter, superstep, /*slice*/0, /*tot*/1, /*flag*/false);
}

// === 静态共享路由缓存 / 阶段事件写入锁 ===
std::mutex SnnPESubComponent::s_stage_csv_mutex_;

// === Stage event hub (Phase5.2-A1): absorbed into Impl ===
void SnnPESubComponent::Impl::markBeginGather(uint32_t seq) {
    if (!core) return;
    const bool trace_step_path =
        core->output_ &&
        core->node_id_ == 0 &&
        seq <= 2 &&
        core->core_id_ >= 15;
    if (trace_step_path) {
        core->output_->verbose(
            CALL_INFO, 2, 0,
            "[[sentinel-step-path]] node=%u core=%d seq=%u markBeginGather enter\n",
            core->node_id_, core->core_id_, seq);
    }
    if (gas_ctrl_) gas_ctrl_->onBeginGather(seq);
    if (core->use_bcsr_) {
        core->logBcsrWindowStats_("prev");
        core->resetBcsrWindowCounters_();
    }
    uint64_t now = core->getCurrentSimTimeNano();
    core->appendStageEventRow_("BeginGather", now, 0);
    if (trace_step_path) {
        core->output_->verbose(
            CALL_INFO, 2, 0,
            "[[sentinel-step-path]] node=%u core=%d seq=%u markBeginGather after_append now=%" PRIu64 "\n",
            core->node_id_, core->core_id_, seq, now);
    }
    t_begin_gather = now;
    have_begin_gather = true;
    have_begin_apply = false;
    have_begin_scatter = false;
    core->window_spikes_all_ = 0;
}

void SnnPESubComponent::Impl::markBeginApply(uint32_t seq) {
    if (!core) return;
    if (gas_ctrl_) gas_ctrl_->onBeginApply(seq);
    uint64_t now = core->getCurrentSimTimeNano();
    core->appendStageEventRow_("BeginApply", now, 0);
    t_begin_apply = now;
    have_begin_apply = true;
}

void SnnPESubComponent::Impl::markBeginScatter(uint32_t seq) {
    if (!core) return;
    if (gas_ctrl_) gas_ctrl_->onBeginScatter(seq);
    uint64_t now = core->getCurrentSimTimeNano();
    core->appendStageEventRow_("BeginScatter", now, 0);
    t_begin_scatter = now;
    have_begin_scatter = true;
}

void SnnPESubComponent::Impl::markEndScatter(uint32_t seq, uint64_t spikes_emitted) {
    if (!core) return;
    if (gas_ctrl_) gas_ctrl_->onEndScatter(seq, spikes_emitted);
    uint64_t now = core->getCurrentSimTimeNano();
    core->appendStageEventRow_("EndScatter", now, spikes_emitted);
    // Always perform end-of-window housekeeping.
    // endScatterWindow() keeps diagnostic printing internally gated by debug flags.
    if (core->weight_mem_subsystem_) {
        core->weight_mem_subsystem_->endScatterWindow(seq);
        const auto retire_obs = core->weight_mem_subsystem_->retireObservabilityStats();
        const auto pulse_obs = core->weight_mem_subsystem_->pulseAgendaObservabilityStats();
        const auto pod_align =
            core->weight_mem_subsystem_->experimentalPeInternalPodPathAlignmentStats();
        const auto& pod_rowdescriptor = pod_align.rowdescriptor;
        if (auto* pe = core->snn_parent_observer_) {
            pe->recordStepRetireStat(
                seq,
                retire_obs.global_hol_cycles_total,
                retire_obs.ready_but_blocked_edges_total,
                retire_obs.per_post_progress_total,
                retire_obs.wait_cycles_total,
                retire_obs.wait_cycles_due_to_hol_total,
                retire_obs.wait_cycles_due_to_barrier_total,
                retire_obs.wait_cycles_due_to_not_ready_total,
                retire_obs.samepost_blocked_edges_total,
                retire_obs.crosspost_blocked_edges_total,
                retire_obs.policy_loss_cycles_total,
                retire_obs.policy_loss_edges_total,
                retire_obs.shadow_per_post_recoverable_cycles_total,
                retire_obs.shadow_per_post_recoverable_edges_total,
                retire_obs.shadow_per_post_ready_posts_peak,
                retire_obs.shadow_per_post_committable_edges_peak,
                retire_obs.head_hol_cycles_dense_total,
                retire_obs.head_hol_cycles_cache_total,
                retire_obs.head_hol_cycles_miss_total,
                retire_obs.head_hol_cycles_bcsr_total,
                retire_obs.head_hol_cycles_bcsr_file_total,
                retire_obs.head_hol_cycles_gcss_total,
                retire_obs.head_blocked_edges_dense_total,
                retire_obs.head_blocked_edges_cache_total,
                retire_obs.head_blocked_edges_miss_total,
                retire_obs.head_blocked_edges_bcsr_total,
                retire_obs.head_blocked_edges_bcsr_file_total,
                retire_obs.head_blocked_edges_gcss_total,
                retire_obs.gcss_head_queued_not_issued_cycles_total,
                retire_obs.gcss_qni_head_wait_episodes_total,
                retire_obs.gcss_qni_head_wait_cycles_max,
                retire_obs.gcss_head_queued_not_issued_blocked_edges_total,
                retire_obs.gcss_head_issued_wait_resp_cycles_total,
                retire_obs.gcss_head_issued_wait_resp_blocked_edges_total,
                retire_obs.gcss_resp_ready_but_hol_cycles_total,
                retire_obs.gcss_resp_ready_but_hol_blocked_edges_total,
                retire_obs.gcss_qni_loader_not_ready_cycles_total,
                retire_obs.gcss_qni_loader_not_ready_blocked_edges_total,
                retire_obs.gcss_qni_weight_sram_stall_cycles_total,
                retire_obs.gcss_qni_weight_sram_stall_blocked_edges_total,
                retire_obs.gcss_qni_vlf_younger_ahead_cycles_total,
                retire_obs.gcss_qni_vlf_younger_ahead_blocked_edges_total,
                retire_obs.gcss_qni_vlf_younger_ahead_depth_total,
                retire_obs.gcss_qni_vlf_younger_ahead_depth_samples_total,
                retire_obs.gcss_qni_vlf_younger_ahead_depth_max,
                retire_obs.gcss_qni_issue_deferred_total,
                retire_obs.gcss_qni_pending_direct_queue_residency_cycles_total,
                retire_obs.gcss_qni_pending_direct_queue_residency_samples_total,
                retire_obs.gcss_qni_pending_direct_queue_residency_cycles_max,
                retire_obs.gcss_qni_vlf_front_inflight_full_cycles_total,
                retire_obs.gcss_qni_vlf_front_inflight_full_blocked_edges_total,
                retire_obs.gcss_qni_vlf_front_waiting_issue_cycles_total,
                retire_obs.gcss_qni_vlf_front_waiting_issue_blocked_edges_total,
                retire_obs.gcss_qni_pending_younger_ahead_cycles_total,
                retire_obs.gcss_qni_pending_younger_ahead_blocked_edges_total,
                retire_obs.gcss_qni_pending_front_inflight_full_cycles_total,
                retire_obs.gcss_qni_pending_front_inflight_full_blocked_edges_total,
                retire_obs.gcss_qni_pending_front_waiting_tick_cycles_total,
                retire_obs.gcss_qni_pending_front_waiting_tick_blocked_edges_total,
                retire_obs.gcss_qni_unknown_cycles_total,
                retire_obs.gcss_qni_unknown_blocked_edges_total,
                retire_obs.begin_apply_windows_total,
                retire_obs.begin_apply_prev_edges_total,
                retire_obs.begin_apply_outstanding_carryin_total,
                retire_obs.begin_apply_outstanding_carryin_windows_total,
                retire_obs.begin_apply_loader_not_ready_windows_total,
                retire_obs.edge_retire_registered_total,
                retire_obs.edge_retire_retired_total,
                retire_obs.end_scatter_gcss_vlf_issue_queue_residual_total,
                retire_obs.end_scatter_pending_direct_reads_residual_total,
                retire_obs.end_scatter_outstanding_residual_total,
                retire_obs.end_scatter_residual_work_windows_total,
                retire_obs.gcss_vlf_issue_prepare_total,
                retire_obs.gcss_vlf_issue_edges_total,
                retire_obs.gcss_vlf_issue_reorder_trigger_total,
                retire_obs.gcss_vlf_issue_line_groups_total,
                retire_obs.ready_queue_peak,
                retire_obs.unblock_events_total);
            pe->recordPulseAgendaObservability(
                seq,
                pulse_obs.candidates_total,
                pulse_obs.accepted_total,
                pulse_obs.rejected_total,
                pulse_obs.reject_gate_total,
                pulse_obs.correctness_ready_blocked_cycles_total,
                pulse_obs.correctness_scoreboard_occupancy_peak,
                pulse_obs.shared_service_hits_total,
                pulse_obs.shared_service_misses_total,
                pulse_obs.region_service_entries_peak,
                pulse_obs.ready_fanout_total,
                pulse_obs.rowdescriptor_ready_transition_total,
                pulse_obs.rowdescriptor_join_ready_total,
                pulse_obs.rowdescriptor_ready_join_shortcut_candidates_total,
                pulse_obs.rowdescriptor_ready_join_shortcut_taken_total,
                pulse_obs.rowdescriptor_ready_join_shortcut_late_release_taken_total,
                pulse_obs.rowdescriptor_ready_join_shortcut_deferred_live_park_total,
                pulse_obs.rowdescriptor_ready_join_shortcut_deferred_live_apply_total,
                pulse_obs.rowdescriptor_owner_form_deferred_park_total,
                pulse_obs.rowdescriptor_owner_form_deferred_activate_total,
                pulse_obs.rowdescriptor_ready_join_shortcut_blocked_not_ready_total,
                pulse_obs.rowdescriptor_ready_join_shortcut_blocked_owner_form_total,
                pulse_obs.rowdescriptor_ready_join_shortcut_blocked_join_live_total,
                pulse_obs.rowdescriptor_ready_join_shortcut_blocked_other_total,
                pulse_obs.rowdescriptor_ready_join_shortcut_release_deferred_total,
                pulse_obs.rowdescriptor_ready_join_shortcut_apply_complete_total,
                pulse_obs.rowdescriptor_ready_join_shortcut_release_forwarded_total,
                pulse_obs.rowdescriptor_ready_join_shortcut_release_missing_total,
                pulse_obs.rowdescriptor_ready_join_descriptor_elide_total,
                pulse_obs.rowdescriptor_ready_join_lines_elide_total,
                pulse_obs.rowdescriptor_owner_first_service_elide_join_live_total,
                pulse_obs.rowdescriptor_owner_first_service_elide_join_ready_total,
                pulse_obs.rowdescriptor_owner_first_service_elide_late_join_total,
                pulse_obs.actual_gate_enable_false_total,
                pulse_obs.actual_gate_window_zero_total,
                pulse_obs.actual_gate_line_too_small_total,
                pulse_obs.actual_gate_taken_total,
                pulse_obs.frontier_windows_total,
                pulse_obs.frontier_lines_exported_total,
                pulse_obs.frontier_overlap_lines_total,
                pulse_obs.frontier_overlap_peer_total,
                pulse_obs.frontier_max_exported_per_window,
                pulse_obs.prebase_lookup_owner_fill_total,
                pulse_obs.prebase_lookup_shared_hits_total,
                pulse_obs.prebase_lookup_entries_peak);
            pe->recordPulsePodRowdescriptorObservability(
                seq,
                pod_rowdescriptor.seam_owner_form_total,
                pod_rowdescriptor.seam_joiner_hit_total,
                pod_rowdescriptor.seam_owner_live_join_total,
                pod_rowdescriptor.seam_owner_ready_join_total,
                pod_rowdescriptor.seam_ready_transition_total,
                pod_rowdescriptor.seam_late_join_total,
                pod_rowdescriptor.seam_guard_total,
                pod_rowdescriptor.seam_guard_disabled_total,
                pod_rowdescriptor.seam_guard_missing_metadata_plane_total,
                pod_rowdescriptor.seam_guard_missing_owner_table_total,
                pod_rowdescriptor.seam_guard_zero_pod_count_total,
                pod_rowdescriptor.seam_guard_window_zero_total,
                pod_rowdescriptor.seam_guard_invalid_cfg_pod_total,
                pod_rowdescriptor.seam_reject_total,
                pod_rowdescriptor.seam_attempted_total,
                pod_rowdescriptor.seam_potential_private_service_elide_total,
                pod_rowdescriptor.seam_owner_first_issue_deferred_total,
                pod_rowdescriptor.seam_owner_first_private_issue_avoided_total,
                pod_rowdescriptor.seam_owner_first_service_elide_total);
            pe->recordCoreStepRetireStat(
                core->core_id_,
                seq,
                retire_obs.global_hol_cycles_total,
                retire_obs.ready_but_blocked_edges_total,
                retire_obs.per_post_progress_total,
                retire_obs.wait_cycles_total,
                retire_obs.wait_cycles_due_to_hol_total,
                retire_obs.wait_cycles_due_to_barrier_total,
                retire_obs.wait_cycles_due_to_not_ready_total,
                retire_obs.samepost_blocked_edges_total,
                retire_obs.crosspost_blocked_edges_total,
                retire_obs.policy_loss_cycles_total,
                retire_obs.policy_loss_edges_total,
                retire_obs.shadow_per_post_recoverable_cycles_total,
                retire_obs.shadow_per_post_recoverable_edges_total,
                retire_obs.shadow_per_post_ready_posts_peak,
                retire_obs.shadow_per_post_committable_edges_peak,
                retire_obs.head_hol_cycles_dense_total,
                retire_obs.head_hol_cycles_cache_total,
                retire_obs.head_hol_cycles_miss_total,
                retire_obs.head_hol_cycles_bcsr_total,
                retire_obs.head_hol_cycles_bcsr_file_total,
                retire_obs.head_hol_cycles_gcss_total,
                retire_obs.head_blocked_edges_dense_total,
                retire_obs.head_blocked_edges_cache_total,
                retire_obs.head_blocked_edges_miss_total,
                retire_obs.head_blocked_edges_bcsr_total,
                retire_obs.head_blocked_edges_bcsr_file_total,
                retire_obs.head_blocked_edges_gcss_total,
                retire_obs.gcss_head_queued_not_issued_cycles_total,
                retire_obs.gcss_qni_head_wait_episodes_total,
                retire_obs.gcss_qni_head_wait_cycles_max,
                retire_obs.gcss_head_queued_not_issued_blocked_edges_total,
                retire_obs.gcss_head_issued_wait_resp_cycles_total,
                retire_obs.gcss_head_issued_wait_resp_blocked_edges_total,
                retire_obs.gcss_resp_ready_but_hol_cycles_total,
                retire_obs.gcss_resp_ready_but_hol_blocked_edges_total,
                retire_obs.gcss_qni_loader_not_ready_cycles_total,
                retire_obs.gcss_qni_loader_not_ready_blocked_edges_total,
                retire_obs.gcss_qni_weight_sram_stall_cycles_total,
                retire_obs.gcss_qni_weight_sram_stall_blocked_edges_total,
                retire_obs.gcss_qni_vlf_younger_ahead_cycles_total,
                retire_obs.gcss_qni_vlf_younger_ahead_blocked_edges_total,
                retire_obs.gcss_qni_vlf_younger_ahead_depth_total,
                retire_obs.gcss_qni_vlf_younger_ahead_depth_samples_total,
                retire_obs.gcss_qni_vlf_younger_ahead_depth_max,
                retire_obs.gcss_qni_issue_deferred_total,
                retire_obs.gcss_qni_pending_direct_queue_residency_cycles_total,
                retire_obs.gcss_qni_pending_direct_queue_residency_samples_total,
                retire_obs.gcss_qni_pending_direct_queue_residency_cycles_max,
                retire_obs.gcss_qni_vlf_front_inflight_full_cycles_total,
                retire_obs.gcss_qni_vlf_front_inflight_full_blocked_edges_total,
                retire_obs.gcss_qni_vlf_front_waiting_issue_cycles_total,
                retire_obs.gcss_qni_vlf_front_waiting_issue_blocked_edges_total,
                retire_obs.gcss_qni_pending_younger_ahead_cycles_total,
                retire_obs.gcss_qni_pending_younger_ahead_blocked_edges_total,
                retire_obs.gcss_qni_pending_front_inflight_full_cycles_total,
                retire_obs.gcss_qni_pending_front_inflight_full_blocked_edges_total,
                retire_obs.gcss_qni_pending_front_waiting_tick_cycles_total,
                retire_obs.gcss_qni_pending_front_waiting_tick_blocked_edges_total,
                retire_obs.gcss_qni_unknown_cycles_total,
                retire_obs.gcss_qni_unknown_blocked_edges_total,
                retire_obs.begin_apply_windows_total,
                retire_obs.begin_apply_prev_edges_total,
                retire_obs.begin_apply_outstanding_carryin_total,
                retire_obs.begin_apply_outstanding_carryin_windows_total,
                retire_obs.begin_apply_loader_not_ready_windows_total,
                retire_obs.edge_retire_registered_total,
                retire_obs.edge_retire_retired_total,
                retire_obs.end_scatter_gcss_vlf_issue_queue_residual_total,
                retire_obs.end_scatter_pending_direct_reads_residual_total,
                retire_obs.end_scatter_outstanding_residual_total,
                retire_obs.end_scatter_residual_work_windows_total,
                retire_obs.gcss_vlf_issue_prepare_total,
                retire_obs.gcss_vlf_issue_edges_total,
                retire_obs.gcss_vlf_issue_reorder_trigger_total,
                retire_obs.gcss_vlf_issue_line_groups_total,
                retire_obs.ready_queue_peak,
                retire_obs.unblock_events_total);
        }
    }
    reportWindowSpikes(static_cast<uint32_t>(seq), spikes_emitted);
    core->spikes_emitted_window_ = 0;
    core->window_spikes_all_ = 0;
    if (core->stat_gas_superstep_total_cycles_) {
        if (have_begin_gather) {
            uint64_t total = (now >= t_begin_gather) ? (now - t_begin_gather) : 0ULL;
            core->stat_gas_superstep_total_cycles_->addData(total);
        }
        if (have_begin_gather && have_begin_apply && core->stat_gas_superstep_gather_cycles_) {
            uint64_t g = (t_begin_apply >= t_begin_gather) ? (t_begin_apply - t_begin_gather) : 0ULL;
            core->stat_gas_superstep_gather_cycles_->addData(g);
        }
        if (have_begin_apply && core->stat_gas_superstep_apply_cycles_) {
            uint64_t a = (t_begin_scatter >= t_begin_apply) ? (t_begin_scatter - t_begin_apply) : 0ULL;
            core->stat_gas_superstep_apply_cycles_->addData(a);
        }
        if (have_begin_scatter && core->stat_gas_superstep_scatter_cycles_) {
            uint64_t s = (now >= t_begin_scatter) ? (now - t_begin_scatter) : 0ULL;
            core->stat_gas_superstep_scatter_cycles_->addData(s);
        }
    }
    have_begin_gather = have_begin_apply = have_begin_scatter = false;
}

void SnnPESubComponent::appendStageEventRow_(const char* event_name, uint64_t now_ns, uint64_t spikes_emitted) {
    // 阶段事件上报（转发给 MultiCorePE 聚合落盘）：
    // 注意：GAS 窗口阶段事件在多核之间并不严格同步，某些窗口的 BeginApply/BeginScatter
    // 可能首先出现在非 core0 上。若只允许 core0 上报，会导致 ga/bs 缺失，从而 gather/apply/scatter
    // 统计被写成 0（p95=0）。
    // 解决：所有核心都上报阶段边界事件，由 MultiCorePE::notifyStageEvent 做 min/max 聚合收敛。
    if (event_name == nullptr) return;
    std::lock_guard<std::mutex> lock(s_stage_csv_mutex_);
    // 改为通知父 PE 统一写入阶段事件（避免多核重复与多次落盘）；同时传递本窗发放数量
    if (auto* pe = parent_pe_cached_) {
        pe->notifyStageEvent(static_cast<uint32_t>(curr_stage_seq_), std::string(event_name), now_ns, spikes_emitted, core_id_);
    }
}

void SnnPESubComponent::handleStageEventWithoutApply_(const GasOpData* op) {
    if (!op) return;
    uint64_t now = getCurrentSimTimeNano();
    switch (op->op) {
        case GasOp::BeginGather:
            curr_stage_seq_ = op->superstep;
            appendStageEventRow_("BeginGather", now, 0);
            break;
        case GasOp::BeginApply:
            appendStageEventRow_("BeginApply", now, 0);
            break;
        case GasOp::EndApply:
            appendStageEventRow_("EndApply", now, 0);
            break;
        case GasOp::BeginScatter:
            appendStageEventRow_("BeginScatter", now, 0);
            break;
        case GasOp::EndScatter:
            appendStageEventRow_("EndScatter", now, 0);
            break;
        default:
            break;
    }
}

void SnnPESubComponent::prepareEdgeWindowForApply_() {
    if (!(apply_acc_enable_ && gas_window_mode_)) return;
    if (weight_mem_subsystem_) {
        weight_mem_subsystem_->beginApplyWindow(curr_stage_seq_, window_read_debug_, output_, core_id_);
    } else {
        windowStateBegin_();
    }
    orchestrateIssueFromEdgesDirect();
}

void SnnPESubComponent::diagEdgeWeight_(const char* tag, uint32_t post_local,
                                        uint32_t pre_global, float weight,
                                        uint32_t count) {
    if (!enable_extended_diagnostics_ && !window_read_debug_) return;
    if (!output_) return;
    if (output_->getVerboseLevel() < 2) return;
    output_->verbose(CALL_INFO, 2, 0,
        "[diag-weight] %s core=%d window=%u post_local=%u pre_global=%u weight=%.6f count=%u\n",
        tag ? tag : "edge", core_id_, curr_stage_seq_, post_local, pre_global,
        (double)weight, count);
}

void SnnPESubComponent::logBcsrWindowStats_(const char* tag) {
    if (!window_read_debug_ || !use_bcsr_ || !output_) return;
    if (output_->getVerboseLevel() < 2) return;
    if (bcsr_req_edges_ == 0 && bcsr_req_wait_rowptr_ == 0 &&
        bcsr_req_block_hit_ == 0 && bcsr_req_block_miss_ == 0) {
        return;
    }
    output_->verbose(CALL_INFO, 2, 0,
        "[diag-bcsr-window] core=%d window=%u tag=%s edges=%" PRIu64
        " rowptr_wait=%" PRIu64 " hits=%" PRIu64 " miss=%" PRIu64 "\n",
        core_id_, curr_stage_seq_, tag ? tag : "-",
        bcsr_req_edges_, bcsr_req_wait_rowptr_,
        bcsr_req_block_hit_, bcsr_req_block_miss_);
}

void SnnPESubComponent::resetBcsrWindowCounters_() {
    bcsr_req_edges_ = 0;
    bcsr_req_wait_rowptr_ = 0;
    bcsr_req_block_hit_ = 0;
    bcsr_req_block_miss_ = 0;
}

void SnnPESubComponent::reserveWindowContainers_() {
    if (weight_mem_subsystem_) {
        weight_mem_subsystem_->reserveWindowContainers(num_neurons_);
    }
}

bool SnnPESubComponent::ensureLoaderReady_() {
    if (!wait_for_loader_done_) return true;
    if (loader_ready_latched_) return true;
    if (!loader_done_shared_initialized_) return true;
    if (loader_done_shared_.size() == 0) return false;
    int ready = loader_done_shared_.mutex_read(0);
    if (ready != 0) {
        loader_ready_latched_ = true;
        if (window_read_debug_ && !loader_ready_logged_ && output_ && output_->getVerboseLevel() >= 2) {
            output_->verbose(CALL_INFO, 2, 0,
                "[diag-loader] core=%d weights_ready at cycle=%" PRIu64 "\n",
                core_id_, total_cycles_);
        }
        loader_ready_logged_ = true;
        return true;
    }
    return false;
}

void SnnPESubComponent::onLoaderReady() {
    if (!wait_for_loader_done_) return;
    loader_ready_latched_ = true;
}

bool SnnPESubComponent::ensureMemoryReady_() const {
    return stdmem_ep_ && stdmem_ep_->available() && memory_ready_;
}

void SnnPESubComponent::issueEdgeWeightFetches_() {
    const size_t prev_edges = weight_mem_subsystem_ ? weight_mem_subsystem_->edgesPrevSize() : 0;
    if (window_read_debug_ && output_ && output_->getVerboseLevel() >= 2) {
        output_->verbose(CALL_INFO, 2, 0,
            "[diag-edge-fetch] core=%d stage=%d prev_edges=%zu issued=%u outstanding=%u budget=%u\n",
            core_id_, static_cast<int>(gas_stage_), prev_edges,
            windowStateIssued_(), windowStateOutstanding_(), window_read_budget_);
    }
    issueFromEdges_();
}

void SnnPESubComponent::recordEdge_(uint32_t post_local, uint32_t pre_global) {
    // Correctness note:
    // Recording edges is a pure "gather" operation and must not depend on memory readiness.
    // Memory/loader readiness is enforced later at issue time (WMS.ensure_loader_ready / memory_warmup).
    if (!enable_weight_fetch_) {
        diag_edges_cond_skips_++;
        return;
    }
    bool stage_ok = false;
    switch (gas_stage_) {
        case GasStage::Gather:
            stage_ok = true;
            break;
        case GasStage::Apply:
            stage_ok = record_edge_apply_enable_;
            break;
        case GasStage::Idle:
            stage_ok = record_edge_idle_enable_;
            break;
        case GasStage::Scatter:
            stage_ok = record_edge_scatter_enable_;
            break;
        default:
            stage_ok = false;
            break;
    }
    if (!(apply_acc_enable_ && gas_window_mode_ && stage_ok)) {
        diag_edges_stage_skips_++;
        if (window_read_debug_ && !record_edge_stage_warned_ && output_ && output_->getVerboseLevel() >= 2) {
            output_->verbose(CALL_INFO, 2, 0,
                "[diag-edges] recordEdge skipped apply_acc=%d gas_window=%d stage=%d (apply_en=%d idle_en=%d scatter_en=%d)\n",
                apply_acc_enable_ ? 1 : 0, gas_window_mode_ ? 1 : 0, static_cast<int>(gas_stage_),
                record_edge_apply_enable_ ? 1 : 0, record_edge_idle_enable_ ? 1 : 0,
                record_edge_scatter_enable_ ? 1 : 0);
            record_edge_stage_warned_ = true;
        }
        return;
    }
    if (!weight_mem_subsystem_) {
        diag_edges_cond_skips_++;
        return;
    }
    // 容量保护：极端情况下避免map无限增长
    const size_t curr_edges = weight_mem_subsystem_ ? weight_mem_subsystem_->edgesCurrSize() : 0;
    if (curr_edges >= edge_collector_max_capacity_) {
        if (!record_edge_capacity_warned_) {
            if (output_ && output_->getVerboseLevel() >= 2) {
                output_->verbose(CALL_INFO, 2, 0,
                    "[diag-edges] ⚠️ edge_collector capacity reached (core=%d seq=%u cap=%zu), stop recording this window\n",
                    core_id_, curr_stage_seq_, edge_collector_max_capacity_);
            }
            record_edge_capacity_warned_ = true;
            if (stat_gas_edge_overflow_) stat_gas_edge_overflow_->addData(1);
        }
        return;
    }
    if (weight_mem_subsystem_) {
        weight_mem_subsystem_->recordEdge(post_local, pre_global);
    }
    diag_edges_record_hits_++;
    const size_t after_edges = weight_mem_subsystem_ ? weight_mem_subsystem_->edgesCurrSize() : 0;
    if (window_read_debug_ && after_edges <= 5 && output_ && output_->getVerboseLevel() >= 2) {
        output_->verbose(CALL_INFO, 2, 0,
            "[diag-edges] recordEdge sample core=%d seq=%u post_local=%u pre_global=%u size_now=%zu\n",
            core_id_, curr_stage_seq_, post_local, pre_global, after_edges);
    }
}

SnnPESubComponent::SnnPESubComponent(ComponentId_t id, Params& params)
    : CoreShellAPI(id, params),
      output_(nullptr),
      memory_link_(nullptr) {
    const SnnPESubComponentConfig cfg = parseSnnPESubComponentConfig(params);

    // 构造期最早哨兵（P2：参数优先，未设置回退到环境变量，以保持兼容）。
    // 避免直接使用 stdout，统一走 SST Output。
    const bool kSentinelOn = cfg.sentinel_enable;
    // 提前构建一个最低等级的输出对象，避免后续早期初始化路径使用 output_ 时发生空指针
    // 真实 verbose 等级稍后在解析完参数后再生效（此处仅用于早期诊断与防护）
    if (!output_) {
        try {
            output_ = new Output("SnnPESubComponent[@p:@l]: ", /*verbose*/0, 0, Output::STDOUT);
        } catch (...) {
            output_ = nullptr; // 最小化风险，保持后续分支都做空指针判定
        }
    }
    // 注意：其余子模块初始化挪到参数解析之后，避免早期未初始化成员被使用
    
    // 读取配置参数
    core_id_ = cfg.core_id;
    // Phase5.2：将实现类型改为指针持有，避免在 control 头文件中包含 synapse 实现头
    if (!stdmem_ep_) stdmem_ep_ = std::make_unique<StdMemEndpoint>();
    if (!bcsr_weights_) bcsr_weights_ = std::make_unique<BcsrWeightManager>();
    if (kSentinelOn && output_) {
        SNNDL_LOG(2, "[[sentinel-core-ctor]] core_ctor enter\n");
        SNNDL_LOG(2, "[[sentinel-core-ctor]] after params: core_id=%d\n", core_id_);
    }
    total_cores_ = cfg.total_cores;
    global_neuron_base_ = cfg.global_neuron_base;
    num_neurons_ = cfg.num_neurons;
    base_addr_ = cfg.base_addr;
    node_id_ = cfg.node_id;
    verbose_ = cfg.verbose;
    enable_extended_diagnostics_ = cfg.enable_extended_diagnostics;
    total_nodes_cfg_ = cfg.total_nodes;
    if (output_) {
        output_->setVerboseLevel(verbose_ < 0 ? 0u : static_cast<uint32_t>(verbose_));
    }

    // Phase6：workload 选择（优先 Params，其次环境变量）
    // 约定：默认 "snn" 仍走 legacy 主链路；仅当显式选择 stream/traffic 时才启用 workload_。
    const WorkloadKind workload_kind = workloadKindFromString(cfg.workload_impl);
    switch (workload_kind) {
        case WorkloadKind::RiscvSnn:
            workload_impl_ = WorkloadImpl::RiscvSnn;
            break;
        case WorkloadKind::Stream:
            workload_impl_ = WorkloadImpl::Stream;
            break;
        case WorkloadKind::Traffic:
            workload_impl_ = WorkloadImpl::Traffic;
            break;
        case WorkloadKind::TrafficMem:
            workload_impl_ = WorkloadImpl::TrafficMem;
            break;
        case WorkloadKind::Tensor:
            workload_impl_ = WorkloadImpl::Tensor;
            break;
        case WorkloadKind::Snn:
        default:
            workload_impl_ = WorkloadImpl::Snn;
            break;
    }
    // Phase6：通过工厂创建 workload（snn/stream/traffic）
    {
        const char* name = workloadKindName(workload_kind);
        if (!workload_) workload_ = createWorkloadByName(std::string(name));
        if (!workload_) {
            if (output_) {
                output_->fatal(CALL_INFO, -1, "❌ createWorkloadByName(\"%s\") returned nullptr\n", name);
            }
            return;
        }
        workload_->configureFromParams(params);
        spike_workload_ = dynamic_cast<ISpikeWorkload*>(workload_.get());
        snn_comm_workload_ = dynamic_cast<ISnnSpikeCommWorkload*>(workload_.get());
        gas_stage_workload_ = dynamic_cast<IGasStageSink*>(workload_.get());
    }
    if (workload_kind == WorkloadKind::RiscvSnn &&
        params.find<std::string>("riscv_snn_backend_name", "null") == "runtime_bridge") {
        if (!riscv_snn_runtime_bridge_) {
            riscv_snn_runtime_bridge_ = std::make_unique<RiscvSnnShadowRuntimeServices>();
            riscv_snn_runtime_bridge_->configureFromParams(params);
        }
        accel_runtime_services_ = riscv_snn_runtime_bridge_.get();
    }
    if (kSentinelOn && output_) {
        SNNDL_LOG(2, "[[sentinel-core-ctor]] after params2: node_id=%u num_neurons=%u base_addr=%" PRIu64 "\n",
                node_id_, num_neurons_, (uint64_t)base_addr_);
    }
    enable_weight_fetch_ = cfg.enable_weight_fetch;
    workload_spike_input_enable_ = cfg.workload_spike_input_enable;
    // Phase5.2-A1：StageEventHub 吸收到 Impl（不再单独编译 platform/core/StageEventHub.*）
    if (!impl_) impl_ = std::make_unique<Impl>();
    impl_->init(this);
    impl_->gas_ctrl_ = std::make_unique<GasPhaseController>();
    if (impl_->gas_ctrl_) {
        impl_->gas_ctrl_->init(this, output_);
    }
#if SNNDL_DEBUG_ENABLED
    if (window_read_debug_) {
        SNNDL_DEBUG_LOG(1, "[diag-init] core=%d enable_weight_fetch=%d\n", core_id_, enable_weight_fetch_ ? 1 : 0);
    }
#endif
    write_weights_on_init_ = cfg.write_weights_on_init;
    memory_warmup_cycles_ = cfg.memory_warmup_cycles;
    init_default_weight_ = cfg.init_default_weight;
    readresp_zero_fallback_ = cfg.readresp_zero_fallback;
    max_outstanding_requests_ = cfg.max_outstanding_requests;
    const uint32_t max_cache_entries = cfg.max_cache_entries;
    const bool use_clock_weight_cache = cfg.use_clock_weight_cache;
    const bool disable_weight_cache = cfg.disable_weight_cache;
    {
        WeightCacheOps::Config cache_cfg{};
        cache_cfg.max_entries = max_cache_entries;
        cache_cfg.use_clock = use_clock_weight_cache;
        cache_cfg.disable_cache = disable_weight_cache;
        if (!weight_cache_ops_) weight_cache_ops_ = std::make_unique<WeightCacheOps>();
        weight_cache_ops_->configure(cache_cfg, [this](uint64_t) {
            if (stat_cache_evictions_) stat_cache_evictions_->addData(1);
            count_cache_evictions_++;
        });
        weight_cache_ops_->reserve(cache_cfg.max_entries ? cache_cfg.max_entries : 1);
    }
    use_event_weight_fallback_ = cfg.use_event_weight_fallback;
    event_weight_fallback_warned_ = false;
    route_summary_enable_ = cfg.route_summary_enable;
    merge_read_cacheline_ = cfg.merge_read_cacheline;
    merge_read_row_ = cfg.merge_read_row;
    gas_enable_ = cfg.gas_enable; // 默认关闭
    gas_window_mode_ = cfg.gas_window_mode; // 当为true时采用GatherBufferIF的window驱动
    // 已弃用：保留参数解析以兼容旧配置，但运行期强制忽略
    bool manual_drive_param = cfg.gas_manual_window_drive_requested;
    if (manual_drive_param && output_ && output_->getVerboseLevel() >= 2) {
        output_->verbose(CALL_INFO, 2, 0,
            "[diag-gas-config] core=%d gas_manual_window_drive 已弃用，仍将使用自动窗口驱动\n",
            core_id_);
    }
    merge_read_auto_ = cfg.merge_read_auto; // default off
    line_size_bytes_ = cfg.line_size_bytes;
    byte_exact_verify_enable_ = cfg.byte_exact_verify_enable;
    byte_exact_verify_mode_ = cfg.byte_exact_verify_mode;
    byte_exact_verify_row_scale_ = cfg.byte_exact_verify_row_scale;
    byte_exact_verify_max_mismatch_ = cfg.byte_exact_verify_max_mismatch;
    bcsr_semantic_verify_enable_ = cfg.bcsr_semantic_verify_enable;
    bcsr_semantic_verify_max_edges_ = cfg.bcsr_semantic_verify_max_edges;
    bcsr_semantic_verify_max_mismatch_ = cfg.bcsr_semantic_verify_max_mismatch;
    bcsr_semantic_verify_abs_tol_ = cfg.bcsr_semantic_verify_abs_tol;
    bcsr_semantic_verify_rel_tol_ = cfg.bcsr_semantic_verify_rel_tol;
    loader_done_key_ = cfg.loader_done_key;
    wait_for_loader_done_ = !loader_done_key_.empty();
    if (wait_for_loader_done_) {
        loader_done_shared_.initialize(loader_done_key_, 1, 0);
        loader_done_shared_initialized_ = true;
        if (window_read_debug_ && output_ && output_->getVerboseLevel() >= 2) {
            output_->verbose(CALL_INFO, 2, 0,
                "[diag-loader] core=%d init loader_done_key=%s\n",
                core_id_, loader_done_key_.c_str());
        }
    }
    window_read_enable_ = cfg.window_read_enable;
    window_read_debug_ = cfg.window_read_debug;
    scatter_diag_limit_ = cfg.scatter_diag_limit;
    scatter_diag_count_ = 0;
    if (impl_ && impl_->gas_ctrl_) impl_->gas_ctrl_->setDebug(window_read_debug_, enable_extended_diagnostics_);
    window_read_budget_ = cfg.window_read_budget;
    windowStateConfigure_();
    read_force_single_ = cfg.read_force_single;
    // 边集合容量上限（极端保护）
    edge_collector_max_capacity_ = static_cast<size_t>(cfg.edge_collector_max_capacity);
    if (window_read_enable_) {
        reserveWindowContainers_();
    if (!record_edge_idle_enable_ && !record_edge_scatter_enable_ && window_read_debug_ &&
        output_ && enable_extended_diagnostics_ && output_->getVerboseLevel() >= 2) {
            output_->verbose(CALL_INFO, 2, 0,
                "[diag-record-edge] core=%d 仅在Gather阶段记录边 (Apply/Idle/Scatter=0)", core_id_);
        }
    } else if (window_read_debug_ && output_ && enable_extended_diagnostics_ && output_->getVerboseLevel() >= 2) {
        output_->verbose(CALL_INFO, 2, 0,
            "[diag-record-edge] core=%d window_read_enable=0 => 忽略 window_read_debug", core_id_);
    }
    // 全网读取扩展参数
    weights_cols_ = cfg.weights_cols;
    synapse_weight_mode_ = cfg.synapse_weight_mode;
    const bool gcss_valueonly_mode =
        (synapse_weight_mode_ == "gcss_valueonly_dstcore") ||
        (synapse_weight_mode_ == "gcss_valueonly_dstcore_idx2") ||
        (synapse_weight_mode_ == "gcss_idx2_rowmphf") ||
        (synapse_weight_mode_ == "gcss_valueonly_dstcore_vlf_premphf") ||
        (synapse_weight_mode_ == "gcss_valueonly_dstcore_vlf_premphf_plp");
    gcss_index_template_ = cfg.gcss_index_template;
    std::string index_mode_str = cfg.index_mode;
    // 神经元状态布局参数已下沉到 compute core（控制层不再持有）
    verify_routing_weights_ = cfg.verify_routing_weights;
    use_post_row_pre_col_ = (index_mode_str == "post_row_pre_col");
    if (index_mode_str == "bcsr_post_row") {
        use_bcsr_ = true;
        use_post_row_pre_col_ = true;
    } else if (index_mode_str == "csr_post_row") {
        // CSR 模式已弃用：统一禁用
        use_post_row_pre_col_ = true;
        SNNDL_DEBUG_LOG(1, "[CSR] 索引模式已禁用，改用密集/BCSR读取\n");
    }
    if (gcss_valueonly_mode) {
        use_bcsr_ = false;
    }
    if (weights_cols_ == 0) weights_cols_ = num_neurons_; // 默认沿用旧行宽
    // 配置权重索引解析器（独立于控制层实现）
    if (!weight_accessor_) weight_accessor_ = std::make_unique<WeightAccessor>();
    weight_accessor_->configure(WeightAccessorConfig{
        static_cast<uint32_t>(core_id_),
        static_cast<uint64_t>(global_neuron_base_),
        static_cast<uint32_t>(num_neurons_),
        static_cast<uint32_t>(weights_cols_),
        use_post_row_pre_col_
    });
    // Dense 权重“物理布局”（实验性；默认 row_major）
    dense_layout_mode_ = toLowerCopy(cfg.dense_layout_mode);
    if (dense_layout_mode_.empty()) dense_layout_mode_ = "row_major";
    dense_phys_dram_row_bytes_ = cfg.dense_phys_dram_row_bytes;
    dense_phys_enable_ = (dense_layout_mode_ == "phys_v1");
    dense_phys_row_stride_bytes_ = 0;
    dense_phys_rows_per_dram_row_ = 1;
    dense_phys_group_stride_bytes_ = 0;

    if (dense_phys_enable_) {
        const uint32_t cols = use_post_row_pre_col_ ? weights_cols_ : num_neurons_;
        DensePhysV1Derived d{};
        const bool ok = computeDensePhysV1Derived(num_neurons_, cols, line_size_bytes_, dense_phys_dram_row_bytes_, d);
        if (!ok) {
            if (output_) {
                output_->fatal(CALL_INFO, -1,
                               "SnnPESubComponent fatal: invalid dense phys_v1 layout rows=%u cols=%u line=%u row_bytes=%u\n",
                               num_neurons_, cols, line_size_bytes_, dense_phys_dram_row_bytes_);
            }
            std::abort();
        }
        if (dense_phys_dram_row_bytes_ != 0 &&
            (base_addr_ % static_cast<uint64_t>(dense_phys_dram_row_bytes_)) != 0) {
            if (output_) {
                output_->fatal(CALL_INFO, -1,
                               "SnnPESubComponent fatal: dense phys_v1 requires base_addr aligned to dram_row_bytes "
                               "(base=0x%llx row_bytes=%u)\n",
                               (unsigned long long)base_addr_,
                               dense_phys_dram_row_bytes_);
            }
            std::abort();
        }
        dense_phys_row_stride_bytes_ = d.row_stride_bytes;
        dense_phys_rows_per_dram_row_ = d.rows_per_dram_row;
        dense_phys_group_stride_bytes_ = d.group_stride_bytes;
        weight_region_end_ = base_addr_ + d.total_bytes;
    } else {
        // 预计算dense权重区域上界（按行*列*4B）
        uint64_t bytes = static_cast<uint64_t>(num_neurons_) * static_cast<uint64_t>(weights_cols_) *
                         static_cast<uint64_t>(sizeof(float));
        weight_region_end_ = base_addr_ + bytes;
    }
    enable_detailed_map_log_ = cfg.enable_detailed_map_log;
    log_weight_details_ = cfg.log_weight_details;
    loader_barrier_cycles_ = cfg.loader_barrier_cycles;
    // BCSR 参数
    bcsr_layout_.rows = num_neurons_;
    bcsr_layout_.cols = cfg.bcsr_cols;
    bcsr_layout_.block_rows = cfg.bcsr_block_rows;
    bcsr_layout_.block_cols = cfg.bcsr_block_cols;
    bcsr_layout_.idx_bytes = cfg.bcsr_idx_bytes;
    bcsr_layout_.val_bytes = cfg.bcsr_val_bytes;
    bcsr_layout_.rowptr_offset = cfg.bcsr_rowptr_offset;
    bcsr_layout_.colidx_offset = cfg.bcsr_colidx_offset;
    bcsr_layout_.blockdata_offset = cfg.bcsr_blockdata_offset;
    bcsr_layout_.blockids_offset = cfg.bcsr_blockids_offset;
    bcsr_layout_.layout_mode = cfg.bcsr_layout_mode;
    bcsr_layout_.colidx_row_stride_bytes = cfg.bcsr_colidx_row_stride_bytes;
    bcsr_layout_.blockdata_row_stride_bytes = cfg.bcsr_blockdata_row_stride_bytes;
    bcsr_layout_.blockids_row_stride_bytes = cfg.bcsr_blockids_row_stride_bytes;
    bcsr_layout_.per_core_stride = cfg.per_core_stride;
    bcsr_layout_.validate(base_addr_, output_, (window_read_debug_ || enable_extended_diagnostics_), core_id_, node_id_);
    bcsr_br_ = bcsr_layout_.block_rows;
    bcsr_bc_ = bcsr_layout_.block_cols;
    bcsr_val_bytes_ = bcsr_layout_.val_bytes;
    bcsr_idx_bytes_ = bcsr_layout_.idx_bytes;
    uint64_t bcsr_rowptr_addr = base_addr_ + bcsr_layout_.rowptr_offset;
    bcsr_colidx_addr_ = base_addr_ + bcsr_layout_.colidx_offset;
    bcsr_blockdata_addr_ = base_addr_ + bcsr_layout_.blockdata_offset;
    bcsr_blockids_addr_ = bcsr_layout_.blockids_offset ? base_addr_ + bcsr_layout_.blockids_offset : 0;
    bcsr_weights_->configure(
        bcsr_rowptr_addr,
        bcsr_colidx_addr_,
        bcsr_blockdata_addr_,
        bcsr_blockids_addr_,
        bcsr_br_, bcsr_bc_, bcsr_idx_bytes_, bcsr_val_bytes_,
        bcsr_layout_.layout_mode,
        bcsr_layout_.colidx_row_stride_bytes,
        bcsr_layout_.blockdata_row_stride_bytes,
        bcsr_layout_.blockids_row_stride_bytes);
    // Revert: 默认值恢复为 64/256，避免影响发放路径；如需禁用由脚本显式传入 0
    bcsr_row_index_cache_cap_ = cfg.bcsr_row_index_cache_cap;
    bcsr_block_cache_cap_ = cfg.bcsr_block_cache_cap;
    // GAS Apply/Scatter Phase‑1
    apply_acc_enable_ = cfg.apply_acc_enable;
    acc_hwm_bytes_cfg_ = cfg.acc_high_watermark_bytes;
    acc_spill_enable_cfg_ = cfg.acc_spill_enable;
    stage_events_csv_ = cfg.stage_events_csv;
    if (impl_ && impl_->gas_ctrl_) impl_->gas_ctrl_->setStageEventsCsv(stage_events_csv_);
    // aosoa_block_rows 默认推导已转移至 compute core
    // CSR 参数已移除
    bcsr_prefetch_all_ = cfg.bcsr_prefetch_all;
    // 权重验证开关（具体采样/文件配置由 compute core 解析并执行）
    verify_weights_ = cfg.verify_weights;
    bcsr_force_file_read_ = cfg.bcsr_force_file_read;
    bcsr_rowptr_file_fallback_enable_ = cfg.bcsr_rowptr_file_fallback_enable;
    quiet_finish_logs_ = cfg.quiet_finish_logs;
    record_edge_apply_enable_ = cfg.record_edge_apply_enable;
    record_edge_idle_enable_ = cfg.record_edge_idle_enable;
    if (record_edge_idle_enable_ && output_ && enable_extended_diagnostics_ && output_->getVerboseLevel() >= 2) {
        output_->verbose(CALL_INFO, 2, 0,
            "[diag-gas-config] core=%d 已启用 record_edge_idle（诊断配置，回归默认关闭）\n",
            core_id_);
    }
    record_edge_scatter_enable_ = cfg.record_edge_scatter_enable;

    // ---- Profiling init (optional; minimal overhead when disabled) ----
#ifdef SNNDL_ENABLE_PROFILING
    profiler_enabled_ = cfg.profiler_enable;
    profiler_csv_prefix_ = cfg.profiler_csv_prefix;
    if (profiler_enabled_) {
        try {
            profiler_ = new SST::SnnDL::Profiler(std::string("SnnPESubComponent_Core") + std::to_string(core_id_));
        } catch (...) {
            profiler_ = nullptr;
            profiler_enabled_ = false;
        }
    }
#endif
    // 默认启用致密累加器（与头文件参数表一致）
    acc_dense_enable_cfg_ = cfg.apply_dense_acc_enable;
    acc_shadow_verify_enable_cfg_ = cfg.acc_shadow_verify_enable;
    // 构建窗口累加器（所有状态收敛到 AccumulatorOps）
    {
        AccumulatorOpsConfig acc_cfg{};
        acc_cfg.num_neurons = num_neurons_;
        acc_cfg.dense_enable = acc_dense_enable_cfg_;
        acc_cfg.spill_enable = acc_spill_enable_cfg_;
        acc_cfg.high_watermark_bytes = acc_hwm_bytes_cfg_;
        acc_cfg.shadow_verify_enable = acc_shadow_verify_enable_cfg_;
        acc_cfg.window_read_debug = window_read_debug_;
        acc_cfg.core_id = core_id_;
        acc_cfg.verbose = verbose_;
        acc_cfg.out = output_;
        acc_cfg.updates_count = &acc_updates_count_;
        acc_cfg.posts_touched_count = &acc_posts_touched_count_;
        acc_cfg.spill_records_count = &acc_spill_records_count_;
        acc_cfg.spilled_bytes_sum = &acc_spilled_bytes_sum_;
        acc_cfg.hwm_bytes_max = &acc_hwm_bytes_max_;
        // Stats pointers will be attached in initializeStatistics().
        acc_ops_ = std::make_unique<AccumulatorOps>(acc_cfg);
    }

    // Weights template is used by both synapse/weights (BCSR) and synapse/route; keep cached here.
    weights_template_ = cfg.weights_template;
    if (gcss_valueonly_mode &&
        gcss_index_template_.empty() &&
        output_ && output_->getVerboseLevel() >= 1) {
        output_->verbose(CALL_INFO, 1, 0,
            "[gcss] warning: synapse_weight_mode=%s but gcss_index_template is empty (core=%d node=%u)\n",
            synapse_weight_mode_.c_str(), core_id_, node_id_);
    }

    // neurons_per_pe：默认按 total_cores*num_neurons 推导；但允许脚本显式覆盖（保持兼容）
    uint32_t np_from_params = cfg.neurons_per_pe;
    uint32_t computed_neurons_per_pe = static_cast<uint32_t>(total_cores_) * static_cast<uint32_t>(num_neurons_);
    if (np_from_params > 0) {
        neurons_per_pe_cfg_ = np_from_params;
        if (np_from_params != computed_neurons_per_pe && output_ && output_->getVerboseLevel() >= 2) {
            output_->verbose(CALL_INFO, 2, 0,
                "[diag-gas-config] core=%d neurons_per_pe=%u (脚本指定) ≠ cores*rows=%u\n",
                core_id_, np_from_params, computed_neurons_per_pe);
        }
    } else {
        neurons_per_pe_cfg_ = computed_neurons_per_pe;
    }

    // 获取权重文件路径（由 WeightLoader 负责加载；控制层仅缓存以便日志/兼容）
    weights_file_path_ = cfg.weights_file;

    if (!isNonSnnWorkload_()) {
        // Phase4 Task6.1：compute core 创建/配置下沉到 workload=snn；
        // 控制层此处仅构建 synapse/weights 的 weight reader 子系统。
        configureWeightReaderSubsystem_(params);

    // 初始化输出对象（若前面已创建则不重复）
    if (!output_) {
        output_ = new Output("SnnPESubComponent[@p:@l]: ", verbose_, 0, Output::STDOUT);
    }
    if (window_read_debug_ && output_ && output_->getVerboseLevel() >= 2) {
        output_->verbose(CALL_INFO, 2, 0,
            "[diag-bcsr-base] node=%u core=%d base=0x%llx rowptr=0x%llx colidx=0x%llx blockdata=0x%llx blockids=0x%llx\n",
            node_id_, core_id_,
            (unsigned long long)base_addr_,
            (unsigned long long)bcsr_weights_->rowptrAddr(),
            (unsigned long long)bcsr_colidx_addr_,
            (unsigned long long)bcsr_blockdata_addr_,
            (unsigned long long)bcsr_blockids_addr_);
    }
    if (use_bcsr_) {
        if (weights_template_.empty() && window_read_debug_ && output_ && output_->getVerboseLevel() >= 2) {
            output_->verbose(CALL_INFO, 2, 0,
                "[diag-bcsr] node=%u core=%d warning: weights_template empty while BCSR enabled\n",
                node_id_, core_id_);
        } else if (bcsr_rowptr_file_fallback_enable_ && !weights_template_.empty() &&
                   !bcsr_weights_->isRowptrReady() && loadBcsrRowptrFromFile_()) {
            if (window_read_debug_ && output_ && output_->getVerboseLevel() >= 2) {
                output_->verbose(CALL_INFO, 2, 0,
                    "[diag-bcsr] core=%u preload rowptr entries=%zu first=%u second=%u\n",
                    core_id_, bcsr_weights_->rowptrHost().size(),
                    bcsr_weights_->rowptrHost().empty()?0u:bcsr_weights_->rowptrHost()[0],
                    bcsr_weights_->rowptrHost().size()>1?bcsr_weights_->rowptrHost()[1]:0u);
            }
        }
    }

    // 输出权重验证开关以便调试（采样/文件参数由 compute core 负责）
    if (output_) {
        output_->verbose(CALL_INFO, 3, 0,
            "🔍 权重验证配置: verify_weights=%d (details in compute core)\n",
            verify_weights_ ? 1 : 0);
    }
    }

    // output_->verbose(CALL_INFO, 1, 0, "🔧 初始化SnnPE SubComponent (核心%d, %u个神经元)\n", 
    //                 core_id_, num_neurons_);
    
    // 神经元状态由 compute core 维护，控制层不再初始化本地副本
    // 去重发放统计位图（默认全0）
    fired_ever_.assign(num_neurons_, 0);
    
    // 初始化内存访问
    memory_link_ = nullptr;

    // 学习窗口状态由 compute core 维护


    
    // 初始化统计变量
    total_cycles_ = 0;
    active_cycles_ = 0;
    boot_read_sent_ = false;
    boot_write_sent_ = false;
    weights_initialized_ = false;
    memory_ready_ = false;
    stat_spikes_received_ = nullptr;
    stat_spikes_generated_ = nullptr;
    stat_neurons_fired_ = nullptr;
    stat_memory_requests_ = nullptr;
    stat_weight_cache_hits_ = nullptr;
    stat_weight_cache_misses_ = nullptr;
    stat_merged_reads_rows_ = nullptr;
    stat_merged_reads_cls_ = nullptr;
    stat_weights_verify_count_ = nullptr;
    stat_weights_mismatch_count_ = nullptr;
    stat_weights_verify_sum_ = nullptr;
    
    // 初始化内部计数器
    count_spikes_received_ = 0;
    count_spikes_generated_ = 0;
    count_neurons_fired_ = 0;
    count_memory_requests_ = 0;
    count_non_spike_packets_received_ = 0;
    count_stream_mem_verify_pass_ = 0;
    count_stream_mem_verify_fail_ = 0;
    count_stream_pkt_sent_ = 0;
    count_stream_pkt_recv_ = 0;
    count_stream_pkt_bad_crc_ = 0;
    count_stream_pkt_bad_magic_ = 0;
    count_route3d_native_activation_total_ = 0;
    count_route3d_native_gating_activation_total_ = 0;
    count_route3d_native_direct_activation_total_ = 0;
    count_route3d_native_unique_sources_total_ = 0;
    
    // 配置时钟
    std::string clock_freq = "1GHz";
    registerClock(clock_freq, new Clock::Handler2<SnnPESubComponent,&SnnPESubComponent::clockTick>(this));
    
    // 立即注册统计，避免在调用 getStatistics 前指针为空
    initializeStatistics();

    // output_->verbose(CALL_INFO, 2, 0, "✅ SnnPE SubComponent核心%d初始化完成\n", core_id_);
}

SnnPESubComponent::~SnnPESubComponent() {
    // output_->verbose(CALL_INFO, 1, 0, "🗑️ 销毁SnnPE SubComponent核心%d\n", core_id_);
    parent_pe_cached_ = nullptr;
    snn_parent_observer_ = nullptr;

#ifdef SNNDL_ENABLE_PROFILING
    if (profiler_) { delete profiler_; profiler_ = nullptr; }
#endif
    // 避免析构次序竞态：不手动delete日志对象
    output_ = nullptr;
}

// === Activity f (per-window active axons ratio) ===
void SnnPESubComponent::activityFlush_() {
    if (!activity_stats_enable_) return;
    if (!parent_pe_cached_) return;
    if (weights_cols_ == 0) return;
    double f = (double)activity_pre_set_.size() / (double)weights_cols_;
    if (auto* pe = parent_pe_cached_) {
        pe->accumulateActivityF(f);
    }
    activityReset_();
}

size_t SnnPESubComponent::pendingMemSize_() const {
    size_t n = 0;
    if (stdmem_ep_ && stdmem_ep_->available()) {
        if (auto* mem = stdmem_ep_->memoryAccess()) n += mem->pendingSize();
    }
    return n;
}

void SnnPESubComponent::accReset_() {
    if (acc_ops_) acc_ops_->reset();
}

void SnnPESubComponent::accUpdate_(uint32_t post, float dv) {
    if (acc_ops_) acc_ops_->update(post, dv);
}

bool SnnPESubComponent::weightCacheTryGet_(uint64_t key, float& out) {
    if (!weight_cache_ops_) return false;
    return weight_cache_ops_->tryGet(key, out);
}

void SnnPESubComponent::weightCacheStore_(uint64_t key, float value) {
    if (!weight_cache_ops_) return;
    weight_cache_ops_->store(key, value);
}

void SnnPESubComponent::windowStateConfigure_() {
    if (weight_mem_subsystem_) {
        weight_mem_subsystem_->configureWindow(window_read_budget_, max_outstanding_requests_);
    }
}

void SnnPESubComponent::windowStateBegin_() {
    if (weight_mem_subsystem_) {
        weight_mem_subsystem_->beginWindow();
    }
}

bool SnnPESubComponent::windowStateCanIssue_(uint32_t n) const {
    return weight_mem_subsystem_ ? weight_mem_subsystem_->canIssue(n) : false;
}

void SnnPESubComponent::windowStateNoteIssue_(uint32_t n) {
    if (weight_mem_subsystem_) {
        weight_mem_subsystem_->noteIssue(n);
        if (impl_) impl_->updatePendingPeak(weight_mem_subsystem_->outstanding());
    }
}

void SnnPESubComponent::windowStateNoteComplete_(uint32_t n) {
    if (weight_mem_subsystem_) {
        weight_mem_subsystem_->noteComplete(n);
    }
}

uint32_t SnnPESubComponent::windowStateIssued_() const {
    return weight_mem_subsystem_ ? weight_mem_subsystem_->issued() : 0;
}

uint32_t SnnPESubComponent::windowStateOutstanding_() const {
    return weight_mem_subsystem_ ? weight_mem_subsystem_->outstanding() : 0;
}

void SnnPESubComponent::fillStreamRuntime_(ICoreWorkload::Runtime& rt) {
    rt.reporting.ctx = this;
    rt.reporting.report_mem_issue = &SnnPESubComponent::reportStreamMemIssueThunk_;
    rt.sinks.mem_verify_pass = &count_stream_mem_verify_pass_;
    rt.sinks.mem_verify_fail = &count_stream_mem_verify_fail_;
    rt.sinks.pkt_sent = &count_stream_pkt_sent_;
    rt.sinks.pkt_recv = &count_stream_pkt_recv_;
    rt.sinks.pkt_bad_crc = &count_stream_pkt_bad_crc_;
    rt.sinks.pkt_bad_magic = &count_stream_pkt_bad_magic_;
    rt.sinks.stat_mem_writes_issued_total = stat_stream_mem_writes_issued_total_;
    rt.sinks.stat_mem_reads_issued_total = stat_stream_mem_reads_issued_total_;
    rt.sinks.stat_mem_bytes_written_total = stat_stream_mem_bytes_written_total_;
    rt.sinks.stat_mem_bytes_read_total = stat_stream_mem_bytes_read_total_;
    rt.sinks.stat_mem_verify_pass_total = stat_stream_mem_verify_pass_total_;
    rt.sinks.stat_mem_verify_fail_total = stat_stream_mem_verify_fail_total_;
    rt.sinks.stat_pkt_sent_total = stat_stream_pkt_sent_total_;
    rt.sinks.stat_pkt_recv_total = stat_stream_pkt_recv_total_;
    rt.sinks.stat_pkt_bad_crc_total = stat_stream_pkt_bad_crc_total_;
    rt.sinks.stat_pkt_bad_magic_total = stat_stream_pkt_bad_magic_total_;
}

uint64_t SnnPESubComponent::workloadNowNsThunk_(void* ctx) {
    auto* self = static_cast<SnnPESubComponent*>(ctx);
    return self ? self->getCurrentSimTimeNano() : 0;
}

void SnnPESubComponent::bindWorkloadRuntime_() {
    if (!workload_) return;

    ICoreWorkload::Runtime rt{};
    rt.log = output_;
    rt.node_id = static_cast<uint32_t>(node_id_);
    rt.core_id = static_cast<uint32_t>(core_id_);
    rt.total_nodes = total_nodes_cfg_;
    rt.total_cores = static_cast<uint32_t>(total_cores_);
    rt.neurons_per_core = num_neurons_;
    rt.neurons_per_pe = neurons_per_pe_cfg_;
    rt.global_neuron_base = global_neuron_base_;
    rt.base_addr = base_addr_;
    rt.mem = (stdmem_ep_ && stdmem_ep_->available()) ? stdmem_ep_->memoryAccess() : nullptr;
    rt.noc = noc_transport_;
    rt.time.ctx = this;
    rt.time.now_ns = &SnnPESubComponent::workloadNowNsThunk_;
    rt.reporting.ctx = this;
    // Non-SNN workloads must not be counted as weight reads (keep analysis semantics clean).
    rt.reporting.report_mem_issue =
        isNonSnnWorkload_() ? &SnnPESubComponent::reportStreamMemIssueThunk_ : &SnnPESubComponent::reportSnnMemIssueThunk_;
    rt.reporting.report_apply_scatter = &SnnPESubComponent::reportApplyScatterThunk_;
    rt.reporting.request_gas_end_gather = &SnnPESubComponent::requestGasEndGatherThunk_;
    rt.reporting.request_gas_end_scatter = &SnnPESubComponent::requestGasEndScatterThunk_;
    rt.sinks.spikes_received = &count_spikes_received_;
    rt.sinks.spikes_generated = &count_spikes_generated_;
    rt.sinks.neurons_fired = &count_neurons_fired_;
    rt.sinks.synaptic_accesses = &count_synaptic_accesses_;
    rt.sinks.window_spikes_all = &window_spikes_all_;
    rt.sinks.spikes_emitted_window = &spikes_emitted_window_;
    rt.sinks.stat_spikes_received_total = stat_spikes_received_;
    rt.sinks.stat_spikes_generated_total = stat_spikes_generated_;
    rt.sinks.stat_neurons_fired_total = stat_neurons_fired_;
    rt.sinks.stat_synaptic_accesses_total = stat_synaptic_accesses_;
    rt.sinks.stat_compute_active_cycles_total = nullptr;
    if (auto* pe = snn_parent_observer_) {
        rt.sinks.stat_compute_active_cycles_total = pe->getComputeActiveCyclesTotalStatistic();
    }
    rt.sinks.stat_gas_scatter_spikes_emitted_total = stat_gas_scatter_spikes_emitted_total_;
    rt.sinks.route3d_native_activation_total = &count_route3d_native_activation_total_;
    rt.sinks.route3d_native_gating_activation_total = &count_route3d_native_gating_activation_total_;
    rt.sinks.route3d_native_direct_activation_total = &count_route3d_native_direct_activation_total_;
    rt.sinks.route3d_native_unique_sources_total = &count_route3d_native_unique_sources_total_;
    rt.sinks.stat_routes_entries_total = stat_routes_entries_;
    rt.sinks.stat_fanout_per_spike_total = stat_fanout_per_spike_;
    rt.sinks.stat_route3d_native_activation_total = stat_route3d_native_activation_total_;
    rt.sinks.stat_route3d_native_gating_activation_total = stat_route3d_native_gating_activation_total_;
    rt.sinks.stat_route3d_native_direct_activation_total = stat_route3d_native_direct_activation_total_;
    rt.sinks.stat_route3d_native_unique_sources_total = stat_route3d_native_unique_sources_total_;

    if (riscv_snn_runtime_bridge_) {
        riscv_snn_runtime_bridge_->bindRuntime(rt);
        accel_runtime_services_ = riscv_snn_runtime_bridge_.get();
    }
    rt.accel_runtime = accel_runtime_services_;

    if (isStreamLikeWorkload_()) {
        fillStreamRuntime_(rt);
    }

    workload_->bindRuntime(rt);

    // Tier2-E (A): CoreShell 统一装配 weight reader（WMS），并将所有权一次性移交给 workload=snn。
    // - 避免 control 与 workload 重复创建 WeightMemorySubsystem/WeightCacheOps；
    // - 不改变脚本侧参数/接口，仅在 C++ 内部通过窄接口移交所有权。
    if (auto* adopter = dynamic_cast<IWeightReaderAdopter*>(workload_.get())) {
        std::unique_ptr<IWeightReader> reader = takeWeightReader_();
        if (reader) adopter->adoptWeightReader(std::move(reader));
    }
}

bool SnnPESubComponent::applyPlatformConfig(const CorePlatformConfig& config,
                                            std::string& error) {
    WorkloadKind configured = WorkloadKind::Snn;
    switch (workload_impl_) {
        case WorkloadImpl::RiscvSnn: configured = WorkloadKind::RiscvSnn; break;
        case WorkloadImpl::Stream: configured = WorkloadKind::Stream; break;
        case WorkloadImpl::Traffic: configured = WorkloadKind::Traffic; break;
        case WorkloadImpl::TrafficMem: configured = WorkloadKind::TrafficMem; break;
        case WorkloadImpl::Tensor: configured = WorkloadKind::Tensor; break;
        case WorkloadImpl::Snn: configured = WorkloadKind::Snn; break;
    }
    const WorkloadKind expected = workloadKindFromString(config.workload_impl);
    if (configured != expected) {
        error = "workload_impl mismatch: core=" +
                std::string(workloadKindName(configured)) + " parent=" +
                std::string(workloadKindName(expected));
        return false;
    }
    if (config.total_nodes == 0 || config.total_cores == 0 ||
        config.neurons_per_core == 0 ||
        config.neurons_per_pe != config.total_cores * config.neurons_per_core) {
        error = "invalid parent topology";
        return false;
    }

    node_id_ = config.node_id;
    core_id_ = static_cast<int>(config.core_id);
    total_nodes_cfg_ = config.total_nodes;
    total_cores_ = static_cast<int>(config.total_cores);
    num_neurons_ = config.neurons_per_core;
    neurons_per_pe_cfg_ = config.neurons_per_pe;
    global_neuron_base_ = config.global_neuron_base;
    base_addr_ = config.base_addr;
    return true;
}

void SnnPESubComponent::setParentInterface(IPeAggregation* parent) {
    parent_pe_cached_ = parent;
    snn_parent_observer_ = dynamic_cast<MultiCorePE*>(parent);
    refreshSharedWeightObjectPlaneBinding_();
    // output_->verbose(CALL_INFO, 2, 0, "🔗 核心%d设置父级接口\n", core_id_);
    bindWorkloadRuntime_();
}

void SnnPESubComponent::refreshSharedWeightObjectPlaneBinding_() {
    if (!weight_mem_subsystem_) return;

    auto* provider = dynamic_cast<IPeWeightObjectPlaneProvider*>(parent_pe_cached_);
    auto* plane = provider ? provider->peWeightObjectPlane() : nullptr;
    auto* pod_provider = dynamic_cast<IPePodSharedMetadataProvider*>(parent_pe_cached_);
    auto* metadata_plane =
        pod_provider ? pod_provider->pePodMetadataObjectPlane() : nullptr;
    auto* owner_table =
        pod_provider ? pod_provider->pePodOwnerServiceTable() : nullptr;
    auto* service_table =
        pod_provider ? pod_provider->peLocalServiceObjectTable() : nullptr;
    auto* fabric_provider = dynamic_cast<IPeSharedCoreFabricProvider*>(parent_pe_cached_);
    auto* fabric = fabric_provider ? fabric_provider->peSharedCoreFabric() : nullptr;
    auto* mc_pe = snn_parent_observer_;

    weight_mem_subsystem_->bindSharedWeightObjectPlane(plane);
    weight_mem_subsystem_->bindPodMetadataObjectPlane(metadata_plane);
    weight_mem_subsystem_->bindPodOwnerServiceTable(owner_table);
    weight_mem_subsystem_->bindPeLocalServiceObjectTable(service_table);
    weight_mem_subsystem_->bindPeSharedCoreFabric(fabric);
    weight_mem_subsystem_->setPulseOsaMetadataTxnConfig(
        mc_pe ? mc_pe->pulseOsaMetadataTxnEnabled() : false,
        mc_pe ? mc_pe->pulseOsaMetadataReadyLeaseEnabled() : false,
        mc_pe ? mc_pe->pulseOsaMetadataObjectMaskBits() : 0u);
    if (mc_pe) {
        const uint32_t pod_count =
            std::max<uint32_t>(1u, mc_pe->peInternalPodCountConfig());
        const uint32_t pod_size =
            std::max<uint32_t>(1u, mc_pe->peInternalPodSizeConfig());
        const uint32_t pod_id =
            std::min<uint32_t>(
                static_cast<uint32_t>(core_id_) / pod_size,
                pod_count - 1u);
        weight_mem_subsystem_->setPeInternalPodRuntimeConfig(
            mc_pe->peInternalCpeEnabledConfig(),
            mc_pe->peInternalPodEnabledConfig(),
            mc_pe->peInternalPodMetadataEnabledConfig(),
            mc_pe->peInternalPodOwnerEnabledConfig(),
            pod_id,
            pod_count,
            pod_size,
            static_cast<uint32_t>(core_id_));
    }
    weight_mem_subsystem_->setSharedWeightObjectPlaneResidencyAuthority(
        plane != nullptr && plane->actualOwnerEnabled());
}

void SnnPESubComponent::setNocTransport(INocTransport* noc) {
    noc_transport_ = noc;
    bindWorkloadRuntime_();
}

bool SnnPESubComponent::deliverPacket(NocPacketEvent* packet) {
    return invokeWorkload_([&]() {
        if (!packet) return true;
        // Experimental PULSE gather-preband isolation:
        // serialize PE-side workload ingress so packet delivery cannot race
        // with stage transitions / workload ticks on the same core.
        if (workload_) {
            return workload_->deliverPacket(packet);
        }

        if (enable_extended_diagnostics_ && output_) {
            output_->verbose(
                CALL_INFO, 1, 0,
                "[core-packet] core=%d kind=%u payload=%zu src=%u:%u dst=%u:%u\n",
                core_id_,
                static_cast<unsigned>(packet->kind),
                packet->payload.size(),
                packet->src_node,
                packet->src_endpoint,
                packet->dst_node,
                packet->dst_endpoint);
        }
        delete packet;
        return true;
    });
}

bool SnnPESubComponent::syntheticEmitNeuronFire(uint32_t neuron_idx, uint64_t now_cycle) {
    if (!snn_comm_workload_ || !snn_comm_workload_->ready()) return false;
    if (neuron_idx >= num_neurons_) return false;
    snn_comm_workload_->emitNeuronFire(neuron_idx, now_cycle);
    return true;
}

uint64_t SnnPESubComponent::syntheticEmitNeuronFireBatch(const std::vector<uint32_t>& neuron_indices, uint64_t now_cycle) {
    if (!snn_comm_workload_ || !snn_comm_workload_->ready() || neuron_indices.empty()) return 0;

    std::vector<uint32_t> filtered;
    filtered.reserve(neuron_indices.size());
    for (uint32_t neuron_idx : neuron_indices) {
        if (neuron_idx < num_neurons_) filtered.push_back(neuron_idx);
    }
    if (filtered.empty()) return 0;
    return snn_comm_workload_->emitNeuronFireBatch(filtered, now_cycle);
}

void SnnPESubComponent::onGlobalStepStart(uint32_t seq) {
    const bool trace_step_path =
        output_ &&
        node_id_ == 0 &&
        seq <= 2 &&
        core_id_ >= 15;
    invokeWorkload_([&]() {
    if (trace_step_path) {
        output_->verbose(
            CALL_INFO, 2, 0,
            "[[sentinel-step-path]] node=%u core=%d seq=%u onGlobalStepStart enter non_snn=%d gas_window=%d\n",
            node_id_, core_id_, seq, isNonSnnWorkload_() ? 1 : 0, gas_window_mode_ ? 1 : 0);
    }
    if (isNonSnnWorkload_()) {
        // stream workload 不参与 SNN/GAS window 编排；
        // 但仍需要打开 memory(GatherBufferIF) 的 step gate，否则 StandardMemAccess 会拒绝请求（write/read 返回失败）。
        curr_stage_seq_ = seq;
        if (stdmem_ep_ && stdmem_ep_->available()) {
            auto* gate = stdmem_ep_->stepGate();
            if (gate) {
                auto* credit_gate = stdmem_ep_->creditGate();
                if (credit_gate &&
                    global_step_apply_bank_credit_seq_ == seq &&
                    global_step_apply_bank_credit_target_ > 0) {
                    credit_gate->setApplyBankCreditTarget(seq, global_step_apply_bank_credit_target_);
                    global_step_apply_bank_credit_seq_ = 0;
                    global_step_apply_bank_credit_target_ = 0;
                }
                gate->openStep(seq);
                if (trace_step_path) {
                    output_->verbose(
                        CALL_INFO, 2, 0,
                        "[[sentinel-step-path]] node=%u core=%d seq=%u onGlobalStepStart non_snn gate_opened\n",
                        node_id_, core_id_, seq);
                }
            }
        }
        if (trace_step_path) {
            output_->verbose(
                CALL_INFO, 2, 0,
                "[[sentinel-step-path]] node=%u core=%d seq=%u onGlobalStepStart non_snn before_workload\n",
                node_id_, core_id_, seq);
        }
        if (workload_) workload_->onGlobalStepStart(seq);
        if (trace_step_path) {
            output_->verbose(
                CALL_INFO, 2, 0,
                "[[sentinel-step-path]] node=%u core=%d seq=%u onGlobalStepStart non_snn after_workload\n",
                node_id_, core_id_, seq);
        }
        if (accel_runtime_services_) accel_runtime_services_->onGlobalStepStart(seq);
        if (trace_step_path) {
            output_->verbose(
                CALL_INFO, 2, 0,
                "[[sentinel-step-path]] node=%u core=%d seq=%u onGlobalStepStart non_snn after_runtime_services\n",
                node_id_, core_id_, seq);
        }
        return;
    }
    // 全局 Step 同步：打开 memory(GatherBufferIF) 的新窗口。
    // 注意：这里不直接操作 GAS 状态机，而是通过 IGasStepGate 保持边界清晰。
    if (!stdmem_ep_ || !stdmem_ep_->available()) {
        if (output_) output_->fatal(CALL_INFO, -1, "core=%d onGlobalStepStart(seq=%u) but stdmem endpoint is unavailable\n", core_id_, seq);
        return;
    }
    auto* gate = stdmem_ep_->stepGate();
    if (gate) {
        auto* credit_gate = stdmem_ep_->creditGate();
        if (credit_gate &&
            global_step_apply_bank_credit_seq_ == seq &&
            global_step_apply_bank_credit_target_ > 0) {
            credit_gate->setApplyBankCreditTarget(seq, global_step_apply_bank_credit_target_);
            global_step_apply_bank_credit_seq_ = 0;
            global_step_apply_bank_credit_target_ = 0;
        }
        gate->openStep(seq);
        if (trace_step_path) {
            output_->verbose(
                CALL_INFO, 2, 0,
                "[[sentinel-step-path]] node=%u core=%d seq=%u onGlobalStepStart gate_opened\n",
                node_id_, core_id_, seq);
        }
        if (trace_step_path) {
            output_->verbose(
                CALL_INFO, 2, 0,
                "[[sentinel-step-path]] node=%u core=%d seq=%u onGlobalStepStart before_workload\n",
                node_id_, core_id_, seq);
        }
        if (workload_) workload_->onGlobalStepStart(seq);
        if (trace_step_path) {
            output_->verbose(
                CALL_INFO, 2, 0,
                "[[sentinel-step-path]] node=%u core=%d seq=%u onGlobalStepStart after_workload\n",
                node_id_, core_id_, seq);
        }
        if (trace_step_path) {
            output_->verbose(
                CALL_INFO, 2, 0,
                "[[sentinel-step-path]] node=%u core=%d seq=%u onGlobalStepStart before_runtime_services\n",
                node_id_, core_id_, seq);
        }
        if (accel_runtime_services_) accel_runtime_services_->onGlobalStepStart(seq);
        if (trace_step_path) {
            output_->verbose(
                CALL_INFO, 2, 0,
                "[[sentinel-step-path]] node=%u core=%d seq=%u onGlobalStepStart after_runtime_services\n",
                node_id_, core_id_, seq);
        }
        return;
    }

    // naive/non-window 模式：memory 可能是 memHierarchy.standardInterface（不实现 IGasStepGate）。
    // 仅在窗口化 GAS 模式下，缺失 step gate 才属于配置错误（fail-fast）。
    if (gas_window_mode_ && window_read_enable_) {
        if (output_) output_->fatal(
            CALL_INFO, -1,
            "core=%d onGlobalStepStart(seq=%u) requires memory to implement IGasStepGate (did you load GatherBufferIF with step_gate_enable=1?)\n",
            core_id_, seq);
        return;
    }
    curr_stage_seq_ = seq;
    if (workload_) workload_->onGlobalStepStart(seq);
    if (accel_runtime_services_) accel_runtime_services_->onGlobalStepStart(seq);
    if (trace_step_path) {
        output_->verbose(
            CALL_INFO, 2, 0,
            "[[sentinel-step-path]] node=%u core=%d seq=%u onGlobalStepStart no_gate_after_callbacks\n",
            node_id_, core_id_, seq);
    }
    });
}

void SnnPESubComponent::onGlobalStepApplyBankCredit(uint32_t seq, uint32_t apply_bank_credit) {
    global_step_apply_bank_credit_seq_ = seq;
    global_step_apply_bank_credit_target_ = apply_bank_credit;
}

// === GAS stage/stat sink (Phase4-Task6.4) ===
void SnnPESubComponent::onGasStageEvent(const GasStageEvent& ev) {
    invokeWorkload_([&]() {
    // CoreShell 保留最小镜像状态用于：
    // - StageEventHub 统计/CSV 口径（仍由 CoreShell 汇聚写出）
    // - activity f 等 PE 级聚合（兼容历史分析脚本）
    // 具体窗口读编排与 scatter 事务由 workload=snn 接管（通过转发完成）。

    curr_stage_seq_ = ev.superstep;

    // Mirror GAS stage enum (minimal).
    switch (ev.op) {
        case GasOp::BeginGather:
            gas_stage_ = GasStage::Gather;
            // Reset per-window diagnostics in CoreShell only (no side effects).
            record_edge_capacity_warned_ = false;
            diag_edges_record_hits_ = 0;
            diag_edges_stage_skips_ = 0;
            diag_edges_cond_skips_ = 0;
            diag_spikes_stage_gather_ = 0;
            diag_spikes_stage_apply_ = 0;
            diag_spikes_stage_scatter_ = 0;
            diag_spikes_stage_idle_ = 0;
            // Stage event bookkeeping (timing + window spikes reset) stays in CoreShell.
            if (impl_) impl_->markBeginGather(curr_stage_seq_);
            break;
        case GasOp::BeginApply:
            gas_stage_ = GasStage::Apply;
            if (impl_) impl_->markBeginApply(curr_stage_seq_);
            break;
        case GasOp::EndApply:
            gas_stage_ = GasStage::Apply; // remain until BeginScatter
            // EndApply is still recorded for stage CSV (legacy analysis scripts).
            appendStageEventRow_("EndApply", getCurrentSimTimeNano(), 0);
            break;
        case GasOp::BeginScatter:
            gas_stage_ = GasStage::Scatter;
            // BeginScatter timing stays in CoreShell; window spikes baseline used by EndScatter fallback.
            spikes_generated_base_ = count_spikes_generated_;
            if (impl_) impl_->markBeginScatter(curr_stage_seq_);
            break;
        case GasOp::EndScatter: {
            gas_stage_ = GasStage::Idle;
            uint64_t to_emit = window_spikes_all_ ? window_spikes_all_ : spikes_emitted_window_;
            if (to_emit == 0) {
                uint64_t delta = 0;
                if (count_spikes_generated_ >= spikes_generated_base_) {
                    delta = count_spikes_generated_ - spikes_generated_base_;
                }
                if (delta > 0) to_emit = delta;
            }
            if (impl_) impl_->markEndScatter(curr_stage_seq_, to_emit);
            break;
        }
        default:
            break;
    }

    // Activity-f window tracking: keep in CoreShell (PE-level aggregation).
    switch (ev.op) {
        case GasOp::BeginGather:
            activity_window_seq_ = ev.superstep;
            activityReset_();
            break;
        case GasOp::BeginApply:
        case GasOp::EndApply:
        case GasOp::BeginScatter:
            activityFlush_();
            break;
        default:
            break;
    }

    // Forward to workload (if it implements IGasStageSink).
    if (gas_stage_workload_) {
        gas_stage_workload_->onGasStageEvent(ev);
    }
    if (accel_runtime_services_) {
        accel_runtime_services_->onGasStageEvent(ev);
    }
    });
}

void SnnPESubComponent::onGasStatEvent(const GasStatEvent& st) {
    // Accumulate at PE level for CSV visibility (keep identical to legacy behavior).
    if (auto* pe = parent_pe_cached_) {
        pe->accumulateGasStatsExt(st.unique_bytes, st.unique_reads,
                                  st.rowwin_triggers, st.rowwin_bytes,
                                  st.bursts, st.payload_bytes,
                                  st.window_inflight_peak, st.window_buffer_max_bytes,
                                  st.frontend_staged_reads,
                                  st.frontend_staged_line_touches,
                                  st.frontend_granules_built,
                                  st.unique_line_count, st.covered_line_count, st.overfetch_bytes,
                                  st.apply_bank_credit_effective,
                                  st.cmd_cost_veto,
                                  st.cmd_cost_veto_fine_gap,
                                  st.cmd_cost_veto_row_window,
                                  st.stall_on_step_gate_cycles);
        if (auto* mpe = snn_parent_observer_) {
            const uint32_t seq = (st.superstep != 0) ? st.superstep : static_cast<uint32_t>(curr_stage_seq_);
            mpe->recordStepGasStat(seq, st);
            mpe->recordCoreStepGasStat(core_id_, seq, st);
        }
    }
    // Local (per-core) copies for unique_* only (optional)
    if (stat_gas_unique_reads_total_ && st.unique_reads) stat_gas_unique_reads_total_->addData(st.unique_reads);
    if (stat_gas_unique_bytes_total_ && st.unique_bytes) stat_gas_unique_bytes_total_->addData(st.unique_bytes);
    if (stat_gas_row_window_triggers_total_ && st.rowwin_triggers) stat_gas_row_window_triggers_total_->addData(st.rowwin_triggers);
    if (stat_gas_row_window_bytes_total_ && st.rowwin_bytes) stat_gas_row_window_bytes_total_->addData(st.rowwin_bytes);
    if (stat_gas_bursts_total_ && st.bursts) stat_gas_bursts_total_->addData(st.bursts);
    if (stat_gas_payload_bytes_total_ && st.payload_bytes) stat_gas_payload_bytes_total_->addData(st.payload_bytes);
    if (stat_gas_gap_absorbed_bytes_total_ && st.gap_absorbed_bytes) stat_gas_gap_absorbed_bytes_total_->addData(st.gap_absorbed_bytes);

    // Optional forward (mostly no-op for workload=snn; kept for completeness).
    if (gas_stage_workload_) {
        gas_stage_workload_->onGasStatEvent(st);
    }
    if (accel_runtime_services_) {
        accel_runtime_services_->onGasStatEvent(st);
    }
}

void SnnPESubComponent::configureWeightReaderSubsystem_(const Params& params) {
    // 构建权重读取子系统（Phase E：内存子系统闭环，控制层不再持有 pending/解析）
    if (!weight_reader_adapter_) {
        auto mem = std::make_unique<WeightMemorySubsystem>();
        mem->configure(
            [this](uint64_t key, float& out) { return weightCacheTryGet_(key, out); },
            [this](uint64_t key, float val) { weightCacheStore_(key, val); }
        );
        const bool disable_weight_cache = params.find<int>("disable_weight_cache", 0) != 0;
        // Phase A/E：窗口读集合/预算/outstanding + 读发起/响应闭环下沉到子系统（保持行为与日志口径）
        {
            WeightMemorySubsystem::OrchestratorConfig ocfg{};
            ocfg.accessor = weight_accessor_.get();
            ocfg.cache_try = [this](uint64_t key, float& out) -> bool {
                return weightCacheTryGet_(key, out);
            };
            ocfg.cache_put = [this](uint64_t key, float v) {
                weightCacheStore_(key, v);
            };
            ocfg.acc_update = [this](uint32_t post_l, float dv) { accUpdate_(post_l, dv); };
            ocfg.diag_edge_weight = [this](const char* tag, uint32_t post_l, uint32_t pre_g, float w, uint32_t cnt) {
                diagEdgeWeight_(tag, post_l, pre_g, w, cnt);
            };
            ocfg.report_cache_access = [this](bool hit) { if (impl_) impl_->reportCacheAccess(hit); };
            ocfg.update_pending_peak = [this](uint32_t ostd) { if (impl_) impl_->updatePendingPeak(ostd); };
            ocfg.report_mem_issue = [this](size_t bytes, bool count_weight_read) {
                if (impl_) impl_->reportMemoryIssue(bytes, count_weight_read);
            };
            ocfg.report_mem_latency = [this](uint64_t lat_cycles, bool is_weight) {
                accum_mem_latency_cycles_ += lat_cycles;
                count_mem_responses_++;
                if (auto* pe = parent_pe_cached_) {
                    pe->accumulateMemReadLatency(lat_cycles, is_weight);
                }
            };
            ocfg.ensure_loader_ready = [this]() { return ensureLoaderReady_(); };
            ocfg.bcsr_rowptr_ready = [this]() { return !use_bcsr_ || bcsr_weights_->isRowptrReady(); };
            ocfg.ensure_rowptr_ready_or_fatal = [this](const char* reason) { ensureRowptrReadyOrFatal_(reason); };
            ocfg.resume_issue_after_rowptr_ready = [this]() {
                if (apply_acc_enable_ && gas_window_mode_ && gas_stage_ == GasStage::Apply) {
                    issueEdgeWeightFetches_();
                }
            };
            ocfg.read_bcsr_from_file = [this](uint32_t post_l, uint32_t pre_g) { return readBcsrWeightFromFile_(post_l, pre_g); };
            ocfg.use_bcsr = use_bcsr_;
            ocfg.bcsr_force_file_read = bcsr_force_file_read_;
            ocfg.bcsr_prefetch_all = bcsr_prefetch_all_;
            ocfg.bcsr_rowptr_file_fallback_enable = bcsr_rowptr_file_fallback_enable_;
            ocfg.bcsr_colidx_inflight_coalesce_enable =
                params.find<int>("bcsr_colidx_inflight_coalesce_enable", 1) != 0;
            ocfg.bcsr_block_inflight_coalesce_enable =
                params.find<int>("bcsr_block_inflight_coalesce_enable", 1) != 0;
            ocfg.bcsr_block_fetch_mode =
                params.find<std::string>("bcsr_block_fetch_mode", "full_block");
            ocfg.bcsr_row_index_prefetch_mode =
                params.find<std::string>("bcsr_row_index_prefetch_mode", "auto");
            ocfg.bcsr_row_index_prefetch_all_rows_threshold =
                params.find<uint32_t>("bcsr_row_index_prefetch_all_rows_threshold", 1024);
            ocfg.bcsr_row_index_prefetch_all_rows_max_bytes =
                params.find<uint64_t>("bcsr_row_index_prefetch_all_rows_max_bytes", 64ull * 1024ull);
            ocfg.experimental_noc_rowidx_prefetch_enable =
                params.find<int>("experimental_noc_rowidx_prefetch_enable", 0) != 0;
            ocfg.experimental_noc_rowidx_prefetch_budget_per_tick =
                params.find<uint32_t>("experimental_noc_rowidx_prefetch_budget_per_tick", 4);
            ocfg.experimental_noc_rowidx_cache_rows =
                params.find<uint32_t>("experimental_noc_rowidx_cache_rows", 1024);
            ocfg.experimental_noc_rowidx_prefetch_gather_only =
                params.find<int>("experimental_noc_rowidx_prefetch_gather_only", 1) != 0;
            ocfg.experimental_noc_rowidx_prefetch_detached_enable =
                params.find<int>("experimental_noc_rowidx_prefetch_detached_enable", 0) != 0;
            ocfg.experimental_noc_rowidx_prefetch_carry_to_apply_enable =
                params.find<int>("experimental_noc_rowidx_prefetch_carry_to_apply_enable", 0) != 0;
            ocfg.experimental_noc_rowidx_hot_touch_min =
                params.find<uint32_t>("experimental_noc_rowidx_hot_touch_min", 1);
            ocfg.experimental_noc_rowidx_budget_adapt_enable =
                params.find<int>("experimental_noc_rowidx_budget_adapt_enable", 0) != 0;
            ocfg.experimental_noc_rowidx_budget_adapt_max_per_tick =
                params.find<uint32_t>("experimental_noc_rowidx_budget_adapt_max_per_tick", 32);
            ocfg.experimental_noc_rowidx_budget_adapt_q_depth =
                params.find<uint32_t>("experimental_noc_rowidx_budget_adapt_q_depth", 16);
            ocfg.experimental_idx2_ingress_prefetch_enable = false;
            ocfg.experimental_idx2_ingress_prefetch_budget_per_tick = 0;
            ocfg.experimental_idx2_ingress_prefetch_cache_entries = 0;
            ocfg.experimental_idx2_ingress_prefetch_max_inflight = 0;
            ocfg.experimental_idx2_ingress_prefetch_gather_only = false;
            ocfg.experimental_idx2_ingress_prefetch_carry_to_apply_enable = false;
            ocfg.experimental_idx2_ingress_prefetch_apply_max_inflight = 0;
            ocfg.experimental_idx2_ingress_prefetch_apply_outstanding_reserve = 0;
            ocfg.experimental_idx2_ingress_prefetch_apply_frontier_keep_pending = 0;
            ocfg.experimental_idx2_ingress_tail_guard_enable = false;
            ocfg.experimental_idx2_ingress_budget_adapt_enable = false;
            ocfg.experimental_idx2_ingress_budget_adapt_max_per_tick = 0;
            ocfg.experimental_idx2_ingress_budget_adapt_q_depth = 0;
            ocfg.bcsr_block_cache_auto_tune =
                params.find<int>("bcsr_block_cache_auto_tune", 1) != 0;
            ocfg.bcsr_block_cache_max_bytes =
                params.find<uint64_t>("bcsr_block_cache_max_bytes", 64ull * 1024ull * 1024ull);
            ocfg.bcsr_block_cache_tune_miss_ratio =
                params.find<float>("bcsr_block_cache_tune_miss_ratio", 0.05f);
            ocfg.bcsr_block_cache_tune_min_misses =
                params.find<uint32_t>("bcsr_block_cache_tune_min_misses", 64);
            ocfg.bcsr_populate_weight_cache_enable =
                params.find<int>("bcsr_populate_weight_cache_enable", 1) != 0;
            if (disable_weight_cache) ocfg.bcsr_populate_weight_cache_enable = false;
            ocfg.bcsr_weight_guard_enable = bcsr_weight_guard_enable_;
            ocfg.bcsr_weight_abs_max = bcsr_weight_abs_max_;
            ocfg.bcsr_semantic_verify_enable = bcsr_semantic_verify_enable_;
            ocfg.bcsr_semantic_verify_max_edges = bcsr_semantic_verify_max_edges_;
            ocfg.bcsr_semantic_verify_max_mismatch = bcsr_semantic_verify_max_mismatch_;
            ocfg.bcsr_semantic_verify_abs_tol = bcsr_semantic_verify_abs_tol_;
            ocfg.bcsr_semantic_verify_rel_tol = bcsr_semantic_verify_rel_tol_;
            ocfg.readresp_zero_fallback = readresp_zero_fallback_;
            ocfg.init_default_weight = init_default_weight_;
            ocfg.num_neurons = num_neurons_;
            ocfg.weights_cols = weights_cols_;
            ocfg.use_post_row_pre_col = use_post_row_pre_col_;
            ocfg.base_addr = static_cast<uint64_t>(base_addr_);
            ocfg.weight_region_end = weight_region_end_;
            ocfg.dense_layout_mode = dense_phys_enable_ ? DenseLayoutMode::PhysV1 : DenseLayoutMode::RowMajor;
            ocfg.dense_phys_dram_row_bytes = dense_phys_dram_row_bytes_;
            ocfg.read_force_single = read_force_single_;
            ocfg.merge_read_cacheline = merge_read_cacheline_;
            ocfg.merge_read_row = merge_read_row_;
            ocfg.merge_read_auto = merge_read_auto_;
            ocfg.line_size_bytes = line_size_bytes_;
            ocfg.byte_exact_verify_enable = byte_exact_verify_enable_;
            ocfg.byte_exact_verify_mode = byte_exact_verify_mode_;
            ocfg.byte_exact_verify_row_scale = byte_exact_verify_row_scale_;
            ocfg.byte_exact_verify_max_mismatch = byte_exact_verify_max_mismatch_;
            ocfg.synapse_weight_mode = synapse_weight_mode_;
            ocfg.weight_sram_model_enable =
                params.find<int>("weight_sram_model_enable", 0) != 0;
            ocfg.weight_idx_sram_enable =
                params.find<int>("weight_idx_sram_enable", 0) != 0;
            ocfg.weight_l0_sram_enable =
                params.find<int>("weight_l0_sram_enable", 0) != 0;
            ocfg.weight_idx_sram_capacity_bytes =
                params.find<uint64_t>("weight_idx_sram_capacity_bytes", 0);
            ocfg.weight_l0_sram_capacity_bytes =
                params.find<uint64_t>("weight_l0_sram_capacity_bytes", 0);
            ocfg.weight_idx_sram_banks =
                params.find<uint32_t>("weight_idx_sram_banks", 16);
            ocfg.weight_l0_sram_banks =
                params.find<uint32_t>("weight_l0_sram_banks", 8);
            ocfg.weight_sram_ports_per_bank =
                params.find<uint32_t>("weight_sram_ports_per_bank", 1);
            ocfg.weight_sram_bank_interleave_bytes =
                params.find<uint64_t>("weight_sram_bank_interleave_bytes", 4);
            ocfg.weight_sram_t_read_cycles =
                params.find<uint32_t>("weight_sram_t_read_cycles", 1);
            ocfg.weight_sram_t_write_cycles =
                params.find<uint32_t>("weight_sram_t_write_cycles", 1);
            ocfg.weight_sram_sample_log2 =
                params.find<uint32_t>("weight_sram_sample_log2", 0);
            ocfg.weight_idx_sram_base =
                params.find<uint64_t>("weight_idx_sram_base", 0x100000000ull);
            ocfg.weight_l0_sram_base =
                params.find<uint64_t>("weight_l0_sram_base", 0x200000000ull);
            ocfg.weight_l0_sram_slots =
                params.find<uint64_t>("weight_l0_sram_slots", (1ull << 20));
            ocfg.memory_warmup_cycles = memory_warmup_cycles_;
            ocfg.loader_barrier_cycles = loader_barrier_cycles_;
            ocfg.node_id = node_id_;
            ocfg.core_id = static_cast<uint32_t>(core_id_);
            ocfg.total_cores = static_cast<uint32_t>(total_cores_);
            ocfg.weights_template = weights_template_;
            ocfg.gcss_index_template = gcss_index_template_;
            ocfg.experimental_retire_policy =
                params.find<std::string>("experimental_retire_policy", "global_inorder");
            ocfg.experimental_gcss_phase_breakdown_enable =
                params.find<int>("experimental_gcss_phase_breakdown_enable", 0) != 0;
            ocfg.experimental_retire_shadow_per_post_enable = false;
            ocfg.experimental_gcss_vlf_queue_policy =
                params.find<std::string>("experimental_gcss_vlf_queue_policy", "locality_first");
            ocfg.experimental_gcss_vlf_fair_band_size =
                params.find<uint32_t>("experimental_gcss_vlf_fair_band_size", 256);
            const bool pulse_enable = params.find<int>("pulse_enable", 0) != 0;
            const bool pulse_agenda_observe_only =
                params.find<int>("pulse_agenda_observe_only", 1) != 0;
            const bool pulse_descriptor_actual_enable =
                params.find<int>("pulse_descriptor_actual_enable", 0) != 0;
            ocfg.pulse_agenda_enable =
                pulse_enable &&
                (pulse_agenda_observe_only || pulse_descriptor_actual_enable);
            ocfg.pulse_descriptor_actual_enable =
                pulse_enable && pulse_descriptor_actual_enable;
            ocfg.pulse_osa_metadata_txn_enable =
                pulse_enable && (params.find<int>("pulse_osa_metadata_txn_enable", 0) != 0);
            ocfg.pulse_osa_metadata_ready_lease_enable =
                pulse_enable && (params.find<int>("pulse_osa_metadata_ready_lease_enable", 0) != 0);
            const std::string pulse_osa_metadata_object_mask =
                pulse_enable
                    ? params.find<std::string>("pulse_osa_metadata_object_mask", "rowdescriptor")
                    : "rowdescriptor";
            ocfg.pulse_osa_metadata_object_mask =
                pulse_enable
                    ? parsePulseMetadataObjectMask_(
                          pulse_osa_metadata_object_mask.empty() ? "rowdescriptor"
                                                                 : pulse_osa_metadata_object_mask)
                    : 0u;
            ocfg.experimental_rowdescriptor_ready_join_dedup_enable =
                ocfg.pulse_descriptor_actual_enable &&
                (params.find<int>("experimental_rowdescriptor_ready_join_dedup_enable", 0) != 0);
            ocfg.pulse_domain_retire_enable = false;
            ocfg.pulse_domain_retire_observe_only = true;
            ocfg.pulse_frontier_observe_enable =
                pulse_enable && (params.find<int>("pulse_frontier_observe_enable", 0) != 0);
            ocfg.pulse_frontier_top_lines =
                std::max<uint32_t>(1u, params.find<uint32_t>("pulse_frontier_top_lines", 32));
            ocfg.pulse_metadata_frontier_observe_enable =
                pulse_enable && (params.find<int>("pulse_metadata_frontier_observe_enable", 0) != 0);
            ocfg.pulse_metadata_frontier_top_items =
                std::max<uint32_t>(1u, params.find<uint32_t>("pulse_metadata_frontier_top_items", 32));
            ocfg.pulse_metadata_frontier_band_slots =
                std::max<uint32_t>(1u, params.find<uint32_t>("pulse_metadata_frontier_band_slots", 128));
            ocfg.pulse_mfb_preband_band_slots =
                params.find<uint32_t>("pulse_mfb_preband_band_slots", 0u);
            ocfg.pulse_metadata_seed_enable = false;
            ocfg.pulse_metadata_seed_top_bases =
                std::max<uint32_t>(1u, params.find<uint32_t>("pulse_metadata_seed_top_bases", 32));
            ocfg.pulse_metadata_seed_window_budget =
                params.find<uint32_t>("pulse_metadata_seed_window_budget", 0);
            ocfg.pulse_mfb_preband_seed_enable = false;
            ocfg.pulse_mfb_preband_top_bands =
                std::max<uint32_t>(1u, params.find<uint32_t>("pulse_mfb_preband_top_bands", 32));
            ocfg.pulse_mfb_preband_lines_per_band =
                std::max<uint32_t>(1u, params.find<uint32_t>("pulse_mfb_preband_lines_per_band", 4));
            ocfg.pulse_mfb_preband_window_budget =
                params.find<uint32_t>("pulse_mfb_preband_window_budget", 0);
            ocfg.pulse_mfb_gather_preband_enable = false;
            ocfg.pulse_mfb_gather_barrier_enable = false;
            ocfg.pulse_mfb_gather_top_bands =
                std::max<uint32_t>(1u, params.find<uint32_t>("pulse_mfb_gather_top_bands", 32));
            ocfg.pulse_mfb_gather_lines_per_band =
                std::max<uint32_t>(1u, params.find<uint32_t>("pulse_mfb_gather_lines_per_band", 4));
            ocfg.pulse_mfb_gather_window_budget =
                params.find<uint32_t>("pulse_mfb_gather_window_budget", 0);
            ocfg.pulse_mfb_gather_min_consumers =
                std::max<uint32_t>(2u, params.find<uint32_t>("pulse_mfb_gather_min_consumers", 2));
            ocfg.pulse_prebase_shared_lookup_enable =
                pulse_enable &&
                (params.find<int>("pulse_prebase_shared_lookup_enable", 0) != 0);
            ocfg.pe_internal_cpe_enable =
                params.find<int>("pe_internal_cpe_enable", 0) != 0;
            ocfg.pe_internal_pod_enable =
                ocfg.pe_internal_cpe_enable &&
                (params.find<int>("pe_internal_pod_enable", 0) != 0);
            ocfg.pe_internal_pod_metadata_enable =
                ocfg.pe_internal_pod_enable &&
                (params.find<int>("pe_internal_pod_metadata_enable", 0) != 0);
            ocfg.pe_internal_pod_owner_enable =
                ocfg.pe_internal_pod_enable &&
                (params.find<int>("pe_internal_pod_owner_enable", 0) != 0);
            {
                const uint32_t configured_pod_count =
                    params.find<uint32_t>("pe_internal_pod_count", 0);
                const uint32_t configured_pod_size =
                    params.find<uint32_t>("pe_internal_pod_size", 0);
                const uint32_t total_cores =
                    std::max<uint32_t>(1u, static_cast<uint32_t>(total_cores_));
                uint32_t pod_count = 1u;
                if (ocfg.pe_internal_pod_enable) {
                    if (configured_pod_count > 0u) {
                        pod_count = configured_pod_count;
                    } else if (configured_pod_size > 0u) {
                        pod_count =
                            static_cast<uint32_t>((total_cores + configured_pod_size - 1u) /
                                                  configured_pod_size);
                    } else {
                        pod_count = total_cores;
                    }
                    pod_count = std::max<uint32_t>(1u, std::min<uint32_t>(pod_count, total_cores));
                }
                uint32_t pod_size = configured_pod_size;
                if (pod_size == 0u) {
                    pod_size =
                        static_cast<uint32_t>((total_cores + pod_count - 1u) / pod_count);
                }
                pod_size = std::max<uint32_t>(1u, pod_size);
                ocfg.pe_internal_pod_count = pod_count;
                ocfg.pe_internal_pod_size = pod_size;
                ocfg.pe_internal_pod_id =
                    std::min<uint32_t>(ocfg.core_id / pod_size, pod_count - 1u);
            }
            ocfg.pulse_domain_retire_mode =
                params.find<std::string>("pulse_domain_retire_mode", "per_post");
            std::transform(
                ocfg.pulse_domain_retire_mode.begin(),
                ocfg.pulse_domain_retire_mode.end(),
                ocfg.pulse_domain_retire_mode.begin(),
                [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
            if (ocfg.pulse_domain_retire_mode != "descriptor_domain") {
                ocfg.pulse_domain_retire_mode = "per_post";
            }
            ocfg.pulse_domain_retire_release_budget =
                params.find<uint32_t>("pulse_domain_retire_release_budget", 0);
            ocfg.experimental_pre_window_profile_export_enable =
                params.find<int>("experimental_pre_window_profile_export_enable", 0) != 0;
            ocfg.experimental_pre_window_profile_export_dir =
                params.find<std::string>("experimental_pre_window_profile_export_dir", "");
            ocfg.bcsr_mgr = bcsr_weights_.get();
            mem->configureOrchestrator(std::move(ocfg));
            if (byte_exact_verify_enable_ && output_) {
                output_->verbose(CALL_INFO, 0, 0,
                    "[byte-exact] enabled=1 mode=%s row_scale=%u max_mismatch=%u node=%u core=%d\n",
                    byte_exact_verify_mode_.c_str(),
                    byte_exact_verify_row_scale_,
                    byte_exact_verify_max_mismatch_,
                    node_id_, core_id_);
            }
        }
        weight_mem_subsystem_ = mem.get();
        refreshSharedWeightObjectPlaneBinding_();
        // Phase E：BCSR 缓存容量配置下沉到 BcsrWeightManager
        uint32_t row_cap = bcsr_row_index_cache_cap_;
        const bool row_auto_fit = params.find<int>("bcsr_row_index_cache_auto_fit", 0) != 0;
        if (row_auto_fit && row_cap > 0 && bcsr_weights_) {
            const uint32_t br_eff = bcsr_weights_->effectiveBlockRows();
            const uint32_t n_block_rows =
                br_eff ? ((num_neurons_ + br_eff - 1u) / br_eff) : static_cast<uint32_t>(num_neurons_);
            if (n_block_rows > 0 && row_cap < n_block_rows) {
                if (window_read_debug_ && output_ && output_->getVerboseLevel() >= 2) {
                    output_->verbose(CALL_INFO, 2, 0,
                                     "[bcsr] auto-fit row_index_cache_cap %u -> %u (node=%u core=%d rows=%u br=%u)\n",
                                     row_cap,
                                     n_block_rows,
                                     node_id_,
                                     core_id_,
                                     num_neurons_,
                                     br_eff);
                }
                row_cap = n_block_rows;
            }
        }
        bcsr_weights_->setRowIndexCacheCapacity(row_cap);
        bcsr_weights_->setBlockCacheCapacity(bcsr_block_cache_cap_);
        {
            std::string pol = params.find<std::string>("bcsr_block_cache_policy", "lru");
            for (auto& ch : pol) {
                if (ch >= 'A' && ch <= 'Z') ch = static_cast<char>(ch - 'A' + 'a');
            }
            if (pol == "fifo") {
                bcsr_weights_->setBlockCachePolicy(BcsrWeightManager::BlockCachePolicy::FIFO);
            } else if (pol == "legacy_unordered" || pol == "legacy") {
                bcsr_weights_->setBlockCachePolicy(BcsrWeightManager::BlockCachePolicy::LegacyUnordered);
            } else {
                bcsr_weights_->setBlockCachePolicy(BcsrWeightManager::BlockCachePolicy::LRU);
            }
        }
        windowStateConfigure_();
        if (window_read_enable_) reserveWindowContainers_();
        weight_reader_adapter_ = std::move(mem);
    }
}

void SnnPESubComponent::init(unsigned int phase) {
    // 提前构建输出对象，避免在init早期使用output_时空指针
    if (!output_) {
        output_ = new Output("SnnPESubComponent[@p:@l]: ", verbose_, 0, Output::STDOUT);
    }
    // output_->verbose(CALL_INFO, 1, 0, "🔄 核心%d init phase %u\n", core_id_, phase);
    
    if (phase == 0) {
        // 初始化统计收集
        initializeStatistics();
        
        // 配置内存端口（可选，但不覆盖已设置的链接）
        if (!memory_link_) {
            memory_link_ = configureLink("mem_link");
            if (memory_link_) output_->verbose(CALL_INFO, 2, 0, "🔗 核心%d配置mem_link\n", core_id_);
        }
        
        initStdMemPhase0_();

        // Phase6/Phase4：workload runtime 绑定（平台：内存/NoC/parent；stream 额外绑定统计 sinks）。
        // Phase4 Task6.3：route/comm 装配已迁入 workload=snn，CoreShell 不再负责通信子系统装配。
        bindWorkloadRuntime_();

        // 权重验证所需的文件加载已下沉到 compute core（DefaultSnnComputeCore::initVerifyFile_）
    }

    // 将 init 相位转发给 StandardMem（通过 stdmem 端点转发）
    stdmem_ep_->init(phase);

    // Phase4 Task6.1：compute core init 下沉到 workload=snn（通过 onInitPhase 转发）。

    // Default weight initialization disabled, relying on WeightLoader
    if (phase == 4) {
        // 所有init阶段结束，允许后续时钟中发起访问
        memory_ready_ = true;
    }

    // Phase4：将生命周期相位转发给 workload（CoreShell 统一出口）。
    if (workload_) workload_->onInitPhase(phase);
    if (accel_runtime_services_) accel_runtime_services_->onInitPhase(phase);
}

void SnnPESubComponent::complete(unsigned int phase) {
    // 转发 complete 给 StandardMem：这是 memHierarchy init 握手的必要阶段（尤其当下游不是 Cache 而是 Bus/Dir）。
    stdmem_ep_->complete(phase);
}

void SnnPESubComponent::setup() {
    // output_->verbose(CALL_INFO, 1, 0, "🔧 核心%d setup 进入\n", core_id_);
    // output_->verbose(CALL_INFO, 1, 0,
    //     "🧩 参数: init_default_weight=%.3f, fallback=%d, merge_row=%d, merge_cl=%d, line=%uB, base_addr=%" PRIu64 ", N=%u\n",
    //     init_default_weight_, use_event_weight_fallback_, merge_read_row_, merge_read_cacheline_, line_size_bytes_, base_addr_, num_neurons_);
    
    // 验证组件状态
    if (!parent_pe_cached_) {
        output_->fatal(CALL_INFO, -1, "❌ 错误: 核心%d没有父级接口\n", core_id_);
    }
    // 注意：此处不直接发起内存访问，避免在setup阶段 MemLink 尚未建立时触发 memHierarchy fatal
    if (!stdmem_ep_ || !stdmem_ep_->available()) {
        // output_->verbose(CALL_INFO, 1, 0, "⚠️ 核心%d未配置StandardMem，检查是否有直接权重文件\n", core_id_);
        
        // 权重将由WeightLoader组件通过内存接口加载
        if (!weights_file_path_.empty()) {
            // output_->verbose(CALL_INFO, 1, 0, "🔧 核心%d权重文件路径: %s (将由WeightLoader加载)\n", core_id_, weights_file_path_.c_str());
        } else {
            output_->verbose(CALL_INFO, 1, 0, "⚠️ 核心%d未配置权重文件，将使用默认权重\n", core_id_);
        }
    }
    // 确保后端已构建（init阶段可能未加载到 StandardMem）

    // 打印映射模式与GAS端到端配置（一次性调试信息）
    {
        const char* idx_name = use_bcsr_ ? "bcsr_post_row" : (use_post_row_pre_col_ ? "post_row_pre_col" : "pre_row_post_col");
        int diag_lvl = window_read_debug_ ? 0 : 1;
        output_->verbose(CALL_INFO, diag_lvl, 0,
            "[GAS-Debug] core=%d index_mode=%s use_post_row_pre_col=%d apply_acc_enable=%d gas_enable=%d gas_window_mode=%d\n",
            core_id_, idx_name, use_post_row_pre_col_ ? 1 : 0, apply_acc_enable_ ? 1 : 0, gas_enable_ ? 1 : 0, gas_window_mode_ ? 1 : 0);
        output_->verbose(CALL_INFO, diag_lvl, 0,
            "[Init] core=%d global_base=%" PRIu64 " num_neurons=%u weights_cols=%u\n",
            core_id_, (uint64_t)global_neuron_base_, num_neurons_, weights_cols_);
    }
    // 配置一致性：启用窗口端到端语义时要求 window 模式的 GAS
    if (isNonSnnWorkload_()) {
        // Non-SNN workloads (stream/traffic) do not depend on GAS/Apply/Scatter.
        if (workload_) workload_->onSetup();
        if (accel_runtime_services_) accel_runtime_services_->onSetup();
        return;
    }
    if (apply_acc_enable_ && (!gas_enable_ || !gas_window_mode_)) {
        output_->fatal(CALL_INFO, -1, "❌ 配置错误：apply_acc_enable=1 需要 GAS 启用且 gas_window_mode=1 (window_auto)。\n");
    }
    // Phase4 Task6.1：compute core setup 下沉到 workload=snn。
    if (workload_) workload_->onSetup();
    if (accel_runtime_services_) accel_runtime_services_->onSetup();
    // output_->verbose(CALL_INFO, 1, 0, "✅ SnnPE SubComponent核心%d setup完成\n", core_id_);
}


bool SnnPESubComponent::clockTick(Cycle_t current_cycle) {
    return invokeWorkload_([&]() {
        (void)current_cycle; // 统一使用内部 cycle 计数，避免不同 SST 调度口径导致漂移
        total_cycles_++;
        const bool did = workload_->onClockTick(static_cast<uint64_t>(total_cycles_));
        // Phase10: active_cycles 由 workload 的返回值定义（SNN/stream 一致）。
        if (did) active_cycles_++;
        return false;
    });
}

std::unique_ptr<IWeightReader> SnnPESubComponent::takeWeightReader_() {
    return std::move(weight_reader_adapter_);
}

void SnnPESubComponent::orchestrateBeginGatherWindowSetup() {
    onStageBeginGatherCore_(curr_stage_seq_);
    if (impl_) impl_->markBeginGather(curr_stage_seq_);
}

void SnnPESubComponent::orchestratePrepareApplyWindow() {
    prepareEdgeWindowForApply_();
}

void SnnPESubComponent::orchestrateApplyWindowEntry() {
    onStageBeginApplyCore_(curr_stage_seq_);
    if (impl_) impl_->markBeginApply(curr_stage_seq_);
}

void SnnPESubComponent::orchestrateBeginApplyIssueFallback(bool strict_active) {
    issueFallbackReadsIfNeeded_(strict_active);
}

void SnnPESubComponent::orchestrateContinueIssueReads() {
    issueFromEdges_();
}

void SnnPESubComponent::orchestrateIssueFromEdgesDirect() {
    issueEdgeWeightFetches_();
}

void SnnPESubComponent::orchestrateBeginScatterSequence() {
    diag_spikes_stage_apply_ = 0;
    onStageEndApplyCore_(curr_stage_seq_);
    onStageBeginScatterCore_(curr_stage_seq_);
    clearFiredWindowCore_();
    if (impl_) impl_->markBeginScatter(curr_stage_seq_);
    spikes_generated_base_ = count_spikes_generated_;
    uint64_t spikes_emitted = applyAccumulatedWindowAndScatter_();
    if (spikes_emitted > 0) {
        if (auto* pe = parent_pe_cached_) pe->accumulateApplyScatterStats(0, 0, spikes_emitted, 0, 0, 0);
    }
}

void SnnPESubComponent::orchestrateEndScatterSequence() {
    uint64_t to_emit = window_spikes_all_ ? window_spikes_all_ : spikes_emitted_window_;
    if (to_emit == 0) {
        uint64_t delta = 0;
        if (count_spikes_generated_ >= spikes_generated_base_) delta = count_spikes_generated_ - spikes_generated_base_;
        if (delta > 0) to_emit = delta;
    }
    if (impl_) impl_->markEndScatter(curr_stage_seq_, to_emit);
    onStageEndScatterCore_(curr_stage_seq_, to_emit);
}

void SnnPESubComponent::resetMembraneState(float v_rest_value) {
    workload_->resetMembraneState(v_rest_value);
    if (accel_runtime_services_) {
        accel_runtime_services_->resetMembraneState(v_rest_value);
    }
    accReset_();
}

void SnnPESubComponent::setMemoryLink(SST::Link* link) {
    memory_link_ = link;
    
    // ★ 关键修正：直接使用提供的Link进行内存操作 ★
    if (memory_link_) {
        // output_->verbose(CALL_INFO, 2, 0, "🔗 核心%d设置内存连接成功\n", core_id_);
        memory_ready_ = true;  // 标记内存已准备就绪
    } else {
        output_->verbose(CALL_INFO, 2, 0, "🔗 核心%d设置内存连接失败 (link=nullptr)\n", core_id_);
        memory_ready_ = false;
    }
}

bool SnnPESubComponent::hasWork() const {
    return workload_ && workload_->hasWork();
}

double SnnPESubComponent::getUtilization() const {
    return workload_ ? workload_->getUtilization() : 0.0;
}

void SnnPESubComponent::getStatistics(std::map<std::string, uint64_t>& stats) const {
    if (workload_) {
        workload_->getStatistics(stats);
    }
    stats["riscv_snn_workload_selected"] = isRiscvSnnWorkload_() ? 1ull : 0ull;
}

void SnnPESubComponent::recordNeuronFires_(const std::vector<uint32_t>& neuron_indices) {
    // Phase4-Task6.3：统计口径仍锚定在 CoreShell；workload 只负责 route/comm 事务。
    for (uint32_t neuron_idx : neuron_indices) {
        if (stat_neurons_fired_) stat_neurons_fired_->addData(1);
        if (stat_spikes_generated_) stat_spikes_generated_->addData(1);
        count_neurons_fired_++;
        count_spikes_generated_++;
        if (apply_acc_enable_ && gas_window_mode_) {
            window_spikes_all_++;
        }
        if (neuron_idx < fired_ever_.size() && fired_ever_[neuron_idx] == 0) {
            fired_ever_[neuron_idx] = 1;
            if (auto* pe = parent_pe_cached_) {
                pe->accumulateUniqueNeuronFired(1);
            }
        }
    }
}

// === Learning writeback (called by compute core) ===
// applyLocalWeightUpdates_ 已拆分到 SnnPESubComponent_mem.cc

// requestWeight / handleMemoryResponse 已拆分到 SnnPESubComponent_mem.cc（StandardMem 控制面）

void SnnPESubComponent::verifyDenseAccumulator_(uint32_t seq) {
    if (acc_ops_) {
        acc_ops_->verifyDense(seq);
    }
}

// === Helpers implementations ===
// prepareDenseRead_ / issueReadCommon_ 已拆分到 SnnPESubComponent_mem.cc


// === BCSR 辅助实现 ===
// BCSR 读/缓存/诊断实现已拆分到 SnnPESubComponent_bcsr.cc
