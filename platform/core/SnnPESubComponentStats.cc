// -*- c++ -*-

#include <sst/core/sst_config.h>

#include "platform/core/SnnPESubComponent.h"
#include "platform/core/SnnPESubComponent_impl.h"
#include "api/ISnnAccelRuntimeServices.h"
#include "components/MultiCorePE.h"
#include "snn/synapse/gas/AccumulatorOps.h"
#include "snn/synapse/weights/WeightMemorySubsystem.h"

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>

using namespace SST;
using namespace SST::SnnDL;

void SnnPESubComponent::finish() {
    // 统计聚合（保持原路径）
    if (stat_pending_reqs_peak_) stat_pending_reqs_peak_->addData(pending_reqs_peak_);
    double avg_lat = (count_mem_responses_ > 0) ? ((double)accum_mem_latency_cycles_ / (double)count_mem_responses_) : 0.0;
    double utilization = (total_cycles_ > 0) ? (double)active_cycles_ / (double)total_cycles_ : 0.0;
    uint64_t core_state_sram_reads_total = 0;
    uint64_t core_state_sram_writes_total = 0;
    uint64_t core_state_sram_bytes_read_total = 0;
    uint64_t core_state_sram_bytes_write_total = 0;
    uint64_t core_state_sram_bank_conflict_ticks_total = 0;
    uint64_t core_state_sram_predicted_extra_cycles_total = 0;
    uint64_t core_state_sram_resident_bytes_peak = 0;
    uint64_t core_state_sram_bank_peak_accesses_per_tick = 0;
    uint64_t core_state_sram_energy_read_pj_total = 0;
    uint64_t core_state_sram_energy_write_pj_total = 0;
    uint64_t core_state_sram_stall_cycles_total = 0;
    const uint64_t riscv_snn_workload_selected = isRiscvSnnWorkload_() ? 1ull : 0ull;
    uint64_t riscv_snn_firmware_elf_present = 0;
    uint64_t riscv_snn_firmware_loaded = 0;
    uint64_t riscv_snn_backend_runtime_bridge = 0;
    uint64_t riscv_snn_firmware_started_count = 0;
    uint64_t riscv_snn_submitted_commands = 0;
    uint64_t riscv_snn_accepted_commands = 0;
    uint64_t riscv_snn_completion_visible_count = 0;
    uint64_t riscv_snn_completion_consumed_count = 0;
    uint64_t riscv_snn_fused_step_completion_count = 0;
    uint64_t riscv_snn_fault_count = 0;
    uint64_t riscv_snn_last_completion_status = 0;
    uint64_t riscv_snn_last_fault_csr = 0;
    uint64_t riscv_snn_backend_runtime_bridge_provider_bound = 0;
    if (stat_riscv_snn_workload_selected_) {
        stat_riscv_snn_workload_selected_->addData(riscv_snn_workload_selected);
    }
    if (!quiet_finish_logs_) {
        // 输出统计信息（使用内部计数器获得正确值）
        output_->verbose(CALL_INFO, 1, 0,
                         "[summary] core=%d spikes_recv=%" PRIu64 " spikes_gen=%" PRIu64 " neurons_fired=%" PRIu64 "\n",
                         core_id_, count_spikes_received_, count_spikes_generated_, count_neurons_fired_);
        if (verify_weights_) {
            uint64_t completed = 0;
            uint64_t mismatch = 0;
            if (compute_core_) {
                std::map<std::string, uint64_t> core_stats;
                compute_core_->getStatistics(core_stats);
                if (core_stats.count("core_verify_completed")) completed = core_stats["core_verify_completed"];
                if (core_stats.count("core_verify_mismatch_count")) mismatch = core_stats["core_verify_mismatch_count"];
            }
            output_->verbose(CALL_INFO, 1, 0,
                             "[summary] weight_verify completed=%" PRIu64 " mismatch=%" PRIu64 "\n",
                             completed, mismatch);
        }
        output_->verbose(CALL_INFO, 1, 0,
            "[summary] core=%d perf total_cycles=%" PRIu64 " active_cycles=%" PRIu64 " util=%.4f mem_req=%" PRIu64 " cache_hit=%" PRIu64 " cache_miss=%" PRIu64 " pending_peak=%u avg_mem_lat=%.2f\n",
            core_id_, total_cycles_, active_cycles_, utilization,
            count_memory_requests_, count_cache_hits_, count_cache_misses_, pending_reqs_peak_, avg_lat);
    }
    if (compute_core_) {
        std::map<std::string, uint64_t> core_stats;
        compute_core_->getStatistics(core_stats);
        auto get_core_u64 = [&core_stats](const char* key) -> uint64_t {
            if (!key) return 0;
            auto it = core_stats.find(key);
            if (it == core_stats.end()) return 0;
            return it->second;
        };
        auto add_core_stat = [&core_stats](const char* key, Statistic<uint64_t>* st) {
            if (!st || !key) return;
            auto it = core_stats.find(key);
            if (it == core_stats.end()) return;
            if (it->second == 0) return;
            st->addData(it->second);
        };
        core_state_sram_reads_total = get_core_u64("core_state_sram_reads_total");
        core_state_sram_writes_total = get_core_u64("core_state_sram_writes_total");
        core_state_sram_bytes_read_total = get_core_u64("core_state_sram_bytes_read_total");
        core_state_sram_bytes_write_total = get_core_u64("core_state_sram_bytes_write_total");
        core_state_sram_bank_conflict_ticks_total = get_core_u64("core_state_sram_bank_conflict_ticks_total");
        core_state_sram_predicted_extra_cycles_total = get_core_u64("core_state_sram_predicted_extra_cycles_total");
        core_state_sram_resident_bytes_peak = get_core_u64("core_state_sram_resident_bytes_peak");
        core_state_sram_bank_peak_accesses_per_tick = get_core_u64("core_state_sram_bank_peak_accesses_per_tick");
        core_state_sram_energy_read_pj_total = get_core_u64("core_state_sram_energy_read_pj_total");
        core_state_sram_energy_write_pj_total = get_core_u64("core_state_sram_energy_write_pj_total");
        core_state_sram_stall_cycles_total = get_core_u64("core_state_sram_stall_cycles_total");
        add_core_stat("core_state_sram_reads_total", stat_core_state_sram_reads_total_);
        add_core_stat("core_state_sram_writes_total", stat_core_state_sram_writes_total_);
        add_core_stat("core_state_sram_bytes_read_total", stat_core_state_sram_bytes_read_total_);
        add_core_stat("core_state_sram_bytes_write_total", stat_core_state_sram_bytes_write_total_);
        add_core_stat("core_state_sram_bank_conflict_ticks_total", stat_core_state_sram_bank_conflict_ticks_total_);
        add_core_stat("core_state_sram_predicted_extra_cycles_total", stat_core_state_sram_predicted_extra_cycles_total_);
        add_core_stat("core_state_sram_resident_bytes_peak", stat_core_state_sram_resident_bytes_peak_);
        add_core_stat("core_state_sram_bank_peak_accesses_per_tick", stat_core_state_sram_bank_peak_accesses_per_tick_);
        add_core_stat("core_state_sram_energy_read_pj_total", stat_core_state_sram_energy_read_pj_total_);
        add_core_stat("core_state_sram_energy_write_pj_total", stat_core_state_sram_energy_write_pj_total_);
        add_core_stat("core_state_sram_stall_cycles_total", stat_core_state_sram_stall_cycles_total_);
        if (workload_) {
            std::map<std::string, uint64_t> workload_stats;
            workload_->getStatistics(workload_stats);
            auto get_workload_u64 = [&workload_stats](const char* key) -> uint64_t {
                if (!key) return 0;
                auto it = workload_stats.find(key);
                if (it == workload_stats.end()) return 0;
                return it->second;
            };
            auto add_workload_stat_allow_zero = [&workload_stats](const char* key, Statistic<uint64_t>* st) {
                if (!st || !key) return;
                auto it = workload_stats.find(key);
                if (it == workload_stats.end()) return;
                st->addData(it->second);
            };
            riscv_snn_firmware_elf_present = get_workload_u64("riscv_snn_firmware_elf_present");
            riscv_snn_firmware_loaded = get_workload_u64("riscv_snn_firmware_loaded");
            riscv_snn_backend_runtime_bridge = get_workload_u64("riscv_snn_backend_runtime_bridge");
            riscv_snn_firmware_started_count = get_workload_u64("riscv_snn_firmware_started_count");
            riscv_snn_submitted_commands = get_workload_u64("riscv_snn_submitted_commands");
            riscv_snn_accepted_commands = get_workload_u64("riscv_snn_accepted_commands");
            riscv_snn_completion_visible_count = get_workload_u64("riscv_snn_completion_visible_count");
            riscv_snn_completion_consumed_count = get_workload_u64("riscv_snn_completion_consumed_count");
            riscv_snn_fused_step_completion_count = get_workload_u64("riscv_snn_fused_step_completion_count");
            riscv_snn_fault_count = get_workload_u64("riscv_snn_fault_count");
            riscv_snn_last_completion_status = get_workload_u64("riscv_snn_last_completion_status");
            riscv_snn_last_fault_csr = get_workload_u64("riscv_snn_last_fault_csr");
            riscv_snn_backend_runtime_bridge_provider_bound = get_workload_u64("riscv_snn_backend_runtime_bridge_provider_bound");
            add_workload_stat_allow_zero("riscv_snn_firmware_elf_present", stat_riscv_snn_firmware_elf_present_);
            add_workload_stat_allow_zero("riscv_snn_firmware_loaded", stat_riscv_snn_firmware_loaded_);
            add_workload_stat_allow_zero("riscv_snn_backend_runtime_bridge", stat_riscv_snn_backend_runtime_bridge_);
            add_workload_stat_allow_zero("riscv_snn_firmware_started_count", stat_riscv_snn_firmware_started_count_);
            add_workload_stat_allow_zero("riscv_snn_submitted_commands", stat_riscv_snn_submitted_commands_);
            add_workload_stat_allow_zero("riscv_snn_accepted_commands", stat_riscv_snn_accepted_commands_);
            add_workload_stat_allow_zero("riscv_snn_completion_visible_count", stat_riscv_snn_completion_visible_count_);
            add_workload_stat_allow_zero("riscv_snn_completion_consumed_count", stat_riscv_snn_completion_consumed_count_);
            add_workload_stat_allow_zero("riscv_snn_fused_step_completion_count", stat_riscv_snn_fused_step_completion_count_);
            add_workload_stat_allow_zero("riscv_snn_fault_count", stat_riscv_snn_fault_count_);
            add_workload_stat_allow_zero("riscv_snn_last_completion_status", stat_riscv_snn_last_completion_status_);
            add_workload_stat_allow_zero("riscv_snn_last_fault_csr", stat_riscv_snn_last_fault_csr_);
            add_workload_stat_allow_zero("riscv_snn_backend_runtime_bridge_provider_bound", stat_riscv_snn_backend_runtime_bridge_provider_bound_);
        }
    } else if (workload_) {
        std::map<std::string, uint64_t> core_stats;
        workload_->getStatistics(core_stats);
        auto get_core_u64 = [&core_stats](const char* key) -> uint64_t {
            if (!key) return 0;
            auto it = core_stats.find(key);
            if (it == core_stats.end()) return 0;
            return it->second;
        };
        auto add_core_stat = [&core_stats](const char* key, Statistic<uint64_t>* st) {
            if (!st || !key) return;
            auto it = core_stats.find(key);
            if (it == core_stats.end()) return;
            if (it->second == 0) return;
            st->addData(it->second);
        };
        auto add_core_stat_allow_zero = [&core_stats](const char* key, Statistic<uint64_t>* st) {
            if (!st || !key) return;
            auto it = core_stats.find(key);
            if (it == core_stats.end()) return;
            st->addData(it->second);
        };
        core_state_sram_reads_total = get_core_u64("core_state_sram_reads_total");
        core_state_sram_writes_total = get_core_u64("core_state_sram_writes_total");
        core_state_sram_bytes_read_total = get_core_u64("core_state_sram_bytes_read_total");
        core_state_sram_bytes_write_total = get_core_u64("core_state_sram_bytes_write_total");
        core_state_sram_bank_conflict_ticks_total = get_core_u64("core_state_sram_bank_conflict_ticks_total");
        core_state_sram_predicted_extra_cycles_total = get_core_u64("core_state_sram_predicted_extra_cycles_total");
        core_state_sram_resident_bytes_peak = get_core_u64("core_state_sram_resident_bytes_peak");
        core_state_sram_bank_peak_accesses_per_tick = get_core_u64("core_state_sram_bank_peak_accesses_per_tick");
        core_state_sram_energy_read_pj_total = get_core_u64("core_state_sram_energy_read_pj_total");
        core_state_sram_energy_write_pj_total = get_core_u64("core_state_sram_energy_write_pj_total");
        core_state_sram_stall_cycles_total = get_core_u64("core_state_sram_stall_cycles_total");
        add_core_stat("core_state_sram_reads_total", stat_core_state_sram_reads_total_);
        add_core_stat("core_state_sram_writes_total", stat_core_state_sram_writes_total_);
        add_core_stat("core_state_sram_bytes_read_total", stat_core_state_sram_bytes_read_total_);
        add_core_stat("core_state_sram_bytes_write_total", stat_core_state_sram_bytes_write_total_);
        add_core_stat("core_state_sram_bank_conflict_ticks_total", stat_core_state_sram_bank_conflict_ticks_total_);
        add_core_stat("core_state_sram_predicted_extra_cycles_total", stat_core_state_sram_predicted_extra_cycles_total_);
        add_core_stat("core_state_sram_resident_bytes_peak", stat_core_state_sram_resident_bytes_peak_);
        add_core_stat("core_state_sram_bank_peak_accesses_per_tick", stat_core_state_sram_bank_peak_accesses_per_tick_);
        add_core_stat("core_state_sram_energy_read_pj_total", stat_core_state_sram_energy_read_pj_total_);
        add_core_stat("core_state_sram_energy_write_pj_total", stat_core_state_sram_energy_write_pj_total_);
        add_core_stat("core_state_sram_stall_cycles_total", stat_core_state_sram_stall_cycles_total_);
        riscv_snn_firmware_elf_present = get_core_u64("riscv_snn_firmware_elf_present");
        riscv_snn_firmware_loaded = get_core_u64("riscv_snn_firmware_loaded");
        riscv_snn_backend_runtime_bridge = get_core_u64("riscv_snn_backend_runtime_bridge");
        riscv_snn_firmware_started_count = get_core_u64("riscv_snn_firmware_started_count");
        riscv_snn_submitted_commands = get_core_u64("riscv_snn_submitted_commands");
        riscv_snn_accepted_commands = get_core_u64("riscv_snn_accepted_commands");
        riscv_snn_completion_visible_count = get_core_u64("riscv_snn_completion_visible_count");
        riscv_snn_completion_consumed_count = get_core_u64("riscv_snn_completion_consumed_count");
        riscv_snn_fused_step_completion_count = get_core_u64("riscv_snn_fused_step_completion_count");
        riscv_snn_fault_count = get_core_u64("riscv_snn_fault_count");
        riscv_snn_last_completion_status = get_core_u64("riscv_snn_last_completion_status");
        riscv_snn_last_fault_csr = get_core_u64("riscv_snn_last_fault_csr");
        riscv_snn_backend_runtime_bridge_provider_bound = get_core_u64("riscv_snn_backend_runtime_bridge_provider_bound");
        add_core_stat_allow_zero("riscv_snn_firmware_elf_present", stat_riscv_snn_firmware_elf_present_);
        add_core_stat_allow_zero("riscv_snn_firmware_loaded", stat_riscv_snn_firmware_loaded_);
        add_core_stat_allow_zero("riscv_snn_backend_runtime_bridge", stat_riscv_snn_backend_runtime_bridge_);
        add_core_stat_allow_zero("riscv_snn_firmware_started_count", stat_riscv_snn_firmware_started_count_);
        add_core_stat_allow_zero("riscv_snn_submitted_commands", stat_riscv_snn_submitted_commands_);
        add_core_stat_allow_zero("riscv_snn_accepted_commands", stat_riscv_snn_accepted_commands_);
        add_core_stat_allow_zero("riscv_snn_completion_visible_count", stat_riscv_snn_completion_visible_count_);
        add_core_stat_allow_zero("riscv_snn_completion_consumed_count", stat_riscv_snn_completion_consumed_count_);
        add_core_stat_allow_zero("riscv_snn_fused_step_completion_count", stat_riscv_snn_fused_step_completion_count_);
        add_core_stat_allow_zero("riscv_snn_fault_count", stat_riscv_snn_fault_count_);
        add_core_stat_allow_zero("riscv_snn_last_completion_status", stat_riscv_snn_last_completion_status_);
        add_core_stat_allow_zero("riscv_snn_last_fault_csr", stat_riscv_snn_last_fault_csr_);
        add_core_stat_allow_zero("riscv_snn_backend_runtime_bridge_provider_bound", stat_riscv_snn_backend_runtime_bridge_provider_bound_);
        if (auto* mc_pe = snn_parent_observer_) {
            mc_pe->accumulateRiscvSnnRuntimeStats(
                riscv_snn_workload_selected,
                riscv_snn_firmware_elf_present,
                riscv_snn_firmware_loaded,
                riscv_snn_backend_runtime_bridge,
                riscv_snn_firmware_started_count,
                riscv_snn_submitted_commands,
                riscv_snn_accepted_commands,
                riscv_snn_completion_visible_count,
                riscv_snn_completion_consumed_count,
                riscv_snn_fused_step_completion_count,
                riscv_snn_fault_count,
                riscv_snn_last_completion_status,
                riscv_snn_last_fault_csr,
                riscv_snn_backend_runtime_bridge_provider_bound);
        }
    }

#ifdef SNNDL_ENABLE_PROFILING
    if (profiler_enabled_ && profiler_) {
        // Keep profiler output consistent with SnnDL logging (avoid raw std::cout).
        if (output_ && output_->getVerboseLevel() >= 1) {
            std::ostringstream oss;
            profiler_->generate_report(oss, 3.0);
            output_->verbose(CALL_INFO, 1, 0, "%s", oss.str().c_str());
        }
        // CSV 导出：prefix 优先；否则回退到工程级 analysis
        std::string csv = profiler_csv_prefix_.empty() ? std::string("analysis/profile_core") : profiler_csv_prefix_;
        csv += std::string("_c") + std::to_string(core_id_) + std::string(".csv");
        profiler_->export_csv(csv, 3.0);
    }
#endif
    // Phase4 Task6.1：compute core finish 下沉到 workload=snn。
    // Phase4-Task6.4：窗口读诊断与 BCSR 语义校验 marker 下沉到 workload=snn。
    // 若未加载 workload（legacy 路径），仍在控制层收尾阶段输出 marker，避免静默缺失。
    if (!workload_) {
        if (window_read_debug_ && weight_mem_subsystem_) {
            weight_mem_subsystem_->finishWindowDiag();
        }
        if (weight_mem_subsystem_) {
            weight_mem_subsystem_->finishSemanticVerify();
        }
    }
    if (workload_) workload_->onFinish();
    if (accel_runtime_services_) accel_runtime_services_->onFinish();

    if (weight_mem_subsystem_) {
        const auto exp_noc_rowidx = weight_mem_subsystem_->experimentalNocRowidxStats();
        if (stat_exp_noc_rowidx_prefetch_rows_total_) {
            stat_exp_noc_rowidx_prefetch_rows_total_->addData(exp_noc_rowidx.prefetch_rows_issued);
        }
        if (stat_exp_noc_rowidx_prefetch_bytes_total_) {
            stat_exp_noc_rowidx_prefetch_bytes_total_->addData(exp_noc_rowidx.prefetch_bytes_issued);
        }
        if (stat_exp_noc_rowidx_prefetch_rows_deferred_total_) {
            stat_exp_noc_rowidx_prefetch_rows_deferred_total_->addData(exp_noc_rowidx.prefetch_rows_deferred);
        }
        if (stat_exp_noc_rowidx_prefetch_rows_failed_total_) {
            stat_exp_noc_rowidx_prefetch_rows_failed_total_->addData(exp_noc_rowidx.prefetch_rows_failed);
        }
        if (stat_exp_noc_rowidx_cache_hits_total_) {
            stat_exp_noc_rowidx_cache_hits_total_->addData(exp_noc_rowidx.cache_hits);
        }
        if (stat_exp_noc_rowidx_cache_misses_total_) {
            stat_exp_noc_rowidx_cache_misses_total_->addData(exp_noc_rowidx.cache_misses);
        }
        if (stat_exp_noc_rowidx_cache_fills_total_) {
            stat_exp_noc_rowidx_cache_fills_total_->addData(exp_noc_rowidx.cache_fills);
        }
        if (stat_exp_noc_rowidx_cache_full_drop_total_) {
            stat_exp_noc_rowidx_cache_full_drop_total_->addData(exp_noc_rowidx.cache_full_drop);
        }
        if (stat_exp_noc_rowidx_cache_entries_final_) {
            stat_exp_noc_rowidx_cache_entries_final_->addData(exp_noc_rowidx.cache_entries);
        }
        if (stat_exp_noc_rowidx_touch_rows_total_) {
            stat_exp_noc_rowidx_touch_rows_total_->addData(exp_noc_rowidx.rows_touched_enqueued);
        }
        if (stat_exp_noc_rowidx_touch_events_total_) {
            stat_exp_noc_rowidx_touch_events_total_->addData(exp_noc_rowidx.touch_events_total);
        }
        if (stat_exp_noc_rowidx_rows_filtered_cold_total_) {
            stat_exp_noc_rowidx_rows_filtered_cold_total_->addData(exp_noc_rowidx.rows_filtered_cold);
        }
        if (stat_exp_noc_rowidx_carry_apply_pending_rows_total_) {
            stat_exp_noc_rowidx_carry_apply_pending_rows_total_->addData(
                exp_noc_rowidx.carry_apply_pending_rows_total);
        }
        if (stat_exp_noc_rowidx_drain_skip_phase_gather_total_) {
            stat_exp_noc_rowidx_drain_skip_phase_gather_total_->addData(
                exp_noc_rowidx.drain_skip_phase_gather_total);
        }
        if (stat_exp_noc_rowidx_drain_skip_phase_apply_disabled_total_) {
            stat_exp_noc_rowidx_drain_skip_phase_apply_disabled_total_->addData(
                exp_noc_rowidx.drain_skip_phase_apply_disabled_total);
        }
        if (stat_exp_noc_rowidx_drain_skip_no_pending_total_) {
            stat_exp_noc_rowidx_drain_skip_no_pending_total_->addData(
                exp_noc_rowidx.drain_skip_no_pending_total);
        }
        if (stat_exp_noc_rowidx_drain_skip_loader_not_ready_total_) {
            stat_exp_noc_rowidx_drain_skip_loader_not_ready_total_->addData(
                exp_noc_rowidx.drain_skip_loader_not_ready_total);
        }
        if (stat_exp_noc_rowidx_drain_skip_rowptr_not_ready_total_) {
            stat_exp_noc_rowidx_drain_skip_rowptr_not_ready_total_->addData(
                exp_noc_rowidx.drain_skip_rowptr_not_ready_total);
        }
        if (stat_exp_noc_rowidx_drain_skip_budget_zero_total_) {
            stat_exp_noc_rowidx_drain_skip_budget_zero_total_->addData(
                exp_noc_rowidx.drain_skip_budget_zero_total);
        }
        if (stat_exp_noc_rowidx_drain_skip_cache_hit_total_) {
            stat_exp_noc_rowidx_drain_skip_cache_hit_total_->addData(
                exp_noc_rowidx.drain_skip_cache_hit_total);
        }
        if (stat_exp_noc_rowidx_drain_skip_detached_inflight_total_) {
            stat_exp_noc_rowidx_drain_skip_detached_inflight_total_->addData(
                exp_noc_rowidx.drain_skip_detached_inflight_total);
        }
        if (stat_exp_noc_rowidx_drain_skip_colidx_inflight_total_) {
            stat_exp_noc_rowidx_drain_skip_colidx_inflight_total_->addData(
                exp_noc_rowidx.drain_skip_colidx_inflight_total);
        }
        if (stat_exp_noc_rowidx_drain_skip_empty_row_total_) {
            stat_exp_noc_rowidx_drain_skip_empty_row_total_->addData(
                exp_noc_rowidx.drain_skip_empty_row_total);
        }
        if (stat_exp_noc_rowidx_budget_ticks_total_) {
            stat_exp_noc_rowidx_budget_ticks_total_->addData(exp_noc_rowidx.budget_ticks_total);
        }
        if (stat_exp_noc_rowidx_budget_effective_total_) {
            stat_exp_noc_rowidx_budget_effective_total_->addData(exp_noc_rowidx.budget_effective_total);
        }
        if (stat_exp_noc_rowidx_budget_adapt_ticks_total_) {
            stat_exp_noc_rowidx_budget_adapt_ticks_total_->addData(exp_noc_rowidx.budget_adapt_ticks);
        }
        if (stat_exp_noc_rowidx_detached_demand_join_total_) {
            stat_exp_noc_rowidx_detached_demand_join_total_->addData(
                exp_noc_rowidx.detached_demand_join_total);
        }
        if (stat_exp_noc_rowidx_detached_demand_waiters_resolved_total_) {
            stat_exp_noc_rowidx_detached_demand_waiters_resolved_total_->addData(
                exp_noc_rowidx.detached_demand_waiters_resolved_total);
        }
        if (stat_exp_noc_rowidx_detached_demand_fallback_zero_total_) {
            stat_exp_noc_rowidx_detached_demand_fallback_zero_total_->addData(
                exp_noc_rowidx.detached_demand_fallback_zero_total);
        }
        if (stat_exp_noc_rowidx_detached_demand_ready_signal_total_) {
            stat_exp_noc_rowidx_detached_demand_ready_signal_total_->addData(
                exp_noc_rowidx.detached_demand_ready_signal_total);
        }
        if (stat_exp_noc_rowidx_detached_demand_ready_transition_total_) {
            stat_exp_noc_rowidx_detached_demand_ready_transition_total_->addData(
                exp_noc_rowidx.detached_demand_ready_transition_total);
        }
        const auto pulse_metadata_txn = weight_mem_subsystem_->pulseOsaMetadataTxnStats();
        if (auto* mc_pe = snn_parent_observer_) {
            mc_pe->recordPulseOsaMetadataTxnObservability(pulse_metadata_txn);
        }
        if (stat_pulse_metadata_txn_export_total_) {
            stat_pulse_metadata_txn_export_total_->addData(
                pulse_metadata_txn.export_total);
        }
        if (stat_pulse_metadata_txn_owner_launch_total_) {
            stat_pulse_metadata_txn_owner_launch_total_->addData(
                pulse_metadata_txn.owner_launch_total);
        }
        if (stat_pulse_metadata_txn_join_live_total_) {
            stat_pulse_metadata_txn_join_live_total_->addData(
                pulse_metadata_txn.join_live_total);
        }
        if (stat_pulse_metadata_txn_join_ready_total_) {
            stat_pulse_metadata_txn_join_ready_total_->addData(
                pulse_metadata_txn.join_ready_total);
        }
        if (stat_pulse_metadata_txn_late_join_total_) {
            stat_pulse_metadata_txn_late_join_total_->addData(
                pulse_metadata_txn.late_join_total);
        }
        if (stat_pulse_metadata_txn_ready_lease_hit_total_) {
            stat_pulse_metadata_txn_ready_lease_hit_total_->addData(
                pulse_metadata_txn.ready_lease_hit_total);
        }
        if (stat_pulse_metadata_txn_ready_lease_expired_total_) {
            stat_pulse_metadata_txn_ready_lease_expired_total_->addData(
                pulse_metadata_txn.ready_lease_expired_total);
        }
        if (stat_pulse_metadata_txn_envelope_size_sum_total_) {
            stat_pulse_metadata_txn_envelope_size_sum_total_->addData(
                pulse_metadata_txn.envelope_size_sum_total);
        }
        if (stat_pulse_metadata_frontier_observed_total_) {
            stat_pulse_metadata_frontier_observed_total_->addData(
                pulse_metadata_txn.frontier_observed_total);
        }
        if (stat_pulse_metadata_frontier_same_window_reobserve_total_) {
            stat_pulse_metadata_frontier_same_window_reobserve_total_->addData(
                pulse_metadata_txn.frontier_same_window_reobserve_total);
        }
        if (stat_pulse_metadata_frontier_owner_form_candidate_total_) {
            stat_pulse_metadata_frontier_owner_form_candidate_total_->addData(
                pulse_metadata_txn.frontier_owner_form_candidate_total);
        }
        if (stat_pulse_metadata_frontier_join_ready_candidate_total_) {
            stat_pulse_metadata_frontier_join_ready_candidate_total_->addData(
                pulse_metadata_txn.frontier_join_ready_candidate_total);
        }
        if (stat_pulse_metadata_frontier_premphf_base_observed_total_) {
            stat_pulse_metadata_frontier_premphf_base_observed_total_->addData(
                pulse_metadata_txn.frontier_premphf_base_observed_total);
        }
        if (stat_pulse_metadata_frontier_premphf_base_same_window_reobserve_total_) {
            stat_pulse_metadata_frontier_premphf_base_same_window_reobserve_total_->addData(
                pulse_metadata_txn.frontier_premphf_base_same_window_reobserve_total);
        }
        if (stat_pulse_metadata_frontier_premphf_base_owner_form_candidate_total_) {
            stat_pulse_metadata_frontier_premphf_base_owner_form_candidate_total_->addData(
                pulse_metadata_txn.frontier_premphf_base_owner_form_candidate_total);
        }
        if (stat_pulse_metadata_frontier_premphf_base_join_ready_candidate_total_) {
            stat_pulse_metadata_frontier_premphf_base_join_ready_candidate_total_->addData(
                pulse_metadata_txn.frontier_premphf_base_join_ready_candidate_total);
        }
        if (stat_pulse_metadata_frontier_premphf_band_observed_total_) {
            stat_pulse_metadata_frontier_premphf_band_observed_total_->addData(
                pulse_metadata_txn.frontier_premphf_band_observed_total);
        }
        if (stat_pulse_metadata_frontier_premphf_band_same_window_reobserve_total_) {
            stat_pulse_metadata_frontier_premphf_band_same_window_reobserve_total_->addData(
                pulse_metadata_txn.frontier_premphf_band_same_window_reobserve_total);
        }
        if (stat_pulse_metadata_frontier_premphf_band_owner_form_candidate_total_) {
            stat_pulse_metadata_frontier_premphf_band_owner_form_candidate_total_->addData(
                pulse_metadata_txn.frontier_premphf_band_owner_form_candidate_total);
        }
        if (stat_pulse_metadata_frontier_premphf_band_join_ready_candidate_total_) {
            stat_pulse_metadata_frontier_premphf_band_join_ready_candidate_total_->addData(
                pulse_metadata_txn.frontier_premphf_band_join_ready_candidate_total);
        }
        if (stat_pulse_metadata_frontier_idx2row_observed_total_) {
            stat_pulse_metadata_frontier_idx2row_observed_total_->addData(
                pulse_metadata_txn.frontier_idx2row_observed_total);
        }
        if (stat_pulse_metadata_frontier_idx2row_same_window_reobserve_total_) {
            stat_pulse_metadata_frontier_idx2row_same_window_reobserve_total_->addData(
                pulse_metadata_txn.frontier_idx2row_same_window_reobserve_total);
        }
        if (stat_pulse_metadata_frontier_idx2row_owner_form_candidate_total_) {
            stat_pulse_metadata_frontier_idx2row_owner_form_candidate_total_->addData(
                pulse_metadata_txn.frontier_idx2row_owner_form_candidate_total);
        }
        if (stat_pulse_metadata_frontier_idx2row_join_ready_candidate_total_) {
            stat_pulse_metadata_frontier_idx2row_join_ready_candidate_total_->addData(
                pulse_metadata_txn.frontier_idx2row_join_ready_candidate_total);
        }
        if (stat_pulse_metadata_frontier_rowindex_observed_total_) {
            stat_pulse_metadata_frontier_rowindex_observed_total_->addData(
                pulse_metadata_txn.frontier_rowindex_observed_total);
        }
        if (stat_pulse_metadata_frontier_rowindex_same_window_reobserve_total_) {
            stat_pulse_metadata_frontier_rowindex_same_window_reobserve_total_->addData(
                pulse_metadata_txn.frontier_rowindex_same_window_reobserve_total);
        }
        if (stat_pulse_metadata_frontier_rowindex_owner_form_candidate_total_) {
            stat_pulse_metadata_frontier_rowindex_owner_form_candidate_total_->addData(
                pulse_metadata_txn.frontier_rowindex_owner_form_candidate_total);
        }
        if (stat_pulse_metadata_frontier_rowindex_join_ready_candidate_total_) {
            stat_pulse_metadata_frontier_rowindex_join_ready_candidate_total_->addData(
                pulse_metadata_txn.frontier_rowindex_join_ready_candidate_total);
        }
        const auto atlas_census = weight_mem_subsystem_->experimentalPeAtlasObjectCensus();
        if (stat_atlas_census_premphf_base_frontier_events_total_) {
            stat_atlas_census_premphf_base_frontier_events_total_->addData(
                atlas_census.premphf_base.frontier_events_total);
        }
        if (stat_atlas_census_premphf_base_producer_events_total_) {
            stat_atlas_census_premphf_base_producer_events_total_->addData(
                atlas_census.premphf_base.producer_events_total);
        }
        if (stat_atlas_census_premphf_base_gate_events_total_) {
            stat_atlas_census_premphf_base_gate_events_total_->addData(
                atlas_census.premphf_base.gate_events_total);
        }
        if (stat_atlas_census_premphf_base_service_events_total_) {
            stat_atlas_census_premphf_base_service_events_total_->addData(
                atlas_census.premphf_base.service_events_total);
        }
        if (stat_atlas_census_premphf_band_frontier_events_total_) {
            stat_atlas_census_premphf_band_frontier_events_total_->addData(
                atlas_census.premphf_band.frontier_events_total);
        }
        if (stat_atlas_census_premphf_band_producer_events_total_) {
            stat_atlas_census_premphf_band_producer_events_total_->addData(
                atlas_census.premphf_band.producer_events_total);
        }
        if (stat_atlas_census_premphf_band_gate_events_total_) {
            stat_atlas_census_premphf_band_gate_events_total_->addData(
                atlas_census.premphf_band.gate_events_total);
        }
        if (stat_atlas_census_premphf_band_service_events_total_) {
            stat_atlas_census_premphf_band_service_events_total_->addData(
                atlas_census.premphf_band.service_events_total);
        }
        if (stat_atlas_census_idx2row_frontier_events_total_) {
            stat_atlas_census_idx2row_frontier_events_total_->addData(
                atlas_census.idx2row.frontier_events_total);
        }
        if (stat_atlas_census_idx2row_producer_events_total_) {
            stat_atlas_census_idx2row_producer_events_total_->addData(
                atlas_census.idx2row.producer_events_total);
        }
        if (stat_atlas_census_idx2row_gate_events_total_) {
            stat_atlas_census_idx2row_gate_events_total_->addData(
                atlas_census.idx2row.gate_events_total);
        }
        if (stat_atlas_census_idx2row_service_events_total_) {
            stat_atlas_census_idx2row_service_events_total_->addData(
                atlas_census.idx2row.service_events_total);
        }
        if (stat_atlas_census_rowindex_frontier_events_total_) {
            stat_atlas_census_rowindex_frontier_events_total_->addData(
                atlas_census.rowindex.frontier_events_total);
        }
        if (stat_atlas_census_rowindex_producer_events_total_) {
            stat_atlas_census_rowindex_producer_events_total_->addData(
                atlas_census.rowindex.producer_events_total);
        }
        if (stat_atlas_census_rowindex_gate_events_total_) {
            stat_atlas_census_rowindex_gate_events_total_->addData(
                atlas_census.rowindex.gate_events_total);
        }
        if (stat_atlas_census_rowindex_service_events_total_) {
            stat_atlas_census_rowindex_service_events_total_->addData(
                atlas_census.rowindex.service_events_total);
        }
        if (stat_atlas_census_rowdescriptor_frontier_events_total_) {
            stat_atlas_census_rowdescriptor_frontier_events_total_->addData(
                atlas_census.rowdescriptor.frontier_events_total);
        }
        if (stat_atlas_census_rowdescriptor_producer_events_total_) {
            stat_atlas_census_rowdescriptor_producer_events_total_->addData(
                atlas_census.rowdescriptor.producer_events_total);
        }
        if (stat_atlas_census_rowdescriptor_gate_events_total_) {
            stat_atlas_census_rowdescriptor_gate_events_total_->addData(
                atlas_census.rowdescriptor.gate_events_total);
        }
        if (stat_atlas_census_rowdescriptor_service_events_total_) {
            stat_atlas_census_rowdescriptor_service_events_total_->addData(
                atlas_census.rowdescriptor.service_events_total);
        }
        const auto atlas_rowindex_ledger =
            weight_mem_subsystem_->experimentalPeAtlasRowIndexLifecycleLedger();
        if (stat_atlas_proxy_rowindex_materialize_total_) {
            stat_atlas_proxy_rowindex_materialize_total_->addData(
                atlas_rowindex_ledger.materialize_total);
        }
        if (stat_atlas_proxy_rowindex_publicize_total_) {
            stat_atlas_proxy_rowindex_publicize_total_->addData(
                atlas_rowindex_ledger.publicize_total);
        }
        if (stat_atlas_proxy_rowindex_owner_form_total_) {
            stat_atlas_proxy_rowindex_owner_form_total_->addData(
                atlas_rowindex_ledger.owner_form_total);
        }
        if (stat_atlas_proxy_rowindex_join_live_total_) {
            stat_atlas_proxy_rowindex_join_live_total_->addData(
                atlas_rowindex_ledger.join_live_total);
        }
        if (stat_atlas_proxy_rowindex_join_ready_total_) {
            stat_atlas_proxy_rowindex_join_ready_total_->addData(
                atlas_rowindex_ledger.join_ready_total);
        }
        if (stat_atlas_proxy_rowindex_ready_total_) {
            stat_atlas_proxy_rowindex_ready_total_->addData(
                atlas_rowindex_ledger.ready_total);
        }
        if (stat_atlas_proxy_rowindex_release_total_) {
            stat_atlas_proxy_rowindex_release_total_->addData(
                atlas_rowindex_ledger.release_total);
        }
        if (stat_atlas_proxy_rowindex_release_missing_total_) {
            stat_atlas_proxy_rowindex_release_missing_total_->addData(
                atlas_rowindex_ledger.release_missing_total);
        }
        if (stat_atlas_proxy_rowindex_fallback_total_) {
            stat_atlas_proxy_rowindex_fallback_total_->addData(
                atlas_rowindex_ledger.fallback_total);
        }
        const auto atlas_idx2row_ledger =
            weight_mem_subsystem_->experimentalPeAtlasIdx2RowLifecycleLedger();
        if (stat_atlas_proxy_idx2row_materialize_total_) {
            stat_atlas_proxy_idx2row_materialize_total_->addData(
                atlas_idx2row_ledger.materialize_total);
        }
        if (stat_atlas_proxy_idx2row_publicize_total_) {
            stat_atlas_proxy_idx2row_publicize_total_->addData(
                atlas_idx2row_ledger.publicize_total);
        }
        if (stat_atlas_proxy_idx2row_owner_form_total_) {
            stat_atlas_proxy_idx2row_owner_form_total_->addData(
                atlas_idx2row_ledger.owner_form_total);
        }
        if (stat_atlas_proxy_idx2row_join_live_total_) {
            stat_atlas_proxy_idx2row_join_live_total_->addData(
                atlas_idx2row_ledger.join_live_total);
        }
        if (stat_atlas_proxy_idx2row_join_ready_total_) {
            stat_atlas_proxy_idx2row_join_ready_total_->addData(
                atlas_idx2row_ledger.join_ready_total);
        }
        if (stat_atlas_proxy_idx2row_ready_total_) {
            stat_atlas_proxy_idx2row_ready_total_->addData(
                atlas_idx2row_ledger.ready_total);
        }
        if (stat_atlas_proxy_idx2row_release_total_) {
            stat_atlas_proxy_idx2row_release_total_->addData(
                atlas_idx2row_ledger.release_total);
        }
        if (stat_atlas_proxy_idx2row_release_missing_total_) {
            stat_atlas_proxy_idx2row_release_missing_total_->addData(
                atlas_idx2row_ledger.release_missing_total);
        }
        if (stat_atlas_proxy_idx2row_fallback_total_) {
            stat_atlas_proxy_idx2row_fallback_total_->addData(
                atlas_idx2row_ledger.fallback_total);
        }
        const auto atlas_premphf_base_ledger =
            weight_mem_subsystem_->experimentalPeAtlasPreMphfBaseProxyLedger();
        if (stat_atlas_proxy_premphf_base_materialize_total_) {
            stat_atlas_proxy_premphf_base_materialize_total_->addData(
                atlas_premphf_base_ledger.materialize_total);
        }
        if (stat_atlas_proxy_premphf_base_publicize_total_) {
            stat_atlas_proxy_premphf_base_publicize_total_->addData(
                atlas_premphf_base_ledger.publicize_total);
        }
        if (stat_atlas_proxy_premphf_base_owner_form_total_) {
            stat_atlas_proxy_premphf_base_owner_form_total_->addData(
                atlas_premphf_base_ledger.owner_form_total);
        }
        if (stat_atlas_proxy_premphf_base_shared_hit_total_) {
            stat_atlas_proxy_premphf_base_shared_hit_total_->addData(
                atlas_premphf_base_ledger.shared_hit_total);
        }
        if (stat_atlas_proxy_premphf_base_lookup_ready_total_) {
            stat_atlas_proxy_premphf_base_lookup_ready_total_->addData(
                atlas_premphf_base_ledger.lookup_ready_total);
        }
        if (stat_atlas_proxy_premphf_base_proxy_only_gap_total_) {
            stat_atlas_proxy_premphf_base_proxy_only_gap_total_->addData(
                atlas_premphf_base_ledger.proxy_only_gap_total);
        }
        const auto atlas_premphf_band_ledger =
            weight_mem_subsystem_->experimentalPeAtlasPreMphfBandProxyLedger();
        const auto atlas_phase_ledger =
            weight_mem_subsystem_->experimentalPeAtlasPhaseStats();
        const auto atlas_pod_runtime =
            weight_mem_subsystem_->peInternalPodStats();
        if (stat_atlas_proxy_premphf_band_materialize_total_) {
            stat_atlas_proxy_premphf_band_materialize_total_->addData(
                atlas_premphf_band_ledger.materialize_total);
        }
        if (stat_atlas_proxy_premphf_band_publicize_total_) {
            stat_atlas_proxy_premphf_band_publicize_total_->addData(
                atlas_premphf_band_ledger.publicize_total);
        }
        if (stat_atlas_proxy_premphf_band_owner_form_candidate_total_) {
            stat_atlas_proxy_premphf_band_owner_form_candidate_total_->addData(
                atlas_premphf_band_ledger.owner_form_candidate_total);
        }
        if (stat_atlas_proxy_premphf_band_join_ready_candidate_total_) {
            stat_atlas_proxy_premphf_band_join_ready_candidate_total_->addData(
                atlas_premphf_band_ledger.join_ready_candidate_total);
        }
        if (stat_atlas_proxy_premphf_band_zero_service_total_) {
            stat_atlas_proxy_premphf_band_zero_service_total_->addData(
                atlas_premphf_band_ledger.zero_service_total);
        }
        if (auto* mc_pe = snn_parent_observer_) {
            mc_pe->recordAtlasObjectObservability(
                atlas_census,
                atlas_rowindex_ledger,
                atlas_idx2row_ledger,
                atlas_premphf_base_ledger,
                atlas_premphf_band_ledger,
                atlas_phase_ledger,
                atlas_pod_runtime);
        }
        weight_mem_subsystem_->flushSramObservability(static_cast<uint64_t>(total_cycles_));
        const auto weight_sram = weight_mem_subsystem_->sramObservabilityStats();
        if (stat_weight_idx_sram_reads_total_) {
            stat_weight_idx_sram_reads_total_->addData(weight_sram.idx_sram.reads_total);
        }
        if (stat_weight_idx_sram_writes_total_) {
            stat_weight_idx_sram_writes_total_->addData(weight_sram.idx_sram.writes_total);
        }
        if (stat_weight_idx_sram_bytes_read_total_) {
            stat_weight_idx_sram_bytes_read_total_->addData(weight_sram.idx_sram.bytes_read_total);
        }
        if (stat_weight_idx_sram_bytes_write_total_) {
            stat_weight_idx_sram_bytes_write_total_->addData(weight_sram.idx_sram.bytes_write_total);
        }
        if (stat_weight_idx_sram_bank_conflict_ticks_total_) {
            stat_weight_idx_sram_bank_conflict_ticks_total_->addData(weight_sram.idx_sram.bank_conflict_ticks_total);
        }
        if (stat_weight_idx_sram_predicted_extra_cycles_total_) {
            stat_weight_idx_sram_predicted_extra_cycles_total_->addData(weight_sram.idx_sram.predicted_extra_cycles_total);
        }
        if (stat_weight_idx_sram_resident_bytes_peak_) {
            stat_weight_idx_sram_resident_bytes_peak_->addData(weight_sram.idx_sram.resident_bytes_peak);
        }
        if (stat_weight_idx_sram_bank_peak_accesses_per_tick_) {
            stat_weight_idx_sram_bank_peak_accesses_per_tick_->addData(weight_sram.idx_sram.bank_peak_accesses_per_tick);
        }
        if (stat_weight_idx_sram_energy_read_pj_total_) {
            stat_weight_idx_sram_energy_read_pj_total_->addData(static_cast<uint64_t>(weight_sram.idx_sram.energy_read_pj_total));
        }
        if (stat_weight_idx_sram_energy_write_pj_total_) {
            stat_weight_idx_sram_energy_write_pj_total_->addData(static_cast<uint64_t>(weight_sram.idx_sram.energy_write_pj_total));
        }
        if (stat_weight_idx_lookup_total_) {
            stat_weight_idx_lookup_total_->addData(weight_sram.idx_lookup_total);
        }
        if (stat_weight_idx_lookup_idx2_total_) {
            stat_weight_idx_lookup_idx2_total_->addData(weight_sram.idx_lookup_idx2_total);
        }
        if (stat_weight_l0_sram_reads_total_) {
            stat_weight_l0_sram_reads_total_->addData(weight_sram.l0_sram.reads_total);
        }
        if (stat_weight_l0_sram_writes_total_) {
            stat_weight_l0_sram_writes_total_->addData(weight_sram.l0_sram.writes_total);
        }
        if (stat_weight_l0_sram_bytes_read_total_) {
            stat_weight_l0_sram_bytes_read_total_->addData(weight_sram.l0_sram.bytes_read_total);
        }
        if (stat_weight_l0_sram_bytes_write_total_) {
            stat_weight_l0_sram_bytes_write_total_->addData(weight_sram.l0_sram.bytes_write_total);
        }
        if (stat_weight_l0_sram_bank_conflict_ticks_total_) {
            stat_weight_l0_sram_bank_conflict_ticks_total_->addData(weight_sram.l0_sram.bank_conflict_ticks_total);
        }
        if (stat_weight_l0_sram_predicted_extra_cycles_total_) {
            stat_weight_l0_sram_predicted_extra_cycles_total_->addData(weight_sram.l0_sram.predicted_extra_cycles_total);
        }
        if (stat_weight_l0_sram_resident_bytes_peak_) {
            stat_weight_l0_sram_resident_bytes_peak_->addData(weight_sram.l0_sram.resident_bytes_peak);
        }
        if (stat_weight_l0_sram_bank_peak_accesses_per_tick_) {
            stat_weight_l0_sram_bank_peak_accesses_per_tick_->addData(weight_sram.l0_sram.bank_peak_accesses_per_tick);
        }
        if (stat_weight_l0_sram_energy_read_pj_total_) {
            stat_weight_l0_sram_energy_read_pj_total_->addData(static_cast<uint64_t>(weight_sram.l0_sram.energy_read_pj_total));
        }
        if (stat_weight_l0_sram_energy_write_pj_total_) {
            stat_weight_l0_sram_energy_write_pj_total_->addData(static_cast<uint64_t>(weight_sram.l0_sram.energy_write_pj_total));
        }
        if (stat_weight_sram_enforced_stall_cycles_total_) {
            stat_weight_sram_enforced_stall_cycles_total_->addData(weight_sram.enforced_stall_cycles_total);
        }
        if (stat_weight_l0_lookup_total_) {
            stat_weight_l0_lookup_total_->addData(weight_sram.l0_lookup_total);
        }
        if (stat_weight_l0_hit_total_) {
            stat_weight_l0_hit_total_->addData(weight_sram.l0_hit_total);
        }
        if (stat_weight_l0_fill_total_) {
            stat_weight_l0_fill_total_->addData(weight_sram.l0_fill_total);
        }
        if (stat_weight_l0_evict_total_) {
            stat_weight_l0_evict_total_->addData(weight_sram.l0_evict_total);
        }
        const auto gcss_lookup = weight_mem_subsystem_->gcssLookupStats();
        if (stat_gcss_lookup_hit_total_) {
            stat_gcss_lookup_hit_total_->addData(gcss_lookup.hit_total);
        }
        if (stat_gcss_lookup_miss_total_) {
            stat_gcss_lookup_miss_total_->addData(gcss_lookup.miss_total);
        }
        const auto read_source = weight_mem_subsystem_->readSourceStats();
        if (stat_weight_read_dense_reqs_total_) {
            stat_weight_read_dense_reqs_total_->addData(read_source.dense_reqs_total);
        }
        if (stat_weight_read_dense_bytes_total_) {
            stat_weight_read_dense_bytes_total_->addData(read_source.dense_bytes_total);
        }
        if (stat_weight_read_rowptr_reqs_total_) {
            stat_weight_read_rowptr_reqs_total_->addData(read_source.rowptr_reqs_total);
        }
        if (stat_weight_read_rowptr_bytes_total_) {
            stat_weight_read_rowptr_bytes_total_->addData(read_source.rowptr_bytes_total);
        }
        if (stat_weight_read_colidx_reqs_total_) {
            stat_weight_read_colidx_reqs_total_->addData(read_source.colidx_reqs_total);
        }
        if (stat_weight_read_colidx_bytes_total_) {
            stat_weight_read_colidx_bytes_total_->addData(read_source.colidx_bytes_total);
        }
        if (stat_weight_read_blockdata_reqs_total_) {
            stat_weight_read_blockdata_reqs_total_->addData(read_source.blockdata_reqs_total);
        }
        if (stat_weight_read_blockdata_bytes_total_) {
            stat_weight_read_blockdata_bytes_total_->addData(read_source.blockdata_bytes_total);
        }
        if (stat_weight_read_gcss_reqs_total_) {
            stat_weight_read_gcss_reqs_total_->addData(read_source.gcss_reqs_total);
        }
        if (stat_weight_read_gcss_bytes_total_) {
            stat_weight_read_gcss_bytes_total_->addData(read_source.gcss_bytes_total);
        }
        const auto retire_obs = weight_mem_subsystem_->retireObservabilityStats();
        if (stat_gas_retire_global_hol_cycles_total_) {
            stat_gas_retire_global_hol_cycles_total_->addData(retire_obs.global_hol_cycles_total);
        }
        if (stat_gas_retire_ready_but_blocked_edges_total_) {
            stat_gas_retire_ready_but_blocked_edges_total_->addData(retire_obs.ready_but_blocked_edges_total);
        }
        if (stat_gas_retire_per_post_progress_total_) {
            stat_gas_retire_per_post_progress_total_->addData(retire_obs.per_post_progress_total);
        }
        if (stat_gas_retire_samepost_blocked_edges_total_) {
            stat_gas_retire_samepost_blocked_edges_total_->addData(retire_obs.samepost_blocked_edges_total);
        }
        if (stat_gas_retire_crosspost_blocked_edges_total_) {
            stat_gas_retire_crosspost_blocked_edges_total_->addData(retire_obs.crosspost_blocked_edges_total);
        }
        if (stat_gas_retire_policy_loss_cycles_total_) {
            stat_gas_retire_policy_loss_cycles_total_->addData(retire_obs.policy_loss_cycles_total);
        }
        if (stat_gas_retire_policy_loss_edges_total_) {
            stat_gas_retire_policy_loss_edges_total_->addData(retire_obs.policy_loss_edges_total);
        }
        if (auto* pe = parent_pe_cached_) {
            ExperimentalNocRowidxPeStats rowidx_pe{};
            rowidx_pe.accumulateCore(
                exp_noc_rowidx.touch_events_total,
                exp_noc_rowidx.rows_touched_enqueued,
                exp_noc_rowidx.rows_filtered_cold,
                exp_noc_rowidx.prefetch_rows_issued,
                exp_noc_rowidx.prefetch_complete_inflight_miss_total,
                exp_noc_rowidx.prefetch_complete_zero_waiters_total,
                exp_noc_rowidx.prefetch_complete_waiters_total,
                exp_noc_rowidx.prefetch_rows_deferred,
                exp_noc_rowidx.prefetch_rows_failed,
                exp_noc_rowidx.prefetch_bytes_issued,
                exp_noc_rowidx.budget_ticks_total,
                exp_noc_rowidx.budget_effective_total,
                exp_noc_rowidx.budget_adapt_ticks,
                exp_noc_rowidx.cache_hits,
                exp_noc_rowidx.cache_misses,
                exp_noc_rowidx.cache_fills,
                exp_noc_rowidx.cache_full_drop,
                exp_noc_rowidx.cache_entries,
                exp_noc_rowidx.bulk_fill_total,
                exp_noc_rowidx.bulk_rows_cached_total,
                exp_noc_rowidx.bulk_waiters_resolved_total,
                exp_noc_rowidx.ready_transition_apply_promote_cached_total,
                exp_noc_rowidx.ready_signal_rowindex_response_total,
                exp_noc_rowidx.ready_transition_rowindex_response_total,
                exp_noc_rowidx.ready_signal_prefetch_response_total,
                exp_noc_rowidx.ready_transition_prefetch_response_total,
                exp_noc_rowidx.ready_signal_rowindex_response_inflight_waiters_total,
                exp_noc_rowidx.ready_transition_rowindex_response_inflight_waiters_total,
                exp_noc_rowidx.ready_signal_rowindex_response_inflight_zero_waiters_total,
                exp_noc_rowidx.ready_transition_rowindex_response_inflight_zero_waiters_total,
                exp_noc_rowidx.ready_signal_rowindex_response_noninflight_prefetch_only_total,
                exp_noc_rowidx.ready_transition_rowindex_response_noninflight_prefetch_only_total,
                exp_noc_rowidx.ready_signal_prefetch_response_inflight_waiters_total,
                exp_noc_rowidx.ready_transition_prefetch_response_inflight_waiters_total,
                exp_noc_rowidx.ready_signal_prefetch_response_inflight_zero_waiters_total,
                exp_noc_rowidx.ready_transition_prefetch_response_inflight_zero_waiters_total,
                exp_noc_rowidx.ready_signal_prefetch_response_noninflight_prefetch_only_total,
                exp_noc_rowidx.ready_transition_prefetch_response_noninflight_prefetch_only_total,
                exp_noc_rowidx.detached_demand_join_total,
                exp_noc_rowidx.detached_demand_waiters_resolved_total,
                exp_noc_rowidx.detached_demand_fallback_zero_total,
                exp_noc_rowidx.detached_demand_ready_signal_total,
                exp_noc_rowidx.detached_demand_ready_transition_total,
                exp_noc_rowidx.ready_bypass_experimental_cache_hit_total,
                exp_noc_rowidx.ready_bypass_rowindex_get_hit_total,
                exp_noc_rowidx.close_attempt_total,
                exp_noc_rowidx.close_attempt_active_owner_total,
                exp_noc_rowidx.close_attempt_already_pending_total,
                exp_noc_rowidx.close_attempt_not_active_total,
                exp_noc_rowidx.close_attempt_not_owner_total);
            auto* mc_pe = snn_parent_observer_;
            if (mc_pe) {
                mc_pe->accumulateExperimentalNocPrefetchStats(rowidx_pe);
            }
            pe->accumulateSynapseReadStats(
                gcss_lookup.hit_total,
                gcss_lookup.miss_total,
                read_source.dense_reqs_total,
                read_source.dense_bytes_total,
                read_source.rowptr_reqs_total,
                read_source.rowptr_bytes_total,
                read_source.colidx_reqs_total,
                read_source.colidx_bytes_total,
                read_source.blockdata_reqs_total,
                read_source.blockdata_bytes_total,
                read_source.gcss_reqs_total,
                read_source.gcss_bytes_total,
                weight_sram.idx_sram.reads_total,
                weight_sram.idx_sram.writes_total,
                weight_sram.idx_sram.bytes_read_total,
                weight_sram.idx_sram.bytes_write_total,
                weight_sram.idx_sram.bank_conflict_ticks_total,
                weight_sram.idx_sram.predicted_extra_cycles_total,
                weight_sram.idx_sram.resident_bytes_peak,
                weight_sram.idx_sram.bank_peak_accesses_per_tick,
                static_cast<uint64_t>(weight_sram.idx_sram.energy_read_pj_total),
                static_cast<uint64_t>(weight_sram.idx_sram.energy_write_pj_total),
                weight_sram.idx_lookup_total,
                weight_sram.idx_lookup_idx2_total,
                weight_sram.l0_sram.reads_total,
                weight_sram.l0_sram.writes_total,
                weight_sram.l0_sram.bytes_read_total,
                weight_sram.l0_sram.bytes_write_total,
                weight_sram.l0_sram.bank_conflict_ticks_total,
                weight_sram.l0_sram.predicted_extra_cycles_total,
                weight_sram.l0_sram.resident_bytes_peak,
                weight_sram.l0_sram.bank_peak_accesses_per_tick,
                static_cast<uint64_t>(weight_sram.l0_sram.energy_read_pj_total),
                static_cast<uint64_t>(weight_sram.l0_sram.energy_write_pj_total),
                weight_sram.enforced_stall_cycles_total,
                weight_sram.l0_lookup_total,
                weight_sram.l0_hit_total,
                weight_sram.l0_fill_total,
                weight_sram.l0_evict_total,
                core_state_sram_reads_total,
                core_state_sram_writes_total,
                core_state_sram_bytes_read_total,
                core_state_sram_bytes_write_total,
                core_state_sram_bank_conflict_ticks_total,
                core_state_sram_predicted_extra_cycles_total,
                core_state_sram_resident_bytes_peak,
                core_state_sram_bank_peak_accesses_per_tick,
                core_state_sram_energy_read_pj_total,
                core_state_sram_energy_write_pj_total,
                core_state_sram_stall_cycles_total,
                retire_obs.global_hol_cycles_total,
                retire_obs.ready_but_blocked_edges_total,
                retire_obs.per_post_progress_total,
                retire_obs.samepost_blocked_edges_total,
                retire_obs.crosspost_blocked_edges_total,
                retire_obs.policy_loss_cycles_total,
                retire_obs.policy_loss_edges_total,
                retire_obs.shadow_per_post_recoverable_cycles_total,
                retire_obs.shadow_per_post_recoverable_edges_total,
                retire_obs.shadow_per_post_ready_posts_peak,
                retire_obs.shadow_per_post_committable_edges_peak);
            if (mc_pe) {
                mc_pe->accumulateRiscvSnnRuntimeStats(
                    riscv_snn_workload_selected,
                    riscv_snn_firmware_elf_present,
                    riscv_snn_firmware_loaded,
                    riscv_snn_backend_runtime_bridge,
                    riscv_snn_firmware_started_count,
                    riscv_snn_submitted_commands,
                    riscv_snn_accepted_commands,
                    riscv_snn_completion_visible_count,
                    riscv_snn_completion_consumed_count,
                    riscv_snn_fused_step_completion_count,
                    riscv_snn_fault_count,
                    riscv_snn_last_completion_status,
                    riscv_snn_last_fault_csr,
                    riscv_snn_backend_runtime_bridge_provider_bound);
            }
        }

        const uint64_t exp_total_observed =
            exp_noc_rowidx.touch_events_total +
            exp_noc_rowidx.rows_touched_enqueued +
            exp_noc_rowidx.rows_filtered_cold +
            exp_noc_rowidx.prefetch_rows_issued +
            exp_noc_rowidx.prefetch_rows_deferred +
            exp_noc_rowidx.prefetch_rows_failed +
            exp_noc_rowidx.budget_ticks_total +
            exp_noc_rowidx.budget_effective_total +
            exp_noc_rowidx.budget_adapt_ticks +
            exp_noc_rowidx.cache_hits +
            exp_noc_rowidx.cache_misses +
            exp_noc_rowidx.cache_fills +
            exp_noc_rowidx.cache_full_drop +
            exp_noc_rowidx.detached_demand_join_total +
            exp_noc_rowidx.detached_demand_waiters_resolved_total +
            exp_noc_rowidx.detached_demand_fallback_zero_total +
            exp_noc_rowidx.detached_demand_ready_signal_total +
            exp_noc_rowidx.detached_demand_ready_transition_total;
        if (exp_total_observed > 0 && output_) {
            output_->verbose(
                CALL_INFO, 1, 0,
                "[exp-rowidx] core=%d touch=%" PRIu64 " touch_events=%" PRIu64
                " cold_filtered=%" PRIu64 " prefetch_rows=%" PRIu64
                " prefetch_bytes=%" PRIu64 " deferred=%" PRIu64 " failed=%" PRIu64
                " budget_ticks=%" PRIu64 " budget_eff=%" PRIu64 " budget_adapt_ticks=%" PRIu64
                " cache_hit=%" PRIu64 " cache_miss=%" PRIu64 " cache_fill=%" PRIu64
                " cache_drop=%" PRIu64 " cache_entries=%" PRIu64
                " carry_pending=%" PRIu64
                " drain_skip_gather=%" PRIu64
                " drain_skip_apply_disabled=%" PRIu64
                " drain_skip_no_pending=%" PRIu64
                " drain_skip_loader=%" PRIu64
                " drain_skip_rowptr=%" PRIu64
                " drain_skip_budget0=%" PRIu64
                " drain_skip_cache=%" PRIu64
                " drain_skip_detached_inflight=%" PRIu64
                " drain_skip_colidx_inflight=%" PRIu64
                " drain_skip_empty_row=%" PRIu64 "\n",
                core_id_,
                exp_noc_rowidx.rows_touched_enqueued,
                exp_noc_rowidx.touch_events_total,
                exp_noc_rowidx.rows_filtered_cold,
                exp_noc_rowidx.prefetch_rows_issued,
                exp_noc_rowidx.prefetch_bytes_issued,
                exp_noc_rowidx.prefetch_rows_deferred,
                exp_noc_rowidx.prefetch_rows_failed,
                exp_noc_rowidx.budget_ticks_total,
                exp_noc_rowidx.budget_effective_total,
                exp_noc_rowidx.budget_adapt_ticks,
                exp_noc_rowidx.cache_hits,
                exp_noc_rowidx.cache_misses,
                exp_noc_rowidx.cache_fills,
                exp_noc_rowidx.cache_full_drop,
                exp_noc_rowidx.cache_entries,
                exp_noc_rowidx.carry_apply_pending_rows_total,
                exp_noc_rowidx.drain_skip_phase_gather_total,
                exp_noc_rowidx.drain_skip_phase_apply_disabled_total,
                exp_noc_rowidx.drain_skip_no_pending_total,
                exp_noc_rowidx.drain_skip_loader_not_ready_total,
                exp_noc_rowidx.drain_skip_rowptr_not_ready_total,
                exp_noc_rowidx.drain_skip_budget_zero_total,
                exp_noc_rowidx.drain_skip_cache_hit_total,
                exp_noc_rowidx.drain_skip_detached_inflight_total,
                exp_noc_rowidx.drain_skip_colidx_inflight_total,
                exp_noc_rowidx.drain_skip_empty_row_total);
        }
        const auto atlas_rowindex =
            weight_mem_subsystem_->experimentalPeAtlasRowIndexLifecycleLedger();
        const auto pod_stats = weight_mem_subsystem_->peInternalPodStats();
        const uint64_t atlas_rowindex_total_observed =
            atlas_rowindex.materialize_total +
            atlas_rowindex.publicize_total +
            atlas_rowindex.owner_form_total +
            atlas_rowindex.join_live_total +
            atlas_rowindex.join_ready_total +
            atlas_rowindex.ready_total +
            atlas_rowindex.release_total +
            atlas_rowindex.release_deferred_total +
            atlas_rowindex.ready_release_total +
            atlas_rowindex.release_missing_total +
            atlas_rowindex.fallback_total;
        const uint64_t pod_guard_total_observed =
            pod_stats.guard_drop_total +
            pod_stats.guard_disabled_total +
            pod_stats.guard_missing_metadata_plane_total +
            pod_stats.guard_missing_owner_table_total +
            pod_stats.guard_zero_pod_count_total +
            pod_stats.guard_window_zero_total +
            pod_stats.guard_invalid_cfg_pod_total +
            pod_stats.guard_rowindex_total;
        const uint64_t pod_service_total_observed =
            pod_stats.frontier_export_total +
            pod_stats.owner_lookup_total +
            pod_stats.owner_alloc_total +
            pod_stats.owner_hit_total +
            pod_stats.owner_reject_total +
            pod_stats.join_request_total +
            pod_stats.join_grant_total +
            pod_stats.join_reject_total +
            pod_stats.service_join_live_total +
            pod_stats.service_join_ready_total +
            pod_stats.service_ready_transition_total +
            pod_stats.service_ready_fanout_total +
            pod_stats.service_release_deferred_total +
            pod_stats.service_ready_release_total +
            pod_stats.service_late_join_total +
            pod_stats.service_potential_private_service_elide_total +
            pod_stats.fallback_private_issue_total;
        if ((atlas_rowindex_total_observed > 0 ||
             pod_guard_total_observed > 0 ||
             pod_service_total_observed > 0) &&
            output_) {
            output_->verbose(
                CALL_INFO, 1, 0,
                "[atlas-rowidx] core=%d materialize=%" PRIu64
                " publicize=%" PRIu64 " owner_form=%" PRIu64
                " join_live=%" PRIu64 " join_ready=%" PRIu64
                " ready=%" PRIu64 " release=%" PRIu64
                " release_deferred=%" PRIu64
                " ready_release=%" PRIu64
                " release_missing=%" PRIu64 " fallback=%" PRIu64
                " guard_drop=%" PRIu64 " guard_disabled=%" PRIu64
                " guard_missing_meta=%" PRIu64 " guard_missing_owner=%" PRIu64
                " guard_zero_pod=%" PRIu64 " guard_window_zero=%" PRIu64
                " guard_invalid_pod=%" PRIu64 " guard_rowindex=%" PRIu64
                " frontier=%" PRIu64 " owner_lookup=%" PRIu64
                " owner_alloc=%" PRIu64 " owner_hit=%" PRIu64
                " owner_reject=%" PRIu64 " join_req=%" PRIu64
                " join_grant=%" PRIu64 " join_reject=%" PRIu64
                " svc_live=%" PRIu64 " svc_ready=%" PRIu64
                " svc_ready_tx=%" PRIu64 " svc_ready_fanout=%" PRIu64
                " svc_late=%" PRIu64 " svc_elide=%" PRIu64 "\n",
                core_id_,
                atlas_rowindex.materialize_total,
                atlas_rowindex.publicize_total,
                atlas_rowindex.owner_form_total,
                atlas_rowindex.join_live_total,
                atlas_rowindex.join_ready_total,
                atlas_rowindex.ready_total,
                atlas_rowindex.release_total,
                atlas_rowindex.release_deferred_total,
                atlas_rowindex.ready_release_total,
                atlas_rowindex.release_missing_total,
                atlas_rowindex.fallback_total,
                pod_stats.guard_drop_total,
                pod_stats.guard_disabled_total,
                pod_stats.guard_missing_metadata_plane_total,
                pod_stats.guard_missing_owner_table_total,
                pod_stats.guard_zero_pod_count_total,
                pod_stats.guard_window_zero_total,
                pod_stats.guard_invalid_cfg_pod_total,
                pod_stats.guard_rowindex_total,
                pod_stats.frontier_export_total,
                pod_stats.owner_lookup_total,
                pod_stats.owner_alloc_total,
                pod_stats.owner_hit_total,
                pod_stats.owner_reject_total,
                pod_stats.join_request_total,
                pod_stats.join_grant_total,
                pod_stats.join_reject_total,
                pod_stats.service_join_live_total,
                pod_stats.service_join_ready_total,
                pod_stats.service_ready_transition_total,
                pod_stats.service_ready_fanout_total,
                pod_stats.service_late_join_total,
                pod_stats.service_potential_private_service_elide_total);
        }
    }
}


void SnnPESubComponent::initializeStatistics() {
    // output_->verbose(CALL_INFO, 2, 0, "📊 核心%d初始化统计收集\n", core_id_);
    
    stat_spikes_received_ = registerStatistic<uint64_t>("spikes_received");
    stat_spikes_generated_ = registerStatistic<uint64_t>("spikes_generated");
    stat_neurons_fired_ = registerStatistic<uint64_t>("neurons_fired");
    stat_memory_requests_ = registerStatistic<uint64_t>("memory_requests");
    stat_weight_cache_hits_ = registerStatistic<uint64_t>("weight_cache_hits");
    stat_weight_cache_misses_ = registerStatistic<uint64_t>("weight_cache_misses");
    stat_merged_reads_rows_ = registerStatistic<uint64_t>("merged_reads_rows");
    stat_merged_reads_cls_ = registerStatistic<uint64_t>("merged_reads_cls");
    stat_weights_verify_count_ = registerStatistic<uint64_t>("weights_verify_count");
    stat_weights_mismatch_count_ = registerStatistic<uint64_t>("weights_mismatch_count");
    stat_weights_verify_sum_ = registerStatistic<double>("weights_verify_sum");
    // 扩展统计
    stat_routes_entries_ = registerStatistic<uint64_t>("routes_entries");
    stat_fanout_per_spike_ = registerStatistic<uint64_t>("fanout_per_spike");
    stat_route3d_native_activation_total_ = registerStatistic<uint64_t>("route3d_native_activation_total");
    stat_route3d_native_gating_activation_total_ = registerStatistic<uint64_t>("route3d_native_gating_activation_total");
    stat_route3d_native_direct_activation_total_ = registerStatistic<uint64_t>("route3d_native_direct_activation_total");
    stat_route3d_native_unique_sources_total_ = registerStatistic<uint64_t>("route3d_native_unique_sources_total");
    stat_cache_evictions_ = registerStatistic<uint64_t>("cache_evictions");
    stat_pending_reqs_peak_ = registerStatistic<uint64_t>("pending_reqs_peak");
    stat_cycles_update_neuron_ = registerStatistic<uint64_t>("cycles_update_neuron");
    stat_synaptic_accesses_ = registerStatistic<uint64_t>("synaptic_accesses");
    // 门控诊断：权重读请求发起次数
    stat_weight_read_requests_ = registerStatistic<uint64_t>("weight_read_requests");
    // GAS totals accumulated from GatherBufferIF via CustomResp
    stat_gas_unique_reads_total_ = registerStatistic<uint64_t>("gas_unique_reads_total");
    stat_gas_unique_bytes_total_ = registerStatistic<uint64_t>("gas_unique_bytes_total");
    stat_gas_row_window_triggers_total_ = registerStatistic<uint64_t>("gas_row_window_triggers_total");
    stat_gas_row_window_bytes_total_ = registerStatistic<uint64_t>("gas_row_window_bytes_total");
    stat_gas_bursts_total_ = registerStatistic<uint64_t>("gas_bursts_total");
    stat_gas_payload_bytes_total_ = registerStatistic<uint64_t>("gas_payload_bytes_total");
    stat_gas_gap_absorbed_bytes_total_ = registerStatistic<uint64_t>("gas_gap_absorbed_bytes_total");
    stat_exp_noc_rowidx_prefetch_rows_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_prefetch_rows_total");
    stat_exp_noc_rowidx_prefetch_bytes_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_prefetch_bytes_total");
    stat_exp_noc_rowidx_prefetch_rows_deferred_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_prefetch_rows_deferred_total");
    stat_exp_noc_rowidx_prefetch_rows_failed_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_prefetch_rows_failed_total");
    stat_exp_noc_rowidx_cache_hits_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_cache_hits_total");
    stat_exp_noc_rowidx_cache_misses_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_cache_misses_total");
    stat_exp_noc_rowidx_cache_fills_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_cache_fills_total");
    stat_exp_noc_rowidx_cache_full_drop_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_cache_full_drop_total");
    stat_exp_noc_rowidx_cache_entries_final_ = registerStatistic<uint64_t>("exp_noc_rowidx_cache_entries_final");
    stat_exp_noc_rowidx_touch_rows_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_touch_rows_total");
    stat_exp_noc_rowidx_touch_events_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_touch_events_total");
    stat_exp_noc_rowidx_rows_filtered_cold_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_rows_filtered_cold_total");
    stat_exp_noc_rowidx_carry_apply_pending_rows_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_carry_apply_pending_rows_total");
    stat_exp_noc_rowidx_drain_skip_phase_gather_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_drain_skip_phase_gather_total");
    stat_exp_noc_rowidx_drain_skip_phase_apply_disabled_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_drain_skip_phase_apply_disabled_total");
    stat_exp_noc_rowidx_drain_skip_no_pending_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_drain_skip_no_pending_total");
    stat_exp_noc_rowidx_drain_skip_loader_not_ready_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_drain_skip_loader_not_ready_total");
    stat_exp_noc_rowidx_drain_skip_rowptr_not_ready_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_drain_skip_rowptr_not_ready_total");
    stat_exp_noc_rowidx_drain_skip_budget_zero_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_drain_skip_budget_zero_total");
    stat_exp_noc_rowidx_drain_skip_cache_hit_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_drain_skip_cache_hit_total");
    stat_exp_noc_rowidx_drain_skip_detached_inflight_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_drain_skip_detached_inflight_total");
    stat_exp_noc_rowidx_drain_skip_colidx_inflight_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_drain_skip_colidx_inflight_total");
    stat_exp_noc_rowidx_drain_skip_empty_row_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_drain_skip_empty_row_total");
    stat_exp_noc_rowidx_budget_ticks_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_budget_ticks_total");
    stat_exp_noc_rowidx_budget_effective_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_budget_effective_total");
    stat_exp_noc_rowidx_budget_adapt_ticks_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_budget_adapt_ticks_total");
    stat_exp_noc_rowidx_detached_demand_join_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_detached_demand_join_total");
    stat_exp_noc_rowidx_detached_demand_waiters_resolved_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_detached_demand_waiters_resolved_total");
    stat_exp_noc_rowidx_detached_demand_fallback_zero_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_detached_demand_fallback_zero_total");
    stat_exp_noc_rowidx_detached_demand_ready_signal_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_detached_demand_ready_signal_total");
    stat_exp_noc_rowidx_detached_demand_ready_transition_total_ = registerStatistic<uint64_t>("exp_noc_rowidx_detached_demand_ready_transition_total");
    stat_pulse_metadata_txn_export_total_ = registerStatistic<uint64_t>("pulse_metadata_txn_export_total");
    stat_pulse_metadata_txn_owner_launch_total_ = registerStatistic<uint64_t>("pulse_metadata_txn_owner_launch_total");
    stat_pulse_metadata_txn_join_live_total_ = registerStatistic<uint64_t>("pulse_metadata_txn_join_live_total");
    stat_pulse_metadata_txn_join_ready_total_ = registerStatistic<uint64_t>("pulse_metadata_txn_join_ready_total");
    stat_pulse_metadata_txn_late_join_total_ = registerStatistic<uint64_t>("pulse_metadata_txn_late_join_total");
    stat_pulse_metadata_txn_ready_lease_hit_total_ = registerStatistic<uint64_t>("pulse_metadata_txn_ready_lease_hit_total");
    stat_pulse_metadata_txn_ready_lease_expired_total_ = registerStatistic<uint64_t>("pulse_metadata_txn_ready_lease_expired_total");
    stat_pulse_metadata_txn_envelope_size_sum_total_ = registerStatistic<uint64_t>("pulse_metadata_txn_envelope_size_sum_total");
    stat_pulse_metadata_frontier_observed_total_ = registerStatistic<uint64_t>("pulse_metadata_frontier_observed_total");
    stat_pulse_metadata_frontier_same_window_reobserve_total_ = registerStatistic<uint64_t>("pulse_metadata_frontier_same_window_reobserve_total");
    stat_pulse_metadata_frontier_owner_form_candidate_total_ = registerStatistic<uint64_t>("pulse_metadata_frontier_owner_form_candidate_total");
    stat_pulse_metadata_frontier_join_ready_candidate_total_ = registerStatistic<uint64_t>("pulse_metadata_frontier_join_ready_candidate_total");
    stat_pulse_metadata_frontier_premphf_base_observed_total_ = registerStatistic<uint64_t>("pulse_metadata_frontier_premphf_base_observed_total");
    stat_pulse_metadata_frontier_premphf_base_same_window_reobserve_total_ = registerStatistic<uint64_t>("pulse_metadata_frontier_premphf_base_same_window_reobserve_total");
    stat_pulse_metadata_frontier_premphf_base_owner_form_candidate_total_ = registerStatistic<uint64_t>("pulse_metadata_frontier_premphf_base_owner_form_candidate_total");
    stat_pulse_metadata_frontier_premphf_base_join_ready_candidate_total_ = registerStatistic<uint64_t>("pulse_metadata_frontier_premphf_base_join_ready_candidate_total");
    stat_pulse_metadata_frontier_premphf_band_observed_total_ = registerStatistic<uint64_t>("pulse_metadata_frontier_premphf_band_observed_total");
    stat_pulse_metadata_frontier_premphf_band_same_window_reobserve_total_ = registerStatistic<uint64_t>("pulse_metadata_frontier_premphf_band_same_window_reobserve_total");
    stat_pulse_metadata_frontier_premphf_band_owner_form_candidate_total_ = registerStatistic<uint64_t>("pulse_metadata_frontier_premphf_band_owner_form_candidate_total");
    stat_pulse_metadata_frontier_premphf_band_join_ready_candidate_total_ = registerStatistic<uint64_t>("pulse_metadata_frontier_premphf_band_join_ready_candidate_total");
    stat_pulse_metadata_frontier_idx2row_observed_total_ = registerStatistic<uint64_t>("pulse_metadata_frontier_idx2row_observed_total");
    stat_pulse_metadata_frontier_idx2row_same_window_reobserve_total_ = registerStatistic<uint64_t>("pulse_metadata_frontier_idx2row_same_window_reobserve_total");
    stat_pulse_metadata_frontier_idx2row_owner_form_candidate_total_ = registerStatistic<uint64_t>("pulse_metadata_frontier_idx2row_owner_form_candidate_total");
    stat_pulse_metadata_frontier_idx2row_join_ready_candidate_total_ = registerStatistic<uint64_t>("pulse_metadata_frontier_idx2row_join_ready_candidate_total");
    stat_pulse_metadata_frontier_rowindex_observed_total_ = registerStatistic<uint64_t>("pulse_metadata_frontier_rowindex_observed_total");
    stat_pulse_metadata_frontier_rowindex_same_window_reobserve_total_ = registerStatistic<uint64_t>("pulse_metadata_frontier_rowindex_same_window_reobserve_total");
    stat_pulse_metadata_frontier_rowindex_owner_form_candidate_total_ = registerStatistic<uint64_t>("pulse_metadata_frontier_rowindex_owner_form_candidate_total");
    stat_pulse_metadata_frontier_rowindex_join_ready_candidate_total_ = registerStatistic<uint64_t>("pulse_metadata_frontier_rowindex_join_ready_candidate_total");
    stat_atlas_census_premphf_base_frontier_events_total_ = registerStatistic<uint64_t>("atlas_census_premphf_base_frontier_events_total");
    stat_atlas_census_premphf_base_producer_events_total_ = registerStatistic<uint64_t>("atlas_census_premphf_base_producer_events_total");
    stat_atlas_census_premphf_base_gate_events_total_ = registerStatistic<uint64_t>("atlas_census_premphf_base_gate_events_total");
    stat_atlas_census_premphf_base_service_events_total_ = registerStatistic<uint64_t>("atlas_census_premphf_base_service_events_total");
    stat_atlas_census_premphf_band_frontier_events_total_ = registerStatistic<uint64_t>("atlas_census_premphf_band_frontier_events_total");
    stat_atlas_census_premphf_band_producer_events_total_ = registerStatistic<uint64_t>("atlas_census_premphf_band_producer_events_total");
    stat_atlas_census_premphf_band_gate_events_total_ = registerStatistic<uint64_t>("atlas_census_premphf_band_gate_events_total");
    stat_atlas_census_premphf_band_service_events_total_ = registerStatistic<uint64_t>("atlas_census_premphf_band_service_events_total");
    stat_atlas_census_idx2row_frontier_events_total_ = registerStatistic<uint64_t>("atlas_census_idx2row_frontier_events_total");
    stat_atlas_census_idx2row_producer_events_total_ = registerStatistic<uint64_t>("atlas_census_idx2row_producer_events_total");
    stat_atlas_census_idx2row_gate_events_total_ = registerStatistic<uint64_t>("atlas_census_idx2row_gate_events_total");
    stat_atlas_census_idx2row_service_events_total_ = registerStatistic<uint64_t>("atlas_census_idx2row_service_events_total");
    stat_atlas_census_rowindex_frontier_events_total_ = registerStatistic<uint64_t>("atlas_census_rowindex_frontier_events_total");
    stat_atlas_census_rowindex_producer_events_total_ = registerStatistic<uint64_t>("atlas_census_rowindex_producer_events_total");
    stat_atlas_census_rowindex_gate_events_total_ = registerStatistic<uint64_t>("atlas_census_rowindex_gate_events_total");
    stat_atlas_census_rowindex_service_events_total_ = registerStatistic<uint64_t>("atlas_census_rowindex_service_events_total");
    stat_atlas_census_rowdescriptor_frontier_events_total_ = registerStatistic<uint64_t>("atlas_census_rowdescriptor_frontier_events_total");
    stat_atlas_census_rowdescriptor_producer_events_total_ = registerStatistic<uint64_t>("atlas_census_rowdescriptor_producer_events_total");
    stat_atlas_census_rowdescriptor_gate_events_total_ = registerStatistic<uint64_t>("atlas_census_rowdescriptor_gate_events_total");
    stat_atlas_census_rowdescriptor_service_events_total_ = registerStatistic<uint64_t>("atlas_census_rowdescriptor_service_events_total");
    stat_atlas_proxy_rowindex_materialize_total_ = registerStatistic<uint64_t>("atlas_proxy_rowindex_materialize_total");
    stat_atlas_proxy_rowindex_publicize_total_ = registerStatistic<uint64_t>("atlas_proxy_rowindex_publicize_total");
    stat_atlas_proxy_rowindex_owner_form_total_ = registerStatistic<uint64_t>("atlas_proxy_rowindex_owner_form_total");
    stat_atlas_proxy_rowindex_join_live_total_ = registerStatistic<uint64_t>("atlas_proxy_rowindex_join_live_total");
    stat_atlas_proxy_rowindex_join_ready_total_ = registerStatistic<uint64_t>("atlas_proxy_rowindex_join_ready_total");
    stat_atlas_proxy_rowindex_ready_total_ = registerStatistic<uint64_t>("atlas_proxy_rowindex_ready_total");
    stat_atlas_proxy_rowindex_release_total_ = registerStatistic<uint64_t>("atlas_proxy_rowindex_release_total");
    stat_atlas_proxy_rowindex_release_missing_total_ = registerStatistic<uint64_t>("atlas_proxy_rowindex_release_missing_total");
    stat_atlas_proxy_rowindex_fallback_total_ = registerStatistic<uint64_t>("atlas_proxy_rowindex_fallback_total");
    stat_atlas_proxy_idx2row_materialize_total_ = registerStatistic<uint64_t>("atlas_proxy_idx2row_materialize_total");
    stat_atlas_proxy_idx2row_publicize_total_ = registerStatistic<uint64_t>("atlas_proxy_idx2row_publicize_total");
    stat_atlas_proxy_idx2row_owner_form_total_ = registerStatistic<uint64_t>("atlas_proxy_idx2row_owner_form_total");
    stat_atlas_proxy_idx2row_join_live_total_ = registerStatistic<uint64_t>("atlas_proxy_idx2row_join_live_total");
    stat_atlas_proxy_idx2row_join_ready_total_ = registerStatistic<uint64_t>("atlas_proxy_idx2row_join_ready_total");
    stat_atlas_proxy_idx2row_ready_total_ = registerStatistic<uint64_t>("atlas_proxy_idx2row_ready_total");
    stat_atlas_proxy_idx2row_release_total_ = registerStatistic<uint64_t>("atlas_proxy_idx2row_release_total");
    stat_atlas_proxy_idx2row_release_missing_total_ = registerStatistic<uint64_t>("atlas_proxy_idx2row_release_missing_total");
    stat_atlas_proxy_idx2row_fallback_total_ = registerStatistic<uint64_t>("atlas_proxy_idx2row_fallback_total");
    stat_atlas_proxy_premphf_base_materialize_total_ = registerStatistic<uint64_t>("atlas_proxy_premphf_base_materialize_total");
    stat_atlas_proxy_premphf_base_publicize_total_ = registerStatistic<uint64_t>("atlas_proxy_premphf_base_publicize_total");
    stat_atlas_proxy_premphf_base_owner_form_total_ = registerStatistic<uint64_t>("atlas_proxy_premphf_base_owner_form_total");
    stat_atlas_proxy_premphf_base_shared_hit_total_ = registerStatistic<uint64_t>("atlas_proxy_premphf_base_shared_hit_total");
    stat_atlas_proxy_premphf_base_lookup_ready_total_ = registerStatistic<uint64_t>("atlas_proxy_premphf_base_lookup_ready_total");
    stat_atlas_proxy_premphf_base_proxy_only_gap_total_ = registerStatistic<uint64_t>("atlas_proxy_premphf_base_proxy_only_gap_total");
    stat_atlas_proxy_premphf_band_materialize_total_ = registerStatistic<uint64_t>("atlas_proxy_premphf_band_materialize_total");
    stat_atlas_proxy_premphf_band_publicize_total_ = registerStatistic<uint64_t>("atlas_proxy_premphf_band_publicize_total");
    stat_atlas_proxy_premphf_band_owner_form_candidate_total_ = registerStatistic<uint64_t>("atlas_proxy_premphf_band_owner_form_candidate_total");
    stat_atlas_proxy_premphf_band_join_ready_candidate_total_ = registerStatistic<uint64_t>("atlas_proxy_premphf_band_join_ready_candidate_total");
    stat_atlas_proxy_premphf_band_zero_service_total_ = registerStatistic<uint64_t>("atlas_proxy_premphf_band_zero_service_total");
    stat_gcss_lookup_hit_total_ = registerStatistic<uint64_t>("gcss_lookup_hit_total");
    stat_gcss_lookup_miss_total_ = registerStatistic<uint64_t>("gcss_lookup_miss_total");
    stat_weight_read_dense_reqs_total_ = registerStatistic<uint64_t>("weight_read_dense_reqs_total");
    stat_weight_read_dense_bytes_total_ = registerStatistic<uint64_t>("weight_read_dense_bytes_total");
    stat_weight_read_rowptr_reqs_total_ = registerStatistic<uint64_t>("weight_read_rowptr_reqs_total");
    stat_weight_read_rowptr_bytes_total_ = registerStatistic<uint64_t>("weight_read_rowptr_bytes_total");
    stat_weight_read_colidx_reqs_total_ = registerStatistic<uint64_t>("weight_read_colidx_reqs_total");
    stat_weight_read_colidx_bytes_total_ = registerStatistic<uint64_t>("weight_read_colidx_bytes_total");
    stat_weight_read_blockdata_reqs_total_ = registerStatistic<uint64_t>("weight_read_blockdata_reqs_total");
    stat_weight_read_blockdata_bytes_total_ = registerStatistic<uint64_t>("weight_read_blockdata_bytes_total");
    stat_weight_read_gcss_reqs_total_ = registerStatistic<uint64_t>("weight_read_gcss_reqs_total");
    stat_weight_read_gcss_bytes_total_ = registerStatistic<uint64_t>("weight_read_gcss_bytes_total");
    stat_weight_idx_sram_reads_total_ = registerStatistic<uint64_t>("weight_idx_sram_reads_total");
    stat_weight_idx_sram_writes_total_ = registerStatistic<uint64_t>("weight_idx_sram_writes_total");
    stat_weight_idx_sram_bytes_read_total_ = registerStatistic<uint64_t>("weight_idx_sram_bytes_read_total");
    stat_weight_idx_sram_bytes_write_total_ = registerStatistic<uint64_t>("weight_idx_sram_bytes_write_total");
    stat_weight_idx_sram_bank_conflict_ticks_total_ = registerStatistic<uint64_t>("weight_idx_sram_bank_conflict_ticks_total");
    stat_weight_idx_sram_predicted_extra_cycles_total_ = registerStatistic<uint64_t>("weight_idx_sram_predicted_extra_cycles_total");
    stat_weight_idx_sram_resident_bytes_peak_ = registerStatistic<uint64_t>("weight_idx_sram_resident_bytes_peak");
    stat_weight_idx_sram_bank_peak_accesses_per_tick_ = registerStatistic<uint64_t>("weight_idx_sram_bank_peak_accesses_per_tick");
    stat_weight_idx_sram_energy_read_pj_total_ = registerStatistic<uint64_t>("weight_idx_sram_energy_read_pj_total");
    stat_weight_idx_sram_energy_write_pj_total_ = registerStatistic<uint64_t>("weight_idx_sram_energy_write_pj_total");
    stat_weight_idx_lookup_total_ = registerStatistic<uint64_t>("weight_idx_lookup_total");
    stat_weight_idx_lookup_idx2_total_ = registerStatistic<uint64_t>("weight_idx_lookup_idx2_total");
    stat_weight_l0_sram_reads_total_ = registerStatistic<uint64_t>("weight_l0_sram_reads_total");
    stat_weight_l0_sram_writes_total_ = registerStatistic<uint64_t>("weight_l0_sram_writes_total");
    stat_weight_l0_sram_bytes_read_total_ = registerStatistic<uint64_t>("weight_l0_sram_bytes_read_total");
    stat_weight_l0_sram_bytes_write_total_ = registerStatistic<uint64_t>("weight_l0_sram_bytes_write_total");
    stat_weight_l0_sram_bank_conflict_ticks_total_ = registerStatistic<uint64_t>("weight_l0_sram_bank_conflict_ticks_total");
    stat_weight_l0_sram_predicted_extra_cycles_total_ = registerStatistic<uint64_t>("weight_l0_sram_predicted_extra_cycles_total");
    stat_weight_l0_sram_resident_bytes_peak_ = registerStatistic<uint64_t>("weight_l0_sram_resident_bytes_peak");
    stat_weight_l0_sram_bank_peak_accesses_per_tick_ = registerStatistic<uint64_t>("weight_l0_sram_bank_peak_accesses_per_tick");
    stat_weight_l0_sram_energy_read_pj_total_ = registerStatistic<uint64_t>("weight_l0_sram_energy_read_pj_total");
    stat_weight_l0_sram_energy_write_pj_total_ = registerStatistic<uint64_t>("weight_l0_sram_energy_write_pj_total");
    stat_weight_sram_enforced_stall_cycles_total_ = registerStatistic<uint64_t>("weight_sram_enforced_stall_cycles_total");
    stat_weight_l0_lookup_total_ = registerStatistic<uint64_t>("weight_l0_lookup_total");
    stat_weight_l0_hit_total_ = registerStatistic<uint64_t>("weight_l0_hit_total");
    stat_weight_l0_fill_total_ = registerStatistic<uint64_t>("weight_l0_fill_total");
    stat_weight_l0_evict_total_ = registerStatistic<uint64_t>("weight_l0_evict_total");
    stat_core_state_sram_reads_total_ = registerStatistic<uint64_t>("core_state_sram_reads_total");
    stat_core_state_sram_writes_total_ = registerStatistic<uint64_t>("core_state_sram_writes_total");
    stat_core_state_sram_bytes_read_total_ = registerStatistic<uint64_t>("core_state_sram_bytes_read_total");
    stat_core_state_sram_bytes_write_total_ = registerStatistic<uint64_t>("core_state_sram_bytes_write_total");
    stat_core_state_sram_bank_conflict_ticks_total_ = registerStatistic<uint64_t>("core_state_sram_bank_conflict_ticks_total");
    stat_core_state_sram_predicted_extra_cycles_total_ = registerStatistic<uint64_t>("core_state_sram_predicted_extra_cycles_total");
    stat_core_state_sram_resident_bytes_peak_ = registerStatistic<uint64_t>("core_state_sram_resident_bytes_peak");
    stat_core_state_sram_bank_peak_accesses_per_tick_ = registerStatistic<uint64_t>("core_state_sram_bank_peak_accesses_per_tick");
    stat_core_state_sram_energy_read_pj_total_ = registerStatistic<uint64_t>("core_state_sram_energy_read_pj_total");
    stat_core_state_sram_energy_write_pj_total_ = registerStatistic<uint64_t>("core_state_sram_energy_write_pj_total");
    stat_core_state_sram_stall_cycles_total_ = registerStatistic<uint64_t>("core_state_sram_stall_cycles_total");
    stat_riscv_snn_workload_selected_ = registerStatistic<uint64_t>("riscv_snn_workload_selected");
    stat_riscv_snn_firmware_elf_present_ = registerStatistic<uint64_t>("riscv_snn_firmware_elf_present");
    stat_riscv_snn_firmware_loaded_ = registerStatistic<uint64_t>("riscv_snn_firmware_loaded");
    stat_riscv_snn_backend_runtime_bridge_ = registerStatistic<uint64_t>("riscv_snn_backend_runtime_bridge");
    stat_riscv_snn_firmware_started_count_ = registerStatistic<uint64_t>("riscv_snn_firmware_started_count");
    stat_riscv_snn_submitted_commands_ = registerStatistic<uint64_t>("riscv_snn_submitted_commands");
    stat_riscv_snn_accepted_commands_ = registerStatistic<uint64_t>("riscv_snn_accepted_commands");
    stat_riscv_snn_completion_visible_count_ = registerStatistic<uint64_t>("riscv_snn_completion_visible_count");
    stat_riscv_snn_completion_consumed_count_ = registerStatistic<uint64_t>("riscv_snn_completion_consumed_count");
    stat_riscv_snn_fused_step_completion_count_ = registerStatistic<uint64_t>("riscv_snn_fused_step_completion_count");
    stat_riscv_snn_fault_count_ = registerStatistic<uint64_t>("riscv_snn_fault_count");
    stat_riscv_snn_last_completion_status_ = registerStatistic<uint64_t>("riscv_snn_last_completion_status");
    stat_riscv_snn_last_fault_csr_ = registerStatistic<uint64_t>("riscv_snn_last_fault_csr");
    stat_riscv_snn_backend_runtime_bridge_provider_bound_ =
        registerStatistic<uint64_t>("riscv_snn_backend_runtime_bridge_provider_bound");
    // 边集合溢出计数（仅在容量保护触发时递增）
    stat_gas_edge_overflow_ = registerStatistic<uint64_t>("gas_edge_overflow");
    // Apply/Scatter端到端统计（Phase‑1）
    stat_gas_apply_acc_updates_total_ = registerStatistic<uint64_t>("gas_apply_acc_updates_total");
    stat_gas_acc_posts_touched_total_ = registerStatistic<uint64_t>("gas_acc_posts_touched_total");
    stat_gas_scatter_spikes_emitted_total_ = registerStatistic<uint64_t>("gas_scatter_spikes_emitted_total");
    stat_gas_acc_hwm_bytes_total_ = registerStatistic<uint64_t>("gas_acc_high_watermark_bytes_total");
    stat_gas_acc_spill_records_total_ = registerStatistic<uint64_t>("gas_acc_spill_records_total");
    stat_gas_acc_spilled_bytes_total_ = registerStatistic<uint64_t>("gas_acc_spilled_bytes_total");
    stat_gas_retire_global_hol_cycles_total_ = registerStatistic<uint64_t>("gas_retire_global_hol_cycles_total");
    stat_gas_retire_ready_but_blocked_edges_total_ = registerStatistic<uint64_t>("gas_retire_ready_but_blocked_edges_total");
    stat_gas_retire_per_post_progress_total_ = registerStatistic<uint64_t>("gas_retire_per_post_progress_total");
    stat_gas_retire_samepost_blocked_edges_total_ = registerStatistic<uint64_t>("gas_retire_samepost_blocked_edges_total");
    stat_gas_retire_crosspost_blocked_edges_total_ = registerStatistic<uint64_t>("gas_retire_crosspost_blocked_edges_total");
    stat_gas_retire_policy_loss_cycles_total_ = registerStatistic<uint64_t>("gas_retire_policy_loss_cycles_total");
    stat_gas_retire_policy_loss_edges_total_ = registerStatistic<uint64_t>("gas_retire_policy_loss_edges_total");
    // GAS superstep durations（cycles@1GHz == ns）
    // 留空注册，默认不向 SST 统计输出；统一使用 stage_events_csv + 离线脚本聚合
    // Batch-A additions（若需要SST统计输出，可在后续版本开放）
    // stat_mem_read_latency_cycles_ = registerStatistic<uint64_t>("mem_read_latency_cycles");
    // stat_mem_read_latency_cycles_weights_ = registerStatistic<uint64_t>("mem_read_latency_cycles_weights");
    // stat_mem_read_latency_cycles_state_ = registerStatistic<uint64_t>("mem_read_latency_cycles_state");
    // 记录单次内存请求大小与发起时的未完成请求数（Mesh 汇总使用）
    stat_mem_req_size_bytes_ = registerStatistic<uint64_t>("mem_req_size_bytes");
    stat_mem_outstanding_at_issue_ = registerStatistic<uint64_t>("mem_outstanding_at_issue");

    // Phase6: stream workload stats (always registered; default no-op when workload_impl=snn)
    stat_stream_mem_writes_issued_total_ = registerStatistic<uint64_t>("stream_mem_writes_issued_total");
    stat_stream_mem_reads_issued_total_ = registerStatistic<uint64_t>("stream_mem_reads_issued_total");
    stat_stream_mem_bytes_written_total_ = registerStatistic<uint64_t>("stream_mem_bytes_written_total");
    stat_stream_mem_bytes_read_total_ = registerStatistic<uint64_t>("stream_mem_bytes_read_total");
    stat_stream_mem_verify_pass_total_ = registerStatistic<uint64_t>("stream_mem_verify_pass_total");
    stat_stream_mem_verify_fail_total_ = registerStatistic<uint64_t>("stream_mem_verify_fail_total");
    stat_stream_pkt_sent_total_ = registerStatistic<uint64_t>("stream_pkt_sent_total");
    stat_stream_pkt_recv_total_ = registerStatistic<uint64_t>("stream_pkt_recv_total");
    stat_stream_pkt_bad_crc_total_ = registerStatistic<uint64_t>("stream_pkt_bad_crc_total");
    stat_stream_pkt_bad_magic_total_ = registerStatistic<uint64_t>("stream_pkt_bad_magic_total");
    
    // Attach stats hooks to accumulator module now that they are registered.
    if (acc_ops_) {
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
        acc_cfg.stat_apply_updates_total = stat_gas_apply_acc_updates_total_;
        acc_cfg.stat_posts_touched_total = stat_gas_acc_posts_touched_total_;
        acc_cfg.stat_spill_records_total = stat_gas_acc_spill_records_total_;
        acc_cfg.stat_spilled_bytes_total = stat_gas_acc_spilled_bytes_total_;
        acc_cfg.stat_hwm_bytes_total = stat_gas_acc_hwm_bytes_total_;
        acc_ops_->configure(acc_cfg);
    }

    // output_->verbose(CALL_INFO, 2, 0, "✅ 核心%d统计收集初始化完成\n", core_id_);
}
