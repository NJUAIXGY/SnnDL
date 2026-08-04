// -*- c++ -*-

#include <algorithm>

#include <sst/core/output.h>

#include "PeOptionalServices.h"
#include "multicore/MultiCorePEConfig.h"
#include "platform/memory/PeDmaScheduler.h"
#include "research/local_storage/LocalStorageHierarchyController.h"
#include "research/local_storage/LocalStorageTypes.h"
#include "research/local_storage/PeLocalServiceObjectTable.h"
#include "research/local_storage/PeWeightObjectPlane.h"
#include "research/local_storage/PodMetadataObjectPlane.h"
#include "research/local_storage/PodOwnerServiceTable.h"

namespace SST { namespace SnnDL {

namespace {

uint64_t ceilDivU64(uint64_t num, uint64_t den) {
    if (den == 0) return 0;
    return (num + den - 1ull) / den;
}

size_t normalizePodCount(size_t num_cores,
                         uint32_t configured_pod_count,
                         uint32_t configured_pod_size) {
    const size_t cores = std::max<size_t>(num_cores, 1u);
    size_t pods = 0;
    if (configured_pod_count > 0) {
        pods = static_cast<size_t>(configured_pod_count);
    } else if (configured_pod_size > 0) {
        pods = static_cast<size_t>(ceilDivU64(cores, configured_pod_size));
    } else {
        pods = cores;
    }
    if (pods == 0) pods = 1;
    return std::min(pods, cores);
}

constexpr uint32_t kPodMetadataShadowEntryBytes = 32u;

} // namespace

PeOptionalServices::PeOptionalServices() = default;
PeOptionalServices::~PeOptionalServices() = default;

PeOptionalServicesState PeOptionalServices::configure(
    const MultiCorePEConfig& cfg,
    bool pure_snn_datapath_workload,
    int num_cores,
    int node_id,
    SST::Output* output) {
    PeOptionalServicesState state{};
    const size_t core_count = static_cast<size_t>(std::max(1, num_cores));

    const bool dma_enable = cfg.dma_enable && pure_snn_datapath_workload;
    if (cfg.dma_enable && !dma_enable && output) {
        output->verbose(CALL_INFO, 1, 0,
                        "[dma] ignore dma_enable for non-SNN workload_impl=%s node=%d\n",
                        cfg.workload_impl.c_str(), node_id);
    }
    if (dma_enable) {
        PeDmaScheduler::Config dma_cfg{};
        dma_cfg.num_cores = core_count;
        dma_cfg.bytes_per_cycle = cfg.dma_bytes_per_cycle;
        dma_cfg.read_engines = cfg.dma_read_engines;
        dma_cfg.max_inflight = cfg.dma_max_inflight;
        dma_cfg.queue_depth = cfg.dma_queue_depth;
        dma_cfg.overflow_policy =
            (cfg.dma_overflow_policy == "fail_fast")
                ? PeDmaScheduler::Config::OverflowPolicy::FailFast
                : PeDmaScheduler::Config::OverflowPolicy::Block;
        dma_cfg.burst_bytes = cfg.dma_burst_bytes;
        dma_cfg.setup_cycles = cfg.dma_setup_cycles;
        dma_cfg.channels = cfg.dma_channels;
        dma_cfg.channel_bytes_per_cycle = cfg.dma_channel_bytes_per_cycle;
        dma_cfg.channel_interleave_bytes = cfg.dma_channel_interleave_bytes;
        dma_cfg.stage_budget_permille = cfg.dma_stage_budget_permille;
        dma_scheduler_ = std::make_unique<PeDmaScheduler>(dma_cfg);
    }

    const bool local_storage_enable = cfg.local_storage_enable && pure_snn_datapath_workload;
    if (cfg.local_storage_enable && !local_storage_enable && output) {
        output->verbose(CALL_INFO, 1, 0,
                        "[local-storage] ignore local_storage_enable for non-SNN workload_impl=%s node=%d\n",
                        cfg.workload_impl.c_str(), node_id);
    }

    state.pe_internal_cpe_enable = local_storage_enable && cfg.pe_internal_cpe_enable;
    state.pe_internal_pod_enable = state.pe_internal_cpe_enable && cfg.pe_internal_pod_enable;
    state.pe_internal_pod_metadata_enable =
        state.pe_internal_pod_enable && cfg.pe_internal_pod_metadata_enable;
    state.pe_internal_pod_owner_enable =
        state.pe_internal_pod_enable && cfg.pe_internal_pod_owner_enable;
    state.pe_internal_pod_count = state.pe_internal_pod_enable
        ? static_cast<uint32_t>(normalizePodCount(core_count,
                                                  cfg.pe_internal_pod_count,
                                                  cfg.pe_internal_pod_size))
        : 1u;
    state.pe_internal_pod_size = cfg.pe_internal_pod_size;
    if (state.pe_internal_pod_size == 0u) {
        state.pe_internal_pod_size = static_cast<uint32_t>(ceilDivU64(
            static_cast<uint64_t>(core_count),
            static_cast<uint64_t>(std::max<uint32_t>(1u, state.pe_internal_pod_count))));
    }
    state.pe_internal_pod_size = std::max<uint32_t>(1u, state.pe_internal_pod_size);

    if (local_storage_enable) {
        const bool pod_scope_enable =
            cfg.pe_internal_cpe_enable && cfg.pe_internal_pod_enable;
        const size_t pod_count = pod_scope_enable
            ? normalizePodCount(core_count, cfg.pe_internal_pod_count, cfg.pe_internal_pod_size)
            : 1u;

        LocalStorageHierarchyController::Config ls_cfg{};
        ls_cfg.enable = true;
        ls_cfg.num_cores = core_count;
        ls_cfg.num_pods = pod_count;
        local_storage_controller_ =
            std::make_unique<LocalStorageHierarchyController>(ls_cfg);

        PhaseALocalStorageRegistrationConfig phase_cfg{};
        phase_cfg.num_cores = ls_cfg.num_cores;
        phase_cfg.num_pods = ls_cfg.num_pods;

        phase_cfg.activation_ingress.name = "activation_ingress_store";
        phase_cfg.activation_ingress.kind = LocalStorageObjectKind::QueueStore;
        phase_cfg.activation_ingress.scope = LocalStorageScope::PerPe;
        phase_cfg.activation_ingress.enable = cfg.ls_activation_ingress_enable;
        phase_cfg.activation_ingress.queue_depth = cfg.ls_activation_ingress_entries;

        phase_cfg.weight_idx.name = "weight_idx_store";
        phase_cfg.weight_idx.kind = LocalStorageObjectKind::AddressableStore;
        phase_cfg.weight_idx.scope = LocalStorageScope::PerPe;
        phase_cfg.weight_idx.enable = cfg.ls_weight_idx_enable;
        phase_cfg.weight_idx.capacity_bytes = cfg.ls_weight_idx_capacity_bytes;
        phase_cfg.weight_idx.banks = cfg.ls_weight_idx_banks;
        phase_cfg.weight_idx.read_ports = cfg.ls_weight_idx_read_ports;
        phase_cfg.weight_idx.write_ports = cfg.ls_weight_idx_write_ports;
        phase_cfg.weight_idx.queue_depth = cfg.ls_weight_idx_queue_depth;

        phase_cfg.weight_value.name = "weight_value_store";
        phase_cfg.weight_value.kind = LocalStorageObjectKind::AddressableStore;
        phase_cfg.weight_value.scope = LocalStorageScope::PerPe;
        phase_cfg.weight_value.enable = cfg.ls_weight_value_enable;
        phase_cfg.weight_value.capacity_bytes = cfg.ls_weight_value_capacity_bytes;
        phase_cfg.weight_value.banks = cfg.ls_weight_value_banks;
        phase_cfg.weight_value.read_ports = cfg.ls_weight_value_read_ports;
        phase_cfg.weight_value.write_ports = cfg.ls_weight_value_write_ports;
        phase_cfg.weight_value.queue_depth = cfg.ls_weight_value_queue_depth;

        if (pod_scope_enable) {
            phase_cfg.pod_metadata_template.name = "pod_metadata_store";
            phase_cfg.pod_metadata_template.kind = LocalStorageObjectKind::AddressableStore;
            phase_cfg.pod_metadata_template.scope = LocalStorageScope::PerPod;
            phase_cfg.pod_metadata_template.enable = cfg.pe_internal_pod_metadata_enable;
            phase_cfg.pod_metadata_template.capacity_bytes =
                cfg.pe_internal_pod_metadata_capacity_bytes;
            phase_cfg.pod_metadata_template.banks =
                std::max<uint32_t>(cfg.pe_internal_pod_metadata_banks, 1u);

            phase_cfg.pod_owner_template.name = "pod_owner_table";
            phase_cfg.pod_owner_template.kind = LocalStorageObjectKind::AddressableStore;
            phase_cfg.pod_owner_template.scope = LocalStorageScope::PerPod;
            phase_cfg.pod_owner_template.enable = cfg.pe_internal_pod_owner_enable;
            phase_cfg.pod_owner_template.capacity_bytes =
                static_cast<uint64_t>(cfg.pe_internal_pod_owner_entries) *
                static_cast<uint64_t>(std::max<uint32_t>(cfg.pe_internal_pod_owner_entry_bytes, 1u));

            phase_cfg.pod_join_template.name = "pod_join_table";
            phase_cfg.pod_join_template.kind = LocalStorageObjectKind::AddressableStore;
            phase_cfg.pod_join_template.scope = LocalStorageScope::PerPod;
            phase_cfg.pod_join_template.enable = cfg.pe_internal_pod_join_enable;
            phase_cfg.pod_join_template.capacity_bytes =
                static_cast<uint64_t>(cfg.pe_internal_pod_join_entries) *
                static_cast<uint64_t>(std::max<uint32_t>(cfg.pe_internal_pod_join_entry_bytes, 1u));

            phase_cfg.pod_ready_template.name = "pod_ready_table";
            phase_cfg.pod_ready_template.kind = LocalStorageObjectKind::QueueStore;
            phase_cfg.pod_ready_template.scope = LocalStorageScope::PerPod;
            phase_cfg.pod_ready_template.enable = cfg.pe_internal_pod_ready_enable;
            phase_cfg.pod_ready_template.queue_depth = cfg.pe_internal_pod_ready_entries;
        }

        phase_cfg.state_template.name = "state_store";
        phase_cfg.state_template.kind = LocalStorageObjectKind::AddressableStore;
        phase_cfg.state_template.scope = LocalStorageScope::PerCore;
        phase_cfg.state_template.enable = cfg.ls_state_enable;
        phase_cfg.state_template.capacity_bytes = cfg.ls_state_capacity_bytes;
        phase_cfg.state_template.banks = cfg.ls_state_banks;
        phase_cfg.state_template.read_ports = cfg.ls_state_read_ports;
        phase_cfg.state_template.write_ports = cfg.ls_state_write_ports;
        phase_cfg.state_template.update_ports = cfg.ls_state_update_ports;
        phase_cfg.state_template.queue_depth = cfg.ls_state_queue_depth;

        phase_cfg.activation_core_template.name = "activation_core_queue";
        phase_cfg.activation_core_template.kind = LocalStorageObjectKind::QueueStore;
        phase_cfg.activation_core_template.scope = LocalStorageScope::PerCore;
        phase_cfg.activation_core_template.enable = cfg.ls_activation_core_enable;
        phase_cfg.activation_core_template.queue_depth = cfg.ls_activation_core_entries;

        phase_cfg.acc_template.name = "accumulator_store";
        phase_cfg.acc_template.kind = LocalStorageObjectKind::AddressableStore;
        phase_cfg.acc_template.scope = LocalStorageScope::PerCore;
        phase_cfg.acc_template.enable = cfg.ls_acc_enable;
        phase_cfg.acc_template.capacity_bytes = cfg.ls_acc_capacity_bytes;
        phase_cfg.acc_template.banks = cfg.ls_acc_banks;
        phase_cfg.acc_template.read_ports = cfg.ls_acc_read_ports;
        phase_cfg.acc_template.write_ports = cfg.ls_acc_write_ports;
        phase_cfg.acc_template.update_ports = cfg.ls_acc_update_ports;
        phase_cfg.acc_template.queue_depth = cfg.ls_acc_queue_depth;

        phase_cfg.rf_template.name = "register_file";
        phase_cfg.rf_template.kind = LocalStorageObjectKind::RegisterFile;
        phase_cfg.rf_template.scope = LocalStorageScope::PerCore;
        phase_cfg.rf_template.enable = cfg.ls_rf_enable;
        phase_cfg.rf_template.entries = cfg.ls_rf_entries;
        phase_cfg.rf_template.entry_bytes = cfg.ls_rf_entry_bytes;
        phase_cfg.rf_template.read_ports = cfg.ls_rf_read_ports;
        phase_cfg.rf_template.write_ports = cfg.ls_rf_write_ports;

        if (!registerDefaultPhaseAObjects(*local_storage_controller_, phase_cfg) && output) {
            output->fatal(CALL_INFO, -1,
                          "LocalStorageHierarchyController registration failed at node=%d\n",
                          node_id);
        }

        if (pod_scope_enable && cfg.pe_internal_pod_metadata_enable) {
            PodMetadataObjectPlane::Config pod_metadata_cfg{};
            pod_metadata_cfg.enable = true;
            pod_metadata_cfg.num_pods = static_cast<uint32_t>(pod_count);
            if (cfg.pe_internal_pod_metadata_capacity_bytes > 0) {
                const uint64_t raw_entries =
                    cfg.pe_internal_pod_metadata_capacity_bytes /
                    static_cast<uint64_t>(kPodMetadataShadowEntryBytes);
                pod_metadata_cfg.capacity_entries_per_pod =
                    static_cast<uint32_t>(std::max<uint64_t>(1u, raw_entries));
            }
            pod_metadata_object_plane_ =
                std::make_unique<PodMetadataObjectPlane>(pod_metadata_cfg);
        }
        if (pod_scope_enable && cfg.pe_internal_pod_owner_enable) {
            PodOwnerServiceTable::Config pod_owner_cfg{};
            pod_owner_cfg.enable = true;
            pod_owner_cfg.num_pods = static_cast<uint32_t>(pod_count);
            pod_owner_cfg.owner_entries_per_pod = cfg.pe_internal_pod_owner_entries;
            pod_owner_cfg.join_entries_per_pod = cfg.pe_internal_pod_join_entries;
            pod_owner_service_table_ =
                std::make_unique<PodOwnerServiceTable>(pod_owner_cfg);
        }
        if (pod_scope_enable && cfg.pe_internal_pod_metadata_enable &&
            cfg.pe_internal_pod_owner_enable &&
            (cfg.pe_internal_pod_join_enable || cfg.pe_internal_pod_ready_enable)) {
            PeLocalServiceObjectTable::Config service_cfg{};
            service_cfg.enable = true;
            service_cfg.num_pods = static_cast<uint32_t>(pod_count);
            service_cfg.active_entries_per_pod =
                std::max(cfg.pe_internal_pod_owner_entries, cfg.pe_internal_pod_join_entries);
            service_cfg.released_entries_per_pod =
                std::max(cfg.pe_internal_pod_ready_entries, cfg.pe_internal_pod_join_entries);
            service_cfg.ready_lease_enable = false;
            service_cfg.ready_lease_ttl = 0u;
            service_cfg.ready_lease_kind_mask = 0u;
            pe_local_service_object_table_ =
                std::make_unique<PeLocalServiceObjectTable>(service_cfg);
        }
    }

    if (local_storage_enable &&
        (cfg.ls_weight_idx_enable || cfg.ls_weight_value_enable)) {
        PeWeightObjectPlane::Config weight_plane_cfg{};
        weight_plane_cfg.enable = true;
        weight_plane_cfg.owner_scope_enable = true;
        weight_plane_cfg.actual_owner_enable = false;
        weight_plane_cfg.idx_enable = cfg.ls_weight_idx_enable;
        weight_plane_cfg.l0_enable = cfg.ls_weight_value_enable;
        weight_plane_cfg.idx_capacity_bytes = cfg.ls_weight_idx_capacity_bytes;
        weight_plane_cfg.l0_capacity_bytes = cfg.ls_weight_value_capacity_bytes;
        weight_plane_cfg.idx_banks = cfg.ls_weight_idx_banks;
        weight_plane_cfg.l0_banks = cfg.ls_weight_value_banks;
        weight_plane_cfg.ports_per_bank = std::max<uint32_t>(
            1u,
            std::max(std::max(cfg.ls_weight_idx_read_ports, cfg.ls_weight_idx_write_ports),
                     std::max(cfg.ls_weight_value_read_ports, cfg.ls_weight_value_write_ports)));
        pe_weight_object_plane_ = std::make_unique<PeWeightObjectPlane>(weight_plane_cfg);
    }

    return state;
}

}} // namespace SST::SnnDL
