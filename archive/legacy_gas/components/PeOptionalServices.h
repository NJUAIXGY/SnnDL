// -*- c++ -*-
// Ownership boundary for PE-local optional services.

#pragma once

#include <cstdint>
#include <memory>

namespace SST {
class Output;
namespace SnnDL {

struct MultiCorePEConfig;
class PeDmaScheduler;
class LocalStorageHierarchyController;
class PodMetadataObjectPlane;
class PodOwnerServiceTable;
class PeLocalServiceObjectTable;
class PeWeightObjectPlane;

struct PeOptionalServicesState {
    bool pe_internal_cpe_enable = false;
    bool pe_internal_pod_enable = false;
    bool pe_internal_pod_metadata_enable = false;
    bool pe_internal_pod_owner_enable = false;
    uint32_t pe_internal_pod_count = 1;
    uint32_t pe_internal_pod_size = 1;
};

/**
 * Owns optional PE services and keeps their configuration out of MultiCorePE.
 * The core component receives only the effective capability state and narrow
 * service pointers needed by existing provider interfaces.
 */
class PeOptionalServices final {
public:
    PeOptionalServices();
    ~PeOptionalServices();

    PeOptionalServices(const PeOptionalServices&) = delete;
    PeOptionalServices& operator=(const PeOptionalServices&) = delete;

    PeOptionalServicesState configure(const MultiCorePEConfig& cfg,
                                      bool pure_snn_datapath_workload,
                                      int num_cores,
                                      int node_id,
                                      SST::Output* output);

    PeDmaScheduler* dmaScheduler() const { return dma_scheduler_.get(); }
    LocalStorageHierarchyController* localStorageHierarchy() const {
        return local_storage_controller_.get();
    }
    PodMetadataObjectPlane* podMetadataObjectPlane() const {
        return pod_metadata_object_plane_.get();
    }
    PodOwnerServiceTable* podOwnerServiceTable() const {
        return pod_owner_service_table_.get();
    }
    PeLocalServiceObjectTable* localServiceObjectTable() const {
        return pe_local_service_object_table_.get();
    }
    PeWeightObjectPlane* weightObjectPlane() const {
        return pe_weight_object_plane_.get();
    }

private:
    std::unique_ptr<PeDmaScheduler> dma_scheduler_;
    std::unique_ptr<LocalStorageHierarchyController> local_storage_controller_;
    std::unique_ptr<PodMetadataObjectPlane> pod_metadata_object_plane_;
    std::unique_ptr<PodOwnerServiceTable> pod_owner_service_table_;
    std::unique_ptr<PeLocalServiceObjectTable> pe_local_service_object_table_;
    std::unique_ptr<PeWeightObjectPlane> pe_weight_object_plane_;
};

} // namespace SnnDL
} // namespace SST
