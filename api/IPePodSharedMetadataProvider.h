// -*- c++ -*-
//
// Narrow provider interface for exposing PE-internal pod-shared metadata services
// without coupling callers to a concrete SST component.

#pragma once

namespace SST { namespace SnnDL {

class PodMetadataObjectPlane;
class PodOwnerServiceTable;
class PeLocalServiceObjectTable;

class IPePodSharedMetadataProvider {
public:
    virtual ~IPePodSharedMetadataProvider() = default;
    virtual PodMetadataObjectPlane* pePodMetadataObjectPlane() = 0;
    virtual PodOwnerServiceTable* pePodOwnerServiceTable() = 0;
    virtual PeLocalServiceObjectTable* peLocalServiceObjectTable() = 0;
};

}} // namespace SST::SnnDL
