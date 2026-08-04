# Active API Contracts

`api/` contains the narrow interfaces shared by the active SnnDL domains.
Implementations depend on these contracts; the contracts do not include
component or research implementation details.

## Communication

- `SnnInterface.h` describes topology-aware packet endpoints.
- `INocTransport.h` and `ISpikeTransport.h` separate packet delivery from
  spike encoding.
- `ISynapseRoute.h` and `SynapseRouteBuildConfig.h` describe route queries and
  BCSR route construction.
- `NocSpikeTransport.h`, `MulticastLimits.h`, and `GlobalNeuronLayout.h`
  provide shared packet and topology utilities.

## Memory and Storage

- `IMemoryAccess.h` is strictly an asynchronous `address + size <-> bytes`
  interface.
- `IDmaTaggedAccess.h` adds opaque tags and priority without adding data
  semantics; `IDmaSchedulerProvider.h` exposes the scheduler where needed.
- `ILocalStorageProvider.h`, `IPePodSharedMetadataProvider.h`, and
  `IPeWeightObjectPlaneProvider.h` expose local-storage services through narrow
  provider handles.

## Shared Utilities

- `BcsrSourceContract.h` binds routing and data consumers to one descriptor and
  content identity per logical source slot.
- `SnnDLStringUtil.h` provides deterministic ASCII normalization and path
  placeholder helpers.
- `SnnDLLogging.h` contains the common lightweight logging macros.

The former PE/workload/weight-control interfaces are preserved under
`archive/legacy_gas/api/` and must not be reintroduced into active manifests.
