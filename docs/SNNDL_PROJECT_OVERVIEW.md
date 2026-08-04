# SnnDL Project Overview

The active SnnDL build is a small, explicit 2D communication and memory
closure:

```text
api/events
   |-- libSnnDLCore       compute and shared data primitives
   |-- libSnnDLComm       2D NIC, packet, and route path
   |-- libSnnDLLocal      byte memory, DMA, SRAM, local storage
   |-- libSnnDLRegistry   BCSR source identity registry
   `-- libSnnDLResearch   opt-in 3D communication extensions
              |
       libSnnDLNextCore   independent synchronous timestep contract
```

`libSnnDL.la` is an aggregate compatibility name over these active libraries;
it does not pull in a hidden PE or workload implementation.  `archive/legacy_gas`
contains the former all-in-one assembly and is excluded by the source
manifests.

The canonical data path is: a BCSR descriptor is validated and registered,
route code builds packet destinations, the 2D NoC transports packets, and the
memory domain services only address/byte requests.  Local-storage and 3D
research components are separate extensions rather than core dependencies.

After changing a boundary, run `make check-boundaries`, `make -j1`, and the
focused timestep, BCSR, DMA, SRAM, or local-storage test that covers the edit.
