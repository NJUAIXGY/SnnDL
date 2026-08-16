# SnnDL Project Overview

The active SnnDL build is an explicit SST-native P8 2D communication, compute,
memory, and evidence closure:

```text
api/events
   |-- libSnnDLCore       compute and shared data primitives
   |-- libSnnDLComm       2D NIC, packet, and route path
   |-- libSnnDLLocal      byte memory, DMA, SRAM, local storage
   |-- libSnnDLRegistry   BCSR source identity registry
   `-- libSnnDLResearch   opt-in 3D communication extensions
              |
       libSnnDLNextCore   independent synchronous timestep contract
              |
       libSnnDLV5Core    Core/NIC/control/artifact runtime
```

`libSnnDL.la` is an aggregate compatibility name over these active libraries;
it does not pull in a hidden PE or workload implementation.  `archive/legacy_gas`
contains the former all-in-one assembly and is excluded by the source
manifests.

The canonical data path is: the parent `snndl/v1` compiler validates a graph,
stimulus, mapping and LIF contract; artifact v2 binds destination-owned rows,
weights, source routes and stimuli; the 2D NoC transports events; and the
memory domain services timed weight/DMA requests. The SST runtime exports
per-timestep releases and Core state hashes for acceptance. Local-storage and
3D research components are separate extensions rather than core dependencies.

After changing a boundary, run `make check-boundaries`, `make -j1`, and the
focused timestep, BCSR, DMA, SRAM, or local-storage test that covers the edit.
