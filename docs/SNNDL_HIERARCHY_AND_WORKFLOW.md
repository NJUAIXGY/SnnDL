# Active SnnDL Workflow

The maintained P8 path has six explicit stages:

1. The parent `snndl/v1` API validates chip, workload, mapping and simulation intent.
2. GraphIR, StimulusIR and MappingIR are compiled with an independent float32
   LIF oracle.
3. Artifact v2 binds destination-owned rows/weights, source routes and
   source-owned stimuli with a dynamic, non-overlapping ChipDram layout.
4. The strict v5 resolver validates owners, regions, digests, capacity and
   memory contracts before SST starts.
5. SST runs Core, typed SRAM, DMA, L1/L2/DRAM, NIC, Merlin/native multicast and
   timed control components.
6. The runner compares traffic, per-timestep releases and per-Core state hashes,
   then writes acceptance/provenance evidence.

`libSnnDLNextCore.la` is a separate pure-domain closure for timestep behavior.
`libSnnDL.la` aggregates the active SST libraries for compatibility, but it has
no hidden PE, workload, or stage-control dependency.  Local-storage and 3D
communication research modules are opt-in extensions.

For every run, record the validated spec, source identity, topology, and
effective timing parameters.  Keep generated output outside the source tree;
use `make check-boundaries` before building.
