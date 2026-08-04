# Active SnnDL Workflow

The maintained path has four explicit stages:

1. A caller validates topology, timestep, memory, and BCSR source parameters.
2. `BcsrRouteBuilder` loads and validates the descriptor, then binds its
   identity through `BcsrSourceContract`.
3. Route and stimulus code encode packet destinations; `NocSubsystem` and the
   SST NIC transport packets across the 2D mesh.
4. The memory domain services address/byte requests, while the synchronous
   timestep core retires deltas and updates neuron state.

`libSnnDLNextCore.la` is a separate pure-domain closure for timestep behavior.
`libSnnDL.la` aggregates the active SST libraries for compatibility, but it has
no hidden PE, workload, or stage-control dependency.  Local-storage and 3D
communication research modules are opt-in extensions.

For every run, record the validated spec, source identity, topology, and
effective timing parameters.  Keep generated output outside the source tree;
use `make check-boundaries` before building.
