# SnnDL

SnnDL is the SST element library for the current 2D SNN communication and
memory model.  The active tree is deliberately small: domain primitives,
packet transport and routing, local-memory models, the BCSR source contract,
and opt-in 3D communication research components.

## Active Layout

- `api/`: narrow contracts for packets, transport, memory, topology, and BCSR
  source identity.
- `events/`: serializable packet, spike, and timestep event types.
- `snn/compute/`: neuron dynamics, learning helpers, and the next-generation
  timestep core.
- `snn/synapse/route/`: BCSR route construction and spike packet codecs.
- `components/` and `platform/noc/`: SST NICs, 2D adapters, and packet rings.
- `platform/memory/` and `research/local_storage/`: address/byte memory,
  DMA scheduling, SRAM timing, and local-storage object models.
- `research/noc3d/` and `research/route3d/`: explicitly separate 3D research
  extensions.
- `archive/legacy_gas/`: the complete historical PE, workload, GAS, weight
  front-end, API, test, and design-record tree.  It is not compiled.

## Build and Checks

Run from this directory after the parent SST tree has been configured:

```bash
make -j1
make check-boundaries
make test-timestep-core
make test-bcsr-source-contract
make test-v5-address
make test-v5-statistics
make test-banked-sram
make test-local-storage
```

`build/core_sources.am` and `build/extension_sources.am` are the active build
boundary; `build/v5_sources.am` is the standalone v5 contract boundary.  Keep new code out of the archive and do not add compatibility
includes that reach into it.  Regenerate `Makefile.in` and `Makefile` with the
parent tree's `automake`/`config.status` flow after changing `Makefile.am`.

The canonical `libSnnDL.la` entry point links the 2D platform only. The 3D
components are built as `libSnnDLResearch.la` and register under the separate
`SnnDLResearch` library namespace; load that library explicitly for research
SDLs. See `docs/SNNDL_V4_STORAGE_CONTRACT.md` for the v4 memory ownership
contract.
