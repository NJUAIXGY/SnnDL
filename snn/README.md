# SNN Domain

The active SNN domain is split by responsibility:

- `compute/`: neuron dynamics, learning helpers, and data-layout utilities.
- `synapse/common/`: BCSR metadata parsing and source validation.
- `synapse/route/`: route construction, fanout queries, and spike packet
  codecs.
- `stimulus/`: external spike injection over the active packet transport.
- `timestep/`: the synchronous timestep tracker used by the next-generation
  core.

Compute consumes parsed values and emits domain events.  Route code owns BCSR
  reachability and packet encoding; NoC only transports packets.  Memory is
  accessed through `api/IMemoryAccess` and remains unaware of SNN semantics.

Use `make test-timestep-core` and the BCSR source-contract test for changes in
the active SNN path.  Historical synapse, weight, and stage implementations are
under `archive/legacy_gas/snn/`.
