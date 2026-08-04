# Active Route Path

`BcsrRouteBuilder` validates one BCSR source and constructs destination edges.
`SnnRouteProvider` and `SynapseRouteSubsystem` expose fanout queries, while the
packet bridge and codec headers convert spikes to and from `NocPacketEvent`.
`SpikeCommSubsystem` owns batching and transport statistics.

Route code may depend on active `api/`, `events/`, and `snn/synapse/common/`.
It must not include a component assembler or a memory implementation.  The
`BcsrSourceContract` registry is the shared guard against route/data source
drift.  Changes should run the BCSR contract and communication compile checks.
