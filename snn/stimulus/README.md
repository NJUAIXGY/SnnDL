# Active Stimulus

`ExternalSpikeInputSubsystem` accepts an externally supplied spike, encodes it
as an active packet, and injects it through `api/INocTransport`.  It owns no
memory scheduling, route-table policy, or timestep controller.  Keep stimulus
tests deterministic and validate packet ownership at the transport boundary.

The former step/window injection implementation is preserved under
`archive/legacy_gas/snn/stimulus/`.
