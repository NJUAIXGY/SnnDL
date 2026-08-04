# Active Platform

The active platform is intentionally limited to transport, memory, and source
identity services:

- `noc/`: packet queues, internal ring scheduling, and NoC transport callbacks.
- `memory/`: StandardMem byte access, DMA scheduling, and banked SRAM timing.
- `registry/`: the process-local `BcsrSourceContract` registry shared by route
  producers and readers.

Memory code must preserve the `IMemoryAccess` address/byte contract and must not
parse neural or route formats.  NoC code transports `NocPacketEvent` values and
must not select fanout or decode BCSR metadata.  New cross-domain behavior
belongs behind an `api/` interface and a focused test.

The former CoreShell, workload statistics, and stage-control platform code is
preserved under `archive/legacy_gas/platform/`.
