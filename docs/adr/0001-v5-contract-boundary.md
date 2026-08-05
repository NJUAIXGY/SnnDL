# ADR-0001: SnnDL v5 Contract Boundary

- Status: accepted for P0 implementation; final P0 sign-off remains required
- Date: 2026-08-04
- Scope: SnnDL v5 2D SST-native cycle-model contract

## Context

The v4 runner is a functional oracle. It closes synchronous timesteps and
provides deterministic BCSR, state, spike, and delivery evidence, but its SST
cycle count is an approximate compatibility statistic. v5 adds explicit
resources and timing, so its semantic ownership must be closed before a new
resolver or component can introduce defaults.

## Decision

The following choices are frozen for v5:

1. The model is a 2D mesh with synchronous, counted-drain timesteps. Held
   spikes are released only at the next timestep; LIF plus BCSR is the first
   closed workload.
2. Mapping is external to SnnDL. A single artifact manifest owns graph,
   placement, route metadata, BCSR values, stimulus, regions, and digests.
   Runtime code may derive views, but may not create a second route or weight
   truth.
3. State, delta, index, and route data use explicit SRAM contracts. Weights
   are DRAM-backed. Each Core has a private L1, each PE has a shared L2, and
   the chip has one shared DRAM. The baseline cache policy is noncoherent and
   read-only for weights.
4. SST native components are preferred for cache, memory, and timing services.
   Merlin is the v5 NoC boundary for P4; SnnDL owns only SNN-specific data
   types and adapters.
5. GAS, Gather/Apply/Scatter stage semantics, 3D NoC/route, and legacy
   Tensor/Stream/Traffic/RISC-V workloads are outside v5. Their files remain
   historical archive material and cannot enter the active v5 dependency
   closure.
6. v4 remains the functional regression oracle. v4 approximate `cycles` and
   SST simulation time are not v5 timing goldens or performance claims.

## Authority and compatibility

When documents conflict, use the order in `docs/adr/README.md`. The v4
architecture and storage contract remain authoritative for v4 behavior; this
ADR governs new v5 contract work. The v4 resolver and runner remain unchanged
until a strict v5 resolver exists. A v5 contract-only validation run must fail
with `runtime_not_implemented` before P1 components exist; it must not fall
back to v4 while claiming v5 execution.

## Not decided here

Neuron and accumulator precision, SRAM bank mapping, index residency, memory
fabric scale-out, cache inclusion details, multicast tree placement, sync
topology, NIR/workload subsets, and energy/area coefficient sources remain
roadmap decisions D1-D12. They must not become hidden resolver defaults.

## Change control

Any change to these choices requires a new or amended ADR, an updated authority
index, and a contract-version change when the wire/spec surface changes. The
progress log records the evidence and review state; it does not override this
ADR.
