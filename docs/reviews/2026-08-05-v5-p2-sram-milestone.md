# SnnDL v5 P2-A SRAM Milestone

- Date: 2026-08-05
- Scope: request-driven local SRAM timing service and SST smoke graph
- Conclusion: `accepted` for P2-A
- Blockers: 0
- Claim boundary: development storage timing evidence only. This milestone
  does not claim Core state/delta/index/route binding, cache/DRAM hierarchy,
  Merlin NoC timing, performance class, power, or area.

## Review Checklist

1. **Finite capacity: PASS.** Every accepted request is bounds-checked against
   one owned byte backing. Invalid and capacity failures are permanent and
   fail closed; queue pressure returns an explicit retryable response.
2. **Bank service: PASS.** Low-order interleave selects a bank. Each bank has
   a configured number of ports; requests wait in a finite FIFO and latency
   starts at actual port service, not enqueue.
3. **Backing correctness: PASS.** The SST probe writes three values and reads
   them back through the same request path. There is no timing-only shadow
   response path.
4. **Evidence: PASS.** The component exports accepted/retry/issued/completed,
   bytes, occupancy, conflict, port-stall, busy, latency, capacity, and
   resident-byte counters. The v5 registry and generated header are checked.
5. **Isolation: PASS.** The P1 acceptance suite still passes after adding the
   P2 component. The canonical aggregate has no GAS, 3D, legacy DMA, or
   research dependency edge.

## Evidence

- Verifier: `bash tools/verify_snndl_v5_p2.sh`
- Acceptance manifest:
  `snndl_runs/p2_v5_acceptance_20260805_rerun2/acceptance.json`
- SRAM summary and raw counters:
  `snndl_runs/p2_v5_acceptance_20260805_rerun2/case/`
- SST 15.0.0 ELI includes `BankedSramV5` and `SramProbeV5`.

## Next Boundary

P2-B has now bound typed `CoreState`, `CoreDelta`, `CoreIndex`, and `PeRoute`
regions to the Core pipeline without retaining vector-backed functional reads.
P3 adds the private-L1, PE-shared-L2, single-chip-DRAM hierarchy; this
milestone's SRAM probe remains the isolated timing oracle for local storage.
