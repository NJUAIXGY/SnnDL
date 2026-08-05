# SnnDL v5 P2-B Core Storage Binding

- Date: 2026-08-05
- Scope: typed CoreState/CoreDelta/CoreIndex/PeRoute binding
- Conclusion: `accepted`
- Claim boundary: development local-storage evidence only. This does not claim
  cache/DRAM hierarchy, DMA, Merlin NoC timing, performance, power, or area.

## Review Checklist

1. **Typed ownership: PASS.** Each region has an explicit address space,
   owner, base, size, and fail-closed bounds check. Core regions use the core
   owner; `PeRoute` uses the PE owner.
2. **Single functional backing: PASS.** `CoreStorageV5` owns one
   `BankedSramV5` per region. State and delta pipeline operations use SRAM
   request/response completion; no vector is used as functional truth.
3. **Deterministic delta: PASS.** Delta counts and serialized retire entries
   live in `CoreDelta`; readback restores the retire key and sorts by the same
   deterministic key as P1.
4. **P1 compatibility: PASS.** The existing 34 SST timing tests and P1
   acceptance scenarios retain the same functional hash and stage counts.
5. **Capacity/configuration: PASS.** Core ELI parameters expose capacity,
   bank, port, interleave, latency, queue, and resident delta geometry. Invalid
   capacity or address requests fail closed.

## Evidence

- Verifier: `bash tools/verify_snndl_v5_p2.sh`
- Acceptance manifest:
  `snndl_runs/p2_v5_acceptance_20260805_p2b/acceptance.json`
- Typed Core evidence:
  `snndl_runs/p2_v5_acceptance_20260805_p2b/p1/cases/minimal/summary.json`
- Focused test: `make -C .../SnnDL test-v5-core-storage`

## Next Boundary

P3 owns the explicit memory hierarchy: private noncoherent L1 per Core,
PE-shared noncoherent L2, and one chip-wide DRAM controller/backend. The P2
region API remains the address contract for that hierarchy.
