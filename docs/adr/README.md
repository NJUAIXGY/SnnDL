# SnnDL Authority Index

This directory records decisions that constrain the v5 contract. An ADR is
normative only when it is linked from this index and has an `accepted` status.
The roadmap remains the process authority; this index prevents an older README
or archived experiment from silently changing the design.

## Authority Order

1. The mandatory contract rules in the v5 roadmap and accepted P0 ADRs.
2. `../plans/2026-08-03-snndl-nextgen-2d-non-gas-architecture.md` for v4
   functional semantics.
3. `../SNNDL_V4_STORAGE_CONTRACT.md` for v4-compatible storage ownership.
4. Active source, generated summaries, and README files as implementation
   evidence only.
5. `../../archive/` for historical explanation only; it is never normative.

## Decisions

| ID | Document | Scope | Status |
|---|---|---|---|
| ADR-0001 | `0001-v5-contract-boundary.md` | v5 semantic and dependency boundary | accepted |

New decisions must add an ADR or update the owning ADR. A change to a wire or
spec contract must also increment its contract version and update the relevant
machine-readable contract. Open choices stay in roadmap section 26.2 until an
ADR resolves them; a default is not a calibration result.
