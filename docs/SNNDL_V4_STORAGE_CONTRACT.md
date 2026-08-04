# SnnDL v4 Storage Contract

This document freezes ownership for the canonical 2D SST model. It is the
source of truth for interpreting cache and memory statistics in v4 reports.

## Address and data ownership

1. The resolved spec produces one BCSR descriptor and one value image. The
   descriptor owns edge order, route metadata, and the descriptor digest. It
   is read during PE construction and is not a second runtime weight source.
2. Every edge value has address `edge.ordinal * memory_bytes`. With local
   storage disabled, value reads travel through the native SST hierarchy:

   ```text
   core StandardMem -> private L1 -> PE bus -> shared PE L2
       -> chip bus -> one MemController -> simpleMem or ramulator2
   ```

3. Neuron state, refractory state, and the current timestep delta are owned by
   `SnnCoreTile` and its domain accumulators. They are core-local state objects;
   they do not silently become DRAM traffic. Future state-SRAM work must add a
   separate, counted interface.

## Local-storage extension

When `extensions.local_storage.enabled` is true, each PE treats the canonical
BCSR value image as a PE-local read-only value store. A task still creates one
logical memory request and one logical response, but completion is scheduled
on the PE-local ready queue with the configured local latency (one cycle in
the v4 runner). The shared DRAM hierarchy remains present in the chip graph,
but receives no value request from that task.

The extension must preserve `state_digest`, `spike_digest`, logical tx/rx,
task counts, and memory request/response counts. `storage_hits` reports the
accepted tasks completed by the local store. A mismatch is a correctness
failure, not an optimization result.

## Reporting rules

Descriptor bytes are startup metadata. BCSR value reads, cache hits, local
storage hits, request/response counts, and timestep cycles are reported
separately. Synchronization events from `TimestepCoordinator` are control
traffic and are not counted as NoC payload bytes.
