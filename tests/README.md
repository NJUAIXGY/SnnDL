# Active Tests

The active tests cover the maintained closure only:

- `test_timestep_core.cc`: synchronous tracker, delta accumulation, and neuron
  firing semantics.
- `test_bcsr_source_contract.cc`: descriptor/content identity binding.
- `test_pe_dma_scheduler.cc`: tagged read scheduling and write passthrough.
- `test_banked_sram_model.cc`: bank conflict and broadcast timing accounting.
- `test_local_storage_hierarchy.cc`: local object, pod, and shared-plane
  contracts.

Run the focused targets from the SnnDL directory (`make test-compile` is the
fast compile-only check).  Historical workload and firmware tests are kept in
`archive/legacy_gas/tests/`.
