# Legacy SnnDL archive

This directory preserves the pre-next-generation SnnDL object graph for
historical inspection and reproducibility only.  It contains the former GAS
controller, PE assembly, weight/StandardMem front-end, workload implementations,
old APIs, tests, generated objects, and their design records.

The retired tagged DMA cluster is preserved under `api/IDma*.h`,
`platform/memory/{DmaMemAccessProxy,PeDmaScheduler}.*`, and
`tests/test_pe_dma_scheduler.cc`.  It is historical material only; the active
tree has no forwarding headers or scheduler target for these names.

Nothing below this directory is part of an active Automake source manifest or
linked into the current SnnDL libraries.  New code must use the active 2D
communication, local-memory, BCSR source-contract, or timestep interfaces;
do not add compatibility includes that reach back into this archive.

The former event-only `components/SnnMemoryModel` source is also outside the
active platform manifest. v4 memory traffic must use SST `StandardMem` and
`memHierarchy`; the event model is retained only for historical reference.
