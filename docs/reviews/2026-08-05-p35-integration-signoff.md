# P3.5 Core-Memory Integration Sign-off

P3.5 closes the gap between the P1/P2 timed SNN core and the P3 SST memory
hierarchy. It does not add a second functional weight truth.

## Closed Data Path

```text
ChipDram image -> SnnDmaEngineV5 -> PeWeightSpm
                                      |
Core row request -> timed StandardMem read -> decoded edge -> SnnCoreV5
                                                        -> typed local SRAM
```

`IdealSynapseSource` retains its P1 ideal mode by default. In the optional
memory-backed mode it writes the initial DRAM image during SST initialization,
waits for the typed DMA completion, and only then sends `Start`. Each edge
returned to the core is decoded from a timed local scratchpad read.

Multiple clients share the Core memory path through SST's
`memHierarchy.multithreadL1` arbiter, followed by Scratchpad, private L1,
PE-shared L2, and the chip MemController. DMA accepts only
`ChipDram <-> PeWeightSpm` descriptors and requires the DRAM owner to be zero.

## Acceptance

Run from the outer `remote/` checkout:

```bash
bash tools/verify_snndl_v5_p35.sh
```

Acceptance covers the P1 ideal-provider regression, valid and invalid typed
DMA descriptors, SimpleMem, Ramulator2, and a 2 PE x 2 Core case. It checks
preload-before-start ordering, decoded weight identity, functional equivalence,
and positive L1/L2/DRAM traffic. The generated `acceptance.json` is the
authoritative sign-off artifact.

## Claim Boundary

P3.5 supports functional and timing claims for one complete Core, local SRAM,
weight SPM, cache, and DRAM timestep path. It does not include Merlin transport,
flit-level contention, multicast, or timed chip-wide synchronization; those
remain P4 and P5 work.
