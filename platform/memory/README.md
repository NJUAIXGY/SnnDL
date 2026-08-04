# Active Memory Domain

`StandardMemAccess` implements the pure `IMemoryAccess` address/byte contract;
it tracks pending requests, validates response sizes, and dispatches callbacks.
`PeDmaScheduler` and `DmaMemAccessProxy` add tagged read scheduling without
embedding neural or route semantics.  The banked SRAM model is an independent
timing extension used by local-storage tests.

Do not add BCSR parsing, weight-cache policy, or packet routing here.  Those
domains consume the byte interface from above.  Run `make test-dma-scheduler`
and `make test-banked-sram` after memory changes.
