# Active Memory Domain

`StandardMemAccess` implements the pure `IMemoryAccess` address/byte contract;
it tracks pending requests, validates response sizes, and dispatches callbacks.
The banked SRAM model is an independent timing extension used by local-storage
tests.  Future DMA accepts explicit byte-copy descriptors and does not expose
the retired GAS-stage scheduler API.

Do not add BCSR parsing, weight-cache policy, or packet routing here.  Those
domains consume the byte interface from above.  Run `make test-banked-sram` and
`make test-local-storage` after memory changes.
