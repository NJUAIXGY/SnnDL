# 2D NoC Components

`SnnNetworkAdapter` and `SimpleNetworkWrapper` adapt SST mesh/torus links and
Merlin SimpleNetwork requests.  `MulticastNIC` and `MulticastRouter` provide
an explicit 2D multicast backend for packet experiments.  All components carry
`NocPacketEvent`; route selection and BCSR interpretation stay in
`snn/synapse/route/`.

The active objects are compiled by `libSnnDLComm.la` and can be loaded through
their ELI names without loading any archived PE or workload component.
