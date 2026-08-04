# Active Components

The active component layer only adapts SST lifecycle and links to the packet
domains.  It does not assemble the historical PE/workload object graph.

- `SnnNIC.{h,cc}`: SimpleNetwork-facing NIC with packet statistics.
- `components/noc/SnnNetworkAdapter`: 2D mesh/torus adapter for directional
  links or Merlin SimpleNetwork.
- `components/noc/SimpleNetworkWrapper`: reusable SimpleNetwork queue proxy.
- `components/noc/MulticastNIC` and `MulticastRouter`: explicit 2D multicast
  experiment backends.

The implementation split is reflected in the build manifests:
`libSnnDLComm.la` owns these components, while memory and local storage live in
`libSnnDLLocal.la`.  The aggregate `libSnnDL.la` only links active domain
libraries.

Archived PE assembly, loaders, control events, and workload components remain
available for historical inspection under `archive/legacy_gas/components/`.
