# Active Research Extensions

Research code is compiled separately from the stable core and must remain
opt-in at the SST configuration boundary.

- `local_storage/`: PE-local object tables, pod metadata/ownership models, and
  shared weight-object accounting.
- `noc3d/`: 3D multicast router, HBM stack stub, and source/sink smoke
  components.
- `route3d/`: 3D node mapping and route-table extension built on the active 2D
  route interfaces.

These modules may depend on active `api/` and communication contracts, but the
stable core and 2D memory layer must not depend on them.  Keep benchmark output
and calibration data outside the source manifests.
