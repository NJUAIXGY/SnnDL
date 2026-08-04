# Active Synapse Domain

Only the BCSR source contract and route path are active:

- `common/BcsrMeta.h` and `common/BcsrDataSource.h` parse and validate one
  canonical BCSR file identity.
- `route/` builds fanout, resolves weight metadata needed for packet creation,
  and encodes spike, tile, and inter-bundle packets.

The route path may call `bindBcsrSourceContract()` so route construction and
data consumers cannot silently use different descriptor or content files.  It
does not own StandardMem scheduling; byte traffic belongs to `platform/memory`.

The former weight cache, window controller, and stage-oriented synapse code is
archived under `archive/legacy_gas/snn/synapse/`.
