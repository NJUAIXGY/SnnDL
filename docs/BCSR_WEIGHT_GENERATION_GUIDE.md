# BCSR Source Contract

The active route path consumes a BCSR data file together with a sibling
`.meta.json`.  `BcsrDataSource` validates rows, columns, block shape, offsets,
value/index widths, and optional row-packed strides before any route is built.

The descriptor and data bytes are fingerprinted into a `BcsrSourceIdentity`.
Call `bindBcsrSourceContract(slot, path, identity, owner, error)` once for each
logical PE/core source slot.  Rebinding an identical identity is allowed;
different metadata or content fails with a diagnostic rather than silently
creating a second truth.

The active memory layer still sees only byte requests.  It does not parse BCSR,
and route code must not depend on a StandardMem implementation.  Keep generated
BCSR files and run summaries outside this source directory.  Historical weight
loaders and layout experiments remain in `archive/legacy_gas/`.
