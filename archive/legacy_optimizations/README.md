# Archived SnnDL Optimizations

This directory is the recovery boundary for optimization experiments that are
not part of the supported SnnDL runtime. Files are moved here with `git mv`
instead of deleted, so their history and source remain available without
advertising them as active components.

## Active boundary

The supported tree contains the baseline PE, memory, NoC, SNN workload, GAS,
BCSR, communication optimization, 3D communication, and local-storage paths.
New work should not add dependencies on this directory.

## Archive categories

- `research/gas/`: experimental global credit and activation prediction policy.
- `research/pe_fabric/`: PULSE shared-core fabric and descriptor experiments.
- `components/gather/apply/`: superseded DRAM-aware apply helpers.
- `snn/synapse/weights/`: GCSS and PULSE weight metadata experiments.
- `components/` and `tests/`: experimental prefetch statistics and tests.

All files in these categories are intentionally outside the active build. The
archive index is authoritative for historical provenance; new code must not
depend on an archived path or extend the legacy mechanism in place.

## Recovery

Use `git log --follow -- <path>` to inspect provenance. To restore an archived
file for a temporary experiment, copy it into a separate worktree or branch;
do not make the active build depend on this directory.
