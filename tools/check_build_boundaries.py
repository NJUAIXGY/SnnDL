#!/usr/bin/env python3
"""Validate the active SnnDL source manifests and their dependency edges."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path


SOURCE_RE = re.compile(r"^\s*([A-Za-z0-9_./-]+\.(?:cc|h))\s*\\?\s*$")
ASSIGN_RE = re.compile(r"^([A-Za-z0-9_]+)_la_SOURCES\s*=")

ACTIVE_MANIFESTS = ("build/core_sources.am", "build/extension_sources.am")
ACTIVE_TARGETS = {
    "libSnnDLCore",
    "libSnnDLComm",
    "libSnnDLLocal",
    "libSnnDLRegistry",
    "libSnnDLResearch",
}

# These edges would reintroduce the archived PE/workload/GAS object graph.
FORBIDDEN = {
    "libSnnDLCore": (
        "components/",
        "platform/",
        "research/",
        "snn/stimulus/",
        "snn/synapse/route/",
        "snn/synapse/weights/",
        "workloads/",
        "Gas",
        "GAS",
        "GatherBuffer",
    ),
    "libSnnDLComm": (
        "components/MultiCorePE",
        "platform/memory/",
        "research/local_storage/",
        "workloads/",
        "SnnPESubComponent",
        "WeightLoader",
        "Gas",
        "GAS",
        "GatherBuffer",
    ),
    "libSnnDLLocal": (
        "components/",
        "platform/noc/",
        "research/noc3d/",
        "research/route3d/",
        "snn/stimulus/",
        "snn/synapse/route/",
        "workloads/",
        "Gas",
        "GAS",
        "GatherBuffer",
    ),
    "libSnnDLRegistry": (
        "components/",
        "platform/",
        "research/",
        "workloads/",
        "Gas",
        "GAS",
        "GatherBuffer",
    ),
}


def read_manifest(path: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    target: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        assignment = ASSIGN_RE.match(raw)
        if assignment:
            target = assignment.group(1)
            result.setdefault(target, [])
            continue
        if target is None:
            continue
        match = SOURCE_RE.match(raw)
        if match:
            result[target].append(match.group(1))
            continue
        if raw.strip() and not raw.rstrip().endswith("\\"):
            target = None
    return result


def read_source_list(path: Path, variable: str) -> list[str]:
    sources: list[str] = []
    active = False
    prefix = f"{variable} ="
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith(prefix):
            active = True
            raw = raw[len(prefix):]
        elif not active:
            continue
        match = SOURCE_RE.match(raw)
        if match:
            sources.append(match.group(1))
        if active and raw.strip() and not raw.rstrip().endswith("\\"):
            active = False
    return sources


def check(root: Path) -> list[str]:
    errors: list[str] = []
    seen: dict[str, str] = {}

    for rel_manifest in ACTIVE_MANIFESTS:
        manifest = root / rel_manifest
        if not manifest.is_file():
            errors.append(f"missing active manifest {rel_manifest}")
            continue
        for target, sources in read_manifest(manifest).items():
            if target not in ACTIVE_TARGETS:
                errors.append(f"unexpected target in active manifest: {target}")
            for rel in sources:
                previous = seen.get(rel)
                if previous and previous != target:
                    errors.append(f"source {rel} appears in both {previous} and {target}")
                seen[rel] = target
                source = root / rel
                if not source.is_file():
                    errors.append(f"{target}: missing source {rel}")
                    continue
                for line_no, line in enumerate(source.read_text(encoding="utf-8").splitlines(), 1):
                    if not line.lstrip().startswith("#include"):
                        continue
                    for token in FORBIDDEN.get(target, ()):
                        if token in line:
                            errors.append(f"{target}: {rel}:{line_no}: forbidden include token {token!r}")

    nextgen_manifest = root / "build" / "nextgen_sources.am"
    if not nextgen_manifest.is_file():
        errors.append("missing next-generation source manifest")
    else:
        for rel in read_source_list(nextgen_manifest, "SNNDL_NEXT_CORE_SOURCES"):
            source = root / rel
            if not source.is_file():
                errors.append(f"nextgen: missing source {rel}")
                continue
            text = source.read_text(encoding="utf-8")
            for token in ("gas", "Gas", "GAS", "GatherBuffer", "GlobalGas", "workloads/", "research/"):
                if token in text:
                    errors.append(f"nextgen: {rel}: legacy dependency token {token!r} present")

    makefile_am = root / "Makefile.am"
    if makefile_am.is_file():
        text = makefile_am.read_text(encoding="utf-8")
        # platform/core/ is now the active GAS-free SnnCoreTile boundary;
        # archived PE code is guarded by the source manifests above.
        for token in ("libSnnDLOpt", "libSnnDLWorkloads", "components/MultiCorePE"):
            if token in text:
                errors.append(f"Makefile.am: archived target/dependency token {token!r} present")

        aggregate = re.search(
            r"libSnnDL_la_LIBADD\s*=\s*(.*?)(?=\n\S|\Z)", text, flags=re.DOTALL
        )
        if aggregate and "libSnnDLResearch.la" in aggregate.group(1):
            errors.append("Makefile.am: canonical libSnnDL aggregate links libSnnDLResearch")

    for source in sorted((root / "research" / "noc3d").glob("*.h")):
        text = source.read_text(encoding="utf-8")
        if '"SnnDL",' in text:
            errors.append(f"{source.relative_to(root)}: 3D ELI still registers under SnnDL")
        if "SST_ELI_REGISTER_COMPONENT" in text and '"SnnDLResearch",' not in text:
            errors.append(f"{source.relative_to(root)}: 3D ELI is missing SnnDLResearch namespace")

    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    errors = check(args.root.resolve())
    if errors:
        print("SnnDL active build boundary violations:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print("SnnDL active build boundaries: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
