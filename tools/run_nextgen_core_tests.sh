#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
BIN="/tmp/snndl_nextgen_core_test_$$"

g++ -std=c++17 -Wall -Wextra -Werror \
  -I"$ROOT" \
  "$ROOT/snn/timestep/TimestepTracker.cc" \
  "$ROOT/snn/compute/DeltaAccumulator.cc" \
  "$ROOT/snn/compute/NextGenNeuronEngine.cc" \
  "$ROOT/tests/test_timestep_core.cc" \
  -o "$BIN"
"$BIN"
rm -f "$BIN"
echo "SnnDL next-generation core tests: PASS"
