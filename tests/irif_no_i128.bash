#!/bin/bash
# Verify OCML bitcode does not contain i128 (irif.h type fix).
# Buggy irif uses unsigned long long for double → 128-bit in OpenCL → i128 → SPIR-V crash.

set -e
BUILD_DIR="$1"
LLVM_DIS="$2"
OCML_BC="${BUILD_DIR}/bitcode/ROCm-Device-Libs/ocml/ocml.lib.bc"

if [[ ! -f "$OCML_BC" ]]; then
  echo "Error: ocml.lib.bc not found. Run: ninja ocml"
  exit 1
fi

IR=$("$LLVM_DIS" "$OCML_BC" -o - 2>/dev/null)
# Match i128 type (load/store/op), not SSA names like %retval.0.i128
if echo "$IR" | grep -qE '(load|store|add|sub|mul|and|or|xor|ptr|trunc|zext|sext) i128'; then
  echo "FAIL: ocml.lib.bc contains i128 - irif.h has wrong OpenCL types (unsigned long long for double)"
  exit 1
fi
echo "PASS: No i128 in OCML (irif fix verified)"
