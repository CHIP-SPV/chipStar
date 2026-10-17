#!/bin/bash
# Reproducer for CHIP-SPV/chipStar#1746: with a 64 wide warp, __chip_ballot
# must put lane 32 in bit 32.
set -eu
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@"
# A ballot where only lane 32 is true.
echo 'uint4 __attribute__((overloadable)) sub_group_ballot(int p) { return (uint4)(0, 1, 0, 0); }' > "${OUT}.stub.cl"
"@CMAKE_CXX_COMPILER@" -x cl -cl-std=CL2.0 --target=spirv64 -O2 -DDEFAULT_WARP_SIZE=64 \
  -include "${OUT}.stub.cl" -S -emit-llvm -o "${OUT}.ll" "@CMAKE_SOURCE_DIR@/bitcode/ballot_native.cl"
if ! grep -q "ret i64 4294967296" "${OUT}.ll"; then
  echo "FAIL: lane 32 is not bit 32"; grep -A2 "__chip_ballot" "${OUT}.ll"; exit 1
fi
echo PASSED
