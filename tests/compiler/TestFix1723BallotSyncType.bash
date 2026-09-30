#!/bin/bash
# Reproducer for CHIP-SPV/chipStar#1723: __ballot_sync must call a
# __chip_ballot_sync whose return type matches the call.
set -eu

HIPCC="@CMAKE_BINARY_DIR@/bin/hipcc"
LLVM_DIS="@LLVM_TOOLS_BINARY_DIR@/llvm-dis"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"

"${HIPCC}" --save-temps=cwd -c "@CMAKE_CURRENT_SOURCE_DIR@/@TEST_NAME@.hip" \
  -o "@TEST_NAME@.o"
# The device library link output: the optimizer inlines matching calls later.
BC=$(ls ./*-link.bc 2>/dev/null | head -1)
[ -n "${BC}" ] || { echo "FAIL: hipcc left no linked device bitcode"; exit 1; }
"${LLVM_DIS}" "${BC}" -o linked.ll

F=_Z18__chip_ballot_syncji
# The type is the token before the callee name.
DEF=$(grep -E "^define " linked.ll | grep -oE "\S+ @${F}\(" | cut -d' ' -f1)
CALLS=$(grep -v '^define' linked.ll | grep -E "call " |
  grep -oE "\S+ @${F}\(" | cut -d' ' -f1 | sort -u)
[ -n "${DEF}" ] || { echo "FAIL: no definition of ${F}"; exit 1; }
[ -n "${CALLS}" ] || { echo "FAIL: no call to ${F}"; exit 1; }
if [ "${CALLS}" != "${DEF}" ]; then
  echo "FAIL: ${F} defined returning ${DEF} but called as returning ${CALLS}"
  exit 1
fi
echo PASSED
