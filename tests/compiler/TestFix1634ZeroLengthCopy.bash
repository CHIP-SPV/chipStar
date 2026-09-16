#!/bin/bash
# Regression test for CHIP-SPV/chipStar#1634.
#
# WORKAROUND(CHIP-SPV/chipStar#1634, llvm/llvm-project#201904,
# KhronosGroup/SPIRV-LLVM-Translator#3827): guards the constant zero length
# case of hip-lower-hint-intrinsics. Delete this test together with that case
# once LLVM 21 and 22 are no longer supported.
#
# hipcc exits 0 on the invalid module, and on LLVM 23 both producers drop the
# copy anyway, so the bitcode check below is what discriminates on every
# supported LLVM. -O0 is the level that reproduces: from -O1 up the optimizer
# deletes the copy before the device link.
set -eu

SRC_DIR="@CMAKE_CURRENT_SOURCE_DIR@"
HIPCC="@CMAKE_BINARY_DIR@/bin/hipcc"
LLVM_DIS="@LLVM_TOOLS_BINARY_DIR@/llvm-dis"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

SPIRV_VAL="@CMAKE_BINARY_DIR@/external/spirv-tools/bin/spirv-val"
[ -x "${SPIRV_VAL}" ] || SPIRV_VAL=$(command -v spirv-val || true)

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"

# --save-temps keeps the lowered device bitcode and the SPIR-V hipcc ships.
"${HIPCC}" -O0 --save-temps=cwd -c "${SRC_DIR}/@TEST_NAME@.hip" -o "@TEST_NAME@.o"
BC=$(ls ./*-lower.bc 2>/dev/null | head -1)
[ -n "${BC}" ] || { echo "FAIL: hipcc left no lowered device bitcode"; exit 1; }
"${LLVM_DIS}" "${BC}" -o lowered.ll

if grep -E 'llvm\.(memcpy|memmove)\.[^(]*\(.*, i64 0, i1' lowered.ll; then
  echo "FAIL: a constant zero length copy reached SPIR-V emission"
  exit 1
fi
KEPT=$(grep -cE 'call void @llvm\.(memcpy|memmove)\.' lowered.ll || true)
if [ "${KEPT}" -ne 2 ]; then
  echo "FAIL: expected the 2 copies that must be kept, found ${KEPT}"
  grep -E 'llvm\.(memcpy|memmove)\.' lowered.ll || true
  exit 1
fi

# End to end, wherever a validator exists: what hipcc ships must validate.
if [ -x "${SPIRV_VAL}" ]; then
  SPV=0
  for F in ./*; do
    [ -f "${F}" ] && [ "$(od -An -tx1 -N4 "${F}" | tr -d ' \n')" = "03022307" ] || continue
    "${SPIRV_VAL}" "${F}"
    SPV=$((SPV + 1))
  done
  [ "${SPV}" -ge 1 ] || { echo "FAIL: hipcc emitted no SPIR-V module"; exit 1; }
fi
echo "PASSED"
