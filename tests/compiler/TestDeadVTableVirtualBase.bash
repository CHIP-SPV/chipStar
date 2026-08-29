#!/bin/bash
# Reproducer for CHIP-SPV/chipStar#1382: dead C++ vtables in device modules.
#
# An explicit instantiation makes clang emit the vtable, VTT and construction
# vtables of a polymorphic class into the device module with weak_odr linkage,
# which GlobalDCE cannot discard, even though nothing on the device dispatches
# through them. A device module is self-contained, so nothing can ever link
# against them, and their inttoptr virtual base offsets are what aborted
# llvm-spirv on Tpetra::RowMatrix's HIP instantiations. Check that the lowered
# device bitcode, the input of the SPIR-V producer, carries no vtable family
# global.
set -eu

HIPCC="@CMAKE_BINARY_DIR@/bin/hipcc"
LLVM_DIS="@CLANG_ROOT_PATH_BIN@/llvm-dis"
SRC="@CMAKE_CURRENT_SOURCE_DIR@/TestDeadVTableVirtualBase.hip"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

if [ ! -x "${LLVM_DIS}" ]; then
  echo "llvm-dis not found at ${LLVM_DIS}; skipping"
  exit 0
fi

rm -rf "${OUT}"
mkdir -p "${OUT}"
cd "${OUT}"

# The lowered device bitcode is *-generic-lower.bc with the old offload driver
# and *.hipfb.<triple>.-lower.bc with the new one; both end in -lower.bc.
"${HIPCC}" --save-temps=cwd -c "${SRC}" -o TestDeadVTableVirtualBase.o
BC=$(ls "${OUT}"/*-lower.bc 2>/dev/null | head -1)
if [ -z "${BC}" ]; then
  echo "FAIL: no lowered device bitcode (*-lower.bc) produced by hipcc"
  exit 1
fi
"${LLVM_DIS}" "${BC}" -o lowered.ll

if grep -qE '^@_ZT[VTC]' lowered.ll; then
  echo "FAIL: dead vtable family globals survive in the device module:"
  grep -E '^@_ZT[VTC]' lowered.ll | cut -c1-100
  exit 1
fi
echo "PASSED"
