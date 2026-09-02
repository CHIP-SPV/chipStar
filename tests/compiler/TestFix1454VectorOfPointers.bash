#!/bin/bash
# Regression test for CHIP-SPV/chipStar#1454.
#
# Device IR that materialises a vector of pointers can only be translated to
# SPIR-V when SPV_INTEL_masked_gather_scatter is permitted. chipStar pins the
# allowed set in the HIPSPV driver with `--spirv-ext=-all,...`, so the
# extension is not merely absent by default, it is switched off, and
# llvm-spirv rejects such a module with
#   RequiresExtension: ... SPV_INTEL_masked_gather_scatter
# rocPRIM's test/rocprim/test_device_reduce_by_key.cpp is a real trigger:
# without the extension the whole rocPRIM build fails and rocThrust,
# rocSPARSE, hipSPARSE, hipMM and zeroRK fail after it.
#
# The driver does not expose its extension list through -### or -v, and no
# small HIP source reliably produces a vector of pointers (adjacent pointer
# copies, indexed gathers and scatters and an array-of-pointers dereference
# were all tried and none vectorised), so the guard has two halves:
#   1. the chipStar patch that sets the list must still list the extension
#   2. the extension must still be what makes such a module translatable
set -eu

SRC_DIR="@CMAKE_CURRENT_SOURCE_DIR@"
LLVM_SPIRV="@LLVM_SPIRV@"
# llvm-as sits beside llvm-spirv in the same LLVM bin directory.
LLVM_AS="$(dirname "@LLVM_SPIRV@")/llvm-as"
SPIRV_VAL="@CMAKE_BINARY_DIR@/external/spirv-tools/bin/spirv-val"
PATCH="@CMAKE_SOURCE_DIR@/llvm-patches/llvm-@LLVM_VERSION_MAJOR@/llvm/0001-hipspv-in-tree-spirv-backend.patch"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

# Part 1: the driver patch must still permit the extension.
if [ ! -f "${PATCH}" ]; then
  echo "no HIPSPV driver patch for LLVM @LLVM_VERSION_MAJOR@; skipping"
  exit 0
fi
if ! grep -q "SPV_INTEL_masked_gather_scatter" "${PATCH}"; then
  echo "FAIL: ${PATCH} no longer permits SPV_INTEL_masked_gather_scatter."
  echo "      Device code holding a vector of pointers will fail to translate."
  exit 1
fi
echo "driver patch permits SPV_INTEL_masked_gather_scatter"

# Part 2: show that this is still the extension that matters. Uses llvm-spirv
# directly, so it is meaningful on the translator configuration only; with the
# in-tree backend the same input aborts inside SPIRVEmitIntrinsics before any
# extension is consulted, which is the unresolved half of #1454.
if [ "${LLVM_SPIRV}" = "NOT_NEEDED" ] || [ ! -x "${LLVM_SPIRV}" ]; then
  echo "external llvm-spirv not in use; skipping the translation half"
  exit 0
fi
if [ ! -x "${SPIRV_VAL}" ] || [ ! -x "${LLVM_AS}" ]; then
  echo "spirv-val or llvm-as not found; skipping the translation half"
  exit 0
fi

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"
BASE="-all,+SPV_INTEL_function_pointers,+SPV_INTEL_subgroups,+SPV_KHR_bit_instructions,+SPV_EXT_shader_atomic_float_add"
"${LLVM_AS}" "${SRC_DIR}/TestFix1454VectorOfPointers.ll" -o vecptr.bc

# Without the extension the module must still be rejected. If this ever
# succeeds the extension is no longer needed and this test can go.
if "${LLVM_SPIRV}" "--spirv-ext=${BASE}" vecptr.bc -o without.spv 2>/dev/null; then
  echo "FAIL: translation unexpectedly succeeded without the extension"
  exit 1
fi

# With it, translation must succeed and the result must validate.
"${LLVM_SPIRV}" "--spirv-ext=${BASE},+SPV_INTEL_masked_gather_scatter" vecptr.bc -o with.spv
"${SPIRV_VAL}" with.spv
echo "PASSED"
