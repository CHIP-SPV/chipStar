#!/bin/bash
# Check that the post-link pass pipeline leaves behind no 8 or 16 bit atomic
# instruction, and that the result still translates to valid SPIR-V.
#
# Usage: run_subword_atomics_pass.bash <input.ll>
#
# OpenCL SPIR-V consumers implement 32 and 64 bit atomics only. An
# OpAtomicLoad / OpAtomicStore / OpAtomicCompareExchange / OpAtomicIAdd ... on
# OpTypeInt 8 or 16 passes spirv-val and only dies inside the driver's JIT
# (IGC: "undefined reference to `_Z18__spirv_AtomicLoadPU3AS4cii'" followed by
# "backend compiler failed build."), which takes every kernel in the module
# down with it. HipLowerSubwordAtomicsPass rewrites those onto the containing
# 32 bit word; this looks at the module after the pipeline to make sure none
# survived. See CHIP-SPV/chipStar#1497.

set -e

if [ $# -ne 1 ]; then
  echo "Usage: $0 <input.ll>"
  exit 1
fi

INPUT_FILE="$1"
BASE_NAME=$(basename "${INPUT_FILE}" .ll)
OUTPUT_BC="${BASE_NAME}.lowered.bc"
OUTPUT_LL="${BASE_NAME}.lowered.ll"
OUTPUT_SPV="${BASE_NAME}.lowered.spv"
SPIRV_OPTS="--spirv-max-version=1.2 --spirv-ext=-all,+SPV_INTEL_function_pointers,+SPV_INTEL_subgroups"

# CHIP_VERIFY_MODE=off: the in-pass IR->SPIR-V re-verification defaults to on
# in Debug builds and is redundant here; the translation below is the check.
CHIP_VERIFY_MODE=off "${LLVM_OPT}" -load-pass-plugin "${HIP_SPV_PASSES_LIB}" \
  -passes=hip-post-link-passes "${INPUT_FILE}" -o "${OUTPUT_BC}"
"${LLVM_DIS}" "${OUTPUT_BC}" -o "${OUTPUT_LL}"

KERNELS=$(grep -c 'spir_kernel' "${OUTPUT_LL}" || true)
if [ "${KERNELS}" -eq 0 ]; then
  echo "ERROR: no spir_kernel survived the pass pipeline; test input is stale"
  exit 1
fi

# load atomic i8 / store atomic i16 %v / atomicrmw add ptr %p, i8 %v /
# cmpxchg weak ptr %p, i16 %c, i16 %n / atomicrmw fadd ptr %p, half %v
LEFTOVER=$(grep -E -e 'load atomic (volatile )?(i8|i16|half|bfloat),' \
                   -e 'store atomic (volatile )?(i8|i16|half|bfloat) ' \
                   -e 'atomicrmw .*, (i8|i16|half|bfloat) ' \
                   -e 'cmpxchg .*, (i8|i16) ' "${OUTPUT_LL}" || true)
if [ -n "${LEFTOVER}" ]; then
  echo "ERROR: 8/16 bit atomic instruction(s) survived the pass pipeline:"
  echo "${LEFTOVER}"
  echo "See ${OUTPUT_LL} for details"
  exit 1
fi

# The lowering must land on 32 bit atomics, not delete the operations.
WORD_CAS=$(grep -c -E 'cmpxchg .*, i32 ' "${OUTPUT_LL}" || true)
WORD_LOAD=$(grep -c -E 'load atomic (volatile )?i32,' "${OUTPUT_LL}" || true)
if [ "${WORD_CAS}" -eq 0 ] || [ "${WORD_LOAD}" -eq 0 ]; then
  echo "ERROR: expected 32 bit cmpxchg and atomic loads after lowering" \
       "(cmpxchg=${WORD_CAS} loads=${WORD_LOAD})"
  echo "See ${OUTPUT_LL} for details"
  exit 1
fi

# The rewrite is only useful if the result is valid SPIR-V.
"${LLVM_SPIRV}" "${OUTPUT_BC}" ${SPIRV_OPTS} -o "${OUTPUT_SPV}"
if [ -n "${SPIRV_VAL}" ] && [ -x "${SPIRV_VAL}" ]; then
  "${SPIRV_VAL}" --target-env opencl2.2 "${OUTPUT_SPV}"
  VALIDATED="spirv-val ok"
else
  VALIDATED="spirv-val not available"
fi

echo "kernels=${KERNELS} no 8/16 bit atomics, word cmpxchg=${WORD_CAS}" \
     "word loads=${WORD_LOAD}, SPIR-V ok, ${VALIDATED}"
exit 0
