#!/bin/bash
# Check that HipLowerRoundIntrinsicsPass leaves behind no llvm math intrinsic
# llvm-spirv cannot translate, and that the result really does translate.
#
# Usage: run_math_intrinsics_pass.bash <input.ll>
#
# llvm.llround, llvm.llrint and llvm.ldexp all reach hipspv-link intact and all
# die there with
#
#   InvalidFunctionCall: Unexpected llvm intrinsic: <name>
#
# with no source location, so the only cheap way to catch a regression is to
# look at the module after the pass pipeline and then hand it to the translator.
# See CHIP-SPV/chipStar#1404.

set -e

if [ $# -ne 1 ]; then
  echo "Usage: $0 <input.ll>"
  exit 1
fi

INPUT_FILE="$1"
BASE_NAME=$(basename "${INPUT_FILE}" .ll)
OUTPUT_BC="${BASE_NAME}.math.bc"
OUTPUT_LL="${BASE_NAME}.math.ll"
OUTPUT_SPV="${BASE_NAME}.math.spv"
SPIRV_OPTS="--spirv-max-version=1.2 --spirv-ext=-all,+SPV_INTEL_function_pointers,+SPV_INTEL_subgroups"

# CHIP_VERIFY_MODE=off for the same reason promoteInt does it: the in-pass
# IR->SPIR-V re-verification is on by default in Debug builds and is redundant
# here, the translation below is the check.
CHIP_VERIFY_MODE=off "${LLVM_OPT}" -load-pass-plugin "${HIP_SPV_PASSES_LIB}" \
  -passes=hip-post-link-passes "${INPUT_FILE}" -o "${OUTPUT_BC}"
"${LLVM_DIS}" "${OUTPUT_BC}" -o "${OUTPUT_LL}"

KERNELS=$(grep -c 'spir_kernel' "${OUTPUT_LL}" || true)
if [ "${KERNELS}" -eq 0 ]; then
  echo "ERROR: no spir_kernel survived the pass pipeline; test input is stale"
  exit 1
fi

LEFTOVER=$(grep -o -E 'llvm\.(llround|llrint|ldexp)\.[a-z0-9.]+' "${OUTPUT_LL}" |
           sort -u || true)
if [ -n "${LEFTOVER}" ]; then
  echo "ERROR: untranslatable intrinsic(s) survived the pass pipeline:"
  echo "${LEFTOVER}"
  echo "See ${OUTPUT_LL} for details"
  exit 1
fi

# The rewrite is only useful if the translator actually accepts the result.
"${LLVM_SPIRV}" "${OUTPUT_BC}" ${SPIRV_OPTS} -o "${OUTPUT_SPV}"

echo "kernels=${KERNELS} no untranslatable math intrinsics, SPIR-V ok"
exit 0
