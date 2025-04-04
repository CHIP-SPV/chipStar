#!/bin/bash
set -e

if [ $# -ne 1 ]; then
    echo "Usage: $0 <input.ll>"
    exit 1
fi

INPUT_FILE="$1"
BASE_NAME=$(basename "${INPUT_FILE}" .ll)
OUTPUT_BC="${BASE_NAME}.bc"
OUTPUT_LL="${BASE_NAME}.out.ll"
OUTPUT_SPV="${BASE_NAME}.spv"
SPIRV_OPTS="--spirv-max-version=1.2 --spirv-ext=-all,+SPV_INTEL_function_pointers,+SPV_INTEL_subgroups"

# Run the promote int pass
${LLVM_OPT} -load-pass-plugin "${HIP_SPV_PASSES_LIB}" \
    -passes=hip-post-link-passes \
    "${INPUT_FILE}" -o "${OUTPUT_BC}" || exit 1

# Disassemble for examination
${LLVM_DIS} "${OUTPUT_BC}" -o "${OUTPUT_LL}" || exit 1

# Check for non-standard integer types
NON_STD_INT_COUNT=$(grep -c -E 'i[0-9]+(?<!i1|i8|i16|i32|i64)' "${OUTPUT_LL}" || true)
if [ "${NON_STD_INT_COUNT}" -gt 0 ]; then
    echo "ERROR: Found ${NON_STD_INT_COUNT} non-standard integer types in output"
    echo "See ${OUTPUT_LL} for details"
    exit 1
fi

# Convert to SPIR-V to check validity
${LLVM_SPIRV} "${OUTPUT_BC}" ${SPIRV_OPTS} -o "${OUTPUT_SPV}" || exit 1

exit 0 