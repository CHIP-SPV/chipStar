#!/bin/bash
# Check that HipLowerVolatileAccessesPass turns the volatile 32 and 64 bit
# global and generic accesses of a module into relaxed device-scope atomics,
# leaves every other volatile access as it is, and that the result still
# translates to valid SPIR-V.
#
# Usage: run_volatile_accesses_pass.bash <input.ll>
#
# The input has two kernels: @rewritten holds only accesses the pass must
# rewrite, @left_alone only accesses it must not touch. Every load and store
# in @rewritten must come out `atomic volatile ... syncscope("device")
# monotonic` on i32 or i64 (floats and pointers are laundered through the
# integer), and @left_alone must contain no `syncscope("device") monotonic`
# access at all.

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
  -passes=hip-lower-volatile-accesses "${INPUT_FILE}" -o "${OUTPUT_BC}" \
  2> "${BASE_NAME}.stderr"
"${LLVM_DIS}" "${OUTPUT_BC}" -o "${OUTPUT_LL}"

kernel_body() {
  sed -n "/^define .*@$1(/,/^}/p" "${OUTPUT_LL}"
}

REWRITTEN=$(kernel_body rewritten)
LEFT=$(kernel_body left_alone)
if [ -z "${REWRITTEN}" ] || [ -z "${LEFT}" ]; then
  echo "ERROR: kernels @rewritten / @left_alone not found after the pass"
  exit 1
fi

# Volatile accesses in @rewritten that are not device-scope monotonic atomics
# on i32 / i64: the 12 volatile accesses of the input must all have become
# `load atomic volatile i32|i64 ... syncscope("device") monotonic` or the
# store equivalent.
STALE=$(echo "${REWRITTEN}" | grep -E '(load|store) volatile' || true)
if [ -n "${STALE}" ]; then
  echo "ERROR: volatile access(es) in @rewritten survived the pass:"
  echo "${STALE}"
  exit 1
fi
ATOMIC=$(echo "${REWRITTEN}" | grep -c -E '(load|store) atomic volatile i(32|64)[, ].*syncscope\("device"\) monotonic' || true)
if [ "${ATOMIC}" -ne 12 ]; then
  echo "ERROR: expected 12 device-scope monotonic i32 / i64 atomics in @rewritten, found ${ATOMIC}"
  echo "See ${OUTPUT_LL} for details"
  exit 1
fi
# Floats and pointers must be laundered through the integer, not loaded as such.
if echo "${REWRITTEN}" | grep -q -E 'atomic volatile (float|double|ptr)'; then
  echo "ERROR: float / pointer typed atomic access in @rewritten:"
  echo "${REWRITTEN}" | grep -E 'atomic volatile (float|double|ptr)'
  exit 1
fi

# Nothing in @left_alone may have been rewritten. Its 15 volatile accesses go
# in as is (two of them atomic already, with their own scope and ordering).
if echo "${LEFT}" | grep -q 'syncscope("device") monotonic'; then
  echo "ERROR: access(es) in @left_alone were rewritten:"
  echo "${LEFT}" | grep 'syncscope("device") monotonic'
  exit 1
fi
KEPT=$(echo "${LEFT}" | grep -c -E '(load|store) (atomic )?volatile' || true)
if [ "${KEPT}" -ne 15 ]; then
  echo "ERROR: expected the 15 volatile accesses of @left_alone to survive, found ${KEPT}"
  echo "See ${OUTPUT_LL} for details"
  exit 1
fi
# The two under-aligned accesses are reported, not silently skipped.
WARNED=$(grep -c "leaving under-aligned volatile access alone" "${BASE_NAME}.stderr" || true)
if [ "${WARNED}" -ne 2 ]; then
  echo "ERROR: expected 2 under-aligned access warnings, got ${WARNED}:"
  cat "${BASE_NAME}.stderr"
  exit 1
fi

# The rewrite is only useful if the result is valid SPIR-V.
"${LLVM_SPIRV}" "${OUTPUT_BC}" ${SPIRV_OPTS} -o "${OUTPUT_SPV}"
if [ -n "${SPIRV_VAL}" ] && [ -x "${SPIRV_VAL}" ]; then
  "${SPIRV_VAL}" "${OUTPUT_SPV}"
  VALIDATED="spirv-val ok"
else
  VALIDATED="spirv-val not available"
fi

echo "rewritten=${ATOMIC} left alone=${KEPT} warnings=${WARNED}, SPIR-V ok, ${VALIDATED}"
exit 0
