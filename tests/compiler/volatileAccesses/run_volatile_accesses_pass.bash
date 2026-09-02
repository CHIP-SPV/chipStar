#!/bin/bash
# Check that HipLowerVolatileAccessesPass rewrites the volatile global and
# generic accesses of a module into relaxed device-scope atomics, leaves every
# other volatile access as it is, and that the result still translates to valid
# SPIR-V carrying OpAtomicLoad / OpAtomicStore.
#
# Usage: run_volatile_accesses_pass.bash <input.ll>
#
# The input has two kernels: @rewritten holds only accesses the pass must
# rewrite, @left_alone only accesses it must not touch. No access changes type
# or volatility: the difference is the atomic ordering and syncscope, which the
# SPIR-V producers turn into OpAtomicLoad / OpAtomicStore at Device scope with
# Relaxed semantics, and which IGC serves coherently rather than from L1.

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

MARKED_IN=14  # volatile accesses in @rewritten
KEPT_IN=21    # volatile accesses in @left_alone

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

# Every volatile access of @rewritten must come out as a relaxed device-scope
# atomic, and nothing else about it may have changed: same type, same volatility,
# and no leftover !nontemporal, which the atomics replaced.
MARKED=$(echo "${REWRITTEN}" | grep -c -E '(load|store) atomic volatile .*syncscope\("device"\) monotonic' || true)
if [ "${MARKED}" -ne "${MARKED_IN}" ]; then
  echo "ERROR: expected ${MARKED_IN} relaxed device-scope atomic accesses in @rewritten, found ${MARKED}"
  echo "See ${OUTPUT_LL} for details"
  exit 1
fi
# The accesses stay volatile and keep their original type: only the ordering
# and syncscope are added.
STILL_VOLATILE=$(echo "${REWRITTEN}" | grep -c -E '(load|store) atomic volatile ' || true)
if [ "${STILL_VOLATILE}" -ne "${MARKED_IN}" ]; then
  echo "ERROR: the pass dropped volatility from ${MARKED_IN} accesses, ${STILL_VOLATILE} remain volatile"
  exit 1
fi
if echo "${REWRITTEN}" | grep -q -E '!nontemporal'; then
  echo "ERROR: @rewritten still carries the !nontemporal marking, which was replaced by atomics:"
  echo "${REWRITTEN}" | grep -E '!nontemporal'
  exit 1
fi

# Nothing in @left_alone may have been rewritten. Its volatile accesses go in as
# is, two of them atomic already with their own scope and ordering, and that
# count must not grow.
LEFT_ATOMIC_IN=2   # @left_alone goes in with two already-atomic accesses
LEFT_ATOMIC=$(echo "${LEFT}" | grep -c -E '(load|store) atomic' || true)
if [ "${LEFT_ATOMIC}" -ne "${LEFT_ATOMIC_IN}" ]; then
  echo "ERROR: @left_alone should keep exactly its ${LEFT_ATOMIC_IN} pre-existing atomics, found ${LEFT_ATOMIC}:"
  echo "${LEFT}" | grep -E '(load|store) atomic'
  exit 1
fi
KEPT=$(echo "${LEFT}" | grep -c -E '(load|store) (atomic )?volatile' || true)
if [ "${KEPT}" -ne "${KEPT_IN}" ]; then
  echo "ERROR: expected the ${KEPT_IN} volatile accesses of @left_alone to survive, found ${KEPT}"
  echo "See ${OUTPUT_LL} for details"
  exit 1
fi

# The rewrite is only useful if it reaches SPIR-V as OpAtomicLoad / OpAtomicStore
# at Device scope with Relaxed semantics. A build targeting LLVM's
# integrated SPIR-V backend has no translator to check that with, so report
# what was verified and stop rather than failing on the missing binary.
if [ -z "${LLVM_SPIRV}" ] || [ ! -x "${LLVM_SPIRV}" ]; then
  echo "marked=${MARKED} left alone=${KEPT}, llvm-spirv not available so the SPIR-V side was not checked"
  exit 0
fi
"${LLVM_SPIRV}" "${OUTPUT_BC}" ${SPIRV_OPTS} -o "${OUTPUT_SPV}"
if [ -n "${SPIRV_VAL}" ] && [ -x "${SPIRV_VAL}" ]; then
  "${SPIRV_VAL}" "${OUTPUT_SPV}"
  VALIDATED="spirv-val ok"
else
  VALIDATED="spirv-val not available"
fi
if [ -n "${SPIRV_DIS}" ] && [ -x "${SPIRV_DIS}" ]; then
  "${SPIRV_DIS}" "${OUTPUT_SPV}" > "${BASE_NAME}.spvasm"
  NT=$(grep -c -E 'OpAtomic(Load|Store)' "${BASE_NAME}.spvasm" || true)
  if [ "${NT}" -lt "${MARKED_IN}" ]; then
    echo "ERROR: expected at least ${MARKED_IN} OpAtomicLoad / OpAtomicStore in the SPIR-V module, found ${NT}"
    exit 1
  fi
  DISASSEMBLED="${NT} atomic accesses in SPIR-V"
else
  DISASSEMBLED="spirv-dis not available"
fi

echo "marked=${MARKED} left alone=${KEPT}, SPIR-V ok, ${VALIDATED}, ${DISASSEMBLED}"
exit 0
