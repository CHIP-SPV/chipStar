#!/bin/bash
# Check that HipLowerVolatileAccessesPass marks the volatile global and generic
# accesses of a module !nontemporal, leaves every other volatile access as it
# is, and that the result still translates to valid SPIR-V carrying the
# Nontemporal memory operand.
#
# Usage: run_volatile_accesses_pass.bash <input.ll>
#
# The input has two kernels: @rewritten holds only accesses the pass must mark,
# @left_alone only accesses it must not touch. No access changes type, opcode,
# ordering or volatility: the only difference is the !nontemporal metadata,
# which the SPIR-V producers turn into the Nontemporal memory operand and IGC
# into an L1 uncached access.

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

MARKED_IN=22  # volatile accesses in @rewritten
KEPT_IN=13    # volatile accesses in @left_alone

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

# Every volatile access of @rewritten must carry !nontemporal and nothing else
# about it may have changed: no atomic form (IGC implements OpAtomicLoad as
# atomic_or and OpAtomicStore as an atomic exchange, which faults on memory a
# Level Zero device reports without ZE_MEMORY_ACCESS_CAP_FLAG_ATOMIC).
VOLATILE=$(echo "${REWRITTEN}" | grep -c -E '(load|store) volatile ' || true)
if [ "${VOLATILE}" -ne "${MARKED_IN}" ]; then
  echo "ERROR: expected the ${MARKED_IN} volatile accesses of @rewritten to stay plain volatile, found ${VOLATILE}"
  echo "See ${OUTPUT_LL} for details"
  exit 1
fi
MARKED=$(echo "${REWRITTEN}" | grep -c -E '(load|store) volatile .*!nontemporal ' || true)
if [ "${MARKED}" -ne "${MARKED_IN}" ]; then
  echo "ERROR: expected ${MARKED_IN} !nontemporal volatile accesses in @rewritten, found ${MARKED}"
  echo "See ${OUTPUT_LL} for details"
  exit 1
fi
if echo "${REWRITTEN}" | grep -q -E '(load|store) atomic'; then
  echo "ERROR: the pass turned an access of @rewritten into an atomic:"
  echo "${REWRITTEN}" | grep -E '(load|store) atomic'
  exit 1
fi

# Nothing in @left_alone may have been marked. Its volatile accesses go in as
# is (two of them atomic already, with their own scope and ordering).
if echo "${LEFT}" | grep -q -E '(load|store) .*!nontemporal '; then
  echo "ERROR: access(es) in @left_alone were marked:"
  echo "${LEFT}" | grep -E '!nontemporal '
  exit 1
fi
KEPT=$(echo "${LEFT}" | grep -c -E '(load|store) (atomic )?volatile' || true)
if [ "${KEPT}" -ne "${KEPT_IN}" ]; then
  echo "ERROR: expected the ${KEPT_IN} volatile accesses of @left_alone to survive, found ${KEPT}"
  echo "See ${OUTPUT_LL} for details"
  exit 1
fi

# The marking is only useful if it reaches SPIR-V as the Nontemporal memory
# operand of an otherwise unchanged OpLoad / OpStore.
"${LLVM_SPIRV}" "${OUTPUT_BC}" ${SPIRV_OPTS} -o "${OUTPUT_SPV}"
if [ -n "${SPIRV_VAL}" ] && [ -x "${SPIRV_VAL}" ]; then
  "${SPIRV_VAL}" "${OUTPUT_SPV}"
  VALIDATED="spirv-val ok"
else
  VALIDATED="spirv-val not available"
fi
if [ -n "${SPIRV_DIS}" ] && [ -x "${SPIRV_DIS}" ]; then
  "${SPIRV_DIS}" "${OUTPUT_SPV}" > "${BASE_NAME}.spvasm"
  NT=$(grep -c -E 'Op(Load|Store) .*Nontemporal' "${BASE_NAME}.spvasm" || true)
  if [ "${NT}" -lt "${MARKED_IN}" ]; then
    echo "ERROR: expected at least ${MARKED_IN} Nontemporal OpLoad / OpStore in the SPIR-V module, found ${NT}"
    exit 1
  fi
  DISASSEMBLED="${NT} Nontemporal accesses in SPIR-V"
else
  DISASSEMBLED="spirv-dis not available"
fi

echo "marked=${MARKED} left alone=${KEPT}, SPIR-V ok, ${VALIDATED}, ${DISASSEMBLED}"
exit 0
