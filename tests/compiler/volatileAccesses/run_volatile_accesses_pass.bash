#!/bin/bash
# Check that HipLowerVolatileAccessesPass rewrites the volatile global and
# generic accesses of a module, leaves every other volatile access as it is,
# and that the result still translates to valid SPIR-V.
#
# Usage: run_volatile_accesses_pass.bash <input.ll>
#
# The pass has two lowerings and the build picks one, so the shape asserted
# here follows the build rather than being fixed:
#
#   cachectl  the default. The access keeps its type and stays a plain volatile
#             load or store; its pointer becomes a GEP decorated
#             CacheControlLoadINTEL / CacheControlStoreINTEL level 0
#             UncachedINTEL, which SPV_INTEL_cache_controls turns into an
#             L1-uncached access.
#   atomic    CHIP_ATOMICS_CACHE_BYPASS_WORKAROUND=ON, for consumers that
#             reject the extension. The access gains device scope and monotonic
#             ordering and becomes OpAtomicLoad / OpAtomicStore.
#
# The input has two kernels: @rewritten holds only accesses the pass must
# rewrite, @left_alone only accesses it must not touch. In neither lowering may
# an access change type or lose volatility.

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

# Set by cmake from CHIP_ATOMICS_CACHE_BYPASS_WORKAROUND: "atomic" or "cachectl".
LOWERING="@VOLATILE_LOWERING@"
if [ "${LOWERING}" = "cachectl" ]; then
  # The decorations do not translate without the extension that defines them.
  SPIRV_OPTS="${SPIRV_OPTS},+SPV_INTEL_cache_controls"
fi

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

if [ "${LOWERING}" = "atomic" ]; then
  # Every volatile access of @rewritten must come out as a relaxed device-scope
  # atomic, and nothing else about it may have changed: same type, same
  # volatility.
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
else
  # Every volatile access of @rewritten must reach a pointer carrying the
  # cache-control decoration, and must itself stay a plain, non-atomic,
  # volatile access of the same type.
  MARKED=$(echo "${REWRITTEN}" | grep -c -E 'getelementptr .*!spirv\.Decorations' || true)
  if [ "${MARKED}" -ne "${MARKED_IN}" ]; then
    echo "ERROR: expected ${MARKED_IN} decorated pointers in @rewritten, found ${MARKED}"
    echo "See ${OUTPUT_LL} for details"
    exit 1
  fi
  STILL_VOLATILE=$(echo "${REWRITTEN}" | grep -c -E '(load|store) volatile ' || true)
  if [ "${STILL_VOLATILE}" -ne "${MARKED_IN}" ]; then
    echo "ERROR: expected ${MARKED_IN} plain volatile accesses in @rewritten, found ${STILL_VOLATILE}"
    exit 1
  fi
  # The whole point of this lowering is that it does not make the access atomic,
  # which is illegal on an allocation whose device reports no atomic support.
  ADDED_ATOMIC=$(echo "${REWRITTEN}" | grep -c -E '(load|store) atomic' || true)
  if [ "${ADDED_ATOMIC}" -ne 0 ]; then
    echo "ERROR: the cache-control lowering made ${ADDED_ATOMIC} accesses atomic"
    exit 1
  fi
  # The decoration operands are load/store 6442/6443, cache level 0, UncachedINTEL 0.
  for OPCODE in 6442 6443; do
    if ! grep -q -E "^![0-9]+ = !\\{i32 ${OPCODE}, i32 0, i32 0\\}" "${OUTPUT_LL}"; then
      echo "ERROR: no !{i32 ${OPCODE}, i32 0, i32 0} decoration in ${OUTPUT_LL}"
      exit 1
    fi
  done
fi
# Neither lowering emits !nontemporal: IGC drops it when it widens adjacent
# stores, so it cannot carry this guarantee.
if echo "${REWRITTEN}" | grep -q -E '!nontemporal'; then
  echo "ERROR: @rewritten carries a !nontemporal marking, which no lowering emits:"
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

# The rewrite is only useful if it survives into SPIR-V. A build targeting
# LLVM's integrated SPIR-V backend has no translator to check that with, so
# report what was verified and stop rather than failing on the missing binary.
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
  if [ "${LOWERING}" = "atomic" ]; then
    WANT='OpAtomic(Load|Store)'
    WHAT="OpAtomicLoad / OpAtomicStore"
  else
    WANT='CacheControl(Load|Store)INTEL 0 UncachedINTEL'
    WHAT="CacheControlLoadINTEL / CacheControlStoreINTEL"
  fi
  NT=$(grep -c -E "${WANT}" "${BASE_NAME}.spvasm" || true)
  if [ "${NT}" -lt "${MARKED_IN}" ]; then
    echo "ERROR: expected at least ${MARKED_IN} ${WHAT} in the SPIR-V module, found ${NT}"
    exit 1
  fi
  DISASSEMBLED="${NT} ${LOWERING} accesses in SPIR-V"
else
  DISASSEMBLED="spirv-dis not available"
fi

echo "marked=${MARKED} left alone=${KEPT}, SPIR-V ok, ${VALIDATED}, ${DISASSEMBLED}"
exit 0
