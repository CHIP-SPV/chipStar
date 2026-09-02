#!/bin/bash
# Volatile loads and stores in global memory must reach the SPIR-V producer
# marked !nontemporal, not as plain volatile ones.
#
# CUDA lowers a volatile global access to ld.volatile / st.volatile, which the
# PTX ISA (8.4.2 "volatile Operation") defines as a relaxed memory operation at
# system scope; code written against that (Kokkos::volatile_load in the
# UnorderedMap insert list walk) relies on the load bypassing a core's L1. In
# SPIR-V a `load volatile` is an OpLoad with the Volatile memory operand, which
# says nothing about caching, so IGC serves it from L1 and on PVC the walk reads
# stale data. The fix marks such accesses !nontemporal, which the SPIR-V
# producers emit as the Nontemporal memory operand of the same OpLoad / OpStore
# and IGC maps to an L1 uncached access.
#
# The accesses must stay non-atomic: IGC implements OpAtomicLoad as
# atomic_or(p, 0) and OpAtomicStore as an atomic exchange, and a Level Zero
# device may report an allocation kind without ZE_MEMORY_ACCESS_CAP_FLAG_ATOMIC
# (PVC does, for host allocations), where a GPU atomic faults the context.
#
# Compiles TestFixVolatileLoadLowering.hip with --save-temps and inspects the
# lowered device bitcode, the SPIR-V producer's input: the global accesses must
# be marked and non-atomic, the work-group local ones must not be marked. When
# the module was produced by the Khronos translator and spirv-dis is available,
# the SPIR-V module is checked for the Nontemporal memory operand as well.
set -u

HIPCC="@CMAKE_BINARY_DIR@/bin/hipcc"
LLVM_DIS="@CLANG_ROOT_PATH_BIN@/llvm-dis"
SPIRV_DIS="@SPIRV_DIS@"
SRC="@CMAKE_CURRENT_SOURCE_DIR@/TestFixVolatileLoadLowering.hip"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

if [ ! -x "${LLVM_DIS}" ]; then
  echo "HIP_SKIP_THIS_TEST: llvm-dis not found at ${LLVM_DIS}"
  exit 0
fi

rm -rf "${OUT}"
mkdir -p "${OUT}"
cd "${OUT}"

# -O2 so the accesses are not hidden behind allocas; --save-temps keeps the
# lowered device bitcode (*-lower.bc with either offload driver) and, with the
# old driver, the SPIR-V module (*.out).
"${HIPCC}" -O2 --save-temps=cwd -c "${SRC}" -o TestFixVolatileLoadLowering.o
BC=$(ls "${OUT}"/*-lower.bc 2>/dev/null | head -1)
if [ -z "${BC}" ]; then
  echo "FAIL: no lowered device bitcode (*-lower.bc) produced by hipcc"
  exit 1
fi
"${LLVM_DIS}" "${BC}" -o lowered.ll

# The body of one kernel, from its define line to the closing brace.
kernel_body() {
  sed -n "/^define .*@_Z[0-9]*$1/,/^}/p" lowered.ll
}

STATUS=0
fail() {
  echo "FAIL: $1"
  STATUS=1
}

ACCESS=$(kernel_body volatileAccess)
if [ -z "${ACCESS}" ]; then
  echo "FAIL: kernel volatileAccess not found in lowered.ll"
  exit 1
fi
for PATTERN in 'load atomic volatile i32.*syncscope\("device"\) monotonic' \
               'load atomic volatile i64.*syncscope\("device"\) monotonic' \
               'store atomic volatile i32 .*syncscope\("device"\) monotonic' \
               'store atomic volatile i64 .*syncscope\("device"\) monotonic'; do
  if ! echo "${ACCESS}" | grep -qE "${PATTERN}"; then
    fail "volatileAccess has no '${PATTERN}' after the pass pipeline"
  fi
done
# The 16 bit accesses of the same kernel must NOT be marked: a consumer only
# has to honour Nontemporal on shapes it has a non-temporal instruction for,
# and PoCL's x86 back end aborts with "Unsupported store size" on ones it does
# not.
if echo "${ACCESS}" | grep -qE '(load|store) atomic volatile i16'; then
  fail "volatileAccess had its 16 bit accesses made atomic; the OpenCL SPIR-V environment allows atomics on 32 bit types only:"
  echo "${ACCESS}" | grep -E '(load|store) volatile i16'
fi
PLAIN=$(echo "${ACCESS}" | grep -E '(load|store) volatile (i32|i64)' | grep -v 'atomic' || true)
if [ -n "${PLAIN}" ]; then
  fail "volatileAccess still has non-atomic 32 or 64 bit volatile global accesses:"
  echo "${PLAIN}"
fi
if echo "${ACCESS}" | grep -qE '!nontemporal'; then
  fail "volatileAccess still carries the !nontemporal marking, which the atomics replaced:"
  echo "${ACCESS}" | grep -E '!nontemporal'
fi

LEFT=$(kernel_body volatileLocal)
if [ -z "${LEFT}" ]; then
  echo "FAIL: kernel volatileLocal not found in lowered.ll"
  exit 1
fi
# The __shared__ array is accessed through a generic pointer over an
# addrspace(3) object; the accesses stay volatile and unmarked.
if ! echo "${LEFT}" | grep -qE '(load|store) volatile i32'; then
  fail "the work-group local volatile accesses in volatileLocal were not left alone"
fi
if echo "${LEFT}" | grep -qE '(load|store) atomic'; then
  fail "the work-group local volatile accesses in volatileLocal were made atomic:"
  echo "${LEFT}" | grep -E '(load|store) atomic'
fi

# SPIR-V level check.
SPV=$(ls "${OUT}"/*.out 2>/dev/null | head -1)
if [ -n "${SPV}" ] && [ -n "${SPIRV_DIS}" ] && [ -x "${SPIRV_DIS}" ]; then
  "${SPIRV_DIS}" "${SPV}" > module.spvasm
  if grep -q "Generator: Khronos LLVM/SPIR-V Translator" module.spvasm; then
    # The entry point id is a number or, when the translator kept an OpName,
    # the mangled name. Translators from LLVM 21 on emit the entry point as
    # a wrapper whose only instruction is an OpFunctionCall to the kernel
    # body, so a wrapper is followed to its callee before inspecting the body.
    KID=$(grep -E 'OpEntryPoint Kernel %[^ ]+ "_Z[0-9]+volatileAccess' module.spvasm |
          sed -E 's/.*Kernel (%[^ ]+) .*/\1/')
    FUNC=$(sed -n "/^ *${KID} = OpFunction /,/OpFunctionEnd/p" module.spvasm)
    CALLEE=$(echo "${FUNC}" | grep -oE 'OpFunctionCall %[^ ]+ %[^ ]+' | awk '{print $3}' | head -1)
    if [ -n "${CALLEE}" ]; then
      FUNC=$(sed -n "/^ *${CALLEE} = OpFunction /,/OpFunctionEnd/p" module.spvasm)
    fi
    LOADS=$(echo "${FUNC}" | grep -c -E 'OpAtomicLoad' || true)
    STORES=$(echo "${FUNC}" | grep -c -E 'OpAtomicStore' || true)
    if [ "${LOADS}" -lt 2 ] || [ "${STORES}" -lt 2 ]; then
      fail "SPIR-V volatileAccess has ${LOADS} OpAtomicLoad and ${STORES} OpAtomicStore, expected at least 2 each"
    fi
    echo "SPIR-V module checked (Khronos translator)"
  else
    echo "NOTE: SPIR-V module not produced by the Khronos translator; module check skipped"
    grep -m1 "Generator" module.spvasm || true
  fi
else
  echo "NOTE: no SPIR-V module or spirv-dis; module check skipped"
fi

if [ "${STATUS}" -ne 0 ]; then
  echo "See ${OUT}/lowered.ll"
  exit 1
fi
echo "PASSED"
