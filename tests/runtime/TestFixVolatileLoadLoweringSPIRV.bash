#!/bin/bash
# Volatile loads and stores of 32 and 64 bit values in global memory must reach
# the SPIR-V producer as relaxed atomic accesses, not as plain volatile ones.
#
# CUDA lowers a volatile global access to ld.volatile / st.volatile, which the
# PTX ISA (8.4.2 "volatile Operation") defines as a relaxed memory operation at
# system scope; code written against that (Kokkos::volatile_load in the
# UnorderedMap insert list walk) relies on the load bypassing a core's L1. In
# SPIR-V a `load volatile` is an OpLoad with the Volatile memory operand, which
# IGC serves from L1 like any other load, so on PVC the walk reads stale data.
# The fix lowers such accesses to `load atomic ... syncscope("device")
# monotonic`, which the SPIR-V producers emit as OpAtomicLoad with Relaxed
# semantics at Device scope.
#
# Compiles TestFixVolatileLoadLowering.hip with --save-temps and inspects the
# lowered device bitcode, the SPIR-V producer's input: the 32 and 64 bit global
# accesses must be atomic, the 16 bit and work-group local ones must not. When
# the module was produced by the Khronos translator and spirv-dis is available,
# the SPIR-V module is checked for OpAtomicLoad / OpAtomicStore as well.
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
for PATTERN in 'load atomic (volatile )?i32,' 'load atomic (volatile )?i64,' \
               'store atomic (volatile )?i32 ' 'store atomic (volatile )?i64 '; do
  if ! echo "${ACCESS}" | grep -qE "${PATTERN}"; then
    fail "volatileAccess has no '${PATTERN}' after the pass pipeline"
  fi
done
PLAIN=$(echo "${ACCESS}" | grep -E 'load volatile (i32|i64),|store volatile (i32|i64) ' || true)
if [ -n "${PLAIN}" ]; then
  fail "volatileAccess still has non-atomic volatile global accesses:"
  echo "${PLAIN}"
fi

LEFT=$(kernel_body volatileLeftAlone)
if [ -z "${LEFT}" ]; then
  echo "FAIL: kernel volatileLeftAlone not found in lowered.ll"
  exit 1
fi
if ! echo "${LEFT}" | grep -qE 'load volatile i16,'; then
  fail "the 16 bit volatile load in volatileLeftAlone was not left alone"
fi
# The __shared__ array is accessed through a generic pointer over an
# addrspace(3) object; the accesses stay volatile and non-atomic either way.
if ! echo "${LEFT}" | grep -qE '(load|store) volatile i32'; then
  fail "the work-group local volatile accesses in volatileLeftAlone were not left alone"
fi
if echo "${LEFT}" | grep -qE 'atomic'; then
  fail "volatileLeftAlone gained atomic accesses:"
  echo "${LEFT}" | grep -E 'atomic'
fi

# SPIR-V level check. The LLVM SPIR-V backend selects every load as OpLoad
# regardless of its atomic ordering (SPIRVInstructionSelector::selectLoad), so
# the module check is only meaningful for the Khronos translator.
SPV=$(ls "${OUT}"/*.out 2>/dev/null | head -1)
if [ -n "${SPV}" ] && [ -n "${SPIRV_DIS}" ] && [ -x "${SPIRV_DIS}" ]; then
  "${SPIRV_DIS}" "${SPV}" > module.spvasm
  if grep -q "Generator: Khronos LLVM/SPIR-V Translator" module.spvasm; then
    KID=$(grep -E 'OpEntryPoint Kernel %[0-9]+ "_Z[0-9]+volatileAccess' module.spvasm |
          sed -E 's/.*Kernel (%[0-9]+) .*/\1/')
    FUNC=$(sed -n "/^ *${KID} = OpFunction /,/OpFunctionEnd/p" module.spvasm)
    LOADS=$(echo "${FUNC}" | grep -c 'OpAtomicLoad' || true)
    STORES=$(echo "${FUNC}" | grep -c 'OpAtomicStore' || true)
    if [ "${LOADS}" -lt 2 ] || [ "${STORES}" -lt 2 ]; then
      fail "SPIR-V volatileAccess has ${LOADS} OpAtomicLoad and ${STORES} OpAtomicStore, expected at least 2 each"
    fi
    if echo "${FUNC}" | grep -qE 'Op(Load|Store) .*Volatile'; then
      fail "SPIR-V volatileAccess still has Volatile OpLoad / OpStore:"
      echo "${FUNC}" | grep -E 'Op(Load|Store) .*Volatile'
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
