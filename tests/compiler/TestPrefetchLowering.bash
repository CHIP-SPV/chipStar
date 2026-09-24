#!/bin/bash
# Every __builtin_prefetch on a global pointer in TestBuiltinPrefetch.hip must
# reach SPIR-V emission as the OpenCL prefetch builtin; the __shared__ one is
# dropped, since OpenCL.std prefetch takes only a CrossWorkgroup pointer.
set -eu

SRC="@CMAKE_CURRENT_SOURCE_DIR@/TestBuiltinPrefetch.hip"
HIPCC="@CMAKE_BINARY_DIR@/bin/hipcc"
LLVM_DIS="@LLVM_TOOLS_BINARY_DIR@/llvm-dis"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

SPIRV_VAL="@CMAKE_BINARY_DIR@/external/spirv-tools/bin/spirv-val"
[ -x "${SPIRV_VAL}" ] || SPIRV_VAL=$(command -v spirv-val || true)

for OPT in -O0 -O2; do
  rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"
  "${HIPCC}" ${OPT} --save-temps=cwd -c "${SRC}" -o k.o
  "${LLVM_DIS}" "$(ls ./*-lower.bc | head -1)" -o lowered.ll
  N=$(grep -c 'call spir_func void @_Z8prefetchPU3AS1Kcm(ptr addrspace(1) .*, i64 1)' lowered.ll || true)
  if [ "${N}" -ne 3 ] || grep -q 'llvm\.prefetch' lowered.ll; then
    echo "FAIL ${OPT}: expected 3 prefetch builtin calls and no llvm.prefetch, found ${N}"
    grep -E 'prefetch' lowered.ll || true
    exit 1
  fi
  if [ -x "${SPIRV_VAL}" ]; then
    SPV=0
    for F in ./*; do
      [ -f "${F}" ] && [ "$(od -An -tx1 -N4 "${F}" | tr -d ' \n')" = "03022307" ] || continue
      "${SPIRV_VAL}" "${F}"
      SPV=$((SPV + 1))
    done
    [ "${SPV}" -ge 1 ] || { echo "FAIL ${OPT}: hipcc emitted no SPIR-V module"; exit 1; }
  fi
done
echo "PASSED"
