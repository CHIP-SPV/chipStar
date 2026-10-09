#!/bin/bash
# Reproducer for CHIP-SPV/chipStar#1911: __syncwarp() must fence local
# (__shared__) memory, i.e. pass CLK_LOCAL_MEM_FENCE (1) to sub_group_barrier.
set -eu

HIPCC="@CMAKE_BINARY_DIR@/bin/hipcc"
LLVM_DIS="@LLVM_TOOLS_BINARY_DIR@/llvm-dis"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"
cat > k.hip <<'HIP'
#include <hip/hip_runtime.h>
__global__ void k(int *o) {
  __shared__ int s[64];
  s[threadIdx.x] = threadIdx.x;
  __syncwarp();
  o[threadIdx.x] = s[threadIdx.x ^ 1];
}
HIP
"${HIPCC}" --save-temps=cwd -c k.hip -o k.o
"${LLVM_DIS}" "$(ls ./*-lower.bc | head -1)" -o lowered.ll

FLAGS=$(grep -oE 'call spir_func void @_Z17sub_group_barrierj\(i32 noundef [0-9]+\)' \
  lowered.ll | grep -oE '[0-9]+\)$' | tr -d ')')
[ -n "${FLAGS}" ] || { echo "FAIL: no sub_group_barrier call"; exit 1; }
for F in ${FLAGS}; do
  if [ $((F & 1)) -eq 0 ]; then
    echo "FAIL: __syncwarp lowers to sub_group_barrier(${F}), no CLK_LOCAL_MEM_FENCE"
    exit 1
  fi
done
echo PASSED
