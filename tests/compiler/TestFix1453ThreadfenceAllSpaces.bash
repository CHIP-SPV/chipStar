#!/bin/bash
# Reproducer for CHIP-SPV/chipStar#1453: __threadfence_block/__threadfence/
# __threadfence_system must be seq_cst fences over local AND global memory
# (flags 3) at work_group (1), device (2) and all_svm_devices (3) scope.
set -eu

HIPCC="@CMAKE_BINARY_DIR@/bin/hipcc"
LLVM_DIS="@LLVM_TOOLS_BINARY_DIR@/llvm-dis"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"
cat > k.hip <<'HIP'
#include <hip/hip_runtime.h>
__global__ void k() {
  __threadfence_block();
  __threadfence();
  __threadfence_system();
}
HIP
"${HIPCC}" --save-temps=cwd -c k.hip -o k.o
"${LLVM_DIS}" "$(ls ./*-lower.bc | head -1)" -o lowered.ll

F='_Z22atomic_work_item_fencej12memory_order12memory_scope'
GOT=$(grep -oE 'call spir_func void @_Z[0-9]+(mem_fence|atomic_work_item_fence)[^)]*\)' \
  lowered.ll | sed -E 's/call spir_func void @//; s/i32 noundef //g' | tr '\n' ' ')
WANT="${F}(3, 5, 1) ${F}(3, 5, 2) ${F}(3, 5, 3) "
if [ "${GOT}" != "${WANT}" ]; then
  echo "FAIL: fences lowered to: ${GOT}"
  echo "      expected:          ${WANT}"
  exit 1
fi
echo PASSED
