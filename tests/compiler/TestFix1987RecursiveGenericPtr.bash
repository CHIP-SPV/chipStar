#!/bin/bash
# Reproducer for CHIP-SPV/chipStar#1987: IGC must build hipcc's SPIR-V for a
# recursive function with a generic pointer argument. Needs ocloc, no GPU.
set -u
if ! command -v ocloc >/dev/null 2>&1; then
  echo "HIP_SKIP_THIS_TEST: ocloc not found"
  exit 0
fi
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"
rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"
cat > k.hip <<'HIP'
#include <hip/hip_runtime.h>
__device__ void fib(int *Acc, int N) {
  if (N < 2) {
    *Acc += N;
    return;
  }
  fib(Acc, N - 1);
  fib(Acc, N - 2);
}
__global__ void k(int *Out, int N) { int Acc = 0; fib(&Acc, N); *Out = Acc; }
HIP
"@CMAKE_BINARY_DIR@/bin/hipcc" -O2 --save-temps=cwd -c k.hip -o k.o > hipcc.log 2>&1 ||
  { echo "FAIL: hipcc"; cat hipcc.log; exit 1; }
SPV=0
for F in ./*; do
  [ -f "${F}" ] && [ "$(od -An -tx1 -N4 "${F}" | tr -d ' \n')" = "03022307" ] || continue
  SPV=$((SPV + 1))
  for DEV in bmg pvc; do
    ocloc compile -spirv_input -file "${F}" -device ${DEV} -output o > ocloc.log 2>&1 ||
      { echo "FAIL: ocloc exit $? building ${F} for ${DEV}"; cat ocloc.log; exit 1; }
  done
done
[ "${SPV}" -ge 1 ] || { echo "FAIL: hipcc emitted no SPIR-V module"; ls; exit 1; }
echo PASSED
