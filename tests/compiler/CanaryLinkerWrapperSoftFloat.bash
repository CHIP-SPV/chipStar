#!/bin/bash
# Canary for https://github.com/CHIP-SPV/chipStar/issues/1952. MEANT TO FAIL once
# clang-linker-wrapper builds its host object with the host ABI; then delete the
# riscv64 USE_NEW_OFFLOAD_DRIVER default, this file and its CMake line.
set -euo pipefail

BIN="@LLVM_TOOLS_BINARY_DIR@"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"

cat > a.hip <<'EOF'
#define __global__ __attribute__((global))
struct dim3 { unsigned x, y, z; };
typedef int hipError_t; typedef void *hipStream_t;
extern "C" hipError_t hipLaunchKernel(const void *, dim3, dim3, void **, unsigned long, hipStream_t);
extern "C" __global__ void k(int *p) { *p = 1; }
int main() { return 0; }
EOF
"${BIN}/clang++" -x hip --offload=spirv64 --offload-new-driver -nogpuinc \
  -nogpulib -fgpu-rdc -c a.hip -o a.o || { echo "FAIL: a.hip does not compile"; exit 1; }
# The link is expected to fail on the ABI mismatch; only the wrapper object matters.
"${BIN}/clang++" --hip-link --offload=spirv64 --offload-new-driver -fgpu-rdc \
  -no-hip-rt a.o -o a.out -save-temps || true
[ -f a.out.hip.image.wrapper.o ] || { echo "FAIL: no wrapper object"; exit 1; }

H=$("${BIN}/llvm-readelf" -h a.o | grep Flags:)
W=$("${BIN}/llvm-readelf" -h a.out.hip.image.wrapper.o | grep Flags:)
echo "a.o $H"; echo "wrapper $W"
[[ $H == *double-float* ]] || { echo "FAIL: the host object is not lp64d"; exit 1; }
if [[ $W == *double-float* ]]; then
  echo "CANARY FIRED: the wrapper object now has the host ABI, see the header"; exit 1
fi
echo PASSED
