#!/bin/bash
# hip_bf16.h must be includable from more than one translation unit of a
# relocatable-device-code build.
#
# spirv_hip_bf16.h defined its free functions as plain '__device__' with no
# 'inline', so every TU emitted a strong device definition. Without RDC each TU
# becomes its own SPIR-V module and nothing notices; with -fgpu-rdc all the
# device modules are llvm-linked together and the link dies with
#
#   error: Linking globals named '_Z20__bfloat162bfloat16214__hip_bfloat16':
#          symbol multiply defined!
#
# That broke every -fgpu-rdc application, which is the configuration Kokkos
# requires for device-side virtual dispatch
# (Kokkos_ENABLE_HIP_RELOCATABLE_DEVICE_CODE, used by kynema-ugf / Nalu-Wind).
#
# This needs its own driver because the failure only shows up at the device
# link of two TUs, which no single-file compile test reaches.
set -eu

OUT_DIR=@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d
HIPCC=@CMAKE_BINARY_DIR@/bin/hipcc

rm -rf ${OUT_DIR}
mkdir -p ${OUT_DIR}

cat > ${OUT_DIR}/a.hip <<'SRC'
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <cstdio>
__global__ void ka(__hip_bfloat16 *Out) { *Out = __hip_bfloat16(1.5f); }
void useB();
int main() { useB(); printf("PASSED\n"); return 0; }
SRC

cat > ${OUT_DIR}/b.hip <<'SRC'
#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
__global__ void kb(__hip_bfloat162 *Out, __hip_bfloat16 X) {
  *Out = __bfloat162bfloat162(X);
}
void useB() {}
SRC

${HIPCC} -fgpu-rdc -c ${OUT_DIR}/a.hip -o ${OUT_DIR}/a.o
${HIPCC} -fgpu-rdc -c ${OUT_DIR}/b.hip -o ${OUT_DIR}/b.o
${HIPCC} -fgpu-rdc ${OUT_DIR}/a.o ${OUT_DIR}/b.o -o ${OUT_DIR}/bf
${OUT_DIR}/bf
