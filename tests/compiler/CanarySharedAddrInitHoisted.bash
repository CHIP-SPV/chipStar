#!/bin/bash
# Canary for https://github.com/CHIP-SPV/chipStar/issues/1691. MEANT TO FAIL
# EVENTUALLY: it asserts clang still hoists a local array initializer that
# takes __shared__ addresses into a constant global copied with llvm.memcpy.
# Once it prints CANARY FIRED on every supported LLVM, delete
# HipSharedAddrLocalInitPass, its registration and the LLVMSPIRV.h include in
# llvm_passes/HipPasses.cpp, this file and its line in CMakeLists.txt. The
# upstream issue is https://github.com/llvm/llvm-project/issues/198078.
set -eu

CLANG="@LLVM_TOOLS_BINARY_DIR@/clang"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"

cat > canary.hip <<'EOF'
__attribute__((global)) void k(unsigned long *Out, unsigned C) {
  __attribute__((shared)) float A[4], B[4];
  float *Buf[2] = {A, B};
  *Out = (unsigned long)Buf[C & 1];
}
EOF
"${CLANG}" -cc1 -triple spirv64 -aux-triple x86_64-unknown-linux-gnu \
    -fcuda-is-device -x hip -emit-llvm -O0 canary.hip -o canary.ll

if ! grep -q '^@[^ ]* = .* constant .*(ptr addrspace(3) @' canary.ll ||
   ! grep -q 'call void @llvm\.memcpy' canary.ll; then
  echo "CANARY FIRED: clang no longer hoists the shared address initializer."
  echo "  HipSharedAddrLocalInitPass may be obsolete; see this file's header."
  exit 1
fi
echo "canary: clang still hoists the initializer, the workaround is still needed"
echo "PASSED"
