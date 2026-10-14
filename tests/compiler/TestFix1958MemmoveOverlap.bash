#!/bin/bash
# Regression test for CHIP-SPV/chipStar#1958: no llvm.memmove may reach the
# in-tree SPIR-V backend, which emits it as OpCopyMemorySized (read as memcpy).
set -eu

OPT="@LLVM_TOOLS_BINARY_DIR@/opt"
LLC="@LLVM_TOOLS_BINARY_DIR@/llc"
PLUGIN="@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"

# Private shifts in both directions, and generic/global pointers that may overlap.
cat > m.ll <<'EOF'
target triple = "spirv64"
define spir_kernel void @k(ptr addrspace(1) %out, ptr addrspace(4) %g, i64 %i) {
  %a = alloca [4 x i32]
  %a1 = getelementptr inbounds i8, ptr %a, i64 4
  %a2 = getelementptr inbounds i8, ptr %a, i64 8
  store i64 %i, ptr %a
  store i64 %i, ptr %a2
  call void @llvm.memmove.p0.p0.i64(ptr %a1, ptr %a, i64 12, i1 false)
  call void @llvm.memmove.p0.p0.i64(ptr %a, ptr %a1, i64 12, i1 false)
  call void @llvm.memmove.p4.p1.i64(ptr addrspace(4) %g, ptr addrspace(1) %out, i64 12, i1 false)
  %p = getelementptr inbounds i32, ptr %a, i64 %i
  %v = load i32, ptr %p
  store i32 %v, ptr addrspace(1) %out
  ret void
}
EOF
"${OPT}" -load-pass-plugin "${PLUGIN}" -passes=hip-post-link-passes m.ll -o m.bc
"${LLC}" -mtriple=spirv64v1.3-unknown-chipstar m.bc -o m.spt
if grep OpCopyMemorySized m.spt; then
  echo "FAIL: a memmove reached the SPIR-V backend as OpCopyMemorySized"
  exit 1
fi
echo PASSED
