#!/bin/bash
# Regression test for CHIP-SPV/chipStar#2108: HipAbortPass must not leave a null in llvm.compiler.used.
set -eu

OPT="@LLVM_TOOLS_BINARY_DIR@/opt"
PLUGIN="@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"

cat > m.ll <<'EOF'
target triple = "spirv64"
@__chipspv_abort_called = weak addrspace(1) global i32 0
@__chipspv_abort_msg = weak addrspace(1) global { i32, [508 x i8] } zeroinitializer
@llvm.compiler.used = appending addrspace(1) global [2 x ptr addrspace(4)] [ptr addrspace(4) addrspacecast (ptr addrspace(1) @__chipspv_abort_called to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @__chipspv_abort_msg to ptr addrspace(4))], section "llvm.metadata"
define spir_kernel void @k(ptr addrspace(1) %out) {
  store i32 1, ptr addrspace(1) %out
  ret void
}
EOF
"${OPT}" -load-pass-plugin "${PLUGIN}" -passes=hip-post-link-passes -verify-each m.ll -o m.bc
echo PASSED
