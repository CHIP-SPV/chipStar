#!/bin/bash
# Regression test for CHIP-SPV/chipStar#2110: HipGlobalVariablesPass must not leave poison in llvm.compiler.used.
set -eu

OPT="@LLVM_TOOLS_BINARY_DIR@/opt"
PLUGIN="@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"

cat > m.ll <<'EOF'
target triple = "spirv64-unknown-unknown"

@G = hidden addrspace(1) externally_initialized global i32 0, align 4
@H = addrspace(1) global i32 0, align 4
@llvm.compiler.used = appending addrspace(1) global [2 x ptr addrspace(4)] [ptr addrspace(4) addrspacecast (ptr addrspace(1) @G to ptr addrspace(4)), ptr addrspace(4) addrspacecast (ptr addrspace(1) @H to ptr addrspace(4))], section "llvm.metadata"

define spir_kernel void @k(ptr addrspace(1) %out) {
  %v = load i32, ptr addrspace(1) @G, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}
EOF
"${OPT}" -load-pass-plugin "${PLUGIN}" -passes=hip-post-link-passes -verify-each m.ll -o m.bc
echo PASSED
