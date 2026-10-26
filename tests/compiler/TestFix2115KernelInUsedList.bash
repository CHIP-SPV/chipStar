#!/bin/bash
# Regression test for CHIP-SPV/chipStar#2115: HipGlobalVariablesPass must not crash on a kernel in llvm.used.
set -eu

OPT="@LLVM_TOOLS_BINARY_DIR@/opt"
PLUGIN="@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"

cat > m.ll <<'EOT'
target triple = "spirv64-unknown-unknown"

@G = hidden addrspace(1) externally_initialized global i32 0, align 4

define spir_kernel void @k(ptr addrspace(1) %out) {
  %v = load i32, ptr addrspace(1) @G, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}

@llvm.used = appending global [1 x ptr] [ptr @k], section "llvm.metadata"
EOT
"${OPT}" -load-pass-plugin "${PLUGIN}" -passes=hip-post-link-passes -verify-each m.ll -o m.bc
echo PASSED
