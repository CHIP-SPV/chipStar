#!/bin/bash
# Regression test for https://github.com/CHIP-SPV/chipStar/issues/2217: a local linkage address holder must keep default visibility.
set -eu

OPT="@LLVM_TOOLS_BINARY_DIR@/opt"
PLUGIN="@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"

cat > m.ll <<'EOF'
target triple = "spirv64-unknown-unknown"

@G = internal addrspace(1) externally_initialized global i32 0, align 4

define spir_kernel void @k(ptr addrspace(1) %out) {
  %v = load i32, ptr addrspace(1) @G, align 4
  store i32 %v, ptr addrspace(1) %out, align 4
  ret void
}
EOF
"${OPT}" -load-pass-plugin "${PLUGIN}" -passes=hip-post-link-passes -S m.ll -o m.out.ll
H=$(grep '^@__chip_var_addr_G ' m.out.ll)
echo "${H}"
[[ "${H}" == *" internal "* && "${H}" != *hidden* ]]
echo PASSED
