#!/bin/bash
# Regression test for CHIP-SPV/chipStar#2247: llvm-spirv emits a function named in a global initializer before the bodyless declarations, but SPIR-V requires declarations first.
set -eu

OPT="@LLVM_TOOLS_BINARY_DIR@/opt"
PLUGIN="@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"

cat > m.ll <<'EOT'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64-unknown-unknown"

@Fns = addrspace(1) constant [1 x ptr] [ptr @one]

declare spir_func i64 @_Z13__chip_balloti(i32)

define spir_func i32 @one(i32 %x) {
  ret i32 %x
}

define spir_kernel void @k(ptr addrspace(1) %out) {
  %b = call spir_func i64 @_Z13__chip_balloti(i32 1)
  %f = load ptr, ptr addrspace(1) @Fns
  %r = call spir_func i32 %f(i32 7)
  %t = trunc i64 %b to i32
  %s = add i32 %r, %t
  store i32 %s, ptr addrspace(1) %out
  ret void
}
EOT
"${OPT}" -load-pass-plugin "${PLUGIN}" -passes=hip-post-link-passes -verify-each m.ll -o m.bc
"@LLVM_SPIRV@" --spirv-ext=+SPV_INTEL_function_pointers --spirv-text m.bc -o m.spt
grep -E ' (Function|Label|FunctionEnd) ' m.spt
# A bodyless Function (no Label before FunctionEnd) must not follow a defined one.
if awk '$2 == "Function" { fn = 1; body = 0 }
        $2 == "Label" && fn { body = 1 }
        $2 == "FunctionEnd" { if (!body && seendef) bad = 1; if (body) seendef = 1; fn = 0 }
        END { exit !bad }' m.spt; then
  echo "FAIL: a function declaration follows a function definition"; exit 1
fi
awk '$2 == "Function"' m.spt | grep -q . || { echo "FAIL: no functions emitted"; exit 1; }
echo PASSED
