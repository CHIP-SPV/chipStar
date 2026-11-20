#!/bin/bash
# Canary for https://github.com/CHIP-SPV/chipStar/issues/2247. MEANT TO FAIL EVENTUALLY.
# Asserts bare llvm-spirv still emits a bodyless declaration after a definition when a
# global initializer names the definition, i.e. still lacks
# https://github.com/KhronosGroup/SPIRV-LLVM-Translator/pull/4092.
# When it fires: delete HipDeclOrderPass in llvm_passes/HipPasses.cpp, its registration,
# TestFix2247DeclBeforeDef if it no longer adds anything, and this canary together in one
# commit, then close #2247.
set -u
cd "$(mktemp -d)"
cat > m.ll <<'EOT'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64-unknown-unknown"

@t = addrspace(1) constant [2 x ptr] [ptr @f, ptr @g]

define spir_func void @f() {
  ret void
}

declare spir_func void @g()
EOT
{ "@LLVM_TOOLS_BINARY_DIR@/llvm-as" m.ll -o m.bc &&
  "@LLVM_SPIRV@" --spirv-ext=+SPV_INTEL_function_pointers --spirv-text m.bc -o m.spt; } > m.log 2>&1 ||
  { echo "FAIL: the module does not translate"; cat m.log; exit 1; }
grep -E ' (Function|Label|FunctionEnd) ' m.spt
# Bad shape: a Function with no Label before FunctionEnd comes after a defined one.
if awk '$2 == "Function" { fn = 1; body = 0 }
        $2 == "Label" && fn { body = 1 }
        $2 == "FunctionEnd" { if (!body && seendef) bad = 1; if (body) seendef = 1; fn = 0 }
        END { exit !bad }' m.spt; then
  echo PASSED
else
  echo "CANARY FIRED: llvm-spirv now emits declarations before definitions, see the header"; exit 1
fi
