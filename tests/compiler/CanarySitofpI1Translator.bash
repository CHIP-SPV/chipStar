#!/bin/bash
# Canary for https://github.com/CHIP-SPV/chipStar/issues/1891. MEANT TO FAIL EVENTUALLY.
# Asserts the configured llvm-spirv still lowers sitofp i1 true to +1, i.e. still
# lacks https://github.com/KhronosGroup/SPIRV-LLVM-Translator/pull/3918.
# HipLowerSitofpI1Pass in llvm_passes/HipPasses.cpp works around it and also
# covers https://github.com/llvm/llvm-project/pull/209232 (CanarySitofpI1Backend).
# Delete the pass, its registration and both canaries together only once both
# canaries fire for every supported LLVM version, then close #1891.
set -u
cd "$(mktemp -d)"
printf '%s\n' 'target triple = "spirv64-unknown-unknown"' \
  'define spir_func float @f(i32 %a) {' '  %c = icmp eq i32 %a, 0' \
  '  %r = sitofp i1 %c to float' '  ret float %r' '}' > m.ll
{ "@LLVM_TOOLS_BINARY_DIR@/llvm-as" m.ll -o m.bc &&
  "@LLVM_SPIRV@" --spirv-text m.bc -o m.spt; } > m.log 2>&1 ||
  { echo "FAIL: the module does not translate"; cat m.log; exit 1; }
cat m.spt
# Bad shape: OpSelect picks the integer constant 1 (4 Constant T Id 1; 6 Select T R C True False).
if awk '$2 == "Constant" && $NF == "1" { one[$(NF-1)] = 1 }
        $2 == "Select" && one[$(NF-1)] { bad = 1 } END { exit !bad }' m.spt; then
  echo PASSED
else
  echo "CANARY FIRED: llvm-spirv no longer lowers sitofp i1 true to +1, see the header"; exit 1
fi
