#!/bin/bash
# Canary for https://github.com/CHIP-SPV/chipStar/issues/1420. MEANT TO FAIL EVENTUALLY.
# Asserts the configured llvm-spirv, with the arguments the configured clang passes it, still emits a
# struct's debug types as forward references (a DebugTypeMember or DebugTypePointer
# naming a DebugTypeComposite defined later, which any struct produces), which
# spirv-val rejects unless SPV_KHR_relaxed_extended_instruction is declared; no
# chipStar consumer implements that extension. No upstream report.
# HipStripDebugInfoPass (llvm_passes/workarounds/HipStripDebugInfo.*) works around it. When this
# fires, check that the translator output with -g validates and loads on every
# consumer; then HipStripDebugInfo can be dropped for the translator lane, and
# this canary with it.
set -u
command -v spirv-val >/dev/null 2>&1 || { echo "HIP_SKIP_THIS_TEST: spirv-val not found"; exit 0; }
cd "$(mktemp -d)"
printf '%s\n' 'struct Node { struct Node *next; int v; };' \
  'kernel void k(global struct Node *n, global int *o) { *o = n->next->v; }' > d.cl
# The translator arguments the HIPSPV toolchain passes, which differ per LLVM version.
echo '__global__ void k() {}' > h.hip
TRARGS=$("@LLVM_TOOLS_BINARY_DIR@/clang" -### -x hip --offload=spirv64 -nogpulib -nogpuinc \
  --cuda-device-only -c h.hip -o h.o 2>&1 | grep '/llvm-spirv"' | grep -o '"--spirv[^"]*"' | tr -d '"')
[ -n "$TRARGS" ] || { echo "FAIL: clang -### printed no llvm-spirv arguments"; exit 1; }
echo "llvm-spirv arguments: $TRARGS"
{ "@LLVM_TOOLS_BINARY_DIR@/clang" -cc1 -triple spir64-unknown-unknown -cl-std=CL2.0 \
    -debug-info-kind=limited -dwarf-version=4 -O0 -emit-llvm-bc d.cl -o d.bc &&
  "@LLVM_SPIRV@" $TRARGS d.bc -o d.spv; } > d.log 2>&1 ||
  { echo "FAIL: the module does not translate"; cat d.log; exit 1; }
spirv-val d.spv > val.log 2>&1
cat val.log
# Bad shape: error: ID '48[%48]' has not been defined / %54 = OpExtInst %void %2 DebugTypePointer %48 ...
# (nonsemantic-shader-200 prints the instruction number: OpExtInst %void %2 3 %56 ...)
if grep -q "has not been defined" val.log && grep -q "OpExtInst" val.log; then
  echo PASSED
else
  echo "CANARY FIRED: llvm-spirv debug info no longer has the forward-referenced type cycle, see the header"; exit 1
fi
