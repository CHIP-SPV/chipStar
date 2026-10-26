#!/bin/bash
# Canary for https://github.com/CHIP-SPV/chipStar/issues/1373. MEANT TO FAIL EVENTUALLY.
# Asserts clang still types SPIR-V C++ vtable slots in the global address space,
# casting each function address with addrspacecast (ptr @f to ptr addrspace(1)),
# which llvm-spirv rejects. The upstream fix is
# https://github.com/llvm/llvm-project/pull/212452 (open). When it fires on
# every supported LLVM, delete llvm_passes/workarounds/HipFunctionPointerAS.*, its
# registration in llvm_passes/HipPasses.cpp and llvm_passes/CMakeLists.txt and
# this file, then close #1373. The in-tree backend accepts the original shape
# (with SPV_INTEL_function_pointers), so only the translator lane needs the pass.
set -eu
CLANG="@LLVM_TOOLS_BINARY_DIR@/clang"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"
rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"

cat > vt.hip <<'HIP'
struct B { __attribute__((device)) virtual int f(int x) { return x + 1; } };
struct D : B { __attribute__((device)) int f(int x) override { return x * 2; } };
__attribute__((device)) int call(B *b, int x) { return b->f(x); }
__attribute__((global)) void k(int *o, int c) { D d; B bb; *o = call(c ? (B *)&d : &bb, c); }
HIP
"${CLANG}" -cc1 -triple spirv64 -aux-triple x86_64-unknown-linux-gnu \
    -fcuda-is-device -x hip -emit-llvm -O1 -disable-llvm-passes vt.hip -o vt.ll

grep -q '^@_ZTV1D = ' vt.ll || { echo "FAIL: clang emitted no vtable for D"; exit 1; }
if ! grep '^@_ZTV1D = ' vt.ll | grep -q 'addrspacecast (ptr @_ZN1D1fEi to ptr addrspace(1))'; then
  echo "CANARY FIRED: clang no longer casts vtable slots to addrspace(1):"
  grep '^@_ZTV1D = ' vt.ll
  echo "  HipFunctionPointerAS may be obsolete; see this file's header."
  exit 1
fi
echo "canary: clang still emits global-AS vtable slots, the workaround is still needed"
echo PASSED
