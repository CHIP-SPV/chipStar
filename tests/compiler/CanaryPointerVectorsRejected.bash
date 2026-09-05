#!/bin/bash
# Canary for CHIP-SPV/chipStar#1577. THIS TEST IS MEANT TO FAIL EVENTUALLY.
#
# It asserts that the SPIR-V path still rejects a vector of pointers, which is
# the only reason the hip-lower-pointer-vectors pass exists. When this test
# goes red, upstream has fixed the gap: delete
# llvm_passes/HipLowerPointerVectors.*, its registration in
# llvm_passes/HipPasses.cpp and llvm_passes/CMakeLists.txt,
# tests/compiler/TestFix1577PointerVectors.*, and this file, then close
# https://github.com/CHIP-SPV/chipStar/issues/1577 naming the upstream fix.
#
# It probes the compiler directly rather than through chipStar, so a failure
# points at the toolchain and nothing else. It deliberately does NOT go red
# when the chipStar pass is reverted; that is the regression test's job.
#
# Two things make the assertion real rather than incidental:
#   - an availability control runs first, so "the tool could not start" can
#     never be read as "the tool rejected our module";
#   - the rejection must carry the known signature, so an unrelated failure
#     fails the test instead of quietly satisfying it.
#
# Known gap: #1577 lists a second removal condition, LLVM ceasing to form
# <N x ptr> for this pattern. This fixture writes the type by hand, so it
# cannot observe that; only the emitter-side condition is covered here.
set -eu

SRC_DIR="@CMAKE_CURRENT_SOURCE_DIR@"
CLANG="@LLVM_TOOLS_BINARY_DIR@/clang"
LLVM_AS="@LLVM_TOOLS_BINARY_DIR@/llvm-as"
LLVM_SPIRV="@LLVM_SPIRV@"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

[ -x "${CLANG}" ] || { echo "clang not found at ${CLANG}; skipping"; exit 0; }

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"

CHECKED=0

# --- in-tree SPIR-V backend -------------------------------------------------
# Control: can this toolchain emit SPIR-V at all? A module valid for any target,
# so a failure here means the target is missing, not that the module was bad.
printf 'define void @probe() {\nentry:\n  ret void\n}\n' > probe.ll
if "${CLANG}" -cc1 -triple spirv64v1.3-unknown-chipstar -emit-obj \
        probe.ll -o probe.img > probe.log 2>&1; then
  # The shell prints "Aborted" for the expected crash below. That line is the
  # evidence the canary really ran, so it is announced rather than hidden: a
  # silenced canary is indistinguishable from one that did nothing.
  echo "canary: emitting the unlowered module, an abort here is expected today"
  if "${CLANG}" -cc1 -triple spirv64v1.3-unknown-chipstar -emit-obj \
          "${SRC_DIR}/TestFix1577PointerVectors.ll" -o canary.img > canary.log 2>&1; then
    echo "CANARY FIRED: the SPIR-V backend now emits <N x ptr> without help."
    echo "  The hip-lower-pointer-vectors workaround is obsolete."
    echo "  Delete the pass, its registration, TestFix1577PointerVectors and this"
    echo "  test, and close https://github.com/CHIP-SPV/chipStar/issues/1577"
    exit 1
  fi
  if ! grep -qiE "isValidElementType|[Vv]ector of pointers|masked_gather_scatter" canary.log; then
    echo "FAIL: the backend rejected the module, but not for the reason this canary tracks."
    echo "      Expected isValidElementType / vector of pointers / masked_gather_scatter."
    head -5 canary.log
    exit 1
  fi
  echo "canary: backend still rejects <N x ptr>, so the workaround is still needed"
  grep -iE "isValidElementType|[Vv]ector of pointers|masked_gather_scatter" canary.log | head -1
  CHECKED=$((CHECKED + 1))
else
  echo "note: no in-tree SPIR-V backend in this toolchain, skipping that half"
fi

# --- external translator ----------------------------------------------------
if [ "${LLVM_SPIRV}" != "NOT_NEEDED" ] && [ -x "${LLVM_SPIRV}" ] \
        && [ -x "${LLVM_AS}" ]; then
  # llvm-spirv reads bitcode, not textual IR, so assemble the fixture first.
  "${LLVM_AS}" "${SRC_DIR}/TestFix1577PointerVectors.ll" -o canary-in.bc
  EXTS="-all,+SPV_INTEL_function_pointers,+SPV_INTEL_subgroups"
  EXTS="${EXTS},+SPV_KHR_bit_instructions,+SPV_EXT_shader_atomic_float_add"
  if "${LLVM_SPIRV}" "--spirv-ext=${EXTS}" canary-in.bc -o canary.spv \
          > canary-tr.log 2>&1; then
    echo "CANARY FIRED: llvm-spirv now translates <N x ptr> without the extension."
    echo "  The hip-lower-pointer-vectors workaround is obsolete."
    echo "  Delete the pass, its registration, TestFix1577PointerVectors and this"
    echo "  test, and close https://github.com/CHIP-SPV/chipStar/issues/1577"
    exit 1
  fi
  if ! grep -qiE "masked_gather_scatter|RequiresExtension" canary-tr.log; then
    echo "FAIL: llvm-spirv failed, but not for the reason this canary tracks."
    echo "      Expected masked_gather_scatter / RequiresExtension."
    head -5 canary-tr.log
    exit 1
  fi
  echo "canary: llvm-spirv still demands SPV_INTEL_masked_gather_scatter"
  grep -iE "masked_gather_scatter|RequiresExtension" canary-tr.log | head -1
  CHECKED=$((CHECKED + 1))
fi

if [ "${CHECKED}" -eq 0 ]; then
  echo "FAIL: no SPIR-V emitter was available, so this canary checked nothing."
  echo "      A canary that cannot observe the bug must not report success."
  exit 1
fi

echo "PASSED"
