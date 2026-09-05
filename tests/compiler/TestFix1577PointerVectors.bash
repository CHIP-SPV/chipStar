#!/bin/bash
# Regression test for CHIP-SPV/chipStar#1577.
#
# WORKAROUND(CHIP-SPV/chipStar#1577): guards the hip-lower-pointer-vectors
# pass, which exists only because neither SPIR-V path can consume a vector of
# pointers. Delete this test together with the pass once the SPIR-V backend
# legalises <N x ptr> itself, or once LLVM stops forming that type for SPIR-V
# targets. See https://github.com/CHIP-SPV/chipStar/issues/1577.
#
# The test runs the real chipStar pipeline and then real SPIR-V emission, so
# without the pass it fails the way the issue reports: the in-tree backend
# aborts inside SPIRVEmitIntrinsics, and llvm-spirv refuses the module for
# want of SPV_INTEL_masked_gather_scatter.
#
# chipStar supports translator-only toolchains (scripts/configure_llvm.sh
# builds LLVM_TARGETS="host" for those), so the backend half is checked only
# where the SPIR-V target exists. At least one emitter must be exercised: a
# run that checks neither is a pass that proves nothing, and fails instead.
set -eu

SRC_DIR="@CMAKE_CURRENT_SOURCE_DIR@"
OPT="@LLVM_TOOLS_BINARY_DIR@/opt"
CLANG="@LLVM_TOOLS_BINARY_DIR@/clang"
LLVM_SPIRV="@LLVM_SPIRV@"
PLUGIN="@CMAKE_BINARY_DIR@/lib/libLLVMHipSpvPasses.so"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

[ -x "${OPT}" ] || { echo "opt not found at ${OPT}; skipping"; exit 0; }
[ -f "${PLUGIN}" ] || { echo "pass plugin not built; skipping"; exit 0; }

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}"

# Does this toolchain have the in-tree SPIR-V backend at all? Probe it with a
# module that is valid for any target, so a failure can only mean the target
# is missing, never that the module was rejected.
printf 'define void @probe() {\nentry:\n  ret void\n}\n' > probe.ll
if "${CLANG}" -cc1 -triple spirv64v1.3-unknown-chipstar -emit-obj \
        probe.ll -o probe.img > probe.log 2>&1; then
  HAVE_BACKEND=1
else
  HAVE_BACKEND=0
  echo "note: no in-tree SPIR-V backend in this toolchain, skipping that half"
  grep -iE "unable to create target" probe.log | head -1 || true
fi

HAVE_TRANSLATOR=0
if [ "${LLVM_SPIRV}" != "NOT_NEEDED" ] && [ -x "${LLVM_SPIRV}" ]; then
  HAVE_TRANSLATOR=1
fi

if [ "${HAVE_BACKEND}" -eq 0 ] && [ "${HAVE_TRANSLATOR}" -eq 0 ]; then
  echo "FAIL: neither the in-tree SPIR-V backend nor llvm-spirv is available,"
  echo "      so this test would prove nothing. Check the toolchain configuration."
  exit 1
fi

# Run the pipeline chipStar actually uses before SPIR-V emission. This is
# deliberately the whole pipeline and not the individual pass, so the test
# reproduces the reported failure rather than merely asserting that some
# named pass exists.
"${OPT}" -load-pass-plugin "${PLUGIN}" -passes=hip-post-link-passes \
    "${SRC_DIR}/TestFix1577PointerVectors.ll" -o lowered.bc

# 1. The in-tree SPIR-V backend. Without the lowering this aborts with
#    "Assertion `isValidElementType(ElementType)' failed" in the
#    'SPIRV emit intrinsics' pass.
if [ "${HAVE_BACKEND}" -eq 1 ]; then
  if ! "${CLANG}" -cc1 -triple spirv64v1.3-unknown-chipstar -emit-obj \
          lowered.bc -o backend.img > backend.log 2>&1; then
    echo "FAIL: the in-tree SPIR-V backend could not emit a module holding a vector of pointers"
    grep -iE "Assertion|error:|Running pass" backend.log | head -4
    exit 1
  fi
  echo "checked: in-tree SPIR-V backend emitted the lowered module"
fi

# 2. The external translator, using the extension set chipStar permits.
#    Without the lowering this stops at
#    "RequiresExtension: ... SPV_INTEL_masked_gather_scatter".
if [ "${HAVE_TRANSLATOR}" -eq 1 ]; then
  EXTS="-all,+SPV_INTEL_function_pointers,+SPV_INTEL_subgroups"
  EXTS="${EXTS},+SPV_KHR_bit_instructions,+SPV_EXT_shader_atomic_float_add"
  if ! "${LLVM_SPIRV}" "--spirv-ext=${EXTS}" lowered.bc -o translated.spv \
          > translate.log 2>&1; then
    echo "FAIL: llvm-spirv could not translate a module holding a vector of pointers"
    head -3 translate.log
    exit 1
  fi
  echo "checked: llvm-spirv translated the lowered module"
fi

# 3. A pointer vector the rewrite cannot remove must be reported by this pass,
#    naming the issue, instead of reaching SPIR-V emission and aborting there.
if "${OPT}" -load-pass-plugin "${PLUGIN}" -passes=hip-post-link-passes \
        "${SRC_DIR}/TestFix1577PointerVectorsUnsupported.ll" -o unsupported.bc \
        > unsupported.log 2>&1; then
  echo "FAIL: a pointer vector that cannot be lowered passed through silently."
  echo "      It would abort later inside SPIR-V emission instead."
  exit 1
fi
if ! grep -q "HipLowerPointerVectors: a vector of pointers survives" unsupported.log; then
  echo "FAIL: the unsupported case failed, but without the diagnostic that names it."
  head -5 unsupported.log
  exit 1
fi
echo "checked: an unlowerable pointer vector is diagnosed by name"
grep -m1 "chipStar/issues/1577" unsupported.log || true

echo "PASSED"
