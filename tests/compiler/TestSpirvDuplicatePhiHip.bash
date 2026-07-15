#!/bin/bash
# Regression test for llvm-patches/spirv-translator/0005
# (coalesce duplicate-predecessor OpPhi). Compiles a small HIP kernel that
# mirrors a real-world crash pattern, keeps the lowered device bitcode, translates
# it to SPIR-V with the build's llvm-spirv, and validates it.
#
# Without 0005 the SPIR-V writer emits an OpPhi that lists a predecessor block
# more than once and spirv-val rejects the module; with 0005 it is valid.
#
set -eu

SRC_DIR="@CMAKE_CURRENT_SOURCE_DIR@"
HIPCC="@CMAKE_BINARY_DIR@/bin/hipcc"
LLVM_SPIRV="@LLVM_SPIRV@"
SPIRV_VAL="@CMAKE_BINARY_DIR@/external/spirv-tools/bin/spirv-val"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

# 0005 patches the external SPIRV-LLVM-Translator. When chipStar uses the in-tree
# LLVM SPIR-V backend (LLVM_SPIRV=NOT_NEEDED) the patch does not apply -- skip.
if [ "${LLVM_SPIRV}" = "NOT_NEEDED" ] || [ ! -x "${LLVM_SPIRV}" ]; then
  echo "external llvm-spirv not in use; skipping (0005 not applicable)"
  exit 0
fi
if [ ! -x "${SPIRV_VAL}" ]; then
  echo "spirv-val not found; skipping"
  exit 0
fi

rm -rf "${OUT}"
mkdir -p "${OUT}"
cd "${OUT}"

# -O3 so the optimizer threads the shared-return switch; --save-temps keeps the
# lowered device bitcode (the input the SPIR-V writer consumes).
"${HIPCC}" -O3 --save-temps=cwd -c "${SRC_DIR}/TestSpirvDuplicatePhiHip.hip" \
  -o "${OUT}/TestSpirvDuplicatePhiHip.o"

BC=$(ls "${OUT}"/*-generic-lower.bc 2>/dev/null | head -1)
if [ -z "${BC}" ]; then
  echo "FAIL: no lowered device bitcode (*-generic-lower.bc) produced by hipcc"
  exit 1
fi

"${LLVM_SPIRV}" --spirv-max-version=1.2 "${BC}" -o "${OUT}/TestSpirvDuplicatePhiHip.spv"
"${SPIRV_VAL}" "${OUT}/TestSpirvDuplicatePhiHip.spv"
echo "PASSED"
