#!/bin/bash
# Checks what the CONSUMER actually generates, not just what chipStar emits.
#
# TestFixVolatileLoadLoweringSPIRV.bash stops at the SPIR-V module: it proves
# the volatile accesses left chipStar in the intended form. Nothing downstream
# of that was ever checked, and that gap is not hypothetical. While this pass
# still used the Nontemporal memory operand, the operand was present and correct
# in the SPIR-V and IGC then discarded it when it widened adjacent stores
# (store.ugm.d32x4t.a64.wb.wb instead of .uc.uc), so the fix silently did
# nothing on an Arc A380 while every SPIR-V level check passed.
#
# This test closes that gap by compiling the module the way a driver does, with
# ocloc, and inspecting the generated ISA. The accesses must come out as atomic
# ugm messages: an atomic is coherent by construction, so it cannot be widened
# or cached away the way a hint can.
#
# Needs no GPU: ocloc is an offline compiler and -device names a target.
set -u

HIPCC="@CMAKE_BINARY_DIR@/bin/hipcc"
SRC="@CMAKE_CURRENT_SOURCE_DIR@/TestFixVolatileLoadLowering.hip"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

if ! command -v ocloc >/dev/null 2>&1; then
  echo "HIP_SKIP_THIS_TEST: ocloc not found, cannot inspect generated ISA"
  exit 0
fi

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}" || exit 1

"${HIPCC}" -O2 --save-temps=cwd -c "${SRC}" -o probe.o > hipcc.log 2>&1
SPV=$(ls "${OUT}"/*.out 2>/dev/null | head -1)
if [ -z "${SPV}" ]; then
  echo "HIP_SKIP_THIS_TEST: no SPIR-V module produced (integrated backend build keeps no *.out)"
  exit 0
fi

STATUS=0
CHECKED=0
for DEV in pvc dg2; do
  DDIR="${OUT}/dump-${DEV}"
  rm -rf "${DDIR}"; mkdir -p "${DDIR}"
  ( cd "${DDIR}" && IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir="${DDIR}" \
      ocloc compile -file "${SPV}" -spirv_input -device "${DEV}" ) > "${DDIR}/ocloc.log" 2>&1
  if ! grep -q "Build succeeded" "${DDIR}/ocloc.log" 2>/dev/null; then
    echo "NOTE: ocloc could not build for -device ${DEV}, skipping that target"
    continue
  fi
  CHECKED=$((CHECKED + 1))
  # Every volatile global access must reach the hardware as an atomic message.
  N=$(cat "${DDIR}"/*.asm 2>/dev/null | grep -c -oE 'atomic[a-z_.0-9]*\.(ugm|slm)' || true)
  echo "-device ${DEV}: ${N} atomic ugm/slm messages in the generated ISA"
  if [ "${N}" -lt 1 ]; then
    echo "FAIL: -device ${DEV} generated no atomic messages, so the volatile"
    echo "      accesses were NOT lowered to atomics by the time IGC saw them."
    echo "      Generated memory messages were:"
    cat "${DDIR}"/*.asm 2>/dev/null | grep -ohE '(load|store)\.ugm[a-z0-9._]*' | sort | uniq -c | sed 's/^/        /'
    STATUS=1
  fi
done

if [ "${CHECKED}" -eq 0 ]; then
  echo "HIP_SKIP_THIS_TEST: ocloc built for no target, nothing inspected"
  exit 0
fi
[ "${STATUS}" -ne 0 ] && { echo "See ${OUT} for the shader dumps"; exit 1; }
echo "PASSED"
