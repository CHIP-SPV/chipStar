#!/bin/bash
# Checks what the CONSUMER actually generates, not just what chipStar emits.
#
# TestFixVolatileLoadLoweringSPIRV.bash stops at the SPIR-V module: it proves
# the volatile accesses left chipStar in the intended form. Nothing downstream
# of that was ever checked, and that gap is not hypothetical: a cache hint can
# be present and correct in the SPIR-V and still be discarded by IGC when it
# widens adjacent stores (store.ugm.d32x4t.a64.wb.wb instead of .uc.uc), which
# made an Arc A380 run unfixed code while every SPIR-V level check passed.
#
# This test closes that gap by compiling the module the way a driver does, with
# ocloc, and inspecting the generated ISA. Under the atomic lowering the
# accesses must come out as atomic ugm messages: an atomic is coherent by
# construction, so it cannot be widened or cached away the way a hint can.
# Under the cache-control lowering they must come out as ordinary messages that
# still carry the L1-uncached control, at both widths and in both directions,
# on the parts that keep it.
#
# Not every part does. IGC's stateless-to-stateful promotion rewrites a
# decorated indexed access to a bindless a32 message and drops the control, so
# on dg2, mtl and arl the lowering is a no-op and those parts need the atomic
# fallback. Each part is asserted against what it actually does, which makes
# this fail in both directions: a control lost where one is expected, and a
# control kept on a part that is only on the fallback because it drops one. The
# second is the canary: it goes red the day IGC stops promoting, and the part
# comes off the fallback list. Only that indexed shape is gated; the
# uniform-address 64 bit store is described in HipLowerVolatileAccesses.cpp,
# untested. See CHIP-SPV/chipStar#1616.
#
# Needs no GPU: ocloc is an offline compiler and -device names a target.
set -u

# Set by cmake from CHIP_ATOMICS_CACHE_BYPASS_WORKAROUND: "atomic" or "cachectl".
LOWERING="@VOLATILE_LOWERING@"

HIPCC="@CMAKE_BINARY_DIR@/bin/hipcc"
SRC="@CMAKE_CURRENT_SOURCE_DIR@/TestFixVolatileLoadLowering.hip"
OUT="@CMAKE_CURRENT_BINARY_DIR@/@TEST_NAME@.d"

if ! command -v ocloc >/dev/null 2>&1; then
  echo "HIP_SKIP_THIS_TEST: ocloc not found, cannot inspect generated ISA"
  exit 0
fi

rm -rf "${OUT}"; mkdir -p "${OUT}"; cd "${OUT}" || exit 1

"${HIPCC}" -O2 --save-temps=cwd -c "${SRC}" -o probe.o > hipcc.log 2>&1
# --save-temps leaves the SPIR-V module as a *.img under clang's new offload
# driver, the default from LLVM 23 on, and as a *.out under the old one, and a
# new-driver *.out is a clang offload binary rather than a module. So the file
# is chosen by its magic number, and finding none is a failure: it means this
# gate would otherwise pass without inspecting anything.
SPV=""
for CAND in "${OUT}"/*.img "${OUT}"/*.out; do
  [ -f "${CAND}" ] || continue
  case "$(od -An -tx1 -N4 "${CAND}" | tr -d ' \n')" in
    03022307|07230203) SPV="${CAND}"; break ;;
  esac
done
if [ -z "${SPV}" ]; then
  echo "FAIL: hipcc produced no SPIR-V module under ${OUT}, which holds:"
  ls -1 "${OUT}" | sed 's/^/        /'
  echo "See ${OUT}/hipcc.log"
  exit 1
fi

STATUS=0
CHECKED=""
for DEV in pvc bmg dg2 mtl arl; do
  DDIR="${OUT}/dump-${DEV}"
  rm -rf "${DDIR}"; mkdir -p "${DDIR}"
  ( cd "${DDIR}" && IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir="${DDIR}" \
      ocloc compile -file "${SPV}" -spirv_input -device "${DEV}" ) > "${DDIR}/ocloc.log" 2>&1
  if ! grep -q "Build succeeded" "${DDIR}/ocloc.log" 2>/dev/null; then
    echo "NOTE: ocloc could not build for -device ${DEV}, skipping that target"
    continue
  fi
  CHECKED="${CHECKED} ${DEV}"
  if [ "${LOWERING}" = "atomic" ]; then
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
    continue
  fi

  # IGC dumps one .asm per kernel and names it on the first line.
  ASM=$(grep -l '^//\.kernel _Z[0-9]*volatileAccess' "${DDIR}"/*.asm 2>/dev/null | head -1)
  if [ -z "${ASM}" ]; then
    echo "FAIL: -device ${DEV} produced no ISA dump for the volatileAccess kernel"
    STATUS=1
    continue
  fi
  echo "-device ${DEV}: memory messages of volatileAccess:"
  grep -ohE '(load|store)[a-z0-9_.]*\.ugm[a-z0-9_.]*' "${ASM}" | sort | uniq -c | sed 's/^/        /'
  case "${DEV}" in
    pvc|bmg) KEEPS=1 ;;
    *)       KEEPS=0 ;;
  esac
  # A 64 bit access is d64 or, where IGC splits it, d32x2; the trailing t marks
  # a transposed (uniform address) message.
  for DIR in load store; do
    for WIDTH in 32 64; do
      if [ "${WIDTH}" = "32" ]; then
        SHAPE='d32(x1)?t?'
      else
        SHAPE='(d64(x1)?t?|d32x2t?)'
      fi
      if grep -qE "${DIR}\.ugm\.${SHAPE}\.a[0-9]+\.uc" "${ASM}"; then
        if [ "${KEEPS}" = "0" ]; then
          echo "FAIL: -device ${DEV} kept the .uc cache control on the ${WIDTH} bit ${DIR}."
          echo "      IGC no longer drops it here, so this part does not need"
          echo "      -DCHIP_ATOMICS_CACHE_BYPASS_WORKAROUND=ON. Take ${DEV} out of the"
          echo "      part list in HipLowerVolatileAccesses.cpp and out of the case above."
          STATUS=1
        fi
      elif [ "${KEEPS}" = "1" ]; then
        echo "FAIL: -device ${DEV} generated no ${WIDTH} bit ${DIR} carrying the .uc"
        echo "      cache control, so the decoration was dropped and the volatile"
        echo "      ${DIR} still hits the core-private cache."
        STATUS=1
      fi
    done
  done
done

if [ -z "${CHECKED}" ]; then
  echo "HIP_SKIP_THIS_TEST: ocloc built for no target, nothing inspected"
  exit 0
fi
[ "${STATUS}" -ne 0 ] && { echo "See ${OUT} for the shader dumps"; exit 1; }
echo "PASSED"
