#!/bin/bash
# Canary for https://github.com/CHIP-SPV/chipStar/issues/1978. MEANT TO FAIL EVENTUALLY.
# Asserts IGC still lowers a lane-plus-constant sub_group_shuffle, rotate or
# shift down, to an indirect r[a0] register gather on bmg and pvc. No upstream
# IGC report exists yet. When it fires, check every shape on every part the
# workaround serves; once none gathers, delete HipShuffleShiftPass from
# llvm_passes/HipPasses.cpp, the bitcode/shflShift_* rtdevlib modules, their
# linking in both backends, CHIP_FAST_SHUFFLE and this canary, and drop the
# speedup requirement from tests/benchmarks/shflRotate.hip.
# Needs no GPU: ocloc is an offline compiler and -device names a target.
set -u
if ! command -v ocloc >/dev/null 2>&1; then
  echo "HIP_SKIP_THIS_TEST: ocloc not found"
  exit 0
fi
cd "$(mktemp -d)"

gathers() { # $1 device, $2 shuffle expression; prints the r[a0] reads in the ISA
  rm -rf d && mkdir d
  printf '__attribute__((intel_reqd_sub_group_size(32)))\nkernel void k(global int *p) {\n  int v = p[get_global_id(0)];\n  uint l = get_sub_group_local_id();\n  p[get_global_id(0)] = %s;\n}\n' "$2" > d/k.cl
  ( cd d && IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir="$PWD" \
      ocloc compile -file k.cl -device "$1" -options -cl-std=CL2.0 ) > ocloc.log 2>&1 ||
    { echo "FAIL: ocloc did not build $2 for $1"; cat ocloc.log; exit 1; }
  cat d/*.asm | grep -c 'r\[a0' || true
}

for DEV in bmg pvc; do
  N=$(gathers $DEV 'intel_sub_group_shuffle_down(v, v, 1)') || { echo "$N"; exit 1; }
  [ "$N" = 0 ] || { echo "FAIL: the control shift already gathers through a0 on $DEV"; exit 1; }
  for E in 'sub_group_shuffle(v, (l + 1) & 31)' 'sub_group_shuffle(v, l + 1 < 32 ? l + 1 : l)'; do
    N=$(gathers $DEV "$E") || { echo "$N"; exit 1; }
    if [ "$N" = 0 ]; then
      echo "CANARY FIRED: IGC no longer gathers $E on $DEV, see the header"
      exit 1
    fi
  done
done
echo PASSED
