#!/bin/bash
# Canary for https://github.com/CHIP-SPV/chipStar/issues/1616. MEANT TO FAIL EVENTUALLY.
# Asserts IGC still drops CacheControlLoadINTEL (UncachedINTEL, level 0) from an
# indexed global load on dg2, mtl and arl, where stateless-to-stateful promotion
# turns it into an a32 message, while pvc keeps it. No upstream report.
# llvm_passes/core/HipLowerVolatileAccesses.cpp works around it only by documentation: those parts
# need -DCHIP_ATOMICS_CACHE_BYPASS_WORKAROUND=ON. When this fires, rerun
# tests/runtime/TestFixVolatileLoadLoweringISA.bash in the cache-control lowering;
# once no part drops the control, delete the dg2/mtl/arl caveat from the pass
# header, the KEEPS=0 list in that test and this canary.
# Needs no GPU: ocloc is an offline compiler and -device names a target.
set -u
command -v ocloc >/dev/null 2>&1 || { echo "HIP_SKIP_THIS_TEST: ocloc not found"; exit 0; }
cd "$(mktemp -d)"
cat > m.ll <<'IR'
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spir64-unknown-unknown"
declare spir_func i64 @_Z13get_global_idj(i32)
define spir_kernel void @k(ptr addrspace(1) %p, ptr addrspace(1) %o) {
  %i = call spir_func i64 @_Z13get_global_idj(i32 0)
  %a = getelementptr inbounds i32, ptr addrspace(1) %p, i64 %i
  %d = getelementptr inbounds i32, ptr addrspace(1) %a, i64 0, !spirv.Decorations !0
  %v = load volatile i32, ptr addrspace(1) %d, align 4
  %b = getelementptr inbounds i32, ptr addrspace(1) %o, i64 %i
  store i32 %v, ptr addrspace(1) %b, align 4
  ret void
}
!0 = !{!1}
!1 = !{i32 6442, i32 0, i32 0}
IR
{ "@LLVM_TOOLS_BINARY_DIR@/llvm-as" m.ll -o m.bc &&
  "@LLVM_SPIRV@" m.bc --spirv-ext=-all,+SPV_INTEL_cache_controls -o m.spv; } > m.log 2>&1 ||
  { echo "FAIL: the module does not translate"; cat m.log; exit 1; }
uncached() { # $1 device; prints the L1-uncached load messages in the ISA
  rm -rf "d-$1" && mkdir "d-$1"
  ( cd "d-$1" && IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir="$PWD" \
      ocloc compile -file ../m.spv -spirv_input -device "$1" ) > "ocloc-$1.log" 2>&1 ||
    { echo "FAIL: ocloc did not build for $1"; cat "ocloc-$1.log"; exit 1; }
  compgen -G "d-$1/*.asm" > /dev/null || { echo "FAIL: no asm dump for $1"; exit 1; }
  cat "d-$1"/*.asm | grep -cE 'load\.ugm\.[a-z0-9]+\.a[0-9]+\.uc' || true
}
N=$(uncached pvc) || { echo "$N"; exit 1; }
[ "$N" -ge 1 ] || { echo "FAIL: pvc dropped the control too, the decoration never reached IGC"; exit 1; }
for DEV in dg2 mtl arl; do
  N=$(uncached $DEV) || { echo "$N"; exit 1; }
  if [ "$N" != 0 ]; then
    echo "CANARY FIRED: IGC keeps the cache control on $DEV, see the header"; exit 1
  fi
done
echo PASSED
