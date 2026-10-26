#!/bin/bash
# Canary for https://github.com/CHIP-SPV/chipStar/issues/1476. MEANT TO FAIL EVENTUALLY.
# Asserts the SPIR-V producers still reject the float-to-integer rounding
# intrinsics and llvm.ldexp that HipLowerRoundIntrinsicsPass expands. ldexp is
# fixed upstream by https://github.com/llvm/llvm-project/pull/195402 (main, not
# release/23.x) and https://github.com/KhronosGroup/SPIRV-LLVM-Translator/pull/3608
# (llvm_release_230 only); lrint/llrint have no upstream fix in either producer.
# Only pairs rejected on every supported LLVM (21, 22, 23) are probed. When it
# fires for a pair, drop that case from the pass once every supported LLVM
# accepts it; delete llvm_passes/workarounds/HipLowerRoundIntrinsics.* and this file when
# all have fired, then close #1476.
set -u
LLVM_AS="@LLVM_TOOLS_BINARY_DIR@/llvm-as"
LLC="@LLVM_TOOLS_BINARY_DIR@/llc"
LLVM_SPIRV="@LLVM_SPIRV@"
cd "$(mktemp -d)"

emit() { # $1 tool (tr|be), $2 declaration, $3 call producing %r of type $4
  printf 'target triple = "spirv64-unknown-unknown"\ndeclare %s\ndefine spir_kernel void @k(float %%x, double %%d, ptr addrspace(1) %%p) {\n  %s\n  store %s %%r, ptr addrspace(1) %%p\n  ret void\n}\n' "$2" "$3" "$4" > m.ll
  if [ "$1" = tr ]; then
    { "$LLVM_AS" m.ll -o m.bc && "$LLVM_SPIRV" m.bc -o m.spv; } > m.log 2>&1
  else
    "$LLC" -mtriple=spirv64v1.3-unknown-chipstar -filetype=obj m.ll -o m.spv > m.log 2>&1
  fi
}

TOOLS=""
[ "$LLVM_SPIRV" != NOT_NEEDED ] && [ -x "$LLVM_SPIRV" ] && TOOLS="$TOOLS tr"
"$LLC" --version | grep -q spirv64 && TOOLS="$TOOLS be"
[ -n "$TOOLS" ] || { echo "HIP_SKIP_THIS_TEST: no SPIR-V producer"; exit 0; }

for T in $TOOLS; do
  emit $T 'float @llvm.rint.f32(float)' '%r = call float @llvm.rint.f32(float %x)' float ||
    { echo "FAIL: the control module does not compile with $T"; cat m.log; exit 1; }
done

while IFS='|' read -r T DECL CALL TY SIG; do
  case " $TOOLS " in *" $T "*) ;; *) continue ;; esac
  if emit "$T" "$DECL" "$CALL" "$TY"; then
    echo "CANARY FIRED: $T now accepts $DECL, see the header"; exit 1
  fi
  grep -q "$SIG" m.log || { echo "FAIL: $T rejected $DECL for another reason"; cat m.log; exit 1; }
done <<'LIST'
tr|i64 @llvm.lrint.i64.f32(float)|%r = call i64 @llvm.lrint.i64.f32(float %x)|i64|Unexpected llvm intrinsic
tr|i64 @llvm.llrint.i64.f64(double)|%r = call i64 @llvm.llrint.i64.f64(double %d)|i64|Unexpected llvm intrinsic
tr|i64 @llvm.llround.i64.f32(float)|%r = call i64 @llvm.llround.i64.f32(float %x)|i64|Unexpected llvm intrinsic
be|i64 @llvm.lrint.i64.f32(float)|%r = call i64 @llvm.lrint.i64.f32(float %x)|i64|unable to legalize instruction
be|i64 @llvm.llrint.i64.f64(double)|%r = call i64 @llvm.llrint.i64.f64(double %d)|i64|unable to legalize instruction
be|double @llvm.ldexp.f64.i32(double, i32)|%r = call double @llvm.ldexp.f64.i32(double %d, i32 3)|double|unable to legalize instruction
LIST
echo PASSED
