; The llvm.llround / llvm.llrint half of HipLowerRoundIntrinsicsPass, which had
; no test of its own. Guards against a regression in the rewrite that
; llvm.ldexp support was bolted onto.
;
; See CHIP-SPV/chipStar#1404.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv64"

declare i64 @llvm.llround.i64.f32(float)
declare i64 @llvm.llround.i64.f64(double)
declare i64 @llvm.llrint.i64.f32(float)
declare i64 @llvm.llrint.i64.f64(double)

define spir_kernel void @llround_variants(ptr addrspace(1) %Out, float %F,
                                          double %D) {
entry:
  %a = call i64 @llvm.llround.i64.f32(float %F)
  %b = call i64 @llvm.llround.i64.f64(double %D)
  %c = call i64 @llvm.llrint.i64.f32(float %F)
  %d = call i64 @llvm.llrint.i64.f64(double %D)
  %ab = add i64 %a, %b
  %cd = add i64 %c, %d
  %sum = add i64 %ab, %cd
  store i64 %sum, ptr addrspace(1) %Out
  ret void
}
