; Every llvm.ldexp variant that can reach a SPIR-V device module.
;
; The value type is one of half / float / double, because those are the only
; floating point types OpenCL declares ldexp over and the only ones clang can
; produce for this target. The exponent is crossed against several integer
; widths: clang itself only ever emits i32, but InstCombine and SimplifyLibCalls
; also synthesise llvm.ldexp out of exp2(sitofp x) and pow(2, sitofp x) and keep
; the source integer's width. The vector form comes from the same rewrite when
; it fires on a vectorised expression.
;
; See CHIP-SPV/chipStar#1404.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv64"

declare double @llvm.ldexp.f64.i32(double, i32)
declare float @llvm.ldexp.f32.i32(float, i32)
declare half @llvm.ldexp.f16.i32(half, i32)
declare double @llvm.ldexp.f64.i64(double, i64)
declare float @llvm.ldexp.f32.i16(float, i16)
declare <4 x float> @llvm.ldexp.v4f32.v4i32(<4 x float>, <4 x i32>)

define spir_kernel void @ldexp_variants(ptr addrspace(1) %DOut,
                                        ptr addrspace(1) %FOut,
                                        ptr addrspace(1) %HOut,
                                        ptr addrspace(1) %VOut,
                                        i32 %E32, i64 %E64, i16 %E16,
                                        <4 x i32> %EV) {
entry:
  %d32 = call double @llvm.ldexp.f64.i32(double 3.000000e+00, i32 %E32)
  %f32 = call float @llvm.ldexp.f32.i32(float 3.000000e+00, i32 %E32)
  %h32 = call half @llvm.ldexp.f16.i32(half 0xH4200, i32 %E32)
  %d64 = call double @llvm.ldexp.f64.i64(double 3.000000e+00, i64 %E64)
  %f16 = call float @llvm.ldexp.f32.i16(float 3.000000e+00, i16 %E16)
  %vec = call <4 x float> @llvm.ldexp.v4f32.v4i32(
             <4 x float> <float 1.000000e+00, float 2.000000e+00,
                          float 3.000000e+00, float 4.000000e+00>,
             <4 x i32> %EV)

  %dsum = fadd double %d32, %d64
  %fsum = fadd float %f32, %f16
  store double %dsum, ptr addrspace(1) %DOut
  store float %fsum, ptr addrspace(1) %FOut
  store half %h32, ptr addrspace(1) %HOut
  store <4 x float> %vec, ptr addrspace(1) %VOut
  ret void
}
