; A template instance named like an OpenCL builtin (fma<__hip_bfloat162>,
; addBiasResidualLayerNorm-hip) must keep its body and not become llvm.fma.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64-unknown-chipstar-shadervulkan"

define spir_kernel void @k(ptr addrspace(1) %o) {
  %r = call spir_func i32 @_Z3fmaI3TagET_S1_S1_S1_(i32 1, i32 2, i32 3)
  store i32 %r, ptr addrspace(1) %o, align 4
  ret void
}

define linkonce_odr spir_func i32 @_Z3fmaI3TagET_S1_S1_S1_(i32 %a, i32 %b, i32 %c) {
  %m = mul i32 %a, %b
  %s = add i32 %m, %c
  ret i32 %s
}
