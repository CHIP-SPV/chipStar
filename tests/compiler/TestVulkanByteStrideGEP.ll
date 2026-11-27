; A GEP stepping through 24 byte rows of a by-value argument's field, read as
; floats (flame-hip), must step in [6 x float]: the SPIR-V backend misorders
; a byte stride GEP (#2398).
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64-unknown-chipstar-shadervulkan"

%P = type { [2 x float], [20 x [6 x float]] }

define spir_kernel void @k(ptr byval(%P) %p, i64 %i, ptr addrspace(1) %o) {
  %f = getelementptr inbounds i8, ptr %p, i64 8
  %row = getelementptr inbounds [24 x i8], ptr %f, i64 %i
  %a = load float, ptr %row, align 4
  %q = getelementptr inbounds i8, ptr %row, i64 4
  %b = load float, ptr %q, align 4
  %s = fadd float %a, %b
  store float %s, ptr addrspace(1) %o, align 4
  ret void
}
