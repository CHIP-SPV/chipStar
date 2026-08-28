; Every volatile access shape HipLowerVolatileAccessesPass has a rule for. The
; 32 and 64 bit integer, float and pointer accesses through global and generic
; pointers must become relaxed device-scope atomics; everything else must come
; out exactly as it went in.

target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-G1"
target triple = "spirv64"

@lmem = internal addrspace(3) global [64 x i32] undef, align 4
@cmem = internal addrspace(2) constant [4 x i32] zeroinitializer, align 4

define spir_kernel void @rewritten(ptr addrspace(1) %P32, ptr addrspace(1) %P64,
                                   ptr addrspace(1) %PF, ptr addrspace(1) %PD,
                                   ptr addrspace(1) %PP,
                                   i32 %V32, i64 %V64, float %VF, double %VD,
                                   ptr addrspace(1) %VP) {
entry:
  %G32 = addrspacecast ptr addrspace(1) %P32 to ptr addrspace(4)
  %GF = addrspacecast ptr addrspace(1) %PF to ptr addrspace(4)
  ; integers, global and generic
  %ld32 = load volatile i32, ptr addrspace(1) %P32, align 4
  %ld64 = load volatile i64, ptr addrspace(1) %P64, align 8
  %ld32g = load volatile i32, ptr addrspace(4) %G32, align 4
  store volatile i32 %V32, ptr addrspace(1) %P32, align 4
  store volatile i64 %V64, ptr addrspace(1) %P64, align 8
  store volatile i32 %V32, ptr addrspace(4) %G32, align 4
  ; floats go through the same-width integer
  %ldf = load volatile float, ptr addrspace(4) %GF, align 4
  %ldd = load volatile double, ptr addrspace(1) %PD, align 8
  store volatile float %VF, ptr addrspace(4) %GF, align 4
  store volatile double %VD, ptr addrspace(1) %PD, align 8
  ; pointers go through i64
  %ldp = load volatile ptr addrspace(1), ptr addrspace(1) %PP, align 8
  store volatile ptr addrspace(1) %VP, ptr addrspace(1) %PP, align 8
  ; keep every loaded value alive
  %f2i = bitcast float %ldf to i32
  %d2i = bitcast double %ldd to i64
  %p2i = ptrtoint ptr addrspace(1) %ldp to i64
  %s1 = add i32 %ld32, %ld32g
  %s2 = add i32 %s1, %f2i
  %s3 = add i64 %ld64, %d2i
  %s4 = add i64 %s3, %p2i
  store i32 %s2, ptr addrspace(1) %P32, align 4
  store i64 %s4, ptr addrspace(1) %P64, align 8
  ret void
}

define spir_kernel void @left_alone(ptr addrspace(1) %P8, ptr addrspace(1) %P16,
                                    ptr addrspace(1) %P32, ptr addrspace(1) %PV,
                                    i8 %V8, i16 %V16, i32 %V32, <2 x i32> %VV) {
entry:
  ; 8 and 16 bit values
  %ld8 = load volatile i8, ptr addrspace(1) %P8, align 1
  %ld16 = load volatile i16, ptr addrspace(1) %P16, align 2
  store volatile i8 %V8, ptr addrspace(1) %P8, align 1
  store volatile i16 %V16, ptr addrspace(1) %P16, align 2
  ; vectors
  %ldv = load volatile <2 x i32>, ptr addrspace(1) %PV, align 8
  store volatile <2 x i32> %VV, ptr addrspace(1) %PV, align 8
  ; work-group local memory reached through a generic pointer
  %L = getelementptr inbounds [64 x i32], ptr addrspace(3) @lmem, i64 0, i64 5
  %LG = addrspacecast ptr addrspace(3) %L to ptr addrspace(4)
  %ldl = load volatile i32, ptr addrspace(4) %LG, align 4
  store volatile i32 %V32, ptr addrspace(4) %LG, align 4
  ; private memory reached through a generic pointer
  %A = alloca i32, align 4
  %AG = addrspacecast ptr %A to ptr addrspace(4)
  %lda = load volatile i32, ptr addrspace(4) %AG, align 4
  store volatile i32 %V32, ptr addrspace(4) %AG, align 4
  ; constant memory
  %C = getelementptr inbounds [4 x i32], ptr addrspace(2) @cmem, i64 0, i64 1
  %ldc = load volatile i32, ptr addrspace(2) %C, align 4
  ; under-aligned
  %ldu = load volatile i32, ptr addrspace(1) %P32, align 2
  store volatile i32 %V32, ptr addrspace(1) %P32, align 2
  ; already atomic
  %lda2 = load atomic volatile i32, ptr addrspace(1) %P32 syncscope("workgroup") acquire, align 4
  store atomic volatile i32 %V32, ptr addrspace(1) %P32 seq_cst, align 4
  ; keep every loaded value alive
  %e8 = zext i8 %ld8 to i32
  %e16 = zext i16 %ld16 to i32
  %v0 = extractelement <2 x i32> %ldv, i32 0
  %s1 = add i32 %e8, %e16
  %s2 = add i32 %s1, %v0
  %s3 = add i32 %s2, %ldl
  %s4 = add i32 %s3, %lda
  %s5 = add i32 %s4, %ldc
  %s6 = add i32 %s5, %ldu
  %s7 = add i32 %s6, %lda2
  store i32 %s7, ptr addrspace(1) %P32, align 4
  ret void
}
