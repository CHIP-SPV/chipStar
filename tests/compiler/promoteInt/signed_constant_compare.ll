; ModuleID = 'signed_constant_compare.bc'
source_filename = "signed_constant_compare.cu"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv64-unknown-unknown"

; Test case to verify correct sign/zero extension during promotion of constants
; used in signed comparisons.

; Function Attrs: convergent norecurse nounwind
define spir_kernel void @signed_constant_compare_kernel(ptr addrspace(1) noundef %out) local_unnamed_addr #0 {
entry:
  %tid = tail call spir_func i64 @_Z12get_local_idj(i32 noundef 0) #1
  %tid_i32 = trunc i64 %tid to i32

  ; Create an i33 value. Subtracting makes some values negative.
  %val_i32 = sub i32 %tid_i32, 10
  %val_i33 = sext i32 %val_i32 to i33

  ; Perform a signed less than comparison.
  ; If the constant -1 is incorrectly zero-extended to i64 (becoming 0x1FFFFFFFF),
  ; this comparison will yield incorrect results for many %val_i33 values.
  ; Correct promotion requires sign-extending the constant -1 to i64 -1 (0xFFFFFFFFFFFFFFFF).
  %cmp = icmp slt i33 %val_i33, -1

  ; Store the boolean result (0 or 1)
  %result = zext i1 %cmp to i32
  %arrayidx = getelementptr inbounds i32, ptr addrspace(1) %out, i64 %tid
  store i32 %result, ptr addrspace(1) %arrayidx, align 4
  ret void
}

; Function Attrs: convergent nounwind readnone
declare spir_func i64 @_Z12get_local_idj(i32 noundef) local_unnamed_addr #1

attributes #0 = { convergent norecurse nounwind "uniform-work-group-size"="true" }
attributes #1 = { convergent nounwind readnone }

!llvm.module.flags = !{!0}
!opencl.ocl.version = !{!1}
!llvm.ident = !{!2}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 2, i32 0}
!2 = !{!"Test Compiler"} 