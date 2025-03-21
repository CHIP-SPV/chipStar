; ModuleID = 'exact_error_reproducer.ll'
target datalayout = "e-p:64:64-p1:64:64-p2:32:32-p3:32:32-p4:64:64-p5:32:32-p6:32:32-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-v2048:2048-n32:64-S32-A5-G1-ni:7"
target triple = "spirv64-unknown-unknown"

declare spir_func i32 @_Z23intel_sub_group_shuffleij(i32, i32)

define spir_kernel void @exact_error_reproducer() {
entry:
  ; Create a value with non-standard bit width (directly from error message)
  ; %warp_prefix.i.sroa.12.sroa.14.0.extract.trunc = trunc i32 %warp_prefix.i.sroa.12.sroa.14.0.extract.shift to i24
  %shift_val = lshr i32 123456, 8
  %trunc24 = trunc i32 %shift_val to i24
  
  ; %warp_prefix.i.sroa.12.sroa.14.0.insert.ext = zext i24 %warp_prefix.i.sroa.12.sroa.14.0.extract.trunc to i32
  %ext32 = zext i24 %trunc24 to i32
  
  ; %warp_prefix.i.sroa.12.sroa.14.0.insert.shift = shl i32 %warp_prefix.i.sroa.12.sroa.14.0.insert.ext, 8
  %shift = shl i32 %ext32, 8
  
  ; Create more operations that match the error message pattern
  %mask = and i32 undef, 255
  %or_val = or i32 %mask, %shift
  %and_val = and i32 %or_val, -256
  
  ; Call the intel_sub_group_shuffle function
  %call_result = call spir_func i32 @_Z23intel_sub_group_shuffleij(i32 %and_val, i32 1)
  
  ; Convert to boolean
  %trunc8 = trunc i32 %call_result to i8
  %or8 = or i8 %trunc8, 1
  %cmp = icmp ne i8 %or8, 0
  
  ; This is the exact problematic instruction:
  ; %frombool.i.i = zext i1 %16 to i8    ============>   %17 = zext i1 %15 to i32
  %frombool = zext i1 %cmp to i8
  
  ; We'll also add a potential chain to i32 that might trigger the optimization
  %final = zext i8 %frombool to i32
  
  ; Use both values to prevent optimization
  call void @use_i8(i8 %frombool)
  call void @use_i32(i32 %final)
  
  ret void
}

declare void @use_i8(i8)
declare void @use_i32(i32) 