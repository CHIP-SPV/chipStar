source_filename = "minimal_rocprim.ll"
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024"
target triple = "spirv64"

define void @test_load_shift_trunc_chain(ptr addrspace(1) %out, ptr addrspace(3) %in) {
entry:
  %val_i48 = load i48, ptr addrspace(3) %in, align 2
  %shift_i48 = lshr i48 %val_i48, 16
  %trunc_i32 = trunc i48 %shift_i48 to i32
  %add_final = add i32 %trunc_i32, 5
  store i32 %add_final, ptr addrspace(1) %out, align 4
  ret void
}
