; Reproducer for https://github.com/CHIP-SPV/chipStar/issues/1640: a lowered
; device global's address nested in a constant aggregate.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

%struct.Agg = type { ptr addrspace(4), [4 x i32] }

@Scalar = hidden addrspace(1) externally_initialized global i32 42, align 4
@S = hidden addrspace(1) externally_initialized global %struct.Agg { ptr addrspace(4) addrspacecast (ptr addrspace(1) @Scalar to ptr addrspace(4)), [4 x i32] [i32 1, i32 2, i32 3, i32 4] }, align 8

define hidden spir_kernel void @k(ptr addrspace(1) %Out) {
entry:
  store <2 x i64> <i64 ptrtoint (ptr addrspace(1) @Scalar to i64), i64 ptrtoint (ptr addrspace(1) @S to i64)>, ptr addrspace(1) %Out, align 8
  ret void
}
