; Reproducer for https://github.com/CHIP-SPV/chipStar/issues/1409 and
; https://github.com/CHIP-SPV/chipStar/issues/1479: kernels that depend on the
; warp width without calling a listed warp primitive.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

@slots = internal addrspace(3) global [32 x i32] poison, align 4

declare spir_func void @_Z17sub_group_barrierj(i32)

; Barrier-free read of lane 0's shared slot through a generic pointer, as in
; a warp lock-step reduction.
define spir_kernel void @lockstep(ptr addrspace(1) %o) {
entry:
  %v = load i32, ptr addrspacecast (ptr addrspace(3) @slots to ptr), align 4
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
}

; Dynamic shared memory, as HipDynMem passes it.
define spir_kernel void @dynshared(ptr addrspace(1) %o, ptr addrspace(3) %s) {
entry:
  %v = load i32, ptr addrspace(3) %s, align 4
  store i32 %v, ptr addrspace(1) %o, align 4
  ret void
}

; The body __syncwarp() lowers to.
define spir_kernel void @syncwarp(ptr addrspace(1) %o) {
entry:
  store i32 1, ptr addrspace(1) %o, align 4
  call spir_func void @_Z17sub_group_barrierj(i32 2)
  ret void
}

; Lanes never exchange data, so no pin is needed.
define spir_kernel void @plain(ptr addrspace(1) %o) {
entry:
  store i32 1, ptr addrspace(1) %o, align 4
  ret void
}

; Must stay unpinned: see tests/runtime/TestIndirectCall.hip.
define spir_kernel void @indirect(ptr addrspace(1) %o, ptr %f) {
entry:
  call spir_func void @_Z17sub_group_barrierj(i32 2)
  call spir_func void %f(ptr addrspace(1) %o)
  ret void
}
