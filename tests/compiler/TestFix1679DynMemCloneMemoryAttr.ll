; Reproducer for https://github.com/CHIP-SPV/chipStar/issues/1679: h gains a
; shared memory argument it writes through, so memory(argmem: none) is wrong.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

@smem = external addrspace(3) global [0 x i32]

define spir_func void @h(i32 %v) #0 {
  store i32 %v, ptr addrspace(3) @smem
  ret void
}

define spir_kernel void @k(i32 %v) {
  call spir_func void @h(i32 %v)
  ret void
}

attributes #0 = { noinline memory(write, argmem: none) }
