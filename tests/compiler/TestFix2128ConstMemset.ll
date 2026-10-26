; Reproducer for https://github.com/CHIP-SPV/chipStar/issues/2128: small constant memsets
; (constant fill byte, length 1 to 256) must not be expanded; others still are.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

define spir_kernel void @k(ptr addrspace(1) %p, i64 %n) {
  call void @llvm.memset.p1.i64(ptr addrspace(1) %p, i8 90, i64 64, i1 false)
  %q = getelementptr i8, ptr addrspace(1) %p, i64 64
  call void @llvm.memset.p1.i64(ptr addrspace(1) %q, i8 90, i64 %n, i1 false)
  %z = getelementptr i8, ptr addrspace(1) %p, i64 128
  call void @llvm.memset.p1.i64(ptr addrspace(1) %z, i8 90, i64 0, i1 false)
  %b = getelementptr i8, ptr addrspace(1) %p, i64 192
  call void @llvm.memset.p1.i64(ptr addrspace(1) %b, i8 90, i64 257, i1 false)
  %x = getelementptr i8, ptr addrspace(1) %p, i64 512
  call void @llvm.memset.inline.p1.i64(ptr addrspace(1) %x, i8 90, i64 64, i1 false)
  ret void
}

declare void @llvm.memset.p1.i64(ptr addrspace(1), i8, i64, i1)
declare void @llvm.memset.inline.p1.i64(ptr addrspace(1), i8, i64, i1)
