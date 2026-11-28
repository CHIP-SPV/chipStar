; A thread that returns before a workgroup barrier must keep reaching it.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64-unknown-chipstar-shadervulkan"

declare i32 @llvm.spv.flattened.thread.id.in.group()
declare void @llvm.spv.group.memory.barrier.with.group.sync()

; syncthreadsExitedThreads: if (tid > 10) return; store; barrier; store.
declare spir_func i64 @_Z12get_local_idj(i32)

define spir_kernel void @early(ptr addrspace(1) %o) {
  %t = call spir_func i64 @_Z12get_local_idj(i32 0)
  %tid = trunc i64 %t to i32
  %c = icmp ugt i32 %tid, 10
  br i1 %c, label %ret0, label %body
ret0:
  br label %ret
body:
  store i32 1, ptr addrspace(1) %o, align 4
  call void @llvm.spv.group.memory.barrier.with.group.sync()
  store i32 2, ptr addrspace(1) %o, align 4
  br label %ret
ret:
  ret void
}

; A return inside a loop whose body has a barrier.
define spir_kernel void @loop(ptr addrspace(1) %o, i32 %n) {
entry:
  %tid = call i32 @llvm.spv.flattened.thread.id.in.group()
  br label %head
head:
  %i = phi i32 [ 0, %entry ], [ %i1, %latch ]
  %c = icmp eq i32 %tid, %i
  br i1 %c, label %ret, label %body
body:
  store i32 %i, ptr addrspace(1) %o, align 4
  call void @llvm.spv.group.memory.barrier.with.group.sync()
  br label %latch
latch:
  %i1 = add i32 %i, 1
  %d = icmp ult i32 %i1, %n
  br i1 %d, label %head, label %ret
ret:
  ret void
}

; The loop condition reads memory a guarded thread would not read.
@s = internal addrspace(3) global [4 x i32] undef, align 4
define spir_kernel void @bail(ptr addrspace(1) %o) {
entry:
  %tid = call i32 @llvm.spv.flattened.thread.id.in.group()
  %c = icmp ugt i32 %tid, 10
  br i1 %c, label %ret, label %head
head:
  %v = load i32, ptr addrspace(3) @s, align 4
  %d = icmp sgt i32 %v, 0
  br i1 %d, label %body, label %ret
body:
  store i32 %v, ptr addrspace(1) %o, align 4
  call void @llvm.spv.group.memory.barrier.with.group.sync()
  br label %head
ret:
  ret void
}

; No barrier after the return.
define spir_kernel void @nobarrier(ptr addrspace(1) %o) {
  %tid = call i32 @llvm.spv.flattened.thread.id.in.group()
  %c = icmp ugt i32 %tid, 10
  br i1 %c, label %ret, label %body
body:
  store i32 1, ptr addrspace(1) %o, align 4
  br label %ret
ret:
  ret void
}

; A loop bound loaded before the early return runs in every thread.
define spir_kernel void @bound(ptr addrspace(1) %o, ptr addrspace(1) %n) {
entry:
  %tid = call i32 @llvm.spv.flattened.thread.id.in.group()
  %nv = load i32, ptr addrspace(1) %n, align 4
  %c = icmp ugt i32 %tid, 10
  br i1 %c, label %ret, label %head
head:
  %i = phi i32 [ 0, %entry ], [ %i1, %body ]
  %d = icmp slt i32 %i, %nv
  br i1 %d, label %body, label %ret
body:
  store i32 %i, ptr addrspace(1) %o, align 4
  call void @llvm.spv.group.memory.barrier.with.group.sync()
  %i1 = add i32 %i, 1
  br label %head
ret:
  ret void
}
