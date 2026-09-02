; Reproducer for CHIP-SPV/chipStar#1454: device IR holding a vector of
; pointers. Reduced with llvm-reduce from a failing Aurora application by the
; issue reporter; rocPRIM's test/rocprim/test_device_reduce_by_key.cpp is a
; real-world trigger. Translating this needs SPV_INTEL_masked_gather_scatter,
; and chipStar's device flag list pins the allowed set with
; `--spirv-ext=-all,...`, so without the extension llvm-spirv rejects it.

target triple = "spirv64"

define spir_kernel void @k(ptr addrspace(1) %src, ptr addrspace(1) %dst) {
entry:
  %v = load <4 x ptr addrspace(4)>, ptr addrspace(1) %src, align 8
  store <4 x ptr addrspace(4)> %v, ptr addrspace(1) %dst, align 8
  ret void
}
