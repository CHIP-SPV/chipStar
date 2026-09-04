; Reproducer for CHIP-SPV/chipStar#1577.
;
; LLVM 23's SROA folds a struct of pointers into a vector of pointers. This is
; the shape it produces for rocPRIM's scan_state in
; test/rocprim/test_device_reduce_by_key.cpp, reduced to the load/store pair:
; the value is only copied, never dereferenced as a vector.
;
; Neither SPIR-V path accepts <N x ptr>. The in-tree backend asserts in
; SPIRVEmitIntrinsics, and llvm-spirv demands SPV_INTEL_masked_gather_scatter,
; which IGC 2.38.2 then refuses. hip-lower-pointer-vectors must rewrite these
; so no vector-of-pointer type survives.

target triple = "spirv64"

define spir_kernel void @copy_ptr_vec(ptr %src, ptr %dst) {
entry:
  %v = load <3 x ptr addrspace(4)>, ptr %src, align 8
  store <3 x ptr addrspace(4)> %v, ptr %dst, align 32
  ret void
}
