; What __ballot, __any and __all lower to: __chip_ballot, linked at runtime.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64"

declare spir_func i32 @_Z13__chip_balloti(i32)

define spir_kernel void @ballot(ptr addrspace(1) %o) {
entry:
  %b = call spir_func i32 @_Z13__chip_balloti(i32 1)
  store i32 %b, ptr addrspace(1) %o, align 4
  ret void
}
