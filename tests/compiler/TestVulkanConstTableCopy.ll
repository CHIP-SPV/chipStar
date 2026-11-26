; A local array initialized from a constant table larger than 256 bytes
; (spm-hip's 388 byte ran[]) must become constant stores, not a byte loop
; reading the float table as bytes.
target datalayout = "e-i64:64-v16:16-v24:32-v32:32-v48:64-v96:128-v192:256-v256:256-v512:512-v1024:1024-n8:16:32:64-G1"
target triple = "spirv64-unknown-chipstar-shadervulkan"

@ran = private unnamed_addr addrspace(1) constant [97 x float] [float 1.0, float 2.0, float 3.0, float 4.0, float 5.0, float 6.0, float 7.0, float 8.0, float 9.0, float 10.0, float 11.0, float 12.0, float 13.0, float 14.0, float 15.0, float 16.0, float 17.0, float 18.0, float 19.0, float 20.0, float 21.0, float 22.0, float 23.0, float 24.0, float 25.0, float 26.0, float 27.0, float 28.0, float 29.0, float 30.0, float 31.0, float 32.0, float 33.0, float 34.0, float 35.0, float 36.0, float 37.0, float 38.0, float 39.0, float 40.0, float 41.0, float 42.0, float 43.0, float 44.0, float 45.0, float 46.0, float 47.0, float 48.0, float 49.0, float 50.0, float 51.0, float 52.0, float 53.0, float 54.0, float 55.0, float 56.0, float 57.0, float 58.0, float 59.0, float 60.0, float 61.0, float 62.0, float 63.0, float 64.0, float 65.0, float 66.0, float 67.0, float 68.0, float 69.0, float 70.0, float 71.0, float 72.0, float 73.0, float 74.0, float 75.0, float 76.0, float 77.0, float 78.0, float 79.0, float 80.0, float 81.0, float 82.0, float 83.0, float 84.0, float 85.0, float 86.0, float 87.0, float 88.0, float 89.0, float 90.0, float 91.0, float 92.0, float 93.0, float 94.0, float 95.0, float 96.0, float 97.0], align 4

define spir_kernel void @k(ptr addrspace(1) %o, i32 %i) {
  %a = alloca [97 x float], align 4
  call void @llvm.memcpy.p0.p1.i64(ptr align 4 %a, ptr addrspace(1) align 4 @ran, i64 388, i1 false)
  %j = urem i32 %i, 97
  %p = getelementptr [97 x float], ptr %a, i32 0, i32 %j
  %v = load float, ptr %p, align 4
  store float %v, ptr addrspace(1) %o, align 4
  ret void
}

declare void @llvm.memcpy.p0.p1.i64(ptr, ptr addrspace(1), i64, i1)
