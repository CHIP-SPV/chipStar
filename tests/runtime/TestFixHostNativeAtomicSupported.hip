// Reproduces: hipDeviceGetAttribute(hipDeviceAttributeHostNativeAtomicSupported)
// throws hipErrorInvalidValue because chipstar::Device::getAttr() has no case
// for this attribute, even though hipDeviceProp_t::hostNativeAtomicSupported is
// populated. CUDA/ROCm expose this attribute; consumers (e.g. PyTorch, HecBench
// atomicSystemWide) call it to gate code paths.

#include <hip/hip_runtime.h>
#include <cstdio>

int main() {
  int value = -1;
  hipError_t err =
      hipDeviceGetAttribute(&value, hipDeviceAttributeHostNativeAtomicSupported, 0);
  int pass = (err == hipSuccess) && (value == 0 || value == 1);
  printf("hipDeviceAttributeHostNativeAtomicSupported: err=%d value=%d => %s\n",
         (int)err, value, pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
