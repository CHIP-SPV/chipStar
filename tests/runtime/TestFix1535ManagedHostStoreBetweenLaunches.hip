// Reproduces: on OpenCL devices that only support coarse grained SVM (Mali G52,
// rusticl) hipMallocManaged memory is undefined for the host outside of a
// clEnqueueSVMMap/Unmap pair, so a plain host store made between two kernel
// launches never reaches the device and a host read after synchronisation may
// return stale cache contents. https://github.com/CHIP-SPV/chipStar/issues/1535
//
// HIP allows the host to dereference a managed pointer whenever the device is
// idle, even when concurrentManagedAccess == 0, so the runtime has to keep the
// allocation mapped for the host between the synchronisation point and the next
// device side command. The host touches the pointer after hipStreamSynchronize,
// hipDeviceSynchronize and a blocking hipMemcpy, the three ways this test
// returns control to the host.

#include <hip/hip_runtime.h>
#include <cstdio>

#define CHECK(Expr)                                                            \
  do {                                                                         \
    hipError_t Err = (Expr);                                                   \
    if (Err != hipSuccess) {                                                   \
      printf("FAIL: %s returned %s\n", #Expr, hipGetErrorString(Err));         \
      return 1;                                                                \
    }                                                                          \
  } while (0)

__global__ void observe(const int *Managed, int *Out, int Slot) {
  Out[Slot] = *Managed;
}

__global__ void bump(int *Managed) { *Managed += 100; }

int main() {
  hipDeviceProp_t Props;
  CHECK(hipGetDeviceProperties(&Props, 0));
  if (!Props.managedMemory) {
    printf("HIP_SKIP_THIS_TEST: managedMemory == 0\n");
    return 0;
  }

  int *Managed = nullptr;
  int *Out = nullptr;
  CHECK(hipMallocManaged(&Managed, sizeof(int)));
  CHECK(hipMalloc(&Out, 3 * sizeof(int)));
  CHECK(hipMemset(Out, 0xff, 3 * sizeof(int)));

  // Host initialisation right after the allocation, before any launch.
  *Managed = 1;
  observe<<<1, 1>>>(Managed, Out, 0);
  CHECK(hipStreamSynchronize(0));

  // Host store between two launches: the store lost on coarse grained SVM.
  *Managed = 2;
  observe<<<1, 1>>>(Managed, Out, 1);
  bump<<<1, 1>>>(Managed);
  CHECK(hipDeviceSynchronize());

  // Host read after a device synchronisation sees the device write.
  int HostSeen = *Managed;

  // Host store after a device synchronisation is visible to the next launch.
  *Managed = 3;
  observe<<<1, 1>>>(Managed, Out, 2);

  // A blocking copy is a synchronisation point too: the host may read the
  // managed pointer right after it.
  int Seen[3] = {-1, -1, -1};
  CHECK(hipMemcpy(Seen, Out, sizeof(Seen), hipMemcpyDeviceToHost));
  int HostSeenAfterCopy = *Managed;

  bool Pass = Seen[0] == 1 && Seen[1] == 2 && Seen[2] == 3 && HostSeen == 102 &&
              HostSeenAfterCopy == 3;
  printf("device saw %d %d %d (expected 1 2 3), host saw %d then %d (expected "
         "102 then 3): %s\n",
         Seen[0], Seen[1], Seen[2], HostSeen, HostSeenAfterCopy,
         Pass ? "PASS" : "FAIL");

  CHECK(hipFree(Out));
  CHECK(hipFree(Managed));
  return Pass ? 0 : 1;
}
