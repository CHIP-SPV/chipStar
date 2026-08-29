// Reproduces: on Intel Data Center GPU Max (PVC) under Level Zero a kernel
// whose only path to a hipMallocManaged buffer is a pointer stored in other
// memory (a pointer member of a by-value struct argument, or an entry of a
// device array of pointers) updates a stale device copy and the host never
// sees the write. Since e7d1f6e52f5b managed memory on PVC is shared USM
// (zeMemAllocShared), and NEO only migrates a shared allocation for a launch
// when the kernel carries ZE_KERNEL_INDIRECT_ACCESS_FLAG_SHARED
// (KernelImp::setIndirectAccess -> indirectSharedAllocationsAllowed ->
// CommandList::migrateSharedAllocations). CHIPQueueLevel0::launchImpl sets
// only the DEVICE and HOST flags. The OpenCL backend sets all three
// CL_KERNEL_EXEC_INFO_INDIRECT_*_ACCESS_INTEL flags, so it is unaffected.
//
// Kokkos hip.mixed_then_host_device_nodes (4 instead of 6) and
// hip.interact_with_hip_node (0 instead of 1) hit exactly this: a Kokkos
// View (a pointer inside the by-value functor) over HIPManagedSpace.
//
// The kernels below follow the patterns HipIGBADetector.cpp flags as indirect
// global buffer accesses (a load of a pointer from memory), so the module is
// marked as having IGBAs and the runtime takes the zeKernelSetIndirectAccess
// path on every launch. The test prints that marking, and fails if the
// detector did not flag the module, so a passing run is known to have
// exercised the indirect access path.
//
// Every path is checked through a hipMemcpy readback, which the issue shows
// stale as well. A control launch with the managed pointer as a direct kernel
// argument decides whether host reads of managed memory are coherent after
// hipDeviceSynchronize on this device; they are not on coarse-grained SVM
// (rusticl), where managed memory is only reachable through copies, so the
// host read checks apply only where the control shows them meaningful.

#include <hip/hip_runtime.h>
#include "CHIPBackend.hh"

#include <cstdio>

#define CHECK(Expr)                                                            \
  do {                                                                         \
    hipError_t E = (Expr);                                                     \
    if (E != hipSuccess) {                                                     \
      printf("FAIL: %s -> %s (%d) at line %d\n", #Expr, hipGetErrorString(E), \
             (int)E, __LINE__);                                                \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// Control: the managed pointer is a direct kernel argument, so the runtime
// hands it to the driver explicitly and no indirect access declaration is
// involved.
__global__ void incrementDirect(unsigned *P) { ++*P; }

// The managed pointer is only reachable through the struct.
struct Holder {
  unsigned *P;
};

__global__ void incrementViaStruct(Holder H) { ++*H.P; }

// The managed pointer is only reachable through a device array of pointers.
__global__ void incrementViaTable(unsigned **Table) { ++*Table[0]; }

static int Failures = 0;

static void expectU(const char *What, unsigned Got, unsigned Want) {
  if (Got != Want) {
    printf("  mismatch: %s = %u, expected %u\n", What, Got, Want);
    Failures++;
  }
}

int main() {
  hipDeviceProp_t Props;
  CHECK(hipGetDeviceProperties(&Props, 0));
  printf("device: %s managedMemory=%d\n", Props.name, Props.managedMemory);
  if (!Props.managedMemory) {
    printf("HIP_SKIP_THIS_TEST: device has no managed memory\n");
    return CHIP_SKIP_TEST;
  }

  // What the runtime concluded from __chip_module_has_no_IGBAs. Level Zero
  // calls zeKernelSetIndirectAccess only when HasNoIGBAs is false.
  auto *ChipKernel = Backend->getActiveDevice()->findKernel(
      HostPtr(reinterpret_cast<const void *>(incrementViaStruct)));
  const bool HasNoIGBAs = ChipKernel->getModule()->getInfo().HasNoIGBAs;
  printf("module HasNoIGBAs=%d: indirect access flags %s at launch\n",
         (int)HasNoIGBAs, HasNoIGBAs ? "NOT set" : "set");
  if (HasNoIGBAs) {
    printf("FAIL: the IGBA detector did not flag this module, the test would "
           "not exercise zeKernelSetIndirectAccess\n");
    return 1;
  }

  unsigned *Counter = nullptr;
  CHECK(hipMallocManaged(&Counter, sizeof(unsigned)));
  unsigned Copy = 0;
  const unsigned Zero = 0;

  // Control launch: the readback must be 1 on any device that reports
  // managed memory. The host read tells whether managed memory is host
  // coherent after synchronization here; the host read checks below are
  // applied only when it is.
  CHECK(hipMemcpy(Counter, &Zero, sizeof(unsigned), hipMemcpyHostToDevice));
  incrementDirect<<<1, 1>>>(Counter);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(&Copy, Counter, sizeof(unsigned), hipMemcpyDeviceToHost));
  const bool HostCoherent = *Counter == 1;
  printf("direct: host read %u, hipMemcpy readback %u, expected 1\n", *Counter,
         Copy);
  printf("managed memory host reads after sync: %s\n",
         HostCoherent ? "coherent, host reads checked"
                      : "NOT coherent on this device (coarse-grained SVM), "
                        "host reads not checked");
  fflush(stdout);
  expectU("direct: hipMemcpy readback", Copy, 1);

  // A host write is the Kokkos pattern and is kept where the host is
  // coherent; elsewhere only a copy reaches the device.
  auto resetCounter = [&]() -> hipError_t {
    if (HostCoherent) {
      *Counter = 0;
      return hipSuccess;
    }
    return hipMemcpy(Counter, &Zero, sizeof(unsigned), hipMemcpyHostToDevice);
  };

  // Pointer inside a by-value struct argument (a Kokkos View in a functor).
  CHECK(resetCounter());
  Holder H{Counter};
  incrementViaStruct<<<1, 1>>>(H);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(&Copy, Counter, sizeof(unsigned), hipMemcpyDeviceToHost));
  printf("struct: host read %u, hipMemcpy readback %u, expected 1\n", *Counter,
         Copy);
  fflush(stdout);
  if (HostCoherent)
    expectU("struct: host read", *Counter, 1);
  expectU("struct: hipMemcpy readback", Copy, 1);

  // Pointer stored in a hipMalloc'd device array of pointers.
  unsigned **Table = nullptr;
  CHECK(hipMalloc(&Table, sizeof(unsigned *)));
  CHECK(hipMemcpy(Table, &Counter, sizeof(unsigned *), hipMemcpyHostToDevice));
  CHECK(resetCounter());
  incrementViaTable<<<1, 1>>>(Table);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(&Copy, Counter, sizeof(unsigned), hipMemcpyDeviceToHost));
  printf("table: host read %u, hipMemcpy readback %u, expected 1\n", *Counter,
         Copy);
  fflush(stdout);
  if (HostCoherent)
    expectU("table: host read", *Counter, 1);
  expectU("table: hipMemcpy readback", Copy, 1);

  CHECK(hipFree(Table));
  CHECK(hipFree(Counter));

  if (Failures) {
    printf("FAIL: %d mismatches\n", Failures);
    return 1;
  }
  printf("PASS\n");
  return 0;
}
