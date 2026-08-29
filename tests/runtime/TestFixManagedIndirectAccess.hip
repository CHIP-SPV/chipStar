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
// stale as well. The host touches managed memory directly (the Kokkos
// pattern: a host write before the launch, a host read after the sync) only
// where the device reports directManagedMemAccessFromHost or
// concurrentManagedAccess. On coarse-grained SVM (Mali, rusticl) host access
// outside clEnqueueSVMMap/Unmap is undefined: an unmapped host write is not
// seen by the next kernel and an unmapped host read returns whatever the CPU
// cache holds. Probing that empirically (a host read after a control launch)
// is not a decision, it is a coin toss: on Mali-G52 the probe read 1, the
// struct and table increments were then applied to a stale value and both
// the host read and the hipMemcpy readback showed 0, in one CI lane and not
// the other. A pure OpenCL run of the same sequence loses the write on every
// repetition with an unmapped host write and keeps it on every repetition
// with a clEnqueueSVMMemcpy reset. Where host access is not reported, the
// counter is reset and read only through hipMemcpy.

#include <hip/hip_runtime.h>
#include "CHIPBackend.hh"

#include <cstdio>
#include <string>

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

  // Whether the host may touch managed memory directly between launches.
  // Where neither attribute is set (coarse-grained SVM under OpenCL) the
  // counter is reset and read through hipMemcpy only.
  const bool HostAccess =
      Props.directManagedMemAccessFromHost || Props.concurrentManagedAccess;
  printf("directManagedMemAccessFromHost=%d concurrentManagedAccess=%d: host "
         "%s\n",
         Props.directManagedMemAccessFromHost, Props.concurrentManagedAccess,
         HostAccess ? "writes and reads managed memory directly"
                    : "reaches managed memory through hipMemcpy only");

  unsigned *Counter = nullptr;
  CHECK(hipMallocManaged(&Counter, sizeof(unsigned)));
  unsigned Copy = 0;
  const unsigned Zero = 0;

  // A host write is the Kokkos pattern and is kept where the device reports
  // host access; elsewhere only a copy reaches the device.
  auto resetCounter = [&]() -> hipError_t {
    if (HostAccess) {
      *Counter = 0;
      return hipSuccess;
    }
    return hipMemcpy(Counter, &Zero, sizeof(unsigned), hipMemcpyHostToDevice);
  };
  // The host read for the log; "skipped" where the host does not access
  // managed memory directly.
  auto hostRead = [&]() -> std::string {
    return HostAccess ? std::to_string(*Counter) : std::string("skipped");
  };

  // Control launch: the managed pointer is a direct kernel argument, so this
  // must give 1 on any device that reports managed memory.
  CHECK(resetCounter());
  incrementDirect<<<1, 1>>>(Counter);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(&Copy, Counter, sizeof(unsigned), hipMemcpyDeviceToHost));
  printf("direct: host read %s, hipMemcpy readback %u, expected 1\n",
         hostRead().c_str(), Copy);
  fflush(stdout);
  if (HostAccess)
    expectU("direct: host read", *Counter, 1);
  expectU("direct: hipMemcpy readback", Copy, 1);

  // Pointer inside a by-value struct argument (a Kokkos View in a functor).
  CHECK(resetCounter());
  Holder H{Counter};
  incrementViaStruct<<<1, 1>>>(H);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(&Copy, Counter, sizeof(unsigned), hipMemcpyDeviceToHost));
  printf("struct: host read %s, hipMemcpy readback %u, expected 1\n",
         hostRead().c_str(), Copy);
  fflush(stdout);
  if (HostAccess)
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
  printf("table: host read %s, hipMemcpy readback %u, expected 1\n",
         hostRead().c_str(), Copy);
  fflush(stdout);
  if (HostAccess)
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
