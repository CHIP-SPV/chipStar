// Reproduces: a device module whose comdat device variables (a function-local
// static __device__ variable in an inline __host__ __device__ function, a C++17
// inline __device__ variable, a __device__ static data member of a class
// template) are not registered by the host faults at module initialisation
// when a second module carries the same variables.
//
// HipGlobalVariables.cpp lowers every externally_initialized global into an
// address slot __chip_var_<X> plus __chip_var_info_<X> / __chip_var_bind_<X>
// shadow kernels, and __chip_var_init_all writes each variable's initializer
// through its slot. clang emits no __hipRegisterVar for these variables, so
// the runtime discovers them from the info kernels instead; that discovery
// de-duplicates by name across all modules of the process, so only the first
// module gets storage bound. In every later module the slot stays 0 and
// __chip_var_init_all writes the initializer to address 0: on Aurora PVC every
// Trilinos STK NGP test executable dies with "Segmentation fault from GPU at
// 0x0 ... access: 1 (Write)" (stk_mesh_utest
// LateFieldsTestFixture.get_ngp_field_multistate_no_seg_fault and all of
// stk_ngp_test_unit_tests).
//
// Two translation units (this file and inputs/FunctionLocalStaticDeviceVarTU2
// .hip) carry the same three variables. Each module must observe the initial
// values, a kernel must see a value written by an earlier kernel from the same
// module, every __chip_var_info_<X> kernel of each module must have storage
// bound for X, and a plain registered __device__ variable must still round
// trip through hipMemcpyToSymbol / hipMemcpyFromSymbol / hipGetSymbolAddress.

#include "inputs/FunctionLocalStaticDeviceVar.hh"
#include "CHIPBackend.hh"

#include <cstdio>
#include <cstring>

#define CHECK(Expr)                                                            \
  do {                                                                         \
    hipError_t E = (Expr);                                                     \
    if (E != hipSuccess) {                                                     \
      printf("FAIL: %s -> %s (%d) at line %d\n", #Expr, hipGetErrorString(E), \
             (int)E, __LINE__);                                                \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// (4) A plain namespace-scope __device__ variable, which clang does register.
__device__ int PlainVar = 5;

__global__ void readShapesTU1(int *Out) { readShapes(Out); }
__global__ void writeShapesTU1(int V) { writeShapes(V); }
__global__ void readPlain(int *Out) { *Out = PlainVar; }

void launchReadShapesTU2(int *Out);
void launchWriteShapesTU2(int V);
const void *readShapesTU2HostPtr();

static int Failures = 0;

static void expectShapes(const char *What, const int *Got, int Want) {
  printf("%s: local static %d, inline var %d, template member %d, expected "
         "%d\n",
         What, Got[0], Got[1], Got[2], Want);
  for (int I = 0; I < 3; I++)
    if (Got[I] != Want)
      Failures++;
}

// Every __chip_var_info_<X> shadow kernel in a module means the module's code
// reads X through a __chip_var_<X> address slot that only the runtime can
// fill. Count the info kernels whose variable has no storage bound.
static int countUnboundVariables(const char *Label, const void *KernelPtr) {
  auto *Kernel = Backend->getActiveDevice()->findKernel(HostPtr(KernelPtr));
  if (!Kernel) {
    printf("FAIL: %s: kernel not found\n", Label);
    return 1;
  }
  auto *Mod = Kernel->getModule();
  int Unbound = 0;
  for (auto *K : Mod->getKernels()) {
    const std::string &Name = K->getName();
    if (Name.rfind(ChipVarInfoPrefix, 0) != 0)
      continue;
    std::string VarName = Name.substr(strlen(ChipVarInfoPrefix));
    auto *Var = Mod->getGlobalVar(VarName.c_str());
    bool Bound = Var && Var->getDevAddr();
    printf("%s: %s -> %s\n", Label, VarName.c_str(),
           Bound ? "bound" : "NOT BOUND");
    if (!Bound)
      Unbound++;
  }
  return Unbound;
}

int main() {
  int *Out = nullptr;
  CHECK(hipMalloc(&Out, 3 * sizeof(int)));
  int Got[3] = {0, 0, 0};

  // Module 1: initial values, then a write seen by a later kernel.
  readShapesTU1<<<1, 1>>>(Out);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(Got, Out, sizeof(Got), hipMemcpyDeviceToHost));
  expectShapes("TU1 initial", Got, 7);

  writeShapesTU1<<<1, 1>>>(11);
  CHECK(hipGetLastError());
  readShapesTU1<<<1, 1>>>(Out);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(Got, Out, sizeof(Got), hipMemcpyDeviceToHost));
  expectShapes("TU1 after write", Got, 11);

  // Module 2: its own copies, starting from their initializers.
  launchReadShapesTU2(Out);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(Got, Out, sizeof(Got), hipMemcpyDeviceToHost));
  expectShapes("TU2 initial", Got, 7);

  launchWriteShapesTU2(13);
  CHECK(hipGetLastError());
  launchReadShapesTU2(Out);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(Got, Out, sizeof(Got), hipMemcpyDeviceToHost));
  expectShapes("TU2 after write", Got, 13);

  // Module 1 is unaffected by module 2's writes.
  readShapesTU1<<<1, 1>>>(Out);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(Got, Out, sizeof(Got), hipMemcpyDeviceToHost));
  expectShapes("TU1 after TU2 write", Got, 11);

  int Unbound =
      countUnboundVariables("TU1 module",
                            reinterpret_cast<const void *>(readShapesTU1)) +
      countUnboundVariables("TU2 module", readShapesTU2HostPtr());
  if (Unbound) {
    printf("FAIL: %d device variables have no storage bound\n", Unbound);
    Failures++;
  }

  // (4) A registered variable keeps working from the host.
  int V = 9;
  CHECK(hipMemcpyToSymbol(HIP_SYMBOL(PlainVar), &V, sizeof(int)));
  readPlain<<<1, 1>>>(Out);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  CHECK(hipMemcpy(Got, Out, sizeof(int), hipMemcpyDeviceToHost));
  int Back = 0;
  CHECK(hipMemcpyFromSymbol(&Back, HIP_SYMBOL(PlainVar), sizeof(int)));
  void *Addr = nullptr;
  CHECK(hipGetSymbolAddress(&Addr, HIP_SYMBOL(PlainVar)));
  printf("plain var: kernel read %d, hipMemcpyFromSymbol %d, expected 9, "
         "hipGetSymbolAddress %p\n",
         Got[0], Back, Addr);
  if (Got[0] != 9 || Back != 9 || !Addr)
    Failures++;

  CHECK(hipFree(Out));

  if (Failures) {
    printf("FAIL: %d mismatches\n", Failures);
    return 1;
  }
  printf("PASS\n");
  return 0;
}
