// Reproduces #2299: a captured hipMemset2DAsync ignores its pitch on graph
// launch and fills Height * Width contiguous bytes instead of 2 rows of 7.
#include <hip/hip_runtime.h>
#include <iostream>

#define CHECK(Call)                                                            \
  do {                                                                         \
    hipError_t Err = (Call);                                                   \
    if (Err != hipSuccess) {                                                   \
      std::cout << "FAIL: " #Call " -> " << hipGetErrorString(Err) << "\n";    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  hipStream_t S;
  CHECK(hipStreamCreate(&S));
  unsigned char *P;
  CHECK(hipMalloc(&P, 32));
  CHECK(hipMemset(P, 0, 32));
  CHECK(hipDeviceSynchronize());

  hipGraph_t Graph;
  hipGraphExec_t Exec;
  CHECK(hipStreamBeginCapture(S, hipStreamCaptureModeGlobal));
  CHECK(hipMemset2DAsync(P, 16, 0xAB, 7, 2, S));
  CHECK(hipStreamEndCapture(S, &Graph));
  CHECK(hipGraphInstantiate(&Exec, Graph, nullptr, nullptr, 0));
  CHECK(hipGraphLaunch(Exec, S));
  CHECK(hipStreamSynchronize(S));

  unsigned char H[32];
  CHECK(hipMemcpy(H, P, 32, hipMemcpyDeviceToHost));
  int Bad = 0;
  for (int I = 0; I < 32; ++I) {
    unsigned char Want = (I % 16) < 7 ? 0xAB : 0;
    if (H[I] != Want) {
      std::cout << "FAIL: offset " << I << " is " << (int)H[I] << ", want "
                << (int)Want << "\n";
      Bad = 1;
    }
  }
  if (Bad)
    return 1;
  std::cout << "PASS\n";
  return 0;
}
