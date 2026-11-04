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

  // An exact fit must write every byte; rows past the allocation are rejected.
  for (int Oob = 0; Oob < 2; ++Oob) {
    CHECK(hipMemset(P, 0, 32));
    hipMemsetParams M = {};
    M.dst = P;
    M.elementSize = 1;
    M.height = 2;
    M.width = M.pitch = Oob ? 32 : 16;
    M.value = Oob ? 0xEF : 0xCD;
    hipGraph_t G;
    hipGraphNode_t N;
    hipGraphExec_t X;
    CHECK(hipGraphCreate(&G, 0));
    CHECK(hipGraphAddMemsetNode(&N, G, nullptr, 0, &M));
    CHECK(hipGraphInstantiate(&X, G, nullptr, nullptr, 0));
    hipError_t E = hipGraphLaunch(X, S);
    if (E != (Oob ? hipErrorInvalidValue : hipSuccess)) {
      std::cout << "FAIL: oob=" << Oob << " launch -> " << hipGetErrorString(E)
                << "\n";
      return 1;
    }
    CHECK(hipStreamSynchronize(S));
    CHECK(hipMemcpy(H, P, 32, hipMemcpyDeviceToHost));
    for (int I = 0; I < 32; ++I)
      if (H[I] != (Oob ? 0 : 0xCD)) {
        std::cout << "FAIL: oob=" << Oob << " offset " << I << " is "
                  << (int)H[I] << "\n";
        return 1;
      }
  }
  std::cout << "PASS\n";
  return 0;
}
