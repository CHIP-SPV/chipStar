// Reproduces: a captured hipMemset3DAsync fills contiguous bytes instead of the
// strided region, https://github.com/CHIP-SPV/chipStar/issues/2304
#include <hip/hip_runtime_api.h>
#include <iostream>
#include <vector>

#define CHECK(Call)                                                            \
  do {                                                                         \
    hipError_t Err = (Call);                                                   \
    if (Err != hipSuccess) {                                                   \
      std::cout << "FAIL: " #Call " -> " << hipGetErrorString(Err) << "\n";    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static hipPitchedPtr P;
static const size_t Rows = 2, Slices = 4;

static size_t total() { return P.pitch * Rows * Slices; }

// Compares the whole allocation against the strided layout, returns 0 if equal.
static int verify(const char *Name, hipExtent E, int Val,
                  const std::vector<unsigned char> &Got) {
  for (size_t I = 0; I < total(); I++) {
    size_t Col = I % P.pitch, Row = (I / P.pitch) % Rows,
           Slice = I / (P.pitch * Rows);
    bool Set = Col < E.width && Row < E.height && Slice < E.depth;
    if (Got[I] != (Set ? Val : 0)) {
      std::cout << "FAIL: " << Name << " offset " << I << " is " << (int)Got[I]
                << " expected " << (Set ? Val : 0) << "\n";
      return 1;
    }
  }
  return 0;
}

static hipError_t zero() { return hipMemset(P.ptr, 0, total()); }

static hipError_t readBack(std::vector<unsigned char> &H) {
  H.assign(total(), 0xee);
  return hipMemcpy(H.data(), P.ptr, total(), hipMemcpyDeviceToHost);
}

int main() {
  CHECK(hipMalloc3D(&P, make_hipExtent(128, Rows, Slices)));
  hipStream_t S;
  CHECK(hipStreamCreate(&S));
  std::vector<unsigned char> H, Eager;

  // Capture a memset3D of E; Exec is left alive for the caller.
  auto capture = [&](hipExtent E, hipGraph_t &G, hipGraphExec_t &X) -> int {
    CHECK(zero());
    CHECK(hipStreamBeginCapture(S, hipStreamCaptureModeGlobal));
    CHECK(hipMemset3DAsync(P, 0x5a, E, S));
    CHECK(hipStreamEndCapture(S, &G));
    CHECK(hipGraphInstantiate(&X, G, nullptr, nullptr, 0));
    return 0;
  };
  auto launch = [&](hipGraphExec_t X) -> int {
    CHECK(hipGraphLaunch(X, S));
    CHECK(hipStreamSynchronize(S));
    return readBack(H) != hipSuccess;
  };

  hipGraph_t G;
  hipGraphExec_t X;
  hipExtent E = make_hipExtent(1, Rows, Slices);
  if (capture(E, G, X))
    return 1;
  if (launch(X))
    return 1;
  if (verify("captured (1,2,4)", E, 0x5a, H))
    return 1;

  // Eager path on the same allocation must produce the same bytes.
  CHECK(zero());
  CHECK(hipMemset3DAsync(P, 0x5a, E, S));
  CHECK(hipStreamSynchronize(S));
  CHECK(readBack(Eager));
  if (Eager != H) {
    std::cout << "FAIL: captured and eager (1,2,4) differ\n";
    return 1;
  }

  // Changing only the value through the exec graph keeps the depth.
  hipGraphNode_t Node;
  size_t N = 1;
  hipMemsetParams MP;
  CHECK(hipGraphGetNodes(G, &Node, &N));
  CHECK(hipGraphMemsetNodeGetParams(Node, &MP));
  MP.value = 0x33;
  CHECK(hipGraphExecMemsetNodeSetParams(X, Node, &MP));
  CHECK(zero());
  if (launch(X))
    return 1;
  if (verify("SetParams (1,2,4)", E, 0x33, H))
    return 1;

  // A zero depth must change nothing.
  E = make_hipExtent(1, Rows, 0);
  if (capture(E, G, X))
    return 1;
  if (launch(X))
    return 1;
  if (verify("captured (1,2,0)", E, 0x5a, H))
    return 1;

  std::cout << "PASS\n";
  return 0;
}
