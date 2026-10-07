// Reproduces: copying a CHIPGraph (hipGraphClone, hipGraphInstantiate) writes
// blank lines to the application's stdout.
#include <hip/hip_runtime_api.h>
#include <cstdio>
#include <iostream>
#include <unistd.h>

#define CHECK(Call)                                                            \
  do {                                                                         \
    hipError_t Err = (Call);                                                   \
    if (Err != hipSuccess) {                                                   \
      std::cout << "FAIL: " #Call " -> " << hipGetErrorString(Err) << "\n";    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

int main() {
  hipGraph_t Graph, Clone;
  hipGraphNode_t Node;
  hipGraphExec_t Exec;
  CHECK(hipGraphCreate(&Graph, 0));
  CHECK(hipGraphAddEmptyNode(&Node, Graph, nullptr, 0));

  // Send stdout to a temporary file while the graph is copied.
  std::cout.flush();
  FILE *Captured = tmpfile();
  int SavedStdout = dup(STDOUT_FILENO);
  if (!Captured || SavedStdout < 0 ||
      dup2(fileno(Captured), STDOUT_FILENO) < 0) {
    std::cout << "FAIL: cannot redirect stdout\n";
    return 1;
  }
  hipError_t CloneErr = hipGraphClone(&Clone, Graph);
  hipError_t InstErr = hipGraphInstantiate(&Exec, Graph, nullptr, nullptr, 0);
  std::cout.flush();
  fflush(stdout);
  dup2(SavedStdout, STDOUT_FILENO);
  fseek(Captured, 0, SEEK_END);
  long Written = ftell(Captured);

  CHECK(CloneErr);
  CHECK(InstErr);
  if (Written != 0) {
    std::cout << "FAIL: copying a graph wrote " << Written
              << " bytes to stdout\n";
    return 1;
  }
  CHECK(hipGraphExecDestroy(Exec));
  CHECK(hipGraphDestroy(Clone));
  CHECK(hipGraphDestroy(Graph));
  std::cout << "PASSED\n";
  return 0;
}
