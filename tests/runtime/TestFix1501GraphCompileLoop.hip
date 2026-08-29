// Reproduces: CHIPGraphExec::compile() spins forever, growing its execution
// queue, when no schedule pass can place a node (the only node of a graph
// depends on a node of another graph, or the graph has a cycle and thus no
// root), and pruneGraph_() removes the redundant edges of the caller's graph
// on every hipGraphLaunch, so hipGraphGetEdges under reports after a launch.
#include <hip/hip_runtime_api.h>
#include <csignal>
#include <iostream>
#include <sys/wait.h>
#include <unistd.h>

#define CHECK(Call)                                                            \
  do {                                                                         \
    hipError_t Err = (Call);                                                   \
    if (Err != hipSuccess) {                                                   \
      std::cout << "FAIL: " #Call " -> " << hipGetErrorString(Err) << "\n";    \
      return 1;                                                                \
    }                                                                          \
  } while (0)

static void onAlarm(int) {
  static const char Msg[] = "FAIL: the graph launch did not return\n";
  write(STDOUT_FILENO, Msg, sizeof(Msg) - 1);
  _exit(1);
}

// Instantiate Graph in a child process, so that a launch that never returns
// only costs this case, and report whether instantiation was rejected.
static int expectRejected(hipGraph_t Graph, const char *What) {
  pid_t Child = fork();
  if (Child < 0) {
    std::cout << "FAIL: fork\n";
    return 1;
  }
  if (Child == 0) {
    signal(SIGALRM, onAlarm);
    alarm(5);
    hipGraphExec_t Exec = nullptr;
    hipError_t Err = hipGraphInstantiate(&Exec, Graph, nullptr, nullptr, 0);
    if (Err == hipErrorInvalidValue)
      _exit(0);
    std::cout << "FAIL: hipGraphInstantiate of " << What << " returned "
              << hipGetErrorString(Err) << ", expected hipErrorInvalidValue\n";
    if (Err == hipSuccess)
      hipGraphLaunch(Exec, 0); // never returns without the fix
    _exit(1);
  }
  int Status = 0;
  waitpid(Child, &Status, 0);
  return WIFEXITED(Status) && WEXITSTATUS(Status) == 0 ? 0 : 1;
}

int main() {
  int Failed = 0;
  std::cout << std::unitbuf; // keep the output when a child _exit()s

  // A -> B -> C plus the redundant edge A -> C. Instantiating and launching
  // the graph twice must leave its nodes and edges as the caller built them.
  hipGraph_t Graph;
  CHECK(hipGraphCreate(&Graph, 0));
  hipGraphNode_t A, B, C;
  CHECK(hipGraphAddEmptyNode(&A, Graph, nullptr, 0));
  CHECK(hipGraphAddEmptyNode(&B, Graph, &A, 1));
  hipGraphNode_t DepsC[] = {A, B};
  CHECK(hipGraphAddEmptyNode(&C, Graph, DepsC, 2));

  hipGraphNode_t Nodes[3];
  size_t NumNodes = 0, NumEdges = 0;
  CHECK(hipGraphGetNodes(Graph, Nodes, &NumNodes));
  CHECK(hipGraphGetEdges(Graph, nullptr, nullptr, &NumEdges));
  if (NumNodes != 3 || NumEdges != 3) {
    std::cout << "FAIL: built " << NumNodes << " nodes and " << NumEdges
              << " edges, expected 3 and 3\n";
    return 1;
  }

  hipGraphExec_t Exec;
  CHECK(hipGraphInstantiate(&Exec, Graph, nullptr, nullptr, 0));
  for (int Launch = 0; Launch < 2; Launch++) {
    CHECK(hipGraphLaunch(Exec, 0));
    CHECK(hipStreamSynchronize(0));
  }

  CHECK(hipGraphGetNodes(Graph, Nodes, &NumNodes));
  CHECK(hipGraphGetEdges(Graph, nullptr, nullptr, &NumEdges));
  if (NumNodes != 3 || NumEdges != 3) {
    std::cout << "FAIL: after two launches the graph has " << NumNodes
              << " nodes and " << NumEdges << " edges, expected 3 and 3\n";
    Failed = 1;
  }
  CHECK(hipGraphExecDestroy(Exec));
  CHECK(hipGraphDestroy(Graph));

  // A graph whose only node depends on a node of another graph can never be
  // scheduled. hipGraphInstantiate must reject it instead of leaving the
  // launch to spin forever.
  hipGraph_t Other, Foreign;
  CHECK(hipGraphCreate(&Other, 0));
  CHECK(hipGraphCreate(&Foreign, 0));
  hipGraphNode_t OtherNode, ForeignNode;
  CHECK(hipGraphAddEmptyNode(&OtherNode, Other, nullptr, 0));
  CHECK(hipGraphAddEmptyNode(&ForeignNode, Foreign, &OtherNode, 1));
  Failed |= expectRejected(Foreign, "the graph with a foreign dependency");
  CHECK(hipGraphDestroy(Foreign));
  CHECK(hipGraphDestroy(Other));

  // A graph whose every node has a dependency has no root, so no node can go
  // first. Here X and Y depend on each other and Z waits on X.
  hipGraph_t Cyclic;
  CHECK(hipGraphCreate(&Cyclic, 0));
  hipGraphNode_t X, Y, Z;
  CHECK(hipGraphAddEmptyNode(&X, Cyclic, nullptr, 0));
  CHECK(hipGraphAddEmptyNode(&Y, Cyclic, &X, 1));
  CHECK(hipGraphAddEmptyNode(&Z, Cyclic, &X, 1));
  CHECK(hipGraphAddDependencies(Cyclic, &Y, &X, 1));
  Failed |= expectRejected(Cyclic, "the cyclic graph");
  CHECK(hipGraphDestroy(Cyclic));

  std::cout << (Failed ? "FAILED\n" : "PASSED\n");
  return Failed;
}
