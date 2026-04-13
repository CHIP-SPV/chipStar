// Regression test for #556: hipGraphAddDependencies and
// hipGraphRemoveDependencies must process all from[i]->to[i] pairs,
// not only the first 'to' node.
#include <hip/hip_runtime_api.h>
#include <iostream>

int main() {
  hipGraph_t graph;
  if (hipGraphCreate(&graph, 0) != hipSuccess)
    return 1;

  // Create four independent nodes.
  hipGraphNode_t nodeA, nodeB, nodeC, nodeD;
  if (hipGraphAddEmptyNode(&nodeA, graph, nullptr, 0) != hipSuccess)
    return 2;
  if (hipGraphAddEmptyNode(&nodeB, graph, nullptr, 0) != hipSuccess)
    return 3;
  if (hipGraphAddEmptyNode(&nodeC, graph, nullptr, 0) != hipSuccess)
    return 4;
  if (hipGraphAddEmptyNode(&nodeD, graph, nullptr, 0) != hipSuccess)
    return 5;

  // Add two dependencies at once: A->C and B->D.
  hipGraphNode_t from[2] = {nodeA, nodeB};
  hipGraphNode_t to[2]   = {nodeC, nodeD};
  if (hipGraphAddDependencies(graph, from, to, 2) != hipSuccess)
    return 6;

  // nodeC must depend on nodeA (first pair).
  size_t numDepsC = 0;
  if (hipGraphNodeGetDependencies(nodeC, nullptr, &numDepsC) != hipSuccess)
    return 7;
  if (numDepsC != 1)
    return 8; // Before fix: nodeC has 2 deps (nodeA and nodeB)

  // nodeD must depend on nodeB (second pair).
  // Before the fix only the first 'to' (nodeC) is updated; nodeD gets no dep.
  size_t numDepsD = 0;
  if (hipGraphNodeGetDependencies(nodeD, nullptr, &numDepsD) != hipSuccess)
    return 9;
  if (numDepsD != 1)
    return 10; // Before fix: nodeD has 0 deps

  // Now test hipGraphRemoveDependencies: remove A->C and B->D.
  if (hipGraphRemoveDependencies(graph, from, to, 2) != hipSuccess)
    return 11;

  // Both nodeC and nodeD should have no dependencies now.
  if (hipGraphNodeGetDependencies(nodeC, nullptr, &numDepsC) != hipSuccess)
    return 12;
  if (numDepsC != 0)
    return 13;

  if (hipGraphNodeGetDependencies(nodeD, nullptr, &numDepsD) != hipSuccess)
    return 14;
  if (numDepsD != 0)
    return 15; // Before fix: nodeD still has 0 deps (was never added)

  hipGraphDestroy(graph);

  std::cout << "PASSED\n";
  return 0;
}
