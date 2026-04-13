// Regression test for #555: hipGraph* API functions must return
// hipErrorInvalidValue for null pointer arguments instead of crashing.
#include <hip/hip_runtime_api.h>
#include <iostream>

int main() {
  // hipGraphCreate: null pGraph must not crash
  if (hipGraphCreate(nullptr, 0) != hipErrorInvalidValue)
    return 1;

  hipGraph_t graph;
  if (hipGraphCreate(&graph, 0) != hipSuccess)
    return 2;

  hipGraphNode_t node;

  // hipGraphAddEmptyNode: null pGraphNode or null graph
  if (hipGraphAddEmptyNode(nullptr, graph, nullptr, 0) != hipErrorInvalidValue)
    return 3;
  if (hipGraphAddEmptyNode(&node, nullptr, nullptr, 0) != hipErrorInvalidValue)
    return 4;

  if (hipGraphAddEmptyNode(&node, graph, nullptr, 0) != hipSuccess)
    return 5;

  // hipGraphNodeGetDependencies: null node or null pNumDependencies
  size_t numDeps = 0;
  if (hipGraphNodeGetDependencies(nullptr, nullptr, &numDeps) !=
      hipErrorInvalidValue)
    return 6;
  if (hipGraphNodeGetDependencies(node, nullptr, nullptr) !=
      hipErrorInvalidValue)
    return 7;

  // hipGraphNodeGetDependentNodes: null node or null pNumDependentNodes
  size_t numDependents = 0;
  if (hipGraphNodeGetDependentNodes(nullptr, nullptr, &numDependents) !=
      hipErrorInvalidValue)
    return 8;
  if (hipGraphNodeGetDependentNodes(node, nullptr, nullptr) !=
      hipErrorInvalidValue)
    return 9;

  // hipGraphExecDestroy: null graphExec must not crash
  if (hipGraphExecDestroy(nullptr) != hipErrorInvalidValue)
    return 10;

  // hipGraphAddMemcpyNode: null pDependencies with non-zero numDependencies
  // (bitwise-AND operator bug fixed to logical-AND)
  hipMemcpy3DParms copyParams = {};
  if (hipGraphAddMemcpyNode(&node, graph, nullptr, 1, &copyParams) !=
      hipErrorInvalidValue)
    return 11;

  hipGraphDestroy(graph);

  std::cout << "PASSED\n";
  return 0;
}
