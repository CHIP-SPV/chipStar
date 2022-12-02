# Graphs Overview

HIP graphs allow you to create a graph of kernels and data transfers that can be executed on the GPU. This is useful for applications that have a complex flow of data and operations. The graph can be optimized and executed asynchronously to improve performance.

## CHIP-SPV Graph Implementation

Graphs can be constructed manually by creating nodes and defining relationships between nodes, or by using `hipStreamCaptureBegin()` and `hipStreamCaptureEnd()` to capture a sequence of operations. The graph is then executed by calling `hipGraphLaunch()`. 

### Manual Graph Construction

Graphs are constructed by creating nodes and edges. Nodes represent operations, such as kernels, data transfers, or other graphs. Edges represent dependencies between nodes. The graph is constructed by adding nodes and edges to the graph object. All graph nodes nodes inherit from `CHIPGraphNode` class which defines the minimal set of functions to be implemted so the implementation can be extended to support different types of nodes in the future. 

### Automatic Graph Construction

When using `hipStreamCaptureBegin()` and `hipStreamCaptureEnd()` to capture a sequence of operations, the graph is constructed automatically. `CHIPQueue` object has fields which track whether it's in recording mode and what was the last node that was recorded. In the current implementation, each graph node has a dependency on the previous node so compared to manually constructured graphs, the automatic graphs have less parallelism. This can be improved in the future by adding graph analysis logic to re-arrange the graph. 

### Graph Compilation

Before a graph can be executed it must be compiled into a `CHIPGraphExec` object. This object contains the compiled graph and the memory allocations for the graph parameters. The graph is executed by calling `hipGraphLaunch()` which takes a `CHIPGraphExec` object and a stream. The graph is executed asynchronously on the stream. 

### Graph Optimization

During compilation of `CHIPGraphExec` object, the graph is analyzed and optimized. First, a pointer to the original graph is stored and a the orignal graph is cloned. Then, the graph is traversed using Depth-First Search (DFS) to acquire all the excecution paths from root to leaf nodes. Unnecessary dependency endges are then trimmed by checking if any of the shorter paths are a subset of longer ones. 

### Graph Execution

Once the unnecessary edges are trimmed away, an execution queue is created. This queue is made up of sets of nodes that can be executed concurrently in any order on a single or multiple streams. This queue is constructed by adding all the root nodes to the first set of the queue. Then, the remaining nodes are analyzed by checking if their dependencies are in the previous set. If so, they are added to the next set. This process is repeated until all the nodes are added to the queue. Once this queue is constructed, the graph can be launched onto a stream. Currently only a single HIP Stream/CHIPQueue is used but this can easily be changed in future. Graph nodes are executed by calling their `execute()` function. At this time, `execute()` calls regular HIP API with the parameters stored in the node so no performance benefit is expected. 

### Future Work

* Add support for executing a graph on multiple streams
* Take advantage of Level Zero command lists
* Take advantage of OpenCL command buffers
* Improve graph optimization by adding graph analysis logic to re-arrange the graph for automatically constructed graphs

