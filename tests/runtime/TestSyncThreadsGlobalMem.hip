// Regression test for chipStar issue #632.
//
// __syncthreads() in CUDA/HIP must order GLOBAL memory as well as shared
// memory: writes to global memory made by a thread before __syncthreads()
// must be visible to all other threads in the block after __syncthreads().
//
// chipStar's __chip_syncthreads() historically lowered to
// barrier(CLK_LOCAL_MEM_FENCE), which only orders local (shared) memory.
// Kernels that hand data between threads through GLOBAL memory across a
// __syncthreads() (the HeCBench b+tree findK traversal is the canonical
// example) therefore get rare, nondeterministic wrong results.
//
// This test faithfully reproduces the b+tree findK single-writer / all-reader
// global handoff: each block traverses a B-ary search tree stored in global
// memory. At every level, the single thread whose key-bucket contains the
// target writes the chosen child into a per-block GLOBAL slot (offsetD);
// after a __syncthreads() thread 0 copies it into currKnodeD (also global);
// after a second __syncthreads() ALL threads read currKnodeD to address the
// next node. If __syncthreads() does not fence global memory, some threads
// read a stale currKnodeD, descend the wrong node, and produce a wrong answer.
//
// The test launches many blocks over many iterations and asserts every block
// of every launch produced the correct record. On a correct (global-fencing)
// barrier this is deterministic; on the buggy local-only barrier it flakes.

#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

#include <hip/hip_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x)                                                               \
  do {                                                                         \
    hipError_t _e = (x);                                                       \
    if (_e != hipSuccess) {                                                    \
      std::printf("HIP error %d at %s:%d\n", (int)_e, __FILE__, __LINE__);     \
      std::printf("FAIL\n");                                                   \
      return 1;                                                                \
    }                                                                          \
  } while (0)

// Mirror the HeCBench b+tree "mil.txt" dataset shape that exposes the race:
// order 256 (all 256 threads participate in the per-node key comparison) and
// height 2. This structure reliably flakes on the buggy local-only barrier;
// shallower/narrower trees do not stress the global handoff enough.
static const int NTHREADS = 256; // threads per block
static const int ORDER = 256;     // branching factor of the search tree
static const int HEIGHT = 2;      // descents from root to leaf
static const int INF = 0x7fffffff;

// Device-side node layout mirroring HeCBench b+tree "knode".
struct knode {
  int location;
  int indices[NTHREADS + 1];
  int keys[NTHREADS + 1];
};
struct record {
  int value;
};

// Faithful HIP port of the HeCBench b+tree findK kernel.
__global__ void findK(long height, const knode *knodesD, long knodes_elem,
                      const record *recordsD, long *currKnodeD, long *offsetD,
                      const int *keysD, record *ansD) {
  int thid = threadIdx.x;
  int bid = blockIdx.x;

  for (long i = 0; i < height; i++) {
    if ((knodesD[currKnodeD[bid]].keys[thid]) <= keysD[bid] &&
        (knodesD[currKnodeD[bid]].keys[thid + 1] > keysD[bid])) {
      if (knodesD[offsetD[bid]].indices[thid] < knodes_elem) {
        offsetD[bid] = knodesD[offsetD[bid]].indices[thid];
      }
    }
    __syncthreads();
    if (thid == 0) {
      currKnodeD[bid] = offsetD[bid];
    }
    __syncthreads();
  }

  if (knodesD[currKnodeD[bid]].keys[thid] == keysD[bid]) {
    ansD[bid].value = recordsD[knodesD[currKnodeD[bid]].indices[thid]].value;
  }
}

// Recursively build a perfect ORDER-ary search tree covering [lo, hi).
// Returns the node id (nodes are appended; root ends up last).
static int buildTree(std::vector<knode> &nodes, int lo, int hi, int level) {
  knode n;
  n.location = 0;
  for (int j = 0; j <= NTHREADS; j++) {
    n.keys[j] = INF;
    n.indices[j] = 0;
  }

  if (level == HEIGHT) {
    // Leaf: covers ORDER consecutive keys [lo, lo+ORDER). keys[j] holds the
    // stored key, indices[j] the matching record id (== the key value).
    for (int j = 0; j < ORDER; j++) {
      n.keys[j] = lo + j;
      n.indices[j] = lo + j;
    }
    n.keys[ORDER] = hi; // = lo + ORDER
    int id = (int)nodes.size();
    nodes.push_back(n);
    return id;
  }

  int step = (hi - lo) / ORDER;
  int childId[ORDER];
  int childLo[ORDER];
  for (int j = 0; j < ORDER; j++) {
    int clo = lo + j * step;
    int chi = lo + (j + 1) * step;
    childLo[j] = clo;
    childId[j] = buildTree(nodes, clo, chi, level + 1);
  }
  for (int j = 0; j < ORDER; j++) {
    n.keys[j] = childLo[j];
    n.indices[j] = childId[j];
  }
  n.keys[ORDER] = hi; // upper sentinel for the last bucket
  int id = (int)nodes.size();
  nodes.push_back(n);
  return id;
}

int main() {
  // Total key space so that each leaf covers exactly ORDER keys.
  long keyspace = 1;
  for (int i = 0; i < HEIGHT + 1; i++)
    keyspace *= ORDER; // ORDER^(HEIGHT+1)

  std::vector<knode> nodes;
  int rootId = buildTree(nodes, 0, (int)keyspace, 0);
  long knodes_elem = (long)nodes.size();

  // Records: value = key*7 + 3, so a correct answer is target*7+3.
  std::vector<record> records((size_t)keyspace);
  for (long k = 0; k < keyspace; k++)
    records[(size_t)k].value = (int)(k * 7 + 3);

  const int NBLOCKS = 10000;
  std::vector<int> keysH((size_t)NBLOCKS);
  std::vector<int> golden((size_t)NBLOCKS);
  // Deterministic spread of targets across the whole key space.
  for (int b = 0; b < NBLOCKS; b++) {
    int target = (int)(((long)b * 2654435761u) % keyspace);
    keysH[b] = target;
    golden[b] = target * 7 + 3;
  }

  knode *knodesD;
  record *recordsD;
  record *ansD;
  long *currKnodeD;
  long *offsetD;
  int *keysD;
  CHECK(hipMalloc(&knodesD, nodes.size() * sizeof(knode)));
  CHECK(hipMalloc(&recordsD, records.size() * sizeof(record)));
  CHECK(hipMalloc(&ansD, NBLOCKS * sizeof(record)));
  CHECK(hipMalloc(&currKnodeD, NBLOCKS * sizeof(long)));
  CHECK(hipMalloc(&offsetD, NBLOCKS * sizeof(long)));
  CHECK(hipMalloc(&keysD, NBLOCKS * sizeof(int)));

  CHECK(hipMemcpy(knodesD, nodes.data(), nodes.size() * sizeof(knode),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(recordsD, records.data(), records.size() * sizeof(record),
                  hipMemcpyHostToDevice));
  CHECK(hipMemcpy(keysD, keysH.data(), NBLOCKS * sizeof(int),
                  hipMemcpyHostToDevice));

  std::vector<long> rootInit((size_t)NBLOCKS, (long)rootId);
  std::vector<record> ansH((size_t)NBLOCKS);

  const int NRUNS = 50;
  long totalWrong = 0;
  for (int run = 0; run < NRUNS; run++) {
    // Reset per-block traversal state and answers.
    CHECK(hipMemcpy(currKnodeD, rootInit.data(), NBLOCKS * sizeof(long),
                    hipMemcpyHostToDevice));
    CHECK(hipMemcpy(offsetD, rootInit.data(), NBLOCKS * sizeof(long),
                    hipMemcpyHostToDevice));
    CHECK(hipMemset(ansD, 0xff, NBLOCKS * sizeof(record))); // sentinel -1

    findK<<<NBLOCKS, NTHREADS>>>(HEIGHT, knodesD, knodes_elem, recordsD,
                                 currKnodeD, offsetD, keysD, ansD);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());
    CHECK(hipMemcpy(ansH.data(), ansD, NBLOCKS * sizeof(record),
                    hipMemcpyDeviceToHost));

    int wrong = 0;
    for (int b = 0; b < NBLOCKS; b++)
      if (ansH[b].value != golden[b])
        wrong++;
    if (wrong)
      std::printf("run %d: %d/%d blocks wrong\n", run, wrong, NBLOCKS);
    totalWrong += wrong;
  }

  std::printf("total wrong over %d runs x %d blocks = %ld\n", NRUNS, NBLOCKS,
              totalWrong);

  hipFree(knodesD);
  hipFree(recordsD);
  hipFree(ansD);
  hipFree(currKnodeD);
  hipFree(offsetD);
  hipFree(keysD);

  if (totalWrong != 0) {
    std::printf("FAIL\n");
    return 1;
  }
  std::printf("PASS\n");
  return 0;
}
