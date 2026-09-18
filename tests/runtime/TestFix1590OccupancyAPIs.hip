// Reproduces CHIP-SPV/chipStar#1590: the occupancy queries are unimplemented.
#include <hip/hip_runtime.h>
#include <cstdio>

__global__ void occupancyKernel(int *Out, int N) {
  int I = threadIdx.x + blockIdx.x * blockDim.x;
  if (I < N)
    Out[I] = I;
}

static int Failures = 0;

#define CHECK(Expr)                                                            \
  do {                                                                         \
    hipError_t Err_ = (Expr);                                                  \
    if (Err_ != hipSuccess) {                                                  \
      printf("FAILED: %s -> %s\n", #Expr, hipGetErrorString(Err_));            \
      return 1;                                                                \
    }                                                                          \
  } while (0)

#define EXPECT(Cond, ...)                                                      \
  do {                                                                         \
    if (!(Cond)) {                                                             \
      printf("FAILED: " __VA_ARGS__);                                          \
      ++Failures;                                                              \
    }                                                                          \
  } while (0)

/// An implemented query rejects a null hipFunction_t; a stub says NotSupported.
static void expectImplemented(const char *Name, hipError_t Err) {
  if (Err != hipErrorInvalidValue) {
    printf("FAILED: %s on a null function -> %s\n", Name,
           hipGetErrorString(Err));
    ++Failures;
  }
}

int main() {
  hipDeviceProp_t Props;
  CHECK(hipGetDeviceProperties(&Props, 0));

  hipFuncAttributes Attr;
  CHECK(hipFuncGetAttributes(&Attr, (const void *)occupancyKernel));
  const int MaxBlock = Attr.maxThreadsPerBlock < Props.maxThreadsPerBlock
                           ? Attr.maxThreadsPerBlock
                           : Props.maxThreadsPerBlock;

  int GridSize = 0, BlockSize = 0;
  CHECK(hipOccupancyMaxPotentialBlockSize(
      &GridSize, &BlockSize, (const void *)occupancyKernel, 0, 0));
  EXPECT(BlockSize > 0 && BlockSize <= MaxBlock,
         "block size %d out of range (1..%d)\n", BlockSize, MaxBlock);
  EXPECT(GridSize > 0, "grid size %d is not positive\n", GridSize);

  // A caller-supplied block size limit must be honored.
  int GridLtd = 0, BlockLtd = 0;
  CHECK(hipOccupancyMaxPotentialBlockSize(
      &GridLtd, &BlockLtd, (const void *)occupancyKernel, 0, 32));
  EXPECT(BlockLtd > 0 && BlockLtd <= 32,
         "block size limit 32 not honored: %d\n", BlockLtd);
  EXPECT(GridLtd > 0, "limited grid size %d is not positive\n", GridLtd);

  // Under a 768 limit the block must keep as many threads resident as 512.
  int Grid768 = 0, Block768 = 0, Active768 = 0, Active512 = 0;
  CHECK(hipOccupancyMaxPotentialBlockSize(
      &Grid768, &Block768, (const void *)occupancyKernel, 0, 768));
  CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &Active768, (const void *)occupancyKernel, Block768, 0));
  CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &Active512, (const void *)occupancyKernel, 512, 0));
  EXPECT(Block768 * Active768 >= 512 * Active512,
         "block %d keeps fewer threads resident than 512\n", Block768);

  // When memory allows one block, the unaligned 49 limit beats aligned 32.
  int Grid49 = 0, Block49 = 0;
  CHECK(hipOccupancyMaxPotentialBlockSize(
      &Grid49, &Block49, (const void *)occupancyKernel,
      Props.maxSharedMemoryPerMultiProcessor - Attr.sharedSizeBytes, 49));
  EXPECT(Block49 == 49, "one-block limit 49 gave block %d\n", Block49);

  // A dynamic shared memory request must not make the query fail.
  int GridShm = 0, BlockShm = 0;
  CHECK(hipOccupancyMaxPotentialBlockSize(
      &GridShm, &BlockShm, (const void *)occupancyKernel, 1024, 0));
  EXPECT(BlockShm > 0 && GridShm > 0,
         "dynamic shared memory query gave grid=%d block=%d\n", GridShm,
         BlockShm);

  int NumBlocks = 0;
  CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &NumBlocks, (const void *)occupancyKernel, BlockSize, 0));
  EXPECT(NumBlocks > 0, "active blocks per multiprocessor is %d\n", NumBlocks);

  int NumBlocksFlagged = 0;
  CHECK(hipOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
      &NumBlocksFlagged, (const void *)occupancyKernel, BlockSize, 0,
      hipOccupancyDefault));
  EXPECT(NumBlocksFlagged == NumBlocks,
         "hipOccupancyDefault changed the answer: %d vs %d\n",
         NumBlocksFlagged, NumBlocks);

  // The suggested geometry must actually be launchable.
  int *Out = nullptr;
  const int N = GridSize * BlockSize;
  CHECK(hipMalloc(&Out, sizeof(int) * N));
  occupancyKernel<<<GridSize, BlockSize>>>(Out, N);
  CHECK(hipGetLastError());
  CHECK(hipDeviceSynchronize());
  CHECK(hipFree(Out));

  // A shared memory request no multiprocessor can hold must be rejected.
  EXPECT(hipOccupancyMaxPotentialBlockSize(&GridShm, &BlockShm,
                                           (const void *)occupancyKernel,
                                           (size_t)-1, 0) != hipSuccess,
         "an impossible shared memory request succeeded\n");

  // hipModuleOccupancyMaxPotentialBlockSize also has to link.
  hipFunction_t NullFunc = nullptr;
  int MGrid = 0, MBlock = 0, MBlocks = 0;
  expectImplemented(
      "hipModuleOccupancyMaxPotentialBlockSize",
      hipModuleOccupancyMaxPotentialBlockSize(&MGrid, &MBlock, NullFunc, 0, 0));
  expectImplemented("hipModuleOccupancyMaxPotentialBlockSizeWithFlags",
                    hipModuleOccupancyMaxPotentialBlockSizeWithFlags(
                        &MGrid, &MBlock, NullFunc, 0, 0, hipOccupancyDefault));
  expectImplemented("hipModuleOccupancyMaxActiveBlocksPerMultiprocessor",
                    hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(
                        &MBlocks, NullFunc, 64, 0));
  expectImplemented(
      "hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags",
      hipModuleOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
          &MBlocks, NullFunc, 64, 0, hipOccupancyDefault));

  if (Failures)
    return 1;
  printf("PASSED\n");
  return 0;
}
