#ifndef COOPERATIVE_GROUPS_H
#define COOPERATIVE_GROUPS_H

#include <cuda_runtime.h>
#include <algorithm>
using std::min, std::max;

namespace cooperative_groups {

class thread_block;

class thread_group {
public:
  // inline __device__ ~thread_group() {
  //   sync();
  // }
  __device__ thread_group(): groupSize(0), threadRank(0) {}

  inline __device__ int thread_rank() const { return threadRank; }

  inline __device__ int size() const { return groupSize; }

  [[gnu::used]] inline __device__ void sync() const { 
    // __syncthreads();
    }

  inline __device__ bool is_valid() const { return true; }

private:
  __device__ thread_group(int size)
      : groupSize(size), threadRank(threadIdx.x % groupSize) {
        int _ = thread_rank(); // This keeps the thread_rank function definition in the IR
    }

  friend __device__ thread_group tiled_partition(thread_block group, int size);
  friend __device__ thread_group tiled_partition(thread_group group, int size);
  template <int N>
  friend __device__ thread_group tiled_partition(thread_group group);

  int groupSize;
  int threadRank;
};


class thread_block: public thread_group {
public:
  inline __device__ thread_block() : thread_group() {}

  inline __device__ int thread_rank() const { return threadIdx.x; }

  inline __device__ int size() const { return blockDim.x; }

  inline __device__ void sync() const { __syncthreads(); }

  inline __device__ bool is_valid() const { return true; }

  inline __device__ dim3 group_index() const {
    return {blockIdx.x, blockIdx.y, blockIdx.z};
  }

  inline __device__ dim3 thread_index() const {
    return {threadIdx.x, threadIdx.y, threadIdx.z};
  }
};

template<unsigned int N>
using thread_block_tile = thread_group;

inline __device__ thread_block this_thread_block() { return thread_block(); }

inline __device__ thread_group tiled_partition(thread_block group, int size) {
  return thread_group(size);
}

inline __device__ thread_group tiled_partition(thread_group group, int size) {
  return thread_group(size);
}

template <int N>
inline __device__ thread_group tiled_partition(thread_group group) {
    return thread_group(N);
}

[[gnu::used]]
inline __device__ void synchronize(thread_block group) { 
  __syncthreads();
}

[[gnu::used]]
inline __device__ void sync(thread_block group) { 
  __syncthreads();
}

} // namespace cooperative_groups

#endif
