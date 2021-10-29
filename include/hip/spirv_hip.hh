#ifndef HIP_H
#define HIP_H

#include <stddef.h>
#include <stdint.h>

#include <hip/driver_types.h>

#if defined(__clang__) && defined(__HIP__)

#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))

#define __noinline__ __attribute__((noinline))
#define __forceinline__ inline __attribute__((always_inline))

#else

/**
 * Function and kernel markers
 */
#define __host__
#define __device__
#define __global__
#define __shared__
#define __constant__

#define __noinline__
#define __forceinline__ inline

#endif

#if defined(__clang__) && defined(__HIP__)
#include "spirv_hip_mathlib.hh"

#define uint uint32_t

#define HIP_KERNEL_NAME(...) __VA_ARGS__
#define HIP_SYMBOL(X) #X

#define HIP_DYNAMIC_SHARED(type, var) __shared__ type var[4294967295];

#define HIP_DYNAMIC_SHARED_ATTRIBUTE

typedef int hipLaunchParm;
#define hipLaunchKernelGGLInternal(kernelName, numBlocks, numThreads,     \
                                   memPerBlock, streamId, ...)            \
  do {                                                                    \
    kernelName<<<(numBlocks), (numThreads), (memPerBlock), (streamId)>>>( \
        __VA_ARGS__);                                                     \
  } while (0)

#define hipLaunchKernelGGL(kernelName, ...) \
  hipLaunchKernelGGLInternal((kernelName), __VA_ARGS__)
// TODO implement this as hipLaunchKernelGGL
// #define hipLaunchKernel(kernelName, numblocks, numthreads, memperblock,   \
//                         streamId, ...)                                    \
//   do {                                                                    \
//     kernelName<<<(numblocks), (numthreads), (memperblock), (streamId)>>>( \
//         hipLaunchParm{}, ##__VA_ARGS__);                                  \
//   } while (0)

//#define hipLaunchKernelGGL(kernelName, numblocks, numthreads, memperblock, \
//                           streamId, ...)                                  \
//  do {                                                                     \
//    kernelName<<<(numblocks), (numthreads), (memperblock), (streamId)>>>(  \
//        __VA_ARGS__);                                                      \
//  } while (0)

// template <std::size_t n, typename... Ts,
//           typename std::enable_if<n == sizeof...(Ts)>::type* = nullptr>
// void pArgs(const std::tuple<Ts...>&, void*) {}

// template <std::size_t n, typename... Ts,
//           typename std::enable_if<n != sizeof...(Ts)>::type* = nullptr>
// void pArgs(const std::tuple<Ts...>& formals, void** _vargs) {
//   using T = typename std::tuple_element<n, std::tuple<Ts...> >::type;

//   static_assert(!std::is_reference<T>{},
//                 "A __global__ function cannot have a reference as one of its
//                 " "arguments.");
// #if defined(HIP_STRICT)
//   static_assert(std::is_trivially_copyable<T>{},
//                 "Only TriviallyCopyable types can be arguments to a
//                 __global__ " "function");
// #endif
//   _vargs[n] =
//       const_cast<void*>(reinterpret_cast<const
//       void*>(&std::get<n>(formals)));
//   return pArgs<n + 1>(formals, _vargs);
// }

// template <typename... Formals, typename... Actuals>
// std::tuple<Formals...> validateArgsCountType(void (*kernel)(Formals...),
//                                              std::tuple<Actuals...>(actuals))
//                                              {
//   static_assert(sizeof...(Formals) == sizeof...(Actuals),
//                 "Argument Count Mismatch");
//   std::tuple<Formals...> to_formals{std::move(actuals)};
//   return to_formals;
// }

// template <typename... Args, typename F = void (*)(Args...)>
// void hipLaunchKernelGGL(F kernel, const dim3& numBlocks, const dim3&
// dimBlocks,
//                         size_t sharedMemBytes, hipStream_t stream,
//                         Args... args) {
//   constexpr size_t count = sizeof...(Args);
//   auto tup_ = std::tuple<Args...>{args...};
//   auto tup = validateArgsCountType(kernel, tup_);
//   void* _Args[count];
//   pArgs<0>(tup, _Args);

//   const auto k = reinterpret_cast<void*>(kernel);

//   hipLaunchKernel(k, const_cast<dim3&>(numBlocks),
//   const_cast<dim3&>(dimBlocks),
//                   _Args, sharedMemBytes, stream);
// }

#pragma push_macro("__DEVICE__")
#define __DEVICE__ static __device__ __forceinline__

extern "C" __device__ size_t _Z12get_local_idj(uint);
__DEVICE__ uint __hip_get_thread_idx_x() { return _Z12get_local_idj(0); }
__DEVICE__ uint __hip_get_thread_idx_y() { return _Z12get_local_idj(1); }
__DEVICE__ uint __hip_get_thread_idx_z() { return _Z12get_local_idj(2); }

extern "C" __device__ size_t _Z12get_group_idj(uint);
__DEVICE__ uint __hip_get_block_idx_x() { return _Z12get_group_idj(0); }
__DEVICE__ uint __hip_get_block_idx_y() { return _Z12get_group_idj(1); }
__DEVICE__ uint __hip_get_block_idx_z() { return _Z12get_group_idj(2); }

extern "C" __device__ size_t _Z14get_local_sizej(uint);
__DEVICE__ uint __hip_get_block_dim_x() { return _Z14get_local_sizej(0); }
__DEVICE__ uint __hip_get_block_dim_y() { return _Z14get_local_sizej(1); }
__DEVICE__ uint __hip_get_block_dim_z() { return _Z14get_local_sizej(2); }

extern "C" __device__ size_t _Z14get_num_groupsj(uint);
__DEVICE__ uint __hip_get_grid_dim_x() { return _Z14get_num_groupsj(0); }
__DEVICE__ uint __hip_get_grid_dim_y() { return _Z14get_num_groupsj(1); }
__DEVICE__ uint __hip_get_grid_dim_z() { return _Z14get_num_groupsj(2); }

#define __HIP_DEVICE_BUILTIN(DIMENSION, FUNCTION)               \
  __declspec(property(get = __get_##DIMENSION)) uint DIMENSION; \
  __DEVICE__ uint __get_##DIMENSION(void) { return FUNCTION; }

struct __hip_builtin_threadIdx_t {
  __HIP_DEVICE_BUILTIN(x, __hip_get_thread_idx_x());
  __HIP_DEVICE_BUILTIN(y, __hip_get_thread_idx_y());
  __HIP_DEVICE_BUILTIN(z, __hip_get_thread_idx_z());
};

struct __hip_builtin_blockIdx_t {
  __HIP_DEVICE_BUILTIN(x, __hip_get_block_idx_x());
  __HIP_DEVICE_BUILTIN(y, __hip_get_block_idx_y());
  __HIP_DEVICE_BUILTIN(z, __hip_get_block_idx_z());
};

struct __hip_builtin_blockDim_t {
  __HIP_DEVICE_BUILTIN(x, __hip_get_block_dim_x());
  __HIP_DEVICE_BUILTIN(y, __hip_get_block_dim_y());
  __HIP_DEVICE_BUILTIN(z, __hip_get_block_dim_z());
};

struct __hip_builtin_gridDim_t {
  __HIP_DEVICE_BUILTIN(x, __hip_get_grid_dim_x());
  __HIP_DEVICE_BUILTIN(y, __hip_get_grid_dim_y());
  __HIP_DEVICE_BUILTIN(z, __hip_get_grid_dim_z());
};

#undef uint
#undef __HIP_DEVICE_BUILTIN
#pragma pop_macro("__DEVICE__")

extern const __device__ __attribute__((weak))
__hip_builtin_threadIdx_t threadIdx;
extern const __device__ __attribute__((weak)) __hip_builtin_blockIdx_t blockIdx;
extern const __device__ __attribute__((weak)) __hip_builtin_blockDim_t blockDim;
extern const __device__ __attribute__((weak)) __hip_builtin_gridDim_t gridDim;

#define hipThreadIdx_x threadIdx.x
#define hipThreadIdx_y threadIdx.y
#define hipThreadIdx_z threadIdx.z

#define hipBlockIdx_x blockIdx.x
#define hipBlockIdx_y blockIdx.y
#define hipBlockIdx_z blockIdx.z

#define hipBlockDim_x blockDim.x
#define hipBlockDim_y blockDim.y
#define hipBlockDim_z blockDim.z

#define hipGridDim_x gridDim.x
#define hipGridDim_y gridDim.y
#define hipGridDim_z gridDim.z

#endif  // defined(__clang__) && defined(__HIP__)

/*************************************************************************************************/

#include "hip_to_chip.hh"

typedef enum {
  EVENT_STATUS_INIT = 5,
  EVENT_STATUS_RECORDING,
  EVENT_STATUS_RECORDED
} event_status_e;

#endif