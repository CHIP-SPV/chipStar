// Regression test for issue #177: unsupported mixed-dimensional texture
// use must produce a diagnostic instead of crashing HipTextureLowering.
#include <hip/hip_runtime.h>

template <typename T>
__global__ void mixedTexture(T *output, hipTextureObject_t texObj, size_t width,
                             size_t height) {
  const unsigned int x = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int y = blockIdx.y * blockDim.y + threadIdx.y;
  const float u = x / static_cast<float>(width);

  if (height == 0) {
    output[x] = tex1D<T>(texObj, u);
  } else {
    const float v = y / static_cast<float>(height);
    output[y * width + x] = tex2D<T>(texObj, u, v);
  }
}

template __global__ void mixedTexture<float>(float *, hipTextureObject_t, size_t,
                                             size_t);
