// Width-parameterized warp shuffle test for issue #726.
//
// Exercises __shfl / __shfl_up / __shfl_down / __shfl_xor with every valid
// width (power of two <= warpSize) over a warpSize-lane warp with a known
// input, comparing device results against host-computed CUDA reference
// semantics. The warp size is queried at runtime so the test adapts to the
// device (32 on NVIDIA/Intel, 64 on AMD, 8 on Mali); widths larger than the
// hardware warp are undefined in CUDA and cannot be expressed via OpenCL
// sub_group_shuffle (which cannot cross subgroup boundaries), so they are
// skipped rather than tested.
//
// CUDA semantics recap (width is a power of two, warp split into contiguous
// segments of `width` lanes, each with logical lane IDs 0..width-1):
//   __shfl(src)        : read segBase + (src % width)
//   __shfl_up(delta)   : read lane-delta, clamped at the segment base (else own)
//   __shfl_down(delta) : read lane+delta, clamped at the segment top  (else own)
//   __shfl_xor(mask)   : read lane^mask if it stays inside the segment, else own

#include <hip/hip_runtime.h>
#include <cstdio>

// Buffers are sized to the largest warp we support; the ACTUAL lane count is
// the device warpSize, queried at runtime.
#define MAX_WARP 64

__global__ void run_shuffles(const int *in, int width, int srcLane, int delta,
                             int xorMask, int *out_shfl, int *out_up,
                             int *out_down, int *out_xor) {
  int lane = threadIdx.x;
  int v = in[lane];
  out_shfl[lane] = __shfl(v, srcLane, width);
  out_up[lane] = __shfl_up(v, delta, width);
  out_down[lane] = __shfl_down(v, delta, width);
  out_xor[lane] = __shfl_xor(v, xorMask, width);
}

// Host reference implementations.
static int ref_shfl(const int *in, int lane, int width, int src) {
  int segBase = (lane / width) * width;
  int idx = segBase + (((unsigned)src) % (unsigned)width);
  return in[idx];
}
static int ref_up(const int *in, int lane, int width, int delta) {
  int laneInSeg = lane % width;
  int srcInSeg = laneInSeg - delta;
  int idx = (srcInSeg < 0) ? lane : (lane - delta);
  return in[idx];
}
static int ref_down(const int *in, int lane, int width, int delta) {
  int laneInSeg = lane % width;
  int srcInSeg = laneInSeg + delta;
  int idx = (srcInSeg >= width) ? lane : (lane + delta);
  return in[idx];
}
static int ref_xor(const int *in, int lane, int width, int mask) {
  int segBase = (lane / width) * width;
  int src = lane ^ mask;
  if (src < segBase || src >= segBase + width)
    src = lane;
  return in[src];
}

int main() {
  int device = 0;
  hipDeviceProp_t props;
  if (hipGetDeviceProperties(&props, device) != hipSuccess) {
    printf("FAILED! (could not query device properties)\n");
    return 1;
  }
  int warp = props.warpSize;
  if (warp < 2 || warp > MAX_WARP) {
    printf("FAILED! (unexpected warpSize=%d)\n", warp);
    return 1;
  }
  printf("warpSize=%d\n", warp);

  int h_in[MAX_WARP];
  for (int i = 0; i < warp; ++i)
    h_in[i] = 100 + i; // distinctive known input

  int *d_in, *d_shfl, *d_up, *d_down, *d_xor;
  hipMalloc(&d_in, warp * sizeof(int));
  hipMalloc(&d_shfl, warp * sizeof(int));
  hipMalloc(&d_up, warp * sizeof(int));
  hipMalloc(&d_down, warp * sizeof(int));
  hipMalloc(&d_xor, warp * sizeof(int));
  hipMemcpy(d_in, h_in, warp * sizeof(int), hipMemcpyHostToDevice);

  const int allWidths[] = {2, 4, 8, 16, 32};
  // For each width test a source lane that forces modulo-wrap, a delta, and
  // two xor masks (one in-segment, one that can cross a small segment).
  const int srcLanes[] = {0, 3, 5};
  const int deltas[] = {1, 2};
  const int xorMasks[] = {1, 2, 4};

  int h_shfl[MAX_WARP], h_up[MAX_WARP], h_down[MAX_WARP], h_xor[MAX_WARP];
  int failures = 0;
  int tested = 0;

  for (int wi = 0; wi < 5; ++wi) {
    int width = allWidths[wi];
    // Only widths that fit within the hardware warp are well-defined; skip the
    // rest (e.g. 16/32 on Mali's 8-wide subgroups).
    if (width > warp)
      continue;
    tested++;
    for (int si = 0; si < 3; ++si) {
      for (int di = 0; di < 2; ++di) {
        for (int xi = 0; xi < 3; ++xi) {
          int src = srcLanes[si], delta = deltas[di], xm = xorMasks[xi];
          hipLaunchKernelGGL(run_shuffles, dim3(1), dim3(warp), 0, 0, d_in,
                             width, src, delta, xm, d_shfl, d_up, d_down,
                             d_xor);
          hipDeviceSynchronize();
          hipMemcpy(h_shfl, d_shfl, warp * sizeof(int), hipMemcpyDeviceToHost);
          hipMemcpy(h_up, d_up, warp * sizeof(int), hipMemcpyDeviceToHost);
          hipMemcpy(h_down, d_down, warp * sizeof(int), hipMemcpyDeviceToHost);
          hipMemcpy(h_xor, d_xor, warp * sizeof(int), hipMemcpyDeviceToHost);

          for (int lane = 0; lane < warp; ++lane) {
            int e_shfl = ref_shfl(h_in, lane, width, src);
            int e_up = ref_up(h_in, lane, width, delta);
            int e_down = ref_down(h_in, lane, width, delta);
            int e_xor = ref_xor(h_in, lane, width, xm);
            if (h_shfl[lane] != e_shfl) {
              printf("FAIL shfl w=%d src=%d lane=%d got=%d exp=%d\n", width,
                     src, lane, h_shfl[lane], e_shfl);
              failures++;
            }
            if (h_up[lane] != e_up) {
              printf("FAIL shfl_up w=%d delta=%d lane=%d got=%d exp=%d\n", width,
                     delta, lane, h_up[lane], e_up);
              failures++;
            }
            if (h_down[lane] != e_down) {
              printf("FAIL shfl_down w=%d delta=%d lane=%d got=%d exp=%d\n",
                     width, delta, lane, h_down[lane], e_down);
              failures++;
            }
            if (h_xor[lane] != e_xor) {
              printf("FAIL shfl_xor w=%d mask=%d lane=%d got=%d exp=%d\n", width,
                     xm, lane, h_xor[lane], e_xor);
              failures++;
            }
          }
        }
      }
    }
  }

  hipFree(d_in);
  hipFree(d_shfl);
  hipFree(d_up);
  hipFree(d_down);
  hipFree(d_xor);

  if (tested == 0) {
    printf("FAILED! (no valid widths for warpSize=%d)\n", warp);
    return 1;
  }
  if (failures == 0) {
    printf("PASSED! (%d widths tested)\n", tested);
    return 0;
  }
  printf("FAILED! (%d mismatches)\n", failures);
  return 1;
}
