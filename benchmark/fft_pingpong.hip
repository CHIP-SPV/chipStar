// shim-queue micro-benchmark.
//
// Times a FFT-heavy ping-pong on chipStar: alternating HIP kernels and
// hipFFT R2C / C2R transforms, mirroring the inner loop of GROMACS PME.
// Reports ms/iteration and FFTs/s. Intended to be run twice — once
// LD_PRELOADing libMKLShim.before.so (entry/exit/post-compute queue.wait()
// calls in MKLShim's exec path), once LD_PRELOADing libMKLShim.after.so
// (those waits removed). The host-fence cost is what we are measuring.
//
// Build:
//   /home/pvelesko/install/HIP/chipStar/2026.04.29/bin/hipcc \
//     -O2 \
//     -I/space/pvelesko/apps/gromacs/build-chipstar/h4i-hipfft/include \
//     -L/space/pvelesko/apps/gromacs/build-chipstar/h4i-hipfft/lib -lhipfft \
//     -L/space/pvelesko/apps/gromacs/build-chipstar/h4i-mklshim/lib -lMKLShim \
//     -Wl,-rpath,/space/pvelesko/apps/gromacs/build-chipstar/h4i-mklshim/lib:\
//                /space/pvelesko/apps/gromacs/build-chipstar/h4i-hipfft/lib \
//     -o fft_pingpong fft_pingpong.hip

#include <hip/hip_runtime.h>
#include <hipfft/hipfft.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define HIPCK(x) do { hipError_t err=(x); if (err != hipSuccess) { \
    fprintf(stderr,"HIP err %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(err)); std::exit(1);} } while(0)
#define FFTCK(x) do { hipfftResult r=(x); if (r != HIPFFT_SUCCESS) { \
    fprintf(stderr,"FFT err %s:%d rc=%d\n",__FILE__,__LINE__,(int)r); std::exit(1);} } while(0)

// Lightweight kernel; the goal is to keep the queue busy with chipStar work
// between FFT calls so any host-side queue.wait() inside MKLShim shows up
// as a serialization stall.
__global__ void touch_real(float *p, int n, int it) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = p[i] * 1.000001f + (float)(it & 0xff);
}

int main(int argc, char** argv) {
    int iters    = (argc > 1) ? std::atoi(argv[1]) : 200;
    int warmup   = (argc > 2) ? std::atoi(argv[2]) : 20;
    int N        = (argc > 3) ? std::atoi(argv[3]) : 200;       // grid edge
    const char* tag = (argc > 4) ? argv[4] : "run";

    int NX = N, NY = N, NZ = N;
    int NZc = NZ/2 + 1;
    size_t realCount    = (size_t)NX * NY * NZ;
    size_t complexCount = (size_t)NX * NY * NZc;

    HIPCK(hipSetDevice(0));
    hipStream_t stream;
    HIPCK(hipStreamCreate(&stream));

    float *d_real = nullptr;
    hipfftComplex *d_complex = nullptr;
    HIPCK(hipMalloc(&d_real,    realCount    * sizeof(float)));
    HIPCK(hipMalloc(&d_complex, complexCount * sizeof(hipfftComplex)));

    int n[3]       = { NX, NY, NZ };
    int inembed[3] = { NX, NY, NZ };
    int onembed[3] = { NX, NY, NZc };
    hipfftHandle planR2C, planC2R;
    FFTCK(hipfftPlanMany(&planR2C, 3, n,
                          inembed, 1, NX*NY*NZ,
                          onembed, 1, NX*NY*NZc,
                          HIPFFT_R2C, 1));
    FFTCK(hipfftPlanMany(&planC2R, 3, n,
                          onembed, 1, NX*NY*NZc,
                          inembed, 1, NX*NY*NZ,
                          HIPFFT_C2R, 1));
    FFTCK(hipfftSetStream(planR2C, stream));
    FFTCK(hipfftSetStream(planC2R, stream));

    int threads = 256;
    int blocksR = (int)((realCount + threads - 1) / threads);

    auto run_iter = [&](int it){
        // PP-like work
        hipLaunchKernelGGL(touch_real, dim3(blocksR), dim3(threads), 0, stream,
                           d_real, (int)realCount, it);
        // R2C
        hipfftExecR2C(planR2C, d_real, d_complex);
        // PME-solve-like
        hipLaunchKernelGGL(touch_real, dim3(blocksR), dim3(threads), 0, stream,
                           d_real, (int)realCount, it);
        // C2R
        hipfftExecC2R(planC2R, d_complex, d_real);
        // Update-like
        hipLaunchKernelGGL(touch_real, dim3(blocksR), dim3(threads), 0, stream,
                           d_real, (int)realCount, it);
    };

    // Warmup (JIT compile, plan settle)
    for (int it = 0; it < warmup; ++it) run_iter(it);
    HIPCK(hipStreamSynchronize(stream));

    // Timed loop
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int it = 0; it < iters; ++it) run_iter(it);
    HIPCK(hipStreamSynchronize(stream));
    auto t1 = std::chrono::high_resolution_clock::now();

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double per_iter_ms = ms / iters;
    double ffts_per_s  = (2.0 * iters) * 1000.0 / ms;   // 2 FFTs (R2C + C2R) per iter

    fprintf(stdout,
            "[%s] grid=%dx%dx%d iters=%d warmup=%d total=%.3f ms  "
            "per_iter=%.3f ms  ffts/s=%.1f\n",
            tag, NX, NY, NZ, iters, warmup, ms, per_iter_ms, ffts_per_s);

    FFTCK(hipfftDestroy(planR2C));
    FFTCK(hipfftDestroy(planC2R));
    HIPCK(hipFree(d_real));
    HIPCK(hipFree(d_complex));
    HIPCK(hipStreamDestroy(stream));
    return 0;
}
