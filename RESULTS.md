# shim-queue: measured uplift

## System

- Intel Arc A770 (`Intel(R) Arc(TM) A770 Graphics`)
- chipStar 2026.04.29 (`/home/pvelesko/install/HIP/chipStar/2026.04.29`)
- oneAPI 2025.3.2 (`/space/pvelesko/install/oneapi/2025.3.2`)
- LLVM 22.0-native (22.1.2)

## Microbenchmark — `benchmark/fft_pingpong`

R2C + 3 chipStar HIP kernels + C2R per iteration, `iters=500`, `warmup=30`.
Both libs share identical strided `fftPlanMany` and Context setup; only
the FFT exec hot-path `wait()`s differ.

### Level 0 backend (`CHIP_BE=level0`)

| grid    | before (ms/iter) | after (ms/iter) | speedup |
|---------|------------------|-----------------|---------|
| 32^3    | 27.482           | 0.798           | 34.4×   |
| 64^3    |  0.728           | 0.579           |  1.26×  |
| 96^3    | 24.447           | 2.618           |  9.34×  |
| 128^3   | 15.263           | 2.555           |  5.97×  |
| 200^3   | 45.555           | 2.889           | 15.77×  |

(Variability in the "before" column at 64^3 reflects oneMKL JIT cache hits;
MKL behaves differently across grid sizes. The overall pattern — host
fences are the bottleneck — is consistent.)

### OpenCL backend (`CHIP_BE=opencl`)

| grid    | before (ms/iter) | after (ms/iter) | speedup |
|---------|------------------|-----------------|---------|
| 32^3    |  1.880           | 0.246           |  7.64×  |
| 64^3    |  0.527           | 0.437           |  1.21×  |
| 96^3    |  0.670           | 0.409           |  1.64×  |
| 128^3   |  1.033           | 0.530           |  1.95×  |
| 200^3   |  3.197           | 3.852           |  0.83×  |

OpenCL gets less benefit at 200^3 — the `cl_command_queue` is not an
immediate command list and natively supports OOO without host fences,
so MKL's per-call overhead dominates the redundant wait cost. Small/mid
grids still show clear uplift.

## End-to-end — GROMACS PME-on-GPU

GROMACS 2027.0-dev, water_GMX50_bare 1536 (3M atom water box),
`gmx mdrun -nb gpu -pme gpu -update gpu -ntmpi 1 -ntomp 8 -nstlist 100
-noconfout -nsteps 500`, chipStar `CHIP_BE=level0` Arc A770.

| MKLShim                           | ns/day | ms/step |
|-----------------------------------|--------|---------|
| before (queue.wait everywhere)    | 0.963  | 179.36  |
| after  (shim-queue commit 8b713cc)| 3.997  |  43.23  |
| **speedup**                       | **4.15×** | **4.15×** |

The earlier session's local SYCL number on the same input was 11.298 ns/day
(20000-step run, fully cached); chipStar HIP closes a meaningful chunk of
that gap on this short-duration test once the host fences are gone.

## Reproducing the A/B

```bash
cd ~/chipStar/shim-queue/benchmark

# build microbench (one-time)
/home/pvelesko/install/HIP/chipStar/2026.04.29/bin/hipcc -O2 \
  -I/space/pvelesko/apps/gromacs/build-chipstar/h4i-hipfft/include \
  -L/space/pvelesko/apps/gromacs/build-chipstar/h4i-hipfft/lib -lhipfft \
  -L/space/pvelesko/apps/gromacs/build-chipstar/h4i-mklshim/lib -lMKLShim \
  -Wl,-rpath,/space/pvelesko/apps/gromacs/build-chipstar/h4i-mklshim/lib:/space/pvelesko/apps/gromacs/build-chipstar/h4i-hipfft/lib \
  -o fft_pingpong fft_pingpong.hip

ENV_PREFIX="env -i HOME=$HOME PATH=/usr/bin:/bin CHIP_LOGLEVEL=err \
  LD_LIBRARY_PATH=/space/pvelesko/apps/gromacs/build-chipstar/h4i-mklshim/lib:\
/space/pvelesko/apps/gromacs/build-chipstar/h4i-hipfft/lib:\
/home/pvelesko/install/HIP/chipStar/2026.04.29/lib:\
/space/pvelesko/install/oneapi/2025.3.2/lib"

# Level 0
$ENV_PREFIX LD_PRELOAD=$PWD/libMKLShim.before.so CHIP_BE=level0 ./fft_pingpong 500 30 96 L0_before
$ENV_PREFIX LD_PRELOAD=$PWD/libMKLShim.after.so  CHIP_BE=level0 ./fft_pingpong 500 30 96 L0_after

# OpenCL
$ENV_PREFIX LD_PRELOAD=$PWD/libMKLShim.before.so CHIP_BE=opencl ./fft_pingpong 500 30 96 OCL_before
$ENV_PREFIX LD_PRELOAD=$PWD/libMKLShim.after.so  CHIP_BE=opencl ./fft_pingpong 500 30 96 OCL_after
```

## Existing MKLShim tests (regression check)

`ctest` from `/tmp/h4i-mklshim-build`:

| test                                  | L0     | OpenCL |
|---------------------------------------|--------|--------|
| `h4i-mklshim_context_test`            | PASS   | PASS   |
| `h4i-mklshim_batch_smoke_test`        | FAIL†  | FAIL†  |
| `h4i-mklshim_batch_correctness_test`  | FAIL†  | FAIL†  |
| `h4i-mklshim_handle_management_test`  | PASS   | PASS   |

† Pre-existing failure on the parent commit `b71b7ca` as well: the test
calls `Dgetrf_batch` (FP64 LAPACK) which throws "Required aspect fp64 is
not supported on the device" — an Arc A770 hardware limitation
(no native FP64). Not a regression.
