# shim-queue: drop redundant host fences in H4I-MKLShim FFT exec path

## Problem

`H4I-MKLShim/src/onemklfft.cpp`'s `fftExecR2C`, `fftExecC2R`, `fftExecC2Cforward`,
`fftExecC2Cbackward` each contained ~6 `ctxt->queue.wait()` calls per single
FFT operation:

1. wait on entry (drain anything pending)
2. wait after `get_value(PLACEMENT)`
3. wait after `set_value(PLACEMENT)` (cold path, placement change only)
4. wait after `commit(queue)` (cold path)
5. wait after `compute_forward / compute_backward`
6. wait on exit

These were added as defensive sync points back when MKLShim created its own
SYCL queue independent of chipStar's HIP stream. With the queue truly
independent, ordering between MKL submissions and surrounding chipStar HIP
kernels could only be enforced via host fences — hence the `wait()`s.

## Why the waits are redundant today

`H4I-MKLShim/src/Context.cpp:Update()` already wraps the **native L0 handles
that chipStar provides via `hipGetBackendNativeHandles`**:

- For Level Zero (6 handles): wraps chipStar's `ZeCmdListImm_` (immediate
  command list) as a SYCL queue with `KeepOwnership=true` and the
  `in_order` property.
- For OpenCL (5 handles): wraps chipStar's `cl_command_queue`.

Since chipStar 2026.04.x:
- chipStar L0 backend submits all HIP kernels to `ZeCmdListImm_` (its
  immediate command list).
- MKLShim's SYCL queue submits oneMKL DFT work onto **the same** immediate
  command list (or onto the same `cl_command_queue` for OpenCL).

So MKL submissions and chipStar kernels land on the same in-order L0/CL
queue and are ordered at the driver level. **Host `wait()` fences between
them are pure overhead** — they serialize what was already correctly ordered.

## Fix

`shim-queue` commit `8b713cc shim-queue: remove redundant queue.wait() in
FFT exec hot path`:

- Remove entry, post-`get_value`, post-`compute_*`, and exit `wait()`s in
  the four exec functions.
- Keep the `wait()` after `commit(queue)` in the cold placement-change path,
  since plan rebuild may JIT and the next `compute_*` call must see the
  freshly-committed plan.
- Eliminate `std::cout` debug noise in the placement-change branches at the
  same time (purely cosmetic).

Plan-construction `wait()`s (descriptor constructors) are kept; plan
construction is one-shot per FFT plan, not on the hot path.

## Why this still works for both backends

The same Context-creation code already handles both paths via
`#if HAS_UR_API` (MKL 2025) and `#elif __INTEL_LLVM_COMPILER >= 20240000`
(MKL 2024) — both branches call `make_queue(KeepOwnership=true,
in_order)`. Tested on Intel Arc A770 with chipStar 2026.04.29 and oneAPI
2025.3.2 in both `CHIP_BE=level0` and `CHIP_BE=opencl`.

## What was *not* changed

- BLAS (`onemklblas.cpp`) and Solver (`onemklsolver.cpp`) wait patterns:
  these are lighter and out of scope for this change. Same audit could be
  applied later.
- `fftDescriptorSR/SC/DR/DC` constructors: still call `commit + wait()`.
  Plan creation is rare, not perf-critical.
- The `__FORCE_MKL_FLUSH__` macro state: orthogonal to this change.
- `Context.h::chipstar_cmd_list` field: was added in earlier WIP for a
  different cross-queue idea but is no longer needed and stays unused.

## Repository layout for this work

- `H4I-MKLShim` branch `shim-queue` at `/home/pvelesko/H4I-MKLShim`
  (commit `8b713cc`).
- This chipStar worktree `~/chipStar/shim-queue` (branch `shim-queue`,
  no chipStar source changes — just docs and the benchmark).
- Snapshot libs at `benchmark/libMKLShim.{before,after}.so` for A/B
  reproduction without rebuilding.

See `RESULTS.md` for measured numbers.
