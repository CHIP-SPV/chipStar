# chipStar Libraries CI Test Report

**Date**: 2026-03-17
**Platform**: x86_64 Linux, Intel Arc A770 dGPU (15 GB)
**LLVM**: 22.0-native (22.1.0-rc3, in-tree SPIR-V backend)
**Backend**: Level0 (dgpu)
**Branch**: libraries-tests-ci

---

## Pass/Fail Summary

| Library | Passed | Total | Rate | Notes |
|---------|--------|-------|------|-------|
| rocPRIM | 29 | 48 | 60% | |
| hipCUB | 19 | 34 | 56% | 1 build failure |
| rocRAND | 41 | 45 | 91% | |
| hipRAND | 2 | 4 | 50% | |
| rocThrust | ~170 | 289 | ~59% | ~4 build failures |
| rocSPARSE | ~458 | ~1500+ | partial | axpby (432) + csr2coo (26) pass |
| hipSPARSE | partial | — | partial | axpyi (64) + bad_arg tests pass |
| hipMM | 42 | 54 | 78% | 12 build failures |
| H4I-MKLShim | 4 | 4 | 100% | conda oneAPI 2025.0.4 |
| H4I-HipBLAS | 4 | 4 | 100% | |
| H4I-HipSOLVER | 2 | 2 | 100% | |
| H4I-HipFFT | 2 | 2 | 100% | required handle bug fix |

---

## Root Cause Categories

### 1. `hipMallocAsync` not implemented
**Severity**: High — blocks most rocSPARSE and hipSPARSE tests
**Affected**: rocSPARSE (bsrmv, bsric0, bsrilu0, csric0, csrilu0, csrsv), hipSPARSE (most operations)
**Error**: `hipErrorNotSupported` at `CHIPBindings.cc:800`
**Fix**: Implement `hipMallocAsync` → `zeMemAllocDevice` with stream context

### 2. Subgroup ballot/shuffle operations
**Severity**: High — causes hangs and wrong results in all sort/scan primitives
**Affected**: rocPRIM (block_sort, device_radix_sort, block_scan, warp_reduce/scan), hipCUB (BlockRadixSort, DeviceRadixSort, DeviceSelect), rocThrust (sort_by_key)
**Error**: Timeouts (300s) or wrong numerical results
**Root cause**: LLVM 22 in-tree SPIR-V backend doesn't correctly emit `OpGroupNonUniformShuffleXor`, `OpGroupNonUniformBallot` for Intel Arc Level0

### 3. `sub_group_barrier()` / `mem_fence()` missing from rtdevlib
**Severity**: Medium — blocks sort operations in sparse libraries
**Affected**: rocSPARSE csrsort, hipSPARSE csrsort/coosort
**Error**: `clLinkProgram` error -17, "Missing definition for sub_group_barrier/mem_fence"
**Fix**: Add OpenCL 1.x built-ins to chipStar device library

### 4. `KeyValuePair` ReduceArgMin/ArgMax wrong results
**Severity**: Medium
**Affected**: rocPRIM device_reduce (arg variants), hipCUB DeviceReduce (arg variants)
**Error**: Returns index 0 instead of correct min/max index
**Root cause**: 64-bit struct comparison in SPIR-V produces wrong results for `KeyValuePair<int, T>`

### 5. `hipGraph` not implemented
**Severity**: Low — one test
**Affected**: rocRAND test_rocrand_hipgraphs
**Fix**: Implement hipGraph API (known limitation)

### 6. CUDA CUB headers missing for rocThrust CUDA compatibility layer
**Severity**: Low — build-time only
**Affected**: 12 hipMM tests (DEVICE_MR_REF, POOL_MR, HOST_MR_REF, PINNED_POOL_MR, THRUST_ALLOCATOR, DEVICE_BUFFER)
**Root cause**: rocThrust installs CUDA Thrust compatibility headers that transitively require CUDA CUB (`cub/util_namespace.cuh` etc.)
**Fix**: Install CUB stubs or patch hipMM cmake to exclude CUDA Thrust headers

---

## Per-Library Detail

### rocPRIM — 29/48 (60%)

Passing: block_adjacent_difference, block_discontinuity, block_exchange, block_histogram, block_load_store, block_sort (non-merge), block_run_length_decode, device_adjacent_difference, device_histogram, device_scan, device_find_first_of, device_memcpy, device_transform, device_nth_element, device_merge, device_adjacent_find, device_upper_lower_bound, intrinsics, iterator (most), temp_storage, texture_cache, warp_sort (bitonic only), zip_iterator

Failing: block_sort_merge/bitonic (timeout), device_radix_sort (timeout), device_segmented_radix_sort (timeout), warp_reduce/scan (wrong results), block_radix_rank/sort, block_reduce/scan (partial), device_reduce (arg variants), device_reduce_by_key, device_partition, device_select, device_merge_sort, device_run_length_encode, device_segmented_reduce, arg_index_iterator

### hipCUB — 19/34 (56%)

Build failure: test_hipcub_thread (`ThreadStore<STORE_CS>` — cache-hint store has no SPIR-V equivalent)

Passing: BlockAdjacentDifference, BlockDiscontinuity, BlockExchange, BlockHistogram, BlockLoad/Store, BlockMergeSort, BlockRunLengthDecode, BlockShuffle, DeviceAdjacentDifference, DeviceMergeSort, DeviceScan, DeviceTopK, WarpLoad/Store, WarpMergeSort, WarpShuffle (partial)

Failing: BlockRadixRank (timeout), BlockRadixSort, BlockScan (partial), DeviceHistogram, DeviceRadixSort, DeviceReduce (arg variants), DeviceReduceByKey, DeviceRunLengthEncode, DeviceSegmentedRadixSort (timeout), DeviceSegmentedReduce, DeviceSegmentedSort (timeout), DeviceSelect, DevicePartition, Grid

### rocRAND — 41/45 (91%)

Passing: All device-side generator tests (xorwow, mrg32k3a, philox, sobol, scrambled_sobol, lfsr113), host-side variants, statistics tests, ordering tests

Failing: test_rocrand_cpp_basic (timeout), test_rocrand_cpp_wrapper (timeout), test_rocrand_hipgraphs (hipGraph not implemented), test_rocrand_mt19937_prng (timeout — i127/i192 SPIR-V lowering)

### hipRAND — 2/4 (50%)

Passing: test_hiprand_kernel, test_hiprand_poisson

Failing: test_hiprand_api (timeout — exercises MT19937), test_hiprand_cpp_wrapper (timeout)

### rocThrust — ~170/289 (~59%)

Build failures (~4 tests): generate_const_iterators, partition, partition_point, is_partitioned — "Appending variables with different element types" when linking hipspv BC module

Runtime failures: adjacent_difference (partial), async_copy/reduce, transform_reduce (partial), sort_by_key (timeout for some configs), counting_iterator, unique, unique_by_key

### rocSPARSE — partial

Pass: axpby (432/432), csr2coo (26/26)
Fail: All tests using hipMallocAsync (bsrmv 992 cases, bsric0, bsrilu0, csric0, csrilu0, csrsv), csrsort (sub_group_barrier missing)

### hipSPARSE — partial

Pass: axpyi (64 + bad_arg), csr2coo_bad_arg, coosort_bad_arg, csrsort_bad_arg
Fail: csrsort, coosort (sub_group_barrier), bsrmv and most sparse operations (hipMallocAsync)

### hipMM — 42/54 (78%)

Pass: ARENA_MR, BINNING_MR, CUDA_ASYNC_VIEW_MR, DEVICE_ACCESSIBLE_RESOURCE, DEVICE_MR, FAILURE_CALLBACK_MR, FIXED_MULTISIZE_MR, FIXED_SIZE_MR, LOGGING_MR, MANAGED_MR, PREFETCH_RESOURCE, STATISTICS_MR, TRACKING_MR, UPSTREAM_RESOURCE_ADAPTOR, and more

Build failures (12): DEVICE_MR_REF, POOL_MR, HOST_MR_REF, PINNED_POOL_MR, THRUST_ALLOCATOR, DEVICE_BUFFER (×2 variants each) — CUDA CUB headers not available via rocThrust compatibility layer

### H4I-MKLShim — 4/4 (100%)

Built with conda oneAPI 2025.0.4 (`icpx` at `/space/pvelesko/install/oneapi/2025.0.4/bin/icpx`, MKL sequential threading). All 4 tests pass: context_test, batch_smoke_test, batch_correctness_test, handle_management_test.

### H4I-HipBLAS — 4/4 (100%)

All 4 tests pass: context_test, simple_batched_test, batched_correctness_test, basic_context_test.

Fix applied: `Scripts/install-hipblas-header.sh` used bash `(( ))` arithmetic in a `#!/bin/sh` script — changed to POSIX `[ "$#" -ne 3 ]`.

### H4I-HipSOLVER — 2/2 (100%)

Both tests pass: basic_context_test, context_test.

### H4I-HipFFT — 2/2 (100%)

Both tests pass: hipfft_real_1d, hipfft_complex_1d.

Fix applied to `src/hipfft.cpp`: `hipfftCreate()` was incorrectly shifting the native handles array and decrementing `nHandles` from 5 to 4 before calling `H4I::MKLShim::Create()`. `MKLShim::Create()` requires ≥5 handles with `handles[0]` = backend name pointer, causing "Error: Invalid handles". Removed the erroneous shift/decrement — the full array is now passed unchanged, matching the pattern used by H4I-HipBLAS.

---

## Priority Fixes

| Priority | Issue | Impact |
|----------|-------|--------|
| P1 | Implement `hipMallocAsync` | Unblocks majority of rocSPARSE + hipSPARSE |
| P1 | Fix subgroup shuffle/ballot emission in LLVM SPIR-V backend | Unblocks rocPRIM/hipCUB/rocThrust sort (most failures) |
| P2 | Add `sub_group_barrier`/`mem_fence` to rtdevlib | Unblocks rocSPARSE/hipSPARSE csrsort |
| P2 | Fix KeyValuePair arg-reduce in SPIR-V | Unblocks rocPRIM/hipCUB ReduceArgMin/ArgMax |
| P3 | ~~Install `icpx` / Intel oneAPI~~ | **DONE** — conda oneAPI 2025.0.4; H4I-* all 12/12 (100%) |
| P3 | Fix CUDA CUB stubs for rocThrust compat layer | Unblocks 12 hipMM build failures |
| P4 | Implement hipGraph | Unblocks rocRAND hipgraphs test |
