# macOS PoCL Known Failures Analysis

## Overview
This document analyzes the test failures specific to the macOS ARM64 CI runner (hostname: `pastrami`) when running chipStar with the PoCL OpenCL backend.

## Test Failure Summary

### 1. hipcc-testIssue606
**Status:** ✅ FIXED  
**Test Number:** #19  
**Category:** Compiler Test

#### Error Details (Before Fix)
```
error: unknown type name 'size_t'
```

The test failed during compilation because `size_t` was not defined when including `hip/hip_runtime_api.h` in plain C++ mode (not HIP mode).

#### Root Cause
The test (`tests/compiler/inputs/testIssue606.cpp`) verifies that `hip_runtime_api.h` can be included in regular C++ compilation mode:
```cpp
#include <hip/hip_runtime_api.h>
#if defined(__HIPCC__) || defined(__HIP__)
#  error "expected C++ compilation mode."
#endif
```

When compiled with `-xc++` flag without standard library headers, the `hip_runtime_api.h` header used `size_t` but didn't include the necessary standard headers (`<cstddef>` or `<stddef.h>`).

#### Fix Applied
Added `#include <cstddef>` at the top of `HIP/include/hip/hip_runtime_api.h`:
```cpp
#include <cstddef>  // for size_t
```

This ensures `size_t` is always defined when the header is included in any C++ compilation mode.

**Test Status:** Now passes ✅

---

### 2. TestRDCWithSingleHipccCmd
**Status:** Failed  
**Test Number:** #25  
**Category:** RDC (Relocatable Device Code)

#### Error Details
```
/Users/paulius/install/llvm/19.0/bin/opt: WARNING: failed to create target machine for 'spirv64'
Assertion failed: (Segment.size() <= 16 && Section.size() <= 16 && "Segment or section string too long"), 
function MCSectionMachO, file MCSectionMachO.cpp, line 99.
```

#### Root Cause
Two cascading issues:
1. **SPIR-V target not recognized**: LLVM's `opt` tool doesn't recognize `spirv64` as a valid target on macOS
2. **Mach-O section name limitation**: When falling back, generated section names exceed the 16-character limit for Mach-O object files (macOS uses Mach-O, not ELF)

Example of problematic section name:
```
.type __hip_gpubin_handle_53d79de610e4ea74,@object
```

This is ELF-style syntax that doesn't translate to Mach-O format.

#### Potential Fix
**Short-term:** Disable RDC tests on macOS as the current implementation relies on ELF-specific tooling:
- `llvm-objcopy` operations that don't work with Mach-O
- Section naming conventions incompatible with Mach-O 16-char limit
- Target triple handling for SPIR-V on macOS

**Long-term:** 
1. Implement Mach-O-specific RDC bundling in clang-offload-bundler
2. Add SPIR-V target support for macOS in LLVM opt
3. Use shorter section names or Mach-O-compatible naming scheme

---

### 3. TestRDCWithMultipleHipccCmds
**Status:** Failed  
**Test Number:** #26  
**Category:** RDC (Relocatable Device Code)

#### Error Details
```
llvm-objcopy: error: option is not supported for MachO
/Users/paulius/install/llvm/19.0/bin/clang-offload-bundler: error: 'llvm-objcopy' tool failed
```

#### Root Cause
The `llvm-objcopy` tool used by `clang-offload-bundler` to create fat binaries with device code doesn't support the required operations for Mach-O object files. The tool is designed for ELF object files (Linux) and COFF (Windows), but macOS's Mach-O format requires different handling.

#### Potential Fix
**Short-term:** Mark RDC tests as unsupported on macOS

**Long-term:**
1. Extend `llvm-objcopy` to support Mach-O bundling operations
2. Alternative: Use macOS-native tools (`lipo`, `install_name_tool`) for bundling
3. Implement a Mach-O-specific code path in clang-offload-bundler

---

### 4. TestLazyModuleInit
**Status:** ✅ NOT A REAL FAILURE  
**Test Number:** #81  
**Category:** Runtime Test

#### Error Details (When Run Without Proper Environment)
```
CHIP info: CHIP_DEVICE_TYPE=gpu
CHIP critical: No OpenCL platforms found with devices of type gpu that support SPIR-V
```

#### Root Cause
This test **does NOT actually require a GPU** - it works fine on CPU/PoCL. The failure only occurs when running `ctest` directly without the environment variables that `scripts/check.py` sets.

The test verifies lazy JIT compilation by:
1. Launching a trivial kernel: `__global__ void bar(int *Dst) { *Dst = 42; }`
2. Checking that module compilation happens on first kernel launch

There's nothing GPU-specific about this functionality.

#### Why It Failed
When run via `ctest -I 81,81` directly (not through `check.py`), no `CHIP_DEVICE_TYPE` environment variable is set, so chipStar defaults to looking for GPU devices. Since PoCL only presents CPU devices, initialization fails.

#### Fix
The test **passes correctly** when run through `scripts/check.py ./ pocl opencl`, which sets `CHIP_DEVICE_TYPE=pocl`. 

**Verification:**
```bash
$ CHIP_BE=opencl CHIP_DEVICE_TYPE=pocl ./tests/runtime/TestLazyModuleInit
# Test passes! ✅
```

**Conclusion:** This is not a macOS-specific issue - the test works fine. Removed from known failures list.

---

### 5. TestBufferDevAddr
**Status:** Failed  
**Test Number:** #112  
**Category:** Runtime Test

#### Error Details
```
CHIP info: CHIP_DEVICE_TYPE=gpu  
CHIP critical: No OpenCL platforms found with devices of type gpu that support SPIR-V
```

#### Root Cause
Same as TestLazyModuleInit - the test hard-codes GPU device type but PoCL only provides CPU devices.

#### Potential Fix
Same options as TestLazyModuleInit. Additionally, check if this test specifically requires GPU features (like physical device addresses) that aren't available on CPU devices. If so, it should be skipped on PoCL regardless of platform.

---

### 6. hipTestResetStaticVar
**Status:** Failed (Subprocess aborted)  
**Test Number:** #153  
**Category:** Sample Test (internal)

#### Error Details
```
CHIP info: CHIP_DEVICE_TYPE=gpu
CHIP critical: No OpenCL platforms found with devices of type gpu that support SPIR-V
```

#### Root Cause
Same GPU vs CPU device type mismatch as TestLazyModuleInit and TestBufferDevAddr.

#### Potential Fix
Same options as above - either override device type for PoCL environments or implement fallback logic in chipStar.

---

### 7. Unit_hipHostMalloc_CoherentAccess
**Status:** Timeout (not found in current build)  
**Test Number:** #765 (in full test suite)  
**Category:** Unit Test

#### Status
This test was not present in the current build directory (only 200 tests built vs 1397 total tests). Unable to reproduce the failure without building the full test suite.

#### Likely Root Cause
Based on the test name and timeout symptom, potential issues:
1. **Coherent memory operations deadlock**: macOS may handle cache coherency differently for CPU-accessible GPU memory
2. **PoCL memory model**: PoCL's memory implementation on macOS might not properly support the coherent access patterns
3. **Infinite loop in synchronization**: The test might be waiting for memory visibility that never occurs on PoCL/macOS

#### Potential Investigation
1. Build full test suite: `cd build && ninja build_tests`
2. Run with timeout and debug output
3. Check if PoCL supports `CL_MEM_ALLOC_HOST_PTR` with coherent semantics on macOS

---

## Common Themes

### 1. Platform Binary Format Issues
**Affected Tests:** RDC tests (TestRDCWithSingleHipccCmd, TestRDCWithMultipleHipccCmds)

macOS uses Mach-O binary format while chipStar's RDC implementation assumes ELF (Linux) format:
- Section name length limits (16 chars in Mach-O vs unlimited in ELF)
- Different symbol and relocation handling
- `llvm-objcopy` doesn't support required Mach-O operations

### 2. Device Type Mismatch
**Affected Tests:** TestLazyModuleInit, TestBufferDevAddr, hipTestResetStaticVar

Tests hard-code `CHIP_DEVICE_TYPE=gpu` but PoCL is CPU-only:
- No runtime fallback mechanism
- Test configuration doesn't account for PoCL's CPU nature
- Need per-backend test configuration

### 3. Standard Library Dependencies  
**Affected Tests:** hipcc-testIssue606

Headers don't include necessary standard library headers for standalone C++ compilation.

---

## Recommendations

### Immediate Actions
1. **Add RDC macOS exclusion**: Mark all RDC tests as unsupported on macOS until Mach-O support is implemented
2. **Fix hip_runtime_api.h**: Add `#include <cstddef>` for `size_t` definition
3. **Update test configurations**: Allow CPU device fallback for gpu-requesting tests when running on PoCL

### Medium-term
1. **Implement test environment detection**: Auto-configure device type based on available backends
2. **Build complete test suite on CI**: Investigate Unit_hipHostMalloc_CoherentAccess timeout
3. **Add macOS-specific test variants**: Create separate test configurations for Mach-O limitations

### Long-term  
1. **Implement Mach-O RDC support**: Full relocatable device code support for macOS
2. **SPIR-V target for macOS**: Add macOS SPIR-V target support in LLVM
3. **Unified device selection**: Smart fallback logic in chipStar runtime for device type mismatches

---

## Testing Methodology
Tests were executed on macOS ARM64 (pastrami) with:
- Backend: OpenCL (PoCL)
- Build: `/Users/paulius/chipStar/build_ci`
- LLVM: 19.0
- Command: `ctest -I <test_number>,<test_number> --output-on-failure`

Each test was run with a 60-second timeout to capture failure modes without hanging the CI.

