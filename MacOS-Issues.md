# macOS Build Issues

This document tracks known issues when building chipStar on macOS (specifically ARM64/Apple Silicon).

## Issue 1: OpenCL Version Mismatch Error

### Symptom
When running chipStar samples or tests with PoCL on macOS, you may encounter OpenCL version mismatch errors during ICD loading.

### Root Cause
The macOS OpenCL framework may conflict with the ocl-icd-loader, or PoCL may report a different OpenCL version than expected by the ICD loader.

### Workaround
- Ensure `OCL_ICD_VENDORS` points to the correct PoCL vendor directory
- Use the ocl-icd-loader explicitly: `-DOpenCL_LIBRARY=/path/to/ocl-icd-loader/lib/libOpenCL.dylib`
- Verify PoCL rpaths include the LLVM library path:
  ```bash
  install_name_tool -add_rpath "$(llvm-config --libdir)" /path/to/pocl/lib/libpocl.dylib
  ```

---

## Issue 2: LLVM Pass Plugin Linker Errors (Undefined Symbols)

### Symptom
When building chipStar with a **statically-linked LLVM** (the default for `configure_llvm.sh`), linking `libLLVMHipSpvPasses.so` fails with hundreds of undefined symbol errors:

```
Undefined symbols for architecture arm64:
  "llvm::BasicBlock::getFirstNonPHI()", referenced from: ...
  "llvm::Instruction::eraseFromParent()", referenced from: ...
  "llvm::Module::getFunction(llvm::StringRef) const", referenced from: ...
  ... (hundreds more)
ld: symbol(s) not found for architecture arm64
```

### Root Cause
On macOS, the linker is stricter than on Linux. MODULE libraries (pass plugins) must have all symbols resolved at link time.

The current CMake logic in `llvm_passes/CMakeLists.txt` only links against LLVM when building with shared LLVM libraries:

```cmake
if(APPLE AND LLVM_LINK_LLVM_DYLIB)
  target_link_libraries(LLVMHipPasses PRIVATE LLVM)
  ...
endif()
```

When LLVM is built statically (the default), `LLVM_LINK_LLVM_DYLIB` is false, so no LLVM libraries are linked. On Linux, this works because:
1. The plugin is loaded by clang/opt which already has the symbols
2. The linker allows unresolved symbols in shared objects by default

On macOS, neither of these apply - all symbols must be resolved.

### Solutions

**Option A: Build LLVM with shared libraries (recommended for macOS development)**

```bash
./scripts/configure_llvm.sh --version 21 --install-dir ~/install/llvm/21.0 --link-type dynamic
```

This sets `LLVM_LINK_LLVM_DYLIB=ON` and the existing CMake logic will link against `libLLVM.dylib`.

**Option B: Fix CMake to link static LLVM libs on macOS**

Modify `llvm_passes/CMakeLists.txt` to also link LLVM components when building static on macOS:

```cmake
# On macOS, MODULE libraries need explicit linking against LLVM
if(APPLE)
  if(LLVM_LINK_LLVM_DYLIB)
    target_link_libraries(LLVMHipPasses PRIVATE LLVM)
    target_link_libraries(LLVMHipDynMem PRIVATE LLVM)
    target_link_libraries(LLVMHipStripUsedIntrinsics PRIVATE LLVM)
    target_link_libraries(LLVMHipDefrost PRIVATE LLVM)
  else()
    # Static LLVM on macOS - need to link individual components
    llvm_map_components_to_libnames(LLVM_LIBS
      Core Support Analysis TransformUtils Scalar IPO Passes
      IRReader Linker Target AggressiveInstCombine InstCombine
      ScalarOpts Vectorize)
    target_link_libraries(LLVMHipPasses PRIVATE ${LLVM_LIBS})
    target_link_libraries(LLVMHipDynMem PRIVATE ${LLVM_LIBS})
    target_link_libraries(LLVMHipStripUsedIntrinsics PRIVATE ${LLVM_LIBS})
    target_link_libraries(LLVMHipDefrost PRIVATE ${LLVM_LIBS})
  endif()
endif()
```

**Option C: Use `-undefined dynamic_lookup` linker flag (not recommended)**

This tells the macOS linker to defer symbol resolution to load time (similar to Linux behavior):

```cmake
if(APPLE AND NOT LLVM_LINK_LLVM_DYLIB)
  set_target_properties(LLVMHipPasses PROPERTIES
    LINK_FLAGS "-undefined dynamic_lookup")
endif()
```

This is fragile and can mask real linking issues.

### Recommended Approach
For macOS development, use **Option A** (dynamic LLVM). The `configure_llvm.sh` script supports this via `--link-type dynamic`.

---

## General macOS Build Notes

### Required Dependencies
- Homebrew packages: `spirv-tools`, `cmake`, `ninja`
- Environment modules: `llvm/19.0` (or newer), `ocl-icd-loader`, `pocl`

### Build Command
```bash
cmake .. -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DCHIP_BUILD_SAMPLES=ON \
  -DCHIP_BUILD_TESTS=ON \
  -DOpenCL_LIBRARY=/path/to/ocl-icd-loader/lib/libOpenCL.dylib
```

### CI Configuration
The macOS ARM64 CI workflow (`.github/workflows/macos-arm64-ci.yml`) currently uses LLVM 19.0 with static linking. If upgrading to LLVM 21+, consider switching to dynamic linking or applying the CMake fix above.

---

## Issue 3: LLVM 21 Darwin Toolchain Crashes (macOS-specific)

When using LLVM 21 with HIP-SPIRV on macOS, the clang driver crashes with assertion failures like:

```
Assertion failed: (TargetInitialized && "Target not initialized!"), function isTargetWatchOSBased, file Darwin.h
```

### Root Cause
The Darwin toolchain in LLVM 21 assumes the target is always initialized when certain methods are called (e.g., `addClangWarningOptions`, `getSupportedSanitizers`). When Darwin is used as the host toolchain for HIP offloading (via HIPSPVToolChain), the target may not be fully initialized.

### Required Patches
Several patches to the LLVM source are needed:

1. **Darwin.cpp - addClangWarningOptions**: Add guard for uninitialized target
2. **Darwin.cpp - getSupportedSanitizers**: Add guard for uninitialized target  
3. **Darwin.cpp - CheckObjCARC**: Add guard for uninitialized target
4. **AlignedAllocation.h**: Handle unknown OS types instead of llvm_unreachable
5. **HIPSPV.cpp**: Don't delegate addClangTargetOptions to host (avoids -faligned-alloc-unavailable on SPIRV)
6. **CGCUDANV.cpp**: Use Mach-O section naming for HIP fatbin on macOS
7. **HIPUtility.cpp**: Use Mach-O section naming when generating fatbin assembly

These patches are in the llvm-project directory used for building LLVM 21.

---

## Issue 4: LLVM 21 HIP Fatbin Section Names (Mach-O)

### Symptom
```
fatal error: error in backend: Global variable '' has an invalid section specifier '.hip_fatbin': mach-o section specifier requires a segment and section separated by a comma.
```

### Root Cause
LLVM's HIP codegen uses ELF-style section names (`.hip_fatbin`, `.hipFatBinSegment`) which are invalid on macOS. Mach-O requires `segment,section` format.

### Fix
Patches to `CGCUDANV.cpp` and `HIPUtility.cpp` to use `__HIP,__hip_fatbin` format on macOS.

---

## Issue 5: Missing C++ Headers for SPIRV Device Compilation

### Symptom
```
fatal error: 'cstddef' file not found
```

### Root Cause
The SPIRV device compilation doesn't find standard library headers when the SDK path isn't set.

### Workaround
Pass the sysroot explicitly:
```bash
hipcc --sysroot=/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk myfile.cpp
```

Or set it in cmake/build system.

---

## Issue 6: Apple SDK TargetConditionals.h Error for SPIRV Targets

### Symptom
```
/Library/Developer/CommandLineTools/SDKs/MacOSX.sdk/usr/include/TargetConditionals.h:353:14: error: unrecognized arch using compiler with __is_target_arch support
```

### Root Cause
Apple's `TargetConditionals.h` uses the `__is_target_arch()` builtin to detect the target architecture. When compiling for SPIRV device targets, this builtin doesn't recognize SPIRV architectures and falls through to an `#error` directive.

### Fix
Two patches to `clang/lib/Lex/PPMacroExpansion.cpp`:

1. **Hide `__is_target_arch` builtin for SPIRV targets**: Make `__has_builtin(__is_target_arch)` return false when the target is spirv/spirv32/spirv64. This causes `TargetConditionals.h` to use fallback logic instead.

2. **Return false for CPU arch queries on SPIRV targets**: When `__is_target_arch(arm64)` etc. is called on a SPIRV target, return false gracefully instead of potentially hitting edge cases.

This patch is included in `llvm-patches/llvm-21-macos.patch`.

---

## Complete LLVM 21 macOS Patch

All LLVM 21 macOS compatibility issues are addressed in a single patch file:

```
llvm-patches/llvm-21-macos.patch
```

### Files Modified
1. `clang/lib/Lex/PPMacroExpansion.cpp` - Hide `__is_target_arch` for SPIRV
2. `clang/lib/Driver/ToolChains/Darwin.cpp` - Guards for uninitialized targets
3. `clang/include/clang/Basic/AlignedAllocation.h` - Handle unknown OS types
4. `clang/lib/Driver/ToolChains/HIPSPV.cpp` - Don't delegate host flags to device
5. `clang/lib/CodeGen/CGCUDANV.cpp` - Mach-O section naming for HIP
6. `clang/lib/Driver/ToolChains/HIPUtility.cpp` - Mach-O section naming in assembly
7. `llvm/lib/Frontend/Offloading/OffloadWrapper.cpp` - Hidden visibility for Darwin
8. `clang/lib/Basic/Targets/SPIR.h` - Data layout adjustments
9. `llvm/lib/Target/SPIRV/SPIRVTargetMachine.cpp` - Data layout adjustments  
10. `llvm/tools/llvm-link/llvm-link.cpp` - Archive linking improvements

### Applying the Patch
```bash
cd /path/to/llvm-project
git apply /path/to/chipStar/llvm-patches/llvm-21-macos.patch
```

### Branch
The patch is also available as a branch: `llvm-21-macos` in the local llvm-project directory.
