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
