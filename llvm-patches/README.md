# chipStar LLVM Patches

```
llvm-patches/
├── llvm/
│   ├── 0001-Allow-up-to-v1.2-SPIR-V-features.patch
│   ├── 0002-fix-SPIR-V-data-layout.patch
│   └── 0003-Unbundle-SDL.patch
├── spirv-translator/
│   ├── 0001-Use-fp_fast_mode-extension.patch
│   ├── 0002-Pretend-SPIR-ver-1.2.patch
│   ├── 0003-Fix-LoopMerge-error.patch
│   └── 0004-fix-blockMerge.patch
└── README.md
```

## Base Versions

- **LLVM:** 21.1.4 (commit `ab0074fe306f` - tag `llvmorg-21.1.4`)
- **SPIRV-LLVM-Translator:** `llvm_release_210` branch

## Applying Patches

### Automatic Application

The `configure_llvm.sh` script automatically applies all patches when setting up LLVM. No manual intervention needed.

### Manual Application

If you need to apply patches manually:

```bash
# For LLVM patches
cd llvm-project
git apply ../llvm-patches/llvm/*.patch

# For SPIRV-Translator patches
cd llvm/projects/SPIRV-LLVM-Translator
git apply ../../../../llvm-patches/spirv-translator/*.patch
```

Patches should be applied in order (they are numbered sequentially).

---

## LLVM Patches (3 patches)

### 1. Allow up to v1.2 SPIR-V features (0001)

**Commit:** `c8c774712a2a`  
**File:** `clang/lib/Driver/ToolChains/HIPSPV.cpp`  
**Purpose:** Enable SPIR-V 1.2 for warp-level primitives via subgroup extensions

**Why needed:**
- SPIR-V 1.2 required for subgroup operations used in RDC (Relocatable Device Code)
- SPIR-V 1.1 lacks features for proper shuffle/warp operations
- SPIR-V 1.3 would be preferred for standard shuffle ops, but not supported by targets yet

**Change:**
- Updates `--spirv-max-version=1.1` to `--spirv-max-version=1.2`

---

### 2. Fix SPIR-V data layout (0002)

**Commit:** `5ebbf7913a57`  
**Files:** `clang/lib/Basic/Targets/SPIR.h`, `llvm/lib/Target/SPIRV/SPIRVTargetMachine.cpp`  
**Purpose:** Fix bitcode linking data layout mismatches  
**Required for:** LLVM 20+ only (not needed for LLVM 17-19)

**Why needed:**
- Starting in LLVM 20, upstream includes `-n8:16:32:64` in SPIR-V data layouts
- chipStar's linking process requires consistent data layouts across modules
- When linking multiple bitcode files, mismatched data layouts cause: `error: linking module 'xxx.bc': Linking two modules of different data layouts`

**Change:**
- Removes `-n8:16:32:64` from SPIR-V data layout strings in both files
- This is **critical** for LLVM 20+ - without it, chipStar builds fail during bitcode linking

**Note:** This is a chipStar-specific workflow requirement, not an upstream LLVM bug. LLVM 17-19 don't have this in the data layout, so this patch can be skipped for those versions.

---

### 3. Unbundle SDL - Static Device Libraries (0003)

**Commit:** `ae0614de05ac`  
**File:** `clang/lib/Driver/ToolChains/HIPSPV.cpp`  
**Purpose:** Enable RDC linking with static libraries containing device code  
**Required for:** All LLVM versions

**Why needed:**
- Without this patch, static libraries (.a files) containing device bitcode cannot be properly linked
- The patch adds support for unbundling archives containing bitcode bundles
- Calls `AddStaticDeviceLibsLinking` helper function to handle static device library linking
- Critical for TestStaticLibRDC and issue #984

**Change:**
- Adds logic to unbundle and link static device libraries in HIPSPV toolchain
- Handles both `-l` library flags and direct `.a` file inputs
- Ensures proper ordering of device code linking

**Authors:** Paulius Velesko, Henry Linjamäki

---

## SPIRV-LLVM-Translator Patches (4 patches)

All patches are critical chipStar-specific fixes.

- **0001** - Use fp_fast_mode extension with the capability (#2028)
  - Compatibility fix for floating-point fast math mode
  - Author: Pekka Jääskeläinen

- **0002** - Pretend the SPIR ver needed by shuffles is 1.2
  - Allows CHIPSPV warp-level functions to compile with SPIR-V 1.2
  - In reality shuffles require v1.3, but Intel's v1.2 implementation works with extensions
  - **Critical for RDC support**
  - Author: Pekka Jääskeläinen

- **0003** - Fix LoopMerge error
  - Fixes OpLoopMerge instruction placement in SPIR-V generation
  - Ensures OpLoopMerge is second-to-last instruction before branch
  - Author: Paulius Velesko

- **0004** - fix blockMerge
  - Fixes block merging logic in SPIR-V writer
  - Correctly handles innermost loop detection for merge blocks
  - Author: Paulius Velesko
