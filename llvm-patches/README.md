# chipStar LLVM Patches

```
llvm-patches/
├── llvm/
│   ├── 0001-Allow-up-to-v1.2-SPIR-V-features.patch
│   ├── 0001-0004-spirv-version-and-extensions-llvm22.patch  (LLVM 22+ only)
│   ├── 0002-fix-SPIR-V-data-layout.patch                    (LLVM 20 only)
│   ├── 0002-fix-SPIR-V-data-layout-llvm21.patch             (LLVM 21 only)
│   ├── 0003-Unbundle-SDL.patch                               (LLVM 17-21, upstream in 22+)
│   ├── 0004-only-necessary-exts.patch                        (LLVM 17-21)
│   ├── 0005-fix-archive-data-layout.patch
│   ├── 0006-fix-macos-hip-spirv.patch                        (LLVM 17-21)
│   └── 0006-macos-hip-spirv-llvm22.patch                     (LLVM 22+ only)
├── spirv-translator/
│   ├── 0001-Use-fp_fast_mode-extension.patch
│   ├── 0002-Pretend-SPIR-ver-1.2.patch
│   ├── 0002-Pretend-the-SPIR-ver-needed-by-shuffles-is-1.2-llvm17-18.patch
│   ├── 0003-Fix-LoopMerge-error.patch                        (LLVM 17-21, upstream in 22+)
│   └── 0004-fix-blockMerge.patch                              (LLVM 17-21, upstream in 22+)
└── README.md
```

## Supported Versions

| LLVM Version | Status |
|---|---|
| 17-19 | Supported |
| 20 | Supported |
| 21 | Supported |
| 22 | Supported (22.1.0-rc3+) |

## Applying Patches

### Automatic Application

The `scripts/configure_llvm.sh` script automatically applies the correct patches for each LLVM version. No manual intervention needed.

```bash
scripts/configure_llvm.sh --version 22 --install-dir /path/to/install
```

### Manual Application

If you need to apply patches manually, see which patches apply to your version in the table below.

---

## LLVM Patches

### 1. Allow up to v1.2 SPIR-V features (0001)

**File:** `clang/lib/Driver/ToolChains/HIPSPV.cpp`
**Versions:** LLVM 17-21 (use `0001-0004-*-llvm22.patch` for 22+)
**Purpose:** Enable SPIR-V 1.2 for warp-level primitives via subgroup extensions

### 2. Fix SPIR-V data layout (0002)

**Files:** `clang/lib/Basic/Targets/SPIR.h`, `llvm/lib/Target/SPIRV/SPIRVTargetMachine.cpp`
**Versions:** LLVM 20-21 only (not needed for 17-19 or 22+, fixed at chipStar level)
**Purpose:** Fix bitcode linking data layout mismatches (`-n8:16:32:64` removal)

### 3. Unbundle SDL - Static Device Libraries (0003)

**File:** `clang/lib/Driver/ToolChains/HIPSPV.cpp`
**Versions:** LLVM 17-21 (upstream in 22+, commit `ae0614de05ac`)
**Purpose:** Enable RDC linking with static libraries containing device code

### 4. Restrict SPIR-V extensions (0004)

**File:** `clang/lib/Driver/ToolChains/HIPSPV.cpp`
**Versions:** LLVM 17-21 (use `0001-0004-*-llvm22.patch` for 22+)
**Purpose:** Replace `--spirv-ext=+all` with only required extensions

### 5. Fix archive data layout (0005)

**File:** `llvm/tools/llvm-link/llvm-link.cpp`
**Versions:** All
**Purpose:** Fix llvm-link creating empty "ArchiveModule" with wrong data layout

### 6. macOS HIP SPIR-V support (0006)

**Files:** `clang/lib/CodeGen/CGCUDANV.cpp`, `clang/lib/CodeGen/HIPUtility.cpp`, `clang/lib/Driver/ToolChains/Darwin.cpp`, `clang/lib/Driver/ToolChains/Darwin.h`, `clang/lib/Driver/ToolChains/HIPSPV.cpp`
**Versions:** LLVM 17-21 use `0006-fix-macos-hip-spirv.patch`; LLVM 22+ use `0006-macos-hip-spirv-llvm22.patch`
**Purpose:** Support HIP SPIR-V compilation on macOS (Mach-O sections, Darwin target init, skip host stdlib for device)

---

## SPIRV-LLVM-Translator Patches

- **0001** - fp_fast_mode extension fix (all versions)
- **0002** - Shuffle version requirement to 1.2 (all versions, version-specific patches for 17/18)
- **0003** - Fix LoopMerge error (LLVM 17-21, upstream in 22+)
- **0004** - Fix blockMerge (LLVM 17-21, upstream in 22+)

## LLVM 22 Patch Summary

For LLVM 22, only these patches are applied:
- `0001-0004-spirv-version-and-extensions-llvm22.patch` (combined SPIR-V version + extensions)
- `0005-fix-archive-data-layout.patch` (llvm-link fix)
- `0006-macos-hip-spirv-llvm22.patch` (macOS support)
- Translator: `0001` (fp_fast_mode) and `0002` (shuffle version)
