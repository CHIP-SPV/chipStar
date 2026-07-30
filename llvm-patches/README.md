# chipStar LLVM Patches

chipStar builds its compiler toolchain from upstream LLVM release branches plus
a small set of patches, kept in one directory per supported LLVM version:

```
llvm-patches/
├── llvm-20/
│   ├── llvm/              patches applied in the llvm-project checkout
│   └── spirv-translator/  patches applied in the SPIRV-LLVM-Translator checkout
├── llvm-21/
│   ├── llvm/
│   └── spirv-translator/
└── llvm-22/
    ├── llvm/
    └── spirv-translator/
```

`scripts/configure_llvm.sh --version <20|21|22>` clones the matching upstream
branches (`release/<version>.x` and `llvm_release_<version>0`) and applies
every patch in the version's directory, in lexicographic (numeric) order, with
`git apply`. There is no per-patch version gating: everything in a version
directory applies to that version, and a failed patch is a hard error.

`--version latest` (experimental) is different: it clones the maintained
branch `chipStar-llvm-23` from
[CHIP-SPV/llvm-project](https://github.com/CHIP-SPV/llvm-project) together
with the translator branch `llvm_release_230`, and applies no patches. That
branch carries the chipStar changes directly; patch directories exist only for
the release-pinned versions.

## Supported Versions

| LLVM Version | Source | Patches |
|---|---|---|
| 20 | `llvm/llvm-project` `release/20.x` | `llvm-patches/llvm-20/` |
| 21 | `llvm/llvm-project` `release/21.x` | `llvm-patches/llvm-21/` |
| 22 | `llvm/llvm-project` `release/22.x` | `llvm-patches/llvm-22/` |
| latest (experimental) | `CHIP-SPV/llvm-project` `chipStar-llvm-23` | none |

LLVM 17, 18, and 19 support was dropped.

## llvm-20

### llvm/

| Patch | Purpose | Upstream status |
|---|---|---|
| 0001-spirv-version-and-extensions | Enable SPIR-V 1.2 (warp-level primitives via subgroup extensions) and restrict `--spirv-ext` to only the required extensions | Upstreamed in LLVM 23+ behind `Triple::ChipStar` ([llvm#179902](https://github.com/llvm/llvm-project/pull/179902)) |
| 0002-preserve-device-debug-info | Keep debug info intact through the HIP SPIR-V device pipeline | Merged upstream ([llvm#210504](https://github.com/llvm/llvm-project/pull/210504)), ships in LLVM 24; upstream additionally adds SPV_INTEL_optnone, which the local patch deliberately omits for now |
| 0003-unbundle-static-device-libraries | Enable RDC linking with static libraries containing device code | Upstream in LLVM 22+ ([llvm#136412](https://github.com/llvm/llvm-project/pull/136412), commit `ae0614de05ac`) |
| 0004-fix-spirv-data-layout | Revert the `-n8:16:32:64` data layout change to avoid bitcode linking mismatches | chipStar-local revert of [llvm#110695](https://github.com/llvm/llvm-project/pull/110695), not upstreamable |
| 0005-macos-hip-spirv | HIP SPIR-V compilation on macOS (Mach-O sections, Darwin toolchain guards, skip host stdlib for device) | Upstreamed via [llvm#183991](https://github.com/llvm/llvm-project/pull/183991) + [llvm#206902](https://github.com/llvm/llvm-project/pull/206902) |

### spirv-translator/

| Patch | Purpose | Upstream status |
|---|---|---|
| 0001-pretend-subgroup-caps-are-spirv-1.2 | Report subgroup shuffle capabilities as requiring SPIR-V 1.2 instead of 1.3 | Deliberate spec deviation, permanent |
| 0002-fix-loop-merge-placement | Fix LoopMerge instruction placement | Upstream in translator 220+ ([KhronosGroup#3277](https://github.com/KhronosGroup/SPIRV-LLVM-Translator/pull/3277)) |
| 0003-fix-block-merge-innermost-loop | Fix block merging in innermost loops | Upstream in translator 220+ ([KhronosGroup#3280](https://github.com/KhronosGroup/SPIRV-LLVM-Translator/pull/3280)) |
| 0004-coalesce-duplicate-phi-predecessors | Coalesce duplicate phi predecessors during translation | Pending upstream ([KhronosGroup#3866](https://github.com/KhronosGroup/SPIRV-LLVM-Translator/pull/3866)) |

## llvm-21

Same patch set as llvm-20 (rebased onto `release/21.x`):

### llvm/

| Patch | Purpose | Upstream status |
|---|---|---|
| 0001-spirv-version-and-extensions | As in llvm-20 | Upstreamed in LLVM 23+ behind `Triple::ChipStar` ([llvm#179902](https://github.com/llvm/llvm-project/pull/179902)) |
| 0002-preserve-device-debug-info | As in llvm-20 | Merged upstream ([llvm#210504](https://github.com/llvm/llvm-project/pull/210504)), ships in LLVM 24; SPV_INTEL_optnone deliberately omitted locally |
| 0003-unbundle-static-device-libraries | As in llvm-20 | Upstream in LLVM 22+ ([llvm#136412](https://github.com/llvm/llvm-project/pull/136412), `ae0614de05ac`) |
| 0004-fix-spirv-data-layout | As in llvm-20 | chipStar-local revert of [llvm#110695](https://github.com/llvm/llvm-project/pull/110695), not upstreamable |
| 0005-macos-hip-spirv | As in llvm-20 | Upstreamed via [llvm#183991](https://github.com/llvm/llvm-project/pull/183991) + [llvm#206902](https://github.com/llvm/llvm-project/pull/206902) |

### spirv-translator/

| Patch | Purpose | Upstream status |
|---|---|---|
| 0001-pretend-subgroup-caps-are-spirv-1.2 | As in llvm-20 | Deliberate spec deviation, permanent |
| 0002-fix-loop-merge-placement | As in llvm-20 | Upstream in translator 220+ ([KhronosGroup#3277](https://github.com/KhronosGroup/SPIRV-LLVM-Translator/pull/3277)) |
| 0003-fix-block-merge-innermost-loop | As in llvm-20 | Upstream in translator 220+ ([KhronosGroup#3280](https://github.com/KhronosGroup/SPIRV-LLVM-Translator/pull/3280)) |
| 0004-coalesce-duplicate-phi-predecessors | As in llvm-20 | Pending upstream ([KhronosGroup#3866](https://github.com/KhronosGroup/SPIRV-LLVM-Translator/pull/3866)) |

## llvm-22

Smaller set: the unbundle-SDL fix is already upstream in LLVM 22, the data
layout revert is no longer needed, and the loop/block merge fixes are already
upstream in translator 220+.

### llvm/

| Patch | Purpose | Upstream status |
|---|---|---|
| 0001-spirv-version-and-extensions | As in llvm-20 | Upstreamed in LLVM 23+ behind `Triple::ChipStar` ([llvm#179902](https://github.com/llvm/llvm-project/pull/179902)) |
| 0002-preserve-device-debug-info | As in llvm-20 | Merged upstream ([llvm#210504](https://github.com/llvm/llvm-project/pull/210504)), ships in LLVM 24; SPV_INTEL_optnone deliberately omitted locally |
| 0003-macos-hip-spirv | As in llvm-20 | Upstreamed via [llvm#183991](https://github.com/llvm/llvm-project/pull/183991) + [llvm#206902](https://github.com/llvm/llvm-project/pull/206902) |

### spirv-translator/

| Patch | Purpose | Upstream status |
|---|---|---|
| 0001-pretend-subgroup-caps-are-spirv-1.2 | As in llvm-20 | Deliberate spec deviation, permanent |
| 0002-coalesce-duplicate-phi-predecessors | As in llvm-20 | Pending upstream ([KhronosGroup#3866](https://github.com/KhronosGroup/SPIRV-LLVM-Translator/pull/3866)) |

## Removed in the layout change

- **archive-data-layout patch** (`llvm-link` empty "ArchiveModule" data layout
  fix): deleted; a no-op versus the upstream IRMover behavior.
- **fp_fast_mode test patch** (translator): deleted; a no-op.
- **LLVM 17/18/19 support** and their version-specific patch variants were
  dropped.
