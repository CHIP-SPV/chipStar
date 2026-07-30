# Profiling and Debugging chipStar Applications on Intel GPUs

Short reference for Intel GPUs with the oneAPI tools. Paths and module names
vary by installation.

## Flag cheat sheet

| Goal | Compile time (hipcc flags) | JIT time (environment) |
|---|---|---|
| CPU/host profiling | `-g -O2` (or `CMAKE_BUILD_TYPE=RelWithDebInfo`) | — |
| GPU kernel source attribution in VTune | `-fdebug-info-for-profiling -gline-tables-only` | — |
| In-kernel debugging, line info + breakpoints | `-g -O0` | — |
| In-kernel debugging, readable locals | `-g -O0` | `CHIP_JIT_FLAGS="-cl-opt-disable"` |

Device debug info requires LLVM built with chipStar's
`preserve-device-debug-info` patch (upstream from LLVM 24,
[llvm#210504](https://github.com/llvm/llvm-project/pull/210504));
`scripts/configure_llvm.sh` applies it for LLVM 20-22.

## Compile time vs JIT time

Kernels are compiled twice, and the two stages are controlled independently:

1. **Compile time (`hipcc`)** produces the SPIR-V embedded in your binary.
   `-g` puts debug info into it (NonSemantic.Shader.DebugInfo form);
   `-O0` additionally marks kernels `optnone`, which chipStar preserves into
   SPIR-V (`OptNoneINTEL`) when the module carries debug info.
2. **JIT time (module load)** is when the device compiler (e.g. IGC) turns
   that SPIR-V into ISA — and its optimizer runs *regardless of the `-O`
   level you compiled with*. JIT behavior is controlled only by the JIT
   flags chipStar passes to `clBuildProgram` / `zeModuleCreate`:

   - `CHIP_JIT_FLAGS` **appends** to chipStar's backend defaults. Use this
     for adding flags such as `-cl-opt-disable`.
   - `CHIP_JIT_FLAGS_OVERRIDE` **replaces** the defaults entirely. On the
     OpenCL backend the defaults (`-cl-kernel-arg-info -cl-std=CL3.0`) are
     required for kernel argument handling, so overriding there will break
     things unless you re-include them. On Level Zero the defaults are
     currently empty, making append and override equivalent.

   So: append (`CHIP_JIT_FLAGS`) unless you deliberately need to discard the
   defaults.

Consequence for debugging: `-g -O0` at compile time is necessary but not
sufficient for readable locals — without `-cl-opt-disable` at JIT time the
device compiler optimizes the kernel anyway and every local reads
`<optimized out>`.

## Profiling with VTune

Source VTune's environment script if there is no module for it:

```bash
source <oneapi-root>/vtune/latest/vtune-vars.sh
export SWIP_NULL_SOCKET=1     # avoids an end-of-collection segfault in older oneAPI; harmless otherwise
vtune -collect gpu-hotspots -r r-hot -- ./app
```

On clusters whose job prologue expects processes inside an mpiexec-launched
cgroup, wrap the command in the site's launcher (e.g.
`mpiexec -n 1 -ppn 1 <site-gpu-binding-script> vtune ...`) even for
single-rank runs; the symptom of skipping this is the job being killed within
seconds with no error message. The result directory may gain a hostname
suffix (`r-hot.<node>`); glob it when generating reports.

Common collections: `gpu-offload` (CPU vs GPU bound, transfers),
`gpu-hotspots` (per-kernel XVE/occupancy/memory), add
`-knob profiling-mode=source-analysis` for source-line attribution.
For a lightweight alternative, THAPI's `iprof ./app` gives Level Zero API
time and memory-traffic tables with no setup.

## Debugging kernels with gdb-oneapi

Four prerequisites, all required (validated in
[chipStar#1350](https://github.com/CHIP-SPV/chipStar/pull/1350)):

```bash
# 1. Enable EU debug on every card (writable by the job user, no root).
for f in /sys/class/drm/card*/prelim_enable_eu_debug; do echo 1 > "$f"; done
export ZET_ENABLE_PROGRAM_DEBUGGING=1

# 2. Point gdb-oneapi's auto-attach at the entry point chipStar actually calls.
export INTELGT_AUTO_ATTACH_HOOK="-qualified zeContextCreateEx"

# 3. Disable device-compiler optimization at module JIT so locals stay
#    readable (appends to chipStar's default JIT flags; see section above).
export CHIP_JIT_FLAGS="-cl-opt-disable"

# 4. Compile -g -O0, then debug; device breakpoints resolve at module load.
export CHIP_BE=level0
hipcc -g -O0 app.hip -o app
gdb-oneapi -ex "set breakpoint pending on" -ex "break app.hip:42" -ex run ./app
```

Failure signatures, in order of appearance: `hipErrorNotInitialized` under the
debugger means step 1 is missing; breakpoints stay pending while the program
runs to completion means step 2; breakpoints hit but every local reads
`<optimized out>` means step 3 (or a toolchain without the `-g` patch).
`INTELGT_AUTO_ATTACH_VERBOSE_LOG=1` traces the attach; a working session logs
`gdbserver-ze started for process N`. Per-lane state is available via
`thread :N`.
