# Cross-building chipStar's LLVM 23 toolchain for aarch64 (salami)

salami is a Hardkernel ODROID-N2+ (6 slow cores, 3.6 GB RAM, Ubuntu 22.04
arm64, Mali-G52). A native LLVM build there is roughly 750 core-minutes and
would very likely die at link time for lack of RAM, so the toolchain is
cross-compiled on an x86_64 runner and copied over.

Everything here is host-agnostic: any `[self-hosted, Linux, X64]` runner with
docker can reproduce it from a clean chipStar checkout. Nothing is specific to
the machine that first built it.

## What is in here

| file | purpose |
| --- | --- |
| `Dockerfile` | Ubuntu 22.04 image with `g++-aarch64-linux-gnu`. 22.04 is load-bearing: its cross toolchain targets glibc 2.35 and gcc 11's libstdc++, which is exactly salami's userland, so no sysroot has to be copied off the target. |
| `aarch64-toolchain.cmake` | CMake cross toolchain file for that compiler. |
| `cross-build.sh` | Runs inside the container: fetches and patches the LLVM sources via `scripts/configure_llvm.sh --version 23 --source-only`, configures/builds the cross LLVM, then builds the openmp runtime standalone, and stages the install. |

`cross-build.sh` deliberately does **not** re-implement the source setup. The pinned
refs (`llvmorg-23.1.0-rc2`, SPIRV-LLVM-Translator `llvm_release_230`) and the
`llvm-patches/llvm-23/` series stay solely in `scripts/configure_llvm.sh`,
which grew a `--source-only` mode for this. What `cross-build.sh` does own is the
cmake configure, because the cross build genuinely differs from the native one:
an explicit `LLVM_TARGETS_TO_BUILD=AArch64;SPIRV` (host detection would pick
x86), `LLVM_HOST_TRIPLE`/`LLVM_DEFAULT_TARGET_TRIPLE`, a native tablegen
sub-build via `CROSS_TOOLCHAIN_FLAGS_NATIVE`, and no host-gcc rpath flags.

The openmp runtime is built standalone from `runtimes/` rather than with
`LLVM_ENABLE_RUNTIMES`: the in-tree mode compiles the runtime with the
freshly built clang, which is an aarch64 binary the x86 host cannot execute.

## Running it by hand

```bash
cd <chipStar checkout>
docker build -t chipstar-llvm-cross-aarch64 scripts/cross-aarch64
WORK=/space/$USER/llvm-cross-build   # any dir with ~40 GB free (~12 GB used)
mkdir -p "$WORK"
docker run --rm \
  --user "$(id -u):$(id -g)" -e HOME=/work \
  -v "$PWD":/chipstar:ro \
  -v "$WORK":/work \
  chipstar-llvm-cross-aarch64 \
  /chipstar/scripts/cross-aarch64/cross-build.sh 8
```

The container is run as the invoking user so the staged tree is not left
root-owned. `/work` is kept on the host between runs, so a rebuild is
incremental and the docker layer cache makes the image build a no-op.

Result: `$WORK/stage/home/pvelesko/install/llvm/23.0`, which is copied to
salami with

```bash
rsync -a --delete \
  "$WORK/stage/home/pvelesko/install/llvm/23.0/" \
  salami:install/llvm/23.0/
```

The prefix is baked into `CMAKE_INSTALL_RPATH`, so it must match the final path
on salami; override both with `LLVM_PREFIX=` if that ever changes.

Set `CONFIGURE_ONLY=on` in the environment to stop after the cmake configure
(useful for validating a runner without spending hours on a full build).

## In CI

`.github/workflows/test-llvm-patches.yml` wires this up as two jobs, both
gated on `detect.outputs.run_23` (a change under `llvm-patches/llvm-23/` or
under this directory):

* `cross-build-llvm23-aarch64` on `[self-hosted, Linux, X64]`: picks a
  workspace, builds the image, runs `cross-build.sh` with `-j(nproc-1)`, rsyncs the
  stage to `salami:install/llvm/23.0/`, writes salami's `llvm/23.0` modulefile,
  and runs `clang --version` on salami as a smoke test.
* `step2-build-test-salami` on `[self-hosted, Linux, ARM64]`: builds chipStar
  against that toolchain at `-j4` (3.6 GB RAM) and runs
  `scripts/check.py ./ igpu opencl`.

## Runner prerequisites

A checklist for any **future** x86 node. All three items are already in place
and verified on both current x86 runners, meatloaf and ramen, so nothing here
is pending work. None of it is automated in CI: it is one-time manual host
setup.

1. **docker usable without sudo** by the runner user (i.e. the user is in the
   `docker` group).
2. **Non-interactive ssh access to salami** as user `pvelesko`, for the rsync:
   the runner user's public key in salami's `authorized_keys`, salami's host
   key in the runner's `known_hosts`, and a `Host salami` block in the runner's
   `~/.ssh/config` (HostName/Port/User).
3. **~40 GB of scratch space** (a finished build measures ~12 GB). The workflow does not hardcode a location: a
   step probes `/space/$USER/llvm-cross-build`, then `$HOME`, then the runner
   temp dir, and uses the first with 40 GB free, failing fast with a `df` dump
   if none qualifies. This matters because the two nodes differ: ramen's `/`
   has only ~48 GB free (and a history of filling up) but a separate 3.6 TB
   `/space`, while meatloaf has no separate `/space` and ~76 GB on `/`.

## Cross-compiling chipStar and its tests for salami

The second use of this directory. A chipStar build takes 61 minutes on
salami (an ODROID-N2+), of which the tests are only 4 minutes to *run*.
So the tests are built on an x86 runner and only executed on salami.

| file | role |
| --- | --- |
| `Dockerfile.chipstar` | Extends the base image with two prebuilt dependencies chipStar cannot produce for itself in a cross build: an x86-hosted LLVM 23 with `X86;AArch64;SPIRV` (`image-llvm.sh`) and an aarch64 SPIRV-Tools (`image-spirv-tools.sh`). |
| `cross-chipstar.sh` | Runs inside that image. Two passes: a native x86 configure that builds only the tools chipStar *executes during its own build* (hipcc.bin, hipconfig.bin, the pass plugin, prepare-builtins), then the aarch64 build of chipStar + `build_tests`, with those x86 tools swapped in. |
| `chipstar-aarch64-toolchain.cmake` | Keeps clang as the compiler (chipStar's CMake rejects anything else, and device bitcode needs it) and retargets it via `CMAKE_<LANG>_COMPILER_TARGET`, which chipStar's `CMAKE_CXX_FLAGS` assignments cannot drop. |
| `libmali-stub/` | Salami's OpenCL is `libmali.so.0` loaded directly, no ICD. The stub carries that SONAME and defines every `cl*` the real driver exports (`cl-symbols.txt`, from `nm -D` on salami) as an empty function, so executables that call the API directly link. Nothing from it ships. |
| `ship-to-salami.sh` | Relocates the baked build-machine paths in every `CTestTestfile.cmake`, drops the tests that invoke a compiler at test time, and rsyncs to `salami:~/ci-stage/<sha>/`. |

Why the toolchain-invoking tests are dropped rather than run: the tree's
hipcc, opt and cucc are x86 binaries (they had to be, to run on the
builder), and a compiler that *did* run on aarch64 would be testing the
aarch64 host compiler, not Mali. Every one of those tests already runs on
the x86 gate with a working toolchain. The split is made by the test's
command shape in `CTestTestfile.cmake`, not by name, so a new test is
classified by what it does. On the last measurement 164 of 260 tests
survive the cut and 110 of those run (the rest skip for fp64); all pass.

Two things in chipStar are worked around here rather than fixed, and
should be fixed there:

* `CMakeLists.txt` derives `HOST_ARCH` from `llvm-config --host-target`
  and passes it as `--target=` to every host compile, after the toolchain
  file's `--target`, so it wins. It is a plain `set()` and cannot be
  overridden with `-D`. Pass 2 hands chipStar an `llvm-config` wrapper
  that answers `--host-target` with the aarch64 triple.
* hipcc drops `HIPCC_LINK_FLAGS_APPEND` on its `-no-hip-rt` link path
  (`hipBin_spirv.h`: the no-hip-rt copy is taken before the append is
  applied). Pass 2 wraps `bin/hipcc` to put the flags on every invocation.

Reproduce locally (image build is a one-off, ~40 min for the LLVM):

```
docker build -t chipstar-cross-aarch64:base scripts/cross-aarch64
docker run --rm -v $PWD:/chipstar:ro -v $WORK:/work chipstar-cross-aarch64:base \
  /chipstar/scripts/cross-aarch64/image-llvm.sh
docker run --rm -v $PWD/scripts/cross-aarch64:/s:ro -v $WORK:/work chipstar-cross-aarch64:base \
  /s/image-spirv-tools.sh
mkdir -p $WORK/image-ctx/opt && cp -a $WORK/x86-stage/opt/* $WORK/spirv-stage/opt/* $WORK/image-ctx/opt/
docker build -f scripts/cross-aarch64/Dockerfile.chipstar -t chipstar-cross-aarch64:llvm23 $WORK/image-ctx
docker run --rm -v $PWD:/chipstar:ro -v $WORK:/work chipstar-cross-aarch64:llvm23 \
  /chipstar/scripts/cross-aarch64/cross-chipstar.sh <sha>
scripts/cross-aarch64/ship-to-salami.sh $WORK/cross-<sha> <sha>
ssh salami "cd ci-stage/<sha> && LD_LIBRARY_PATH=\$PWD python3 src/scripts/check.py ./ igpu opencl"
```
