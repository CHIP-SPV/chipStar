
# CHIP-SPV internals for developers

The code of CHIP-SPV can be divided in two large parts: compilation and runtime library.

## Compilation

Compilation of HIP sources with CHIP-SPV is done by Clang. The Clang compiler has (since version 14) upstreamed changes that support the compilation process. In particular, the Clang driver (the program that "drives" the actual compilation process by running other programs) knows the steps to compile a CHIP-SPV program. The steps are:

1) compilation of the source in device mode (host code is ignored) to LLVM IR
2) linking of the LLVM IR (from previous step) to the CHIP-SPV bitcode library (see 'CHIP-SPV bitcode library' below)
3) running the necessary transformation passes (see LLVM passes below)
4) converting the device bitcode to SPIR-V binary. Currently, SPIRV-LLVM Translator tool is used for this, but in the future we may switch to using LLVM’s SPIR-V backend
5) bundling the SPIR-V in clang's "offload bundle" binary format (in theory, multiple binary formats could be bundled here, CHIP-SPV only uses SPIR-V)
6) compilation of the source in host mode (device code is ignored in this step) to LLVM IR. The offload bundle binary is included in the compilation unit as a "magic" global variable
7) assembling & linking of the LLVM IR from previous step to produce final object.


These steps with the exact arguments can be seen if Clang is run with "-v" argument. Additionally, if "--save-temps" argument is used, Clang will also keep the temporary output files of the compilation steps. These are useful for debugging the process. E.g. lets say we want to debug LLVM passes on the `samples/hipmath` example. Running `make VERBOSE=1` gives us the command line CMake uses to compile the example:

```bash
/usr/local/bin/clang++ -DLLVMHipPasses_EXPORTS -I/usr/lib/llvm-14/include -Wall .... -D__HIP_PLATFORM_SPIRV__= -x hip --target=x86_64-linux-gnu --offload=spirv64 -nohipwrapperinc --hip-path=/path/to/chip/build -std=c++17 -MD -MT samples/hipmath/CMakeFiles/hipmath.dir/hipmath.cc.o -MF CMakeFiles/hipmath.dir/hipmath.cc.o.d -o CMakeFiles/hipmath.dir/hipmath.cc.o -c samples/hipmath/hipmath.cc
```

running this same command with additional "-v --save-temps" shows this:

```bash
clang version 14.0.5 (https://github.com/llvm/llvm-project.git b950bd2ce7ff79b203b2acba02e1c468836989ae)
Target: x86_64-unknown-linux-gnu
Thread model: posix
InstalledDir: /usr/local/bin
Found candidate GCC installation: /usr/lib/gcc/x86_64-linux-gnu/9
Selected GCC installation: /usr/lib/gcc/x86_64-linux-gnu/9
Candidate multilib: .;@m64
Selected multilib: .;@m64
Found HIP installation: /home/michal/0/build/b_chip_sycl, version 5.1.0

# preprocessing (in device mode)
/usr/local/bin/clang -cc1 -triple spirv64 -aux-triple x86_64-unknown-linux-gnu .... -o hipmath-hip-spirv64-generic.cui -x hip samples/hipmath/hipmath.cc
# compilation to LLVM IR (device mode)
/usr/local/bin/clang -cc1 -triple spirv64 -aux-triple x86_64-unknown-linux-gnu .... -o hipmath-hip-spirv64-generic.bc -x hip-cpp-output hipmath-hip-spirv64-generic.cui
# this step links in the builtin library and internalizes the functions
/usr/local/bin/clang -cc1 -triple spirv64 -aux-triple x86_64-unknown-linux-gnu .... -mlink-builtin-bitcode /path../lib/hip-device-lib/hipspv-spirv64.bc -o hipmath-hip-spirv64-generic.bc -x ir hipmath-hip-spirv64-generic.bc

# this step is meant to link device code objects from multiple TUs (translation units, if there are multiple)
/usr/local/bin/llvm-link  hipmath-hip-spirv64-generic.bc -o hipmath-hip-spirv64-generic-link.bc
# runs the CHIP-SPV LLVM passes
/usr/local/bin/opt hipmath-hip-spirv64-generic-link.bc -load-pass-plugin /chip_build_dir/lib/libLLVMHipSpvPasses.so -passes=hip-post-link-passes -o hipmath-hip-spirv64-generic-lower.bc
# convert device LLVM IR bitcode to SPIR-V
/usr/local/bin/llvm-spirv  --spirv-max-version=1.1 --spirv-ext=+all hipmath-hip-spirv64-generic-lower.bc -o hipmath-hip-spirv64-generic.out
# converts the SPIR-V to a clang "offload bundle" format
/usr/local/bin/clang-offload-bundler -type=o -bundle-align=4096 -targets=host-x86_64-unknown-linux,hip-spirv64----generic -inputs=... -outputs=...

# preprocessing (in host mode)
/usr/local/bin/clang -cc1 -triple x86_64-unknown-linux-gnu -aux-triple spirv64 ... -o hipmath-host-x86_64-unknown-linux-gnu.cui -x hip samples/hipmath/hipmath.cc
# compilation to LLVM IR (host mode) with the offload binary
/usr/local/bin/clang -cc1 -triple x86_64-unknown-linux-gnu -aux-triple spirv64 ... -fcuda-include-gpubinary hipmath.cc-hip-spirv64.hipfb -o hipmath-host-x86_64-unknown-linux-gnu.bc -x hip-cpp-output hipmath-host-x86_64-unknown-linux-gnu.cui
# target code (ASM) emitting step
/usr/local/bin/clang -cc1 -triple x86_64-unknown-linux-gnu -aux-triple spirv64 ... -o hipmath-host-x86_64-unknown-linux-gnu.s -x ir hipmath-host-x86_64-unknown-linux-gnu.bc
# produce final object binary
/usr/local/bin/clang -cc1as -triple x86_64-unknown-linux-gnu -filetype obj ... -o CMakeFiles/hipmath.dir/hipmath.cc.o hipmath-host-x86_64-unknown-linux-gnu.s

```

### CHIP-SPV bitcode library

This is a library which contains HIP device-side functions (math, workgroup and others) which either don't exist in OpenCL at all, or have different implementation (different name, function signature etc), and implements these by using OpenCL device-side functions (or OpenCL extensions where possible).

For example, the `__syncthreads()` call is implemented by calling `barrier(CLK_LOCAL_MEM_FENCE)`, `rhypot()` has no equivalent in OpenCL so it's implemented via OCML, and shuffle functions are implemented using intel's `cl_intel_subgroups` extension (there is now a khronos subgroup extension, `cl_khr_subgroup_shuffle`, but it did not exist at the time).


### LLVM passes

There are several transformations (LLVM passes) done on the LLVM IR of the device code, because of differences between CUDA and OpenCL/Level0/SPIR-V capabilities. They can be found in `<CHIP-SPV>/llvm-passes`. Most of them have more documentation in the source. Note that for debugging LLVM passes, it's recommended to compile Clang+LLVM from source in debug mode and enable assertions (`cmake -DLLVM_ENABLE_ASSERTIONS=ON ...`); the Clang/LLVM packages in linux distributions are usually compiled in release mode, without assertions.

* HipAbort.cpp - special handling for abort() calls from device side (these must terminate the program)
* HipDefrost.cpp - removes freeze from instructions (workaround for the llvm-spirv translator)
* HipDynMem.cpp - replaces dynamically sized shared-memory variables (`extern __shared__ type variable[];`) with a kernel argument. This is because in OpenCL, dynamically-sized local memory can only be passed as kernel argument.
* HipGlobalVariable.cpp - creates special kernels that handle access and modification of global scope variables.
* HipPasses.cpp - defines a pass plugin that runs a collection of LLVM passes (= rest of the files in this directory)
* HipPrintf.cpp - pass to convert calls to the CUDA/HIP printf() to OpenCL/SPIR-V compatible printf() calls.
* HipStripUsedIntrinsics.cpp - pass to remove llvm.used and llvm.compiler.used intrinsic variables
* HipTextureLowering.cpp - pass that transforms kernels (and texturing functions) with `hipTextureObject_t` argument to kernels with actual opencl image+sampler arguments

## Runtime

The runtime implements the HIP API. It consists mainly of these files:

* `src/CHIPBindings.cc` - implementation of HIP API functions like e.g. `hipEventRecord` by using the abstract backend classes
* `src/CHIPBackend.{cc,hh}` - a set of abstract backend classes used by the HIP API functions
* `src/backend/Level0/CHIPBackendLevel0.{cc,hh}` - Level0 implementation of the abstract backend
* `src/backend/OpenCL/CHIPBackendOpenCL.{cc,hh}` - OpenCL implementation of the abstract backend
* `src/Driver.{cc,hh}` - implementation of (un)initialization routines
* `src/spirv.{cc,hh}` - a very simple SPIR-V binary parser, only enough to parse function names & their arguments
* `src/logging.{cc,hh}` - spdlog logging library setup
* `src/hip_conversions.hh` - contains various conversion helpers e.g. converting hipTextureAddressMode to corresponding OpenCL's address mode constants

### application lifetime

For every application compiled in HIP mode, Clang inserts several hidden initialization calls which are called before main().
These called functions are implemented in the runtime, `src/CHIPBindings.cc`; the names start with `__hip`:

`__hipRegisterFatBinary`: this is called with the "clang offload bundle" binary blob. The function parses the binary's header and ignores all bundles except the SPIR-V one. The SPIR-V binary is than handed over to the backends for compilation, but it's also parsed by the runtime (with the mini SPIR-V parser) to obtain function signatures and arguments. This is necessary to properly set up arguments for function calls.

`__hipRegisterFunction`: this is called with each kernel name. It connects a function name with a host pointer (void*). That host pointer is later used by launch API.

`__hipRegisterVar`: this registers global variables, again connecting a host pointer with a variable name.

there is an additional `__hipUnregisterFatBinary()` called after main() returns; in CHIP-SPV, this calls `CHIPUninitialize()` once the number of loaded modules becomes zero.
