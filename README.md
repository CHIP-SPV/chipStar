# CHIP-SPV

CHIP-SPV is a HIP implementation that abstracts HIP API, providing a
set of base classes that can be derived from to implement an
additional, SPIR-V capable backend on which to execute HIP
calls.

Currently CHIP-SPV supports OpenCL and Level Zero as backend alternatives.

This project is an integration of [HIPCL](https://github.com/cpc/hipcl) and
[HIPLZ](https://github.com/jz10/anl-gt-gpu/) projects.

For CHIP-SPV User documentation, read [this.](docs/Using.md)
For CHIP-SPV Developer documentation, read [this.](docs/Development.md)
For a list of (un)supported features in CHIP-SPV, read [this.](docs/Features.md)

## Prerequisites

* Cmake >= 3.18.0
* Clang 14 or 15
  * Can be installed, for example, by adding the [LLVM's Debian/Ubuntu repository](https://apt.llvm.org/) and installing packages 'clang-15 llvm-15 clang-tools-15'. *NOTE*: The Ubuntu clang package does not provide a symlink for `clang++`, only `clang++-14` is availble. If you plan on using `hipcc` you will need to make this symlink manually to ensure that `clang++` is available in `HIP_CLANG_PATH`.
* SPIRV-LLVM-Translator from a branch matching to the clang version:
  (e.g. llvm\_release\_150 for Clang 15.0)
  [llvm-spirv](https://github.com/KhronosGroup/SPIRV-LLVM-Translator)
* For Level Zero Backend
  * [Intel Compute Runtime](https://github.com/intel/compute-runtime) and
  * [oneAPI Level Zero Loader](https://github.com/oneapi-src/level-zero/releases)
* For OpenCL Backend
  * OpenCL implementation with Shared Virtual Memory and SPIR-V
    support.

## Downloading Sources

```bash
git clone https://github.com/CHIP-SPV/chip-spv.git
cd chip-spv
git submodule update --init --recursive
```

## Building

```bash
mkdir build
cd build

cmake .. \
 -DCMAKE_CXX_COMPILER=clang++ \
 -DCMAKE_C_COMPILER=clang \
 -DCMAKE_INSTALL_PREFIX=<install location>
 # optional: -DCMAKE_BUILD_TYPE=<Debug(default), Release, RelWithDebInfo>
 # optional: to provide a path to separately built SPIRV-LLVM translator, use -DLLVM_SPIRV_BINARY=/path
make
make install
```

Be sure you refer to the clang++ and clang binaries you want to build against in
the above command. For example, the LLVM debian packages might install binaries with
a version suffix in the names: 'clang++-15' and 'clang-15'.

## Building documentation

Make sure you have doxygen installed. Then:

```bash
....
cd build
cmake .....
make gendocs
```

The documentation will be placed in `doxygen/html`.

## Building & Running Unit Tests

```bash
make build_tests_standalone
ctest --timeout 40 # currently some tests might hang
```

## Environment Flags

### CHIP_BE

Select the backend to execute on.
Possible values: opencl(default), level0

### CHIP_LOGLEVEL

Select the verbosity of debug info during execution.
Possible values: trace, debug(default), warn, err, crit

Setting this value to `debug` will print information coming from the CHIP-SPV functions which are shared between the backends.

Settings this value to `trace` will print `debug`, as well as debug infomarmation from the backend implementation itself such as results from low-level Level Zero API calls.

### HIP_PLATFORM

Select which HIP implementation to execute on. Possible values: amd, nvidia, spirv.
If you do not provide this value, `hipcc` will check for existance of the following directions and try to guess which implementation is available:

```bash
/usr/local/cuda
/opt/rocm
```

## Compiling CUDA application directly with CHIP-SPV

Compilation of CUDA sources without changing the sources, can be done in two ways. The first is to replace calls of the nvcc compiler with calls of the wrapper script <CHIP-install-path>/bin/cuspv in Makefiles. The wrapper script will call clang with the correct flags. The other way is possible when using CMake: use `find_package(HIP REQUIRED CONFIG)` and then use `target_link_libraries(<YOUR_TARGET> hip::device)`. However the project must be compiled with Clang (a version supported by HIP). Note that it's not necessary to have Nvidia's CUDA installed.

## Compiling a HIP application using CHIP-SPV

Compiling a HIP application with CHIP-SPV will allow you to execute HIP code on any device that supports SPIR-V, such as Intel GPUs. To compile a HIP application, all you need to do is to use the `hipcc` compiler wrapper provided by this project. In case you have AMD implementation installed as well, you can switch between them by using `HIP_PLATFORM` environment variable.

You can find various HIP applications here: <https://github.com/CHIP-SPV/hip-testsuite>

```bash
export HIP_PLATFORM=spirv
hipcc ./hip_app.cpp -o hip_app
```

## Interoperability with OpenCL / Level0 APIs from HIP code

CHIP-SPV provides several extra APIs (not present in AMD's HIP) for interoperability with its backends:

* the hipGetBackendNativeHandles API returns native object handles, but does not give up ownership of these objects (HIP will keep using them asynchronously with the application). 
* hipInitFromNativeHandles API creates HIP objects from native handles, but again does not take ownership (they're still usable from application).

Synchronization of context switching is left to the application.

Both APIs take an array as argument. In both cases, the NumHandles size must be set to 4 before the call (because the APIs currently take/return 4 handles).
With OpenCL the array contains: cl_platform_id, cl_device_id, cl_context, cl_command_queue.
With Level0 the array contains: ze_driver_handle_t, ze_device_handle_t, ze_context_handle_t, ze_command_queue_handle_t.

there are also two APIs for asynchronous interoperability:

* hipGetNativeEventFromHipEvent takes hipEvent_t and returns an OpenCL/Level0 event handle as void*
* hipGetHipEventFromNativeEvent takes OpenCL/Level0 event handle, and returns a hipEvent_t as void*

Before using the hipGetNativeEvent, the event must be recorded in a Stream.

With OpenCL, both get*Event APIs increase the refcount of the cl_event and each side (HIP and the user application) are responsible for releasing the event when they’re done with it.
With Level0, the event pool from which it was allocated remains the responsibility of the side that allocated it (e.g. getNativeEvent returns a ze_event handle but the pool is still managed by CHIP-SPV). This could lead to issues if e.g. a pool is released but an event allocated from it still exists as a dependency in some command enqueued by the opposite API. Since CHIP-SPV's Level0 backend never releases event pools, this can be resolved by not releasing eventpools allocated on the application side.

Simplified example code with OpenCL interop:

```C
    void* runNativeKernel(void *NativeEventDep, uintptr_t *NativeHandles, int NumHandles, unsigned Blocks, unsigned Threads, unsigned Arg1, void *Arg2, void *Arg3) {
      cl_device_id Dev = (cl_device_id)NativeHandles[1];
      cl_context Ctx = (cl_context)NativeHandles[2];
      cl_command_queue CQ = (cl_command_queue)NativeHandles[3];

      cl_event DepEv = (cl_event)NativeEventDep;

      if (Program == 0) {
        Program = clCreateProgramWithIL(Ctx, KernelSpirV, KernelSpirVLength, &Err);

        clBuildProgram(Program, 1, &Dev, NULL, NULL, NULL);

        Kernel = clCreateKernel(Program, "binomial_options.1", &Err);

        clSetKernelArg(Kernel, 0, sizeof(int), &Arg1);
        clSetKernelArgSVMPointer(Kernel, 1, Arg2);
        clSetKernelArgSVMPointer(Kernel, 2, Arg3);
      }

      size_t Goffs0[3] = { 0, 0, 0 };
      size_t GWS[3] = { Blocks*Threads, 0, 0 };
      size_t LWS[3] = { Threads, 0, 0 };
      cl_event RetEvent = 0;
      clEnqueueNDRangeKernel(CQ, Kernel, 1, Goffs0, GWS, LWS, 1, &DepEv, &RetEvent);
      return (void*)RetEvent;
    }

    int main() {

        hipEvent_t Start, Stop;

        uintptr_t NativeHandles[4];
        int NumHandles = 4;
        hipGetBackendNativeHandles((uintptr_t)0, NativeHandles, &NumHandles);

        ....

        hipEventCreate(&Start);
        hipEventCreate(&Stop);
        hipEventRecord(Start, NULL);

        hipLaunchKernelGGL(binomial_options, ...);

        hipEventRecord(Stop, NULL);

        void *NativeEventDep = hipGetNativeEventFromHipEvent(Stop);
        assert (NativeEventDep != nullptr);
        void *NativeEvtRun = runNativeKernel(NativeEventDep, NativeHandles, NumHandles,
                                             blocks, threads,
                                             arg1, (void*)input, (void*)output);
        hipEvent_t EventRun = (hipEvent_t)hipGetHipEventFromNativeEvent(NativeEvtRun);
        assert (EventRun);
        hipStreamWaitEvent(NULL, EventRun, 0);

        ....
    }
```

This example launches the `binomial_options` HIP kernel using `hipLaunchKernelGGL`, gets the native event of that launch, and launches a native kernel with that event as dependency. The event returned by that native launch, can in turn be used by HIP code as dependency (in this case it's used with `hipStreamWaitEvent`). The full example with both Level0 and OpenCL interoperability can be found in CHIP-SPV sources: `<CHIP-SPV>/samples/hip_async_interop`.

## Using CHIP-SPV With CMake

CHIP-SPV provides a `FindHIP.cmake` module so you can verify that HIP is installed:

```bash
list(APPEND CMAKE_MODULES_PREFIX <CHIP-SPV install location>/cmake)
find_package(HIP REQUIRED)
```
