## Using chipStar

### Environment Variables

These variables can be used to control `hipcc` when targeting chipStar.

#### CHIP_BE

Selects the backend to execute on.
Possible values: opencl, level0, default

If set to "default" (or unset), it automatically selects any available backend in order: Level0, OpenCL.

#### CHIP_LOGLEVEL

Selects the verbosity of debug info during execution.
Possible values: `trace`, `debug` (default for Debug builds), `warn` (default for non-Debug builds), `err`, `crit`.

Note that the values `trace` and `debug` need chipStar to be compiled in DEBUG mode.

Setting this value to `debug` will print information coming from the chipStar functions which are shared between the backends.

Settings this value to `trace` will print `debug`, as well as debug infomarmation from the backend implementation itself such as results from low-level Level Zero API calls.

#### CHIP_DEVICE_TYPE
Selects which type of device to use. 

Possible values for Level 0 backend are: `gpu` and `fpga`. Leaving it empty uses all available devices.

Possible values for OpenCL backend are: `gpu`, `cpu` , `accel`. Leaving it empty uses GPU devices.

#### HIP_PLATFORM

Selects which HIP implementation to execute on. Possible values: amd, nvidia, spirv.
If you do not provide this value, `hipcc` will check for existance of the following directions and try to guess which implementation is available:

```bash
/usr/local/cuda
/opt/rocm
```

#### CHIP_RTC_SAVE_TEMPS

Preserves runtime temporary compilation files when this variable is set to `1`.

#### CHIP\_LAZY\_JIT

When set to `0`, chipStar will compile all device modules at the runtime
initialization. Default setting is `1` meaning the device modules are
compiled just before kernel launches.

#### CHIP\_OCL\_DISABLE\_QUEUE\_PROFILING

A debug option for forcing queue profiling to be disabled in the
OpenCL backend. The default setting is `0`.

#### CHIP\_OCL\_USE\_ALLOC\_STRATEGY

Defines the allocation strategy the OpenCL backend uses for managing
HIP allocations. The valid case-insensitive choices and their meaning
are:

* `intelusm` or `usm`: The backend uses Intel Unified Shared Memory
  (USM) extension for HIP allocations. The OpenCL devices must support
  `cl_intel_unified_shared_memory` extension to use this strategy.

* `svm`: The backend uses shared virtual memory (SVM). The OpenCL
  devices must support at least coarse grain SVM to use this strategy.

* `bufferdevaddr`: The backend uses `cl_mem` buffers and experimental
  `cl_ext_buffer_devive_address` extension. Note: unified virtual
  addressing is not available when using this strategy. Consequently,
  `hipMemcpyDefault` flag is not supported and `hipMallocHost()`
  allocations are not implicitly mapped and portable. The OpenCL
  devices must support the `cl_ext_buffer_devive_address`
  extension to use this strategy.

If this variable is not set, the backend chooses the first available
strategy in this order: `usm`, `svm`, `bufferdevaddr`.

### Disabling GPU hangcheck

Note that long-running GPU compute kernels can trigger hang detection mechanism in the GPU driver, which will cause the kernel execution to be terminated and the runtime will report an error. Consult the documentation of your GPU driver on how to disable this hangcheck.

### Compiling CUDA applications directly with chipStar

CUDA sources can be compiled directly without converting the sources
to HIP first. CUDA compilation can be driven with the `cucc` compiler
driver which can be used as drop-in replacement for
[nvcc](https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html).
Beware though - the tool, and the CUDA runtime headers, are still a
work in progress and some features are missing not not supported. Here
is a non-exhaustive list of known issues and limitations:

* Only a subset of nvcc options are supported. Some of them are
  ignored now, others are not processed and they may cause compile
  errors.

* Inline assembly snippets within device code are not supported.

* CUDA sources without any includes do not compile for now.
  Workaround: include <cuda_runtime.h>.

* Some nvcc preprocessor defines are not set currently -
  e.g. `__CUDA_ARCH__` and `__CUDA_ARCH_LIST__`.

* Sources depending on CUDA libraries (thrust, cuBLAS, cuFFT, etc)
  won't compile for now and might produce possibly obscure error
  messages.

* Warp functions ending with `_sync` are not currently
  supported. Workaround: call the legacy warp functions instead if
  possible - the current requirement of chipStar is that all threads
  in the warp must converge at the same warp function.

* Some runtime API types and functions might be undeclared.

#### Configuring CMake-based CUDA Projects

CUDA projects using CMake can be targeted to utilize chipStar. The
feature is experimental, and has been only tested with CMake 3.24.3.
Additionally, thie feature currently only covers CMake projects that
use CMake's built-in CUDA compilation flow instead of relying
`find_package(CUDA)` or `find_package(CUDAToolkit)`. Here is an
example of `CMakeLists.txt` for enabling CUDA.

```cmake
project(MyCudaProject LANGUAGES CUDA CXX C)
add_executable(myexe main.cu some.cu some.cpp some.c)
```

The CUDA-on-chipStar is enabled by making the `cucc` tool appear as
`nvcc` for CMake. To make CMake target chipStar, define a following
environment variable:

```
export CUCC_VERSION_STRING="nvcc: NVIDIA (R) Cuda compiler driver"
```

Create dummy `libcudart.so`, `libcudart_static.a` and `libcudadevrt.so`
files and place them under the chipStar library install directory:

```
touch <chipStar-install-dir>/lib/libcudart.so
touch <chipStar-install-dir>/lib/libcudart_static.a
touch <chipStar-install-dir>/lib/libcudadevrt.so
```

And invoke cmake as follows:

```
cmake -DCMAKE_CUDA_COMPILER=<chipStar-install-dir>/bin/cucc \
      -DCMAKE_CUDA_ARCHITECTURES=72 /path/to/your/cuda-project
```

Note that the `CMAKE_CUDA_ARCHITECTURES` only needs to be set to some valid
architecture for the command to succeed. Otherwise, the value does not
matter and chipStar ignores it.

### Compiling a HIP application using chipStar

Compiling a HIP application with chipStar will create a fat binary which can be executed on any device that fulfils the prerequisities. To compile a HIP application, all you need to do is to use the `hipcc` compiler wrapper provided by this project. In case you have AMD implementation installed as well, you can switch between them by using `HIP_PLATFORM` environment variable.

You can find various HIP applications here for testing: <https://github.com/CHIP-SPV/hip-testsuite>

```bash
hipcc ./hip_app.cpp -o hip_app
```

### Interoperability with OpenCL / Level0 APIs from HIP code

chipStar provides several extra APIs which are not present in the AMD's HIP API for interoperability with its backends:

* the hipGetBackendNativeHandles() function returns native object handles, but does not give up ownership of these objects (HIP will keep using them asynchronously with the application).
* hipInitFromNativeHandles () creates HIP objects from native handles, but again does not take ownership (they're still usable from application).

Proper synchronization of context switching is left to the application.

Both APIs take an array as argument. In both cases, the NumHandles size must be set to 4 before the call (because the APIs currently take/return 4 handles).
With OpenCL the array contains: cl_platform_id, cl_device_id, cl_context, cl_command_queue.
With Level0 the array contains: ze_driver_handle_t, ze_device_handle_t, ze_context_handle_t, ze_command_queue_handle_t.

there are also two APIs for asynchronous interoperability:

* hipGetNativeEventFromHipEvent takes hipEvent_t and returns an OpenCL/Level0 event handle as void*
* hipGetHipEventFromNativeEvent takes OpenCL/Level0 event handle, and returns a hipEvent_t as void*

Before using the hipGetNativeEvent, the event must be recorded in a Stream.

With OpenCL, both get*Event APIs increase the refcount of the cl_event and each side (HIP and the user application) are responsible for releasing the event when they’re done with it.
With Level0, the event pool from which it was allocated remains the responsibility of the side that allocated it (e.g. getNativeEvent returns a ze_event handle but the pool is still managed by chipStar). This could lead to issues if e.g. a pool is released but an event allocated from it still exists as a dependency in some command enqueued by the opposite API. Since chipStar's Level0 backend never releases event pools, this can be resolved by not releasing eventpools allocated on the application side.

A simple example code that uses the OpenCL interop:

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

This example launches the `binomial_options` HIP kernel using `hipLaunchKernelGGL`, gets the native event of that launch, and launches a native kernel with that event as dependency. The event returned by that native launch, can in turn be used by HIP code as dependency (in this case it's used with `hipStreamWaitEvent`). The full example with both Level0 and OpenCL interoperability can be found in chipStar sources: `<chipStar>/samples/hip_async_interop`.

### Using chipStar in own projects (with CMake)

chipStar provides a `FindHIP.cmake` module so you can verify that HIP is installed:

```bash
list(APPEND CMAKE_MODULES_PREFIX <chipStar install location>/cmake)
find_package(HIP REQUIRED)
```

### Compiling HIP sources in relocatable device mode (RDC) with CMake

```bash
addLibrary(yourLib <sources>)
target_link_libraries(yourLib hip::deviceRDC)
```

### Compiling HIP sources in relocatable device mode (RDC) with hipcc

With single command:

```bash
hipcc -fgpu-gpu a.hip b.hip c.hip -o abc

```

Or separately:

```bash
hipcc -fgpu-rdc -c a.hip -o a.o
hipcc -fgpu-rdc -c b.hip -o b.o
hipcc -fgpu-rdc -c c.hip -o c.o
hipcc -fgpu-rdc a.o b.o c.o -o abc
```
