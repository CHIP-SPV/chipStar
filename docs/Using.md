## Using CHIP-SPV

### Environment Flags

#### CHIP_BE

Select the backend to execute on.
Possible values: opencl, level0, default

If set to "default" (or unset), it automatically selects any available backend in order: Level0, OpenCL.

#### CHIP_LOGLEVEL

Select the verbosity of debug info during execution.
Possible values: trace, debug(default for Debug builds), warn(default for non-Debug builds), err, crit

Setting this value to `debug` will print information coming from the CHIP-SPV functions which are shared between the backends.

Settings this value to `trace` will print `debug`, as well as debug infomarmation from the backend implementation itself such as results from low-level Level Zero API calls.

#### HIP_PLATFORM

Select which HIP implementation to execute on. Possible values: amd, nvidia, spirv.
If you do not provide this value, `hipcc` will check for existance of the following directions and try to guess which implementation is available:

```bash
/usr/local/cuda
/opt/rocm
```

### Disabling GPU hangcheck

Note that long-running GPU compute kernels can trigger hang detection mechanism in the GPU driver, which will cause the kernel execution to be terminated and the runtime will report an error. Consult the documentation of your GPU driver on how to disable this hangcheck.

### Compiling CUDA application directly with CHIP-SPV

Compilation of CUDA sources without changing the sources, can be done in two ways. The first is to replace calls of the nvcc compiler with calls of the wrapper script <CHIP-install-path>/bin/cuspv in Makefiles. The wrapper script will call clang with the correct flags. The other way is possible when using CMake: use `find_package(HIP REQUIRED CONFIG)` and then use `target_link_libraries(<YOUR_TARGET> hip::device)`. However the project must be compiled with Clang (a version supported by HIP). Note that it's not necessary to have Nvidia's CUDA installed.

### Compiling a HIP application using CHIP-SPV

Compiling a HIP application with CHIP-SPV will allow you to execute HIP code on any device that supports SPIR-V, such as Intel GPUs. To compile a HIP application, all you need to do is to use the `hipcc` compiler wrapper provided by this project. In case you have AMD implementation installed as well, you can switch between them by using `HIP_PLATFORM` environment variable.

You can find various HIP applications here: <https://github.com/CHIP-SPV/hip-testsuite>

```bash
hipcc ./hip_app.cpp -o hip_app
```

### Interoperability with OpenCL / Level0 APIs from HIP code

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

### Using CHIP-SPV in own projects (with CMake)

CHIP-SPV provides a `FindHIP.cmake` module so you can verify that HIP is installed:

```bash
list(APPEND CMAKE_MODULES_PREFIX <CHIP-SPV install location>/cmake)
find_package(HIP REQUIRED)
```
