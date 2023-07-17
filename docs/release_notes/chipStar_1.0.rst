********************************
Release Notes for chipStar 1.0
********************************

=============================
Major New Features Since 0.9
=============================

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Support for Clang/LLVM 16.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added supports for Clang/LLVM 16. LLVM 15 and 14 might still work but are not
prioritized. Other Clang/LLVM versions are unsupported.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Support for hiprtc API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implemented essential hiprtc API for invoking C++ kernels in HIP code and added support for few whitelisted compilation options such as -D, -O and -std. All unrecognized options are ignored by default and warnings are given when they are encountered.

Supported features:

* Invoking ``extern "C" __global__`` kernels.

Unsupported features:

* Non-whitelisted compile options
* Invoking non-extern "C" ``__global__`` kernels
* Accessing global variables.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Lazy Compilation of Device Code
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implemented lazy compilation of device code that compiles device modules when needed instead of compiling all of them at once during chipStar runtime initialization. This reduces the startup time of applications which have a lot of device code (for example, neural network node libraries) but only use a small portion of it per application launch. For a Pytorch application case the initial compilation time decreased from ~40 minutes to ~40 seconds.

=============================
Minor New Features Since 0.9
=============================

* Synchronized chipStar’s FP16 support to the latest one in AMD HIP.

* Emulation of warp functions using the required sub group metadata added to kernels that are sensitive to the warp size.

* Added support for kernel launches up to 4KB arg size to circumvent the lower limit in Intel GPU drivers. The workaround is modifying the kernels so that some of the large pass-by-value arguments are read from runtime allocated device buffer to meet the kernel parameter size limit.

* Added support for ``__device__ std::complex`` type and its operations.

* Added support for compiling HIP sources in relocatable device code (RDC) mode. Refer to ``docs/Using.md`` for instructions.

* Added missing device-side ``__assert_fail()`` function, needed for calling ``assert()`` from the device side.

* Implemented callbacks that have correct HIP semantics for OpenCL backend by using OpenCL callbacks and user events.

* Added experimental support for emitting device code (SPIR-V) with indirect calls which requires support for ``SPV_INTEL_function_pointers`` in the driver.

* Added a HIP-SPIRV pass for sanity checks starting with checks for unsupported indirect calls. This checks for accidental indirect calls, and caller-callee function type mismatch.

=============================
Bugfixes
=============================

* Fixed compiler error with ``__attribute__((__noinline__))``.

* Added missing ``ihipEvent_t`` forward declaration.

* Fixed hip_runtime.h failing to compile in C++ mode.

* Added missing HIP undocumented device ``atomicAddNoRet`` function, required for certain applications.

* Added some missing __host__ and many __device__ implementations of ``cmath.h``.

* Fixed HIP sources fail to compile with ``-std=gnu++14`` (and probably with other gnu++ versions too) due to inclusion of ``__float128`` type.

* Fixed size calculation logic for kernel aggregate arguments passed by value; in some cases it underestimated argument sizes.

* Fixed ``error: call to '__shfl*' is ambiguous``. fixed by implementing missing ``__shfl`` overloads.

* Fixed ``error: no matching conversion for C-style cast from 'float' to '__half'``, with ``-D__HIP_NO_HALF_CONVERSIONS__=1``.

* Fixed very slow compilation/execution of ``__device__`` variable initialization shadow kernel when large ``__device__`` variables are involved.

* Some tests pass invalid pointers to kernels when they know the kernel is not going to dereference them. This caused chipStar to crash as it assumed valid pointers when setting kernel arguments. The workaround used by chipStar is to set the kernel argument to NULL when an invalid pointer is detected.

* Fix ``__device__`` variables not fully initialized.

* Fix shadow kernel causing shared local memory limit to be exceeded.

* Memset intrinsics should be marked external linkage.

* Having invalid printf specifiers crashed a LLVM lowering pass.

* Fixed a race condition related to hipstar’s event resource management.

* Don't drop linkage attributes in SPIR-V modules too eagerly.

* Indirectly accessed buffers were not synchronized because the runtime didn’t detect them and indirect buffer uses were not informed to OpenCL and Level Zero driver.

* Fix crashes in multi-threaded applications due to ``cl_kernel`` objects being shared among threads.

* Hipcc fails to compile C source to executable, producing object file instead.

* Add missing implementations of ``__{u,}int_as_float()`` and ``__float_as_{u,}int()`` HIP conversion intrinsics.

* Add missing undocumented ``atomicAdd(ulong)`` since it’s required by rocPRIM.

* Fix uninitialized ``gcnArchName`` property.

* Uninitialize runtime at program exit despite of unreleased user modules.

* Fix missing ``__device__`` declaration for ``pow(double, double)``.

* Fix behaviour of ``-x`` option to ``hipcc``.

* Fix a compile error from HIP vector operation: ``error: invalid operands to binary expression ('uchar2' (aka 'HIP_vector_type<unsigned char, 2>') and 'uchar2')``.

* Fix hipcc compilation failing when mixing source and object files in the same command.

* Added a SPIRV lowering pass for dealing with switch instructions with selectors that uses “non-standard” bitwidths (e.g. i4).


=============================
Known Issues
=============================

* `__align__ attribute is broken <https://github.com/CHIP-SPV/chipStar/issues/484>`_

* `compile errors from hip/hip_fp16.h inclusion <https://github.com/CHIP-SPV/chipStar/issues/515>`_

* `error: no matching function for call to atomicMax <https://github.com/CHIP-SPV/chipStar/issues/516>`_

* chipStar 1.0 is not yet optimized for performance. The initial focus has been to get functionality first. Optimizations come next.


