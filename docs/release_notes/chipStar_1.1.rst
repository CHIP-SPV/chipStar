********************************
Release Notes for chipStar 1.1
********************************

This release cycle focused on stabilization and performance improvements
over the 1.0 release. The release was measured to run some benchmarks up
to 80% faster than 1.0. Further highlights are described in the following.

==================
Major New Features
==================

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Support for Clang/LLVM 17.0
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added support for Clang/LLVM 17. LLVM 15 and 16 are still supported.
Older Clang/LLVM versions are unsupported.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Ability to Use the Intel Unified Shared Memory Extension with OpenCL
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previously, the OpenCL device was checked for fine-grained SVM support,
and if it was unavailable, the backend would assume only coarse-grained
support and insert SVMMap & SVMUnmap for each buffer used by a kernel.

In case the target supports the USM extension, but not fine-grain
SVM, chipStar can now use USM automatically to avoid the overhead of the
unnecessary mapping.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Optimized Atomic Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

chipStar now uses the generic address space versions of atomics (available in
OpenCL 3.0) to avoid branches to select the correct call for the given address space.
It can also now utilize the ``cl_ext_float_atomics`` extension instead of
a compare-and-swap loop emulation with floating point atomics.

The atomics also now use the relaxed ordering which is allowed by the HIP
specs. This gave major improvements with atomics-heavy benchmarks.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Use Level Zero Immediate Command Lists for Low Latency Dispatch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Using immediate command lists by default enables significant latency improvements
and shows its impact especially on benchmarks which queue lots of small kernels.
This doesn't apply to the OpenCL, since its command queues can be executed
immediately by default.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Improved Asynchronous Execution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Previous versions of chipStar used command queue barriers excessively for
synchronization, which led to limited opportunities for asynchronous execution.
In chipStar 1.1, command queue synchronization is done using event dependencies,
which leads to more task parallelism opportunities presented to the drivers,
speeding up various workloads significantly. Workloads that do not exploit
parallelism but enqueue a lot of very small kernels (in the 10's of microseconds
range) may also benefit as the barrier itself could dominate the execution time.

==============
Minor Features
==============

* Added missing __device__ std::pow overloads (#569).

* Implemented system atomics.

* There were unneccessary 'volatile' keywords used in ``__chip_mem{set,cpy}``. Removing them sped up these builtins greatly.

* Support __align__ attribute in HIP code (#659).

* Level Zero command lists are now recycled (#665).

==============
Major Bugfixes
==============

* Fixed multiple memory leaks.

* Issues related to running libCEED fixed, and added it to the test suite.

* Do not pass ``-x spir`` to clBuildProgram() in OpenCL backend. '-x spir' requests compilation of the old SPIR 1.2/2.0 whereas we use the new SPIR-V which doesn't need the build flag, as long as we call clCreateProgramWithIL(). Passing the flag might fail if the device doesn't support the old SPIR even though it supports the new SPIR-V.

