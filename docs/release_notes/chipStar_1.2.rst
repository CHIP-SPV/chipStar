********************************
Release Notes for chipStar 1.2
********************************

This release brings significant stability and performance improvements,
enhanced support for CUDA, new HIP/ROCm library ports and integrations for
hipBLAS, hipFFT and hipRAND/rocRAND. It also includes initial testing of
running HIP/CUDA applications on RISC-V.

================
Tested Platforms
================

* Intel, AMD CPUs via Intel Compute Runtime

* Intel GPUs via Neo i915 driver

* ARM Mali GPUs (Quartz64 SBC)

* RISC-V (Starfive Visionfive 2 SBC Debian, experimental)

* AMD GPUs via `rusticl <https://docs.mesa3d.org/rusticl>`_ (exploratory work)

==================
Major New Features
==================

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Introduced ``cucc``, a Drop-In Replacement for ``nvcc``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``cucc`` enables direct compilation of CUDA sources. An ``nvcc`` softlink is
also installed, allowing CUDA sources to be compiled without making any
changes to the build system.

The CUDA headers were adjusted to improve compatibility with CUDA sources,
including a dummy ``cublas_v2.h`` header to prevent conflicts with system
headers.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Enhanced OpenCL Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Added support for devices featuring the ``cl_ext_buffer_device_address``
extension, improving memory management capabilities.

Queue profiling was optimized: the OpenCL backend now uses non-profiling
queues by default and switches to profiling queues only when needed,
resulting in performance improvements.

Various other performance optimizations were made.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Rebased to HIP 6.x
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The codebase was updated to be compatible with HIP 6.x, and hip-tests was
updated to match.

========================
Library Support Changes
========================

* hipBLAS integration: introduced the ``CHIP_BUILD_HIPBLAS`` option to enable
  building hipBLAS.

* hipFFT integration: introduced the ``CHIP_BUILD_HIPFFT`` option to enable
  building hipFFT.

* rocRAND port: https://github.com/CHIP-SPV/rocRAND

==============
Major Bugfixes
==============

* Fixed out-of-memory (OOM) errors in the Level Zero backend: memory leaks were
  fixed and resource management improved to prevent OOM errors during heavy
  workloads.

* Improved thread safety in the Level Zero backend by implementing mutexes and
  synchronization mechanisms.

=============================
Update Release chipStar 1.2.1
=============================

This minor release adds some fixes and performance improvements, most notably
module caching.

* Add JIT timings (#940).

* Integrate hipSOLVER (#941).

* Remove stl sycl header include (#942).

* Module caching (#943).

* Fences fix (#944).

* Integrate HIPCC macro fix (#946).

* Print JIT logs to Info, always (#948).

* Use Level Zero copy queues (#949).

* Prune known_failures.yaml & arg checks (#950).

* Support specifying include directories (``-I``) through hipRTC (#951).

* Implement a missing atomicMax (#953).

* JIT flags append instead of override (#954).
