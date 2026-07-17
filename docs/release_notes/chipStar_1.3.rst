********************************
Release Notes for chipStar 1.3
********************************

chipStar 1.3 is a major release with approximately 700 commits since 1.2.1.
The main user-visible changes are official CMake integration, macOS support,
broader LLVM and HIP compatibility, expanded HIP runtime API coverage, HIPRTC
improvements, and many correctness and performance fixes.

=====================
New Platform Support
=====================

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
macOS (ARM64 + x86)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

macOS is now a supported chipStar platform, including Apple Silicon CPUs via
PoCL. LLVM 21/22-based macOS builds are supported, and runtime fixes make
queue handling and unified-memory behavior work correctly on macOS.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Intel Arc B570 / GPU Kernel 6.8+
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Intel Arc B570 systems are supported with newer Linux kernels. Timestamp
handling was fixed on devices with 64-bit timestamps.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
ARM Mali GPU
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Device-side ``printf`` is now supported on Mali GPUs with the
``cl_arm_printf`` extension. Workarounds were added for a Mali driver deadlock
(flush queues before finish, ``clFlush`` for marker/callback events).

===================
Supported Libraries
===================

``install_chipstar.py`` now installs chipStar together with supported CHIP-SPV
library ports.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Platform-Independent Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* `rocPRIM <https://github.com/CHIP-SPV/rocPRIM>`_ - Parallel primitives

* `hipCUB <https://github.com/CHIP-SPV/hipCUB>`_ - CUB-like primitives for HIP

* `rocThrust <https://github.com/CHIP-SPV/rocThrust>`_ - Thrust parallel algorithms

* `rocRAND <https://github.com/CHIP-SPV/rocRAND>`_ - Random number generation

* `hipRAND <https://github.com/CHIP-SPV/hipRAND>`_ - HIP random number interface

* `rocSPARSE <https://github.com/CHIP-SPV/rocSPARSE>`_ (new) - Sparse matrix operations

* `hipSPARSE <https://github.com/CHIP-SPV/hipSPARSE>`_ (new) - HIP sparse matrix interface

* `hipMM <https://github.com/CHIP-SPV/hipMM>`_ (new) - HIP memory manager (RMM port)

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Intel MKL-Based Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* `H4I-MKLShim <https://github.com/CHIP-SPV/H4I-MKLShim>`_ - Intel MKL shim layer

* `H4I-HipBLAS <https://github.com/CHIP-SPV/H4I-HipBLAS>`_ - hipBLAS via Intel MKL

* `H4I-HipFFT <https://github.com/CHIP-SPV/H4I-HipFFT>`_ - hipFFT via Intel MKL

* `H4I-HipSOLVER <https://github.com/CHIP-SPV/H4I-HipSOLVER>`_ - hipSOLVER via Intel MKL

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Application Support
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* `libCEED <https://github.com/CEED/libCEED>`_

* `zero-rk <https://github.com/llnl/zero-rk>`_

* `GMSHPC <https://doi.org/10.1063/5.0005188>`_

* Multiple other applications that are not open source

==========================
LLVM and HIP Compatibility
==========================

* Added support for LLVM 20, LLVM 21, and LLVM 22.

* LLVM 22's integrated SPIR-V backend is detected automatically and is the
  default/preferred option.

* Added compatibility work for upcoming LLVM 23 / Clang offload driver changes.

* Updated the bundled HIP stack to HIP 7.2.0 (``chipStar-hip-7``).

* Fixed HIP 7 API signature changes, including ``hipMemcpyHtoD`` /
  ``hipMemcpyHtoDAsync``.

* Fixed HIPRTC API parameter types for compatibility with newer HIP headers.

=============================
Build and Project Integration
=============================

* chipStar is officially supported by CMake starting with **CMake 4.3.0**,
  making it easier for projects to use chipStar through CMake's HIP language
  support.

* New ``install_chipstar.py`` with unified install dir, ``--with-tests`` flag
  and ``rocPRIM`` prefix support. It now builds from ``cwd`` when run inside
  the repo.

* Fixed ``make install`` failure on macOS (hardcoded ``.so`` extension).

* Excluded ``spdlog`` from install to avoid downstream symbol conflicts;
  renamed the namespace to ``chipStar_spdlog``.

* Installed ``hip-lang-config.cmake`` for CMake HIP language support.

* Installed ``FindHIP.cmake``.

============
New Features
============

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Async Memory Allocation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implemented ``hipMallocAsync``, ``hipFreeAsync`` and
``hipMallocFromPoolAsync``.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Managed Memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Implemented ``hipMemAdvise`` and ``hipMemRangeGetAttribute(s)``.

* Implemented ``hipMemPrefetchAsync``.

* OpenCL: use ``clHostMemAllocINTEL`` for managed memory; added
  ``clEnqueueMigrateMemINTEL`` support.

* Level Zero / OpenCL: managed memory support reporting.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Device-Side ``malloc`` / ``free``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Implemented device-side ``__chip_malloc`` / ``__chip_free`` with C++ ``new`` /
``delete`` wrappers.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
HIP Graphs
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Fixed ``hipGraphAddDependencies`` / ``hipGraphRemoveDependencies`` to iterate
  ``from[i]`` → ``to[i]`` pairs correctly.

* Added null-pointer parameter validation across the ``hipGraph*`` API
  functions.

* Fixed ``getCaptureStatus()``, which was hardcoded to return ``None``.

* Fixed ``hipStreamWaitEvent`` incorrectly flipping stream capture status.

* Fixed ``hipStreamEndCapture`` returning ``IllegalState`` (401).

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Separate Compilation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Handle the ``-dc`` flag in ``hipcc`` for separate compilation workflows
  (#893).

* Support unbundling of static device libraries in the HIPSPV toolchain.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
HIPRTC Improvements
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Runtime compilation handles more SPIR-V constructs correctly.

* HIPRTC compilation-output caching support.

* Auto-include fp16 headers in HIPRTC.

* Device variable registration for compiled modules (constant memory in
  HIPRTC).

* Shell metacharacter escaping in ``hiprtcCompileProgram``.

* Fixed hipRTC compile error with Clang 22.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fp16 / Device Library
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Added float→half conversion functions with rounding modes.

* Added ``__device__`` / ``__host__`` decorators to the fp16 header.

* Fixed raw bit extraction and error messages in ``fp16_conversion.hpp``.

* Fixed ``__ocml_cvtrtn_f16_f32`` and other missing conversion functions.

* Added float/double ``atomicMin`` / ``atomicMax`` devicelib implementations.

~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
SPIR-V Compatibility
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

* Promote narrow integer kernel args to ``i32`` for SPIR-V conformance (#849).

* Inline kernel arg promotion to avoid the wrapper function pattern (#849).

* Improved linking behavior for programs that use atomics and ballot
  operations.

============
SYCL Interop
============

* Fixed ``hip_sycl_interop`` and ``hip_sycl_interop_no_buffers`` for the Level
  Zero backend with the MKL 2025 UR API.

==================
Notable Bug Fixes
==================

* Memory-copy validation is stricter and more compatible with HIP behavior,
  including invalid pointer handling and ``hipMemcpyDefault`` direction
  inference.

* Atomics and ballot intrinsics received several correctness fixes, including
  ``atomicMin`` / ``atomicMax`` on floating-point types, ``__chip_all()`` and
  ``__byte_perm``.

* Stream, event, and queue synchronization fixes address hangs and races seen
  with OpenCL, Level Zero, ARM Mali, and default-stream behavior.

* Module loading/unloading and backend selection are more robust, including
  fixes for default backend selection and ``hipModuleUnload``.

* C and CMake integration fixes improve mixed C/HIP projects, CMake HIP
  language support, and downstream package integration.

==================
Submodule Updates
==================

* HIP → 7.2.0 (``chipStar-hip-7``).

* ROCm-Device-Libs: dynamic datalayout from clang; ``irif.h`` type-punning fix.

* HIPCC: various fixes including ``-no-hip-rt``, LLVM 21, and shell
  metacharacter escaping.

* PoCL support updated to version 7.

===============
Full Changelog
===============

For the complete list of changes, see::

  git log v1.2.1..v1.3
