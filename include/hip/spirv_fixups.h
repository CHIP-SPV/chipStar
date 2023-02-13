/*
 * Copyright (c) 2023 CHIP-SPV developers
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */

// A header for applying fixups. To take effect, this header must be first to be
// included before anything else (before any include in user code, before
// all -include options).

#ifndef SPIRV_COMPILER_FIXUPS_H
#define SPIRV_COMPILER_FIXUPS_H

// A workaround for device side compilation when using -std=gnu++##. Clang
// incorrectly defines these macros which enable unsupported float128
// code in some libraries (e.g. in libstdc++).
//
// AFAIK, these macros leak from the host target. Clang bundles target specific
// macro defines for both the device and host target together and passes them to
// both the host and device compilation pipelines. SPIR-V target does not define
// the macros but x86_64 does.
#ifdef __HIP_DEVICE_COMPILE__
#undef __FLOAT128__
#undef __SIZEOF_FLOAT128__
#endif

#endif // SPIRV_COMPILER_FIXUPS_H
