/*
 * Copyright (c) 2021-23 chipStar developers
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

void _cl_print_str(__generic const char *S) {
  if (S == 0) {
    return;  /* Match AMD/nvidia: null %s prints nothing, format provides newline */
  }
  unsigned Pos = 0;
  char C;
  while ((C = S[Pos]) != 0) {
    printf("%c", C);
    ++Pos;
  }
}

/*
 * Prints a device-side assertion-failure message:
 *
 *   <file>:<line>: <function>: Device-side assertion `<assertion>' failed.
 *
 * Implemented here in OpenCL C (rather than directly in the HIP C++
 * __assert_fail in spirv_hip.hh) so the printf *format* string literals live
 * in the constant address space. A HIP C++ definition places the literals in
 * the generic address space, which forces the SPIR-V translator to emit
 * SPV_EXT_relaxed_printf_string_address_space -- an extension the Intel
 * compute runtime rejects at clBuildProgram. CUDA and ROCm likewise keep
 * assert message handling in their device libraries for this reason.
 *
 * The user-visible assertion/file/function strings are generic pointers, so
 * they are printed via _cl_print_str (one %c at a time with a constant format
 * string) instead of being passed to a %s conversion. Aborting is left to the
 * caller (HIP's abort(), handled by the HipAbort pass).
 */
void _cl_assert_fail_print(__generic const char *file, unsigned int line,
                           __generic const char *function,
                           __generic const char *assertion) {
  _cl_print_str(file);
  printf(":%u: ", line);
  _cl_print_str(function);
  printf(": Device-side assertion `");
  _cl_print_str(assertion);
  printf("' failed.\n");
}
