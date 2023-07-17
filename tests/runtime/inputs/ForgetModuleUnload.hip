/*
 * Copyright (c) 2023 chipStar developers
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

#include <hip/hip_runtime_api.h>
#include <fstream>
#include <vector>

int main(int ArgC, char *ArgV[]) {
  if (ArgC != 2)
    return 2;
  // Loaded binary is created with hiprtc, binary content does not matter as
  // long as it's valid.
  std::ifstream File(ArgV[1], std::ios::binary | std::ios::ate);
  if (!File)
    return 3;
  auto Size = File.tellg();
  File.seekg(0, std::ios::beg);
  std::vector<char> ModuleBin(Size);
  if (!File.read(ModuleBin.data(), Size))
    return 4;

  hipModule_t Module;
  if (hipModuleLoadData(&Module, ModuleBin.data()) != hipSuccess)
    return 5;
  // Return without hipModuleUnload(Module) call.
  return 0;
}
