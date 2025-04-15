// Copyright (c) 2024 chipStar developers
//
// Part of the chipStar project, under the MIT License.
// See https://github.com/CHIP-SPV/chipStar/blob/main/LICENSE for license
// information.
// SPDX-License-Identifier: MIT

// Check indirect global buffer access (IGBA) detection.
#include <hip/hip_runtime.h>
#include "CHIPBackend.hh"
#include <iostream>

__global__ void k(uintptr_t Arg) {
  auto *Ptr = reinterpret_cast<int *>(Arg);
  *Ptr = 123;
}

int main() {
  auto *ChipKernel = Backend->getActiveDevice()->findKernel(
      HostPtr(reinterpret_cast<const void *>(k)));
  // HasNoIGBAs should be false here.
  std::cout << "PASSED\n";
  return ChipKernel->getModule()->getInfo().HasNoIGBAs;
}
