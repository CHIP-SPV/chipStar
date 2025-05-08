// Copyright (c) 2024 chipStar developers
//
// Part of the chipStar project, under the MIT License.
// See https://github.com/CHIP-SPV/chipStar/blob/main/LICENSE for license
// information.
// SPDX-License-Identifier: MIT

// Check we detect that the device code does not have indirect global
// buffer accesses.
#include <hip/hip_runtime.h>
#include "CHIPBackend.hh"
#include <iostream>

__global__ void k(int *Out) { *Out = 123; }

int main() {
  auto *ChipKernel = Backend->getActiveDevice()->findKernel(
      HostPtr(reinterpret_cast<const void *>(k)));
  // HasNoIGBAs should be true here.
  if (ChipKernel->getModule()->getInfo().HasNoIGBAs) {
    std::cout << "PASSED\n";
  } else {
    std::cout << "FAILED\n";
  }
  return 0;
}
