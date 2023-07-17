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

// Kernel argument visitors
//
// Kernel argument visitors methods provides a way to iterate over kernel
// arguments from both the HIP client point of view and the the actual, compiled
// kernel point of view and access the corresponding kernel argument value from
// an argument list fed by the hipLaunchKernel().
//
// The actual kernel may have some arguments modified, respect to the
// original source, and some new ones. For example:
//
//  __global__ void aKernel(
//    |<- Client-visible kernel args ->|<- Actual kernel args       ->|
//    |   visitClientArgs(...)         |   visitKernelArgs(...)       |
//    float *a,             [Pointer] --> [Pointer],  ArgList[0]
//    hipTextureObject_t b, [Pointer] --> [Image],    ArgList[1]  *1
//                                    --> [Sampler],  ArgList[1]  *1
//    float c,              [POD]     --> [POD],      ArgList[2]
//    hipTextureObject_t d, [Pointer] --> [Image],    ArgList[3]  *1
//                                    --> [Sampler],  ArgList[3]  *1
//    float e,              [POD]     --> [POD],      ArgList[4]
//    BigStruct f)          [POD]     --> [PODByRef], ArgList[5]  *3
//
//                                    --> [Pointer, isWorkgroupPtr()==true] *2
//                            ^- SPVTypeKind -^
//    {
//      extern __shared__ int smem[]; // *2
//      ...
//    }
//
//  uint16_t __chip_spilled_args_[] = {8, sizeof(BigStruct)}; // *3
//
//  *1: Emitted by HipTextureLowering.cpp.
//  *2: Emitted by HipDynMem.cpp when dynamic shared memory objects are present
//      in the kernel.
//  *3: Emitted by HipKernelArgSpiller.cpp. Instead of being passed directly,
//      a device pointer, which points to a device buffer having the argument
//      value, is passed instead at the argument position.

#include "SPIRVFuncInfo.hh"

#include "logging.hh"

#include <cassert>

std::string_view SPVFuncInfo::Arg::getKindAsString() const {
  switch (Kind) {
  default:
    assert(false && "Missing TypeKind to string conversion!");
    // FALLTHROUGH
  case SPVTypeKind::Unknown:
    return "Unknown";
  case SPVTypeKind::POD:
    return "POD";
  case SPVTypeKind::PODByRef:
    return "PODByRef";
  case SPVTypeKind::Pointer:
    return "Pointer";
  case SPVTypeKind::Image:
    return "Image";
  case SPVTypeKind::Sampler:
    return "Sampler";
  }
}

bool SPVFuncInfo::isSpilledArg(unsigned KernelArgIndex) const {
  return SpilledArgs_.count(KernelArgIndex);
}

unsigned SPVFuncInfo::getSpilledArgSize(unsigned KernelArgIndex) const {
  assert(isSpilledArg(KernelArgIndex));
  assert(KernelArgIndex < (1u << 16u) && "Definitely out-of-bounds");
  return SpilledArgs_.at(static_cast<uint16_t>(KernelArgIndex));
}

/// Client side kernel argument visitor.
void SPVFuncInfo::visitClientArgsImpl(const std::vector<void *> &ClientArgList,
                                      ClientArgVisitor Visitor) const {

  unsigned ArgListIndex = 0;
  for (unsigned KernelArgIdx = 0; KernelArgIdx < ArgTypeInfo_.size();
       KernelArgIdx++) {
    const auto &ArgTI = ArgTypeInfo_[KernelArgIdx];
    auto ArgKind = ArgTI.Kind;
    auto ArgSize = ArgTI.Size;

    if (isSpilledArg(KernelArgIdx)) {
      assert(ArgKind != SPVTypeKind::Image && ArgKind != SPVTypeKind::Sampler &&
             "Impossible arg kind to spill!");
      ArgKind = SPVTypeKind::POD;
      ArgSize = getSpilledArgSize(KernelArgIdx);
    }

    // Additional argument created by the texture lowering pass.
    if (ArgKind == SPVTypeKind::Sampler)
      continue;
    // Dynamic shared memory which is passed in hipLaunchKernel() or
    // <<<>>>-syntax - not in a kernel parameter list.
    if (ArgTI.isWorkgroupPtr())
      continue;

    // Map kernel argument types to types as defined in HIP source code.
    if (ArgKind == SPVTypeKind::Image)
      // Image argument replaced hipTextureObject_t argument.
      ArgKind = SPVTypeKind::Pointer;

    auto *ArgData =
        ClientArgList.empty() ? nullptr : ClientArgList[ArgListIndex];

    // Clang generated argument list should not have nullptrs in it.
    assert((ClientArgList.empty() || ArgData) &&
           "nullptr in the argument list");

    ClientArg CArg{
        {{ArgKind, ArgTI.StorageClass, ArgSize}, ArgListIndex, ArgData}};
    Visitor(CArg);
    ArgListIndex++;
  }
}

/// Visit client-visible kernel arguments
void SPVFuncInfo::visitClientArgs(const std::vector<void *> &ClientArgList,
                                  ClientArgVisitor Visitor) const {
  assert(ClientArgList.size() == getNumClientArgs());
  visitClientArgsImpl(ClientArgList, Visitor);
}

/// Visit client-visible kernel arguments without the argument value
/// (Arg::Data will be nullptr).
void SPVFuncInfo::visitClientArgs(ClientArgVisitor Visitor) const {
  visitClientArgsImpl(std::vector<void *>(), Visitor);
}

void SPVFuncInfo::visitKernelArgsImpl(const std::vector<void *> &ClientArgList,
                                      KernelArgVisitor Visitor) const {
  unsigned ArgIndex = 0;
  unsigned ArgListIndex = 0;
  for (const auto &ArgTI : ArgTypeInfo_) {
    auto ArgKind = ArgTI.Kind;
    auto ArgSize = ArgTI.Size;

    if (isSpilledArg(ArgIndex)) {
      assert(ArgKind != SPVTypeKind::Image && ArgKind != SPVTypeKind::Sampler &&
             "Impossible arg kind to spill!");
      ArgKind = SPVTypeKind::PODByRef;
      ArgSize = getSpilledArgSize(ArgIndex);
      logDebug("*** Arg {} spill size: {}", ArgIndex, ArgSize); // DEBUG
    }

    // Sampler is additional argument generated by HipTextureLowering
    // pass and it appears after SPVTypeKind::Image argument. Pass
    // the same argument that was passed for the image argument
    // (hipTextureObject_t*);
    if (ArgKind == SPVTypeKind::Sampler)
      ArgListIndex--;

    const void *ArgData = nullptr;
    if (!ClientArgList.empty() && !ArgTI.isWorkgroupPtr()) {
      ArgData = ClientArgList[ArgListIndex];

      // Clang geerated  argument list should not have nullptrs in it.
      assert(ArgData && "nullptr in the argument list");
    }

    KernelArg KArg{{{ArgKind, ArgTI.StorageClass, ArgSize}, ArgIndex, ArgData}};
    Visitor(KArg);

    ArgIndex++;
    ArgListIndex++;
  }
}

// Visit kernel arguments
void SPVFuncInfo::visitKernelArgs(const std::vector<void *> &ClientArgList,
                                  KernelArgVisitor Visitor) const {
  assert(ClientArgList.size() == getNumClientArgs());
  visitKernelArgsImpl(ClientArgList, Visitor);
}

/// Visit kernel arguments without argument list (Arg::Data will be nullptr)
void SPVFuncInfo::visitKernelArgs(KernelArgVisitor Visitor) const {
  visitKernelArgsImpl(std::vector<void *>(), Visitor);
}

/// Return HIP user visible kernel argument count.
unsigned SPVFuncInfo::getNumClientArgs() const {
  unsigned Count = getNumKernelArgs();
  for (const auto &ArgTI : ArgTypeInfo_) {
    auto ArgKind = ArgTI.Kind;
    Count -= ArgKind == SPVTypeKind::Sampler || ArgTI.isWorkgroupPtr();
  }
  return Count;
}
