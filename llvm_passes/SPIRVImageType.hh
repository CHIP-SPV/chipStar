//===- SPIRVImageType.cc -------------------------------------------------===//
//
// copied from clang/lib/CodeGen/Targets/SPIR.cpp
//
// Originally part of the LLVM Project, under the Apache License v2.0 with
// LLVM Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

enum ImageAccessQualifier : unsigned { AQ_ro = 0, AQ_wo = 1, AQ_rw = 2 };

namespace llvm {
  class Type;
  class LLVMContext;
  class StringRef;
}


/// Construct a SPIR-V target extension type for the given OpenCL image type.
llvm::Type *getSPIRVImageType(llvm::LLVMContext &Ctx, llvm::StringRef BaseType,
                                     llvm::StringRef OpenCLName,
                                     unsigned ImageAccessQualifier);
