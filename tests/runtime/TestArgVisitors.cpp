#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>

#include "SPIRVFuncInfo.hh"

int main() {
  // Test setup.

  std::vector<SPVArgTypeInfo> ArgInfo;

  // Arg 0: Simulate a pointer
  ArgInfo.emplace_back(
      SPVArgTypeInfo{SPVTypeKind::Pointer, SPVStorageClass::CrossWorkgroup, 8});

  // Arg 1: Simulate a hipTextureObject_t as emitted by HipTextureLowering.
  ArgInfo.emplace_back(
      SPVArgTypeInfo{SPVTypeKind::Image, SPVStorageClass::CrossWorkgroup, 8});
  ArgInfo.emplace_back(SPVArgTypeInfo{SPVTypeKind::Sampler,
                                      SPVStorageClass::UniformConstant, 8});

  // Arg 2: Simulate a POD.
  ArgInfo.emplace_back(
      SPVArgTypeInfo{SPVTypeKind::POD, SPVStorageClass::CrossWorkgroup, 4});

  // Arg 3: Simulate another hipTextureObject_t.
  ArgInfo.emplace_back(
      SPVArgTypeInfo{SPVTypeKind::Image, SPVStorageClass::CrossWorkgroup});
  ArgInfo.emplace_back(
      SPVArgTypeInfo{SPVTypeKind::Sampler, SPVStorageClass::UniformConstant});

  // Arg 4: Simulate another POD.
  ArgInfo.emplace_back(
      SPVArgTypeInfo{SPVTypeKind::POD, SPVStorageClass::CrossWorkgroup, 16});

  // Simulate dynamic shared memory pointer. HipDynMem.cpp inserts it
  // at the end of the paremeter list
  ArgInfo.emplace_back(
      SPVArgTypeInfo{SPVTypeKind::Pointer, SPVStorageClass::Workgroup, 8});

  SPVFuncInfo FI(ArgInfo);

  // Simulate client-side arguments.
  int a, b, c, d, e;
  std::vector<void *> ArgList{&a, &b, &c, &d, &e};

  // Test visitors.

  assert(FI.getNumClientArgs() == 5);
  assert(FI.getNumKernelArgs() == 8);

  unsigned ArgIdx = 0;
  FI.visitClientArgs([&](const SPVFuncInfo::ClientArg &Arg) {
    assert(Arg.Index == ArgIdx++);
    assert(Arg.Data == nullptr);
    // Arg.Kind is checked in the below.
  });

  ArgIdx = 0;
  FI.visitClientArgs(ArgList, [&](const SPVFuncInfo::ClientArg &Arg) {
    assert(Arg.Index == ArgIdx++);
    assert(Arg.Data == ArgList.at(Arg.Index));
    if (Arg.Index == 0)
      assert(Arg.Kind == SPVTypeKind::Pointer);
    else if (Arg.Index == 1)
      assert(Arg.Kind == SPVTypeKind::Pointer);
    // Skip sampler argument.
    else if (Arg.Index == 2)
      assert(Arg.Kind == SPVTypeKind::POD);
    else if (Arg.Index == 3)
      assert(Arg.Kind == SPVTypeKind::Pointer);
    // Skip sampler argument.
    else if (Arg.Index == 4)
      assert(Arg.Kind == SPVTypeKind::POD);
    // Skip workgroup pointer for dynamic shared pointer.
    else {
      assert(false && "Broken test.");
      exit(1);
    }
  });

  ArgIdx = 0;
  FI.visitKernelArgs([&](const SPVFuncInfo::KernelArg &Arg) {
    assert(Arg.Index == ArgIdx++);
    assert(Arg.Data == nullptr);
    // Arg.Kind is checked in the below.
  });

  ArgIdx = 0;
  FI.visitKernelArgs(ArgList, [&](const SPVFuncInfo::KernelArg &Arg) {
    assert(Arg.Index == ArgIdx++);

    if (Arg.Index == 0) {
      assert(Arg.Kind == SPVTypeKind::Pointer);
      assert(Arg.Data == ArgList.at(0));
    } else if (Arg.Index == 1) {
      assert(Arg.Kind == SPVTypeKind::Image);
      assert(Arg.Data == ArgList.at(1));
    } else if (Arg.Index == 2) {
      assert(Arg.Kind == SPVTypeKind::Sampler);
      assert(Arg.Data == ArgList.at(1));
    } else if (Arg.Index == 3) {
      assert(Arg.Kind == SPVTypeKind::POD);
      assert(Arg.Data == ArgList.at(2));
    } else if (Arg.Index == 4) {
      assert(Arg.Kind == SPVTypeKind::Image);
      assert(Arg.Data == ArgList.at(3));
    } else if (Arg.Index == 5) {
      assert(Arg.Kind == SPVTypeKind::Sampler);
      assert(Arg.Data == ArgList.at(3));
    } else if (Arg.Index == 6) {
      assert(Arg.Kind == SPVTypeKind::POD);
      assert(Arg.Data == ArgList.at(4));
    } else if (Arg.Index == 7) {
      assert(Arg.Kind == SPVTypeKind::Pointer);
      assert(Arg.isWorkgroupPtr());
      assert(Arg.Data == nullptr);
    } else {
      assert(false && "Broken test.");
      exit(1);
    }
  });

  return 0;
}
