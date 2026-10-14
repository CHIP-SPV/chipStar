/*
 * Copyright (c) 2024-26 chipStar developers
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

#include "SPVReflection.hh"
#include "logging.hh"

#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <vector>

using InstWord = uint32_t;

namespace {
// Decodes the literal string starting at Words[Begin]; sets End past it.
std::string decodeString(const InstWord *Words, size_t Begin, size_t WC,
                         size_t *End = nullptr) {
  std::string S;
  size_t K = Begin;
  for (; K < WC; ++K) {
    InstWord X = Words[K];
    bool Done = false;
    for (int B = 0; B < 4 && !Done; ++B) {
      char C = static_cast<char>((X >> (8 * B)) & 0xff);
      if (C == 0)
        Done = true;
      else
        S.push_back(C);
    }
    if (Done)
      break;
  }
  if (End)
    *End = K + 1;
  return S;
}
} // namespace

// The SPIR-V backend lowers each HIP kernel for the shadervulkan environment
// to a GLCompute entry point whose arguments are the members of one push
// constant block, in order: a scalar its value, a pointer its byte offset into
// the storage buffer named "<kernel>.<argument number>[~<view>][.<argument
// name>]", one per type the kernel accesses it as; a null pointer passes 1 << 63.
bool tryAnalyzeVulkanReflection(const InstWord *Stream, size_t NumWords,
                                SPVModuleInfo &Output) {
  std::map<InstWord, std::string> Names;
  std::map<InstWord, uint32_t> Bindings;
  std::map<std::pair<InstWord, uint32_t>, uint32_t> MemberOffsets;
  std::map<InstWord, std::vector<InstWord>> StructMembers;
  std::map<InstWord, size_t> TypeSizes; // Scalars and vectors.
  std::map<InstWord, std::pair<InstWord, InstWord>> ArrayTypes; // elem, len.
  std::map<InstWord, InstWord> RuntimeArrayTypes;               // elem.
  std::map<InstWord, uint32_t> ArrayStrides;
  std::map<InstWord, uint64_t> Constants;
  std::map<InstWord, std::pair<uint32_t, InstWord>> PtrTypes; // SC, pointee.
  std::map<InstWord, std::pair<InstWord, uint32_t>> Vars;     // type, SC.
  struct EntryPoint {
    std::string Name;
    std::vector<InstWord> Interface;
  };
  std::vector<EntryPoint> EntryPoints;
  bool Logical = false;

  for (size_t I = 5; I < NumWords;) {
    const InstWord *W = &Stream[I];
    uint16_t WC = W[0] >> 16, Op = W[0] & 0xffff;
    if (WC == 0)
      return false;
    switch (Op) {
    case 14: // OpMemoryModel
      Logical = WC >= 3 && W[1] == 0;
      break;
    case 15: // OpEntryPoint
      if (WC >= 4 && W[1] == 5 /*GLCompute*/) {
        size_t End;
        EntryPoint EP;
        EP.Name = decodeString(W, 3, WC, &End);
        if (EP.Name.rfind("__hipspv_error: ", 0) == 0) {
          Output.BuildError += EP.Name.substr(strlen("__hipspv_error: ")) + "\n";
          // As for the unresolved imports of an OpenCL module.
          constexpr char Undef[] = "call to undefined function ";
          if (size_t P = EP.Name.find(Undef); P != std::string::npos)
            logWarn("Missing definition for '{}'",
                    EP.Name.substr(P + strlen(Undef)));
          break;
        }
        EP.Interface.assign(W + std::min<size_t>(End, WC), W + WC);
        EntryPoints.push_back(std::move(EP));
      }
      break;
    case 5: // OpName
      if (WC >= 3)
        Names[W[1]] = decodeString(W, 2, WC);
      break;
    case 71: // OpDecorate
      if (WC >= 4 && W[2] == 33 /*Binding*/)
        Bindings[W[1]] = W[3];
      if (WC >= 4 && W[2] == 6 /*ArrayStride*/)
        ArrayStrides[W[1]] = W[3];
      break;
    case 43: // OpConstant
      if (WC >= 4)
        Constants[W[2]] = W[3];
      break;
    case 28: // OpTypeArray
      if (WC >= 4)
        ArrayTypes[W[1]] = {W[2], W[3]};
      break;
    case 29: // OpTypeRuntimeArray
      if (WC >= 3)
        RuntimeArrayTypes[W[1]] = W[2];
      break;
    case 72: // OpMemberDecorate
      if (WC >= 5 && W[3] == 35 /*Offset*/)
        MemberOffsets[{W[1], W[2]}] = W[4];
      break;
    case 21: // OpTypeInt
    case 22: // OpTypeFloat
      if (WC >= 3)
        TypeSizes[W[1]] = W[2] / 8;
      break;
    case 23: // OpTypeVector
      if (WC >= 4 && TypeSizes.count(W[2]))
        TypeSizes[W[1]] = TypeSizes[W[2]] * W[3];
      break;
    case 30: // OpTypeStruct
      StructMembers[W[1]].assign(W + 2, W + WC);
      break;
    case 32: // OpTypePointer
      if (WC >= 4)
        PtrTypes[W[1]] = {W[2], W[3]};
      break;
    case 59: // OpVariable
      if (WC >= 4)
        Vars[W[2]] = {W[1], W[3]};
      break;
    }
    I += WC;
  }
  if (!Logical || EntryPoints.empty())
    return false;

  // Byte size of a push constant member: a struct or array passed by value
  // ends at its last member, which is all the kernel reads.
  std::function<size_t(InstWord)> sizeOf = [&](InstWord T) -> size_t {
    if (auto S = TypeSizes.find(T); S != TypeSizes.end())
      return S->second;
    TypeSizes[T] = 0; // Ends a malformed type cycle.
    size_t Size = 0;
    if (auto A = ArrayTypes.find(T); A != ArrayTypes.end()) {
      size_t N = Constants[A->second.second];
      auto Stride = ArrayStrides.find(T);
      Size = N * (Stride != ArrayStrides.end() ? Stride->second
                                               : sizeOf(A->second.first));
    } else if (auto S = StructMembers.find(T); S != StructMembers.end()) {
      for (uint32_t M = 0; M < S->second.size(); ++M)
        Size = std::max<size_t>(Size, MemberOffsets[{T, M}] +
                                          sizeOf(S->second[M]));
    }
    return TypeSizes[T] = Size;
  };
  auto storageClassOf = [&](InstWord Var) -> uint32_t {
    auto V = Vars.find(Var);
    return V == Vars.end() ? ~0u : V->second.second;
  };
  for (const EntryPoint &EP : EntryPoints) {
    // The push constant block: the one in the interface, else (when every
    // argument is unused) the variable named after the kernel.
    InstWord PC = 0;
    for (InstWord V : EP.Interface)
      if (storageClassOf(V) == 9 /*PushConstant*/)
        PC = V;
    if (!PC)
      for (auto &[Id, Ty] : Vars)
        if (Ty.second == 9 && Names[Id].rfind(EP.Name + ".pc", 0) == 0)
          PC = Id;
    // Or, beyond 128 bytes, the elements of the storage buffer
    // "<kernel>.args".
    InstWord Struct = PC ? PtrTypes[Vars[PC].first].second : 0;
    int ArgsBinding = -1;
    for (auto &[Id, Ty] : Vars)
      if (!PC && Ty.second == 12 && Names[Id] == EP.Name + ".args" &&
          Bindings.count(Id)) {
        const std::vector<InstWord> &Block =
            StructMembers[PtrTypes[Ty.first].second];
        if (!Block.empty())
          Struct = RuntimeArrayTypes[Block[0]];
        ArgsBinding = static_cast<int>(Bindings[Id]);
      }
    std::vector<SPVArgTypeInfo> Args;
    if (Struct) {
      const std::vector<InstWord> &Members = StructMembers[Struct];
      for (uint32_t M = 0; M < Members.size(); ++M) {
        SPVArgTypeInfo Ti{};
        Ti.Kind = SPVTypeKind::POD;
        Ti.StorageClass = SPVStorageClass::Private;
        Ti.Size = sizeOf(Members[M]);
        auto Off = MemberOffsets.find({Struct, M});
        Ti.PushConstOffset = Off == MemberOffsets.end() ? -1 : Off->second;
        Args.push_back(Ti);
      }
    }
    // Storage buffers mark the pointer arguments.
    for (auto &[Id, Ty] : Vars) {
      if (Ty.second != 12 /*StorageBuffer*/)
        continue;
      const std::string &N = Names[Id];
      std::string Prefix = EP.Name + ".";
      if (N.rfind(Prefix, 0) != 0 || !Bindings.count(Id))
        continue;
      unsigned long ArgNo;
      size_t NumLen;
      try {
        ArgNo = std::stoul(N.substr(Prefix.size()), &NumLen);
      } catch (...) {
        continue;
      }
      if (ArgNo >= Args.size())
        continue;
      // Skip a "~<view>" suffix: "<kernel>.<n>~<view>.__chip_dg_<global>"
      // holds a device global.
      size_t P = Prefix.size() + NumLen;
      if (N.compare(P, 1, "~") == 0)
        P = std::min(N.find('.', P), N.size());
      std::string DG = std::string(".") + ChipDevGlobalArgPrefix;
      std::string AF = std::string(".") + ChipArgFieldPrefix;
      if (N.compare(P, DG.size(), DG) == 0) {
        Args[ArgNo].Kind = SPVTypeKind::DeviceGlobal;
        Args[ArgNo].DevGlobalName = N.substr(P + DG.size());
      } else if (N.compare(P, AF.size(), AF) == 0) {
        // Bound like a device global, from a pointer the client passes.
        Args[ArgNo].Kind = SPVTypeKind::DeviceGlobal;
        Args[ArgNo].DevGlobalName = N.substr(P + 1);
      } else
        Args[ArgNo].Kind = SPVTypeKind::Pointer;
      Args[ArgNo].StorageClass = SPVStorageClass::CrossWorkgroup;
      Args[ArgNo].Binding = static_cast<int>(Bindings[Id]);
    }
    logDebug("Reflect kernel='{}': {} arguments", EP.Name, Args.size());
    auto FInfo = std::make_shared<SPVFuncInfo>(Args);
    FInfo->ArgsBinding = ArgsBinding;
    Output.FuncInfoMap.emplace(EP.Name, FInfo);
  }
  Output.HasNoIGBAs = true;
  return !Output.FuncInfoMap.empty();
}
