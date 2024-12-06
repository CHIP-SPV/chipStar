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

#ifndef SRC_SPIRV_FUNCINFO_H
#define SRC_SPIRV_FUNCINFO_H

#include <map>j
#include <memory>
#include <vector>
#include <functional>
#include <string_view>

enum class SPVTypeKind : unsigned {
  Unknown = 0,

  // Kinds that may appear in both SPVFuncInfo::ClientArg and
  // SPVFuncInfo::KernelArg:
  POD,     // The type is a non-pointer, primitive value or an
           // aggregate passed-by-value.
  Pointer, // The type is a pointer of any storage class.

  // Kinds that may only appear in SPVFuncInfo::KernelArg.
  PODByRef, // Same as POD except the value is passed indirectly via
            // a device buffer.
  Image,    // The type is a image.
  Sampler,  // The type is a sample.

  // Should not appear in kernel parameter lists.
  Opaque, // The type is an unresolved, special SPIR-V type.
};

inline std::string_view to_string(SPVTypeKind kind) {
  switch (kind) {
    case SPVTypeKind::POD: return "POD";
    case SPVTypeKind::Pointer: return "Pointer";
    case SPVTypeKind::PODByRef: return "PODByRef";
    case SPVTypeKind::Image: return "Image";
    case SPVTypeKind::Sampler: return "Sampler";
    case SPVTypeKind::Opaque: return "Opaque";
    default: return "Unknown";
  }
}

// TODO: Redundant. Could use SPV::StorageClass instead.
enum class SPVStorageClass : unsigned {
  Private = 0,
  CrossWorkgroup = 1,
  UniformConstant = 2,
  Workgroup = 3,
  Unknown = 1000
};

struct SPVArgTypeInfo {
  SPVTypeKind Kind;
  SPVStorageClass StorageClass;
  size_t Size;

  bool isWorkgroupPtr() const {
    return Kind == SPVTypeKind::Pointer &&
           StorageClass == SPVStorageClass::Workgroup;
  }
};

class SPVFuncInfo {
  friend class SPIRVmodule;
  friend class SPIRVinst;

  std::vector<SPVArgTypeInfo> ArgTypeInfo_;

  /// Set to true if any argument is SPVTypeKind::PODByRef.
  bool HasByRefArgs_ = false;

public:
  /// A structure for argument info passed by the visitor methods.
  struct Arg : SPVArgTypeInfo {
    size_t Index;
    /// Argument data (an address to argument value). In case of
    /// isWorkgroupPtr()==true this member is nullptr.
    const void *Data;

    /// Return SPVArgTypeInfo::Kind as a string
    std::string_view getKindAsString() const;
  };
  struct KernelArg : public Arg {};
  // Represetns an Argument visible to the HIP client. This only
  // includes source code defined arguments (e.g. excludes samplers,
  // images and pointers with workgroup storage class).
  struct ClientArg : public Arg {};

  using ClientArgVisitor = std::function<void(const ClientArg &)>;
  using KernelArgVisitor = std::function<void(const KernelArg &)>;

  SPVFuncInfo() = default;
  SPVFuncInfo(const std::vector<SPVArgTypeInfo> &Info) : ArgTypeInfo_(Info) {}

  void visitClientArgs(ClientArgVisitor Fn) const;
  void visitClientArgs(void **ArgList, ClientArgVisitor Fn) const;
  void visitKernelArgs(KernelArgVisitor Fn) const;
  void visitKernelArgs(void **ArgList, KernelArgVisitor Fn) const;

  /// Return visible kernel argument count.
  ///
  /// The count only accounts arguments defined in the HIP source code.
  unsigned getNumClientArgs() const;

  /// Return actual kernel argument count (includes arguments not
  /// defined in HIP source code)
  unsigned getNumKernelArgs() const { return ArgTypeInfo_.size(); }

  /// Return true is any argument is passed via intermediate buffer.
  bool hasByRefArgs() const noexcept { return HasByRefArgs_; }

private:
  void visitClientArgsImpl(void **ArgList, ClientArgVisitor Fn) const;
  void visitKernelArgsImpl(void **ArgList, KernelArgVisitor Fn) const;
};

typedef std::map<int32_t, std::shared_ptr<SPVFuncInfo>> SPVFuncInfoMap;

#endif
