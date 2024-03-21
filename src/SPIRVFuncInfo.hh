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

#include <map>
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
  PODByRef, // Same as PODB except the value is passed in an
            // intermediate device buffer and a pointer to its
            // location given to the kernel.
  Image,    // The type is a image.
  Sampler,  // The type is a sample.

  // Should not appear in kernel parameter lists.
  Opaque, // The type is an unresolved, special SPIR-V type.
};

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

  /// Spilled argument annotations represented as pairs of argument
  /// index (key) and argument size (value).
  std::map<uint16_t, uint16_t> SpilledArgs_;

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
  void visitClientArgs(const std::vector<void *> &ArgList,
                       ClientArgVisitor Fn) const;
  void visitKernelArgs(KernelArgVisitor Fn) const;
  void visitKernelArgs(const std::vector<void *> &ArgList,
                       KernelArgVisitor Fn) const;

  /// Return visible kernel argument count.
  ///
  /// The count only accounts arguments defined in the HIP source code.
  unsigned getNumClientArgs() const;

  /// Return actual kernel argument count (includes arguments not
  /// defined in HIP source code)
  unsigned getNumKernelArgs() const { return ArgTypeInfo_.size(); }

  /// Return true is any argument is passed via intermediate buffer.
  bool hasByRefArgs() const { return SpilledArgs_.size(); }

private:
  void visitClientArgsImpl(const std::vector<void *> &ArgList,
                           ClientArgVisitor Fn) const;
  void visitKernelArgsImpl(const std::vector<void *> &ArgList,
                           KernelArgVisitor Fn) const;
  bool isSpilledArg(unsigned KernelArgIndex) const;
  unsigned getSpilledArgSize(unsigned KernelArgIndex) const;
};

typedef std::map<int32_t, std::shared_ptr<SPVFuncInfo>> SPVFuncInfoMap;

#endif
