/*
 * Copyright (c) 2023 CHIP-SPV developers
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

enum class SPVTypeKind : unsigned {
  POD,
  Pointer,
  Image,
  Sampler,
  Opaque,
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
};

struct SPVFuncInfo {
  std::vector<SPVArgTypeInfo> ArgTypeInfo;
};

typedef std::map<int32_t, std::shared_ptr<SPVFuncInfo>> SPVFuncInfoMap;

typedef std::map<std::string, std::shared_ptr<SPVFuncInfo>>
    OpenCLFunctionInfoMap;

#endif
