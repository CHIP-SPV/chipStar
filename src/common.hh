/*
 * Copyright (c) 2021-22 CHIP-SPV developers
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

#ifndef HIP_COMMON_HH
#define HIP_COMMON_HH

#include "Config.hh"

#include <map>
#include <set>
#include <vector>
#include <stdint.h>
#include <string>
#include <memory>
#include <unordered_set>
#include <utility>
#include <sstream>
#include <algorithm>
#include <iostream>
#include <mutex>
#include <queue>
#include <stack>

/// The implementation of ihipEvent_t. The CHIPEvent class inherits this so
/// ihipEvent_t pointers may carry CHIPEvent instances.
struct ihipEvent_t {};
struct ihipCtx_t {};
struct ihipStream_t {};
struct ihipModule_t {};
struct ihipModuleSymbol_t {};
// struct hipFunction_t {};

enum class OCLType : unsigned {
  POD,
  Pointer,
  Image,
  Sampler,
  Opaque,
};

enum class OCLSpace : unsigned {
  Private = 0,
  Global = 1,
  Constant = 2,
  Local = 3,
  Unknown = 1000
};

struct OCLArgTypeInfo {
  OCLType Type;
  OCLSpace Space;
  size_t Size;
};

struct OCLFuncInfo {
  std::vector<OCLArgTypeInfo> ArgTypeInfo;
  OCLArgTypeInfo RetTypeInfo;
};

typedef std::map<int32_t, std::shared_ptr<OCLFuncInfo>> OCLFuncInfoMap;

typedef std::map<std::string, std::shared_ptr<OCLFuncInfo>>
    OpenCLFunctionInfoMap;

bool filterSPIRV(const char *Bytes, size_t NumBytes, std::string &Dst);
bool parseSPIR(int32_t *Stream, size_t NumWords,
               OpenCLFunctionInfoMap &FuncInfoMap);

/// A prefix given to lowered global scope device variables.
constexpr char ChipVarPrefix[] = "__chip_var_";
/// A prefix used for a shadow kernel used for querying device
/// variable properties.
constexpr char ChipVarInfoPrefix[] = "__chip_var_info_";
/// A prefix used for a shadow kernel used for binding storage to
/// device variables.
constexpr char ChipVarBindPrefix[] = "__chip_var_bind_";
/// A prefix used for a shadow kernel used for initializing device
/// variables.
constexpr char ChipVarInitPrefix[] = "__chip_var_init_";
/// A structure to where properties of a device variable are written.
/// CHIPVarInfo[0]: Size in bytes.
/// CHIPVarInfo[1]: Requested alignment.
/// CHIPVarInfo[2]: Non-zero if variable has initializer. Otherwise zero.
using CHIPVarInfo = int64_t[3];

/// The name of the shadow kernel responsible for resetting host-inaccessible
/// global device variables (e.g. static local variables in device code).
constexpr char ChipNonSymbolResetKernelName[] = "__chip_reset_non_symbols";

#endif
