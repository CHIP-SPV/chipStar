/*
 * Copyright (c) 2021-22 chipStar developers
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

#include "chipStarConfig.hh"
#include "SPIRVFuncInfo.hh"

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

using SPVFunctionInfoMap = std::map<std::string, std::shared_ptr<SPVFuncInfo>>;

struct SPVModuleInfo {
  SPVFunctionInfoMap FuncInfoMap;

  /// Set to true if the module is known not to have indirect global
  /// buffer accesses (IGBA) in any kernel.
  bool HasNoIGBAs = false;
};

// Processing done before analysis.
bool preprocessSPIRV(const char *Bytes, size_t NumBytes,
                     bool PreventNameDemangling, std::vector<uint32_t> &Dst);
bool analyzeSPIRV(uint32_t *Stream, size_t NumWords, SPVModuleInfo &ModuleInfo);
// Processing done after analysis.
bool postprocessSPIRV(std::vector<uint32_t> &Binary);

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
/// The name of a single combined shadow kernel that initializes ALL
/// host-accessible program-scope variables in one launch. Emitted (in the
/// program-scope-globals lowering) instead of per-variable init kernels to
/// avoid O(N) single-work-item kernel launches. See issue #582.
constexpr char ChipVarInitAllName[] = "__chip_var_init_all";
/// Zero-initialized program-scope variables of at least this size (in bytes)
/// are initialized by a host-issued device fill (DMA) instead of by the init
/// shadow kernel. Below this size a fill is slower than just letting the
/// combined init kernel store the zeros (measured on Intel Arc B570: a 4 KiB
/// grid-stride kernel fill is on par with memFillAsync, while per-variable
/// host fills do not amortize -- 256 x 8 B variables cost 1782 us as
/// individual fills vs 99 us in one combined kernel).
///
/// This constant is shared by llvm_passes/HipGlobalVariables.cpp (which
/// decides which variables to leave out of the init kernel) and by the runtime
/// (which issues the fills), so the two cannot disagree.
constexpr size_t ChipVarFillThreshold = 65536;
/// A structure to where properties of a device variable are written.
/// CHIPVarInfo[0]: Size in bytes.
/// CHIPVarInfo[1]: Requested alignment.
/// CHIPVarInfo[2]: Initializer kind. A TRI-STATE:
///   0 = no initializer.
///   1 = has an initializer which the init shadow kernel applies.
///   2 = zero initializer of at least ChipVarFillThreshold bytes: the init
///       shadow kernel does NOT touch the variable, the runtime fills it
///       with zeros directly instead.
/// Any other non-zero value must be treated as 1 by the runtime.
///
/// NOTE: this slot deliberately stays a tri-state instead of the array growing
/// a fourth element. The HIPRTC on-disk cache (~/.cache/chipStar/hiprtc, see
/// src/spirv_hiprtc.cc) keys SPIR-V on nothing derived from the pass plugin, so
/// after a chipStar rebuild a module produced by an OLD pass can be handed to a
/// NEW runtime. With a tri-state such a module reports 1 and is correctly
/// treated as kernel-initialized; a fourth slot would instead be read out of
/// uninitialized device memory.
using CHIPVarInfo = int64_t[3];

/// The name of the shadow kernel responsible for resetting host-inaccessible
/// global device variables (e.g. static local variables in device code).
constexpr char ChipNonSymbolResetKernelName[] = "__chip_reset_non_symbols";

/// The prefix for global-scope variables in SPIR-V modules for carrying
/// information about "spilled" arguments
///
/// see HipKernelArgSpiller.cpp for details. Full name of such
/// variables is '<ChipSpilledArgsVarPrefix><kernel-name>'
constexpr char ChipSpilledArgsVarPrefix[] = "__chip_spilled_args_";

/// The prefix for global-scope annotation variables recording which device
/// globals feed a kernel's implicit trailing DeviceGlobal arguments
/// (globals-as-kernel-args lowering, used when program-scope globals are
/// disabled). The variable '<ChipGVarArgPrefix><kernel-name>' holds the
/// NUL-separated original global names in trailing-argument order. See
/// HipGlobalVariables.cpp for details.
constexpr char ChipGVarArgPrefix[] = "__chip_gvararg_";

/// The name of a global variable which indicates, when non-zero, if
/// the abort() function was called by a kernel.
constexpr char ChipDeviceAbortFlagName[] = "__chipspv_abort_called";

/// The name of a global variable which is the device heap.
constexpr char ChipDeviceHeapName[] = "__chipspv_device_heap";

#endif
