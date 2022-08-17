#ifndef HIP_COMMON_H
#define HIP_COMMON_H

#include <map>
#include <vector>
#include <stdint.h>
#include <string>
#include <memory>

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
