#ifndef HIP_COMMON_H
#define HIP_COMMON_H

#include <map>
#include <vector>
#include <stdint.h>
#include <string>

enum class OCLType : unsigned { POD = 0, Pointer = 1, Image = 2, Sampler = 3 };

enum class OCLSpace : unsigned {
  Private = 0,
  Global = 1,
  Constant = 2,
  Local = 3,
  Unknown = 1000
};

struct OCLArgTypeInfo {
  OCLType type;
  OCLSpace space;
  size_t size;
};

struct OCLFuncInfo {
  std::vector<OCLArgTypeInfo> ArgTypeInfo;
  OCLArgTypeInfo retTypeInfo;
};

typedef std::map<int32_t, OCLFuncInfo *> OCLFuncInfoMap;

typedef std::map<std::string, OCLFuncInfo *> OpenCLFunctionInfoMap;

bool parseSPIR(int32_t *stream, size_t numWords, OpenCLFunctionInfoMap &output);

#endif