/*
 * Copyright (c) 2022 chipStar developers
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

#include "Utils.hh"

#include "logging.hh"

#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>

#ifdef __APPLE__
#include <dlfcn.h>
#else
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <link.h>
#endif

bool isConvertibleToInt(const std::string &str) {
  try {
    std::stoi(str);
    return true;
  } catch (const std::invalid_argument &) {
    return false;
  } catch (const std::out_of_range &) {
    return false;
  }
}

/// Read an environment variable and return its value as a string.
bool readEnvVar(std::string EnvVar, std::string &Value, bool Lower) {
  const char *EnvVarIn = std::getenv(EnvVar.c_str());
  if (EnvVarIn == nullptr)
    return false;
  Value = std::string(EnvVarIn);
  if (Lower)
    std::transform(Value.begin(), Value.end(), Value.begin(),
                   [](unsigned char Ch) { return std::tolower(Ch); });

  logDebug("readEnvVar value picked up form env: {} = {}", EnvVar, Value);
  return true;
}

std::string generateShortHash(std::string_view input, size_t length) {
  std::hash<std::string_view> hasher;
  std::size_t hashValue = hasher(input);

  std::stringstream ss;
  ss << std::hex << std::setw(length) << std::setfill('0') << hashValue;

  return ss.str().substr(0, length);
}

/// Dump the SPIR-V to a file
///
/// On success return the path to the file.
std::optional<fs::path> dumpSpirv(std::string_view Spirv) {
  std::string HashSum = generateShortHash(Spirv, 6);
  std::string FileName = "hip-spirv-" + HashSum + ".spv";
  std::ofstream SpirvFile(FileName, std::ios::binary);
  if (!SpirvFile) {
    std::cerr << "Error: Could not open file " << FileName << " for writing"
              << std::endl;
    return std::nullopt;
  }

  SpirvFile.write(Spirv.data(), Spirv.size());
  SpirvFile.close();
  return FileName;
}

/// Dump the SPIR-V to a file with a descriptive name
///
/// On success return the path to the file.
std::optional<fs::path> dumpSpirv(std::string_view Spirv, std::string_view Name) {
  std::string HashSum = generateShortHash(Spirv, 6);
  std::string FileName = "hip-spirv-" + std::string(Name) + "-" + HashSum + ".spv";
  std::ofstream SpirvFile(FileName, std::ios::binary);
  if (!SpirvFile) {
    std::cerr << "Error: Could not open file " << FileName << " for writing"
              << std::endl;
    return std::nullopt;
  }

  SpirvFile.write(Spirv.data(), Spirv.size());
  SpirvFile.close();
  return FileName;
}

/// Returns true if the hipcc can be executed by the user.
static bool canExecuteHipcc(const fs::path &HipccPath) {
  if (!fs::exists(HipccPath))
    return false;

  auto Cmd = std::string("sh -c '\"") + HipccPath.string() +
             "\" --version >/dev/null 2>&1'";
  return std::system(Cmd.c_str()) == 0;
}

std::string getRandomString(size_t Length) {
  constexpr std::string_view CharacterSet =
      "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

  std::random_device RandomDevice;
  std::mt19937 Generator(RandomDevice());
  std::uniform_int_distribution<> Distribution(0, CharacterSet.size() - 1);

  std::string Result(Length, '\0');
  for (auto &C : Result)
    C = CharacterSet[Distribution(Generator)];
  return Result;
}

std::optional<fs::path> createTemporaryDirectory() {
  const auto Prefix = fs::temp_directory_path() / "chip-temp-";
  for (unsigned Tries = 100; Tries; Tries--) {
    auto TempDir = Prefix;
    TempDir += getRandomString(8);
    // fs::create_directories returning false means that somebody else
    // has already created the directory and is using it. Try to
    // create a directory for exclusive use.
    if (fs::create_directories(TempDir))
      return TempDir;
  }

  return std::nullopt;
}

bool writeToFile(const fs::path Path, const std::string &Data) {
  std::ofstream File(Path);
  File << Data;
  return File.good();
}

std::optional<std::string> readFromFile(const fs::path Path) {
  if (auto File = std::ifstream(Path)) {
    std::stringstream Buffer;
    Buffer << File.rdbuf();
    return Buffer.str();
  }
  return std::nullopt;
}

#ifndef __APPLE__
static int dlIterateCallback(struct dl_phdr_info *Info, size_t Size,
                             void *Data) {
  std::string *Res = static_cast<std::string *>(Data);
  std::string DlName(Info->dlpi_name);
  size_t Pos = DlName.find("/libCHIP.so");
  if (Pos == std::string::npos)
    return 0;

  DlName.erase(Pos);
  Res->assign(DlName);
  return 1;
}
#endif

std::optional<fs::path> getHIPCCPath() {
  static std::once_flag Flag;
  static std::optional<fs::path> HIPCCPath;

  std::string LibCHIPPath("/dev/null");
#ifdef __APPLE__
  // On macOS, use dladdr to find the library path
  Dl_info info;
  if (dladdr((void*)getHIPCCPath, &info) && info.dli_fname) {
    std::string DlName(info.dli_fname);
    size_t Pos = DlName.find("/libCHIP.dylib");
    if (Pos != std::string::npos) {
      DlName.erase(Pos);
      LibCHIPPath = DlName;
    }
  }
#else
  dl_iterate_phdr(dlIterateCallback, static_cast<void *>(&LibCHIPPath));
#endif

  std::call_once(Flag, [&]() {
    for (const auto &ExeCand : {fs::path(LibCHIPPath) / "bin/hipcc",
                                fs::path(CHIP_INSTALL_DIR) / "bin/hipcc",
                                fs::path(CHIP_BUILD_DIR) / "bin/hipcc"})
      if (canExecuteHipcc(ExeCand)) {
        HIPCCPath = ExeCand;
        return;
      }
  });

  logDebug("HIPCC path: {}", HIPCCPath->c_str());
  return HIPCCPath;
}

std::optional<fs::path> getChipKernelVerifyPath() {
  std::string LibCHIPPath("/dev/null");
#ifdef __APPLE__
  Dl_info info;
  if (dladdr((void*)getChipKernelVerifyPath, &info) && info.dli_fname) {
    std::string DlName(info.dli_fname);
    size_t Pos = DlName.find("/libCHIP.dylib");
    if (Pos != std::string::npos) {
      DlName.erase(Pos);
      LibCHIPPath = DlName;
    }
  }
#else
  dl_iterate_phdr(dlIterateCallback, static_cast<void *>(&LibCHIPPath));
#endif
  for (const auto &Cand : {fs::path(LibCHIPPath) / "bin/chip-kernel-verify",
                           fs::path(CHIP_INSTALL_DIR) / "bin/chip-kernel-verify",
                           fs::path(CHIP_BUILD_DIR) / "bin/chip-kernel-verify"}) {
    std::error_code Ec;
    if (fs::is_regular_file(Cand, Ec) && access(Cand.c_str(), X_OK) == 0)
      return Cand;
  }
  return std::nullopt;
}

std::vector<void *>
convertExtraArgsToPointerArray(void *ExtraArgBuf, const SPVFuncInfo &FuncInfo) {
  auto *BaseAddr = (uint8_t *)ExtraArgBuf;
  std::vector<void *> PointerArray;
  PointerArray.reserve(FuncInfo.getNumClientArgs());
  unsigned Offset = 0;

  auto ArgVisitor = [&](const SPVFuncInfo::ClientArg &Arg) {
    assert((Arg.Kind == SPVTypeKind::POD || Arg.Kind == SPVTypeKind::Pointer) &&
           "Unexpected argument kind.");

    // Default argument size and alignment.
    size_t Size = Arg.Size;
    size_t Alignment = roundUpToPowerOfTwo(Size);
    assert(Size && Alignment && "Couldn't determine arg size or alignment!");
    Offset = roundUp(Offset, Alignment);
    logDebug("Extra arg {} offset: {}", Arg.Index, Offset);
    PointerArray.push_back(BaseAddr + Offset);
    Offset += Size;
  };
  FuncInfo.visitClientArgs(ArgVisitor);

  return PointerArray;
}

std::string_view trim(std::string_view Str) {
  auto IsWhitespace = [](char C) -> bool { return (C == ' ' || C == '\t'); };
  while (!Str.empty() && IsWhitespace(Str.front()))
    Str.remove_prefix(1);
  while (!Str.empty() && IsWhitespace(Str.back()))
    Str.remove_suffix(1);
  return Str;
}

/// Return true if the 'Str' string starts with the 'WithStr' string.
bool startsWith(std::string_view Str, std::string_view WithStr) {
  // NOTE: With C++20 this function could be deprecated in favor of
  //       std::string_view::starts_with().

  return Str.size() >= WithStr.size() &&
         Str.substr(0, WithStr.size()) == WithStr;
}

uint64_t fnv1a64(const std::string &S) {
  uint64_t Hash = UINT64_C(14695981039346656037);
  for (unsigned char C : S) {
    Hash ^= C;
    Hash *= UINT64_C(1099511628211);
  }
  return Hash;
}

/// True when 'Name' has the shape of a Compute Runtime debug key.
///
/// NEO reads every declared key by bare name: EnvironmentVariableReader::
/// getSetting() calls getenv(prefix + settingName) for each valid prefix and
/// "" is one of them (shared/source/os_interface/debug_env_reader.cpp; the
/// prefixes come from api_specific_config_{ocl,l0}.cpp and are
/// {"NEO_OCL_"/"NEO_L0_", "NEO_", ""}). Under NEOReadDebugKeys any variable
/// named like a declared key can therefore reach the compiler, and the cache
/// key has to cover all of them.
///
/// Every one of the 750 names in shared/source/debug_settings/
/// debug_variables_base.inl is a CamelCase C++ identifier: none contains an
/// underscore and each has at least one lower-case letter. That is what
/// separates them from the batch scheduler and launcher variables (PBS_*,
/// PMIX_*, SLURM_*, HOSTNAME, PALS_APID, TMPDIR), which take a fresh value on
/// every launch and, when hashed, make the key unique per run so the cache can
/// never hit.
static bool looksLikeNeoDebugKey(std::string_view Name) {
  if (Name.empty() || !std::isalpha(static_cast<unsigned char>(Name.front())))
    return false;
  bool HasLower = false;
  for (char C : Name) {
    if (C == '_')
      return false;
    HasLower |= std::islower(static_cast<unsigned char>(C)) != 0;
  }
  return HasLower;
}

std::string collectCompilerEnvironmentVariables() {
  // Environment variables that reach the device compiler and change the
  // binary it produces.
  //
  // Prefixes:
  //  - IGC_: complete for IGC by construction; its regkey reader literally
  //    prepends "IGC_" to every declared flag before calling getenv
  //    (intel-graphics-compiler, igc_regkeys.cpp, ReadIGCEnv).
  //  - NEO: Compute Runtime's spelling prefix. Every NEO debug/release
  //    variable is also readable as NEO_<name> (api_specific_config_ocl.cpp,
  //    validClPrefixes = {"NEO_OCL_", "NEO_", ""}), and this also catches
  //    NEOReadDebugKeys itself.
  //  - Override: the bare-name NEO family that includes
  //    OverrideDefaultFP64Settings, the motivating case: it switches on fp64
  //    emulation and the x86 CI exports it on every job.
  //
  // Exact names: NEO release variables that match no prefix but are honored
  // ungated by stock release drivers (release_variables_base.inl), plus the
  // two ZET_ program-instrumentation switches that change the produced
  // binary (L1 cache policy build options on debugger-attach products).
  // ZE_* as a class is deliberately NOT matched: those select devices and
  // layers, and device identity is keyed separately; hashing them only
  // causes false invalidation.
  static constexpr const char *CompilerEnvPrefixes[] = {"IGC_", "NEO",
                                                        "Override"};
  static constexpr const char *CompilerEnvExact[] = {
      "ZET_ENABLE_PROGRAM_DEBUGGING", "ZET_ENABLE_PROGRAM_INSTRUMENTATION",
      "EnableLEO", "ZEX_NUMBER_OF_CCS", "ONEAPI_PVC_SEND_WAR_WA"};

  std::vector<std::string> Vars;
  logDebug("Collecting device compiler environment variables...");

  // Access the environment variables through the global environ variable
  extern char **environ;

  // With NEOReadDebugKeys set to a nonzero value, Compute Runtime reads all
  // of its ~750 debug variables by bare name, and any of them may reach the
  // compiler (InjectInternalBuildOptions is an arbitrary string appended to
  // the build options). The prefix list above cannot cover those, so while the
  // gate is on every variable shaped like a debug key is hashed as well. With
  // the gate off (the normal case) none of them is read at all.
  const char *DebugKeys = std::getenv("NEOReadDebugKeys");
  const bool DebugKeysEnabled = DebugKeys && std::atoll(DebugKeys) != 0;

  for (char **Env = environ; *Env != nullptr; ++Env) {
    std::string_view EnvVar(*Env);
    auto Eq = EnvVar.find('=');
    if (Eq == std::string_view::npos)
      continue; // Not a NAME=VALUE entry.
    auto Name = EnvVar.substr(0, Eq);

    bool Match = DebugKeysEnabled && looksLikeNeoDebugKey(Name);
    if (!Match)
      for (const char *Prefix : CompilerEnvPrefixes)
        if (startsWith(Name, Prefix)) {
          Match = true;
          break;
        }
    if (!Match)
      for (const char *Exact : CompilerEnvExact)
        if (Name == Exact) {
          Match = true;
          break;
        }
    if (Match) {
      logDebug("Found compiler variable: {}", EnvVar);
      Vars.emplace_back(EnvVar);
    }
  }

  // Sort so the key does not depend on the order the environment happens to be
  // laid out in.
  std::sort(Vars.begin(), Vars.end());

  std::string Result;
  for (const auto &Var : Vars) {
    if (!Result.empty()) {
      Result += ";";
    }
    Result += Var;
  }

  logDebug("Collected compiler variables string: '{}'", Result);
  return Result;
}

/// Deep copies kernel arguments pointed by 'CopyArg'. Bytes of the
/// argument values are stored in 'ArgData'. 'ArgList[I]' points to
/// the argument value in 'ArgData' for Ith kernel argument.
void copyKernelArgs(std::vector<void *> &ArgList, std::vector<char> &ArgData,
                    void **CopyFrom, const SPVFuncInfo &FuncInfo) {

  ArgList.clear();
  ArgData.clear();

  std::vector<size_t> Offsets;
  size_t CurrOffset = 0;

  auto CopyArgData = [&](const SPVFuncInfo::ClientArg &Arg) {
    assert((Arg.Kind == SPVTypeKind::POD || Arg.Kind == SPVTypeKind::Pointer) &&
           "Unexpected argument kind.");

    size_t Size = Arg.Size;
    size_t Alignment = roundUpToPowerOfTwo(Size);
    assert(Size && Alignment && "Invalid arg size or alignment!");

    CurrOffset = roundUp(CurrOffset, Alignment);
    logDebug("arg {} tgt offset: {}", Arg.Index, CurrOffset);
    Offsets.push_back(CurrOffset);
    assert(CurrOffset >= ArgData.size());

    ArgData.resize(CurrOffset + Size, 0);
    std::memcpy(ArgData.data() + CurrOffset, Arg.Data, Size);

    CurrOffset += Size;
  };
  FuncInfo.visitClientArgs(CopyFrom, CopyArgData);

  ArgList.reserve(Offsets.size());
  char *BasePtr = ArgData.data();
  for (auto Offset : Offsets)
    ArgList.push_back(static_cast<void *>(BasePtr + Offset));
}
