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

#include <fstream>
#include <random>

/// Read an environment variable and return its value as a string.
std::string readEnvVar(std::string EnvVar, bool Lower) {
  logDebug("Reading {} from env", EnvVar);
  const char *EnvVarIn = std::getenv(EnvVar.c_str());
  if (EnvVarIn == nullptr) {
    return std::string();
  }
  std::string Var = std::string(EnvVarIn);
  if (Lower)
    std::transform(Var.begin(), Var.end(), Var.begin(),
                   [](unsigned char Ch) { return std::tolower(Ch); });

  return Var;
}

std::string generateShortHash(std::string_view input, size_t length) {
  std::hash<std::string_view> hasher;
  std::size_t hashValue = hasher(input);

  std::stringstream ss;
  ss << std::hex << std::setw(length) << std::setfill('0') << hashValue;

  return ss.str().substr(0, length);
}

/// Dump the SPIR-V to a file
void dumpSpirv(std::string_view Spirv) {
  auto dump = readEnvVar("CHIP_DUMP_SPIRV");
  if (dump.empty()) {
    return;
  }

  std::string hashSum = generateShortHash(Spirv, 6);
  std::string fileName = "hip-spirv-" + hashSum + ".spv";
  std::ofstream SpirvFile(fileName, std::ios::binary);

  if (!SpirvFile) {
    std::cerr << "Error: Could not open file " << fileName << " for writing"
              << std::endl;
    return;
  }

  SpirvFile.write(Spirv.data(), Spirv.size());
  SpirvFile.close();
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

std::optional<fs::path> getHIPCCPath() {
  static std::once_flag Flag;
  static std::optional<fs::path> HIPCCPath;

  std::call_once(Flag, []() {
    for (const auto &ExeCand : {
#if !CHIP_DEBUG_BUILD
           fs::path(CHIP_INSTALL_DIR) / "bin/hipcc",
#endif
           fs::path(CHIP_BUILD_DIR) / "bin/hipcc"
         })
      if (canExecuteHipcc(ExeCand)) {
        HIPCCPath = ExeCand;
        return;
      }
  });

  return HIPCCPath;
}

std::string_view extractSPIRVModule(const void *Bundle, std::string &ErrorMsg) {
  // NOTE: This method is designed to read from possibly misaligned buffer.

  std::string Magic(reinterpret_cast<const char *>(Bundle),
                    sizeof(CLANG_OFFLOAD_BUNDLER_MAGIC) - 1);
  if (Magic != CLANG_OFFLOAD_BUNDLER_MAGIC) {
    ErrorMsg = "The bundled binaries are not Clang bundled "
               "(CLANG_OFFLOAD_BUNDLER_MAGIC is missing)";
    return std::string_view();
  }

  using HeaderT = __ClangOffloadBundleHeader;
  using EntryT = __ClangOffloadBundleDesc;
  const auto *Header = (const char *)Bundle;
  auto NumBundles = copyAs<uint64_t>(Header, offsetof(HeaderT, numBundles));

  const char *Desc = Header + offsetof(HeaderT, desc);
  for (size_t i = 0; i < NumBundles; i++) {
    auto Offset = copyAs<uint64_t>(Desc, offsetof(EntryT, offset));
    auto Size = copyAs<uint64_t>(Desc, offsetof(EntryT, size));
    auto TripleSize = copyAs<uint64_t>(Desc, offsetof(EntryT, tripleSize));
    const char *Triple = Desc + offsetof(EntryT, triple);
    std::string_view EntryID(Triple, TripleSize);

    logDebug("Bundle entry ID {} is: '{}'\n", i, EntryID);

    // SPIR-V bundle entry ID for HIP-Clang 14+. Additional components
    // are ignored for now.
    std::string_view SPIRVBundleID = "hip-spirv64";
    if (EntryID.substr(0, SPIRVBundleID.size()) == SPIRVBundleID ||
        // Legacy entry ID used during early development.
        EntryID == "hip-spir64-unknown-unknown")
      return std::string_view(Header + Offset, Size);

    logDebug("Not a SPIR-V triple, ignoring\n");
    Desc = Triple + TripleSize; // Next bundle entry.
  }

  ErrorMsg = "Couldn't find SPIR-V binary in the bundle!";
  return std::string_view();
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
