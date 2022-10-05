/*
 * Copyright (c) 2022 CHIP-SPV developers
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

namespace fs = std::experimental::filesystem;

template <class T>
static T copyAs(const void *BaseAddr, size_t ByteOffset = 0) {
  T Res;
  std::memcpy(&Res, (const char *)BaseAddr + ByteOffset, sizeof(T));
  return Res;
}

/// Returns true if the file can be executed.
static bool canExecute(const fs::path &Path) {
  if (!fs::exists(Path))
    return false;

  const auto &Perms = fs::status(Path).permissions();

  // Only considering file to be executable if it has others_exec bit
  // set on. For owner_exec and group_exec bits we need to check
  // user's file access privileges.
  return (Perms & fs::perms::others_exec) != fs::perms::none;
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

std::optional<std::experimental::filesystem::path> getHIPCCPath() {
  // TODO: Probably should detect if we are using a built or an
  //       installed CHIP library. Mixing the installed and the built
  //       resources could lead to obscure issues.
  for (const auto &ExeCand : {fs::path(CHIP_INSTALL_DIR) / "bin/hipcc",
                              fs::path(CHIP_BUILD_DIR) / "bin/hipcc",
                              fs::path(CHIP_BUILD_DIR) / "bin/hipcc.bin"})
    if (canExecute(ExeCand))
      return ExeCand;

  return std::nullopt;
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
convertExtraArgsToPointerArray(void *ExtraArgBuf, const OCLFuncInfo &FuncInfo) {
  auto *BaseAddr = (uint8_t *)ExtraArgBuf;
  std::vector<void *> PointerArray;
  PointerArray.reserve(FuncInfo.ArgTypeInfo.size());
  unsigned Offset = 0;
  unsigned i = 0;
  for (const auto &ArgInfo : FuncInfo.ArgTypeInfo) {
    // Default argument size and alignment.
    size_t Size = ArgInfo.Size;
    size_t Alignment = roundUpToPowerOfTwo(Size);
    switch (ArgInfo.Type) {
    default:
      assert(false && "Unknown OpenCL type!");
      // FALLTHROUGH.
    case OCLType::Pointer:
      if (ArgInfo.Space == OCLSpace::Local)
        // Not passed by the client. The parameter is created when
        // there is dynamic shared memory references in the kernel.
        continue;
      // FALLTHROUGH.
    case OCLType::POD:
      break; // Use default size & alignment.

    case OCLType::Sampler:
      // Not passed by the client. HipTextureLoweringPass creates these.
      continue;

    case OCLType::Image:
      // In device code texture objects are presented as image types.
      Size = Alignment = 8; // Texture objects are pointers.
      break;
    }

    assert(Size && Alignment && "Couldn't determine arg size or alignment!");
    Offset = roundUp(Offset, Alignment);
    logDebug("Extra arg {} offset: {}", i++, Offset);
    PointerArray.push_back(BaseAddr + Offset);
    Offset += Size;
  }
  return PointerArray;
}
