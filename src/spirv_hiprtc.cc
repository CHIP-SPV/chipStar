/*
Copyright (c) 2021 - 2021 Advanced Micro Devices, Inc. All rights reserved.
Copyright (c) 2022 chipStar developers.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <hip/hiprtc.h>
// #include "macros.hh"
#include "CHIPBackend.hh"
#include "Utils.hh"
#include "logging.hh"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <regex>
#include <set>
#include <vector>
#include <chrono>

struct CompileOptions {
  std::vector<std::string> Options; /// All accepted user options.
  bool HasO = false; /// True if the user provided an optimization flag.
};

/// Checks the name is valid string for #include (both the "" and <>
/// forms) and is sensible name for shells as is (doesn't need escaping).
static bool checkIncludeName(std::string_view Name) {
  // TODO: capture more valid cases (e.g. foo/bar.h).
  static const auto RegEx = std::regex("[.\\w][\\-.\\w]*");
  return std::regex_match(Name.begin(), Name.end(), RegEx);
}

static bool createHeaderFiles(const chipstar::Program &Program,
                              const fs::path &DestDir) {
  for (auto &Header : Program.getHeaders())
    if (!writeToFile(DestDir / Header.first, Header.second))
      return false;
  return true;
}

static bool createSourceFile(const chipstar::Program &Program,
                             const fs::path OutputFile,
                             // A file to output lowered name expressions.
                             const fs::path LoweredNamesFile) {

  std::ofstream File(OutputFile);
  File << Program.getSource() << "\n";

  // Insert name expressions at the end of the program. They are used
  // for mapping name expressions to mangled kernels and they may
  // instantiate templates.
  const auto &NameExprMap = Program.getNameExpressionMap();
  if (NameExprMap.empty())
    return File.good();

  // CUDA NVRTC guide: "... The characters in the name expression
  // string are parsed as a C++ constant expression at the end of the
  // user program."
  unsigned Num = 0;
  for (auto &Kv : Program.getNameExpressionMap())
    File << "extern \"C\" __device__ constexpr auto *_chip_name_expr_"
         << std::to_string(Num++) << " = " << Kv.first << ";\n";

  // The lowered name expressions are extracted by a LLVM pass which
  // detects the magic variable in the above. Because HIPSPV tool
  // chain does not support passing options to the our pass plugin, we
  // emit another magic variable for pointing the output file where
  // the results are written to.
  File << R"---(
extern  "C" __device__ const char *_chip_name_expr_output_file =
    )---"
       << LoweredNamesFile << ";";

  return File.good();
}

/// Filter and translate user given options. Return true if an error
/// was encountered.
static bool processOptions(chipstar::Program &Program, int NumOptions,
                           const char* const* Options, CompileOptions &OptionsOut) {

  // Already checked in hiprtcCompileProgram().
  assert(NumOptions >= 0);
  assert(Options || NumOptions == 0);

  auto Match = [&](std::string_view Str, std::string_view RegEx) -> bool {
    return std::regex_match(Str.begin(), Str.end(),
                            std::regex(RegEx.data(), RegEx.size()));
  };

  // Pass whitelisted options. Unrecognized options are ignored.
  for (int OptIdx = 0; OptIdx < NumOptions; OptIdx++) {
    if (!Options[OptIdx])
      continue; // Consider NULL pointers are empty.
    auto OptionIn = trim(std::string_view(Options[OptIdx]));
    if (Match(OptionIn, "-D.*") || Match(OptionIn, "--?std=[cC][+][+][0-9]*") ||
        Match(OptionIn, "-I.*") || Match(OptionIn, "-g") ||
        Match(OptionIn, "-fno-eliminate-unused-debug-types") ||
        Match(OptionIn, "-fno-eliminate-unused-debug-symbols")) {
      logDebug("hiprtc: accept option '{}'", std::string(OptionIn));
      OptionsOut.Options.emplace_back(OptionIn);
      continue;
    }

    if (Match(OptionIn, "-O[0-3]")) {
      OptionsOut.Options.emplace_back(OptionIn);
      OptionsOut.HasO = true;
      continue;
    }

    // TODO: match and translate nvrtc options?

    logWarn("hiprtc: ignored option: '{}'", OptionIn);
    Program.appendToLog(std::string("warning: ignored option '") +
                        std::string(OptionIn) + "'\n");
  }

  return false;
}

/// Escapes the string with single quotes for bourne shell.
static std::string escapeWithSingleQuotes(const std::string &Str) {
  std::string Result;
  Result.reserve(Str.size() + 2);
  Result += "'";
  for (auto C : Str) {
    if (C == '\'') // Escape '.
      Result += "'\"'\"'";
    else
      Result += C;
  }
  Result += "'";
  return Result;
}

static std::string createCompileCommand(const CompileOptions &Options,
                                        const fs::path &WorkingDirectory,
                                        const fs::path &SourceFile,
                                        const fs::path &OutputFile) {

  std::string CompileCommand;

  auto Append = [&](const std::string Str) -> void {
    if (Str.empty())
      return;
    if (CompileCommand.size())
      CompileCommand += " ";
    // Put arguments into single quotes for avoiding misinterpretations
    // and shell injections.
    CompileCommand += escapeWithSingleQuotes(Str);
  };

  // Get path to hipcc tool. If not found use "hipcc" and hope it's
  // found in PATH.
  Append(getHIPCCPath().value_or("hipcc"));

  // Adding this option will cause the compiler to emit device code only.
  // Ommitting the use of this option will create a fatbin.
  // We can use readelf --debug-dump=info <fatbin> to get useful debug info.
  // In particular, this allows HipInterceptLayer to figure out struct sizes
  // without having to parse source code. 
  // Append("--cuda-device-only");

#ifdef CHIP_SOURCE_DIR
  // For making the compilation work in the build directory.
  //
  // TODO: Could we detect if we are using installed chipStar and omit
  //       these options?
  Append(std::string("-I") + CHIP_SOURCE_DIR + "/HIP/include");
  Append(std::string("-I") + CHIP_SOURCE_DIR + "/include/hip");
  Append(std::string("-I") + CHIP_SOURCE_DIR + "/include");
#endif

  // For locating headers added by hiprtcCreateProgram().
  Append("-I" + WorkingDirectory.string());

  // Clients don't need to include hip_runtime.h by themselves.
  Append("--include=hip/hip_runtime.h");
  Append("--include=hip/hip_fp16_gcc.h");

  // User options.
  for (const auto &Opt : Options.Options)
    Append(Opt);

  // By default optimizations are on (-dopt).
  if (!Options.HasO)
    Append("-O2");

  Append("-c");

  Append(SourceFile.string());
  Append("-o");
  Append(OutputFile.string());

  return CompileCommand;
}

static bool executeCommand(const fs::path &WorkingDirectory,
                           std::string_view Command,
                           const fs::path &CompileLogFile) {

  auto ScriptFile = WorkingDirectory / "compile.sh";

  std::string ShellScript;
#ifdef CHIP_CLANG_PATH
  // For convenience, add the clang (needed by hipcc) found during
  // project configuration into the PATH in case it's not there
  // already. The path is appended to the back so we don't override
  // search paths the client has possibly set.
  ShellScript += std::string("export PATH=$PATH:") + CHIP_CLANG_PATH + "; ";
#endif
  ShellScript += Command;
  ShellScript += " >'" + CompileLogFile.string() + "' 2>&1";

  if (!writeToFile(ScriptFile, ShellScript)) {
    logError("Could not create shell command script.");
    return false;
  }

  logDebug("Executing shell command '{}'", ShellScript);

  auto ShellCommand = std::string("sh ") + ScriptFile.string();
  int ReturnCode = std::system(ShellCommand.c_str());
  logDebug("Return code: {}", ReturnCode);
  return ReturnCode == 0;
}

static void getLoweredNameExpressions(chipstar::Program &Program,
                                      const fs::path &WorkingDirectory,
                                      const fs::path &LoweredNamesFile) {
  auto &NameExprMap = Program.getNameExpressionMap();
  if (NameExprMap.empty())
    return;

  if (auto InputStream = std::ifstream(LoweredNamesFile)) {
    auto It = NameExprMap.begin();
    std::string LoweredName;
    while (std::getline(InputStream, LoweredName)) {
      assert(It != NameExprMap.end());
      It++->second = LoweredName;
    }
  }
}

// Compiles sources stored in 'chipstar::Program'. Uses 'WorkingDirectory' for
// temporary compilation I/O.
static hiprtcResult compile(chipstar::Program &Program, int NumRawOptions,
                            const char* const* RawOptions,
                            fs::path WorkingDirectory) {
  // Create source and header files.
  auto SourceFile = WorkingDirectory / "program.hip";
  auto OutputFile = WorkingDirectory / "program.o";
  auto LoweredNamesFile = WorkingDirectory / "lowerednames.txt";
  auto CompileLogFile = WorkingDirectory / "compile.log";

  CompileOptions ProcessedOptions;
  if (processOptions(Program, NumRawOptions, RawOptions, ProcessedOptions))
    return HIPRTC_ERROR_INVALID_INPUT;

  if (!createHeaderFiles(Program, WorkingDirectory)) {
    logError("hiprtc: could not create user header files.");
    return HIPRTC_ERROR_COMPILATION;
  }

  if (!createSourceFile(Program, SourceFile, LoweredNamesFile)) {
    logError("hiprtc: could not create user source file.");
    return HIPRTC_ERROR_COMPILATION;
  }

  std::string CompileCommand = createCompileCommand(
      ProcessedOptions, WorkingDirectory, SourceFile, OutputFile);

  bool ExecSuccess =
      executeCommand(WorkingDirectory, CompileCommand, CompileLogFile);

  if (auto Log = readFromFile(CompileLogFile))
    Program.appendToLog(*Log);
  else
    logError("Could not read logfile (unknown reason).");

  if (!ExecSuccess)
    return HIPRTC_ERROR_COMPILATION;

  auto Bundle = readFromFile(OutputFile);
  if (!Bundle) {
    logError("Could not read compilation output '{}' (unknown reason).",
             OutputFile.string());
    return HIPRTC_ERROR_COMPILATION;
  }

  Program.addCode(*Bundle);

  getLoweredNameExpressions(Program, WorkingDirectory, LoweredNamesFile);

  return HIPRTC_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

#include <stdlib.h>

#if !defined(_WIN32)
#pragma GCC visibility push(default)
#endif

const char *hiprtcGetErrorString(hiprtcResult Result) {
#define HANDLE_CASE(E)                                                         \
  case E:                                                                      \
    return #E

  switch (Result) {
  default:
    logError("Invalid hiprtc error code: {}.", Result);
    return nullptr;

    HANDLE_CASE(HIPRTC_SUCCESS);
    HANDLE_CASE(HIPRTC_ERROR_OUT_OF_MEMORY);
    HANDLE_CASE(HIPRTC_ERROR_PROGRAM_CREATION_FAILURE);
    HANDLE_CASE(HIPRTC_ERROR_INVALID_INPUT);
    HANDLE_CASE(HIPRTC_ERROR_INVALID_PROGRAM);
    HANDLE_CASE(HIPRTC_ERROR_INVALID_OPTION);
    HANDLE_CASE(HIPRTC_ERROR_COMPILATION);
    HANDLE_CASE(HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE);
    HANDLE_CASE(HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION);
    HANDLE_CASE(HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION);
    HANDLE_CASE(HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID);
    HANDLE_CASE(HIPRTC_ERROR_INTERNAL_ERROR);
  }
#undef HANDLE_CASE
}

hiprtcResult hiprtcVersion(int *Major, int *Minor) {
  if (!Major || !Minor)
    return HIPRTC_ERROR_INVALID_INPUT;

  // TODO: Choose a version to return. Returning dummy version values for now.
  *Major = 1;
  *Minor = 0;
  return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcAddNameExpression(hiprtcProgram Prog,
                                     const char *NameExpression) {
  if (!Prog || !NameExpression)
    return HIPRTC_ERROR_INVALID_INPUT;

  auto &Program = *(chipstar::Program *)Prog;
  if (Program.isAfterCompilation())
    return HIPRTC_ERROR_NO_NAME_EXPRESSIONS_AFTER_COMPILATION;

  // Reject some unreasonable name expressions upfront that would
  // cause failures at the compilation step or cause other issues.
  using RE = std::regex;
  if ( // Empty name expressions.
      std::string_view(NameExpression).empty() ||
      std::regex_match(NameExpression, RE("[[:space:]]*")) ||
      // Characters not expected in expressions.
      std::regex_search(NameExpression, RE("[;]")))
    return HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID;

  Program.addNameExpression(NameExpression);
  return HIPRTC_SUCCESS;
}

/// FNV-1a 64-bit hash — portable, deterministic across runs and platforms.
/// std::hash<std::string> is implementation-defined and may produce different
/// values across program runs (some libc++ randomize it) or across different
/// compiler/stdlib versions, making the cache effectively write-only on those
/// platforms. The same change was previously proposed in #1231 (closed
/// unmerged); we re-introduce it here because a stable on-disk cache key is
/// part of the new versioned cache file format ('CHC1' magic) — without a
/// stable hash, the version header doesn't help.
static uint64_t fnv1a64(const std::string &s) {
  uint64_t hash = UINT64_C(14695981039346656037);
  for (unsigned char c : s) {
    hash ^= c;
    hash *= UINT64_C(1099511628211);
  }
  return hash;
}

/// Compute a cache key for HIPRTC output based on source, headers, options,
/// and registered name expressions.
/// The key is a portable, stable hash of all inputs that affect the SPIRV
/// output AND the set of name expressions that need a lowered-name mapping.
/// Without including name expressions, two compilations with the same
/// source/options but a different set of registered name expressions would
/// alias to the same cache entry, leaving some lowered-name lookups unmapped
/// on cache hit.
static std::string computeHiprtcCacheKey(const chipstar::Program &Program,
                                         int NumOptions,
                                         const char *const *Options) {
  std::string combined;
  combined += Program.getSource();
  combined += "\n---headers---\n";
  // std::map is sorted by key, so iteration order is deterministic
  for (auto &[name, content] : Program.getHeaders()) {
    combined += name + ":" + content + "\n";
  }
  combined += "\n---options---\n";
  for (int i = 0; i < NumOptions; i++) {
    if (Options[i])
      combined += Options[i];
    combined += "\n";
  }
  combined += "\n---name-expressions---\n";
  // Map values are empty at hash time (filled in only after compilation), so
  // only keys (the expressions) contribute. std::map iteration is sorted.
  for (auto &[expr, lowered] : Program.getNameExpressionMap()) {
    combined += expr + "\n";
  }
  return std::to_string(fnv1a64(combined));
}

// Cache file format (binary, little-endian):
//   magic[4]      = 'C','H','C','1'
//   spirv_size    = uint32_t
//   spirv_bytes   = spirv_size bytes
//   name_count    = uint32_t
//   for each entry:
//     expr_len    = uint32_t
//     expr_bytes  = expr_len bytes (UTF-8)
//     lowered_len = uint32_t
//     lowered_bytes = lowered_len bytes (UTF-8)
//
// 'CHC1' marks format v1. A future schema change should bump to 'CHC2' etc.
// loadHiprtcCache rejects any file lacking the magic — old (pre-fix) caches
// are auto-evicted on first use.
static constexpr char kCacheMagic[4] = {'C', 'H', 'C', '1'};

static void writeU32LE(std::ostream &out, uint32_t v) {
  char b[4];
  b[0] = (char)(v & 0xff);
  b[1] = (char)((v >> 8) & 0xff);
  b[2] = (char)((v >> 16) & 0xff);
  b[3] = (char)((v >> 24) & 0xff);
  out.write(b, 4);
}

static bool readU32LE(std::istream &in, uint32_t &v) {
  unsigned char b[4];
  if (!in.read((char *)b, 4)) return false;
  v = (uint32_t)b[0]
    | ((uint32_t)b[1] << 8)
    | ((uint32_t)b[2] << 16)
    | ((uint32_t)b[3] << 24);
  return true;
}

/// Try to load a cached HIPRTC compilation result.
/// Returns true and populates Program.Code_ and the name-expression map if a
/// cache hit is found. Returns false on miss, missing magic (old format), or
/// any read/parse error (treated as cache miss).
static bool loadHiprtcCache(chipstar::Program &Program,
                             const std::string &cacheKey) {
  if (!ChipEnvVars.getModuleCacheDir().has_value())
    return false;
  auto cacheFile = fs::path(ChipEnvVars.getModuleCacheDir().value())
                   / "hiprtc" / cacheKey;
  std::ifstream in(cacheFile, std::ios::binary);
  if (!in)
    return false;

  char magic[4];
  if (!in.read(magic, 4) ||
      std::memcmp(magic, kCacheMagic, 4) != 0) {
    logDebug("hiprtc: cache file '{}' has missing/old magic; ignoring",
             cacheFile.string());
    return false;
  }

  uint32_t spirvSize = 0;
  if (!readU32LE(in, spirvSize))
    return false;
  std::string spirv(spirvSize, '\0');
  if (spirvSize && !in.read(spirv.data(), spirvSize))
    return false;

  uint32_t nameCount = 0;
  if (!readU32LE(in, nameCount))
    return false;
  // Stage parsed names locally; only commit to Program after a fully
  // successful read so a partial/corrupt cache file doesn't leave half-set
  // state on the program object.
  std::vector<std::pair<std::string, std::string>> names;
  names.reserve(nameCount);
  for (uint32_t i = 0; i < nameCount; i++) {
    uint32_t exprLen = 0;
    if (!readU32LE(in, exprLen)) return false;
    std::string expr(exprLen, '\0');
    if (exprLen && !in.read(expr.data(), exprLen)) return false;
    uint32_t loweredLen = 0;
    if (!readU32LE(in, loweredLen)) return false;
    std::string lowered(loweredLen, '\0');
    if (loweredLen && !in.read(lowered.data(), loweredLen)) return false;
    names.emplace_back(std::move(expr), std::move(lowered));
  }

  Program.addCode(spirv);
  // Populate the name-expression map. The user-registered expressions are
  // already present (with empty values) from prior hiprtcAddNameExpression()
  // calls; we fill in the lowered names from the cache.
  auto &NameExprMap = Program.getNameExpressionMap();
  for (auto &[expr, lowered] : names) {
    auto It = NameExprMap.find(expr);
    if (It != NameExprMap.end())
      It->second = lowered;
  }
  logInfo("hiprtc: Loaded SPIRV from cache (key={}, names={})",
          cacheKey, nameCount);
  return true;
}

/// Save HIPRTC compilation result to cache.
/// Writes both the SPIR-V binary and the name-expression -> lowered-name
/// table so subsequent cache hits can satisfy hiprtcGetLoweredName().
static void saveHiprtcCache(const chipstar::Program &Program,
                             const std::string &cacheKey) {
  if (!ChipEnvVars.getModuleCacheDir().has_value())
    return;
  auto cacheDir = fs::path(ChipEnvVars.getModuleCacheDir().value()) / "hiprtc";
  std::error_code ec;
  fs::create_directories(cacheDir, ec);
  if (ec) {
    logDebug("hiprtc: Could not create cache directory: {}", ec.message());
    return;
  }
  auto cacheFile = cacheDir / cacheKey;
  // Write to a temp file then rename, so a crash mid-write does not leave a
  // partial cache file that would later be loaded as if valid.
  auto tmpFile = cacheFile;
  tmpFile += ".tmp";
  {
    std::ofstream out(tmpFile, std::ios::binary | std::ios::trunc);
    if (!out) {
      logDebug("hiprtc: Could not open cache temp file for writing: {}",
               tmpFile.string());
      return;
    }
    out.write(kCacheMagic, 4);
    const auto &code = Program.getCode();
    if (code.size() > UINT32_MAX) {
      logDebug("hiprtc: SPIR-V too large for cache file format ({} bytes)",
               code.size());
      out.close();
      fs::remove(tmpFile, ec);
      return;
    }
    writeU32LE(out, (uint32_t)code.size());
    out.write(code.data(), code.size());
    const auto &NameExprMap = Program.getNameExpressionMap();
    writeU32LE(out, (uint32_t)NameExprMap.size());
    for (auto &[expr, lowered] : NameExprMap) {
      writeU32LE(out, (uint32_t)expr.size());
      out.write(expr.data(), expr.size());
      writeU32LE(out, (uint32_t)lowered.size());
      out.write(lowered.data(), lowered.size());
    }
    if (!out) {
      logDebug("hiprtc: Error while writing cache temp file: {}",
               tmpFile.string());
      fs::remove(tmpFile, ec);
      return;
    }
  }
  fs::rename(tmpFile, cacheFile, ec);
  if (ec) {
    logDebug("hiprtc: Could not rename cache temp file: {}", ec.message());
    fs::remove(tmpFile, ec);
    return;
  }
  logInfo("hiprtc: Saved SPIRV to cache (key={}, names={})",
          cacheKey, Program.getNameExpressionMap().size());
}

hiprtcResult hiprtcCompileProgram(hiprtcProgram Prog, int NumOptions,
                                  const char* const* Options) {
  logTrace("{}", __func__);

  if (NumOptions < 0) {
    logError("Invalid option count ({}).", NumOptions);
    return HIPRTC_ERROR_INVALID_INPUT;
  }

  if (NumOptions && !Options) {
    logError("Option array may not be NULL.");
    return HIPRTC_ERROR_INVALID_INPUT;
  }

  if (!Prog)
    return HIPRTC_ERROR_INVALID_INPUT;
  try {
    auto &Program = *(chipstar::Program *)Prog;

    // Check HIPRTC output cache before invoking clang.
    auto cacheKey = computeHiprtcCacheKey(Program, NumOptions, Options);
    auto t0 = std::chrono::steady_clock::now();
    if (loadHiprtcCache(Program, cacheKey)) {
      auto t1 = std::chrono::steady_clock::now();
      double elapsed = std::chrono::duration<double>(t1 - t0).count();
      logInfo("hiprtc: Cache hit — skipped clang compilation ({:.3f}s saved)", elapsed);
      return HIPRTC_SUCCESS;
    }

    // Create temporary directory for compilation I/O.
    auto TmpDir = createTemporaryDirectory();
    if (!TmpDir) {
      logError(
          "hiprtc: Failed to create a temporary directory for compilation.");
      return HIPRTC_ERROR_COMPILATION;
    }

    logDebug("hiprtc: Temp directory: '{}'", TmpDir->string());
    hiprtcResult Result = compile(Program, NumOptions, Options, *TmpDir);

    if (!ChipEnvVars.getSaveTemps()) {
      assert(!TmpDir->empty() && *TmpDir != TmpDir->root_path() &&
             "Attempted to delete a root directory!");

      logDebug("Removing '{}'", TmpDir->string());
      std::error_code IgnoreErrors;
      fs::remove_all(*TmpDir, IgnoreErrors);
    }

    // Cache the compiled SPIRV for future runs.
    if (Result == HIPRTC_SUCCESS)
      saveHiprtcCache(Program, cacheKey);

    return Result;
  } catch (...) {
    logDebug("Caught an unknown exception\n");
    return HIPRTC_ERROR_INTERNAL_ERROR;
  }
}

hiprtcResult hiprtcCreateProgram(hiprtcProgram *Prog, const char *Src,
                                 const char *Name, int NumHeaders,
                                 const char* const* Headers,
                                 const char* const* IncludeNames) {
  if (!Prog)
    return HIPRTC_ERROR_INVALID_INPUT;
  if (!Src)
    return HIPRTC_ERROR_INVALID_INPUT;

  try {
    // From NVRTC: 'CUDA program name. name can be NULL;
    // "default_program" is used when name is NULL or "". '.
    auto Program =
        std::make_unique<chipstar::Program>(Name ? Name : "default_program");
    Program->setSource(Src);

    for (int i = 0; i < NumHeaders; i++) {
      auto *IncludeNamePtr = IncludeNames[i];
      auto *HeaderPtr = Headers[i];

      if (!IncludeNamePtr) {
        logError("Include name must not be null.");
        return HIPRTC_ERROR_INVALID_INPUT;
      }
      if (!HeaderPtr) {
        logError("Include contents buffer must not be null.");
        return HIPRTC_ERROR_INVALID_INPUT;
      }

      std::string_view IncludeName(IncludeNamePtr);
      if (IncludeName.empty()) // Nameless headers can't be included.
        continue;

      if (!checkIncludeName(IncludeNamePtr)) {
        logError("Invalid include name.");
        return HIPRTC_ERROR_INVALID_INPUT;
      }

      Program->addHeader(IncludeName, HeaderPtr);
    }

    *Prog = (hiprtcProgram)Program.release();
    return HIPRTC_SUCCESS;
  } catch (...) {
    logDebug("Caught an unknown exception\n");
    return HIPRTC_ERROR_INTERNAL_ERROR;
  }
}

hiprtcResult hiprtcDestroyProgram(hiprtcProgram *Prog) {
  if (!Prog || !*Prog)
    return HIPRTC_ERROR_INVALID_PROGRAM;
  try {
    delete (chipstar::Program *)*Prog;
    *Prog = nullptr;
  } catch (...) {
    logDebug("Caught an unknown exception\n");
    return HIPRTC_ERROR_INTERNAL_ERROR;
  }
  return HIPRTC_SUCCESS;
}

hiprtcResult hiprtcGetLoweredName(hiprtcProgram WrappedProg,
                                  const char *NameExpression,
                                  const char **LoweredName) {
  if (!WrappedProg || !NameExpression || !LoweredName)
    return HIPRTC_ERROR_INVALID_INPUT;

  auto &Prog = *(chipstar::Program *)WrappedProg;
  if (!Prog.isAfterCompilation())
    return HIPRTC_ERROR_NO_LOWERED_NAMES_BEFORE_COMPILATION;

  const auto &NameExprMap = Prog.getNameExpressionMap();
  auto It = NameExprMap.find(NameExpression);
  if (It != NameExprMap.end()) {
    if (It->second.empty()) {
      // Defensive: a populated map entry should never have an empty lowered
      // name. If it does, the compilation pipeline (or cache load) failed
      // to record the mapping. Returning success with an empty string makes
      // the caller pass "" to hipModuleGetFunction, which fails downstream
      // with the misleading error "Failed to find kernel via kernel name: ".
      logError("hiprtc: lowered name for '{}' is empty; refusing to return "
               "empty string as success",
               NameExpression);
      return HIPRTC_ERROR_INTERNAL_ERROR;
    }
    *LoweredName = It->second.data();
    return HIPRTC_SUCCESS;
  }

  return HIPRTC_ERROR_NAME_EXPRESSION_NOT_VALID;
}

hiprtcResult hiprtcGetProgramLog(hiprtcProgram Prog, char *Log) {
  if (!Prog || !Log)
    return HIPRTC_ERROR_INVALID_INPUT;
  try {
    const auto &LogSrc = ((chipstar::Program *)Prog)->getProgramLog();
    std::memcpy(Log, LogSrc.c_str(), LogSrc.size());
    return HIPRTC_SUCCESS;
  } catch (...) {
    logDebug("Caught an unknown exception\n");
    return HIPRTC_ERROR_INTERNAL_ERROR;
  }
}

hiprtcResult hiprtcGetProgramLogSize(hiprtcProgram Prog, size_t *LogSizeRet) {
  if (!Prog || !LogSizeRet)
    return HIPRTC_ERROR_INVALID_INPUT;
  try {
    *LogSizeRet = ((chipstar::Program *)Prog)->getProgramLog().size();
    return HIPRTC_SUCCESS;
  } catch (...) {
    logDebug("Caught an unknown exception\n");
    return HIPRTC_ERROR_INTERNAL_ERROR;
  }
}

hiprtcResult hiprtcGetCode(hiprtcProgram Prog, char *Code) {
  if (!Prog)
    return HIPRTC_ERROR_INVALID_PROGRAM;
  if (!Code)
    return HIPRTC_ERROR_INVALID_INPUT;
  try {
    auto &SavedCode = ((chipstar::Program *)Prog)->getCode();
    std::memcpy(Code, SavedCode.c_str(), SavedCode.size());
    return HIPRTC_SUCCESS;
  } catch (...) {
    logDebug("Caught an unknown exception\n");
    return HIPRTC_ERROR_INTERNAL_ERROR;
  }
}

hiprtcResult hiprtcGetCodeSize(hiprtcProgram Prog, size_t *CodeSizeRet) {
  if (!Prog)
    return HIPRTC_ERROR_INVALID_PROGRAM;
  if (!CodeSizeRet)
    return HIPRTC_ERROR_INVALID_INPUT;
  try {
    *CodeSizeRet = ((chipstar::Program *)Prog)->getCode().size();
    return HIPRTC_SUCCESS;
  } catch (...) {
    logDebug("Caught an unknown exception\n");
    return HIPRTC_ERROR_INTERNAL_ERROR;
  }
}

#if !defined(_WIN32)
#pragma GCC visibility pop
#endif

#ifdef __cplusplus
}
#endif /* __cplusplus */
