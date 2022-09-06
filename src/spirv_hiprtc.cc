/*
Copyright (c) 2021 - 2021 Advanced Micro Devices, Inc. All rights reserved.
Copyright (c) 2022 CHIP-SPV developers.

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
#include "macros.hh"
#include "logging.hh"
#include "CHIPBackend.hh"
#include "Utils.hh"

#include <filesystem>
#include <cstdlib>
#include <regex>
#include <set>

namespace fs = std::filesystem;

/// Checks the name is valid string for #include (both the "" and <>
/// forms) and is sensible name for shells as is (doesn't need escaping).
static bool checkIncludeName(std::string_view Name) {
  // TODO: capture more valid cases (e.g. foo/bar.h).
  static const auto RegEx = std::regex("[.\\w][\\-.\\w]*");
  return std::regex_match(Name.begin(), Name.end(), RegEx);
}

static bool createHeaderFiles(const CHIPProgram &Program,
                              const fs::path &DestDir) {
  for (auto &Header : Program.getHeaders())
    if (!writeToFile(DestDir / Header.first, Header.second))
      return false;
  return true;
}

static bool createSourceFile(const CHIPProgram &Program,
                             const fs::path OutputFile) {
  return writeToFile(OutputFile, Program.getSource());
}

static bool createCompileCommand(const fs::path &WorkingDirectory,
                                 const fs::path &SourceFile,
                                 const fs::path &OutputFile,
                                 const fs::path &CompileLogFile,
                                 std::string &CompileCommandRet) {

  CompileCommandRet.clear();

  auto Append = [&](const std::string Str) -> void {
    if (CompileCommandRet.size())
      CompileCommandRet += " ";
    CompileCommandRet += Str;
  };

  Append("sh -c '");

#ifdef CHIP_CLANG_PATH
  // For convenience, add the clang (needed by hipcc) found during
  // project configuration into the PATH in case it's not there
  // already. The path is appended to the back so we don't override
  // search paths the client has possibly set.
  CompileCommandRet +=
      std::string("export PATH=$PATH:") + CHIP_CLANG_PATH + ";";
#endif

  // Get path to hipcc tool. If not found use "hipcc" and hope it's
  // found in PATH.
  Append(getHIPCCPath().value_or("hipcc"));

  // Emit device code only. Resulting output file is a clang offload bundle.
  Append("--cuda-device-only");

#ifdef CHIP_SOURCE_DIR
  // For making the compilation work in the build directory.
  //
  // TODO: Could we detect if we are using installed CHIP-SPV and omit
  //       these options?
  Append(std::string("-I") + CHIP_SOURCE_DIR + "/HIP/include");
  Append(std::string("-I") + CHIP_SOURCE_DIR + "/include/hip");
  Append(std::string("-I") + CHIP_SOURCE_DIR + "/include");
#endif

  // For locating headers added by hiprtcCreateProgram().
  Append("-I" + WorkingDirectory.string());

  // Clients don't need to include hip_runtime.h by themselves.
  Append("--include=hip/hip_runtime.h");

  // By default optimizations are on (-dopt).
  Append("-O2 -c");

  Append(SourceFile.string());
  Append("-o " + OutputFile.string());
  Append(">" + CompileLogFile.string() + " 2>&1");

  CompileCommandRet += "'";

  return true;
}

static bool executeCommand(const std::string &Command) {
  logDebug("Executing shell command '{}'", Command);
  int ReturnCode = std::system(Command.c_str());
  logDebug("Return code: {}", ReturnCode);
  return ReturnCode == 0;
}

// Compiles sources stored in 'Program'. Uses 'WorkingDirectory' for
// temporary compilation I/O.
static hiprtcResult compile(CHIPProgram &Program, fs::path WorkingDirectory) {
  // Create source and header files.
  auto SourceFile = WorkingDirectory / "program.hip";
  auto OutputFile = WorkingDirectory / "program.o";
  auto CompileLogFile = WorkingDirectory / "compile.log";

  if (!createHeaderFiles(Program, WorkingDirectory)) {
    logError("hiprtc: could not create user header files.");
    return HIPRTC_ERROR_COMPILATION;
  }

  if (!createSourceFile(Program, SourceFile)) {
    logError("hiprtc: could not create user source file.");
    return HIPRTC_ERROR_COMPILATION;
  }

  std::string CompileCommand;
  if (!createCompileCommand(WorkingDirectory, SourceFile, OutputFile,
                            CompileLogFile, CompileCommand)) {
    logError("hiprtc: Could not create compilation command.");
    return HIPRTC_ERROR_COMPILATION;
  }

  bool ExecSuccess = executeCommand(CompileCommand);

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
  UNIMPLEMENTED(HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE);
}

hiprtcResult hiprtcCompileProgram(hiprtcProgram Prog, int NumOptions,
                                  const char **Options) {
  logTrace("{}", __func__);

  if (NumOptions < 0)
    return HIPRTC_ERROR_INVALID_INPUT;

  if (NumOptions)
    logWarn("hiprtc: compile options are ignored (not supported yet).");

  for (int OptIdx = 0; OptIdx < NumOptions; OptIdx++) {
    const auto *Option = Options[OptIdx];
    if (!Option)
      return HIPRTC_ERROR_INVALID_INPUT;
    logDebug("hiprtc: option: '{}'", Option);
  }

  if (!Prog)
    return HIPRTC_ERROR_INVALID_INPUT;
  try {
    auto &Program = *(CHIPProgram *)Prog;

    // Create temporary directory for compilation I/O.
    auto TmpDir = createTemporaryDirectory();
    if (!TmpDir) {
      logError(
          "hiprtc: Failed to create a temporary directory for compilation.");
      return HIPRTC_ERROR_COMPILATION;
    }

    logDebug("hiprtc: Temp directory: '{}'", TmpDir->string());
    hiprtcResult Result = compile(Program, *TmpDir);

    // TODO: Add a debug option for preserving the directory.
    assert(!TmpDir->empty() && *TmpDir != TmpDir->root_path() &&
           "Attempted to delete a root directory!");

    logDebug("Removing '{}'", TmpDir->string());
    std::error_code IgnoreErrors;
    fs::remove_all(*TmpDir, IgnoreErrors);

    return Result;
  } catch (...) {
    logDebug("Caught an unknown exception\n");
    return HIPRTC_ERROR_INTERNAL_ERROR;
  }
}

hiprtcResult hiprtcCreateProgram(hiprtcProgram *Prog, const char *Src,
                                 const char *Name, int NumHeaders,
                                 const char **Headers,
                                 const char **IncludeNames) {
  if (!Prog)
    return HIPRTC_ERROR_INVALID_INPUT;
  if (!Src)
    return HIPRTC_ERROR_INVALID_INPUT;

  try {
    // From NVRTC: 'CUDA program name. name can be NULL;
    // "default_program" is used when name is NULL or "". '.
    auto Program =
        std::make_unique<CHIPProgram>(Name ? Name : "default_program");
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

    *Prog = (hiprtcProgram)Program.get();
    Backend->addProgram(std::move(Program));
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
    return Backend->eraseProgram((CHIPProgram *)*Prog)
               ? HIPRTC_SUCCESS
               : HIPRTC_ERROR_INVALID_PROGRAM;
  } catch (...) {
    logDebug("Caught an unknown exception\n");
    return HIPRTC_ERROR_INTERNAL_ERROR;
  }
}

hiprtcResult hiprtcGetLoweredName(hiprtcProgram Prog,
                                  const char *NameExpression,
                                  const char **LoweredName) {
  UNIMPLEMENTED(HIPRTC_ERROR_BUILTIN_OPERATION_FAILURE);
}

hiprtcResult hiprtcGetProgramLog(hiprtcProgram Prog, char *Log) {
  if (!Prog || !Log)
    return HIPRTC_ERROR_INVALID_INPUT;
  try {
    const auto &LogSrc = ((CHIPProgram *)Prog)->getProgramLog();
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
    *LogSizeRet = ((CHIPProgram *)Prog)->getProgramLog().size();
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
    auto &SavedCode = ((CHIPProgram *)Prog)->getCode();
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
    *CodeSizeRet = ((CHIPProgram *)Prog)->getCode().size();
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
