//=============================================================================
//
//   Copyright (c) 2023 chipStar developers
//
//   Permission is hereby granted, free of charge, to any person
//   obtaining a copy of this software and associated documentation
//   files (the "Software"), to deal in the Software without
//   restriction, including without limitation the rights to use, copy,
//   modify, merge, publish, distribute, sublicense, and/or sell copies
//   of the Software, and to permit persons to whom the Software is
//   furnished to do so, subject to the following conditions:
//
//   The above copyright notice and this permission notice shall be
//   included in all copies or substantial portions of the Software.
//
//   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
//   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
//   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
//   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
//   HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
//   WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
//   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
//   DEALINGS IN THE SOFTWARE.
//
//=============================================================================
/** Compiler wrapper aiming to be a drop-in replacement for nvcc. */
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <filesystem>
#include <set>
#include "hipBin_spirv.h"

namespace fs = std::filesystem;

HipBinSpirv hipBinSpirv;

// If true, we change the behavior of this tool so that the CMake
// considers this tool to be NVidia's nvcc. This allows us to compile
// CUDA code in CMake without find_package().
// TODO: Add an option or env-var for controlling the value of this.
bool MASQUERADE_AS_NVCC = true;

void error_exit(const std::string& msg) {
    /** Print an error message to stderr and then exit with an error. */
    std::cerr << "error: " << msg << std::endl;
    std::exit(1);
}

void warn(const std::string& msg) {
    /** Print an warning message to stderr. */
    std::cerr << "warning: " << msg << std::endl;
}

class IgnoredOption {
    /** A class that ignores the option and gives a warning about it. */
public:
    IgnoredOption(const std::string& option_string) {
        warn("Ignored option '" + option_string + "'");
    }
};

std::map<std::string, std::string> prepare_argparser(const std::vector<std::string>& arg_list) {
    /** Prepares argument parser intended for parse_known_args() */
    std::map<std::string, std::string> args;

    // Define options we want to capture and act upon - for example, translate NVCC specific options to corresponding one in hipcc, filter out options, raise errors or warnings on unsupported options, etc.
    args["-expt-relaxed-constexpr"] = "false";
    args["-extended-lambda"] = "false";
    args["-std"] = "";
    args["-x"] = "";
    args["-default-stream"] = "";
    args["--version"] = "false";
    args["-v"] = "false";

    for (const auto& arg : arg_list) {
        if (args.find(arg) != args.end()) {
            args[arg] = "true";
        } else if (arg.rfind("-std=", 0) == 0) {
            args["-std"] = arg.substr(5);
        } else if (arg == "-x") {
            args["-x"] = arg;
        }
    }

    return args;
}

std::string get_hip_path() {
    /** Get HIP path */
    std::cout << "hipInfo.hipPath: " << hipBinSpirv.hipInfo_.hipPath << std::endl;
    return hipBinSpirv.hipInfo_.hipPath;
}

std::string get_hipcc() {
    /** Get path to hipcc executable */
    return get_hip_path() + "/bin/hipcc";
}

std::string get_cuda_include_dir() {
    /** Return include directory for chipStar's CUDA headers. */
    return get_hip_path() + "/include/cuspv";
}

std::string get_cuda_library_dir() {
    /** Return directory for chipStar's CUDA libraries. */
    return get_hip_path() + "/lib";
}

std::set<std::string> determine_input_languages(const std::vector<std::string>& arg_list, const std::string& xmode) {
    /** Determine input language modes from the argument list */
    std::set<std::string> modes;

    for (const auto& arg : arg_list) {
        if (arg[0] == '-') continue; // Skip options

        auto ext = fs::path(arg).extension().string();

        if (ext == ".c") modes.insert("c");
        else if (ext == ".cu") modes.insert("cuda");
        else if (ext == ".cpp" || ext == ".cc") modes.insert("c++");
    }

    return modes;
}

std::vector<std::string> filter_args_for_hipcc(const std::vector<std::string>& arg_list) {
    /** Filter out arguments on the way to hipcc. */
    std::vector<std::string> filtered;

    for (const auto& arg : arg_list) {
        if (arg != "-Xcompiler") {
            filtered.push_back(arg);
        }
    }

    return filtered;
}

int main(int argc, char* argv[]) {
    warn("cucc is a work-in-progress."
         " It is incomplete and may behave incorrectly."
         "\nPlease, report issues at "
         "https://github.com/CHIP-SPV/chipStar/issues.");

    
    hipBinSpirv.detectPlatform();
    /** Driver implementation. */
    const char* verbose_env = std::getenv("CHIP_CUCC_VERBOSE");
    if (verbose_env && std::string(verbose_env) == "1") {
        std::cerr << "cucc args: ";
        for (int i = 0; i < argc; ++i) {
            std::cerr << argv[i] << " ";
        }
        std::cerr << std::endl;
    }

    std::vector<std::string> other_args(argv + 1, argv + argc);
    auto args = prepare_argparser(other_args);

    if (args["-default-stream"] == "per-thread") {
        error_exit("per-thread stream function is not implemented.");
    }

    auto languages = determine_input_languages(other_args, args["-x"]);

    std::vector<std::string> hipcc_args = {get_hipcc(), "-isystem", get_cuda_include_dir()};
    hipcc_args.push_back("-D__NVCC__");
    hipcc_args.push_back("-D__CHIP_CUDA_COMPATIBILITY__");

    if (!args["-std"].empty() && languages.find("c") == languages.end()) {
        hipcc_args.push_back("-std=" + args["-std"]);
    }

    if (!args["-x"].empty()) {
        hipcc_args.push_back("-x");
        hipcc_args.push_back(args["-x"]);
    }

    if (args["--version"] == "true") {
        std::string alt_version_str = std::getenv("CUCC_VERSION_STRING") ? std::getenv("CUCC_VERSION_STRING") : "";
        if (MASQUERADE_AS_NVCC && !alt_version_str.empty()) {
            std::cout << alt_version_str << std::endl;
            return 0;
        } else {
            hipcc_args.push_back("--version");
        }
    }

    if (args["-v"] == "true") {
        if (MASQUERADE_AS_NVCC) {
            std::cout << "#$ PATH=" << std::getenv("PATH") << std::endl;
            std::cout << "#$ TOP=" << get_hip_path() << std::endl;
            std::cout << "#$ LIBRARIES= -L" << get_cuda_library_dir() << std::endl;
            std::cout << "#$ g++ -L" << get_cuda_library_dir() << " -lCHIP" << std::endl;
        } else {
            hipcc_args.push_back("-v");
        }
    }

    auto filtered_args = filter_args_for_hipcc(other_args);
    hipcc_args.insert(hipcc_args.end(), filtered_args.begin(), filtered_args.end());

    if (std::getenv("CHIP_CUCC_VERBOSE") && std::string(std::getenv("CHIP_CUCC_VERBOSE")) == "1") {
        std::cerr << "Executing: ";
        for (const auto& arg : hipcc_args) {
            std::cerr << arg << " ";
        }
        std::cerr << std::endl;
    }

    std::string command = "";
    for (const auto& arg : hipcc_args) {
        command += arg + " ";
    }

    int result = std::system(command.c_str());
    return result;
}