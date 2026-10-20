#!/usr/bin/env python3
#=============================================================================
#
#   Copyright (c) 2023 chipStar developers
#
#   Permission is hereby granted, free of charge, to any person
#   obtaining a copy of this software and associated documentation
#   files (the "Software"), to deal in the Software without
#   restriction, including without limitation the rights to use, copy,
#   modify, merge, publish, distribute, sublicense, and/or sell copies
#   of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be
#   included in all copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#   EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#   MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
#   NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
#   HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
#   WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
#   DEALINGS IN THE SOFTWARE.
#
#=============================================================================
"""Compiler wrapper aiming to be a drop-in replacement for nvcc."""
# Enable v3.9+ type annotations in older versions (down to 3.7).
import argparse
import os
import shlex
import subprocess
import sys
from shlex import quote
from typing import List, Set, Optional

# If true, we change the behavior of this tool so that the CMake
# considers this tool to be NVvidia's nvcc. This allows us to compile
# CUDA code in CMake without find_package().
# TODO: Add an option or env-var for controlling the value of this.
MASQUERADE_AS_NVCC = True
DEFAULT_VERSION_BANNER = """\
nvcc: CHIPSTAR Cuda compiler driver
 V12.5.40"""
DEFAULT_COMPUTE_CAPABILITY = 80

def error_exit(msg : str):
    """Print an error message to stderr and then exit with an error.
    """
    print("error: " + msg, file=sys.stderr)
    sys.exit(1)

def warn(msg : str):
    """Print an warning message to stderr.
    """
    print("warning: " + msg, file=sys.stderr)

class IgnoredOption(argparse.Action):
    """An action that ignores the option and gives a warning about it.
    """
    def __init__(self, option_strings, dest, quiet=False, **kwargs):
        super().__init__(option_strings, dest, **kwargs)
        self.quiet = quiet

    def __call__(self, parser, namespace, values, option_string=None):
        if not self.quiet:
            warn(f"Ignored option '{self.option_strings[-1]}'")

def prepare_argparser() -> argparse.ArgumentParser:
    """Prepares argument parser intended for parse_known_args()

    Here we define options we want to capture and act upon - for
    example, translate NVCC specific options to corresponding one in
    hipcc, filter out options, raise errors or warnings on unsupported
    options, etc.

    Unknown arguments are passed to hipcc.
    """
    parser = argparse.ArgumentParser(
        description=("CUDA compiler driver for chipStar and a drop-in "
                     + "replacement for nvcc."))

    # --expt-relaxed-constexpr (-expt-relaxed-constexpr)
    # Experimental flag: Allow host code to invoke ``__device__
    # constexpr`` functions, and device code to invoke ``__host__
    # constexpr`` functions.
    # [NVCC v12.2 4.2.7.7]
    #
    # In HIP constexpr functions can be called from __device__ and
    # __host__ code. Not sure if this option means to allow:
    #   * __device__ -> __host__ constexpr calls and
    #   * __host__ -> __device__ constexpr calls.
    parser.add_argument("-expt-relaxed-constexpr", "--expt-relaxed-constexpr",
                        action='store_true')

    # "Allow __host__, __device__ annotations in lambda declarations."
    # [NVCC v12.2 4.2.3.21]
    # HIP allows this intrinsically so this will be ignored silently.
    parser.add_argument("-extended-lambda", "--extended-lambda",
                        action='store_true')

    # nvcc allows -std=c++ mixed with C files which in case the option is
    # ineffective for C files.
    parser.add_argument(
        '-std', '--std',
        choices=['c++03', 'c++11', 'c++14', 'c++17', 'c++20', 'c++23'],
        help="Specify language standard for CUDA and C++ inputs.")

    # The nvcc differs from clang, hipcc and other compilers by that the
    # option applies globally and only the last option is considered.
    parser.add_argument('-x', '--x', choices=['c', 'c++', 'cu'],
                        help="Specify language of the input files")

    parser.add_argument("-default-stream", "--default-stream",
                        choices=['legacy', 'null', 'per-thread'])

    parser.add_argument("--version", action='store_true')
    parser.add_argument("-v", "--verbose", action='store_true')

    # TODO: define __CUDA_ARCH__ and __CUDA_ARCH_LIST__ based on
    #       '-arch' values.
    parser.add_argument("-arch", action=IgnoredOption, quiet=True)
    # TODO: handle -fmad. Investigate what is the nvcc's default for this too.
    parser.add_argument('-fmad', '--fmad', choices=['true', 'false'],
                        action=IgnoredOption)
    parser.add_argument("-maxrregcount", "--maxrregcount",
                        action=IgnoredOption, help="Ignored option.")
    parser.add_argument("--keep", nargs=0, action=IgnoredOption, quiet=True)
    parser.add_argument("--keep-dir", action=IgnoredOption, quiet=True)
    parser.add_argument("--generate-code", action=IgnoredOption, quiet=True)
    parser.add_argument("-G", "--device-debug", nargs=0, action=IgnoredOption,
                        help="Generate debug information for device code.")
    # "The compiler has an option (-use_fast_math) that forces each
    # function in Table 9 to compile to its intrinsic counterpart. "
    # [CUDA PM v12.2 13.2].
    #
    # CUDA intrinsics have defined numerical accuracy. This means -ffast-math
    # is potentially more relaxed than --use_fast_math so we may not use it.
    parser.add_argument("-use_fast_math", "--use_fast_math",
                        nargs=0, action=IgnoredOption)
    parser.add_argument("-gencode", "--gencode",
                        nargs="?", action=IgnoredOption, quiet=True)

    # Add new arguments for relocatable device code and device compilation
    parser.add_argument("--relocatable-device-code", choices=['true', 'false'])
    parser.add_argument("-rdc", choices=['true', 'false'])
    parser.add_argument("--device-c", action='store_true')
    parser.add_argument("-dc", action='store_true')

    return parser

def get_hip_path() -> str:
    """Get HIP path
    cucc resides in <instal_dir/build_dir>/../bin
    so HIP path is always one level up.
    """
    import os
    path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return path

def get_hipcc() -> str:
    """Get path to hipcc executable
    """
    return get_hip_path() + "/bin/hipcc"

def get_cuda_include_dir() -> str:
    """Return include directory for chipStar's CUDA headers.
    """
    return get_hip_path() + "/include/cuspv"

def get_cuda_library_dir() -> str:
    """Return directory for chipStar's CUDA libraries.
    """
    return get_hip_path() + "/lib"

def determine_input_languages(arg_list: List[str], xmode: Optional[str] = None) -> Set[str]:
    """Determine input language modes from the argument list

    The 'xmode' specifies value of the last '-x' option if it was specified.
    returns a set of languages (c, c++ and cu) involved.
    """

    # '-x' applies globally - unlike in other compilers usually.
    if xmode is not None:
        return {xmode}

    # Scan over arguments and look for file extensions for determining
    # input languages. This may give incorrect answer if unknown
    # options with values are involved ('-opt-with-value silly.cu').
    modes : set = set()
    for arg in arg_list:
        if len(modes) == 3: # All supported languages are present.
            break

        if arg.startswith("-"):
            continue # An option, skip it.

        _, ext = os.path.splitext(arg)

        if not ext: # Perhaps, a value for an unknown option.
            continue

        if ext == ".c":
            modes.add('c')
        elif ext == ".cu":
            modes.add('cuda')
        elif ext in {'.cpp', '.cc'}:
            modes.add('c++')

    return modes

def expand_xcompiler_options(value: str) -> List[str]:
    """Expand nvcc's -Xcompiler value into individual host options.

    nvcc -Xcompiler behavior:
      -Xcompiler -Wall                       -> ["-Wall"]
      -Xcompiler "-Wall","-Werror"           -> ["-Wall", "-Werror"]
      -Xcompiler \"-Wl,--no-as-needed\"      -> ["-Wl,--no-as-needed"]
    """
    try:
        lexer = shlex.shlex(value, posix=True)
        lexer.whitespace_split = True # Split on separators only, not wordchars.
        lexer.whitespace = ","        # The comma is the sole separator.
        lexer.commenters = ""         # '#' has no special meaning here.
        opts = [opt.strip() for opt in lexer if opt.strip()]
    except ValueError as err:
        # Malformed value (unbalanced quote): pass it through as is.
        warn(f"could not parse -Xcompiler value '{value}': {err}")
        opts = [value]
    return opts


def expand_options_file(path: str) -> List[str]:
    """Expand nvcc's --options-file value into individual options.

    The file holds one option per line, with an option and its argument
    separated by spaces. Blank lines are skipped and '#' starts a comment,
    as in nvcc.
    """
    try:
        with open(path) as stream:
            lines = stream.readlines()
    except OSError as err:
        warn(f"could not read options file '{path}': {err}")
        return []

    opts : List[str] = []
    for line in lines:
        try:
            opts.extend(shlex.split(line, comments=True))
        except ValueError as err:
            warn(f"could not parse options file '{path}': {err}")
    return opts


def filter_args_for_hipcc(arg_list: List[str]) -> List[str]:
    """Filter out arguments on the way to hipcc.
    """
    filtered = []
    i = 0
    while i < len(arg_list):
        arg = arg_list[i]
        i += 1

        if arg == "--threads":
            if i < len(arg_list):
                i += 1
                continue
            # Without a value it is not the --threads we handle; pass it on.

        if arg.startswith("-ftz=") or arg.startswith("--ftz="):
            continue

        # nvcc uses -ccbin (aliased --compiler-bindir) to select the host
        # compiler. chipStar compiles host code with clang through hipcc, so
        # the path is not needed - drop it along with its value.
        if arg == "-ccbin" or arg == "--compiler-bindir":
            if i < len(arg_list):
                i += 1
            continue
        if arg.startswith("-ccbin=") or arg.startswith("--compiler-bindir="):
            continue
        if arg == "-forward-unknown-to-host-compiler":
            continue

        if arg == "--options-file":
            if i < len(arg_list):
                filtered.extend(expand_options_file(arg_list[i]))
                i += 1
            continue
        if arg.startswith("--options-file="):
            filtered.extend(expand_options_file(arg[len("--options-file="):]))
            continue

        # -Xcompiler takes the next token as its value. That value may be a
        # single option or a comma-separated list of host options; expand it
        # into individual flags instead of passing it through verbatim.
        if arg == "-Xcompiler":
            if i < len(arg_list):
                filtered.extend(expand_xcompiler_options(arg_list[i]))
                i += 1
            continue
        if arg.startswith("-Xcompiler="):
            filtered.extend(expand_xcompiler_options(arg[len("-Xcompiler="):]))
            continue

        filtered.append(arg)

    return filtered

def main():
    """Driver implementation.
    """
    if os.environ.get('CHIP_CUCC_VERBOSE') == "1":
        print(f"cucc args: {' '.join(map(quote, sys.argv))}",
              file=sys.stderr)

    known_args, other_args = prepare_argparser().parse_known_args()

    if known_args.default_stream == "per-thread":
        error_exit("per-thread stream function is not implemented.")

    languages = determine_input_languages(other_args, known_args.x)

    hipcc_args = [get_hipcc(), "-isystem", get_cuda_include_dir()]
    hipcc_args.append("-include")
    hipcc_args.append(get_cuda_include_dir()+"/cuda_runtime.h")

    hipcc_args.append("-D__NVCC__")

    # This tells our CUDA/HIP headers to adjust their code for CUDA compilation.
    hipcc_args.append("-D__CHIP_CUDA_COMPATIBILITY__")

    # Handle -std option. It should be ineffective for C inputs (nvcc
    # behavior).  NOTE: this does not behave correctly for mixed C and
    # C/CUDA input cases currently.
    if known_args.std is not None and languages != {'c'}:
        hipcc_args.append("-std=" + known_args.std)

    # Only the last -x option applies and is applied globally so place
    # it before any file argument.
    if known_args.x is not None:
        if known_args.x == 'cu':
            hipcc_args.extend(["-x", "hip"])
        else:
            hipcc_args.extend(["-x", known_args.x])

    if known_args.version:
        alt_version_str = os.environ.get('CUCC_VERSION_STRING')
        if MASQUERADE_AS_NVCC:
            print(alt_version_str if alt_version_str
                  else DEFAULT_VERSION_BANNER)
            sys.exit(0)
        else:
            hipcc_args.append("--version")

    if known_args.verbose:
        if MASQUERADE_AS_NVCC:
            # Mimic nvcc partially that makes our compiler to look
            # like nvcc for CMake for the needed parts.

            print(f"#$ PATH={os.environ.get('PATH')}")

            # CMake infers CMAKE_CUDA_COMPILER_TOOLKIT_ROOT from this line.
            print(f"#$ TOP={get_hip_path()}")

            # CMake fills CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES from this.
            print(f'#$ INCLUDES="-I{get_cuda_include_dir()}" '
                  f'"-I{get_hip_path()}/include"')

            # CMake parses the default architecture from '-arch compute_NN'.
            print(f"#$ nvlink -arch compute_{DEFAULT_COMPUTE_CAPABILITY}")

            # Contents of the LIBRARIES variable is used by CMake for locating
            # a 'link line'.
            print(f"#$ LIBRARIES= -L{get_cuda_library_dir()}")

            # The 'link line': CMake uses its first word as the link launcher.
            print(f"#$ {get_hipcc()} -L{get_cuda_library_dir()} -lCHIP")

            # --verbose option is not propagated so CMake does not get
            # confused by the extra output from the hipcc.
        else:
            hipcc_args.append("-v")

    # Handle relocatable device code and device compilation flags
    if (known_args.relocatable_device_code == 'true' or
        known_args.device_c or
        known_args.dc):
            hipcc_args.append("-fgpu-rdc")
    
    if known_args.device_c or known_args.dc or "-dc" in other_args or "--device-c" in other_args:
        hipcc_args.append("-c")

    if other_args is not None:
        hipcc_args.extend(filter_args_for_hipcc(other_args))

    if os.environ.get('CHIP_CUCC_VERBOSE') == "1":
        print(f"Executing: {' '.join(map(quote, hipcc_args))}",
              file=sys.stderr)

    result = subprocess.run(hipcc_args, check=False, shell=False)
    sys.exit(result.returncode)

if __name__ == '__main__':
    warn("cucc is a work-in-progress."
         + " It is incomplete and may behave incorrectly."
         + "\nPlease, report issues at "
         + "https://github.com/CHIP-SPV/chipStar/issues.")
    main()
