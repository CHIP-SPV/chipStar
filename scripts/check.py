#!/usr/bin/env python3
import os
import argparse
import subprocess
import hashlib
import time
import platform


parser = argparse.ArgumentParser(
                    prog='check.py',
                    description='Run the unit tests for the specified device type and backend',
                    epilog='have a nice day')

parser.add_argument('work_dir', type=str, help='Path to build directory')
parser.add_argument('device_type', type=str, choices=['cpu', 'igpu', 'dgpu', 'pocl'], help='Device type')
parser.add_argument('backend', type=str, choices=['opencl', 'level0'], help='Backend to use')
parser.add_argument('--num-threads', type=int, nargs='?', default=os.cpu_count(), help='Number of threads to use (default: number of cores on the system)')
parser.add_argument('--timeout', type=int, nargs='?', default=200, help='Timeout in seconds (default: 200)')
parser.add_argument('-m', '--modules', type=str, choices=['on', 'off'], default="off", help='load modulefiles automatically (default: off)')
parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
parser.add_argument('-d', '--dry-run', '-N', action='store_true', help='dry run')
parser.add_argument('--regex-include', type=str, nargs='?', default="", help='Tests to be run must also match this regex (known failures will still be excluded)')
parser.add_argument('--regex-exclude', type=str, nargs='?', default="", help='Specifically exclude tests that match this regex (known failures will still be excluded)')

# --total-runtime cannot be used with --num-tries
group = parser.add_mutually_exclusive_group()
group.add_argument('--total-runtime', type=str, nargs='?', default=None, help='Set --num-tries such that the total runtime is approximately this value in hours')
group.add_argument('--num-tries', type=int, nargs='?', default=1, help='Number of tries (default: 1)')

args = parser.parse_args()

# make sure that args.total_runtime end in either m or h
if args.total_runtime is not None:
    if str(args.total_runtime[-1]) not in ["m", "h"]:
        print("Error: --total-runtime should end in either 'm' or 'h'")
        exit(1)

# execute a command and return the output along with the return code
def run_cmd(cmd):
    # get current milliseconds
    cur_ms = subprocess.check_output("date +%s%3N", shell=True).decode('utf-8').strip()
    cmd_hash = hashlib.sha256(cmd.encode('utf-8')+ cur_ms.encode('utf-8')).hexdigest()

    file_name = f"/tmp/{cmd_hash}_cmd.txt"
    cmd = f"{cmd} | tee {file_name}"
    if args.verbose:
        print(f"check.py: {cmd}")
    if args.dry_run:
        print(cmd)
        return "", 0
    return_code = subprocess.call(cmd, shell=True)
    with open(file_name, "r") as f:
        return f.read(), return_code

# print out the arguments
if args.verbose:
  print(f"work_dir: {args.work_dir}")
  print(f"device_type: {args.device_type}")
  print(f"backend: {args.backend}")
  print(f"num_threads: {args.num_threads}")
  print(f"num_tries: {args.num_tries}")
  print(f"timeout: {args.timeout}")

# platform agnostic way of getting hostname
hostname = platform.uname().node

if args.device_type in ["cpu", "pocl"]:
    device_type_stripped = "cpu"
elif args.device_type in ["dgpu", "igpu"]:
    device_type_stripped = "gpu"

env_vars = f"CHIP_BE={args.backend} CHIP_DEVICE_TYPE={device_type_stripped}"
    
# setup module load line
modules = ""
if args.modules == "on":
  modules =  ". /etc/profile.d/modules.sh && export MODULEPATH=/home/pvelesko/modulefiles && module load "
  if args.backend == "opencl" and args.device_type == "cpu":
      modules += "opencl/cpu"
  elif args.backend == "opencl" and args.device_type == "igpu":
      modules += "opencl/igpu"
  elif args.backend == "opencl" and args.device_type == "dgpu":
      modules += "opencl/dgpu"
  elif args.backend == "level0" and args.device_type == "igpu":
      modules += "level-zero/igpu"
  elif args.backend == "level0" and args.device_type == "dgpu":
      modules += "level-zero/dgpu"
  elif args.backend == "opencl" and args.device_type == "pocl":
      modules += "opencl/pocl"
  modules += " &&  module list;"

os.chdir(args.work_dir)

cmd = f"{modules} {env_vars} ./hipInfo"
out, _ = run_cmd(cmd)

texture_support = False
if ("maxTexture1DLinear:" in out):
    texture_support = 0 < int(out.split("maxTexture1DLinear:")[1].split("\n")[0].strip())

double_support = False
if ("arch.hasDoubles:" in out):
    double_support = 0 < int(out.split("arch.hasDoubles:")[1].split("\n")[0].strip())

if double_support:
    double_cmd = ""
else:
    double_cmd = "|[Dd]ouble"
if not texture_support:
    texture_cmd = "|[Tt]ex"
else:
    texture_cmd = ""

all_test_list = f"./test_lists/ALL.txt"
failed_test_list = f"./test_lists/{args.backend.upper()}_{device_type_stripped.upper()}_{hostname.upper()}.txt"

def run_tests(num_tries):
  if len(args.regex_exclude) > 0:
      args.regex_exclude = f"{args.regex_exclude}|"
  if len(args.regex_include) > 0:
      args.regex_include = f"-R {args.regex_include}"
  # if failed_test_list is not empty, separator is |, otherwise it is empty
  separator = "|" if os.path.exists(failed_test_list) and os.path.getsize(failed_test_list) > 0 else ""
  cmd = f"{modules} {env_vars} ctest --output-on-failure --timeout {args.timeout} --repeat until-fail:{num_tries} -j {args.num_threads} {args.regex_include} -E \"{args.regex_exclude}`cat {failed_test_list}`{separator}`cat {all_test_list}`{texture_cmd}\" -O checkpy_{args.backend}_{args.device_type}.txt"
  res, err = run_cmd(cmd)
  return res, err


# if --total-runtime is set, calculate the number of tries by running run_tests and checking the time
num_tries = 1
if args.total_runtime:
    t_start = time.time()
    run_tests(1)
    t_end = time.time()
    # calculate the total time
    total_time = t_end - t_start
    # calculate the number of tries
    # make sure that args.total_runtime ends in either m or h
    if args.total_runtime[-1] == "m":
        num_tries = int(float(args.total_runtime[:-1]) * 60 / total_time)
    elif args.total_runtime[-1] == "h":
        num_tries = int(float(args.total_runtime[:-1]) * 60 * 60 / total_time)
    else:
        print("Error: --total-runtime should end in either m or h")
        exit(1)
    print(f"Running tests {num_tries} times to get a total runtime of {args.total_runtime} hours")
else:
    num_tries = args.num_tries

res, err = run_tests(num_tries)
if "0 tests failed" in res:
    exit(0)
else:
    exit(1)
