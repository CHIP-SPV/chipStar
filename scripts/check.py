#!/usr/bin/env python3
import os
import argparse
import subprocess
import hashlib
import time


parser = argparse.ArgumentParser(
                    prog='check.py',
                    description='Run the unit tests for the specified device type and backend',
                    epilog='have a nice day')

parser.add_argument('work_dir', type=str, help='Path to build directory')
parser.add_argument('device_type', type=str, choices=['cpu', 'igpu', 'dgpu'], help='Device type')
parser.add_argument('backend', type=str, choices=['opencl', 'level0-reg', 'level0-imm', 'pocl'], help='Backend to use')
parser.add_argument('--num-threads', type=int, nargs='?', default=os.cpu_count(), help='Number of threads to use (default: number of cores on the system)')
parser.add_argument('--timeout', type=int, nargs='?', default=200, help='Timeout in seconds (default: 200)')
parser.add_argument('-m', '--modules', type=str, choices=['on', 'off'], default="off", help='load modulefiles automatically (default: off)')
parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')
parser.add_argument('-d', '--dry-run', '-N', action='store_true', help='dry run')
parser.add_argument('-c', '--categories', action='store_true', help='run tests by categories, including running a set of tests in a single thread')

# --total-runtime cannot be used with --num-tries
group = parser.add_mutually_exclusive_group()
group.add_argument('--total-runtime', type=int, nargs='?', default=0, help='Set --num-tries such that the total runtime is approximately this value in minutes')
group.add_argument('--num-tries', type=int, nargs='?', default=1, help='Number of tries (default: 1)')

args = parser.parse_args()

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

if args.device_type == "cpu":
    device_type_stripped = "cpu"
elif args.device_type in ["dgpu", "igpu"]:
    device_type_stripped = "gpu"

if args.backend in ["pocl", "opencl"]:
    env_vars = f"CHIP_BE=opencl CHIP_DEVICE_TYPE={device_type_stripped}"
else:
    env_vars = f"CHIP_BE=level0 CHIP_DEVICE_TYPE={device_type_stripped}"

if args.backend == "level0-reg":
    level0_cmd_list = "reg_"
    args.backend = "level0"
    backend_full = "level0_reg"
    env_vars += " CHIP_L0_IMM_CMD_LISTS=OFF"
elif args.backend == "level0-imm":
    level0_cmd_list = "imm_"
    args.backend = "level0"
    backend_full = "level0_imm"
    env_vars += " CHIP_L0_IMM_CMD_LISTS=ON"
else:
    level0_cmd_list = ""
    backend_full = args.backend

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
  elif args.backend == "pocl" and args.device_type == "cpu":
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

def run_tests(num_tries):
  if args.categories:
    cmd_deviceFunc = f"{modules} {env_vars} ctest --output-on-failure --timeout {args.timeout} --repeat until-fail:{num_tries} -j 100 -E \"`cat ./test_lists/{args.device_type}_{args.backend}_failed_{level0_cmd_list}tests.txt`{texture_cmd}{double_cmd}\" -R deviceFunc -O checkpy_{args.device_type}_{args.backend}_device.txt"
    cmd_graph = f"{modules} {env_vars} ctest --output-on-failure --timeout {args.timeout} --repeat until-fail:{num_tries} -j 100 -E \"`cat ./test_lists/{args.device_type}_{args.backend}_failed_{level0_cmd_list}tests.txt`{texture_cmd}{double_cmd}\" -R \"[Gg]raph\" -O checkpy_{args.device_type}_{args.backend}_graph.txt"
    cmd_single = f"{modules} {env_vars} ctest --output-on-failure --timeout {args.timeout} --repeat until-fail:{num_tries} -j 1 -E \"`cat ./test_lists/{args.device_type}_{args.backend}_failed_{level0_cmd_list}tests.txt`{texture_cmd}{double_cmd}\" -R \"`cat ./test_lists/non_parallel_tests.txt`\" -O checkpy_{args.device_type}_{args.backend}_single.txt"
    cmd_other = f"{modules} {env_vars} ctest --output-on-failure --timeout {args.timeout} --repeat until-fail:{num_tries} -j {args.num_threads} -E \"`cat ./test_lists/{args.device_type}_{args.backend}_failed_{level0_cmd_list}tests.txt`{texture_cmd}{double_cmd}|deviceFunc|[Gg]raph|`cat ./test_lists/non_parallel_tests.txt`\" -O checkpy_{args.device_type}_{args.backend}_other.txt"

    res_deviceFunc, err  = run_cmd(cmd_deviceFunc)
    res_graph, err = run_cmd(cmd_graph)
    res_single, err = run_cmd(cmd_single)
    res_other, err = run_cmd(cmd_other)

    if "0 tests failed" in res_deviceFunc and "0 tests failed" in res_graph and "0 tests failed" in res_single and "0 tests failed" in res_other:
        exit(0)
    else:
        exit(1)
  else:
    cmd = f"{modules} {env_vars} ctest --output-on-failure --timeout {args.timeout} --repeat until-fail:{num_tries} -j {args.num_threads} -E \"`cat ./test_lists/{args.device_type}_{args.backend}_failed_{level0_cmd_list}tests.txt`{texture_cmd}\" -O checkpy_{args.device_type}_{backend_full}.txt"
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
    num_tries = int(args.total_runtime * 60 / total_time)
    print(f"Running tests {num_tries} times to get a total runtime of {args.total_runtime} minutes")
else:
    num_tries = args.num_tries

res, err = run_tests(num_tries)
if "0 tests failed" in res:
    exit(0)
else:
    exit(1)
