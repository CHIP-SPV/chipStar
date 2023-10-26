#!/usr/bin/env python3
import os
import argparse
import subprocess
import hashlib

# execute a command and return the output along with the return code
def run_cmd(cmd):
    cmd_hash = hashlib.md5(cmd.encode()).hexdigest()[0:10]
    file_name = f"/tmp/{cmd_hash}_cmd.txt"
    cmd = f"{cmd} | tee {file_name}"
    # print(f"Running command: {cmd}")
    return_code = subprocess.call(cmd, shell=True)
    with open(file_name, "r") as f:
        return f.read(), return_code

parser = argparse.ArgumentParser(
                    prog='check.py',
                    description='Run the unit tests for the specified device type and backend',
                    epilog='have a nice day')

parser.add_argument('work_dir', type=str, help='Path to build directory')
parser.add_argument('device_type', type=str, choices=['cpu', 'igpu', 'dgpu'], help='Device type')
parser.add_argument('backend', type=str, choices=['opencl', 'level0-reg', 'level0-imm', 'pocl'], help='Backend to use')
parser.add_argument('--num-threads', type=int, nargs='?', default=os.cpu_count(), help='Number of threads to use (default: number of cores on the system)')
parser.add_argument('--num-tries', type=int, nargs='?', default=1, help='Number of tries (default: 1)')
parser.add_argument('--timeout', type=int, nargs='?', default=200, help='Timeout in seconds (default: 200)')
parser.add_argument('-m', '--modules', type=str, choices=['on', 'off'], default="on", help='load modulefiles automatically (default: on)')
parser.add_argument('-v', '--verbose', action='store_true', help='verbose output')

args = parser.parse_args()

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
    env_vars += " CHIP_L0_IMM_CMD_LISTS=OFF"
elif args.backend == "level0-imm":
    level0_cmd_list = "imm_"
    args.backend = "level0"
    env_vars += " CHIP_L0_IMM_CMD_LISTS=ON"
else:
    level0_cmd_list = ""

# setup module load line
modules = ""
if args.modules == "on":
  modules =  ". /etc/profile.d/modules.sh && module load "
  if args.backend == "opencl" and args.device_type == "cpu":
      modules += "opencl/cpu"
  elif args.backend == "opencl" and args.device_type == "igpu":
      modules += "opencl/igpu"
  elif args.backend == "opencl" and args.device_type == "dgpu":
      modules += "opencl/dgpu"
  elif args.backend == "level0" and args.device_type == "igpu":
      modules += "intel/opencl level-zero/igpu"
  elif args.backend == "level0" and args.device_type == "dgpu":
      modules += "intel/opencl level-zero/dgpu"
  elif args.backend == "pocl" and args.device_type == "cpu":
      modules += "opencl/pocl"
  modules += " &&  module list;"

os.chdir(args.work_dir)

cmd = f"{modules} {env_vars} ./samples/hipInfo/hipInfo"
out, _ = run_cmd(cmd)
texture_support = 0 < int(out.split("maxTexture1DLinear:")[1].split("\n")[0].strip())
if not texture_support:
    texture_cmd = "|[Tt]ex"
else:
    texture_cmd = ""

cmd = f"{modules} {env_vars} ctest --schedule-random --output-on-failure --timeout {args.timeout} --repeat until-fail:{args.num_tries} -j {args.num_threads} -E \"`cat ./test_lists/{args.device_type}_{args.backend}_failed_{level0_cmd_list}tests.txt`{texture_cmd}\"  -O checkpy_{args.device_type}_{args.backend}.txt"
res, ctest_return_code = run_cmd(cmd)
# check if "0 tests failed" is in the output, if so return 0
if "0 tests failed" in res:
    exit(0)
else:
    exit(ctest_return_code)
