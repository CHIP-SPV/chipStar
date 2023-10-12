#!/usr/bin/env python3
import os
import argparse
from util import run_cmd

parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')

parser.add_argument('work_dir', type=str, help='Path to build directory')
parser.add_argument('device_type', type=str, choices=['cpu', 'igpu', 'dgpu'], help='Device type')
parser.add_argument('backend', type=str, choices=['opencl', 'level0-reg', 'level0-imm', 'pocl'], help='Backend to use')
parser.add_argument('num_threads', type=int, help='Number of threads to use')
parser.add_argument('num_tries', type=int, help='Number of tries')

args = parser.parse_args()

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

if args.device_type == "cpu":
    timeout = 1800
else:
    timeout = 1800

os.chdir(args.work_dir)

cmd = "./samples/hipInfo/hipInfo"
out = run_cmd(cmd)
texture_support = 0 < int(out.split("maxTexture1DLinear:")[1].split("\n")[0].strip())
if not texture_support:
    texture_cmd = "|[Tt]ex"
else:
    texture_cmd = ""

cmd = f"{env_vars} ctest --output-on-failure --timeout {timeout} --repeat until-fail:{args.num_tries} -j {args.num_threads} -E \"`cat ./test_lists/{args.device_type}_{args.backend}_failed_{level0_cmd_list}tests.txt`{texture_cmd}\"  -O checkpy_{args.device_type}_{args.backend}.txt"
res = run_cmd(cmd)
if '***Failed' in res:
    exit(1)
