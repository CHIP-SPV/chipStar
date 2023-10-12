#!/usr/bin/env python3
import sys
import re
import json
import argparse
import os
from util import run_cmd

parser = argparse.ArgumentParser(
                    prog = 'Resolved UnitTest Checker',
                    description = 'Runs thet tests that have failed for a specified backend and device type',
                    epilog = '')
parser.add_argument('path', help='Path to the build directory')
# parser.add_argument('test_case', help='Test case to check for resolved tests', choices=['cpu,opencl', 'cpu,pocl', 'igpu,opencl', 'igpu,level0', 'dgpu,opencl', 'dgpu,level0'])
parser.add_argument('device_type', help='Device type to check for resolved tests', choices=['cpu', 'igpu', 'dgpu'])
parser.add_argument('backend', type=str, choices=['opencl', 'level0-reg', 'level0-imm', 'pocl'], help='Backend to use')
parser.add_argument('num_threads', help='Number of threads to use for ctest', type=int)
parser.add_argument('num_tries', help='Number of times to run each test', type=int)
# Add an action argument either "candidates" or "verify"
parser.add_argument('action', help='Action to perform', choices=['candidates', 'verify'])
args = parser.parse_args()
print(args)

env_vars = ""

if args.backend == "level0-reg":
    level0_cmd_list = "reg_"
    args.backend = "level0"
    env_vars += " CHIP_L0_IMM_CMD_LISTS=OFF "

elif args.backend == "level0-imm":
    level0_cmd_list = "imm_"
    args.backend = "level0"
    env_vars += " CHIP_L0_IMM_CMD_LISTS=ON "
else:
    level0_cmd_list = ""

possible_tests = [
    ("cpu", "opencl"),
    ("cpu", "pocl"),
    ("igpu", "opencl"),
    ("igpu", "level0"),
    ("dgpu", "opencl"),
    ("dgpu", "level0")
]

test_tuple = (args.device_type, args.backend)
if test_tuple not in possible_tests:
    print("Invalid device type and backend combination")
    sys.exit(1)

precise_device_type = args.device_type
agg_device_type = args.device_type
if agg_device_type == "igpu" or agg_device_type == "dgpu":
    agg_device_type = "gpu"

test_case = "{precise_device_type}_{backend}_{level0_cmd_list}".format(precise_device_type=precise_device_type, backend=args.backend, level0_cmd_list=level0_cmd_list)

if args.action == "candidates":
  env_vars += "CHIP_BE={backend} CHIP_DEVICE_TYPE={device_type}".format(backend=args.backend, device_type=agg_device_type)
  print("Checking for resolved tests for {device_type} {backend}".format(device_type=precise_device_type, backend=args.backend))
  os.chdir(args.path)
  out = run_cmd("{env_vars} ctest --timeout 10 -j {num_threads}  -R \"`cat ./test_lists/{precise_device_type}_{backend}_failed_{level0_cmd_list}tests.txt`\"".format(level0_cmd_list=level0_cmd_list, backend=args.backend, precise_device_type=precise_device_type, env_vars=env_vars, num_threads=args.num_threads))
  potentially_resolved_tests = re.findall(r".*?\: (.*?) \.+   Passed", out)
  with open("potentially_resolved_tests_{test_case}.txt".format(test_case=test_case), 'w') as file:
      for test in potentially_resolved_tests:
          file.write(test + "\n")
  tests = []
  exit(0)

if args.action == "verify":
    # change to args.path dir 
    os.chdir(args.path)
    candidates = open("potentially_resolved_tests_{test_case}.txt".format(path=args.path, test_case=test_case), 'r').readlines()
    candidates = [test.strip() for test in candidates]

    tests_to_run = "|".join(["{}$".format(test) for test in candidates])
    cmd = "{env_vars} ctest --timeout 600 -j {num_threads} --repeat until-fail:{num_tries} -R '{tests_to_run}'".format(num_threads=args.num_threads, env_vars=env_vars, num_tries=args.num_tries, tests_to_run=tests_to_run)
    failed_after_retest = run_cmd(cmd)

    #filter out everything before "The following tests FAILED:"
    failed_after_retest = failed_after_retest.split("The following tests FAILED:")[1]
    
    # go line by line and extract the test name. The test name is after - and before (
    failed_after_retest = failed_after_retest.splitlines()
    failed_after_retest = [line.split("-")[1].split("(")[0].strip() for line in failed_after_retest if "-" in line]
    resolved_tests = [test for test in candidates if test not in failed_after_retest]

    with open("resolved_tests_{test_case}.txt".format(test_case=test_case), 'w') as file:
        for test in resolved_tests:
            file.write(test + "\n")
