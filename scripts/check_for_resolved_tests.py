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
args = parser.parse_args()
print(args)

if args.backend == "level0-reg":
    level0_cmd_list = "reg_"
    args.backend = "level0"
elif args.backend == "level0-imm":
    level0_cmd_list = "imm_"
    args.backend = "level0"
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

# TODO if adding more than one check for invocation
# resolved_tests = {}

precise_device_type = args.device_type
agg_device_type = args.device_type
if agg_device_type == "igpu" or agg_device_type == "dgpu":
    agg_device_type = "gpu"

env_vars = "CHIP_BE={backend} CHIP_DEVICE_TYPE={device_type}".format(backend=args.backend, device_type=agg_device_type)
print("Checking for resolved tests for {device_type} {backend}".format(device_type=precise_device_type, backend=args.backend))
os.chdir(args.path)
out = run_cmd("{env_vars} ctest --timeout 120 -j {num_threads}  -R \"`cat ./test_lists/{precise_device_type}_{backend}_failed_{level0_cmd_list}tests.txt`\"".format(level0_cmd_list=level0_cmd_list, backend=args.backend, precise_device_type=precise_device_type, env_vars=env_vars, num_threads=args.num_threads))
with open("1.initial_passed_tests.txt", 'w') as file:
    file.write(out)
potentially_resolved_tests = re.findall(r".*?\: (.*?) \.+   Passed", out)
with open("2.potentially_resolved_tests.txt", 'w') as file:
    file.write(str(potentially_resolved_tests))
tests = []
for test in potentially_resolved_tests:
    cmd = "{env_vars} ctest --timeout 120 -j {num_threads} --repeat until-fail:{num_tries} -R '^{test}$'".format(num_threads=args.num_threads, env_vars=env_vars, num_tries=1, test=test)
    out = run_cmd(cmd)
    if "100% tests passed" in out:
        print("{test} has been resolved!".format(test=test))
        tests.append(test)

test_case = "{precise_device_type}_{backend}".format(precise_device_type=precise_device_type, backend=args.backend)
# TODO if adding more than one check for invocation
# resolved_tests[test_case] = tests 
# print("\n\n")
# for test_case in resolved_tests:
#     for test in resolved_tests[test_case]:
#         print("Resolved: {case} {test}".format(case=test_case, test=test))


with open("{test_case}_resolved_tests.txt".format(test_case=test_case), 'w') as file:
    for test in tests:
        file.write(test + "\n")
    #  file.write(json.dumps(resolved_tests)) # use `json.loads` to do the reverse
