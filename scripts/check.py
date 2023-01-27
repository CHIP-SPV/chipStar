#!/usr/bin/env python3
import subprocess
import sys
import os
import re
import json

# args: <path to build dir> <cpu/gpu> <opencl/level0/PoCL> <num threads>
def process_args():
    if (len(sys.argv) != 5):
        print(sys.argv)
        print("Usage: python3 check.py <path to build dir> <cpu/gpu> <opencl/level0/PoCL> <num threads>")
        sys.exit(1)

    work_dir = sys.argv[1]
    if sys.argv[2] == "cpu":
        device_type = "cpu"
    elif sys.argv[2] == "gpu":
        device_type = "gpu"
    else:
        print("Unrecognized device type: " + sys.argv[2])
        print("Usage: python3 check.py <path to build dir> <cpu/gpu> <opencl/level0/PoCL> <num threads>")
        sys.exit(1)
    if sys.argv[3] == "opencl":
        backend = "opencl"
    elif sys.argv[3] == "level0":
        backend = "level0"
    elif sys.argv[3] == "pocl":
        backend = "pocl"
    else:
        print("Unrecognized backend: " + sys.argv[3])
        print("Usage: python3 check.py <path to build dir> <cpu/gpu> <opencl/level0/PoCL> <num threads>")
        sys.exit(1)
    num_threads = sys.argv[4] 
    return work_dir, device_type, backend, num_threads

test_cases = [
    ("CHIP_BE=opencl CHIP_DEVICE_TYPE=cpu", "cpu_pocl"),
    ("CHIP_BE=opencl CHIP_DEVICE_TYPE=cpu", "cpu_opencl"),
    ("CHIP_BE=opencl CHIP_DEVICE_TYPE=gpu", "igpu_opencl"),
    ("CHIP_BE=opencl CHIP_DEVICE_TYPE=gpu", "dgpu_opencl"),
    ("CHIP_BE=level0 CHIP_DEVICE_TYPE=gpu", "igpu_level0"),
    ("CHIP_BE=level0 CHIP_DEVICE_TYPE=gpu", "dgpu_level0")
]

def run_cmd(cmd):
   subprocess.call("rm -f last_cmd.txt", shell=True)  
   cmd = cmd + " | tee last_cmd.txt"
   print("Running command: " + cmd)
   subprocess.call(cmd, shell=True) 
   with open("last_cmd.txt", "r") as f:
    return f.read()

resolved_tests = {}


work_dir, device_type, backend, num_threads = process_args()
os.chdir(work_dir)
if(backend == "pocl" or backend == "opencl"):
    env_vars = "CHIP_BE=opencl CHIP_DEVICE_TYPE={device_type}".format(backend=backend, device_type=device_type)
else:
    env_vars = "CHIP_BE=level0 CHIP_DEVICE_TYPE={device_type} CHIP_DEVICE_NUM=0".format(backend=backend, device_type=device_type)

cmd = "{env_vars} ctest --timeout 180 -j {num_threads} -E \"`cat ./test_lists/{device_type}_{backend}_failed_tests.txt`\"".format(env_vars=env_vars, num_threads=num_threads, device_type=device_type, backend=backend)
run_cmd(cmd)

# for env_vars, case in test_cases:
#     if "opencl" in case:
#         num_threads = 1
#     else:
#         num_threads = 1
#     out = run_cmd("{env_vars} ctest --timeout 180 -j {num_threads}  -R \"`cat ./test_lists/{case}_failed_tests.txt`\"".format(num_threads=num_threads, env_vars=env_vars, case=case))
#     potentially_resolved_tests = re.findall(r".*?\: (.*?) .*Passed", out)
#     tests = []
#     for test in potentially_resolved_tests:
#         cmd = "{env_vars} ctest --timeout 180 --repeat until-fail:{num_tries} -R '^{test}$'".format(env_vars=env_vars, num_tries=1, test=test)
#         out = run_cmd(cmd)
#         if "100% tests passed" in out:
#             print("Test " + test + " has been resolved for " + case)
#             tests.append(test)
#     resolved_tests[case] = tests 

# print("\n\n")
# for case in resolved_tests:
#     for test in resolved_tests[case]:
#         print("Resolved: {case} {test}".format(case=case, test=test))

# with open('resolved_unit_tests.txt', 'w') as file:
#      file.write(json.dumps(resolved_tests)) # use `json.loads` to do the reverse

# if len(resolved_tests) > 0:
#     sys.exit(1)
