#!/usr/bin/env python3
import subprocess
import sys
import os
import re
import json

def process_args():
    usage_line = "Usage: python3 check.py <path to build dir> <cpu/gpu> <opencl/level0/PoCL> <num threads> <num_tries>"
    if (len(sys.argv) != 5):
        print(sys.argv)
        print(usage_line)
        sys.exit(1)

    work_dir = sys.argv[1]
    if sys.argv[2] == "cpu":
        device_type = "cpu"
    elif sys.argv[2] == "gpu":
        device_type = "gpu"
    else:
        print("Unrecognized device type: " + sys.argv[2])
        print(usage_line)
        sys.exit(1)
    if sys.argv[3] == "opencl":
        backend = "opencl"
    elif sys.argv[3] == "level0":
        backend = "level0"
    elif sys.argv[3] == "pocl":
        backend = "pocl"
    else:
        print("Unrecognized backend: " + sys.argv[3])
        print(usage_line)
        sys.exit(1)
    num_threads = sys.argv[4] 
    num_tries = sys.argv[5]
    return work_dir, device_type, backend, num_threads, num_tries

def run_cmd(cmd):
   subprocess.call("rm -f last_cmd.txt", shell=True)  
   cmd = cmd + " | tee last_cmd.txt"
   print("Running command: " + cmd)
   subprocess.call(cmd, shell=True) 
   with open("last_cmd.txt", "r") as f:
    return f.read()

resolved_tests = {}


work_dir, device_type, backend, num_threads, num_tries = process_args()
os.chdir(work_dir)
if(backend == "pocl" or backend == "opencl"):
    env_vars = "CHIP_BE=opencl CHIP_DEVICE_TYPE={device_type}".format(backend=backend, device_type=device_type)
else:
    env_vars = "CHIP_BE=level0 CHIP_DEVICE_TYPE={device_type} CHIP_DEVICE_NUM=0".format(backend=backend, device_type=device_type)

cmd = "{env_vars} ctest --timeout 180 --repeat until-fail:{num_tries} -j {num_threads} -E \"`cat ./test_lists/{device_type}_{backend}_failed_tests.txt`\"".format(num_tries=num_tries, env_vars=env_vars, num_threads=num_threads, device_type=device_type, backend=backend)
run_cmd(cmd)