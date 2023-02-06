#!/usr/bin/env python3
import sys
import os
import argparse
from util import run_cmd

parser = argparse.ArgumentParser(
                    prog = 'ProgramName',
                    description = 'What the program does',
                    epilog = 'Text at the bottom of help')

def process_args():
    usage_line = "Usage: python3 check.py <path to build dir> <cpu/igpu/dgpu> <opencl/level0/PoCL> <num threads> <num_tries>"
    if (len(sys.argv) != 6):
        print("not enough args")
        print(sys.argv)
        print(usage_line)
        sys.exit(1)

    work_dir = sys.argv[1]
    if sys.argv[2] == "cpu":
        device_type = "cpu"
    elif sys.argv[2] == "dgpu":
        device_type = "dgpu"
    elif sys.argv[2] == "igpu":
        device_type = "igpu"
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

resolved_tests = {}


work_dir, device_type, backend, num_threads, num_tries = process_args()
os.chdir(work_dir)
if(backend == "pocl" or backend == "opencl"):
    env_vars = "CHIP_BE=opencl CHIP_DEVICE_TYPE={device_type}".format(backend=backend, device_type=device_type)
else:
    env_vars = "CHIP_BE=level0 CHIP_DEVICE_TYPE={device_type} CHIP_DEVICE_NUM=0".format(backend=backend, device_type=device_type)

cmd = "{env_vars} ctest --timeout 180 --repeat until-fail:{num_tries} -j {num_threads} -E \"`cat ./test_lists/{device_type}_{backend}_failed_tests.txt`\"".format(work_dir=work_dir, num_tries=num_tries, env_vars=env_vars, num_threads=num_threads, device_type=device_type, backend=backend)
run_cmd(cmd)
