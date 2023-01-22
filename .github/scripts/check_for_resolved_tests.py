#!/usr/bin/env python3
import subprocess
import sys
import os
import re

test_cases = [
    ("CHIP_BE=opencl CHIP_DEVICE_TYPE=gpu CHIP_PLATFORM=4 CHIP_DEVICE=0 ", "igpu_opencl"),
    ("CHIP_BE=opencl CHIP_DEVICE_TYPE=gpu CHIP_PLATFORM=3 CHIP_DEVICE=0", "dgpu_opencl"),
    ("CHIP_BE=opencl CHIP_DEVICE_TYPE=cpu CHIP_PLATFORM=1 CHIP_DEVICE=0", "cpu_opencl"),
    ("CHIP_BE=level0 CHIP_DEVICE=1", "igpu_level0"),
    ("CHIP_BE=level0 CHIP_DEVICE=0 ", "dgpu_level0")
]

def test_cmd(cmd):
   subprocess.call("rm log.txt", shell=True)  
   cmd = cmd + " | tee log.txt"
   print("Running command: " + cmd)
   subprocess.call(cmd, shell=True) 
   with open("log.txt", "r") as f:
    return f.read()

def run_cmd(cmd):
    stdout, stderr = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE).communicate()
    return stdout.decode(), stderr.decode()
    
resolved_tests = {}

print("Checking for tests that have been resolved in " ,sys.argv[1])
os.chdir(sys.argv[1])
for env_vars, case in test_cases:
    out = test_cmd("{env_vars} ctest --timeout 180 -j 8  -R \"`cat ./test_lists/{case}_failed_tests.txt`\"".format(env_vars=env_vars, case=case))
    potentially_resolved_tests = re.findall(r".*?\: (.*?) .*Passed", out)
    tests = []
    for test in potentially_resolved_tests:
        cmd = "{env_vars} ctest --timeout 180 --repeat until-fail:{num_tries} -R '^{test}$'".format(env_vars=env_vars, num_tries=100, test=test)
        out = test_cmd(cmd)
        if "100% tests passed" in out:
            print("Test " + test + " has been resolved for " + case)
            tests.append(test)
    resolved_tests[case] = tests 

print("\n\n")
for test in resolved_tests:
    print("Test " + test + " has been resolved")

with open('resolved_unit_tests.txt', 'w') as file:
     file.write(json.dumps(resolved_tests)) # use `json.loads` to do the reverse

if len(resolved_tests) > 0:
    sys.exit(1)