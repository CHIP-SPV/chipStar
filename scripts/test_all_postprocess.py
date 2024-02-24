#!/usr/bin/env python3
import re

def extract_test_name_and_reason(line :str) -> tuple[str, str]:
    match = re.search(r' - (.*) \((.*)\)', line)
    if match:
        name =  match.group(1)
        reason = match.group(2)
    else:
        raise Exception(f"Failed to extract name and reason from line: {line}") 

    # print (f"Name: {name}, Reason: {reason}")
    return name, reason


def extract_failed_tests(log_file : str) -> list[tuple[str, str]]:
    failed_tests = []
    lines = open(log_file, 'r').readlines()
    # if the first line is "The following tests FAILED:", skip it
    if "The following tests FAILED:" in lines[0]:
        lines = lines[1:]
    
    for line in lines:
        failed_tests.append(extract_test_name_and_reason(line))
    return failed_tests

def find_commonly_failed_tests(test_failures : list[list[tuple[str, str]]]) -> list[tuple[int, list[str]]]:
    # Create a dictionary to count the number of times each test has failed
    test_counts : dict[str, list[str]] = {}
    for test_list in test_failures:
        for test in test_list:
            if test[0] in test_counts:
                test_counts[test[0]].append(test[1])
            else:
                test_counts[test[0]] = [test[1]]
    # Find the tests that have failed in all lists
    common_failures = []
    for test in test_counts:
        if len(test_counts[test]) == len(test_failures):
            common_failures.append((test, test_counts[test]))
    
    return common_failures
                
                
def generate_cmakeformat(failures, format):
    print("\n\n")
    lines = []
    for fail in failures:
        # check if fail[1] is a list
        if isinstance(fail[1], list):
            reasons_str = ""
            for r in fail[1]:
                reasons_str += f"{r}|"
        else:
            reasons_str = fail[1]
        pass
        test_name = fail[0]
        line = f"list(APPEND {format} \"{test_name}\") # {reasons_str}"
        lines.append(line)
        print(line)
    return lines

opencl_dgpu = extract_failed_tests("alltest_opencl_dgpu_failed.log")
opencl_igpu = extract_failed_tests("alltest_opencl_igpu_failed.log")
level0icl_dgpu = extract_failed_tests("alltest_level0icl_dgpu_failed.log")
level0icl_igpu = extract_failed_tests("alltest_level0icl_igpu_failed.log")
level0rcl_dgpu = extract_failed_tests("alltest_level0rcl_dgpu_failed.log")
level0rcl_igpu = extract_failed_tests("alltest_level0rcl_igpu_failed.log")
opencl_cpu = extract_failed_tests("alltest_opencl_cpu_failed.log")
opencl_pocl = extract_failed_tests("alltest_pocl_failed.log")

common_failures = find_commonly_failed_tests([opencl_dgpu, opencl_igpu, level0icl_dgpu, level0icl_igpu, opencl_cpu])
print(f"Number of tests failing for all platforms: {len(common_failures)}")

# remove commonly failing tests from the individual lists
for test in common_failures:
    opencl_dgpu = [x for x in opencl_dgpu if x[0] != test[0]]
    opencl_igpu = [x for x in opencl_igpu if x[0] != test[0]]
    level0icl_dgpu = [x for x in level0icl_dgpu if x[0] != test[0]]
    level0icl_igpu = [x for x in level0icl_igpu if x[0] != test[0]]
    level0rcl_dgpu = [x for x in level0rcl_dgpu if x[0] != test[0]]
    level0rcl_igpu = [x for x in level0rcl_igpu if x[0] != test[0]]
    opencl_cpu = [x for x in opencl_cpu if x[0] != test[0]]
    opencl_pocl = [x for x in opencl_pocl if x[0] != test[0]]

generate_cmakeformat(common_failures, "FAILING_FOR_ALL")
generate_cmakeformat(opencl_dgpu, "DGPU_OPENCL_FAILED_TESTS")
generate_cmakeformat(opencl_igpu, "IGPU_OPENCL_FAILED_TESTS")
generate_cmakeformat(level0icl_dgpu, "DGPU_LEVEL0_ICL_FAILED_TESTS")
generate_cmakeformat(level0icl_igpu, "IGPU_LEVEL0_ICL_FAILED_TESTS")
generate_cmakeformat(level0rcl_dgpu, "DGPU_LEVEL0_RCL_FAILED_TESTS")
generate_cmakeformat(level0rcl_igpu, "IGPU_LEVEL0_RCL_FAILED_TESTS")
generate_cmakeformat(opencl_cpu, "CPU_OPENCL_FAILED_TESTS")
generate_cmakeformat(opencl_pocl, "CPU_POCL_FAILED_TESTS")
