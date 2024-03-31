#!/usr/bin/env python3
import yaml
import argparse
import os

parser = argparse.ArgumentParser(
    prog="check.py",
    description="Run the unit tests for the specified device type and backend",
    epilog="have a nice day",
)

parser.add_argument(
    "known_failures_path", help="Path to the known_failures.yaml file", type=str
)

parser.add_argument(
    "--generate",
    type=str,
    help="Generate test_lists files at the specified output path",
)

parser.add_argument(
    "--cleanup", action="store_true", help="Cleanup the known_failures.yaml file"
)

parser.add_argument(
    "--print", action="store_true", help="Pretty print the known_failures.yaml file"
)


args = parser.parse_args()

categories = ["ALL", "OPENCL_GPU", "OPENCL_CPU", "OPENCL_POCL", "LEVEL0_GPU"]


def dump_known_failures_to_yaml(known_failures, yaml_path, total_tests):
    known_failures["TOTAL_TESTS"] = total_tests
    with open(yaml_path, 'w') as file:
        yaml.dump(known_failures, file)


def load_known_failures_from_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        known_failures = yaml.safe_load(file)
    total_tests = known_failures.pop("TOTAL_TESTS", None)
    return known_failures, int(total_tests)


def prune_tests_map(tests_map):
    # Define the categories to check
    categories_to_check = ["OPENCL_GPU", "OPENCL_CPU", "OPENCL_POCL", "LEVEL0_GPU"]
    opencl_categories = ["OPENCL_GPU", "OPENCL_CPU", "OPENCL_POCL"]

    # Find all tests that are in all categories
    common_tests = set(tests_map[categories_to_check[0]].keys())
    for category in categories_to_check[1:]:
        common_tests &= set(tests_map[category].keys())

    # Remove common tests from their categories and add them to "ALL"
    for test in common_tests:
        for category in categories_to_check:
            del tests_map[category][test]
        tests_map["ALL"][test] = ""

    # If a test appears in any of the OPENCL categories and also in LEVEL0_GPU, add it to "ALL"
    for test in tests_map["LEVEL0_GPU"].keys():
        for category in opencl_categories:
            if test in tests_map[category]:
                del tests_map[category][test]
                tests_map["ALL"][test] = ""

    # Ensure tests in "ALL" do not appear in any other category
    for test in tests_map["ALL"].keys():
        for category in tests_map:
            if category != "ALL" and test in tests_map[category]:
                del tests_map[category][test]

    # Sort the tests in each category by their names
    for category in tests_map:
        tests_map[category] = dict(sorted(tests_map[category].items()))

    return tests_map

def pretty_print_known_failures(known_failures, total_tests):
    all_tests = set(known_failures.get("ALL", {}).keys())
    for category, tests in known_failures.items():
        if category == "ALL":
            continue
        category_failures = set(tests.keys())
        unique_failures = category_failures - all_tests
        total_failures = category_failures.union(all_tests)
        num_unique_failures = len(unique_failures)
        num_total_failures = len(total_failures)
        pass_rate = ((total_tests - num_total_failures) / total_tests) * 100
        print(f"{category} - Unique Failures: {num_unique_failures}, Total Failures: {num_total_failures}, Pass Rate: {pass_rate:.2f}%")
        for test in unique_failures:
            print(f"\t{test}")
    num_all_failures = len(all_tests)
    all_pass_rate = ((total_tests - num_all_failures) / total_tests) * 100
    print(f"ALL - Total Failures: {num_all_failures}, Pass Rate: {all_pass_rate:.2f}%")


def generate_test_string(tests_map, output_dir):
    test_string_map = {}
    for category, tests in tests_map.items():
        if tests is not None:  # Add this check
            test_string = "$|".join(tests.keys()) + "$"
            test_string_map[category] = test_string
            # print(f"{category}\n {test_string}")
            with open(f"{output_dir}/{category}.txt", "+w") as file:
                file.write(test_string)
    return test_string_map


def main():
    known_failures, total_tests = load_known_failures_from_yaml(args.known_failures_path)
    if args.generate:
        print("Generating test_lists files")
        # make sure output_dir exists, if not create it
        if not os.path.exists(args.generate):
            os.makedirs(args.generate)
        generate_test_string(known_failures, args.generate)
    elif args.cleanup:
        print("Cleaning up known_failures.yaml")
        known_failures = prune_tests_map(known_failures)
        dump_known_failures_to_yaml(known_failures, args.known_failures_path, total_tests)
    elif args.print:
        pretty_print_known_failures(known_failures, total_tests)
    else:
        print("No action specified. Use --generate, --cleanup, or --print")


if __name__ == "__main__":
    main()
