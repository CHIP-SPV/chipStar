# Utility Scripts for CHIP-SPV

This directory contains a number of utility scripts for working with CHIP-SPV.
## all_check_for_resolved_tests.sh

This script runs `check_for_resolved_tests.py` for all the backends. It is intended to be run on the CI machine `cupcake`.

## check_for_resolved_tests.py

This script tests if any of excluded tests found in `build/test_lists/` are now passing after N nubmer of tries. 
For usage instructions, run `python3 check_for_resolved_tests.py -h`

## check.py

Replacement for running `make check`. Allows for selecting the backend and running `ctest` in parallel. You can also specify a `num_tries` argument to run the tests multiple times to identify flaky tests.

## unit_tests.sh

Run the unit tests on the CI machine `cupcake`. This script will exclude the known failures and bind/unbind iGPU/dGPU from the i915 driver to mitigate the driver instability.