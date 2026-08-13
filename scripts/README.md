# Utility Scripts for chipStar

This directory contains a number of utility scripts for working with chipStar.
## all_check_for_resolved_tests.sh

This script runs `check_for_resolved_tests.py` for all the backends. It is intended to be run on the CI machine `cupcake`.

## check_for_resolved_tests.py

This script tests if any of excluded tests found in `build/test_lists/` are now passing after N nubmer of tries. 
For usage instructions, run `python3 check_for_resolved_tests.py -h`

## check.py

Replacement for running `make check`. Allows for selecting the backend and running `ctest` in parallel. You can also specify a `num_tries` argument to run the tests multiple times to identify flaky tests.

## module-env.sh

Source this to initialize Environment Modules (user-local install, Lmod, or
the system package, in that order) and register the shared modulefiles trees.
CI workflow steps use it as a one-line replacement for the init boilerplate;
building and testing is then explicit per step (cmake + make + `check.py`),
which is what the removed `unit_tests.sh` wrapper used to do behind one
opaque invocation.
