#!/bin/bash
CURRENT_COMMIT=`git_hash=$(git rev-parse --short "$GITHUB_SHA")`
# CURRENT_COMMIT=`git log --oneline --skip=1 -1 | awk '{print $1}'`
echo "CURRENT COMMIT: ${CURRENT_COMMIT}"


OPENCL_UNIT_TEST_COMMIT=`cat opencl_unit_tests_iris.log | awk 'NR==1{print $1}'`
LEVEL0_UNIT_TEST_COMMIT=`cat level0_unit_tests_iris.log | awk 'NR==1{print $1}'`
echo "OpenCL Unit Test Commit: ${OPENCL_UNIT_TEST_COMMIT}"
echo "LevelZero Unit Test Commit: ${LEVEL0_UNIT_TEST_COMMIT}"

if [[ "$CURRENT_COMMIT" = "$OPENCL_UNIT_TEST_COMMIT" ]]; then
    echo "OpenCL unit tests up-to-date"
else
    echo "OpenCL unit tests out-of-date"
    exit 1  
fi

if [[ "$CURRENT_COMMIT" = "$LEVEL0_UNIT_TEST_COMMIT" ]]; then
    echo "LevelZero unit tests up-to-date"
else
    echo "LevelZero unit tests out-of-date"
    exit 1
fi

exit 0