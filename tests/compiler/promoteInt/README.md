# Promote Int LLVM Pass Tests

This directory contains tests for the `HipPromoteInts` LLVM pass that promotes non-standard integer types (e.g., i24, i3) to standard types (i32, i8, etc.) that are supported by SPIR-V.

## Test Files

- `small-i24-repro.ll`: Tests handling of i24 integer type with phi nodes
- `illegalZext.ll`: Tests handling of non-standard integer types in combined control flow with zext operations
- `dominance.ll`: Tests handling of non-standard integer types across basic blocks with dominance relationships
- `ext-repro.ll`: Tests more complex external reproduction case for integer promotion
- `warpCalc.ll`: Complex test case from the hipcc test suite converted for direct LLVM testing

## How Tests Work

The tests work by:
1. Running the LLVM IR through the HipPromoteInts pass
2. Verifying that the pass correctly transforms the non-standard integer types
3. Converting the result to SPIR-V to validate correctness

## Running Tests

The tests are run as part of the normal test suite:

```bash
cd /path/to/build/directory
ctest -R compiler_promote_int_tests
```

## Adding New Tests

To add a new test:
1. Create an LLVM IR file (.ll) that contains non-standard integer types
2. Place it in this directory
3. The test script will automatically pick it up and run it

## Known Issues

None at the moment. 