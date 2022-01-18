# hiptest

file(READ strings.xstdout STRINGS_XSTDOUT)
add_chip_test(strings PrintfSimple ${STRINGS_XSTDOUT} strings.cc)

file(READ nop_printfs.xstdout NOP_PRINTFS_XSTDOUT)
add_chip_test(nop_printfs PrintfNOP ${NOP_PRINTFS_XSTDOUT} nop_printfs.cc)

file(READ dynamic_str_args.xstdout DYNAMIC_STR_ARGS_XSTDOUT)
add_chip_test(dynamic_str_args PrintfDynamic ${DYNAMIC_STR_ARGS_XSTDOUT} dynamic_str_args.cc)

# Fails on Level0, so disabled for now.
# We'd need to check this at test execution time.
set_tests_properties(PrintfDynamic PROPERTIES DISABLED YES)

