
add_hiplz_test(fp16 fp16 PASSED haxpy-base.cpp)

# add_hiplz_test(fp16_math fp16_math PASSED half_math.cpp)
add_hiplz_binary(fp16_math half_math.cpp)
