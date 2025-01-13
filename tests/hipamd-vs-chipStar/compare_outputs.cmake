# Get arguments passed from CMake
set(hipamd_exe ${CMAKE_ARGV3})
set(chipstar_exe ${CMAKE_ARGV4})
set(hipamd_ld_preload ${CMAKE_ARGV5})
set(chipstar_ld_preload ${CMAKE_ARGV6})

# Run HIP-AMD version
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env "LD_PRELOAD=${hipamd_ld_preload}" ${hipamd_exe}
    OUTPUT_VARIABLE hipamd_output
    ERROR_VARIABLE hipamd_error
    RESULT_VARIABLE hipamd_result)

if(NOT hipamd_result EQUAL 0)
    message(FATAL_ERROR "HIP-AMD execution failed: ${hipamd_error}")
endif()

# Run chipStar version
execute_process(
    COMMAND ${CMAKE_COMMAND} -E env "LD_PRELOAD=${chipstar_ld_preload}" ${chipstar_exe}
    OUTPUT_VARIABLE chipstar_output
    ERROR_VARIABLE chipstar_error
    RESULT_VARIABLE chipstar_result)

if(NOT chipstar_result EQUAL 0)
    message(FATAL_ERROR "chipStar execution failed: ${chipstar_error}")
endif()

# Compare outputs
if(NOT hipamd_output STREQUAL chipstar_output)
    message(FATAL_ERROR "Outputs differ!\nHIP-AMD output:\n${hipamd_output}\nchipStar output:\n${chipstar_output}")
endif()

message(STATUS "Test passed - outputs match") 