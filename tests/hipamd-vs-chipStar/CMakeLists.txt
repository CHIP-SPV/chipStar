set(hipamd_compiler /usr/bin/hipcc)
set(chipstar_compiler ${CMAKE_BINARY_DIR}/bin/hipcc)

set(hipamd_ld_preload /opt/rocm-6.2.4/lib/libamdhip64.so.6.2.60204)
set(chipstar_ld_preload ${CMAKE_BINARY_DIR}/lib/libCHIP.so)

message(STATUS "hipamd_compiler: ${hipamd_compiler}")
message(STATUS "chipstar_compiler: ${chipstar_compiler}")
message(STATUS "hipamd_ld_preload: ${hipamd_ld_preload}")
message(STATUS "chipstar_ld_preload: ${chipstar_ld_preload}")

function(add_hip_test_hipamd_vs_chipstar test_name source_file)
    set(hipamd_output ${CMAKE_CURRENT_BINARY_DIR}/${test_name}_hipamd)
    set(chipstar_output ${CMAKE_CURRENT_BINARY_DIR}/${test_name}_chipstar)
    
    # Build HIP-AMD version
    add_custom_command(
        OUTPUT ${hipamd_output}
        COMMAND ${hipamd_compiler} -D__HIP_PLATFORM_AMD__=1 -w
                ${CMAKE_CURRENT_SOURCE_DIR}/${source_file} 
                -o ${hipamd_output}
        DEPENDS ${source_file}
        COMMENT "Building HIP-AMD version of ${test_name}"
    )

    # Build chipStar version
    add_custom_command(
        OUTPUT ${chipstar_output}
        COMMAND ${chipstar_compiler} -D__HIP_PLATFORM_SPIRV__=1 -w
                ${CMAKE_CURRENT_SOURCE_DIR}/${source_file} 
                -o ${chipstar_output}
        DEPENDS ${source_file} CHIP
        COMMENT "Building chipStar version of ${test_name}"
    )

    # Add custom target to trigger the builds
    add_custom_target(${test_name} ALL
        DEPENDS ${hipamd_output} ${chipstar_output} CHIP
    )

    # Add test that compares outputs
    add_test(NAME ${test_name}_compare
        COMMAND ${CMAKE_COMMAND} -P 
            ${CMAKE_CURRENT_SOURCE_DIR}/compare_outputs.cmake
            ${hipamd_output}
            ${chipstar_output}
            ${hipamd_ld_preload}
            ${chipstar_ld_preload})
endfunction()

add_hip_test_hipamd_vs_chipstar(bpermute bpermute.cpp)