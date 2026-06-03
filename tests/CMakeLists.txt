# Each *.cpp here becomes one CTest test. doctest.h is in this dir.
set(MATSAR_TESTS
        test_smoke
        test_core
        test_public_header
        test_arithmetic
        test_comparison
        test_build_config
        test_print
        test_construct
)

foreach(t IN LISTS MATSAR_TESTS)
    add_executable(${t} ${t}.cpp)
    target_link_libraries(${t} PRIVATE matsar)
    target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${t} COMMAND ${t})
endforeach()

# Death test: invalid-shape op prints a *Exception message and exit()s.
add_executable(death_invalid_shape death_invalid_shape.cpp)
target_link_libraries(death_invalid_shape PRIVATE matsar)
target_include_directories(death_invalid_shape PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME death_invalid_shape COMMAND death_invalid_shape)
set_tests_properties(death_invalid_shape PROPERTIES PASS_REGULAR_EXPRESSION "(Broadcast|Shape)Exception")

# BLAS-backed tests (only when OpenBLAS is enabled)
if(MATSAR_WITH_BLAS)
    set(MATSAR_BLAS_TESTS
            test_mmul
    )
    foreach(t IN LISTS MATSAR_BLAS_TESTS)
        add_executable(${t} ${t}.cpp)
        target_link_libraries(${t} PRIVATE matsar)
        target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
        add_test(NAME ${t} COMMAND ${t})
        # Stage the bundled OpenBLAS DLL next to the test exe on Windows.
        if(WIN32 AND DEFINED MATSAR_BLAS_DLL)
            add_custom_command(TARGET ${t} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${MATSAR_BLAS_DLL}" "$<TARGET_FILE_DIR:${t}>")
        endif()
    endforeach()
endif()
