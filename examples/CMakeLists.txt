# Each *.cpp here is a small, self-contained runnable example linked against the
# MatSAR interface target. Enabled by MATSAR_BUILD_EXAMPLES (ON when MatSAR is
# the top-level project).

set(MATSAR_EXAMPLES
        basic
        broadcasting
)

foreach(ex IN LISTS MATSAR_EXAMPLES)
    add_executable(example_${ex} ${ex}.cpp)
    target_link_libraries(example_${ex} PRIVATE MatSAR::matsar)
    set_target_properties(example_${ex} PROPERTIES OUTPUT_NAME ${ex})
endforeach()

# The matrix-product example needs the BLAS-backed build (mmul / operator%).
if(MATSAR_WITH_BLAS)
    add_executable(example_matmul matmul.cpp)
    target_link_libraries(example_matmul PRIVATE MatSAR::matsar)
    set_target_properties(example_matmul PROPERTIES OUTPUT_NAME matmul)
    # Stage the bundled OpenBLAS DLL next to the exe on Windows (matches tests/).
    if(WIN32 AND DEFINED MATSAR_BLAS_DLL)
        add_custom_command(TARGET example_matmul POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${MATSAR_BLAS_DLL}" "$<TARGET_FILE_DIR:example_matmul>")
    endif()
endif()
