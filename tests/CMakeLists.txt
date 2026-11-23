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
        test_number_ops
)

foreach(t IN LISTS MATSAR_TESTS)
    add_executable(${t} ${t}.cpp)
    target_link_libraries(${t} PRIVATE matsar)
    target_include_directories(${t} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
    add_test(NAME ${t} COMMAND ${t})
endforeach()

# Two translation units including MatSAR.hpp: catches non-inline definitions in headers.
add_executable(test_multi_tu test_multi_tu.cpp multi_tu_other.cpp)
target_link_libraries(test_multi_tu PRIVATE matsar)
target_include_directories(test_multi_tu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
add_test(NAME test_multi_tu COMMAND test_multi_tu)

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

# Experimental math-module tests (NumPy-golden + property). Opt-in; excluded from the public mirror.
if(MATSAR_BUILD_EXPERIMENTAL_TESTS)
    add_executable(test_mathfunc math/test_mathfunc.cpp)
    target_link_libraries(test_mathfunc PRIVATE matsar)
    target_include_directories(test_mathfunc PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}                 # doctest.h
            ${CMAKE_CURRENT_SOURCE_DIR}/math/include    # approx.hpp/golden.hpp/rng.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/math/golden)    # golden_mathfunc.hpp
    target_compile_definitions(test_mathfunc PRIVATE MATSAR_ENABLE_EXPERIMENTAL)
    add_test(NAME test_mathfunc COMMAND test_mathfunc)

    # NumPy-golden replay for Trigonometry.hpp (golden_trig.hpp).
    add_executable(test_mathtrig math/test_trig.cpp)
    target_link_libraries(test_mathtrig PRIVATE matsar)
    target_include_directories(test_mathtrig PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}                 # doctest.h
            ${CMAKE_CURRENT_SOURCE_DIR}/math/include    # approx.hpp/golden.hpp/rng.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/math/golden)    # golden_trig.hpp
    target_compile_definitions(test_mathtrig PRIVATE MATSAR_ENABLE_EXPERIMENTAL)
    add_test(NAME test_mathtrig COMMAND test_mathtrig)

    # Value tests for MatrixOperation.hpp (golden_matop.hpp + hand-derived cases).
    add_executable(test_mathmatop math/test_matop.cpp)
    target_link_libraries(test_mathmatop PRIVATE matsar)
    target_include_directories(test_mathmatop PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}                 # doctest.h
            ${CMAKE_CURRENT_SOURCE_DIR}/math/include    # approx.hpp/golden.hpp/rng.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/math/golden)    # golden_matop.hpp
    target_compile_definitions(test_mathmatop PRIVATE MATSAR_ENABLE_EXPERIMENTAL)
    add_test(NAME test_mathmatop COMMAND test_mathmatop)

    # Death tests for MatrixOperation: one executable, the case is chosen by argv[1].
    add_executable(death_matop math/death_matop.cpp)
    target_link_libraries(death_matop PRIVATE matsar)
    target_compile_definitions(death_matop PRIVATE MATSAR_ENABLE_EXPERIMENTAL)
    foreach(case_regex IN ITEMS
            "ipermute_range|ValueError: ipermute: The number in dimorder must be from 1 to n"
            "ipermute_repeated|ValueError: ipermute: The numbers in dimorder are repeated"
            "factorial_nonint|ValueError: Factorial requires an integer"
            "factorial_negative|ValueError: Factorial cannot be negative"
            "numden_zero|ValueError: numden: 0/0 is undefined"
            "expm_nonsquare|ValueError: expm: requires a square matrix"
            "logm_negative|ValueError: logm: .*negative real axis.*logm<complex128>"
            "logm_singular|ValueError: logm: the matrix is singular"
            "mpower_singular|ValueError: mpower: the matrix is singular"
            "mpower_two_matrices|ValueError: mpower: the base or the exponent must be a scalar"
            "mpower_negative_base|ValueError: mpower: .*negative base.*complex")
        string(FIND "${case_regex}" "|" bar)
        string(SUBSTRING "${case_regex}" 0 ${bar} case)
        math(EXPR msg_start "${bar} + 1")
        string(SUBSTRING "${case_regex}" ${msg_start} -1 regex)
        add_test(NAME death_matop_${case} COMMAND death_matop ${case})
        set_tests_properties(death_matop_${case} PROPERTIES PASS_REGULAR_EXPRESSION "${regex}")
    endforeach()

    # Value tests for Probability.hpp (golden_prob.hpp + statistical checks of the generators).
    add_executable(test_mathprob math/test_prob.cpp)
    target_link_libraries(test_mathprob PRIVATE matsar)
    target_include_directories(test_mathprob PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}                 # doctest.h
            ${CMAKE_CURRENT_SOURCE_DIR}/math/include    # approx.hpp/golden.hpp/rng.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/math/golden)    # golden_prob.hpp
    target_compile_definitions(test_mathprob PRIVATE MATSAR_ENABLE_EXPERIMENTAL)
    add_test(NAME test_mathprob COMMAND test_mathprob)

    # Death tests for Probability: one executable, the case is chosen by argv[1].
    add_executable(death_prob math/death_prob.cpp)
    target_link_libraries(death_prob PRIVATE matsar)
    target_compile_definitions(death_prob PRIVATE MATSAR_ENABLE_EXPERIMENTAL)
    foreach(case_regex IN ITEMS
            "randi_order|ValueError: randi requires the minimum"
            "randperm_k|ValueError: randperm: k"
            "randperm_negative|ValueError: randperm: n"
            "normrnd_shape|ShapeException: normrnd"
            "unifpdf_shape|ShapeException: unifpdf"
            "pdf_shape|ShapeException: poisspdf")
        string(FIND "${case_regex}" "|" bar)
        string(SUBSTRING "${case_regex}" 0 ${bar} case)
        math(EXPR msg_start "${bar} + 1")
        string(SUBSTRING "${case_regex}" ${msg_start} -1 regex)
        add_test(NAME death_prob_${case} COMMAND death_prob ${case})
        set_tests_properties(death_prob_${case} PROPERTIES PASS_REGULAR_EXPRESSION "${regex}")
    endforeach()

    # Every math header in two translation units of one program (link-time check).
    add_executable(test_math_multi_tu math/multi_tu_a.cpp math/multi_tu_b.cpp)
    target_link_libraries(test_math_multi_tu PRIVATE matsar)
    target_compile_definitions(test_math_multi_tu PRIVATE MATSAR_ENABLE_EXPERIMENTAL)
    add_test(NAME test_math_multi_tu COMMAND test_math_multi_tu)
    if(WIN32 AND MATSAR_WITH_BLAS AND DEFINED MATSAR_BLAS_DLL)
        add_custom_command(TARGET test_math_multi_tu POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${MATSAR_BLAS_DLL}" "$<TARGET_FILE_DIR:test_math_multi_tu>")
    endif()

    # Self-test for the harness's own helpers (approx_eq / build<T> / check_matrix).
    add_executable(selftest_helpers math/selftest_helpers.cpp)
    target_link_libraries(selftest_helpers PRIVATE matsar)
    target_include_directories(selftest_helpers PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}                 # doctest.h
            ${CMAKE_CURRENT_SOURCE_DIR}/math/include)   # approx.hpp/golden.hpp/rng.hpp
    target_compile_definitions(selftest_helpers PRIVATE MATSAR_ENABLE_EXPERIMENTAL)
    add_test(NAME selftest_helpers COMMAND selftest_helpers)

    # Hand-derived value tests: regressions for fixed wrong-result bugs.
    add_executable(test_mathvalues math/test_values.cpp)
    target_link_libraries(test_mathvalues PRIVATE matsar)
    target_include_directories(test_mathvalues PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}                 # doctest.h
            ${CMAKE_CURRENT_SOURCE_DIR}/math/include)
    target_compile_definitions(test_mathvalues PRIVATE MATSAR_ENABLE_EXPERIMENTAL)
    add_test(NAME test_mathvalues COMMAND test_mathvalues)

    # Death test: permute() rejects a repeated dimension with its own ValueError.
    add_executable(death_permute_repeated math/death_permute_repeated.cpp)
    target_link_libraries(death_permute_repeated PRIVATE matsar)
    target_compile_definitions(death_permute_repeated PRIVATE MATSAR_ENABLE_EXPERIMENTAL)
    add_test(NAME death_permute_repeated COMMAND death_permute_repeated)
    set_tests_properties(death_permute_repeated PROPERTIES
            PASS_REGULAR_EXPRESSION "ValueError: permute: The numbers in dimorder are repeated")

    # Value tests for the BLAS-free part of LinearAlgebra.hpp (golden_linalg.hpp).
    add_executable(test_mathlinalg math/test_linalg.cpp)
    target_include_directories(test_mathlinalg PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}                 # doctest.h
            ${CMAKE_CURRENT_SOURCE_DIR}/math/include    # approx.hpp/golden.hpp/rng.hpp
            ${CMAKE_CURRENT_SOURCE_DIR}/math/golden)    # golden_linalg.hpp
    add_test(NAME test_mathlinalg COMMAND test_mathlinalg)

    # Death tests for LinearAlgebra: one executable, the case is chosen by argv[1].
    add_executable(death_linalg math/death_linalg.cpp)
    set(MATSAR_LINALG_DEATH
            "trace_nonsquare|ValueError: trace requires the matrix to be square"
            "kron_3d|ValueError: kron: kron requires"
            "ppval_pieces|ValueError: ppval: coefs must have one row per piece"
            "ppval_breaks|ValueError: ppval: at least two breaks")
    set(MATSAR_LINALG_TARGETS test_mathlinalg death_linalg)

    # LAPACK-backed value tests (svd, det, inv, pinv, lu, eig, schur, ...): only when OpenBLAS
    # (+ LAPACKE) is enabled.
    if(MATSAR_WITH_BLAS)
        add_executable(test_mathlapack math/test_lapack.cpp)
        target_include_directories(test_mathlapack PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}                 # doctest.h
                ${CMAKE_CURRENT_SOURCE_DIR}/math/include    # approx.hpp/golden.hpp/rng.hpp
                ${CMAKE_CURRENT_SOURCE_DIR}/math/golden)    # golden_lapack.hpp
        add_test(NAME test_mathlapack COMMAND test_mathlapack)
        list(APPEND MATSAR_LINALG_TARGETS test_mathlapack)
        list(APPEND MATSAR_LINALG_DEATH
                "eig_complex_real|ValueError: eig: .*complex eigenvalues.*eig<complex128>"
                "condeig_complex_real|ValueError: condeig: .*complex eigenvalues.*condeig<complex128>"
                "schur_flag|ValueError: schur: .*'real' or 'complex'"
                "schur_complex_real|ValueError: schur: .*complex eigenvalues.*schur<complex128>"
                "det_nonsquare|ValueError: det: requires a square matrix"
                "inv_nonsquare|ValueError: inv: requires a 2-D square matrix"
                "pinv_3d|ValueError: pinv: requires a 2-D matrix"
                "lu_3d|ValueError: lu requires a 2D matrix"
                "svd_1d|ValueError: svd: svd requires Matrix at least 2 dimensions"
                "polyvalm_nonsquare|ValueError: polyvalm: requires a square matrix")
    endif()
    foreach(t IN LISTS MATSAR_LINALG_TARGETS)
        target_link_libraries(${t} PRIVATE matsar)
        target_compile_definitions(${t} PRIVATE MATSAR_ENABLE_EXPERIMENTAL)
        if(WIN32 AND MATSAR_WITH_BLAS AND DEFINED MATSAR_BLAS_DLL)
            add_custom_command(TARGET ${t} POST_BUILD
                    COMMAND ${CMAKE_COMMAND} -E copy_if_different
                    "${MATSAR_BLAS_DLL}" "$<TARGET_FILE_DIR:${t}>")
        endif()
    endforeach()
    foreach(case_regex IN LISTS MATSAR_LINALG_DEATH)
        string(FIND "${case_regex}" "|" bar)
        string(SUBSTRING "${case_regex}" 0 ${bar} case)
        math(EXPR msg_start "${bar} + 1")
        string(SUBSTRING "${case_regex}" ${msg_start} -1 regex)
        add_test(NAME death_linalg_${case} COMMAND death_linalg ${case})
        set_tests_properties(death_linalg_${case} PROPERTIES PASS_REGULAR_EXPRESSION "${regex}")
    endforeach()
endif()
