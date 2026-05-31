cmake_minimum_required(VERSION 3.22)
project(MatSAR)

#set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD 11)

set(INC_DIR ./include)
include_directories(${INC_DIR})
set(LINK_DIR ./lib)
link_directories(${LINK_DIR})

set(MATH_DIR ./math)
include_directories(${MATH_DIR})

add_executable(MatSAR main.cpp old/Property.h Matrix.hpp
        util/VectorUtil.hpp util/TypeUtil.hpp util/Utils util/UtilHeadCheck.hpp util/ExceptionUtil.hpp util/IgnoreWarningUtil.hpp
        math/Trigonometry.hpp math/LinearAlgebra.hpp NumberOperation.hpp math/DigitalSignal.hpp math/MatrixOperation.hpp math/Probability.hpp)
add_executable(test test/test.cpp old/Property.h Matrix.hpp
        util/VectorUtil.hpp util/TypeUtil.hpp util/Utils util/UtilHeadCheck.hpp util/ExceptionUtil.hpp util/IgnoreWarningUtil.hpp
        math/Trigonometry.hpp math/LinearAlgebra.hpp NumberOperation.hpp math/DigitalSignal.hpp math/MatrixOperation.hpp math/Probability.hpp)

#add_executable(Matrix Matrix.hpp util/IgnoreWarningUtil.hpp math/LinearAlgebra.hpp NumberOperation.hpp math/DigitalSignal.hpp math/MatrixOperation.hpp math/Probability.hpp)
add_executable(pure_test test/pure_test.cpp)

add_executable(broadcast_test test/broadcast_test.cpp)
add_executable(numberop_test test/numberop_test.cpp)

add_executable(probability_test test/probability_test.cpp)
target_link_libraries(probability_test libopenblas)

add_executable(sortsearchcount_test test/sortsearchcount_test.cpp)
target_link_libraries(sortsearchcount_test libopenblas)

add_executable(logic_test test/logic_test.cpp)
target_link_libraries(logic_test libopenblas)

add_executable(matrix_creation_test test/matrix_creation.cpp)
target_link_libraries(matrix_creation_test libopenblas)

add_executable(mathematical_function_test test/mathematical_function_test.cpp)
target_link_libraries(mathematical_function_test libopenblas)

add_executable(linear_algebra_test test/linear_algebra_test.cpp)
target_link_libraries(linear_algebra_test libopenblas)

add_executable(matrix_operation_test test/matrix_operation_test.cpp)
add_executable(core_test test/core_test.cpp)
target_link_libraries(core_test libopenblas)

add_executable(openblas_test test/openblas_test.cpp)
target_link_libraries(openblas_test libopenblas)

add_executable(trigonometry_test test/trigonometry_test.cpp)
target_link_libraries(trigonometry_test libopenblas)

add_executable(matrix_operation_test2 test/matrix_operation_test2.cpp)
target_link_libraries(matrix_operation_test2 libopenblas)

add_executable(linear_algebra_test2 test/linear_algebra_test2.cpp)
target_link_libraries(linear_algebra_test2 libopenblas)

add_executable(probability_test2 test/probability_test2.cpp)
target_link_libraries(probability_test2 libopenblas)

add_executable(digital_signal_test test/digital_signal_test.cpp)
target_link_libraries(digital_signal_test libopenblas)
