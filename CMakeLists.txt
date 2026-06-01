cmake_minimum_required(VERSION 3.22)
project(MatSAR LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(MATSAR_WITH_BLAS "Enable OpenBLAS-backed matrix multiply (mmul/%)" ON)
option(MATSAR_ENABLE_EXPERIMENTAL "Include unfinished experimental modules" OFF)
option(MATSAR_LEGACY_COLUMN_INITIALIZER_LIST "Revert nested {{...}} to legacy column-wise construction (temporary migration aid)" OFF)
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    option(MATSAR_BUILD_TESTS "Build MatSAR tests" ON)
else()
    option(MATSAR_BUILD_TESTS "Build MatSAR tests" OFF)
endif()

# Header-only interface library
add_library(matsar INTERFACE)
add_library(matsar::matsar ALIAS matsar)
target_include_directories(matsar INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/math)
target_compile_features(matsar INTERFACE cxx_std_17)

if(MATSAR_ENABLE_EXPERIMENTAL)
    target_compile_definitions(matsar INTERFACE MATSAR_ENABLE_EXPERIMENTAL)
endif()

if(MATSAR_LEGACY_COLUMN_INITIALIZER_LIST)
    target_compile_definitions(matsar INTERFACE MATSAR_LEGACY_COLUMN_INITIALIZER_LIST)
endif()

if(MATSAR_WITH_BLAS)
    include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/MatSARBlas.cmake)
    matsar_find_blas()              # defines MatSAR::blas + MATSAR_BLAS_SOURCE
    message(STATUS "MatSAR BLAS: ${MATSAR_BLAS_SOURCE}")
    target_link_libraries(matsar INTERFACE MatSAR::blas)
    target_compile_definitions(matsar INTERFACE MATSAR_WITH_BLAS)
endif()

if(MATSAR_BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()
