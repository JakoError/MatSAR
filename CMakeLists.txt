cmake_minimum_required(VERSION 3.22)
project(MatSAR VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

string(COMPARE EQUAL "${CMAKE_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}" _matsar_is_top_level)

option(MATSAR_WITH_BLAS "Enable OpenBLAS-backed matrix multiply (mmul/%)" ON)
option(MATSAR_ENABLE_EXPERIMENTAL "Include unfinished experimental modules" OFF)
option(MATSAR_LEGACY_COLUMN_INITIALIZER_LIST "Revert nested {{...}} to legacy column-wise construction (temporary migration aid)" OFF)
option(MATSAR_BUILD_TESTS "Build MatSAR tests" ${_matsar_is_top_level})
option(MATSAR_INSTALL "Generate MatSAR install + package-config rules" ${_matsar_is_top_level})

# Header-only interface library
add_library(matsar INTERFACE)
add_library(MatSAR::matsar ALIAS matsar)
target_include_directories(matsar INTERFACE
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>)
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

# ---- install + package config (find_package(MatSAR) / FetchContent) ----
if(MATSAR_INSTALL)
    include(GNUInstallDirs)
    include(CMakePackageConfigHelpers)

    set(MATSAR_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/MatSAR"
            CACHE STRING "Install location for MatSAR CMake package files")

    # The public, stable headers. Experimental subtrees (math/, SparseMatrix.hpp,
    # UniversalFunctions.hpp) are intentionally NOT installed.
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/MatSAR
            DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
            PATTERN "math" EXCLUDE
            PATTERN "SparseMatrix.hpp" EXCLUDE
            PATTERN "UniversalFunctions.hpp" EXCLUDE)

    install(TARGETS matsar EXPORT MatSARTargets)
    install(EXPORT MatSARTargets
            NAMESPACE MatSAR::
            FILE MatSARTargets.cmake
            DESTINATION ${MATSAR_INSTALL_CMAKEDIR})

    configure_package_config_file(
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/MatSARConfig.cmake.in
            ${CMAKE_CURRENT_BINARY_DIR}/MatSARConfig.cmake
            INSTALL_DESTINATION ${MATSAR_INSTALL_CMAKEDIR})

    write_basic_package_version_file(
            ${CMAKE_CURRENT_BINARY_DIR}/MatSARConfigVersion.cmake
            VERSION ${PROJECT_VERSION}
            COMPATIBILITY SameMinorVersion)   # 0.x: API may change between minors

    # Ship MatSARConfig + version file + the BLAS discovery module (installed
    # consumers reuse matsar_find_blas() instead of a bare find_dependency).
    install(FILES
            ${CMAKE_CURRENT_BINARY_DIR}/MatSARConfig.cmake
            ${CMAKE_CURRENT_BINARY_DIR}/MatSARConfigVersion.cmake
            ${CMAKE_CURRENT_SOURCE_DIR}/cmake/MatSARBlas.cmake
            DESTINATION ${MATSAR_INSTALL_CMAKEDIR})
endif()
