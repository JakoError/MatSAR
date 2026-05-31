# Defines IMPORTED INTERFACE target MatSAR::blas and MATSAR_BLAS_SOURCE.
#
# OpenBLAS ships inconsistent CMake packaging across distros (the upstream
# OpenBLASConfig.cmake calls enable_language(Fortran) and doesn't always define
# an OpenBLAS::OpenBLAS target). So discovery is ordered to prefer the robust
# mechanisms first:
#   1. pkg-config            (Linux libopenblas-dev, MSYS2/MinGW pacman openblas)
#   2. FindBLAS (OpenBLAS)   (vcpkg on MSVC; finds the lib without the Fortran config)
#   3. find_package CONFIG   (last resort; target-or-variables)
#   4. bundled binaries shipped in the repo

# Repo root captured at include() time (CMAKE_CURRENT_LIST_DIR inside a function
# resolves to the CALLER's dir, so it must be resolved here, at module scope).
get_filename_component(_MATSAR_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

function(matsar_find_blas)
    if(TARGET MatSAR::blas)
        return()
    endif()
    set(_root ${_MATSAR_ROOT})

    # 1) pkg-config -- reliable, provides include dirs + link libs, avoids the
    #    Fortran-requiring OpenBLAS config.
    find_package(PkgConfig QUIET)
    if(PkgConfig_FOUND)
        pkg_check_modules(OPENBLAS QUIET openblas)
        if(OPENBLAS_FOUND)
            add_library(MatSAR::blas INTERFACE IMPORTED GLOBAL)
            target_include_directories(MatSAR::blas INTERFACE ${OPENBLAS_INCLUDE_DIRS})
            target_link_libraries(MatSAR::blas INTERFACE ${OPENBLAS_LINK_LIBRARIES})
            set(MATSAR_BLAS_SOURCE "pkg-config openblas (${OPENBLAS_VERSION})" PARENT_SCOPE)
            return()
        endif()
    endif()

    # 2) CMake's FindBLAS with the OpenBLAS vendor (e.g. vcpkg). Finds the library
    #    directly, so it does NOT invoke OpenBLAS's enable_language(Fortran) config.
    set(BLA_VENDOR OpenBLAS)
    find_package(BLAS QUIET)
    unset(BLA_VENDOR)
    if(BLAS_FOUND)
        add_library(MatSAR::blas INTERFACE IMPORTED GLOBAL)
        target_link_libraries(MatSAR::blas INTERFACE BLAS::BLAS)
        find_path(MATSAR_CBLAS_INCLUDE cblas.h PATH_SUFFIXES openblas)
        if(MATSAR_CBLAS_INCLUDE)
            target_include_directories(MatSAR::blas INTERFACE ${MATSAR_CBLAS_INCLUDE})
        endif()
        set(MATSAR_BLAS_SOURCE "find_package(BLAS, vendor OpenBLAS)" PARENT_SCOPE)
        return()
    endif()

    # 3) OpenBLAS CMake config. Use the imported target if it exists, else its
    #    variables (Debian ships the config without an OpenBLAS::OpenBLAS target).
    find_package(OpenBLAS CONFIG QUIET)
    if(OpenBLAS_FOUND)
        add_library(MatSAR::blas INTERFACE IMPORTED GLOBAL)
        if(TARGET OpenBLAS::OpenBLAS)
            target_link_libraries(MatSAR::blas INTERFACE OpenBLAS::OpenBLAS)
        else()
            target_include_directories(MatSAR::blas INTERFACE ${OpenBLAS_INCLUDE_DIRS})
            target_link_libraries(MatSAR::blas INTERFACE ${OpenBLAS_LIBRARIES})
        endif()
        set(MATSAR_BLAS_SOURCE "find_package(OpenBLAS CONFIG)" PARENT_SCOPE)
        return()
    endif()

    # 4) Bundled fallback (binaries shipped in the repo)
    add_library(MatSAR::blas INTERFACE IMPORTED GLOBAL)
    target_include_directories(MatSAR::blas INTERFACE ${_root}/include)
    if(WIN32)
        if(MSVC)
            target_link_libraries(MatSAR::blas INTERFACE ${_root}/lib/libopenblas.lib)
        else() # MinGW links the DLL directly
            target_link_libraries(MatSAR::blas INTERFACE ${_root}/bin/libopenblas.dll)
        endif()
        set(MATSAR_BLAS_DLL ${_root}/bin/libopenblas.dll CACHE INTERNAL "bundled DLL to stage")
        set(MATSAR_BLAS_SOURCE "bundled bin/libopenblas.dll" PARENT_SCOPE)
    else() # Linux/Unix: bundled static ELF archive
        target_link_libraries(MatSAR::blas INTERFACE ${_root}/lib/libopenblas.a pthread)
        set(MATSAR_BLAS_SOURCE "bundled lib/libopenblas.a" PARENT_SCOPE)
    endif()
endfunction()
