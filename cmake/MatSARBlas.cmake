# Defines IMPORTED INTERFACE target MatSAR::blas and MATSAR_BLAS_SOURCE.
# Order: find_package(OpenBLAS CONFIG) -> pkg-config -> bundled fallback.

# Repo root captured at include() time (CMAKE_CURRENT_LIST_DIR inside a function
# resolves to the CALLER's dir, so it must be resolved here, at module scope).
get_filename_component(_MATSAR_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

function(matsar_find_blas)
    if(TARGET MatSAR::blas)
        return()
    endif()
    set(_root ${_MATSAR_ROOT})

    # 1) CMake config package (vcpkg on MSVC, some system installs)
    find_package(OpenBLAS CONFIG QUIET)
    if(OpenBLAS_FOUND)
        add_library(MatSAR::blas INTERFACE IMPORTED GLOBAL)
        target_link_libraries(MatSAR::blas INTERFACE OpenBLAS::OpenBLAS)
        set(MATSAR_BLAS_SOURCE "find_package(OpenBLAS CONFIG)" PARENT_SCOPE)
        return()
    endif()

    # 2) pkg-config (MSYS2 pacman openblas, Linux libopenblas-dev)
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

    # 3) Bundled fallback (binaries shipped in the repo)
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
