
include_directories(include)

option(NEXUS_REQUIRE_CUDA "Fail build if CUDA compiler not found" OFF)
option(NEXUS_REQUIRE_TT   "Fail build if TT Metalium not found"   OFF)

add_subdirectory(cpu)

if(LINUX)

    include(CheckLanguage)
    check_language(CUDA)

    if(CMAKE_CUDA_COMPILER)

        message(STATUS "CUDA compiler found")

        enable_language(CUDA)
        set(CUDA_AVAILABLE TRUE)

        find_package(CUDAToolkit REQUIRED)

        add_subdirectory(cuda)

    else()

        if(NEXUS_REQUIRE_CUDA)
            message(FATAL_ERROR "CUDA compiler not found but NEXUS_REQUIRE_CUDA=ON")
        endif()

        set(CUDA_AVAILABLE FALSE)

    endif()

    add_subdirectory(hip)

    set(TT_METAL_CMAKE_PATH_CHECK $ENV{TT_METAL_CMAKE_PATH})
    if(NOT EXISTS "${TT_METAL_CMAKE_PATH_CHECK}" AND NEXUS_REQUIRE_TT)
        message(FATAL_ERROR "TT Metalium not found at '${TT_METAL_CMAKE_PATH_CHECK}' but NEXUS_REQUIRE_TT=ON")
    endif()

    add_subdirectory(tenstorrent)

elseif(MACOS)

    add_subdirectory(metal)

endif()
