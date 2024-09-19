################################################################################
# Required cmake version
################################################################################

cmake_minimum_required(VERSION 3.16.0)

################################################################################
# Project
################################################################################

include(CheckLanguage) # check for CUDA/HIP language support

project(CUDA_memtest LANGUAGES CXX)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}" CACHE PATH "install prefix" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

# own modules for find_packages
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${CMAKE_CURRENT_SOURCE_DIR}/cmake/)

set(CUDA_MEMTEST_BACKEND "cuda" CACHE STRING "Select the backend API used for the test.")
set_property(CACHE CUDA_MEMTEST_BACKEND PROPERTY STRINGS "cuda;hip")

option(CUDA_MEMTEST_ADD_RPATH "Add RPATH's to binaries." ON)

################################################################################
# CMake policies
#
# Search in <PackageName>_ROOT:
#   https://cmake.org/cmake/help/v3.12/policy/CMP0074.html
################################################################################

if(POLICY CMP0074)
    cmake_policy(SET CMP0074 NEW)
endif()


if(CUDA_MEMTEST_BACKEND STREQUAL "cuda")
    ################################################################################
    # Find CUDA
    ################################################################################

    enable_language(CUDA)
    # cuda toolkit contains CUDA::nvml
    find_package(CUDAToolkit)

    if(SAME_NVCC_FLAGS_IN_SUBPROJECTS)
        if(CUDA_SHOW_CODELINES)
            target_compile_options(cuda_memtest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--source-in-ptx -lineinfo>)
            set(CUDA_KEEP_FILES ON CACHE BOOL "activate keep files" FORCE)
        endif(CUDA_SHOW_CODELINES)

        if(CUDA_SHOW_REGISTER)
            target_compile_options(cuda_memtest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcuda-ptxas=-v>)
        endif()

        if(CUDA_KEEP_FILES)
            target_compile_options(cuda_memtest PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--keep>)
        endif()
    endif()

    ################################################################################
    # Find NVML
    ################################################################################

    set(GPU_DEPLOYMENT_KIT_ROOT_DIR "$ENV{GDK_ROOT}")
    find_package(NVML)
else()
    ################################################################################
    # Find HIP
    ################################################################################
    # supported HIP version range
    check_language(HIP)

    if(CMAKE_HIP_COMPILER)
        enable_language(HIP)
        find_package(hip REQUIRED)

        set(CUDA_MEMTEST_HIP_MIN_VER 5.2)
        set(CUDA_MEMTEST_HIP_MAX_VER 6.1)

        # construct hip version only with major and minor level
        # cannot use hip_VERSION because of the patch level
        # 6.0 is smaller than 6.0.1234, so CUDA_MEMTEST_HIP_MAX_VER would have to be defined with a large patch level or
        # the next minor level, e.g. 6.1, would have to be used.
        set(_hip_MAJOR_MINOR_VERSION "${hip_VERSION_MAJOR}.${hip_VERSION_MINOR}")

        if(${_hip_MAJOR_MINOR_VERSION} VERSION_LESS ${CUDA_MEMTEST_HIP_MIN_VER} OR ${_hip_MAJOR_MINOR_VERSION} VERSION_GREATER ${CUDA_MEMTEST_HIP_MAX_VER})
            message(WARNING "HIP ${_hip_MAJOR_MINOR_VERSION} is not official supported by cuda_memtest. Supported versions: ${CUDA_MEMTEST_HIP_MIN_VER} - ${CUDA_MEMTEST_HIP_MAX_VER}")
        endif()

    else()
        message(FATAL_ERROR "Optional alpaka dependency HIP could not be found!")
    endif()
endif()


################################################################################
# Find PThreads
################################################################################

if(NOT MSVC)
    set(THREADS_PREFER_PTHREAD_FLAG TRUE)
endif()

find_package(Threads REQUIRED)


################################################################################
# Warnings
################################################################################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")


################################################################################
# Compile & Link cuda_memtest
################################################################################

if(CUDA_MEMTEST_BACKEND STREQUAL "cuda")
    set_source_files_properties(tests.cpp cuda_memtest.cpp misc.cpp PROPERTIES LANGUAGE CUDA)
endif()

add_executable(cuda_memtest
        tests.cpp
        misc.cpp
        cuda_memtest.cpp
        )

if(CUDA_MEMTEST_BACKEND STREQUAL "cuda")
    target_link_libraries(cuda_memtest INTERFACE CUDA::cudart)
    target_link_libraries(cuda_memtest INTERFACE CUDA::cuda_driver)

    if(TARGET CUDA::nvml)
        message(STATUS "nvml found")
        target_link_libraries(cuda_memtest PRIVATE CUDA::nvml)
        target_compile_definitions(cuda_memtest PRIVATE "ENABLE_NVML=1")
    else()
        target_compile_definitions(cuda_memtest PRIVATE "ENABLE_NVML=0")
    endif()

    if(NOT CUDA_MEMTEST_RELEASE)
        set(CMAKE_BUILD_TYPE Debug)
    endif()
else()
    target_link_libraries(cuda_memtest PRIVATE hip::host)
    target_link_libraries(cuda_memtest PRIVATE hip::device)
    target_compile_definitions(cuda_memtest PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:__HIP_PLATFORM_AMD__>")
    target_compile_definitions(cuda_memtest PRIVATE "ENABLE_NVML=0")
endif()

if(NOT MSVC)
    target_link_libraries(cuda_memtest PRIVATE Threads::Threads)
endif()

## annotate with RPATH's
if(CUDA_MEMTEST_ADD_RPATH)
    if(NOT DEFINED CMAKE_INSTALL_RPATH)
        if(APPLE)
            set_target_properties(cuda_memtest PROPERTIES INSTALL_RPATH "@loader_path")
        elseif(CMAKE_SYSTEM_NAME MATCHES "Linux")
            set_target_properties(cuda_memtest PROPERTIES INSTALL_RPATH "$ORIGIN")
        endif()
    endif()
    if(NOT DEFINED CMAKE_INSTALL_RPATH_USE_LINK_PATH)
        set_target_properties(cuda_memtest PROPERTIES INSTALL_RPATH_USE_LINK_PATH ON)
    endif()
endif()

################################################################################
# Build type (debug, release)
################################################################################

option(CUDA_MEMTEST_RELEASE "disable all runtime asserts" ON)
if(CUDA_MEMTEST_RELEASE)
    target_compile_definitions(cuda_memtest PRIVATE NDEBUG)
endif(CUDA_MEMTEST_RELEASE)

################################################################################
# Install cuda_memtest
################################################################################

install(TARGETS cuda_memtest
        RUNTIME DESTINATION bin)
